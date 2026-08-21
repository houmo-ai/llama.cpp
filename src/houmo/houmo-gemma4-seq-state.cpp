#include "houmo-gemma4.h"
#include "houmo-state-io.h"

#include <algorithm>
#include <stdexcept>

namespace {
constexpr uint32_t GEMMA4_STATE_VERSION = 1;
}

class HoumoGemma4SeqState {
    public:
        static void write(const HouMoGemma4 & owner, llama_io_write_i & io,
                                            int seq_id, uint32_t flags);
        static void read(HouMoGemma4 & owner, llama_io_read_i & io,
                                         int seq_id, uint32_t flags);
        static int position(const HouMoGemma4 & owner, int seq_id);
};

void HoumoGemma4SeqState::write(const HouMoGemma4 & owner,
                                llama_io_write_i & io, int seq_id,
                                uint32_t flags) {
    houmo_state_io::reject_unsupported_flags(flags, "Gemma4");
    auto pending_it = owner.pending_verify_.find(seq_id);
    if (pending_it != owner.pending_verify_.end() && pending_it->second.valid) {
        throw std::runtime_error("cannot save Gemma4 state during pending verify");
    }
    auto chunk_it = owner.chunk_history_.find(seq_id);
    if (chunk_it != owner.chunk_history_.end() && !chunk_it->second.empty()) {
        throw std::runtime_error("cannot save Gemma4 state with pending prefill chunks");
    }
    const int32_t position = owner.memory_->seq_pos_min(seq_id);
    const auto valid_it = owner.decode_valid_length_.find(seq_id);
    const int32_t decode_valid = valid_it == owner.decode_valid_length_.end()
                                     ? position : valid_it->second;
    const auto history_it = owner.input_history_.find(seq_id);
    const auto kv_it = owner.kv_cache_.find(seq_id);
    const auto committed_it = owner.last_committed_tokens_.find(seq_id);
    const int32_t last_committed = committed_it == owner.last_committed_tokens_.end()
                                       ? 0 : committed_it->second;
    const auto start_it = owner.device_kv_abs_start_.find(seq_id);
    const auto end_it = owner.device_kv_abs_end_.find(seq_id);
    const int64_t range_start = start_it == owner.device_kv_abs_start_.end()
                                    ? std::max<int64_t>(0, position - owner.sliding_window_)
                                    : start_it->second;
    const int64_t range_end = end_it == owner.device_kv_abs_end_.end()
                                  ? position : end_it->second;
    if (position < 0 || decode_valid != position || history_it == owner.input_history_.end() ||
        houmo_state_io::count_chunk_tokens(history_it->second) != static_cast<uint64_t>(position) ||
        kv_it == owner.kv_cache_.end() || range_end != position || range_start < 0 ||
        range_start > range_end || last_committed < 0) {
        throw std::runtime_error("Gemma4 sequence is not in a committed state");
    }

    houmo_state_io::write_value(io, GEMMA4_STATE_VERSION);
    houmo_state_io::write_value(io, position);
    houmo_state_io::write_value(io, decode_valid);
    houmo_state_io::write_value(io, range_start);
    houmo_state_io::write_value(io, range_end);
    houmo_state_io::write_value(io, last_committed);
    houmo_state_io::write_chunks(io, history_it->second, owner.n_embd_);
    houmo_state_io::write_tensors(io, kv_it->second.names, kv_it->second.tensors);
}

void HoumoGemma4SeqState::read(HouMoGemma4 & owner,
                               llama_io_read_i & io, int seq_id,
                               uint32_t flags) {
    houmo_state_io::reject_unsupported_flags(flags, "Gemma4");
    uint32_t version = 0;
    int32_t position = 0;
    int32_t decode_valid = 0;
    int64_t range_start = 0;
    int64_t range_end = 0;
    int32_t last_committed = 0;
    houmo_state_io::read_value(io, version);
    houmo_state_io::read_value(io, position);
    houmo_state_io::read_value(io, decode_valid);
    houmo_state_io::read_value(io, range_start);
    houmo_state_io::read_value(io, range_end);
    houmo_state_io::read_value(io, last_committed);
    if (version != GEMMA4_STATE_VERSION || position < 0 || decode_valid != position ||
        range_end != position || range_start < 0 || range_start > range_end ||
        last_committed < 0 || last_committed > position) {
        throw std::runtime_error("invalid Gemma4 state metadata");
    }
    auto history = houmo_state_io::read_chunks(io, owner.n_embd_, position);
    auto kv_it = owner.kv_cache_.find(seq_id);
    if (kv_it == owner.kv_cache_.end()) {
        throw std::runtime_error("Gemma4 destination KV cache not found");
    }
    auto pending_kv = houmo_state_io::read_tensors(
        io, kv_it->second.names, kv_it->second.tensors);
    auto backup_kv = houmo_state_io::backup_tensors(kv_it->second.tensors);
    try {
        houmo_state_io::commit_tensors(pending_kv, kv_it->second.tensors);
    } catch (...) {
        houmo_state_io::restore_tensors(backup_kv, kv_it->second.tensors);
        throw;
    }

    owner.input_history_[seq_id] = std::move(history);
    owner.decode_valid_length_[seq_id] = decode_valid;
    owner.device_kv_abs_start_[seq_id] = std::max<int64_t>(0, position - owner.sliding_window_);
    owner.device_kv_abs_end_[seq_id] = position;
    owner.last_committed_tokens_[seq_id] = last_committed;
    owner.chunk_history_[seq_id].clear();
    owner.pending_verify_.erase(seq_id);
    owner.kv_reuse_cache_.clear_seq(seq_id);
}

int HoumoGemma4SeqState::position(const HouMoGemma4 & owner, int seq_id) {
    const auto it = owner.decode_valid_length_.find(seq_id);
    return it == owner.decode_valid_length_.end() ? -1 : it->second;
}

void HouMoGemma4::state_write(llama_io_write_i & io, int seq_id,
                              llama_state_seq_flags flags) const {
    HoumoGemma4SeqState::write(*this, io, seq_id, flags);
}

void HouMoGemma4::state_read(llama_io_read_i & io, int seq_id,
                             llama_state_seq_flags flags) {
    HoumoGemma4SeqState::read(*this, io, seq_id, flags);
}

int HouMoGemma4::state_position(int seq_id) const {
    return HoumoGemma4SeqState::position(*this, seq_id);
}
