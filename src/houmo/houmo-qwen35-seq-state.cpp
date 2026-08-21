#include "houmo-qwen35llm.h"
#include "houmo-state-io.h"

#include <stdexcept>

namespace {
constexpr uint32_t QWEN35_STATE_VERSION = 1;
}

class HoumoQwen35SeqState {
    public:
        static void write(const HoumoQwen35LLM & owner, llama_io_write_i & io,
                                            int seq_id, uint32_t flags);
        static void read(HoumoQwen35LLM & owner, llama_io_read_i & io,
                                         int seq_id, uint32_t flags);
        static int position(const HoumoQwen35LLM & owner, int seq_id);
};

void HoumoQwen35SeqState::write(const HoumoQwen35LLM & owner,
                                llama_io_write_i & io, int seq_id,
                                uint32_t flags) {
    houmo_state_io::reject_unsupported_flags(flags, "Qwen3.5");
    auto pending_it = owner.pending_verify_.find(seq_id);
    if (pending_it != owner.pending_verify_.end() && pending_it->second.valid) {
        throw std::runtime_error("cannot save Qwen3.5 state during pending verify");
    }
    auto chunk_it = owner.chunk_history_.find(seq_id);
    if (chunk_it != owner.chunk_history_.end() && !chunk_it->second.empty()) {
        throw std::runtime_error("cannot save Qwen3.5 state with pending prefill chunks");
    }
    const int32_t position = owner.memory_->seq_pos_min(seq_id);
    const auto valid_it = owner.decode_valid_length_.find(seq_id);
    const int32_t decode_valid = valid_it == owner.decode_valid_length_.end()
                                     ? position : valid_it->second;
    const auto offset_it = owner.decode_position_offset_.find(seq_id);
    const int32_t position_offset = offset_it == owner.decode_position_offset_.end()
                                        ? 0 : offset_it->second;
    const auto * snapshot = owner.state_cache_.find_snapshot(seq_id);
    const auto * state_entry = owner.state_cache_.find_entry(seq_id);
    const auto kv_it = owner.kv_cache_.find(seq_id);
    if (position < 0 || decode_valid != position || snapshot == nullptr ||
        !snapshot->valid || snapshot->last_processed_pos != position ||
        houmo_state_io::count_chunk_tokens(snapshot->chunks) != static_cast<uint64_t>(position) ||
        state_entry == nullptr || kv_it == owner.kv_cache_.end()) {
        throw std::runtime_error("Qwen3.5 sequence is not in a committed state");
    }

    houmo_state_io::write_value(io, QWEN35_STATE_VERSION);
    houmo_state_io::write_value(io, position);
    houmo_state_io::write_value(io, decode_valid);
    houmo_state_io::write_value(io, position_offset);
    houmo_state_io::write_chunks(io, snapshot->chunks, owner.n_embd_);
    houmo_state_io::write_tensors(io, kv_it->second.names, kv_it->second.tensors);
    houmo_state_io::write_tensors(io, owner.state_cache_.state_names(),
                                  state_entry->tensors);
}

void HoumoQwen35SeqState::read(HoumoQwen35LLM & owner,
                               llama_io_read_i & io, int seq_id,
                               uint32_t flags) {
    houmo_state_io::reject_unsupported_flags(flags, "Qwen3.5");
    uint32_t version = 0;
    int32_t position = 0;
    int32_t decode_valid = 0;
    int32_t position_offset = 0;
    houmo_state_io::read_value(io, version);
    houmo_state_io::read_value(io, position);
    houmo_state_io::read_value(io, decode_valid);
    houmo_state_io::read_value(io, position_offset);
    // Vision M-RoPE may legitimately use a negative offset: the physical
    // number of image tokens can be larger than the logical M-RoPE advance.
    // Validate the resulting logical decode position instead of the offset
    // itself.  The upper-bound check also avoids signed overflow before the
    // range comparison below.
    const int64_t logical_position = static_cast<int64_t>(position) +
                                     static_cast<int64_t>(position_offset);
    if (version != QWEN35_STATE_VERSION || position < 0 ||
        position > owner.context_length_ || decode_valid != position ||
        logical_position < 0 || logical_position > owner.context_length_) {
        throw std::runtime_error("invalid Qwen3.5 state metadata");
    }
    auto chunks = houmo_state_io::read_chunks(io, owner.n_embd_, position);
    auto kv_it = owner.kv_cache_.find(seq_id);
    auto * state_entry = owner.state_cache_.find_entry(seq_id);
    if (kv_it == owner.kv_cache_.end() || state_entry == nullptr) {
        throw std::runtime_error("Qwen3.5 destination cache not found");
    }
    auto pending_kv = houmo_state_io::read_tensors(
        io, kv_it->second.names, kv_it->second.tensors);
    auto pending_state = houmo_state_io::read_tensors(
        io, owner.state_cache_.state_names(), state_entry->tensors);

    auto backup_kv = houmo_state_io::backup_tensors(kv_it->second.tensors);
    auto backup_state = houmo_state_io::backup_tensors(state_entry->tensors);
    try {
        houmo_state_io::commit_tensors(pending_kv, kv_it->second.tensors);
        houmo_state_io::commit_tensors(pending_state, state_entry->tensors);
    } catch (...) {
        houmo_state_io::restore_tensors(backup_kv, kv_it->second.tensors);
        houmo_state_io::restore_tensors(backup_state, state_entry->tensors);
        throw;
    }

    auto & snapshot = owner.state_cache_.snapshot(seq_id);
    snapshot.chunks = std::move(chunks);
    snapshot.last_processed_pos = position;
    snapshot.valid = true;
    owner.state_cache_.mark_initialized(seq_id, true);
    owner.state_cache_.clear_all_checkpoints(seq_id);
    owner.decode_valid_length_[seq_id] = decode_valid;
    owner.decode_position_offset_[seq_id] = position_offset;
    owner.chunk_history_[seq_id].clear();
    owner.pending_verify_.erase(seq_id);
}

int HoumoQwen35SeqState::position(const HoumoQwen35LLM & owner, int seq_id) {
    const auto it = owner.decode_valid_length_.find(seq_id);
    return it == owner.decode_valid_length_.end() ? -1 : it->second;
}

void HoumoQwen35LLM::state_write(llama_io_write_i & io, int seq_id,
                                 llama_state_seq_flags flags) const {
    HoumoQwen35SeqState::write(*this, io, seq_id, flags);
}

void HoumoQwen35LLM::state_read(llama_io_read_i & io, int seq_id,
                                llama_state_seq_flags flags) {
    HoumoQwen35SeqState::read(*this, io, seq_id, flags);
}

int HoumoQwen35LLM::state_position(int seq_id) const {
    return HoumoQwen35SeqState::position(*this, seq_id);
}
