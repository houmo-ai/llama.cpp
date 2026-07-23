#include "houmo-qwen35llm.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstring>
#include <filesystem>
#include <fstream>
#include "ggml.h"
#include "llama-impl.h"
#include "time-perf.h"

#pragma pack(push, 1)
struct VisionCompactHeader {
    int16_t magic;
    int16_t nx;
    int16_t ny;
    int16_t chunksize;
};
#pragma pack(pop)

namespace {
std::string find_input_name(const std::map<std::string, tcim::Tensor> &inputs,
                            const std::string &key) {
    for (const auto &pair : inputs) {
        if (pair.first.find(key) != std::string::npos) {
            return pair.first;
        }
    }
    return "";
}

std::string find_input_name_any(const std::map<std::string, tcim::Tensor> &inputs,
                                const std::vector<std::string> &keys) {
    for (const auto &key : keys) {
        auto name = find_input_name(inputs, key);
        if (!name.empty()) {
            return name;
        }
    }
    return "";
}

int extract_batch_id(const std::string &name) {
    auto pos = name.find("_batch");
    if (pos == std::string::npos) {
        return -1;
    }
    auto start = pos + std::strlen("_batch");
    if (start >= name.size()) {
        return -1;
    }
    try {
        return std::stoi(name.substr(start));
    } catch (...) {
        return -1;
    }
}

std::vector<std::string> collect_input_names(
    const std::map<std::string, tcim::Tensor> &inputs,
    const std::string &key) {
    std::vector<std::string> names;
    for (const auto &pair : inputs) {
        if (pair.first.find(key) != std::string::npos) {
            names.push_back(pair.first);
        }
    }
    std::sort(names.begin(), names.end(), [](const std::string &a, const std::string &b) {
        int batch_a = extract_batch_id(a);
        int batch_b = extract_batch_id(b);
        if (batch_a >= 0 && batch_b >= 0) {
            return batch_a < batch_b;
        }
        return a < b;
    });
    return names;
}

std::vector<std::string> collect_input_names_any(
    const std::map<std::string, tcim::Tensor> &inputs,
    const std::vector<std::string> &keys) {
    std::vector<std::string> names;
    for (const auto &key : keys) {
        auto partial = collect_input_names(inputs, key);
        names.insert(names.end(), partial.begin(), partial.end());
    }
    std::sort(names.begin(), names.end(), [](const std::string &a, const std::string &b) {
        int batch_a = extract_batch_id(a);
        int batch_b = extract_batch_id(b);
        if (batch_a >= 0 && batch_b >= 0) {
            return batch_a < batch_b;
        }
        return a < b;
    });
    names.erase(std::unique(names.begin(), names.end()), names.end());
    return names;
}

std::string resolve_batch_name(const std::vector<std::string> &names,
                               int batch_id) {
    if (names.empty()) {
        return "";
    }
    if (names.size() == 1) {
        return names.front();
    }
    for (const auto &name : names) {
        if (extract_batch_id(name) == batch_id) {
            return name;
        }
    }
    if (batch_id >= 0 && static_cast<size_t>(batch_id) < names.size()) {
        return names[batch_id];
    }
    return names.front();
}

void zero_tensor(tcim::Tensor &tensor) {
    if (tensor.Device() == tcim::Device::COMPND) {
        for (const auto & sub_tensor : tcim::CompTensor(tensor).SubTensors()) {
            sub_tensor.Buffer().MemSet(0);
        }
    } else {
        tensor.Buffer().MemSet(0);
    }
}

} // namespace

void HoumoQwen35LLM::houmo_init(houmo_memory_i *memory, int seq_max,
                                ggml_abort_callback abort_callback,
                                void *abort_callback_data) {
    abort_callback_data_ = abort_callback_data;
    abort_callback_ = abort_callback;
    model_inout_init(seq_max);
    if (is_embedding()) {
        memory_ = nullptr;
    } else {
        memory_ = memory;
    }
    state_cache_.configure_state_cache(prefill_model_, decode_model_, prefill_length_);
    resolve_input_names();
    state_cache_.initialize_state_cache(seq_max, n_prefill_batch_, n_decode_batch_,
                                        n_context_length(), device_ids_.size() > 1);
}

void HoumoQwen35LLM::resolve_input_names() {
    prefill_time_pos_name_ = find_input_name_any(
        prefill_input_map_, {"time_position", "time_position.hmcc.format"});
    prefill_height_pos_name_ = find_input_name_any(
        prefill_input_map_,
        {"height_position", "hight_position", "height_position.hmcc.format",
         "hight_position.hmcc.format"});
    prefill_width_pos_name_ = find_input_name_any(
        prefill_input_map_, {"width_position", "width_position.hmcc.format"});
    prefill_valid_length_name_ = find_input_name_any(
        prefill_input_map_,
        {"valid_length", "past_seq_length", "valid_length.hmcc.format",
         "past_seq_length.hmcc.format"});
    prefill_current_length_name_ = find_input_name_any(
        prefill_input_map_,
        {"current_length", "current_input_length", "current_length.hmcc.format",
         "current_input_length.hmcc.format"});
    prefill_linear_mask_name_ = find_input_name_any(
        prefill_input_map_,
        {"linear_attn_mask", "linear_attn_mask.hmcc.format"});

    decode_time_pos_names_.clear();
    decode_height_pos_names_.clear();
    decode_width_pos_names_.clear();
    decode_linear_mask_names_.clear();
    decode_valid_length_names_.clear();
    decode_current_length_names_.clear();

    decode_time_pos_names_ = collect_input_names_any(
        decode_input_map_, {"time_position", "time_position.hmcc.format"});
    decode_height_pos_names_ = collect_input_names_any(
        decode_input_map_,
        {"height_position", "hight_position", "height_position.hmcc.format",
         "hight_position.hmcc.format"});
    decode_width_pos_names_ = collect_input_names_any(
        decode_input_map_, {"width_position", "width_position.hmcc.format"});
    decode_valid_length_names_ = collect_input_names_any(
        decode_input_map_,
        {"valid_length", "past_seq_length", "valid_length.hmcc.format",
         "past_seq_length.hmcc.format"});
    decode_current_length_names_ = collect_input_names_any(
        decode_input_map_,
        {"current_length", "current_input_length", "current_length.hmcc.format",
         "current_input_length.hmcc.format"});
    decode_linear_mask_names_ = collect_input_names_any(
        decode_input_map_,
        {"linear_attn_mask", "linear_attn_mask.hmcc.format"});
}


int HoumoQwen35LLM::commit_verify_accept(int seq_id, int accepted_steps) {
    PERF_START(commit_verify_accept);
    auto it = pending_verify_.find(seq_id);
    if (it == pending_verify_.end() || !it->second.valid) {
        return 0;
    }

    auto &pending = it->second;
    if (accepted_steps <= 0) {
        pending.valid = false;
        pending.tokens.clear();
        pending.n_verify = 0;
        return 0;
    }

    accepted_steps = std::min(accepted_steps, pending.n_verify);
    const int32_t keep_pos = pending.start_pos + accepted_steps;

    if (state_cache_.enabled()) {
        if (state_cache_.bind_verify_state_to_cache(seq_id, accepted_steps - 1) != 0) {
            LLAMA_LOG_ERROR("%s: failed to bind verify state outputs seq_id=%d accept_pos=%d\n",
                            __func__, seq_id, accepted_steps - 1);
            return -1;
        }
    }

    if (memory_) {
        memory_->seq_add(seq_id, 0, 0, accepted_steps);
    }

    auto &snapshot = state_cache_.snapshot(seq_id);
    append_snapshot_text_history(seq_id, pending.tokens.data(),
                                 static_cast<size_t>(accepted_steps));
    snapshot.last_processed_pos = keep_pos;
    snapshot.valid = true;

    decode_valid_length_[seq_id] = keep_pos;
    pending.valid = false;
    pending.tokens.clear();
    pending.n_verify = 0;
    PERF_STOP(commit_verify_accept);

    return 0;
}

std::vector<int32_t> HoumoQwen35LLM::build_rope_position_ids(int valid_length,
                                                             int length) const {
    std::vector<int32_t> position_ids(static_cast<size_t>(length), 0);
    for (int i = 0; i < length; ++i) {
        position_ids[static_cast<size_t>(i)] = valid_length + i;
    }
    return position_ids;
}

void HoumoQwen35LLM::set_linear_attn_mask(tcim::Tensor &tensor, int fill_length,
                                          int new_cache_length) const {
    int batch_size = 1;
    auto shape = tensor.Info().Shape();
    if (!shape.empty()) {
        batch_size = static_cast<int>(shape[0]);
    }
    if (fill_length <= 0 || new_cache_length <= 0) {
        zero_tensor(tensor);
        return;
    }

    if (tensor.Info().DataType() == tcim::DataType::FLOAT16) {
        size_t elem_count = tensor.MemSize() / sizeof(ggml_fp16_t);
        std::vector<ggml_fp16_t> data(elem_count, ggml_fp32_to_fp16(0.0f));
        size_t stride = elem_count / static_cast<size_t>(batch_size);
        ggml_fp16_t one = ggml_fp32_to_fp16(1.0f);
        for (int b = 0; b < batch_size; ++b) {
            size_t offset = static_cast<size_t>(b) * stride;
            size_t limit = std::min(static_cast<size_t>(new_cache_length), stride);
            for (size_t i = 0; i < limit; ++i) {
                data[offset + i] = one;
            }
        }
        tensor.Buffer().CopyFromHost(data.data(), tensor.MemSize());
    }  else {
        zero_tensor(tensor);
    }
}

int HoumoQwen35LLM::set_lora_inputs(
    tcim::Module &model,
    std::map<std::string, tcim::Tensor> &input_map) {
    namespace fs = std::filesystem;
    if (lora_map.empty()) {
        return 0;
    }
    std::string enabled_path;
    for (const auto &pair : lora_map) {
        if (pair.second.scale != 0.0f) {
            if (!fs::is_directory(pair.first)) {
                LLAMA_LOG_ERROR("%s: LoRA directory not found: %s\n",
                                __func__, pair.first.c_str());
                return -1;
            }
            enabled_path = pair.first;
            break; // 一次一个lora
        }
    }

    if (!enabled_path.empty()) {
        for (const auto &entry : fs::directory_iterator(enabled_path)) {
            if (!entry.is_regular_file() || entry.path().extension() != ".bin") {
                continue;
            }
            const std::string input_name = entry.path().stem().string();
            LLAMA_LOG_INFO("%s: loading LoRA input '%s' from %s\n",
                            __func__, input_name.c_str(),
                            entry.path().string().c_str());
            auto input = input_map.find(input_name);
            if (input == input_map.end()) {
                LLAMA_LOG_ERROR("%s: LoRA input '%s' from %s is not present "
                                "in the model\n",
                                __func__, input_name.c_str(),
                                entry.path().string().c_str());
                return -1;
            }

            const uintmax_t file_size = fs::file_size(entry.path());
            std::vector<char> bytes(static_cast<size_t>(file_size));
            std::ifstream stream(entry.path(), std::ios::binary);
            if (!stream.read(bytes.data(),
                             static_cast<std::streamsize>(bytes.size()))) {
                LLAMA_LOG_ERROR("%s: failed to read LoRA input: %s\n",
                                __func__, entry.path().string().c_str());
                return -1;
            }
            input->second.Buffer().CopyFromHost(bytes.data(), bytes.size());
            if (model.SetInput(input_name, input->second) != tcim::Status::OK) {
                LLAMA_LOG_ERROR("%s: failed to set LoRA input '%s'\n",
                                __func__, input_name.c_str());
                return -1;
            }
        }
        return 0;
    }

    for (auto &input : input_map) {
        // 只能清空lora
        if (input.first.find("lora") != std::string::npos) {
            zero_tensor(input.second);
            if (model.SetInput(input.first, input.second) != tcim::Status::OK) {
                LLAMA_LOG_ERROR("%s: failed to clear LoRA input '%s'\n",
                                __func__, input.first.c_str());
                return -1;
            }
        }
    }
    return 0;
}

int HoumoQwen35LLM::run_prefill_round(int seq_id,
                                      tcim::Tensor &input_data,
                                      int32_t valid_length,
                                      int32_t current_length,
                                      const int32_t *time_pos_slice,
                                      const int32_t *height_pos_slice,
                                      const int32_t *width_pos_slice,
                                      bool is_first_round) {
    PERF_START(PREFILL_SETINPUT_ONCE);
    if (!prefill_valid_length_name_.empty()) {
        auto vl = prefill_input_map_.at(prefill_valid_length_name_);
        set_input_data(vl, n_prefill_batch_, valid_length);
        prefill_model_->SetInput(prefill_valid_length_name_, vl);
    }
    if (!prefill_current_length_name_.empty()) {
        auto cl = prefill_input_map_.at(prefill_current_length_name_);
        set_input_data(cl, n_prefill_batch_, current_length);
        prefill_model_->SetInput(prefill_current_length_name_, cl);
    }

    auto bind_position = [&](const std::string &name,
                             const int32_t *values) {
        if (name.empty() || values == nullptr) {
            return;
        }
        auto input = prefill_input_map_.find(name);
        if (input == prefill_input_map_.end()) {
            LLAMA_LOG_ERROR("%s: missing prefill position input '%s'\n",
                            __func__, name.c_str());
            return;
        }
        auto tensor = input->second;
        std::vector<int32_t> positions(values, values + prefill_length_);
        set_input_data(tensor, positions);
        prefill_model_->SetInput(name, tensor);
    };
    bind_position(prefill_time_pos_name_, time_pos_slice);
    bind_position(prefill_height_pos_name_, height_pos_slice);
    bind_position(prefill_width_pos_name_, width_pos_slice);

    if (!prefill_linear_mask_name_.empty()) {
        auto linear_mask = prefill_input_map_.at(prefill_linear_mask_name_);
        set_linear_attn_mask(linear_mask, prefill_length_, current_length);
        prefill_model_->SetInput(prefill_linear_mask_name_, linear_mask);
    }

    if (is_first_round) {
        if (set_lora_inputs(*prefill_model_, prefill_input_map_) != 0) {
            return -1;
        }
        if (set_lora_inputs(*decode_model_, decode_input_map_) != 0) {
            LLAMA_LOG_ERROR("%s: failed to set decode LoRA inputs\n", __func__);
            return -1;
        }
    }

    prefill_model_->SetInput(prefill_model_->GetInputName(0), input_data);
    if (is_first_round) {
        state_cache_.set_state_cache_for_prefill(seq_id);
        set_kv_cache_for_prefill(seq_id);
    }
    PERF_STOP(PREFILL_SETINPUT_ONCE);

    PERF_START(PREFILL_RUN_ONCE);
    const auto status = prefill_model_->Run();
    if (status != tcim::Status::OK) {
        PERF_STOP(PREFILL_RUN_ONCE);
        LLAMA_LOG_ERROR("%s: prefill_model_->Run() failed, result = %d\n",
                        __func__, static_cast<int>(status));
        const bool fatal = status == tcim::Status::TIMEOUT ||
                           status == tcim::Status::ERR_KERNEL ||
                           status == tcim::Status::ERR_FATAL;
        if (fatal) {
            // 严重错误，退出程序
            exit(-1);
        }
        return -1;
    }
    prefill_model_->Sync();
    PERF_STOP(PREFILL_RUN_ONCE);
    return 0;
}

bool HoumoQwen35LLM::has_vision_chunk(int seq_id) const {
    auto it = chunk_history_.find(seq_id);
    if (it == chunk_history_.end()) {
        return false;
    }
    return std::any_of(it->second.begin(), it->second.end(),
                       [](const PrefillChunk &chunk) {
                           return chunk.is_embedding;
                       });
}

size_t HoumoQwen35LLM::chunk_token_count(
    const std::vector<PrefillChunk> &chunks) const {
    size_t count = 0;
    for (const auto &chunk : chunks) {
        count += chunk.size();
    }
    return count;
}

void HoumoQwen35LLM::append_text_chunk(
    std::vector<PrefillChunk> &chunks,
    const std::vector<llama_token> &tokens) const {
    if (tokens.empty()) {
        return;
    }
    PrefillChunk chunk;
    chunk.n_tokens = tokens.size();
    chunk.tokens = tokens;
    chunks.push_back(std::move(chunk));
}

void HoumoQwen35LLM::append_snapshot_text_history(
    int seq_id, const llama_token *tokens, size_t n_tokens) {
    if (!state_cache_.enabled() || tokens == nullptr || n_tokens == 0) {
        return;
    }

    auto &snapshot = state_cache_.snapshot(seq_id);
    if (!snapshot.chunks.empty() && !snapshot.chunks.back().is_embedding) {
        auto &chunk = snapshot.chunks.back();
        chunk.tokens.insert(chunk.tokens.end(), tokens, tokens + n_tokens);
        chunk.n_tokens = chunk.tokens.size();
    } else {
        PrefillChunk chunk;
        chunk.n_tokens = n_tokens;
        chunk.tokens.assign(tokens, tokens + n_tokens);
        snapshot.chunks.push_back(std::move(chunk));
    }
}

bool HoumoQwen35LLM::slice_snapshot_history(
    int seq_id, size_t begin, size_t end,
    std::vector<PrefillChunk> &out) const {
    out.clear();
    const auto *snapshot = state_cache_.find_snapshot(seq_id);
    if (snapshot == nullptr || begin > end ||
        end > chunk_token_count(snapshot->chunks)) {
        return false;
    }
    if (begin == end) {
        return true;
    }

    size_t position = 0;
    for (const auto &chunk : snapshot->chunks) {
        const size_t chunk_begin = position;
        const size_t chunk_end = position + chunk.size();
        position = chunk_end;
        const size_t copy_begin = std::max(begin, chunk_begin);
        const size_t copy_end = std::min(end, chunk_end);
        if (copy_begin >= copy_end) {
            continue;
        }
        const size_t offset = copy_begin - chunk_begin;
        const size_t count = copy_end - copy_begin;
        if (chunk.is_embedding &&
            (offset != 0 || count != chunk.size())) {
            LLAMA_LOG_ERROR(
                "%s: range [%zu..%zu) cuts vision chunk [%zu..%zu) "
                "for seq_id=%d\n",
                __func__, begin, end, chunk_begin, chunk_end, seq_id);
            return false;
        }
        PrefillChunk part;
        part.is_embedding = chunk.is_embedding;
        part.bidirectional_attention = chunk.bidirectional_attention;
        part.n_tokens = count;
        part.nx = chunk.nx;
        part.ny = chunk.ny;
        if (chunk.is_embedding) {
            const size_t elem_begin = offset * static_cast<size_t>(n_embd_);
            const size_t elem_end = (offset + count) * static_cast<size_t>(n_embd_);
            if (elem_end > chunk.embeddings.size()) {
                return false;
            }
            part.embeddings.assign(chunk.embeddings.begin() + elem_begin,
                                   chunk.embeddings.begin() + elem_end);
        } else {
            if (offset + count > chunk.tokens.size()) {
                return false;
            }
            part.tokens.assign(chunk.tokens.begin() + offset,
                               chunk.tokens.begin() + offset + count);
        }
        out.push_back(std::move(part));
    }
    return true;
}

bool HoumoQwen35LLM::position_at_token_boundary(
    const std::vector<PrefillChunk> &chunks, size_t token_boundary,
    int32_t &position) const {
    size_t consumed = 0;
    position = 0;
    for (const auto &chunk : chunks) {
        if (consumed == token_boundary) {
            return true;
        }
        if (consumed > token_boundary) {
            return false;
        }

        const size_t remaining = token_boundary - consumed;
        const size_t take = std::min(remaining, chunk.size());
        if (chunk.is_embedding) {
            const size_t grid_tokens =
                static_cast<size_t>(std::max<int>(0, chunk.nx)) *
                static_cast<size_t>(std::max<int>(0, chunk.ny));
            if (chunk.nx <= 0 || chunk.ny <= 0 ||
                grid_tokens != chunk.size() || take != chunk.size()) {
                return false;
            }
            position += std::max<int>(chunk.nx, chunk.ny);
        } else {
            position += static_cast<int32_t>(take);
        }
        consumed += take;
    }
    return consumed == token_boundary;
}

bool HoumoQwen35LLM::prepare_prefill_replay(
    int seq_id,
    int32_t kv_offset,
    const std::vector<PrefillChunk> &pending_chunks,
    bool require_snapshot,
    std::vector<PrefillChunk> &effective_chunks,
    int32_t &effective_kv_offset,
    int32_t &effective_position_base) {
    effective_chunks = pending_chunks;
    effective_kv_offset = kv_offset;
    effective_position_base = kv_offset + decode_position_offset_[seq_id];

    if (!state_cache_.enabled()) {
        return true;
    }
    if (kv_offset == 0) {
        state_cache_.reset_for_fresh_prefill(seq_id);
        decode_position_offset_[seq_id] = 0;
        effective_position_base = 0;
        return true;
    }

    auto *snapshot = state_cache_.find_snapshot(seq_id);
    if (snapshot == nullptr || !snapshot->valid) {
        LLAMA_LOG_ERROR("%s: no state snapshot for seq_id=%d at kv_offset=%d\n",
                        __func__, seq_id, kv_offset);
        if (require_snapshot) {
            return false;
        }
        state_cache_.mark_initialized(seq_id, true);
        return true;
    }
    if (snapshot->last_processed_pos == kv_offset) {
        return true;
    }

    const size_t saved_history_size = chunk_token_count(snapshot->chunks);
    int best_ckpt = state_cache_.find_and_restore_best_checkpoint(seq_id,
                                                                  kv_offset);

    const int rollback_pos = best_ckpt > 0 ? best_ckpt : 0;
    if (rollback_pos >= kv_offset) {
        if (!position_at_token_boundary(
                snapshot->chunks, static_cast<size_t>(kv_offset),
                effective_position_base)) {
            LLAMA_LOG_ERROR(
                "%s: checkpoint boundary %d is unsafe for seq_id=%d\n",
                __func__, kv_offset, seq_id);
            state_cache_.invalidate_snapshot(seq_id, true);
            return false;
        }
        state_cache_.mark_initialized(seq_id, true);
        snapshot->valid = false;
        return true;
    }

    if (static_cast<size_t>(kv_offset) > saved_history_size) {
        LLAMA_LOG_WARN(
            "%s: saved history too short (%zu) to fill gap [%d..%d) "
            "for seq_id=%d, refusing replay; state may be inaccurate. "
            "should never come here\n",
            __func__, saved_history_size, rollback_pos, kv_offset, seq_id);
        state_cache_.mark_initialized(seq_id, true);
        state_cache_.invalidate_snapshot(seq_id, true);
        return false;
    }

    std::vector<PrefillChunk> replay_chunks;
    if (!slice_snapshot_history(seq_id, static_cast<size_t>(rollback_pos),
                                static_cast<size_t>(kv_offset),
                                replay_chunks) ||
        !position_at_token_boundary(snapshot->chunks,
                                    static_cast<size_t>(rollback_pos),
                                    effective_position_base)) {
        LLAMA_LOG_ERROR(
            "%s: unsafe replay range [%d..%d) for seq_id=%d\n",
            __func__, rollback_pos, kv_offset, seq_id);
        state_cache_.invalidate_snapshot(seq_id, true);
        return false;
    }
    LLAMA_LOG_DEBUG(
        "%s: replay positions seq_id=%d physical=[%d..%d) "
        "logical_base=%d replay_tokens=%zu decode_pos_offset_before=%d\n",
        __func__, seq_id, rollback_pos, kv_offset, effective_position_base,
        chunk_token_count(replay_chunks), decode_position_offset_[seq_id]);

    replay_chunks.insert(replay_chunks.end(), pending_chunks.begin(),
                         pending_chunks.end());
    effective_chunks = std::move(replay_chunks);
    effective_kv_offset = rollback_pos;
    if (memory_) {
        memory_->seq_rm(seq_id, rollback_pos, kv_offset);
    }
    state_cache_.mark_initialized(seq_id, best_ckpt > 0);
    LLAMA_LOG_INFO(
        "%s: rollback/replay seq_id=%d state_pos=%d kv=%d->%d "
        "replay=%d input=%zu source=%s rope_base=%d\n",
        __func__, seq_id, snapshot->last_processed_pos, kv_offset,
        rollback_pos, kv_offset - rollback_pos,
        chunk_token_count(pending_chunks),
        best_ckpt > 0 ? "checkpoint" : "zero",
        effective_position_base);

    snapshot->valid = false;
    if (best_ckpt <= 0) {
        state_cache_.clear_all_checkpoints(seq_id);
    }
    return true;
}

void HoumoQwen35LLM::export_prefill_hidden(
    int32_t kv_offset, int32_t valid_length, int32_t current_length) {
    if (pre_norm_hidden_dst_ == nullptr || pre_norm_hidden_dst_rows_ <= 0) {
        return;
    }

    const auto hidden_it = std::find_if(
        prefill_output_names_.begin(), prefill_output_names_.end(),
        [](const std::string &name) {
            return name.find("hidden") != std::string::npos;
        });
    if (hidden_it == prefill_output_names_.end()) {
        LLAMA_LOG_WARN("%s: no hidden output found in prefill model for MTP export\n",
                       __func__);
        return;
    }

    auto hidden = prefill_model_->GetOutput(*hidden_it);
    if (hidden.Info().DataType() != tcim::DataType::FLOAT16) {
        LLAMA_LOG_WARN(
            "%s: unexpected prefill hidden dtype=%d for MTP export, "
            "expected FLOAT16\n",
            __func__, static_cast<int>(hidden.Info().DataType()));
        return;
    }

    const int copy_begin = std::max(0, kv_offset - valid_length);
    const int dst_begin = valid_length + copy_begin - kv_offset;
    const int n_copy = std::min(current_length - copy_begin,
                                pre_norm_hidden_dst_rows_ - dst_begin);
    if (dst_begin < 0 || dst_begin >= pre_norm_hidden_dst_rows_ || n_copy <= 0) {
        return;
    }

    const size_t row_bytes = static_cast<size_t>(n_embd_) * sizeof(ggml_fp16_t);
    const auto *src = static_cast<const uint8_t *>(hidden.Buffer().Data());
    auto *dst = reinterpret_cast<uint8_t *>(pre_norm_hidden_dst_);
    for (int row = 0; row < n_copy; ++row) {
        std::memcpy(dst + static_cast<size_t>(dst_begin + row) * row_bytes,
                    src + static_cast<size_t>(copy_begin + row) * row_bytes,
                    row_bytes);
    }
}

void HoumoQwen35LLM::maybe_save_prefill_checkpoint(
    int seq_id, size_t round, size_t rounds, int32_t valid_length,
    bool is_prompt_final_prefill) {
    if (valid_length <= 0) {
        return;
    }

    // llama.cpp can split a long prompt into multiple batches (for example,
    // 2048 tokens each). Only force the penultimate model round for the batch
    // that actually requests logits, which marks the completed prompt prefill.
    // Intermediate batches continue to use checkpoint_min_interval_.
    const bool is_second_last = round + 2 == rounds;
    const bool force = is_prompt_final_prefill && is_second_last;
    if (state_cache_.should_save_checkpoint(seq_id, valid_length,
                                            force)) {
        state_cache_.save_state_checkpoint(seq_id, valid_length);
    }
}

int HoumoQwen35LLM::copy_prefill_logits(float *logits,
                                        int32_t current_length) {
    if (logits == nullptr) {
        return 0;
    }

    PERF_START(PREFILL_GETOUTPUT);
    auto output = prefill_model_->GetOutput(prefill_output_names_[0]);
    const auto output_info = output.Info();
    PERF_STOP(PREFILL_GETOUTPUT);

    PERF_START(PREFILL_CAST_OUTPUT);
    if (output_info.DataType() != tcim::DataType::FLOAT16) {
        PERF_STOP(PREFILL_CAST_OUTPUT);
        LLAMA_LOG_ERROR("%s: unsupported prefill output dtype=%d\n", __func__,
                        static_cast<int>(output_info.DataType()));
        return -1;
    }
    const size_t row_elems = static_cast<size_t>(n_vocab_);
    const size_t n_rows = output.MemSize() / sizeof(ggml_fp16_t) / row_elems;
    const size_t row_idx = std::min<size_t>(
        n_rows - 1,
        current_length > 0 ? static_cast<size_t>(current_length - 1) : 0);
    const auto *src =
        static_cast<const ggml_fp16_t *>(output.Buffer().Data()) +
        row_idx * row_elems;
    ggml_fp16_to_fp32_row(src, logits, static_cast<int64_t>(row_elems));
    PERF_STOP(PREFILL_CAST_OUTPUT);
    return 0;
}

bool HoumoQwen35LLM::update_text_prefill_snapshot(
    int seq_id,
    int32_t effective_kv_offset,
    int32_t valid_length,
    const std::vector<PrefillChunk> &effective_chunks) {
    auto &snapshot = state_cache_.snapshot(seq_id);
    const size_t history_size = chunk_token_count(snapshot.chunks);
    if (history_size < static_cast<size_t>(effective_kv_offset)) {
        LLAMA_LOG_WARN(
            "%s: snapshot chunk history (%zu) < effective_kv_offset (%d) "
            "for seq_id=%d\n",
            __func__, history_size, effective_kv_offset, seq_id);
    }
    // Checkpoints for the newly processed suffix were saved before this
    // snapshot update. Only rewrite history here; releasing checkpoints after
    // effective_kv_offset would immediately discard those fresh checkpoints.
    if (!state_cache_.trim_snapshot_to_kv_offset(
            seq_id, effective_kv_offset, /*release_checkpoints=*/false)) {
        return false;
    }
    for (const auto &chunk : effective_chunks) {
        snapshot.chunks.push_back(chunk);
    }
    snapshot.last_processed_pos = valid_length;
    snapshot.valid = true;
    return true;
}

int HoumoQwen35LLM::houmo_prefill(std::vector<llama_token> &tokens, int seq_id,
                                 float *logits) {
    if (has_vision_chunk(seq_id)) {
        return houmo_prefill_vision(tokens, seq_id, logits);
    }

    const int32_t kv_offset = memory_ ? memory_->seq_pos_min(seq_id) : 0;
    // Pending vision chunks are normally caught above. A text-only request
    // can still need to replay already committed vision chunks from the state
    // snapshot after KV prefix reuse. Route that case through the chunk-aware
    // M-RoPE executor as well.
    if (state_cache_.enabled()) {
        const auto *snapshot = state_cache_.find_snapshot(seq_id);
        const bool state_needs_replay =
            snapshot != nullptr && snapshot->valid &&
            snapshot->last_processed_pos != kv_offset;
        const bool snapshot_has_vision =
            state_needs_replay &&
            std::any_of(snapshot->chunks.begin(), snapshot->chunks.end(),
                        [](const PrefillChunk &chunk) {
                            return chunk.is_embedding;
                        });
        if (snapshot_has_vision) {
            return houmo_prefill_vision(tokens, seq_id, logits);
        }
    }

    if (kv_offset == 0) {
        decode_position_offset_[seq_id] = 0;
    }

    std::vector<PrefillChunk> pending_chunks;
    append_text_chunk(pending_chunks, tokens);
    std::vector<PrefillChunk> effective_chunks;
    int32_t effective_kv_offset = 0;
    int32_t effective_position_base = 0;
    if (!prepare_prefill_replay(seq_id, kv_offset, pending_chunks,
                                /*require_snapshot=*/false,
                                effective_chunks, effective_kv_offset,
                                effective_position_base)) {
        LLAMA_LOG_ERROR("%s: failed to prepare replay for seq_id=%d\n",
                        __func__, seq_id);
        return -1;
    }

    if (state_cache_.enabled()) {
        if (!state_cache_.trim_snapshot_to_kv_offset(
                seq_id, effective_kv_offset)) {
            LLAMA_LOG_ERROR(
                "%s: unsafe snapshot boundary %d for seq_id=%d\n",
                __func__, effective_kv_offset, seq_id);
            return -1;
        }
        if (effective_kv_offset == 0) {
            state_cache_.mark_initialized(seq_id, false);
        }
    }

    const size_t process_tokens = chunk_token_count(effective_chunks);
    if ((process_tokens + effective_kv_offset) >=
        static_cast<uint32_t>(context_length_)) {
        LLAMA_LOG_ERROR(
            "%s: input size %zu + offset %d >= context_length_ %d\n",
            __func__, process_tokens, effective_kv_offset,
            context_length_);
        return 1;
    }

    if (process_tokens == 0) {
        auto &snapshot = state_cache_.snapshot(seq_id);
        if (!state_cache_.trim_snapshot_to_kv_offset(
                seq_id, effective_kv_offset)) {
            return -1;
        }
        snapshot.last_processed_pos = effective_kv_offset;
        snapshot.valid = true;
        decode_valid_length_[seq_id] = effective_kv_offset;
        return 0;
    }

    int32_t current_length = 0;
    int32_t valid_length = effective_kv_offset;
    const size_t rounds =
        (process_tokens + prefill_length_ - 1) / prefill_length_;
    auto &input_data = prefill_input_map_.at(prefill_model_->GetInputName(0));
    for (size_t round = 0; round < rounds; ++round) {
        const size_t round_start = round * prefill_length_;
        current_length = static_cast<int>(std::min<size_t>(
            prefill_length_, process_tokens - round_start));

        memset(input_data.Data(), 0, input_data.MemSize());
        size_t stream_pos = 0;
        size_t copied = 0;
        auto *round_dst = static_cast<uint8_t *>(input_data.Data());
        for (const auto &chunk : effective_chunks) {
            const size_t chunk_start = stream_pos;
            const size_t chunk_end = stream_pos + chunk.size();
            stream_pos = chunk_end;
            const size_t copy_start = std::max(round_start, chunk_start);
            const size_t copy_end = std::min(round_start +
                                                static_cast<size_t>(current_length),
                                            chunk_end);
            if (copy_start >= copy_end) {
                continue;
            }
            const size_t chunk_offset = copy_start - chunk_start;
            const size_t round_offset = copy_start - round_start;
            const size_t count = copy_end - copy_start;
            auto *dst = round_dst + round_offset *
                                    static_cast<size_t>(n_embd_) *
                                    sizeof(uint16_t);
            if (chunk.is_embedding) {
                const size_t elem_offset = chunk_offset * static_cast<size_t>(n_embd_);
                const size_t elem_count = count * static_cast<size_t>(n_embd_);
                if (elem_offset + elem_count > chunk.embeddings.size()) {
                    LLAMA_LOG_ERROR(
                        "%s: embedding history is incomplete seq_id=%d "
                        "offset=%zu count=%zu size=%zu\n",
                        __func__, seq_id, elem_offset, elem_count,
                        chunk.embeddings.size());
                    return -1;
                }
                std::memcpy(dst, chunk.embeddings.data() + elem_offset,
                            elem_count * sizeof(uint16_t));
            } else {
                if (chunk_offset + count > chunk.tokens.size()) {
                    LLAMA_LOG_ERROR(
                        "%s: token history is incomplete seq_id=%d "
                        "offset=%zu count=%zu size=%zu\n",
                        __func__, seq_id, chunk_offset, count,
                        chunk.tokens.size());
                    return -1;
                }
                embedding_layer_->get_embedding_batch(
                    chunk.tokens.data() + chunk_offset, count, dst);
            }
            copied += count;
        }
        if (copied != static_cast<size_t>(current_length)) {
            LLAMA_LOG_ERROR("%s: mixed history copied=%zu expected=%d seq_id=%d\n",
                            __func__, copied, current_length, seq_id);
            return -1;
        }

        const int prefill_position_base =
            effective_position_base + (valid_length - effective_kv_offset);
        auto position_ids = build_rope_position_ids(prefill_position_base, prefill_length_);
        if (run_prefill_round(seq_id, input_data, valid_length, current_length,
                              position_ids.data(), position_ids.data(),
                              position_ids.data(),
                              /*is_first_round=*/round == 0) != 0) {
            return -1;
        }

        export_prefill_hidden(kv_offset, valid_length, current_length);
        valid_length += current_length;
        maybe_save_prefill_checkpoint(seq_id, round, rounds, valid_length,
                                      logits != nullptr);
    }

    decode_valid_length_[seq_id] = valid_length;
    if (copy_prefill_logits(logits, current_length) != 0) {
        return -1;
    }

    if (memory_) {
        memory_->seq_add(seq_id, 0, 0, static_cast<int32_t>(process_tokens));
    }

    if (!update_text_prefill_snapshot(seq_id, effective_kv_offset,
                                      valid_length, effective_chunks)) {
        return -1;
    }

    return 0;
}

int HoumoQwen35LLM::houmo_prefill(const float *embeddings, int n_tokens,
                                 int seq_id, float *logits) {
    (void)logits;
    if (n_tokens <= 0) {
        return 0;
    }

    VisionCompactHeader header;
    int embedding_offset = 0;
    int16_t nx = 0, ny = 0;
    std::memcpy(&header, embeddings, sizeof(VisionCompactHeader));
    if (header.magic == 0x1234) {
        ny = header.ny;
        nx = header.nx;
        embedding_offset = sizeof(VisionCompactHeader);
    }

    const int8_t *ptr = reinterpret_cast<const int8_t *>(embeddings);
    const size_t data_elems = static_cast<size_t>(n_tokens) * n_embd_;
    const size_t data_bytes = data_elems * sizeof(uint16_t);

    PrefillChunk chunk;
    chunk.is_embedding = true;
    chunk.n_tokens = static_cast<size_t>(n_tokens);
    chunk.nx = nx;
    chunk.ny = ny;
    chunk.embeddings.resize(data_elems);
    std::memcpy(chunk.embeddings.data(), ptr + embedding_offset, data_bytes);
    chunk_history_[seq_id].push_back(std::move(chunk));

    return 0;
}

void HoumoQwen35LLM::clear_vision_cache(int seq_id) {
    chunk_history_[seq_id].clear();
}

void HoumoQwen35LLM::compute_vision_position_ids(
    int seq_id, int kv_offset,
    const std::vector<llama_token> & /*trailing_tokens*/,
    std::vector<int32_t> &time_positions,
    std::vector<int32_t> &height_positions,
    std::vector<int32_t> &width_positions,
    int32_t &final_position) {
    // Build M-RoPE position IDs matching Python get_rope_index:
    //   - text chunk: all 3 dims = start_pos + i
    //   - vision chunk (nx*ny tokens): t = start_pos,
    //       h = start_pos + y, w = start_pos + x  (y in [0,ny), x in [0,nx))
    //       then start_pos advances by max(nx, ny)
    auto &chunks = chunk_history_[seq_id];
    // Continue from the accumulated M-RoPE position. This matters when a new
    // vision segment is prefetched after an earlier vision segment was
    // already committed without requesting logits.
    int32_t pos = static_cast<int32_t>(kv_offset) +
                  decode_position_offset_[seq_id];
    for (size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
        const auto &chunk = chunks[chunk_idx];
        const int nx = chunk.nx;
        const int ny = chunk.ny;
        if (chunk.is_embedding && nx > 0 && ny > 0) {
            // Vision tokens: grid layout height=ny, width=nx
            for (int y = 0; y < ny; y++) {
                for (int x = 0; x < nx; x++) {
                    time_positions.push_back(pos);
                    height_positions.push_back(pos + y);
                    width_positions.push_back(pos + x);
                }
            }
            pos += std::max(nx, ny);
        } else {
            // Text tokens
            for (size_t i = 0; i < chunk.n_tokens; i++) {
                time_positions.push_back(pos + i);
                height_positions.push_back(pos + i);
                width_positions.push_back(pos + i);
            }
            pos += static_cast<int32_t>(chunk.n_tokens);
        }
    }
    final_position = pos;
}

int HoumoQwen35LLM::houmo_prefill_vision(std::vector<llama_token> &tokens,
                                          int seq_id, float *logits) {
    if (!tokens.empty()) {
        PrefillChunk chunk;
        chunk.n_tokens = tokens.size();
        chunk.tokens = tokens;
        chunk.embeddings.resize(tokens.size() * n_embd_);
        embedding_layer_->get_embedding_batch(
            tokens.data(), tokens.size(), chunk.embeddings.data());
        chunk_history_[seq_id].push_back(std::move(chunk));
    }

    const int32_t kv_offset = memory_ ? memory_->seq_pos_min(seq_id) : 0;
    if (kv_offset == 0) {
        decode_position_offset_[seq_id] = 0;
    }
    const int32_t old_decode_position_offset = decode_position_offset_[seq_id];

    int32_t effective_kv_offset = 0;
    int32_t effective_position_base = 0;
    std::vector<PrefillChunk> effective_chunks;
    if (!prepare_prefill_replay(
            seq_id, kv_offset, chunk_history_[seq_id],
            /*require_snapshot=*/kv_offset > 0,
            effective_chunks, effective_kv_offset,
            effective_position_base)) {
        clear_vision_cache(seq_id);
        return -1;
    }
    chunk_history_[seq_id] = std::move(effective_chunks);

    size_t total_tokens = 0;
    size_t last_vision_end_local = 0;
    for (const auto &chunk : chunk_history_[seq_id]) {
        if (chunk.is_embedding) {
            const size_t grid_tokens =
                static_cast<size_t>(std::max<int>(0, chunk.nx)) *
                static_cast<size_t>(std::max<int>(0, chunk.ny));
            if (chunk.nx <= 0 || chunk.ny <= 0 ||
                grid_tokens != chunk.n_tokens) {
                LLAMA_LOG_ERROR(
                    "%s: invalid replay vision chunk nx=%d ny=%d "
                    "n_tokens=%zu seq_id=%d\n",
                    __func__, chunk.nx, chunk.ny, chunk.n_tokens, seq_id);
                clear_vision_cache(seq_id);
                return -1;
            }
        } else if (chunk.tokens.size() < chunk.n_tokens) {
            LLAMA_LOG_ERROR(
                "%s: replay text chunk has %zu tokens, expected %zu "
                "for seq_id=%d\n",
                __func__, chunk.tokens.size(), chunk.n_tokens, seq_id);
            clear_vision_cache(seq_id);
            return -1;
        }
        total_tokens += chunk.n_tokens;
        if (chunk.is_embedding) {
            last_vision_end_local = total_tokens;
        }
    }

    if ((total_tokens + effective_kv_offset) >=
        static_cast<size_t>(context_length_)) {
        LLAMA_LOG_ERROR(
            "%s: total %zu + effective_kv_offset %d >= context_length %d\n",
            __func__, total_tokens, effective_kv_offset, context_length_);
        clear_vision_cache(seq_id);
        return 1;
    }

    decode_position_offset_[seq_id] =
        effective_position_base - effective_kv_offset;

    std::vector<int32_t> time_pos_ids, height_pos_ids, width_pos_ids;
    int32_t final_pos = 0;
    compute_vision_position_ids(seq_id, effective_kv_offset, tokens,
                                time_pos_ids, height_pos_ids, width_pos_ids,
                                final_pos);

    const int32_t new_decode_position_offset =
        final_pos - static_cast<int32_t>(total_tokens) - effective_kv_offset;
    decode_position_offset_[seq_id] = old_decode_position_offset;

    // Pad position IDs.  We use a variable-step round loop below so the
    // start of any round is bounded by total_tokens, and each slice reads
    // prefill_length_ entries.  Pad enough to cover the worst case where
    // the last round started just before total_tokens.
    size_t padded_size = total_tokens + prefill_length_;
    time_pos_ids.resize(padded_size, time_pos_ids.empty() ? 0 : time_pos_ids.back());
    height_pos_ids.resize(padded_size, height_pos_ids.empty() ? 0 : height_pos_ids.back());
    width_pos_ids.resize(padded_size, width_pos_ids.empty() ? 0 : width_pos_ids.back());

    int32_t valid_length = effective_kv_offset;
    int32_t current_length = 0;
    const int32_t safe_pos_abs =
        effective_kv_offset + static_cast<int32_t>(last_vision_end_local);
    bool safe_ckpt_saved = false;

    auto &input_data = prefill_input_map_.at(prefill_model_->GetInputName(0));
    const auto &chunks = chunk_history_[seq_id];
    size_t chunk_idx = 0;
    size_t chunk_token_offset = 0;

    // Split at the end of vision data so post-vision state can be checkpointed.
    size_t cursor = 0;
    size_t round_idx = 0;
    while (cursor < total_tokens) {
        size_t step = std::min<size_t>(prefill_length_, total_tokens - cursor);
        if (last_vision_end_local > 0 &&
            cursor < last_vision_end_local &&
            cursor + step > last_vision_end_local) {
            step = last_vision_end_local - cursor;
        }
        size_t round_start = cursor;
        current_length = static_cast<int>(step);
        const bool is_first_round = (round_idx == 0);

        memset(input_data.Data(), 0, input_data.MemSize());
        size_t copied_tokens = 0;
        auto *round_dst = static_cast<uint8_t *>(input_data.Data());
        while (copied_tokens < step) {
            while (chunk_idx < chunks.size() &&
                   chunk_token_offset >= chunks[chunk_idx].n_tokens) {
                ++chunk_idx;
                chunk_token_offset = 0;
            }
            if (chunk_idx >= chunks.size()) {
                LLAMA_LOG_ERROR(
                    "%s: vision chunk stream ended after %zu/%zu tokens "
                    "for seq_id=%d\n",
                    __func__, cursor + copied_tokens, total_tokens, seq_id);
                clear_vision_cache(seq_id);
                return -1;
            }

            const auto &chunk = chunks[chunk_idx];
            const size_t available_tokens =
                chunk.n_tokens - chunk_token_offset;
            const size_t copy_tokens =
                std::min(available_tokens, step - copied_tokens);
            const size_t dst_elem = copied_tokens *
                                    static_cast<size_t>(n_embd_);
            auto *dst = round_dst + dst_elem * sizeof(uint16_t);
            if (chunk.is_embedding) {
                const size_t required_elems = chunk.n_tokens *
                                              static_cast<size_t>(n_embd_);
                if (chunk.embeddings.size() < required_elems) {
                    LLAMA_LOG_ERROR(
                        "%s: chunk %zu embedding data too short: have %zu "
                        "elements, need %zu for seq_id=%d\n",
                        __func__, chunk_idx, chunk.embeddings.size(),
                        required_elems, seq_id);
                    clear_vision_cache(seq_id);
                    return -1;
                }
                const size_t src_elem = chunk_token_offset *
                                        static_cast<size_t>(n_embd_);
                const size_t copy_bytes = copy_tokens *
                                          static_cast<size_t>(n_embd_) *
                                          sizeof(uint16_t);
                std::memcpy(dst, chunk.embeddings.data() + src_elem,
                            copy_bytes);
            } else {
                embedding_layer_->get_embedding_batch(
                    chunk.tokens.data() + chunk_token_offset,
                    copy_tokens, dst);
            }

            copied_tokens += copy_tokens;
            chunk_token_offset += copy_tokens;
        }

        if (run_prefill_round(seq_id, input_data, valid_length, current_length,
                              time_pos_ids.data() + round_start,
                              height_pos_ids.data() + round_start,
                              width_pos_ids.data() + round_start,
                              is_first_round) != 0) {
            clear_vision_cache(seq_id);
            return -1;
        }

        if (abort_callback_ != nullptr && abort_callback_(abort_callback_data_)) {
            clear_vision_cache(seq_id);
            return 2;
        }

        valid_length += current_length;
        cursor += step;
        ++round_idx;

        if (!safe_ckpt_saved && last_vision_end_local > 0 &&
            valid_length >= safe_pos_abs) {
            state_cache_.save_state_checkpoint(seq_id, valid_length);
            safe_ckpt_saved = true;
        }
    }

    decode_valid_length_[seq_id] =
        effective_kv_offset + static_cast<int32_t>(total_tokens);

    if (logits != nullptr) {
        PERF_START(PREFILL_GETOUTPUT);
        tcim::Tensor prefill_output_data =
            prefill_model_->GetOutput(prefill_output_names_[0]);
        const auto out_info = prefill_output_data.Info();
        PERF_STOP(PREFILL_GETOUTPUT);

        PERF_START(PREFILL_CAST_OUTPUT);
        if (out_info.DataType() != tcim::DataType::FLOAT16) {
            PERF_STOP(PREFILL_CAST_OUTPUT);
            LLAMA_LOG_ERROR("%s: unsupported prefill output dtype=%d\n", __func__,
                            static_cast<int>(out_info.DataType()));
            clear_vision_cache(seq_id);
            return -1;
        }
        const size_t row_elems = static_cast<size_t>(n_vocab_);
        const size_t n_rows = prefill_output_data.MemSize() /
                    sizeof(ggml_fp16_t) / row_elems;
        const size_t row_idx = std::min<size_t>(
            n_rows - 1,
            current_length > 0 ? static_cast<size_t>(current_length - 1) : 0);
        auto *src = static_cast<const ggml_fp16_t *>(prefill_output_data.Buffer().Data()) +
                    row_idx * row_elems;
        ggml_fp16_to_fp32_row(src, logits, static_cast<int64_t>(row_elems));
        PERF_STOP(PREFILL_CAST_OUTPUT);
    }

    if (memory_) {
        memory_->seq_add(seq_id, 0, 0, static_cast<int32_t>(total_tokens));
    }

    // Update snapshot. Vision chunks are stored directly in mixed chunk
    // history, so future text prefills can replay text + embedding chunks.
    {
        auto &snapshot = state_cache_.snapshot(seq_id);
        if (effective_kv_offset > 0) {
            if (chunk_token_count(snapshot.chunks) <
                static_cast<size_t>(effective_kv_offset)) {
                LLAMA_LOG_WARN(
                    "%s: existing snapshot chunk history (%zu) shorter than "
                    "effective_kv_offset (%d) for seq_id=%d\n",
                    __func__, chunk_token_count(snapshot.chunks),
                    effective_kv_offset, seq_id);
            }
            if (!state_cache_.trim_snapshot_to_kv_offset(
                    seq_id, effective_kv_offset,
                    /*release_checkpoints=*/false)) {
                clear_vision_cache(seq_id);
                return -1;
            }
        } else {
            snapshot.chunks.clear();
        }
        for (auto &chunk : chunk_history_[seq_id]) {
            snapshot.chunks.push_back(chunk);
        }
        if (valid_length > 0 && (!safe_ckpt_saved || valid_length == safe_pos_abs)) {
            state_cache_.save_state_checkpoint(seq_id, valid_length);
        }
        snapshot.last_processed_pos = valid_length;
        snapshot.valid = true;
    }

    decode_position_offset_[seq_id] = new_decode_position_offset;
    LLAMA_LOG_DEBUG(
        "%s: commit position seq_id=%d valid_length=%d "
        "decode_pos_offset=%d decode_position=%d\n",
        __func__, seq_id, valid_length, decode_position_offset_[seq_id],
        valid_length + decode_position_offset_[seq_id]);
    clear_vision_cache(seq_id);
    return 0;
}

int HoumoQwen35LLM::set_decode_rope_and_mask(const std::vector<int> &seq_ids) {
    const int n_slots = std::max(1, state_cache_.decode_verify_seq());
    for (size_t i = 0; i < seq_ids.size(); i++) {
        int seq_id = seq_ids[i];
        int batch_id = seq_id;
        if (n_decode_batch_ == 1) {
            batch_id = 0;
        }
        int valid_length = decode_valid_length_[seq_id];
        int position = valid_length + decode_position_offset_[seq_id];
        auto position_ids = build_rope_position_ids(position, n_slots);
        auto time_name = resolve_batch_name(decode_time_pos_names_, batch_id);
        auto height_name = resolve_batch_name(decode_height_pos_names_, batch_id);
        auto width_name = resolve_batch_name(decode_width_pos_names_, batch_id);
        auto mask_name = resolve_batch_name(decode_linear_mask_names_, batch_id);

        if (!time_name.empty()) {
            auto tensor = decode_input_map_.at(time_name);
            set_input_data(tensor, position_ids);
            decode_model_->SetInput(time_name, tensor);
        }
        if (!height_name.empty()) {
            auto tensor = decode_input_map_.at(height_name);
            set_input_data(tensor, position_ids);
            decode_model_->SetInput(height_name, tensor);
        }
        if (!width_name.empty()) {
            auto tensor = decode_input_map_.at(width_name);
            set_input_data(tensor, position_ids);
            decode_model_->SetInput(width_name, tensor);
        }
        if (!mask_name.empty()) {
            auto tensor = decode_input_map_.at(mask_name);
            set_linear_attn_mask(tensor, n_slots, n_slots);
            decode_model_->SetInput(mask_name, tensor);
        }
    }
    return 0;
}

int HoumoQwen35LLM::houmo_decode(std::vector<llama_token> &batches,
                                std::vector<int> &seq_ids,
                                std::vector<float *> logits) {
    for (auto seq_id : seq_ids) {
        if (memory_) {
            decode_valid_length_[seq_id] = memory_->seq_pos_min(seq_id);
        }
    }

    // Prefill repairs state gaps before single-token decode.
    if (state_cache_.enabled()) {
        for (size_t i = 0; i < seq_ids.size(); ++i) {
            int seq_id = seq_ids[i];
            int target_vl = decode_valid_length_[seq_id];

            int state_pos = state_cache_.state_position(seq_id);

            if (state_pos >= 0 && state_pos != target_vl) {
                std::vector<llama_token> token_vec = {batches[i]};
                int ret = houmo_prefill(token_vec, seq_id, logits[i]);
                if (ret != 0) {
                    return ret;
                }
                if (seq_ids.size() == 1) {
                    return 0;
                }
                seq_ids.erase(seq_ids.begin() + static_cast<ptrdiff_t>(i));
                batches.erase(batches.begin() + static_cast<ptrdiff_t>(i));
                logits.erase(logits.begin() + static_cast<ptrdiff_t>(i));
                --i;
                continue;
            }

            if (target_vl == 0) {
                state_cache_.restore_zero_state(seq_id);
                state_cache_.mark_initialized(seq_id, true);
            }
        }
    }
    if (seq_ids.size() > static_cast<size_t>(n_decode_batch_)) {
        LLAMA_LOG_ERROR("%s: seq_ids.size() > n_decode_batch_ not supported\n",
                        __func__);
        return -1;
    }

    PERF_START(DECODE_SETINPUT);
    int ret = set_input_for_decode(batches, seq_ids);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: set_input_for_decode failed\n", __func__);
        return ret;
    }
    ret = set_xlenght_for_decode(seq_ids);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: set_xlenght_for_decode failed\n", __func__);
        return ret;
    }
    ret = set_decode_rope_and_mask(seq_ids);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: set_decode_rope_and_mask failed\n", __func__);
        return ret;
    }
    state_cache_.set_state_cache_for_decode(seq_ids, /*accept_pos=*/0);
    set_kv_cache_for_decode(seq_ids);

    PERF_STOP(DECODE_SETINPUT);

    PERF_START(DECODE_RUN);
    decode_model_->Run();
    decode_model_->Sync();
    PERF_STOP(DECODE_RUN);

    PERF_START(DECODE_GETOUTPUT);
    tcim::Tensor decode_data = decode_model_->GetOutput(decode_output_names_[0]);
    const auto out_dtype = decode_data.Info().DataType();
    void *decode_ptr = decode_data.Buffer().Data();
    PERF_STOP(DECODE_GETOUTPUT);
    PERF_START(DECODE_POSTPROCESS);
    for (size_t i = 0; i < seq_ids.size(); i++) {
        auto batch_id = seq_ids[i];
        decode_valid_length_[batch_id]++;
        if (n_decode_batch_ == 1) {
            batch_id = 0;
        }
        float *logits_out = logits[i];
        if (logits_out != nullptr) {
            if (out_dtype == tcim::DataType::FLOAT16) {
                auto *output_ptr =
                    static_cast<ggml_fp16_t *>(decode_ptr) +
                    batch_id * n_vocab_;
                ggml_fp16_to_fp32_row(output_ptr, logits_out,
                                      static_cast<int>(n_vocab_));
            } else {
                LLAMA_LOG_ERROR("%s: unsupported decode output dtype=%d\n", __func__,
                                static_cast<int>(out_dtype));
                return -1;
            }
        }
        if (memory_) {
            memory_->seq_add(seq_ids[i], 0, 0, 1);
        }

        {
            auto &snapshot = state_cache_.snapshot(seq_ids[i]);
            int vl = decode_valid_length_[seq_ids[i]];
            append_snapshot_text_history(seq_ids[i], &batches[i], 1);
            snapshot.last_processed_pos = vl;
            snapshot.valid = true;
        }
    }

    if (pre_norm_hidden_dst_ != nullptr && pre_norm_hidden_dst_rows_ > 0) {
        for (size_t oi = 0; oi < decode_output_names_.size(); ++oi) {
            if (decode_output_names_[oi].find("hidden") == std::string::npos) {
                continue;
            }
            auto h_dev = decode_model_->GetOutput(decode_output_names_[oi]);
            const auto h_info = h_dev.Info();
            if (h_info.DataType() != tcim::DataType::FLOAT16) {
                break;
            }
            const auto h_shape = h_info.Shape();
            const int h_embd = h_shape.empty() ? 0
                                               : static_cast<int>(h_shape.back());
            if (h_embd <= 0) {
                break;
            }
            const int n_rows = std::min(1, pre_norm_hidden_dst_rows_);
            const size_t row_bytes = static_cast<size_t>(h_embd) * sizeof(ggml_fp16_t);
            const auto *src = static_cast<const uint8_t *>(h_dev.Buffer().Data());
            for (int r = 0; r < n_rows; ++r) {
                std::memcpy(reinterpret_cast<uint8_t *>(pre_norm_hidden_dst_ + static_cast<size_t>(r) * h_embd),
                            src + static_cast<size_t>(r) * row_bytes,
                            row_bytes);
            }
            break;
        }
    }

    PERF_STOP(DECODE_POSTPROCESS);
    return 0;
}

int HoumoQwen35LLM::houmo_verify(std::vector<llama_token> &tokens, int seq_id,
                                 std::vector<float *> &logits_out) {
    PERF_START(HOUMO_VERIFY);
    if (tokens.size() != logits_out.size()) {
        LLAMA_LOG_ERROR("%s: tokens.size()=%zu != logits_out.size()=%zu\n",
                        __func__, tokens.size(), logits_out.size());
        return -1;
    }
    if (tokens.empty()) {
        return 0;
    }

    if (decode_model_ == nullptr || state_cache_.decode_verify_seq() <= 1) {
        LLAMA_LOG_WARN("%s: no verify-style decode model, fallback to prefill loop\n",
                       __func__);
        return HouMoLLModel::Impl::houmo_verify(tokens, seq_id, logits_out);
    }
    if (tokens.size() > static_cast<size_t>(state_cache_.decode_verify_seq())) {
        LLAMA_LOG_WARN(
            "%s: verify tokens=%zu exceed decode_verify_seq=%d. MTP draft "
            "length must be <= verify_seq-1 so sampled+draft fits one verify "
            "decode pass; fallback to prefill loop for this long batch.\n",
            __func__, tokens.size(), state_cache_.decode_verify_seq());
        return HouMoLLModel::Impl::houmo_verify(tokens, seq_id, logits_out);
    }

    const int32_t start_pos = memory_ ? memory_->seq_pos_min(seq_id)
                                      : decode_valid_length_[seq_id];

    PERF_START(VERIFY_SETINPUT);
    const int n_verify = static_cast<int>(tokens.size());
    const int n_slots = state_cache_.decode_verify_seq();
    const auto act_name = decode_model_->GetInputName(0);
    auto act_iter = decode_input_map_.find(act_name);
    if (act_iter == decode_input_map_.end()) {
        LLAMA_LOG_ERROR("%s: missing decode activation input %s\n",
                        __func__, act_name.c_str());
        return -1;
    }
    auto input_data = act_iter->second;
    zero_tensor(input_data);
    try {
        embedding_layer_->get_embedding_batch(tokens.data(), n_verify,
                                             input_data.Data());
    } catch (...) {
        LLAMA_LOG_ERROR("%s: failed to get verify embedding batch\n", __func__);
        return -1;
    }
    decode_model_->SetInput(act_name, input_data);

    auto set_scalar_by_names = [&](const std::vector<std::string> &names,
                                   int32_t value,
                                   const char *what) -> int {
        if (names.empty()) {
            LLAMA_LOG_ERROR("%s: missing decode %s input name\n", __func__, what);
            return -1;
        }
        const std::string name = resolve_batch_name(names, 0);
        auto tensor = decode_input_map_.at(name);
        set_input_data(tensor, 1, value);
        if (decode_model_->SetInput(name, tensor) != tcim::Status::OK) {
            LLAMA_LOG_ERROR("%s: failed to set %s input %s\n", __func__, what,
                            name.c_str());
            return -1;
        }
        return 0;
    };
    if (set_scalar_by_names(decode_valid_length_names_, start_pos,
                            "valid_length") != 0 ||
        set_scalar_by_names(decode_current_length_names_, n_verify,
                            "current_length") != 0) {
        return -1;
    }

    auto position_ids = build_rope_position_ids(
        start_pos + decode_position_offset_[seq_id], n_slots);
    auto set_pos = [&](const std::vector<std::string> &names,
                       const char *what) -> int {
        const std::string name = resolve_batch_name(names, 0);
        if (name.empty()) {
            return 0;
        }
        auto tensor = decode_input_map_.at(name);
        set_input_data(tensor, position_ids);
        if (decode_model_->SetInput(name, tensor) != tcim::Status::OK) {
            LLAMA_LOG_ERROR("%s: failed to set %s input %s\n", __func__, what,
                            name.c_str());
            return -1;
        }
        return 0;
    };
    if (set_pos(decode_time_pos_names_, "time_position_ids") != 0 ||
        set_pos(decode_height_pos_names_, "height_position_ids") != 0 ||
        set_pos(decode_width_pos_names_, "width_position_ids") != 0) {
        return -1;
    }
    const std::string mask_name = resolve_batch_name(decode_linear_mask_names_, 0);
    if (!mask_name.empty()) {
        auto tensor = decode_input_map_.at(mask_name);
        set_linear_attn_mask(tensor, n_slots, n_slots);
        decode_model_->SetInput(mask_name, tensor);
    }

    std::vector<int> seq_ids = {seq_id};
    state_cache_.set_state_cache_for_decode(seq_ids, /*accept_pos=*/0, /*bind_outputs=*/false);
    set_kv_cache_for_decode(seq_ids);
    PERF_STOP(VERIFY_SETINPUT);

    PERF_START(VERIFY_RUN);
    auto status = decode_model_->Run();
    if (status != tcim::Status::OK) {
        PERF_STOP(VERIFY_RUN);
        LLAMA_LOG_ERROR("%s: decode verify Run() failed, result=%d\n",
                        __func__, static_cast<int>(status));
        return -1;
    }
    decode_model_->Sync();
    PERF_STOP(VERIFY_RUN);

    PERF_START(VERIFY_GETOUTPUT);
    tcim::Tensor decode_data = decode_model_->GetOutput(decode_output_names_[0]);
    const auto out_dtype = decode_data.Info().DataType();
    void *decode_ptr = decode_data.Buffer().Data();
    PERF_STOP(VERIFY_GETOUTPUT);

    PERF_START(VERIFY_POSTPROCESS);
    if (out_dtype != tcim::DataType::FLOAT16) {
        LLAMA_LOG_ERROR("%s: unsupported verify output dtype=%d\n", __func__,
                        static_cast<int>(out_dtype));
        return -1;
    }
    for (int i = 0; i < n_verify; ++i) {
        if (logits_out[static_cast<size_t>(i)] == nullptr) {
            continue;
        }
        auto *output_ptr = static_cast<ggml_fp16_t *>(decode_ptr) +
                           static_cast<size_t>(i) * n_vocab_;
        ggml_fp16_to_fp32_row(output_ptr, logits_out[static_cast<size_t>(i)],
                              static_cast<int>(n_vocab_));
    }

    if (pre_norm_hidden_dst_ != nullptr && pre_norm_hidden_dst_rows_ > 0) {
        for (size_t oi = 0; oi < decode_output_names_.size(); ++oi) {
            if (decode_output_names_[oi].find("hidden") == std::string::npos) {
                continue;
            }
            auto h_dev = decode_model_->GetOutput(decode_output_names_[oi]);
            const auto h_info = h_dev.Info();
            if (h_info.DataType() != tcim::DataType::FLOAT16) {
                LLAMA_LOG_WARN("%s: hidden output '%s' dtype=%d, expected fp16\n",
                               __func__, decode_output_names_[oi].c_str(),
                               static_cast<int>(h_info.DataType()));
                break;
            }
            const auto h_shape = h_info.Shape();
            const int h_seq = (h_shape.size() >= 2)
                                  ? static_cast<int>(h_shape[h_shape.size() - 2])
                                  : 1;
            const int h_embd = h_shape.empty()
                                   ? 0
                                   : static_cast<int>(h_shape.back());
            const int n_rows = std::min({n_verify, h_seq,
                                         pre_norm_hidden_dst_rows_});
            const size_t row_bytes = static_cast<size_t>(h_embd) * sizeof(ggml_fp16_t);
            const auto *src = static_cast<const uint8_t *>(h_dev.Buffer().Data());
            for (int r = 0; r < n_rows; ++r) {
                std::memcpy(
                    reinterpret_cast<uint8_t *>(pre_norm_hidden_dst_ + static_cast<size_t>(r) * h_embd),
                    src + static_cast<size_t>(r) * row_bytes,
                    row_bytes);
            }
            break;
        }
    }

    auto &pending = pending_verify_[seq_id];
    pending.valid = true;
    pending.start_pos = start_pos;
    pending.n_verify = n_verify;
    pending.tokens = tokens;
    PERF_STOP(VERIFY_POSTPROCESS);
    PERF_STOP(HOUMO_VERIFY);
    return 0;
}

int HoumoQwen35LLM::houmo_verify_accept(int seq_id, uint16_t n_accepted) {
    return commit_verify_accept(seq_id, static_cast<int>(n_accepted) + 1);
}

int HoumoQwen35LLM::houmo_embedding(
    const std::map<int32_t, std::vector<llama_token>> &batches,
    std::map<int, float *> &embeddings) {
    for (auto &[seq_id, tokens] : batches) {
        if (tokens.empty()) {
            continue;
        }
        auto &embedding = embeddings[seq_id];
        if (embedding == nullptr) {
            LLAMA_LOG_ERROR("%s: null embedding pointer for seq_id %d\n", __func__,
                            seq_id);
            return -1;
        }
        int ret = houmo_prefill(const_cast<std::vector<llama_token> &>(tokens),
                                seq_id, embedding);
        if (ret != 0) {
            LLAMA_LOG_ERROR(
                "%s: houmo_prefill failed for seq_id %d with error code %d\n",
                __func__, seq_id, ret);
            return -1;
        }
    }
    return 0;
}

bool HoumoQwen35LLM::is_embedding() {
    return decode_model_ == nullptr;
}

