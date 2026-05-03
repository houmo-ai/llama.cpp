#include "houmo-qwen35llm.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include "ggml.h"
#include "llama-impl.h"
#include "time-perf.h"

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
    size_t elem_count = tensor.MemSize();
    switch (tensor.Info().DataType()) {
    case tcim::DataType::FLOAT16: {
        elem_count /= sizeof(ggml_fp16_t);
        std::vector<ggml_fp16_t> zeros(elem_count, ggml_fp32_to_fp16(0.0f));
        tensor.Buffer().CopyFromHost(zeros.data(), tensor.MemSize());
        break;
    }
    case tcim::DataType::FLOAT32: {
        elem_count /= sizeof(float);
        std::vector<float> zeros(elem_count, 0.0f);
        tensor.Buffer().CopyFromHost(zeros.data(), tensor.MemSize());
        break;
    }
    case tcim::DataType::INT16: {
        elem_count /= sizeof(int16_t);
        std::vector<int16_t> zeros(elem_count, 0);
        tensor.Buffer().CopyFromHost(zeros.data(), tensor.MemSize());
        break;
    }
    case tcim::DataType::INT32: {
        elem_count /= sizeof(int32_t);
        std::vector<int32_t> zeros(elem_count, 0);
        tensor.Buffer().CopyFromHost(zeros.data(), tensor.MemSize());
        break;
    }
    default:
        break;
    }
}

bool is_cache_input_name(const std::string &name) {
    return name.find("past_key_cache_") != std::string::npos ||
           name.find("_self_attn_vcache") != std::string::npos ||
           name.find("_self_attn_kcache") != std::string::npos ||
           name.find("past_conv_cache") != std::string::npos ||
           name.find("past_recurrent_state") != std::string::npos;
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
    resolve_input_names();
    init_state_cache(seq_max);

    if (const char *env = std::getenv("LLAMA_CHECKPOINT_INTERVAL")) {
        errno = 0;
        char *end = nullptr;
        long v = std::strtol(env, &end, 10);
        if (errno == 0 && end != env && *end == '\0' && v > 0 && v <= 4096) {
            checkpoint_min_interval_ = static_cast<int>(v);
            LLAMA_LOG_INFO("%s: set checkpoint_min_interval_ from env LLAMA_CHECKPOINT_INTERVAL=%d\n",
                           __func__, checkpoint_min_interval_);
        } else {
            LLAMA_LOG_WARN("%s: invalid LLAMA_CHECKPOINT_INTERVAL='%s', keep default=%d\n",
                           __func__, env, checkpoint_min_interval_);
        }
    }

}

void HoumoQwen35LLM::resolve_input_names() {
    prefill_time_pos_name_ = prefill_model_->GetInputName(1);
    prefill_height_pos_name_ = prefill_model_->GetInputName(2);
    prefill_width_pos_name_ = prefill_model_->GetInputName(3);
    prefill_valid_length_name_ = prefill_model_->GetInputName(4);
    prefill_current_length_name_ = prefill_model_->GetInputName(5);
    prefill_linear_mask_name_ = prefill_model_->GetInputName(6);

    decode_time_pos_names_.clear();
    decode_height_pos_names_.clear();
    decode_width_pos_names_.clear();
    decode_linear_mask_names_.clear();
    decode_valid_length_names_.clear();
    decode_current_length_names_.clear();

    decode_time_pos_names_.push_back(decode_model_->GetInputName(1));
    decode_height_pos_names_.push_back(decode_model_->GetInputName(2));
    decode_width_pos_names_.push_back(decode_model_->GetInputName(3));
    decode_valid_length_names_.push_back(decode_model_->GetInputName(4));
    decode_current_length_names_.push_back(decode_model_->GetInputName(5));
    decode_linear_mask_names_.push_back(decode_model_->GetInputName(6));
}


void HoumoQwen35LLM::init_state_cache(int seq_max) {
    state_cache_names_.clear();
    state_cache_output_names_.clear();
    state_cache_.clear();
    state_cache_initialized_.clear();
    prefill_snapshots_.clear();
    if (decode_model_ == nullptr) {
        return;
    }
    auto add_name = [&](const std::string &name, const std::string &output_name) {
        auto iter = std::find(state_cache_names_.begin(), state_cache_names_.end(), name);
        if (iter == state_cache_names_.end()) {
            state_cache_names_.push_back(name);
            state_cache_output_names_.push_back(output_name);
        }
    };
    for (size_t idx = 0; idx < decode_model_->GetInputNum(); ++idx) {
        auto input_name = decode_model_->GetInputName(idx);
        if (input_name.find("conv_cache") != std::string::npos ||
            input_name.find("recurrent_state") != std::string::npos) {
            std::string output_name;
            if (input_name.find("conv_cache") != std::string::npos) {
                output_name = input_name;
                auto pos = output_name.find("past_conv_cache_");
                if (pos != std::string::npos) {
                    output_name.replace(pos, std::strlen("past_conv_cache_"),
                                        "conv_cache_out_");
                }
            } else if (input_name.find("recurrent_state") != std::string::npos) {
                output_name = input_name;
                auto pos = output_name.find("past_recurrent_state_");
                if (pos != std::string::npos) {
                    output_name.replace(pos, std::strlen("past_recurrent_state_"),
                                        "recurrent_state_out_");
                }
            }
            add_name(input_name, output_name);
        }
    }
    if (state_cache_names_.empty()) {
        return;
    }
    // Build checkpoint_tensor_indices_: only conv_cache and recurrent_state
    // tensors need to be saved/restored in checkpoints.  KV cache tensors
    // are managed by the KV cache system and must be excluded.
    checkpoint_tensor_indices_.clear();
    for (size_t idx = 0; idx < state_cache_names_.size(); ++idx) {
        const auto &name = state_cache_names_[idx];
        if (name.find("conv_cache") != std::string::npos ||
            name.find("recurrent_state") != std::string::npos) {
            checkpoint_tensor_indices_.push_back(idx);
        }
    }
    LLAMA_LOG_INFO("%s: checkpoint_tensor_indices_ size=%zu (total state_cache_names_=%zu)\n",
                   __func__, checkpoint_tensor_indices_.size(), state_cache_names_.size());
    int size = (seq_max + n_prefill_batch_ - 1) / n_prefill_batch_;
    size = std::max(size, n_decode_batch_);
    for (int i = 0; i < size; ++i) {
        StateCacheEntry entry;
        entry.tensors.reserve(state_cache_names_.size());
        for (const auto &name : state_cache_names_) {
            auto cache = decode_model_->GetDevInput(name);
            if (i > 0) {
                cache = cache.Clone();
            }
            entry.tensors.push_back(cache);
        }
        state_cache_[i] = std::move(entry);
        state_cache_initialized_[i] = false;
    }
    // Create device-side zero-initialized tensors for fast state restoration.
    // These serve as the single source of truth for position-0 state,
    // avoiding redundant per-seq_id checkpoint 0 storage.
    zero_state_tensors_.clear();
    zero_state_tensors_.reserve(state_cache_names_.size());
    for (const auto &name : state_cache_names_) {
        auto cache = decode_model_->GetDevInput(name);
        auto dev_tensor = cache.Clone();
        zero_tensor(dev_tensor);
        zero_state_tensors_.push_back(dev_tensor);
    }
    // Zero-initialize all state cache entries so model I/O is connected.
    for (auto &[seq_id, entry] : state_cache_) {
        for (size_t idx = 0; idx < entry.tensors.size(); ++idx) {
            zero_state_tensors_[idx].CopyTo(entry.tensors[idx]);
        }
        state_cache_initialized_[seq_id] = true;
        auto &snap = prefill_snapshots_[seq_id];
        snap.last_processed_pos = 0;
        snap.valid = true;
        // Pre-allocate checkpoint pool once — slots are reused across all
        // save/restore operations to avoid heap fragmentation.
        init_checkpoint_pool(seq_id);
    }
}

void HoumoQwen35LLM::init_checkpoint_pool(int seq_id) {
    if (decode_model_ == nullptr || checkpoint_tensor_indices_.empty() ||
        checkpoint_max_count_ == 0) {
        return;
    }
    checkpoint_max_count_ = std::min(
        checkpoint_max_count_,
        static_cast<size_t>(n_context_length() / checkpoint_min_interval_));
    auto &snapshot = prefill_snapshots_[seq_id];
    snapshot.checkpoint_pool.clear();
    snapshot.checkpoint_index.clear();
    snapshot.checkpoint_pool.resize(checkpoint_max_count_);
    size_t allocated_slots = 0;
    bool alloc_failed = false;
    for (auto &slot : snapshot.checkpoint_pool) {
        slot.used = false;
        slot.position = -1;
        slot.state_tensors.clear();
        slot.state_tensors.reserve(checkpoint_tensor_indices_.size());
        for (size_t idx : checkpoint_tensor_indices_) {
            const auto &name = state_cache_names_[idx];
            auto src = decode_model_->GetDevInput(name);
            tcim::Tensor dst;
            if (checkpoint_storage_ == CheckpointStorage::kDevice) {
                // Independent device-side buffer of matching shape/dtype.
                dst = src.Clone();
            } else {
                // ToHost(true) yields a contiguous host copy of matching
                // shape — content is irrelevant, will be overwritten by
                // CopyTo on first save.
                dst = src.ToHost(true);
            }
            if (dst.GetInitStatus() != tcim::Status::OK) {
                LLAMA_LOG_WARN("%s: failed to pre-allocate checkpoint tensor "
                               "seq_id=%d name=%s storage=%s — stopping pool "
                               "growth at %zu slot(s)\n",
                               __func__, seq_id, name.c_str(),
                               checkpoint_storage_ == CheckpointStorage::kDevice
                                   ? "device" : "host",
                               allocated_slots);
                alloc_failed = true;
                break;
            }
            slot.state_tensors.push_back(std::move(dst));
        }
        if (alloc_failed) {
            // Drop the partially-filled current slot.
            slot.state_tensors.clear();
            break;
        }
        ++allocated_slots;
    }
    // Shrink pool to successfully allocated slots; keep checkpoint feature
    // enabled if at least one slot was allocated.
    if (allocated_slots < snapshot.checkpoint_pool.size()) {
        snapshot.checkpoint_pool.resize(allocated_slots);
    }
    LLAMA_LOG_INFO("%s: pre-allocated checkpoint pool seq_id=%d slots=%zu "
                   "tensors_per_slot=%zu storage=%s\n",
                   __func__, seq_id, snapshot.checkpoint_pool.size(),
                   checkpoint_tensor_indices_.size(),
                   checkpoint_storage_ == CheckpointStorage::kDevice
                       ? "device" : "host");
}

size_t HoumoQwen35LLM::acquire_checkpoint_slot(PrefillSnapshot &snapshot,
                                                int position) {
    // Prefer a free slot — no eviction needed.
    for (size_t i = 0; i < snapshot.checkpoint_pool.size(); ++i) {
        if (!snapshot.checkpoint_pool[i].used) {
            return i;
        }
    }
    // All slots in use — evict the in-use slot whose distance to its nearest
    // neighbor (by position) is smallest, but never the lowest or highest
    // checkpoint (preserve maximal rollback range).
    auto &index = snapshot.checkpoint_index;
    auto best_evict = index.end();
    int best_gap = INT_MAX;
    for (auto it = index.begin(); it != index.end(); ++it) {
        if (it == index.begin()) continue;
        auto next = std::next(it);
        if (next == index.end()) continue;
        auto prev = std::prev(it);
        int gap = std::min(it->first - prev->first, next->first - it->first);
        if (gap < best_gap) {
            best_gap = gap;
            best_evict = it;
        }
    }
    // Fallback: if pool size <= 2 (degenerate), evict the closest neighbor
    // of `position` to keep the spread balanced.
    if (best_evict == index.end()) {
        int best_dist = INT_MAX;
        for (auto it = index.begin(); it != index.end(); ++it) {
            int d = std::abs(it->first - position);
            if (d < best_dist) {
                best_dist = d;
                best_evict = it;
            }
        }
    }
    size_t slot_idx = best_evict->second;
    LLAMA_LOG_INFO("%s: evicting checkpoint pos=%d slot=%zu (pool full, "
                   "max=%zu)\n",
                   __func__, best_evict->first, slot_idx,
                   checkpoint_max_count_);
    snapshot.checkpoint_pool[slot_idx].used = false;
    index.erase(best_evict);
    return slot_idx;
}

void HoumoQwen35LLM::release_checkpoints_after(PrefillSnapshot &snapshot,
                                                int after_pos) {
    auto &index = snapshot.checkpoint_index;
    auto erase_begin = index.upper_bound(after_pos);
    if (erase_begin == index.end()) {
        return;
    }
    for (auto it = erase_begin; it != index.end(); ++it) {
        snapshot.checkpoint_pool[it->second].used = false;
    }
    index.erase(erase_begin, index.end());
}

void HoumoQwen35LLM::clear_all_checkpoints(PrefillSnapshot &snapshot) {
    for (auto &slot : snapshot.checkpoint_pool) {
        slot.used = false;
        slot.position = -1;
    }
    snapshot.checkpoint_index.clear();
}

void HoumoQwen35LLM::set_state_cache_for_prefill(int seq_id) {
    if (prefill_model_ == nullptr) {
        return;
    }
    auto iter = state_cache_.find(seq_id);
    if (iter == state_cache_.end()) {
        return;
    }
    auto &tensors = iter->second.tensors;
    // If not initialized, restore zero state directly from template.
    if (!state_cache_initialized_[seq_id]) {
        restore_zero_state(seq_id);
        state_cache_initialized_[seq_id] = true;
    }
    // Set inputs/outputs for the model - tensors are already in place
    for (size_t idx = 0; idx < tensors.size(); ++idx) {
        prefill_model_->SetDevInput(state_cache_names_[idx], tensors[idx]);
        if (idx < state_cache_output_names_.size() &&
            !state_cache_output_names_[idx].empty()) {
            prefill_model_->SetDevOutput(state_cache_output_names_[idx],
                                         tensors[idx]);
        }
    }
}

void HoumoQwen35LLM::set_state_cache_for_decode(
    const std::vector<int> &seq_ids) {
    if (decode_model_ == nullptr) {
        return;
    }
    for (auto seq_id : seq_ids) {
        auto iter = state_cache_.find(seq_id);
        if (iter == state_cache_.end()) {
            continue;
        }
        auto &tensors = iter->second.tensors;
        for (size_t idx = 0; idx < tensors.size(); ++idx) {
            decode_model_->SetDevInput(state_cache_names_[idx], tensors[idx]);
            if (idx < state_cache_output_names_.size() &&
                !state_cache_output_names_[idx].empty()) {
                decode_model_->SetDevOutput(state_cache_output_names_[idx],
                                            tensors[idx]);
            }
        }
    }
}

int HoumoQwen35LLM::find_common_prefix_length(
    const std::vector<llama_token> &saved,
    const std::vector<llama_token> &incoming) const {
    size_t min_len = std::min(saved.size(), incoming.size());
    for (size_t i = 0; i < min_len; ++i) {
        if (saved[i] != incoming[i]) {
            LLAMA_LOG_INFO("%s: common prefix length=%zu (saved[%zu]=%d vs incoming[%zu]=%d)\n",
                       __func__, i, i, saved[i], i, incoming[i]);
            return static_cast<int>(i);
        }
    }
    return static_cast<int>(min_len);
}

void HoumoQwen35LLM::save_state_checkpoint(int seq_id, int position) {
    if (prefill_model_ == nullptr) {
        return;
    }
    auto cache_iter = state_cache_.find(seq_id);
    if (cache_iter == state_cache_.end()) {
        return;
    }
    auto snap_iter = prefill_snapshots_.find(seq_id);
    if (snap_iter == prefill_snapshots_.end()) {
        return;
    }
    auto &snapshot = snap_iter->second;
    if (snapshot.checkpoint_pool.empty()) {
        // Pool not pre-allocated (degenerate config) — nothing to do.
        return;
    }
    auto &tensors = cache_iter->second.tensors;

    // Erase stale checkpoints AFTER position first so those slots become
    // candidates for reuse if the pool is full.
    release_checkpoints_after(snapshot, position);

    // If a checkpoint already exists exactly at this position, reuse its slot.
    size_t slot_idx;
    auto existing = snapshot.checkpoint_index.find(position);
    if (existing != snapshot.checkpoint_index.end()) {
        slot_idx = existing->second;
    } else {
        slot_idx = acquire_checkpoint_slot(snapshot, position);
        snapshot.checkpoint_index[position] = slot_idx;
    }

    auto &slot = snapshot.checkpoint_pool[slot_idx];
    slot.position = position;
    slot.used = true;

    // Pre-allocated buffers — only memcpy, no alloc/free.
    for (size_t ci = 0; ci < checkpoint_tensor_indices_.size(); ++ci) {
        size_t ti = checkpoint_tensor_indices_[ci];
        tensors[ti].CopyTo(slot.state_tensors[ci]);
    }

    LLAMA_LOG_INFO("%s: saved checkpoint seq_id=%d pos=%d slot=%zu "
                   "state_tensors=%zu storage=%s in_use=%zu/%zu\n",
                   __func__, seq_id, position, slot_idx,
                   slot.state_tensors.size(),
                   checkpoint_storage_ == CheckpointStorage::kDevice
                       ? "device" : "host",
                   snapshot.checkpoint_index.size(),
                   snapshot.checkpoint_pool.size());
}

int HoumoQwen35LLM::find_and_restore_best_checkpoint(int seq_id,
                                                      int max_position) {
    auto snap_iter = prefill_snapshots_.find(seq_id);
    if (snap_iter == prefill_snapshots_.end() || !snap_iter->second.valid) {
        return -1;
    }
    auto &snapshot = snap_iter->second;
    auto &index = snapshot.checkpoint_index;
    if (index.empty()) {
        return -1;
    }
    // Find the highest checkpoint with position <= max_position
    auto it = index.upper_bound(max_position);
    if (it == index.begin()) {
        return -1; // no checkpoint at or below max_position
    }
    --it;

    auto cache_iter = state_cache_.find(seq_id);
    if (cache_iter == state_cache_.end()) {
        return -1;
    }
    auto &tensors = cache_iter->second.tensors;
    auto &slot = snapshot.checkpoint_pool[it->second];
    auto &ckpt_tensors = slot.state_tensors;
    if (ckpt_tensors.size() != checkpoint_tensor_indices_.size()) {
        LLAMA_LOG_WARN("%s: checkpoint state_tensors size mismatch seq_id=%d "
                       "(%zu vs %zu)\n",
                       __func__, seq_id, ckpt_tensors.size(),
                       checkpoint_tensor_indices_.size());
        return -1;
    }
    for (size_t ci = 0; ci < checkpoint_tensor_indices_.size(); ++ci) {
        size_t ti = checkpoint_tensor_indices_[ci];
        if (ckpt_tensors[ci].Info().Shape() != tensors[ti].Info().Shape() ||
            ckpt_tensors[ci].Info().DataType() != tensors[ti].Info().DataType()) {
            LLAMA_LOG_WARN("%s: checkpoint tensor %zu info mismatch "
                           "seq_id=%d\n",
                           __func__, ci, seq_id);
            return -1;
        }
        ckpt_tensors[ci].CopyTo(tensors[ti]);
    }
    int restored_pos = slot.position;

    // Erase stale checkpoints after restored_pos (free slots, no realloc).
    release_checkpoints_after(snapshot, restored_pos);

    LLAMA_LOG_INFO("%s: restored checkpoint seq_id=%d pos=%d "
                   "reused_tokens=%d (max_allowed=%d)\n",
                   __func__, seq_id, restored_pos, restored_pos, max_position);
    return restored_pos;
}

void HoumoQwen35LLM::restore_zero_state(int seq_id) {
    auto cache_iter = state_cache_.find(seq_id);
    if (cache_iter == state_cache_.end()) {
        return;
    }
    auto &tensors = cache_iter->second.tensors;
    for (size_t idx : checkpoint_tensor_indices_) {
        zero_state_tensors_[idx].CopyTo(tensors[idx]);
    }
}

void HoumoQwen35LLM::trim_snapshot_to_kv_offset(int seq_id,
                                                int32_t kv_offset) {
    auto snap_iter = prefill_snapshots_.find(seq_id);
    if (snap_iter == prefill_snapshots_.end()) {
        return;
    }

    auto &snapshot = snap_iter->second;
    const int32_t trim_pos = std::max<int32_t>(0, kv_offset);

    bool changed = false;
    const size_t trim_size = static_cast<size_t>(trim_pos);

    if (snapshot.tokens.size() > trim_size) {
        snapshot.tokens.resize(trim_size);
        changed = true;
    }

    if (snapshot.checkpoint_index.upper_bound(trim_pos) !=
        snapshot.checkpoint_index.end()) {
        release_checkpoints_after(snapshot, trim_pos);
        changed = true;
    }

    // IMPORTANT: do NOT clamp last_processed_pos here.
    // last_processed_pos reflects where state tensors actually are.
    // memory_seq_rm() only truncates KV/memory, it does not rewind state
    // tensors. If we force last_processed_pos down to kv_offset, a later
    // prefill may mis-detect "continuation" and skip required rollback/
    // replay, causing duplicated-token state advance.

    if (snapshot.first_safe_position > trim_pos) {
        snapshot.first_safe_position = trim_pos;
        changed = true;
    }

    if (changed) {
        LLAMA_LOG_INFO(
            "%s: trimmed snapshot by boundary seq_id=%d valid_length=%d "
            "tokens=%zu ckpts=%zu state_pos=%d first_safe_position=%d\n",
            __func__, seq_id, trim_pos, snapshot.tokens.size(),
            snapshot.checkpoint_index.size(), snapshot.last_processed_pos,
            snapshot.first_safe_position);
    }
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
    } else if (tensor.Info().DataType() == tcim::DataType::FLOAT32) {
        size_t elem_count = tensor.MemSize() / sizeof(float);
        std::vector<float> data(elem_count, 0.0f);
        size_t stride = elem_count / static_cast<size_t>(batch_size);
        for (int b = 0; b < batch_size; ++b) {
            size_t offset = static_cast<size_t>(b) * stride;
            size_t limit = std::min(static_cast<size_t>(new_cache_length), stride);
            for (size_t i = 0; i < limit; ++i) {
                data[offset + i] = 1.0f;
            }
        }
        tensor.Buffer().CopyFromHost(data.data(), tensor.MemSize());
    } else {
        zero_tensor(tensor);
    }
}

int HoumoQwen35LLM::run_prefill_round(int seq_id,
                                      tcim::Tensor &input_data,
                                      int32_t valid_length,
                                      int32_t current_length,
                                      const int32_t *time_pos_slice,
                                      const int32_t *height_pos_slice,
                                      const int32_t *width_pos_slice,
                                      bool is_first_round) {
    LLAMA_LOG_INFO("%s: seq_id=%d, valid_length=%d, current_length=%d, is_first_round=%d\n",
                   __func__, seq_id, valid_length, current_length,
                   is_first_round);
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

    auto bind_pos = [&](const std::string &name, const int32_t *slice) {
        if (name.empty() || slice == nullptr) {
            return;
        }
        auto tensor = prefill_input_map_.at(name);
        std::vector<int32_t> v(slice, slice + prefill_length_);
        set_input_data(tensor, v);
        prefill_model_->SetInput(name, tensor);
    };
    bind_pos(prefill_time_pos_name_, time_pos_slice);
    bind_pos(prefill_height_pos_name_, height_pos_slice);
    bind_pos(prefill_width_pos_name_, width_pos_slice);

    if (!prefill_linear_mask_name_.empty()) {
        auto linear_mask = prefill_input_map_.at(prefill_linear_mask_name_);
        set_linear_attn_mask(linear_mask, prefill_length_, current_length);
        prefill_model_->SetInput(prefill_linear_mask_name_, linear_mask);
    }

    PERF_START(PREFILL_SETINPUT_ONCE);
    if (is_first_round) {
        for (const auto &pair : lora_map) {
            if (prefill_input_map_.count(pair.first) > 0) {
                auto lora_mask = prefill_input_map_[pair.first];
                lora_mask_float_.Buffer().CopyFromHost(&pair.second.scale,
                                                       lora_mask_float_.MemSize());
                lora_mask_float_.CastTo(lora_mask);
                prefill_model_->SetInput(pair.first, lora_mask);
            }
        }
    }
    prefill_model_->SetInput(prefill_model_->GetInputName(0), input_data);
    if (is_first_round) {
        set_state_cache_for_prefill(seq_id);
        set_kv_cache_for_prefill(seq_id);
    }
    PERF_STOP(PREFILL_SETINPUT_ONCE);

    PERF_START(PREFILL_RUN_ONCE);
    auto status = prefill_model_->Run();
    if (status != tcim::Status::OK) {
        PERF_STOP(PREFILL_RUN_ONCE);
        LLAMA_LOG_ERROR("%s: prefill_model_->Run() failed, result = %d\n",
                        __func__, static_cast<int>(status));
        return -1;
    }
    prefill_model_->Sync();
    PERF_STOP(PREFILL_RUN_ONCE);
    return 0;
}

int HoumoQwen35LLM::houmo_prefill(std::vector<llama_token> &tokens, int seq_id,
                                 float *logits) {
    LLAMA_LOG_INFO("%s seq_id = %d, tokens size is = %zu\n", "prefill stage",
                   seq_id, tokens.size());

    // ---- Vision multi-call caching ----
    // Only cache text embeddings when this sequence is already in a
    // multimodal flow (i.e. at least one vision chunk has been received
    // via the embd overload).  Pure-text prompts that llama.cpp chunks
    // into multiple batches — where only the last batch carries logits —
    // must NOT enter the vision cache path, otherwise:
    //   1. The prefill loop is bypassed, so no intermediate state
    //      checkpoints are saved;
    //   2. The final vision path stores a sentinel-zero token history,
    //      making gap replay for future prefix reuse impossible.
    // For such pure-text continuation batches, fall through to the normal
    // text prefill path.  It already supports `logits==nullptr` (skipping
    // the memcpy to logits) and maintains snapshot history + checkpoints.
    auto has_vision_chunk = [&]() {
        auto it = pending_chunks_.find(seq_id);
        if (it == pending_chunks_.end()) return false;
        for (const auto & c : it->second) {
            if (c.is_vision) return true;
        }
        return false;
    };

    if (logits == nullptr && has_vision_chunk()) {
        // Text chunk interleaved between vision chunks — cache it.
        if (!tokens.empty()) {
            auto &cache = embedding_cache_[seq_id];
            size_t cur = cache.size();
            size_t add_elems = tokens.size() * n_embd_;
            cache.resize(cur + add_elems);
            embedding_layer_->get_embedding_batch(
                tokens.data(), tokens.size(), cache.data() + cur);
            pending_chunks_[seq_id].push_back(
                {static_cast<int>(tokens.size()), /*is_vision=*/false, 0, 0,
                 tokens});
            LLAMA_LOG_INFO("%s: cached %zu text tokens for seq_id=%d (vision flow)\n",
                           __func__, tokens.size(), seq_id);
        }
        return 0;
    }

    // When logits is non-null and a vision chunk is pending, flush via
    // vision path (handles M-RoPE, cached embeddings, etc.).
    if (logits != nullptr && has_vision_chunk()) {
        return houmo_prefill_vision(tokens, seq_id, logits);
    }
    // ---- End vision caching ----

    // kv_offset: how many tokens the KV cache already has (prefix reuse
    // computed by the outer layer).  The incoming `tokens` do NOT include
    // those prefix tokens — they have already been stripped.
    int32_t kv_offset = memory_ ? memory_->seq_pos_min(seq_id) : 0;

    if (kv_offset == 0) {
        decode_position_offset_[seq_id] = 0;
    }

    // ---- state cache checkpoint reuse ----
    // The outer layer reuses kv_offset tokens of KV cache.  But the state
    // cache (conv_cache / recurrent_state) checkpoint may be at a position
    // earlier than kv_offset.  In that case we need to:
    //   1. Restore state to the best checkpoint position
    //   2. Retrieve the "gap" tokens [checkpoint_pos .. kv_offset) from our
    //      saved token history
    //   3. Roll back the KV cache to the checkpoint position
    //   4. Prepend gap tokens to incoming tokens and re-compute from there
    //
    // Continuation batches (multi-batch prefill for the same prompt) are
    // detected by snapshot.last_processed_pos == kv_offset, meaning the
    // state is already up-to-date at kv_offset — no rollback needed.
    std::vector<llama_token> effective_tokens;
    int32_t effective_kv_offset = kv_offset;

    if (!state_cache_names_.empty() && kv_offset > 0) {
        auto snap_iter = prefill_snapshots_.find(seq_id);
        bool is_continuation = false;

        if (snap_iter != prefill_snapshots_.end() &&
            snap_iter->second.valid) {
            // If the snapshot's last processed position matches kv_offset
            // exactly, this is a continuation batch — the previous
            // houmo_prefill call (or decode) already brought the state to
            // this position.
            if (snap_iter->second.last_processed_pos == kv_offset) {
                is_continuation = true;
            }
        }

        if (is_continuation) {
            // Continuation batch — state is already correct, just proceed.
            effective_tokens = tokens;
            LLAMA_LOG_INFO(
                "%s: continuation batch seq_id=%d kv_offset=%d "
                "last_processed_pos=%d\n",
                __func__, seq_id, kv_offset,
                snap_iter->second.last_processed_pos);
        } else if (snap_iter != prefill_snapshots_.end() &&
                   snap_iter->second.valid) {
            // First batch of a new request with KV prefix reuse.
            // Try to find a checkpoint at or before kv_offset.
            const auto &saved_history = snap_iter->second.tokens;
            int best_ckpt = -1;

            if (!snap_iter->second.checkpoint_index.empty()) {
                best_ckpt =
                    find_and_restore_best_checkpoint(seq_id, kv_offset);
            }

            // Determine the rollback position.
            // - best_ckpt > 0  → state restored to that position
            // - best_ckpt == -1 → no checkpoint found, state must start
            //   from position 0 (will be zero-initialized)
            //
            // In both cases, if rollback_pos < kv_offset we need to
            // retrieve the gap tokens [rollback_pos .. kv_offset) from
            // saved history and roll back the KV cache, because the
            // state cache is sequential and must see ALL tokens.
            int rollback_pos = (best_ckpt > 0) ? best_ckpt : 0;

            // Sentinel guard: history positions in
            // [0 .. first_safe_position) may be vision-sentinel zeros
            // (placeholders for image embeddings), which cannot be replayed
            // through the text embedding path.  If our rollback would land
            // before first_safe_position the gap would contain sentinels
            // and produce wrong state — treat this case as "unrecoverable"
            // and degrade gracefully.
            const int first_safe_pos = snap_iter->second.first_safe_position;
            const bool sentinel_block =
                (rollback_pos < first_safe_pos) && (kv_offset > 0);

            if (sentinel_block) {
                LLAMA_LOG_WARN(
                    "%s: rollback_pos=%d < first_safe_position=%d for "
                    "seq_id=%d (vision sentinels in gap), refusing replay; "
                    "state will be inaccurate. should never come here\n",
                    __func__, rollback_pos, first_safe_pos, seq_id);
                effective_tokens = tokens;
                state_cache_initialized_[seq_id] = true;
                snap_iter->second.valid = false;
                clear_all_checkpoints(snap_iter->second);
                snap_iter->second.first_safe_position = 0;
            } else if (rollback_pos < kv_offset) {
                // Need gap tokens [rollback_pos .. kv_offset).
                if (static_cast<size_t>(kv_offset) <= saved_history.size()) {
                    std::vector<llama_token> gap(
                        saved_history.begin() + rollback_pos,
                        saved_history.begin() + kv_offset);

                    // Roll back KV cache to the rollback position.
                    if (memory_) {
                        memory_->seq_rm(seq_id, rollback_pos, kv_offset);
                    }

                    // Build effective tokens: gap + incoming.
                    effective_tokens.reserve(gap.size() + tokens.size());
                    effective_tokens.insert(effective_tokens.end(),
                                           gap.begin(), gap.end());
                    effective_tokens.insert(effective_tokens.end(),
                                           tokens.begin(), tokens.end());
                    effective_kv_offset = rollback_pos;

                    if (best_ckpt > 0) {
                        state_cache_initialized_[seq_id] = true;
                    } else {
                        // No checkpoint — zero-init state.
                        state_cache_initialized_[seq_id] = false;
                    }

                    LLAMA_LOG_INFO(
                        "%s: rollback seq_id=%d, best_ckpt=%d, "
                        "kv_offset=%d, last_processed_pos=%d, "
                        "gap=%zu, effective=%zu, "
                        "effective_kv_offset=%d\n",
                        __func__, seq_id, best_ckpt, kv_offset,
                        snap_iter->second.last_processed_pos,
                        gap.size(), effective_tokens.size(),
                        effective_kv_offset);
                } else {
                    // History too short to fill gap — degrade gracefully.
                    LLAMA_LOG_WARN(
                        "%s: saved history too short (%zu) to fill gap "
                        "[%d..%d) for seq_id=%d, refusing replay; state may be inaccurate. should never come here\n",
                        __func__, saved_history.size(),
                        rollback_pos, kv_offset, seq_id);
                    effective_tokens = tokens;
                    state_cache_initialized_[seq_id] = true;
                    snap_iter->second.valid = false;
                    clear_all_checkpoints(snap_iter->second);
                    snap_iter->second.first_safe_position = 0;
                }
                snap_iter->second.valid = false;
                if (best_ckpt <= 0) {
                    // No usable checkpoint — clear all stale checkpoints.
                    clear_all_checkpoints(snap_iter->second);
                }
                // When best_ckpt > 0, find_and_restore_best_checkpoint
                // already erased stale checkpoints after the restored
                // position.  Keep the restored checkpoint so that the
                // NEXT request can still reuse it.
            } else {
                // rollback_pos == kv_offset → checkpoint exactly matches.
                effective_tokens = tokens;
                state_cache_initialized_[seq_id] = true;
                snap_iter->second.valid = false;
                LLAMA_LOG_INFO(
                    "%s: state checkpoint matches kv_offset=%d, "
                    "no gap, seq_id=%d\n",
                    __func__, kv_offset, seq_id);
            }
        } else {
            // No snapshot at all — state must start from position 0.
            // If kv_offset > 0 we must roll back KV cache and recompute
            // from scratch, but we have no saved history to retrieve
            // tokens [0..kv_offset).  This only happens if the first
            // request had kv_offset > 0 somehow (shouldn't happen in
            // practice).
            if (kv_offset > 0) {
                LLAMA_LOG_WARN(
                    "%s: no snapshot for seq_id=%d but kv_offset=%d, "
                    "cannot retrieve prefix tokens, refusing replay; "
                    "state may be inaccurate. should never come here!!!\n",
                    __func__, seq_id, kv_offset);
            }
            effective_tokens = tokens;
            state_cache_initialized_[seq_id] = true;
            auto snap_iter2 = prefill_snapshots_.find(seq_id);
            if (snap_iter2 != prefill_snapshots_.end()) {
                snap_iter2->second.first_safe_position = 0;
            }
        }
    } else if (!state_cache_names_.empty() && kv_offset == 0) {
        // Fresh start — reset state.
        effective_tokens = tokens;
        state_cache_initialized_[seq_id] = false;
        auto snap_iter = prefill_snapshots_.find(seq_id);
        if (snap_iter != prefill_snapshots_.end()) {
            snap_iter->second.valid = false;
            clear_all_checkpoints(snap_iter->second);
            snap_iter->second.last_processed_pos = 0;
            snap_iter->second.first_safe_position = 0;
        }
    } else {
        // No state cache — use tokens as-is.
        effective_tokens = tokens;
    }
    // ---- end state cache checkpoint reuse ----

    // Prefill executes from effective_kv_offset, which is the actual
    // valid_length fed into prefill rounds. Clean snapshot metadata to this
    // boundary before running rounds so stale token/checkpoint/sentinel
    // markers beyond valid_length do not affect subsequent decisions.
    if (!state_cache_names_.empty()) {
        trim_snapshot_to_kv_offset(seq_id, effective_kv_offset);
        auto snap_iter2 = prefill_snapshots_.find(seq_id);
        if (snap_iter2 != prefill_snapshots_.end()) {
            LLAMA_LOG_INFO(
                "%s: cleanup-by-valid_length seq_id=%d kv_offset=%d "
                "effective_kv_offset=%d first_safe_position=%d "
                "history=%zu ckpts=%zu\n",
                __func__, seq_id, kv_offset, effective_kv_offset,
                snap_iter2->second.first_safe_position,
                snap_iter2->second.tokens.size(),
                snap_iter2->second.checkpoint_index.size());
        }
    }

    // Mark state as needing zero-init if effective_kv_offset is 0
    if (!state_cache_names_.empty() && effective_kv_offset == 0) {
        state_cache_initialized_[seq_id] = false;
    }

    if ((effective_tokens.size() + effective_kv_offset) >=
        static_cast<uint32_t>(context_length_)) {
        LLAMA_LOG_ERROR(
            "%s: input size %zu + offset %d >= context_length_ %d\n",
            __func__, effective_tokens.size(), effective_kv_offset,
            context_length_);
        return 1;
    }

    size_t process_tokens = effective_tokens.size();
    if (process_tokens == 0) {
        LLAMA_LOG_INFO(
            "%s: no tokens to process for seq_id=%d (kv_offset=%d)\n",
            __func__, seq_id, kv_offset);
        // Still update snapshot history.
        auto &snapshot = prefill_snapshots_[seq_id];
        if (snapshot.tokens.size() >= static_cast<size_t>(effective_kv_offset)) {
            snapshot.tokens.resize(static_cast<size_t>(effective_kv_offset));
        }
        snapshot.tokens.insert(snapshot.tokens.end(),
                               tokens.begin(), tokens.end());
        if (effective_kv_offset == 0) {
            // Pure-text prefill rebuilt from position 0; any prior
            // vision-sentinel boundary is stale and must be cleared.
            snapshot.first_safe_position = 0;
        }
        snapshot.last_processed_pos = effective_kv_offset;
        snapshot.valid = true;
        decode_valid_length_[seq_id] = effective_kv_offset;
        return 0;
    }

    int32_t current_length = 0;
    int32_t valid_length = effective_kv_offset;

    size_t rounds = (process_tokens + prefill_length_ - 1) / prefill_length_;
    auto &input_data = prefill_input_map_.at(prefill_model_->GetInputName(0));
    for (size_t round = 0; round < rounds; ++round) {
        size_t round_start = round * prefill_length_;
        size_t remaining = process_tokens > round_start ? process_tokens - round_start : 0;
        current_length = remaining >= static_cast<size_t>(prefill_length_)
                             ? prefill_length_
                             : static_cast<int>(remaining);

        const int32_t *token_ptr = effective_tokens.data() + round_start;
        memset(input_data.Data(), 0, input_data.MemSize());
        embedding_layer_->get_embedding_batch(token_ptr, current_length,
                                             input_data.Data());

        int prefill_position_base = valid_length + decode_position_offset_[seq_id];
        auto position_ids = build_rope_position_ids(prefill_position_base, prefill_length_);
        if (run_prefill_round(seq_id, input_data, valid_length, current_length,
                              position_ids.data(), position_ids.data(),
                              position_ids.data(),
                              /*is_first_round=*/round == 0) != 0) {
            return -1;
        }
        valid_length += current_length;

        // Save a checkpoint after this round when:
        //   a) Not the last round (last round state is unreliable due to chat
        //      template token instability across turns).
        //   b) Either the interval threshold is exceeded, OR this is the
        //      second-to-last round (always ensure one stable checkpoint just
        //      before the final round so future turns can reuse it).
        if (valid_length > 0 && round + 1 < rounds) {
            auto &ckpts = prefill_snapshots_[seq_id].checkpoint_index;
            bool is_second_last = (round + 2 == rounds);
            bool should_save = is_second_last && ckpts.find(valid_length) == ckpts.end();
            if (!should_save) {
                bool by_interval = ckpts.empty();
                if (!by_interval) {

                    auto it = ckpts.upper_bound(valid_length);
                    if (it != ckpts.end()) {
                        LLAMA_LOG_ERROR(
                            "%s: BUG: stale checkpoint pos=%d found after "
                            "valid_length=%d for seq_id=%d — should "
                            "have been released before this round\n",
                            __func__, it->first, valid_length, seq_id);
                    }
                    int dist = INT_MAX;
                    if (it != ckpts.begin()) {
                        --it;
                        dist = (int)valid_length - it->first;
                    }
                    by_interval = (dist >= checkpoint_min_interval_);
                }
                should_save = by_interval;
            }
            if (should_save) {
                save_state_checkpoint(seq_id, valid_length);
            }
        }
    }

    decode_valid_length_[seq_id] = effective_kv_offset + static_cast<int32_t>(process_tokens);
    PERF_START(PREFILL_GETOUTPUT);
    tcim::Tensor prefill_output_data =
        prefill_model_->GetOutput(prefill_output_names_[0]);
    auto &prefill_output_float = prefill_output_map_[prefill_output_names_[0]];
    prefill_output_data.CastTo(prefill_output_float);
    PERF_STOP(PREFILL_GETOUTPUT);
    if (logits != nullptr) {
        memcpy(logits, prefill_output_float.Data(), prefill_output_float.MemSize());
    }

    if (memory_) {
        memory_->seq_add(seq_id, 0, 0, static_cast<int32_t>(process_tokens));
    }

    // ---- update snapshot token history ----
    // Maintain a complete token history so that future checkpoint restores
    // can retrieve gap tokens that are not passed in by the outer layer.
    //
    // Layout: snapshot.tokens = [0 .. effective_kv_offset) ++ effective_tokens
    //
    // For rollback case (effective_kv_offset < kv_offset):
    //   keeps history[0..ckpt) + gap + incoming  → full history restored
    // For continuation batch:
    //   keeps history[0..kv_offset) + incoming   → history extended
    // For fresh start (kv_offset == 0):
    //   effective_tokens == tokens               → history starts fresh
    {
        auto &snapshot = prefill_snapshots_[seq_id];

        if (snapshot.tokens.size() >=
            static_cast<size_t>(effective_kv_offset)) {
            // Truncate to effective_kv_offset (keep the prefix) and append
            // the tokens we just processed.
            snapshot.tokens.resize(
                static_cast<size_t>(effective_kv_offset));
        } else {
            // History is shorter than expected — pad with zeros.
            // This is a degraded case; future gap retrieval for positions
            // within the padding won't produce correct tokens.
            LLAMA_LOG_WARN(
                "%s: snapshot history (%zu) < effective_kv_offset (%d) "
                "for seq_id=%d, padding\n",
                __func__, snapshot.tokens.size(), effective_kv_offset,
                seq_id);
            snapshot.tokens.resize(
                static_cast<size_t>(effective_kv_offset), 0);
        }
        snapshot.tokens.insert(snapshot.tokens.end(),
                               effective_tokens.begin(),
                               effective_tokens.end());

        if (effective_kv_offset == 0) {
            // Text prefill from offset 0 rebuilds history with real text
            // token ids, so any previous vision sentinel guard is stale.
            snapshot.first_safe_position = 0;
        }

        snapshot.last_processed_pos = valid_length;
        snapshot.valid = true;
    }
    // ---- end snapshot ----

    return 0;
}

int HoumoQwen35LLM::houmo_prefill(const float *embeddings, int n_tokens,
                                 int seq_id, float *logits) {
    (void)logits;
    LLAMA_LOG_INFO("%s: seq_id=%d, n_tokens=%d (vision embedding)\n",
                   __func__, seq_id, n_tokens);

    if (n_tokens <= 0) {
        return 0;
    }

    // Parse compact header to extract grid dimensions
    VisionCompactHeader header;
    int embedding_offset = 0;
    int16_t nx = 0, ny = 0;
    std::memcpy(&header, embeddings, sizeof(VisionCompactHeader));
    if (header.magic == 0x1234) {
        ny = header.ny;
        nx = header.nx;
        embedding_offset = sizeof(VisionCompactHeader);
        LLAMA_LOG_INFO("%s: header magic=0x%04x nx=%d ny=%d chunksize=%d\n",
                       __func__, header.magic, nx, ny, header.chunksize);
    }

    // Cache raw embedding data (fp16 / int16 format)
    const int8_t *ptr = reinterpret_cast<const int8_t *>(embeddings);
    const size_t data_elems = static_cast<size_t>(n_tokens) * n_embd_;
    const size_t data_bytes = data_elems * sizeof(int16_t);

    auto &cache = embedding_cache_[seq_id];
    size_t cur = cache.size();
    cache.resize(cur + data_elems);
    std::memcpy(cache.data() + cur, ptr + embedding_offset, data_bytes);

    // Track chunk info (vision chunk has no real token ids)
    pending_chunks_[seq_id].push_back(
        {n_tokens, /*is_vision=*/true, nx, ny, {}});

    return 0;
}

void HoumoQwen35LLM::clear_vision_cache(int seq_id) {
    embedding_cache_[seq_id].clear();
    pending_chunks_[seq_id].clear();
}

void HoumoQwen35LLM::compute_vision_position_ids(
    int seq_id, int kv_offset,
    const std::vector<llama_token> & /*trailing_tokens*/,
    std::vector<int32_t> &time_positions,
    std::vector<int32_t> &height_positions,
    std::vector<int32_t> &width_positions,
    int32_t &final_position) {
    (void)seq_id;
    // Build M-RoPE position IDs matching Python get_rope_index:
    //   - text chunk: all 3 dims = start_pos + i
    //   - vision chunk (nx*ny tokens): t = start_pos,
    //       h = start_pos + y, w = start_pos + x  (y in [0,ny), x in [0,nx))
    //       then start_pos advances by max(nx, ny)
    auto &chunks = pending_chunks_[seq_id];
    int32_t pos = static_cast<int32_t>(kv_offset);
    for (auto &chunk : chunks) {
        if (chunk.is_vision && chunk.nx > 0 && chunk.ny > 0) {
            // Vision tokens: grid layout height=ny, width=nx
            for (int y = 0; y < chunk.ny; y++) {
                for (int x = 0; x < chunk.nx; x++) {
                    time_positions.push_back(pos);
                    height_positions.push_back(pos + y);
                    width_positions.push_back(pos + x);
                }
            }
            pos += std::max(chunk.nx, chunk.ny);
        } else {
            // Text tokens
            for (int i = 0; i < chunk.size; i++) {
                time_positions.push_back(pos + i);
                height_positions.push_back(pos + i);
                width_positions.push_back(pos + i);
            }
            pos += chunk.size;
        }
    }
    final_position = pos;
}

int HoumoQwen35LLM::houmo_prefill_vision(std::vector<llama_token> &tokens,
                                          int seq_id, float *logits) {
    LLAMA_LOG_INFO("%s: seq_id=%d, trailing text tokens=%zu\n",
                   __func__, seq_id, tokens.size());

    // Append trailing text tokens to cache
    if (!tokens.empty()) {
        auto &cache = embedding_cache_[seq_id];
        size_t cur = cache.size();
        size_t add_elems = tokens.size() * n_embd_;
        cache.resize(cur + add_elems);
        embedding_layer_->get_embedding_batch(
            tokens.data(), tokens.size(), cache.data() + cur);
        pending_chunks_[seq_id].push_back(
            {static_cast<int>(tokens.size()), /*is_vision=*/false, 0, 0,
             tokens});
    }

    // Compute total tokens, and the local position right after the LAST
    // vision chunk completes.  Token positions in the snapshot covered by
    // text-only suffix [last_vision_end_local .. total_tokens) are safe to
    // replay (real ids), while [0 .. last_vision_end_local) may contain
    // vision sentinels — checkpoints that fall before last_vision_end_local
    // must NOT be used as a rollback point for gap replay.
    size_t total_tokens = 0;
    size_t last_vision_end_local = 0;
    for (auto &chunk : pending_chunks_[seq_id]) {
        total_tokens += chunk.size;
        if (chunk.is_vision) {
            last_vision_end_local = total_tokens;
        }
    }

    int32_t kv_offset = memory_ ? memory_->seq_pos_min(seq_id) : 0;

    if ((total_tokens + kv_offset) >= static_cast<size_t>(context_length_)) {
        LLAMA_LOG_ERROR("%s: total %zu + kv_offset %d >= context_length %d\n",
                        __func__, total_tokens, kv_offset, context_length_);
        clear_vision_cache(seq_id);
        return 1;
    }

    // Reset state for fresh vision prefill
    if (!state_cache_names_.empty()) {
        state_cache_initialized_[seq_id] = false;
        auto snap_iter = prefill_snapshots_.find(seq_id);
        if (snap_iter != prefill_snapshots_.end()) {
            snap_iter->second.valid = false;
            clear_all_checkpoints(snap_iter->second);
            snap_iter->second.last_processed_pos = 0;
            snap_iter->second.first_safe_position = 0;
        }
    }

    // Compute M-RoPE position IDs
    std::vector<int32_t> time_pos_ids, height_pos_ids, width_pos_ids;
    int32_t final_pos = 0;
    compute_vision_position_ids(seq_id, kv_offset, tokens,
                                time_pos_ids, height_pos_ids, width_pos_ids,
                                final_pos);

    // The position offset for decode = (final M-RoPE position) - (total tokens + kv_offset)
    // This is the "rope_deltas" equivalent from Python
    decode_position_offset_[seq_id] = final_pos - static_cast<int32_t>(total_tokens) - kv_offset;
    LLAMA_LOG_INFO("%s: total_tokens=%zu kv_offset=%d final_pos=%d decode_position_offset=%d\n",
                   __func__, total_tokens, kv_offset, final_pos,
                   decode_position_offset_[seq_id]);

    // Pad position IDs.  We use a variable-step round loop below so the
    // start of any round is bounded by total_tokens, and each slice reads
    // prefill_length_ entries.  Pad enough to cover the worst case where
    // the last round started just before total_tokens.
    size_t padded_size = total_tokens + prefill_length_;
    time_pos_ids.resize(padded_size, time_pos_ids.empty() ? 0 : time_pos_ids.back());
    height_pos_ids.resize(padded_size, height_pos_ids.empty() ? 0 : height_pos_ids.back());
    width_pos_ids.resize(padded_size, width_pos_ids.empty() ? 0 : width_pos_ids.back());

    int32_t valid_length = kv_offset;
    int32_t current_length = 0;
    // Absolute position at which the last vision chunk ends.  Once a round
    // boundary reaches or passes this position, the trailing state cache is
    // safe to checkpoint for future regenerate / prefix-reuse, because the
    // suffix [safe_pos_abs .. total_end) contains only real text tokens.
    const int32_t safe_pos_abs =
        kv_offset + static_cast<int32_t>(last_vision_end_local);
    const int32_t total_end =
        kv_offset + static_cast<int32_t>(total_tokens);
    bool safe_ckpt_saved = false;

    auto &input_data = prefill_input_map_.at(prefill_model_->GetInputName(0));
    auto &emb_cache = embedding_cache_[seq_id];

    // Variable-step round loop:
    // - Each step processes at most prefill_length_ valid tokens.
    // - The step never crosses last_vision_end_local, so we get a clean
    //   round boundary right after the vision portion.  Without this
    //   forced break, a single-round prefill (total_tokens <= prefill_length_)
    //   would leave us no chance to snapshot the post-vision state.
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

        // Copy cached embeddings into prefill input
        memset(input_data.Data(), 0, input_data.MemSize());
        size_t emb_start = round_start * n_embd_;
        size_t copy_bytes = current_length * n_embd_ * sizeof(int16_t);
        std::memcpy(input_data.Data(), emb_cache.data() + emb_start, copy_bytes);

        if (run_prefill_round(seq_id, input_data, valid_length, current_length,
                              time_pos_ids.data() + round_start,
                              height_pos_ids.data() + round_start,
                              width_pos_ids.data() + round_start,
                              is_first_round) != 0) {
            clear_vision_cache(seq_id);
            return -1;
        }

        if (abort_callback_ != nullptr && abort_callback_(abort_callback_data_)) {
            LLAMA_LOG_INFO("%s: aborted seq_id=%d\n", __func__, seq_id);
            clear_vision_cache(seq_id);
            return 2;
        }

        valid_length += current_length;
        cursor += step;
        ++round_idx;

        // Save a safe checkpoint as soon as we cross the end of the last
        // vision chunk, provided some text still follows (otherwise the
        // end-of-prefill checkpoint already covers this exact position).
        // The variable-step loop above guarantees there is a round
        // boundary at safe_pos_abs, so this branch fires reliably even
        // when total_tokens fits in a single prefill_length_ window.
        if (!safe_ckpt_saved && last_vision_end_local > 0 &&
            valid_length >= safe_pos_abs && valid_length < total_end) {
            save_state_checkpoint(seq_id, valid_length);
            safe_ckpt_saved = true;
            LLAMA_LOG_INFO(
                "%s: saved safe post-vision checkpoint seq_id=%d pos=%d "
                "(safe_pos_abs=%d total_end=%d)\n",
                __func__, seq_id, valid_length, safe_pos_abs, total_end);
        }
    }

    decode_valid_length_[seq_id] = kv_offset + static_cast<int32_t>(total_tokens);

    PERF_START(PREFILL_GETOUTPUT);
    tcim::Tensor prefill_output_data =
        prefill_model_->GetOutput(prefill_output_names_[0]);
    auto &prefill_output_float = prefill_output_map_[prefill_output_names_[0]];
    prefill_output_data.CastTo(prefill_output_float);
    PERF_STOP(PREFILL_GETOUTPUT);
    if (logits != nullptr) {
        memcpy(logits, prefill_output_float.Data(), prefill_output_float.MemSize());
    }

    if (memory_) {
        memory_->seq_add(seq_id, 0, 0, static_cast<int32_t>(total_tokens));
    }

    // Update snapshot.  Vision prefill always restarts state from 0 above
    // (kv_offset > 0 is supported but state is reset), so the snapshot
    // history is rebuilt from scratch from pending_chunks_:
    //   - text chunks contribute their real token ids
    //   - vision chunks contribute sentinel zeros of equal length
    // first_safe_position marks where vision sentinels end, so future text
    // prefills know they cannot roll back further than this without
    // corrupting state-cache replay.
    {
        auto &snapshot = prefill_snapshots_[seq_id];
        if (kv_offset > 0) {
            if (snapshot.tokens.size() >= static_cast<size_t>(kv_offset)) {
                snapshot.tokens.resize(static_cast<size_t>(kv_offset));
            } else {
                LLAMA_LOG_WARN(
                    "%s: existing snapshot history (%zu) shorter than "
                    "kv_offset (%d) for seq_id=%d, padding with sentinels\n",
                    __func__, snapshot.tokens.size(), kv_offset, seq_id);
                snapshot.tokens.resize(static_cast<size_t>(kv_offset),
                                       static_cast<llama_token>(0));
            }
        } else {
            snapshot.tokens.clear();
        }
        snapshot.tokens.reserve(static_cast<size_t>(kv_offset) + total_tokens);
        for (auto &chunk : pending_chunks_[seq_id]) {
            if (chunk.is_vision) {
                snapshot.tokens.insert(snapshot.tokens.end(),
                                       static_cast<size_t>(chunk.size),
                                       static_cast<llama_token>(0));
            } else {
                // Defensive: text chunks should have stored their ids.
                if (static_cast<int>(chunk.text_tokens.size()) ==
                    chunk.size) {
                    snapshot.tokens.insert(snapshot.tokens.end(),
                                           chunk.text_tokens.begin(),
                                           chunk.text_tokens.end());
                } else {
                    LLAMA_LOG_WARN(
                        "%s: text chunk missing token ids (have %zu need %d) "
                        "seq_id=%d, padding with sentinels\n",
                        __func__, chunk.text_tokens.size(), chunk.size,
                        seq_id);
                    snapshot.tokens.insert(snapshot.tokens.end(),
                                           static_cast<size_t>(chunk.size),
                                           static_cast<llama_token>(0));
                }
            }
        }
        // first_safe_position is also an ABSOLUTE position — translate the
        // local end-of-vision marker into the snapshot coordinate space.
        snapshot.first_safe_position =
            kv_offset + static_cast<int>(last_vision_end_local);
        if (valid_length > 0) {
            save_state_checkpoint(seq_id, valid_length);
        }
        snapshot.last_processed_pos = valid_length;
        snapshot.valid = true;
    }

    clear_vision_cache(seq_id);
    LLAMA_LOG_INFO("%s: done seq_id=%d total_tokens=%zu\n",
                   __func__, seq_id, total_tokens);
    return 0;
}

int HoumoQwen35LLM::set_decode_rope_and_mask(const std::vector<int> &seq_ids) {
    for (size_t i = 0; i < seq_ids.size(); i++) {
        int seq_id = seq_ids[i];
        int batch_id = seq_id;
        if (n_decode_batch_ == 1) {
            batch_id = 0;
        }
        int valid_length = decode_valid_length_[seq_id];
        int position = valid_length + decode_position_offset_[seq_id];
        auto position_ids = build_rope_position_ids(position, 1);
        auto time_name = resolve_batch_name(decode_time_pos_names_, batch_id);
        auto height_name = resolve_batch_name(decode_height_pos_names_, batch_id);
        auto width_name = resolve_batch_name(decode_width_pos_names_, batch_id);
        auto mask_name = resolve_batch_name(decode_linear_mask_names_, batch_id);

        if (!time_name.empty()) {
            auto tensor = decode_input_map_.at(time_name);
            set_input_data(tensor, position_ids);
            LLAMA_LOG_DEBUG("%s: seq_id=%d batch_id=%d time_pos=%d\n", __func__,
                            seq_id, batch_id, position_ids[0]);
            decode_model_->SetInput(time_name, tensor);
        }
        if (!height_name.empty()) {
            auto tensor = decode_input_map_.at(height_name);
            set_input_data(tensor, position_ids);
            LLAMA_LOG_DEBUG("%s: seq_id=%d batch_id=%d height_pos=%d\n",
                            __func__, seq_id, batch_id, position_ids[0]);
            decode_model_->SetInput(height_name, tensor);
        }
        if (!width_name.empty()) {
            auto tensor = decode_input_map_.at(width_name);
            set_input_data(tensor, position_ids);
            LLAMA_LOG_DEBUG("%s: seq_id=%d batch_id=%d width_pos=%d\n", __func__,
                           seq_id, batch_id, position_ids[0]);
            decode_model_->SetInput(width_name, tensor);
        }
        if (!mask_name.empty()) {
            auto tensor = decode_input_map_.at(mask_name);
            set_linear_attn_mask(tensor, 1, 1);
            decode_model_->SetInput(mask_name, tensor);
        }
    }
    return 0;
}

int HoumoQwen35LLM::houmo_decode(std::vector<llama_token> &batches,
                                std::vector<int> &seq_ids,
                                std::vector<float *> logits) {
    LLAMA_LOG_DEBUG("%s: seq_ids size = %zu, batches size = %zu\n", __func__,
                    seq_ids.size(), batches.size());
    for (auto seq_id : seq_ids) {
        if (memory_) {
            decode_valid_length_[seq_id] = memory_->seq_pos_min(seq_id);
        }
    }

    // ---- Detect state cache gap and redirect to prefill if needed ----
    // When the state cache has a gap (e.g. KV prefix reused but state
    // checkpoint is behind target_vl), running decode alone produces wrong
    // results.  Since decode processes only 1 token per seq, the simplest
    // fix is to redirect the token through houmo_prefill, which already
    // handles rollback, gap replay, and checkpoint management.
    if (!state_cache_names_.empty()) {
        for (size_t i = 0; i < seq_ids.size(); ++i) {
            int seq_id = seq_ids[i];
            int target_vl = decode_valid_length_[seq_id];

            auto snap_iter = prefill_snapshots_.find(seq_id);
            int state_pos = -1;
            if (snap_iter != prefill_snapshots_.end() &&
                snap_iter->second.valid) {
                state_pos = snap_iter->second.last_processed_pos;
            }

            if (state_pos >= 0 && state_pos != target_vl) {
                // State gap detected — route through prefill instead.
                LLAMA_LOG_INFO(
                    "%s: state gap seq_id=%d state_pos=%d target_vl=%d, "
                    "redirecting token to prefill\n",
                    __func__, seq_id, state_pos, target_vl);
                std::vector<llama_token> token_vec = {batches[i]};
                int ret = houmo_prefill(token_vec, seq_id, logits[i]);
                if (ret != 0) {
                    return ret;
                }
                // prefill updated decode_valid_length_ and snapshot;
                // done for this seq_id.
                if (seq_ids.size() == 1) {
                    return 0;
                }
                // Multi-batch: remove this seq from decode and continue.
                // (rare path — n_decode_batch_ > 1 with gap)
                seq_ids.erase(seq_ids.begin() + static_cast<ptrdiff_t>(i));
                batches.erase(batches.begin() + static_cast<ptrdiff_t>(i));
                logits.erase(logits.begin() + static_cast<ptrdiff_t>(i));
                --i;
                continue;
            }

            if (target_vl == 0) {
                restore_zero_state(seq_id);
                state_cache_initialized_[seq_id] = true;
            }
        }
    }
    // ---- end state cache gap detection ----

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
    set_state_cache_for_decode(seq_ids);
    set_kv_cache_for_decode(seq_ids);

    for (const auto &pair : lora_map) {
        if (decode_input_map_.count(pair.first) > 0) {
            auto lora_mask = decode_input_map_[pair.first];
            lora_mask_float_.Buffer().CopyFromHost(&pair.second.scale,
                                                   lora_mask_float_.MemSize());
            lora_mask_float_.CastTo(lora_mask);
            decode_model_->SetInput(pair.first, lora_mask);
        }
    }
    PERF_STOP(DECODE_SETINPUT);

    PERF_START(DECODE_RUN);
    decode_model_->Run();
    decode_model_->Sync();
    PERF_STOP(DECODE_RUN);

    PERF_START(DECODE_GETOUTPUT);
    tcim::Tensor decode_data = decode_model_->GetOutput(decode_output_names_[0]);
    auto &decode_output_float = decode_output_map_[decode_output_names_[0]];
    decode_data.CastTo(decode_output_float);
    auto n_vocab =
        decode_output_float.MemSize() / (n_decode_batch_ * sizeof(float));
    PERF_STOP(DECODE_GETOUTPUT);

    for (size_t i = 0; i < seq_ids.size(); i++) {
        auto batch_id = seq_ids[i];
        decode_valid_length_[batch_id]++;
        if (n_decode_batch_ == 1) {
            batch_id = 0;
        }
        float *logits_out = logits[i];
        float *output_ptr = static_cast<float *>(decode_output_float.Data()) +
                            batch_id * n_vocab;
        if (logits_out != nullptr) {
            memcpy(logits_out, output_ptr, n_vocab * sizeof(float));
        }
        if (memory_) {
            memory_->seq_add(seq_ids[i], 0, 0, 1);
        }

        // Append the decoded token to snapshot history and save a checkpoint
        // when far enough from the nearest existing one.
        {
            auto &snapshot = prefill_snapshots_[seq_ids[i]];
            if (i < batches.size()) {
                snapshot.tokens.push_back(batches[i]);
            }
            int vl = decode_valid_length_[seq_ids[i]];
            snapshot.last_processed_pos = vl;

            bool should_save = false;
            if (vl > 0) {
                auto &ckpts = snapshot.checkpoint_index;
                if (ckpts.empty()) {
                    should_save = true;
                } else {
                    // Distance to the nearest checkpoint.
                    auto it = ckpts.lower_bound(vl);
                    int dist = INT_MAX;
                    if (it != ckpts.end()) {
                        dist = std::min(dist, it->first - vl);
                    }
                    if (it != ckpts.begin()) {
                        --it;
                        dist = std::min(dist, vl - it->first);
                    }
                    should_save = (dist >= checkpoint_min_interval_);
                }
            }
            if (should_save) {
                save_state_checkpoint(seq_ids[i], vl);
                snapshot.valid = true;
            }
        }
    }
    return 0;
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
