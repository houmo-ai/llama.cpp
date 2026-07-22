#include "houmo-qwen35-state-cache.h"

#include "llama-impl.h"
#include "time-perf.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>

namespace {
void zero_tensor(tcim::Tensor &tensor) {
    if (tensor.Device() == tcim::Device::COMPND) {
        for (const auto &sub_tensor : tcim::CompTensor(tensor).SubTensors()) {
            sub_tensor.Buffer().MemSet(0);
        }
    } else {
        tensor.Buffer().MemSet(0);
    }
}

bool is_state_cache_name(const std::string &name) {
    return name.find("conv_cache") != std::string::npos ||
           name.find("recurrent_state") != std::string::npos;
}

size_t chunk_token_count(const std::vector<PrefillChunk> &chunks) {
    size_t count = 0;
    for (const auto &chunk : chunks) {
        count += chunk.size();
    }
    return count;
}

std::string make_state_output_name(const std::string &input_name) {
    std::string output_name = input_name;
    auto pos = output_name.find("past_conv_cache_");
    if (pos != std::string::npos) {
        output_name.replace(pos, std::strlen("past_conv_cache_"),
                            "conv_cache_out_");
        return output_name;
    }
    pos = output_name.find("past_recurrent_state_");
    if (pos != std::string::npos) {
        output_name.replace(pos, std::strlen("past_recurrent_state_"),
                            "recurrent_state_out_");
    }
    return output_name;
}
} // namespace

void HoumoQwen35StateCache::configure_state_cache(
    std::shared_ptr<tcim::Module> prefill,
    std::shared_ptr<tcim::Module> decode,
    int prefill_length) {
    state_prefill_model_ = std::move(prefill);
    state_decode_model_ = std::move(decode);
    checkpoint_min_interval_ = prefill_length * 4;

    HoumoCheckpointPool config;
    config.configure_from_env(
        checkpoint_min_interval_, static_cast<int>(checkpoint_max_count_),
        /*min_interval=*/1, /*max_interval=*/0xffffff,
        /*max_count_limit=*/80, checkpoint_storage_);
    checkpoint_min_interval_ = config.checkpoint_interval();
    checkpoint_max_count_ =
        static_cast<size_t>(config.checkpoint_max_count());
    checkpoint_storage_ = config.checkpoint_storage();
}

void HoumoQwen35StateCache::reset_state_cache_metadata() {
    state_cache_names_.clear();
    state_cache_output_names_.clear();
    state_cache_split_output_names_.clear();
    checkpoint_tensor_indices_.clear();
    decode_verify_seq_ = 1;
    state_cache_.clear();
    state_cache_initialized_.clear();
    prefill_snapshots_.clear();
    checkpoint_pools_.clear();
}

bool HoumoQwen35StateCache::should_save_checkpoint(int seq_id, int position,
                                                   bool force) {
    auto *pool = find_checkpoint_pool(seq_id);
    if (find_snapshot(seq_id) == nullptr || pool == nullptr ||
        !pool->enabled() || position <= 0) {
        return false;
    }
    auto *exact = pool->find_exact_covering(seq_id, position, position);
    if (exact != nullptr) {
        return false;
    }
    if (force) {
        return true;
    }
    auto *slot = pool->find_nearest_before(seq_id, INT64_MAX, position + 1);
    if (slot == nullptr) {
        return true;
    }
    return position - static_cast<int>(slot->abs_end) >= checkpoint_min_interval_;
}

void HoumoQwen35StateCache::reset_for_fresh_prefill(int seq_id) {
    mark_initialized(seq_id, false);
    auto *snap = find_snapshot(seq_id);
    if (snap == nullptr) {
        return;
    }
    snap->valid = false;
    snap->chunks.clear();
    clear_all_checkpoints(seq_id);
    snap->last_processed_pos = 0;
}

void HoumoQwen35StateCache::invalidate_snapshot(int seq_id,
                                                bool clear_checkpoints) {
    auto *snap = find_snapshot(seq_id);
    if (snap == nullptr) {
        return;
    }
    snap->valid = false;
    snap->chunks.clear();
    if (clear_checkpoints) {
        clear_all_checkpoints(seq_id);
    }
}

void HoumoQwen35StateCache::initialize_state_cache(
    int seq_max, int n_prefill_batch, int n_decode_batch, int context_length,
    bool multi_device) {
    reset_state_cache_metadata();
    if (state_decode_model_ == nullptr) {
        return;
    }

    const auto all_output_names = collect_decode_output_names();
    discover_state_cache_bindings(all_output_names);
    if (state_cache_names_.empty()) {
        return;
    }

    decode_verify_seq_ = static_cast<int>(
        state_decode_model_->GetDevInput(state_decode_model_->GetInputName(0))
            .Info().Shape()[1]);
    LLAMA_LOG_INFO("%s: detected verify-style decode model, verify_seq=%d\n",
                   __func__, decode_verify_seq_);

    init_checkpoint_tensor_indices();
    allocate_state_cache_entries(seq_max, n_prefill_batch, n_decode_batch);

    if (multi_device) {
        LLAMA_LOG_INFO("%s: using host checkpoint storage for multi device "
                       "model\n", __func__);
        checkpoint_storage_ = HoumoCheckpointPool::Storage::kHost;
    }
    init_checkpoint_pool(context_length, multi_device);

    for (auto &[seq_id, entry] : state_cache_) {
        for (auto &tensor : entry.tensors) {
            zero_tensor(tensor);
        }
        state_cache_initialized_[seq_id] = true;
        auto &snap = prefill_snapshots_[seq_id];
        snap.last_processed_pos = 0;
        snap.valid = true;
    }
}

std::vector<std::string> HoumoQwen35StateCache::collect_decode_output_names() const {
    std::vector<std::string> names;
    if (state_decode_model_ == nullptr) {
        return names;
    }
    names.reserve(state_decode_model_->GetOutputNum());
    for (size_t i = 0; i < state_decode_model_->GetOutputNum(); ++i) {
        names.push_back(state_decode_model_->GetOutputName(i));
    }
    return names;
}

std::vector<std::string> HoumoQwen35StateCache::collect_state_split_outputs(
    const std::vector<std::string> &all_output_names,
    const std::string &bare_name) const {
    std::vector<std::pair<int, std::string>> indexed;
    const std::string prefix = bare_name + "_";
    for (const auto &out : all_output_names) {
        if (out.size() <= prefix.size() ||
            out.compare(0, prefix.size(), prefix) != 0) {
            continue;
        }
        const std::string suffix = out.substr(prefix.size());
        if (suffix.empty() ||
            !std::all_of(suffix.begin(), suffix.end(), ::isdigit)) {
            continue;
        }
        try {
            indexed.emplace_back(std::stoi(suffix), out);
        } catch (...) {
            continue;
        }
    }

    std::sort(indexed.begin(), indexed.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    std::vector<std::string> ordered;
    ordered.reserve(indexed.size());
    for (auto &p : indexed) {
        ordered.push_back(std::move(p.second));
    }
    return ordered;
}

void HoumoQwen35StateCache::discover_state_cache_bindings(
    const std::vector<std::string> &all_output_names) {
    auto add_name = [&](const std::string &name, const std::string &output_name,
                        std::vector<std::string> &&split_outputs) {
        auto iter = std::find(state_cache_names_.begin(), state_cache_names_.end(), name);
        if (iter == state_cache_names_.end()) {
            state_cache_names_.push_back(name);
            state_cache_output_names_.push_back(output_name);
            state_cache_split_output_names_.push_back(std::move(split_outputs));
        }
    };
    for (size_t idx = 0; idx < state_decode_model_->GetInputNum(); ++idx) {
        auto input_name = state_decode_model_->GetInputName(idx);
        if (!is_state_cache_name(input_name)) {
            continue;
        }

        const std::string output_name = make_state_output_name(input_name);
        // 优先看 bare 名是否真的存在于 decode 输出里 (普通 1-token decode);
        // 否则按 verify 形态尝试 <bare>_<pos> 收集 split 输出。
        std::vector<std::string> split_outs;
        const bool has_bare =
            std::find(all_output_names.begin(), all_output_names.end(),
                      output_name) != all_output_names.end();
        if (!has_bare) {
            split_outs = collect_state_split_outputs(all_output_names, output_name);
        }
        add_name(input_name, output_name, std::move(split_outs));
    }
}

void HoumoQwen35StateCache::init_checkpoint_tensor_indices() {
    checkpoint_tensor_indices_.clear();
    for (size_t idx = 0; idx < state_cache_names_.size(); ++idx) {
        checkpoint_tensor_indices_.push_back(idx);
    }
    LLAMA_LOG_INFO("%s: checkpoint_tensor_indices_ size=%zu (total state_cache_names_=%zu)\n",
                   __func__, checkpoint_tensor_indices_.size(), state_cache_names_.size());
}

void HoumoQwen35StateCache::allocate_state_cache_entries(
    int seq_max, int n_prefill_batch, int n_decode_batch) {
    int size = (seq_max + n_prefill_batch - 1) / n_prefill_batch;
    size = std::max(size, n_decode_batch);
    for (int i = 0; i < size; ++i) {
        StateCacheEntry entry;
        entry.tensors.reserve(state_cache_names_.size());
        for (const auto &name : state_cache_names_) {
            auto cache = state_decode_model_->GetDevInput(name);
            if (i > 0) {
                cache = cache.Clone();
            }
            entry.tensors.push_back(cache);
        }
        state_cache_[i] = std::move(entry);
        state_cache_initialized_[i] = false;
    }
}

void HoumoQwen35StateCache::init_checkpoint_pool(int context_length,
                                                 bool multi_device) {
    (void)multi_device; // currently unused
    checkpoint_pools_.clear();
    if (checkpoint_max_count_ == 0) {
        LLAMA_LOG_INFO("%s: skip checkpoint pools init because checkpoint_max_count_=0\n",
                       __func__);
        return;
    }
    if (state_decode_model_ == nullptr || checkpoint_tensor_indices_.empty()) {
        return;
    }
    checkpoint_max_count_ = std::min(
        checkpoint_max_count_,
        static_cast<size_t>(context_length / checkpoint_min_interval_));
    double total_mb = 0.0;
    for (const auto &[seq_id, entry] : state_cache_) {
        std::vector<tcim::Tensor> prototypes;
        prototypes.reserve(checkpoint_tensor_indices_.size());
        for (size_t idx : checkpoint_tensor_indices_) {
            if (idx < entry.tensors.size()) {
                prototypes.push_back(entry.tensors[idx]);
            }
        }
        auto &pool = checkpoint_pools_[seq_id];
        pool.configure(checkpoint_min_interval_,
                       static_cast<int>(checkpoint_max_count_),
                       checkpoint_storage_);
        if (!pool.allocate(prototypes, true)) {
            checkpoint_pools_.erase(seq_id);
            LLAMA_LOG_WARN("%s: failed Qwen3.5 checkpoint pool seq_id=%d\n",
                           __func__, seq_id);
            continue;
        }
        total_mb += pool.allocated_mb();
    }
    if (checkpoint_pools_.empty()) {
        checkpoint_max_count_ = 0;
    }
    LLAMA_LOG_INFO("%s: pre-allocated Qwen3.5 state checkpoint pools "
                   "seqs=%zu slots_per_seq=%zu tensors_per_slot=%zu "
                   "storage=%s total_mb=%.2f\n",
                   __func__, checkpoint_pools_.size(), checkpoint_max_count_,
                   checkpoint_tensor_indices_.size(),
                   checkpoint_storage_ == HoumoCheckpointPool::Storage::kDevice
                       ? "device" : "host",
                   total_mb);
}

HoumoCheckpointPool *HoumoQwen35StateCache::find_checkpoint_pool(int seq_id) {
    auto it = checkpoint_pools_.find(seq_id);
    return it == checkpoint_pools_.end() ? nullptr : &it->second;
}

const HoumoCheckpointPool *HoumoQwen35StateCache::find_checkpoint_pool(
    int seq_id) const {
    auto it = checkpoint_pools_.find(seq_id);
    return it == checkpoint_pools_.end() ? nullptr : &it->second;
}

void HoumoQwen35StateCache::clear_all_checkpoints(int seq_id) {
    auto *pool = find_checkpoint_pool(seq_id);
    if (pool != nullptr) {
        pool->clear_seq(seq_id);
    }
}

void HoumoQwen35StateCache::set_state_cache_for_prefill(int seq_id) {
    if (state_prefill_model_ == nullptr) {
        return;
    }
    auto iter = state_cache_.find(seq_id);
    if (iter == state_cache_.end()) {
        return;
    }
    auto &tensors = iter->second.tensors;
    // If not initialized, restore zero state in-place.
    if (!state_cache_initialized_[seq_id]) {
        restore_zero_state(seq_id);
        state_cache_initialized_[seq_id] = true;
    }
    // Set inputs/outputs for the model - tensors are already in place
    for (size_t idx = 0; idx < tensors.size(); ++idx) {
        state_prefill_model_->SetDevInput(state_cache_names_[idx], tensors[idx]);
        if (idx < state_cache_output_names_.size() &&
            !state_cache_output_names_[idx].empty()) {
            state_prefill_model_->SetDevOutput(state_cache_output_names_[idx],
                                         tensors[idx]);
        }
    }
}

void HoumoQwen35StateCache::set_state_cache_for_decode(
    const std::vector<int> &seq_ids, int accept_pos, bool bind_outputs) {
    if (state_decode_model_ == nullptr) {
        return;
    }
    for (auto seq_id : seq_ids) {
        auto iter = state_cache_.find(seq_id);
        if (iter == state_cache_.end()) {
            continue;
        }
        auto &tensors = iter->second.tensors;
        for (size_t idx = 0; idx < tensors.size(); ++idx) {
            state_decode_model_->SetDevInput(state_cache_names_[idx], tensors[idx]);
            const std::string out_name = state_output_name(idx, accept_pos);
            if (bind_outputs && !out_name.empty()) {
                state_decode_model_->SetDevOutput(out_name, tensors[idx]);
            }
        }
    }
}

std::string HoumoQwen35StateCache::state_output_name(size_t idx, int accept_pos) const {
    if (idx < state_cache_split_output_names_.size() &&
        !state_cache_split_output_names_[idx].empty()) {
        const auto &split = state_cache_split_output_names_[idx];
        const int last = static_cast<int>(split.size()) - 1;
        int sel = accept_pos;
        if (sel < 0) sel = 0;
        if (sel > last) sel = last;
        return split[static_cast<size_t>(sel)];
    }
    if (idx < state_cache_output_names_.size()) {
        return state_cache_output_names_[idx];
    }
    return "";
}

int HoumoQwen35StateCache::bind_verify_state_to_cache(int seq_id, int accept_pos) {
    if (state_decode_model_ == nullptr) {
        return -1;
    }
    auto iter = state_cache_.find(seq_id);
    if (iter == state_cache_.end()) {
        return -1;
    }

    auto &tensors = iter->second.tensors;
    for (size_t idx = 0; idx < tensors.size(); ++idx) {
        const std::string out_name = state_output_name(idx, accept_pos);
        if (out_name.empty()) {
            continue;
        }

        auto out = state_decode_model_->GetDevOutput(out_name);
        if (state_decode_model_->SetDevInput(state_cache_names_[idx], out) != tcim::Status::OK) {
            LLAMA_LOG_ERROR("%s: failed to bind verify state output '%s' to input '%s'\n",
                            __func__, out_name.c_str(), state_cache_names_[idx].c_str());
            return -1;
        }
        tensors[idx] = out;
    }

    state_cache_initialized_[seq_id] = true;
    return 0;
}

void HoumoQwen35StateCache::save_state_checkpoint(int seq_id, int position) {
    auto *pool = find_checkpoint_pool(seq_id);
    if (pool == nullptr || !pool->enabled()) {
        return;
    }
    if (state_prefill_model_ == nullptr) {
        return;
    }
    if (position <= 0) {
        // Position 0 is represented by in-place zeroing of state tensors;
        // never allocate or save a checkpoint slot for it.
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
    auto &tensors = cache_iter->second.tensors;

    std::vector<tcim::Tensor> state_tensors;
    state_tensors.reserve(checkpoint_tensor_indices_.size());
    for (size_t ti : checkpoint_tensor_indices_) {
        if (ti < tensors.size()) {
            state_tensors.push_back(tensors[ti]);
        }
    }
    PERF_START(SAVE_CHECKPOINT_COPY);
    const bool saved = pool->save_full(seq_id, position, position,
                                       state_tensors);
    PERF_STOP(SAVE_CHECKPOINT_COPY);
    if (!saved) {
        return;
    }

    LLAMA_LOG_INFO("%s: saved checkpoint seq_id=%d pos=%d "
                   "state_tensors=%zu storage=%s active_ckpts=%zu\n",
                   __func__, seq_id, position, state_tensors.size(),
                   pool->checkpoint_storage() ==
                           HoumoCheckpointPool::Storage::kDevice
                       ? "device" : "host",
                   pool->active_count(seq_id));
}

int HoumoQwen35StateCache::find_and_restore_best_checkpoint(int seq_id,
                                                      int max_position) {
    auto *pool = find_checkpoint_pool(seq_id);
    if (pool == nullptr || !pool->enabled()) {
        return -1;
    }
    auto snap_iter = prefill_snapshots_.find(seq_id);
    if (snap_iter == prefill_snapshots_.end() || !snap_iter->second.valid) {
        return -1;
    }
    const size_t history_tokens =
        chunk_token_count(snap_iter->second.chunks);
    // Checkpoints are only saved at completed prefill boundaries. Limit the
    // lookup to history represented by the snapshot; replay code validates
    // the selected boundary when slicing mixed text/vision history.
    const int64_t search_max = std::min<int64_t>(
        max_position, static_cast<int64_t>(history_tokens));
    auto *slot = pool->find_nearest_before(seq_id, INT64_MAX, search_max + 1);
    if (slot == nullptr) {
        return -1;
    }

    auto cache_iter = state_cache_.find(seq_id);
    if (cache_iter == state_cache_.end()) {
        return -1;
    }
    auto &tensors = cache_iter->second.tensors;
    std::vector<tcim::Tensor> state_tensors;
    state_tensors.reserve(checkpoint_tensor_indices_.size());
    for (size_t ti : checkpoint_tensor_indices_) {
        if (ti < tensors.size()) {
            state_tensors.push_back(tensors[ti]);
        }
    }
    PERF_START(RESTORE_CHECKPOINT_COPY);
    const bool restored = pool->restore_full(*slot, state_tensors);
    PERF_STOP(RESTORE_CHECKPOINT_COPY);
    if (!restored) {
        return -1;
    }
    int restored_pos = static_cast<int>(slot->abs_end);

    // Erase stale checkpoints after restored_pos (free slots, no realloc).
    pool->release_after(seq_id, restored_pos);

    LLAMA_LOG_INFO("%s: restored checkpoint seq_id=%d pos=%d "
                   "reused_tokens=%d (max_allowed=%d)\n",
                   __func__, seq_id, restored_pos, restored_pos, max_position);
    return restored_pos;
}

void HoumoQwen35StateCache::restore_zero_state(int seq_id) {
    auto cache_iter = state_cache_.find(seq_id);
    if (cache_iter == state_cache_.end()) {
        LLAMA_LOG_WARN("%s: state cache not found for seq_id=%d\n",
                       __func__, seq_id);
        return;
    }
    auto &tensors = cache_iter->second.tensors;
    size_t restored = 0;
    for (size_t idx : checkpoint_tensor_indices_) {
        if (idx < tensors.size()) {
            zero_tensor(tensors[idx]);
            ++restored;
        }
    }
    LLAMA_LOG_INFO("%s: restored zero state seq_id=%d state_tensors=%zu\n",
                   __func__, seq_id, restored);
}

bool HoumoQwen35StateCache::trim_snapshot_to_kv_offset(
    int seq_id, int32_t kv_offset, bool release_checkpoints) {
    auto snap_iter = prefill_snapshots_.find(seq_id);
    if (snap_iter == prefill_snapshots_.end()) {
        return true;
    }

    auto &snapshot = snap_iter->second;
    const int32_t trim_pos = std::max<int32_t>(0, kv_offset);

    bool changed = false;
    const size_t trim_size = static_cast<size_t>(trim_pos);

    size_t position = 0;
    size_t keep_chunks = 0;
    for (; keep_chunks < snapshot.chunks.size(); ++keep_chunks) {
        auto &chunk = snapshot.chunks[keep_chunks];
        if (position + chunk.size() <= trim_size) {
            position += chunk.size();
            continue;
        }
        const size_t old_tokens = chunk.size();
        const size_t elems_per_token =
            chunk.is_embedding && old_tokens > 0
                ? chunk.embeddings.size() / old_tokens
                : 0;
        const size_t keep = trim_size > position ? trim_size - position : 0;
        if (chunk.is_embedding && keep > 0 && keep < old_tokens) {
            LLAMA_LOG_ERROR(
                "%s: trim position %d cuts vision chunk [%zu..%zu) "
                "for seq_id=%d\n",
                __func__, trim_pos, position, position + old_tokens, seq_id);
            return false;
        }
        chunk.n_tokens = keep;
        if (chunk.is_embedding) {
            chunk.embeddings.resize(keep * elems_per_token);
        } else {
            chunk.tokens.resize(keep);
        }
        if (keep > 0) {
            ++keep_chunks;
        }
        changed = true;
        break;
    }
    if (snapshot.chunks.size() > keep_chunks) {
        snapshot.chunks.resize(keep_chunks);
        changed = true;
    }

    if (release_checkpoints) {
        auto *pool = find_checkpoint_pool(seq_id);
        if (pool != nullptr) {
            pool->release_after(seq_id, trim_pos);
        }
    }

    // IMPORTANT: do NOT clamp last_processed_pos here.
    // last_processed_pos reflects where state tensors actually are.
    // memory_seq_rm() only truncates KV/memory, it does not rewind state
    // tensors. If we force last_processed_pos down to kv_offset, a later
    // prefill may mis-detect "continuation" and skip required rollback/
    // replay, causing duplicated-token state advance.

    if (changed) {
        const auto *pool = find_checkpoint_pool(seq_id);
        LLAMA_LOG_INFO(
            "%s: trimmed snapshot by boundary seq_id=%d valid_length=%d "
            "history_tokens=%zu ckpts=%zu state_pos=%d "
            "release_checkpoints=%d\n",
            __func__, seq_id, trim_pos, chunk_token_count(snapshot.chunks),
            pool == nullptr ? 0 : pool->active_count(seq_id),
            snapshot.last_processed_pos,
            release_checkpoints ? 1 : 0);
    }
    return true;
}

