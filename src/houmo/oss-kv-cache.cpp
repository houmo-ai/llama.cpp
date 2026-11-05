#include "oss-kv-cache.h"

#include "llama-impl.h"

#include <algorithm>
#include <climits>

bool HoumoOSSKVCache::parse_layer_id(const std::string &name, int &layer_id) {
    const std::string prefix = "model_layers_";
    auto pos = name.find(prefix);
    if (pos == std::string::npos) {
        return false;
    }
    pos += prefix.size();
    const auto end = name.find("_self_attn_", pos);
    if (end == std::string::npos || end == pos) {
        return false;
    }
    try {
        layer_id = std::stoi(name.substr(pos, end - pos));
    } catch (...) {
        return false;
    }
    return true;
}

bool HoumoOSSKVCache::is_sliding_kv_name(const std::string &name) {
    int layer_id = -1;
    if (!parse_layer_id(name, layer_id) || layer_id % 2 != 0) {
        return false;
    }
    return name.find("_self_attn_kcache") != std::string::npos ||
           name.find("_self_attn_vcache") != std::string::npos;
}

void HoumoOSSKVCache::init(const std::map<int, kv_cache_item> &kv_cache,
                           int sliding_window, int prefill_length) {
    sliding_window_ = sliding_window > 0 ? sliding_window : 128;
    prefill_length_ = prefill_length > 0 ? prefill_length : 256;
    snapshot_size_ = sliding_window_ + prefill_length_;
    HoumoCheckpointPool config;
    config.configure_from_env(4096, 5, prefill_length_, INT_MAX / 2, 20,
                              HoumoCheckpointPool::Storage::kDevice);
    checkpoint_interval_ = config.checkpoint_interval();
    checkpoint_max_count_ = config.checkpoint_max_count();
    checkpoint_storage_ = config.checkpoint_storage();
    bindings_.clear();
    checkpoint_pools_.clear();

    if (checkpoint_max_count() == 0 || kv_cache.empty()) {
        LLAMA_LOG_INFO("%s: GPT-OSS KV checkpoint disabled max_count=%d\n",
                       __func__, checkpoint_max_count());
        return;
    }
    discover_bindings(kv_cache.begin()->second);
    if (bindings_.empty()) {
        LLAMA_LOG_WARN("%s: no GPT-OSS sliding KV binding discovered\n",
                       __func__);
        checkpoint_max_count_ = 0;
        return;
    }

    double total_mb = 0.0;
    for (const auto &[seq_id, item] : kv_cache) {
        if (allocate_pool(seq_id, item)) {
            total_mb += checkpoint_pools_.at(seq_id).allocated_mb();
        }
    }
    if (checkpoint_pools_.empty()) {
        checkpoint_max_count_ = 0;
        LLAMA_LOG_WARN("%s: failed to allocate any GPT-OSS KV checkpoint pool\n",
                       __func__);
        return;
    }
    LLAMA_LOG_INFO(
        "%s: GPT-OSS KV checkpoint pools=%zu slots_per_seq=%d "
        "tensors_per_slot=%zu storage=%s total_mb=%.2f interval=%d\n",
        __func__, checkpoint_pools_.size(), checkpoint_max_count(),
        bindings_.size(),
        checkpoint_storage_ == HoumoCheckpointPool::Storage::kDevice
            ? "device"
            : "host",
        total_mb, checkpoint_interval());
}

void HoumoOSSKVCache::discover_bindings(const kv_cache_item &item) {
    for (size_t i = 0; i < item.names.size() && i < item.tensors.size(); ++i) {
        if (!is_sliding_kv_name(item.names[i])) {
            continue;
        }
        int layer_id = -1;
        parse_layer_id(item.names[i], layer_id);
        const auto &shape = item.tensors[i].Info().Shape();
        if (shape.size() < 3 ||
            static_cast<int>(shape[shape.size() - 2]) != snapshot_size_) {
            LLAMA_LOG_WARN(
                "%s: skip GPT-OSS sliding KV name=%s unexpected width=%ld "
                "expected=%d\n",
                __func__, item.names[i].c_str(),
                shape.size() >= 2 ? shape[shape.size() - 2] : -1,
                snapshot_size_);
            continue;
        }
        bindings_.push_back({i, layer_id, item.names[i]});
    }
    std::sort(bindings_.begin(), bindings_.end(),
              [](const TensorBinding &a, const TensorBinding &b) {
                  if (a.layer_id != b.layer_id) {
                      return a.layer_id < b.layer_id;
                  }
                  return a.name < b.name;
              });
    LLAMA_LOG_INFO("%s: discovered GPT-OSS sliding KV bindings=%zu\n",
                   __func__, bindings_.size());
}

bool HoumoOSSKVCache::allocate_pool(int seq_id, const kv_cache_item &item) {
    std::vector<tcim::Tensor> prototypes;
    prototypes.reserve(bindings_.size());
    for (const auto &binding : bindings_) {
        if (binding.kv_index >= item.tensors.size()) {
            LLAMA_LOG_WARN(
                "%s: skip GPT-OSS checkpoint pool seq_id=%d invalid index=%zu "
                "name=%s\n",
                __func__, seq_id, binding.kv_index, binding.name.c_str());
            return false;
        }
        prototypes.push_back(item.tensors[binding.kv_index]);
    }
    auto &pool = checkpoint_pools_[seq_id];
    pool.configure(checkpoint_interval_, checkpoint_max_count_,
                   checkpoint_storage_);
    if (!pool.allocate(prototypes)) {
        checkpoint_pools_.erase(seq_id);
        LLAMA_LOG_WARN("%s: failed GPT-OSS checkpoint pool seq_id=%d\n",
                       __func__, seq_id);
        return false;
    }
    return true;
}

HoumoCheckpointPool *HoumoOSSKVCache::find_pool(int seq_id) {
    auto it = checkpoint_pools_.find(seq_id);
    return it == checkpoint_pools_.end() ? nullptr : &it->second;
}

const HoumoCheckpointPool *HoumoOSSKVCache::find_pool(int seq_id) const {
    auto it = checkpoint_pools_.find(seq_id);
    return it == checkpoint_pools_.end() ? nullptr : &it->second;
}

bool HoumoOSSKVCache::restore_from_slot(HoumoCheckpointPool::Slot &slot,
                                        int64_t valid_length,
                                        kv_cache_item &item) {
    if (valid_length != slot.abs_end) {
        LLAMA_LOG_WARN(
            "%s: checkpoint end mismatch valid_length=%ld snapshot=[%ld,%ld)\n",
            __func__, valid_length, slot.abs_start, slot.abs_end);
        return false;
    }
    for (size_t bi = 0; bi < bindings_.size(); ++bi) {
        const size_t idx = bindings_[bi].kv_index;
        if (idx >= item.tensors.size() || bi >= slot.tensors.size()) {
            LLAMA_LOG_WARN("%s: invalid GPT-OSS restore binding=%zu index=%zu\n",
                           __func__, bi, idx);
            return false;
        }
        slot.tensors[bi].CopyTo(item.tensors[idx]);
    }
    LLAMA_LOG_INFO(
        "%s: restored GPT-OSS checkpoint seq_id=%d snapshot=[%ld,%ld) "
        "tensors=%zu\n",
        __func__, slot.seq_id, slot.abs_start, slot.abs_end, bindings_.size());
    return true;
}

bool HoumoOSSKVCache::should_save(int seq_id, int64_t abs_end,
                                  bool force) const {
    const auto *pool = find_pool(seq_id);
    if (pool == nullptr || !pool->enabled() || abs_end <= 0) {
        return false;
    }
    if (force) {
        return true;
    }
    if (abs_end <= sliding_window_ + prefill_length_) {
        return false;
    }
    const int64_t last = pool->latest_position(seq_id);
    if (last < 0) {
        return abs_end >= checkpoint_interval();
    }
    return abs_end - last >= checkpoint_interval();
}

bool HoumoOSSKVCache::prepare_for_prefill(int seq_id, int64_t valid_length,
                                          kv_cache_item &item) {
    auto *pool = find_pool(seq_id);
    if (pool == nullptr || !pool->enabled() || valid_length <= 0) {
        return false;
    }
    const int64_t history_len =
        std::min<int64_t>(valid_length, sliding_window_);
    const int64_t history_start = valid_length - history_len;
    auto *slot = pool->find_exact_covering(seq_id, history_start, valid_length);
    return slot != nullptr && restore_from_slot(*slot, valid_length, item);
}

int64_t HoumoOSSKVCache::find_nearest_before(int seq_id,
                                             int64_t valid_length) {
    auto *pool = find_pool(seq_id);
    if (pool == nullptr || !pool->enabled() || valid_length <= 0) {
        return -1;
    }
    const int64_t history_len =
        std::min<int64_t>(valid_length, sliding_window_);
    auto *slot = pool->find_nearest_before(
        seq_id, valid_length - history_len, valid_length);
    return slot == nullptr ? -1 : slot->abs_end;
}

bool HoumoOSSKVCache::restore_at(int seq_id, int64_t position,
                                 kv_cache_item &item) {
    auto *pool = find_pool(seq_id);
    if (pool == nullptr || !pool->enabled() || position <= 0) {
        return false;
    }
    const int64_t history_len = std::min<int64_t>(position, sliding_window_);
    auto *slot = pool->find_exact_covering(seq_id, position - history_len,
                                           position);
    return slot != nullptr && restore_from_slot(*slot, position, item);
}

void HoumoOSSKVCache::save_after_prefill(int seq_id, int64_t abs_end,
                                         kv_cache_item &item, bool force) {
    if (!should_save(seq_id, abs_end, force)) {
        return;
    }
    auto *pool = find_pool(seq_id);
    if (pool == nullptr) {
        return;
    }
    std::vector<tcim::Tensor> src_tensors;
    src_tensors.reserve(bindings_.size());
    for (const auto &binding : bindings_) {
        if (binding.kv_index >= item.tensors.size()) {
            LLAMA_LOG_WARN("%s: invalid GPT-OSS save index=%zu name=%s\n",
                           __func__, binding.kv_index, binding.name.c_str());
            return;
        }
        src_tensors.push_back(item.tensors[binding.kv_index]);
    }
    const int64_t abs_start = std::max<int64_t>(0, abs_end - snapshot_size_);
    if (!pool->save_full(seq_id, abs_start, abs_end, src_tensors,
                         /*protect_first_slot=*/true)) {
        return;
    }
    release_after(seq_id, abs_end);
    LLAMA_LOG_INFO(
        "%s: saved GPT-OSS sliding KV seq_id=%d range=[%ld,%ld) tensors=%zu "
        "force=%d\n",
        __func__, seq_id, abs_start, abs_end, bindings_.size(), force ? 1 : 0);
}

void HoumoOSSKVCache::release_after(int seq_id, int64_t position) {
    auto *pool = find_pool(seq_id);
    if (pool != nullptr) {
        pool->release_after(seq_id, position);
    }
}

void HoumoOSSKVCache::clear_seq(int seq_id) {
    auto *pool = find_pool(seq_id);
    if (pool != nullptr) {
        pool->clear_seq(seq_id);
    }
}
