#include "houmo-gemma4-kv-cache.h"

#include "llama-impl.h"

#include <algorithm>
#include <climits>
#include <cstdint>

bool HoumoGemma4KVCache::parse_layer_id(const std::string &name,
                                        int &layer_id) {
    const std::string prefix = "model_layers_";
    auto pos = name.find(prefix);
    if (pos == std::string::npos) {
        return false;
    }
    pos += prefix.size();
    auto end = name.find("_self_attn_", pos);
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

bool HoumoGemma4KVCache::is_full_attention_layer(int layer_id) {
    return layer_id == 5 || layer_id == 11 || layer_id == 17 ||
           layer_id == 23 || layer_id == 29;
}

bool HoumoGemma4KVCache::is_sliding_kv_name(const std::string &name) {
    int layer_id = -1;
    if (!parse_layer_id(name, layer_id)) {
        return false;
    }
    if (is_full_attention_layer(layer_id)) {
        return false;
    }
    return name.find("_self_attn_kcache") != std::string::npos ||
           name.find("_self_attn_vcache") != std::string::npos;
}

size_t HoumoGemma4KVCache::dtype_size(tcim::DataType dtype) {
    switch (dtype) {
    case tcim::DataType::INT8:
    case tcim::DataType::UINT8:
        return 1;
    case tcim::DataType::INT16:
    case tcim::DataType::UINT16:
    case tcim::DataType::FLOAT16:
        return 2;
    case tcim::DataType::INT32:
    case tcim::DataType::UINT32:
    case tcim::DataType::FLOAT32:
        return 4;
    default:
        return 0;
    }
}

bool HoumoGemma4KVCache::copy_history_window_to_device(
    const tcim::Tensor &snapshot, tcim::Tensor &dst, int64_t src_offset,
    int64_t dst_offset, int64_t length) {
    if (length <= 0) {
        return true;
    }
    if (snapshot.Info().Shape() != dst.Info().Shape() ||
        snapshot.Info().DataType() != dst.Info().DataType()) {
        LLAMA_LOG_WARN(
            "%s: shape/dtype mismatch src_mem=%zu dst_mem=%zu length=%ld\n",
            __func__, snapshot.MemSize(), dst.MemSize(), length);
        return false;
    }
    const auto &shape = dst.Info().Shape();
    if (shape.size() < 3) {
        LLAMA_LOG_WARN("%s: invalid rank=%zu\n", __func__, shape.size());
        return false;
    }
    const int64_t width = static_cast<int64_t>(shape[shape.size() - 2]);
    if (src_offset < 0 || dst_offset < 0 || src_offset + length > width ||
        dst_offset + length > width) {
        LLAMA_LOG_WARN("%s: invalid copy range src_offset=%ld dst_offset=%ld "
                       "length=%ld width=%ld\n",
                       __func__, src_offset, dst_offset, length, width);
        return false;
    }
    const size_t elem_size = dtype_size(dst.Info().DataType());
    const int64_t dim_after = static_cast<int64_t>(shape.back());
    int64_t outer = 1;
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
        outer *= static_cast<int64_t>(shape[i]);
    }
    const size_t row_bytes =
        static_cast<size_t>(length * dim_after) * elem_size;
    const size_t src_stride =
        static_cast<size_t>(width * dim_after) * elem_size;
    const size_t dst_stride = src_stride;
    const size_t src_col =
        static_cast<size_t>(src_offset * dim_after) * elem_size;
    const size_t dst_col =
        static_cast<size_t>(dst_offset * dim_after) * elem_size;

    const bool host_snapshot = snapshot.Buffer().Device() == tcim::Device::CPU;
    const auto *src_ptr =
        host_snapshot ? static_cast<const uint8_t *>(snapshot.Buffer().Data())
                      : nullptr;
    if (host_snapshot && src_ptr == nullptr) {
        LLAMA_LOG_WARN("%s: host snapshot data is null\n", __func__);
        return false;
    }

    for (int64_t i = 0; i < outer; ++i) {
        const size_t row_src_offset =
            static_cast<size_t>(i) * src_stride + src_col;
        const size_t row_dst_offset =
            static_cast<size_t>(i) * dst_stride + dst_col;
        tcim::Status status = tcim::Status::OK;
        if (host_snapshot) {
            status = dst.Buffer().CopyFromHost(src_ptr + row_src_offset,
                                               row_bytes, row_dst_offset);
        } else {
            status = snapshot.Buffer().CopyTo(dst.Buffer(), row_bytes,
                                              row_src_offset, row_dst_offset);
        }
        if (status != tcim::Status::OK) {
            LLAMA_LOG_WARN("%s: row copy failed row=%ld src_offset=%zu "
                           "dst_offset=%zu bytes=%zu storage=%s status=%d\n",
                           __func__, i, row_src_offset, row_dst_offset,
                           row_bytes, host_snapshot ? "host" : "device",
                           static_cast<int>(status));
            return false;
        }
    }
    LLAMA_LOG_DEBUG(
        "%s: restored history window src_offset=%ld dst_offset=%ld length=%ld "
        "width=%ld outer=%ld row_bytes=%zu total_bytes=%zu storage=%s\n",
        __func__, src_offset, dst_offset, length, width, outer, row_bytes,
        row_bytes * static_cast<size_t>(outer),
        host_snapshot ? "host" : "device");
    return true;
}

void HoumoGemma4KVCache::init(const std::map<int, kv_cache_item> &kv_cache,
                              int sliding_window, int prefill_length) {
    sliding_window_ = sliding_window > 0 ? sliding_window : 1024;
    const int chunk_length = prefill_length > 0 ? prefill_length : 256;
    prefill_length_ = chunk_length;
    snapshot_size_ = sliding_window_ + chunk_length;
    HoumoCheckpointPool config;
    config.configure_from_env(4096, 5, chunk_length, INT_MAX / 2, 20,
                              HoumoCheckpointPool::Storage::kDevice);
    checkpoint_interval_ = config.checkpoint_interval();
    checkpoint_max_count_ = config.checkpoint_max_count();
    checkpoint_storage_ = config.checkpoint_storage();
    bindings_.clear();
    checkpoint_pools_.clear();

    if (checkpoint_max_count() == 0 || kv_cache.empty()) {
        LLAMA_LOG_INFO("%s: Gemma4 KV checkpoint disabled max_count=%d\n",
                       __func__, checkpoint_max_count());
        return;
    }
    discover_bindings(kv_cache.begin()->second);
    if (bindings_.empty()) {
        LLAMA_LOG_WARN("%s: no Gemma4 sliding KV binding discovered\n",
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
        LLAMA_LOG_WARN("%s: failed to allocate any Gemma4 KV checkpoint pool\n",
                       __func__);
        return;
    }
    LLAMA_LOG_INFO(
        "%s: Gemma4 KV checkpoint pools=%zu slots_per_seq=%d "
        "tensors_per_slot=%zu storage=%s total_mb=%.2f interval=%d\n",
        __func__, checkpoint_pools_.size(), checkpoint_max_count(),
        bindings_.size(),
        checkpoint_storage_ == HoumoCheckpointPool::Storage::kDevice
            ? "device"
            : "host",
        total_mb, checkpoint_interval());
}

void HoumoGemma4KVCache::discover_bindings(const kv_cache_item &item) {
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
                "%s: skip sliding KV name=%s unexpected shape width=%ld\n",
                __func__, item.names[i].c_str(),
                shape.size() >= 2 ? shape[shape.size() - 2] : -1);
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
    LLAMA_LOG_INFO("%s: discovered Gemma4 sliding KV bindings=%zu\n", __func__,
                   bindings_.size());
}

bool HoumoGemma4KVCache::allocate_pool(int seq_id,
                                      const kv_cache_item &item) {
    std::vector<tcim::Tensor> prototypes;
    prototypes.reserve(bindings_.size());
    for (const auto &binding : bindings_) {
        if (binding.kv_index >= item.tensors.size()) {
            LLAMA_LOG_WARN(
                "%s: skip Gemma4 checkpoint pool seq_id=%d invalid index=%zu "
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
        LLAMA_LOG_WARN("%s: failed Gemma4 checkpoint pool seq_id=%d\n",
                       __func__, seq_id);
        return false;
    }
    return true;
}

HoumoCheckpointPool *HoumoGemma4KVCache::find_pool(int seq_id) {
    auto it = checkpoint_pools_.find(seq_id);
    return it == checkpoint_pools_.end() ? nullptr : &it->second;
}

const HoumoCheckpointPool *HoumoGemma4KVCache::find_pool(int seq_id) const {
    auto it = checkpoint_pools_.find(seq_id);
    return it == checkpoint_pools_.end() ? nullptr : &it->second;
}

bool HoumoGemma4KVCache::restore_from_slot(HoumoCheckpointPool::Slot &slot,
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
        if (idx >= item.tensors.size()) {
            LLAMA_LOG_WARN("%s: restore binding=%zu kv_index=%zu out of range "
                           "tensors=%zu name=%s\n",
                           __func__, bi, idx, item.tensors.size(),
                           bindings_[bi].name.c_str());
            return false;
        }
        if (bi >= slot.tensors.size()) {
            LLAMA_LOG_WARN("%s: restore binding=%zu out of range "
                           "snapshot_tensors=%zu name=%s\n",
                           __func__, bi, slot.tensors.size(),
                           bindings_[bi].name.c_str());
            return false;
        }
        slot.tensors[bi].CopyTo(item.tensors[idx]);
    }
    LLAMA_LOG_INFO("%s: restored exact checkpoint seq_id=%d snapshot=[%ld,%ld) "
                   "tensors=%zu\n",
                   __func__, slot.seq_id, slot.abs_start, slot.abs_end,
                   bindings_.size());
    return true;
}

bool HoumoGemma4KVCache::should_save(int seq_id, int64_t abs_end,
                                     bool force) const {
    const auto *pool = find_pool(seq_id);
    if (pool == nullptr || !pool->enabled() || abs_end <= 0) {
        return false;
    }
    if (force) {
        return true;
    }
    if (abs_end <= (sliding_window_ + prefill_length_)) {
        return false;
    }
    const int64_t last = pool->latest_position(seq_id);
    if (last < 0) {
        return abs_end >= checkpoint_interval();
    }
    return abs_end - last >= checkpoint_interval();
}

bool HoumoGemma4KVCache::prepare_for_prefill(int seq_id, int64_t valid_length,
                                             kv_cache_item &item) {
    auto *pool = find_pool(seq_id);
    if (pool == nullptr || !pool->enabled() || valid_length <= 0) {
        LLAMA_LOG_INFO(
            "%s: skip restore seq_id=%d valid_length=%ld enabled=%d\n",
            __func__, seq_id, valid_length,
            pool != nullptr && pool->enabled() ? 1 : 0);
        return false;
    }
    const int64_t history_len =
        std::min<int64_t>(valid_length, sliding_window_);
    const int64_t history_start = valid_length - history_len;
    const int64_t history_end = valid_length;
    LLAMA_LOG_INFO(
        "%s: try restore seq_id=%d valid_length=%ld history=[%ld,%ld) "
        "history_len=%ld sliding_window=%d bindings=%zu\n",
        __func__, seq_id, valid_length, history_start, history_end, history_len,
        sliding_window_, bindings_.size());
    HoumoCheckpointPool::Slot *slot = pool->find_exact_covering(
        seq_id, history_start, history_end);
    if (slot == nullptr) {
        return false;
    }
    if (!restore_from_slot(*slot, valid_length, item)) {
        return false;
    }
    LLAMA_LOG_INFO("%s: restored Gemma4 sliding KV seq_id=%d history=[%ld,%ld) "
                   "from snapshot=[%ld,%ld)\n",
                   __func__, seq_id, history_start, history_end,
                   slot->abs_start, slot->abs_end);
    return true;
}

int64_t HoumoGemma4KVCache::find_nearest_before(int seq_id,
                                                int64_t valid_length) {
    auto *pool = find_pool(seq_id);
    if (pool == nullptr || !pool->enabled() || valid_length <= 0) {
        return -1;
    }
    const int64_t history_len =
        std::min<int64_t>(valid_length, sliding_window_);
    const int64_t history_start = valid_length - history_len;
    HoumoCheckpointPool::Slot *slot = pool->find_nearest_before(
        seq_id, history_start, valid_length);
    if (slot == nullptr) {
        return -1;
    }
    return slot->abs_end;
}

bool HoumoGemma4KVCache::restore_at(int seq_id, int64_t position,
                                    kv_cache_item &item) {
    auto *pool = find_pool(seq_id);
    if (pool == nullptr || !pool->enabled() || position <= 0) {
        return false;
    }
    const int64_t history_len = std::min<int64_t>(position, sliding_window_);
    const int64_t history_start = position - history_len;
    HoumoCheckpointPool::Slot *slot =
        pool->find_exact_covering(seq_id, history_start, position);
    if (slot == nullptr || !restore_from_slot(*slot, position, item)) {
        LLAMA_LOG_WARN("%s: checkpoint unavailable seq_id=%d position=%ld\n",
                       __func__, seq_id, position);
        return false;
    }
    LLAMA_LOG_INFO("%s: restored Gemma4 sliding KV seq_id=%d position=%ld from "
                   "snapshot=[%ld,%ld)\n",
                   __func__, seq_id, position, slot->abs_start, slot->abs_end);
    return true;
}

void HoumoGemma4KVCache::save_after_prefill(int seq_id, int64_t abs_end,
                                            kv_cache_item &item, bool force) {
    if (!should_save(seq_id, abs_end, force)) {
        const auto *pool = find_pool(seq_id);
        const int64_t last = pool == nullptr ? -1 : pool->latest_position(seq_id);
        LLAMA_LOG_INFO("%s: skip Gemma4 sliding KV save seq_id=%d abs_end=%ld "
                       "last_saved=%ld interval=%d force=%d enabled=%d\n",
                       __func__, seq_id, abs_end, last, checkpoint_interval(),
                       force ? 1 : 0, enabled() ? 1 : 0);
        return;
    }
    auto *pool = find_pool(seq_id);
    if (pool == nullptr) {
        return;
    }

    std::vector<tcim::Tensor> src_tensors;
    src_tensors.reserve(bindings_.size());
    for (size_t bi = 0; bi < bindings_.size(); ++bi) {
        const size_t idx = bindings_[bi].kv_index;
        if (idx >= item.tensors.size()) {
            LLAMA_LOG_WARN(
                "%s: skip save binding=%zu kv_index=%zu tensors=%zu name=%s\n",
                __func__, bi, idx, item.tensors.size(),
                bindings_[bi].name.c_str());
            return;
        }
        LLAMA_LOG_DEBUG("%s: save tensor binding=%zu layer=%d kv_index=%zu "
                        "name=%s bytes=%zu\n",
                        __func__, bi, bindings_[bi].layer_id, idx,
                        bindings_[bi].name.c_str(),
                        item.tensors[idx].MemSize());
        src_tensors.push_back(item.tensors[idx]);
    }

    const int64_t abs_start = std::max<int64_t>(0, abs_end - snapshot_size_);
    if (!pool->save_full(seq_id, abs_start, abs_end, src_tensors,
                         /*protect_first_slot=*/true)) {
        return;
    }
    release_after(seq_id, abs_end);
    LLAMA_LOG_INFO("%s: saved Gemma4 sliding KV seq_id=%d range=[%ld,%ld) "
                   "tensors=%zu force=%d\n",
                   __func__, seq_id, abs_start, abs_end, bindings_.size(),
                   force ? 1 : 0);
}

void HoumoGemma4KVCache::release_after(int seq_id, int64_t position) {
    auto *pool = find_pool(seq_id);
    if (pool != nullptr) {
        pool->release_after(seq_id, position);
    }
}

void HoumoGemma4KVCache::clear_seq(int seq_id) {
    auto *pool = find_pool(seq_id);
    if (pool != nullptr) {
        pool->clear_seq(seq_id);
    }
}
