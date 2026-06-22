#include "houmo-gemma4-kv-cache.h"

#include "llama-impl.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>

namespace {
constexpr int64_t kUnsetSavedPos = -1;
}

int HoumoGemma4KVCache::env_int(const char *name, int default_value,
                                int min_value, int max_value) {
    const char *env = std::getenv(name);
    if (env == nullptr || *env == '\0') {
        return default_value;
    }
    errno = 0;
    char *end = nullptr;
    long value = std::strtol(env, &end, 10);
    if (errno == 0 && end != env && *end == '\0' &&
        value >= min_value && value <= max_value) {
        return static_cast<int>(value);
    }
    LLAMA_LOG_WARN("%s: invalid %s='%s', keep default=%d\n", __func__, name,
                   env, default_value);
    return default_value;
}

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
        LLAMA_LOG_WARN("%s: shape/dtype mismatch src_mem=%zu dst_mem=%zu length=%ld\n",
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
        LLAMA_LOG_WARN("%s: invalid copy range src_offset=%ld dst_offset=%ld length=%ld width=%ld\n",
                       __func__, src_offset, dst_offset, length, width);
        return false;
    }
    const size_t elem_size = dtype_size(dst.Info().DataType());
    const int64_t dim_after = static_cast<int64_t>(shape.back());
    int64_t outer = 1;
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
        outer *= static_cast<int64_t>(shape[i]);
    }
    const size_t row_bytes = static_cast<size_t>(length * dim_after) * elem_size;
    const size_t src_stride = static_cast<size_t>(width * dim_after) * elem_size;
    const size_t dst_stride = src_stride;
    const size_t src_col = static_cast<size_t>(src_offset * dim_after) * elem_size;
    const size_t dst_col = static_cast<size_t>(dst_offset * dim_after) * elem_size;

    const auto *src_ptr = static_cast<const uint8_t *>(snapshot.Buffer().Data());
    if (src_ptr == nullptr) {
        LLAMA_LOG_WARN("%s: host snapshot data is null\n", __func__);
        return false;
    }

    // Only the history window before valid_length is restored.  The writable
    // tail for the current 256-token prefill chunk is produced by graph Run(),
    // so there is no need to rebuild/upload a full 1280-token host image.
    for (int64_t i = 0; i < outer; ++i) {
        const auto *row_src = src_ptr + static_cast<size_t>(i) * src_stride + src_col;
        const size_t row_dst_offset = static_cast<size_t>(i) * dst_stride + dst_col;
        dst.Buffer().CopyFromHost(row_src, row_bytes, row_dst_offset);
    }
    LLAMA_LOG_DEBUG("%s: uploaded history window src_offset=%ld dst_offset=%ld length=%ld width=%ld outer=%ld row_bytes=%zu total_bytes=%zu\n",
                    __func__, src_offset, dst_offset, length, width, outer,
                    row_bytes, row_bytes * static_cast<size_t>(outer));
    return true;
}

bool HoumoGemma4KVCache::is_host_tensor(const tcim::Tensor &tensor) {
    return tensor.Device() == tcim::Device::CPU;
}

void HoumoGemma4KVCache::init(const std::map<int, kv_cache_item> &kv_cache,
                              int sliding_window) {
    sliding_window_ = sliding_window > 0 ? sliding_window : 1024;
    snapshot_size_ = sliding_window_ + 256;
    checkpoint_interval_ = env_int("LLAMA_CHECKPOINT_INTERVAL", 4096, 1,
                                   INT_MAX / 2);
    checkpoint_max_count_ = env_int("LLAMA_CHECKPOINT_MAX_COUNT", 10, 0, 256);
    checkpoint_storage_ = env_int("LLAMA_CHECKPOINT_STORAGE", 1, 0, 1) == 0
                              ? CheckpointStorage::kHost
                              : CheckpointStorage::kDevice;
    bindings_.clear();
    pool_.clear();
    last_saved_pos_.clear();
    access_clock_ = 0;

    if (checkpoint_max_count_ == 0 || kv_cache.empty()) {
        LLAMA_LOG_INFO("%s: Gemma4 KV checkpoint disabled max_count=%d\n",
                       __func__, checkpoint_max_count_);
        return;
    }
    discover_bindings(kv_cache.begin()->second);
    if (bindings_.empty()) {
        LLAMA_LOG_WARN("%s: no Gemma4 sliding KV binding discovered\n",
                       __func__);
        checkpoint_max_count_ = 0;
        return;
    }
    allocate_pool(kv_cache.begin()->second);
}

void HoumoGemma4KVCache::discover_bindings(const kv_cache_item &item) {
    for (size_t i = 0; i < item.names.size() && i < item.tensors.size(); ++i) {
        if (!is_sliding_kv_name(item.names[i])) {
            continue;
        }
        int layer_id = -1;
        parse_layer_id(item.names[i], layer_id);
        const auto &shape = item.tensors[i].Info().Shape();
        if (shape.size() < 3 || static_cast<int>(shape[shape.size() - 2]) != snapshot_size_) {
            LLAMA_LOG_WARN("%s: skip sliding KV name=%s unexpected shape width=%ld\n",
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
    LLAMA_LOG_INFO("%s: discovered Gemma4 sliding KV bindings=%zu\n",
                   __func__, bindings_.size());
}

void HoumoGemma4KVCache::allocate_pool(const kv_cache_item &item) {
    pool_.resize(static_cast<size_t>(checkpoint_max_count_));
    size_t allocated_slots = 0;
    double total_mb = 0.0;
    for (auto &slot : pool_) {
        slot.used = false;
        slot.seq_id = -1;
        slot.tensors.clear();
        slot.tensors.reserve(bindings_.size());
        bool failed = false;
        for (const auto &binding : bindings_) {
            const auto &src = item.tensors[binding.kv_index];
            tcim::Tensor checkpoint;
            if (checkpoint_storage_ == CheckpointStorage::kDevice) {
                auto src_copy = src;
                checkpoint = src_copy.Clone(false);
            } else {
                checkpoint = src.ToHost(false);
            }
            total_mb += checkpoint.Buffer().Size() / 1024.0 / 1024.0;
            if (checkpoint.GetInitStatus() != tcim::Status::OK) {
                failed = true;
                break;
            }
            slot.tensors.push_back(std::move(checkpoint));
        }
        if (failed) {
            slot.tensors.clear();
            break;
        }
        ++allocated_slots;
    }
    if (allocated_slots < pool_.size()) {
        pool_.resize(allocated_slots);
        checkpoint_max_count_ = static_cast<int>(allocated_slots);
    }
    if (pool_.empty()) {
        checkpoint_max_count_ = 0;
    }
    LLAMA_LOG_INFO(
        "%s: Gemma4 KV checkpoint pool slots=%zu tensors_per_slot=%zu "
        "storage=%s total_mb=%.2f interval=%d\n",
        __func__, pool_.size(), bindings_.size(),
        checkpoint_storage_ == CheckpointStorage::kDevice ? "device" : "host",
        total_mb, checkpoint_interval_);
}

HoumoGemma4KVCache::Slot *HoumoGemma4KVCache::find_covering_slot(
    int seq_id, int64_t history_start, int64_t history_end) {
    Slot *best = nullptr;
    int used_slots = 0;
    for (auto &slot : pool_) {
        if (!slot.used || slot.seq_id != seq_id) {
            continue;
        }
        ++used_slots;
        LLAMA_LOG_INFO("%s: candidate seq_id=%d snapshot=[%ld,%ld) request=[%ld,%ld) hits=%ld last_access=%ld\n",
                       __func__, seq_id, slot.abs_start, slot.abs_end,
                       history_start, history_end, slot.hit_count,
                       slot.last_access);
        // Exact-lookup by valid_length: require snapshot end to match request
        // valid_length (= history_end). If there is no exact match, caller
        // should fallback to restore_nearest_before().
        if (slot.abs_end == history_end && slot.abs_start <= history_start) {
            best = &slot;
            break;
        }
    }
    if (best != nullptr) {
        best->last_access = ++access_clock_;
        best->hit_count++;
        LLAMA_LOG_INFO("%s: exact hit seq_id=%d request=[%ld,%ld) snapshot=[%ld,%ld) hits=%ld\n",
                       __func__, seq_id, history_start, history_end,
                       best->abs_start, best->abs_end, best->hit_count);
    } else {
        LLAMA_LOG_INFO("%s: exact miss seq_id=%d request=[%ld,%ld) used_slots_for_seq=%d pool_slots=%zu\n",
                       __func__, seq_id, history_start, history_end,
                       used_slots, pool_.size());
    }
    return best;
}

HoumoGemma4KVCache::Slot *HoumoGemma4KVCache::find_nearest_slot_before(
    int seq_id, int64_t history_start, int64_t history_end) {
    Slot *best = nullptr;
    for (auto &slot : pool_) {
        if (!slot.used || slot.seq_id != seq_id) {
            continue;
        }
        if (slot.abs_end >= history_end || slot.abs_start >= history_start) {
            continue;
        }
        if (best == nullptr || slot.abs_end > best->abs_end) {
            best = &slot;
        }
    }
    if (best != nullptr) {
        best->last_access = ++access_clock_;
        LLAMA_LOG_INFO("%s: nearest seq_id=%d request=[%ld,%ld) snapshot=[%ld,%ld)\n",
                       __func__, seq_id, history_start, history_end,
                       best->abs_start, best->abs_end);
    } else {
        LLAMA_LOG_INFO("%s: no nearest seq_id=%d request=[%ld,%ld)\n",
                       __func__, seq_id, history_start, history_end);
    }
    return best;
}

bool HoumoGemma4KVCache::restore_from_slot(Slot &slot, int64_t valid_length,
                                           kv_cache_item &item) {
    const int64_t history_len = std::min<int64_t>(valid_length, sliding_window_);
    const int64_t history_start = valid_length - history_len;
    const int64_t history_end = valid_length;
    if (slot.abs_start > history_start || slot.abs_end < history_end) {
        LLAMA_LOG_WARN("%s: snapshot=[%ld,%ld) does not cover history=[%ld,%ld)\n",
                       __func__, slot.abs_start, slot.abs_end, history_start,
                       history_end);
        return false;
    }
    // Restore a complete snapshot window when we restore at checkpoint end
    // (exact hit or rollback hit). This avoids stale writable-tail content
    // contaminating the next prefill round.
    int64_t src_offset = history_start - slot.abs_start;
    int64_t dst_offset = 0;
    int64_t copy_len = history_len;
    if (valid_length == slot.abs_end) {
        copy_len = std::min<int64_t>(snapshot_size_, slot.abs_end - slot.abs_start);
        src_offset = 0;
        dst_offset = 0;
    }
    LLAMA_LOG_INFO("%s: restore offsets seq_id=%d snapshot=[%ld,%ld) src_offset=%ld dst_offset=%ld length=%ld\n",
                   __func__, slot.seq_id, slot.abs_start, slot.abs_end,
                   src_offset, dst_offset, copy_len);
    for (size_t bi = 0; bi < bindings_.size(); ++bi) {
        const size_t idx = bindings_[bi].kv_index;
        if (idx >= item.tensors.size()) {
            LLAMA_LOG_WARN("%s: restore binding=%zu kv_index=%zu out of range tensors=%zu name=%s\n",
                           __func__, bi, idx, item.tensors.size(),
                           bindings_[bi].name.c_str());
            return false;
        }
        if (!is_host_tensor(slot.tensors[bi])) {
            slot.tensors[bi].CopyTo(item.tensors[idx]);
            continue;
        }
        if (!copy_history_window_to_device(slot.tensors[bi], item.tensors[idx],
                                           src_offset, dst_offset, copy_len)) {
            LLAMA_LOG_WARN("%s: restore failed binding=%zu layer=%d kv_index=%zu name=%s\n",
                           __func__, bi, bindings_[bi].layer_id, idx,
                           bindings_[bi].name.c_str());
            return false;
        }
    }
    return true;
}

HoumoGemma4KVCache::Slot &HoumoGemma4KVCache::acquire_slot(int seq_id,
                                                           int64_t abs_end) {
    for (auto &slot : pool_) {
        if (slot.used && slot.seq_id == seq_id && slot.abs_end == abs_end) {
            return slot;
        }
    }
    for (auto &slot : pool_) {
        if (!slot.used) {
            return slot;
        }
    }
    auto victim = std::min_element(pool_.begin(), pool_.end(),
                                   [](const Slot &a, const Slot &b) {
                                       return a.last_access < b.last_access;
                                   });
    if (victim != pool_.end() && victim == pool_.begin() && pool_.size() > 1) {
        // The first checkpoint usually contains the system prompt in agent
        // scenarios. Keep it hotter than the rest so LRU prefers later slots.
        victim->last_access = ++access_clock_;
        victim = std::min_element(pool_.begin() + 1, pool_.end(),
                                  [](const Slot &a, const Slot &b) {
                                      return a.last_access < b.last_access;
                                  });
    }
    LLAMA_LOG_INFO("%s: evict Gemma4 KV checkpoint seq_id=%d range=[%ld,%ld)\n",
                   __func__, victim->seq_id, victim->abs_start,
                   victim->abs_end);
    return *victim;
}

bool HoumoGemma4KVCache::should_save(int seq_id, int64_t abs_end,
                                     bool force) const {
    if (!enabled() || abs_end <= 0) {
        return false;
    }
    if (force) {
        return true;
    }
    if (abs_end <= sliding_window_) {
        return false;
    }
    auto it = last_saved_pos_.find(seq_id);
    const int64_t last = it == last_saved_pos_.end() ? kUnsetSavedPos : it->second;
    if (last < 0) {
        return abs_end >= checkpoint_interval_;
    }
    return abs_end - last >= checkpoint_interval_;
}

bool HoumoGemma4KVCache::prepare_for_prefill(int seq_id, int64_t valid_length,
                                             kv_cache_item &item) {
    if (!enabled() || valid_length <= 0) {
        LLAMA_LOG_INFO("%s: skip restore seq_id=%d valid_length=%ld enabled=%d\n",
                       __func__, seq_id, valid_length, enabled() ? 1 : 0);
        return false;
    }
    const int64_t history_len = std::min<int64_t>(valid_length, sliding_window_);
    const int64_t history_start = valid_length - history_len;
    const int64_t history_end = valid_length;
    LLAMA_LOG_INFO("%s: try restore seq_id=%d valid_length=%ld history=[%ld,%ld) history_len=%ld sliding_window=%d bindings=%zu\n",
                   __func__, seq_id, valid_length, history_start, history_end,
                   history_len, sliding_window_, bindings_.size());
    Slot *slot = find_covering_slot(seq_id, history_start, history_end);
    if (slot == nullptr) {
        return false;
    }
    if (!restore_from_slot(*slot, valid_length, item)) {
        return false;
    }
    LLAMA_LOG_INFO("%s: restored Gemma4 sliding KV seq_id=%d history=[%ld,%ld) from snapshot=[%ld,%ld)\n",
                   __func__, seq_id, history_start, history_end,
                   slot->abs_start, slot->abs_end);
    return true;
}

int64_t HoumoGemma4KVCache::restore_nearest_before(int seq_id,
                                                   int64_t valid_length,
                                                   kv_cache_item &item) {
    if (!enabled() || valid_length <= 0) {
        return -1;
    }
    const int64_t history_len = std::min<int64_t>(valid_length, sliding_window_);
    const int64_t history_start = valid_length - history_len;
    const int64_t history_end = valid_length;
    Slot *slot = find_nearest_slot_before(seq_id, history_start, history_end);
    if (slot == nullptr) {
        return -1;
    }
    const int64_t rollback_pos = slot->abs_end;
    if (!restore_from_slot(*slot, rollback_pos, item)) {
        return -1;
    }
    LLAMA_LOG_INFO("%s: restored nearest Gemma4 sliding KV seq_id=%d rollback_pos=%ld from snapshot=[%ld,%ld)\n",
                   __func__, seq_id, rollback_pos, slot->abs_start,
                   slot->abs_end);
    return rollback_pos;
}

void HoumoGemma4KVCache::save_after_prefill(int seq_id, int64_t abs_end,
                                            kv_cache_item &item, bool force) {
    if (!should_save(seq_id, abs_end, force)) {
        auto it = last_saved_pos_.find(seq_id);
        const int64_t last = it == last_saved_pos_.end() ? kUnsetSavedPos : it->second;
        LLAMA_LOG_INFO("%s: skip Gemma4 sliding KV save seq_id=%d abs_end=%ld last_saved=%ld interval=%d force=%d enabled=%d\n",
                       __func__, seq_id, abs_end, last, checkpoint_interval_,
                       force ? 1 : 0, enabled() ? 1 : 0);
        return;
    }
    Slot &slot = acquire_slot(seq_id, abs_end);
    slot.used = true;
    slot.seq_id = seq_id;
    slot.abs_end = abs_end;
    slot.abs_start = std::max<int64_t>(0, abs_end - snapshot_size_);
    slot.last_access = ++access_clock_;
    slot.hit_count = 0;
    for (size_t bi = 0; bi < bindings_.size(); ++bi) {
        const size_t idx = bindings_[bi].kv_index;
        if (idx >= item.tensors.size()) {
            LLAMA_LOG_WARN("%s: skip save binding=%zu kv_index=%zu tensors=%zu name=%s\n",
                           __func__, bi, idx, item.tensors.size(),
                           bindings_[bi].name.c_str());
            continue;
        }
        // Device sliding KV -> host checkpoint tensor.
        LLAMA_LOG_DEBUG("%s: save tensor binding=%zu layer=%d kv_index=%zu name=%s bytes=%zu\n",
                       __func__, bi, bindings_[bi].layer_id, idx,
                       bindings_[bi].name.c_str(), item.tensors[idx].MemSize());
        item.tensors[idx].CopyTo(slot.tensors[bi]);
    }
    last_saved_pos_[seq_id] = abs_end;
    release_after(seq_id, abs_end);
    LLAMA_LOG_INFO("%s: saved Gemma4 sliding KV seq_id=%d range=[%ld,%ld) tensors=%zu force=%d\n",
                   __func__, seq_id, slot.abs_start, slot.abs_end,
                   bindings_.size(), force ? 1 : 0);
}

void HoumoGemma4KVCache::release_after(int seq_id, int64_t position) {
    for (auto &slot : pool_) {
        if (slot.used && slot.seq_id == seq_id && slot.abs_end > position) {
            slot.used = false;
            slot.seq_id = -1;
        }
    }
}

void HoumoGemma4KVCache::clear_seq(int seq_id) {
    for (auto &slot : pool_) {
        if (slot.used && slot.seq_id == seq_id) {
            slot.used = false;
            slot.seq_id = -1;
        }
    }
    last_saved_pos_.erase(seq_id);
}
