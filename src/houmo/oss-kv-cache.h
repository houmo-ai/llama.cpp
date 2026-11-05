#pragma once

#include "houmo-checkpoint-pool.h"
#include "houmo-llmodel.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

// GPT-OSS sliding KV snapshot/checkpoint pool.
//
// Even-numbered transformer layers use sliding attention. Their writable KV
// inputs have layout [1, 8, sliding_window + prefill_length, head_dim], e.g.
// [1, 8, 384, 64] for a 128-token window and a 256-token prefill chunk.
class HoumoOSSKVCache {
  public:
    void init(const std::map<int, kv_cache_item> &kv_cache,
              int sliding_window = 128, int prefill_length = 256);

    bool enabled() const { return !checkpoint_pools_.empty(); }
    int checkpoint_interval() const { return checkpoint_interval_; }
    int checkpoint_max_count() const { return checkpoint_max_count_; }

    bool prepare_for_prefill(int seq_id, int64_t valid_length,
                             kv_cache_item &item);
    int64_t find_nearest_before(int seq_id, int64_t valid_length);
    bool restore_at(int seq_id, int64_t position, kv_cache_item &item);
    void save_after_prefill(int seq_id, int64_t abs_end, kv_cache_item &item,
                            bool force = false);
    void release_after(int seq_id, int64_t position);
    void clear_seq(int seq_id);

  private:
    struct TensorBinding {
        size_t kv_index = 0;
        int layer_id = -1;
        std::string name;
    };

    static bool parse_layer_id(const std::string &name, int &layer_id);
    static bool is_sliding_kv_name(const std::string &name);
    void discover_bindings(const kv_cache_item &item);
    bool allocate_pool(int seq_id, const kv_cache_item &item);
    HoumoCheckpointPool *find_pool(int seq_id);
    const HoumoCheckpointPool *find_pool(int seq_id) const;
    bool restore_from_slot(HoumoCheckpointPool::Slot &slot,
                           int64_t valid_length, kv_cache_item &item);
    bool should_save(int seq_id, int64_t abs_end, bool force) const;

  private:
    int sliding_window_ = 128;
    int snapshot_size_ = 384;
    int prefill_length_ = 256;
    int checkpoint_interval_ = 4096;
    int checkpoint_max_count_ = 0;
    HoumoCheckpointPool::Storage checkpoint_storage_ =
      HoumoCheckpointPool::Storage::kDevice;
    std::vector<TensorBinding> bindings_;
    std::map<int, HoumoCheckpointPool> checkpoint_pools_;
};
