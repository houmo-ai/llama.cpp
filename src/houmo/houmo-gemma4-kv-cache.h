#pragma once

#include "houmo-checkpoint-pool.h"
#include "houmo-llmodel.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

// Gemma4 sliding KV snapshot/checkpoint pool.
//
// The Gemma4 prefill graph uses writable KV cache inputs.  Each sliding layer
// input has layout [1, 8, 1280, 256], where only the last 1024 history tokens
// are reused for the next prefill round and the final 256 slots are writable by
// the current prefill chunk.  This helper keeps a fixed-size pool of complete
// 1280-window snapshots and rebuilds the writable KV input from a snapshot when
// the requested history window is covered.
class HoumoGemma4KVCache {
  public:
    void init(const std::map<int, kv_cache_item> &kv_cache,
          int sliding_window = 1024, int prefill_length = 320);

    bool enabled() const { return !checkpoint_pools_.empty(); }
    int checkpoint_interval() const { return checkpoint_interval_; }
    int checkpoint_max_count() const { return checkpoint_max_count_; }

    // Rebuild the sliding KV tensors in @p item for target @p valid_length.
    // Returns true when a snapshot covered [valid_length-1024, valid_length).
    bool prepare_for_prefill(int seq_id, int64_t valid_length,
                             kv_cache_item &item);

    // Return the closest usable checkpoint position before @p valid_length
    // without modifying the live device KV. Returns -1 when unavailable.
    int64_t find_nearest_before(int seq_id, int64_t valid_length);

    // Restore the checkpoint whose end position is @p position.
    bool restore_at(int seq_id, int64_t position, kv_cache_item &item);

    // Save a post-run complete 1280-window snapshot.  Non-force saves are
    // throttled by LLAMA_CHECKPOINT_INTERVAL; force saves always refresh the
    // pool slot, e.g. at prefill end.
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
    static bool is_full_attention_layer(int layer_id);
    static bool is_sliding_kv_name(const std::string &name);
    static size_t dtype_size(tcim::DataType dtype);
    static bool copy_history_window_to_device(const tcim::Tensor &snapshot,
                          tcim::Tensor &dst,
                          int64_t src_offset,
                          int64_t dst_offset,
                          int64_t length);
    void discover_bindings(const kv_cache_item &item);
    bool allocate_pool(int seq_id, const kv_cache_item &item);
    HoumoCheckpointPool *find_pool(int seq_id);
    const HoumoCheckpointPool *find_pool(int seq_id) const;
    bool restore_from_slot(HoumoCheckpointPool::Slot &slot, int64_t valid_length,
                 kv_cache_item &item);
    bool should_save(int seq_id, int64_t abs_end, bool force) const;

  private:
    int sliding_window_ = 1024;
    int snapshot_size_ = 1344;
    int prefill_length_ = 320;
    int checkpoint_interval_ = 4096;
    int checkpoint_max_count_ = 0;
    HoumoCheckpointPool::Storage checkpoint_storage_ =
      HoumoCheckpointPool::Storage::kDevice;

    std::vector<TensorBinding> bindings_;
    std::map<int, HoumoCheckpointPool> checkpoint_pools_;
};
