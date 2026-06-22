#pragma once

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
              int sliding_window = 1024);

    bool enabled() const { return checkpoint_max_count_ > 0 && !pool_.empty(); }
    int checkpoint_interval() const { return checkpoint_interval_; }
    int checkpoint_max_count() const { return checkpoint_max_count_; }

    // Rebuild the sliding KV tensors in @p item for target @p valid_length.
    // Returns true when a snapshot covered [valid_length-1024, valid_length).
    bool prepare_for_prefill(int seq_id, int64_t valid_length,
                             kv_cache_item &item);

    // Restore the closest checkpoint before @p valid_length and return the
    // position that the restored device KV represents. Returns -1 when no
    // usable checkpoint is available.
    int64_t restore_nearest_before(int seq_id, int64_t valid_length,
                     kv_cache_item &item);

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

    struct Slot {
        bool used = false;
        int seq_id = -1;
        int64_t abs_start = 0;
        int64_t abs_end = 0;
        int64_t last_access = 0;
        int64_t hit_count = 0;
        std::vector<tcim::Tensor> tensors; // checkpoint tensors, one per binding
    };

    enum class CheckpointStorage {
      kHost,
      kDevice,
    };

    static int env_int(const char *name, int default_value, int min_value,
                       int max_value);
    static bool parse_layer_id(const std::string &name, int &layer_id);
    static bool is_full_attention_layer(int layer_id);
    static bool is_sliding_kv_name(const std::string &name);
    static size_t dtype_size(tcim::DataType dtype);
    static bool copy_history_window_to_device(const tcim::Tensor &snapshot,
                          tcim::Tensor &dst,
                          int64_t src_offset,
                          int64_t dst_offset,
                          int64_t length);
    static bool is_host_tensor(const tcim::Tensor &tensor);

    void discover_bindings(const kv_cache_item &item);
    void allocate_pool(const kv_cache_item &item);
    Slot *find_covering_slot(int seq_id, int64_t history_start,
                             int64_t history_end);
    Slot *find_nearest_slot_before(int seq_id, int64_t history_start,
                     int64_t history_end);
    bool restore_from_slot(Slot &slot, int64_t valid_length,
                 kv_cache_item &item);
    Slot &acquire_slot(int seq_id, int64_t abs_end);
    bool should_save(int seq_id, int64_t abs_end, bool force) const;

  private:
    int sliding_window_ = 1024;
    int snapshot_size_ = 1280;
    int checkpoint_interval_ = 4096;
    int checkpoint_max_count_ = 5;
    CheckpointStorage checkpoint_storage_ = CheckpointStorage::kDevice;

    std::vector<TensorBinding> bindings_;
    std::vector<Slot> pool_;
    std::map<int, int64_t> last_saved_pos_;
    int64_t access_clock_ = 0;
};
