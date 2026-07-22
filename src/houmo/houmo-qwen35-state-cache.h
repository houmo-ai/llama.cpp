#pragma once

#include "houmo-checkpoint-pool.h"
#include "houmo-llmodel.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

// Owns Qwen3.5 linear-attention state tensors and the metadata used to
// checkpoint/replay them.
class HoumoQwen35StateCache {
  public:
    struct StateCacheEntry {
        std::vector<tcim::Tensor> tensors;
    };

    struct PrefillSnapshot {
        std::vector<PrefillChunk> chunks;
        int last_processed_pos = 0;
        bool valid = false;
    };

    void configure_state_cache(std::shared_ptr<tcim::Module> prefill,
                               std::shared_ptr<tcim::Module> decode,
                               int prefill_length);
    void initialize_state_cache(int seq_max, int n_prefill_batch,
                                int n_decode_batch, int context_length,
                                bool multi_device);
    bool enabled() const { return !state_cache_names_.empty(); }
    size_t state_count() const { return state_cache_names_.size(); }
    int decode_verify_seq() const { return decode_verify_seq_; }
    int checkpoint_min_interval() const { return checkpoint_min_interval_; }
    PrefillSnapshot &snapshot(int seq_id) { return prefill_snapshots_[seq_id]; }
    PrefillSnapshot *find_snapshot(int seq_id) {
        auto it = prefill_snapshots_.find(seq_id);
        return it == prefill_snapshots_.end() ? nullptr : &it->second;
    }
    const PrefillSnapshot *find_snapshot(int seq_id) const {
        auto it = prefill_snapshots_.find(seq_id);
        return it == prefill_snapshots_.end() ? nullptr : &it->second;
    }
    void mark_initialized(int seq_id, bool initialized) {
        state_cache_initialized_[seq_id] = initialized;
    }
    int state_position(int seq_id) const {
        const auto *snap = find_snapshot(seq_id);
        return snap != nullptr && snap->valid ? snap->last_processed_pos : -1;
    }
    bool should_save_checkpoint(int seq_id, int position,
                                bool force);
    void reset_for_fresh_prefill(int seq_id);
    void invalidate_snapshot(int seq_id, bool clear_checkpoints);
    void set_state_cache_for_prefill(int seq_id);
    void set_state_cache_for_decode(const std::vector<int> &seq_ids,
                                    int accept_pos = 0,
                                    bool bind_outputs = true);
    int bind_verify_state_to_cache(int seq_id, int accept_pos);
    void save_state_checkpoint(int seq_id, int position);
    int find_and_restore_best_checkpoint(int seq_id, int max_position);
    void restore_zero_state(int seq_id);
    bool trim_snapshot_to_kv_offset(int seq_id, int32_t kv_offset,
                                    bool release_checkpoints = true);
    void clear_all_checkpoints(int seq_id);

  private:
    void reset_state_cache_metadata();
    std::vector<std::string> collect_decode_output_names() const;
    std::vector<std::string> collect_state_split_outputs(
        const std::vector<std::string> &all_output_names,
        const std::string &bare_name) const;
    void discover_state_cache_bindings(
        const std::vector<std::string> &all_output_names);
    void init_checkpoint_tensor_indices();
    void allocate_state_cache_entries(int seq_max, int n_prefill_batch,
                                      int n_decode_batch);
    void init_state_cache_snapshots();
    void init_checkpoint_pool(int context_length, bool multi_device);
    HoumoCheckpointPool *find_checkpoint_pool(int seq_id);
    const HoumoCheckpointPool *find_checkpoint_pool(int seq_id) const;
    std::string state_output_name(size_t idx, int accept_pos) const;

    std::shared_ptr<tcim::Module> state_prefill_model_;
    std::shared_ptr<tcim::Module> state_decode_model_;
    std::vector<std::string> state_cache_names_;
    std::vector<std::string> state_cache_output_names_;
    std::vector<std::vector<std::string>> state_cache_split_output_names_;
    int decode_verify_seq_ = 1;
    std::map<int, StateCacheEntry> state_cache_;
    std::map<int, bool> state_cache_initialized_;
    std::vector<size_t> checkpoint_tensor_indices_;
    int checkpoint_min_interval_ = 4096;
    size_t checkpoint_max_count_ = 10;
    HoumoCheckpointPool::Storage checkpoint_storage_ = HoumoCheckpointPool::Storage::kHost;
    std::map<int, HoumoCheckpointPool> checkpoint_pools_;
    std::map<int, PrefillSnapshot> prefill_snapshots_;
};
