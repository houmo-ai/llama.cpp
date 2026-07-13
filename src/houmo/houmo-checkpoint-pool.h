#pragma once

#include "houmo-llmodel.h"

#include <cstdint>
#include <vector>

class HoumoCheckpointPool {
  public:
    enum class Storage { kHost, kDevice };

    struct Slot {
        bool used = false;
        int seq_id = -1;
        int64_t abs_start = 0;
        int64_t abs_end = 0;
        int64_t last_access = 0;
        int64_t hit_count = 0;
        std::vector<tcim::Tensor> tensors;
    };

    static int env_int(const char *name, int default_value, int min_value,
                       int max_value);
    static Storage env_storage(const char *name, Storage default_value);

    void configure(int checkpoint_interval, int checkpoint_max_count,
                   Storage checkpoint_storage);
    void configure_from_env(int default_interval, int default_max_count,
                            int min_interval, int max_interval,
                            int max_count_limit, Storage default_storage);

    int checkpoint_interval() const { return checkpoint_interval_; }
    int checkpoint_max_count() const { return checkpoint_max_count_; }
    Storage checkpoint_storage() const { return checkpoint_storage_; }
    bool enabled() const { return checkpoint_max_count_ > 0 && !pool_.empty(); }
    size_t size() const { return pool_.size(); }

    bool allocate(const std::vector<tcim::Tensor> &prototypes,
                  bool force_host_for_compound = true);
    void clear();

    Slot *find_exact_covering(int seq_id, int64_t history_start,
                              int64_t history_end);
    Slot *find_nearest_before(int seq_id, int64_t history_start,
                              int64_t history_end);
    Slot &acquire_slot(int seq_id, int64_t abs_end,
                       bool protect_first_slot = false);

    bool save_full(int seq_id, int64_t abs_start, int64_t abs_end,
                   const std::vector<tcim::Tensor> &src_tensors,
                   bool protect_first_slot = false);
    bool restore_full(const Slot &slot,
                      std::vector<tcim::Tensor> &dst_tensors) const;

    void release_after(int seq_id, int64_t position);
    void clear_seq(int seq_id);

    int64_t latest_position(int seq_id) const;
    size_t active_count(int seq_id) const;
    double allocated_mb() const { return allocated_mb_; }

  private:
    int checkpoint_interval_ = 4096;
    int checkpoint_max_count_ = 0;
    Storage checkpoint_storage_ = Storage::kHost;
    std::vector<Slot> pool_;
    int64_t access_clock_ = 0;
    double allocated_mb_ = 0.0;
};
