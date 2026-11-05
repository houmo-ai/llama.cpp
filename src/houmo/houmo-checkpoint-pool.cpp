#include "houmo-checkpoint-pool.h"

#include "llama-impl.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>

namespace {
constexpr int64_t kNoPosition = -1;
}

int HoumoCheckpointPool::env_int(const char *name, int default_value,
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

HoumoCheckpointPool::Storage HoumoCheckpointPool::env_storage(
    const char *name, Storage default_value) {
    const char *env = std::getenv(name);
    if (env == nullptr || *env == '\0') {
        return default_value;
    }
    if (std::strcmp(env, "1") == 0 || std::strcmp(env, "device") == 0) {
        return Storage::kDevice;
    }
    if (std::strcmp(env, "0") == 0 || std::strcmp(env, "host") == 0) {
        return Storage::kHost;
    }
    LLAMA_LOG_WARN("%s: invalid %s='%s', keep default=%s\n", __func__, name,
                   env, default_value == Storage::kDevice ? "device" : "host");
    return default_value;
}

void HoumoCheckpointPool::configure(int checkpoint_interval,
                                    int checkpoint_max_count,
                                    Storage checkpoint_storage) {
    checkpoint_interval_ = checkpoint_interval;
    checkpoint_max_count_ = checkpoint_max_count;
    checkpoint_storage_ = checkpoint_storage;
}

void HoumoCheckpointPool::configure_from_env(
    int default_interval, int default_max_count, int min_interval,
    int max_interval, int max_count_limit, Storage default_storage) {
    configure(env_int("LLAMA_CHECKPOINT_INTERVAL", default_interval,
                      min_interval, max_interval),
              env_int("LLAMA_CHECKPOINT_MAX_COUNT", default_max_count, 0,
                      max_count_limit),
              env_storage("LLAMA_CHECKPOINT_STORAGE", default_storage));
}

bool HoumoCheckpointPool::allocate(const std::vector<tcim::Tensor> &prototypes,
                                   bool force_host_for_compound) {
    pool_.clear();
    allocated_mb_ = 0.0;
    access_clock_ = 0;
    if (checkpoint_max_count_ <= 0 || prototypes.empty()) {
        checkpoint_max_count_ = 0;
        return false;
    }

    if (force_host_for_compound) {
        for (const auto &src : prototypes) {
            if (src.Device() == tcim::Device::COMPND) {
                checkpoint_storage_ = Storage::kHost;
                break;
            }
        }
    }

    pool_.resize(static_cast<size_t>(checkpoint_max_count_));
    size_t allocated_slots = 0;
    for (auto &slot : pool_) {
        slot.used = false;
        slot.seq_id = -1;
        slot.tensors.clear();
        slot.tensors.reserve(prototypes.size());
        bool failed = false;
        for (const auto &src : prototypes) {
            tcim::Tensor checkpoint;
            if (checkpoint_storage_ == Storage::kDevice) {
                auto src_copy = src;
                checkpoint = src_copy.Clone(false);
            } else {
                checkpoint = src.ToHost(false);
            }
            allocated_mb_ += checkpoint.Buffer().Size() / 1024.0 / 1024.0;
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
        return false;
    }
    return true;
}

void HoumoCheckpointPool::clear() {
    pool_.clear();
    checkpoint_max_count_ = 0;
    allocated_mb_ = 0.0;
    access_clock_ = 0;
}

HoumoCheckpointPool::Slot *HoumoCheckpointPool::find_exact_covering(
    int seq_id, int64_t history_start, int64_t history_end) {
    Slot *best = nullptr;
    for (auto &slot : pool_) {
        if (!slot.used || slot.seq_id != seq_id) {
            continue;
        }
        if (slot.abs_end == history_end && slot.abs_start <= history_start) {
            best = &slot;
            break;
        }
    }
    if (best != nullptr) {
        best->last_access = ++access_clock_;
        best->hit_count++;
    }
    return best;
}

HoumoCheckpointPool::Slot *HoumoCheckpointPool::find_nearest_before(
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
        best->hit_count++;
    }
    return best;
}

HoumoCheckpointPool::Slot &HoumoCheckpointPool::acquire_slot(
    int seq_id, int64_t abs_end, bool protect_first_slot) {
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
    auto begin = pool_.begin();
    auto victim = std::min_element(begin, pool_.end(),
                                   [](const Slot &a, const Slot &b) {
                                       return a.last_access < b.last_access;
                                   });
    if (protect_first_slot && victim != pool_.end() && victim == begin &&
        pool_.size() > 1) {
        victim->last_access = ++access_clock_;
        victim = std::min_element(begin + 1, pool_.end(),
                                  [](const Slot &a, const Slot &b) {
                                      return a.last_access < b.last_access;
                                  });
    }
    return *victim;
}

bool HoumoCheckpointPool::save_full(int seq_id, int64_t abs_start,
                                    int64_t abs_end,
                                    const std::vector<tcim::Tensor> &src_tensors,
                                    bool protect_first_slot) {
    if (!enabled() || src_tensors.empty()) {
        return false;
    }
    Slot &slot = acquire_slot(seq_id, abs_end, protect_first_slot);
    if (slot.tensors.size() != src_tensors.size()) {
        LLAMA_LOG_WARN("%s: tensor count mismatch src=%zu dst=%zu\n", __func__,
                       src_tensors.size(), slot.tensors.size());
        return false;
    }
    slot.used = true;
    slot.seq_id = seq_id;
    slot.abs_start = abs_start;
    slot.abs_end = abs_end;
    slot.last_access = ++access_clock_;
    slot.hit_count = 0;
    for (size_t i = 0; i < src_tensors.size(); ++i) {
        src_tensors[i].CopyTo(slot.tensors[i]);
    }
    return true;
}

bool HoumoCheckpointPool::restore_full(
    const Slot &slot, std::vector<tcim::Tensor> &dst_tensors) const {
    if (!slot.used || slot.tensors.size() != dst_tensors.size()) {
        LLAMA_LOG_WARN("%s: invalid restore slot used=%d src=%zu dst=%zu\n",
                       __func__, slot.used ? 1 : 0, slot.tensors.size(),
                       dst_tensors.size());
        return false;
    }
    for (size_t i = 0; i < dst_tensors.size(); ++i) {
        if (slot.tensors[i].Info().Shape() != dst_tensors[i].Info().Shape() ||
            slot.tensors[i].Info().DataType() != dst_tensors[i].Info().DataType()) {
            LLAMA_LOG_WARN("%s: tensor %zu info mismatch\n", __func__, i);
            return false;
        }
        slot.tensors[i].CopyTo(dst_tensors[i]);
    }
    return true;
}

void HoumoCheckpointPool::release_after(int seq_id, int64_t position) {
    for (auto &slot : pool_) {
        if (slot.used && slot.seq_id == seq_id && slot.abs_end > position) {
            slot.used = false;
            slot.seq_id = -1;
        }
    }
}

void HoumoCheckpointPool::clear_seq(int seq_id) {
    for (auto &slot : pool_) {
        if (slot.used && slot.seq_id == seq_id) {
            slot.used = false;
            slot.seq_id = -1;
        }
    }
}

int64_t HoumoCheckpointPool::latest_position(int seq_id) const {
    int64_t latest = kNoPosition;
    for (const auto &slot : pool_) {
        if (slot.used && slot.seq_id == seq_id) {
            latest = std::max(latest, slot.abs_end);
        }
    }
    return latest;
}

size_t HoumoCheckpointPool::active_count(int seq_id) const {
    return static_cast<size_t>(std::count_if(
        pool_.begin(), pool_.end(), [seq_id](const Slot &slot) {
            return slot.used && slot.seq_id == seq_id;
        }));
}
