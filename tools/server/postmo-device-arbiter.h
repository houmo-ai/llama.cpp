#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <set>
#include <utility>

namespace postmo {

struct device_arbiter_stats {
    bool busy = false;
    size_t waiting = 0;
    uint64_t acquisitions = 0;
    uint64_t wait_us_total = 0;
};

class device_arbiter {
public:
    class permit {
    public:
        permit() = default;
        ~permit();

        permit(const permit &) = delete;
        permit & operator=(const permit &) = delete;

        permit(permit && other) noexcept;
        permit & operator=(permit && other) noexcept;

        uint64_t wait_us() const;

    private:
        friend class device_arbiter;
        permit(device_arbiter * owner, uint64_t wait_us);

        device_arbiter * owner_ = nullptr;
        uint64_t wait_us_ = 0;
    };

    permit acquire(int32_t priority);
    device_arbiter_stats stats() const;

private:
    using ticket = std::pair<int32_t, uint64_t>;

    void release();

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::set<ticket> waiting_;
    bool busy_ = false;
    uint64_t next_ticket_ = 0;
    uint64_t acquisitions_ = 0;
    uint64_t wait_us_total_ = 0;
};

} // namespace postmo
