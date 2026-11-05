#include "postmo-device-arbiter.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace postmo {

device_arbiter::permit::permit(device_arbiter * owner, uint64_t wait_us)
    : owner_(owner), wait_us_(wait_us) {
}

device_arbiter::permit::~permit() {
    if (owner_ != nullptr) {
        owner_->release();
    }
}

device_arbiter::permit::permit(permit && other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)),
      wait_us_(std::exchange(other.wait_us_, 0)) {
}

device_arbiter::permit & device_arbiter::permit::operator=(permit && other) noexcept {
    if (this != &other) {
        if (owner_ != nullptr) {
            owner_->release();
        }
        owner_ = std::exchange(other.owner_, nullptr);
        wait_us_ = std::exchange(other.wait_us_, 0);
    }
    return *this;
}

uint64_t device_arbiter::permit::wait_us() const {
    return wait_us_;
}

device_arbiter::permit device_arbiter::acquire(int32_t priority) {
    if (priority < 0) {
        throw std::invalid_argument("device arbiter priority must be >= 0");
    }

    const auto started = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mutex_);
    const ticket current {priority, next_ticket_++};
    waiting_.insert(current);
    cv_.wait(lock, [&]() {
        return !busy_ && !waiting_.empty() && *waiting_.begin() == current;
    });

    waiting_.erase(current);
    busy_ = true;
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - started);
    const uint64_t wait_us = static_cast<uint64_t>(elapsed.count());
    ++acquisitions_;
    wait_us_total_ += wait_us;
    return permit(this, wait_us);
}

device_arbiter_stats device_arbiter::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {
        busy_,
        waiting_.size(),
        acquisitions_,
        wait_us_total_,
    };
}

void device_arbiter::release() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        busy_ = false;
    }
    cv_.notify_all();
}

} // namespace postmo
