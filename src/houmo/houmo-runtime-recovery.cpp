#include "houmo-runtime-recovery.h"

#include "llama-impl.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <map>
#include <mutex>
#include <string>
#include <thread>

namespace {

constexpr int64_t k_default_api_timeout_us = 5'000'000;
constexpr int64_t k_default_retry_sleep_us = 100'000;

int64_t steady_time_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

int64_t retry_sleep_us() {
    static const int64_t value = []() {
        const char * raw = std::getenv("HOUMO_RUN_RETRY_SLEEP_US");
        if (raw == nullptr || *raw == '\0') {
            return k_default_retry_sleep_us;
        }
        try {
            size_t parsed_chars = 0;
            const auto parsed = std::stoll(raw, &parsed_chars);
            if (parsed >= 0 && raw[parsed_chars] == '\0') {
                return static_cast<int64_t>(parsed);
            }
        } catch (...) {
        }
        LLAMA_LOG_WARN("invalid HOUMO_RUN_RETRY_SLEEP_US='%s', using %lld us\n", raw,
                       static_cast<long long>(k_default_retry_sleep_us));
        return k_default_retry_sleep_us;
    }();
    return value;
}

void configure_api_timeout_once() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        if (std::getenv("HDPL_API_TIMEOUT") != nullptr) {
            return;
        }
        const std::string value = std::to_string(k_default_api_timeout_us);
#if defined(_WIN32)
        if (_putenv_s("HDPL_API_TIMEOUT", value.c_str()) != 0) {
            LLAMA_LOG_WARN("failed to set default HDPL_API_TIMEOUT\n");
            return;
        }
#else
        if (setenv("HDPL_API_TIMEOUT", value.c_str(), 0) != 0) {
            LLAMA_LOG_WARN("failed to set default HDPL_API_TIMEOUT\n");
            return;
        }
#endif
        LLAMA_LOG_INFO("HDPL_API_TIMEOUT was unset; defaulting to %s us\n", value.c_str());
    });
}

struct device_recovery_state {
    std::atomic<int64_t> retry_after_us{ 0 };
};

std::mutex                                          registry_mutex;
std::map<int, std::weak_ptr<device_recovery_state>> registry;

std::shared_ptr<device_recovery_state> acquire_device_state(int device_id) {
    auto & weak = registry[device_id];
    if (auto existing = weak.lock()) {
        return existing;
    }
    auto created = std::make_shared<device_recovery_state>();
    weak         = created;
    return created;
}

void extend_retry_after(const std::shared_ptr<device_recovery_state> & state, int64_t retry_after_us) {
    int64_t current = state->retry_after_us.load(std::memory_order_relaxed);
    while (current < retry_after_us &&
           !state->retry_after_us.compare_exchange_weak(current, retry_after_us, std::memory_order_release,
                                                        std::memory_order_relaxed)) {
    }
}

const char * safe_operation(const char * operation) {
    return operation != nullptr ? operation : "tcim::Module::Run";
}

}  // namespace

struct houmo_runtime_recovery_handle {
    std::vector<std::shared_ptr<device_recovery_state>> devices;

    void wait_until_ready() const {
        while (true) {
            int64_t ready_at = 0;
            for (const auto & device : devices) {
                ready_at = std::max(ready_at, device->retry_after_us.load(std::memory_order_acquire));
            }
            const int64_t now = steady_time_us();
            if (ready_at <= now) {
                return;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(ready_at - now));
        }
    }

    void note_collateral_error() const {
        const int64_t ready_at = steady_time_us() + retry_sleep_us();
        for (const auto & device : devices) {
            extend_retry_after(device, ready_at);
        }
    }
};

houmo_runtime_recovery_handle_ptr houmo_runtime_recovery_acquire(const std::vector<int> & device_ids) {
    configure_api_timeout_once();

    std::vector<int> normalized = device_ids;
    if (normalized.empty()) {
        normalized.push_back(0);
    }
    std::sort(normalized.begin(), normalized.end());
    normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());

    auto handle = std::make_shared<houmo_runtime_recovery_handle>();
    handle->devices.reserve(normalized.size());
    std::lock_guard<std::mutex> lock(registry_mutex);
    for (int device_id : normalized) {
        handle->devices.push_back(acquire_device_state(device_id));
    }
    return handle;
}

tcim::Status houmo_run_with_retry(tcim::Module &                            module,
                                  const houmo_runtime_recovery_handle_ptr & recovery,
                                  const char *                              operation) {
    const auto handle  = recovery != nullptr ? recovery : houmo_runtime_recovery_acquire({ 0 });
    uint64_t   attempt = 0;
    while (true) {
        ++attempt;
        handle->wait_until_ready();

        tcim::Status status = tcim::Status::ERR_FATAL;
        try {
            status = module.Run(true);
        } catch (const std::exception & error) {
            LLAMA_LOG_ERROR("%s threw an exception: %s\n", safe_operation(operation), error.what());
            return tcim::Status::ERR_FATAL;
        } catch (...) {
            LLAMA_LOG_ERROR("%s threw an unknown exception\n", safe_operation(operation));
            return tcim::Status::ERR_FATAL;
        }

        switch (status) {
            case tcim::Status::OK:
                if (attempt > 1) {
                    LLAMA_LOG_INFO("%s recovered after %llu attempts\n", safe_operation(operation),
                                   static_cast<unsigned long long>(attempt));
                }
                return status;

            case tcim::Status::TIMEOUT:
                LLAMA_LOG_WARN("%s timed out on attempt %llu; TCIM reset completed, retrying\n",
                               safe_operation(operation), static_cast<unsigned long long>(attempt));
                continue;

            case tcim::Status::ERR_KERNEL:
            case tcim::Status::RESOURCE_EXHAUSTED:
            case tcim::Status::RECOVERING:
                handle->note_collateral_error();
                LLAMA_LOG_WARN("%s returned recoverable status %d on attempt %llu; retrying after %lld us\n",
                               safe_operation(operation), static_cast<int>(status),
                               static_cast<unsigned long long>(attempt), static_cast<long long>(retry_sleep_us()));
                continue;

            case tcim::Status::ERR_FATAL:
                LLAMA_LOG_ERROR("%s returned ERR_FATAL; automatic reset failed, run hm_smi -rd before retrying\n",
                                safe_operation(operation));
                return status;

            default:
                LLAMA_LOG_ERROR("%s failed with non-retryable status %d\n", safe_operation(operation),
                                static_cast<int>(status));
                return status;
        }
    }
}
