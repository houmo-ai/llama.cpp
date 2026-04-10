#pragma once
#include <chrono>
#include <vector>
#include <iomanip>
#include <string>
#include <mutex>
#include <stdio.h>
// --- 全局性能仓库 (单例) ---
class PerfRegistry {
public:
    struct Metric {
        std::string tag;
        double* total_ns;
        uint64_t* count;
        double* current_ns;
        double* max_ns;
    };
    static PerfRegistry& Instance() {
        static PerfRegistry instance;
        return instance;
    }
    void Register(Metric m) {
        std::lock_guard<std::mutex> lock(mutex_);
        metrics_.push_back(m);
    }
    void PrintAll() {
        printf(
            "\n================ GLOBAL PERFORMANCE REPORT ================\n");
        printf("%-25s %-12s %-12s %-12s %-10s\n", "Tag", "Cur (ms)", "Avg (ms)",
                 "Max (ms)", "Count");
        for (const auto &m : metrics_) {
            double total = *m.total_ns;
            uint64_t cnt = *m.count;
            double cur = *m.current_ns;
            double maxv = *m.max_ns;
            double avg = (cnt > 0) ? (total / cnt) : 0;
            // 直接传入浮点数，不再使用 std::to_string
            printf("%-25s %-12.4f %-12.4f %-12.4f %-10llu\n",
                     m.tag.c_str(),
                     cur / 1e6,
                     avg / 1e6,
                     maxv / 1e6,
                     (unsigned long long)cnt);
            *m.total_ns = 0;
            *m.count = 0;
            *m.current_ns = 0;
            *m.max_ns = 0; // <-- 重置最大值
        }
    }
private:
    std::vector<Metric> metrics_;
    std::mutex mutex_;
};
// --- 宏实现 ---
#define PERF_START(tag) \
    static double total_ns_##tag = 0; \
    static uint64_t count_##tag = 0; \
    static double current_ns_##tag = 0; \
    static double max_ns_##tag = 0; \
    static bool registered_##tag = []() { \
        PerfRegistry::Instance().Register({#tag, &total_ns_##tag, &count_##tag, &current_ns_##tag, &max_ns_##tag}); \
        return true; \
    }(); \
    (void) registered_##tag; \
    auto start_##tag = std::chrono::high_resolution_clock::now();
#define PERF_STOP(tag) \
    { \
        auto end_##tag = std::chrono::high_resolution_clock::now(); \
        current_ns_##tag = std::chrono::duration_cast<std::chrono::nanoseconds>(end_##tag - start_##tag).count(); \
        total_ns_##tag += current_ns_##tag; \
        if (current_ns_##tag > max_ns_##tag) max_ns_##tag = current_ns_##tag; \
        count_##tag++; \
    }
#define PERF_REPORT() PerfRegistry::Instance().PrintAll()
