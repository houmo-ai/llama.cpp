#pragma once

#include "log.h"
#include <chrono>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#endif
#include <string>
#include "nlohmann/json.hpp"
#include "server-common.h"
#include "server-context.h"
using json = nlohmann::ordered_json;

class InfoPrinter {
  private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    TimePoint start_time_;
    bool is_running_;

  private:
    // 开始计时
    void start() {
        start_time_ = Clock::now();
        is_running_ = true;
    }
    // 获取从 start() 到现在的 elapsed 时间，单位为秒
    double elapsed_seconds() const {
        if (!is_running_) {
            return 0.0;
        }
        TimePoint current_time = Clock::now();
        return std::chrono::duration<double>(current_time - start_time_)
            .count();
    }

    // 获取从 start() 到现在的 elapsed 时间，单位为毫秒
    long long elapsed_milliseconds() const {
        if (!is_running_) {
            return 0;
        }
        TimePoint current_time = Clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   current_time - start_time_)
            .count();
    }

  public:
    InfoPrinter() : is_running_(false) { start(); }
    void print_process_memory_info() {
        int64_t virtual_size = 0;
        int64_t virtual_peak = 0;
        int64_t working_set_size = 0;
        int64_t working_set_peak = 0;
#if defined(_WIN32)
        // --- Windows 平台实现 ---
        PROCESS_MEMORY_COUNTERS_EX pmc;
        pmc.cb = sizeof(pmc);

        if (!GetProcessMemoryInfo(
                GetCurrentProcess(),
                reinterpret_cast<PPROCESS_MEMORY_COUNTERS>(&pmc), pmc.cb)) {
            throw std::runtime_error("GetProcessMemoryInfo failed");
        }
        // Convert to MB
        virtual_size = pmc.PrivateUsage / (1024 * 1024);
        virtual_peak = pmc.PeakPagefileUsage / (1024 * 1024);
        working_set_size = pmc.WorkingSetSize / (1024 * 1024);
        working_set_peak = pmc.PeakWorkingSetSize / (1024 * 1024);

#elif defined(__linux__)
        // --- Linux 平台实现 ---
        std::ifstream proc_file("/proc/self/status");
        if (!proc_file.is_open()) {
            throw std::runtime_error("Failed to open /proc/self/status");
        }

        std::string line;
        while (std::getline(proc_file, line)) {
            std::istringstream iss(line);
            std::string key;
            iss >> key;

            // Linux 下 /proc/self/status 的单位通常是 KB, 转换为 MB
            if (key == "VmSize:") {
                iss >> virtual_size;
                virtual_size /= 1024;
            } else if (key == "VmPeak:") {
                iss >> virtual_peak;
                virtual_peak /= 1024;
            } else if (key == "VmRSS:") {
                iss >> working_set_size;
                working_set_size /= 1024;
            } else if (key == "VmHWM:") {
                iss >> working_set_peak;
                working_set_peak /= 1024;
            }
        }

        if (virtual_size == 0 && working_set_size == 0) {
            return;
        }
#else
#error "Unsupported platform"
#endif
        LOG_INF("\n===========up info=============================\n"
                "=VmPeak: %lu MB\n"
                "=VmSize: %lu MB\n"
                "=VmHWM: %lu MB\n"
                "=VmRSS: %lu MB\n"
                "=UpTime: %f seconds\n"
                "===========up info=============================\n",
                virtual_peak, virtual_size , working_set_peak, working_set_size,
                elapsed_seconds());
    }
    void print_model_info(server_context_meta &meta) {
        LOG_INF(
            "\n==========model info===========================\n"
            "=n_ctx_length = %5d\n"
            "=n_ctx_train  = %5d\n"
            "=n_embd       = %5d\n"
            "=allow_image  = %5d\n"
            "=allow_audio  = %5d\n"
            "=vision_width = %5d\n"
            "=vision_height= %5d\n"
            "=model_desc   = %s\n"
            "==========model info===========================\n",
            meta.model_n_ctx,
            meta.model_n_ctx_train,
            meta.model_n_embd_inp,
            meta.allow_image,
            meta.allow_audio,
            meta.vision_width,
            meta.vision_height,
            meta.model_desc.c_str());
    }
};
