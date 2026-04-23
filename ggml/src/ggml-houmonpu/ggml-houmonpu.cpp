#include "ggml-houmonpu.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <cstdlib>

#ifndef _WIN32
#include <dlfcn.h>
#endif

#ifdef _WIN32
#include <windows.h>
#endif

// ==================== HoumoNPU SDK 动态加载 ====================

// 从 hm_sys.h 导入的类型和函数签名
struct hm_device_info {
    uint32_t num_devices;
    uint32_t device_ids[32];
};

struct hm_mem_info {
    uint32_t mem_total;
    uint32_t mem_used;
    uint32_t mem_avail;
};

typedef uint32_t (*hm_sys_get_device_info_t)(struct hm_device_info *info);
typedef int (*hm_sys_get_device_name_t)(int dev_index, char name[], int len);
typedef int (*hm_sys_get_ddr_size_t)(int dev_index, uint64_t *ddr_size);
typedef int (*hm_sys_get_mem_info_t)(int dev_index, struct hm_mem_info *mem_info);
typedef int (*hm_sys_get_computing_power_t)(int dev_index);
typedef int (*hm_sys_get_core_count_t)(int dev_index);
typedef float (*hm_sys_get_ipu_utili_rate_t)(int dev_index);
typedef int (*hm_sys_get_temperature_t)(int dev_index, float * temperature);
typedef int (*hm_sys_get_driver_version_t)(char *version, size_t len);
typedef int (*hm_sys_get_device_version_t)(int dev_index, char *version, size_t len);

struct houmonpu_sdk {
    void* handle = nullptr;
    hm_sys_get_device_info_t get_device_info = nullptr;
    hm_sys_get_device_name_t get_device_name = nullptr;
    hm_sys_get_ddr_size_t get_ddr_size = nullptr;
    hm_sys_get_mem_info_t get_mem_info = nullptr;
    hm_sys_get_computing_power_t get_computing_power = nullptr;
    hm_sys_get_core_count_t get_core_count = nullptr;
    hm_sys_get_ipu_utili_rate_t get_ipu_utili_rate = nullptr;
    hm_sys_get_temperature_t get_temperature = nullptr;
    hm_sys_get_driver_version_t get_driver_version = nullptr;
    hm_sys_get_device_version_t get_device_version = nullptr;
    bool initialized = false;

    static const char* get_library_name() {
#ifdef _WIN32
        return "libhal_xh2a.dll";
#else
        return "libhal_xh2a.so";
#endif
        }

    bool load() {
        if (handle) return true;

        GGML_LOG_INFO("[HoumoNPU] Loading HoumoNPU SDK...\n");

#ifdef _WIN32
        // Windows: 首先尝试 HOUMO_SDK_PATH 环境变量
        const char* houmo_sdk_path = getenv("HOUMO_SDK_PATH");
        if (houmo_sdk_path) {
            std::string path = std::string(houmo_sdk_path) + "/hal/lib/" + get_library_name();
            GGML_LOG_INFO("[HoumoNPU] Trying to load: %s\n", path.c_str());
            handle = (void*)LoadLibraryA(path.c_str());
            if (handle) {
                GGML_LOG_INFO("[HoumoNPU] Successfully loaded: %s\n", path.c_str());
            } else {
                GGML_LOG_INFO("[HoumoNPU] Failed to load: %s - Error %lu\n", path.c_str(), GetLastError());
            }
        }

        // 尝试其他环境变量
        if (!handle) {
            const char* env_paths[] = {
                getenv("TCIM_RUNTIME_PATH"),
                ".",
                nullptr
            };

            for (int i = 0; env_paths[i] != nullptr; i++) {
                if (!env_paths[i]) continue;

                std::string path = std::string(env_paths[i]) + "/" + get_library_name();
                GGML_LOG_INFO("[HoumoNPU] Trying to load from env: %s\n", path.c_str());
                handle = (void*)LoadLibraryA(path.c_str());
                if (handle) {
                    GGML_LOG_INFO("[HoumoNPU] Successfully loaded: %s\n", path.c_str());
                    break;
                } else {
                    GGML_LOG_INFO("[HoumoNPU] Failed to load from env: %s - Error %lu\n", path.c_str(), GetLastError());
                }
            }
        }

        // 尝试系统路径
        if (!handle) {
            GGML_LOG_INFO("[HoumoNPU] Trying to load from system path: %s\n", get_library_name());
            handle = (void*)LoadLibraryA(get_library_name());
            if (handle) {
                GGML_LOG_INFO("[HoumoNPU] Successfully loaded from system path\n");
            } else {
                GGML_LOG_INFO("[HoumoNPU] Failed to load from system path: Error %lu\n", GetLastError());
            }
        }
#else
        // Linux: 标准 SDK 路径
        const char* sdk_paths[] = {
            "/usr/local/houmo-sdk/hal/lib",
            "/usr/local/houmo-drv/hal/lib",
            nullptr
        };

        for (int i = 0; sdk_paths[i] != nullptr; i++) {
            std::string path = std::string(sdk_paths[i]) + "/" + get_library_name();
            GGML_LOG_INFO("[HoumoNPU] Trying to load: %s\n", path.c_str());
            handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
            if (handle) {
                GGML_LOG_INFO("[HoumoNPU] Successfully loaded: %s\n", path.c_str());
                break;
            } else {
                GGML_LOG_INFO("[HoumoNPU] Failed to load: %s - %s\n", path.c_str(), dlerror());
            }
        }

        // 尝试环境变量
        if (!handle) {
            const char* env_paths[] = {
                getenv("HOUMO_SDK_PATH"),
                getenv("HOUMO_DRV_PATH"),
                getenv("TCIM_RUNTIME_PATH"),
                ".",
                nullptr
            };

            for (int i = 0; env_paths[i] != nullptr; i++) {
                if (!env_paths[i]) continue;

                std::string path = std::string(env_paths[i]) + "/" + get_library_name();
                GGML_LOG_INFO("[HoumoNPU] Trying to load from env: %s\n", path.c_str());
                handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
                if (handle) {
                    GGML_LOG_INFO("[HoumoNPU] Successfully loaded: %s\n", path.c_str());
                    break;
                } else {
                    GGML_LOG_INFO("[HoumoNPU] Failed to load from env: %s - %s\n", path.c_str(), dlerror());
                }
            }
        }

        // 尝试系统路径
        if (!handle) {
            GGML_LOG_INFO("[HoumoNPU] Trying to load from system path: %s\n", get_library_name());
            handle = dlopen(get_library_name(), RTLD_LAZY | RTLD_LOCAL);
            if (handle) {
                GGML_LOG_INFO("[HoumoNPU] Successfully loaded from system path\n");
            } else {
                GGML_LOG_INFO("[HoumoNPU] Failed to load from system path: %s\n", dlerror());
            }
        }
#endif

        if (!handle) {
            return false;
        }

        // 加载函数
#ifdef _WIN32
        get_device_info = (hm_sys_get_device_info_t)GetProcAddress((HMODULE)handle, "hm_sys_get_device_info");
        get_device_name = (hm_sys_get_device_name_t)GetProcAddress((HMODULE)handle, "hm_sys_get_device_name");
        get_ddr_size = (hm_sys_get_ddr_size_t)GetProcAddress((HMODULE)handle, "hm_sys_get_ddr_size");
        get_mem_info = (hm_sys_get_mem_info_t)GetProcAddress((HMODULE)handle, "hm_sys_get_mem_info");
        get_computing_power = (hm_sys_get_computing_power_t)GetProcAddress((HMODULE)handle, "hm_sys_get_computing_power");
        get_core_count = (hm_sys_get_core_count_t)GetProcAddress((HMODULE)handle, "hm_sys_get_core_count");
        get_ipu_utili_rate = (hm_sys_get_ipu_utili_rate_t)GetProcAddress((HMODULE)handle, "hm_sys_get_ipu_utili_rate");
        get_temperature = (hm_sys_get_temperature_t)GetProcAddress((HMODULE)handle, "hm_sys_get_temperature");
        get_driver_version = (hm_sys_get_driver_version_t)GetProcAddress((HMODULE)handle, "hm_sys_get_driver_version");
        get_device_version = (hm_sys_get_device_version_t)GetProcAddress((HMODULE)handle, "hm_sys_get_device_version");
    #else
        get_device_info = (hm_sys_get_device_info_t)dlsym(handle, "hm_sys_get_device_info");
        get_device_name = (hm_sys_get_device_name_t)dlsym(handle, "hm_sys_get_device_name");
        get_ddr_size = (hm_sys_get_ddr_size_t)dlsym(handle, "hm_sys_get_ddr_size");
        get_mem_info = (hm_sys_get_mem_info_t)dlsym(handle, "hm_sys_get_mem_info");
        get_computing_power = (hm_sys_get_computing_power_t)dlsym(handle, "hm_sys_get_computing_power");
        get_core_count = (hm_sys_get_core_count_t)dlsym(handle, "hm_sys_get_core_count");
        get_ipu_utili_rate = (hm_sys_get_ipu_utili_rate_t)dlsym(handle, "hm_sys_get_ipu_utili_rate");
        get_temperature = (hm_sys_get_temperature_t)dlsym(handle, "hm_sys_get_temperature");
        get_driver_version = (hm_sys_get_driver_version_t)dlsym(handle, "hm_sys_get_driver_version");
        get_device_version = (hm_sys_get_device_version_t)dlsym(handle, "hm_sys_get_device_version");
    #endif

        if (!get_device_info || !get_device_name || !get_ddr_size || !get_mem_info) {
#ifdef _WIN32
            FreeLibrary((HMODULE)handle);
#else
            dlclose(handle);
#endif
            handle = nullptr;
            return false;
        }

        return true;
    }

    void unload() {
        if (handle) {
#ifdef _WIN32
            FreeLibrary((HMODULE)handle);
#else
            dlclose(handle);
#endif
            handle = nullptr;
            get_device_info = nullptr;
            get_device_name = nullptr;
            get_ddr_size = nullptr;
            get_mem_info = nullptr;
            get_computing_power = nullptr;
            get_core_count = nullptr;
            get_ipu_utili_rate = nullptr;
            get_temperature = nullptr;
            get_driver_version = nullptr;
            get_device_version = nullptr;
            initialized = false;
        }
    }

    ~houmonpu_sdk() {
        unload();
    }
};

// ==================== HoumoNPU 设备信息 ====================

struct ggml_houmonpu_device_info {
    int device_count = 0;
    int device_ids[32] = {};  // 设备 ID 列表
    size_t device_memory_total[32] = {};  // 总内存
    size_t device_memory_free[32] = {};   // 可用内存
    char device_descriptions[32][512] = {};  // 设备描述
};

static ggml_houmonpu_device_info ggml_houmonpu_init() {
    ggml_houmonpu_device_info info = {};

    static houmonpu_sdk sdk;

    if (!sdk.load()) {
        // SDK 加载失败，返回 0 个设备
        GGML_LOG_INFO("[HoumoNPU] SDK load failed, no devices available\n");
        return info;
    }

    // 获取设备信息
    struct hm_device_info hm_info = {};
    uint32_t count = sdk.get_device_info(&hm_info);

    GGML_LOG_INFO("[HoumoNPU] hm_sys_get_device_info returned: %d devices\n", count);

    if (count == 0 || count > 32) {
        GGML_LOG_INFO("[HoumoNPU] Invalid device count: %d\n", count);
        return info;
    }

    if (sdk.get_driver_version) {
        char driver_version[256] = {0};
        int version_ret = sdk.get_driver_version(driver_version, sizeof(driver_version));
        if (version_ret == 0) {
            GGML_LOG_INFO("[HoumoNPU] driver_version=%s\n", driver_version);
        } else {
            GGML_LOG_INFO("[HoumoNPU] get_driver_version failed: ret=%d\n", version_ret);
        }
    } else {
        GGML_LOG_INFO("[HoumoNPU] driver_version=<symbol unavailable>\n");
    }

    info.device_count = (int)count;

    // 获取每个设备的信息
    for (uint32_t i = 0; i < count; i++) {
        int dev_index = (int)hm_info.device_ids[i];
        info.device_ids[i] = dev_index;
        GGML_LOG_INFO("[HoumoNPU] Device %u: device_id=%d\n", i, dev_index);

        // 获取设备名称用于描述
        char name[256] = {0};
        int ret = sdk.get_device_name(dev_index, name, sizeof(name));
        if (ret == 0) {
            snprintf(info.device_descriptions[i], sizeof(info.device_descriptions[i]),
                     "HoumoNPU %s (ID: %d)", name, dev_index);
        } else {
            snprintf(info.device_descriptions[i], sizeof(info.device_descriptions[i]),
                     "HoumoNPU Device %d", dev_index);
        }

        if (sdk.get_device_version) {
            char device_version[256] = {0};
            int version_ret = sdk.get_device_version(dev_index, device_version, sizeof(device_version));
            if (version_ret == 0) {
                GGML_LOG_INFO("[HoumoNPU]   device_version=%s\n", device_version);
            } else {
                GGML_LOG_INFO("[HoumoNPU]   get_device_version failed: ret=%d\n", version_ret);
            }
        } else {
            GGML_LOG_INFO("[HoumoNPU]   device_version=<symbol unavailable>\n");
        }

        // 获取内存信息
        struct hm_mem_info mem_info = {};
        ret = sdk.get_mem_info(dev_index, &mem_info);
        GGML_LOG_INFO("[HoumoNPU]   get_mem_info returned: %d, mem_total=%u, mem_used=%u, mem_avail=%u\n",
                ret, mem_info.mem_total, mem_info.mem_used, mem_info.mem_avail);
        if (ret == 0 && mem_info.mem_total > 0) {
            // mem_total 和 mem_avail 单位是 MB，转换为字节
            info.device_memory_total[i] = (size_t)mem_info.mem_total * 1024 * 1024;
            info.device_memory_free[i] = (size_t)mem_info.mem_avail * 1024 * 1024;
        } else {
            // 回退到 ddr_size
            uint64_t ddr_size = 0;
            ret = sdk.get_ddr_size(dev_index, &ddr_size);
            GGML_LOG_INFO("[HoumoNPU]   get_ddr_size returned: %d, ddr_size=%lu\n", ret, (unsigned long)ddr_size);
            if (ret == 0) {
                info.device_memory_total[i] = (size_t)ddr_size;
                info.device_memory_free[i] = (size_t)ddr_size;
            } else {
                // 默认 12GB
                info.device_memory_total[i] = 12ULL * 1024 * 1024 * 1024;
                info.device_memory_free[i] = 12ULL * 1024 * 1024 * 1024;
            }
        }
        GGML_LOG_INFO("[HoumoNPU]   Final device memory: total=%zu MB, free=%zu MB\n",
                info.device_memory_total[i] / 1024 / 1024, info.device_memory_free[i] / 1024 / 1024);

        int computing_power = -1;
        if (sdk.get_computing_power) {
            computing_power = sdk.get_computing_power(dev_index);
        }

        float ipu_util_rate = -1.0f;
        if (sdk.get_ipu_utili_rate) {
            ipu_util_rate = sdk.get_ipu_utili_rate(dev_index);
        }

        float temperature = -1.0f;
        if (sdk.get_temperature) {
            int temp_ret = sdk.get_temperature(dev_index, &temperature);
            if (temp_ret != 0) {
                temperature = -1.0f;
                GGML_LOG_INFO("[HoumoNPU]   get_temperature failed: ret=%d\n", temp_ret);
            }
        }

        GGML_LOG_INFO("[HoumoNPU]   Extra device info: computing_power=%d, ipu_util_rate=%.4f, temperature=%.2fC\n"
            , computing_power, ipu_util_rate, temperature);
    }

    sdk.initialized = true;
    GGML_LOG_INFO("[HoumoNPU] Initialization complete, %d devices available\n", info.device_count);

    return info;
}

static const ggml_houmonpu_device_info& ggml_houmonpu_info() {
    static ggml_houmonpu_device_info info = ggml_houmonpu_init();
    return info;
}

// ==================== HoumoNPU Device ====================

struct ggml_backend_houmonpu_device_context {
    int device;
};

static const char* ggml_backend_houmonpu_device_get_name(ggml_backend_dev_t dev) {
    auto* ctx = (ggml_backend_houmonpu_device_context*)dev->context;
    // 返回设备 ID 作为名称
    static char name_buf[16];
    snprintf(name_buf, sizeof(name_buf), "%d", ctx->device);
    return name_buf;
}

static const char* ggml_backend_houmonpu_device_get_description(ggml_backend_dev_t dev) {
    auto* ctx = (ggml_backend_houmonpu_device_context*)dev->context;
    if (ctx->device < ggml_houmonpu_info().device_count) {
        return ggml_houmonpu_info().device_descriptions[ctx->device];
    }
    return "Unknown HoumoNPU Device";
}

static void ggml_backend_houmonpu_device_get_memory(ggml_backend_dev_t dev,
                                                    size_t* free, size_t* total) {
    auto* ctx = (ggml_backend_houmonpu_device_context*)dev->context;

    const auto& info = ggml_houmonpu_info();
    if (ctx->device >= info.device_count) {
        *free = 0;
        *total = 0;
        return;
    }

    *total = info.device_memory_total[ctx->device];
    *free = info.device_memory_free[ctx->device];
}

static enum ggml_backend_dev_type ggml_backend_houmonpu_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    // HW_ACCEL: 可被 --list-devices 显示，但不会被自动初始化为计算后端
    // 后端完整实现后改为 GGML_BACKEND_DEVICE_TYPE_ACCEL
    return GGML_BACKEND_DEVICE_TYPE_HM_ACCEL;
}

static void ggml_backend_houmonpu_device_get_props(ggml_backend_dev_t dev,
                                                   struct ggml_backend_dev_props* props) {
    props->name = ggml_backend_houmonpu_device_get_name(dev);
    props->description = ggml_backend_houmonpu_device_get_description(dev);
    ggml_backend_houmonpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->type = ggml_backend_houmonpu_device_get_type(dev);
    props->device_id = nullptr;
    props->caps.async = false;
    props->caps.host_buffer = false;
    props->caps.buffer_from_host_ptr = false;
    props->caps.events = false;
}

// TODO: 以下 stub 函数在后端完整实现后替换
static ggml_backend_t ggml_backend_houmonpu_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
    return nullptr;
}

static ggml_backend_buffer_type_t ggml_backend_houmonpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return nullptr;
}

static bool ggml_backend_houmonpu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    return false;
}

static bool ggml_backend_houmonpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    GGML_UNUSED(buft);
    return false;
}

static struct ggml_backend_device_i ggml_backend_houmonpu_device_interface = {
    /* .get_name              = */ ggml_backend_houmonpu_device_get_name,
    /* .get_description       = */ ggml_backend_houmonpu_device_get_description,
    /* .get_memory            = */ ggml_backend_houmonpu_device_get_memory,
    /* .get_type              = */ ggml_backend_houmonpu_device_get_type,
    /* .get_props             = */ ggml_backend_houmonpu_device_get_props,
    /* .init_backend          = */ ggml_backend_houmonpu_device_init_backend,
    /* .get_buffer_type       = */ ggml_backend_houmonpu_device_get_buffer_type,
    /* .get_host_buffer_type  = */ NULL,
    /* .buffer_from_host_ptr  = */ NULL,
    /* .supports_op           = */ ggml_backend_houmonpu_device_supports_op,
    /* .supports_buft         = */ ggml_backend_houmonpu_device_supports_buft,
    /* .offload_op            = */ NULL,
    /* .event_new             = */ NULL,
    /* .event_free            = */ NULL,
    /* .event_synchronize     = */ NULL,
};

// ==================== HoumoNPU Reg ====================

struct ggml_backend_houmonpu_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char* ggml_backend_houmonpu_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_HOUMONPU_NAME;
}

static size_t ggml_backend_houmonpu_reg_get_device_count(ggml_backend_reg_t reg) {
    auto* ctx = (ggml_backend_houmonpu_reg_context*)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_houmonpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto* ctx = (ggml_backend_houmonpu_reg_context*)reg->context;
    return ctx->devices[index];
}

static struct ggml_backend_reg_i ggml_backend_houmonpu_reg_interface = {
    /* .get_name         = */ ggml_backend_houmonpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_houmonpu_reg_get_device_count,
    /* .get_device       = */ ggml_backend_houmonpu_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_houmonpu_reg() {
    static struct ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_houmonpu_reg_interface,
        /* .context     = */ nullptr,
    };

    static ggml_backend_houmonpu_reg_context ctx;

    if (!ctx.devices.empty()) {
        return &reg;
    }

    const auto& info = ggml_houmonpu_info();
    for (int i = 0; i < info.device_count; i++) {
        auto* dev_ctx = new ggml_backend_houmonpu_device_context;
        dev_ctx->device = i;

        static ggml_backend_device devices[32] = {};
        devices[i] = {
            /* .iface   = */ ggml_backend_houmonpu_device_interface,
            /* .reg     = */ &reg,
            /* .context = */ dev_ctx,
        };

        ctx.devices.push_back(&devices[i]);
    }

    reg.context = &ctx;

    return &reg;
}

#ifdef GGML_BACKEND_DL
GGML_BACKEND_DL_IMPL(ggml_backend_houmonpu_reg)
#endif

// ==================== 公共 API ====================

int ggml_backend_houmonpu_get_device_count(void) {
    return ggml_houmonpu_info().device_count;
}

void ggml_backend_houmonpu_get_device_description(int device, char* description, size_t description_size) {
    if (device < 0 || device >= ggml_houmonpu_info().device_count) {
        return;
    }
    snprintf(description, description_size, "%s", ggml_houmonpu_info().device_descriptions[device]);
}

void ggml_backend_houmonpu_get_device_memory(int device, size_t* free, size_t* total) {
    if (device < 0 || device >= ggml_houmonpu_info().device_count) {
        *free = 0;
        *total = 0;
        return;
    }

    const auto& info = ggml_houmonpu_info();
    *total = info.device_memory_total[device];
    *free = info.device_memory_free[device];
}
