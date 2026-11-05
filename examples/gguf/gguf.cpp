#include "ggml.h"
#include "gguf.h"

#include <clocale>
#include <cstdio>
#include <string>
#include <sstream>
#include <vector>

// 截断阈值
static const size_t MAX_STRING_LEN = 88;    // 字符串最大显示长度
static const size_t MAX_ARRAY_ITEMS = 6;     // 数组最大显示元素数

template <typename T>
static std::string to_string(const T & val) {
    std::stringstream ss;
    ss << val;
    return ss.str();
}

// 截断字符串显示
static std::string truncate_str(const std::string & s, size_t max_len = MAX_STRING_LEN) {
    if (s.size() <= max_len) {
        return "'" + s + "'";
    }
    return "'" + s.substr(0, max_len) + "...'";
}

// 截断数组显示
static std::string truncate_arr_str(const std::string & s, size_t max_len = MAX_STRING_LEN) {
    if (s.size() <= max_len) {
        return s;
    }
    return s.substr(0, max_len) + "...";
}

static std::string gguf_kv_to_str(const struct gguf_context * ctx, int64_t i, bool truncate = true) {
    const enum gguf_type type = gguf_get_kv_type(ctx, i);
    switch (type) {
    case GGUF_TYPE_UINT8:   return to_string(gguf_get_val_u8(ctx, i));
    case GGUF_TYPE_INT8:    return to_string(gguf_get_val_i8(ctx, i));
    case GGUF_TYPE_UINT16:  return to_string(gguf_get_val_u16(ctx, i));
    case GGUF_TYPE_INT16:  return to_string(gguf_get_val_i16(ctx, i));
    case GGUF_TYPE_UINT32: return to_string(gguf_get_val_u32(ctx, i));
    case GGUF_TYPE_INT32:  return to_string(gguf_get_val_i32(ctx, i));
    case GGUF_TYPE_FLOAT32: return to_string(gguf_get_val_f32(ctx, i));
    case GGUF_TYPE_UINT64: return to_string(gguf_get_val_u64(ctx, i));
    case GGUF_TYPE_INT64:  return to_string(gguf_get_val_i64(ctx, i));
    case GGUF_TYPE_FLOAT64: return to_string(gguf_get_val_f64(ctx, i));
    case GGUF_TYPE_BOOL:   return gguf_get_val_bool(ctx, i) ? "true" : "false";
    case GGUF_TYPE_STRING: {
        const std::string s(gguf_get_val_str(ctx, i));
        return truncate ? truncate_str(s) : "'" + s + "'";
    }
    case GGUF_TYPE_ARRAY:
    {
        const enum gguf_type arr_type = gguf_get_arr_type(ctx, i);
        const size_t arr_n = gguf_get_arr_n(ctx, i);
        std::string result = "[";
        const size_t show_n = truncate ? std::min(arr_n, MAX_ARRAY_ITEMS) : arr_n;
        for (size_t j = 0; j < show_n; ++j) {
            if (j > 0) result += ", ";
            if (arr_type == GGUF_TYPE_STRING) {
                std::string item(gguf_get_arr_str(ctx, i, j));
                result += truncate ? truncate_str(item, 16) : "'" + item + "'";
            } else {
                const void * data = gguf_get_arr_data(ctx, i);
                switch (arr_type) {
                case GGUF_TYPE_UINT8:   result += to_string(((const uint8_t *)data)[j]); break;
                case GGUF_TYPE_INT8:    result += to_string(((const int8_t *)data)[j]); break;
                case GGUF_TYPE_UINT16:  result += to_string(((const uint16_t *)data)[j]); break;
                case GGUF_TYPE_INT16:   result += to_string(((const int16_t *)data)[j]); break;
                case GGUF_TYPE_UINT32:  result += to_string(((const uint32_t *)data)[j]); break;
                case GGUF_TYPE_INT32:   result += to_string(((const int32_t *)data)[j]); break;
                case GGUF_TYPE_FLOAT32: result += to_string(((const float *)data)[j]); break;
                case GGUF_TYPE_UINT64:  result += to_string(((const uint64_t *)data)[j]); break;
                case GGUF_TYPE_INT64:   result += to_string(((const int64_t *)data)[j]); break;
                case GGUF_TYPE_FLOAT64: result += to_string(((const double *)data)[j]); break;
                case GGUF_TYPE_BOOL:    result += ((const int8_t *)data)[j] ? "true" : "false"; break;
                default: result += "?"; break;
                }
            }
        }
        if (arr_n > show_n) {
            result += ", ...";
        }
        result += "]";
        return result;
    }
    default:
        return "?";
    }
}

// 类型名称映射（用于打印）
static const char * gguf_type_name_short(enum gguf_type type) {
    switch (type) {
    case GGUF_TYPE_UINT8:   return "UINT8";
    case GGUF_TYPE_INT8:    return "INT8";
    case GGUF_TYPE_UINT16:  return "UINT16";
    case GGUF_TYPE_INT16:  return "INT16";
    case GGUF_TYPE_UINT32: return "UINT32";
    case GGUF_TYPE_INT32:  return "INT32";
    case GGUF_TYPE_FLOAT32: return "FLOAT32";
    case GGUF_TYPE_UINT64: return "UINT64";
    case GGUF_TYPE_INT64:  return "INT64";
    case GGUF_TYPE_FLOAT64: return "FLOAT64";
    case GGUF_TYPE_BOOL:   return "BOOL";
    case GGUF_TYPE_STRING: return "STRING";
    case GGUF_TYPE_ARRAY:  return "ARRAY";
    default: return "UNKNOWN";
    }
}

// 读取并按照 gguf-dump 风格打印
static bool gguf_ex_read_0(const std::string & fname) {
    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx = */ NULL,
    };

    struct gguf_context * ctx = gguf_init_from_file(fname.c_str(), params);

    if (!ctx) {
        fprintf(stderr, "%s: failed to load '%s'\n", __func__, fname.c_str());
        return false;
    }

    printf("File is LITTLE endian, script is running on a LITTLE endian host.\n");
    printf("* Dumping %ld key/value pair(s)\n", gguf_get_n_kv(ctx));

    // kv pairs
    {
        const int n_kv = gguf_get_n_kv(ctx);

        for (int i = 0; i < n_kv; ++i) {
            const char * key = gguf_get_key(ctx, i);
            const enum gguf_type type = gguf_get_kv_type(ctx, i);
            const std::string val = gguf_kv_to_str(ctx, i, true);  // 截断模式

            // 获取元素数量
            uint64_t count = 1;
            if (type == GGUF_TYPE_ARRAY) {
                count = gguf_get_arr_n(ctx, i);
            }

            // 格式化类型名称
            char type_str[32];
            if (type == GGUF_TYPE_ARRAY) {
                const enum gguf_type arr_type = gguf_get_arr_type(ctx, i);
                snprintf(type_str, sizeof(type_str), "[%s]", gguf_type_name_short(arr_type));
            } else {
                snprintf(type_str, sizeof(type_str), "%s", gguf_type_name_short(type));
            }

            printf("%4d: %12s | %zu | %s = %s\n",
                   i + 1, type_str, (size_t)count, key, val.c_str());
        }
    }

    // tensor info
    {
        const int n_tensors = gguf_get_n_tensors(ctx);
        printf("* Dumping %d tensor(s)\n", n_tensors);

        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx, i);
            const size_t size = gguf_get_tensor_size(ctx, i);
            // const size_t offset = gguf_get_tensor_offset(ctx, i);
            const enum ggml_type type = gguf_get_tensor_type(ctx, i);

            printf("%4d: %12zu | %12zu, 1, 1, 1 | %s | %s\n",
                   i + 1, size, size, ggml_type_name(type), name);
        }
    }

    gguf_free(ctx);

    return true;
}

// 按 key 读取对应值
static bool gguf_ex_read_key(const std::string & fname, const std::string & key) {
    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx = */ NULL,
    };

    struct gguf_context * ctx = gguf_init_from_file(fname.c_str(), params);

    if (!ctx) {
        fprintf(stderr, "%s: failed to load '%s'\n", __func__, fname.c_str());
        return false;
    }

    const int64_t key_id = gguf_find_key(ctx, key.c_str());
    if (key_id < 0) {
        fprintf(stderr, "%s: key not found: %s\n", __func__, key.c_str());
        gguf_free(ctx);
        return false;
    }

    const std::string val = gguf_kv_to_str(ctx, key_id, false);
    printf("%s\n", val.c_str());

    gguf_free(ctx);
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    if (argc < 2) {
        printf("usage: %s data.gguf [r] [-k keyword]\n", argv[0]);
        printf("default mode: r\n");
        printf("r: read and dump data.gguf file (print key-value pairs and tensor info)\n");
        printf("r -k keyword: print the value of the specified keyword\n");
        return -1;
    }

    const std::string fname(argv[1]);
    std::string mode = "r";
    std::string keyword;

    int argi = 2;
    if (argc > argi && std::string(argv[argi]) == "r") {
        mode = argv[argi];
        argi++;
    } else if (argc > argi && std::string(argv[argi]) == "w") {
        fprintf(stderr, "error: w mode is not supported\n");
        return -1;
    }

    while (argi < argc) {
        const std::string arg = argv[argi];
        if (arg == "-k") {
            if (argi + 1 >= argc) {
                fprintf(stderr, "error: missing value after -k\n");
                return -1;
            }
            keyword = argv[argi + 1];
            argi += 2;
            continue;
        }

        fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
        return -1;
    }

    GGML_ASSERT(mode == "r" && "mode must be r");

    if (!keyword.empty()) {
        GGML_ASSERT(gguf_ex_read_key(fname, keyword) && "failed to read keyword from gguf file");
    } else {
        GGML_ASSERT(gguf_ex_read_0(fname) && "failed to read gguf file");
    }

    return 0;
}