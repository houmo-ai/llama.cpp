#include "houmo-weight-manager-pool.h"
#include "houmo-embedding_layer.h"
#include "llama-impl.h"

#define XXH_INLINE_ALL
#include "../../examples/gguf-hash/deps/xxhash/xxhash.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>

namespace {

std::string make_key(
        const std::string & model_path,
        const std::vector<int> & device_ids,
        const std::string & backend) {
    std::error_code ec;
    auto path = std::filesystem::weakly_canonical(model_path, ec);
    const std::string normalized = ec ? model_path : path.string();

    std::ostringstream key;
    key << normalized << '\n' << backend;
    const std::filesystem::path identity_path(normalized);
    const auto size = std::filesystem::file_size(identity_path, ec);
    if (!ec) {
        key << '\n' << size;
    }
    ec.clear();
    const auto modified = std::filesystem::last_write_time(identity_path, ec);
    if (!ec) {
        key << '\n' << modified.time_since_epoch().count();
    }
    for (int device_id : device_ids) {
        key << '\n' << device_id;
    }
    return key.str();
}

std::mutex pool_mutex;
std::map<std::string, std::weak_ptr<houmo_weight_manager_handle>> pool;
std::map<std::string, std::weak_ptr<HoumoEmbeddingLayer>> embedding_pool;

std::mutex identity_cache_mutex;
std::map<std::string, std::string> identity_cache;

std::string make_identity_cache_key(
        const std::string & model_path,
        const std::vector<houmo_content_range> & ranges) {
    std::ostringstream key;
    key << make_key(model_path, {}, "content-identity");
    for (const auto & range : ranges) {
        key << '\n' << range.name << '\n' << range.offset << '\n' << range.size;
    }
    return key.str();
}

void hash_u64(XXH3_state_t & state, uint64_t value) {
    std::array<unsigned char, sizeof(value)> bytes{};
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<unsigned char>(value >> (i * 8));
    }
    XXH3_128bits_update(&state, bytes.data(), bytes.size());
}

std::string digest_to_string(XXH128_hash_t digest) {
    std::ostringstream out;
    out << std::hex << std::setfill('0')
        << std::setw(16) << digest.high64
        << std::setw(16) << digest.low64;
    return out.str();
}

} // namespace

std::string houmo_content_identity(
        const std::string & model_path,
        const std::vector<houmo_content_range> & input_ranges) {
    if (input_ranges.empty()) {
        return {};
    }

    std::vector<houmo_content_range> ranges = input_ranges;
    std::sort(ranges.begin(), ranges.end(), [](const auto & lhs, const auto & rhs) {
        return lhs.name < rhs.name;
    });
    const std::string cache_key = make_identity_cache_key(model_path, ranges);
    {
        std::lock_guard<std::mutex> lock(identity_cache_mutex);
        const auto it = identity_cache.find(cache_key);
        if (it != identity_cache.end()) {
            return it->second;
        }
    }

    std::ifstream file(model_path, std::ios::binary);
    if (!file) {
        LLAMA_LOG_ERROR("%s: failed to open %s\n", __func__, model_path.c_str());
        return {};
    }

    XXH3_state_t state;
    if (XXH3_128bits_reset(&state) == XXH_ERROR) {
        return {};
    }
    std::array<char, 1024 * 1024> buffer{};
    for (const auto & range : ranges) {
        if (range.offset < 0 || range.size < 0) {
            return {};
        }
        hash_u64(state, range.name.size());
        XXH3_128bits_update(&state, range.name.data(), range.name.size());
        hash_u64(state, static_cast<uint64_t>(range.size));

        file.clear();
        file.seekg(range.offset, std::ios::beg);
        if (!file) {
            return {};
        }
        int64_t remaining = range.size;
        while (remaining > 0) {
            const auto chunk = static_cast<std::streamsize>(
                    std::min<int64_t>(remaining, buffer.size()));
            file.read(buffer.data(), chunk);
            if (file.gcount() != chunk) {
                LLAMA_LOG_ERROR("%s: failed to read %s range %s\n", __func__,
                                model_path.c_str(), range.name.c_str());
                return {};
            }
            XXH3_128bits_update(&state, buffer.data(), static_cast<size_t>(chunk));
            remaining -= chunk;
        }
    }

    const std::string identity = digest_to_string(XXH3_128bits_digest(&state));
    {
        std::lock_guard<std::mutex> lock(identity_cache_mutex);
        identity_cache[cache_key] = identity;
    }
    return identity;
}

std::shared_ptr<houmo_weight_manager_handle> houmo_weight_manager_acquire(
        const std::string & model_path,
        const std::vector<int> & device_ids,
        const std::string & backend,
        const std::string & embedding_identity) {
    // If content identification failed, retain the old path-based behavior so
    // unrelated models are never merged accidentally.
    std::ostringstream key_builder;
    if (embedding_identity.empty()) {
        key_builder << "path\n" << make_key(model_path, device_ids, backend);
    } else {
        key_builder << "content\n" << backend << '\n' << embedding_identity;
        for (int device_id : device_ids) {
            key_builder << '\n' << device_id;
        }
    }
    const std::string key = key_builder.str();
    std::lock_guard<std::mutex> lock(pool_mutex);

    if (auto existing = pool[key].lock()) {
        LLAMA_LOG_INFO("%s: reusing shared TCIM WeightManager for %s\n",
                       __func__, model_path.c_str());
        return existing;
    }

    tcim::DevManager dev_manager = tcim::DevManager::Create(device_ids, backend);
    auto handle = std::make_shared<houmo_weight_manager_handle>();
    handle->manager = tcim::Module::WeightManager::CreateWeightManager(dev_manager);
    pool[key] = handle;
    LLAMA_LOG_INFO("%s: created shared TCIM WeightManager for %s\n",
                   __func__, model_path.c_str());
    return handle;
}

std::shared_ptr<HoumoEmbeddingLayer> houmo_embedding_layer_acquire(
        const std::string & model_path,
        int64_t offset,
        int64_t size,
        const std::string & embedding_identity,
        int npu_device_id) {
    const std::string content_key = embedding_identity.empty()
            ? make_key(model_path, {}, "host-embedding") + '\n' +
                    std::to_string(offset) + '\n' + std::to_string(size)
            : "content\n" + embedding_identity;
    const std::string key = content_key +
            (npu_device_id < 0
                 ? "\nmode\ncpu"
                 : "\nmode\nnpu\ndevice\n" + std::to_string(npu_device_id));
    std::lock_guard<std::mutex> lock(pool_mutex);

    if (auto existing = embedding_pool[key].lock()) {
        LLAMA_LOG_INFO("%s: reusing shared embedding for %s\n",
                       __func__, model_path.c_str());
        return existing;
    }

    auto embedding = std::make_shared<HoumoEmbeddingLayer>(model_path, offset, size);
    if (npu_device_id >= 0) {
        embedding->init_npu(npu_device_id);
    }
    embedding_pool[key] = embedding;
    LLAMA_LOG_INFO("%s: created shared embedding for %s\n",
                   __func__, model_path.c_str());
    return embedding;
}
