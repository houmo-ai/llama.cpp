#include "houmo-weight-manager-pool.h"
#include "houmo-embedding_layer.h"
#include "llama-impl.h"

#include <filesystem>
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

} // namespace

std::shared_ptr<houmo_weight_manager_handle> houmo_weight_manager_acquire(
        const std::string & model_path,
        const std::vector<int> & device_ids,
        const std::string & backend) {
    const std::string key = make_key(model_path, device_ids, backend);
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
        int64_t size) {
    const std::string key = make_key(model_path, {}, "host-embedding") +
            '\n' + std::to_string(offset) + '\n' + std::to_string(size);
    std::lock_guard<std::mutex> lock(pool_mutex);

    if (auto existing = embedding_pool[key].lock()) {
        LLAMA_LOG_INFO("%s: reusing shared embedding for %s\n",
                       __func__, model_path.c_str());
        return existing;
    }

    auto embedding = std::make_shared<HoumoEmbeddingLayer>(model_path, offset, size);
    embedding_pool[key] = embedding;
    LLAMA_LOG_INFO("%s: created shared embedding for %s\n",
                   __func__, model_path.c_str());
    return embedding;
}
