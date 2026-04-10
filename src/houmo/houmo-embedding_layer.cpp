#include "houmo-embedding_layer.h"
#include <fstream>
#include <cstring>
#include "llama-impl.h"

HoumoEmbeddingLayer::HoumoEmbeddingLayer(const std::string &weight_path, int64_t offset,
                               int64_t size) {
    load_weight(weight_path, offset, size);
    // 预计算嵌入维度（n_embd）
    if (shape_.size() != 2) {
        throw std::runtime_error(
            "Embedding weight must be 2D tensor (vocab_size, n_embd)");
    }
    vocab_size_ = shape_[0];
    n_embd_ = shape_[1];
    LLAMA_LOG_INFO("EmbeddingLayer init success: vocab_size=%d, n_embd=%d, dtype=%d",
                   vocab_size_, n_embd_, static_cast<int>(dtype_));
}

void HoumoEmbeddingLayer::get_embedding(int32_t token_id, void *out) {
    validate_token_id(token_id);

    size_t offset = token_id * n_embd_ * sizeof(int16_t);
    std::vector<int16_t> embedding(n_embd_);
    // 直接拷贝int16原始数据
    memcpy(out, &data_[offset], n_embd_ * sizeof(int16_t));
}

// ========== 新增：批量获取int16格式embedding（高性能） ==========
void HoumoEmbeddingLayer::get_embedding_batch(
    const std::vector<int32_t> &token_ids, void *out_batch) {
    char *out_batch_char = static_cast<char *>(out_batch);
    for (size_t i = 0; i < token_ids.size(); ++i) {
        uint32_t token_id = token_ids[i];
        validate_token_id(token_id);
        size_t offset = token_id * n_embd_ * sizeof(int16_t);
        memcpy(out_batch_char + i * n_embd_ * sizeof(int16_t),
                    &data_[offset], n_embd_ * sizeof(int16_t));
    }
}

// Overload: accept raw pointer to token ids and length to avoid allocations
void HoumoEmbeddingLayer::get_embedding_batch(const int32_t *token_ids, size_t n, void *out_batch) {
    char *out_batch_char = static_cast<char *>(out_batch);
    for (size_t i = 0; i < n; ++i) {
        uint32_t token_id = token_ids[i];
        validate_token_id(token_id);
        size_t offset = token_id * n_embd_ * sizeof(int16_t);
        memcpy(out_batch_char + i * n_embd_ * sizeof(int16_t),
               &data_[offset], n_embd_ * sizeof(int16_t));
    }
}

// 核心函数：加载二进制权重文件
void HoumoEmbeddingLayer::load_weight(const std::string &weight_path, int64_t offset,
                               int64_t size) {
    (void) size;
    std::ifstream f(weight_path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open weight file: " + weight_path);
    }
    // 跳过偏移量
    f.seekg(offset, std::ios::beg);
    uint32_t tmp;

    // 1. 读取维度数
    f.read(reinterpret_cast<char *>(&tmp), sizeof(uint32_t));
    if (f.gcount() != sizeof(uint32_t)) {
        throw std::runtime_error("Failed to read dim count from weight file");
    }
    uint32_t dim_count = tmp;

    // 2. 读取各维度值
    shape_.resize(dim_count);
    for (uint32_t i = 0; i < dim_count; ++i) {
        f.read(reinterpret_cast<char *>(&tmp), sizeof(uint32_t));
        if (f.gcount() != sizeof(uint32_t)) {
            throw std::runtime_error("Failed to read dim " + std::to_string(i) +
                                     " from weight file");
        }
        shape_[i] = tmp;
    }

    // 3. 读取数据类型标识
    f.read(reinterpret_cast<char *>(&tmp), sizeof(uint32_t));
    if (f.gcount() != sizeof(uint32_t)) {
        throw std::runtime_error("Failed to read dtype from weight file");
    }
    dtype_ = static_cast<EmbeddingDtype>(tmp);
    if (dtype_ != EmbeddingDtype::INT16 && dtype_ != EmbeddingDtype::FLOAT16) {
        throw std::runtime_error("Unsupported dtype: " + std::to_string(tmp) +
                                 " (only INT16/FLOAT16 are supported)");
    }

    // 4. 计算总元素数
    size_t elem_count = 1;
    for (uint32_t d : shape_) {
        elem_count *= d;
    }

    data_.resize(elem_count * sizeof(int16_t));
    f.read(reinterpret_cast<char *>(data_.data()),
            elem_count * sizeof(int16_t));
    if (f.gcount() !=
        static_cast<std::streamsize>(elem_count * sizeof(int16_t))) {
        throw std::runtime_error(
            "Failed to read int16 data from weight file");
    }
    f.close();
}