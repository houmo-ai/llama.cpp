#include "houmo-embedding_layer.h"
#include <fstream>
#include <cstring>
#include "llama-impl.h"
#include "tcim/tcim_runtime.h"

// NpuState: PIMPL 隐藏 tcim 依赖 (仅在 --embed-npu 时创建)
struct HoumoEmbeddingLayer::NpuState {
    tcim::Tensor device_emb;       // embedding 权重 [vocab_size, 1, n_embd] fp16 on NPU
};

// PIMPL 析构: 必须在 NpuState 完整定义之后 (在 .cpp 中)
HoumoEmbeddingLayer::~HoumoEmbeddingLayer() = default;

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

// ===== NPU 路径 (--embed-npu) =====
void HoumoEmbeddingLayer::init_npu(int device_id) {
    if (npu_ready_) return;  // 已经初始化过了

    // device_id < 0 时从 HOUMO_DEVICE_ID 环境变量读取
    if (device_id < 0) {
        device_id = 0;
        const char *env = std::getenv("HOUMO_DEVICE_ID");
        if (env) {
            device_id = std::stoi(env);
        }
    }

    size_t elem_count = vocab_size_ * n_embd_;
    // ① 创建 NPU Device Tensor [vocab_size, 1, n_embd] fp16
    npu_ = std::make_unique<NpuState>();
    auto info = tcim::TensorInfo::CreateNDInfo(
        {static_cast<int>(vocab_size_), 1, static_cast<int>(n_embd_)},
        tcim::DataType::FLOAT16);
    npu_->device_emb = tcim::Tensor::CreateDeviceTensor(info, 0, device_id);

    // ② Host → Device: 直接拷贝原始字节
    size_t n_bytes = elem_count * sizeof(int16_t);
    npu_->device_emb.Buffer().CopyFromHost(data_.data(), n_bytes);

    npu_ready_ = true;
    // ③ 释放 CPU 端 data_ 副本 (swap 彻底归还) → 真正省 CPU 内存
    std::vector<int8_t>().swap(data_);
    LLAMA_LOG_INFO("%s: NPU embedding ready, device=%d, vocab=%d, dim=%d, %.1f MB, CPU data_ freed\n",
        __func__, device_id, vocab_size_, n_embd_, n_bytes / 1048576.0);
}

void HoumoEmbeddingLayer::get_embedding_batch_npu(const int32_t *token_ids, size_t n, void *dst, int offset_rows) {
    if (!npu_ready_) return;

    const size_t row_bytes = n_embd_ * sizeof(int16_t);
    uint8_t *host_base = static_cast<uint8_t *>(dst);

    for (size_t i = 0; i < n; ++i) {
        validate_token_id(token_ids[i]);
        // SelectROI 从 NPU Device Tensor 中取出一行 embedding
        auto src = npu_->device_emb.SelectROI(
            {static_cast<int>(token_ids[i]), 0, 0}, {1, 1, static_cast<int>(n_embd_)});
        // D2H: 从 NPU Device 拷贝到目标 buffer 对应位置
        uint8_t *row_dst = host_base + (static_cast<size_t>(offset_rows) + i) * row_bytes;
        src.Buffer().CopyToHost(row_dst, row_bytes);
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