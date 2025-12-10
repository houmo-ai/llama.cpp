#pragma once
#include <cstdint>
#include <string>
#include <stdexcept>
#include <vector>

// 数据类型枚举（与Python导出的二进制格式对齐）
enum class EmbeddingDtype {
    INT16 = 0,   // 对应Python端的torch.int16
    FLOAT16 = 1, // 对应Python端的torch.float16
    UNKNOWN = 255
};

// 嵌入层核心类（支持直接返回int16/float16原始格式）
class HoumoEmbeddingLayer {
  public:
    HoumoEmbeddingLayer(const std::string &weight_path, int64_t offset, int64_t size);

    void get_embedding(int32_t token_id, void *data);

    void get_embedding_batch(const std::vector<int32_t> &token_ids,
                                   void *data);
    // ========== 工具接口 ==========
    // 获取词汇表大小
    uint32_t get_vocab_size() const { return vocab_size_; }

    // 获取嵌入维度
    uint32_t get_n_embd() const { return n_embd_; }

    // 获取原始数据类型
    EmbeddingDtype get_dtype() const { return dtype_; }

    // 检查是否为int16类型
    bool is_int16() const { return dtype_ == EmbeddingDtype::INT16; }

    // 检查是否为float16类型
    bool is_float16() const { return dtype_ == EmbeddingDtype::FLOAT16; }

  private:
    // 私有成员变量
    std::vector<uint32_t> shape_; // 权重形状 [vocab_size, n_embd]
    EmbeddingDtype dtype_ = EmbeddingDtype::UNKNOWN; // 原始数据类型
    std::vector<int8_t> data_;                        // int8格式权重
    uint32_t vocab_size_ = 0;                        // 词汇表大小
    uint32_t n_embd_ = 0;                            // 嵌入维度

    // ========== 私有工具函数 ==========
    // 验证token_id是否合法
    void validate_token_id(uint32_t token_id) const {
        if (token_id >= vocab_size_) {
            throw std::out_of_range("Token_id " + std::to_string(token_id) +
                                    " exceeds vocab_size " +
                                    std::to_string(vocab_size_));
        }
    }

    // 核心函数：加载二进制权重文件
    void load_weight(const std::string &weight_path, int64_t offset, int64_t size);
};