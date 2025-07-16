#pragma once

#include <map>
#include <string>
#include <vector>

#include "llama-batch.h"
#include "tcim/tcim_runtime.h"
#include "torch/torch.h"
// houmo llm model defination
//
// @guoxing.xu 2025-05-23
//

/**
 * @brief HouMoLLModel is the context for HouMo model.
 * @details It provides the interface for HouMo model inference.
 */
class HouMoLLModel {
  public:
    HouMoLLModel();
    ~HouMoLLModel();

    bool houmo_load(const std::string &file_name);
    void houmo_init(int seq_max = 4);
    int houmo_prefill(std::vector<llama_token> &tokens, int seq_id,
                      float *logits);
    int houmo_prefill(const float *embeddings, int n_tokens, int seq_id,
                      float *logits);
    int houmo_decode(std::vector<llama_token> &batches,
                     std::vector<int> &seq_ids, std::vector<float *> logits);
    int n_decode_batch();
    int n_prefill_batch();
    uint32_t n_context_length();
    class Impl;  // Forward declaration of the implementation class
  private:
    Impl *impl_; // Pointer to the implementation
};

struct kv_cache_item {
  std::vector<tcim::Tensor> tensors;
  std::vector<std::string> names;
  int seq_id = 0;
};
/**
* @brief HouMoLLModel is the base class for HouMo model.
* @details It contains the common interface and data members for HouMo model
* inference. Derived classes will implement specific behavior for single-card
* and dual-card models.
*/
class HouMoLLModel::Impl {
public:
Impl() = default;
  virtual ~Impl() = default;

  virtual bool houmo_load(const std::string &file_name) = 0;
  virtual void houmo_init(int seq_max = 4) = 0;
  virtual int houmo_prefill(std::vector<llama_token> &tokens, int seq_id,
                            float *logits) = 0;
  virtual int houmo_prefill(const float *embeddings, int n_tokens, int seq_id,
                            float *logits) {
      return 0;
  }

  virtual int houmo_decode(std::vector<llama_token> &batches,
                           std::vector<int> &seq_ids,
                           std::vector<float *> logits) = 0;

  int n_decode_batch() const { return n_decode_batch_; }
  int n_prefill_batch() const { return n_prefill_batch_; }
  uint32_t n_context_length() const { return context_length_;}

protected:
  void
  prefill_input_init(std::shared_ptr<tcim::Module> prefill_model,
                     std::map<std::string, tcim::Tensor> &prefill_input_map);
  void
  prefill_output_init(std::shared_ptr<tcim::Module> prefill_model,
                      std::map<std::string, tcim::Tensor> &prefill_output_map,
                      std::vector<std::string> &prefill_output_names);
  void
  decode_input_init(std::shared_ptr<tcim::Module> decode_model,
                    std::map<std::string, tcim::Tensor> &decode_input_map);
  void
  decode_output_init(std::shared_ptr<tcim::Module> decode_model,
                     std::map<std::string, tcim::Tensor> &decode_output_map,
                     std::vector<std::string> &decode_output_names);
  void fill_parameters(struct gguf_context *ctx);

  void read_tensor_to_buffer(const std::string &file_name, size_t offset,
    size_t size, std::vector<char> &buffer);
protected:
  int prefill_length_ = 256;
  int context_length_ = 4096;
  std::map<int, int32_t> decode_valid_length_;
  int n_blocks_ = 28;
  int64_t n_embd_ = 3584;
  int64_t n_vocab_ = 30522;
  int n_prefill_batch_ = 1;
  int n_decode_batch_ = 1;
  torch::nn::Embedding embedding_ = nullptr;
  std::map<std::string, tcim::Tensor> prefill_input_map_;
  std::map<std::string, tcim::Tensor> prefill_output_map_;
  std::map<std::string, tcim::Tensor> decode_input_map_;
  std::map<std::string, tcim::Tensor> decode_output_map_;
  std::vector<std::string> decode_output_names_;
  std::vector<std::string> prefill_output_names_;

  std::map<int, kv_cache_item> kv_cache_;
  int need_set_kv_cache_ = 0; // 0: no need, 1: need
  std::string kGeneralArch = "general.architecture";
  std::string kEmbeddingLengthEx = ".embedding_length";
  std::string kBlockCountEx = ".block_count";
  bool xh1_ = true;
  int model_type_ = 0; // 0:text, 1: image..
};

// 封装数据类型无关的操作
template <typename T>
void SetInputData(tcim::Tensor& tensor, int batch_size, int value) {
    std::vector<T> data_vec(batch_size, value);
    tensor.Buffer().CopyFromHost(
        data_vec.data(), tensor.MemSize());
}
template <typename T>
void SetInputData(tcim::Tensor &tensor, std::vector<int32_t> &int32_vec) {
    std::vector<T> data_vec;
    for (auto val: int32_vec) {
        data_vec.push_back(static_cast<T>(val));
    }
    tensor.Buffer().CopyFromHost(data_vec.data(), tensor.MemSize());
}