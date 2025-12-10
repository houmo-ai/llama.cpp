#pragma once

#include <map>
#include <string>
#include <vector>
#include <cmath>
#include "llama-batch.h"
#include "tcim/tcim_runtime.h"
#include "houmo-embedding_layer.h"
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
    int houmo_embedding(const std::map<int32_t, std::vector<llama_token>> &batches,
                        std::map<int, float *> &embeddings);
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
  virtual bool houmo_load(const std::string &file_name);
  virtual void houmo_init(int seq_max = 4) = 0;
  virtual int houmo_prefill(std::vector<llama_token> &tokens, int seq_id,
                            float *logits) = 0;
  virtual int houmo_prefill(const float *embeddings, int n_tokens, int seq_id,
                            float *logits) {
      (void)embeddings; // unused parameter
      (void)n_tokens;   // unused parameter
      (void)seq_id;     // unused parameter
      (void)logits;     // unused parameter
      return 0;
  }
  virtual int
  houmo_embedding(const std::map<int32_t, std::vector<llama_token>> &batches,
                  std::map<int, float *> &embeddings) {
      (void)batches;    // unused parameter
      (void)embeddings; // unused parameter
      return 0;
  }
  virtual int houmo_decode(std::vector<llama_token> &batches,
                           std::vector<int> &seq_ids,
                           std::vector<float *> logits) = 0;

  int n_decode_batch() const { return n_decode_batch_; }
  int n_prefill_batch() const { return n_prefill_batch_; }
  uint32_t n_context_length() const { return context_length_;}

protected:
  /**
   * @brief Initialize the model input and output.
   * @details Initialize the model input and output with the given seq_max.
   * @param seq_max The maximum sequences.
   */
  void model_inout_init(int seq_max);
  /**
   * @brief Set the kv cache for prefill.
   * @details Set the kv cache for prefill with the given seq_id.
   * @param seq_id The sequence id.
   */
  void set_kv_cache_for_prefill(int seq_id);
  /**
   * @brief Set the kv cache for decode.
   * @details Set the kv cache for decode with the given seq_ids.
   * @param seq_ids The sequence ids.
   */
  void set_kv_cache_for_decode(const std::vector<int> &seq_ids);
  /**
   * @brief Set the input for decode.
   * @details Set the input for decode with the given batches and seq_ids.
   * @param batches The input batches.
   * @param seq_ids The sequence ids.
   * @return 0 if success, -1 if failed.
   */
  int set_input_for_decode(std::vector<llama_token> &batches,
                           std::vector<int> &seq_ids);
  /**
   * @brief Set the (valid_length&current_length) for decode.
   * @details Set the valid_length&current_length for decode with the given seq_ids.
   * @param seq_ids The sequence ids.
   * @return 0 if success, -1 if failed.
   */
  int set_xlenght_for_decode(std::vector<int> &seq_ids);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  /**
   * @brief Initialize functions.
   * @details Initialize the prefill input with the given batch_size.
   * @param batch_size The batch size.
   */
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void prefill_input_init();
  void prefill_output_init();
  void decode_input_init();
  void decode_output_init();
  void initialize_kv_cache(int size);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  // helper functions                                                                           //
  ////////////////////////////////////////////////////////////////////////////////////////////////
  /**
   * @brief Parse the device ids.
   * @details Parse the device ids form env. If single_card is true, return the first device id.
   * @param single_card The single card.
   * @return The device ids.
   */
  std::vector<int> parse_deviceids(bool single_card = true);
  void read_tensor_to_buffer(const std::string &file_name, size_t offset,
                             size_t size, std::vector<char> &buffer);
  /**
   * @brief Parse the gguf parameters.(n_blocks, n_embd, n_vocab)
   * @details Parse the gguf parameters with the given ctx.
   * @param ctx The gguf context.
   */
  void parser_gguf_parameters(struct gguf_context *ctx);
  /**
   * @brief Set the input data for the tensor.
   * @details Set the input data for the tensor with the given batch_size and value.
   * @param tensor The tensor.
   * @param batch_size The batch size.
   * @param value The value.
   */
  void set_input_data(tcim::Tensor &tensor, int batch_size, int value);
  /**
   * @brief Set the input data for the tensor.
   * @details Set the input data for the tensor with the given int32_vec and offset.
   * @param tensor The tensor.
   * @param int32_vec The int32 vector.
   * @param offset The offset.
   */
  void set_input_data(tcim::Tensor &tensor, std::vector<int32_t> &int32_vec,
                      size_t offset = 0);
private:
  bool create_infer_engine(bool prefill, const std::vector<char> &raw_data, bool load_from_file = false);

protected:
  int prefill_length_ = 256;
  int context_length_ = 4096;
  std::map<int, int32_t> decode_valid_length_;
  int n_blocks_ = 28;
  int64_t n_embd_ = 3584;
  int64_t n_vocab_ = 30522;
  int n_prefill_batch_ = 1;
  int n_decode_batch_ = 1;
  bool has_history_sum_ = false;

  std::shared_ptr<HoumoEmbeddingLayer> embedding_layer_;
  std::shared_ptr<tcim::Module> prefill_model_;
  std::shared_ptr<tcim::Module> decode_model_;
  tcim::Module::WeightManager wm_;

  std::map<std::string, tcim::Tensor> prefill_input_map_;
  std::map<std::string, tcim::Tensor> prefill_output_map_;
  std::map<std::string, tcim::Tensor> decode_input_map_;
  std::map<std::string, tcim::Tensor> decode_output_map_;
  std::vector<std::string> decode_output_names_;
  std::vector<std::string> prefill_output_names_;
  std::vector<std::string> decode_valid_length_names_;
  std::vector<std::string> decode_current_length_names_;
  std::map<int, kv_cache_item> kv_cache_;
  int need_set_kv_cache_ = 0; // 0: no need, 1: need
  std::string kGeneralArch = "general.architecture";
  std::string kEmbeddingLengthEx = ".embedding_length";
  std::string kBlockCountEx = ".block_count";
  int kSeqMax = 4; // maximum sequence number for single-card model
  std::string model_arch_ = "";
  bool xh1_ = true;
};