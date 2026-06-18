#pragma once

#include <map>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include "llama-batch.h"
#include "tcim/tcim_runtime.h"
#include "houmo-embedding_layer.h"
#include "houmo-memory.h"
#include "llama-model-loader.h"
#include "ggml.h"
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

    bool houmo_load(llama_model_loader & ml, std::vector<int> device_ids);
    // is_mtp: true → 本 ctx 是 MTP 草稿 context (cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP)，
    //         false → target context。由上层 llama_context::houmo_init 根据 cparams.ctx_type 传入。
    void houmo_init(houmo_memory_i *memory, int seq_max = 8,
          ggml_abort_callback abort_callback = nullptr,
          void *abort_callback_data = nullptr,
          bool is_mtp = false);
    int houmo_prefill(std::vector<llama_token> &tokens, int seq_id,
                      float *logits);
    int houmo_prefill(const float *embeddings, int n_tokens, int seq_id,
                      float *logits);
    int houmo_decode(std::vector<llama_token> &batches,
                     std::vector<int> &seq_ids, std::vector<float *> logits);
    int houmo_asr_force_align(std::vector<llama_token> &tokens, int seq_id,
                              std::vector<float *> &logits_out);
    /**
     * @brief Verify k draft tokens for the same sequence in a single NPU forward.
     * @details Equivalent to prefill(k tokens) but returns the logits of every
     * position in @p tokens (not only the last one). Used for speculative
     * decoding verification where the target model has to sample from each
     * of the k draft positions.
     *
     * Semantics:
     *   - All @p tokens belong to the same @p seq_id and are placed at
     *     consecutive positions base..base+k-1 where base = current KV length.
     *   - logits_out.size() must equal tokens.size().
     *   - logits_out[i] may be nullptr to skip writing that position.
     *   - On success KV state advances by k tokens; caller is responsible for
     *     rolling back rejected tail if needed.
     *
     * Default implementation falls back to a per-token houmo_prefill loop
     * (correct but slow, ~k× the cost). Backends that can emit per-position
     * logits from a single forward should override it.
     */
    int houmo_verify(std::vector<llama_token> &tokens, int seq_id,
                     std::vector<float *> &logits_out);
    int houmo_verify_accept(int seq_id, uint16_t n_accepted);
    int houmo_embedding(const std::map<int32_t, std::vector<llama_token>> &batches,
                        std::map<int, float *> &embeddings);
    bool is_embedding();
    bool is_asr_force_aligner();
    bool has_decode_model();

    // ---------------- MTP (Multi-Token Prediction) ----------------
    // 适配 llama.cpp 上游 commit 255582687 的 draft-mtp speculative 流程：
    //   - target 模型与 MTP 模型分别为两份独立 gguf，各自加载为 llama_model +
    //     llama_context（外部 common/speculative.cpp 已编排好）。
    //   - 这里只在 houmo backend 内部识别"当前 ctx 是 MTP 还是 target"，
    //     角色由 houmo_init(..., is_mtp) 参数传入，不再从 gguf tensor 名推断。
    //   - 不修改任何公共 llama.h API。
    //
    // is_mtp_model(): true 表示本 ctx 用作 MTP 草稿模型。
    bool is_mtp_model() const;

    // has_mtp_modules(): true 表示该 HMM GGUF 里实际含 prefill_mtp.hmm /
    // decode_mtp.hmm 模块，可被创建为 LLAMA_CONTEXT_TYPE_MTP。
    bool has_mtp_modules() const;

    // MTP 草稿一次推理：吃 (target_hidden_state[n], token[n]) 返回
    //   logits[n] 以及新的 hidden_state[n] (用于 AR 串接)。
    //   - h_in:        [n_tokens, n_embd]，通过 llama.cpp 的 float* 通道承载
    //                  原始 fp16 hidden（每行有效数据为 n_embd * 2 bytes，
    //                  行步长仍为 n_embd * sizeof(float)）
    //   - tokens:      [n_tokens] 草稿当前要送入的 token id
    //   - logits_out:  每行 n_vocab 个 fp32; nullptr 行表示该位置不取 logits
    //   - h_out:       [n_tokens, n_embd]，同样以 float* 为 carrier 回写原始
    //                  fp16 hidden（供下一轮草稿继续 AR 推进）；
    //                  nullptr 表示不需要回写
    //   - start_pos:   该批第 0 个 token 的绝对位置；MTP 无 KV cache，必须
    //                  信任上层 speculative/seq_rm 后传入的位置，避免内部
    //                  自增 past_seq_length 在 rollback 后漂移。
    //   返回 0 成功，<0 失败。仅 MTP gguf (`is_mtp_model()`) 上有效。
    int houmo_decode_mtp(const float *h_in,
               const std::vector<llama_token> &tokens,
               int seq_id,
               std::vector<float *> &logits_out,
               float *h_out,
               int32_t start_pos = -1);

    // 设置 target ctx 的 pre-norm hidden 输出去向。
    // 调用约定: llama_context::houmo_decode 在调用 prefill/decode/verify 前
    // 设置 dst (= cparams.embeddings_pre_norm ? embd_pre_norm.data + base : nullptr)，
    // backend 在 Run() 完成后把 hidden_state ishmm output 拷过去。
    // dst==nullptr 表示不需要输出。
    // n_rows == 该次调用预计写出的行数 (与 logits_out.size() 对齐)。
    // row_stride_floats == n_embd_。
    void set_pre_norm_hidden_dst(float *dst, int n_rows);
    int n_decode_batch();
    int n_prefill_batch();
    uint32_t n_context_length();
    void lora_init(std::string file_name, void *adapter);
    void lora_set_scale(void *adapter, float scale);
    void lora_clear();
    class Impl;  // Forward declaration of the implementation class
  private:
    Impl *impl_;     // target/current ctx implementation
    Impl *impl_mtp_; // MTP implementation for single-GGUF MTP
};

struct kv_cache_item {
  std::vector<tcim::Tensor> tensors;
  std::vector<std::string> names;
  int seq_id = 0;
};
struct lora_item {
    void * adapter; //记录上层lora contex指针
    float scale;  // scale 0.0 表示disable， 1.0 表示enable
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
  virtual bool houmo_load(llama_model_loader & ml, std::vector<int> device_ids);
  virtual void houmo_init(houmo_memory_i *memory, int seq_max = 4,
                          ggml_abort_callback abort_callback = nullptr,
                          void *abort_callback_data = nullptr) = 0;
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
  virtual int houmo_asr_force_align(std::vector<llama_token> &tokens, int seq_id,
                                    std::vector<float *> &logits_out) {
      (void) tokens;
      (void) seq_id;
      (void) logits_out;
      return -1;
  }
  /**
   * @brief Batched verification for speculative decoding.
   * @details See HouMoLLModel::houmo_verify. Default fallback implementation
   * repeatedly calls houmo_prefill with a single token — correct but ~k×
   * slower than a proper one-shot implementation. Backends should override
   * this when they can produce per-position logits in a single forward.
   */
  virtual int houmo_verify(std::vector<llama_token> &tokens, int seq_id,
                           std::vector<float *> &logits_out) {
      if (tokens.size() != logits_out.size()) {
          return -1;
      }
      for (size_t i = 0; i < tokens.size(); ++i) {
          std::vector<llama_token> one = {tokens[i]};
          int ret = houmo_prefill(one, seq_id, logits_out[i]);
          if (ret != 0) {
              return ret;
          }
      }
      return 0;
  }
  virtual int houmo_verify_accept(int /*seq_id*/, uint16_t /*n_accepted*/) {
    return -1;
  }
  virtual bool is_embedding() {
    return false;
  }
  virtual bool is_asr_force_aligner() {
    return asr_force_aligner_;
  }
  bool has_decode_model() const { return decode_model_ != nullptr; }

  // ---------------- MTP hooks (default: no-op / not supported) ----------------
  // MTP 角色由具体 Impl 类型决定。
  virtual bool is_mtp_model() const { return wants_mtp_modules(); }
  virtual bool has_mtp_modules() const { return wants_mtp_modules(); }
  virtual bool wants_mtp_modules() const { return false; }

  // 默认不支持 MTP draft；只有 HoumoQwen35MTP 等具体 backend 重写。
  virtual int houmo_decode_mtp(const float * /*h_in*/,
                               const std::vector<llama_token> & /*tokens*/,
                               int /*seq_id*/,
                               std::vector<float *> & /*logits_out*/,
                               float * /*h_out*/,
                               int32_t /*start_pos*/ = -1) {
      return -1;
  }

  // 默认 no-op。Qwen3.5 在 Run() 后会把 ishmm 的 hidden_state 输出 memcpy
  // 到 pre_norm_hidden_dst_ (若非空)。
  void set_pre_norm_hidden_dst(float *dst, int n_rows) {
      pre_norm_hidden_dst_      = dst;
      pre_norm_hidden_dst_rows_ = n_rows;
  }
  int n_decode_batch() const { return n_decode_batch_; }
  int n_prefill_batch() const { return n_prefill_batch_; }
  uint32_t n_context_length() const { return context_length_;}
  void lora_init(std::string file_name, void *adapter);
  void lora_set_scale(void *adapter, float scale);
  void lora_clear();
  std::shared_ptr<HoumoEmbeddingLayer> embedding_layer() {
      return embedding_layer_;
  }
  void set_embedding_layer(std::shared_ptr<HoumoEmbeddingLayer> layer) {
      embedding_layer_ = layer;
  }
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
   * @details Parse the device ids form env.
   * @return The device ids.
   */
  void parse_deviceids();
  void read_tensor_to_buffer(const std::string &file_name, size_t offset,
                             size_t size, std::vector<char> &buffer);
  /**
   * @brief Parse the gguf parameters.(n_blocks, n_embd, n_vocab)
   * @details Parse the gguf parameters with the given ctx.
   * @param ctx The gguf context.
   * @param model_arch The model architecture.
   */
  void parser_gguf_parameters(struct gguf_context *ctx, std::string &model_arch);
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
  std::shared_ptr<tcim::Module>
  create_infer_engine(bool prefill, void *raw_data, int64_t size,
                      bool load_from_file = false,
                      std::vector<std::string> dummy_tensor_names = {});
  std::shared_ptr<tcim::Module>
  create_infer_engine(const std::string &file_name, size_t offset, size_t size,
                      std::vector<std::string> dummy_tensor_names = {});

protected:
  int prefill_length_ = 256;
  int context_length_ = 4096;
  std::map<int, int32_t> decode_valid_length_;
  int n_blocks_ = 28;
  int64_t n_embd_ = 3584;
  int64_t n_vocab_ = 30522;
  int n_prefill_batch_ = 1;
  int n_decode_batch_ = 1;
  int n_split_count_ = 1;

  std::shared_ptr<HoumoEmbeddingLayer> embedding_layer_ = nullptr;
  std::shared_ptr<tcim::Module> prefill_model_ = nullptr;
  std::shared_ptr<tcim::Module> decode_model_ = nullptr;
  // MTP 相关：
  //   - MTP 角色由 wants_mtp_modules() / 具体 Impl 类型决定。
  //   - pre_norm_hidden_dst_/_rows_: 调用方在 prefill/decode 前通过
  //                      set_pre_norm_hidden_dst() 注入的写出地址；Run() 后由
  //                      具体 backend 把 hidden_state ishmm 输出 memcpy 过去。
  //                      空指针表示本次调用无需写出。
  float *pre_norm_hidden_dst_     = nullptr;
  int    pre_norm_hidden_dst_rows_ = 0;
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
  std::string arch_name = "unknown";
  std::string kEmbeddingLengthEx = ".embedding_length";
  std::string kBlockCountEx = ".block_count";
  std::string kSpliteCount = "split.tensors.count";
  std::string kHmmInfo = "hmm.info";
  std::string kCacheKeyWord = "model_layers_";
  int kSeqMax = 8; // maximum sequence number for single-card model
  int device_num_ = 1;
  std::vector<int> device_ids_ = {0};
  bool enable_lazy_mode_ = false;
  bool asr_force_aligner_ = false;
  houmo_memory_i *memory_ = nullptr;
  std::map<std::string, lora_item> lora_map;
  tcim::Tensor lora_mask_float_;
  std::vector<std::string> input_names_;
  std::map<int32_t, std::vector<uint16_t>> input_embedding_ids_;
  std::map<int32_t, std::vector<llama_token>> input_tokens_;
  ggml_abort_callback abort_callback_      = nullptr;
  void *              abort_callback_data_ = nullptr;
};
