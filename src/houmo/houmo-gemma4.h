#pragma once

#include "houmo-gemma4-kv-cache.h"
#include "houmo-llmodel.h"

#include <cstdint>
#include <string>
#include <vector>

class HouMoGemma4MTP;

/**
 * @brief HouMoGemma4 is the derived class for Gemma4 HouMo VLM models.
 * @details The runtime input layout follows src/houmo/demo.py:
 *          input_embeds, valid_length, current_length, full_attention_mask,
 *          sliding_attention_mask, plus KV cache inputs.
 *          Decode stage only feeds sliding/local attention mask.
 */
class HouMoGemma4 : public HouMoLLModel::Impl {
    friend class HouMoGemma4MTP;

  public:
    HouMoGemma4() = default;
    ~HouMoGemma4() override = default;
    bool houmo_load(llama_model_loader &ml, std::vector<int> device_ids) override;
    void houmo_init(houmo_memory_i *memory, int seq_max = 4,
                    ggml_abort_callback abort_callback = nullptr,
                    void *abort_callback_data = nullptr) override;
    int houmo_prefill(std::vector<llama_token> &tokens, int seq_id,
                      float *logits) override;
    int houmo_prefill(const float *embeddings, int n_tokens, int seq_id,
                      float *logits) override;
    int houmo_decode(std::vector<llama_token> &batches,
                     std::vector<int> &seq_ids,
                     std::vector<float *> logits) override;
    int houmo_verify(std::vector<llama_token> &tokens, int seq_id,
             std::vector<float *> &logits_out) override;
    int houmo_verify_accept(int seq_id, uint16_t n_accepted) override;

  private:
    int32_t committed_pos(int seq_id) const {
        return memory_ ? memory_->seq_pos_min(seq_id) : -1;
    }

  protected:
    void resolve_attention_names();
    int set_per_layer_inputs_for_prefill(const llama_token *tokens,
                                         int current_length);
    int set_per_layer_inputs_for_decode(std::vector<llama_token> &batches,
                                        std::vector<int> &seq_ids);
    int set_attentions_for_prefill(int current_length, int valid_length,
                                   int round_start, int seq_id);
    int set_attentions_for_decode(std::vector<int> &seq_ids);
    std::vector<uint16_t> generate_mask_f16(int batch_size, int fill_length,
                                            int current_length,
                                            int past_seq_length, int width,
                                            bool is_local,
                                            const std::vector<PrefillChunk> *chunks,
                                            int round_start) const;
    void clear_cached_inputs(int seq_id);
    bool has_cached_inputs(int seq_id) const;
    size_t cached_token_count(int seq_id) const;
    bool device_kv_covers_history(int seq_id, int64_t valid_length) const;
    void update_device_kv_range(int seq_id, int64_t abs_end);
    void clear_device_kv_range(int seq_id);
    size_t history_token_count(int seq_id) const;
    bool slice_history(int seq_id, size_t begin, size_t end,
               std::vector<PrefillChunk> &out) const;
    void truncate_history(int seq_id, size_t keep_tokens);
    void append_history(int seq_id, const std::vector<PrefillChunk> &chunks);
    void append_history_token(int seq_id, size_t position, llama_token token);

    int32_t sliding_window_ = 1024;
    std::shared_ptr<HoumoEmbeddingLayer> per_layer_embedding_layer_ = nullptr;
    std::string prefill_per_layer_input_name_;
    std::string decode_per_layer_input_name_;
    std::vector<std::string> prefill_local_attention_names_;
    std::vector<std::string> prefill_global_attention_names_;
    std::vector<std::string> local_attention_names_;
    std::vector<std::string> global_attention_names_;
    bool visual_bidirectional_attention_ = true;

    // Inputs waiting for the next graph prefill. Chunks retain their token IDs
    // or fp16 embeddings so mixed text/VIT input can be replayed in order.
    std::map<int32_t, std::vector<PrefillChunk>> chunk_history_;
    // Successfully committed input history, including VIT embeddings and
    // visual attention metadata required by rollback/replay.
    std::map<int32_t, std::vector<PrefillChunk>> input_history_;
    std::map<int, int64_t> device_kv_abs_start_;
    std::map<int, int64_t> device_kv_abs_end_;
    std::string accepted_count_input_name_;
    std::map<int, int32_t> last_committed_tokens_;
    struct VerifyPending {
      bool valid = false;
      int32_t start_pos = 0;
      int32_t n_verify = 0;
      std::vector<llama_token> tokens;
    };
    std::map<int, VerifyPending> pending_verify_;
    HoumoGemma4KVCache kv_reuse_cache_;
};
