#pragma once

#include "houmo-llmodel.h"

#include <cstdint>
#include <string>
#include <vector>

/**
 * @brief HouMoGemma4 is the derived class for Gemma4 HouMo VLM models.
 * @details The runtime input layout follows src/houmo/demo.py:
 *          input_embeds, valid_length, current_length, local_attention_mask,
 *          global_attention_mask, plus KV cache inputs.
 */
class HouMoGemma4 : public HouMoLLModel::Impl {
  public:
    HouMoGemma4() = default;
    ~HouMoGemma4() override = default;

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

  private:
    struct ChunkInfo {
        int size = 0;
        bool is_vision = false;
    };

    void resolve_attention_names();
    int set_attentions_for_prefill(int current_length, int valid_length,
                                   int round_start, int seq_id);
    int set_attentions_for_decode(std::vector<int> &seq_ids);
    std::vector<uint16_t> generate_mask_f16(int batch_size, int fill_length,
                                            int current_length,
                                            int past_seq_length, int width,
                                            bool is_local,
                                            const std::vector<ChunkInfo> *chunks,
                                            int round_start) const;
    void clear_cached_inputs(int seq_id);
    bool has_cached_inputs(int seq_id) const;
    size_t cached_token_count(int seq_id) const;

  private:
    int32_t sliding_window_ = 1024;
    std::vector<std::string> local_attention_names_;
    std::vector<std::string> global_attention_names_;

    // External fp16 embeddings supplied via llama_batch.embd.  This is how
    // the Gemma4 vision tower output from demo.py is passed into the LLM graph.
    std::map<int, std::vector<uint16_t>> embedding_cache_;
    std::map<int, std::vector<ChunkInfo>> pending_chunks_;
};
