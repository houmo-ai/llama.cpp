#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include "houmo-llmodel.h"
#include "houmo-qwen35-state-cache.h"

/**
 * @brief HoumoQwen35LLM is the derived class for Qwen3.5 HouMo model.
 */
class HoumoQwen35LLM : public HouMoLLModel::Impl {
  friend class HoumoQwen35SeqState;
  public:
    HoumoQwen35LLM() = default;
    ~HoumoQwen35LLM() override = default;

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
    int houmo_embedding(const std::map<int32_t, std::vector<llama_token>> &batches,
                        std::map<int, float *> &embeddings) override;
    bool is_embedding() override;
    void state_write(llama_io_write_i & io, int local_seq_id,
             llama_state_seq_flags flags) const override;
    void state_read(llama_io_read_i & io, int local_seq_id,
            llama_state_seq_flags flags) override;
    int state_position(int local_seq_id) const override;

  protected:
    void resolve_input_names();
    void set_linear_attn_mask(tcim::Tensor &tensor, int fill_length,
                              int new_cache_length) const;
    std::vector<int32_t> build_rope_position_ids(int valid_length,
                                                 int length) const;
    int set_decode_rope_and_mask(const std::vector<int> &seq_ids);

    // 执行一轮 prefill。
    // - input_data：本轮 embeddings（文本走 embedding_layer_，vision 从 cache memcpy）
    // - *_pos_slice：每个指针至少 prefill_length_ 个 int32_t；三者相同即退化为 1D RoPE
    // - is_first_round：仅首轮绑定 LoRA mask 与 state/KV cache
    // 返回 0 成功，-1 表示 Run() 失败。
    int run_prefill_round(int seq_id,
                          tcim::Tensor &input_data,
                          int32_t valid_length,
                          int32_t current_length,
                          const int32_t *time_pos_slice,
                          const int32_t *height_pos_slice,
                          const int32_t *width_pos_slice,
                          bool is_first_round);

    bool has_vision_chunk(int seq_id) const;
    size_t chunk_token_count(const std::vector<PrefillChunk> &chunks) const;
    bool slice_snapshot_history(int seq_id, size_t begin, size_t end,
                  std::vector<PrefillChunk> &out) const;
    bool position_at_token_boundary(
      const std::vector<PrefillChunk> &chunks, size_t token_boundary,
      int32_t &position) const;
    void append_text_chunk(std::vector<PrefillChunk> &chunks,
                 const std::vector<llama_token> &tokens) const;
    void append_snapshot_text_history(int seq_id,
                      const llama_token *tokens,
                      size_t n_tokens);
    bool prepare_prefill_replay(
      int seq_id, int32_t kv_offset,
      const std::vector<PrefillChunk> &pending_chunks,
      bool require_snapshot,
      std::vector<PrefillChunk> &effective_chunks,
      int32_t &effective_kv_offset,
      int32_t &effective_position_base);
    void export_prefill_hidden(int32_t kv_offset, int32_t valid_length,
                               int32_t current_length);
    void maybe_save_prefill_checkpoint(int seq_id, size_t round, size_t rounds,
                                       int32_t valid_length,
                                       bool is_prompt_final_prefill);
    int copy_prefill_logits(float *logits, int32_t current_length);
    int set_lora_inputs(tcim::Module &model,
                        std::map<std::string, tcim::Tensor> &input_map);
    bool update_text_prefill_snapshot(
        int seq_id, int32_t effective_kv_offset, int32_t valid_length,
      const std::vector<PrefillChunk> &effective_chunks);

    // Vision support
    int houmo_prefill_vision(std::vector<llama_token> &tokens, int seq_id,
                             float *logits);
    void compute_vision_position_ids(
        int seq_id, int kv_offset,
        const std::vector<llama_token> &trailing_tokens,
        std::vector<int32_t> &time_positions,
        std::vector<int32_t> &height_positions,
        std::vector<int32_t> &width_positions,
        int32_t &final_position);
    void clear_vision_cache(int seq_id);

  protected:
    HoumoQwen35StateCache state_cache_;

  private:
    std::map<int32_t, std::vector<PrefillChunk>> chunk_history_;
    std::map<int, int32_t> decode_position_offset_;        // position offset for decode after vision
    std::string prefill_time_pos_name_;
    std::string prefill_height_pos_name_;
    std::string prefill_width_pos_name_;
    std::string prefill_valid_length_name_;
    std::string prefill_current_length_name_;
    std::string prefill_linear_mask_name_;

    std::vector<std::string> decode_time_pos_names_;
    std::vector<std::string> decode_height_pos_names_;
    std::vector<std::string> decode_width_pos_names_;
    std::vector<std::string> decode_linear_mask_names_;

    struct VerifyPending {
      bool valid = false;
      int32_t start_pos = 0;
      int n_verify = 0;
      std::vector<llama_token> tokens;
    };
    std::map<int, VerifyPending> pending_verify_;

    int commit_verify_accept(int seq_id, int accepted_steps);
};
