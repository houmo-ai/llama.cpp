#include "houmo-llmodel.h"
/**
 * @brief HouMoVLLModel is a vision language model for HouMoLLModel.
 * @ 暂时分开实现，后续考虑合并到houmo-llmmode.h
 */
class HouMoVLLModel : public HouMoLLModel::Impl {
  public:
    HouMoVLLModel();
    ~HouMoVLLModel() override = default;
    virtual void houmo_init(houmo_memory_i *memory, int seq_max = 4,
                          ggml_abort_callback abort_callback = nullptr,
                          void *abort_callback_data = nullptr) override;
    virtual int houmo_prefill(std::vector<llama_token> &tokens, int seq_id,
                      float *logits) override;
    virtual int houmo_prefill(const float *embeddings, int n_tokens, int seq_id,
                      float *logits) override;
    virtual int houmo_decode(std::vector<llama_token> &batches,
                             std::vector<int> &seq_ids, std::vector<float *> logits) override;
  private:
    void update_position_ids(int n_tokens,
                             std::vector<int32_t> &time_position_ids,
                             std::vector<int32_t> &high_position_ids,
                             std::vector<int32_t> &width_position_ids,
                             int32_t &position_index);
    void compute_position_ids(int seq_id,
                              std::vector<int32_t> &time_position_ids,
                              std::vector<int32_t> &high_position_ids,
                              std::vector<int32_t> &width_position_ids);
    void reset_position_ids(int seq_id);

    void fill_deepstack_embed(int seq_id, int valid_length);

  private:
    int ny = 13; // 364/patch_size_(14)*2 = 13;
    int nx = 23; // 644/patch_size_(14)*2 = 23;
    std::vector<std::vector<int16_t>> input_embedding_ids_;
    std::vector<std::vector<int16_t>> deepstack_features_0_; // only for qwen3vl
    std::vector<std::vector<int16_t>> deepstack_features_1_; // only for qwen3vl
    std::vector<std::vector<int16_t>> deepstack_features_2_; // only for qwen3vl
    std::vector<int32_t> offset_embedding_ids_;
    std::vector<std::vector<int32_t>> prefill_input_lengths_;
    std::vector<int32_t> position_indexs_;
    std::vector<std::vector<int32_t>> chunk_history_;
};
