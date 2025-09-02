#include "houmo-llmodel.h"
/**
 * @brief HouMoVLLModel is a vision language model for HouMoLLModel.
 * @ 暂时分开实现，后续考虑合并到houmo-llmmode.h
 * @TODO: 多用户的支持..
 */
class HouMoVLLModel : public HouMoLLModel::Impl {
  public:
    HouMoVLLModel();
    ~HouMoVLLModel() override = default;

    bool houmo_load(const std::string &file_name) override;
    void houmo_init(int seq_max = 4) override;
    int houmo_prefill(std::vector<llama_token> &tokens, int seq_id,
                      float *logits) override;
    int houmo_prefill(const float *embeddings, int n_tokens, int seq_id,
                      float *logits) override;
    int houmo_decode(std::vector<llama_token> &batches,
                     std::vector<int> &seq_ids,
                     std::vector<float *> logits) override;

  private:
    void update_position_ids(int seq_id, int n_tokens, bool is_embedding);
    void reset_position_ids();
  private:
    int kSeqMax = 4; // maximum sequence number for single-card model
    std::shared_ptr<tcim::Module> prefill_model_;
    std::shared_ptr<tcim::Module> decode_model_;
    int ny = 13; // 364/patch_size_(14)*2 = 13;
    int nx = 23; // 644/patch_size_(14)*2 = 23;
    std::vector<int16_t> input_embedding_ids_;
    int32_t offset_embedding_id_ = 0;
    std::vector<int32_t> time_position_ids_;
    std::vector<int32_t> hight_position_ids_;
    std::vector<int32_t> width_position_ids_;
    int32_t start_pos_ = 0;
    const std::string kVisualHeight = "hmm.vit.height";
    const std::string kVisualWidth = "hmm.vit.width";
};
