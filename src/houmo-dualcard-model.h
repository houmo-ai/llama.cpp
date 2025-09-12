#include "houmo-llmodel.h"
/**
 * @brief HouMoDualCardModel is the derived class for dual-card HouMo model.
 * @details for 14B+ model, 适合xh1拆分模型的情况。
 */
class HouMoDualCardModel : public HouMoLLModel::Impl {
  public:
    HouMoDualCardModel() = default;
    ~HouMoDualCardModel() override = default;

    bool houmo_load(const std::string &file_name) override;
    void houmo_init(int seq_max = 4) override;
    int houmo_prefill(std::vector<llama_token> &tokens, int seq_id,
                      float *logits) override;
    int houmo_decode(std::vector<llama_token> &batches,
                     std::vector<int> &seq_ids,
                     std::vector<float *> logits) override;
  private:
    std::shared_ptr<tcim::Module> prefill_model_card0_;
    std::shared_ptr<tcim::Module> prefill_model_card1_;
    std::shared_ptr<tcim::Module> decode_model_card0_;
    std::shared_ptr<tcim::Module> decode_model_card1_;
    std::vector<std::string> decode_inner_output_names_;
    std::vector<std::string> prefill_inner_output_names_;
    int kSeqMax = 2;        // maximum sequence number for dual-card model
    size_t split_size_ = 0; // split size for kv cache
};
