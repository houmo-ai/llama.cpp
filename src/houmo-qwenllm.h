#include "houmo-llmodel.h"
/**
 * @brief HoumoQwenLLM is the derived class for single-card HouMo model.
 */
class HoumoQwenLLM : public HouMoLLModel::Impl {
  public:
    HoumoQwenLLM() = default;
    ~HoumoQwenLLM() override = default;
    virtual void houmo_init(houmo_memory_i *memory, int seq_max = 4) override;
    virtual int houmo_prefill(std::vector<llama_token> &tokens, int seq_id,
                              float *logits) override;
    virtual int houmo_decode(std::vector<llama_token> &batches,
                             std::vector<int> &seq_ids,
                             std::vector<float *> logits) override;
    virtual int houmo_embedding(const std::map<int32_t, std::vector<llama_token>> &batches,
                        std::map<int, float *> &embeddings) override;
    virtual bool is_embedding() override;
  private:
    std::vector<llama_token> prefill_tokens_;
};