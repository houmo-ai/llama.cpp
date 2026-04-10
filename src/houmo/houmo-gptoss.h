#include "houmo-llmodel.h"
/**
 * @brief HouMoGPTOSSModel is the derived class for GPT-OSS model.
 * @details GPT-OSS model is a single-card model.
 * @note GPT-OSS model only supports M50
 */

class HouMoGPTOSS : public HouMoLLModel::Impl {
  public:
    HouMoGPTOSS() = default;
    ~HouMoGPTOSS() override = default;

    virtual void houmo_init(houmo_memory_i *memory, int seq_max = 4,
                          ggml_abort_callback abort_callback = nullptr,
                          void *abort_callback_data = nullptr) override;
    virtual int houmo_prefill(std::vector<llama_token> &tokens, int seq_id,
                              float *logits) override;
    virtual int houmo_decode(std::vector<llama_token> &batches,
                             std::vector<int> &seq_ids,
                             std::vector<float *> logits) override;

  private:
    int set_attentions_for_prefill(int current_length, int valid_length);
    int set_attentions_for_decode(std::vector<int> &seq_ids);
    std::vector<float> generate_mask(int batch_size, int query_length,
                                     int past_seq_length, bool is_local);
    std::vector<uint16_t> generate_mask_f16(int batch_size, int query_length,
                                           int past_seq_length, bool is_local);

  private:
    int32_t sliding_window_ = 128;
    std::vector<std::string> local_attention_names_;
    std::vector<std::string> global_attention_names_;
};