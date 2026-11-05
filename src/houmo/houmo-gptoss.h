#include "houmo-llmodel.h"
#include "oss-kv-cache.h"
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
    void state_write(llama_io_write_i & io, int local_seq_id,
             llama_state_seq_flags flags) const override;
    void state_read(llama_io_read_i & io, int local_seq_id,
            llama_state_seq_flags flags) override;
    int state_position(int local_seq_id) const override;

  private:
    int set_attentions_for_prefill(int current_length, int valid_length);
    int set_attentions_for_decode(std::vector<int> &seq_ids);
    bool device_kv_covers_history(int seq_id, int64_t valid_length) const;
    void update_device_kv_range(int seq_id, int64_t abs_end);
    void clear_device_kv_range(int seq_id);
    std::vector<float> generate_mask(int batch_size, int query_length,
                                     int past_seq_length, bool is_local);
    std::vector<uint16_t> generate_mask_f16(int batch_size, int fill_length,
                         int query_length,
                         int past_seq_length, bool is_local);

  private:
    int32_t sliding_window_ = 128;
    HoumoOSSKVCache kv_reuse_cache_;
    std::map<int32_t, std::vector<llama_token>> input_history_;
    std::map<int32_t, int64_t> device_kv_abs_start_;
    std::map<int32_t, int64_t> device_kv_abs_end_;
    std::vector<std::string> prefill_local_attention_names_;
    std::vector<std::string> prefill_global_attention_names_;
    std::vector<std::string> local_attention_names_;
    std::vector<std::string> global_attention_names_;
};