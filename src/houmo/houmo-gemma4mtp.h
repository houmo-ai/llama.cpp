#pragma once

#include "houmo-gemma4.h"

#include <cstdint>
#include <utility>
#include <string>
#include <vector>

struct HouMoGemma4MTPNames {
    std::string mtp_input;
    std::string valid_length;
    std::string current_length;
    std::string local_mask;
    std::string logits_out;
    std::string hidden_out;
};

// Gemma4 MTP draft implementation.
// The target Gemma4 path stays in HouMoGemma4; this class only loads/runs
// decode_mtp.hmm. Its shared KV inputs are bound to the target prefill cache.
class HouMoGemma4MTP : public HouMoGemma4 {
  public:
    explicit HouMoGemma4MTP(HouMoGemma4 *target = nullptr) : target_(target) {}
    ~HouMoGemma4MTP() override = default;

    void houmo_init(houmo_memory_i *memory, int seq_max = 4,
                    ggml_abort_callback abort_callback = nullptr,
                    void *abort_callback_data = nullptr) override;

    int houmo_decode_mtp(const float *h_in,
                         const std::vector<llama_token> &tokens,
                         int seq_id,
                         std::vector<float *> &logits_out,
                         float *h_out,
                         int32_t start_pos = -1) override;

    bool wants_mtp_modules() const override { return true; }

  private:
    int houmo_decode_mtp_one(const float *h_in,
                             llama_token token,
                             int seq_id,
                             float *logits_out,
                             float *h_out,
                             int32_t start_pos);
    void align_draft_memory(int seq_id, int32_t start_pos);
    void resolve_mtp_names(tcim::Module *model);
    std::vector<std::pair<std::string, tcim::Tensor>> collect_target_cache_inputs() const;
    void bind_target_cache_inputs();

  private:
    HouMoGemma4MTPNames mtp_names_;
    HouMoGemma4 *target_ = nullptr;
    std::vector<std::pair<std::string, tcim::Tensor>> target_cache_inputs_;
};