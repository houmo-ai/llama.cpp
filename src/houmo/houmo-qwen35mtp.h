#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include "houmo-qwen35llm.h"

// Qwen3.5 MTP-only metadata extracted from houmo-qwen35llm.*
//
// This header intentionally keeps only MTP-specific naming/state structures so
// MTP implementation can evolve independently from non-MTP qwen35 target logic.
struct HoumoQwen35MtpNameSet {
    std::string hidden_in;          // hidden_states / post_norm_hidden / pre_norm_hidden
    std::string input_embedding_in; // input_embedding / next_token_embedding
    std::vector<std::string> position_ids;
    std::string past_seq_length;
    std::string current_input_length;
    std::string past_key_cache_in;
    std::string past_value_cache_in;
    std::string mtp_logits_out;
    std::string mtp_hidden_out;
};

struct HoumoQwen35MtpState {
    HoumoQwen35MtpNameSet mtp_prefill;
    HoumoQwen35MtpNameSet mtp_decode;
    std::map<int, int32_t> mtp_past_seq_len;
    // Track the MTP TCIM device cache end position per seq_id.
    // After speculative rollback (load_dft + seq_rm), the llama context
    // position is restored while rejected draft entries can remain physically
    // in the TCIM cache. The current TCIM runtime does not reliably mask that
    // stale tail using only past_seq_length/position_ids, so rollback clears
    // the MTP device cache and rewinds this tracker before rebuilding context.
    std::map<int, int32_t> mtp_cache_end;
};

// Dedicated MTP implementation for qwen3.5.
//
// Non-MTP target logic remains in HoumoQwen35LLM (houmo-qwen35llm.cpp), while
// all MTP-only name resolution / inference lives here.
class HoumoQwen35MTP : public HoumoQwen35LLM {
  public:
    HoumoQwen35MTP() = default;
    ~HoumoQwen35MTP() override = default;

    void houmo_init(houmo_memory_i *memory, int seq_max = 4,
            ggml_abort_callback abort_callback = nullptr,
            void *abort_callback_data = nullptr) override;

    int houmo_decode_mtp(const float *h_in,
               const std::vector<llama_token> &tokens, int seq_id,
               std::vector<float *> &logits_out, float *h_out,
               int32_t start_pos = -1) override;
    bool wants_mtp_modules() const override { return true; }

  private:
    void init_mtp_model_names_fixed(tcim::Module *model,
                                    HoumoQwen35MtpNameSet *out);

    // Separate MTP prefill (multi-token) and decode (single-token) paths.
    // Both share the same TCIM cache; split for clearer logging/debugging.
    int run_mtp_prefill(const float *h_in,
                        const std::vector<llama_token> &tokens,
                        int seq_id,
                        std::vector<float *> &logits_out,
                        float *h_out,
                        int32_t start_pos);

    int run_mtp_decode(const float *h_in,
                       const std::vector<llama_token> &tokens,
                       int seq_id,
                       std::vector<float *> &logits_out,
                       float *h_out,
                       int32_t start_pos);

    // Zero the MTP TCIM device cache (past_key_cache + past_value_cache)
    // on both prefill and decode modules. Called after speculative rollback.
    void clear_mtp_device_cache(int seq_id);

  private:
    HoumoQwen35MtpState mtp_state_;
};
