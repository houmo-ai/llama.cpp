#pragma once

#include <map>
#include <string>
#include <vector>

#include "llama-batch.h"

// houmo llm model defination
//
// @guoxing.xu 2025-05-23
//

/**
 * @brief HouMoLLModel is the context for HouMo model.
 * @details It provides the interface for HouMo model inference.
 */
class HouMoLLModel {
  public:
    HouMoLLModel();
    ~HouMoLLModel();

    bool houmo_load(const std::string &file_name);
    void houmo_init(int seq_max = 4);
    int houmo_prefill(std::vector<llama_token> &tokens, int seq_id,
                      float *logits);
    int houmo_decode(std::vector<llama_token> &batches,
                     std::vector<int> &seq_ids, std::vector<float *> logits);
    int n_decode_batch();
    int n_prefill_batch();
    uint32_t n_context_length();
    class Impl;  // Forward declaration of the implementation class
  private:
    Impl *impl_; // Pointer to the implementation
};