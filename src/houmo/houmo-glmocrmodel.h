#pragma once

#include "houmo-llmodel.h"

#include <cstdint>
#include <vector>

class HouMoGLMOCRModel : public HouMoLLModel::Impl {
  public:
    HouMoGLMOCRModel() = default;
    ~HouMoGLMOCRModel() override = default;

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

  private:
    struct PositionChunk {
        int n_tokens = 0;
        int nx = 0;
        int ny = 0;
        bool image = false;
    };

    void append_positions(const PositionChunk &chunk,
                          std::vector<int32_t> &time_position_ids,
                          std::vector<int32_t> &height_position_ids,
                          std::vector<int32_t> &width_position_ids,
                          int32_t &start_pos) const;
    void compute_position_ids(int seq_id,
                              std::vector<int32_t> &time_position_ids,
                              std::vector<int32_t> &height_position_ids,
                              std::vector<int32_t> &width_position_ids);
    int32_t replay_position_until(int seq_id, int target_tokens) const;
    void set_position_input(tcim::Tensor &tensor,
                            const std::vector<int32_t> &time_position_ids,
                            const std::vector<int32_t> &height_position_ids,
                            const std::vector<int32_t> &width_position_ids,
                            size_t offset);
    void set_decode_position_input(tcim::Tensor &tensor,
                                   const std::vector<int32_t> &position_vec);
    void reset_cached_inputs(int seq_id);

  private:
    houmo_memory_i *memory_ = nullptr;
    std::vector<std::vector<int16_t>> input_embedding_ids_;
    std::vector<int32_t> offset_embedding_ids_;
    std::vector<int32_t> position_indexs_;
    std::vector<std::vector<PositionChunk>> prefill_chunks_;
    std::vector<std::vector<PositionChunk>> chunk_history_;
};
