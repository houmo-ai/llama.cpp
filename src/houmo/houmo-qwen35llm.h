#pragma once

#include <map>
#include <string>
#include <vector>
#include "houmo-llmodel.h"

/**
 * @brief HoumoQwen35LLM is the derived class for Qwen3.5 HouMo model.
 */
class HoumoQwen35LLM : public HouMoLLModel::Impl {
  public:
    HoumoQwen35LLM() = default;
    ~HoumoQwen35LLM() override = default;

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
    int houmo_embedding(const std::map<int32_t, std::vector<llama_token>> &batches,
                        std::map<int, float *> &embeddings) override;
    bool is_embedding() override;

  private:
    void resolve_input_names();
    void init_cache_io_tensors();
    void init_state_cache(int seq_max);
    void set_state_cache_for_prefill(int seq_id);
    void set_state_cache_for_decode(const std::vector<int> &seq_ids);
    void set_linear_attn_mask(tcim::Tensor &tensor, int fill_length,
                              int new_cache_length) const;
    std::vector<int32_t> build_rope_position_ids(int valid_length,
                                                 int length) const;
    int set_decode_rope_and_mask(const std::vector<int> &seq_ids);

    // State checkpoint helpers for prefix cache reuse
    int find_common_prefix_length(const std::vector<llama_token> &saved,
                                  const std::vector<llama_token> &incoming) const;
    void save_state_checkpoint(int seq_id, int position);
    int find_and_restore_best_checkpoint(int seq_id, int max_position);

  private:
    std::map<int32_t, std::vector<uint16_t>> input_embedding_ids_;
    std::string prefill_time_pos_name_;
    std::string prefill_height_pos_name_;
    std::string prefill_width_pos_name_;
    std::string prefill_valid_length_name_;
    std::string prefill_current_length_name_;
    std::string prefill_linear_mask_name_;

    std::vector<std::string> decode_time_pos_names_;
    std::vector<std::string> decode_height_pos_names_;
    std::vector<std::string> decode_width_pos_names_;
    std::vector<std::string> decode_linear_mask_names_;

    struct StateCacheEntry {
      std::vector<tcim::Tensor> tensors;
    };

    std::vector<std::string> state_cache_names_;
    std::vector<std::string> state_cache_output_names_;
    std::map<int, StateCacheEntry> state_cache_;
    std::map<int, bool> state_cache_initialized_;

    // Indices into state_cache_names_ / StateCacheEntry::tensors that
    // correspond to recurrent state (conv_cache + recurrent_state) — only
    // these are saved/restored in checkpoints.  KV cache tensors are managed
    // separately and must NOT be included.
    std::vector<size_t> checkpoint_tensor_indices_;

    // Prefix cache with periodic checkpoints for state reuse
    int kCheckpointInterval = prefill_length_ * 2;

    // Where to store state checkpoint tensors.
    //   kHost   — copy to host (CPU) memory; saves device memory but
    //             save/restore involves device↔host transfers.
    //   kDevice — keep on device memory; faster save/restore (device↔device
    //             copy) but consumes extra device memory per checkpoint.
    enum class CheckpointStorage { kHost, kDevice };
    CheckpointStorage checkpoint_storage_ = CheckpointStorage::kDevice;

    struct StateCheckpoint {
        int position = 0;                           // token position after processing
        std::vector<tcim::Tensor> state_tensors;    // copies of device state (host or device per checkpoint_storage_)
    };

    struct PrefillSnapshot {
        std::vector<llama_token> tokens;               // cumulative tokens processed
        std::map<int, StateCheckpoint> checkpoints;    // position -> checkpoint
        int last_processed_pos = 0;                    // state cache valid up to this position
        bool valid = false;
    };
    std::map<int, PrefillSnapshot> prefill_snapshots_;
};
