#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include "houmo-llmodel.h"

#pragma pack(push, 1)
struct VisionCompactHeader {
    int16_t magic;
    int16_t nx;
    int16_t ny;
    int16_t chunksize;
};
#pragma pack(pop)

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
    void init_state_cache(int seq_max);
    void set_state_cache_for_prefill(int seq_id);
    void set_state_cache_for_decode(const std::vector<int> &seq_ids);
    void set_linear_attn_mask(tcim::Tensor &tensor, int fill_length,
                              int new_cache_length) const;
    std::vector<int32_t> build_rope_position_ids(int valid_length,
                                                 int length) const;
    int set_decode_rope_and_mask(const std::vector<int> &seq_ids);

    // 执行一轮 prefill。
    // - input_data：本轮 embeddings（文本走 embedding_layer_，vision 从 cache memcpy）
    // - *_pos_slice：每个指针至少 prefill_length_ 个 int32_t；三者相同即退化为 1D RoPE
    // - is_first_round：仅首轮绑定 LoRA mask 与 state/KV cache
    // 返回 0 成功，-1 表示 Run() 失败。
    int run_prefill_round(int seq_id,
                          tcim::Tensor &input_data,
                          int32_t valid_length,
                          int32_t current_length,
                          const int32_t *time_pos_slice,
                          const int32_t *height_pos_slice,
                          const int32_t *width_pos_slice,
                          bool is_first_round);

    // State checkpoint helpers for prefix cache reuse
    int find_common_prefix_length(const std::vector<llama_token> &saved,
                                  const std::vector<llama_token> &incoming) const;
    void save_state_checkpoint(int seq_id, int position);
    int find_and_restore_best_checkpoint(int seq_id, int max_position);
    void restore_zero_state(int seq_id);
    void trim_snapshot_to_kv_offset(int seq_id, int32_t kv_offset);

    // Vision support
    int houmo_prefill_vision(std::vector<llama_token> &tokens, int seq_id,
                             float *logits);
    void compute_vision_position_ids(
        int seq_id, int kv_offset,
        const std::vector<llama_token> &trailing_tokens,
        std::vector<int32_t> &time_positions,
        std::vector<int32_t> &height_positions,
        std::vector<int32_t> &width_positions,
        int32_t &final_position);
    void clear_vision_cache(int seq_id);

  private:
    // Vision embedding cache and chunk tracking
    struct ChunkInfo {
        int size;       // number of tokens in this chunk
        bool is_vision; // true = vision embedding, false = text
        int nx;         // grid width after merge (only for vision)
        int ny;         // grid height after merge (only for vision)
        // 文本 chunk 的真实 token ids；vision chunk 为空。
        // vision flush 后用于回填 snapshot.tokens，便于后续重生成/前缀复用。
        std::vector<llama_token> text_tokens;
    };
    std::map<int, std::vector<int16_t>> embedding_cache_; // accumulated embeddings (fp16)
    std::map<int, std::vector<ChunkInfo>> pending_chunks_; // chunk types
    std::map<int, int32_t> decode_position_offset_;        // position offset for decode after vision

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
    // Device-side zero-initialized state tensors for fast initialization
    std::vector<tcim::Tensor> zero_state_tensors_;

    // Indices into state_cache_names_ / StateCacheEntry::tensors that
    // correspond to recurrent state (conv_cache + recurrent_state) — only
    // these are saved/restored in checkpoints.  KV cache tensors are managed
    // separately and must NOT be included.
    std::vector<size_t> checkpoint_tensor_indices_;

    // Prefix cache with periodic checkpoints for state reuse.
    // checkpoint_min_interval_: minimum distance (in tokens) between two
    //   consecutive checkpoints.  Prevents over-saving during fast prefill.
    // checkpoint_max_count_: maximum number of checkpoints kept per seq_id.
    //   When exceeded, the checkpoint closest to a neighbor is evicted
    //   (preserving first and last for maximum rollback coverage).
    int checkpoint_min_interval_ = prefill_length_ * 4;
    size_t checkpoint_max_count_ = 40;

    // Where to store state checkpoint tensors.
    //   kHost   — copy to host (CPU) memory; saves device memory but
    //             save/restore involves device↔host transfers.
    //   kDevice — keep on device memory; faster save/restore (device↔device
    //             copy) but consumes extra device memory per checkpoint.
    enum class CheckpointStorage { kHost, kDevice };
    CheckpointStorage checkpoint_storage_ = CheckpointStorage::kDevice;

    struct StateCheckpoint {
        int position = 0;                           // token position after processing
        bool used = false;                          // slot occupancy flag (pool-managed)
        // Pre-allocated state tensor buffers — owned by the slot for the
        // entire snapshot lifetime.  save/restore only memcpy via CopyTo,
        // avoiding per-checkpoint alloc/free that fragments memory.
        std::vector<tcim::Tensor> state_tensors;
    };

    struct PrefillSnapshot {
        std::vector<llama_token> tokens;               // cumulative tokens processed
        // Fixed-size pre-allocated checkpoint pool (size == checkpoint_max_count_).
        // Slots are reused via the used flag — no dynamic alloc/free per save.
        std::vector<StateCheckpoint> checkpoint_pool;
        // Position → pool slot index for ordered lookup (upper_bound, gap eviction).
        std::map<int, size_t> checkpoint_index;
        int last_processed_pos = 0;                    // state cache valid up to this position
        bool valid = false;
        // tokens 中保证为真实 token id 的最小绝对位置。
        // [0, first_safe_position) 可能是代表 vision embedding 的占位 0，
        // 不能作为 gap-replay 起点。0 表示整段历史均为真实文本。
        int first_safe_position = 0;
    };
    std::map<int, PrefillSnapshot> prefill_snapshots_;

    // Mark all slots as free and clear the index.  Pre-allocated tensor
    // buffers are kept — only the occupancy bookkeeping is reset.
    void clear_all_checkpoints(PrefillSnapshot &snapshot);
    // Allocate the pre-allocated checkpoint pool for one seq_id.  Called from
    // init_state_cache after state_cache_names_/checkpoint_tensor_indices_
    // are built.  Allocates checkpoint_max_count_ slots, each with Cloned
    // (kDevice) or host-allocated (kHost) tensors of the right shape.
    void init_checkpoint_pool(int seq_id);
    // Find a free slot (or evict the gap-best in-use slot) and return its
    // index inside snapshot.checkpoint_pool.  Caller is responsible for
    // populating the slot and inserting into checkpoint_index.
    size_t acquire_checkpoint_slot(PrefillSnapshot &snapshot, int position);
    // Mark all slots whose position > after_pos as free and erase from index.
    void release_checkpoints_after(PrefillSnapshot &snapshot, int after_pos);
};
