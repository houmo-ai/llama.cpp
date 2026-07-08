#include "houmo-gemma4.h"

#include "llama-impl.h"
#include "time-perf.h"
#include "ggml.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace {
static constexpr uint16_t F16_NEG_MIN_UINT = 0xFBFF; // torch.finfo(torch.float16).min == -65504

int aligned(int size, int align) {
    return ((size + align - 1) / align) * align;
}

int tensor_dim_or(const tcim::Tensor &tensor, size_t dim, int fallback) {
    const auto shape = tensor.Info().Shape();
    if (dim < shape.size()) {
        return static_cast<int>(shape[dim]);
    }
    return fallback;
}

bool is_per_layer_input(const std::string &name) {
    return name.find("per_layer_inputs") != std::string::npos;
}

bool is_local_mask(const std::string &name) {
    return name.find("local_attention_mask") != std::string::npos ||
           name.find("sliding_attention_mask") != std::string::npos;
}

bool is_global_mask(const std::string &name) {
    return name.find("global_attention_mask") != std::string::npos ||
           name.find("full_attention_mask") != std::string::npos;
}

bool is_accepted_count_input(const std::string &name) {
    return name == "accepted_count" || name.find("accepted_count.") == 0;
}

int top_token_id(const float *logits, size_t n) {
    if (logits == nullptr || n == 0) {
        return -1;
    }
    int id = 0;
    float best = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < n; ++i) {
        if (logits[i] > best) {
            best = logits[i];
            id = static_cast<int>(i);
        }
    }
    return id;
}

} // namespace

bool HouMoGemma4::houmo_load(llama_model_loader &ml, std::vector<int> device_ids) {
    HouMoLLModel::Impl::houmo_load(ml, device_ids);
    std::string file_name = ml.fname_;

    // 加载 per_layer_embedding.bin
    gguf_context *ctx = ml.metadata_ptr.get();
    const int n_tensors = gguf_get_n_tensors(ctx);
    for (int i = 0; i < n_tensors; ++i) {
        const char *name = gguf_get_tensor_name(ctx, i);
        if (name == nullptr) {
            continue;
        }
        if (strcmp(name, "per_layer_embedding.bin") == 0) {
            if (per_layer_embedding_layer_ == nullptr) {
                per_layer_embedding_layer_ =
                    std::make_shared<HoumoEmbeddingLayer>(
                        file_name,
                        gguf_get_tensor_offset(ctx, i) +
                            gguf_get_meta_size(ctx),
                        gguf_get_tensor_size(ctx, i));
                break;
            }
        }
    }
    // 解析gemma4.attention.shared_kv_layers
    auto key_id = gguf_find_key(ctx, "gemma4.attention.shared_kv_layers");
    if (key_id < 0) {
        LLAMA_LOG_INFO("%s: not found key %s in gguf file\n", __func__,
                       "gemma4.attention.shared_kv_layers");
    } else {
        auto shared_layers = gguf_get_val_u32(ctx, key_id);
        n_blocks_ -= shared_layers;
        LLAMA_LOG_INFO("%s: gemma4.attention.shared_kv_layers = %u, n_blocks_ = %d\n", __func__, shared_layers, n_blocks_);
    }
    key_id = gguf_find_key(ctx, "gemma4.attention.sliding_window");
    if (key_id < 0) {
        LLAMA_LOG_INFO("%s: not found key %s in gguf file\n", __func__,
                       "gemma4.attention.sliding_window");
    } else {
        sliding_window_ = gguf_get_val_u32(ctx, key_id);
        LLAMA_LOG_INFO("%s: gemma4.attention.sliding_window = %u\n", __func__, sliding_window_);
    }
    return true;
}

void HouMoGemma4::houmo_init(houmo_memory_i *memory, int seq_max,
                             ggml_abort_callback abort_callback,
                             void *abort_callback_data) {
    abort_callback_data_ = abort_callback_data;
    abort_callback_ = abort_callback;
    model_inout_init(seq_max);
    memory_ = memory;
    resolve_attention_names();
    kv_reuse_cache_.init(kv_cache_, sliding_window_, prefill_length_);

    if (decode_model_ != nullptr && decode_input_map_.count(decode_model_->GetInputName(2)) > 0) {
        auto current_length = decode_input_map_.at(decode_model_->GetInputName(2));
        set_input_data(current_length, n_decode_batch_, 1);
        decode_model_->SetInput(decode_model_->GetInputName(2), current_length);
    }
}

void HouMoGemma4::resolve_attention_names() {
    prefill_local_attention_names_.clear();
    prefill_global_attention_names_.clear();
    local_attention_names_.clear();
    global_attention_names_.clear();
    prefill_per_layer_input_name_.clear();
    decode_per_layer_input_name_.clear();
    accepted_count_input_name_.clear();

    if (prefill_model_ != nullptr) {
        for (size_t i = 0; i < prefill_model_->GetInputNum(); ++i) {
            std::string name = prefill_model_->GetInputName(i);
            if (is_per_layer_input(name)) {
                prefill_per_layer_input_name_ = name;
            } else if (is_local_mask(name)) {
                prefill_local_attention_names_.push_back(name);
            } else if (is_global_mask(name)) {
                prefill_global_attention_names_.push_back(name);
            }
        }
        if (prefill_global_attention_names_.empty() && prefill_local_attention_names_.empty()) {
            if (prefill_model_->GetInputNum() > 3) {
                std::string name = prefill_model_->GetInputName(3);
                if (!is_per_layer_input(name)) {
                    prefill_global_attention_names_.push_back(name);
                }
            }
            if (prefill_model_->GetInputNum() > 4) {
                std::string name = prefill_model_->GetInputName(4);
                if (!is_per_layer_input(name)) {
                    prefill_local_attention_names_.push_back(name);
                }
            }
        }
    }
    if (decode_model_ == nullptr) {
        visual_bidirectional_attention_ = prefill_per_layer_input_name_.empty();
        return;
    }
    for (size_t i = 0; i < decode_model_->GetInputNum(); ++i) {
        std::string name = decode_model_->GetInputName(i);
        if (is_local_mask(name)) {
            local_attention_names_.push_back(name);
        } else if (is_global_mask(name)) {
            global_attention_names_.push_back(name);
        } else if (is_per_layer_input(name)) {
            decode_per_layer_input_name_ = name;
        } else if (is_accepted_count_input(name)) {
            accepted_count_input_name_ = name;
        }
    }
    if (!accepted_count_input_name_.empty()) {
        LLAMA_LOG_INFO("%s: Decode accepted_count input enabled: %s\n",
                       __func__, accepted_count_input_name_.c_str());
    }
    if (local_attention_names_.empty() && decode_input_map_.count(decode_model_->GetInputName(3)) > 0) {
        std::string name = decode_model_->GetInputName(3);
        if (!is_per_layer_input(name)) {
            local_attention_names_.push_back(name);
        }
    }
    if (global_attention_names_.empty() && decode_input_map_.count(decode_model_->GetInputName(4)) > 0) {
        std::string name = decode_model_->GetInputName(4);
        if (!is_per_layer_input(name) && !is_accepted_count_input(name)) {
            global_attention_names_.push_back(name);
        }
    }
    // gemma4-26B dense text path uses visual bidirectional attention, while
    // gemma4_e E2B/E4B path has per_layer_inputs and keeps Base default
    // visual_bidirectional_attention=False. Detect E-series by PLIB input.
    visual_bidirectional_attention_ = prefill_per_layer_input_name_.empty() &&
                                      decode_per_layer_input_name_.empty();
}

int HouMoGemma4::set_per_layer_inputs_for_prefill(const llama_token *tokens,
                                                  int current_length) {
    if (prefill_per_layer_input_name_.empty()) {
        return 0;
    }
    if (per_layer_embedding_layer_ == nullptr) {
        LLAMA_LOG_ERROR("%s: prefill requires per_layer_inputs but per_layer_embedding.bin is not loaded\n", __func__);
        return -1;
    }
    auto &input = prefill_input_map_.at(prefill_per_layer_input_name_);
    const auto shape = input.Info().Shape();
    if (shape.size() != 4) {
        LLAMA_LOG_ERROR("%s: unsupported prefill per_layer_inputs rank=%zu\n", __func__, shape.size());
        return -1;
    }
    const int seq_len = static_cast<int>(shape[1]);
    const size_t per_token_elems = static_cast<size_t>(shape[2]) * static_cast<size_t>(shape[3]);
    if (per_token_elems != per_layer_embedding_layer_->get_n_embd()) {
        LLAMA_LOG_ERROR("%s: per_layer embedding dim mismatch, input=%zu, embedding=%u\n",
                        __func__, per_token_elems, per_layer_embedding_layer_->get_n_embd());
        return -1;
    }
    if (current_length > seq_len) {
        LLAMA_LOG_ERROR("%s: current_length=%d > seq_len=%d\n", __func__, current_length, seq_len);
        return -1;
    }
    std::memset(input.Data(), 0, input.MemSize());
    if (current_length > 0) {
        per_layer_embedding_layer_->get_embedding_batch(tokens, current_length, input.Data());
    }
    prefill_model_->SetInput(prefill_per_layer_input_name_, input);
    return 0;
}

int HouMoGemma4::set_per_layer_inputs_for_decode(std::vector<llama_token> &batches,
                                                 std::vector<int> &seq_ids) {
    if (decode_per_layer_input_name_.empty()) {
        return 0;
    }
    if (per_layer_embedding_layer_ == nullptr) {
        LLAMA_LOG_ERROR("%s: decode requires per_layer_inputs but per_layer_embedding.bin is not loaded\n", __func__);
        return -1;
    }
    auto &input = decode_input_map_.at(decode_per_layer_input_name_);
    const auto shape = input.Info().Shape();
    if (shape.size() != 4) {
        LLAMA_LOG_ERROR("%s: unsupported decode per_layer_inputs rank=%zu\n", __func__, shape.size());
        return -1;
    }
    const size_t batch = static_cast<size_t>(shape[0]);
    const size_t seq_len = static_cast<size_t>(shape[1]);
    const size_t per_token_elems = static_cast<size_t>(shape[2]) * static_cast<size_t>(shape[3]);
    if (seq_len < 1) {
        LLAMA_LOG_ERROR("%s: invalid decode per_layer_inputs seq_len=%zu\n", __func__, seq_len);
        return -1;
    }
    if (per_token_elems != per_layer_embedding_layer_->get_n_embd()) {
        LLAMA_LOG_ERROR("%s: per_layer embedding dim mismatch, input=%zu, embedding=%u\n",
                        __func__, per_token_elems, per_layer_embedding_layer_->get_n_embd());
        return -1;
    }
    std::memset(input.Data(), 0, input.MemSize());
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        int batch_id = n_decode_batch_ == 1 ? 0 : seq_ids[i];
        if (batch_id < 0 || static_cast<size_t>(batch_id) >= batch) {
            LLAMA_LOG_ERROR("%s: batch_id=%d out of range, batch=%zu\n", __func__, batch_id, batch);
            return -1;
        }
        const size_t offset = static_cast<size_t>(batch_id) * seq_len * per_token_elems;
        per_layer_embedding_layer_->get_embedding_batch(&batches[i], 1,
            static_cast<uint16_t *>(input.Data()) + offset);
    }
    decode_model_->SetInput(decode_per_layer_input_name_, input);
    return 0;
}

bool HouMoGemma4::has_cached_inputs(int seq_id) const {
    auto it = embedding_cache_.find(seq_id);
    return it != embedding_cache_.end() && !it->second.empty();
}

size_t HouMoGemma4::cached_token_count(int seq_id) const {
    auto it = embedding_cache_.find(seq_id);
    if (it == embedding_cache_.end()) {
        return 0;
    }
    return it->second.size() / static_cast<size_t>(n_embd_);
}

bool HouMoGemma4::device_kv_covers_history(int seq_id,
                                           int64_t valid_length) const {
    if (valid_length <= 0) {
        return true;
    }
    const int64_t history_len = std::min<int64_t>(valid_length, sliding_window_);
    const int64_t history_start = valid_length - history_len;
    const int64_t history_end = valid_length;
    auto start_it = device_kv_abs_start_.find(seq_id);
    auto end_it = device_kv_abs_end_.find(seq_id);
    if (start_it == device_kv_abs_start_.end() ||
        end_it == device_kv_abs_end_.end()) {
        return false;
    }
    return start_it->second <= history_start && end_it->second >= history_end;
}

void HouMoGemma4::update_device_kv_range(int seq_id, int64_t abs_end) {
    device_kv_abs_end_[seq_id] = abs_end;
    device_kv_abs_start_[seq_id] = std::max<int64_t>(0, abs_end - sliding_window_);
}

void HouMoGemma4::clear_device_kv_range(int seq_id) {
    device_kv_abs_start_.erase(seq_id);
    device_kv_abs_end_.erase(seq_id);
}

void HouMoGemma4::clear_cached_inputs(int seq_id) {
    embedding_cache_[seq_id].clear();
    chunk_history_[seq_id].clear();
}

std::vector<uint16_t> HouMoGemma4::generate_mask_f16(
    int batch_size, int fill_length, int current_length, int past_seq_length,
    int width, bool is_local, const std::vector<PrefillChunk> *chunks,
    int round_start) const {
    const int mask_size = batch_size * fill_length * width;
    std::vector<uint16_t> mask_tensor(static_cast<size_t>(mask_size), F16_NEG_MIN_UINT);

    const int clamped_past = is_local
                                 ? std::min(past_seq_length, sliding_window_ - 1)
                                 : past_seq_length;
    const int valid_k = std::min(context_length_, std::max(1, past_seq_length + current_length));

    for (int b = 0; b < batch_size; ++b) {
        const int batch_offset = b * fill_length * width;
        for (int q = 0; q < fill_length; ++q) {
            if (q >= current_length) {
                // Python _build_masks() keeps padded query rows valid at key 0.
                // Leaving a full -inf row can make softmax produce bad values.
                if (width > 0) {
                    mask_tensor[static_cast<size_t>(batch_offset + q * width)] = 0;
                }
                continue;
            }
            if (is_local) {
                const int causal_end = std::min(width, clamped_past + q + 1);
                const int sw_start = std::max(0, clamped_past + q - sliding_window_ + 1);
                for (int k = sw_start; k < causal_end; ++k) {
                    mask_tensor[static_cast<size_t>(batch_offset + q * width + k)] = 0;
                }
            } else {
                const int full_end = std::min(valid_k, past_seq_length + q + 1);
                for (int k = 0; k < full_end; ++k) {
                    mask_tensor[static_cast<size_t>(batch_offset + q * width + k)] = 0;
                }
            }
        }
    }

    // Gemma4Processor marks image soft tokens with mm_token_type_ids 1/2.
    if (chunks != nullptr && current_length > 0) {
        int local_pos = 0;
        for (const auto &chunk : *chunks) {
            const int chunk_start = local_pos;
            const int chunk_end = local_pos + static_cast<int>(chunk.size());
            local_pos = chunk_end;
            if (!chunk.is_embedding || !chunk.bidirectional_attention) {
                continue;
            }

            const int group_start = std::max(round_start, chunk_start);
            const int group_end = std::min(round_start + current_length, chunk_end);
            if (group_start >= group_end) {
                continue;
            }

            const int q_start = group_start - round_start;
            const int q_end = group_end - round_start;
            const int abs_start = past_seq_length + q_start;
            const int abs_end = past_seq_length + q_end;
            const int cache_offset = std::max(0, past_seq_length - clamped_past);
            const int k_start = is_local ? std::max(0, abs_start - cache_offset) : abs_start;
            const int k_end = is_local ? std::min(width, abs_end - cache_offset) : std::min(width, abs_end);
            if (k_start >= k_end) {
                continue;
            }
            for (int b = 0; b < batch_size; ++b) {
                const int batch_offset = b * fill_length * width;
                for (int q = q_start; q < q_end; ++q) {
                    for (int k = k_start; k < k_end; ++k) {
                        mask_tensor[static_cast<size_t>(batch_offset + q * width + k)] = 0;
                    }
                }
            }
        }
    }

    return mask_tensor;
}

int HouMoGemma4::set_attentions_for_prefill(int current_length, int valid_length,
                                            int round_start, int seq_id) {
    for (const auto &name : prefill_global_attention_names_) {
        auto &global_mask = prefill_input_map_.at(name);
        const int global_fill = tensor_dim_or(global_mask, 2, prefill_length_);
        const int global_width = tensor_dim_or(global_mask, 3, context_length_);
        auto global_host = generate_mask_f16(1, global_fill, current_length, valid_length,
                                             global_width, false, &chunk_history_[seq_id], round_start);
        global_mask.Buffer().CopyFromHost(global_host.data(), global_mask.MemSize());
        prefill_model_->SetInput(name, global_mask);
    }

    for (const auto &name : prefill_local_attention_names_) {
        auto &local_mask = prefill_input_map_.at(name);
        const int local_fill = tensor_dim_or(local_mask, 2, prefill_length_);
        const int local_width = tensor_dim_or(local_mask, 3, aligned(sliding_window_ + local_fill - 1, 16));
        auto local_host = generate_mask_f16(1, local_fill, current_length, valid_length,
                                            local_width, true, &chunk_history_[seq_id], round_start);
        local_mask.Buffer().CopyFromHost(local_host.data(), local_mask.MemSize());
        prefill_model_->SetInput(name, local_mask);
    }
    return 0;
}

int HouMoGemma4::set_attentions_for_decode(std::vector<int> &seq_ids) {
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        int seq_id = seq_ids[i];
        int batch_id = seq_id;
        if (n_decode_batch_ == 1) {
            batch_id = 0;
        }
        if (batch_id >= static_cast<int>(local_attention_names_.size())) {
            LLAMA_LOG_ERROR("%s: missing sliding/local attention mask input for batch_id=%d\n", __func__, batch_id);
            return -1;
        }

        const int valid_length = decode_valid_length_[seq_id];
        std::string name = local_attention_names_[batch_id];
        auto &local_mask = decode_input_map_.at(name);
        const int local_fill = tensor_dim_or(local_mask, 2, std::max(1, tensor_dim_or(decode_input_map_.at(decode_model_->GetInputName(0)), 1, 1)));
        const int local_width = tensor_dim_or(local_mask, 3, aligned(sliding_window_ + local_fill - 1, 16));
        auto local_host = generate_mask_f16(1, local_fill, 1, valid_length,
                                            local_width, true, nullptr, 0);
        local_mask.Buffer().CopyFromHost(local_host.data(), local_mask.MemSize());
        decode_model_->SetInput(name, local_mask);
    }
    return 0;
}

int HouMoGemma4::houmo_prefill(std::vector<llama_token> &tokens, int seq_id,
                               float *logits) {
    LLAMA_LOG_INFO("%s seq_id = %d, tokens size is = %zu\n", "prefill stage",
                   seq_id, tokens.size());

    if (logits == nullptr && has_cached_inputs(seq_id)) {
        if (!tokens.empty()) {
            auto &cache = embedding_cache_[seq_id];
            const size_t offset = cache.size();
            cache.resize(offset + tokens.size() * static_cast<size_t>(n_embd_));
            embedding_layer_->get_embedding_batch(tokens.data(), tokens.size(),
                                                 cache.data() + offset);
            PrefillChunk chunk;
            chunk.n_tokens = tokens.size();
            chunk_history_[seq_id].push_back(std::move(chunk));
        }
        return 0;
    }

    int32_t kv_offset = memory_ ? memory_->seq_pos_min(seq_id) : 0;
    const size_t cached_tokens = cached_token_count(seq_id);
    std::vector<llama_token> effective_tokens = tokens;
    int32_t effective_kv_offset = kv_offset;
    if (kv_offset == 0) {
        clear_device_kv_range(seq_id);
    }

    if (kv_offset > 0 && kv_cache_.find(seq_id) != kv_cache_.end() &&
        !device_kv_covers_history(seq_id, kv_offset)) {
        auto &kv_item = kv_cache_[seq_id];
        const bool restored = kv_reuse_cache_.prepare_for_prefill(seq_id, kv_offset, kv_item);
        if (restored) {
            update_device_kv_range(seq_id, kv_offset);
            LLAMA_LOG_INFO("%s: restored exact Gemma4 sliding KV seq_id=%d kv_offset=%d\n",
                           __func__, seq_id, kv_offset);
        } else {
            const int64_t rollback_pos =
                kv_reuse_cache_.restore_nearest_before(seq_id, kv_offset, kv_item);
            if (rollback_pos >= 0 && rollback_pos < kv_offset) {
                const auto hist_it = token_history_.find(seq_id);
                const bool can_replay_text =
                    cached_tokens == 0 && hist_it != token_history_.end() &&
                    static_cast<size_t>(kv_offset) <= hist_it->second.size();
                if (can_replay_text) {
                    std::vector<llama_token> replay_tokens;
                    replay_tokens.reserve(static_cast<size_t>(kv_offset - rollback_pos) + tokens.size());
                    replay_tokens.insert(replay_tokens.end(),
                                         hist_it->second.begin() + rollback_pos,
                                         hist_it->second.begin() + kv_offset);
                    replay_tokens.insert(replay_tokens.end(), tokens.begin(), tokens.end());
                    if (memory_) {
                        memory_->seq_rm(seq_id, static_cast<int32_t>(rollback_pos), kv_offset);
                    }
                    const int32_t gap = kv_offset - static_cast<int32_t>(rollback_pos);
                    effective_tokens.swap(replay_tokens);
                    effective_kv_offset = static_cast<int32_t>(rollback_pos);
                    update_device_kv_range(seq_id, rollback_pos);
                    LLAMA_LOG_INFO("%s: rollback seq_id=%d kv_offset=%d rollback_pos=%ld gap=%d effective_tokens=%zu\n",
                                   __func__, seq_id, kv_offset, rollback_pos,
                                   gap, effective_tokens.size());
                } else {
                    LLAMA_LOG_WARN("%s: nearest Gemma4 checkpoint found but cannot replay gap seq_id=%d kv_offset=%d rollback_pos=%ld cached_tokens=%zu history=%zu\n",
                                   __func__, seq_id, kv_offset, rollback_pos,
                                   cached_tokens,
                                   hist_it == token_history_.end() ? 0 : hist_it->second.size());
                }
            } else {
                const auto hist_it = token_history_.find(seq_id);
                const bool can_replay_from_zero =
                    cached_tokens == 0 && hist_it != token_history_.end() &&
                    static_cast<size_t>(kv_offset) <= hist_it->second.size();
                if (can_replay_from_zero) {
                    std::vector<llama_token> replay_tokens;
                    replay_tokens.reserve(static_cast<size_t>(kv_offset) + tokens.size());
                    replay_tokens.insert(replay_tokens.end(),
                                         hist_it->second.begin(),
                                         hist_it->second.begin() + kv_offset);
                    replay_tokens.insert(replay_tokens.end(), tokens.begin(), tokens.end());
                    if (memory_) {
                        memory_->seq_rm(seq_id, 0, kv_offset);
                    }
                    effective_tokens.swap(replay_tokens);
                    effective_kv_offset = 0;
                    clear_device_kv_range(seq_id);
                    LLAMA_LOG_INFO("%s: no checkpoint, rollback seq_id=%d kv_offset=%d rollback_pos=0 gap=%d effective_tokens=%zu\n",
                                   __func__, seq_id, kv_offset, kv_offset,
                                   effective_tokens.size());
                } else {
                    LLAMA_LOG_WARN("%s: no Gemma4 sliding KV checkpoint covers seq_id=%d kv_offset=%d and cannot replay from zero cached_tokens=%zu history=%zu; continue without rollback\n",
                                   __func__, seq_id, kv_offset, cached_tokens,
                                   hist_it == token_history_.end() ? 0 : hist_it->second.size());
                }
            }
        }
    }

    const size_t total_tokens = cached_tokens + effective_tokens.size();
    if ((total_tokens + effective_kv_offset) >= static_cast<size_t>(context_length_)) {
        LLAMA_LOG_ERROR("%s: input size %zu + kv_offset %d >= context_length_%d\n",
                        __func__, total_tokens, effective_kv_offset, context_length_);
        clear_cached_inputs(seq_id);
        return 1;
    }
    if (total_tokens == 0) {
        return 0;
    }

    if (!effective_tokens.empty() && cached_tokens == 0) {
        PrefillChunk chunk;
        chunk.n_tokens = effective_tokens.size();
        chunk_history_[seq_id].push_back(std::move(chunk));
    } else if (!tokens.empty()) {
        PrefillChunk chunk;
        chunk.n_tokens = tokens.size();
        chunk_history_[seq_id].push_back(std::move(chunk));
    }

    size_t rounds = (total_tokens + prefill_length_ - 1) / prefill_length_;
    int32_t valid_length = effective_kv_offset;
    int32_t current_length = 0;
    auto &input_data = prefill_input_map_.at(prefill_model_->GetInputName(0));

    for (size_t round = 0; round < rounds; ++round) {
        const size_t round_start = round * static_cast<size_t>(prefill_length_);
        const size_t remaining = total_tokens > round_start ? total_tokens - round_start : 0;
        current_length = remaining >= static_cast<size_t>(prefill_length_)
                             ? prefill_length_
                             : static_cast<int32_t>(remaining);
        LLAMA_LOG_INFO("%s: prefill round seq_id=%d round=%zu/%zu round_start=%zu valid_length=%d current_length=%d total_tokens=%zu cached_tokens=%zu kv_offset=%d\n",
                   __func__, seq_id, round + 1, rounds, round_start,
                   valid_length, current_length, total_tokens,
                   cached_tokens, effective_kv_offset);

        std::memset(input_data.Data(), 0, input_data.MemSize());
        if (round_start + static_cast<size_t>(current_length) <= cached_tokens) {
            const size_t emb_start = round_start * static_cast<size_t>(n_embd_);
            std::memcpy(input_data.Data(), embedding_cache_[seq_id].data() + emb_start,
                        current_length * static_cast<size_t>(n_embd_) * sizeof(uint16_t));
        } else if (round_start >= cached_tokens) {
            const size_t token_offset = round_start - cached_tokens;
            embedding_layer_->get_embedding_batch(effective_tokens.data() + token_offset,
                                                 current_length, input_data.Data());
        } else {
            const size_t emb_part = cached_tokens - round_start;
            const size_t token_part = static_cast<size_t>(current_length) - emb_part;
            const size_t emb_start = round_start * static_cast<size_t>(n_embd_);
            std::memcpy(input_data.Data(), embedding_cache_[seq_id].data() + emb_start,
                        emb_part * static_cast<size_t>(n_embd_) * sizeof(uint16_t));
            embedding_layer_->get_embedding_batch(effective_tokens.data(), token_part,
                                                 static_cast<uint16_t *>(input_data.Data()) +
                                                     emb_part * static_cast<size_t>(n_embd_));
        }

        auto valid_tensor = prefill_input_map_.at(prefill_model_->GetInputName(1));
        auto current_tensor = prefill_input_map_.at(prefill_model_->GetInputName(2));
        set_input_data(valid_tensor, n_prefill_batch_, valid_length);
        set_input_data(current_tensor, n_prefill_batch_, current_length);

        PERF_START(PREFILL_SETINPUT_ONCE);
        prefill_model_->SetInput(prefill_model_->GetInputName(0), input_data);
        prefill_model_->SetInput(prefill_model_->GetInputName(1), valid_tensor);
        prefill_model_->SetInput(prefill_model_->GetInputName(2), current_tensor);
        const llama_token *pli_tokens = nullptr;
        std::vector<llama_token> zero_tokens;
        if (round_start >= cached_tokens) {
            const size_t token_offset = round_start - cached_tokens;
            pli_tokens = effective_tokens.data() + token_offset;
        } else if (round_start + static_cast<size_t>(current_length) <= cached_tokens) {
            zero_tokens.assign(static_cast<size_t>(current_length), 0);
            pli_tokens = zero_tokens.data();
        } else {
            const size_t emb_part = cached_tokens - round_start;
            const size_t token_part = static_cast<size_t>(current_length) - emb_part;
            zero_tokens.assign(static_cast<size_t>(current_length), 0);
            std::copy(effective_tokens.begin(), effective_tokens.begin() + token_part,
                      zero_tokens.begin() + static_cast<std::ptrdiff_t>(emb_part));
            pli_tokens = zero_tokens.data();
        }
        if (set_per_layer_inputs_for_prefill(pli_tokens, current_length) != 0) {
            PERF_STOP(PREFILL_SETINPUT_ONCE);
            clear_cached_inputs(seq_id);
            return -1;
        }
        set_attentions_for_prefill(current_length, valid_length, static_cast<int>(round_start), seq_id);
        if (kv_cache_.find(seq_id) != kv_cache_.end()) {
            auto &kv_item = kv_cache_[seq_id];
            for (size_t idx = 0; idx < kv_item.tensors.size(); ++idx) {
                prefill_model_->SetInput(kv_item.names[idx], kv_item.tensors[idx]);
            }
        }
        PERF_STOP(PREFILL_SETINPUT_ONCE);

        PERF_START(PREFILL_RUN_ONCE);
        auto status = prefill_model_->Run();
        if (status != tcim::Status::OK) {
            PERF_STOP(PREFILL_RUN_ONCE);
            LLAMA_LOG_ERROR("%s: prefill_model_->Run() failed, result = %d\n",
                            __func__, static_cast<int>(status));
            clear_cached_inputs(seq_id);
            return -1;
        }
        prefill_model_->Sync();
        PERF_STOP(PREFILL_RUN_ONCE);

        if (pre_norm_hidden_dst_ != nullptr && pre_norm_hidden_dst_rows_ > 0) {
            for (const auto &name : prefill_output_names_) {
                if (name.find("hidden") == std::string::npos) {
                    continue;
                }
                auto hidden = prefill_model_->GetOutput(name);
                const size_t row_bytes = static_cast<size_t>(n_embd_) * sizeof(ggml_fp16_t);
                const size_t hidden_rows = hidden.MemSize() / row_bytes;
                const int dst_row = static_cast<int>(round_start) + current_length - 1;
                if (hidden_rows == 0 || dst_row < 0 || dst_row >= pre_norm_hidden_dst_rows_) {
                    break;
                }
                const auto *src = static_cast<const uint8_t *>(hidden.Buffer().Data()) +
                                  (hidden_rows - 1) * row_bytes;
                auto *dst = reinterpret_cast<uint8_t *>(pre_norm_hidden_dst_) +
                            static_cast<size_t>(dst_row) * n_embd_ * sizeof(float);
                std::memcpy(dst, src, row_bytes);
                LLAMA_LOG_INFO("%s: exported prefill hidden name='%s' rows=%zu dst_row=%d row_bytes=%zu\n",
                               __func__, name.c_str(), hidden_rows, dst_row, row_bytes);
                break;
            }
        }

        valid_length += current_length;

        if (kv_cache_.find(seq_id) != kv_cache_.end()) {
            const int32_t abs_end = valid_length;
            update_device_kv_range(seq_id, abs_end);
            const bool is_final_round =
                round_start + static_cast<size_t>(current_length) >= total_tokens;
            const bool is_prompt_final_prefill = is_final_round && logits != nullptr;
            const bool force_save = is_prompt_final_prefill && valid_length > sliding_window_;
            LLAMA_LOG_INFO("%s: KV save decision seq_id=%d round=%zu abs_end=%d valid_length=%d current_length=%d is_final=%d logits=%d prompt_final=%d force=%d\n",
                           __func__, seq_id, round + 1, abs_end, valid_length,
                           current_length, is_final_round ? 1 : 0,
                           logits != nullptr ? 1 : 0,
                           is_prompt_final_prefill ? 1 : 0,
                           force_save ? 1 : 0);
            kv_reuse_cache_.save_after_prefill(seq_id, abs_end, kv_cache_[seq_id], force_save);
        }
        if (abort_callback_ != nullptr && abort_callback_(abort_callback_data_)) {
            LLAMA_LOG_INFO("%s: abort_callback_ is called. seq_id = %d\n", __func__, seq_id);
            clear_cached_inputs(seq_id);
            return 2;
        }
    }

    decode_valid_length_[seq_id] = effective_kv_offset + static_cast<int32_t>(total_tokens);
    if (logits != nullptr) {
        PERF_START(PREFILL_GETOUTPUT);
        tcim::Tensor prefill_output_data = prefill_model_->GetOutput(prefill_output_names_[0]);
        const auto out_info = prefill_output_data.Info();
        PERF_STOP(PREFILL_GETOUTPUT);
        if (out_info.DataType() != tcim::DataType::FLOAT16) {
            LLAMA_LOG_ERROR("%s: unsupported prefill output dtype=%d\n", __func__,
                            static_cast<int>(out_info.DataType()));
            clear_cached_inputs(seq_id);
            return -1;
        }
        const size_t row_elems = static_cast<size_t>(n_vocab_);
        const size_t n_rows = prefill_output_data.MemSize() /
                              sizeof(ggml_fp16_t) / row_elems;
        if (n_rows == 0) {
            LLAMA_LOG_ERROR("%s: empty prefill logits output\n", __func__);
            clear_cached_inputs(seq_id);
            return -1;
        }
        const size_t row_idx = std::min<size_t>(
            n_rows - 1,
            current_length > 0 ? static_cast<size_t>(current_length - 1) : 0);
        auto *src = static_cast<const ggml_fp16_t *>(prefill_output_data.Buffer().Data()) +
                    row_idx * row_elems;
        ggml_fp16_to_fp32_row(src, logits, static_cast<int64_t>(row_elems));
    }

    if (memory_) {
        memory_->seq_add(seq_id, 0, 0, static_cast<int32_t>(total_tokens));
    }
    if (cached_tokens == 0) {
        auto &history = token_history_[seq_id];
        if (history.size() > static_cast<size_t>(effective_kv_offset)) {
            history.resize(static_cast<size_t>(effective_kv_offset));
        }
        history.insert(history.end(), effective_tokens.begin(), effective_tokens.end());
    }
    clear_cached_inputs(seq_id);
    return 0;
}

int HouMoGemma4::houmo_prefill(const float *embeddings, int n_tokens,
                               int seq_id, float *logits) {
    (void)logits;
    if (n_tokens <= 0) {
        return 0;
    }
    // HoumoVision 在 embedding buffer 前 8 字节放置 CompactHeader{magic, nx, ny, chunksize}，
    // 真正的 fp16 vision 数据从 sizeof(CompactHeader) 之后才开始。这里需要跳过 header，
    // 否则会让整段视觉嵌入向后偏移并破坏前若干个 fp16，导致 prefill/decode 结果乱码。
    struct CompactHeader { int16_t magic; int16_t nx; int16_t ny; int16_t chunksize; };
    const int8_t *ptr = reinterpret_cast<const int8_t *>(embeddings);
    size_t embedding_offset = 0;
    CompactHeader header{};
    std::memcpy(&header, ptr, sizeof(CompactHeader));
    const bool has_compact_header = header.magic == 0x1234;
    if (has_compact_header) {
        embedding_offset = sizeof(CompactHeader);
    }

    auto &cache = embedding_cache_[seq_id];
    const size_t old_size = cache.size();
    const size_t data_elems = static_cast<size_t>(n_tokens) * static_cast<size_t>(n_embd_);
    cache.resize(old_size + data_elems);
    // HMM multimodal encoders provide fp16 embeddings. Vision may prepend a
    // CompactHeader, while audio does not; both payloads are copied as fp16.
    std::memcpy(cache.data() + old_size, ptr + embedding_offset,
                data_elems * sizeof(uint16_t));
    PrefillChunk chunk;
    chunk.is_embedding = true;
    chunk.bidirectional_attention = has_compact_header && visual_bidirectional_attention_;
    chunk.n_tokens = static_cast<size_t>(n_tokens);
    chunk_history_[seq_id].push_back(std::move(chunk));
    LLAMA_LOG_INFO("%s: cached external embeddings seq_id=%d n_tokens=%d header_magic=0x%04x bidir=%d\n",
                   __func__, seq_id, n_tokens, header.magic,
                   chunk_history_[seq_id].back().bidirectional_attention ? 1 : 0);
    return 0;
}

int HouMoGemma4::houmo_decode(std::vector<llama_token> &batches,
                              std::vector<int> &seq_ids,
                              std::vector<float *> logits) {
    for (auto seq_id : seq_ids) {
        decode_valid_length_[seq_id] = memory_ ? memory_->seq_pos_min(seq_id) : decode_valid_length_[seq_id];
    }
    if (seq_ids.empty()) {
        return 0;
    }
    if (seq_ids.size() > static_cast<size_t>(n_decode_batch_)) {
        LLAMA_LOG_ERROR("%s: seq_ids.size() > n_decode_batch_ not supported\n", __func__);
        return -1;
    }

    PERF_START(DECODE_SETINPUT);
    int ret = set_input_for_decode(batches, seq_ids);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: set_input_for_decode failed\n", __func__);
        return ret;
    }
    ret = set_xlenght_for_decode(seq_ids);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: set_xlenght_for_decode failed\n", __func__);
        return ret;
    }
    ret = set_attentions_for_decode(seq_ids);
    if (ret != 0) {
        return ret;
    }
    ret = set_per_layer_inputs_for_decode(batches, seq_ids);
    if (ret != 0) {
        return ret;
    }
    set_kv_cache_for_decode(seq_ids);
    PERF_STOP(DECODE_SETINPUT);

    PERF_START(DECODE_RUN);
    decode_model_->Run();
    decode_model_->Sync();
    PERF_STOP(DECODE_RUN);

    PERF_START(DECODE_GETOUTPUT);
    tcim::Tensor decode_data = decode_model_->GetOutput(decode_output_names_[0]);
    const auto out_dtype = decode_data.Info().DataType();
    void *decode_ptr = decode_data.Buffer().Data();
    PERF_STOP(DECODE_GETOUTPUT);

    for (size_t i = 0; i < seq_ids.size(); ++i) {
        const int seq_id = seq_ids[i];
        decode_valid_length_[seq_id]++;
        int batch_id = seq_id;
        if (n_decode_batch_ == 1) {
            batch_id = 0;
        }
        if (logits[i] != nullptr) {
            if (out_dtype == tcim::DataType::FLOAT16) {
                auto *output_ptr = static_cast<ggml_fp16_t *>(decode_ptr) +
                                   static_cast<size_t>(batch_id) * n_vocab_;
                ggml_fp16_to_fp32_row(output_ptr, logits[i],
                                      static_cast<int>(n_vocab_));
            } else {
                LLAMA_LOG_ERROR("%s: unsupported decode output dtype=%d\n",
                                __func__, static_cast<int>(out_dtype));
                return -1;
            }
        }
        if (memory_) {
            memory_->seq_add(seq_id, 0, 0, 1);
        }
        update_device_kv_range(seq_id, decode_valid_length_[seq_id]);
    }
    return 0;
}

int HouMoGemma4::houmo_verify(std::vector<llama_token> &tokens, int seq_id,
                              std::vector<float *> &logits_out) {
    if (tokens.size() != logits_out.size()) {
        LLAMA_LOG_ERROR("%s: tokens.size()=%zu != logits_out.size()=%zu\n",
                        __func__, tokens.size(), logits_out.size());
        return -1;
    }
    if (tokens.empty()) {
        return 0;
    }
    if (decode_model_ == nullptr) {
        return HouMoLLModel::Impl::houmo_verify(tokens, seq_id, logits_out);
    }

    auto &input_data = decode_input_map_.at(decode_model_->GetInputName(0));
    const int verify_slots = tensor_dim_or(input_data, 1, 1);
    const int n_verify = static_cast<int>(tokens.size());
    if (verify_slots <= 1 || n_verify > verify_slots) {
        LLAMA_LOG_WARN("%s: verify_slots=%d n_verify=%d, fallback to prefill loop\n",
                       __func__, verify_slots, n_verify);
        return HouMoLLModel::Impl::houmo_verify(tokens, seq_id, logits_out);
    }

    const int32_t start_pos = memory_ ? memory_->seq_pos_min(seq_id)
                                      : decode_valid_length_[seq_id];

    // {
    //     std::ostringstream oss;
    //     for (size_t i = 0; i < tokens.size(); ++i) {
    //         if (i > 0) {
    //             oss << ',';
    //         }
    //         oss << static_cast<int>(tokens[i]);
    //     }
    //     LLAMA_LOG_INFO("%s: verify input seq_id=%d start_pos=%d n_verify=%d tokens=[%s]\n",
    //                    __func__, seq_id, start_pos, n_verify, oss.str().c_str());
    // }

    PERF_START(VERIFY_SETINPUT);
    std::memset(input_data.Data(), 0, input_data.MemSize());
    embedding_layer_->get_embedding_batch(tokens.data(), n_verify, input_data.Data());
    decode_model_->SetInput(decode_model_->GetInputName(0), input_data);

    auto valid_tensor = decode_input_map_.at(decode_model_->GetInputName(1));
    auto current_tensor = decode_input_map_.at(decode_model_->GetInputName(2));
    set_input_data(valid_tensor, n_decode_batch_, start_pos);
    set_input_data(current_tensor, n_decode_batch_, n_verify);
    decode_model_->SetInput(decode_model_->GetInputName(1), valid_tensor);
    decode_model_->SetInput(decode_model_->GetInputName(2), current_tensor);

    auto set_verify_masks = [&](const std::vector<std::string> &names,
                                bool local) -> int {
        for (const auto &name : names) {
            if (is_accepted_count_input(name)) {
                continue;
            }
            auto it = decode_input_map_.find(name);
            if (it == decode_input_map_.end()) {
                continue;
            }
            auto &mask = it->second;
            const int fill = tensor_dim_or(mask, 2, verify_slots);
            const int width = tensor_dim_or(mask, 3,
                local ? aligned(sliding_window_ + fill - 1, 16) : context_length_);
            auto host = generate_mask_f16(1, fill, n_verify, start_pos,
                                          width, local, nullptr, 0);
            mask.Buffer().CopyFromHost(host.data(), mask.MemSize());
            decode_model_->SetInput(name, mask);
        }
        return 0;
    };
    set_verify_masks(local_attention_names_, true);
    set_verify_masks(global_attention_names_, false);

    if (!decode_per_layer_input_name_.empty()) {
        if (per_layer_embedding_layer_ == nullptr) {
            LLAMA_LOG_ERROR("%s: verify requires per_layer_inputs but embedding is not loaded\n",
                            __func__);
            return -1;
        }
        auto &pli = decode_input_map_.at(decode_per_layer_input_name_);
        const auto shape = pli.Info().Shape();
        if (shape.size() != 4) {
            LLAMA_LOG_ERROR("%s: unsupported verify per_layer_inputs rank=%zu\n",
                            __func__, shape.size());
            return -1;
        }
        const size_t per_token_elems = static_cast<size_t>(shape[2]) * static_cast<size_t>(shape[3]);
        if (per_token_elems != per_layer_embedding_layer_->get_n_embd()) {
            LLAMA_LOG_ERROR("%s: per_layer embedding dim mismatch, input=%zu, embedding=%u\n",
                            __func__, per_token_elems,
                            per_layer_embedding_layer_->get_n_embd());
            return -1;
        }
        std::memset(pli.Data(), 0, pli.MemSize());
        per_layer_embedding_layer_->get_embedding_batch(tokens.data(), n_verify,
                                                        pli.Data());
        decode_model_->SetInput(decode_per_layer_input_name_, pli);
    }
    if (!accepted_count_input_name_.empty()) {
        auto accepted_tensor = decode_input_map_.at(accepted_count_input_name_);
        const int32_t accepted_count = last_committed_tokens_[seq_id];
        set_input_data(accepted_tensor, n_decode_batch_, accepted_count);
        decode_model_->SetInput(accepted_count_input_name_, accepted_tensor);
        LLAMA_LOG_INFO("%s: verify accepted_count seq_id=%d value=%d\n",
                       __func__, seq_id, accepted_count);
    }
    std::vector<int> seq_ids = {seq_id};
    set_kv_cache_for_decode(seq_ids);
    PERF_STOP(VERIFY_SETINPUT);

    PERF_START(VERIFY_RUN);
    auto status = decode_model_->Run();
    if (status != tcim::Status::OK) {
        PERF_STOP(VERIFY_RUN);
        LLAMA_LOG_ERROR("%s: decode verify Run() failed, result=%d\n",
                        __func__, static_cast<int>(status));
        return -1;
    }
    decode_model_->Sync();
    PERF_STOP(VERIFY_RUN);

    PERF_START(VERIFY_GETOUTPUT);
    tcim::Tensor decode_data = decode_model_->GetOutput(decode_output_names_[0]);
    if (decode_data.Info().DataType() != tcim::DataType::FLOAT16) {
        PERF_STOP(VERIFY_GETOUTPUT);
        LLAMA_LOG_ERROR("%s: unsupported verify output dtype=%d\n", __func__,
                        static_cast<int>(decode_data.Info().DataType()));
        return -1;
    }
    auto *decode_ptr = static_cast<ggml_fp16_t *>(decode_data.Buffer().Data());
    const size_t vocab_row = static_cast<size_t>(tensor_dim_or(decode_data, decode_data.Info().Shape().size() - 1, n_vocab_));
    const size_t vocab_rows = std::max<size_t>(1, decode_data.MemSize() / (vocab_row * sizeof(ggml_fp16_t)));
    const size_t vocab_base_row = vocab_rows > static_cast<size_t>(n_verify)
                                      ? vocab_rows - static_cast<size_t>(n_verify)
                                      : 0;
    for (int i = 0; i < n_verify; ++i) {
        if (logits_out[static_cast<size_t>(i)] != nullptr) {
            const size_t row_idx = std::min(vocab_base_row + static_cast<size_t>(i),
                                            vocab_rows - 1);
            ggml_fp16_to_fp32_row(decode_ptr + row_idx * vocab_row,
                                  logits_out[static_cast<size_t>(i)],
                                  static_cast<int>(vocab_row));
            LLAMA_LOG_INFO("%s: verify output seq_id=%d row=%d pos=%d input_token=%d top1=%d row_idx=%zu vocab=%zu\n",
                           __func__, seq_id, i, start_pos + i,
                           static_cast<int>(tokens[static_cast<size_t>(i)]),
                           top_token_id(logits_out[static_cast<size_t>(i)], vocab_row),
                           row_idx, vocab_row);
        }
    }

    if (pre_norm_hidden_dst_ != nullptr && pre_norm_hidden_dst_rows_ > 0) {
        for (const auto &name : decode_output_names_) {
            if (name.find("hidden") == std::string::npos) {
                continue;
            }
            auto hidden = decode_model_->GetOutput(name);
            if (hidden.Info().DataType() != tcim::DataType::FLOAT16) {
                continue;
            }
            const auto h_shape = hidden.Info().Shape();
            const int h_seq = h_shape.size() >= 2
                                  ? static_cast<int>(h_shape[h_shape.size() - 2])
                                  : 1;
            const int h_embd = h_shape.empty()
                                   ? static_cast<int>(n_embd_)
                                   : static_cast<int>(h_shape.back());
            const int rows = std::min({n_verify, h_seq, pre_norm_hidden_dst_rows_});
            const int src_begin = h_seq > rows ? h_seq - rows : 0;
            const auto *src = static_cast<const uint8_t *>(hidden.Buffer().Data());
            for (int r = 0; r < rows; ++r) {
                std::memcpy(reinterpret_cast<uint8_t *>(pre_norm_hidden_dst_ +
                            static_cast<size_t>(r) * h_embd),
                            src + static_cast<size_t>(src_begin + r) * h_embd * sizeof(ggml_fp16_t),
                            static_cast<size_t>(h_embd) * sizeof(ggml_fp16_t));
            }
            break;
        }
    }
    PERF_STOP(VERIFY_GETOUTPUT);

    pending_verify_[seq_id] = {true, start_pos, n_verify};
    return 0;
}

int HouMoGemma4::houmo_verify_accept(int seq_id, uint16_t n_accepted) {
    auto it = pending_verify_.find(seq_id);
    if (it == pending_verify_.end() || !it->second.valid) {
        return 0;
    }
    const int accepted = std::min<int>(static_cast<int>(n_accepted) + 1,
                                       it->second.n_verify);
    last_committed_tokens_[seq_id] = accepted;
    const int32_t keep_pos = it->second.start_pos + accepted;
    decode_valid_length_[seq_id] = keep_pos;
    update_device_kv_range(seq_id, keep_pos);
    if (memory_) {
        const int32_t cur = memory_->seq_pos_min(seq_id);
        if (cur < keep_pos) {
            memory_->seq_add(seq_id, 0, 0, keep_pos - cur);
        } else if (cur > keep_pos) {
            memory_->seq_rm(seq_id, keep_pos, cur);
        }
    }
    it->second.valid = false;
    return 0;
}
