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
    if (!HouMoLLModel::Impl::houmo_load(ml, device_ids)) {
        return false;
    }
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
    auto it = chunk_history_.find(seq_id);
    return it != chunk_history_.end() && !it->second.empty();
}

size_t HouMoGemma4::cached_token_count(int seq_id) const {
    auto it = chunk_history_.find(seq_id);
    if (it == chunk_history_.end()) {
        return 0;
    }
    size_t count = 0;
    for (const auto &chunk : it->second) {
        count += chunk.size();
    }
    return count;
}

size_t HouMoGemma4::history_token_count(int seq_id) const {
    auto it = input_history_.find(seq_id);
    if (it == input_history_.end()) {
        return 0;
    }
    size_t count = 0;
    for (const auto &chunk : it->second) {
        count += chunk.size();
    }
    return count;
}

bool HouMoGemma4::slice_history(int seq_id, size_t begin, size_t end,
                                std::vector<PrefillChunk> &out) const {
    out.clear();
    if (begin > end || end > history_token_count(seq_id)) {
        return false;
    }
    if (begin == end) {
        return true;
    }
    const auto it = input_history_.find(seq_id);
    if (it == input_history_.end()) {
        return false;
    }
    size_t position = 0;
    for (const auto &chunk : it->second) {
        const size_t chunk_begin = position;
        const size_t chunk_end = position + chunk.size();
        position = chunk_end;
        const size_t copy_begin = std::max(begin, chunk_begin);
        const size_t copy_end = std::min(end, chunk_end);
        if (copy_begin >= copy_end) {
            continue;
        }
        const size_t offset = copy_begin - chunk_begin;
        const size_t count = copy_end - copy_begin;
        PrefillChunk part;
        part.is_embedding = chunk.is_embedding;
        part.bidirectional_attention = chunk.bidirectional_attention;
        part.n_tokens = count;
        part.nx = chunk.nx;
        part.ny = chunk.ny;
        if (chunk.is_embedding) {
            const size_t elem_begin = offset * static_cast<size_t>(n_embd_);
            const size_t elem_end = (offset + count) * static_cast<size_t>(n_embd_);
            if (elem_end > chunk.embeddings.size()) {
                return false;
            }
            part.embeddings.assign(chunk.embeddings.begin() + elem_begin,
                                   chunk.embeddings.begin() + elem_end);
        } else {
            if (offset + count > chunk.tokens.size()) {
                return false;
            }
            part.tokens.assign(chunk.tokens.begin() + offset,
                               chunk.tokens.begin() + offset + count);
        }
        out.push_back(std::move(part));
    }
    return position >= end;
}

void HouMoGemma4::truncate_history(int seq_id, size_t keep_tokens) {
    auto &history = input_history_[seq_id];
    size_t position = 0;
    size_t keep_chunks = 0;
    for (; keep_chunks < history.size(); ++keep_chunks) {
        auto &chunk = history[keep_chunks];
        if (position + chunk.size() <= keep_tokens) {
            position += chunk.size();
            continue;
        }
        const size_t keep = keep_tokens > position ? keep_tokens - position : 0;
        chunk.n_tokens = keep;
        if (chunk.is_embedding) {
            chunk.embeddings.resize(keep * static_cast<size_t>(n_embd_));
        } else {
            chunk.tokens.resize(keep);
        }
        if (keep > 0) {
            ++keep_chunks;
        }
        break;
    }
    history.resize(keep_chunks);
}

void HouMoGemma4::append_history(int seq_id,
                                 const std::vector<PrefillChunk> &chunks) {
    auto &history = input_history_[seq_id];
    history.insert(history.end(), chunks.begin(), chunks.end());
}

void HouMoGemma4::append_history_token(int seq_id, size_t position,
                                       llama_token token) {
    truncate_history(seq_id, position);
    PrefillChunk chunk;
    chunk.n_tokens = 1;
    chunk.tokens.push_back(token);
    input_history_[seq_id].push_back(std::move(chunk));
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
            PrefillChunk chunk;
            chunk.n_tokens = tokens.size();
            chunk.tokens = tokens;
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
        input_history_[seq_id].clear();
        kv_reuse_cache_.clear_seq(seq_id);
        last_committed_tokens_[seq_id] = 0;
        pending_verify_.erase(seq_id);
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
                kv_reuse_cache_.find_nearest_before(seq_id, kv_offset);
            bool rollback_committed = false;
            if (rollback_pos >= 0 && rollback_pos < kv_offset) {
                std::vector<PrefillChunk> replay_chunks;
                if (slice_history(seq_id, static_cast<size_t>(rollback_pos),
                                  static_cast<size_t>(kv_offset), replay_chunks) &&
                    kv_reuse_cache_.restore_at(seq_id, rollback_pos, kv_item)) {
                    auto &pending = chunk_history_[seq_id];
                    replay_chunks.insert(replay_chunks.end(), pending.begin(), pending.end());
                    pending.swap(replay_chunks);
                    if (memory_) {
                        memory_->seq_rm(seq_id, static_cast<int32_t>(rollback_pos), kv_offset);
                    }
                    const int32_t gap = kv_offset - static_cast<int32_t>(rollback_pos);
                    effective_kv_offset = static_cast<int32_t>(rollback_pos);
                    update_device_kv_range(seq_id, rollback_pos);
                    kv_reuse_cache_.release_after(seq_id, rollback_pos);
                    rollback_committed = true;
                    LLAMA_LOG_INFO("%s: multimodal rollback seq_id=%d kv_offset=%d rollback_pos=%ld gap=%d replay_tokens=%zu\n",
                                   __func__, seq_id, kv_offset, rollback_pos,
                                   gap, cached_token_count(seq_id));
                } else {
                    LLAMA_LOG_WARN("%s: nearest Gemma4 checkpoint rollback unavailable; try replay from zero seq_id=%d kv_offset=%d rollback_pos=%ld pending=%zu history=%zu\n",
                                   __func__, seq_id, kv_offset, rollback_pos,
                                   cached_tokens,
                                   history_token_count(seq_id));
                }
            }
            if (!rollback_committed) {
                std::vector<PrefillChunk> replay_chunks;
                if (slice_history(seq_id, 0, static_cast<size_t>(kv_offset),
                                  replay_chunks)) {
                    auto &pending = chunk_history_[seq_id];
                    replay_chunks.insert(replay_chunks.end(), pending.begin(), pending.end());
                    pending.swap(replay_chunks);
                    if (memory_) {
                        memory_->seq_rm(seq_id, 0, kv_offset);
                    }
                    effective_kv_offset = 0;
                    clear_device_kv_range(seq_id);
                    kv_reuse_cache_.release_after(seq_id, 0);
                    LLAMA_LOG_INFO("%s: no checkpoint, multimodal rollback seq_id=%d kv_offset=%d rollback_pos=0 gap=%d replay_tokens=%zu\n",
                                   __func__, seq_id, kv_offset, kv_offset,
                                   cached_token_count(seq_id));
                } else {
                    LLAMA_LOG_WARN("%s: no Gemma4 sliding KV checkpoint covers seq_id=%d kv_offset=%d and multimodal history cannot replay from zero pending=%zu history=%zu; continue without rollback\n",
                                   __func__, seq_id, kv_offset, cached_tokens,
                                   history_token_count(seq_id));
                }
            }
        }
    }

    if (!effective_tokens.empty()) {
        PrefillChunk chunk;
        chunk.n_tokens = effective_tokens.size();
        chunk.tokens = effective_tokens;
        chunk_history_[seq_id].push_back(std::move(chunk));
    }
    const size_t total_tokens = cached_token_count(seq_id);
    if ((total_tokens + effective_kv_offset) >= static_cast<size_t>(context_length_)) {
        LLAMA_LOG_ERROR("%s: input size %zu + kv_offset %d >= context_length_%d\n",
                        __func__, total_tokens, effective_kv_offset, context_length_);
        clear_cached_inputs(seq_id);
        return 1;
    }
    if (total_tokens == 0) {
        return 0;
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
        std::vector<llama_token> pli_round(static_cast<size_t>(current_length), 0);
        size_t stream_pos = 0;
        size_t copied = 0;
        for (const auto &chunk : chunk_history_[seq_id]) {
            const size_t chunk_start = stream_pos;
            const size_t chunk_end = stream_pos + chunk.size();
            stream_pos = chunk_end;
            const size_t copy_start = std::max(round_start, chunk_start);
            const size_t copy_end = std::min(round_start + static_cast<size_t>(current_length), chunk_end);
            if (copy_start >= copy_end) {
                continue;
            }
            const size_t chunk_offset = copy_start - chunk_start;
            const size_t round_offset = copy_start - round_start;
            const size_t count = copy_end - copy_start;
            auto *dst = static_cast<uint16_t *>(input_data.Data()) +
                        round_offset * static_cast<size_t>(n_embd_);
            if (chunk.is_embedding) {
                const size_t elem_offset = chunk_offset * static_cast<size_t>(n_embd_);
                const size_t elem_count = count * static_cast<size_t>(n_embd_);
                if (elem_offset + elem_count > chunk.embeddings.size()) {
                    LLAMA_LOG_ERROR("%s: embedding history is incomplete seq_id=%d offset=%zu count=%zu size=%zu\n",
                                    __func__, seq_id, elem_offset, elem_count,
                                    chunk.embeddings.size());
                    clear_cached_inputs(seq_id);
                    return -1;
                }
                std::memcpy(dst, chunk.embeddings.data() + elem_offset,
                            elem_count * sizeof(uint16_t));
            } else {
                if (chunk_offset + count > chunk.tokens.size()) {
                    LLAMA_LOG_ERROR("%s: token history is incomplete seq_id=%d offset=%zu count=%zu size=%zu\n",
                                    __func__, seq_id, chunk_offset, count,
                                    chunk.tokens.size());
                    clear_cached_inputs(seq_id);
                    return -1;
                }
                if (embed_npu_ && embedding_layer_->is_npu_ready()) {
                    embedding_layer_->get_embedding_batch_npu(
                        chunk.tokens.data() + chunk_offset, count,
                        dst, 0);
                } else {
                    embedding_layer_->get_embedding_batch(chunk.tokens.data() + chunk_offset,
                                                          count, dst);
                }
                std::copy(chunk.tokens.begin() + chunk_offset,
                          chunk.tokens.begin() + chunk_offset + count,
                          pli_round.begin() + static_cast<std::ptrdiff_t>(round_offset));
            }
            copied += count;
        }
        if (copied != static_cast<size_t>(current_length)) {
            LLAMA_LOG_ERROR("%s: input history copied=%zu expected=%d seq_id=%d\n",
                            __func__, copied, current_length, seq_id);
            clear_cached_inputs(seq_id);
            return -1;
        }

        auto valid_tensor = prefill_input_map_.at(prefill_model_->GetInputName(1));
        auto current_tensor = prefill_input_map_.at(prefill_model_->GetInputName(2));
        set_input_data(valid_tensor, n_prefill_batch_, valid_length);
        set_input_data(current_tensor, n_prefill_batch_, current_length);

        PERF_START(PREFILL_SETINPUT_ONCE);
        prefill_model_->SetInput(prefill_model_->GetInputName(0), input_data);
        prefill_model_->SetInput(prefill_model_->GetInputName(1), valid_tensor);
        prefill_model_->SetInput(prefill_model_->GetInputName(2), current_tensor);
        if (set_per_layer_inputs_for_prefill(pli_round.data(), current_length) != 0) {
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
    truncate_history(seq_id, static_cast<size_t>(effective_kv_offset));
    append_history(seq_id, chunk_history_[seq_id]);
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

    const size_t data_elems = static_cast<size_t>(n_tokens) * static_cast<size_t>(n_embd_);
    PrefillChunk chunk;
    chunk.is_embedding = true;
    chunk.bidirectional_attention = has_compact_header && visual_bidirectional_attention_;
    chunk.n_tokens = static_cast<size_t>(n_tokens);
    chunk.nx = header.nx;
    chunk.ny = header.ny;
    chunk.embeddings.resize(data_elems);
    // HMM multimodal encoders provide fp16 embeddings. Vision may prepend a
    // CompactHeader, while audio does not; both payloads are copied as fp16.
    std::memcpy(chunk.embeddings.data(), ptr + embedding_offset,
                data_elems * sizeof(uint16_t));
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
        const int32_t token_pos = decode_valid_length_[seq_id];
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
        append_history_token(seq_id, static_cast<size_t>(token_pos), batches[i]);
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
    if (embed_npu_ && embedding_layer_->is_npu_ready()) {
        embedding_layer_->get_embedding_batch_npu(
            tokens.data(), n_verify, input_data.Data(), 0);
    } else {
        embedding_layer_->get_embedding_batch(tokens.data(), n_verify, input_data.Data());
    }
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
        const auto committed_it = last_committed_tokens_.find(seq_id);
        const int32_t accepted_count = committed_it != last_committed_tokens_.end()
                                           ? committed_it->second
                                           : 0;
        set_input_data(accepted_tensor, n_decode_batch_, accepted_count);
        decode_model_->SetInput(accepted_count_input_name_, accepted_tensor);
        LLAMA_LOG_INFO("%s: verify accepted_count seq_id=%d committed=%d\n",
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
    for (int i = 0; i < n_verify; ++i) {
        if (logits_out[static_cast<size_t>(i)] != nullptr) {
            const size_t row_idx = std::min(static_cast<size_t>(i),
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
            const auto *src = static_cast<const uint8_t *>(hidden.Buffer().Data());
            for (int r = 0; r < rows; ++r) {
                std::memcpy(reinterpret_cast<uint8_t *>(pre_norm_hidden_dst_ +
                            static_cast<size_t>(r) * h_embd),
                            src + static_cast<size_t>(r) * h_embd * sizeof(ggml_fp16_t),
                            static_cast<size_t>(h_embd) * sizeof(ggml_fp16_t));
            }
            break;
        }
    }
    PERF_STOP(VERIFY_GETOUTPUT);

    pending_verify_[seq_id] = {true, start_pos, n_verify, tokens};
    return 0;
}

int HouMoGemma4::houmo_verify_accept(int seq_id, uint16_t n_accepted) {
    auto it = pending_verify_.find(seq_id);
    if (it == pending_verify_.end() || !it->second.valid) {
        return 0;
    }
    const int accepted_drafts = std::min<int>(
        static_cast<int>(n_accepted), std::max(0, it->second.n_verify - 1));
    const int committed = std::min(accepted_drafts + 1, it->second.n_verify);
    last_committed_tokens_[seq_id] = committed;
    const int32_t keep_pos = it->second.start_pos + committed;
    decode_valid_length_[seq_id] = keep_pos;
    update_device_kv_range(seq_id, keep_pos);
    kv_reuse_cache_.release_after(seq_id, keep_pos);
    if (memory_) {
        const int32_t cur = memory_->seq_pos_min(seq_id);
        if (cur < keep_pos) {
            memory_->seq_add(seq_id, 0, 0, keep_pos - cur);
        } else if (cur > keep_pos) {
            memory_->seq_rm(seq_id, keep_pos, cur);
        }
    }
    truncate_history(seq_id, static_cast<size_t>(it->second.start_pos));
    PrefillChunk accepted_chunk;
    accepted_chunk.n_tokens = static_cast<size_t>(committed);
    accepted_chunk.tokens.assign(it->second.tokens.begin(),
                                 it->second.tokens.begin() + committed);
    input_history_[seq_id].push_back(std::move(accepted_chunk));
    LLAMA_LOG_INFO("%s: verify commit seq_id=%d drafts=%d committed=%d keep_pos=%d\n",
                   __func__, seq_id, accepted_drafts, committed, keep_pos);
    it->second.valid = false;
    return 0;
}
