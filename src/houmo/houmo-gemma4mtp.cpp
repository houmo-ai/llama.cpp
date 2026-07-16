#include "houmo-gemma4mtp.h"

#include "ggml.h"
#include "llama-impl.h"
#include "time-perf.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <sstream>

namespace {

static constexpr uint16_t F16_NEG_MIN_UINT =
    0xFBFF; // torch.finfo(torch.float16).min == -65504

size_t last_dim(const tcim::TensorInfo &info) {
    const auto shape = info.Shape();
    return shape.empty() ? 0 : static_cast<size_t>(shape.back());
}

bool contains_any(const std::string &name, const char *a, const char *b) {
    return name.find(a) != std::string::npos ||
           name.find(b) != std::string::npos;
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

std::string topk_token_ids(const float *logits, size_t n, int k) {
    std::vector<std::pair<float, int>> top;
    top.reserve(static_cast<size_t>(k));
    for (size_t i = 0; i < n; ++i) {
        const float v = logits[i];
        auto it = std::lower_bound(top.begin(), top.end(), v,
                                   [](const std::pair<float, int> &a, float b) {
                                       return a.first > b;
                                   });
        if (static_cast<int>(top.size()) < k) {
            top.insert(it, {v, static_cast<int>(i)});
        } else if (it != top.end()) {
            top.insert(it, {v, static_cast<int>(i)});
            top.pop_back();
        }
    }
    std::ostringstream oss;
    for (size_t i = 0; i < top.size(); ++i) {
        if (i > 0) {
            oss << ',';
        }
        oss << top[i].second << ':' << top[i].first;
    }
    return oss.str();
}

std::vector<uint16_t> build_draft_mask_f16(int fill_length, int width,
                                           int visible_length,
                                           int cache_valid_end,
                                           int sliding_window,
                                           bool local) {
    std::vector<uint16_t> mask(static_cast<size_t>(std::max(1, fill_length)) *
                                   width,
                               F16_NEG_MIN_UINT);
    if (width <= 0 || fill_length <= 0) {
        return mask;
    }
    // Before the cache full this
    // is a prefix [0, N); after it exceeds the window it becomes
    // [N - sliding_window, N). The current assistant token is passed through
    // input_1 and position/valid_length and is not visible in this mask.
    const int visible = std::min(sliding_window, std::max(1, visible_length));
    const int end = std::min(width, std::max(1, cache_valid_end));
    const int start = local ? std::max(0, end - visible) : 0;
    for (int k = start; k < end; ++k) {
        mask[static_cast<size_t>(k)] = 0;
    }
    for (int q = 1; q < fill_length; ++q) {
        mask[static_cast<size_t>(q) * width] = 0;
    }
    return mask;
}

void set_scalar_i32(tcim::Module *m, const std::string &name,
                    tcim::Tensor &tensor, int32_t value) {
    const size_t n = tensor.MemSize() / sizeof(int32_t);
    std::vector<int32_t> buf(n, value);
    tensor.Buffer().CopyFromHost(buf.data(), n * sizeof(int32_t));
    m->SetInput(name, tensor);
}

} // namespace

void HouMoGemma4MTP::houmo_init(houmo_memory_i *memory, int seq_max,
                                ggml_abort_callback abort_callback,
                                void *abort_callback_data) {
    abort_callback_data_ = abort_callback_data;
    abort_callback_ = abort_callback;
    model_inout_init(seq_max);
    memory_ = memory;
    if (target_ != nullptr) {
        target_cache_inputs_ = collect_target_cache_inputs();
    }
    bind_target_cache_inputs();
    resolve_mtp_names(decode_model_.get());
    resolve_target_sliding_mask_width();
}

void HouMoGemma4MTP::resolve_target_sliding_mask_width() {
    target_sliding_mask_width_ = 0;
    if (target_ == nullptr) {
        return;
    }
    for (const auto &name : target_->local_attention_names_) {
        const auto it = target_->decode_input_map_.find(name);
        if (it == target_->decode_input_map_.end()) {
            continue;
        }
        const auto shape = it->second.Info().Shape();
        if (shape.size() > 3 && shape[3] > 0) {
            target_sliding_mask_width_ = static_cast<int32_t>(shape[3]);
            LLAMA_LOG_INFO(
                "%s: target local mask name='%s' physical width=%d\n",
                __func__, name.c_str(), target_sliding_mask_width_);
            return;
        }
    }
    LLAMA_LOG_WARN(
        "%s: unable to resolve target local mask width; MTP mask will use "
        "the absolute committed position\n",
        __func__);
}

std::vector<std::pair<std::string, tcim::Tensor>> HouMoGemma4MTP::collect_target_cache_inputs() const {
    if (target_ == nullptr) {
        return {};
    }
    if (target_->prefill_model_ == nullptr) {
        LLAMA_LOG_ERROR("%s: target prefill module is unavailable for MTP cache binding\n",
                        __func__);
        return {};
    }
    if (!target_->kv_cache_.empty()) {
        const auto it = target_->kv_cache_.find(0);
        const kv_cache_item *item = it != target_->kv_cache_.end()
                                      ? &it->second
                                      : &target_->kv_cache_.begin()->second;
        if (item->names.size() >= 4) {
            const size_t n = item->names.size();
            std::vector<std::pair<std::string, tcim::Tensor>> out;
            out.reserve(4);
            for (size_t idx = n - 4; idx < n; ++idx) {
                const auto &name = item->names[idx];
                out.emplace_back(name, target_->prefill_model_->GetDevInput(name));
            }
            LLAMA_LOG_INFO("%s: Gemma4 MTP target prefill cache mapping: sliding=(%s,%s) full=(%s,%s)\n",
                           __func__, out[0].first.c_str(), out[1].first.c_str(),
                           out[2].first.c_str(), out[3].first.c_str());
            return out;
        }
        LLAMA_LOG_WARN("%s: runtime KV cache names are insufficient for MTP binding: names=%zu\n",
                       __func__, item->names.size());
    }

    LLAMA_LOG_ERROR("%s: no runtime KV cache found for MTP binding\n", __func__);
    return {};
}

void HouMoGemma4MTP::bind_target_cache_inputs() {
    static const char *const kDraftCacheNames[] = {
        "shared_key_cache_sliding",
        "shared_value_cache_sliding",
        "shared_key_cache_full",
        "shared_value_cache_full",
    };
    if (target_cache_inputs_.size() != 4) {
        LLAMA_LOG_WARN("%s: skip Gemma4 MTP cache binding, target caches=%zu\n",
                       __func__, target_cache_inputs_.size());
        return;
    }
    if (decode_model_ == nullptr) {
        return;
    }
    for (size_t i = 0; i < target_cache_inputs_.size(); ++i) {
        const auto status = decode_model_->SetDevInput(
            kDraftCacheNames[i], target_cache_inputs_[i].second);
        if (status != tcim::Status::OK) {
            LLAMA_LOG_WARN("%s: failed to bind draft cache '%s', status=%d\n",
                           __func__, kDraftCacheNames[i], (int)status);
        }
    }
}

void HouMoGemma4MTP::resolve_mtp_names(tcim::Module *model) {
    mtp_names_ = {};
    if (model == nullptr) {
        return;
    }
    if (model->GetInputNum() > 0)
        mtp_names_.mtp_input = model->GetInputName(0);
    for (size_t i = 1; i < model->GetInputNum(); ++i) {
        const std::string name = model->GetInputName(i);
        if (mtp_names_.valid_length.empty() &&
            contains_any(name, "valid_length", "past_seq_length")) {
            mtp_names_.valid_length = name;
        } else if (mtp_names_.current_length.empty() &&
                   contains_any(name, "current_length",
                                "current_input_length")) {
            mtp_names_.current_length = name;
        } else if (mtp_names_.local_mask.empty() &&
                   contains_any(name, "local_attention_mask",
                                "sliding_attention_mask")) {
            mtp_names_.local_mask = name;
        }
    }
    if (model->GetOutputNum() > 0)
        mtp_names_.logits_out = model->GetOutputName(0);
    if (model->GetOutputNum() > 1)
        mtp_names_.hidden_out = model->GetOutputName(1);

    LLAMA_LOG_INFO(
        "%s: input='%s' valid='%s' current='%s' local='%s' logits='%s' "
        "hidden='%s'\n",
        __func__, mtp_names_.mtp_input.c_str(), mtp_names_.valid_length.c_str(),
        mtp_names_.current_length.c_str(), mtp_names_.local_mask.c_str(),
        mtp_names_.logits_out.c_str(), mtp_names_.hidden_out.c_str());
}

int HouMoGemma4MTP::houmo_decode_mtp(const float *h_in,
                                     const std::vector<llama_token> &tokens,
                                     int seq_id,
                                     std::vector<float *> &logits_out,
                                     float *h_out, int32_t start_pos) {
    if (tokens.size() != logits_out.size() || tokens.empty()) {
        return tokens.empty() ? 0 : -1;
    }

    const bool is_prefill = std::all_of(
        logits_out.begin(), logits_out.end(),
        [](const float *out) { return out == nullptr; });
    if (is_prefill) {
        if (memory_ != nullptr) {
            memory_->seq_add(seq_id, 0, 0,
                             static_cast<int32_t>(tokens.size()));
        }
        LLAMA_LOG_INFO("%s: Gemma4 MTP prefill seq_id=%d tokens=%zu "
                       "start_pos=%d advance_memory=%d\n",
                       __func__, seq_id, tokens.size(), start_pos,
                       memory_ != nullptr ? 1 : 0);
        return 0;
    }

    if (tokens.size() != 1 || logits_out[0] == nullptr || h_in == nullptr) {
        LLAMA_LOG_ERROR("%s: Gemma4 MTP decode requires exactly one token, "
                        "one logits output and one hidden row; tokens=%zu "
                        "logits=%p hidden=%p\n",
                        __func__, tokens.size(),
                        tokens.empty() ? nullptr : (void *) logits_out[0],
                        (const void *) h_in);
        return -1;
    }

    align_draft_memory(seq_id, start_pos);
    return houmo_decode_mtp_one(h_in, tokens[0], seq_id, logits_out[0], h_out,
                                start_pos);
}

void HouMoGemma4MTP::align_draft_memory(int seq_id, int32_t start_pos) {
    if (memory_ == nullptr || target_ == nullptr || start_pos < 0) {
        return;
    }
    const int32_t target_pos = target_->committed_pos(seq_id);
    if (target_pos < 0 || start_pos != target_pos) {
        return;
    }
    const int32_t keep_pos = std::max<int32_t>(0, target_pos - 1);
    const int32_t draft_pos = memory_->seq_pos_min(seq_id);
    if (draft_pos < keep_pos) {
        memory_->seq_add(seq_id, 0, 0, keep_pos - draft_pos);
    } else if (draft_pos > keep_pos) {
        memory_->seq_rm(seq_id, keep_pos, draft_pos);
    }
    // The target module can rebind its runtime KV inputs after initialization
    // and verification. Refresh the handles at each accepted draft boundary;
    // keeping only the initialization-time handles makes the assistant read a
    // stale target-cache view.
    target_cache_inputs_ = collect_target_cache_inputs();
    bind_target_cache_inputs();
    LLAMA_LOG_INFO("%s: Gemma4 MTP new round seq_id=%d target_pos=%d "
                   "draft_pos=%d -> keep_pos=%d\n",
                   __func__, seq_id, target_pos, draft_pos, keep_pos);
}

int HouMoGemma4MTP::houmo_decode_mtp_one(const float *h_in,
                                         llama_token token,
                                         int seq_id,
                                         float *logits_out,
                                         float *h_out,
                                         int32_t start_pos) {
    PERF_START(MTP_PRERUN);
    auto &input_tensor = decode_input_map_.at(mtp_names_.mtp_input);
    const size_t input_row = last_dim(input_tensor.Info());
    const size_t embed_row =
        static_cast<size_t>(embedding_layer_->get_n_embd());
    const size_t hidden_row = static_cast<size_t>(n_embd_);
    if (input_row != embed_row + hidden_row) {
        LLAMA_LOG_ERROR("%s: invalid MTP input row, input_row=%zu expect=%zu\n",
                        __func__, input_row, embed_row + hidden_row);
        PERF_STOP(MTP_PRERUN);
        return -1;
    }
    const int32_t base_pos =
        start_pos >= 0 ? start_pos : (memory_ ? memory_->seq_pos_min(seq_id) : 0);
    std::vector<uint8_t> input_buf(input_tensor.MemSize(), 0);
    const size_t embed_row_bytes = embed_row * sizeof(ggml_fp16_t);
    const size_t hidden_row_bytes = hidden_row * sizeof(ggml_fp16_t);
    const auto *hidden_src = reinterpret_cast<const uint8_t *>(h_in);
    embedding_layer_->get_embedding_batch(&token, 1, input_buf.data());
    std::memcpy(input_buf.data() + embed_row_bytes, hidden_src,
                hidden_row_bytes);
    input_tensor.Buffer().CopyFromHost(input_buf.data(), input_tensor.MemSize());
    decode_model_->SetInput(mtp_names_.mtp_input, input_tensor);

    const int32_t pos = base_pos;
    const int32_t target_pos = target_ ? target_->committed_pos(seq_id) : -1;
    // committed_pos() and base_pos are exclusive sequence ends.  The MTP
    // graph, however, consumes (h_{i-1}, token_i) and expects valid_length
    // to be the position before token_i.  This matches the Python reference:
    //     past_len=N, pos_idx=N, valid_length=N-1, mask_source=N.
    const int32_t graph_valid_length = std::max<int32_t>(0, pos - 1);
    const int32_t absolute_mask_past = std::max<int32_t>(
        1, target_pos >= 0 ? target_pos : base_pos);
    // The shared sliding cache grows from index zero and is capped by the
    // target decode cache capacity. Do not tail-align every MTP mask to 1040:
    // Python uses [0, N) before the cache fills and [N - 1024, N) afterwards.
    const int32_t cache_valid_end = target_sliding_mask_width_ > 0
                                        ? std::min(absolute_mask_past,
                                                   target_sliding_mask_width_)
                                        : absolute_mask_past;
    const int32_t visible_length =
        std::min(cache_valid_end, sliding_window_);

    LLAMA_LOG_INFO(
        "%s: MTP pos-calc seq_id=%d start_pos=%d "
        "draft_pos_min=%d draft_pos_max=%d target_pos_min=%d "
        "-> pos=%d valid_length=%d "
        "absolute_mask_past=%d visible_length=%d cache_valid_end=%d "
        "target_mask_width=%d\n",
        __func__, seq_id, start_pos,
        memory_ ? memory_->seq_pos_min(seq_id) : -1,
        memory_ ? memory_->seq_pos_max(seq_id) : -1, target_pos, pos,
        graph_valid_length, absolute_mask_past, visible_length,
        cache_valid_end, target_sliding_mask_width_);

    set_scalar_i32(decode_model_.get(), mtp_names_.valid_length,
                   decode_input_map_.at(mtp_names_.valid_length),
                   graph_valid_length);
    set_scalar_i32(decode_model_.get(), mtp_names_.current_length,
                   decode_input_map_.at(mtp_names_.current_length), 1);

    auto &mask = decode_input_map_.at(mtp_names_.local_mask);
    const auto shape = mask.Info().Shape();
    const int fill = shape.size() > 2 ? static_cast<int>(shape[2]) : 1;
    const int width =
        shape.size() > 3 ? static_cast<int>(shape[3]) : context_length_;
    auto host = build_draft_mask_f16(fill, width, visible_length,
                                     cache_valid_end, sliding_window_, true);
    mask.Buffer().CopyFromHost(host.data(), mask.MemSize());
    decode_model_->SetInput(mtp_names_.local_mask, mask);
    const int zero_end = std::min(width, std::max(1, cache_valid_end));
    const int zero_start = std::max(0, zero_end - visible_length);
    LLAMA_LOG_INFO("%s: MTP mask name='%s' local=%d fill=%d width=%d "
                   "visible_length=%d cache_valid_end=%d "
                   "zero_range=[%d,%d) current=%zu\n",
                   __func__, mtp_names_.local_mask.c_str(), 1, fill, width,
                   visible_length, cache_valid_end, zero_start, zero_end,
                   size_t{1});
    PERF_STOP(MTP_PRERUN);

    PERF_START(MTP_RUN);
    auto status = decode_model_->Run();
    if (status != tcim::Status::OK) {
        LLAMA_LOG_ERROR("%s: MTP Run failed status=%d\n", __func__,
                        (int)status);
        PERF_STOP(MTP_RUN);
        return -1;
    }
    decode_model_->Sync();
    PERF_STOP(MTP_RUN);
    PERF_START(MTP_OUTPUT);
    auto logits = decode_model_->GetOutput(mtp_names_.logits_out);
    const size_t vocab_row = last_dim(logits.Info());
    const size_t logits_bytes = vocab_row * sizeof(ggml_fp16_t);
    if (vocab_row == 0 || logits.MemSize() < logits_bytes) {
        LLAMA_LOG_ERROR(
            "%s: invalid MTP logits output row=%zu bytes=%zu expect_bytes=%zu\n",
            __func__, vocab_row, logits.MemSize(), logits_bytes);
        PERF_STOP(MTP_OUTPUT);
        return -1;
    }
    auto *logit_base = static_cast<const ggml_fp16_t *>(logits.Buffer().Data());
    ggml_fp16_to_fp32_row(logit_base, logits_out,
                          static_cast<int64_t>(vocab_row));
    LLAMA_LOG_DEBUG(
        "%s: MTP output seq_id=%d input_token=%d valid_length=%d "
        "top1=%d top4=[%s] vocab=%zu\n",
        __func__, seq_id, static_cast<int>(token),
        std::max<int32_t>(0, pos - 1),
        top_token_id(logits_out, vocab_row),
        topk_token_ids(logits_out, vocab_row, 4).c_str(), vocab_row);

    if (h_out != nullptr && !mtp_names_.hidden_out.empty()) {
        auto hidden = decode_model_->GetOutput(mtp_names_.hidden_out);
        const size_t output_row = last_dim(hidden.Info());
        if (output_row != hidden_row || hidden.MemSize() < hidden_row_bytes) {
            LLAMA_LOG_ERROR(
                "%s: invalid MTP hidden output row=%zu bytes=%zu "
                "expect_row=%zu expect_bytes=%zu\n",
                __func__, output_row, hidden.MemSize(), hidden_row,
                hidden_row_bytes);
            PERF_STOP(MTP_OUTPUT);
            return -1;
        }
        const auto *src = static_cast<const uint8_t *>(hidden.Buffer().Data());
        auto *dst = reinterpret_cast<uint8_t *>(h_out);
        std::memcpy(dst, src, hidden_row_bytes);
    }
    if (memory_) {
        memory_->seq_add(seq_id, 0, 0, 1);
    }
    PERF_STOP(MTP_OUTPUT);
    return 0;
}
