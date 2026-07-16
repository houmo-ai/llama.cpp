#include "houmo-gptoss.h"
#include "ggml.h"
#include <algorithm>
#include <cstring>
#include "llama-impl.h"
static constexpr int ALIGN_NUM = 16; // 对齐值
static constexpr uint16_t F16_NEG_INF_UINT = 0xFBFF; // torch.finfo(torch.float16).min == -65504
// 辅助函数：内存对齐
inline int aligned(int size, int align) {
    return ((size + align - 1) / align) * align;
}

std::vector<uint16_t> HouMoGPTOSS::generate_mask_f16(int batch_size,
                                                    int fill_length,
                                                    int query_length,
                                                    int past_seq_length,
                                                    bool is_local) {
    int new_cache_length = query_length;
    int old_cache_length = past_seq_length;

    // ========== 2. 计算窗口大小 ==========
    int width; // nk（对应Python input_act_length/context_length）
    if (is_local) {
        width = aligned(fill_length + sliding_window_ - 1, 16);
    } else {
        width = context_length_;
    }

    // ========== 3. 初始化掩码 ==========
    const int mask_size = batch_size * 1 * fill_length * width;
    std::vector<uint16_t> mask_tensor(mask_size, F16_NEG_INF_UINT);

    // ========== 4. 核心填充逻辑 ==========
    for (int b = 0; b < batch_size; ++b) {
        // 仅循环new_cache_length次
        for (int i = 0; i < new_cache_length; ++i) {
            if (is_local) {
                // 对应 Python _gen_mask_v2(attention_max_length=sliding_window_):
                //   b_valid_length = min(valid_length, attention_max_length - 1)
                //   causal:  tril(diagonal=b_valid)         → visible if j <= i + b_valid
                //   window:  tril(diagonal=b_valid - sw)    → masked  if j <= i + b_valid - sw
                //   combined visible: i + b_valid - sw + 1 <= j <= i + b_valid
                int b_valid = std::min(old_cache_length, sliding_window_ - 1);
                int start_idx = std::max(0, i + b_valid - sliding_window_ + 1);
                int end_idx   = std::min(width - 1, i + b_valid);
                LLAMA_LOG_DEBUG("%s: i = %d, b_valid = %d, start_idx = %d, end_idx = %d\n",
                               __func__, i, b_valid, start_idx, end_idx);

                for (int j = start_idx; j <= end_idx; ++j) {
                    int idx = b * 1 * fill_length * width + i * width + j;
                    mask_tensor[idx] = 0;
                }
            } else {
                // -------- 全局掩码（global mask） --------
                int end_idx = old_cache_length + i + 1;
                if (end_idx > width)
                    end_idx = width; // 防止超过context_length
                // 有效区域置0（前old_cache_length+i个token）
                for (int j = 0; j < end_idx; ++j) {
                    int idx = b * 1 * fill_length * width + i * width + j;
                    mask_tensor[idx] = 0.0f;
                }
            }
        }
    }
    return mask_tensor;
}

void HouMoGPTOSS::houmo_init(houmo_memory_i *memory, int seq_max,
                             ggml_abort_callback abort_callback,
                             void *abort_callback_data) {
    abort_callback_data_ = abort_callback_data;
    abort_callback_ = abort_callback;
    model_inout_init(seq_max);
    prefill_local_attention_names_.clear();
    prefill_global_attention_names_.clear();
    local_attention_names_.clear();
    global_attention_names_.clear();
    for (size_t i = 0; i < prefill_model_->GetInputNum(); ++i) {
        std::string name = prefill_model_->GetInputName(i);
        if (name.find("local_attention_mask") != std::string::npos ||
            name.find("sliding_attention_mask") != std::string::npos) {
            prefill_local_attention_names_.push_back(name);
        } else if (name.find("global_attention_mask") != std::string::npos ||
                   name.find("full_attention_mask") != std::string::npos) {
            prefill_global_attention_names_.push_back(name);
        }
    }
    for (size_t i = 0; i < decode_model_->GetInputNum(); ++i) {
        std::string name = decode_model_->GetInputName(i);
        if (name.find("local_attention_mask") != std::string::npos ||
            name.find("sliding_attention_mask") != std::string::npos) {
            local_attention_names_.push_back(name);
        } else if (name.find("global_attention_mask") != std::string::npos ||
                   name.find("full_attention_mask") != std::string::npos) {
            global_attention_names_.push_back(name);
        }
    }
    memory_ = memory;
    // GPT-OSS alternates local/global attention. Even-numbered layers are
    // local layers and expose [1, 8, sliding_window + prefill_length, 64]
    // writable KV inputs (for example width 384 = 128 + 256).
    kv_reuse_cache_.init(kv_cache_, sliding_window_, prefill_length_);
}

bool HouMoGPTOSS::device_kv_covers_history(int seq_id,
                                           int64_t valid_length) const {
    if (valid_length <= 0) {
        return true;
    }
    const int64_t history_len = std::min<int64_t>(valid_length, sliding_window_);
    const int64_t history_start = valid_length - history_len;
    auto start = device_kv_abs_start_.find(seq_id);
    auto end = device_kv_abs_end_.find(seq_id);
    return start != device_kv_abs_start_.end() &&
           end != device_kv_abs_end_.end() && start->second <= history_start &&
           end->second >= valid_length;
}

void HouMoGPTOSS::update_device_kv_range(int seq_id, int64_t abs_end) {
    device_kv_abs_end_[seq_id] = abs_end;
    device_kv_abs_start_[seq_id] =
        std::max<int64_t>(0, abs_end - sliding_window_);
}

void HouMoGPTOSS::clear_device_kv_range(int seq_id) {
    device_kv_abs_start_.erase(seq_id);
    device_kv_abs_end_.erase(seq_id);
}

int HouMoGPTOSS::houmo_prefill(std::vector<llama_token> &tokens, int seq_id,
                           float *logits) {
    LLAMA_LOG_INFO("%s seq_id = %d, tokens size is = %zu\n", "prefill stage",
                   seq_id, tokens.size());
    int kv_offset = memory_->seq_pos_min(seq_id);
    if (kv_offset == 0) {
        input_history_[seq_id].clear();
        clear_device_kv_range(seq_id);
        kv_reuse_cache_.clear_seq(seq_id);
    }

    std::vector<llama_token> effective_tokens = tokens;
    int effective_kv_offset = kv_offset;
    if (kv_offset > 0 && kv_cache_.find(seq_id) != kv_cache_.end() &&
        !device_kv_covers_history(seq_id, kv_offset)) {
        auto &kv_item = kv_cache_[seq_id];
        if (kv_reuse_cache_.prepare_for_prefill(seq_id, kv_offset, kv_item)) {
            update_device_kv_range(seq_id, kv_offset);
        } else {
            int64_t rollback_pos =
                kv_reuse_cache_.find_nearest_before(seq_id, kv_offset);
            if (rollback_pos < 0) {
                rollback_pos = 0;
            }
            const auto &history = input_history_[seq_id];
            if (rollback_pos <= kv_offset &&
                static_cast<size_t>(kv_offset) <= history.size() &&
                (rollback_pos == 0 || kv_reuse_cache_.restore_at(
                                          seq_id, rollback_pos, kv_item))) {
                std::vector<llama_token> replay(
                    history.begin() + static_cast<std::ptrdiff_t>(rollback_pos),
                    history.begin() + static_cast<std::ptrdiff_t>(kv_offset));
                replay.insert(replay.end(), effective_tokens.begin(),
                              effective_tokens.end());
                effective_tokens.swap(replay);
                memory_->seq_rm(seq_id, static_cast<int32_t>(rollback_pos),
                                kv_offset);
                effective_kv_offset = static_cast<int>(rollback_pos);
                if (rollback_pos == 0) {
                    clear_device_kv_range(seq_id);
                } else {
                    update_device_kv_range(seq_id, rollback_pos);
                }
                kv_reuse_cache_.release_after(seq_id, rollback_pos);
                LLAMA_LOG_INFO(
                    "%s: GPT-OSS rollback seq_id=%d kv_offset=%d "
                    "rollback_pos=%ld replay_tokens=%zu\n",
                    __func__, seq_id, kv_offset, rollback_pos,
                    effective_tokens.size());
            } else {
                LLAMA_LOG_WARN(
                    "%s: GPT-OSS KV restore unavailable seq_id=%d "
                    "kv_offset=%d history=%zu; continue without rollback\n",
                    __func__, seq_id, kv_offset, history.size());
            }
        }
    }
    // Do not cache tokens anymore. Process the provided tokens immediately.
    if ((effective_tokens.size() + effective_kv_offset) >=
        static_cast<uint32_t>(context_length_)) {
        LLAMA_LOG_ERROR("%s: tokens size %zu + kv_offset %d >= context_length_%d, reset to 0\n",
                        __func__, effective_tokens.size(), effective_kv_offset,
                        context_length_);
        return 1;
    }

    size_t input_echo_len = effective_tokens.size();
    // compute number of rounds needed (integer ceil)
    int32_t prefill_loop_round = 0;
    prefill_loop_round = static_cast<int32_t>((input_echo_len + prefill_length_ - 1) / prefill_length_);
    int32_t current_length = 0;

    std::vector<llama_token> input_ids;
    int32_t valid_length = effective_kv_offset;
    for (int round = 0; round < prefill_loop_round; round++) {

        if (round == prefill_loop_round - 1) {
            current_length = static_cast<int32_t>(input_echo_len - round * prefill_length_);
            auto first = effective_tokens.begin() + round * prefill_length_;
            auto last = effective_tokens.begin() + input_echo_len;
            input_ids.assign(first, last);
        } else {
            current_length = prefill_length_;
            auto first = effective_tokens.begin() + round * prefill_length_;
            auto last = effective_tokens.begin() + (round + 1) * prefill_length_;
            input_ids.assign(first, last);
        }
        // LLAMA_LOG_DEBUG("%s: round %d/%d, valid_length = %d, current_length = %d\n", __func__, round + 1,
        //             prefill_loop_round, valid_length, current_length);
        auto &input_data = prefill_input_map_.at(prefill_model_->GetInputName(0));
        memset(input_data.Data(), 0, input_data.MemSize());
        embedding_layer_->get_embedding_batch(input_ids, input_data.Data());

        auto prefill_valid_length =
            prefill_input_map_.at(prefill_model_->GetInputName(1));
        auto prefill_current_length =
            prefill_input_map_.at(prefill_model_->GetInputName(2));
        set_input_data(prefill_valid_length, n_prefill_batch_, valid_length);
        set_input_data(prefill_current_length, n_prefill_batch_,
                       current_length);

        prefill_model_->SetInput(prefill_model_->GetInputName(0), input_data);
        prefill_model_->SetInput(prefill_model_->GetInputName(1),
                                 prefill_valid_length);
        prefill_model_->SetInput(prefill_model_->GetInputName(2),
                                 prefill_current_length);
        if (set_attentions_for_prefill(current_length, valid_length) != 0) {
            LLAMA_LOG_ERROR("%s: set_attentions_for_prefill failed\n",
                            __func__);
            return -1;
        }
        set_kv_cache_for_prefill(seq_id);
        auto status = prefill_model_->Run();
        if (status != tcim::Status::OK) {
            LLAMA_LOG_ERROR("%s: prefill_model_->Run() failed, result = %d\n",
                            __func__, static_cast<int>(status));
            return -1;
        }
        prefill_model_->Sync();
        valid_length += current_length;
        update_device_kv_range(seq_id, valid_length);
        const bool final_round = round == prefill_loop_round - 1;
        kv_reuse_cache_.save_after_prefill(
            seq_id, valid_length, kv_cache_[seq_id],
            final_round && valid_length > sliding_window_);
        if (abort_callback_ != nullptr && abort_callback_(abort_callback_data_)) {
            LLAMA_LOG_INFO("%s: abort_callback_ is called. seq_id = %d\n", __func__, seq_id);
            return 2;
        }
    }
    // update decode valid length and optionally copy logits
    decode_valid_length_[seq_id] = valid_length;
    tcim::Tensor prefill_output_data =
        prefill_model_->GetOutput(prefill_output_names_[0]);
    // 转浮点
    auto &prefill_output_float = prefill_output_map_[prefill_output_names_[0]];
    prefill_output_data.CastTo(prefill_output_float);
    if (logits != nullptr) {
        memcpy(logits, prefill_output_float.Data(), prefill_output_float.MemSize());
    }
    memory_->seq_add(seq_id, 0, 0, static_cast<int32_t>(input_echo_len));
    auto &history = input_history_[seq_id];
    history.resize(static_cast<size_t>(effective_kv_offset));
    history.insert(history.end(), effective_tokens.begin(),
                   effective_tokens.end());
    return 0;
}

int HouMoGPTOSS::houmo_decode(std::vector<llama_token> &batches,
                              std::vector<int> &seq_ids,
                              std::vector<float *> logits) {
    // LLAMA_LOG_DEBUG("%s: seq_ids size = %zu, batches size = %zu\n", __func__,
    //                 seq_ids.size(), batches.size());
    for (auto seq_id : seq_ids)
        decode_valid_length_[seq_id] = memory_->seq_pos_min(seq_id);
    int ret = 0;
    if (seq_ids.size() == 0) {
        return ret;
    }
    if (seq_ids.size() > (size_t)n_decode_batch_) {
        LLAMA_LOG_ERROR("%s: seq_ids.size() > n_decode_batch_ not supported\n", __func__);
        return -1;
    }
    // 1. 构造input_tokens
    ret = set_input_for_decode(batches, seq_ids);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: set_input_for_decode failed\n", __func__);
        return ret;
    }
    // 2. set valid_length & current_length
    ret = set_xlenght_for_decode(seq_ids);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: set_xlenght_for_decode failed\n", __func__);
        return ret;
    }
    // 3. set global_attention&current_length
    ret = set_attentions_for_decode(seq_ids);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: set_global_attention_for_decode failed\n",
                        __func__);
        return ret;
    }
    // 4. set kv_cache
    set_kv_cache_for_decode(seq_ids);

    // 5. run & sync
    decode_model_->Run();
    decode_model_->Sync();

    // 6. get output to logits
    tcim::Tensor decode_data =
        decode_model_->GetOutput(decode_output_names_[0]);
    const auto out_dtype = decode_data.Info().DataType();
    void *decode_ptr = decode_data.Buffer().Data();
    for (size_t i = 0; i < seq_ids.size(); i++) {
        auto seq_id = seq_ids[i];
        decode_valid_length_[seq_id]++;

        auto batch_id = seq_id;
        // 单batch模式下所有序列共用batch 0
        if (n_decode_batch_ == 1)
            batch_id = 0;
        float *logits_out = logits[i];
        if (logits_out != nullptr) {
            if (out_dtype == tcim::DataType::FLOAT16) {
                auto *output_ptr = static_cast<ggml_fp16_t *>(decode_ptr) +
                                   batch_id * n_vocab_;
                ggml_fp16_to_fp32_row(output_ptr, logits_out,
                                      static_cast<int>(n_vocab_));
            } else {
                LLAMA_LOG_ERROR("%s: unsupported decode output dtype=%d\n",
                                __func__, static_cast<int>(out_dtype));
                return -1;
            }
        }
        memory_->seq_add(seq_id, 0, 0, 1);
        auto &history = input_history_[seq_id];
        history.resize(static_cast<size_t>(decode_valid_length_[seq_id] - 1));
        history.push_back(batches[i]);
        update_device_kv_range(seq_id, decode_valid_length_[seq_id]);
    }
    return 0;
}

int HouMoGPTOSS::set_attentions_for_prefill(int current_length, int valid_length) {
    for (const auto &name : prefill_local_attention_names_) {
        auto &mask = prefill_input_map_.at(name);
        auto host = generate_mask_f16(1, prefill_length_, current_length,
                                      valid_length, true);
        mask.Buffer().CopyFromHost(host.data(), mask.MemSize());
        prefill_model_->SetInput(name, mask);
    }
    for (const auto &name : prefill_global_attention_names_) {
        auto &mask = prefill_input_map_.at(name);
        auto host = generate_mask_f16(1, prefill_length_, current_length,
                                      valid_length, false);
        mask.Buffer().CopyFromHost(host.data(), mask.MemSize());
        prefill_model_->SetInput(name, mask);
    }
    return 0;
}
int HouMoGPTOSS::set_attentions_for_decode(std::vector<int> &seq_ids) {
    int current_length = 1;
    for (size_t i = 0; i < seq_ids.size(); i++) {
        auto seq_id = seq_ids[i];
        int batch_id = seq_id;
        if (n_decode_batch_ == 1) batch_id = 0;

        int valid_length = decode_valid_length_[seq_id];
        // set local attention mask
        if (batch_id < static_cast<int>(local_attention_names_.size())) {
            std::string name = local_attention_names_[batch_id];
            auto &local_mask_input_data = decode_input_map_.at(name);
            auto local_mask_host = generate_mask_f16(
                1, 1, current_length, valid_length, true);
            local_mask_input_data.Buffer().CopyFromHost(
                local_mask_host.data(), local_mask_input_data.MemSize());
            decode_model_->SetInput(name, local_mask_input_data);
        }

        // set global attention mask
        if (batch_id < static_cast<int>(global_attention_names_.size())) {
            std::string name = global_attention_names_[batch_id];
            auto &global_mask_input_data = decode_input_map_.at(name);
            auto global_mask_host = generate_mask_f16(
                1, 1, current_length, valid_length, false);
            global_mask_input_data.Buffer().CopyFromHost(
                global_mask_host.data(), global_mask_input_data.MemSize());
            decode_model_->SetInput(name, global_mask_input_data);
        }
    }

    return 0;
}