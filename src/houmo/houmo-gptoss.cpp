#include "houmo-gptoss.h"
#include <cstring>
#include "llama-impl.h"
static constexpr int ALIGN_NUM = 16; // 对齐值
static constexpr uint16_t F16_NEG_INF_UINT = 0xFC00; // 二进制：1111110000000000

// 辅助函数：内存对齐
inline int aligned(int size, int align) {
    return ((size + align - 1) / align) * align;
}
std::vector<uint16_t> HouMoGPTOSS::generate_mask_f16(int batch_size,
                                                    int query_length,
                                                    int past_seq_length,
                                                    bool is_local) {
    // ========== 1. 兼容原有参数，==========
    int fill_length = (query_length == 1) ? 1 : prefill_length_; // prefill=256, decode=1（固定值）
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
    int sum = decode_model_->GetInputNum();
    for (int i = 0; i < sum; i++) {
        std::string name = decode_model_->GetInputName(i);
        if (name.find("local_attention_mask") != std::string::npos) {
            local_attention_names_.push_back(name);
        } else if (name.find("global_attention_mask") != std::string::npos) {
            global_attention_names_.push_back(name);
        }
    }
    memory_ = memory;
}
int HouMoGPTOSS::houmo_prefill(std::vector<llama_token> &tokens, int seq_id,
                           float *logits) {
    LLAMA_LOG_INFO("%s seq_id = %d, tokens size is = %zu\n", "prefill stage",
                   seq_id, tokens.size());
    int kv_offset = memory_->seq_pos_min(seq_id);
    // Do not cache tokens anymore. Process the provided tokens immediately.
    if ((tokens.size() + kv_offset) >= (uint32_t)context_length_) {
        LLAMA_LOG_ERROR("%s: tokens size %zu + kv_offset %d >= context_length_%d, reset to 0\n",
                        __func__, tokens.size(), kv_offset, context_length_);
        return 1;
    }

    size_t input_echo_len = tokens.size();
    // compute number of rounds needed (integer ceil)
    int32_t prefill_loop_round = 0;
    prefill_loop_round = static_cast<int32_t>((input_echo_len + prefill_length_ - 1) / prefill_length_);
    int32_t current_length = 0;

    std::vector<llama_token> input_ids;
    int32_t valid_length = kv_offset;
    for (int round = 0; round < prefill_loop_round; round++) {

        if (round == prefill_loop_round - 1) {
            current_length = static_cast<int32_t>(input_echo_len - round * prefill_length_);
            auto first = tokens.begin() + round * prefill_length_;
            auto last = tokens.begin() + input_echo_len;
            input_ids.assign(first, last);
        } else {
            current_length = prefill_length_;
            auto first = tokens.begin() + round * prefill_length_;
            auto last = tokens.begin() + (round + 1) * prefill_length_;
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
        set_attentions_for_prefill(current_length, valid_length);
        set_kv_cache_for_prefill(seq_id);
        prefill_model_->Run();
        prefill_model_->Sync();
        valid_length += current_length;
    }
    // update decode valid length and optionally copy logits
    decode_valid_length_[seq_id] = static_cast<int32_t>(input_echo_len) + kv_offset;
    tcim::Tensor prefill_output_data =
        prefill_model_->GetOutput(prefill_output_names_[0]);
    // 转浮点
    auto &prefill_output_float = prefill_output_map_[prefill_output_names_[0]];
    prefill_output_data.CastTo(prefill_output_float);
    if (logits != nullptr) {
        memcpy(logits, prefill_output_float.Data(), prefill_output_float.MemSize());
    }
    memory_->seq_add(seq_id, 0, 0, static_cast<int32_t>(input_echo_len));
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
    // 转浮点
    auto &decode_output_float = decode_output_map_[decode_output_names_[0]];
    decode_data.CastTo(decode_output_float);
    auto n_vocab =
        decode_output_float.MemSize() / (n_decode_batch_ * sizeof(float));
    for (size_t i = 0; i < seq_ids.size(); i++) {
        auto seq_id = seq_ids[i];
        decode_valid_length_[seq_id]++;

        auto batch_id = seq_id;
        // 单batch模式下所有序列共用batch 0
        if (n_decode_batch_ == 1)
            batch_id = 0;
        float *logits_out = logits[i];
        float *output_ptr = static_cast<float *>(decode_output_float.Data()) +
                            batch_id * n_vocab;
        memcpy(logits_out, output_ptr, n_vocab * sizeof(float));
        memory_->seq_add(seq_id, 0, 0, 1);
    }
    return 0;
}

int HouMoGPTOSS::set_attentions_for_prefill(int current_length, int valid_length) {
    // set local
    std::string name = prefill_model_->GetInputName(3);
    auto& local_mask_input_data = prefill_input_map_.at(name);
    auto prefill_local_mask_host = generate_mask_f16(1, current_length, valid_length, true);
    local_mask_input_data.Buffer().CopyFromHost(
        prefill_local_mask_host.data(), local_mask_input_data.MemSize());
    prefill_model_->SetInput(name, local_mask_input_data);

    // set global attention mask
    name = prefill_model_->GetInputName(4);
    auto &global_mask_input_data = prefill_input_map_.at(name);
    auto prefill_global_mask_host = generate_mask_f16(1, current_length, valid_length, false);
    global_mask_input_data.Buffer().CopyFromHost(prefill_global_mask_host.data(), global_mask_input_data.MemSize());
    prefill_model_->SetInput(name, global_mask_input_data);
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
        std::string name = local_attention_names_[batch_id];
        auto& local_mask_input_data = decode_input_map_.at(name);
        auto local_mask_host = generate_mask_f16(1, current_length, valid_length, true);
        local_mask_input_data.Buffer().CopyFromHost(local_mask_host.data(), local_mask_input_data.MemSize());
        decode_model_->SetInput(name, local_mask_input_data);

        // set global attention mask
        name = global_attention_names_[batch_id];
        auto& global_mask_input_data = decode_input_map_.at(name);
        auto global_mask_host = generate_mask_f16(1, current_length, valid_length, false);
        global_mask_input_data.Buffer().CopyFromHost(global_mask_host.data(), global_mask_input_data.MemSize());
        decode_model_->SetInput(name, global_mask_input_data);
    }

    return 0;
}