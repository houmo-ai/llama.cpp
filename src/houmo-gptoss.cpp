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
    int fill_length =
        (query_length == 1) ? 1 : 256; // prefill=256, decode=1（固定值）
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
                // 1. 计算当前token的绝对位置
                int current_abs_pos = old_cache_length + i;

                // 2. 正确计算滑窗的起始点（核心修正）
                // 也就是：当前位置 减去 窗口大小，再加1（因为是闭区间或长度计算），并防止为负
                int start_idx = std::max(0, current_abs_pos - sliding_window_ + 1);

                // 3. 计算结束点（通常是当前位置+1，即包含自身）
                int end_idx = current_abs_pos + 1;

                // 4. 双重保险（防止 end_idx 超过 width 边界，虽然逻辑上不应发生，但为了安全）
                if (fill_length == 256 &&  end_idx > (width - sliding_window_)) {
                    end_idx = end_idx - sliding_window_ -1;
                    start_idx = start_idx - sliding_window_ -1;
                } else if (end_idx > width) {
                    end_idx = sliding_window_;
                    start_idx = 0;
                }
                LLAMA_LOG_DEBUG("%s: current_abs_pos = %d, start_idx = %d, end_idx = %d\n",
                               __func__, current_abs_pos, start_idx, end_idx);
                
                // 有效区域置0（覆盖负无穷）
                for (int j = start_idx; j < end_idx; ++j) {
                    if (j >= width)
                        break; // 防止越界
                    // 索引计算：[batch, 1, fill_length, width]
                    int idx = b * 1 * fill_length * width // batch维度
                              + i * width                 // fill_length维度
                              + j;                        // width维度
                    mask_tensor[idx] = 0.0f;
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

void HouMoGPTOSS::houmo_init(houmo_memory_i *memory, int seq_max) {
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
    prefill_tokens_.insert(prefill_tokens_.end(), tokens.begin(), tokens.end());
    if (prefill_tokens_.size() >= (uint32_t)context_length_) {
        LLAMA_LOG_ERROR("%s: prefill_tokens_ size %zu > context_length_%d, "
                        "reset to 0\n",
                        __func__, prefill_tokens_.size(), context_length_);
        prefill_tokens_.resize(0);
        return 1;
    }
    if (logits == nullptr) {
        LLAMA_LOG_INFO(
            "%s: logits is null, just cache return, total size = %zu\n",
            __func__, prefill_tokens_.size());
        return 0;
    }
    size_t input_echo_len = prefill_tokens_.size();
    int16_t prefill_loop_round = ceil(static_cast<float>(input_echo_len) /
                                      static_cast<float>(prefill_length_));
    int32_t current_length = 0;

    std::vector<llama_token> input_ids;
    int kv_offset = memory_->seq_pos_min(seq_id);
    int32_t valid_length = kv_offset;
    for (int round = 0; round < prefill_loop_round; round++) {
        if (round == prefill_loop_round - 1) {
            current_length = input_echo_len - round * prefill_length_;
            auto first = prefill_tokens_.begin() + round * prefill_length_;
            auto last = prefill_tokens_.begin() + input_echo_len;
            std::vector<llama_token> round_tokens(first, last);
            input_ids = round_tokens;
        } else {
            current_length = prefill_length_;
            auto first = prefill_tokens_.begin() + round * prefill_length_;
            auto last = prefill_tokens_.begin() + (round + 1) * prefill_length_;
            std::vector<llama_token> round_tokens(first, last);
            input_ids = round_tokens;
        }
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
        set_attentions_for_prefill(current_length, valid_length - kv_offset);
        set_kv_cache_for_prefill(seq_id);
        prefill_model_->Run();
        prefill_model_->Sync();
        valid_length += prefill_length_;
    }
    decode_valid_length_[seq_id] = input_echo_len + kv_offset;
    tcim::Tensor prefill_output_data =
        prefill_model_->GetOutput(prefill_output_names_[0]);
    // 转浮点
    auto &prefill_output_float = prefill_output_map_[prefill_output_names_[0]];
    prefill_output_data.CastTo(prefill_output_float);
    memcpy(logits, prefill_output_float.Data(), prefill_output_float.MemSize());
    prefill_tokens_.resize(0);
    memory_->seq_add(seq_id, 0, 0, input_echo_len);
    return 0;
}

int HouMoGPTOSS::houmo_decode(std::vector<llama_token> &batches,
                              std::vector<int> &seq_ids,
                              std::vector<float *> logits) {
    LLAMA_LOG_DEBUG("%s: seq_ids size = %zu, batches size = %zu\n", __func__,
                    seq_ids.size(), batches.size());
    if (seq_ids.size() > (size_t)n_decode_batch_) {
        LLAMA_LOG_ERROR("%s: seq_ids.size() > n_decode_batch_ not supported\n", __func__);
        return -1;
    }
    // 1. 构造input_tokens
    auto ret = set_input_for_decode(batches, seq_ids);
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
        auto batch_id = seq_ids[i];
        decode_valid_length_[batch_id]++;
        // 单batch模式下所有序列共用batch 0
        if (n_decode_batch_ == 1)
            batch_id = 0;
        float *logits_out = logits[i];
        float *output_ptr = static_cast<float *>(decode_output_float.Data()) +
                            batch_id * n_vocab;
        memcpy(logits_out, output_ptr, n_vocab * sizeof(float));
        memory_->seq_add(batch_id, 0, 0, 1);
    }
    return 0;
}

int HouMoGPTOSS::set_attentions_for_prefill(int current_length, int valid_length) {
    // set local
    std::string name = prefill_model_->GetInputName(3);
    auto& local_mask_input_data = prefill_input_map_.at(name);
    auto prefill_local_mask_host = generate_mask_f16(1, current_length, valid_length, true);
    LLAMA_LOG_DEBUG("%s: current_length = %d, valid_length = %d, "
                   "prefill_local_mask_host.size() = %zu\n",
                   __func__, current_length, valid_length,
                   prefill_local_mask_host.size());
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
    for (auto seq_id : seq_ids) {
        int valid_length = decode_valid_length_[seq_ids[0]];
        // set local attention mask
        std::string name = local_attention_names_[seq_id];
        auto& local_mask_input_data = decode_input_map_.at(name);
        auto local_mask_host = generate_mask_f16(1, current_length, valid_length, true);
        local_mask_input_data.Buffer().CopyFromHost(local_mask_host.data(), local_mask_input_data.MemSize());
        decode_model_->SetInput(name, local_mask_input_data);

        // set global attention mask
        name = global_attention_names_[seq_id];
        auto& global_mask_input_data = decode_input_map_.at(name);
        auto global_mask_host = generate_mask_f16(1, current_length, valid_length, false);
        global_mask_input_data.Buffer().CopyFromHost(global_mask_host.data(), global_mask_input_data.MemSize());
        decode_model_->SetInput(name, global_mask_input_data);
    }

    return 0;
}