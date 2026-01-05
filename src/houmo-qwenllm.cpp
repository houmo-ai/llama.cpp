#include "houmo-qwenllm.h"
#include <cstring>
#include "llama-impl.h"

void HoumoQwenLLM::houmo_init(houmo_memory_i *memory, int seq_max) {
    model_inout_init(seq_max);
    memory_= memory;
}
int HoumoQwenLLM::houmo_prefill(std::vector<llama_token> &tokens, int seq_id,
                                float *logits) {

    LLAMA_LOG_INFO("%s seq_id = %d, tokens size is = %zu\n", "prefill stage",
                   seq_id, tokens.size());
    prefill_tokens_.insert(prefill_tokens_.end(), tokens.begin(), tokens.end());
    int32_t kv_offset = memory_->seq_pos_min(seq_id);
    if ((prefill_tokens_.size() + kv_offset) >= (uint32_t)context_length_) {
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
    int32_t valid_length = kv_offset;
    LLAMA_LOG_INFO("%s: seq_id = %d, kv_offset = %d\n", __func__, seq_id, kv_offset);
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
        if (prefill_input_map_.count(prefill_model_->GetInputName(3)) > 0) {
            // 设置 position_ids
            auto prefill_position_ids =
                prefill_input_map_.at(prefill_model_->GetInputName(3));
            std::vector<int32_t> position_ids(prefill_length_, 1);
            for (int i = 0; i < current_length; i++) {
                position_ids[i] = valid_length + i;
            }
            set_input_data(prefill_position_ids, position_ids);
            prefill_model_->SetInput(prefill_model_->GetInputName(3),
                                     prefill_position_ids);
        }
        LLAMA_LOG_INFO(
            "%s: valid_length = %d, current_length = %d, kv_offset = %d\n",
            __func__, valid_length, current_length, kv_offset);
        set_input_data(prefill_valid_length, n_prefill_batch_, valid_length);
        set_input_data(prefill_current_length, n_prefill_batch_,
                       current_length);

        prefill_model_->SetInput(prefill_model_->GetInputName(0), input_data);
        prefill_model_->SetInput(prefill_model_->GetInputName(1),
                                 prefill_valid_length);
        prefill_model_->SetInput(prefill_model_->GetInputName(2),
                                 prefill_current_length);
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

int HoumoQwenLLM::houmo_decode(std::vector<llama_token> &batches,
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
    // 3. set kv_cache
    set_kv_cache_for_decode(seq_ids);

    // 4. run & sync
    decode_model_->Run();
    decode_model_->Sync();

    // 5. get output to logits
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

int HoumoQwenLLM::houmo_embedding(
    const std::map<int32_t, std::vector<llama_token>> &batches,
    std::map<int, float *> &embeddings) {
    // 处理每个批次
    for (auto &[seq_id, tokens] : batches) {
        // 跳过空tokens
        if (tokens.empty()) {
            continue;
        }
        
        auto &embedding = embeddings[seq_id];
        
        // 验证embedding指针
        if (embedding == nullptr) {
            LLAMA_LOG_ERROR("%s: null embedding pointer for seq_id %d\n", __func__, seq_id);
            return -1;
        }

        // 调用houmo_prefill并检查返回值
        int ret = houmo_prefill(const_cast<std::vector<llama_token>&>(tokens), seq_id, embedding);
        if (ret != 0) {
            LLAMA_LOG_ERROR("%s: houmo_prefill failed for seq_id %d with error code %d\n",
                           __func__, seq_id, ret);
            return -1;
        }
    }
    return 0;
}
bool HoumoQwenLLM::is_embedding() {
    if (decode_model_ == nullptr) {
        return true;
    }
    return false;
}