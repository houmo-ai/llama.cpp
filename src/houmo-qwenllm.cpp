#include "houmo-qwenllm.h"
#include <cstring>
#include "llama-impl.h"
#include "time-perf.h"
void HoumoQwenLLM::houmo_init(houmo_memory_i *memory, int seq_max) {
    model_inout_init(seq_max);
    if (is_embedding()) {
        memory_= nullptr;
    } else {
        memory_= memory;
    }

    // input, valid_lenght, current_length, lora_mask... from index 3.
    for (size_t index = 3; index < input_names_.size(); ++index) {
        if (prefill_input_map_.count(input_names_[index]) > 0
            && decode_input_map_.count(input_names_[index]) > 0) {
            auto lora_mask = prefill_input_map_[input_names_[index]];
            set_input_data(lora_mask, 1, 0);
            prefill_model_->SetInput(input_names_[index], lora_mask);
            auto decode_lora_mask = decode_input_map_[input_names_[index]];
            set_input_data(decode_lora_mask, 1, 0);
            decode_model_->SetInput(input_names_[index], decode_lora_mask);

            lora_mask_float_ =
                tcim::Tensor::CreateHostTensor(lora_mask.Info().AsContiguous().AsType(tcim::DataType::FLOAT32));
            LLAMA_LOG_INFO("set %d to %s\n", 0, input_names_[index].c_str());
        }
    }
}
int HoumoQwenLLM::houmo_prefill(std::vector<llama_token> &tokens, int seq_id,
                                float *logits) {

    LLAMA_LOG_INFO("%s seq_id = %d, tokens size is = %zu\n", "prefill stage",
                   seq_id, tokens.size());
    int32_t kv_offset = memory_ ? memory_->seq_pos_min(seq_id) : 0;
    int32_t cache_tokens = std::max(input_tokens_[seq_id].size(), input_embedding_ids_[seq_id].size() / n_embd_);
    if (( cache_tokens + tokens.size() + kv_offset) >= (uint32_t)context_length_) {
        LLAMA_LOG_ERROR("%s: input size %zu > context_length_%d, "
                        "reset to 0\n",
                        __func__, cache_tokens + tokens.size() + kv_offset, context_length_);
        input_tokens_[seq_id].resize(0);
        input_embedding_ids_[seq_id].resize(0);
        return 1;
    }
    input_tokens_[seq_id].insert(input_tokens_[seq_id].end(), tokens.begin(), tokens.end());
    if (logits == nullptr) {
        LLAMA_LOG_INFO(
            "%s: logits is null, just cache return, total size = %zu\n",
            __func__, cache_tokens + tokens.size());
        return 0;
    }


    int32_t current_length = 0;
    int32_t valid_length = kv_offset;
    std::vector<llama_token> input_ids;
    LLAMA_LOG_INFO("%s: seq_id = %d, kv_offset = %d\n", __func__, seq_id, kv_offset);

    auto &prefill_tokens_ = input_tokens_[seq_id];
    if (input_embedding_ids_[seq_id].size() >0) {
        int existing_embedding_tokens = input_embedding_ids_[seq_id].size();
        input_embedding_ids_[seq_id].resize(existing_embedding_tokens + prefill_tokens_.size() * n_embd_);
        uint16_t *input_data_ptr = input_embedding_ids_[seq_id].data() + existing_embedding_tokens;

        embedding_layer_->get_embedding_batch(prefill_tokens_, input_data_ptr);
        prefill_tokens_.resize(0);
    }
    size_t input_echo_len = input_tokens_[seq_id].size() == 0?  input_embedding_ids_[seq_id].size() / n_embd_: input_tokens_[seq_id].size();
    int16_t prefill_loop_round = ceil(static_cast<float>(input_echo_len) /
                                      static_cast<float>(prefill_length_));
    for (int round = 0; round < prefill_loop_round; round++) {
        auto &input_data = prefill_input_map_.at(prefill_model_->GetInputName(0));
        if (input_tokens_[seq_id].size() != 0) {
            if (round == prefill_loop_round - 1) {
                current_length = input_echo_len - round * prefill_length_;
                auto first = prefill_tokens_.begin() + round * prefill_length_;
                auto last = prefill_tokens_.begin() + input_echo_len;
                input_ids = std::vector<llama_token>(first, last);
            } else {
                current_length = prefill_length_;
                auto first = prefill_tokens_.begin() + round * prefill_length_;
                auto last = prefill_tokens_.begin() + (round + 1) * prefill_length_;
                input_ids = std::vector<llama_token>(first, last);
            }
            memset(input_data.Data(), 0, input_data.MemSize());
            embedding_layer_->get_embedding_batch(input_ids, input_data.Data());
        } else {
            // set embedding input directly
            if (round == prefill_loop_round - 1) {
                current_length = input_echo_len - round * prefill_length_;
            } else {
                current_length = prefill_length_;
            }
            memcpy(input_data.Data(), input_embedding_ids_[seq_id].data() + round * prefill_length_ * n_embd_, current_length * n_embd_ * sizeof(uint16_t));
        }
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
        PERF_START(PREFILL_SETINPUT_ONCE);
        for (const auto& pair : lora_map) {
            if (prefill_input_map_.count(pair.first) > 0) {
                auto lora_mask = prefill_input_map_[pair.first];
                lora_mask_float_.Buffer().CopyFromHost(&pair.second.scale, lora_mask_float_.MemSize());
                lora_mask_float_.CastTo(lora_mask);
                prefill_model_->SetInput(pair.first, lora_mask);
                LLAMA_LOG_DEBUG("%s input set val = %f\n", pair.first.c_str(), pair.second.scale);
            }
        }
        prefill_model_->SetInput(prefill_model_->GetInputName(0), input_data);
        prefill_model_->SetInput(prefill_model_->GetInputName(1),
                                 prefill_valid_length);
        prefill_model_->SetInput(prefill_model_->GetInputName(2),
                                 prefill_current_length);
        set_kv_cache_for_prefill(seq_id);
        PERF_STOP(PREFILL_SETINPUT_ONCE);
        PERF_START(PREFILL_RUN_ONCE);
        prefill_model_->Run();
        prefill_model_->Sync();
        PERF_STOP(PREFILL_RUN_ONCE);
        valid_length += prefill_length_;
    }
    decode_valid_length_[seq_id] = input_echo_len + kv_offset;
    PERF_START(PREFILL_GETOUTPUT);
    tcim::Tensor prefill_output_data =
        prefill_model_->GetOutput(prefill_output_names_[0]);
    // 转浮点
    auto &prefill_output_float = prefill_output_map_[prefill_output_names_[0]];
    prefill_output_data.CastTo(prefill_output_float);
    PERF_STOP(PREFILL_GETOUTPUT);
    memcpy(logits, prefill_output_float.Data(), prefill_output_float.MemSize());
    input_embedding_ids_[seq_id].resize(0);
    input_tokens_[seq_id].resize(0);
    if (memory_) {
        memory_->seq_add(seq_id, 0, 0, input_echo_len);
    }
    return 0;
}

int HoumoQwenLLM::houmo_prefill(const float *embeddings, int n_tokens,
                                int seq_id, float *logits) {
    (void) logits;
    int size = input_embedding_ids_[seq_id].size();
    input_embedding_ids_[seq_id].resize(
        input_embedding_ids_[seq_id].size() + (n_tokens + input_tokens_[seq_id].size()) * n_embd_);
    uint16_t *input_data_ptr = input_embedding_ids_[seq_id].data() + size;
    if (input_tokens_[seq_id].size() > 0) {
        // do embedding
        embedding_layer_->get_embedding_batch(input_tokens_[seq_id], input_data_ptr);
        input_tokens_[seq_id].resize(0);
    }
    std::memcpy(input_data_ptr + input_tokens_[seq_id].size() * n_embd_, embeddings,
                n_tokens * n_embd_ * sizeof(uint16_t));
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
    PERF_START(DECODE_SETINPUT);
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
    // 4. set lora_mask
    for (const auto& pair : lora_map) {
        if (decode_input_map_.count(pair.first) > 0) {
            auto lora_mask = decode_input_map_[pair.first];
            lora_mask_float_.Buffer().CopyFromHost(&pair.second.scale, lora_mask_float_.MemSize());
            lora_mask_float_.CastTo(lora_mask);
            decode_model_->SetInput(pair.first, lora_mask);
        }
    }
    PERF_STOP(DECODE_SETINPUT);
    // 5. run & sync
    PERF_START(DECODE_RUN);
    decode_model_->Run();
    decode_model_->Sync();
    PERF_STOP(DECODE_RUN);

    // 6.get output to logits
    PERF_START(DECODE_GETOUTPUT);
    tcim::Tensor decode_data =
        decode_model_->GetOutput(decode_output_names_[0]);
    // 7.转浮点
    auto &decode_output_float = decode_output_map_[decode_output_names_[0]];
    decode_data.CastTo(decode_output_float);
    auto n_vocab =
        decode_output_float.MemSize() / (n_decode_batch_ * sizeof(float));
    PERF_STOP(DECODE_GETOUTPUT);
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
        if (memory_) {
            memory_->seq_add(batch_id, 0, 0, 1);
        }
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