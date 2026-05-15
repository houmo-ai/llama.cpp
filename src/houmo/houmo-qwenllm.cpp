#include "houmo-qwenllm.h"
#include "ggml.h"
#include <cstring>
#include <algorithm>
#include "llama-impl.h"
#include "time-perf.h"

void HoumoQwenLLM::houmo_init(houmo_memory_i *memory, int seq_max,
                              ggml_abort_callback abort_callback,
                              void *abort_callback_data) {
    abort_callback_data_ = abort_callback_data;
    abort_callback_ = abort_callback;
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
    size_t embeddings_available = input_embedding_ids_[seq_id].size() / n_embd_;
    if ((embeddings_available + tokens.size() + kv_offset) >= (uint32_t)context_length_) {
        LLAMA_LOG_ERROR("%s: input size %zu > context_length_%d, "
                        "reset to 0\n",
                        __func__, embeddings_available + tokens.size() + kv_offset, context_length_);
        // clear embedding cache
        input_embedding_ids_[seq_id].resize(0);
        return 1;
    }

    // Determine total available tokens. We support mixed caches: existing
    // embeddings (stored first) plus the incoming token ids (stored after).
    size_t tokens_available = tokens.size();
    size_t total_tokens = embeddings_available + tokens_available;

    // Decide how many tokens to actually process now.
    // Process all available tokens regardless of whether logits is provided.
    size_t process_tokens = total_tokens;
    if (process_tokens == 0) {
        LLAMA_LOG_INFO(
            "%s: no tokens to process, just cache return, total size = %zu\n",
            __func__, total_tokens);
        return 0;
    }


    int32_t current_length = 0;
    int32_t valid_length = kv_offset;
    std::vector<llama_token> input_ids;
    LLAMA_LOG_INFO("%s: seq_id = %d, kv_offset = %d\n", __func__, seq_id, kv_offset);

    auto &prefill_tokens_ = tokens; // process incoming tokens directly, no caching

    // number of rounds needed to process `process_tokens` tokens
    size_t rounds = (process_tokens + prefill_length_ - 1) / prefill_length_;

    // reuse vector for token slices to avoid reallocations
    std::vector<llama_token> token_slice;
    token_slice.reserve(prefill_length_);

    for (size_t round = 0; round < rounds; ++round) {
        auto &input_data = prefill_input_map_.at(prefill_model_->GetInputName(0));

        size_t round_start = round * prefill_length_;
        size_t remaining = process_tokens > round_start ? process_tokens - round_start : 0;
        current_length = remaining >= (size_t)prefill_length_ ? prefill_length_ : (int)remaining;

        // three cases: entirely from embeddings, entirely from tokens, or spanning both
        if (round_start + current_length <= embeddings_available) {
            // entirely in embeddings
            size_t emb_start = round_start * n_embd_;
            size_t copy_bytes = current_length * n_embd_ * sizeof(uint16_t);
            memcpy(input_data.Data(),
                   input_embedding_ids_[seq_id].data() + emb_start,
                   copy_bytes);
        } else if (round_start >= embeddings_available) {
            // entirely in tokens
            size_t token_offset = round_start - embeddings_available;
            // avoid creating a temporary vector: call overload that accepts raw pointer
            const int32_t *token_ptr = prefill_tokens_.data() + token_offset;
            memset(input_data.Data(), 0, input_data.MemSize());
            embedding_layer_->get_embedding_batch(token_ptr, current_length, input_data.Data());
        } else {
            // spans embeddings and tokens
            size_t emb_part = embeddings_available > round_start ? (embeddings_available - round_start) : 0;
            size_t token_part = current_length - emb_part;
            // copy embeddings portion
            size_t emb_start = round_start * n_embd_;
            memcpy(input_data.Data(),
                   input_embedding_ids_[seq_id].data() + emb_start,
                   emb_part * n_embd_ * sizeof(uint16_t));
            // prepare token slice (tokens start at index 0 in prefill_tokens_)
            auto token_ptr = prefill_tokens_.data();
            // embed token part into tail of input_data
            embedding_layer_->get_embedding_batch(token_ptr, token_part,
                                                 static_cast<uint16_t*>(input_data.Data()) + emb_part * n_embd_);
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
        LLAMA_LOG_DEBUG(
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
        if (round == 0)
            set_kv_cache_for_prefill(seq_id);
        PERF_STOP(PREFILL_SETINPUT_ONCE);
        PERF_START(PREFILL_RUN_ONCE);
        auto status = prefill_model_->Run();
        if (status != tcim::Status::OK) {
            LLAMA_LOG_ERROR("%s: prefill_model_->Run() failed, result = %d\n",
                            __func__, static_cast<int>(status));
            return -1;
        }
        prefill_model_->Sync();
        PERF_STOP(PREFILL_RUN_ONCE);
        valid_length += current_length;
        if (abort_callback_ != nullptr && abort_callback_(abort_callback_data_)) {
            LLAMA_LOG_INFO("%s: abort_callback_ is called. seq_id = %d\n", __func__, seq_id);
            return 2;
        }
    }
    decode_valid_length_[seq_id] = kv_offset + (int32_t)process_tokens;
    PERF_START(PREFILL_GETOUTPUT);
    tcim::Tensor prefill_output_data =
        prefill_model_->GetOutput(prefill_output_names_[0]);
    // 转浮点
    auto &prefill_output_float = prefill_output_map_[prefill_output_names_[0]];
    prefill_output_data.CastTo(prefill_output_float);
    PERF_STOP(PREFILL_GETOUTPUT);
    if (logits != nullptr) {
        memcpy(logits, prefill_output_float.Data(), prefill_output_float.MemSize());
    }

    // clear embedding cache
    input_embedding_ids_[seq_id].resize(0);

    if (memory_) {
        memory_->seq_add(seq_id, 0, 0, (int32_t)process_tokens);
    }
    return 0;
}

int HoumoQwenLLM::houmo_prefill(const float *embeddings, int n_tokens,
                                int seq_id, float *logits) {
    (void) logits;
    int size = input_embedding_ids_[seq_id].size();
    input_embedding_ids_[seq_id].resize(
        input_embedding_ids_[seq_id].size() + n_tokens * n_embd_);
    uint16_t *input_data_ptr = input_embedding_ids_[seq_id].data() + size;
    std::memcpy(input_data_ptr, embeddings, n_tokens * n_embd_ * sizeof(uint16_t));
    return 0;
}

int HoumoQwenLLM::houmo_decode(std::vector<llama_token> &batches,
                               std::vector<int> &seq_ids,
                               std::vector<float *> logits) {
    LLAMA_LOG_DEBUG("%s: seq_ids size = %zu, batches size = %zu\n", __func__,
                    seq_ids.size(), batches.size());
    for (auto seq_id : seq_ids)
        decode_valid_length_[seq_id] = memory_->seq_pos_min(seq_id);
    int ret = 0;
    if (seq_ids.size() > (size_t)n_decode_batch_) {
        LLAMA_LOG_ERROR("%s: seq_ids.size() > n_decode_batch_ not supported\n", __func__);
        return -1;
    }
    PERF_START(DECODE_SETINPUT);
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
    // 3. set kv_cache
    set_kv_cache_for_decode(seq_ids);
    // 4. set lora_mask
    for (const auto &pair : lora_map) {
        if (decode_input_map_.count(pair.first) > 0) {
            auto lora_mask = decode_input_map_[pair.first];
            lora_mask_float_.Buffer().CopyFromHost(&pair.second.scale,
                                                   lora_mask_float_.MemSize());
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
    const auto out_dtype = decode_data.Info().DataType();
    void *decode_ptr = decode_data.Buffer().Data();
    PERF_STOP(DECODE_GETOUTPUT);
    for (size_t i = 0; i < seq_ids.size(); i++) {
        auto batch_id = seq_ids[i];
        decode_valid_length_[batch_id]++;
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
        if (memory_) {
            memory_->seq_add(seq_ids[i], 0, 0, 1);
        }
    }
    return 0;
}

int HoumoQwenLLM::houmo_verify(std::vector<llama_token> &tokens, int seq_id,
                               std::vector<float *> &logits_out) {
    if (tokens.size() != logits_out.size()) {
        LLAMA_LOG_ERROR("%s: tokens.size(%zu) != logits_out.size(%zu)\n",
                        __func__, tokens.size(), logits_out.size());
        return -1;
    }
    const int k = (int)tokens.size();
    if (k <= 0) {
        return 0;
    }

    // 判断 decode 图的输入形状是否支持 [1, k, n_embedding] 批量验证：
    //   - 维度 >= 3
    //   - shape[0] == 1 (单序列)
    //   - shape[1] >= k (序列长度维度足以容纳 k 个 token)
    // 若不满足，则退化为逐 token 调用 houmo_decode。
    int graph_seq_len = 1;
    bool batched_capable = false;
    if (decode_model_ != nullptr && !decode_input_map_.empty()) {
        auto input_name = decode_model_->GetInputName(0);
        auto it = decode_input_map_.find(input_name);
        if (it != decode_input_map_.end()) {
            auto shape = it->second.Info().Shape();
            if (shape.size() >= 3 && shape[0] == 1 && shape[1] > 1 &&
                (int64_t)k <= shape[1]) {
                graph_seq_len = (int)shape[1];
                batched_capable = true;
            }
        }
    }

    if (!batched_capable) {
        // 退化路径：一个 token 一个 token 调用 houmo_decode
        std::vector<int> seq_ids = {seq_id};
        for (int i = 0; i < k; ++i) {
            std::vector<llama_token> one = {tokens[i]};
            std::vector<float *> one_logits = {logits_out[i]};
            int ret = houmo_decode(one, seq_ids, one_logits);
            if (ret != 0) {
                LLAMA_LOG_ERROR("%s: fallback houmo_decode failed at i=%d\n",
                                __func__, i);
                return ret;
            }
        }
        return 0;
    }

    // 批量路径：一次 decode 处理 k 个 token
    LLAMA_LOG_DEBUG("%s: batched verify, seq_id=%d k=%d graph_seq_len=%d\n",
                    __func__, seq_id, k, graph_seq_len);
    decode_valid_length_[seq_id] = memory_->seq_pos_min(seq_id);
    const int base_pos = decode_valid_length_[seq_id];
    if (base_pos + k > context_length_) {
        LLAMA_LOG_WARN("%s: base_pos(%d) + k(%d) exceeds context_length_(%d)\n",
                       __func__, base_pos, k, context_length_);
        return 1;
    }

    PERF_START(DECODE_SETINPUT);
    // 1. 填充 k 个 token 的 embedding 到输入张量 [1, graph_seq_len, n_embd]
    //    未使用的尾部位置保持原有内容（由 causal mask + valid_length 屏蔽）。
    auto input_name = decode_model_->GetInputName(0);
    auto input_data = decode_input_map_.at(input_name);
    try {
        embedding_layer_->get_embedding_batch(tokens, input_data.Data());
    } catch (...) {
        LLAMA_LOG_ERROR("%s: get_embedding_batch failed\n", __func__);
        return -1;
    }
    decode_model_->SetInput(input_name, input_data);

    // 2. 设置 valid_length / current_length
    //    valid_length = 当前已有的历史 token 数 (base_pos)
    //    current_length = 本次新增 token 数 (k)
    if (decode_valid_length_names_.size() != 1 ||
        decode_current_length_names_.size() != 1) {
        LLAMA_LOG_ERROR("%s: batched verify requires single valid/current "
                        "length inputs, got %zu/%zu\n",
                        __func__, decode_valid_length_names_.size(),
                        decode_current_length_names_.size());
        return -1;
    }
    {
        auto valid_tensor =
            decode_input_map_.at(decode_valid_length_names_[0]);
        set_input_data(valid_tensor, 1, base_pos);
        decode_model_->SetInput(decode_valid_length_names_[0], valid_tensor);

        auto current_tensor =
            decode_input_map_.at(decode_current_length_names_[0]);
        set_input_data(current_tensor, 1, k);
        decode_model_->SetInput(decode_current_length_names_[0],
                                current_tensor);
    }

    // 3. 设置 kv cache (复用单 batch 逻辑)
    std::vector<int> seq_ids_one = {seq_id};
    set_kv_cache_for_decode(seq_ids_one);

    // 4. lora mask
    for (const auto &pair : lora_map) {
        if (decode_input_map_.count(pair.first) > 0) {
            auto lora_mask = decode_input_map_[pair.first];
            lora_mask_float_.Buffer().CopyFromHost(&pair.second.scale,
                                                   lora_mask_float_.MemSize());
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

    // 6. 取输出、转浮点
    PERF_START(DECODE_GETOUTPUT);
    tcim::Tensor decode_data =
        decode_model_->GetOutput(decode_output_names_[0]);
    const auto out_dtype = decode_data.Info().DataType();
    void *decode_ptr = decode_data.Buffer().Data();
    // 输出形状 [1, graph_seq_len, n_vocab]
    GGML_ASSERT(out_dtype == tcim::DataType::FLOAT16);
    auto total_floats = decode_data.MemSize() / sizeof(uint16_t);
    auto n_vocab = total_floats / (size_t)graph_seq_len;
    PERF_STOP(DECODE_GETOUTPUT);

    // 7. 复制每个位置的 logits
    for (int i = 0; i < k; ++i) {
        if (logits_out[i] != nullptr) {
            if (out_dtype == tcim::DataType::FLOAT16) {
                auto *base_ptr = static_cast<ggml_fp16_t *>(decode_ptr);
                ggml_fp16_to_fp32_row(base_ptr + (size_t)i * n_vocab,
                                      logits_out[i],
                                      static_cast<int>(n_vocab));
            } else {
                LLAMA_LOG_ERROR("%s: unsupported decode output dtype=%d\n",
                                __func__, static_cast<int>(out_dtype));
                return -1;
            }
        }
    }

    // 8. 更新 KV / valid_length：本次提交了 k 个 token
    decode_valid_length_[seq_id] = base_pos + k;
    if (memory_) {
        memory_->seq_add(seq_id, 0, 0, k);
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