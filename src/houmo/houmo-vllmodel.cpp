#include "houmo-vllmodel.h"
#include "ggml.h"
#include "gguf.h"
#include "llama-impl.h"
#include <regex>
#include <cstring>
#include "time-perf.h"

#pragma pack(push, 1)
struct CompactHeader {
    int16_t magic;      // 16位标识
    int16_t nx;         // 16位宽度
    int16_t ny;         // 16位高度
    int16_t chunksize;  // 16位块大小
};
#pragma pack(pop)

HouMoVLLModel::HouMoVLLModel() : HouMoLLModel::Impl() {
    // do nothing
}

void HouMoVLLModel::houmo_init(houmo_memory_i *memory, int seq_max,
                               ggml_abort_callback abort_callback,
                               void *abort_callback_data) {
    abort_callback_data_ = abort_callback_data;
    abort_callback_ = abort_callback;
    /**
     *  qwen2.5vl：input1, time_position_id, height_position_id, width_position_id, valid_length, current_length
     *  qwen3vl：..., deepstack_features_0, deepstack_features_1, deepstack_features_2
     */
    memory_ = memory;
    int actual_seq_max = std::min(seq_max, kSeqMax);
    model_inout_init(actual_seq_max);
    // do init
    input_embedding_ids_.resize(actual_seq_max);
    offset_embedding_ids_.resize(actual_seq_max);
    position_indexs_.resize(actual_seq_max);
    prefill_input_lengths_.resize(actual_seq_max);
    deepstack_features_0_.resize(actual_seq_max);
    deepstack_features_1_.resize(actual_seq_max);
    deepstack_features_2_.resize(actual_seq_max);
    for (int i = 0; i < actual_seq_max; i++) {
        input_embedding_ids_[i].resize(context_length_ * n_embd_, 0);
        if (prefill_input_map_.size() == 9) {
            deepstack_features_0_[i].resize(context_length_ * n_embd_, 0);
            deepstack_features_1_[i].resize(context_length_ * n_embd_, 0);
            deepstack_features_2_[i].resize(context_length_ * n_embd_, 0);
        }
    }

    // qwen3vl decoder模型需要初始化 deepstack_features_ 为0
    if (decode_input_map_.size() == 9) {
        for (int i = 0; i < 3; i++) {
            auto name = decode_model_->GetInputName(6 + i);
            if (n_decode_batch_ > 1) {
                if (decode_input_map_.count(name) == 0) {
                    name = name + "_batch" + std::to_string(i);
                }
            }
            auto deepstack =
                decode_input_map_.at(name);
            memset(deepstack.Data(), 0, deepstack.MemSize());
            decode_model_->SetInput(name, deepstack);
        }
    }
    chunk_history_.resize(actual_seq_max);
}

int HouMoVLLModel::houmo_prefill(std::vector<llama_token> &tokens, int seq_id,
                                 float *logits) {

    LLAMA_LOG_INFO("%s seq_id = %d, tokens size is = %zu\n", "prefill stage",
                    seq_id, tokens.size());
    int32_t kv_offset = memory_->seq_pos_min(seq_id);

    if ((offset_embedding_ids_[seq_id] + tokens.size() + kv_offset) >=
        (size_t)context_length_) {
        LLAMA_LOG_WARN(
            "excceed context length, reset offset_embedding_id_ to 0\n");
        reset_position_ids(seq_id);
        return 1;
    }
    int32_t current_length = 0;
    // 先计算embedding.
    LLAMA_LOG_INFO("%s: empty tokens, skip embedding. seq_id = %d\n", __func__, seq_id);
    embedding_layer_->get_embedding_batch(
        tokens, input_embedding_ids_[seq_id].data() +
                    offset_embedding_ids_[seq_id] * n_embd_);

    offset_embedding_ids_[seq_id] += tokens.size();
    prefill_input_lengths_[seq_id].push_back(tokens.size());
    chunk_history_[seq_id].push_back(tokens.size());
    // 计算所有embeddings
    if (logits == nullptr && tokens.size() > 0) {
        // 只缓存输入
        LLAMA_LOG_INFO("%s: logits is nullptr, only cache input. seq_id = %d\n", __func__, seq_id);
        return 0;
    }
    size_t input_echo_len = offset_embedding_ids_[seq_id];
    if (input_echo_len == 0) {
        return 0;
    }
    size_t prefill_loop_round = ceil(static_cast<float>(input_echo_len) /
    static_cast<float>(prefill_length_));

    std::vector<int32_t> time_position_ids;
    std::vector<int32_t> high_position_ids;
    std::vector<int32_t> width_position_ids;
    compute_position_ids(seq_id, time_position_ids, high_position_ids,
                         width_position_ids);
    time_position_ids.resize(prefill_loop_round * prefill_length_);
    high_position_ids.resize(prefill_loop_round * prefill_length_);
    width_position_ids.resize(prefill_loop_round * prefill_length_);

    int valid_length = kv_offset;
    LLAMA_LOG_INFO("%s: valid_length = %d, current_length = %d\n", __func__, valid_length, current_length);
    // 计算prefill输出
    for (size_t round = 0; round < prefill_loop_round; round++) {
        if (round == prefill_loop_round - 1) {
            current_length = input_echo_len - round * prefill_length_;
        } else {
            current_length = prefill_length_;
        }
        auto input_data =
            prefill_input_map_.at(prefill_model_->GetInputName(0));
        input_data.Buffer().CopyFromHost(input_embedding_ids_[seq_id].data() +
                                             round * prefill_length_ * n_embd_,
                                         input_data.MemSize());
        auto time_position_id =
            prefill_input_map_.at(prefill_model_->GetInputName(1));
        auto high_position_id =
            prefill_input_map_.at(prefill_model_->GetInputName(2));
        auto width_position_id =
            prefill_input_map_.at(prefill_model_->GetInputName(3));
        auto prefill_valid_length =
            prefill_input_map_.at(prefill_model_->GetInputName(4));
        auto prefill_current_length =
            prefill_input_map_.at(prefill_model_->GetInputName(5));
        LLAMA_LOG_DEBUG("%s: valid_length = %d, current_length = %d, kv_offset = %d\n", __func__, valid_length, current_length, kv_offset);
        set_input_data(time_position_id, time_position_ids, valid_length - kv_offset);
        set_input_data(high_position_id, high_position_ids, valid_length - kv_offset);
        set_input_data(width_position_id, width_position_ids, valid_length - kv_offset);
        set_input_data(prefill_valid_length, n_prefill_batch_, valid_length);
        set_input_data(prefill_current_length, n_prefill_batch_,
                       current_length);
        // 当前仅qwen3vl生效
        fill_deepstack_embed(seq_id, valid_length - kv_offset);
        prefill_model_->SetInput(prefill_model_->GetInputName(0), input_data);
        prefill_model_->SetInput(prefill_model_->GetInputName(1),
                                 time_position_id);
        prefill_model_->SetInput(prefill_model_->GetInputName(2),
                                 high_position_id);
        prefill_model_->SetInput(prefill_model_->GetInputName(3),
                                 width_position_id);
        prefill_model_->SetInput(prefill_model_->GetInputName(4),
                                 prefill_valid_length);
        prefill_model_->SetInput(prefill_model_->GetInputName(5),
                                 prefill_current_length);
        set_kv_cache_for_prefill(seq_id);
        prefill_model_->Run();
        prefill_model_->Sync();
        if (abort_callback_ != nullptr && abort_callback_(abort_callback_data_)) {
            LLAMA_LOG_INFO("%s: abort_callback_ is called. seq_id = %d\n", __func__, seq_id);
            reset_position_ids(seq_id);
            return 2;
        }
        valid_length += prefill_length_;
    }
    decode_valid_length_[seq_id] = input_echo_len + kv_offset;
    memory_->seq_add(seq_id, 0, 0, input_echo_len);
    tcim::Tensor prefill_output_data =
        prefill_model_->GetOutput(prefill_output_names_[0]);
    // 转浮点
    auto &prefill_output_float = prefill_output_map_[prefill_output_names_[0]];
    prefill_output_data.CastTo(prefill_output_float);
    memcpy(logits, prefill_output_float.Data(), prefill_output_float.MemSize());
    // reset input_embedding_ids_
    reset_position_ids(seq_id);
    return 0;
}

int HouMoVLLModel::houmo_prefill(const float *embeddings, int n_tokens,
                                 int seq_id, float *logits) {
    LLAMA_LOG_INFO("%s seq_id = %d, n_tokens is = %d\n", "prefill stage", seq_id,
                   n_tokens);
    if (logits == nullptr) {
        // 只缓存输入
        if ((n_tokens + offset_embedding_ids_[seq_id]) >= context_length_) {
            LLAMA_LOG_WARN(
                "excceed context length, reset offset_embedding_id_ to 0\n");
            reset_position_ids(seq_id);
            return 1;
        }
        CompactHeader header;
        int embedding_offset = 0;
        std::memcpy(&header, embeddings, sizeof(CompactHeader));
        if (header.magic == 0x1234) {
            ny = header.ny;
            nx = header.nx;
            embedding_offset = sizeof(CompactHeader);
        }
        const int8_t* embeddings_ptr = reinterpret_cast<const int8_t *>(embeddings);
        const int data_size_bytes = n_tokens * n_embd_ * sizeof(int16_t);
        memcpy(input_embedding_ids_[seq_id].data() +
                   offset_embedding_ids_[seq_id] * n_embd_,
               embeddings_ptr + embedding_offset, data_size_bytes);
        if (header.magic == 0x1234 && header.chunksize == 4) {
            // for qwenvl3
            embedding_offset += data_size_bytes;
            LLAMA_LOG_INFO("deepstack feature 0 offset = %d, addr = %p\n", embedding_offset, embeddings_ptr + embedding_offset);
            memcpy(deepstack_features_0_[seq_id].data() +
                   offset_embedding_ids_[seq_id] * n_embd_,
                   embeddings_ptr + embedding_offset, data_size_bytes);
            embedding_offset += data_size_bytes;

            memcpy(deepstack_features_1_[seq_id].data() +
                   offset_embedding_ids_[seq_id] * n_embd_,
                   embeddings_ptr + embedding_offset, data_size_bytes);
            embedding_offset += data_size_bytes;
            memcpy(deepstack_features_2_[seq_id].data() +
                   offset_embedding_ids_[seq_id] * n_embd_,
                   embeddings_ptr + embedding_offset, data_size_bytes);
        }
        offset_embedding_ids_[seq_id] += n_tokens;
        prefill_input_lengths_[seq_id].push_back(n_tokens);
        chunk_history_[seq_id].push_back(n_tokens);
        return 0;
    } else {
        LLAMA_LOG_ERROR("%s: logits is not null, not supported\n", __func__);
                    reset_position_ids(seq_id);
        return -1;
    }
}

int HouMoVLLModel::houmo_decode(std::vector<llama_token> &batches,
                                std::vector<int> &seq_ids, std::vector<float *> logits) {
    LLAMA_LOG_DEBUG("%s: seq_ids size = %zu, batches size = %zu\n", __func__,
                   seq_ids.size(), batches.size());
    for (auto seq_id : seq_ids) {
        if (prefill_input_lengths_[seq_id].size() != 0) {
            LLAMA_LOG_INFO("%s: seq_id = %d, process cache input\n", __func__, seq_id);
            std::vector<llama_token> empty_tokens;
            houmo_prefill(empty_tokens, seq_id, nullptr);
        }
        int target_tokens = memory_->seq_pos_min(seq_id);

        if (target_tokens < decode_valid_length_[seq_id]) {
            int current_tokens = 0;
            int current_pos = 0;

            // Replay history to find exact position mapping
            for (int len : chunk_history_[seq_id]) {
                if (current_tokens >= target_tokens) break;

                if (len == nx * ny) {
                    current_tokens += len;
                    current_pos += nx;
                } else {
                    int step = std::min(len, target_tokens - current_tokens);
                    current_tokens += step;
                    current_pos += step;
                }
            }

            // Account for any single tokens generated during decode so far
            if (target_tokens > current_tokens) {
                current_pos += (target_tokens - current_tokens);
            }

            decode_valid_length_[seq_id] = target_tokens;
            position_indexs_[seq_id] = current_pos;

            LLAMA_LOG_INFO("%s: seq_id = %d, rolled back to valid_length = %d, position_index = %d\n",
                            __func__, seq_id, target_tokens, current_pos);
        } else {
            decode_valid_length_[seq_id] = target_tokens;
        }
    }
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

    // 2. set valid_length & current_length &position_ids
    ret = set_xlenght_for_decode(seq_ids);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: fill_decode_xlengths failed\n", __func__);
        return ret;
    }

    // 3. set position_ids
    std::vector<int32_t> position_vec(n_decode_batch_, 1);
    for (size_t k = 0; k < seq_ids.size(); k++) {
        int batch_id = seq_ids[k];
        // 单batch模式下所有序列共用batch 0
        if (n_decode_batch_ == 1)
            batch_id = 0;
        position_vec[batch_id]= position_indexs_[seq_ids[k]];
    }
    auto time_position_id =
        decode_input_map_.at(decode_model_->GetInputName(1));
    auto hight_position_id =
        decode_input_map_.at(decode_model_->GetInputName(2));
    auto width_position_id =
        decode_input_map_.at(decode_model_->GetInputName(3));

    set_input_data(time_position_id, position_vec);
    set_input_data(hight_position_id, position_vec);
    set_input_data(width_position_id, position_vec);
    decode_model_->SetInput(decode_model_->GetInputName(1), time_position_id);
    decode_model_->SetInput(decode_model_->GetInputName(2), hight_position_id);
    decode_model_->SetInput(decode_model_->GetInputName(3), width_position_id);

    // 4. set kv_cache
    set_kv_cache_for_decode(seq_ids);
    PERF_STOP(DECODE_SETINPUT);
    PERF_START(DECODE_RUN);
    // 5. run & sync
    decode_model_->Run();
    decode_model_->Sync();
    PERF_STOP(DECODE_RUN);

    // 6. get output to logits
    PERF_START(DECODE_GETOUTPUT);
    tcim::Tensor decode_data =
        decode_model_->GetOutput(decode_output_names_[0]);
    // 转浮点
    auto &decode_output_float = decode_output_map_[decode_output_names_[0]];
    decode_data.CastTo(decode_output_float);
    auto n_vocab =
        decode_output_float.MemSize() / (n_decode_batch_ * sizeof(float));
    PERF_STOP(DECODE_GETOUTPUT);
    // 7. 拷贝output到logits
    for (size_t i = 0; i < seq_ids.size(); i++) {
        auto batch_id = seq_ids[i];
        decode_valid_length_[batch_id]++;
        position_indexs_[batch_id]++;
        // 单batch模式下所有序列共用batch 0
        if (n_decode_batch_ == 1)
            batch_id = 0;
        float *logits_out = logits[i];
        float *output_ptr = static_cast<float *>(decode_output_float.Data()) +
                            batch_id * n_vocab;
        memcpy(logits_out, output_ptr, n_vocab * sizeof(float));
        memory_->seq_add(seq_ids[i], 0, 0, 1);
    }
    return 0;
}

void HouMoVLLModel::update_position_ids(int n_tokens,
                                    std::vector<int32_t> &time_position_ids,
        std::vector<int32_t> &high_position_ids,
        std::vector<int32_t> &width_position_ids,
        int32_t &start_pos) {
    if (n_tokens == nx * ny) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                time_position_ids.push_back(start_pos);
                high_position_ids.push_back(start_pos + y);
                width_position_ids.push_back(start_pos + x);
            }
        }

        start_pos += nx;
    } else {
        for (int i = 0; i < n_tokens; i++) {
            time_position_ids.push_back(start_pos + i);
            high_position_ids.push_back(start_pos + i);
            width_position_ids.push_back(start_pos + i);
        }
        start_pos += n_tokens;
    }
}

void HouMoVLLModel::compute_position_ids(
    int seq_id, std::vector<int32_t> &time_position_ids,
    std::vector<int32_t> &high_position_ids,
    std::vector<int32_t> &width_position_ids) {
    auto &lens = prefill_input_lengths_[seq_id];
    // mutimedia call prefill three times.
    position_indexs_[seq_id] = memory_->seq_pos_min(seq_id);
    LLAMA_LOG_DEBUG("%s: seq_id = %d, position_indexs_[seq_id] = %d\n",
                    __func__, seq_id, position_indexs_[seq_id]);
    for (size_t i = 0; i < lens.size(); ++i) {
        update_position_ids(lens[i], time_position_ids,
                            high_position_ids, width_position_ids,
                            position_indexs_[seq_id]);
    }
}

void HouMoVLLModel::reset_position_ids(int seq_id) {
    offset_embedding_ids_[seq_id] = 0;
    prefill_input_lengths_[seq_id].resize(0);
}

void HouMoVLLModel::fill_deepstack_embed(int seq_id, int valid_length) {
    if (prefill_input_map_.size() != 9)
        return;
    auto name = prefill_model_->GetInputName(6);
    auto deepstack_tensor0 = prefill_input_map_.at(name);
    deepstack_tensor0.Buffer().CopyFromHost(
        deepstack_features_0_[seq_id].data() + valid_length * n_embd_,
        deepstack_tensor0.MemSize());
    prefill_model_->SetInput(name, deepstack_tensor0);
    memset(deepstack_features_0_[seq_id].data() + valid_length * n_embd_, 0,
           deepstack_tensor0.MemSize());

    name = prefill_model_->GetInputName(7);
    auto deepstack_tensor1 = prefill_input_map_.at(name);
    deepstack_tensor1.Buffer().CopyFromHost(
        deepstack_features_1_[seq_id].data() + valid_length * n_embd_,
        deepstack_tensor1.MemSize());
    prefill_model_->SetInput(name, deepstack_tensor1);
    memset(deepstack_features_1_[seq_id].data() + valid_length * n_embd_, 0,
           deepstack_tensor1.MemSize());

    name = prefill_model_->GetInputName(8);
    auto deepstack_tensor2 = prefill_input_map_.at(name);
    deepstack_tensor2.Buffer().CopyFromHost(
        deepstack_features_2_[seq_id].data() + valid_length * n_embd_,
        deepstack_tensor2.MemSize());
    prefill_model_->SetInput(name, deepstack_tensor2);
    memset(deepstack_features_2_[seq_id].data() + valid_length * n_embd_, 0,
           deepstack_tensor2.MemSize());

}
