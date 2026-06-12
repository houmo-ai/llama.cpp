#include "houmo-glmocrmodel.h"

#include "llama-impl.h"
#include "time-perf.h"

#include <algorithm>
#include <cstring>

#pragma pack(push, 1)
struct GLMOCRCompactHeader {
    int16_t magic;
    int16_t nx;
    int16_t ny;
    int16_t chunksize;
};
#pragma pack(pop)

void HouMoGLMOCRModel::houmo_init(houmo_memory_i *memory, int seq_max,
                                  ggml_abort_callback abort_callback,
                                  void *abort_callback_data) {
    memory_ = memory;
    abort_callback_ = abort_callback;
    abort_callback_data_ = abort_callback_data;

    const int actual_seq_max = std::min(seq_max, kSeqMax);

    prefill_input_init();
    prefill_output_init();
    decode_input_init();
    decode_output_init();

    int kv_cache_count = 0;
    for (size_t i = 0; i < decode_model_->GetInputNum(); i++) {
        const std::string name = decode_model_->GetInputName(i);
        if (name.find("_self_attn_kcache_input") != std::string::npos ||
            name.find("past_key_cache_") != std::string::npos) {
            kv_cache_count++;
        }
    }
    if (kv_cache_count > 0 && kv_cache_count != n_blocks_) {
        LLAMA_LOG_INFO("%s: GLM-OCR KV cache layers = %d, GGUF block_count = %d\n",
                       __func__, kv_cache_count, n_blocks_);
        n_blocks_ = kv_cache_count;
    }

    int size = (actual_seq_max + n_prefill_batch_ - 1) / n_prefill_batch_;
    size = std::max(size, n_decode_batch_);
    initialize_kv_cache(size);

    input_embedding_ids_.resize(actual_seq_max);
    offset_embedding_ids_.resize(actual_seq_max, 0);
    position_indexs_.resize(actual_seq_max, 0);
    prefill_chunks_.resize(actual_seq_max);
    chunk_history_.resize(actual_seq_max);

    for (int i = 0; i < actual_seq_max; i++) {
        input_embedding_ids_[i].resize(context_length_ * n_embd_, 0);
    }
}

int HouMoGLMOCRModel::houmo_prefill(std::vector<llama_token> &tokens,
                                    int seq_id, float *logits) {
    LLAMA_LOG_INFO("%s seq_id = %d, tokens size is = %zu\n", "glmocr prefill",
                   seq_id, tokens.size());

    const int32_t kv_offset = memory_->seq_pos_min(seq_id);
    if ((offset_embedding_ids_[seq_id] + tokens.size() + kv_offset) >=
        (size_t)context_length_) {
        LLAMA_LOG_WARN("exceed context length, reset cached GLM-OCR inputs\n");
        reset_cached_inputs(seq_id);
        return 1;
    }

    if (!tokens.empty()) {
        if (kv_offset == 0 && offset_embedding_ids_[seq_id] == 0 &&
            prefill_chunks_[seq_id].empty()) {
            chunk_history_[seq_id].clear();
        }

        embedding_layer_->get_embedding_batch(
            tokens, input_embedding_ids_[seq_id].data() +
                        offset_embedding_ids_[seq_id] * n_embd_);

        PositionChunk chunk;
        chunk.n_tokens = (int)tokens.size();
        prefill_chunks_[seq_id].push_back(chunk);
        chunk_history_[seq_id].push_back(chunk);
        offset_embedding_ids_[seq_id] += tokens.size();
    }

    if (logits == nullptr && !tokens.empty()) {
        LLAMA_LOG_INFO("%s: logits is nullptr, only cache input. seq_id = %d\n",
                       __func__, seq_id);
        return 0;
    }

    const size_t input_echo_len = offset_embedding_ids_[seq_id];
    if (input_echo_len == 0) {
        return 0;
    }

    const size_t prefill_loop_round =
        (input_echo_len + prefill_length_ - 1) / prefill_length_;

    std::vector<int32_t> time_position_ids;
    std::vector<int32_t> height_position_ids;
    std::vector<int32_t> width_position_ids;
    compute_position_ids(seq_id, time_position_ids, height_position_ids,
                         width_position_ids);
    time_position_ids.resize(prefill_loop_round * prefill_length_,
                             time_position_ids.empty() ? 0 : time_position_ids.back());
    height_position_ids.resize(prefill_loop_round * prefill_length_,
                               height_position_ids.empty() ? 0 : height_position_ids.back());
    width_position_ids.resize(prefill_loop_round * prefill_length_,
                              width_position_ids.empty() ? 0 : width_position_ids.back());

    int valid_length = kv_offset;
    int current_length = 0;

    for (size_t round = 0; round < prefill_loop_round; round++) {
        current_length = (round == prefill_loop_round - 1)
                             ? (int)(input_echo_len - round * prefill_length_)
                             : prefill_length_;

        auto input_data = prefill_input_map_.at(prefill_model_->GetInputName(0));
        input_data.Buffer().CopyFromHost(
            input_embedding_ids_[seq_id].data() + round * prefill_length_ * n_embd_,
            input_data.MemSize());

        auto position_ids = prefill_input_map_.at(prefill_model_->GetInputName(1));
        set_position_input(position_ids, time_position_ids, height_position_ids,
                           width_position_ids, round * prefill_length_);

        auto prefill_valid_length =
            prefill_input_map_.at(prefill_model_->GetInputName(2));
        auto prefill_current_length =
            prefill_input_map_.at(prefill_model_->GetInputName(3));

        set_input_data(prefill_valid_length, n_prefill_batch_, valid_length);
        set_input_data(prefill_current_length, n_prefill_batch_, current_length);

        prefill_model_->SetInput(prefill_model_->GetInputName(0), input_data);
        prefill_model_->SetInput(prefill_model_->GetInputName(1), position_ids);
        prefill_model_->SetInput(prefill_model_->GetInputName(2),
                                 prefill_valid_length);
        prefill_model_->SetInput(prefill_model_->GetInputName(3),
                                 prefill_current_length);

        if (round == 0) {
            set_kv_cache_for_prefill(seq_id);
        }

        const auto status = prefill_model_->Run();
        if (status != tcim::Status::OK) {
            LLAMA_LOG_ERROR("%s: prefill_model_->Run() failed, result = %d\n",
                            __func__, static_cast<int>(status));
            reset_cached_inputs(seq_id);
            return -1;
        }
        prefill_model_->Sync();

        if (abort_callback_ != nullptr && abort_callback_(abort_callback_data_)) {
            LLAMA_LOG_INFO("%s: abort_callback_ is called. seq_id = %d\n",
                           __func__, seq_id);
            reset_cached_inputs(seq_id);
            return 2;
        }

        valid_length += prefill_length_;
    }

    decode_valid_length_[seq_id] = input_echo_len + kv_offset;
    memory_->seq_add(seq_id, 0, 0, input_echo_len);

    if (logits != nullptr) {
        tcim::Tensor prefill_output_data =
            prefill_model_->GetOutput(prefill_output_names_[0]);
        auto &prefill_output_float = prefill_output_map_[prefill_output_names_[0]];
        prefill_output_data.CastTo(prefill_output_float);
        memcpy(logits, prefill_output_float.Data(), prefill_output_float.MemSize());
    }

    reset_cached_inputs(seq_id);
    return 0;
}

int HouMoGLMOCRModel::houmo_prefill(const float *embeddings, int n_tokens,
                                    int seq_id, float *logits) {
    LLAMA_LOG_INFO("%s seq_id = %d, n_tokens is = %d\n", "glmocr image prefill",
                   seq_id, n_tokens);
    if (logits != nullptr) {
        LLAMA_LOG_ERROR("%s: logits is not null, not supported\n", __func__);
        reset_cached_inputs(seq_id);
        return -1;
    }

    if ((n_tokens + offset_embedding_ids_[seq_id]) >= context_length_) {
        LLAMA_LOG_WARN("exceed context length, reset cached GLM-OCR inputs\n");
        reset_cached_inputs(seq_id);
        return 1;
    }

    if (memory_->seq_pos_min(seq_id) == 0 && offset_embedding_ids_[seq_id] == 0 &&
        prefill_chunks_[seq_id].empty()) {
        chunk_history_[seq_id].clear();
    }

    GLMOCRCompactHeader header = {};
    int embedding_offset = 0;
    std::memcpy(&header, embeddings, sizeof(GLMOCRCompactHeader));
    PositionChunk chunk;
    chunk.n_tokens = n_tokens;
    if (header.magic == 0x1234) {
        chunk.nx = header.nx;
        chunk.ny = header.ny;
        chunk.image = chunk.nx > 0 && chunk.ny > 0;
        embedding_offset = sizeof(GLMOCRCompactHeader);
    }

    const int8_t *embeddings_ptr = reinterpret_cast<const int8_t *>(embeddings);
    const int data_size_bytes = n_tokens * n_embd_ * sizeof(int16_t);
    memcpy(input_embedding_ids_[seq_id].data() +
               offset_embedding_ids_[seq_id] * n_embd_,
           embeddings_ptr + embedding_offset, data_size_bytes);

    offset_embedding_ids_[seq_id] += n_tokens;
    prefill_chunks_[seq_id].push_back(chunk);
    chunk_history_[seq_id].push_back(chunk);
    return 0;
}

int HouMoGLMOCRModel::houmo_decode(std::vector<llama_token> &batches,
                                   std::vector<int> &seq_ids,
                                   std::vector<float *> logits) {
    LLAMA_LOG_DEBUG("%s: seq_ids size = %zu, batches size = %zu\n", __func__,
                    seq_ids.size(), batches.size());

    for (auto seq_id : seq_ids) {
        if (!prefill_chunks_[seq_id].empty()) {
            LLAMA_LOG_INFO("%s: seq_id = %d, process cached GLM-OCR input\n",
                           __func__, seq_id);
            std::vector<llama_token> empty_tokens;
            const int ret = houmo_prefill(empty_tokens, seq_id, nullptr);
            if (ret != 0) {
                return ret;
            }
        }

        const int target_tokens = memory_->seq_pos_min(seq_id);
        if (target_tokens < decode_valid_length_[seq_id]) {
            decode_valid_length_[seq_id] = target_tokens;
            position_indexs_[seq_id] = replay_position_until(seq_id, target_tokens);
            LLAMA_LOG_INFO(
                "%s: seq_id = %d, rolled back to valid_length = %d, position_index = %d\n",
                __func__, seq_id, target_tokens, position_indexs_[seq_id]);
        } else {
            decode_valid_length_[seq_id] = target_tokens;
        }
    }

    if (seq_ids.size() > (size_t)n_decode_batch_) {
        LLAMA_LOG_ERROR("%s: seq_ids.size() > n_decode_batch_ not supported\n",
                        __func__);
        return -1;
    }

    PERF_START(DECODE_SETINPUT);
    int ret = set_input_for_decode(batches, seq_ids);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: set_input_for_decode failed\n", __func__);
        return ret;
    }

    std::vector<int32_t> valid_length_vec(n_decode_batch_, 1);
    std::vector<int32_t> current_length_vec(n_decode_batch_, 1);
    for (size_t k = 0; k < seq_ids.size(); k++) {
        int batch_id = seq_ids[k];
        if (n_decode_batch_ == 1) {
            batch_id = 0;
        }
        valid_length_vec[batch_id] = decode_valid_length_[seq_ids[k]];
        if (valid_length_vec[batch_id] >= context_length_) {
            LLAMA_LOG_WARN("%s: valid_length = %d >= context_length_, end of decoding\n",
                           __func__, valid_length_vec[batch_id]);
            return 1;
        }
    }

    if (decode_valid_length_names_.size() == 1 &&
        decode_current_length_names_.size() == 1) {
        auto decode_valid_length =
            decode_input_map_.at(decode_valid_length_names_[0]);
        auto decode_current_length =
            decode_input_map_.at(decode_current_length_names_[0]);
        set_input_data(decode_valid_length, valid_length_vec);
        set_input_data(decode_current_length, current_length_vec);
        decode_model_->SetInput(decode_valid_length_names_[0],
                                decode_valid_length);
        decode_model_->SetInput(decode_current_length_names_[0],
                                decode_current_length);
    } else {
        if (decode_valid_length_names_.size() != (size_t)n_decode_batch_ ||
            decode_current_length_names_.size() != (size_t)n_decode_batch_) {
            LLAMA_LOG_ERROR(
                "%s: invalid GLM-OCR decode length inputs: valid=%zu, current=%zu, batch=%d\n",
                __func__, decode_valid_length_names_.size(),
                decode_current_length_names_.size(), n_decode_batch_);
            return -1;
        }

        for (int batch_id = 0; batch_id < n_decode_batch_; batch_id++) {
            const std::string &valid_name =
                decode_valid_length_names_[batch_id];
            auto valid_length = decode_input_map_.at(valid_name);
            set_input_data(valid_length, 1, valid_length_vec[batch_id]);
            decode_model_->SetInput(valid_name, valid_length);

            const std::string &current_name =
                decode_current_length_names_[batch_id];
            auto current_length = decode_input_map_.at(current_name);
            set_input_data(current_length, 1, current_length_vec[batch_id]);
            decode_model_->SetInput(current_name, current_length);
        }
    }

    std::vector<int32_t> position_vec(n_decode_batch_, 1);
    for (size_t k = 0; k < seq_ids.size(); k++) {
        int batch_id = seq_ids[k];
        if (n_decode_batch_ == 1) {
            batch_id = 0;
        }
        position_vec[batch_id] = position_indexs_[seq_ids[k]];
    }

    const std::string position_name = decode_model_->GetInputName(1);
    const bool split_position_inputs =
        n_decode_batch_ > 1 &&
        decode_input_map_.count(position_name + "_batch1") > 0;
    if (split_position_inputs) {
        for (int batch_id = 0; batch_id < n_decode_batch_; batch_id++) {
            std::string name = position_name;
            if (batch_id > 0) {
                name += "_batch" + std::to_string(batch_id);
            }
            auto position_ids = decode_input_map_.at(name);
            std::vector<int32_t> position = { position_vec[batch_id] };
            set_decode_position_input(position_ids, position);
            decode_model_->SetInput(name, position_ids);
        }
    } else {
        auto position_ids = decode_input_map_.at(position_name);
        set_decode_position_input(position_ids, position_vec);
        decode_model_->SetInput(position_name, position_ids);
    }

    set_kv_cache_for_decode(seq_ids);
    PERF_STOP(DECODE_SETINPUT);

    PERF_START(DECODE_RUN);
    const auto status = decode_model_->Run();
    if (status != tcim::Status::OK) {
        PERF_STOP(DECODE_RUN);
        LLAMA_LOG_ERROR("%s: decode_model_->Run() failed, result = %d\n",
                        __func__, static_cast<int>(status));
        return -1;
    }
    decode_model_->Sync();
    PERF_STOP(DECODE_RUN);

    PERF_START(DECODE_GETOUTPUT);
    tcim::Tensor decode_data = decode_model_->GetOutput(decode_output_names_[0]);
    auto &decode_output_float = decode_output_map_[decode_output_names_[0]];
    decode_data.CastTo(decode_output_float);
    auto n_vocab =
        decode_output_float.MemSize() / (n_decode_batch_ * sizeof(float));
    PERF_STOP(DECODE_GETOUTPUT);

    for (size_t i = 0; i < seq_ids.size(); i++) {
        auto batch_id = seq_ids[i];
        decode_valid_length_[batch_id]++;
        position_indexs_[batch_id]++;
        if (n_decode_batch_ == 1) {
            batch_id = 0;
        }
        float *logits_out = logits[i];
        float *output_ptr = static_cast<float *>(decode_output_float.Data()) +
                            batch_id * n_vocab;
        if (logits_out != nullptr) {
            memcpy(logits_out, output_ptr, n_vocab * sizeof(float));
        }
        memory_->seq_add(seq_ids[i], 0, 0, 1);
    }
    return 0;
}

void HouMoGLMOCRModel::append_positions(
    const PositionChunk &chunk, std::vector<int32_t> &time_position_ids,
    std::vector<int32_t> &height_position_ids,
    std::vector<int32_t> &width_position_ids, int32_t &start_pos) const {
    if (chunk.image && chunk.nx > 0 && chunk.ny > 0 &&
        chunk.n_tokens == chunk.nx * chunk.ny) {
        for (int y = 0; y < chunk.ny; y++) {
            for (int x = 0; x < chunk.nx; x++) {
                time_position_ids.push_back(start_pos);
                height_position_ids.push_back(start_pos + y);
                width_position_ids.push_back(start_pos + x);
            }
        }
        start_pos += std::max(chunk.nx, chunk.ny);
    } else {
        for (int i = 0; i < chunk.n_tokens; i++) {
            time_position_ids.push_back(start_pos + i);
            height_position_ids.push_back(start_pos + i);
            width_position_ids.push_back(start_pos + i);
        }
        start_pos += chunk.n_tokens;
    }
}

void HouMoGLMOCRModel::compute_position_ids(
    int seq_id, std::vector<int32_t> &time_position_ids,
    std::vector<int32_t> &height_position_ids,
    std::vector<int32_t> &width_position_ids) {
    position_indexs_[seq_id] = memory_->seq_pos_min(seq_id);
    for (const auto &chunk : prefill_chunks_[seq_id]) {
        append_positions(chunk, time_position_ids, height_position_ids,
                         width_position_ids, position_indexs_[seq_id]);
    }
}

int32_t HouMoGLMOCRModel::replay_position_until(int seq_id,
                                                int target_tokens) const {
    int current_tokens = 0;
    int32_t current_pos = 0;
    for (const auto &chunk : chunk_history_[seq_id]) {
        if (current_tokens >= target_tokens) {
            break;
        }
        if (chunk.image && chunk.nx > 0 && chunk.ny > 0 &&
            chunk.n_tokens == chunk.nx * chunk.ny) {
            if (target_tokens - current_tokens >= chunk.n_tokens) {
                current_tokens += chunk.n_tokens;
                current_pos += std::max(chunk.nx, chunk.ny);
            } else {
                current_pos += target_tokens - current_tokens;
                current_tokens = target_tokens;
            }
        } else {
            const int step = std::min(chunk.n_tokens, target_tokens - current_tokens);
            current_tokens += step;
            current_pos += step;
        }
    }
    if (target_tokens > current_tokens) {
        current_pos += target_tokens - current_tokens;
    }
    return current_pos;
}

void HouMoGLMOCRModel::set_position_input(
    tcim::Tensor &tensor, const std::vector<int32_t> &time_position_ids,
    const std::vector<int32_t> &height_position_ids,
    const std::vector<int32_t> &width_position_ids, size_t offset) {
    const size_t n = tensor.Info().Shape().back();
    if (tensor.Info().DataType() == tcim::DataType::INT32) {
        std::vector<int32_t> data(n * 3, 0);
        for (size_t i = 0; i < n; i++) {
            data[i] = time_position_ids[offset + i];
            data[n + i] = height_position_ids[offset + i];
            data[2 * n + i] = width_position_ids[offset + i];
        }
        tensor.Buffer().CopyFromHost(data.data(), tensor.MemSize());
        return;
    }

    std::vector<float> data(n * 3, 0.0f);
    for (size_t i = 0; i < n; i++) {
        data[i] = (float)time_position_ids[offset + i];
        data[n + i] = (float)height_position_ids[offset + i];
        data[2 * n + i] = (float)width_position_ids[offset + i];
    }
    auto fp32_tensor = tcim::Tensor::CreateHostTensor(
        tensor.Info().AsContiguous().AsType(tcim::DataType::FLOAT32));
    fp32_tensor.Buffer().CopyFromHost(data.data(), fp32_tensor.MemSize());
    fp32_tensor.CastTo(tensor);
}

void HouMoGLMOCRModel::set_decode_position_input(
    tcim::Tensor &tensor, const std::vector<int32_t> &position_vec) {
    const size_t batch = position_vec.size();
    if (tensor.Info().DataType() == tcim::DataType::INT32) {
        std::vector<int32_t> data(batch * 3, 0);
        for (size_t i = 0; i < batch; i++) {
            data[i] = position_vec[i];
            data[batch + i] = position_vec[i];
            data[2 * batch + i] = position_vec[i];
        }
        tensor.Buffer().CopyFromHost(data.data(), tensor.MemSize());
        return;
    }

    std::vector<float> data(batch * 3, 0.0f);
    for (size_t i = 0; i < batch; i++) {
        data[i] = (float)position_vec[i];
        data[batch + i] = (float)position_vec[i];
        data[2 * batch + i] = (float)position_vec[i];
    }
    auto fp32_tensor = tcim::Tensor::CreateHostTensor(
        tensor.Info().AsContiguous().AsType(tcim::DataType::FLOAT32));
    fp32_tensor.Buffer().CopyFromHost(data.data(), fp32_tensor.MemSize());
    fp32_tensor.CastTo(tensor);
}

void HouMoGLMOCRModel::reset_cached_inputs(int seq_id) {
    offset_embedding_ids_[seq_id] = 0;
    prefill_chunks_[seq_id].clear();
}
