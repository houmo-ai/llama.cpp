#include "houmo-qwen35llm.h"

#include <algorithm>
#include <cstring>
#include "ggml.h"
#include "llama-impl.h"
#include "time-perf.h"

namespace {
std::string find_input_name(const std::map<std::string, tcim::Tensor> &inputs,
                            const std::string &key) {
    for (const auto &pair : inputs) {
        if (pair.first.find(key) != std::string::npos) {
            return pair.first;
        }
    }
    return "";
}

std::string find_input_name_any(const std::map<std::string, tcim::Tensor> &inputs,
                                const std::vector<std::string> &keys) {
    for (const auto &key : keys) {
        auto name = find_input_name(inputs, key);
        if (!name.empty()) {
            return name;
        }
    }
    return "";
}

int extract_batch_id(const std::string &name) {
    auto pos = name.find("_batch");
    if (pos == std::string::npos) {
        return -1;
    }
    auto start = pos + std::strlen("_batch");
    if (start >= name.size()) {
        return -1;
    }
    try {
        return std::stoi(name.substr(start));
    } catch (...) {
        return -1;
    }
}

std::vector<std::string> collect_input_names(
    const std::map<std::string, tcim::Tensor> &inputs,
    const std::string &key) {
    std::vector<std::string> names;
    for (const auto &pair : inputs) {
        if (pair.first.find(key) != std::string::npos) {
            names.push_back(pair.first);
        }
    }
    std::sort(names.begin(), names.end(), [](const std::string &a, const std::string &b) {
        int batch_a = extract_batch_id(a);
        int batch_b = extract_batch_id(b);
        if (batch_a >= 0 && batch_b >= 0) {
            return batch_a < batch_b;
        }
        return a < b;
    });
    return names;
}

std::vector<std::string> collect_input_names_any(
    const std::map<std::string, tcim::Tensor> &inputs,
    const std::vector<std::string> &keys) {
    std::vector<std::string> names;
    for (const auto &key : keys) {
        auto partial = collect_input_names(inputs, key);
        names.insert(names.end(), partial.begin(), partial.end());
    }
    std::sort(names.begin(), names.end(), [](const std::string &a, const std::string &b) {
        int batch_a = extract_batch_id(a);
        int batch_b = extract_batch_id(b);
        if (batch_a >= 0 && batch_b >= 0) {
            return batch_a < batch_b;
        }
        return a < b;
    });
    names.erase(std::unique(names.begin(), names.end()), names.end());
    return names;
}

std::string resolve_batch_name(const std::vector<std::string> &names,
                               int batch_id) {
    if (names.empty()) {
        return "";
    }
    if (names.size() == 1) {
        return names.front();
    }
    for (const auto &name : names) {
        if (extract_batch_id(name) == batch_id) {
            return name;
        }
    }
    if (batch_id >= 0 && static_cast<size_t>(batch_id) < names.size()) {
        return names[batch_id];
    }
    return names.front();
}

void zero_tensor(tcim::Tensor &tensor) {
    size_t elem_count = tensor.MemSize();
    switch (tensor.Info().DataType()) {
    case tcim::DataType::FLOAT16: {
        elem_count /= sizeof(ggml_fp16_t);
        std::vector<ggml_fp16_t> zeros(elem_count, ggml_fp32_to_fp16(0.0f));
        tensor.Buffer().CopyFromHost(zeros.data(), tensor.MemSize());
        break;
    }
    case tcim::DataType::FLOAT32: {
        elem_count /= sizeof(float);
        std::vector<float> zeros(elem_count, 0.0f);
        tensor.Buffer().CopyFromHost(zeros.data(), tensor.MemSize());
        break;
    }
    case tcim::DataType::INT16: {
        elem_count /= sizeof(int16_t);
        std::vector<int16_t> zeros(elem_count, 0);
        tensor.Buffer().CopyFromHost(zeros.data(), tensor.MemSize());
        break;
    }
    case tcim::DataType::INT32: {
        elem_count /= sizeof(int32_t);
        std::vector<int32_t> zeros(elem_count, 0);
        tensor.Buffer().CopyFromHost(zeros.data(), tensor.MemSize());
        break;
    }
    default:
        break;
    }
}

bool is_cache_input_name(const std::string &name) {
    return name.find("past_key_cache_") != std::string::npos ||
           name.find("_self_attn_vcache") != std::string::npos ||
           name.find("_self_attn_kcache") != std::string::npos ||
           name.find("past_conv_cache") != std::string::npos ||
           name.find("past_recurrent_state") != std::string::npos;
}

} // namespace

void HoumoQwen35LLM::houmo_init(houmo_memory_i *memory, int seq_max,
                                ggml_abort_callback abort_callback,
                                void *abort_callback_data) {
    abort_callback_data_ = abort_callback_data;
    abort_callback_ = abort_callback;
    model_inout_init(seq_max);
    if (is_embedding()) {
        memory_ = nullptr;
    } else {
        memory_ = memory;
    }
    resolve_input_names();
    init_cache_io_tensors();
    init_state_cache(seq_max);
}

void HoumoQwen35LLM::resolve_input_names() {
    prefill_time_pos_name_ = prefill_model_->GetInputName(1);
    prefill_height_pos_name_ = prefill_model_->GetInputName(2);
    prefill_width_pos_name_ = prefill_model_->GetInputName(3);
    prefill_valid_length_name_ = prefill_model_->GetInputName(4);
    prefill_current_length_name_ = prefill_model_->GetInputName(5);
    prefill_linear_mask_name_ = prefill_model_->GetInputName(6);

    decode_time_pos_names_.clear();
    decode_height_pos_names_.clear();
    decode_width_pos_names_.clear();
    decode_linear_mask_names_.clear();
    decode_valid_length_names_.clear();
    decode_current_length_names_.clear();

    decode_time_pos_names_.push_back(decode_model_->GetInputName(1));
    decode_height_pos_names_.push_back(decode_model_->GetInputName(2));
    decode_width_pos_names_.push_back(decode_model_->GetInputName(3));
    decode_valid_length_names_.push_back(decode_model_->GetInputName(4));
    decode_current_length_names_.push_back(decode_model_->GetInputName(5));
    decode_linear_mask_names_.push_back(decode_model_->GetInputName(6));
}

void HoumoQwen35LLM::init_cache_io_tensors() {
    if (prefill_model_ == nullptr || decode_model_ == nullptr) {
        return;
    }
    for (size_t i = 0; i < prefill_model_->GetInputNum(); ++i) {
        auto input_name = prefill_model_->GetInputName(i);
        if (input_name.find("conv_cache") != std::string::npos) {
            auto cache = prefill_model_->GetDevInput(input_name);
            auto tensor = tcim::Tensor::CreateHostTensor(cache.Info());
            zero_tensor(tensor);
            prefill_model_->SetInput(input_name, tensor);
        }
        if (input_name.find("recurrent_state") != std::string::npos) {
            auto cache = prefill_model_->GetDevInput(input_name);
            auto tensor = tcim::Tensor::CreateHostTensor(cache.Info());
            zero_tensor(tensor);
            prefill_model_->SetInput(input_name, tensor);
        }
    }
}

void HoumoQwen35LLM::init_state_cache(int seq_max) {
    state_cache_names_.clear();
    state_cache_output_names_.clear();
    state_cache_.clear();
    state_cache_initialized_.clear();
    if (decode_model_ == nullptr) {
        return;
    }
    auto add_name = [&](const std::string &name, const std::string &output_name) {
        auto iter = std::find(state_cache_names_.begin(), state_cache_names_.end(), name);
        if (iter == state_cache_names_.end()) {
            state_cache_names_.push_back(name);
            state_cache_output_names_.push_back(output_name);
        }
    };
    for (size_t idx = 0; idx < decode_model_->GetInputNum(); ++idx) {
        auto input_name = decode_model_->GetInputName(idx);
        if (input_name.find("model_layers") != std::string::npos ||
            input_name.find("conv_cache") != std::string::npos ||
            input_name.find("recurrent_state") != std::string::npos) {
            std::string output_name;
            if (input_name.find("conv_cache") != std::string::npos) {
                output_name = input_name;
                auto pos = output_name.find("past_conv_cache_");
                if (pos != std::string::npos) {
                    output_name.replace(pos, std::strlen("past_conv_cache_"),
                                        "conv_cache_out_");
                }
            } else if (input_name.find("recurrent_state") != std::string::npos) {
                output_name = input_name;
                auto pos = output_name.find("past_recurrent_state_");
                if (pos != std::string::npos) {
                    output_name.replace(pos, std::strlen("past_recurrent_state_"),
                                        "recurrent_state_out_");
                }
            }
            add_name(input_name, output_name);
        }
    }
    if (state_cache_names_.empty()) {
        return;
    }
    int size = (seq_max + n_prefill_batch_ - 1) / n_prefill_batch_;
    size = std::max(size, n_decode_batch_);
    for (int i = 0; i < size; ++i) {
        StateCacheEntry entry;
        entry.tensors.reserve(state_cache_names_.size());
        for (const auto &name : state_cache_names_) {
            auto cache = decode_model_->GetDevInput(name);
            if (i > 0) {
                cache = cache.Clone();
            }
            entry.tensors.push_back(cache);
        }
        state_cache_[i] = std::move(entry);
        state_cache_initialized_[i] = false;
    }
}

void HoumoQwen35LLM::set_state_cache_for_prefill(int seq_id) {
    if (prefill_model_ == nullptr) {
        return;
    }
    auto iter = state_cache_.find(seq_id);
    if (iter == state_cache_.end()) {
        return;
    }
    auto &tensors = iter->second.tensors;
    if (!state_cache_initialized_[seq_id]) {
        for (size_t idx = 0; idx < tensors.size(); ++idx) {
            auto host_tensor = tcim::Tensor::CreateHostTensor(tensors[idx].Info());
            zero_tensor(host_tensor);
            tensors[idx].Buffer().CopyFromHost(host_tensor.Data(), host_tensor.MemSize());
        }
        state_cache_initialized_[seq_id] = true;
    }
    for (size_t idx = 0; idx < tensors.size(); ++idx) {
        prefill_model_->SetDevInput(state_cache_names_[idx], tensors[idx]);
        if (idx < state_cache_output_names_.size() &&
            !state_cache_output_names_[idx].empty()) {
            prefill_model_->SetDevOutput(state_cache_output_names_[idx],
                                         tensors[idx]);
        }
    }
}

void HoumoQwen35LLM::set_state_cache_for_decode(
    const std::vector<int> &seq_ids) {
    if (decode_model_ == nullptr) {
        return;
    }
    for (auto seq_id : seq_ids) {
        auto iter = state_cache_.find(seq_id);
        if (iter == state_cache_.end()) {
            continue;
        }
        auto &tensors = iter->second.tensors;
        for (size_t idx = 0; idx < tensors.size(); ++idx) {
            decode_model_->SetDevInput(state_cache_names_[idx], tensors[idx]);
            if (idx < state_cache_output_names_.size() &&
                !state_cache_output_names_[idx].empty()) {
                decode_model_->SetDevOutput(state_cache_output_names_[idx],
                                            tensors[idx]);
            }
        }
    }
}

std::vector<int32_t> HoumoQwen35LLM::build_rope_position_ids(int valid_length,
                                                             int length) const {
    std::vector<int32_t> position_ids(static_cast<size_t>(length), 0);
    for (int i = 0; i < length; ++i) {
        position_ids[static_cast<size_t>(i)] = valid_length + i;
    }
    return position_ids;
}

void HoumoQwen35LLM::set_linear_attn_mask(tcim::Tensor &tensor, int fill_length,
                                          int new_cache_length) const {
    int batch_size = 1;
    auto shape = tensor.Info().Shape();
    if (!shape.empty()) {
        batch_size = static_cast<int>(shape[0]);
    }
    if (fill_length <= 0 || new_cache_length <= 0) {
        zero_tensor(tensor);
        return;
    }

    if (tensor.Info().DataType() == tcim::DataType::FLOAT16) {
        size_t elem_count = tensor.MemSize() / sizeof(ggml_fp16_t);
        std::vector<ggml_fp16_t> data(elem_count, ggml_fp32_to_fp16(0.0f));
        size_t stride = elem_count / static_cast<size_t>(batch_size);
        ggml_fp16_t one = ggml_fp32_to_fp16(1.0f);
        for (int b = 0; b < batch_size; ++b) {
            size_t offset = static_cast<size_t>(b) * stride;
            size_t limit = std::min(static_cast<size_t>(new_cache_length), stride);
            for (size_t i = 0; i < limit; ++i) {
                data[offset + i] = one;
            }
        }
        tensor.Buffer().CopyFromHost(data.data(), tensor.MemSize());
    } else if (tensor.Info().DataType() == tcim::DataType::FLOAT32) {
        size_t elem_count = tensor.MemSize() / sizeof(float);
        std::vector<float> data(elem_count, 0.0f);
        size_t stride = elem_count / static_cast<size_t>(batch_size);
        for (int b = 0; b < batch_size; ++b) {
            size_t offset = static_cast<size_t>(b) * stride;
            size_t limit = std::min(static_cast<size_t>(new_cache_length), stride);
            for (size_t i = 0; i < limit; ++i) {
                data[offset + i] = 1.0f;
            }
        }
        tensor.Buffer().CopyFromHost(data.data(), tensor.MemSize());
    } else {
        zero_tensor(tensor);
    }
}

int HoumoQwen35LLM::houmo_prefill(std::vector<llama_token> &tokens, int seq_id,
                                 float *logits) {
    LLAMA_LOG_INFO("%s seq_id = %d, tokens size is = %zu\n", "prefill stage",
                   seq_id, tokens.size());
    int32_t kv_offset = memory_ ? memory_->seq_pos_min(seq_id) : 0;
    if (kv_offset == 0 && state_cache_initialized_.count(seq_id) > 0) {
        state_cache_initialized_[seq_id] = false;
    }
    if ((tokens.size() + kv_offset) >=
        static_cast<uint32_t>(context_length_)) {
        LLAMA_LOG_ERROR(
            "%s: input size %zu > context_length_%d, reset to 0\n",
            __func__, tokens.size() + kv_offset,
            context_length_);
        return 1;
    }

    size_t process_tokens = tokens.size();
    if (process_tokens == 0) {
        LLAMA_LOG_INFO(
            "%s: no tokens to process, just cache return, total size = %zu\n",
            __func__, process_tokens);
        return 0;
    }

    int32_t current_length = 0;
    int32_t valid_length = kv_offset;

    size_t rounds = (process_tokens + prefill_length_ - 1) / prefill_length_;

    for (size_t round = 0; round < rounds; ++round) {
        auto &input_data = prefill_input_map_.at(prefill_model_->GetInputName(0));

        size_t round_start = round * prefill_length_;
        size_t remaining = process_tokens > round_start ? process_tokens - round_start : 0;
        current_length = remaining >= static_cast<size_t>(prefill_length_)
                             ? prefill_length_
                             : static_cast<int>(remaining);

        const int32_t *token_ptr = tokens.data() + round_start;
        memset(input_data.Data(), 0, input_data.MemSize());
        embedding_layer_->get_embedding_batch(token_ptr, current_length,
                                             input_data.Data());

        if (!prefill_valid_length_name_.empty()) {
            auto prefill_valid_length = prefill_input_map_.at(prefill_valid_length_name_);
            set_input_data(prefill_valid_length, n_prefill_batch_, valid_length);
            prefill_model_->SetInput(prefill_valid_length_name_, prefill_valid_length);
        }
        if (!prefill_current_length_name_.empty()) {
            auto prefill_current_length = prefill_input_map_.at(prefill_current_length_name_);
            set_input_data(prefill_current_length, n_prefill_batch_, current_length);
            prefill_model_->SetInput(prefill_current_length_name_, prefill_current_length);
        }

        auto position_ids = build_rope_position_ids(valid_length, prefill_length_);
        if (!prefill_time_pos_name_.empty()) {
            auto time_pos = prefill_input_map_.at(prefill_time_pos_name_);
            set_input_data(time_pos, position_ids);
            prefill_model_->SetInput(prefill_time_pos_name_, time_pos);
        }
        if (!prefill_height_pos_name_.empty()) {
            auto height_pos = prefill_input_map_.at(prefill_height_pos_name_);
            set_input_data(height_pos, position_ids);
            prefill_model_->SetInput(prefill_height_pos_name_, height_pos);
        }
        if (!prefill_width_pos_name_.empty()) {
            auto width_pos = prefill_input_map_.at(prefill_width_pos_name_);
            set_input_data(width_pos, position_ids);
            prefill_model_->SetInput(prefill_width_pos_name_, width_pos);
        }
        if (!prefill_linear_mask_name_.empty()) {
            auto linear_mask = prefill_input_map_.at(prefill_linear_mask_name_);
            set_linear_attn_mask(linear_mask, prefill_length_, current_length);
            prefill_model_->SetInput(prefill_linear_mask_name_, linear_mask);
        }

        PERF_START(PREFILL_SETINPUT_ONCE);
        for (const auto &pair : lora_map) {
            if (prefill_input_map_.count(pair.first) > 0) {
                auto lora_mask = prefill_input_map_[pair.first];
                lora_mask_float_.Buffer().CopyFromHost(&pair.second.scale,
                                                       lora_mask_float_.MemSize());
                lora_mask_float_.CastTo(lora_mask);
                prefill_model_->SetInput(pair.first, lora_mask);
            }
        }
        prefill_model_->SetInput(prefill_model_->GetInputName(0), input_data);
        set_state_cache_for_prefill(seq_id);
        set_kv_cache_for_prefill(seq_id);
        PERF_STOP(PREFILL_SETINPUT_ONCE);

        PERF_START(PREFILL_RUN_ONCE);
        prefill_model_->Run();
        prefill_model_->Sync();
        PERF_STOP(PREFILL_RUN_ONCE);
        valid_length += current_length;
    }

    decode_valid_length_[seq_id] = kv_offset + static_cast<int32_t>(process_tokens);
    PERF_START(PREFILL_GETOUTPUT);
    tcim::Tensor prefill_output_data =
        prefill_model_->GetOutput(prefill_output_names_[0]);
    auto &prefill_output_float = prefill_output_map_[prefill_output_names_[0]];
    prefill_output_data.CastTo(prefill_output_float);
    PERF_STOP(PREFILL_GETOUTPUT);
    if (logits != nullptr) {
        memcpy(logits, prefill_output_float.Data(), prefill_output_float.MemSize());
    }

    if (memory_) {
        memory_->seq_add(seq_id, 0, 0, static_cast<int32_t>(process_tokens));
    }
    return 0;
}

int HoumoQwen35LLM::houmo_prefill(const float *embeddings, int n_tokens,
                                 int seq_id, float *logits) {
    (void)embeddings;
    (void)n_tokens;
    (void)seq_id;
    (void)logits;
    return 0;
}

int HoumoQwen35LLM::set_decode_rope_and_mask(const std::vector<int> &seq_ids) {
    for (size_t i = 0; i < seq_ids.size(); i++) {
        int seq_id = seq_ids[i];
        int batch_id = seq_id;
        if (n_decode_batch_ == 1) {
            batch_id = 0;
        }
        int valid_length = decode_valid_length_[seq_id];
        auto position_ids = build_rope_position_ids(valid_length, 1);
        auto time_name = resolve_batch_name(decode_time_pos_names_, batch_id);
        auto height_name = resolve_batch_name(decode_height_pos_names_, batch_id);
        auto width_name = resolve_batch_name(decode_width_pos_names_, batch_id);
        auto mask_name = resolve_batch_name(decode_linear_mask_names_, batch_id);

        if (!time_name.empty()) {
            auto tensor = decode_input_map_.at(time_name);
            set_input_data(tensor, position_ids);
            decode_model_->SetInput(time_name, tensor);
        }
        if (!height_name.empty()) {
            auto tensor = decode_input_map_.at(height_name);
            set_input_data(tensor, position_ids);
            decode_model_->SetInput(height_name, tensor);
        }
        if (!width_name.empty()) {
            auto tensor = decode_input_map_.at(width_name);
            set_input_data(tensor, position_ids);
            decode_model_->SetInput(width_name, tensor);
        }
        if (!mask_name.empty()) {
            auto tensor = decode_input_map_.at(mask_name);
            set_linear_attn_mask(tensor, 1, 1);
            decode_model_->SetInput(mask_name, tensor);
        }
    }
    return 0;
}

int HoumoQwen35LLM::houmo_decode(std::vector<llama_token> &batches,
                                std::vector<int> &seq_ids,
                                std::vector<float *> logits) {
    LLAMA_LOG_DEBUG("%s: seq_ids size = %zu, batches size = %zu\n", __func__,
                    seq_ids.size(), batches.size());
    for (auto seq_id : seq_ids) {
        if (memory_) {
            decode_valid_length_[seq_id] = memory_->seq_pos_min(seq_id);
        }
    }
    if (seq_ids.size() > static_cast<size_t>(n_decode_batch_)) {
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
    ret = set_xlenght_for_decode(seq_ids);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: set_xlenght_for_decode failed\n", __func__);
        return ret;
    }
    ret = set_decode_rope_and_mask(seq_ids);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: set_decode_rope_and_mask failed\n", __func__);
        return ret;
    }
    set_state_cache_for_decode(seq_ids);
    set_kv_cache_for_decode(seq_ids);

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

    PERF_START(DECODE_RUN);
    decode_model_->Run();
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
        if (n_decode_batch_ == 1) {
            batch_id = 0;
        }
        float *logits_out = logits[i];
        float *output_ptr = static_cast<float *>(decode_output_float.Data()) +
                            batch_id * n_vocab;
        memcpy(logits_out, output_ptr, n_vocab * sizeof(float));
        if (memory_) {
            memory_->seq_add(seq_ids[i], 0, 0, 1);
        }
    }
    return 0;
}

int HoumoQwen35LLM::houmo_embedding(
    const std::map<int32_t, std::vector<llama_token>> &batches,
    std::map<int, float *> &embeddings) {
    for (auto &[seq_id, tokens] : batches) {
        if (tokens.empty()) {
            continue;
        }
        auto &embedding = embeddings[seq_id];
        if (embedding == nullptr) {
            LLAMA_LOG_ERROR("%s: null embedding pointer for seq_id %d\n", __func__,
                            seq_id);
            return -1;
        }
        int ret = houmo_prefill(const_cast<std::vector<llama_token> &>(tokens),
                                seq_id, embedding);
        if (ret != 0) {
            LLAMA_LOG_ERROR(
                "%s: houmo_prefill failed for seq_id %d with error code %d\n",
                __func__, seq_id, ret);
            return -1;
        }
    }
    return 0;
}

bool HoumoQwen35LLM::is_embedding() {
    return decode_model_ == nullptr;
}
