#include "houmo-llmodel.h"
#include "houmo-vllmodel.h"
#include "ggml.h"
#include "ggml-hmutils.h"
#include "gguf.h"
#include "llama-batch.h"
#include "llama-impl.h"
#include "tcim/tcim_runtime.h"
#include "torch/torch.h"
#include <fstream>
#include <map>
#include <regex>
#include <string>
#include <vector>
static std::vector<int> parse_deviceids() {
    std::vector<int> deviceIds;
    const char *device_id_str = std::getenv("HOUMO_DEVICE_ID");

    if (device_id_str == nullptr) {
        deviceIds.push_back(0);
        deviceIds.push_back(1);
        return deviceIds;
    }

    std::string deviceList(device_id_str);
    std::istringstream iss(deviceList);
    std::string token;

    while (std::getline(iss, token, ',')) {
        try {
            // 转换为整数并添加到列表
            deviceIds.push_back(std::stoi(token));
        } catch (const std::invalid_argument &) {
            std::cerr << "Invalid device ID: " << token << std::endl;
        } catch (const std::out_of_range &) {
            std::cerr << "Device ID out of range: " << token << std::endl;
        }
    }

    return deviceIds;
}
/**
 * @brief HouMoSingleCardModel is the derived class for single-card HouMo model.
 * @details 试用一张卡可以加载整个模型，不需要拆分模型。
 */
class HouMoSingleCardModel : public HouMoLLModel::Impl {
  public:
    HouMoSingleCardModel() = default;
    ~HouMoSingleCardModel() override = default;

    bool houmo_load(const std::string &file_name) override;
    void houmo_init(int seq_max = 4) override;
    int houmo_prefill(std::vector<llama_token> &tokens, int seq_id,
                      float *logits) override;
    int houmo_decode(std::vector<llama_token> &batches,
                     std::vector<int> &seq_ids,
                     std::vector<float *> logits) override;

  private:
    int kSeqMax = 4; // maximum sequence number for single-card model
    std::shared_ptr<tcim::Module> prefill_model_;
    std::shared_ptr<tcim::Module> decode_model_;
    std::vector<llama_token> prefill_tokens_;
};

/**
 * @brief HouMoDualCardModel is the derived class for dual-card HouMo model.
 * @details for 14B+ model, 拆分模型到两张卡.
 */
class HouMoDualCardModel : public HouMoLLModel::Impl {
  public:
    HouMoDualCardModel() = default;
    ~HouMoDualCardModel() override = default;

    bool houmo_load(const std::string &file_name) override;
    void houmo_init(int seq_max = 4) override;
    int houmo_prefill(std::vector<llama_token> &tokens, int seq_id,
                      float *logits) override;
    int houmo_decode(std::vector<llama_token> &batches,
                     std::vector<int> &seq_ids,
                     std::vector<float *> logits) override;

  private:
    std::shared_ptr<tcim::Module> prefill_model_card0_;
    std::shared_ptr<tcim::Module> prefill_model_card1_;
    std::shared_ptr<tcim::Module> decode_model_card0_;
    std::shared_ptr<tcim::Module> decode_model_card1_;
    std::vector<std::string> decode_inner_output_names_;
    std::vector<std::string> prefill_inner_output_names_;
    int kSeqMax = 2;        // maximum sequence number for dual-card model
    size_t split_size_ = 0; // split size for kv cache
};

void HouMoLLModel::Impl::prefill_input_init(
    std::shared_ptr<tcim::Module> prefill_model,
    std::map<std::string, tcim::Tensor> &prefill_input_map) {
    size_t input_num = prefill_model->GetInputNum();
    for (size_t idx = 0; idx < input_num; idx++) {
        auto input_name = prefill_model->GetInputName(idx);
        LLAMA_LOG_DEBUG("%s: prefill input_name: %s\n", __func__,
                        input_name.c_str());
    }
    // only for text model
    input_num = 3;
    if (model_type_ == 1) {
        input_num = 6;
    }
    for (size_t idx = 0; idx < input_num; idx++) {
        auto input_name = prefill_model->GetInputName(idx);
        LLAMA_LOG_INFO("%s: prefill input_name: %s\n", __func__,
                       input_name.c_str());
        auto input_info =
            prefill_model->GetInputInfo(input_name).AsContiguous();
        auto shape = input_info.Shape();
        for (size_t i = 0; i < shape.size(); i++) {
            LLAMA_LOG_DEBUG("%s: prefill input_shape: %ld, type = %d\n", __func__,
                           shape[i], (int)input_info.DataType());
        }
        auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
        prefill_input_map.insert(
            std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
    }
    // 填充prefill_length_ & n_prefill_batch_
    if (model_type_ == 1) {
        n_prefill_batch_ = prefill_input_map_[prefill_model->GetInputName(4)]
                               .Info()
                               .Shape()[0];
    } else {
        n_prefill_batch_ = prefill_input_map_[prefill_model->GetInputName(1)]
                               .Info()
                               .Shape()[0];
    }
    prefill_length_ =
        (prefill_input_map_[prefill_model->GetInputName(0)].Info().Shape()[0] *
         prefill_input_map_[prefill_model->GetInputName(0)].Info().Shape()[1]) /
        n_prefill_batch_;
    xh1_ = !(
        prefill_input_map_[prefill_model->GetInputName(0)].Info().DataType() ==
        tcim::DataType::FLOAT16);
}

void HouMoLLModel::Impl::prefill_output_init(
    std::shared_ptr<tcim::Module> prefill_model,
    std::map<std::string, tcim::Tensor> &prefill_output_map,
    std::vector<std::string> &prefill_output_names) {
    int output_num = prefill_model->GetOutputNum();
    for (int idx = 0; idx < output_num; idx++) {
        auto output_name = prefill_model->GetOutputName(idx);
        LLAMA_LOG_INFO("%s: prefill output_name: %s\n", __func__,
                       output_name.c_str());
        auto output_info = prefill_model->GetOutputInfo(output_name)
                               .AsContiguous()
                               .AsType(tcim::DataType::FLOAT32);
        auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
        prefill_output_map.insert(
            std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
        prefill_output_names.push_back(output_name);
    }
}

void HouMoLLModel::Impl::decode_input_init(
    std::shared_ptr<tcim::Module> decode_model,
    std::map<std::string, tcim::Tensor> &decode_input_map) {
    size_t input_num_decode = decode_model->GetInputNum();
    for (size_t idx = 0; idx < input_num_decode; idx++) {
        auto input_name = decode_model->GetInputName(idx);
        LLAMA_LOG_DEBUG("%s: decode_model input_name: %s\n", __func__,
                        input_name.c_str());
    }
    // only for text model
    input_num_decode = 3;
    // for image model
    if (model_type_ == 1) {
        input_num_decode = 6;
    }
    for (size_t idx = 0; idx < input_num_decode; idx++) {
        auto input_name = decode_model->GetInputName(idx);
        LLAMA_LOG_INFO("%s: decode input_name: %s\n", __func__,
                       input_name.c_str());
        auto input_info = decode_model->GetInputInfo(input_name).AsContiguous();
        auto shape = input_info.Shape();
        for (size_t i = 0; i < shape.size(); i++) {
            LLAMA_LOG_INFO("%s: decode input_shape: %ld\n", __func__, shape[i]);
        }
        auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
        decode_input_map.insert(
            std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
    }

    // set current_length of decode input
    int32_t current_length_index = 2;
    if (model_type_ == 1) {
        current_length_index = 5;
    }
    n_decode_batch_ = decode_input_map_[decode_model->GetInputName(1)]
                               .Info()
                               .Shape()[0];

    auto decode_current_length =
        decode_input_map.at(decode_model->GetInputName(current_length_index));
    if (decode_current_length.Info().DataType() == tcim::DataType::INT16) {
        SetInputData<int16_t>(decode_current_length, n_decode_batch_, 1);
    } else if (decode_current_length.Info().DataType() ==
               tcim::DataType::INT32) {
        SetInputData<int32_t>(decode_current_length, n_decode_batch_, 1);
    }
    decode_model->SetInput(decode_model->GetInputName(current_length_index),
                           decode_current_length);
}

void HouMoLLModel::Impl::decode_output_init(
    std::shared_ptr<tcim::Module> decode_model,
    std::map<std::string, tcim::Tensor> &decode_output_map,
    std::vector<std::string> &decode_output_names) {

    // 填充decode 的output
    int output_num_decode = decode_model->GetOutputNum();
    for (int idx = 0; idx < output_num_decode; idx++) {
        auto output_name = decode_model->GetOutputName(idx);
        auto output_info = decode_model->GetOutputInfo(output_name)
                               .AsContiguous()
                               .AsType(tcim::DataType::FLOAT32);
        auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
        decode_output_map.insert(
            std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
        decode_output_names.push_back(output_name);
    }
}

void HouMoLLModel::Impl::fill_parameters(struct gguf_context *ctx) {
    auto key_id = gguf_find_key(ctx, kGeneralArch.c_str());
    if (key_id < 0) {
        LLAMA_LOG_ERROR("%s: failed to find key %s in gguf file\n", __func__,
                        kGeneralArch.c_str());
        return;
    }
    std::string arch = gguf_get_val_str(ctx, key_id);
    std::string key = arch + kBlockCountEx;
    key_id = gguf_find_key(ctx, key.c_str());
    if (key_id < 0) {
        LLAMA_LOG_ERROR("%s: failed to find key %s in gguf file\n", __func__,
                        key.c_str());
        return;
    }
    n_blocks_ = gguf_get_val_u32(ctx, key_id);
    key = arch + kEmbeddingLengthEx;
    key_id = gguf_find_key(ctx, key.c_str());
    if (key_id < 0) {
        LLAMA_LOG_ERROR("%s: failed to find key %s in gguf file\n", __func__,
                        key.c_str());
        return;
    }
    n_embd_ = gguf_get_val_u32(ctx, key_id);
}

void HouMoLLModel::Impl::read_tensor_to_buffer(const std::string &file_name,
                                               size_t offset, size_t size,
                                               std::vector<char> &buffer) {
    if (endsWithMem(file_name)) {
        void **buffers;
        void *bufferSize;
        int buffers_count = 0;
        if (parseMemoryString(file_name.c_str(), &buffers, &bufferSize,
                              &buffers_count)) {
            LLAMA_LOG_INFO("buffers (void**): 0x%p\n", (void *)buffers);
            LLAMA_LOG_INFO("bufferSize (void*): 0x%p\n", bufferSize);
            LLAMA_LOG_INFO("buffers_count (int): %d\n", buffers_count);
        }
        size_t model_size = *(size_t *)bufferSize;
        LLAMA_LOG_INFO("buffers[0]: %p\n", buffers[0]);
        LLAMA_LOG_INFO("model_size: %zu\n", model_size);
        memcpy(buffer.data(), (char *)buffers[0] + offset, size);
        LLAMA_LOG_INFO("read membuffer offset:%zu  size:%zu to buffer\r\n",
                       offset, size);
        return;
    }

    LLAMA_LOG_INFO("fread mem offset:%zu  size:%zu to buffer\r\n",offset,size);

    std::ifstream input_file(file_name, std::ios::binary);
    if (!input_file.is_open()) {
        std::cerr << "##Failed to open file: " << file_name << std::endl;
        return;
    }

    // 移动到偏移位置
    input_file.seekg(offset, std::ios::beg);
    if (input_file.fail()) {
        std::cerr << "Failed to seek to offset: " << offset
                  << " in file: " << file_name << std::endl;
        input_file.close();
        return;
    }

    // 读取数据
    input_file.read(buffer.data(), size);
    if (input_file.fail()) {
        std::cerr << "Failed to read " << size
                  << " bytes from file: " << file_name << std::endl;
        input_file.close();
        return;
    }
}

bool HouMoSingleCardModel::houmo_load(const std::string &file_name) {

    struct ggml_context *ctx_data = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_data,
    };
    struct gguf_context *ctx = gguf_init_from_file(file_name.c_str(), params);
    fill_parameters(ctx);
    LLAMA_LOG_INFO("HouMoSingleCardModel after gguf_init_from_file\n");
    // 从gguf文件中加载hmmodels
    auto device_ids = parse_deviceids();
    if (device_ids.empty()) {
        LLAMA_LOG_ERROR("%s: no device id found, please set HOUMO_DEVICE_ID\n",
                        __func__);
        device_ids.push_back(0);
    }
    tcim::DevManager dev_manager = tcim::DevManager::Create(device_ids, "");
    auto wm = tcim::Module::WeightManager::CreateWeightManager(dev_manager);
    auto option_decoder = tcim::Module::Option(wm);

    std::vector<std::string> dummy_tensor_names;
    // 防止更高效使用内存，这个将prefill 设置为dummy_tensor_names
    for (int idx = 0; idx < n_blocks_; idx++) {
        std::string k_cache_name =
            "model_layers_" + std::to_string(idx) + "_self_attn_kcache_input";
        std::string v_cache_name =
            "model_layers_" + std::to_string(idx) + "_self_attn_vcache_input";
        dummy_tensor_names.push_back(k_cache_name);
        dummy_tensor_names.push_back(v_cache_name);
    }
    auto option_prefill = tcim::Module::Option(wm);
    option_prefill.SetDummyTensors(dummy_tensor_names);
    const std::regex kPrefillPattern("^prefill(_\\d+)?\\.hmm$");
    const std::regex kDecodePattern("^decoder(_\\d+)?\\.hmm$");

    // 1. 加载 embedding
    const int n_tensors = gguf_get_n_tensors(ctx);
    for (int i = 0; i < n_tensors; ++i) {
        const char *name = gguf_get_tensor_name(ctx, i);
        LLAMA_LOG_INFO("tensor[%d]: name = %s\n", i, name);
        const size_t size = gguf_get_tensor_size(ctx, i);
        size_t offset =
            gguf_get_tensor_offset(ctx, i) + gguf_get_meta_size(ctx);
        // auto *tensor = ggml_get_tensor(ctx_data, name);
        std::vector<char> buffer(size);

        read_tensor_to_buffer(file_name, offset, size, buffer);
        if (strncmp(name, "quant_embedding.pt", strlen("quant_embedding.pt")) ==
            0) {
            torch::Tensor input;
            try {
                std::istringstream stream(
                    std::string((char *)buffer.data(), size));
                torch::load(input, stream);
                input = input.view({-1, n_embd_});
                embedding_ = torch::nn::Embedding::from_pretrained(input);
            } catch (const c10::Error &e) {
                LLAMA_LOG_ERROR("%s: failed to load quant_embedding.pt %s\n",
                                __func__, e.what());
                return false;
            }

        } else if (std::regex_match(name, kPrefillPattern)) {
            // load prefill
            prefill_model_ = std::make_shared<tcim::Module>();
            auto status =
                prefill_model_->LoadModel(buffer.data(), size, option_prefill);
            if (status != tcim::Status::OK) {
                LLAMA_LOG_ERROR("%s: failed to load prefill.hmm\n", __func__);
                return false;
            }
            LLAMA_LOG_INFO("%s: succeed to load prefill.hmm\n", __func__);
        } else if (std::regex_match(name, kDecodePattern)) {
            // load decode
            decode_model_ = std::make_shared<tcim::Module>();
            auto status =
                decode_model_->LoadModel(buffer.data(), size, option_decoder);
            if (status != tcim::Status::OK) {
                LLAMA_LOG_ERROR("%s: failed to load decode.hmm\n", __func__);
                return false;
            }
            LLAMA_LOG_INFO("%s: succeed to load decode.hmm\n", __func__);
        } else {
            LLAMA_LOG_INFO("tensor[%d]: name = %s, size = %zu\n", i, name,
                           size);
            return false;
        }
    }
    ggml_free(ctx_data);
    gguf_free(ctx);
    return true;
}

void HouMoSingleCardModel::houmo_init(int seq_max) {
    // init prefill
    prefill_input_init(prefill_model_, prefill_input_map_);
    // 填充prefill 的output
    prefill_output_init(prefill_model_, prefill_output_map_,
                        prefill_output_names_);
    // init decode
    decode_input_init(decode_model_, decode_input_map_);
    // 填充decode 的output
    decode_output_init(decode_model_, decode_output_map_, decode_output_names_);
    if (seq_max > 4) {
        LLAMA_LOG_ERROR("%s: seq_max > 4 not supported, set seq_max to 4\n",
                        __func__);
        seq_max = 4;
    }
    int size = (seq_max + n_prefill_batch_ - 1) / n_prefill_batch_;
    size = std::max(size, n_decode_batch_);
    bool has_history_sum = false;
    auto input_num_decode = decode_model_->GetInputNum();
    if (input_num_decode > (size_t)(3 * n_blocks_ * n_decode_batch_)) {
        has_history_sum = true;
    }
    for (int i = 0; i < size; i++) {
        kv_cache_item item;
        item.seq_id = i;
        for (int idx = 0; idx < n_blocks_; idx++) {
            std::string k_cache_name = "model_layers_" + std::to_string(idx) +
                                       "_self_attn_kcache_input";
            std::string v_cache_name = "model_layers_" + std::to_string(idx) +
                                       "_self_attn_vcache_input";

            std::string k_cache_history_sum_name =
                "model_layers_" + std::to_string(idx) +
                "_self_attn_kcache_history_sum";
            // FIXME: FOR XH2 temporary, will removed later. restore...
            if (!xh1_ && prefill_model_->GetInputName(4).find(
                             "past_key_cache") != std::string::npos) {
                k_cache_name =
                    "past_key_cache_" + std::to_string(idx) + ".hmcc.format";
                v_cache_name =
                    "past_value_cache_" + std::to_string(idx) + ".hmcc.format";
            } else if (!xh1_ && prefill_model_->GetInputName(4).find(
                                    ".hmcc.format") != std::string::npos) {
                k_cache_name += ".hmcc.format";
                v_cache_name += ".hmcc.format";
            }
            item.names.push_back(k_cache_name);
            item.names.push_back(v_cache_name);
            if (has_history_sum) {
                item.names.push_back(k_cache_history_sum_name);
            }

            if (n_decode_batch_ > 1) {
                k_cache_name = k_cache_name + "_batch" + std::to_string(i);
                v_cache_name = v_cache_name + "_batch" + std::to_string(i);
                k_cache_history_sum_name =
                    k_cache_history_sum_name + "_batch" + std::to_string(i);
            }
            auto kcache = decode_model_->GetInput(k_cache_name);
            auto vcache = decode_model_->GetInput(v_cache_name);
            if (n_decode_batch_ > 1 || seq_max == 1) {
                item.tensors.push_back(kcache);
                item.tensors.push_back(vcache);
            } else {
                item.tensors.push_back(kcache.Clone());
                item.tensors.push_back(vcache.Clone());
                need_set_kv_cache_ = 1;
            }

            if (has_history_sum) {
                auto kcache_history_sum =
                    decode_model_->GetInput(k_cache_history_sum_name);
                if (n_decode_batch_ > 1) {
                    item.tensors.push_back(kcache_history_sum);
                } else {
                    item.tensors.push_back(kcache_history_sum.Clone());
                }
            }
            context_length_ = std::max(
                kcache.Info().Shape()[kcache.Info().Shape().size() - 1],
                kcache.Info().Shape()[kcache.Info().Shape().size() - 2]);
        }
        kv_cache_[i] = item;
    }
    LLAMA_LOG_INFO("%s: HMM graph init success\n", __func__);
    LLAMA_LOG_INFO("prefill_length_ = %d\n context_length_ = %d\n "
                   "n_blocks_ = %d\n n_embd_ = %ld\n n_prefill_batch_ = %d\n "
                   "n_decode_batch_ = "
                   "%d\n   kv_cache_size = %zu\n need_set_kv_cache_ = "
                   "%d\n",
                   prefill_length_, context_length_, n_blocks_, n_embd_,
                   n_prefill_batch_, n_decode_batch_, kv_cache_.size(),
                   need_set_kv_cache_);
}

int HouMoSingleCardModel::houmo_prefill(std::vector<llama_token> &tokens,
                                        int seq_id, float *logits) {

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
    //[only for deepseek] clear kcache_history_sum before prefill.
    auto &kv_item = kv_cache_[seq_id];
    for (size_t idx = 0; idx < kv_item.names.size(); idx++) {
        if (kv_item.names[idx].find("kcache_history_sum") == std::string::npos)
            continue;
        auto &input_data = kv_item.tensors[idx];
        std::vector<int16_t> kcache_history_sum_init(input_data.MemSize() / 2,
                                                     0);
        input_data.Buffer().CopyFromHost(kcache_history_sum_init.data(),
                                         input_data.MemSize());
        decode_model_->SetInput(kv_item.names[idx], input_data);
    }

    std::vector<llama_token> input_ids;
    int32_t valid_length = 0;
    for (int round = 0; round < prefill_loop_round; round++) {
        valid_length = round * prefill_length_;
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
        torch::Tensor inputs_embeds;
        auto opts = torch::TensorOptions().dtype(torch::kInt);
        auto inp = torch::from_blob(input_ids.data(),
                                    {int64_t(input_ids.size())}, opts)
                       .clone();
        try {
            inputs_embeds = embedding_->forward(inp);
        } catch (const c10::Error &e) {
            std::cerr << "error loading the model\n"
                      << e.what_without_backtrace();
            prefill_tokens_.resize(0);
            return -1;
        }
        auto effective_length = input_ids.size();
        auto pad_embeds = torch::zeros(
            {(int64_t)(prefill_length_ - effective_length), n_embd_},
            inputs_embeds.dtype());
        auto padded_embeds = torch::cat({inputs_embeds, pad_embeds});
        auto input_data =
            prefill_input_map_.at(prefill_model_->GetInputName(0));
        padded_embeds = padded_embeds.view(input_data.Info().Shape());
        // 区分一下更健壮的写法
        if (xh1_) {
            padded_embeds = padded_embeds.to(torch::kInt16);
            input_data.Buffer().CopyFromHost(padded_embeds.data_ptr<int16_t>(),
                                            input_data.MemSize());
        } else {
            input_data.Buffer().CopyFromHost(padded_embeds.data_ptr<at::Half>(),
                                            input_data.MemSize());
        }

        auto prefill_valid_length =
            prefill_input_map_.at(prefill_model_->GetInputName(1));
        auto prefill_current_length =
            prefill_input_map_.at(prefill_model_->GetInputName(2));
        if (prefill_valid_length.Info().DataType() == tcim::DataType::INT16) {
            SetInputData<int16_t>(prefill_valid_length, n_prefill_batch_,
                                  valid_length);
            SetInputData<int16_t>(prefill_current_length, n_prefill_batch_,
                                  current_length);
        } else if (prefill_valid_length.Info().DataType() ==
                   tcim::DataType::INT32) {
            SetInputData<int32_t>(prefill_valid_length, n_prefill_batch_,
                                  valid_length);
            SetInputData<int32_t>(prefill_current_length, n_prefill_batch_,
                                  current_length);
        }

        prefill_model_->SetInput(prefill_model_->GetInputName(0), input_data);
        prefill_model_->SetInput(prefill_model_->GetInputName(1),
                                 prefill_valid_length);
        prefill_model_->SetInput(prefill_model_->GetInputName(2),
                                 prefill_current_length);
        auto &item = kv_cache_[seq_id];
        assert(item.seq_id == seq_id);
        for (size_t idx = 0; idx < item.tensors.size(); idx++) {
            std::string name = item.names[idx];
            auto tensor = item.tensors[idx];
            prefill_model_->SetInput(name, tensor);
        }
        prefill_model_->Run();
        prefill_model_->Sync();
    }
    decode_valid_length_[seq_id] = input_echo_len;
    tcim::Tensor prefill_output_data =
        prefill_model_->GetOutput(prefill_output_names_[0]);
    assert(prefill_output_data.Info().DataType() == tcim::DataType::INT16);
    // 转浮点
    auto &prefill_output_float = prefill_output_map_[prefill_output_names_[0]];
    prefill_output_data.CastTo(prefill_output_float);
    memcpy(logits, prefill_output_float.Data(), prefill_output_float.MemSize());
    prefill_tokens_.resize(0);
    return 0;
}

int HouMoSingleCardModel::houmo_decode(std::vector<llama_token> &batches,
                                       std::vector<int> &seq_ids,
                                       std::vector<float *> logits) {
    LLAMA_LOG_DEBUG("%s: seq_ids size = %zu, batches size = %zu\n", __func__,
                    seq_ids.size(), batches.size());
    if (seq_ids.size() > (size_t)n_decode_batch_) {
        std::cerr << "seq_ids.size() > n_decode_batch_ not supported\n";
        return -1;
    }
    torch::Tensor input_emd;
    auto decode_opts = torch::TensorOptions().dtype(torch::kInt);
    auto decode_inp =
        torch::from_blob(batches.data(), {int64_t(batches.size())}, decode_opts)
            .clone();
    try {
        input_emd = embedding_->forward(decode_inp);
    } catch (const c10::Error &e) {
        std::cerr << "error loading the model\n" << e.what_without_backtrace();
        return -1;
    }
    // set input_1, xh2 是float16
    input_emd = input_emd.view({int64_t(batches.size()), 1, -1});
    auto input_data = decode_input_map_.at(decode_model_->GetInputName(0));

    // 计算每个序列的大小（以int16_t为单位）
    int64_t embedding_dim = input_emd.size(2);
    assert(input_data.MemSize() >=
           embedding_dim * seq_ids.size() * sizeof(int16_t));

    std::vector<int16_t> decode_input_data(
        input_data.MemSize() / sizeof(int16_t), 1);
    // 填充数据
    for (size_t i = 0; i < seq_ids.size(); i++) {
        auto batch_id = seq_ids[i];
        // 单batch模式下所有序列共用batch 0
        if (n_decode_batch_ == 1)
            batch_id = 0;
        // 验证seq_id有效性
        if (batch_id >= n_decode_batch_) {
            std::cerr << "Error: Invalid seq_id " << batch_id << " (max "
                      << n_decode_batch_ - 1 << ")" << std::endl;
            return -1;
        }
        int16_t *input_data_ptr =
            decode_input_data.data() + batch_id * embedding_dim;
        // 根据目标类型拷贝更健壮
        if (xh1_) {
            input_emd = input_emd.to(torch::kInt16);
            memcpy(input_data_ptr, input_emd[i][0].data_ptr<int16_t>(),
                embedding_dim * sizeof(int16_t));
        } else {
            memcpy(input_data_ptr, input_emd[i][0].data_ptr<at::Half>(),
                embedding_dim * sizeof(at::Half));
        }
    }

    input_data.Buffer().CopyFromHost(decode_input_data.data(),
                                     input_data.MemSize());
    // set valid_length
    std::vector<int32_t> valid_length_vec(n_decode_batch_, 1);
    for (size_t k = 0; k < seq_ids.size(); k++) {
        int batch_id = seq_ids[k];
        // 单batch模式下所有序列共用batch 0
        if (n_decode_batch_ == 1)
            batch_id = 0;
        valid_length_vec[batch_id] = decode_valid_length_[seq_ids[k]];
    }
    auto decode_valid_length =
        decode_input_map_.at(decode_model_->GetInputName(1));
    if (decode_valid_length.Info().DataType() == tcim::DataType::INT16) {
        SetInputData<int16_t>(decode_valid_length, valid_length_vec);
    } else if (decode_valid_length.Info().DataType() == tcim::DataType::INT32) {
        SetInputData<int32_t>(decode_valid_length, valid_length_vec);
    }

    decode_model_->SetInput(decode_model_->GetInputName(0), input_data);
    decode_model_->SetInput(decode_model_->GetInputName(1),
                            decode_valid_length);
    if (need_set_kv_cache_ != 0) {
        for (auto seq_id : seq_ids) {
            auto &item = kv_cache_[seq_id];
            for (size_t idx = 0; idx < item.tensors.size(); idx++) {
                std::string name = item.names[idx];
                if (n_decode_batch_ > 1)
                    name = name + "_batch" + std::to_string(item.seq_id);
                auto tensor = item.tensors[idx];
                decode_model_->SetInput(name, tensor);
            }
        }
    }

    decode_model_->Run();
    decode_model_->Sync();
    tcim::Tensor decode_data =
        decode_model_->GetOutput(decode_output_names_[0]);
    assert(decode_data.Info().DataType() == tcim::DataType::INT16);
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
    }
    return 0;
}

bool HouMoDualCardModel::houmo_load(const std::string &file_name) {
    struct ggml_context *ctx_data = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ctx_data,
    };
    struct gguf_context *ctx = gguf_init_from_file(file_name.c_str(), params);
    LLAMA_LOG_INFO("HouMoDualCardModel after gguf_init_from_file\n");
    fill_parameters(ctx);
    // 从gguf文件中加载hmmodels
    auto device_ids = parse_deviceids();
    if (device_ids.size() != 2) {
        LLAMA_LOG_ERROR(
            "HOUMO_DEVICE_ID should contain exactly two device IDs.\n");
        return false;
    }
    auto wm0 = tcim::Module::WeightManager::CreateWeightManager(device_ids[0]);
    auto wm1 = tcim::Module::WeightManager::CreateWeightManager(device_ids[1]);
    auto option_prefill0 = tcim::Module::Option(wm0);
    auto option_prefill1 = tcim::Module::Option(wm1);
    auto option_decoder0 = tcim::Module::Option(wm0);
    auto option_decoder1 = tcim::Module::Option(wm1);

    std::vector<std::string> dummy_tensor_names;
    // 防止更高效使用内存，这个将prefill 设置为dummy_tensor_names
    for (int idx = 0; idx < n_blocks_; idx++) {
        std::string k_cache_name =
            "model_layers_" + std::to_string(idx) + "_self_attn_kcache_input";
        std::string v_cache_name =
            "model_layers_" + std::to_string(idx) + "_self_attn_vcache_input";
        dummy_tensor_names.push_back(k_cache_name);
        dummy_tensor_names.push_back(v_cache_name);
    }
    option_prefill0.SetDummyTensors(dummy_tensor_names);
    option_prefill1.SetDummyTensors(dummy_tensor_names);

    // 1. 加载 embedding
    const int n_tensors = gguf_get_n_tensors(ctx);
    for (int i = 0; i < n_tensors; ++i) {
        const char *name = gguf_get_tensor_name(ctx, i);
        LLAMA_LOG_INFO("tensor[%d]: name = %s\n", i, name);
        const size_t size = gguf_get_tensor_size(ctx, i);
        auto *tensor = ggml_get_tensor(ctx_data, name);

        if (strncmp(name, "quant_embedding.pt", strlen("quant_embedding.pt")) ==
            0) {
            torch::Tensor input;
            try {
                std::istringstream stream(
                    std::string((char *)tensor->data, size));
                torch::load(input, stream);
                input = input.view({-1, n_embd_});
                embedding_ = torch::nn::Embedding::from_pretrained(input);
            } catch (const c10::Error &e) {
                LLAMA_LOG_ERROR("%s: failed to load quant_embedding.pt\n",
                                __func__);
                return false;
            }

        } else if (strncmp(name, "prefill_0.hmm", strlen("prefill_0.hmm")) ==
                   0) {
            // load prefill
            prefill_model_card0_ = std::make_shared<tcim::Module>();
            auto status = prefill_model_card0_->LoadModel(tensor->data, size,
                                                          option_prefill0);
            if (status != tcim::Status::OK) {
                LLAMA_LOG_ERROR("%s: failed to load prefill_0.hmm\n", __func__);
                return false;
            }
            LLAMA_LOG_INFO("%s: succeed to load prefill_0.hmm\n", __func__);
        } else if (strncmp(name, "prefill_1.hmm", strlen("prefill_1.hmm")) ==
                   0) {
            // load prefill
            prefill_model_card1_ = std::make_shared<tcim::Module>();
            auto status = prefill_model_card1_->LoadModel(tensor->data, size,
                                                          option_prefill1);
            if (status != tcim::Status::OK) {
                LLAMA_LOG_ERROR("%s: failed to load prefill_1.hmm\n", __func__);
                return false;
            }
            LLAMA_LOG_INFO("%s: succeed to load prefill_1.hmm\n", __func__);
        } else if (strncmp(name, "decoder_0.hmm", strlen("decoder_0.hmm")) ==
                   0) {
            // load decode
            decode_model_card0_ = std::make_shared<tcim::Module>();
            auto status = decode_model_card0_->LoadModel(tensor->data, size,
                                                         option_decoder0);
            if (status != tcim::Status::OK) {
                LLAMA_LOG_ERROR("%s: failed to load decoder_0.hmm\n", __func__);
                return false;
            }
            LLAMA_LOG_INFO("%s: succeed to load decoder_0.hmm\n", __func__);
        } else if (strncmp(name, "decoder_1.hmm", strlen("decoder_1.hmm")) ==
                   0) {
            // load decode
            decode_model_card1_ = std::make_shared<tcim::Module>();
            auto status = decode_model_card1_->LoadModel(tensor->data, size,
                                                         option_decoder1);
            if (status != tcim::Status::OK) {
                LLAMA_LOG_ERROR("%s: failed to load decoder_1.hmm\n", __func__);
                return false;
            }
            LLAMA_LOG_INFO("%s: succeed to load decoder_1.hmm\n", __func__);
        } else {
            LLAMA_LOG_INFO("tensor[%d]: name = %s, size = %zu\n", i, name,
                           size);
            return false;
        }
    }
    ggml_free(ctx_data);
    gguf_free(ctx);
    return true;
}

void HouMoDualCardModel::houmo_init(int seq_max) {
    // init prefill
    prefill_input_init(prefill_model_card0_, prefill_input_map_);
    // 填充prefill 的output
    prefill_output_init(prefill_model_card1_, prefill_output_map_,
                        prefill_output_names_);
    // init decode
    decode_input_init(decode_model_card0_, decode_input_map_);
    // 填充decode 的output
    decode_output_init(decode_model_card1_, decode_output_map_,
                       decode_output_names_);
    for (size_t i = 0; i < prefill_model_card0_->GetOutputNum(); i++) {
        auto output_name = prefill_model_card0_->GetOutputName(i);
        prefill_inner_output_names_.push_back(output_name);
    }
    for (size_t i = 0; i < decode_model_card0_->GetOutputNum(); i++) {
        auto output_name = decode_model_card0_->GetOutputName(i);
        decode_inner_output_names_.push_back(output_name);
    }
    auto decode_current_length = decode_input_map_.at("current_length");
    decode_model_card1_->SetInput("current_length", decode_current_length);

    if (seq_max > kSeqMax) {
        LLAMA_LOG_ERROR("%s: seq_max > 4 not supported, set seq_max to 4\n",
                        __func__);
        seq_max = kSeqMax;
    }
    int size = (seq_max + n_prefill_batch_ - 1) / n_prefill_batch_;
    size = std::max(size, n_decode_batch_);
    bool has_history_sum = false;
    int input_num_decode =
        decode_model_card0_->GetInputNum() + decode_model_card1_->GetInputNum();
    int split_blocks = 0;
    split_size_ = decode_model_card0_->GetInputNum() - 3;

    if (input_num_decode > 3 * n_blocks_ * n_decode_batch_) {
        has_history_sum = true;
        split_blocks = split_size_ / 3 * n_decode_batch_;
    } else {
        split_blocks = split_size_ / 2 * n_decode_batch_;
    }
    // 计算切了多少个kvcache
    for (int i = 0; i < size; i++) {
        kv_cache_item item;
        item.seq_id = i;
        for (int idx = 0; idx < n_blocks_; idx++) {
            std::string k_cache_name = "model_layers_" + std::to_string(idx) +
                                       "_self_attn_kcache_input";
            std::string v_cache_name = "model_layers_" + std::to_string(idx) +
                                       "_self_attn_vcache_input";

            std::string k_cache_history_sum_name =
                "model_layers_" + std::to_string(idx) +
                "_self_attn_kcache_history_sum";
            item.names.push_back(k_cache_name);
            item.names.push_back(v_cache_name);
            if (has_history_sum) {
                item.names.push_back(k_cache_history_sum_name);
            }

            if (n_decode_batch_ > 1) {
                k_cache_name = k_cache_name + "_batch" + std::to_string(i);
                v_cache_name = v_cache_name + "_batch" + std::to_string(i);
                k_cache_history_sum_name =
                    k_cache_history_sum_name + "_batch" + std::to_string(i);
            }
            auto decode_model = decode_model_card0_;
            // 如果是第二张卡，切换decode_model
            if (idx >= split_blocks) {
                decode_model = decode_model_card1_;
            }
            auto kcache = decode_model->GetInput(k_cache_name);
            auto vcache = decode_model->GetInput(v_cache_name);
            if (n_decode_batch_ > 1 || seq_max == 1) {
                item.tensors.push_back(kcache);
                item.tensors.push_back(vcache);
            } else {
                item.tensors.push_back(kcache.Clone());
                item.tensors.push_back(vcache.Clone());
                need_set_kv_cache_ = 1;
            }

            if (has_history_sum) {
                auto kcache_history_sum =
                    decode_model->GetInput(k_cache_history_sum_name);
                if (n_decode_batch_ > 1) {
                    item.tensors.push_back(kcache_history_sum);
                } else {
                    item.tensors.push_back(kcache_history_sum.Clone());
                }
            }
            context_length_ =
                kcache.Info().Shape()[kcache.Info().Shape().size() - 1];
        }
        kv_cache_[i] = item;
    }

    // KV cache init.
    LLAMA_LOG_INFO("%s: kvcache init finished\n", __func__);

    LLAMA_LOG_INFO("%s: HMM graph init success\n", __func__);
    LLAMA_LOG_INFO("prefill_length_ = %d\n context_length_ = %d\n "
                   "n_blocks_ = %d\n n_embd_ = %ld\n n_prefill_batch_ = %d\n "
                   "n_decode_batch_ = "
                   "%d\n   kv_cache_size = %zu\n need_set_kv_cache_ = %d\n",
                   prefill_length_, context_length_, n_blocks_, n_embd_,
                   n_prefill_batch_, n_decode_batch_, kv_cache_.size(),
                   need_set_kv_cache_);
}

int HouMoDualCardModel::houmo_prefill(std::vector<llama_token> &tokens,
                                      int seq_id, float *logits) {

    LLAMA_LOG_DEBUG("%s seq_id = %d, tokens size is = %zu\n", "prefill stage",
                    seq_id, tokens.size());
    size_t input_echo_len = tokens.size();
    int16_t prefill_loop_round = ceil(static_cast<float>(input_echo_len) /
                                      static_cast<float>(prefill_length_));
    int16_t current_length = 0;
    //[only for deepseek] clear kcache_history_sum before prefill.
    auto &kv_item = kv_cache_[seq_id];
    for (size_t idx = 0; idx < kv_item.names.size(); idx++) {
        if (kv_item.names[idx].find("kcache_history_sum") == std::string::npos)
            continue;
        auto &input_data = kv_item.tensors[idx];
        std::vector<int16_t> kcache_history_sum_init(input_data.MemSize() / 2,
                                                     0);
        input_data.Buffer().CopyFromHost(kcache_history_sum_init.data(),
                                         input_data.MemSize());
        if (idx < split_size_) {
            decode_model_card0_->SetInput(kv_item.names[idx], input_data);
        } else {
            decode_model_card1_->SetInput(kv_item.names[idx], input_data);
        }
    }

    std::vector<llama_token> input_ids;
    int16_t valid_length = 0;
    for (int round = 0; round < prefill_loop_round; round++) {
        valid_length = round * prefill_length_;
        if (round == prefill_loop_round - 1) {
            current_length = input_echo_len - round * prefill_length_;
            auto first = tokens.begin() + round * prefill_length_;
            auto last = tokens.begin() + input_echo_len;
            std::vector<llama_token> round_tokens(first, last);
            input_ids = round_tokens;
        } else {
            current_length = prefill_length_;
            auto first = tokens.begin() + round * prefill_length_;
            auto last = tokens.begin() + (round + 1) * prefill_length_;
            std::vector<llama_token> round_tokens(first, last);
            input_ids = round_tokens;
        }
        torch::Tensor inputs_embeds;
        auto opts = torch::TensorOptions().dtype(torch::kInt);
        auto inp = torch::from_blob(input_ids.data(),
                                    {int64_t(input_ids.size())}, opts)
                       .clone();
        try {
            inputs_embeds = embedding_->forward(inp);
        } catch (const c10::Error &e) {
            std::cerr << "error loading the model\n"
                      << e.what_without_backtrace();
            return -1;
        }
        auto effective_length = input_ids.size();
        auto pad_embeds = torch::zeros(
            {(int64_t)(prefill_length_ - effective_length), n_embd_},
            torch::kInt16);
        auto padded_embeds = torch::cat({inputs_embeds, pad_embeds});

        auto input_data = prefill_input_map_.at("input_1");
        padded_embeds =
            padded_embeds.view(input_data.Info().Shape()).to(torch::kInt16);
        input_data.Buffer().CopyFromHost(padded_embeds.data_ptr<int16_t>(),
                                         padded_embeds.numel() *
                                             sizeof(int16_t));

        auto prefill_valid_length = prefill_input_map_.at("valid_length");
        std::vector<int16_t> valid_length_vec(n_prefill_batch_, valid_length);
        prefill_valid_length.Buffer().CopyFromHost(
            valid_length_vec.data(), sizeof(int16_t) * valid_length_vec.size());

        std::vector<int16_t> current_length_vec(n_prefill_batch_,
                                                current_length);
        auto prefill_current_length = prefill_input_map_.at("current_length");
        prefill_current_length.Buffer().CopyFromHost(
            current_length_vec.data(),
            sizeof(int16_t) * valid_length_vec.size());
        prefill_model_card0_->SetInput("input_1", input_data);
        prefill_model_card0_->SetInput("valid_length", prefill_valid_length);
        prefill_model_card0_->SetInput("current_length",
                                       prefill_current_length);
        auto &item = kv_cache_[seq_id];
        assert(item.seq_id == seq_id);
        for (size_t idx = 0; idx < split_size_; idx++) {
            std::string name = item.names[idx];
            auto tensor = item.tensors[idx];
            prefill_model_card0_->SetInput(name, tensor);
        }
        prefill_model_card0_->Run();
        prefill_model_card0_->Sync();
        LLAMA_LOG_DEBUG("%s: prefill_model_card0_ run success\n", __func__);
        auto inner_tensor =
            prefill_model_card0_->GetOutput(prefill_inner_output_names_[0]);
        prefill_model_card1_->SetInput(prefill_inner_output_names_[0],
                                       inner_tensor);
        prefill_model_card1_->SetInput("valid_length", prefill_valid_length);
        prefill_model_card1_->SetInput("current_length",
                                       prefill_current_length);
        for (size_t idx = split_size_; idx < item.names.size(); idx++) {
            std::string name = item.names[idx];
            auto tensor = item.tensors[idx];
            prefill_model_card1_->SetInput(name, tensor);
        }
        prefill_model_card1_->Run();
        prefill_model_card1_->Sync();
        LLAMA_LOG_DEBUG("%s: prefill_model_card1_ run success\n", __func__);
    }
    decode_valid_length_[seq_id] = input_echo_len;
    tcim::Tensor prefill_output_data =
        prefill_model_card1_->GetOutput(prefill_output_names_[0]);
    assert(prefill_output_data.Info().DataType() == tcim::DataType::INT16);
    // 转浮点
    auto &prefill_output_float = prefill_output_map_[prefill_output_names_[0]];
    prefill_output_data.CastTo(prefill_output_float);
    memcpy(logits, prefill_output_float.Data(), prefill_output_float.MemSize());
    return 0;
}

int HouMoDualCardModel::houmo_decode(std::vector<llama_token> &batches,
                                     std::vector<int> &seq_ids,
                                     std::vector<float *> logits) {
    LLAMA_LOG_DEBUG("%s: seq_ids size = %zu, batches size = %zu\n", __func__,
                    seq_ids.size(), batches.size());
    if (seq_ids.size() > (size_t)n_decode_batch_) {
        std::cerr << "seq_ids.size() > n_decode_batch_ not supported\n";
        return -1;
    }
    torch::Tensor input_emd;
    auto decode_opts = torch::TensorOptions().dtype(torch::kInt);
    auto decode_inp =
        torch::from_blob(batches.data(), {int64_t(batches.size())}, decode_opts)
            .clone();
    try {
        input_emd = embedding_->forward(decode_inp);
    } catch (const c10::Error &e) {
        std::cerr << "error loading the model\n" << e.what_without_backtrace();
        return -1;
    }
    // set input_1
    input_emd =
        input_emd.view({int64_t(batches.size()), 1, -1}).to(torch::kInt16);
    auto input_data = decode_input_map_.at("input_1");
    assert(input_data.Info().DataType() == tcim::DataType::INT16);

    // 计算每个序列的大小（以int16_t为单位）
    int64_t embedding_dim = input_emd.size(2);
    assert(input_data.MemSize() >=
           embedding_dim * seq_ids.size() * sizeof(int16_t));

    std::vector<int16_t> decode_input_data(
        input_data.MemSize() / sizeof(int16_t), 1);
    // 填充数据
    for (size_t i = 0; i < seq_ids.size(); i++) {
        auto batch_id = seq_ids[i];
        // 单batch模式下所有序列共用batch 0
        if (n_decode_batch_ == 1)
            batch_id = 0;
        // 验证seq_id有效性
        if (batch_id >= n_decode_batch_) {
            std::cerr << "Error: Invalid seq_id " << batch_id << " (max "
                      << n_decode_batch_ - 1 << ")" << std::endl;
            return -1;
        }
        int16_t *input_data_ptr =
            decode_input_data.data() + batch_id * embedding_dim;
        memcpy(input_data_ptr, input_emd[i][0].data_ptr<int16_t>(),
               embedding_dim * sizeof(int16_t));
    }

    input_data.Buffer().CopyFromHost(decode_input_data.data(),
                                     input_data.MemSize());
    // set valid_length
    std::vector<int16_t> valid_length_vec(n_decode_batch_, 1);
    for (size_t k = 0; k < seq_ids.size(); k++) {
        int batch_id = seq_ids[k];
        // 单batch模式下所有序列共用batch 0
        if (n_decode_batch_ == 1)
            batch_id = 0;
        valid_length_vec[batch_id] = decode_valid_length_[seq_ids[k]];
    }
    auto decode_valid_length = decode_input_map_.at("valid_length");
    decode_valid_length.Buffer().CopyFromHost(
        valid_length_vec.data(), sizeof(int16_t) * valid_length_vec.size());

    decode_model_card0_->SetInput("input_1", input_data);
    decode_model_card0_->SetInput("valid_length", decode_valid_length);
    if (need_set_kv_cache_ != 0) {
        for (auto seq_id : seq_ids) {
            auto &item = kv_cache_[seq_id];
            for (size_t idx = 0; idx < (size_t)split_size_; idx++) {
                std::string name = item.names[idx];
                if (n_decode_batch_ > 1)
                    name = name + "_batch" + std::to_string(item.seq_id);
                auto tensor = item.tensors[idx];
                decode_model_card0_->SetInput(name, tensor);
            }
        }
    }
    decode_model_card0_->Run();
    decode_model_card0_->Sync();

    auto inner_tensor =
        decode_model_card0_->GetOutput(decode_inner_output_names_[0]);
    decode_model_card1_->SetInput(decode_inner_output_names_[0], inner_tensor);
    decode_model_card1_->SetInput("valid_length", decode_valid_length);
    if (need_set_kv_cache_ != 0) {
        for (auto seq_id : seq_ids) {
            auto &item = kv_cache_[seq_id];
            for (size_t idx = split_size_; idx < item.names.size(); idx++) {
                std::string name = item.names[idx];
                if (n_decode_batch_ > 1)
                    name = name + "_batch" + std::to_string(item.seq_id);
                auto tensor = item.tensors[idx];
                decode_model_card1_->SetInput(name, tensor);
            }
        }
    }
    decode_model_card1_->Run();
    decode_model_card1_->Sync();
    tcim::Tensor decode_data =
        decode_model_card1_->GetOutput(decode_output_names_[0]);

    assert(decode_data.Info().DataType() == tcim::DataType::INT16);
    // 转浮点
    auto &decode_output_float = decode_output_map_[decode_output_names_[0]];
    decode_data.CastTo(decode_output_float);
    auto n_vocab = decode_output_float.Info().Shape()[1];
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
    }
    return 0;
}

// HouMoLLModel constructor and destructor
HouMoLLModel::HouMoLLModel() : impl_(nullptr) {}

HouMoLLModel::~HouMoLLModel() {
    if (impl_) {
        delete impl_;
    }
}

// Forward method calls to the implementation
bool HouMoLLModel::houmo_load(const std::string &file_name) {

    struct ggml_context *ctx_data = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_data,
    };
    struct gguf_context *ctx = gguf_init_from_file(file_name.c_str(), params);
    const int n_tensors = gguf_get_n_tensors(ctx);
    auto key_id = gguf_find_key(ctx, "general.architecture");
    if (key_id < 0) {
        return false;
    }
    std::string arch = gguf_get_val_str(ctx, key_id);

    if (impl_ == nullptr) {
        if (n_tensors == 3) {
            if (arch.find("vl") != std::string::npos) {
                impl_ = new HouMoVLLModel();
            } else {
                impl_ = new HouMoSingleCardModel();
            }
        } else {
            impl_ = new HouMoDualCardModel();
        }
    }

    return impl_->houmo_load(file_name);
}

void HouMoLLModel::houmo_init(int seq_max) { impl_->houmo_init(seq_max); }

int HouMoLLModel::houmo_prefill(std::vector<llama_token> &tokens, int seq_id,
                                float *logits) {
    return impl_->houmo_prefill(tokens, seq_id, logits);
}

int HouMoLLModel::houmo_decode(std::vector<llama_token> &batches,
                               std::vector<int> &seq_ids,
                               std::vector<float *> logits) {
    return impl_->houmo_decode(batches, seq_ids, logits);
}

int HouMoLLModel::n_decode_batch() { return impl_->n_decode_batch(); }

int HouMoLLModel::n_prefill_batch() { return impl_->n_prefill_batch(); }

uint32_t HouMoLLModel::n_context_length() { return impl_->n_context_length(); }
int HouMoLLModel::houmo_prefill(const float *embeddings, int n_tokens,
                                int seq_id, float *logits) {

    return impl_->houmo_prefill(embeddings, n_tokens, seq_id, logits);
}
