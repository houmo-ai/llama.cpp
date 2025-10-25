#include "houmo-llmodel.h"

#include "ggml-hmutils.h"
#include "ggml.h"
#include "gguf.h"
#include "houmo-dualcard-model.h"
#include "houmo-vllmodel.h"
#include "llama-batch.h"
#include "llama-impl.h"
#include "tcim/tcim_runtime.h"
#include "torch/torch.h"
#include <fstream>
#include <map>
#include <regex>
#include <string>
#include <vector>
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
   void fill_decode_valid_length(std::vector<int32_t> &valid_length_vec);
  private:
    int kSeqMax = 4; // maximum sequence number for single-card model
    std::shared_ptr<tcim::Module> prefill_model_;
    std::shared_ptr<tcim::Module> decode_model_;
    std::vector<llama_token> prefill_tokens_;
};

void HouMoLLModel::Impl::prefill_input_init(
    std::shared_ptr<tcim::Module> prefill_model,
    std::map<std::string, tcim::Tensor> &prefill_input_map) {
    for (size_t idx = 0; idx < prefill_model->GetInputNum(); idx++) {
        auto input_name = prefill_model->GetInputName(idx);
        LLAMA_LOG_DEBUG("%s: prefill input_name: %s\n", __func__,
                        input_name.c_str());
        if (input_name.find("past_key_cache_") != std::string::npos ||
            input_name.find("_self_attn_vcache") != std::string::npos ||
            input_name.find("_self_attn_kcache") != std::string::npos) {
            continue;
        }
        auto input_info =
            prefill_model->GetInputInfo(input_name).AsContiguous();
        auto shape = input_info.Shape();
        for (size_t i = 0; i < shape.size(); i++) {
            LLAMA_LOG_INFO("%s: prefill input_shape: %ld, type = %d\n",
                           input_name.c_str(), shape[i],
                           (int)input_info.DataType());
        }
        auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
        prefill_input_map.insert(
            std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
    }
    // 填充prefill_length_ & n_prefill_batch_ only support batch_size = 1
    n_prefill_batch_ = 1;
    prefill_length_ =
        prefill_input_map_[prefill_model->GetInputName(0)].Info().Shape()[1] *
        prefill_input_map_[prefill_model->GetInputName(0)].Info().Shape()[0];
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
        if (input_name.find("past_key_cache_") != std::string::npos ||
            input_name.find("_self_attn_vcache") != std::string::npos ||
            input_name.find("_self_attn_kcache") != std::string::npos) {
            continue;
        }

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
    n_decode_batch_ =
        decode_input_map_[decode_model->GetInputName(0)].Info().Shape()[0];

    auto decode_current_length =
        decode_input_map.at(decode_model->GetInputName(current_length_index));
    set_input_data(decode_current_length, n_decode_batch_, 1);
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
        void **buffers = nullptr;
        void *bufferSize = nullptr;
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

std::vector<int> HouMoLLModel::Impl::parse_deviceids(bool single_card) {
    std::vector<int> deviceIds;
    const char *device_id_str = std::getenv("HOUMO_DEVICE_ID");

    if (device_id_str == nullptr) {
        deviceIds.push_back(0);
        if (!single_card)
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

void HouMoLLModel::Impl::set_input_data(tcim::Tensor &tensor, int batch_size,
                                        int value) {
    if (tensor.Info().DataType() == tcim::DataType::INT16) {
        std::vector<int16_t> data_vec(batch_size, value);
        tensor.Buffer().CopyFromHost(data_vec.data(), tensor.MemSize());
    } else if (tensor.Info().DataType() == tcim::DataType::INT32) {
        std::vector<int32_t> data_vec(batch_size, value);
        tensor.Buffer().CopyFromHost(data_vec.data(), tensor.MemSize());
    }
}

void HouMoLLModel::Impl::set_input_data(tcim::Tensor &tensor,
                                        std::vector<int32_t> &int32_vec,
                                        size_t offset) {
    if (offset >= int32_vec.size()) {
        throw std::out_of_range("Offset is out of range");
    }
    if (tensor.Info().DataType() == tcim::DataType::INT16) {
        std::vector<int16_t> data_vec(int32_vec.size() - offset, 0);
        for (size_t i = offset; i < int32_vec.size(); i++) {
            data_vec[i - offset] = static_cast<int16_t>(int32_vec[i]);
        }
        tensor.Buffer().CopyFromHost(data_vec.data(), tensor.MemSize());
    } else if (tensor.Info().DataType() == tcim::DataType::INT32) {
        std::vector<int32_t> data_vec(int32_vec.size() - offset, 0);
        for (size_t i = offset; i < int32_vec.size(); i++) {
            data_vec[i - offset] = static_cast<int32_t>(int32_vec[i]);
        }
        tensor.Buffer().CopyFromHost(data_vec.data(), tensor.MemSize());
    } else {
        throw std::invalid_argument("Unsupported tensor data type");
    }
}

void HouMoSingleCardModel::fill_decode_valid_length(
    std::vector<int32_t> &valid_length_vec) {
    if (xh1_) {
        auto decode_valid_length =
        decode_input_map_.at(decode_model_->GetInputName(1));
        set_input_data(decode_valid_length, valid_length_vec);
        decode_model_->SetInput(decode_model_->GetInputName(1),
                                decode_valid_length);
        return;
    }

    // for xh2
    for (size_t i = 0; i < valid_length_vec.size(); i++) {
        // valid_length
        std::string name = decode_model_->GetInputName(1);
        if (i > 0 && n_decode_batch_ > 1) {
            name += "_batch" + std::to_string(i);
        }
        auto decode_valid_length = decode_input_map_.at(name);
        set_input_data(decode_valid_length, 1, valid_length_vec[i]);
        auto status = decode_model_->SetInput(name, decode_valid_length);
        if (status != tcim::Status::OK) {
            LLAMA_LOG_ERROR("%s: failed to set input %s\n", __func__,
                            name.c_str());
            return;
        }
        // current_length
        if (n_decode_batch_ > 1) {
            std::string name = decode_model_->GetInputName(2);
            // FIXME: @guoxing.xu 当前XH2模型batch0 没有后缀
            if (i > 0) {
                name += "_batch" + std::to_string(i);
            }
            auto current_valid_length = decode_input_map_.at(name);
            set_input_data(current_valid_length, 1, 1);
            LLAMA_LOG_DEBUG("name = %s, current_valid_length: %d\n",name.c_str(), 1);
            status = decode_model_->SetInput(name, current_valid_length);
            if (status != tcim::Status::OK) {
                LLAMA_LOG_ERROR("%s: failed to set input %s\n", __func__,
                            name.c_str());
                return;
            }
        }
        //FIXME: @guoxing.xu 文本模型(for moe 测试模型,模型修改后可以移除)
        if (model_type_ == 0 && decode_input_map_.count(decode_model_->GetInputName(3)) > 0) {
            auto position_ids = decode_input_map_.at(decode_model_->GetInputName(3));
            std::string name = decode_model_->GetInputName(3);
            if (i > 0 && n_decode_batch_ > 1) {
                name += "_batch" + std::to_string(i);
            }
            set_input_data(position_ids, 1, valid_length_vec[i]);
            auto status = decode_model_->SetInput(name, position_ids);
            if (status != tcim::Status::OK) {
                LLAMA_LOG_ERROR("%s: failed to set input %s\n", __func__,
                            name.c_str());
                return;
            }
        }
    }
}

bool HouMoSingleCardModel::houmo_load(const std::string &file_name) {

    struct ggml_context *ctx_data = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_data,
    };
    bool load_from_file =
        (std::getenv("LOAD_FROM_FILE") == nullptr) ? false : true;
    struct gguf_context *ctx = gguf_init_from_file(file_name.c_str(), params);
    fill_parameters(ctx);
    LLAMA_LOG_INFO("HouMoSingleCardModel after gguf_init_from_file\n");
    // 从gguf文件中加载hmmodels
    auto device_ids = parse_deviceids();
#ifdef XH2_ENABLED
    tcim::DevManager dev_manager =
        tcim::DevManager::Create(device_ids, "Xh2HalBackend");
    auto wm = tcim::Module::WeightManager::CreateWeightManager(dev_manager);
#else
    // FIXME: xh1 还未发布v2.5.0, 还不支持DevManager
    auto wm = tcim::Module::WeightManager::CreateWeightManager(device_ids[0]);
#endif
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
    const std::regex kDecodePattern("^decode(r)?(_\\d+)?\\.hmm$");

    // 1. 加载 embedding
    const int n_tensors = gguf_get_n_tensors(ctx);
    for (int i = 0; i < n_tensors; ++i) {
        const char *name = gguf_get_tensor_name(ctx, i);
        LLAMA_LOG_INFO("tensor[%d]: name = %s\n", i, name);
        const size_t size = gguf_get_tensor_size(ctx, i);
        size_t offset =
            gguf_get_tensor_offset(ctx, i) + gguf_get_meta_size(ctx);
        tcim::Status status;
        if (strncmp(name, "quant_embedding.pt", strlen("quant_embedding.pt")) ==
            0) {
            std::vector<char> buffer(size);
            read_tensor_to_buffer(file_name, offset, size, buffer);
            torch::Tensor input;
            try {
                torch::load(input, buffer.data(), size);
                if (!input.is_floating_point()) {
                    input = input.to(torch::kFloat32);
                }
                input = input.detach();
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
            if (!load_from_file) {
                std::vector<char> buffer(size);
                read_tensor_to_buffer(file_name, offset, size, buffer);
                status = prefill_model_->LoadModel(buffer.data(), size,
                                                   option_prefill);
            } else {
                LLAMA_LOG_INFO("load %s\n", "./hmms/prefill.hmm");
                status = prefill_model_->LoadModel("./hmms/prefill.hmm",
                                                   option_prefill);
            }
            if (status != tcim::Status::OK) {
                LLAMA_LOG_ERROR("%s: failed to load prefill.hmm\n", __func__);
                return false;
            }
            LLAMA_LOG_INFO("%s: succeed to load prefill.hmm\n", __func__);
        } else if (std::regex_match(name, kDecodePattern)) {
            // load decode
            decode_model_ = std::make_shared<tcim::Module>();
            if (!load_from_file) {
                std::vector<char> buffer(size);
                read_tensor_to_buffer(file_name, offset, size, buffer);
                status = decode_model_->LoadModel(buffer.data(), size,
                                                  option_decoder);
            } else {
                status = decode_model_->LoadModel("./hmms/decoder.hmm",
                                                  option_decoder);
            }
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
    if (embedding_.get() == nullptr || prefill_model_ == nullptr ||
        decode_model_ == nullptr) {
        LLAMA_LOG_ERROR("%s: failed to load model. load embedding=%d,load "
                        "prefill=%d,load decode=%d\n",
                        __func__, (embedding_.get() == nullptr),
                        (prefill_model_ == nullptr),
                        (decode_model_ == nullptr));
        return false;
    }

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
                if (xh1_) {
                    k_cache_name = k_cache_name + "_batch" + std::to_string(i);
                    v_cache_name = v_cache_name + "_batch" + std::to_string(i);
                    k_cache_history_sum_name =
                        k_cache_history_sum_name + "_batch" + std::to_string(i);
                } else if (i > 0) {
                    // FIXME: @guoxing.xu 当前XH2模型 batch0 没有后缀
                    k_cache_name = k_cache_name + "_batch" + std::to_string(i);
                    v_cache_name = v_cache_name + "_batch" + std::to_string(i);
                    k_cache_history_sum_name =
                        k_cache_history_sum_name + "_batch" + std::to_string(i);
                }
            }
            #ifdef XH2_ENABLED
            auto kcache = decode_model_->GetDevInput(k_cache_name);
            auto vcache = decode_model_->GetDevInput(v_cache_name);
            #else
            auto kcache = decode_model_->GetInput(k_cache_name);
            auto vcache = decode_model_->GetInput(v_cache_name);
            #endif
            if (n_decode_batch_ > 1 || seq_max == 1) {
                item.tensors.push_back(kcache);
                item.tensors.push_back(vcache);
            } else {
                item.tensors.push_back(kcache.Clone());
                item.tensors.push_back(vcache.Clone());
                need_set_kv_cache_ = 1;
            }

            if (has_history_sum) {
                #ifdef XH2_ENABLED
                auto kcache_history_sum =
                    decode_model_->GetDevInput(k_cache_history_sum_name);
                #else
                auto kcache_history_sum =
                    decode_model_->GetInput(k_cache_history_sum_name);
                #endif
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
        if (model_type_ == 0 && prefill_input_map_.count(prefill_model_->GetInputName(3)) > 0) {
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
        set_input_data(prefill_valid_length, n_prefill_batch_, valid_length);
        set_input_data(prefill_current_length, n_prefill_batch_,
                       current_length);

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
    // 构造input_tokens
    std::vector<llama_token> input_ids(n_decode_batch_, 0);
    for (size_t i = 0; i < seq_ids.size(); i++) {
        int batch_id = seq_ids[i];
        // 单batch模式下所有序列共用batch 0
        if (n_decode_batch_ == 1)
            batch_id = 0;
        input_ids[batch_id] = batches[i];
    }

    torch::Tensor input_emd;
    auto decode_opts = torch::TensorOptions().dtype(torch::kInt);
    auto decode_inp =
        torch::from_blob(input_ids.data(), {int64_t(input_ids.size())}, decode_opts)
            .clone();
    try {
        input_emd = embedding_->forward(decode_inp);
    } catch (const c10::Error &e) {
        std::cerr << "error loading the model\n" << e.what_without_backtrace();
        return -1;
    }
    auto input_data = decode_input_map_.at(decode_model_->GetInputName(0));
    // 填充数据
    if (xh1_) {
        input_emd = input_emd.to(torch::kInt16);
        input_data.Buffer().CopyFromHost(input_emd.data_ptr<int16_t>(),
                                         input_data.MemSize());
    } else {
        input_data.Buffer().CopyFromHost(input_emd.data_ptr<at::Half>(),
                                         input_data.MemSize());
    }
    // set valid_length
    std::vector<int32_t> valid_length_vec(n_decode_batch_, 1);
    for (size_t k = 0; k < seq_ids.size(); k++) {
        int batch_id = seq_ids[k];
        // 单batch模式下所有序列共用batch 0
        if (n_decode_batch_ == 1)
            batch_id = 0;
        valid_length_vec[batch_id] = decode_valid_length_[seq_ids[k]];
        if (valid_length_vec[batch_id] >= context_length_) {
            LLAMA_LOG_WARN("%s: valid_length_vec[%d] = %d >= context_length_ , "
                           "end of decoding\n",
                           __func__, batch_id, valid_length_vec[batch_id]);
            return 1;
        }
    }
    fill_decode_valid_length(valid_length_vec);
    decode_model_->SetInput(decode_model_->GetInputName(0), input_data);

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
            #ifndef XH2_ENABLED
            impl_ = new HouMoDualCardModel();
            #else
            return false;
            #endif
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
