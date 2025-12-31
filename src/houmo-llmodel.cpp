#include "houmo-llmodel.h"
#include "ggml-hmutils.h"
#include "ggml.h"
#include "gguf.h"
#include "houmo-embed.h"
#include "houmo-gptoss.h"
#include "houmo-qwenllm.h"
#include "houmo-vllmodel.h"
#include "llama-arch.h"
#include "llama-batch.h"
#include "llama-impl.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>
using json = nlohmann::ordered_json;
static bool string_replace_once(std::string &str, const std::string &from,
                                const std::string &to) {
    if (from.empty())
        return false;
    size_t pos = str.find(from);
    if (pos == std::string::npos)
        return false;
    str.replace(pos, from.length(), to);
    return true;
}
std::shared_ptr<tcim::Module>
HouMoLLModel::Impl::create_infer_engine(bool prefill, void *raw_data,
                                        int64_t size, bool load_from_file,
                                        std::vector<std::string> dummy_tensor_names) {
    if (wm_.GetInitStatus() != tcim::Status::OK) {
        auto device_ids = parse_deviceids();
        std::string backend = "";
#ifdef XH2_ENABLED
        backend = "Xh2HalBackend";
#endif
        if (device_ids.size() == 1 && multi_device_model_) {
            // 单卡模型，但是模型是双卡模型，需要将device_ids 扩展为{0,1}
            device_ids = {0, 1};
            LLAMA_LOG_WARN("dual card model, need to set device_ids to {0,1}");
        } else if (device_ids.size() == 2 && !multi_device_model_) {
            // 模型是单卡模型，需要将device_ids 缩小为{0}
            device_ids = {0};
            LLAMA_LOG_WARN(
                "single card model, set device_ids size is 2, need to "
                "resize device_ids to {0}");
        }
        tcim::DevManager dev_manager =
            tcim::DevManager::Create(device_ids, backend);
        wm_ = tcim::Module::WeightManager::CreateWeightManager(dev_manager);
    }
    tcim::Status status;
    auto infer_model = std::make_shared<tcim::Module>();
    auto option = tcim::Module::Option(wm_);
    option.SetDummyTensors(dummy_tensor_names);
    option.EnableLazyMode(enable_lazy_mode_);
    if (prefill) {
        if (!load_from_file)
            status = infer_model->LoadModel(raw_data, size, option);
        else {
            std::string prefill_name((char *)raw_data, size);
            string_replace_once(prefill_name, "-00001-", "-00002-");
            LLAMA_LOG_INFO("LoadModel prefill_name: %s\n",
                           prefill_name.c_str());
            status = infer_model->LoadModel(prefill_name, option);
        }
    } else {
        if (!load_from_file)
            status = infer_model->LoadModel(raw_data, size, option);
        else {
            std::string decoder_name((char *)raw_data, size);
            string_replace_once(decoder_name, "-00001-", "-00003-");
            LLAMA_LOG_INFO("LoadModel decoder_name: %s\n",
                           decoder_name.c_str());
            status = infer_model->LoadModel(decoder_name, option);
        }
    }
    if (status != tcim::Status::OK) {
        LLAMA_LOG_ERROR("LoadModel failed: prefill == %d\n", prefill);
        return nullptr;
    }
    return infer_model;
}

std::shared_ptr<tcim::Module> HouMoLLModel::Impl::create_infer_engine(
    const std::string &file_name, size_t offset, size_t size,
    std::vector<std::string> dummy_tensor_names) {
    if (wm_.GetInitStatus() != tcim::Status::OK) {
        std::string backend = "";
#ifdef XH2_ENABLED
        backend = "Xh2HalBackend";
#endif
        auto device_ids = parse_deviceids();
        if (device_ids.size() == 1 && multi_device_model_) {
            // 单卡模型，但是模型是双卡模型，需要将device_ids 扩展为{0,1}
            device_ids = {0, 1};
            LLAMA_LOG_WARN("dual card model, need to set device_ids to {0,1}");
        } else if (device_ids.size() == 2 && !multi_device_model_) {
            // 模型是单卡模型，需要将device_ids 缩小为{0}
            device_ids = {0};
            LLAMA_LOG_WARN(
                "single card model, set device_ids size is 2, need to "
                "resize device_ids to {0}");
        }
        tcim::DevManager dev_manager =
            tcim::DevManager::Create(device_ids, backend);
        wm_ = tcim::Module::WeightManager::CreateWeightManager(dev_manager);
    }
    tcim::Status status;
    auto infer_model = std::make_shared<tcim::Module>();
    auto option = tcim::Module::Option(wm_);
    option.SetDummyTensors(dummy_tensor_names);
    option.EnableLazyMode(enable_lazy_mode_);
#ifdef XH2_ENABLED
    option.SetModelOffset(offset, size);
#else
    (void) offset;
    (void) size;
#endif
    status = infer_model->LoadModel(file_name, option);
    if (status != tcim::Status::OK) {
        LLAMA_LOG_ERROR("LoadModel failed: status == %d\n", (int)status);
        return nullptr;
    }
    return infer_model;
}

bool HouMoLLModel::Impl::houmo_load(const std::string &file_name,
                                    bool use_mmap) {
    struct ggml_context *ctx_data = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_data,
    };
    struct gguf_context *ctx = gguf_init_from_file(file_name.c_str(), params);
    parser_gguf_parameters(ctx);
    LLAMA_LOG_INFO("HouMoLLModel::houmo_load after gguf_init_from_file\n");
    // 从gguf文件中加载hmmodels
    const std::regex kPrefillPattern("^prefill(_\\d+)?\\.hmm$");
    const std::regex kDecodePattern("^decode(r)?(_\\d+)?\\.hmm$");
    // 环境变量中是否开启了LLAMA_ENABLE_LAZY_MODE
    enable_lazy_mode_ = false;
    if (const char *env = std::getenv("LLAMA_ENABLE_LAZY_MODE")) {
        enable_lazy_mode_ = (std::string(env) == "1");
    }
    LLAMA_LOG_INFO("LLAMA_ENABLE_LAZY_MODE: %d\n", enable_lazy_mode_);
    std::vector<std::string> dummy_tensor_names;
    // 防止更高效使用内存，这个将prefill 设置为dummy_tensor_names
    for (int idx = 0; idx < n_blocks_; idx++) {
        std::string k_cache_name = "model_layers_" + std::to_string(idx) +
                                    "_self_attn_kcache_input";
        std::string v_cache_name = "model_layers_" + std::to_string(idx) +
                                    "_self_attn_vcache_input";
        dummy_tensor_names.push_back(k_cache_name);
        dummy_tensor_names.push_back(v_cache_name);
    }
#ifndef XH2_ENABLED
    use_mmap = false;
#endif
    // 1. 加载 embedding
    const int n_tensors = gguf_get_n_tensors(ctx);
    for (int i = 0; i < n_tensors; ++i) {
        const char *name = gguf_get_tensor_name(ctx, i);
        LLAMA_LOG_INFO("tensor[%d]: name = %s\n", i, name);
        const size_t size = gguf_get_tensor_size(ctx, i);
        size_t offset =
            gguf_get_tensor_offset(ctx, i) + gguf_get_meta_size(ctx);
        if (strncmp(name, "quant_embedding.bin",
                    strlen("quant_embedding.bin")) == 0) {
            embedding_layer_ =
                std::make_shared<HoumoEmbeddingLayer>(file_name, offset, size);
            if (embedding_layer_->get_dtype() == EmbeddingDtype::INT16) {
                xh1_ = true;
            } else {
                xh1_ = false;
            }
        } else if (std::regex_match(name, kPrefillPattern)) {
            // load prefill
            if (use_mmap) {
                prefill_model_ = create_infer_engine(file_name, offset, size, dummy_tensor_names);
            } else {
                std::vector<char> buffer_vec;
                buffer_vec.resize(size);
                read_tensor_to_buffer(file_name, offset, size, buffer_vec);
                prefill_model_ =
                    create_infer_engine(true, buffer_vec.data(), size, false, dummy_tensor_names);
            }
            if (prefill_model_ == nullptr) {
                LLAMA_LOG_ERROR("%s: failed to load prefill.hmm\n", __func__);
                return false;
            }
            LLAMA_LOG_INFO("%s: succeed to load prefill.hmm\n", __func__);
        } else if (std::regex_match(name, kDecodePattern)) {
            // load decode
            if (use_mmap) {
                decode_model_ = create_infer_engine(file_name, offset, size);
            } else {
                std::vector<char> buffer_vec;
                buffer_vec.resize(size);
                read_tensor_to_buffer(file_name, offset, size, buffer_vec);
                decode_model_ =
                    create_infer_engine(false, buffer_vec.data(), size, false);
            }
            if (decode_model_ == nullptr) {
                LLAMA_LOG_ERROR("%s: failed to load decode.hmm\n", __func__);
                return false;
            }
            LLAMA_LOG_INFO("%s: succeed to load decode.hmm\n", __func__);
        } else {
            LLAMA_LOG_INFO("tensor[%d]: name = %s, size = %zu, skipped\n", i, name,
                           size);
        }
    }
    if (n_split_count_ == 3) {
        prefill_model_ =
            create_infer_engine(true, const_cast<char *>(file_name.c_str()),
                                file_name.length(), true);
        decode_model_ =
            create_infer_engine(false, const_cast<char *>(file_name.c_str()),
                                file_name.length(), true);
    }
    ggml_free(ctx_data);
    gguf_free(ctx);
    if (prefill_model_ == nullptr || embedding_layer_ == nullptr) {
        LLAMA_LOG_ERROR("%s: failed to load model. load "
                        "prefill=%d,load decode=%d\n",
                        __func__, (prefill_model_ == nullptr),
                        (embedding_layer_ == nullptr));
        return false;
    }

    return true;
}

void HouMoLLModel::Impl::model_inout_init(int seq_max) {
    // init prefill
    prefill_input_init();
    // 填充prefill 的output
    prefill_output_init();
    // init decode
    decode_input_init();
    // 填充decode 的output
    decode_output_init();
    // init kv cache
    if (seq_max > kSeqMax) {
        LLAMA_LOG_ERROR("%s: seq_max > %d not supported, set seq_max to %d\n",
                        __func__, seq_max, kSeqMax);
        seq_max = kSeqMax;
    }
    int size = (seq_max + n_prefill_batch_ - 1) / n_prefill_batch_;
    size = std::max(size, n_decode_batch_);
    initialize_kv_cache(size);

    LLAMA_LOG_INFO("%s: HMM graph init success\n", __func__);
    LLAMA_LOG_INFO("prefill_length_ = %d\n context_length_ = %d\n "
                   "n_blocks_ = %d\n n_embd_ = %ld\n n_prefill_batch_ = %d\n "
                   "n_decode_batch_ = "
                   "%d\n kv_cache_size = %zu\n need_set_kv_cache_ = "
                   "%d\n seq_max = %d\n",
                   prefill_length_, context_length_, n_blocks_, n_embd_,
                   n_prefill_batch_, n_decode_batch_, kv_cache_.size(),
                   need_set_kv_cache_, seq_max);
}

void HouMoLLModel::Impl::prefill_input_init() {
    if (prefill_model_ == nullptr) {
        return;
    }
    for (size_t idx = 0; idx < prefill_model_->GetInputNum(); idx++) {
        auto input_name = prefill_model_->GetInputName(idx);
        LLAMA_LOG_DEBUG("%s: prefill input_name: %s\n", __func__,
                        input_name.c_str());
        if (input_name.find("past_key_cache_") != std::string::npos ||
            input_name.find("_self_attn_vcache") != std::string::npos ||
            input_name.find("_self_attn_kcache") != std::string::npos) {
            continue;
        }
        auto input_info =
            prefill_model_->GetInputInfo(input_name).AsContiguous();
        auto shape = input_info.Shape();
        for (size_t i = 0; i < shape.size(); i++) {
            LLAMA_LOG_INFO("%s: prefill input_shape: %ld, type = %d\n",
                           input_name.c_str(), shape[i],
                           (int)input_info.DataType());
        }
        auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
        prefill_input_map_.insert(
            std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
    }
    // 填充prefill_length_ & n_prefill_batch_ only support batch_size = 1
    n_prefill_batch_ = 1;
    prefill_length_ =
        prefill_input_map_[prefill_model_->GetInputName(0)].Info().Shape()[1] *
        prefill_input_map_[prefill_model_->GetInputName(0)].Info().Shape()[0];
}

void HouMoLLModel::Impl::prefill_output_init() {
    if (prefill_model_ == nullptr) {
        return;
    }
    int output_num = prefill_model_->GetOutputNum();
    for (int idx = 0; idx < output_num; idx++) {
        auto output_name = prefill_model_->GetOutputName(idx);
        LLAMA_LOG_INFO("%s: prefill output_name: %s\n", __func__,
                       output_name.c_str());
        auto output_info = prefill_model_->GetOutputInfo(output_name)
                               .AsContiguous()
                               .AsType(tcim::DataType::FLOAT32);
        auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
        prefill_output_map_.insert(
            std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
        prefill_output_names_.push_back(output_name);
    }
}

void HouMoLLModel::Impl::decode_input_init() {
    if (decode_model_ == nullptr) {
        return;
    }
    size_t input_num_decode = decode_model_->GetInputNum();
    for (size_t idx = 0; idx < input_num_decode; idx++) {
        auto input_name = decode_model_->GetInputName(idx);
        LLAMA_LOG_DEBUG("%s: decode_model input_name: %s\n", __func__,
                        input_name.c_str());
        if (input_name.find("past_key_cache_") != std::string::npos ||
            input_name.find("_self_attn_vcache") != std::string::npos ||
            input_name.find("_self_attn_kcache") != std::string::npos) {
            continue;
        }
        if (input_name.find("valid_length") != std::string::npos) {
            decode_valid_length_names_.push_back(input_name);
        }
        if (input_name.find("current_length") != std::string::npos) {
            decode_current_length_names_.push_back(input_name);
        }
        LLAMA_LOG_INFO("%s: decode input_name: %s\n", __func__,
                       input_name.c_str());
        auto input_info =
            decode_model_->GetInputInfo(input_name).AsContiguous();
        auto shape = input_info.Shape();
        for (size_t i = 0; i < shape.size(); i++) {
            LLAMA_LOG_INFO("%s: decode input_shape: %ld\n", __func__, shape[i]);
        }
        auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
        decode_input_map_.insert(
            std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
    }
    if (input_num_decode > 0)
        n_decode_batch_ =
            decode_input_map_[decode_model_->GetInputName(0)].Info().Shape()[0];
}

void HouMoLLModel::Impl::decode_output_init() {
    // 填充decode 的output
    if (decode_model_ == nullptr) {
        return;
    }
    int output_num_decode = decode_model_->GetOutputNum();
    for (int idx = 0; idx < output_num_decode; idx++) {
        auto output_name = decode_model_->GetOutputName(idx);
        auto output_info = decode_model_->GetOutputInfo(output_name)
                               .AsContiguous()
                               .AsType(tcim::DataType::FLOAT32);
        auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
        decode_output_map_.insert(
            std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
        decode_output_names_.push_back(output_name);
    }
}

void HouMoLLModel::Impl::parser_gguf_parameters(struct gguf_context *ctx) {
    auto key_id = gguf_find_key(ctx, kGeneralArch.c_str());
    if (key_id < 0) {
        LLAMA_LOG_INFO("%s: failed to find key %s in gguf file\n", __func__,
                        kGeneralArch.c_str());
        return;
    }
    model_arch_ = gguf_get_val_str(ctx, key_id);
    std::string key = model_arch_ + kBlockCountEx;
    key_id = gguf_find_key(ctx, key.c_str());
    if (key_id < 0) {
        LLAMA_LOG_INFO("%s: not found key %s in gguf file\n", __func__,
                        key.c_str());
    } else {
        n_blocks_ = gguf_get_val_u32(ctx, key_id);
    }
    key = model_arch_ + kEmbeddingLengthEx;
    key_id = gguf_find_key(ctx, key.c_str());
    if (key_id < 0) {
        LLAMA_LOG_INFO("%s: not found key %s in gguf file\n", __func__,
                        key.c_str());
    } else {
        n_embd_ = gguf_get_val_u32(ctx, key_id);
    }
    auto key_n_split = gguf_find_key(ctx, kSpliteCount.c_str());
    if (key_n_split < 0) {
        LLAMA_LOG_INFO("%s: not found key %s in gguf file\n", __func__,
                       kSpliteCount.c_str());
    } else {
        n_split_count_ = gguf_get_val_i32(ctx, key_n_split);
    }
    // 解析hmm.info
    key_id = gguf_find_key(ctx, kHmmInfo.c_str());
    if (key_id < 0) {
        LLAMA_LOG_INFO("%s: not found key %s in gguf file\n", __func__,
                        kHmmInfo.c_str());
        return;
    }
    auto hmm_info_str = gguf_get_val_str(ctx, key_id);
    LLAMA_LOG_INFO("%s: hmm_info: %s\n", __func__, hmm_info_str);
    try {
        // 解析hmm.info
        auto hmm_info = json::parse(hmm_info_str);
        if (hmm_info.contains("target")) {
            if (hmm_info["target"] == "xh1") {
                xh1_ = true;
            }
        }
        if (hmm_info.contains("device_num")) {
            if (hmm_info["device_num"] > 1) {
                multi_device_model_ = true;
            }
        }

    } catch (const json::parse_error &e) {
        LLAMA_LOG_ERROR("%s: failed to parse hmm_info: %s\n", __func__,
                        e.what());
        return;
    }
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

    LLAMA_LOG_INFO("fread mem offset:%zu  size:%zu to buffer\r\n", offset,
                   size);

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
        LLAMA_LOG_ERROR(
            "%s: offset is out of range, offset: %zu, int32_vec.size(): %zu\n",
            __func__, offset, int32_vec.size());
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
void HouMoLLModel::Impl::initialize_kv_cache(int size) {
    if (decode_model_ == nullptr) {
        // 非decode模型, 无需初始化kv_cache
        std::string name =
            prefill_model_->GetInputName(prefill_input_map_.size());
        context_length_ =
            std::max(prefill_model_->GetInputInfo(name).Shape()[2],
                     prefill_model_->GetInputInfo(name).Shape()[3]);
        return;
    }
    // FIXME: M30 QWENVL 的命名和M50 不一致, 量化后续会统一后移除
    std::string kv_prefix_name = "model_layers_";
    for (size_t i = 0; i < prefill_model_->GetInputNum(); i++) {
        auto input_name = prefill_model_->GetInputName(i);
        if (input_name.find("kcache") != std::string::npos ||
            input_name.find("vcache") != std::string::npos) {
            // 判断是否有model_language_model_layers_ 前缀
            if (input_name.find("model_language_model_layers_") !=
                std::string::npos) {
                kv_prefix_name = "model_language_model_layers_";
            }
            break;
        }
    }
    // 检查是否有.hmcc.format 后缀. 开发编译的模型会添加后缀
    std::string ends_with = "";
    if (!xh1_ && prefill_model_->GetInputName(0).find(".hmcc.format") !=
                     std::string::npos) {
        ends_with = ".hmcc.format";
    }

    for (int i = 0; i < size; i++) {
        kv_cache_item item;
        item.seq_id = i;
        for (int idx = 0; idx < n_blocks_; idx++) {
            std::string k_cache_name = kv_prefix_name + std::to_string(idx) +
                                       "_self_attn_kcache_input" + ends_with;
            std::string v_cache_name = kv_prefix_name + std::to_string(idx) +
                                       "_self_attn_vcache_input" + ends_with;
            item.names.push_back(k_cache_name);
            item.names.push_back(v_cache_name);
            if (n_decode_batch_ > 1) {
                if (i > 0) {
                    k_cache_name = k_cache_name + "_batch" + std::to_string(i);
                    v_cache_name = v_cache_name + "_batch" + std::to_string(i);
                }
            }
            auto kcache = decode_model_->GetDevInput(k_cache_name);
            auto vcache = decode_model_->GetDevInput(v_cache_name);
            if (n_decode_batch_ > 1 || size == 1) {
                item.tensors.push_back(kcache);
                item.tensors.push_back(vcache);
            } else {
                item.tensors.push_back(kcache.Clone());
                item.tensors.push_back(vcache.Clone());
                need_set_kv_cache_ = 1;
            }
            context_length_ = std::max(
                kcache.Info().Shape()[kcache.Info().Shape().size() - 1],
                kcache.Info().Shape()[kcache.Info().Shape().size() - 2]);
        }
        kv_cache_[i] = item;
    }
}

void HouMoLLModel::Impl::set_kv_cache_for_prefill(int seq_id) {
    if (kv_cache_.find(seq_id) == kv_cache_.end()) {
        return;
    }
    auto &item = kv_cache_[seq_id];
    for (size_t idx = 0; idx < item.tensors.size(); idx++) {
        std::string name = item.names[idx];
        auto tensor = item.tensors[idx];
        prefill_model_->SetInput(name, tensor);
    }
}

void HouMoLLModel::Impl::set_kv_cache_for_decode(
    const std::vector<int> &seq_ids) {
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
}

int HouMoLLModel::Impl::set_input_for_decode(std::vector<llama_token> &batches,
                                             std::vector<int> &seq_ids) {
    std::vector<llama_token> input_ids(n_decode_batch_, 0);
    for (size_t i = 0; i < seq_ids.size(); i++) {
        int batch_id = seq_ids[i];
        // 单batch模式下所有序列共用batch 0
        if (n_decode_batch_ == 1)
            batch_id = 0;
        input_ids[batch_id] = batches[i];
    }
    auto input_data = decode_input_map_.at(decode_model_->GetInputName(0));
    try {
        embedding_layer_->get_embedding_batch(input_ids, input_data.Data());
    } catch (...) {
        std::cerr << "error get embedding batch\n";
        return -1;
    }
    decode_model_->SetInput(decode_model_->GetInputName(0), input_data);
    return 0;
}

int HouMoLLModel::Impl::set_xlenght_for_decode(std::vector<int> &seq_ids) {
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
    if (decode_valid_length_names_.size() == 1 && n_decode_batch_ > 1) {
        // 量化输出的是多batch的模型
        auto decode_valid_length =
            decode_input_map_.at(decode_valid_length_names_[0]);
        set_input_data(decode_valid_length, valid_length_vec);
        decode_model_->SetInput(decode_valid_length_names_[0],
                                decode_valid_length);
        std::vector<int32_t> current_length_vec(n_decode_batch_, 1);
        auto decode_current_length =
            decode_input_map_.at(decode_current_length_names_[0]);
        set_input_data(decode_current_length, current_length_vec);
        decode_model_->SetInput(decode_current_length_names_[0],
                                decode_current_length);
    } else {
        // 工具链编译成多batch模型,特征如下
        // current_length, current_length_batch1, ...current_length_batchx
        // valid_length, valid_length_batch1, ...valid_length_batchx
        if (decode_current_length_names_.size() != (size_t)n_decode_batch_ ||
            decode_valid_length_names_.size() != (size_t)n_decode_batch_) {
            LLAMA_LOG_ERROR("%s: decode_current_length_names_.size() = %zu != "
                            "n_decode_batch_ = %d\n",
                            __func__, decode_current_length_names_.size(),
                            n_decode_batch_);
            return -1;
        }

        for (size_t i = 0; i < valid_length_vec.size(); i++) {
            // valid_length
            std::string name = decode_valid_length_names_[i];
            auto decode_valid_length = decode_input_map_.at(name);
            set_input_data(decode_valid_length, 1, valid_length_vec[i]);
            auto status = decode_model_->SetInput(name, decode_valid_length);
            if (status != tcim::Status::OK) {
                LLAMA_LOG_ERROR("%s: failed to set input %s\n", __func__,
                                name.c_str());
                return -1;
            }
            // current_length
            name = decode_current_length_names_[i];
            auto current_valid_length = decode_input_map_.at(name);
            set_input_data(current_valid_length, 1, 1);
            LLAMA_LOG_DEBUG("name = %s, current_valid_length: %d\n",
                            name.c_str(), 1);
            status = decode_model_->SetInput(name, current_valid_length);
            if (status != tcim::Status::OK) {
                LLAMA_LOG_ERROR("%s: failed to set input %s\n", __func__,
                                name.c_str());
                return -1;
            }

            // FIXME: @guoxing.xu 文本模型(for moe 测试模型,模型修改后可以移除)
            if (decode_input_map_.count(decode_model_->GetInputName(3)) > 0) {
                auto position_ids =
                    decode_input_map_.at(decode_model_->GetInputName(3));
                std::string name = decode_model_->GetInputName(3);
                if (i > 0 && n_decode_batch_ > 1) {
                    name += "_batch" + std::to_string(i);
                }
                set_input_data(position_ids, 1, valid_length_vec[i]);
                auto status = decode_model_->SetInput(name, position_ids);
                if (status != tcim::Status::OK) {
                    LLAMA_LOG_ERROR("%s: failed to set input %s\n", __func__,
                                    name.c_str());
                    return -1;
                }
            }
        }
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
bool HouMoLLModel::houmo_load(const std::string &file_name, bool use_mmap) {

    struct ggml_context *ctx_data = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_data,
    };
    struct gguf_context *ctx = gguf_init_from_file(file_name.c_str(), params);
    auto key_id = gguf_find_key(ctx, "general.architecture");
    if (key_id < 0) {
        return false;
    }
    std::string arch = gguf_get_val_str(ctx, key_id);

    if (impl_ == nullptr) {
        if (arch.find("vl") != std::string::npos) {
            impl_ = new HouMoVLLModel();
        } else if (arch.find("gpt-oss") != std::string::npos) {
            impl_ = new HouMoGPTOSS();
        } else if (arch.find("bert") != std::string::npos) {
            impl_ = new HouMoEmbed();
        } else if (arch.find("qwen") != std::string::npos) {
            impl_ = new HoumoQwenLLM();
        } else {
            LLAMA_LOG_ERROR("%s: unknown architecture %s\n", __func__,
                            arch.c_str());
            exit(1);
            return false;
        }
    }

    return impl_->houmo_load(file_name, use_mmap);
}

void HouMoLLModel::houmo_init(houmo_memory_i *memory, int seq_max) {
    impl_->houmo_init(memory, seq_max);
}

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

int HouMoLLModel::houmo_embedding(
    const std::map<int32_t, std::vector<llama_token>> &batches,
    std::map<int, float *> &embeddings) {
    return impl_->houmo_embedding(batches, embeddings);
}
bool HouMoLLModel::is_embedding() {
    return impl_->is_embedding();
}