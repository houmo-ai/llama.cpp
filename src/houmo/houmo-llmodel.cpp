#include "houmo-llmodel.h"
#include "ggml-hmutils.h"
#include "gguf.h"
#include "houmo-embed.h"
#include "houmo-gptoss.h"
#include "houmo-qwenllm.h"
#include "houmo-qwen35llm.h"
#include "houmo-qwen35mtp.h"
#include "houmo-vllmodel.h"
#include "houmo-glmocrmodel.h"
#include "houmo-gemma4.h"
#include "llama-arch.h"
#include "llama-batch.h"
#include "llama-impl.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cctype>
#include <cstring>
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

static bool qwen35_has_mtp_tensors(const llama_model_loader &loader) {
    if (!loader.metadata_ptr) {
        return false;
    }
    gguf_context *ctx = loader.metadata_ptr.get();
    const int n_tensors = gguf_get_n_tensors(ctx);
    const std::regex kPrefillMtpPattern("^prefill_mtp(_\\d+)?\\.hmm$");
    const std::regex kDecodeMtpPattern("^decode(r)?_mtp(_\\d+)?\\.hmm$");
    for (int i = 0; i < n_tensors; ++i) {
        const char *name = gguf_get_tensor_name(ctx, i);
        if (name == nullptr) {
            continue;
        }
        if (std::regex_match(name, kPrefillMtpPattern) ||
            std::regex_match(name, kDecodeMtpPattern)) {
            return true;
        }
    }
    return false;
}

static bool is_kv_cache_io_name(const std::string &name) {
    return name.find("past_key_cache") != std::string::npos ||
           name.find("past_value_cache") != std::string::npos ||
           name.find("_self_attn_vcache") != std::string::npos ||
           name.find("_self_attn_kcache") != std::string::npos ||
           name.find("conv_cache_out") != std::string::npos ||
           name.find("recurrent_state_out") != std::string::npos ||
           name.find("past_conv_cache") != std::string::npos ||
           name.find("past_recurrent_state") != std::string::npos;
}

std::shared_ptr<tcim::Module> HouMoLLModel::Impl::create_infer_engine(
    bool prefill, void *raw_data, int64_t size, bool load_from_file,
    std::vector<std::string> dummy_tensor_names) {
    if (wm_.GetInitStatus() != tcim::Status::OK) {
        std::string backend = "";
        backend = "Xh2HalBackend";
        tcim::DevManager dev_manager =
            tcim::DevManager::Create(device_ids_, backend);
        wm_ = tcim::Module::WeightManager::CreateWeightManager(dev_manager);
    }
    tcim::Status status;
    auto infer_model = std::make_shared<tcim::Module>();
    auto option = tcim::Module::Option(wm_);
    option.SetDummyTensors(dummy_tensor_names);
    option.EnableHostLazyLoading(enable_lazy_mode_);
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
        backend = "Xh2HalBackend";
        tcim::DevManager dev_manager =
            tcim::DevManager::Create(device_ids_, backend);
        wm_ = tcim::Module::WeightManager::CreateWeightManager(dev_manager);
    }
    tcim::Status status;
    auto infer_model = std::make_shared<tcim::Module>();
    auto option = tcim::Module::Option(wm_);
    option.SetDummyTensors(dummy_tensor_names);
    option.EnableHostLazyLoading(enable_lazy_mode_);
    option.SetModelOffset(offset, size);
    status = infer_model->LoadModel(file_name, option);
    if (status != tcim::Status::OK) {
        LLAMA_LOG_ERROR("LoadModel failed: status == %d\n", (int)status);
        return nullptr;
    }
    return infer_model;
}

bool HouMoLLModel::Impl::houmo_load(llama_model_loader &ml, std::vector<int> device_ids) {
    std::string file_name = ml.fname_;
    arch_name = ml.arch_name;
    device_ids_ = device_ids;
    gguf_context *ctx = ml.metadata_ptr.get();
    parser_gguf_parameters(ctx, ml.arch_name);
    const bool load_mtp_modules = wants_mtp_modules();
    // 从 gguf 文件中加载 hmmodels
    const std::regex kPrefillPattern("^prefill(_\\d+)?\\.hmm$");
    const std::regex kPrefillMtpPattern("^prefill_mtp(_\\d+)?\\.hmm$");
    const std::regex kDecodePattern("^decode(r)?(_\\d+)?\\.hmm$");
    const std::regex kDecodeMtpPattern("^decode(r)?_mtp(_\\d+)?\\.hmm$");
    // 环境变量中是否开启了LLAMA_ENABLE_LAZY_MODE
    enable_lazy_mode_ = !ml.use_mmap;
    if (const char *env = std::getenv("LLAMA_ENABLE_LAZY_MODE")) {
        enable_lazy_mode_ = (std::string(env) == "1");
    }
    LLAMA_LOG_INFO("LLAMA_ENABLE_LAZY_MODE: %d\n", enable_lazy_mode_);
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
    parse_deviceids();
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
        } else if (std::regex_match(name, kPrefillMtpPattern)) {
            if (!load_mtp_modules) {
                LLAMA_LOG_INFO("%s: skip prefill_mtp.hmm for target impl\n", __func__);
                continue;
            }
            // MTP 草稿 prefill：与普通 prefill 共用 prefill_model_ 槽位
            // （同一个 HouMoLLModel 实例只承载一个 gguf，要么是 target 要么是 MTP）。
            prefill_model_ = create_infer_engine(file_name, offset, size,
                                                 dummy_tensor_names);
            if (prefill_model_ == nullptr) {
                LLAMA_LOG_ERROR("%s: failed to load prefill_mtp.hmm\n", __func__);
                return false;
            }
            LLAMA_LOG_INFO("%s: succeed to load prefill_mtp.hmm (MTP draft module)\n",
                           __func__);
        } else if (std::regex_match(name, kPrefillPattern)) {
            if (load_mtp_modules) {
                LLAMA_LOG_INFO("%s: skip prefill.hmm for MTP impl\n", __func__);
                continue;
            }
            // load prefill
            prefill_model_ = create_infer_engine(file_name, offset, size,
                                                 dummy_tensor_names);
            if (prefill_model_ == nullptr) {
                LLAMA_LOG_ERROR("%s: failed to load prefill.hmm\n", __func__);
                return false;
            }
            LLAMA_LOG_INFO("%s: succeed to load prefill.hmm\n", __func__);
        } else if (std::regex_match(name, kDecodeMtpPattern)) {
            if (!load_mtp_modules) {
                LLAMA_LOG_INFO("%s: skip decode_mtp.hmm for target impl\n", __func__);
                continue;
            }
            decode_model_ = create_infer_engine(file_name, offset, size);
            if (decode_model_ == nullptr) {
                LLAMA_LOG_ERROR("%s: failed to load decode_mtp.hmm\n", __func__);
                return false;
            }
            LLAMA_LOG_INFO("%s: succeed to load decode_mtp.hmm (MTP draft module)\n",
                           __func__);
        } else if (std::regex_match(name, kDecodePattern)) {
            if (load_mtp_modules) {
                LLAMA_LOG_INFO("%s: skip decode.hmm for MTP impl\n", __func__);
                continue;
            }
            // load decode
            decode_model_ = create_infer_engine(file_name, offset, size);

            if (decode_model_ == nullptr) {
                LLAMA_LOG_ERROR("%s: failed to load decode.hmm\n", __func__);
                return false;
            }
            LLAMA_LOG_INFO("%s: succeed to load decode.hmm\n", __func__);
        } else {
            LLAMA_LOG_INFO("tensor[%d]: name = %s, size = %zu, skipped\n", i,
                           name, size);
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
    n_vocab_ = embedding_layer_->get_vocab_size();
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
    if (is_mtp_model()) {
        // MTP draft HMM does not expose/use the target model KV cache fields.
        // Its state is driven by hidden/input_embedding/position length inputs,
        // so the generic per-layer KV initializer must not probe target KV names.
        kv_cache_.clear();
        need_set_kv_cache_ = 0;
        LLAMA_LOG_INFO("%s: skip generic KV cache init for MTP model\n", __func__);
    } else {
        initialize_kv_cache(size);
    }
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
    if (prefill_input_map_.size() > 0) {
        return;
    }
    for (size_t idx = 0; idx < prefill_model_->GetInputNum(); idx++) {
        auto input_name = prefill_model_->GetInputName(idx);
        LLAMA_LOG_DEBUG("%s: prefill input_name: %s\n", __func__,
                        input_name.c_str());
        if (input_name.find("past_key_cache_") != std::string::npos) {
            kCacheKeyWord = "past_key_cache_";
        }
        if (is_kv_cache_io_name(input_name)) {
            continue;
        }
        input_names_.push_back(input_name);
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
    if (prefill_output_map_.size() > 0) {
        return;
    }
    int output_num = prefill_model_->GetOutputNum();
    for (int idx = 0; idx < output_num; idx++) {
        auto output_name = prefill_model_->GetOutputName(idx);
        LLAMA_LOG_INFO("%s: prefill output_name: %s\n", __func__,
                       output_name.c_str());
        if (is_kv_cache_io_name(output_name)) {
            continue;
        }
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
    if (decode_input_map_.size() > 0) {
        return;
    }

    size_t input_num_decode = decode_model_->GetInputNum();
    for (size_t idx = 0; idx < input_num_decode; idx++) {
        auto input_name = decode_model_->GetInputName(idx);
        LLAMA_LOG_DEBUG("%s: decode_model input_name: %s\n", __func__,
                        input_name.c_str());
        if (input_name.find("past_key_cache_") != std::string::npos) {
            kCacheKeyWord = "past_key_cache_";
        }
        if (is_kv_cache_io_name(input_name)) {
            continue;
        }
        if (input_name.find("valid_length") != std::string::npos ||
            input_name.find("past_seq_length") != std::string::npos) {
            decode_valid_length_names_.push_back(input_name);
        }
        if (input_name.find("current_length") != std::string::npos ||
            input_name.find("current_input_length") != std::string::npos) {
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
    n_decode_batch_ = decode_input_map_[decode_model_->GetInputName(0)].Info().Shape()[0];
}

void HouMoLLModel::Impl::decode_output_init() {
    // 填充decode 的output
    if (decode_model_ == nullptr) {
        return;
    }
    if (decode_output_map_.size() > 0) {
        return;
    }
    int output_num_decode = decode_model_->GetOutputNum();
    for (int idx = 0; idx < output_num_decode; idx++) {
        auto output_name = decode_model_->GetOutputName(idx);
        LLAMA_LOG_INFO("%s: decode output_name: %s\n", __func__,
                       output_name.c_str());
        if (is_kv_cache_io_name(output_name)) {
            continue;
        }
        auto output_info = decode_model_->GetOutputInfo(output_name)
                               .AsContiguous()
                               .AsType(tcim::DataType::FLOAT32);
        auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
        decode_output_map_.insert(
            std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
        decode_output_names_.push_back(output_name);
    }
}

void HouMoLLModel::Impl::parser_gguf_parameters(struct gguf_context *ctx,
                                                std::string &model_arch) {
    std::string key = model_arch + kBlockCountEx;
    auto key_id = gguf_find_key(ctx, key.c_str());
    if (key_id < 0) {
        LLAMA_LOG_INFO("%s: not found key %s in gguf file\n", __func__,
                       key.c_str());
    } else {
        n_blocks_ = gguf_get_val_u32(ctx, key_id);
    }
    key = model_arch + kEmbeddingLengthEx;
    key_id = gguf_find_key(ctx, key.c_str());
    if (key_id < 0) {
        LLAMA_LOG_INFO("%s: not found key %s in gguf file\n", __func__,
                       key.c_str());
    } else {
        n_embd_ = gguf_get_val_u32(ctx, key_id);
    }
    key = "tokenizer.ggml.tokens";
    key_id = gguf_find_key(ctx, key.c_str());
    if (key_id >= 0) {
        n_vocab_ = (int64_t) gguf_get_arr_n(ctx, key_id);
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
        if (hmm_info.contains("device_num")) {
            device_num_ = hmm_info["device_num"];
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
        LLAMA_LOG_ERROR("Failed to open file: %s\n", file_name.c_str());
        return;
    }

    // 移动到偏移位置
    input_file.seekg(offset, std::ios::beg);
    if (input_file.fail()) {
        LLAMA_LOG_ERROR("Failed to seek to offset: %zu in file: %s\n", offset,
                        file_name.c_str());
        input_file.close();
        return;
    }

    // 读取数据
    input_file.read(buffer.data(), size);
    if (input_file.fail()) {
        LLAMA_LOG_ERROR("Failed to read %zu bytes from file: %s\n", size,
                        file_name.c_str());
        input_file.close();
        return;
    }
}

void HouMoLLModel::Impl::parse_deviceids() {
    const char *device_id_str = std::getenv("HOUMO_DEVICE_ID");
    if (device_id_str == nullptr) {
        if (device_ids_.size() == 0)
            device_ids_.push_back(0);
    } else {
        LLAMA_LOG_INFO("HOUMO_DEVICE_ID: %s\n", device_id_str);
        device_ids_.clear();
        std::string deviceList(device_id_str);
        std::istringstream iss(deviceList);
        std::string token;

        while (std::getline(iss, token, ',')) {
            try {
                // 转换为整数并添加到列表
                device_ids_.push_back(std::stoi(token));
            } catch (const std::invalid_argument &) {
                LLAMA_LOG_ERROR("Invalid device ID: %s\n", token.c_str());
            } catch (const std::out_of_range &) {
                LLAMA_LOG_ERROR("Device ID out of range: %s\n", token.c_str());
            }
        }
    }
    // 校验设置的是否合法
    if (device_num_ == 2 && device_ids_.size() == 1) {
        // 单卡模型，但是模型是双卡模型，需要将device_ids 扩展为{0,1}
        device_ids_ = {0, 1};
        LLAMA_LOG_WARN("dual card model, need to set device_ids to {0,1}");
    } else if (device_num_ == 1 && device_ids_.size() > 1) {
        // 模型是单卡模型，需要将device_ids 缩小为{0}
        device_ids_ = {0};
        LLAMA_LOG_WARN("single card model, set device_ids size > 1, need to "
                       "resize device_ids to {0}");
    } else if (device_num_ == 4 && device_ids_.size() == 1) {
        // 模型是四卡模型，需要将device_ids 扩展为{0,1,2,3}
        device_ids_ = {0, 1, 2, 3};
        LLAMA_LOG_WARN("4 card model, need to set device_ids to {0,1,2,3}");
    } else {
        LLAMA_LOG_INFO("run models in %zu cards", device_ids_.size());
    }
}

void HouMoLLModel::Impl::set_input_data(tcim::Tensor &tensor, int batch_size,
                                        int value) {
    if (tensor.Info().DataType() == tcim::DataType::INT16) {
        std::vector<int16_t> data_vec(batch_size, value);
        tensor.Buffer().CopyFromHost(data_vec.data(), tensor.MemSize());
    } else if (tensor.Info().DataType() == tcim::DataType::INT32) {
        std::vector<int32_t> data_vec(batch_size, value);
        tensor.Buffer().CopyFromHost(data_vec.data(), tensor.MemSize());
    } else if (tensor.Info().DataType() == tcim::DataType::INT8) {
        std::vector<int8_t> data_vec(batch_size, value);
        tensor.Buffer().CopyFromHost(data_vec.data(), tensor.MemSize());
    } else if (tensor.Info().DataType() == tcim::DataType::FLOAT16) {
        std::vector<uint16_t> data_vec(batch_size, value);
        tensor.Buffer().CopyFromHost(data_vec.data(), tensor.MemSize());
    } else {
        throw std::invalid_argument("Unsupported tensor data type");
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
    std::string kv_prefix_name = kCacheKeyWord;
    // 检查是否有.hmcc.format 后缀. 开发编译的模型会添加后缀
    std::string ends_with = "";
    if (prefill_model_->GetInputName(0).find(".hmcc.format") !=
                     std::string::npos) {
        ends_with = ".hmcc.format";
    }
    int kv_cache_count = n_blocks_;
    if (arch_name.find("qwen35") != std::string::npos) {
        kv_cache_count = n_blocks_ / 4;
    }
    if ((size_t)size != kv_cache_.size()) {
        LLAMA_LOG_INFO("%s: initialize kv cache, size = %d\n", __func__, size);
        kv_cache_.clear();
    }
    for (int i = 0; i < size; i++) {
        kv_cache_item item;
        item.seq_id = i;
        for (int idx = 0; idx < kv_cache_count; idx++) {
            std::string k_cache_name = kv_prefix_name + std::to_string(idx) +
                                       "_self_attn_kcache_input" + ends_with;
            std::string v_cache_name = kv_prefix_name + std::to_string(idx) +
                                       "_self_attn_vcache_input" + ends_with;
            if (kCacheKeyWord == "past_key_cache_") {
                k_cache_name = "past_key_cache_" + std::to_string(idx);
                v_cache_name = "past_value_cache_" + std::to_string(idx);
            }
            item.names.push_back(k_cache_name);
            item.names.push_back(v_cache_name);
            LLAMA_LOG_DEBUG("%s: kv_cache_name: %s, n_blocks = %d\n", __func__, k_cache_name.c_str(), n_blocks_);
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
        LLAMA_LOG_ERROR("%s: failed to get embedding batch\n", __func__);
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
            // LLAMA_LOG_DEBUG("name = %s, current_valid_length: %d\n",
            //                 name.c_str(), 1);
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
void HouMoLLModel::Impl::lora_init(std::string file_name, void *adapter) {
    struct lora_item item;
    item.adapter = adapter;
    item.scale = 0.0;
    lora_map[file_name] = item;
    LLAMA_LOG_INFO("lora FILE_name IS %s\n", file_name.c_str());
}
void HouMoLLModel::Impl::lora_set_scale(void *adapter, float scale) {
    for (auto iter = lora_map.begin(); iter != lora_map.end(); ++iter) {
        if (iter->second.adapter == adapter) {
            iter->second.scale = scale;
        }
    }
}
void HouMoLLModel::Impl::lora_clear() {
    // scale set 0
    for (auto iter = lora_map.begin(); iter != lora_map.end(); ++iter) {
        iter->second.scale = 0;
    }
}

// HouMoLLModel constructor and destructor
HouMoLLModel::HouMoLLModel() : impl_(nullptr), impl_mtp_(nullptr) {}

HouMoLLModel::~HouMoLLModel() {
    if (impl_mtp_) {
        delete impl_mtp_;
    }
    if (impl_ && impl_ != impl_mtp_) {
        delete impl_;
    }
}

// Forward method calls to the implementation
bool HouMoLLModel::houmo_load(llama_model_loader &ml, std::vector<int> device_ids) {
    const bool is_qwen35_arch =
        ml.arch_name.find("qwen35") != std::string::npos ||
        ml.arch_name.find("qwen3.5") != std::string::npos ||
        ml.arch_name.find("qwen3_5") != std::string::npos;

    if (impl_ == nullptr) {
        bool has_prefill_hmm = false;
        bool has_decode_hmm = false;
        const std::regex kPrefillPattern("^prefill(_\\d+)?\\.hmm$");
        const std::regex kDecodePattern("^decode(r)?(_\\d+)?\\.hmm$");
        const int n_tensors = gguf_get_n_tensors(ml.metadata_ptr.get());
        for (int i = 0; i < n_tensors; ++i) {
            const char *name = gguf_get_tensor_name(ml.metadata_ptr.get(), i);
            if (name == nullptr) {
                continue;
            }
            has_prefill_hmm = has_prefill_hmm || std::regex_match(name, kPrefillPattern);
            has_decode_hmm = has_decode_hmm || std::regex_match(name, kDecodePattern);
        }
        std::string general_name;
        const int general_name_key = gguf_find_key(ml.metadata_ptr.get(), "general.name");
        if (general_name_key >= 0) {
            general_name = gguf_get_val_str(ml.metadata_ptr.get(), general_name_key);
            std::transform(general_name.begin(), general_name.end(), general_name.begin(),
                           [](unsigned char c) { return (char) std::tolower(c); });
        }
        const bool is_qwen = ml.arch_name.find("qwen") != std::string::npos;
        const auto name_has_any = [&general_name](std::initializer_list<const char *> needles) {
            for (const char * needle : needles) {
                if (general_name.find(needle) != std::string::npos) {
                    return true;
                }
            }
            return false;
        };
        const bool is_glm_asr =
            name_has_any({"glm-asr", "glmasr", "glm asr"});
        const bool is_glm_ocr =
            name_has_any({"glm-ocr", "glmocr", "glm ocr"});
        const bool is_qwen_asr_force_aligner =
            is_qwen && has_prefill_hmm && !has_decode_hmm &&
            name_has_any({"forcealigner", "forcedaligner", "forced aligner"});
        if (ml.arch_name.find("gemma4") != std::string::npos) {
            impl_ = new HouMoGemma4();
        } else if (is_glm_ocr) {
            impl_ = new HouMoGLMOCRModel();
        } else if (ml.arch_name.find("vl") != std::string::npos) {
            impl_ = new HouMoVLLModel();
        } else if (ml.arch_name.find("gpt-oss") != std::string::npos) {
            impl_ = new HouMoGPTOSS();
        } else if (ml.arch_name.find("bert") != std::string::npos) {
            impl_ = new HouMoEmbed();
        } else if (is_qwen35_arch) {
            if (qwen35_has_mtp_tensors(ml)) {
                impl_ = new HoumoQwen35LLM();
                impl_mtp_ = new HoumoQwen35MTP();
                LLAMA_LOG_INFO("%s: qwen3.5 gguf has MTP tensors, loading target + MTP impl\n",
                               __func__);

                if (!impl_->houmo_load(ml, device_ids)) {
                    LLAMA_LOG_WARN("%s: failed to load qwen3.5 target impl\n", __func__);
                    delete impl_;
                    impl_ = nullptr;
                }
                if (!impl_mtp_->houmo_load(ml, device_ids)) {
                    LLAMA_LOG_ERROR("%s: failed to load qwen3.5 MTP impl\n", __func__);
                    return false;
                }
                if (impl_ == nullptr) impl_ = impl_mtp_; // MTP impl 包含 target impl 的功能, 直接使用MTP impl
                return true;
            } else {
                impl_ = new HoumoQwen35LLM();
                LLAMA_LOG_INFO("%s: qwen3.5 gguf has no MTP tensors, using HoumoQwen35LLM\n",
                               __func__);
            }
        } else if (is_glm_asr || is_qwen) {
            impl_ = new HoumoQwenLLM();
        } else {
            LLAMA_LOG_ERROR("%s: unknown architecture %s\n", __func__,
                            ml.arch_name.c_str());
            exit(1);
            return false;
        }
        impl_->set_asr_force_aligner(is_qwen_asr_force_aligner);
        if (is_qwen_asr_force_aligner) {
            LLAMA_LOG_INFO("%s: detected Qwen ASR ForceAligner prefill-only model\n", __func__);
        }
    }

    return impl_->houmo_load(ml, device_ids);
}

void HouMoLLModel::houmo_init(houmo_memory_i *memory, int seq_max,
                              ggml_abort_callback abort_callback,
                              void *abort_callback_data,
                              bool is_mtp) {
    if (is_mtp) {
        if (impl_mtp_ == nullptr) {
            LLAMA_LOG_ERROR("%s: requested MTP ctx but no MTP impl loaded\n", __func__);
            return;
        }
        impl_mtp_->houmo_init(memory, seq_max, abort_callback, abort_callback_data);
    } else {
        impl_->houmo_init(memory, seq_max, abort_callback, abort_callback_data);
    }
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

int HouMoLLModel::houmo_asr_force_align(std::vector<llama_token> &tokens, int seq_id,
                                        std::vector<float *> &logits_out) {
    return impl_->houmo_asr_force_align(tokens, seq_id, logits_out);
}

int HouMoLLModel::houmo_verify(std::vector<llama_token> &tokens, int seq_id,
                               std::vector<float *> &logits_out) {
    return impl_->houmo_verify(tokens, seq_id, logits_out);
}

int HouMoLLModel::houmo_verify_accept(int seq_id, uint16_t n_accepted) {
    return impl_->houmo_verify_accept(seq_id, n_accepted);
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
bool HouMoLLModel::is_embedding() { return impl_->is_embedding(); }
bool HouMoLLModel::is_asr_force_aligner() { return impl_->is_asr_force_aligner(); }
bool HouMoLLModel::has_decode_model() { return impl_->has_decode_model(); }

// ---- MTP forwarders ----
bool HouMoLLModel::is_mtp_model() const {
    return impl_ ? impl_->is_mtp_model() : false;
}

bool HouMoLLModel::has_mtp_modules() const {
    return impl_mtp_ != nullptr || (impl_ && impl_->has_mtp_modules());
}

int HouMoLLModel::houmo_decode_mtp(const float *h_in,
                                   const std::vector<llama_token> &tokens,
                                   int seq_id,
                                   std::vector<float *> &logits_out,
                                   float *h_out,
                                   int32_t start_pos) {
    return impl_mtp_ ? impl_mtp_->houmo_decode_mtp(h_in, tokens, seq_id, logits_out, h_out,
                                   start_pos) : -1;
}

void HouMoLLModel::set_pre_norm_hidden_dst(float *dst, int n_rows) {
    if (impl_) {
        impl_->set_pre_norm_hidden_dst(dst, n_rows);
    }
}

void HouMoLLModel::lora_init(std::string file_name, void *adapter) {
    impl_->lora_init(file_name, adapter);
}
void HouMoLLModel::lora_set_scale(void *adapter, float scale) {
    impl_->lora_set_scale(adapter, scale);
}
void HouMoLLModel::lora_clear() { impl_->lora_clear(); }
