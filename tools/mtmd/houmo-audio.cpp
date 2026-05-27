#include "houmo-audio.h"

#include "ggml.h"
#include "gguf.h"
#include "clip-impl.h"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

void HoumoAudio::read_tensor_to_buffer(const std::string &file_name,
                                       size_t offset, size_t size,
                                       std::vector<char> &buffer) {
    std::ifstream input_file(file_name, std::ios::binary);
    if (!input_file.is_open()) {
        return;
    }
    input_file.seekg(offset, std::ios::beg);
    if (input_file.fail()) {
        input_file.close();
        return;
    }

    input_file.read(buffer.data(), size);
    if (input_file.fail()) {
        input_file.close();
        return;
    }
}

bool HoumoAudio::houmo_load(const std::string &file_name, std::vector<int> device_ids) {
    struct ggml_context *ctx_data = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_data,
    };
    device_ids_ = device_ids;
    struct gguf_context *ctx = gguf_init_from_file(file_name.c_str(), params);
    if (ctx == nullptr) {
        LOG_ERR("%s: failed to open gguf file %s\n", __func__, file_name.c_str());
        return false;
    }

    std::string backend = "Xh2HalBackend";
    parse_deviceids();

    const int general_name_key = gguf_find_key(ctx, "general.name");
    if (general_name_key >= 0) {
        std::string general_name = gguf_get_val_str(ctx, general_name_key);
        std::transform(general_name.begin(), general_name.end(), general_name.begin(),
                       [](unsigned char c) { return (char) std::tolower(c); });
        is_force_aligner_ =
            general_name.find("forcealigner") != std::string::npos ||
            general_name.find("forcedaligner") != std::string::npos ||
            general_name.find("forced aligner") != std::string::npos;
    }

    const int n_tensors = gguf_get_n_tensors(ctx);
    for (int i = 0; i < n_tensors; ++i) {
        const char *name = gguf_get_tensor_name(ctx, i);
        LOG_INF("tensor[%d]: name = %s\n", i, name);
        if (std::string(name).find("encode.hmm") == std::string::npos) {
            continue;
        }
        const size_t size = gguf_get_tensor_size(ctx, i);
        size_t offset =
            gguf_get_tensor_offset(ctx, i) + gguf_get_meta_size(ctx);
        std::vector<char> buffer(size);
        read_tensor_to_buffer(file_name, offset, size, buffer);

        tcim::DevManager dev_manager =
            tcim::DevManager::Create(device_ids_, backend);
        auto wm = tcim::Module::WeightManager::CreateWeightManager(dev_manager);
        auto option = tcim::Module::Option(wm);

        module_ = std::make_shared<tcim::Module>();
        auto status = module_->LoadModel(buffer.data(), size, option);
        if (status != tcim::Status::OK) {
            LOG_ERR("%s: failed to load encode.hmm\n", __func__);
            ggml_free(ctx_data);
            gguf_free(ctx);
            return false;
        }
        break;
    }

    ggml_free(ctx_data);
    gguf_free(ctx);

    if (module_ == nullptr) {
        LOG_ERR("%s: failed to find encode.hmm in %s\n", __func__, file_name.c_str());
        return false;
    }
    return true;
}

void HoumoAudio::parse_deviceids() {
    const char *device_id_str = std::getenv("HOUMO_DEVICE_ID");
    if (device_id_str == nullptr) {
        if (device_ids_.size() == 0) {
            device_ids_.push_back(0);
        }
        return;
    }

    std::string deviceList(device_id_str);
    std::istringstream iss(deviceList);
    std::string token;
    device_ids_.clear();
    while (std::getline(iss, token, ',')) {
        try {
            device_ids_.push_back(std::stoi(token));
        } catch (const std::invalid_argument &) {
            LOG_ERR("Invalid device ID: %s\n", token.c_str());
        } catch (const std::out_of_range &) {
            LOG_ERR("Device ID out of range: %s\n", token.c_str());
        }
    }
}

bool HoumoQwen3Audio::init(const std::string &model_path,
                           std::vector<int> device_ids) {
    if (!houmo_load(model_path, device_ids)) {
        LOG_ERR("Failed to load HoumoQwen3Audio model from %s\n",
                model_path.c_str());
        return false;
    }
    int input_size = module_->GetInputNum();
    for (auto i = 0; i < input_size; ++i) {
        auto input_name = module_->GetInputName(i);
        LOG_INF("%s: input_name: %s\n", __func__, input_name.c_str());
        auto input_info = module_->GetInputInfo(input_name).AsContiguous();
        if (i == 0 && input_info.Shape().size() == 3) {
            audio_input_name_ = input_name;
            audio_n_mel_ = static_cast<int32_t>(input_info.Shape()[1]);
            audio_n_frames_ = static_cast<int32_t>(input_info.Shape()[2]);
        } else if (input_name.find("feature_lens") != std::string::npos) {
            audio_feature_lens_name_ = input_name;
        }
        auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
        input_tensors_.insert(
            std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
    }
    int out_size = module_->GetOutputNum();
    for (auto i = 0; i < out_size; ++i) {
        auto output_name = module_->GetOutputName(i);
        LOG_INF("%s: output_name: %s\n", __func__, output_name.c_str());
        auto output_info = module_->GetOutputInfo(output_name).AsContiguous();
        if (i == 0 && output_info.Shape().size() >= 3) {
            output_tokens_ = static_cast<int32_t>(output_info.Shape()[1]);
            LOG_INF("%s: output_tokens: %d\n", __func__, output_tokens_);
        }
        output_names_.push_back(output_name);
    }

    if (audio_input_name_.empty()) {
        audio_input_name_ = module_->GetInputName(0);
    }

    if (output_tokens_ <= 0) {
        output_tokens_ = 195;
    }

    return true;
}

bool HoumoQwen3Audio::encoding(const std::vector<float> &input_data,
                               std::vector<float> &output_data,
                               int out_offset,
                               int32_t feature_lens) {
    auto input_tensor = input_tensors_[audio_input_name_];

    // Qwen3-ASR encoder expects fixed shape [1, n_mel, n_frames], e.g. [1,128,1500].
    const int32_t n_mel = std::max<int32_t>(audio_n_mel_, 1);
    const int32_t n_frames = std::max<int32_t>(audio_n_frames_, 1);
    const size_t expected_count = (size_t) n_mel * (size_t) n_frames;
    std::vector<float> padded_input(expected_count, 0.0f);
    const size_t input_frames = audio_n_mel_ > 0 ? input_data.size() / (size_t) audio_n_mel_ : 0;
    const size_t copy_frames = std::min<size_t>(input_frames, (size_t) n_frames);
    if (copy_frames > 0) {
        for (int32_t m = 0; m < n_mel; ++m) {
            const size_t src_off = (size_t) m * input_frames;
            const size_t dst_off = (size_t) m * (size_t) n_frames;
            if (src_off >= input_data.size() || dst_off >= padded_input.size()) {
                break;
            }
            const size_t frames_this_row = std::min(copy_frames, input_data.size() - src_off);
            std::memcpy(padded_input.data() + dst_off, input_data.data() + src_off, frames_this_row * sizeof(float));
        }
    }

    tcim::Tensor input_fp32_tensor = tcim::Tensor::CreateHostTensor(
        input_tensor.Info().AsContiguous().AsType(tcim::DataType::FLOAT32));
    input_fp32_tensor.Buffer().CopyFromHost(padded_input.data(),
                                            input_fp32_tensor.MemSize());
    input_fp32_tensor.CastTo(input_tensor);
    module_->SetInput(audio_input_name_, input_tensor);

    if (!audio_feature_lens_name_.empty() && input_tensors_.count(audio_feature_lens_name_)) {
        auto feature_lens_tensor = input_tensors_[audio_feature_lens_name_];
        int32_t feature_lens_value = feature_lens > 0 ? feature_lens : 0;
        if (feature_lens_value <= 0 && audio_n_mel_ > 0) {
            feature_lens_value = (int32_t) (input_data.size() / (size_t) audio_n_mel_);
        }
        if (feature_lens_value <= 0) {
            feature_lens_value = audio_n_frames_ > 0 ? audio_n_frames_ : 1500;
        }
        if (audio_n_frames_ > 0) {
            feature_lens_value = std::min<int32_t>(feature_lens_value, audio_n_frames_);
        }
        feature_lens_value = std::max<int32_t>(feature_lens_value, 1);
        LOG_INF("%s: feature_lens=%d, input_frames=%zu, padded_frames=%d\n",
                __func__, feature_lens_value,
                input_frames,
                audio_n_frames_);
        feature_lens_tensor.Buffer().CopyFromHost(&feature_lens_value, sizeof(feature_lens_value));
        module_->SetInput(audio_feature_lens_name_, feature_lens_tensor);
    }

    module_->Run();
    module_->Sync();
    for (auto &name : output_names_) {
        tcim::Tensor output_tensor = module_->GetOutput(name);
        const size_t offset = (size_t) std::max(out_offset, 0);
        const size_t remaining = output_data.size() > offset ? (output_data.size() - offset) * sizeof(float) : 0;
        const size_t copy_bytes = std::min((size_t) output_tensor.MemSize(), remaining);
        if (copy_bytes > 0) {
            memcpy(output_data.data() + offset, output_tensor.Buffer().Data(), copy_bytes);
        }
    }
    return true;
}
