#pragma once

#include "tcim/tcim_runtime.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

class HoumoAudio {
  public:
    HoumoAudio() = default;
    virtual ~HoumoAudio() = default;
    virtual bool init(const std::string &model_path, std::vector<int> device_ids) = 0;
    virtual bool encoding(const std::vector<float> &input_data,
                          std::vector<float> &output_data, int out_offset = 0) = 0;
    int32_t output_tokens() const {
        return output_tokens_;
    }
    int32_t audio_n_frames() const {
        return audio_n_frames_;
    }

  protected:
    bool houmo_load(const std::string &file_name, std::vector<int> device_ids);
    void read_tensor_to_buffer(const std::string &file_name, size_t offset,
                               size_t size, std::vector<char> &buffer);
    void parse_deviceids();

  protected:
    std::shared_ptr<tcim::Module> module_;
    std::map<std::string, tcim::Tensor> input_tensors_;
    std::vector<std::string> output_names_;
    std::vector<int> device_ids_;
    int32_t output_tokens_ = 0;
    int32_t audio_n_mel_ = 0;
    int32_t audio_n_frames_ = 0;
    std::string audio_input_name_;
    std::string audio_feature_lens_name_;
};

class HoumoQwen3Audio : public HoumoAudio {
  public:
    HoumoQwen3Audio() : HoumoAudio() {}
    ~HoumoQwen3Audio() override = default;
    virtual bool init(const std::string &model_path, std::vector<int> device_ids) override;
    virtual bool encoding(const std::vector<float> &input_data,
                          std::vector<float> &output_data, int out_offset = 0) override;
};