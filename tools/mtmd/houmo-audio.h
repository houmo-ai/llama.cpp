#pragma once

#include "tcim/tcim_runtime.h"
#include "houmo-runtime-recovery.h"

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
                          std::vector<float> &output_data,
                          int out_offset = 0,
                          int32_t feature_lens = -1) = 0;
    int32_t output_tokens() const {
        return output_tokens_;
    }
    int32_t audio_n_frames() const {
        return audio_n_frames_;
    }
    bool is_force_aligner() const {
        return is_force_aligner_;
    }
    bool is_glm_asr() const {
        return is_glm_asr_;
    }

  protected:
    bool houmo_load(const std::string &file_name, std::vector<int> device_ids);
    void read_tensor_to_buffer(const std::string &file_name, size_t offset,
                               size_t size, std::vector<char> &buffer);
    void parse_deviceids();

  protected:
    std::shared_ptr<tcim::Module> module_;
    houmo_runtime_recovery_handle_ptr recovery_handle_;
    std::map<std::string, tcim::Tensor> input_tensors_;
    std::vector<std::string> output_names_;
    std::vector<int> device_ids_;
    int32_t output_tokens_ = 0;
    int32_t audio_n_mel_ = 0;
    int32_t audio_n_frames_ = 0;
    bool is_force_aligner_ = false;
    bool is_glm_asr_ = false;
    std::string audio_input_name_;
    std::string audio_feature_lens_name_;
};

class HoumoQwen3Audio : public HoumoAudio {
  public:
    HoumoQwen3Audio() : HoumoAudio() {}
    ~HoumoQwen3Audio() override = default;
    virtual bool init(const std::string &model_path, std::vector<int> device_ids) override;
    virtual bool encoding(const std::vector<float> &input_data,
                          std::vector<float> &output_data,
                          int out_offset = 0,
                          int32_t feature_lens = -1) override;
};

class HoumoGLMAsrAudio : public HoumoAudio {
  public:
    HoumoGLMAsrAudio() : HoumoAudio() {}
    ~HoumoGLMAsrAudio() override = default;
    virtual bool init(const std::string &model_path, std::vector<int> device_ids) override;
    virtual bool encoding(const std::vector<float> &input_data,
                          std::vector<float> &output_data,
                          int out_offset = 0,
                          int32_t feature_lens = -1) override;
};

class HoumoGemmaAudio : public HoumoAudio {
  public:
    HoumoGemmaAudio() : HoumoAudio() {}
    ~HoumoGemmaAudio() override = default;
    virtual bool init(const std::string &model_path, std::vector<int> device_ids) override;
    virtual bool encoding(const std::vector<float> &input_data,
                          std::vector<float> &output_data,
                          int out_offset = 0,
                          int32_t feature_lens = -1) override;

  private:
    int32_t output_tokens_from_frames(int32_t input_frames) const;
    size_t shape_num_elements(const std::vector<int64_t> & shape) const;
    void copy_mask_to_tensor(tcim::Tensor & dst_tensor, const std::vector<float> & src) const;
    std::vector<float> build_feature_mask(int32_t total_frames, int32_t valid_frames) const;
    std::vector<float> build_attention_mask(int32_t total_frames, int32_t valid_frames,
                                            const std::vector<int64_t> & shape) const;
    size_t count_valid_mask_entries(const tcim::Tensor & output_tensor) const;

  private:
    std::string audio_input_mask_name_;
    std::string audio_attention_mask_name_;
    std::string audio_output_embeds_name_;
    std::string audio_output_mask_name_;
};
