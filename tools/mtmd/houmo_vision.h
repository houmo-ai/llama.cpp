#pragma once
#include "clip-impl.h"
#include "tcim/tcim_runtime.h"
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
class HoumoVision {
  public:
    HoumoVision() = default;
    ~HoumoVision();
    bool init(const std::string &model_path);
    bool encoding(const std::vector<float> &input_data,
                  std::vector<float> &output_data);
    int32_t image_width() const;
    int32_t image_height() const;
    int32_t image_channels() const;
    bool image_resize(clip_image_u8_ptr &input, clip_image_f32_ptr &output);

  private:
    bool houmo_load(const std::string &file_name);
    void rgb_to_yuv444(const std::vector<float> &bgr_data, int width,
                       int height, std::vector<uint8_t> &yuv444sp_data);
    void read_tensor_to_buffer(const std::string &file_name, size_t offset,
                               size_t size, std::vector<char> &buffer);
    /**
     * 分辨率固定，则index和mask可以预计算
     * 否则需要根据分辨率计算
     * porting from clip.cpp
     */
    void set_windows_attention();

  private:
    std::shared_ptr<tcim::Module> module_;
    std::map<std::string, tcim::Tensor> input_tensors_;
    std::map<std::string, tcim::Tensor> output_tensors_;
    uint32_t image_width_ = 644;
    uint32_t image_height_ = 364;
    uint32_t image_channels_ = 3;
    std::string input_name_;
    uint32_t patch_size_ = 14;
};