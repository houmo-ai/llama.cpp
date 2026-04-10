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
    virtual ~HoumoVision() = default;
    virtual bool init(const std::string &model_path, std::vector<int> device_ids) = 0;
    virtual bool encoding(const std::vector<float> &input_data,
                          std::vector<float> &output_data, int out_offset = 0) = 0;
    int16_t image_width() const  {
        return image_width_;
    }
    int16_t image_height() const {
        return image_height_;
    }
    int16_t image_channels() const {
        return image_channels_;
    }
    int16_t batch_size() const {
        return batch_size_;
    }
    bool image_resize(clip_image_u8_ptr &input, clip_image_f32_ptr &output);

  protected:
    bool houmo_load(const std::string &file_name, std::vector<int> device_ids);
    void rgb_to_yuv444(const std::vector<float> &bgr_data, int width,
                       int height, std::vector<uint8_t> &yuv444sp_data);
    void read_tensor_to_buffer(const std::string &file_name, size_t offset,
                               size_t size, std::vector<char> &buffer);
    void rgb_to_nchw(const std::vector<float> &rgb_data);
    void resize_and_pad_image(const clip_image_u8 &image, clip_image_u8 &dst,
                              const clip_image_size &target_resolution,
                              std::array<uint8_t, 3> pad_color);
    bool bicubic_resize(const clip_image_u8 &img, clip_image_u8 &dst,
                        int target_width, int target_height);
    bool area_resize(const clip_image_u8 &img, clip_image_u8 &dst,
                     int target_width, int target_height);
    void fill_header(std::vector<float> &output_data);

    void parse_deviceids();
  protected:
    std::shared_ptr<tcim::Module> module_;
    std::map<std::string, tcim::Tensor> input_tensors_;
    std::vector<std::string> output_names_;
    std::string kPatchSizeKey = "clip.vision.patch_size";
    std::vector<float> nchw_data_;
    uint16_t image_width_ = 644;
    uint16_t image_height_ = 364;
    uint16_t image_channels_ = 3;
    uint16_t batch_size_ = 1;
    uint16_t image_frames_ = 2;
    std::string input_name_;
    uint32_t patch_size_ = 14;
    std::vector<int> device_ids_;
};
/*
*@ 基于qwen架构的视觉模型, 支持Qwen2.5vl, Qwen3vl模型
*/
class HoumoQwenVision : public HoumoVision {
  public:
    HoumoQwenVision(): HoumoVision() {}
    ~HoumoQwenVision() override = default;
    virtual bool init(const std::string &model_path, std::vector<int> device_ids) override;
    virtual bool encoding(const std::vector<float> &input_data,
                          std::vector<float> &output_data, int out_offset = 0) override;

  private:
    /**
     * 分辨率固定，则index和mask可以预计算
     * 否则需要根据分辨率计算
     * porting from clip.cpp
     */
    void set_windows_attention();
};

/*
*@ 基于minicpm的视觉模型, 支持M50模型
*/
class HoumoMinicpmVision : public HoumoVision {
  public:
    HoumoMinicpmVision() : HoumoVision() {}
    ~HoumoMinicpmVision() override = default;
    virtual bool init(const std::string &model_path, std::vector<int> device_ids) override;
    virtual bool encoding(const std::vector<float> &input_data,
                          std::vector<float> &output_data, int out_offset = 0) override;
  private:
    /**
     * set resampler for minicpmv
     * 否则需要根据分辨率计算
     * porting from clip.cpp
     */
    void set_attention_resampler();
    // ---------------------------------------------------------
    // 核心算法辅助函数
    // ---------------------------------------------------------
    void precompute_global_pos_embed();
    void compute_1d_sincos(float* out, int length, int dim);
    void reorder_hwc_to_target(const float* input_hwc, float* output_buffer);
    // ---------------------------------------------------------
    // 成员变量
    // ---------------------------------------------------------
    // 预计算的 70x70 全局位置编码表 (FP32)
    std::vector<float> global_pos_embed_table_;
    // 常量定义
    const int MAX_GRID_H = 70;
    const int MAX_GRID_W = 70;
    int32_t n_embedding_ = 3584;
};