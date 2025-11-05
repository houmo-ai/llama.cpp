#include "houmo_vision.h"
#include "ggml.h"
#include "gguf.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <torch/torch.h>

static torch::Tensor floatArrayToFloat16Tensor(std::vector<float> &float_data) {
    // 1. 将float数组转换为float32张量
    torch::Tensor tensor =
        torch::from_blob(float_data.data(),           // 数据指针
                         uint64_t{float_data.size()}, // 张量形状
                         torch::kFloat32              // 原始数据类型
        );

    // 2. 转换为float16（half）类型
    torch::Tensor float16_tensor = tensor.to(torch::kHalf);
    return float16_tensor;
}

static torch::Tensor
Int16ArrayToFloat16Tensor(std::vector<int16_t> &int16_data) {
    // 1. 将float数组转换为float32张量
    torch::Tensor tensor =
        torch::from_blob(int16_data.data(),           // 数据指针
                         uint64_t{int16_data.size()}, // 张量形状
                         torch::kInt16                // 原始数据类型
        );

    // 2. 转换为float16（half）类型
    torch::Tensor float16_tensor = tensor.to(torch::kHalf);

    return float16_tensor;
}
HoumoVision::~HoumoVision() {
    // do nothing
}
void HoumoVision::read_tensor_to_buffer(const std::string &file_name,
                                        size_t offset, size_t size,
                                        std::vector<char> &buffer) {
    // 打开输入文件
    std::ifstream input_file(file_name, std::ios::binary);
    if (!input_file.is_open()) {
        return;
    }
    // 移动到偏移位置
    input_file.seekg(offset, std::ios::beg);
    if (input_file.fail()) {
        input_file.close();
        return;
    }

    // 读取数据
    input_file.read(buffer.data(), size);
    if (input_file.fail()) {
        input_file.close();
        return;
    }
}

void HoumoVision::rgb_to_nchw(const std::vector<float> &rgb_data) {

    // 计算预期的数据大小
    const size_t expected_size = image_width_ * image_height_ * image_channels_ * batch_size_ * image_frames_;
    nchw_data_.resize(expected_size);
    bool only_one_frame = (rgb_data.size() == image_width_ * image_height_ * image_channels_);
    // 转换逻辑：N→C→F→H→W
    const size_t frame_size =image_height_ * image_width_ * image_channels_;
    for (size_t b = 0; b < batch_size_; ++b) {// 批量维度（1）
        for (size_t c = 0; c < image_channels_; ++c) {// 通道维度（3：R/G/B）
            for (size_t f = 0; f < image_frames_; ++f) {// 帧维度（2）
                // 计算当前帧在输入数据中的起始偏移（输入是按帧拼接的HWC）
                const size_t img_index = only_one_frame ? 0 : f;
                const size_t frame_offset = img_index * frame_size;

                for (size_t h = 0; h < image_height_; ++h) {// 高度维度（364）
                    for (size_t w = 0; w < image_width_; ++w) { // 宽度维度（644）
                        // 输入HWC格式的索引：[帧偏移] + [行偏移] + [列×通道] + [通道]
                        const size_t hwc_index = frame_offset
                                                + h * image_width_ * image_channels_
                                                + w * image_channels_
                                                + c;

                        // 输出NCFHW格式的索引：[N偏移] + [C偏移] + [F偏移] + [H偏移] + [W]
                        const size_t ncfhw_index =
                                    b * image_channels_ * image_frames_ * image_height_ * image_width_
                                    + c * image_frames_ * image_height_ * image_width_
                                    + f * image_height_ * image_width_
                                    + h * image_width_
                                    + w;
                        // 赋值
                        nchw_data_[ncfhw_index] = rgb_data[hwc_index];
                    }
                }
            }
        }
    }
}

bool HoumoVision::houmo_load(const std::string &file_name) {
    struct ggml_context *ctx_data = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_data,
    };
    struct gguf_context *ctx = gguf_init_from_file(file_name.c_str(), params);
    // 从gguf文件中加载hmmodels
    const char *device_id_str = std::getenv("HOUMO_DEVICE_ID");
    int device_id = 0;
    if (device_id_str != nullptr) {
        device_id = std::stoi(device_id_str);
    }
    auto wm = tcim::Module::WeightManager::CreateWeightManager(device_id);
    auto option_decoder = tcim::Module::Option(wm);

    auto option = tcim::Module::Option(wm);
    // 1. 加载 embedding
    const int n_tensors = gguf_get_n_tensors(ctx);
    for (int i = 0; i < n_tensors; ++i) {
        const char *name = gguf_get_tensor_name(ctx, i);
        LOG_INF("tensor[%d]: name = %s\n", i, name);
        const size_t size = gguf_get_tensor_size(ctx, i);
        size_t offset =
            gguf_get_tensor_offset(ctx, i) + gguf_get_meta_size(ctx);
        std::vector<char> buffer(size);
        read_tensor_to_buffer(file_name, offset, size, buffer);
        module_ = std::make_shared<tcim::Module>();
        auto status = module_->LoadModel(buffer.data(), size, option);
        if (status != tcim::Status::OK) {
            LOG_ERR("%s: failed to load prefill.hmm\n", __func__);
            return false;
        }
        break;
    }
    ggml_free(ctx_data);
    gguf_free(ctx);
    return true;
}

bool HoumoVision::init(const std::string &model_path) {
    if (!houmo_load(model_path)) {
        LOG_ERR("Failed to load HoumoVision model from %s\n",
                model_path.c_str());
        return false;
    }
    int input_size = module_->GetInputNum();
    for (auto i = 0; i < input_size; ++i) {
        auto input_name = module_->GetInputName(i);
        LOG_INF("%s: input_name: %s\n", __func__, input_name.c_str());
        auto input_info = module_->GetInputInfo(input_name).AsContiguous();
        if (input_info.Shape().size() == 4) {
            image_width_ = input_info.Shape()[3];
            image_height_ = input_info.Shape()[2];
            image_channels_ = input_info.Shape()[1];
            batch_size_ = input_info.Shape()[0];
            image_frames_ = 1;
            input_name_ = input_name;
        } else if (input_info.Shape().size() == 5) {
            image_width_ = input_info.Shape()[4];
            image_height_ = input_info.Shape()[3];
            image_frames_ = input_info.Shape()[2];
            image_channels_ = input_info.Shape()[1];
            batch_size_ = input_info.Shape()[0];
            input_name_ = input_name;
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
        auto input_tensor = tcim::Tensor::CreateHostTensor(output_info);
        output_tensors_.insert(
            std::pair<std::string, tcim::Tensor>(output_name, input_tensor));
    }
    set_windows_attention();

    return true;
}

void HoumoVision::rgb_to_yuv444(const std::vector<float> &rgb_data, int width,
                                int height,
                                std::vector<uint8_t> &yuv444sp_data) {
    int pixel_count = width * height;
    // YUV444SP总大小：Y(1*H*W) + UV(2*H*W)
    yuv444sp_data.resize(pixel_count * 3);

    for (int i = 0; i < pixel_count; ++i) {
        float r = rgb_data[i * 3];
        float g = rgb_data[i * 3 + 1];
        float b = rgb_data[i * 3 + 2];

        // BT.601转换公式（输入BGR为[0,1]范围）
        float y = 0.299f * r + 0.587f * g + 0.114f * b;
        float u = -0.14713f * r - 0.28886f * g + 0.436f * b + 128;
        float v = 0.615f * r - 0.51499f * g - 0.10001f * b + 128;

        // 映射到8位范围（BT.601标准）
        uint8_t y_uint8 = static_cast<uint8_t>(std::clamp(y, 0.0f, 255.0f));
        uint8_t u_uint8 = static_cast<uint8_t>(std::clamp(u, 0.0f, 255.0f));
        uint8_t v_uint8 = static_cast<uint8_t>(std::clamp(v, 0.0f, 255.0f));

        // YUV444SP存储：先存所有Y，再存UV交错
        yuv444sp_data[i] = y_uint8; // Y平面（索引0 ~ pixel_count-1）
        yuv444sp_data[pixel_count + 2 * i] = u_uint8; // UV平面（U在偶数索引）
        yuv444sp_data[pixel_count + 2 * i + 1] =
            v_uint8; // UV平面（V在奇数索引）
    }
}

bool HoumoVision::encoding(const std::vector<float> &input_data,
                           std::vector<float> &output_data) {
    auto input_tensor = input_tensors_[input_name_];
    if (input_tensor.Info().DataType() == tcim::DataType::UINT8 ||
        input_tensor.Info().DataType() == tcim::DataType::INT8) {
        // xh1  input_data 转成uint8_t, 再转成yuv44sp
        std::vector<uint8_t> yuv444_data;
        rgb_to_yuv444(input_data, image_width_, image_height_, yuv444_data);
        input_tensor.Buffer().CopyFromHost(yuv444_data.data(),
                                           input_tensor.MemSize());
    } else {
        rgb_to_nchw(input_data);
        // xh2  input_data 转成float16
        auto input_fp16_tensor = floatArrayToFloat16Tensor(nchw_data_);
        input_tensor.Buffer().CopyFromHost(
            input_fp16_tensor.data_ptr<at::Half>(), input_tensor.MemSize());
    }
    auto window_index_tensor = input_tensors_[module_->GetInputName(1)];
    auto window_mask_tensor = input_tensors_[module_->GetInputName(2)];
    module_->SetInput(module_->GetInputName(1), window_index_tensor);
    module_->SetInput(module_->GetInputName(2), window_mask_tensor);
    module_->SetInput(input_name_, input_tensor);
    module_->Run();
    module_->Sync();
    tcim::Tensor output_tensor = module_->GetOutput(module_->GetOutputName(0));
    memcpy(output_data.data(), output_tensor.Buffer().Data(),
           output_tensor.MemSize());
    LOG_INF("houmo_vision: run model done, output_data size: %d\n",
            output_tensor.MemSize());
    return true;
}

int32_t HoumoVision::image_width() const { return image_width_; }
int32_t HoumoVision::image_height() const { return image_height_; }
int32_t HoumoVision::image_channels() const { return image_channels_; }
static inline int clip(int x, int lower, int upper) {
    return std::max(lower, std::min(x, upper));
}

bool HoumoVision::image_resize(clip_image_u8_ptr &src,
                               clip_image_f32_ptr &dst) {
    // 先resize到合适的尺寸
    clip_image_u8 resized_img;
    // bicubic_resize(*src, resized_img, image_width_, image_height_);
    const clip_image_size target_size = {static_cast<int>(image_width_), static_cast<int>(image_height_)};
    resize_and_pad_image(*src, resized_img, target_size, {114, 114, 114});
    // 再转换成f32
    dst->nx = image_width_;
    dst->ny = image_height_;
    dst->buf.resize(resized_img.buf.size());
    for (size_t i = 0; i < resized_img.buf.size(); ++i) {
        // 将uint8_t转换为float
        dst->buf[i] = static_cast<float>(resized_img.buf[i]);
    }
    return true;
}

void HoumoVision::set_windows_attention() {
    // pw * ph = number of tokens output by ViT after apply patch merger
    // ipw * ipw = number of vision token been processed inside ViT
    const int merge_ratio = 2;
    const int pw = image_width_ / patch_size_ / merge_ratio;
    const int ph = image_height_ / patch_size_ / merge_ratio;
    const int ipw = image_width_ / patch_size_;
    const int iph = image_height_ / patch_size_;
    LOG_INF("===Qwen2.5VL: image_width_=%d, patch_size_=%d, merge_ratio=%d\n",
            image_width_, patch_size_, merge_ratio);
    LOG_INF("===Qwen2.5VL: pw=%d, ph=%d, ipw=%d, iph=%d,\n", pw, ph, ipw, iph);

    std::vector<int32_t> idx(ph * pw);
    std::vector<int32_t> inv_idx(ph * pw);

    const int attn_window_size = 112;
    const int grid_window = attn_window_size / patch_size_ / merge_ratio;
    int dst = 0;
    // [num_vision_tokens, num_vision_tokens] attention mask tensor
    std::vector<int16_t> mask(pow(ipw * iph, 2),
                              std::numeric_limits<int16_t>::lowest());
    int mask_row = 0;

    for (int y = 0; y < ph; y += grid_window) {
        for (int x = 0; x < pw; x += grid_window) {
            const int win_h = std::min(grid_window, ph - y);
            const int win_w = std::min(grid_window, pw - x);
            const int dst_0 = dst;
            // group all tokens belong to the same window togather (to a
            // continue range)
            for (int dy = 0; dy < win_h; dy++) {
                for (int dx = 0; dx < win_w; dx++) {
                    const int src = (y + dy) * pw + (x + dx);
                    idx[src] = dst;
                    inv_idx[dst] = src;
                    dst++;
                }
            }

            for (int r = 0; r < win_h * win_w * merge_ratio * merge_ratio;
                 r++) {
                int row_offset = mask_row * (ipw * iph);
                std::fill(mask.begin() + row_offset +
                              (dst_0 * merge_ratio * merge_ratio),
                          mask.begin() + row_offset +
                              (dst * merge_ratio * merge_ratio),
                          0.0);
                mask_row++;
            }
        }
    }
    auto &tensor = input_tensors_[module_->GetInputName(1)];
    tensor.Buffer().CopyFromHost(inv_idx.data(), tensor.MemSize());
    module_->SetInput(module_->GetInputName(1), tensor);

    auto &mask_tensor = input_tensors_[module_->GetInputName(2)];
    if (mask_tensor.Info().DataType() == tcim::DataType::FLOAT16) {
        auto mask_fp16 = Int16ArrayToFloat16Tensor(mask);
        mask_tensor.Buffer().CopyFromHost(mask_fp16.data_ptr<at::Half>(),
                                          mask_tensor.MemSize());
    } else {
        mask_tensor.Buffer().CopyFromHost(mask.data(), mask_tensor.MemSize());
    }
    module_->SetInput(module_->GetInputName(2), mask_tensor);
}

/*更接近PIL的bicubic_resize*/
bool HoumoVision::bicubic_resize(const clip_image_u8 &img, clip_image_u8 &dst,
                                 int target_width, int target_height) {
    const int nx = img.nx;
    const int ny = img.ny;

    dst.nx = target_width;
    dst.ny = target_height;
    dst.buf.resize(3 * target_width * target_height);

    const float scale_x = static_cast<float>(nx) / target_width;
    const float scale_y = static_cast<float>(ny) / target_height;
    const float a = -0.5f; // Catmull–Rom spline (same as PIL Image.BICUBIC)

    auto cubic_weight = [a](float x) -> float {
        x = fabsf(x);
        if (x <= 1.0f)
            return (a + 2.0f) * x * x * x - (a + 3.0f) * x * x + 1.0f;
        else if (x < 2.0f)
            return a * x * x * x - 5.0f * a * x * x + 8.0f * a * x - 4.0f * a;
        else
            return 0.0f;
    };

    auto sample = [&](int px, int py, int c) -> float {
        px = std::clamp(px, 0, nx - 1);
        py = std::clamp(py, 0, ny - 1);
        return static_cast<float>(img.buf[(py * nx + px) * 3 + c]);
    };

    for (int j = 0; j < target_height; ++j) {
        float gy = (j + 0.5f) * scale_y - 0.5f;
        int y_int = static_cast<int>(floorf(gy));
        float dy = gy - y_int;

        for (int i = 0; i < target_width; ++i) {
            float gx = (i + 0.5f) * scale_x - 0.5f;
            int x_int = static_cast<int>(floorf(gx));
            float dx = gx - x_int;

            for (int c = 0; c < 3; ++c) {
                double sum = 0.0f;
                double wsum = 0.0f;
                for (int m = -1; m <= 2; ++m) {
                    float wy = cubic_weight(m - dy);
                    int yy = y_int + m;
                    for (int n = -1; n <= 2; ++n) {
                        float wx = cubic_weight(n - dx);
                        int xx = x_int + n;
                        float w = wx * wy;
                        sum += sample(xx, yy, c) * w;
                        wsum += w;
                    }
                }
                float v = (wsum != 0.0f) ? (sum / wsum) : 0.0f;
                dst.buf[(j * target_width + i) * 3 + c] =
                    static_cast<uint8_t>(std::clamp(std::round(v), 0.0f, 255.0f));
            }
        }
    }
    return true;
}

void HoumoVision::resize_and_pad_image(const clip_image_u8 &image,
                                       clip_image_u8 &dst,
                                       const clip_image_size &target_resolution,
                                       std::array<uint8_t, 3> pad_color) {
    int target_width = target_resolution.width;
    int target_height = target_resolution.height;

    float scale_w = static_cast<float>(target_width) / image.nx;
    float scale_h = static_cast<float>(target_height) / image.ny;

    int new_width, new_height;

    if (scale_w < scale_h) {
        new_width = target_width;
        new_height = std::min(static_cast<int>(std::ceil(image.ny * scale_w)),
                              target_height);
    } else {
        new_height = target_height;
        new_width = std::min(static_cast<int>(std::ceil(image.nx * scale_h)),
                             target_width);
    }
    float min_scale =
        std::min(static_cast<float>(new_width) / image.nx,
                 static_cast<float>(new_height) / image.ny);
    clip_image_u8 resized_image;
    if (min_scale < 1.0f) {
        // 缩小
        area_resize(image, resized_image, new_width, new_height);
    } else {
        // 放大
        bicubic_resize(image, resized_image, new_width, new_height);
    }
    clip_image_u8 padded_image;
    padded_image.nx = target_width;
    padded_image.ny = target_height;
    padded_image.buf.resize(3 * target_width * target_height);

    // Fill the padded image with the fill color
    for (size_t i = 0; i < padded_image.buf.size(); i += 3) {
        padded_image.buf[i] = pad_color[0];
        padded_image.buf[i + 1] = pad_color[1];
        padded_image.buf[i + 2] = pad_color[2];
    }

    // Calculate padding offsets
    int pad_x = 0;
    int pad_y = 0;

    // Copy the resized image into the top/left of the padded buffer
    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            for (int c = 0; c < 3; ++c) {
                padded_image
                    .buf[3 * ((y + pad_y) * target_width + (x + pad_x)) + c] =
                    resized_image.buf[3 * (y * new_width + x) + c];
            }
        }
    }
    dst = std::move(padded_image);
}

bool HoumoVision::area_resize(const clip_image_u8 &img, clip_image_u8 &dst,
                              int target_width, int target_height) {
    const int src_w = img.nx;
    const int src_h = img.ny;

    dst.nx = target_width;
    dst.ny = target_height;
    dst.buf.resize(3 * target_width * target_height);

    const float scale_x = static_cast<float>(src_w) / target_width;
    const float scale_y = static_cast<float>(src_h) / target_height;

    auto sample = [&](int x, int y, int c) -> float {
        x = std::clamp(x, 0, src_w - 1);
        y = std::clamp(y, 0, src_h - 1);
        return static_cast<float>(img.buf[(y * src_w + x) * 3 + c]);
    };

    // Loop over each pixel of destination
    for (int dy = 0; dy < target_height; ++dy) {
        // Corresponding source region y range
        float sy1 = dy * scale_y;
        float sy2 = sy1 + scale_y;
        int sy_start = static_cast<int>(floorf(sy1));
        int sy_end = static_cast<int>(ceilf(sy2));

        for (int dx = 0; dx < target_width; ++dx) {
            float sx1 = dx * scale_x;
            float sx2 = sx1 + scale_x;
            int sx_start = static_cast<int>(floorf(sx1));
            int sx_end = static_cast<int>(ceilf(sx2));

            for (int c = 0; c < 3; ++c) {
                float sum = 0.0f;
                float area = 0.0f;

                // Average over all relevant input pixels
                for (int sy = sy_start; sy < sy_end; ++sy) {
                    float y1 = std::max(sy1, static_cast<float>(sy));
                    float y2 = std::min(sy2, static_cast<float>(sy + 1));
                    float wy = y2 - y1;
                    if (wy <= 0) continue;

                    for (int sx = sx_start; sx < sx_end; ++sx) {
                        float x1 = std::max(sx1, static_cast<float>(sx));
                        float x2 = std::min(sx2, static_cast<float>(sx + 1));
                        float wx = x2 - x1;
                        if (wx <= 0) continue;

                        float w = wx * wy;
                        sum += sample(sx, sy, c) * w;
                        area += w;
                    }
                }

                float val = (area > 0.0f) ? (sum / area) : 0.0f;
                dst.buf[(dy * target_width + dx) * 3 + c] =
                    static_cast<uint8_t>(std::clamp(std::round(val), 0.0f, 255.0f));
            }
        }
    }

    return true;
}