#include "houmo-vision.h"
#include "ggml.h"
#include "gguf.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <array>
#include "nlohmann/json.hpp"
using json = nlohmann::ordered_json;
void reorder_hwc_to_target(const float* input_hwc, float* output_buffer);
#pragma pack(push, 1)
struct CompactHeader {
    int16_t magic;      // 16位标识
    int16_t nx;         // 16位宽度
    int16_t ny;         // 16位高度
    int16_t chunksize;  // 16位块大小
};
#pragma pack(pop)

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

bool HoumoVision::houmo_load(const std::string &file_name, std::vector<int> device_ids) {
    struct ggml_context *ctx_data = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_data,
    };
    device_ids_ = device_ids;
    struct gguf_context *ctx = gguf_init_from_file(file_name.c_str(), params);
    auto key_id = gguf_find_key(ctx, kPatchSizeKey.c_str());
    if (key_id < 0) {
        LOG_ERR("%s: failed to find key %s in gguf file\n", __func__,
                        kPatchSizeKey.c_str());
    } else {
        patch_size_ = gguf_get_val_u32(ctx, key_id);
    }
    std::string backend = "Xh2HalBackend";
    parse_deviceids();
    // 从gguf文件中加载hmmodels
    const int n_tensors = gguf_get_n_tensors(ctx);
    for (int i = 0; i < n_tensors; ++i) {
        const char *name = gguf_get_tensor_name(ctx, i);
        LOG_INF("tensor[%d]: name = %s\n", i, name);
        if (std::string(name).find("visual.hmm") == std::string::npos) {
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
            LOG_ERR("%s: failed to load prefill.hmm\n", __func__);
            return false;
        }
        break;
    }
    ggml_free(ctx_data);
    gguf_free(ctx);
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

void HoumoVision::fill_header(std::vector<float>& output_data) {
    // 1. 构造紧凑数据结构
    int16_t magic = 0x1234; 
    // 计算 nx：uint32_t（image_width_）→ int16_t，显式转换+范围校验
    uint32_t nx_uint = image_width_ / (patch_size_ * 2);
    int16_t nx = (nx_uint > INT16_MAX) ? INT16_MAX : static_cast<int16_t>(nx_uint);

    // 计算 ny：uint32_t（image_height_）→ int16_t，显式转换+范围校验
    uint32_t ny_uint = image_height_ / (patch_size_ * 2);
    int16_t ny = (ny_uint > INT16_MAX) ? INT16_MAX : static_cast<int16_t>(ny_uint);

    // 计算 chunksize：size_t（output_names_.size()）→ int16_t，显式转换+范围校验
    size_t chunksize_size = output_names_.size();
    int16_t chunksize = (chunksize_size > INT16_MAX) ? INT16_MAX : static_cast<int16_t>(chunksize_size);
    CompactHeader header = {
        magic,    // 第一个字段：.magic
        nx,       // 第二个字段：.nx
        ny,       // 第三个字段：.ny
        chunksize // 第四个字段：.chunksize
    };

    // 2. 确保 vector 至少有 2 个元素（前两个用于存储 header）
    if (output_data.size() < 2) {
        output_data.resize(2, 0.0f); // 扩容并初始化为 0，避免垃圾数据
    }

    // 3. 关键：将 8 字节 header 拷贝到 vector 前两个 float 元素的内存
    // &output_data[0] 是第一个 float 的地址，连续 8 字节覆盖前两个 float
    std::memcpy(
        &output_data[0],  // 目标地址：vector 第一个元素（前 4 字节）
        &header,          // 源地址：紧凑结构体（8 字节）
        sizeof(CompactHeader) // 拷贝长度：8 字节（覆盖前两个 float）
    );
}

void HoumoVision::parse_deviceids() {
    const char *device_id_str = std::getenv("HOUMO_DEVICE_ID");
    if (device_id_str == nullptr) {
        if (device_ids_.size() == 0)
            device_ids_.push_back(0);
        return;
    }

    std::string deviceList(device_id_str);
    std::istringstream iss(deviceList);
    std::string token;
    device_ids_.clear();
    while (std::getline(iss, token, ',')) {
        try {
            // 转换为整数并添加到列表
            device_ids_.push_back(std::stoi(token));
        } catch (const std::invalid_argument &) {
            LOG_ERR("Invalid device ID: %s\n", token.c_str());
        } catch (const std::out_of_range &) {
            LOG_ERR("Device ID out of range: %s\n", token.c_str());
        }
    }
}

void HoumoQwenVision::set_windows_attention() {
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
    if (module_->GetInputNum() < 3) {
        // Some resolution models do not have a windows_mask input.
        return;
    }
    auto &mask_tensor = input_tensors_[module_->GetInputName(2)];
    if (mask_tensor.Info().DataType() == tcim::DataType::FLOAT16) {
        // 创建int16的tcim tensor
        tcim::Tensor mask_int16 = tcim::Tensor::CreateHostTensor(
            mask_tensor.Info().AsContiguous().AsType(tcim::DataType::INT16));
        // 拷贝数据到int16的tcim tensor
        mask_int16.Buffer().CopyFromHost(mask.data(), mask_int16.MemSize());
        mask_int16.CastTo(mask_tensor);
    } else {
        mask_tensor.Buffer().CopyFromHost(mask.data(), mask_tensor.MemSize());
    }
    module_->SetInput(module_->GetInputName(2), mask_tensor);
}
bool HoumoQwenVision::init(const std::string &model_path, std::vector<int> device_ids) {
    if (!houmo_load(model_path, device_ids)) {
        LOG_ERR("Failed to load HoumoQwenVision model from %s\n",
                model_path.c_str());
        return false;
    }
    int input_size = module_->GetInputNum();
    for (auto i = 0; i < input_size; ++i) {
        auto input_name = module_->GetInputName(i);
        LOG_INF("%s: input_name: %s\n", __func__, input_name.c_str());
        auto input_info = module_->GetInputInfo(input_name).AsContiguous();
        if (input_info.Shape().size() == 4 && i == 0) {
            image_width_ = static_cast<uint16_t>(input_info.Shape()[3]);
            image_height_ = static_cast<uint16_t>(input_info.Shape()[2]);
            image_channels_ = static_cast<uint16_t>(input_info.Shape()[1]);
            batch_size_ = static_cast<uint16_t>(input_info.Shape()[0]);
            image_frames_ = 1;
            input_name_ = input_name;
        } else if (input_info.Shape().size() == 5 && i == 0) {
            image_width_ = static_cast<uint16_t>(input_info.Shape()[4]);
            image_height_ = static_cast<uint16_t>(input_info.Shape()[3]);
            image_frames_ = static_cast<uint16_t>(input_info.Shape()[2]);
            image_channels_ = static_cast<uint16_t>(input_info.Shape()[1]);
            batch_size_ = static_cast<uint16_t>(input_info.Shape()[0]);
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
        output_names_.push_back(output_name);
    }
    // Now Qwenvl2.5 clip three input. Qwen3vl clip one input.
    if (input_tensors_.size() > 1) {
        set_windows_attention();
    }
    return true;
}

bool HoumoQwenVision::encoding(const std::vector<float> &input_data,
                           std::vector<float> &output_data, int out_offset) {
    (void)out_offset;
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
        tcim::Tensor input_fp32_tensor = tcim::Tensor::CreateHostTensor(
            input_tensor.Info().AsContiguous().AsType(tcim::DataType::FLOAT32));
        input_fp32_tensor.Buffer().CopyFromHost(nchw_data_.data(),
                                                input_fp32_tensor.MemSize());
        input_fp32_tensor.CastTo(input_tensor);
    }
    // fixed resolution. index & mask already set in init()
    module_->SetInput(input_name_, input_tensor);
    module_->Run();
    module_->Sync();
    fill_header(output_data);
    int offset = sizeof(CompactHeader);
    output_data.resize(output_data.size() * output_names_.size());
    int8_t *output_data_ptr = reinterpret_cast<int8_t *>(output_data.data());
    for (auto& name:output_names_) {
        tcim::Tensor output_tensor = module_->GetOutput(name);
        memcpy(output_data_ptr + offset, output_tensor.Buffer().Data(),
           output_tensor.MemSize());
        offset += output_tensor.MemSize();
    }
    LOG_INF("houmo_vision: run model done\n");
    return true;
}
bool HoumoMinicpmVision::init(const std::string &model_path, std::vector<int> device_ids) {
    // TODO: not implemented
    if (!houmo_load(model_path, device_ids)) {
        LOG_ERR("Failed to load HoumoMinicpmVision model from %s\n",
                model_path.c_str());
        return false;
    }
    int input_size = module_->GetInputNum();
    for (auto i = 0; i < input_size; ++i) {
        auto input_name = module_->GetInputName(i);
        LOG_INF("%s: input_name: %s\n", __func__, input_name.c_str());
        auto input_info = module_->GetInputInfo(input_name).AsContiguous();
        auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
        input_tensors_.insert(
            std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
    }
    image_width_ = 560;
    image_height_ = 560;
    image_channels_ = 3;
    batch_size_ = 1;
    image_frames_ = 1;
    int out_size = module_->GetOutputNum();
    for (auto i = 0; i < out_size; ++i) {
        auto output_name = module_->GetOutputName(i);
        LOG_INF("%s: output_name: %s\n", __func__, output_name.c_str());
        output_names_.push_back(output_name);
    }

    this->precompute_global_pos_embed();
    this->set_attention_resampler();
    return true;
}
// ----------------------------------------------------------
// 辅助算法实现
// ----------------------------------------------------------

void HoumoMinicpmVision::precompute_global_pos_embed() {
    int dim_per_axis = n_embedding_ / 2; // 3584 / 2 = 1792

    // 分配内存
    global_pos_embed_table_.resize(MAX_GRID_H * MAX_GRID_W * n_embedding_);

    std::vector<float> embed_h(MAX_GRID_H * dim_per_axis);
    std::vector<float> embed_w(MAX_GRID_W * dim_per_axis);

    compute_1d_sincos(embed_h.data(), MAX_GRID_H, dim_per_axis);
    compute_1d_sincos(embed_w.data(), MAX_GRID_W, dim_per_axis);

    // 组合成 2D 表 [70, 70, 3584]
    for (int h = 0; h < MAX_GRID_H; ++h) {
        for (int w = 0; w < MAX_GRID_W; ++w) {
            long long offset = ((long long)h * MAX_GRID_W + w) * n_embedding_;
            float* dst = global_pos_embed_table_.data() + offset;

            // H (前一半)
            std::memcpy(dst, embed_h.data() + h * dim_per_axis, dim_per_axis * sizeof(float));
            // W (后一半)
            std::memcpy(dst + dim_per_axis, embed_w.data() + w * dim_per_axis, dim_per_axis * sizeof(float));
        }
    }
}

void HoumoMinicpmVision::compute_1d_sincos(float* out_ptr, int length, int dim) {
    int half_dim = dim / 2;
    std::vector<float> omega(half_dim);

    // Omega
    for (int i = 0; i < half_dim; ++i) {
        float exponent = (float)i / (float)half_dim;
        omega[i] = 1.0f / std::pow(10000.0f, exponent);
    }

    // Sin/Cos
    for (int pos = 0; pos < length; ++pos) {
        float* cur_vec = out_ptr + pos * dim;
        for (int i = 0; i < half_dim; ++i) {
            float val = (float)pos * omega[i];
            cur_vec[i]            = std::sin(val);
            cur_vec[i + half_dim] = std::cos(val);
        }
    }
}
void HoumoMinicpmVision::set_attention_resampler() {
    int32_t pos_h = image_height_ / patch_size_;
    int32_t pos_w = image_width_ / patch_size_;
    std::vector<int32_t> positions(pos_h * pos_w);
    int bucket_coords_h[1024];
    int bucket_coords_w[1024];
    for (int i = 0; i < pos_h; i++) {
        bucket_coords_h[i] = std::floor(70.0 * i / pos_h);
    }
    for (int i = 0; i < pos_w; i++) {
        bucket_coords_w[i] = std::floor(70.0 * i / pos_w);
    }
    for (int i = 0, id = 0; i < pos_h; i++) {
        for (int j = 0; j < pos_w; j++) {
            positions[id++] = bucket_coords_h[i] * 70 + bucket_coords_w[j];
        }
    }
    auto &tensor = input_tensors_[module_->GetInputName(1)];
    tensor.Buffer().CopyFromHost(positions.data(), tensor.MemSize());
    module_->SetInput(module_->GetInputName(1), tensor);
    // set attention_mask
    auto &mask_tensor = input_tensors_[module_->GetInputName(2)];
    memset(mask_tensor.Buffer().Data(), 0, mask_tensor.MemSize());
    module_->SetInput(module_->GetInputName(2), mask_tensor);

    // set global_pos_embed
    auto &pos_embed_tensor = input_tensors_[module_->GetInputName(3)];
    tcim::Tensor input_fp32_tensor = tcim::Tensor::CreateHostTensor(
        pos_embed_tensor.Info().AsContiguous().AsType(tcim::DataType::FLOAT32));
    //input_fp32_tensor.Buffer().CopyFromHost(global_pos_embed_table_.data(), input_fp32_tensor.MemSize());
    float* dst_ptr = (float*)input_fp32_tensor.Buffer().Data();
    const float* src_ptr = global_pos_embed_table_.data();

    int src_stride = 70 * n_embedding_; // 全局表的宽度 (70)
    int copy_bytes = 40 * n_embedding_ * sizeof(float); // 目标每行拷贝的字节数 (40)

    for (int h = 0; h < 40; ++h) {
        // 每一行只拷贝前 40 个 Patch 的 Embedding
        memcpy(dst_ptr + h * 40 * n_embedding_,
            src_ptr + h * src_stride,
            copy_bytes);
    }
    input_fp32_tensor.CastTo(pos_embed_tensor);
    module_->SetInput(module_->GetInputName(3), pos_embed_tensor);

    // resampler_key_padding_mask
    auto &resampler_mask_tensor = input_tensors_[module_->GetInputName(4)];
    memset(resampler_mask_tensor.Buffer().Data(), 0, resampler_mask_tensor.MemSize());
    module_->SetInput(module_->GetInputName(4), resampler_mask_tensor);
}

/**
 * @brief 输入 HWC 格式的 Normalized Float，输出 Unfold+Permute 后的数据
 * * 输入: [560, 560, 3] (HWC)
 * 输出: [3, 14, 22400] (对应 Python: 3, 14, -1)
 */
void HoumoMinicpmVision::reorder_hwc_to_target(const float *input_hwc,
                                               float *output_buffer) {
    int out_idx = 0;
    // 派生常量
    int GRID_W = image_width_ / patch_size_; // 40
    int NUM_PATCHES =
        (image_height_ / patch_size_) * (image_width_ / patch_size_); // 1600
    // 目标内存顺序 (3, 14, 1600, 14) -> (C, PH, P_IDX, PW)
    for (uint16_t c = 0; c < image_channels_; ++c) {    // Dim 0: Channel (3)
        for (size_t ph = 0; ph < patch_size_; ++ph) { // Dim 1: Patch Height (14)
            for (int p_idx = 0; p_idx < NUM_PATCHES;
                 ++p_idx) {
                for (size_t pw = 0; pw < patch_size_; ++pw) {
                    // --- 坐标映射计算 ---

                    // 1. 当前 Patch 在 Grid 中的坐标 (行, 列)
                    int grid_y = p_idx / GRID_W;
                    int grid_x = p_idx % GRID_W;

                    // 2. 还原到原图的 (h, w)
                    int h = grid_y * patch_size_ + ph;
                    int w = grid_x * patch_size_ + pw;

                    // 3. 计算 HWC 输入的偏移量
                    // Input Layout: [H][W][C] -> index = (h * Width + w) *
                    // Channels + c
                    int in_offset =
                        (h * image_width_ + w) * image_channels_ + c;

                    // 4. 赋值 (纯拷贝)
                    output_buffer[out_idx++] = input_hwc[in_offset];
                }
            }
        }
    }
}
bool HoumoMinicpmVision::encoding(const std::vector<float> &input_data,
                                  std::vector<float> &output_data,
                                  int out_offset) {
    LOG_DBG("houmo_minicpm_vision: encoding image slice... %d,%d\n", out_offset,
            input_data.size());
    // input_data 强制转化为int8 保存为图片
    std::vector<int8_t> input_int8_data(input_data.size());
    for (size_t i = 0; i < input_data.size(); i++) {
        input_int8_data[i] = static_cast<int8_t>(input_data[i]);
    }
    // 保存为图片, stdc++ 11 实现
    // std::ofstream image_file("input_image.png", std::ios::binary);
    // image_file.write(reinterpret_cast<const char *>(input_int8_data.data()),
    //                  input_int8_data.size());
    // image_file.close();
    // set image input.
    auto input_tensor = input_tensors_[module_->GetInputName(0)];
    // rgb_to_nchw(input_data);
    const size_t expected_size = image_width_ * image_height_ *
                                 image_channels_ * batch_size_ * image_frames_;
    nchw_data_.resize(expected_size);
    reorder_hwc_to_target(input_data.data(), nchw_data_.data());
    tcim::Tensor input_fp32_tensor = tcim::Tensor::CreateHostTensor(
        input_tensor.Info().AsContiguous().AsType(tcim::DataType::FLOAT32));
    input_fp32_tensor.Buffer().CopyFromHost(nchw_data_.data(),
                                            input_fp32_tensor.MemSize());
    input_fp32_tensor.CastTo(input_tensor);
    module_->SetInput(module_->GetInputName(0), input_tensor);
    module_->Run();
    module_->Sync();
    for (auto &name : output_names_) {
        tcim::Tensor output_tensor = module_->GetOutput(name);
        memcpy(output_data.data() + out_offset, output_tensor.Buffer().Data(),
               output_tensor.MemSize());
    }
    return true;
}
