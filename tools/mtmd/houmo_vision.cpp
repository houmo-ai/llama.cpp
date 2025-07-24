#include "houmo_vision.h"
#include "ggml.h"
#include "gguf.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
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

bool HoumoVision::houmo_load(const std::string &file_name) {
#if 1
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
#else
    // for debug...
    auto wm = tcim::Module::WeightManager::CreateWeightManager(0);
    auto option_decoder = tcim::Module::Option(wm);

    auto option = tcim::Module::Option(wm);
    module_ = std::make_shared<tcim::Module>();
    auto status =
        module_->LoadModel("/houmo/models/qwenvl7b/visual.hmm", option);
    if (status != tcim::Status::OK) {
        LOG_ERR("%s: failed to load prefill.hmm\n", __func__);
        return false;
    }
#endif
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
            input_name_ = input_name;
        }
        // for (size_t kk = 0; kk < input_info.Shape().size(); kk++) {
        //     LOG_INF("%s: input_info.Shape()[%d]: %d, type= %d\n", __func__, kk,
        //             input_info.Shape()[kk], input_info.DataType());
        // }
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

void HoumoVision::rgb_to_yuv444(const std::vector<float> &bgr_data, int width,
                                int height,
                                std::vector<uint8_t> &yuv444sp_data) {
    int pixel_count = width * height;
    // YUV444SP总大小：Y(1*H*W) + UV(2*H*W)
    yuv444sp_data.resize(pixel_count * 3);

    for (int i = 0; i < pixel_count; ++i) {
        float r = bgr_data[i * 3];
        float g = bgr_data[i * 3 + 1];
        float b = bgr_data[i * 3 + 2];

        // BT.601转换公式（输入BGR为[0,1]范围）
        float y = 0.299f * r + 0.587f * g + 0.114f * b;
        float u =
            -0.14713f * r - 0.28886f * g + 0.436f * b + 128;
        float v =
            0.615f * r - 0.51499f * g - 0.10001f * b + 128;

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
    // input_data 转成uint8_t, 再转成yuv44sp
    auto input_tensor = input_tensors_[input_name_];
    std::vector<uint8_t> yuv444_data;
    rgb_to_yuv444(input_data, image_width_, image_height_, yuv444_data);
    input_tensor.Buffer().CopyFromHost(yuv444_data.data(),
                                       input_tensor.MemSize());
    auto window_index_tensor = input_tensors_["window_index"];
    auto window_mask_tensor = input_tensors_["window_mask"];
    module_->SetInput("window_index", window_index_tensor);
    module_->SetInput(input_name_, input_tensor);
    module_->Run();
    module_->Sync();
    tcim::Tensor output_tensor = module_->GetOutput(module_->GetOutputName(0));
    auto output_contiguous = output_tensors_[module_->GetOutputName(0)];
    output_tensor.CopyTo(output_contiguous);
    memcpy(output_data.data(), output_contiguous.Buffer().Data(),
           output_contiguous.MemSize());
    LOG_ERR("houmo_vision: run model done, output_data size: %d\n",
            output_contiguous.MemSize());
    return true;
}

int32_t HoumoVision::image_width() const { return image_width_; }
int32_t HoumoVision::image_height() const { return image_height_; }
int32_t HoumoVision::image_channels() const { return image_channels_; }
static inline float lerp(float s, float e, float t) { return s + (e - s) * t; }
bool HoumoVision::image_resize(clip_image_u8_ptr &src,
                               clip_image_f32_ptr &dst) {
    dst->nx = image_width_;
    dst->ny = image_height_;
    dst->buf.resize(3 * image_width_ * image_height_);
    float x_ratio = static_cast<float>(src->nx - 1) / image_width_;
    float y_ratio = static_cast<float>(src->ny - 1) / image_height_;

    for (size_t y = 0; y < image_height_; y++) {
        for (size_t x = 0; x < image_width_; x++) {
            float px = x_ratio * x;
            float py = y_ratio * y;
            int x_floor = static_cast<int>(px);
            int y_floor = static_cast<int>(py);
            float x_lerp = px - x_floor;
            float y_lerp = py - y_floor;

            for (int c = 0; c < 3; c++) {
                float top = lerp(
                    static_cast<float>(
                        src->buf[3 * (y_floor * src->nx + x_floor) + c]),
                    static_cast<float>(
                        src->buf[3 * (y_floor * src->nx + (x_floor + 1)) + c]),
                    x_lerp);
                float bottom = lerp(
                    static_cast<float>(
                        src->buf[3 * ((y_floor + 1) * src->nx + x_floor) + c]),
                    static_cast<float>(
                        src->buf[3 * ((y_floor + 1) * src->nx + (x_floor + 1)) +
                                 c]),
                    x_lerp);
                dst->buf[3 * (y * image_width_ + x) + c] =
                    static_cast<float>(lerp(top, bottom, y_lerp));
            }
        }
    }
    return 0;
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
    auto &tensor = input_tensors_["window_index"];
    tensor.Buffer().CopyFromHost(inv_idx.data(), tensor.MemSize());
    module_->SetInput("window_index", tensor);
    auto &mask_tensor = input_tensors_["window_mask"];
    mask_tensor.Buffer().CopyFromHost(mask.data(), mask_tensor.MemSize());
    module_->SetInput("window_mask", mask_tensor);
}