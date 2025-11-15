#include "houmo-vllmodel.h"
#include "ggml.h"
#include "gguf.h"
#include "llama-impl.h"
#include <regex>
#pragma pack(push, 1)
struct CompactHeader {
    int16_t magic;      // 16位标识
    int16_t nx;         // 16位宽度
    int16_t ny;         // 16位高度
    int16_t chunksize;  // 16位块大小
};
#pragma pack(pop)

HouMoVLLModel::HouMoVLLModel() : HouMoLLModel::Impl() {
    model_type_ = 1; // 1: image
}
bool HouMoVLLModel::houmo_load(const std::string &file_name) {

    struct ggml_context *ctx_data = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_data,
    };
    struct gguf_context *ctx = gguf_init_from_file(file_name.c_str(), params);
    fill_parameters(ctx);
    // 从gguf中读取visual 宽度&高度
    auto key_id = gguf_find_key(ctx, kVisualHeight.c_str());
    if (key_id > 0) {
        auto visual_height_ = gguf_get_val_u32(ctx, key_id);
        ny = visual_height_ / (14 * 2);
        LLAMA_LOG_INFO("visual height is %d, ny is %d\n", visual_height_, ny);
    }
    key_id = gguf_find_key(ctx, kVisualWidth.c_str());
    if (key_id > 0) {
        auto visual_width_ = gguf_get_val_u32(ctx, key_id);
        nx = visual_width_ / (14 * 2);
        LLAMA_LOG_INFO("visual width is %d, nx is %d\n", visual_width_, nx);
    }

    LLAMA_LOG_INFO("HouMoVLLModel after gguf_init_from_file\n");
    // 从gguf文件中加载hmmodels
    const char *device_id_str = std::getenv("HOUMO_DEVICE_ID");
    int device_id = 0;
    if (device_id_str != nullptr) {
        device_id = std::stoi(device_id_str);
    }
    auto wm = tcim::Module::WeightManager::CreateWeightManager(device_id);
    auto option_decoder = tcim::Module::Option(wm);

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
    auto option_prefill = tcim::Module::Option(wm);
    option_prefill.SetDummyTensors(dummy_tensor_names);
    const std::regex kPrefillPattern("^prefill(_\\d+)?\\.hmm$");
    const std::regex kDecodePattern("^decoder(_\\d+)?\\.hmm$");
    // 1. 加载 embedding
    const int n_tensors = gguf_get_n_tensors(ctx);
    for (int i = 0; i < n_tensors; ++i) {
        const char *name = gguf_get_tensor_name(ctx, i);
        LLAMA_LOG_INFO("tensor[%d]: name = %s\n", i, name);
        const size_t size = gguf_get_tensor_size(ctx, i);
        size_t offset =
            gguf_get_tensor_offset(ctx, i) + gguf_get_meta_size(ctx);
        // auto *tensor = ggml_get_tensor(ctx_data, name);
        std::vector<char> buffer(size);
        read_tensor_to_buffer(file_name, offset, size, buffer);
        if (strncmp(name, "quant_embedding.pt", strlen("quant_embedding.pt")) ==
            0) {
            torch::Tensor input;
            try {
                torch::load(input, buffer.data(), size);
                if (!input.is_floating_point()) {
                    input = input.to(torch::kFloat32);
                }
                input = input.detach();
                input = input.view({-1, n_embd_});
                embedding_ = torch::nn::Embedding::from_pretrained(input);
            } catch (const c10::Error &e) {
                LLAMA_LOG_ERROR("%s: failed to load quant_embedding.pt %s\n",
                                __func__, e.what());
                return false;
            }

        } else if (std::regex_match(name, kPrefillPattern)) {
            // load prefill
            prefill_model_ = std::make_shared<tcim::Module>();
            auto status =
                prefill_model_->LoadModel(buffer.data(), size, option_prefill);
            if (status != tcim::Status::OK) {
                LLAMA_LOG_ERROR("%s: failed to load prefill.hmm\n", __func__);
                return false;
            }
            LLAMA_LOG_INFO("%s: succeed to load prefill.hmm\n", __func__);
        } else if (std::regex_match(name, kDecodePattern)) {
            // load decode
            decode_model_ = std::make_shared<tcim::Module>();
            auto status =
                decode_model_->LoadModel(buffer.data(), size, option_decoder);
            if (status != tcim::Status::OK) {
                LLAMA_LOG_ERROR("%s: failed to load decode.hmm\n", __func__);
                return false;
            }
            LLAMA_LOG_INFO("%s: succeed to load decode.hmm\n", __func__);
        } else {
            LLAMA_LOG_INFO("tensor[%d]: name = %s, size = %zu\n", i, name,
                           size);
            return false;
        }
    }
    ggml_free(ctx_data);
    gguf_free(ctx);
    return true;
}

void HouMoVLLModel::houmo_init(int seq_max) {
    // init prefill
    prefill_input_init(prefill_model_, prefill_input_map_);
    // 填充prefill 的output
    prefill_output_init(prefill_model_, prefill_output_map_,
                        prefill_output_names_);
    // init decode
    decode_input_init(decode_model_, decode_input_map_);
    // 填充decode 的output
    decode_output_init(decode_model_, decode_output_map_, decode_output_names_);
    if (seq_max > 4) {
        LLAMA_LOG_ERROR("%s: seq_max > 4 not supported, set seq_max to 4\n",
                        __func__);
        seq_max = 4;
    }
    int size = (seq_max + n_prefill_batch_ - 1) / n_prefill_batch_;
    size = std::max(size, n_decode_batch_);
    LLAMA_LOG_INFO("seq_max = %d, n_prefill_batch_ = %d, n_decode_batch_ = %d\n",
                   seq_max, n_prefill_batch_, n_decode_batch_);
    for (int i = 0; i < size; i++) {
        kv_cache_item item;
        item.seq_id = i;
        for (int idx = 0; idx < n_blocks_; idx++) {
            std::string k_cache_name = "model_layers_" + std::to_string(idx) +
                                       "_self_attn_kcache_input";
            std::string v_cache_name = "model_layers_" + std::to_string(idx) +
                                       "_self_attn_vcache_input";
            item.names.push_back(k_cache_name);
            item.names.push_back(v_cache_name);

            if (n_decode_batch_ > 1) {
                k_cache_name = k_cache_name + "_batch" + std::to_string(i);
                v_cache_name = v_cache_name + "_batch" + std::to_string(i);
            }
            #ifdef XH2_ENABLED
            //FIXME: v2.5.0 will support GetDevInput
            auto kcache = decode_model_->GetDevInput(k_cache_name);
            auto vcache = decode_model_->GetDevInput(v_cache_name);
            #else
            auto kcache = decode_model_->GetInput(k_cache_name);
            auto vcache = decode_model_->GetInput(v_cache_name);
            #endif
            if (n_decode_batch_ > 1 || seq_max == 1) {
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
    input_embedding_ids_.resize(seq_max);
    offset_embedding_ids_.resize(seq_max);
    position_indexs_.resize(seq_max);
    prefill_input_lengths_.resize(seq_max);
    deepstack_features_0_.resize(seq_max);
    deepstack_features_1_.resize(seq_max);
    deepstack_features_2_.resize(seq_max);
    for (int i = 0; i < seq_max; i++) {
        input_embedding_ids_[i].resize(context_length_ * n_embd_, 0);
        if (prefill_input_map_.size() == 9) {
            deepstack_features_0_[i].resize(context_length_ * n_embd_, 0);
            deepstack_features_1_[i].resize(context_length_ * n_embd_, 0);
            deepstack_features_2_[i].resize(context_length_ * n_embd_, 0);
        }
    }
    if (decode_input_map_.size() == 9) {
        for (int i = 0; i < 3; i++) {
            auto deepstack =
                decode_input_map_.at(decode_model_->GetInputName(6 + i));
            memset(deepstack.Data(), 0, deepstack.MemSize());
            decode_model_->SetInput(decode_model_->GetInputName(6 + i),
                                    deepstack);
        }
    }
    LLAMA_LOG_INFO("%s: HMM graph init success\n", __func__);
    LLAMA_LOG_INFO("prefill_length_ = %d\n context_length_ = %d\n "
                   "n_blocks_ = %d\n n_embd_ = %ld\n n_prefill_batch_ = %d\n "
                   "n_decode_batch_ = "
                   "%d\n   kv_cache_size = %zu\n need_set_kv_cache_ = "
                   "%d\n, seq_max = %d\n",
                   prefill_length_, context_length_, n_blocks_, n_embd_,
                   n_prefill_batch_, n_decode_batch_, kv_cache_.size(),
                   need_set_kv_cache_, seq_max);
}

int HouMoVLLModel::houmo_prefill(std::vector<llama_token> &tokens, int seq_id,
                                 float *logits) {

    LLAMA_LOG_INFO("%s seq_id = %d, tokens size is = %zu\n", "prefill stage",
                    seq_id, tokens.size());
    if ((offset_embedding_ids_[seq_id] + tokens.size()) >= (size_t) context_length_) {
        LLAMA_LOG_WARN("excceed context length, reset offset_embedding_id_ to 0\n");
        reset_position_ids(seq_id);
        return 1;
    }
    int32_t current_length = 0;
    // 先计算embedding.
    torch::Tensor inputs_embeds;
    auto opts = torch::TensorOptions().dtype(torch::kInt);
    auto inp = torch::from_blob(tokens.data(),
                                {int64_t(tokens.size())}, opts)
                    .clone();
    try {
        inputs_embeds = embedding_->forward(inp);
    } catch (const c10::Error &e) {
        std::cerr << "error loading the model\n"
                    << e.what_without_backtrace();
        return -1;
    }

    // 1. 将半精度浮点转换为int16_t
    if (xh1_) {
        auto inputs_embeds_int16 = inputs_embeds.to(torch::kInt16);
        inputs_embeds_int16 = inputs_embeds_int16.view({-1, n_embd_});
        // 缓存输入
        memcpy(input_embedding_ids_[seq_id].data() + offset_embedding_ids_[seq_id] * n_embd_,
            inputs_embeds_int16.data_ptr<int16_t>(),
            tokens.size() * n_embd_ * sizeof(int16_t));
    } else {
        memcpy(input_embedding_ids_[seq_id].data() + offset_embedding_ids_[seq_id] * n_embd_,
            inputs_embeds.data_ptr<at::Half>(),
            tokens.size() * n_embd_ * sizeof(at::Half));
    }
    offset_embedding_ids_[seq_id] += tokens.size();
    prefill_input_lengths_[seq_id].push_back(tokens.size());
    // 计算所有embeddings
    if (logits == nullptr) {
        // 只缓存输入
        LLAMA_LOG_INFO("%s: logits is nullptr, only cache input. seq_id = %d\n", __func__, seq_id);
        return 0;
    }
    size_t input_echo_len = offset_embedding_ids_[seq_id];
    size_t prefill_loop_round = ceil(static_cast<float>(input_echo_len) /
    static_cast<float>(prefill_length_));

    std::vector<int32_t> time_position_ids;
    std::vector<int32_t> high_position_ids;
    std::vector<int32_t> width_position_ids;
    compute_position_ids(seq_id, time_position_ids, high_position_ids,
                         width_position_ids);
    time_position_ids.resize(prefill_loop_round * prefill_length_);
    high_position_ids.resize(prefill_loop_round * prefill_length_);
    width_position_ids.resize(prefill_loop_round * prefill_length_);
    int valid_length = 0;
    // 计算prefill输出
    for (size_t round = 0; round < prefill_loop_round; round++) {
        valid_length = round * prefill_length_;
        if (round == prefill_loop_round - 1) {
            current_length = input_echo_len - round * prefill_length_;
        } else {
            current_length = prefill_length_;
        }
        auto input_data =
            prefill_input_map_.at(prefill_model_->GetInputName(0));
        input_data.Buffer().CopyFromHost(input_embedding_ids_[seq_id].data() +
                                             round * prefill_length_ * n_embd_,
                                         input_data.MemSize());
        auto time_position_id =
            prefill_input_map_.at(prefill_model_->GetInputName(1));
        auto high_position_id =
            prefill_input_map_.at(prefill_model_->GetInputName(2));
        auto width_position_id =
            prefill_input_map_.at(prefill_model_->GetInputName(3));
        auto prefill_valid_length =
            prefill_input_map_.at(prefill_model_->GetInputName(4));
        auto prefill_current_length =
            prefill_input_map_.at(prefill_model_->GetInputName(5));
        set_input_data(time_position_id, time_position_ids, valid_length);
        set_input_data(high_position_id, high_position_ids, valid_length);
        set_input_data(width_position_id, width_position_ids, valid_length);
        set_input_data(prefill_valid_length, n_prefill_batch_, valid_length);
        set_input_data(prefill_current_length, n_prefill_batch_,
                       current_length);
        // 当前仅qwen3vl生效
        fill_deepstack_embed(seq_id, valid_length);
        prefill_model_->SetInput(prefill_model_->GetInputName(0), input_data);
        prefill_model_->SetInput(prefill_model_->GetInputName(1),
                                 time_position_id);
        prefill_model_->SetInput(prefill_model_->GetInputName(2),
                                 high_position_id);
        prefill_model_->SetInput(prefill_model_->GetInputName(3),
                                 width_position_id);
        prefill_model_->SetInput(prefill_model_->GetInputName(4),
                                 prefill_valid_length);
        prefill_model_->SetInput(prefill_model_->GetInputName(5),
                                 prefill_current_length);
        auto &item = kv_cache_[seq_id];
        assert(item.seq_id == seq_id);
        for (size_t idx = 0; idx < item.tensors.size(); idx++) {
            std::string name = item.names[idx];
            auto tensor = item.tensors[idx];
            prefill_model_->SetInput(name, tensor);
        }
        prefill_model_->Run();
        prefill_model_->Sync();
    }
    decode_valid_length_[seq_id] = input_echo_len;
    tcim::Tensor prefill_output_data =
        prefill_model_->GetOutput(prefill_output_names_[0]);
    // 转浮点
    auto &prefill_output_float = prefill_output_map_[prefill_output_names_[0]];
    prefill_output_data.CastTo(prefill_output_float);
    memcpy(logits, prefill_output_float.Data(), prefill_output_float.MemSize());

    // reset input_embedding_ids_
    reset_position_ids(seq_id);
    return 0;
}

int HouMoVLLModel::houmo_prefill(const float *embeddings, int n_tokens,
                                 int seq_id, float *logits) {
    LLAMA_LOG_DEBUG("%s seq_id = %d, n_tokens is = %d\n", "prefill stage", seq_id,
                   n_tokens);
    if (logits == nullptr) {
        // 只缓存输入
        if ((n_tokens + offset_embedding_ids_[seq_id]) >= context_length_) {
            LLAMA_LOG_WARN(
                "excceed context length, reset offset_embedding_id_ to 0\n");
            reset_position_ids(seq_id);
            return 1;
        }
        CompactHeader header;
        int embedding_offset = 0;
        std::memcpy(&header, embeddings, sizeof(CompactHeader));
        if (header.magic == 0x1234) {
            ny = header.ny;
            nx = header.nx;
            embedding_offset = sizeof(CompactHeader);
        }
        const int8_t* embeddings_ptr = reinterpret_cast<const int8_t *>(embeddings);
        const int data_size_bytes = n_tokens * n_embd_ * sizeof(int16_t);
        memcpy(input_embedding_ids_[seq_id].data() +
                   offset_embedding_ids_[seq_id] * n_embd_,
               embeddings_ptr + embedding_offset, data_size_bytes);
        if (header.magic == 0x1234 && header.chunksize == 4) {
            // for qwenvl3
            embedding_offset += data_size_bytes;
            LLAMA_LOG_INFO("deepstack feature 0 offset = %d, addr = %p\n", embedding_offset, embeddings_ptr + embedding_offset);
            memcpy(deepstack_features_0_[seq_id].data() +
                   offset_embedding_ids_[seq_id] * n_embd_,
                   embeddings_ptr + embedding_offset, data_size_bytes);
            embedding_offset += data_size_bytes;

            memcpy(deepstack_features_1_[seq_id].data() +
                   offset_embedding_ids_[seq_id] * n_embd_,
                   embeddings_ptr + embedding_offset, data_size_bytes);
            embedding_offset += data_size_bytes;
            memcpy(deepstack_features_2_[seq_id].data() +
                   offset_embedding_ids_[seq_id] * n_embd_,
                   embeddings_ptr + embedding_offset, data_size_bytes);
        }
        offset_embedding_ids_[seq_id] += n_tokens;
        prefill_input_lengths_[seq_id].push_back(n_tokens);
        return 0;
    } else {
        // FIXME:
        LLAMA_LOG_ERROR("%s: logits is not null, not supported\n", __func__);
                    reset_position_ids(seq_id);
        return -1;
    }
}

int HouMoVLLModel::houmo_decode(std::vector<llama_token> &batches,
                                std::vector<int> &seq_ids,
                                std::vector<float *> logits) {
    LLAMA_LOG_DEBUG("%s: seq_ids size = %zu, batches size = %zu\n", __func__,
                   seq_ids.size(), batches.size());
    if (seq_ids.size() > (size_t)n_decode_batch_) {
        std::cerr << "seq_ids.size() > n_decode_batch_ not supported\n";
        return -1;
    }
    torch::Tensor input_emd;
    auto decode_opts = torch::TensorOptions().dtype(torch::kInt);
    auto decode_inp =
        torch::from_blob(batches.data(), {int64_t(batches.size())}, decode_opts)
            .clone();
    try {
        input_emd = embedding_->forward(decode_inp);
    } catch (const c10::Error &e) {
        std::cerr << "error loading the model\n" << e.what_without_backtrace();
        return -1;
    }
    // set input_1, xh2 是float16
    input_emd = input_emd.view({int64_t(batches.size()), 1, -1});
    auto input_data = decode_input_map_.at(decode_model_->GetInputName(0));

    // 计算每个序列的大小（以int16_t为单位）
    int64_t embedding_dim = input_emd.size(2);
    assert(input_data.MemSize() >=
           embedding_dim * seq_ids.size() * sizeof(int16_t));

    std::vector<int16_t> decode_input_data(
        input_data.MemSize() / sizeof(int16_t), 1);
    // 填充数据
    for (size_t i = 0; i < seq_ids.size(); i++) {
        auto batch_id = seq_ids[i];
        // 单batch模式下所有序列共用batch 0
        if (n_decode_batch_ == 1)
            batch_id = 0;
        // 验证seq_id有效性
        if (batch_id >= n_decode_batch_) {
            std::cerr << "Error: Invalid seq_id " << batch_id << " (max "
                      << n_decode_batch_ - 1 << ")" << std::endl;
            return -1;
        }
        int16_t *input_data_ptr =
            decode_input_data.data() + batch_id * embedding_dim;
        if (xh1_) {
            input_emd = input_emd.to(torch::kInt16);
            memcpy(input_data_ptr, input_emd[i][0].data_ptr<int16_t>(),
                embedding_dim * sizeof(int16_t));
        } else {
            memcpy(input_data_ptr, input_emd[i][0].data_ptr<at::Half>(), embedding_dim * sizeof(at::Half));
        }
    }

    input_data.Buffer().CopyFromHost(decode_input_data.data(),
                                     input_data.MemSize());
    // set valid_length
    std::vector<int32_t> valid_length_vec(n_decode_batch_, 1);
    std::vector<int32_t> current_length_vec(n_decode_batch_, 1);
    std::vector<int32_t> position_vec(n_decode_batch_, 1);
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
        position_vec[batch_id]= position_indexs_[seq_ids[k]];
    }
    auto time_position_id =
        decode_input_map_.at(decode_model_->GetInputName(1));
    auto hight_position_id =
        decode_input_map_.at(decode_model_->GetInputName(2));
    auto width_position_id =
        decode_input_map_.at(decode_model_->GetInputName(3));
    auto decode_valid_length =
        decode_input_map_.at(decode_model_->GetInputName(4));
    auto decode_current_length =
        decode_input_map_.at(decode_model_->GetInputName(5));

    set_input_data(time_position_id, position_vec);
    set_input_data(hight_position_id, position_vec);
    set_input_data(width_position_id, position_vec);
    set_input_data(decode_valid_length, valid_length_vec);
    set_input_data(decode_current_length, current_length_vec);

    decode_model_->SetInput(decode_model_->GetInputName(0), input_data);
    decode_model_->SetInput(decode_model_->GetInputName(1), time_position_id);
    decode_model_->SetInput(decode_model_->GetInputName(2), hight_position_id);
    decode_model_->SetInput(decode_model_->GetInputName(3), width_position_id);
    decode_model_->SetInput(decode_model_->GetInputName(4),
                            decode_valid_length);
    decode_model_->SetInput(decode_model_->GetInputName(5),
                            decode_current_length);
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

    decode_model_->Run();
    decode_model_->Sync();
    tcim::Tensor decode_data =
        decode_model_->GetOutput(decode_output_names_[0]);
    assert(decode_data.Info().DataType() == tcim::DataType::INT16);
    // 转浮点
    auto &decode_output_float = decode_output_map_[decode_output_names_[0]];
    decode_data.CastTo(decode_output_float);
    auto n_vocab =
        decode_output_float.MemSize() / (n_decode_batch_ * sizeof(float));
    for (size_t i = 0; i < seq_ids.size(); i++) {
        auto batch_id = seq_ids[i];
        decode_valid_length_[batch_id]++;
        position_indexs_[batch_id]++;
        // 单batch模式下所有序列共用batch 0
        if (n_decode_batch_ == 1)
            batch_id = 0;
        float *logits_out = logits[i];
        float *output_ptr = static_cast<float *>(decode_output_float.Data()) +
                            batch_id * n_vocab;
        memcpy(logits_out, output_ptr, n_vocab * sizeof(float));
    }
    return 0;
}

void HouMoVLLModel::update_position_ids(int n_tokens,
                                    std::vector<int32_t> &time_position_ids,
        std::vector<int32_t> &high_position_ids,
        std::vector<int32_t> &width_position_ids,
        int32_t &start_pos) {
    if (time_position_ids.size() == 0) {
        start_pos = 0;
    }
    if (n_tokens == nx * ny) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                time_position_ids.push_back(start_pos);
                high_position_ids.push_back(start_pos + y);
                width_position_ids.push_back(start_pos + x);
            }
        }

        start_pos += nx;
    } else {
        for (int i = 0; i < n_tokens; i++) {
            time_position_ids.push_back(start_pos + i);
            high_position_ids.push_back(start_pos + i);
            width_position_ids.push_back(start_pos + i);
        }
        start_pos += n_tokens;
    }
}

void HouMoVLLModel::compute_position_ids(
    int seq_id, std::vector<int32_t> &time_position_ids,
    std::vector<int32_t> &high_position_ids,
    std::vector<int32_t> &width_position_ids) {
    auto &lens = prefill_input_lengths_[seq_id];
    // mutimedia call prefill three times.
    for (size_t i = 0; i < lens.size(); ++i) {
        update_position_ids(lens[i], time_position_ids,
                            high_position_ids, width_position_ids,
                            position_indexs_[seq_id]);
    }
}

void HouMoVLLModel::reset_position_ids(int seq_id) {
    offset_embedding_ids_[seq_id] = 0;
    prefill_input_lengths_[seq_id].resize(0);
}

void HouMoVLLModel::fill_deepstack_embed(int seq_id, int valid_length) {
    if (prefill_input_map_.size() != 9)
        return;
    auto name = prefill_model_->GetInputName(6);
    auto deepstack_tensor0 = prefill_input_map_.at(name);
    deepstack_tensor0.Buffer().CopyFromHost(
        deepstack_features_0_[seq_id].data() + valid_length * n_embd_,
        deepstack_tensor0.MemSize());
    prefill_model_->SetInput(name, deepstack_tensor0);
    memset(deepstack_features_0_[seq_id].data() + valid_length * n_embd_, 0,
           deepstack_tensor0.MemSize());

    name = prefill_model_->GetInputName(7);
    auto deepstack_tensor1 = prefill_input_map_.at(name);
    deepstack_tensor1.Buffer().CopyFromHost(
        deepstack_features_1_[seq_id].data() + valid_length * n_embd_,
        deepstack_tensor1.MemSize());
    prefill_model_->SetInput(name, deepstack_tensor1);
    memset(deepstack_features_1_[seq_id].data() + valid_length * n_embd_, 0,
           deepstack_tensor1.MemSize());

    name = prefill_model_->GetInputName(8);
    auto deepstack_tensor2 = prefill_input_map_.at(name);
    deepstack_tensor2.Buffer().CopyFromHost(
        deepstack_features_2_[seq_id].data() + valid_length * n_embd_,
        deepstack_tensor2.MemSize());
    prefill_model_->SetInput(name, deepstack_tensor2);
    memset(deepstack_features_2_[seq_id].data() + valid_length * n_embd_, 0,
           deepstack_tensor2.MemSize());

}