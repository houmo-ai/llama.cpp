#include "houmo-embed.h"
#include "ggml.h"
#include "gguf.h"
#include "llama-batch.h"
#include "llama-impl.h"
#include <string.h>
HouMoEmbed::HouMoEmbed() {
    // do nothing
}

HouMoEmbed::~HouMoEmbed() {
    // do nothing
}

bool HouMoEmbed::houmo_load(const std::string &file_name, bool use_mmap) {
    struct ggml_context *ctx_data = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_data,
    };
    struct gguf_context *ctx = gguf_init_from_file(file_name.c_str(), params);

    // 1. 加载 embedding
    const int n_tensors = gguf_get_n_tensors(ctx);
    for (int i = 0; i < n_tensors; ++i) {
        const char *name = gguf_get_tensor_name(ctx, i);
        LLAMA_LOG_INFO("tensor[%d]: name = %s\n", i, name);
        const size_t size = gguf_get_tensor_size(ctx, i);
        size_t offset =
            gguf_get_tensor_offset(ctx, i) + gguf_get_meta_size(ctx);
        if (strncmp(name, "embedding", strlen("embedding")) == 0) {
            if (use_mmap) {
                embed_model_ = create_infer_engine(file_name, offset, size);
            } else {
                std::vector<char> buffer_vec;
                buffer_vec.resize(size);
                read_tensor_to_buffer(file_name, offset, size, buffer_vec);
                embed_model_ = create_infer_engine(false, buffer_vec.data(), size, false);
            }
            if (embed_model_ == nullptr) {
                LLAMA_LOG_ERROR("%s: failed to load embed.hmm\n", __func__);
                return false;
            }
        }
    }
    ggml_free(ctx_data);
    gguf_free(ctx);

    return true;
}

void HouMoEmbed::houmo_init(int seq_max) {
    size_t input_num = embed_model_->GetInputNum();
    for (size_t idx = 0; idx < input_num; idx++) {
        auto input_name = embed_model_->GetInputName(idx);
        auto input_info = embed_model_->GetInputInfo(input_name).AsContiguous();
        auto shape = input_info.Shape();
        for (size_t i = 0; i < shape.size(); i++) {
            LLAMA_LOG_INFO("%s: %s input_shape: %ld\n", __func__,
                           input_name.c_str(), shape[i]);
        }
        if (idx == 0) {
            n_embedding_batch_ = shape[0];
            n_context_length_ = shape[1];
        }
        auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
        input_map_.insert(
            std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
    }

    size_t output_num = embed_model_->GetOutputNum();
    for (size_t idx = 0; idx < output_num; idx++) {
        auto output_name = embed_model_->GetOutputName(idx);

        auto output_info = embed_model_->GetOutputInfo(output_name)
                               .AsContiguous()
                               .AsType(tcim::DataType::FLOAT32);
        auto shape = output_info.Shape();
        for (size_t i = 0; i < shape.size(); i++) {
            LLAMA_LOG_INFO("%s: %s output_shape: %ld\n", __func__,
                           output_name.c_str(), shape[i]);
        }
        auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
        output_map_.insert(
            std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
    }
    if (seq_max > n_embedding_batch_) {
        LLAMA_LOG_INFO("houmo_init: seq_max(%d) is larger than "
                       "embedding_batch(%d), set seq_max "
                       "to embedding_batch\n",
                       seq_max, n_embedding_batch_);
    }
    LLAMA_LOG_INFO("embedding init finished. n_embedding_batch= %d, "
                   "n_context_lenght = %d, n_embed_=%d\n",
                   n_embedding_batch_, n_context_length_, n_embed_);
}

int HouMoEmbed::houmo_embedding(
    const std::map<int32_t, std::vector<llama_token>> &seqid_map,
    std::map<int, float *> &embeddings) {
    LLAMA_LOG_INFO("houmo_embedding: start\n");
    if (seqid_map.size() == 0 || embeddings.size() == 0) {
        LLAMA_LOG_WARN("houmo_embedding: seqid_map is empty\n");
        return 0;
    }
    auto input = input_map_.at(embed_model_->GetInputName(0));
    auto input_ptr = static_cast<int32_t *>(input.Data());
    memset(input_ptr, 0, input.MemSize());

    auto attention_mask = input_map_.at(embed_model_->GetInputName(2));
    std::vector<int16_t> attention_mask_vec(
        attention_mask.MemSize() / sizeof(int16_t), 0);

    for (int32_t i = 0; i < n_embedding_batch_; i++) {
        if (seqid_map.count(i) == 0) {
            continue;
        }

        auto &token_vec = seqid_map.at(i);
        for (size_t j = 0; j < token_vec.size(); j++) {
            LLAMA_LOG_INFO("Putting token %d at batch %d, position %lu\n",
                           token_vec[j], i, i * n_context_length_ + j);
            input_ptr[i * n_context_length_ + j] = token_vec[j];
            attention_mask_vec[i * n_context_length_ + j] = 1;
        }
    }
    attention_mask.Buffer().CopyFromHost(attention_mask_vec.data(),
                                         attention_mask.MemSize());
    auto token_type_ids = input_map_.at(embed_model_->GetInputName(1));
    std::vector<int32_t> token_type_vec(
        token_type_ids.MemSize() / sizeof(int32_t), 0);
    token_type_ids.Buffer().CopyFromHost(token_type_vec.data(),
                                         token_type_ids.MemSize());

    embed_model_->SetInput(embed_model_->GetInputName(0), input);
    embed_model_->SetInput(embed_model_->GetInputName(1), token_type_ids);
    embed_model_->SetInput(embed_model_->GetInputName(2), attention_mask);
    embed_model_->Run();
    embed_model_->Sync();

    auto output = embed_model_->GetOutput(embed_model_->GetOutputName(0));
    auto output_float = output_map_.at(embed_model_->GetOutputName(0));
    output.CastTo(output_float);
    auto output_ptr = static_cast<float *>(output_float.Data());
    // FIXME: pooling 只取最前面的n_embed_维度
    for (int i = 0; i < n_embedding_batch_; i++) {
        if (seqid_map.count(i) == 0) {
            continue;
        }
        LLAMA_LOG_INFO("houmo_embedding: get embedding for seq_id=%d\n", i);
        float *embedding_addr = embeddings[i];
        memcpy(embedding_addr, output_ptr + i * n_embed_,
               n_embed_ * sizeof(float));
    }

    return 0;
}
