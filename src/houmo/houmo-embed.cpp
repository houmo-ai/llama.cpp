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

bool HouMoEmbed::houmo_load(llama_model_loader & ml) {
    std::string file_name = ml.fname_;
    gguf_context *ctx = ml.metadata_ptr.get();

    // 1. 加载 embedding
    const int n_tensors = gguf_get_n_tensors(ctx);
    for (int i = 0; i < n_tensors; ++i) {
        const char *name = gguf_get_tensor_name(ctx, i);
        LLAMA_LOG_INFO("tensor[%d]: name = %s\n", i, name);
        const size_t size = gguf_get_tensor_size(ctx, i);
        size_t offset =
            gguf_get_tensor_offset(ctx, i) + gguf_get_meta_size(ctx);
        if (strncmp(name, "embedding", strlen("embedding")) == 0 ||
            strncmp(name, "reranker", strlen("reranker")) == 0) {
            if (ml.use_mmap) {
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
    return true;
}

void HouMoEmbed::houmo_init(houmo_memory_i *memory, int seq_max,
                            ggml_abort_callback abort_callback,
                            void *abort_callback_data) {
    (void)memory;
    abort_callback_data_ = abort_callback_data;
    abort_callback_ = abort_callback;
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
            context_length_ = shape[1];
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
                   "context_length_ = %d, n_embed_=%d\n",
                   n_embedding_batch_, context_length_, n_embed_);
}

int HouMoEmbed::houmo_embedding(
    const std::map<int32_t, std::vector<llama_token>> &seqid_map,
    std::map<int, float *> &embeddings) {
    LLAMA_LOG_INFO("houmo_embedding: start\n");
    if (seqid_map.size() == 0 || embeddings.size() == 0) {
        LLAMA_LOG_WARN("houmo_embedding: seqid_map is empty\n");
        return 0;
    }
    if (seqid_map.size() == 1 &&
        seqid_map.begin()->second.size() > (size_t) context_length_) {
        LLAMA_LOG_INFO("houmo_embedding: exceed context length %d\n",
                       context_length_);
        return 1;
    }
    auto input = input_map_.at(embed_model_->GetInputName(0));
    std::vector<int32_t> input_data(input.MemSize() / sizeof(int32_t), 0);
    auto attention_mask = input_map_.at(embed_model_->GetInputName(2));
    std::vector<int16_t> attention_mask_vec(
        attention_mask.MemSize() / sizeof(int16_t), 0);
    for (int32_t i = 0; i < n_embedding_batch_; i++) {
        if (seqid_map.count(i) == 0) {
            continue;
        }

        auto &token_vec = seqid_map.at(i);
        for (size_t j = 0; j < token_vec.size(); j++) {
            LLAMA_LOG_DEBUG("Putting token %d at batch %d, position %lu\n",
                           token_vec[j], i, i * context_length_ + j);
            input_data[i * context_length_ + j] = token_vec[j];
            attention_mask_vec[i * context_length_ + j] = 1;
        }
    }
    input.Buffer().CopyFromHost(input_data.data(), input.MemSize());
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
        memcpy(embedding_addr, output_ptr + i * n_embed_ * context_length_,
               n_embed_ * sizeof(float));
    }

    return 0;
}
