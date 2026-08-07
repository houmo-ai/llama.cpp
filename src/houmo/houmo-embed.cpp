#include "houmo-embed.h"

#include "ggml.h"
#include "gguf.h"
#include "llama-batch.h"
#include "llama-impl.h"

#include <cstring>
#include <limits>
#include <sstream>

HouMoEmbed::HouMoEmbed() {
    // do nothing
}

HouMoEmbed::~HouMoEmbed() {
    // do nothing
}

bool HouMoEmbed::houmo_load(llama_model_loader & ml, std::vector<int> device_ids) {
    std::string    file_name = ml.fname_;
    gguf_context * ctx       = ml.metadata_ptr.get();

    const int architecture_key = gguf_find_key(ctx, "general.architecture");
    if (architecture_key < 0) {
        LLAMA_LOG_ERROR("%s: general.architecture is missing from the GGUF metadata\n", __func__);
        return false;
    }
    std::string model_arch = gguf_get_val_str(ctx, architecture_key);
    parser_gguf_parameters(ctx, model_arch);
    if (n_embd_ <= 0) {
        LLAMA_LOG_ERROR("%s: invalid embedding width=%lld from GGUF metadata\n", __func__,
                        static_cast<long long>(n_embd_));
        return false;
    }

    // 0. 获取设置的device_ids. env>params>default.
    device_ids_ = device_ids;
    parse_deviceids();

    // 1. 加载 embedding
    const int n_tensors = gguf_get_n_tensors(ctx);
    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        LLAMA_LOG_INFO("tensor[%d]: name = %s\n", i, name);
        const size_t size   = gguf_get_tensor_size(ctx, i);
        size_t       offset = gguf_get_tensor_offset(ctx, i) + gguf_get_meta_size(ctx);
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

void HouMoEmbed::houmo_init(houmo_memory_i *    memory,
                            int                 seq_max,
                            ggml_abort_callback abort_callback,
                            void *              abort_callback_data) {
    (void) memory;
    abort_callback_data_ = abort_callback_data;
    abort_callback_      = abort_callback;
    size_t input_num     = embed_model_->GetInputNum();
    if (input_map_.size() > 0) {
        return;
    }
    for (size_t idx = 0; idx < input_num; idx++) {
        auto input_name = embed_model_->GetInputName(idx);
        auto input_info = embed_model_->GetInputInfo(input_name).AsContiguous();
        auto shape      = input_info.Shape();
        for (size_t i = 0; i < shape.size(); i++) {
            LLAMA_LOG_INFO("%s: %s input_shape: %lld\n", __func__, input_name.c_str(),
                           static_cast<long long>(shape[i]));
        }
        if (idx == 0) {
            if (shape.size() < 2 || shape[0] <= 0 || shape[1] <= 0 || shape[0] > std::numeric_limits<int>::max() ||
                shape[1] > std::numeric_limits<int>::max()) {
                LLAMA_LOG_ERROR("%s: invalid primary input shape for %s\n", __func__, input_name.c_str());
                return;
            }
            n_embedding_batch_ = static_cast<int>(shape[0]);
            context_length_    = static_cast<int>(shape[1]);
        }
        auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
        input_map_.insert(std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
    }

    size_t output_num = embed_model_->GetOutputNum();
    for (size_t idx = 0; idx < output_num; idx++) {
        auto output_name = embed_model_->GetOutputName(idx);

        auto output_info = embed_model_->GetOutputInfo(output_name).AsContiguous().AsType(tcim::DataType::FLOAT32);
        auto shape       = output_info.Shape();
        for (size_t i = 0; i < shape.size(); i++) {
            LLAMA_LOG_INFO("%s: %s output_shape: %lld\n", __func__, output_name.c_str(),
                           static_cast<long long>(shape[i]));
        }

        if (idx == 0) {
            size_t element_count = 1;
            bool   valid_shape   = !shape.empty();
            for (const auto dim : shape) {
                if (dim <= 0 || static_cast<uint64_t>(dim) > std::numeric_limits<size_t>::max() / element_count) {
                    valid_shape = false;
                    break;
                }
                element_count *= static_cast<size_t>(dim);
            }

            if (!valid_shape || shape[0] != n_embedding_batch_) {
                LLAMA_LOG_ERROR("%s: unsupported output shape for %s: expected batch=%d\n", __func__,
                                output_name.c_str(), n_embedding_batch_);
            } else {
                output_element_count_ = element_count;
                output_batch_stride_  = element_count / static_cast<size_t>(n_embedding_batch_);
                output_width_         = shape.size() >= 3 ? static_cast<size_t>(shape.back()) : output_batch_stride_;
                output_layout_valid_  = output_batch_stride_ > 0 && output_width_ > 0 &&
                                       output_width_ <= output_batch_stride_ &&
                                       output_width_ <= static_cast<size_t>(n_embd_);

                std::ostringstream shape_stream;
                shape_stream << '[';
                for (size_t i = 0; i < shape.size(); ++i) {
                    if (i > 0) {
                        shape_stream << ',';
                    }
                    shape_stream << shape[i];
                }
                shape_stream << ']';

                LLAMA_LOG_INFO(
                    "%s: output layout name=%s shape=%s elements=%zu "
                    "batch_stride=%zu output_width=%zu destination_width=%lld\n",
                    __func__, output_name.c_str(), shape_stream.str().c_str(), output_element_count_,
                    output_batch_stride_, output_width_, static_cast<long long>(n_embd_));
            }
        }
        auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
        output_map_.insert(std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
    }
    (void) seq_max;
    if (!output_layout_valid_) {
        LLAMA_LOG_ERROR("%s: failed to initialize a safe output layout\n", __func__);
        return;
    }
    LLAMA_LOG_INFO(
        "embedding init finished. n_embedding_batch=%d, "
        "context_length=%d, output_batch_stride=%zu, output_width=%zu\n",
        n_embedding_batch_, context_length_, output_batch_stride_, output_width_);
}

int HouMoEmbed::houmo_embedding(const std::map<int32_t, std::vector<llama_token>> & seqid_map,
                                std::map<int, float *> &                            embeddings) {
    LLAMA_LOG_INFO("houmo_embedding: start\n");
    if (seqid_map.size() == 0 || embeddings.size() == 0) {
        LLAMA_LOG_WARN("houmo_embedding: seqid_map is empty\n");
        return 0;
    }
    if (!output_layout_valid_) {
        LLAMA_LOG_ERROR("%s: output layout is not initialized\n", __func__);
        return -1;
    }

    for (const auto & [seq_id, token_vec] : seqid_map) {
        if (seq_id < 0 || seq_id >= n_embedding_batch_) {
            LLAMA_LOG_ERROR("%s: invalid seq_id=%d, embedding_batch=%d\n", __func__, seq_id, n_embedding_batch_);
            return -1;
        }
        if (token_vec.size() > static_cast<size_t>(context_length_)) {
            LLAMA_LOG_ERROR("%s: sequence %d has %zu tokens, context_length=%d\n", __func__, seq_id, token_vec.size(),
                            context_length_);
            return 1;
        }
        const auto embedding = embeddings.find(seq_id);
        if (embedding == embeddings.end() || embedding->second == nullptr) {
            LLAMA_LOG_ERROR("%s: missing output buffer for seq_id=%d\n", __func__, seq_id);
            return -1;
        }
    }

    auto                 input = input_map_.at(embed_model_->GetInputName(0));
    std::vector<int32_t> input_data(input.MemSize() / sizeof(int32_t), 0);
    auto                 attention_mask = input_map_.at(embed_model_->GetInputName(2));
    std::vector<int16_t> attention_mask_vec(attention_mask.MemSize() / sizeof(int16_t), 0);
    for (const auto & [seq_id, token_vec] : seqid_map) {
        const size_t input_offset = static_cast<size_t>(seq_id) * context_length_;
        if (input_offset > input_data.size() || token_vec.size() > input_data.size() - input_offset ||
            input_offset > attention_mask_vec.size() || token_vec.size() > attention_mask_vec.size() - input_offset) {
            LLAMA_LOG_ERROR("%s: input buffer overflow for seq_id=%d, tokens=%zu\n", __func__, seq_id,
                            token_vec.size());
            return -1;
        }
        for (size_t j = 0; j < token_vec.size(); j++) {
            LLAMA_LOG_DEBUG("Putting token %d at batch %d, position %lu\n", token_vec[j], seq_id,
                            static_cast<unsigned long>(input_offset + j));
            input_data[input_offset + j]         = token_vec[j];
            attention_mask_vec[input_offset + j] = 1;
        }
    }
    input.Buffer().CopyFromHost(input_data.data(), input.MemSize());
    attention_mask.Buffer().CopyFromHost(attention_mask_vec.data(), attention_mask.MemSize());
    auto                 token_type_ids = input_map_.at(embed_model_->GetInputName(1));
    std::vector<int32_t> token_type_vec(token_type_ids.MemSize() / sizeof(int32_t), 0);
    token_type_ids.Buffer().CopyFromHost(token_type_vec.data(), token_type_ids.MemSize());

    embed_model_->SetInput(embed_model_->GetInputName(0), input);
    embed_model_->SetInput(embed_model_->GetInputName(1), token_type_ids);
    embed_model_->SetInput(embed_model_->GetInputName(2), attention_mask);
    if (auto status = embed_model_->Run(); status != tcim::Status::OK) {
        LLAMA_LOG_ERROR("%s: embed_model_->Run() failed, result = %d\n", __func__, static_cast<int>(status));
        return -1;
    }
    embed_model_->Sync();

    auto output       = embed_model_->GetOutput(embed_model_->GetOutputName(0));
    auto output_float = output_map_.at(embed_model_->GetOutputName(0));
    output.CastTo(output_float);
    auto         output_ptr             = static_cast<float *>(output_float.Data());
    const size_t output_buffer_elements = output_float.MemSize() / sizeof(float);
    if (output_ptr == nullptr || output_element_count_ > output_buffer_elements) {
        LLAMA_LOG_ERROR("%s: invalid output buffer: data=%p, shape_elements=%zu, buffer_elements=%zu\n", __func__,
                        static_cast<void *>(output_ptr), output_element_count_, output_buffer_elements);
        return -1;
    }

    for (const auto & [seq_id, token_vec] : seqid_map) {
        (void) token_vec;
        const size_t output_offset = static_cast<size_t>(seq_id) * output_batch_stride_;
        if (output_offset > output_element_count_ || output_width_ > output_element_count_ - output_offset) {
            LLAMA_LOG_ERROR(
                "%s: output buffer overflow for seq_id=%d, offset=%zu, "
                "width=%zu, elements=%zu\n",
                __func__, seq_id, output_offset, output_width_, output_element_count_);
            return -1;
        }

        float * embedding_addr = embeddings.at(seq_id);
        std::memcpy(embedding_addr, output_ptr + output_offset, output_width_ * sizeof(float));
        LLAMA_LOG_DEBUG("%s: copied seq_id=%d output_offset=%zu output_width=%zu\n", __func__, seq_id, output_offset,
                        output_width_);
    }

    return 0;
}
