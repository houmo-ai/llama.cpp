#include "houmo-state-io.h"

#include <limits>
#include <stdexcept>

namespace houmo_state_io {
namespace {
constexpr uint64_t MAX_CHUNKS = 1ULL << 20;
constexpr uint64_t MAX_ELEMENTS = 1ULL << 34;
constexpr uint32_t MAX_TENSORS = 1U << 20;
constexpr uint32_t MAX_RANK = 32;
}

void reject_unsupported_flags(llama_state_seq_flags flags, const char * model) {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) != 0) {
        throw std::runtime_error(std::string(model) + " Houmo state does not support ON_DEVICE");
    }
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != 0) {
        throw std::runtime_error(std::string(model) + " partial Houmo state is not implemented yet");
    }
}

uint64_t count_chunk_tokens(const std::vector<PrefillChunk> & chunks) {
    uint64_t total = 0;
    for (const auto & chunk : chunks) {
        if (chunk.n_tokens > std::numeric_limits<uint64_t>::max() - total) {
            throw std::runtime_error("Houmo chunk token count overflow");
        }
        total += chunk.n_tokens;
    }
    return total;
}

void write_chunks(llama_io_write_i & io,
                  const std::vector<PrefillChunk> & chunks,
                  int64_t n_embd) {
    if (chunks.size() > MAX_CHUNKS || n_embd <= 0) {
        throw std::runtime_error("invalid Houmo chunk schema");
    }
    const uint64_t count = chunks.size();
    write_value(io, count);
    for (const auto & chunk : chunks) {
        const uint8_t is_embedding = chunk.is_embedding;
        const uint8_t bidirectional = chunk.bidirectional_attention;
        const uint64_t n_tokens = chunk.n_tokens;
        const uint64_t token_count = chunk.tokens.size();
        const uint64_t embedding_count = chunk.embeddings.size();
        if (n_tokens > MAX_ELEMENTS || token_count > MAX_ELEMENTS ||
            embedding_count > MAX_ELEMENTS) {
            throw std::runtime_error("Houmo chunk size is too large");
        }
        if (chunk.is_embedding) {
            if (n_tokens > MAX_ELEMENTS / static_cast<uint64_t>(n_embd) ||
                !chunk.tokens.empty() ||
                embedding_count != n_tokens * static_cast<uint64_t>(n_embd)) {
                throw std::runtime_error("invalid Houmo embedding chunk payload");
            }
        } else {
            if (n_tokens > MAX_ELEMENTS / static_cast<uint64_t>(n_embd)) {
                throw std::runtime_error("Houmo token embedding size overflow");
            }
            const uint64_t expected_embeddings =
                n_tokens * static_cast<uint64_t>(n_embd);
            if (token_count != n_tokens ||
                (embedding_count != 0 &&
                 embedding_count != expected_embeddings)) {
                throw std::runtime_error(
                    "invalid Houmo token chunk payload: n_tokens=" +
                    std::to_string(n_tokens) + " token_count=" +
                    std::to_string(token_count) + " embedding_count=" +
                    std::to_string(embedding_count) + " expected_embeddings=" +
                    std::to_string(expected_embeddings));
            }
        }
        write_value(io, is_embedding);
        write_value(io, bidirectional);
        write_value(io, n_tokens);
        write_value(io, chunk.nx);
        write_value(io, chunk.ny);
        write_value(io, token_count);
        if (token_count > 0) {
            io.write(chunk.tokens.data(), token_count * sizeof(llama_token));
        }
        write_value(io, embedding_count);
        if (embedding_count > 0) {
            io.write(chunk.embeddings.data(), embedding_count * sizeof(uint16_t));
        }
    }
}

std::vector<PrefillChunk> read_chunks(llama_io_read_i & io,
                                      int64_t n_embd,
                                      uint64_t expected_tokens) {
    if (n_embd <= 0 || expected_tokens > MAX_ELEMENTS) {
        throw std::runtime_error("invalid Houmo chunk schema");
    }
    uint64_t count = 0;
    read_value(io, count);
    if (count > MAX_CHUNKS) {
        throw std::runtime_error("Houmo state contains too many chunks");
    }
    std::vector<PrefillChunk> chunks;
    chunks.reserve(count);
    for (uint64_t i = 0; i < count; ++i) {
        uint8_t is_embedding = 0;
        uint8_t bidirectional = 0;
        uint64_t n_tokens = 0;
        uint64_t token_count = 0;
        uint64_t embedding_count = 0;
        PrefillChunk chunk;
        read_value(io, is_embedding);
        read_value(io, bidirectional);
        read_value(io, n_tokens);
        read_value(io, chunk.nx);
        read_value(io, chunk.ny);
        read_value(io, token_count);
        if (n_tokens > MAX_ELEMENTS || token_count > MAX_ELEMENTS) {
            throw std::runtime_error("Houmo chunk size is too large");
        }
        chunk.is_embedding = is_embedding != 0;
        chunk.bidirectional_attention = bidirectional != 0;
        chunk.n_tokens = n_tokens;
        chunk.tokens.resize(token_count);
        if (token_count > 0) {
            io.read(chunk.tokens.data(), token_count * sizeof(llama_token));
        }
        read_value(io, embedding_count);
        if (embedding_count > MAX_ELEMENTS) {
            throw std::runtime_error("Houmo embedding chunk is too large");
        }
        chunk.embeddings.resize(embedding_count);
        if (embedding_count > 0) {
            io.read(chunk.embeddings.data(), embedding_count * sizeof(uint16_t));
        }
        if (chunk.is_embedding) {
            if (n_tokens > MAX_ELEMENTS / static_cast<uint64_t>(n_embd)) {
                throw std::runtime_error("Houmo embedding chunk size overflow");
            }
            const uint64_t expected = n_tokens * static_cast<uint64_t>(n_embd);
            if (!chunk.tokens.empty() || embedding_count != expected) {
                throw std::runtime_error("invalid Houmo embedding chunk payload");
            }
        } else {
            if (n_tokens > MAX_ELEMENTS / static_cast<uint64_t>(n_embd)) {
                throw std::runtime_error("Houmo token embedding size overflow");
            }
            const uint64_t expected_embeddings =
                n_tokens * static_cast<uint64_t>(n_embd);
            if (token_count != n_tokens ||
                (embedding_count != 0 &&
                 embedding_count != expected_embeddings)) {
                throw std::runtime_error(
                    "invalid Houmo token chunk payload: n_tokens=" +
                    std::to_string(n_tokens) + " token_count=" +
                    std::to_string(token_count) + " embedding_count=" +
                    std::to_string(embedding_count) + " expected_embeddings=" +
                    std::to_string(expected_embeddings));
            }
        }
        chunks.push_back(std::move(chunk));
    }
    if (count_chunk_tokens(chunks) != expected_tokens) {
        throw std::runtime_error("Houmo chunk history position mismatch");
    }
    return chunks;
}

void write_tensors(llama_io_write_i & io,
                   const std::vector<std::string> & names,
                   const std::vector<tcim::Tensor> & tensors) {
    if (names.size() != tensors.size()) {
        throw std::runtime_error("Houmo tensor metadata mismatch");
    }
    if (tensors.size() > MAX_TENSORS) {
        throw std::runtime_error("too many Houmo tensors");
    }
    const uint32_t count = static_cast<uint32_t>(tensors.size());
    write_value(io, count);
    for (size_t i = 0; i < tensors.size(); ++i) {
        io.write_string(names[i]);
        const auto & info = tensors[i].Info();
        const auto & shape = info.Shape();
        const auto & stride = info.Stride();
        if (shape.size() > MAX_RANK || stride.size() > MAX_RANK) {
            throw std::runtime_error("invalid Houmo tensor rank");
        }
        const uint32_t rank = shape.size();
        const uint32_t stride_count = stride.size();
        const int32_t dtype = static_cast<int32_t>(info.DataType());
        const uint64_t byte_size = tensors[i].MemSize();
        write_value(io, rank);
        if (rank > 0) io.write(shape.data(), rank * sizeof(shape[0]));
        write_value(io, stride_count);
        if (stride_count > 0) io.write(stride.data(), stride_count * sizeof(stride[0]));
        write_value(io, dtype);
        write_value(io, byte_size);
        auto host = tensors[i].ToHost(false);
        if (host.GetInitStatus() != tcim::Status::OK || host.MemSize() != byte_size) {
            throw std::runtime_error("failed to copy Houmo tensor to host");
        }
        io.write(host.Data(), byte_size);
    }
}

std::vector<pending_tensor> read_tensors(
    llama_io_read_i & io,
    const std::vector<std::string> & names,
    const std::vector<tcim::Tensor> & tensors) {
    uint32_t count = 0;
    read_value(io, count);
    if (count > MAX_TENSORS || count != tensors.size() || names.size() != tensors.size()) {
        throw std::runtime_error("Houmo tensor count mismatch");
    }
    std::vector<pending_tensor> pending;
    pending.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        std::string name;
        uint32_t rank = 0;
        uint32_t stride_count = 0;
        int32_t dtype = 0;
        uint64_t byte_size = 0;
        io.read_string(name);
        read_value(io, rank);
        if (rank > 32) throw std::runtime_error("invalid Houmo tensor rank");
        std::vector<int64_t> shape(rank);
        if (rank > 0) io.read(shape.data(), rank * sizeof(shape[0]));
        read_value(io, stride_count);
        if (stride_count > 32) throw std::runtime_error("invalid Houmo tensor stride rank");
        std::vector<int64_t> stride(stride_count);
        if (stride_count > 0) io.read(stride.data(), stride_count * sizeof(stride[0]));
        read_value(io, dtype);
        read_value(io, byte_size);
        const auto & dst = tensors[i];
        const auto & info = dst.Info();
        if (name != names[i] || shape != info.Shape() || stride != info.Stride() ||
            dtype != static_cast<int32_t>(info.DataType()) || byte_size != dst.MemSize()) {
            throw std::runtime_error("Houmo tensor schema mismatch");
        }
        auto host = tcim::Tensor::CreateHostTensor(info, byte_size);
        if (host.GetInitStatus() != tcim::Status::OK) {
            throw std::runtime_error("failed to allocate Houmo host restore tensor");
        }
        io.read(host.Data(), byte_size);
        pending.push_back({std::move(host), i});
    }
    return pending;
}

void commit_tensors(std::vector<pending_tensor> & pending,
                    std::vector<tcim::Tensor> & tensors) {
    for (auto & entry : pending) {
        if (entry.host.CopyTo(tensors[entry.index]) != tcim::Status::OK) {
            throw std::runtime_error("failed to restore Houmo tensor");
        }
    }
}

std::vector<tcim::Tensor> backup_tensors(
    const std::vector<tcim::Tensor> & tensors) {
    std::vector<tcim::Tensor> backup;
    backup.reserve(tensors.size());
    for (const auto & tensor : tensors) {
        auto host = tensor.ToHost(false);
        if (host.GetInitStatus() != tcim::Status::OK ||
            host.MemSize() != tensor.MemSize()) {
            throw std::runtime_error("failed to back up Houmo tensor");
        }
        backup.push_back(std::move(host));
    }
    return backup;
}

void restore_tensors(const std::vector<tcim::Tensor> & backup,
                     std::vector<tcim::Tensor> & tensors) {
    if (backup.size() != tensors.size()) {
        throw std::runtime_error("Houmo tensor rollback count mismatch");
    }
    for (size_t i = 0; i < backup.size(); ++i) {
        if (backup[i].CopyTo(tensors[i]) != tcim::Status::OK) {
            throw std::runtime_error("failed to roll back Houmo tensor");
        }
    }
}

} // namespace houmo_state_io
