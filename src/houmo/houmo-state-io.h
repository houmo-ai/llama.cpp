#pragma once

#include "houmo-llmodel.h"

#include <cstdint>
#include <string>
#include <vector>

namespace houmo_state_io {

struct pending_tensor {
    tcim::Tensor host;
    size_t index = 0;
};

template <typename T>
void write_value(llama_io_write_i & io, const T & value) {
    io.write(&value, sizeof(value));
}

template <typename T>
void read_value(llama_io_read_i & io, T & value) {
    io.read(&value, sizeof(value));
}

void reject_unsupported_flags(llama_state_seq_flags flags, const char * model);

void write_chunks(llama_io_write_i & io,
                  const std::vector<PrefillChunk> & chunks,
                  int64_t n_embd);
std::vector<PrefillChunk> read_chunks(llama_io_read_i & io,
                                      int64_t n_embd,
                                      uint64_t expected_tokens);
uint64_t count_chunk_tokens(const std::vector<PrefillChunk> & chunks);

void write_tensors(llama_io_write_i & io,
                   const std::vector<std::string> & names,
                   const std::vector<tcim::Tensor> & tensors);
std::vector<pending_tensor> read_tensors(
    llama_io_read_i & io,
    const std::vector<std::string> & names,
    const std::vector<tcim::Tensor> & tensors);
void commit_tensors(std::vector<pending_tensor> & pending,
                    std::vector<tcim::Tensor> & tensors);
std::vector<tcim::Tensor> backup_tensors(
    const std::vector<tcim::Tensor> & tensors);
void restore_tensors(const std::vector<tcim::Tensor> & backup,
                     std::vector<tcim::Tensor> & tensors);

} // namespace houmo_state_io
