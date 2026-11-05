#pragma once

#include "tcim/tcim_runtime.h"

#include <memory>
#include <string>
#include <vector>

class HoumoEmbeddingLayer;

struct houmo_weight_manager_handle {
    tcim::Module::WeightManager manager;
};

struct houmo_content_range {
    std::string name;
    int64_t offset;
    int64_t size;
};

// Returns a content-based identity for the named file ranges. An empty string
// means that the identity could not be calculated.
std::string houmo_content_identity(
        const std::string & model_path,
        const std::vector<houmo_content_range> & ranges);

// Returns a process-wide shared WeightManager for identical embedding content,
// backend, and device topology. The pool keeps weak references only, so device
// weights are released after the last Module using the handle is destroyed.
std::shared_ptr<houmo_weight_manager_handle> houmo_weight_manager_acquire(
        const std::string & model_path,
        const std::vector<int> & device_ids,
        const std::string & backend,
        const std::string & embedding_identity);

std::shared_ptr<HoumoEmbeddingLayer> houmo_embedding_layer_acquire(
        const std::string & model_path,
        int64_t offset,
        int64_t size,
        const std::string & embedding_identity,
        int npu_device_id);
