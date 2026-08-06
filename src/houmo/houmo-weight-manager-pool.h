#pragma once

#include "tcim/tcim_runtime.h"

#include <memory>
#include <string>
#include <vector>

class HoumoEmbeddingLayer;

struct houmo_weight_manager_handle {
    tcim::Module::WeightManager manager;
};

// Returns a process-wide shared WeightManager for the same model and device
// topology. The pool keeps weak references only, so device weights are released
// after the last Module using the handle is destroyed.
std::shared_ptr<houmo_weight_manager_handle> houmo_weight_manager_acquire(
        const std::string & model_path,
        const std::vector<int> & device_ids,
        const std::string & backend);

std::shared_ptr<HoumoEmbeddingLayer> houmo_embedding_layer_acquire(
        const std::string & model_path,
        int64_t offset,
        int64_t size);
