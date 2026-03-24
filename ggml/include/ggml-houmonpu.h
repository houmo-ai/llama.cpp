#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_HOUMONPU_NAME "HoumoNPU"
#define GGML_HOUMONPU_MAX_DEVICES 16

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_houmonpu_reg(void);

GGML_BACKEND_API int ggml_backend_houmonpu_get_device_count(void);
GGML_BACKEND_API void ggml_backend_houmonpu_get_device_description(int device, char *description, size_t description_size);
GGML_BACKEND_API void ggml_backend_houmonpu_get_device_memory(int device, size_t *free, size_t *total);

#ifdef  __cplusplus
}
#endif
