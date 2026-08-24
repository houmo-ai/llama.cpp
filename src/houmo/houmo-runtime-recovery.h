#pragma once

#include "tcim/tcim_runtime.h"

#include <memory>
#include <vector>

struct houmo_runtime_recovery_handle;
using houmo_runtime_recovery_handle_ptr = std::shared_ptr<houmo_runtime_recovery_handle>;

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef HOUMO_RUNTIME_RECOVERY_BUILD
#            define HOUMO_RUNTIME_RECOVERY_API __declspec(dllexport)
#        else
#            define HOUMO_RUNTIME_RECOVERY_API __declspec(dllimport)
#        endif
#    else
#        define HOUMO_RUNTIME_RECOVERY_API __attribute__((visibility("default")))
#    endif
#else
#    define HOUMO_RUNTIME_RECOVERY_API
#endif

// Returns a process-wide recovery handle for the supplied physical devices.
// Handles with overlapping device-id sets share the state for every device in
// the overlap, even when the complete device topology is different.
HOUMO_RUNTIME_RECOVERY_API houmo_runtime_recovery_handle_ptr
houmo_runtime_recovery_acquire(const std::vector<int> & device_ids);

// Runs exactly one TCIM inference operation, retrying only that operation after
// a recoverable runtime reset. Module inputs, caches and bindings are preserved
// by TCIM across reset, so callers must not replay their enclosing prefill or
// decode operation.
HOUMO_RUNTIME_RECOVERY_API tcim::Status houmo_run_with_retry(tcim::Module &                            module,
                                                             const houmo_runtime_recovery_handle_ptr & recovery,
                                                             const char *                              operation);
