// Shared fail-closed execution policy for the live server and performance
// benchmarks. Pure/header-only so it is covered by multimodal-util-tests.
#pragma once

#include <cstddef>
#include <string>

inline size_t gpu_allocation_delta(size_t free_before, size_t free_after) {
    return free_before > free_after ? free_before - free_after : 0;
}

// Require evidence from the backend's device-memory counters after model load,
// not merely a requested --n-gpu-layers value. CPU-only execution requires the
// explicit --allow-cpu opt-in.
inline bool gpu_execution_allowed(bool allow_cpu, size_t actual_gpu_model_bytes) {
    return allow_cpu || actual_gpu_model_bytes > 0;
}

inline std::string gpu_execution_error(const std::string & program) {
    return "GPU load was not successful; " + program +
           " cannot run without GPU model offload. Pass --allow-cpu to "
           "explicitly allow intentional CPU execution.";
}
