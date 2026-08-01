// Shared fail-closed execution policy for the live server and performance
// benchmarks. Pure/header-only so it is covered by multimodal-util-tests.
#pragma once

#include <string>

// `gpu_model_layers` is the number of model layers assigned to a GPU after
// model load. CPU-only execution requires the explicit --allow-cpu opt-in.
inline bool gpu_execution_allowed(bool allow_cpu, int gpu_model_layers) {
    return allow_cpu || gpu_model_layers > 0;
}

inline std::string gpu_execution_error(const std::string & program) {
    return "GPU load was not successful; " + program +
           " cannot run without GPU model offload. Pass --allow-cpu to "
           "explicitly allow intentional CPU execution.";
}
