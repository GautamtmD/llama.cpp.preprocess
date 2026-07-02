#!/usr/bin/env bash
# Copy the CUDA backend + runtime DLLs needed for GPU inference next to the
# multimodal-server exe.
#
# WHY: our CMake build produces the CPU-only ggml backends. Compiling ggml-cuda
# from source requires a CUDA-supported host compiler (VS 2019/2022); the
# installed VS 2026 is too new for CUDA 12.9/13.0's nvcc (cudafe++ crashes). So
# we reuse a prebuilt, known-compatible ggml-cuda.dll from a reference llama.cpp
# release, plus the CUDA 13 runtime DLLs it depends on (which ship with that
# release, not with the CUDA Toolkit install here).
#
# This was validated empirically: 1.2 tok/s (CPU) -> 45.9 tok/s (GPU), 38x.
# ABI compatibility is verified by the test suite (11/11 pass) + coherent output.
#
# Usage:  bash audio.../nope -- see engine/multimodal/scripts/fetch_cuda_dlls.sh
#   bash engine/multimodal/scripts/fetch_cuda_dlls.sh [REFERENCE_DIR] [EXE_DIR]
#
#   REFERENCE_DIR  dir with a working ggml-cuda.dll + CUDA runtime DLLs
#                  (default: C:/Programming/llamacpp)
#   EXE_DIR        where to drop the DLLs (default: the built exe dir)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REF="${1:-/c/Programming/llamacpp}"
EXE="${2:-$HERE/../../build/bin/Release}"

DLLS=(ggml-cuda.dll cudart64_13.dll cublas64_13.dll cublasLt64_13.dll)

echo "copying CUDA backend DLLs:"
echo "  from: $REF"
echo "  to:   $EXE"
mkdir -p "$EXE"
for d in "${DLLS[@]}"; do
  if [[ ! -f "$REF/$d" ]]; then
    echo "  MISSING: $REF/$d" >&2
    exit 1
  fi
  cp "$REF/$d" "$EXE/$d"
  echo "  copied $d"
done
echo "done. 'multimodal-server --n-gpu-layers 99' will now offload to GPU."
