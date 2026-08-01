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
#   bash engine/multimodal/scripts/fetch_cuda_dlls.sh [REFERENCE_DIR] [EXE_DIR] [CUDA_DLL]
#
#   REFERENCE_DIR  dir with CUDA runtime DLLs (default: C:/Programming/llamacpp)
#   EXE_DIR        where to drop the DLLs (default: the built exe dir)
#   CUDA_DLL       ABI-compatible ggml-cuda.dll; defaults to REFERENCE_DIR's copy
#                  (override when the reference runtime has a different ggml ABI)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REF="${1:-/c/Programming/llamacpp}"
EXE="${2:-$HERE/../build/bin/Release}"
CUDA_DLL="${3:-$REF/ggml-cuda.dll}"

if [[ ! -f "$EXE/multimodal-server.exe" ]]; then
  echo "ERROR: multimodal-server.exe not found in destination: $EXE" >&2
  echo "       Build first with: bash scripts/build_engine.sh" >&2
  exit 1
fi

DLLS=(cudart64_13.dll cublas64_13.dll cublasLt64_13.dll)

echo "copying CUDA backend DLLs:"
echo "  runtime source: $REF"
echo "  CUDA backend:   $CUDA_DLL"
echo "  destination:    $EXE"
mkdir -p "$EXE"
if [[ ! -f "$CUDA_DLL" ]]; then
  echo "  MISSING: $CUDA_DLL" >&2
  exit 1
fi
cp "$CUDA_DLL" "$EXE/ggml-cuda.dll"
if ! cmp -s "$CUDA_DLL" "$EXE/ggml-cuda.dll"; then
  echo "  ERROR: failed to verify copied DLL: $EXE/ggml-cuda.dll" >&2
  exit 1
fi
echo "  copied and verified ggml-cuda.dll"
for d in "${DLLS[@]}"; do
  if [[ ! -f "$REF/$d" ]]; then
    echo "  MISSING: $REF/$d" >&2
    exit 1
  fi
  cp "$REF/$d" "$EXE/$d"
  if [[ ! -f "$EXE/$d" ]] || ! cmp -s "$REF/$d" "$EXE/$d"; then
    echo "  ERROR: failed to verify copied DLL: $EXE/$d" >&2
    exit 1
  fi
  echo "  copied and verified $d"
done
echo "done. GPU offload will be verified by multimodal-server at startup."
