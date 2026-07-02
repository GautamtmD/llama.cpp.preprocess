# multimodal/

Our (MultiModalAgent) C++ code, kept **separate** from upstream llama.cpp so
upstream merges stay clean.

## Rules (see also the top-level repo's AGENTS.md)

- All our engine code lives under this folder. **Do not modify or subclass
  `llama-server`** or other upstream files; reference them for understanding
  (especially the batching / inference loop) only.
- **TDD, perf-first.** Every change is preceded by a test in `tests/` that
  measures performance and latency, not just correctness. Write the failing test
  and its acceptance threshold first; implement until it passes.
- Engine user stories / acceptance criteria live in [`user_stories.md`](user_stories.md).

## Layout

```
multimodal/
  src/          our C++ (session API, KV-cache management, HTTP server)
  tests/        perf + correctness tests (written first, per TDD)
  scripts/      fetch_cuda_dlls.sh — get the CUDA backend DLLs in place
  user_stories.md   engine-side acceptance criteria + latency budgets
```

## Build (Windows / MSVC)

Our `CMakeLists.txt` is a **top-level project** that pulls the llama.cpp fork
in via `add_subdirectory(../ ...)` with `LLAMA_STANDALONE=OFF`, linking
`llama` + `llama-common` + the vendored `cpp-httplib`. No upstream files are
modified.

```bat
:: from the parent repo root, in a VS x64 Developer Prompt
 cmake -S engine/multimodal -B engine/multimodal/build ^
   -G "Visual Studio 18 2026" -A x64 -DCMAKE_BUILD_TYPE=Release
 cmake --build engine/multimodal/build --config Release --target multimodal-server
```

The exe lands in `engine/multimodal/build/bin/Release/` (co-located with the
runtime DLLs so `ggml_backend_load_all()` can find the backends).

### Enabling CUDA (GPU offload)

The CPU-only build runs the 12B model at ~1.2 tok/s. For real perf (~46 tok/s on
an RTX 5070 Ti) you need the `ggml-cuda` backend.

**Compiling ggml-cuda from source currently fails** on this machine: the
installed VS 2026 (MSVC 14.4x+) is too new for CUDA 12.9 / 13.0's `nvcc` —
`cudafe++` crashes with `0xC0000005` even with `-allow-unsupported-compiler`.
Until a CUDA-supported VS (2019/2022) is installed, **reuse a prebuilt,
known-compatible `ggml-cuda.dll`** plus the CUDA 13 runtime DLLs it depends on:

```bash
bash engine/multimodal/scripts/fetch_cuda_dlls.sh
# copies ggml-cuda.dll + cudart64_13/cublas64_13/cublasLt64_13.dll
# from C:/Programming/llamacpp into engine/multimodal/build/bin/Release/
```

Then run with `--n-gpu-layers 99`. Validated empirically: 38× speedup, 11/11
tests still pass, output is coherent. ABI compat is verified by the suite —
if the upstream ggml commit drifts, this will start crashing and we'll need
to rebuild ggml-cuda from source (install VS 2022 BuildTools).

## Run

```bash
engine/multimodal/build/bin/Release/multimodal-server.exe \
  --model "/c/ML Models/Gemma4 12b/gemma-4-12b-it-qat-q4_0.gguf" \
  --port 8080 --n-gpu-layers 99
```

Endpoints are documented in the parent repo's `docs/ipc-protocol.md`. Tests:

```bash
pytest engine/multimodal/tests   # see tests/conftest.py for how to start the server
```
