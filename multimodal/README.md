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
- **GPU only.** Acceptance criteria, perf thresholds and the test suite target
  the GPU build. Do not add CPU test suites or CPU perf gates; `--allow-cpu`
  exists for a deliberate dry check of one test, never as a validation target.

## Layout

```
multimodal/
  src/          our C++ (session API, KV-cache management, HTTP server)
  tests/        perf + correctness tests (written first, per TDD)
  user_stories.md   engine-side acceptance criteria + latency budgets
```

## Build (Windows / MSVC)

Our `CMakeLists.txt` is a **top-level project** that pulls the llama.cpp fork
in via `add_subdirectory(../ ...)` with `LLAMA_STANDALONE=OFF`, linking
`llama` + `llama-common` + the vendored `cpp-httplib`. No upstream files are
modified.

Use the parent repo's `scripts/build_engine.sh` (Ninja + shared ccache; it
loads vcvars and copies the DLLs next to the exe):

```bash
bash scripts/build_engine.sh --cuda   # GPU build (ggml-cuda from source)
bash scripts/build_engine.sh          # CPU-only build
```

The exe lands in `engine/multimodal/build/bin/Release/` (co-located with the
runtime DLLs so `ggml_backend_load_all()` can find the backends).

### Enabling CUDA (GPU offload)

`build_engine.sh --cuda` compiles the `ggml-cuda` backend from source
(`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120`, RTX 50xx) and copies the
CUDA runtime DLLs (`cudart64_13`, `cublas64_13`, `cublasLt64_13`) from the
toolkit next to the exe. nvcc 12.9 / 13.0 rejects VS 2026's default MSVC 19.5x
(`cudafe++` crashes), so the script loads the **VS 2022 v143 toolset (MSVC
14.4x)** with `vcvars64.bat -vcvars_ver=14.44`; VS 2026 installs it as the
"MSVC v143 build tools" component. Overrides: `CUDA_PATH_V`, `CUDA_ARCHS`,
`MSVC_TOOLSET`. The old prebuilt-DLL route (`fetch_cuda_dlls.sh`) is
**removed**: a `ggml-cuda.dll` from another llama.cpp build does not register
against this fork's `ggml-base` and the server saw 0 GPUs.

Run with `--n-gpu-layers 99` (the default). GPU execution is fail-closed: the
server and performance benchmarks exit nonzero if no model layer is assigned
to a GPU. Pass `--allow-cpu` only for an intentional CPU-only run. The error
names that opt-in so a CPU fallback cannot be mistaken for a GPU benchmark.

**Pool size on 16 GB GPUs.** The default pool is `--max-sequences 4` x
`--ctx-size 4096`. The scheduler needs the full-size SWA cache, so 8 x 4096
with Gemma 4 12B Q4 fills a 16 GB card and spills into system memory (WDDM
fallback, ~5x slower decodes); 4 uses ~12.6 GB.

## Model reasoning configuration

`reasoning_effort` is available only when the loaded model has a complete
reasoning configuration. Automatic configuration requires exact GGUF
`general.architecture: gemma4`; description matches and other Gemma generations
do not qualify. An explicit model JSON config can declare another model's
equivalent:

```json
{
  "reasoning": {
    "start_marker": "<think>",
    "end_marker": "</think>",
    "effort_budgets": {
      "none": 0,
      "minimal": 64,
      "low": 256,
      "medium": 1024,
      "high": -1
    }
  }
}
```

`none` must be `0`; finite budgets must increase strictly; `high` is either
larger than `medium` or `-1` for unrestricted reasoning. Missing markers,
budgets, or invalid ordering make the capability unavailable, so every explicit
effort request fails with HTTP 400 rather than becoming a no-op.

Per-session generated replay restores `common/`'s reasoning marker/budget state
when chunked `/generate` calls rebuild the sampler. It is cleared by successful
inject, copied by fork, retained through offload/load, and unchanged by
cancellation rewind.

The effective budget locks when a turn first commits generated output. The
engine compares resolved budgets: for the shipped Gemma 4 mapping, omission and
`high` both resolve to unrestricted (`-1`) and may alternate. A custom finite
`high` differs from omission. Any mid-turn request whose resolved budget differs
from the lock returns HTTP 400. Successful inject clears the lock for the next
turn.

## Run

```bash
engine/multimodal/build/bin/Release/multimodal-server.exe \
  --model "/c/ML Models/Gemma4 12b/gemma-4-12b-it-qat-q4_0.gguf" \
  --port 8080 --n-gpu-layers 99
```

For deliberate CPU diagnosis only, add `--allow-cpu --n-gpu-layers 0`.

The pooled scheduler requires `--n-batch >= --max-sequences`; otherwise startup
exits with an argument error explaining which value to increase or reduce.
`--ctx-size`, `--n-batch`, and `--max-sequences` must be positive, their pooled
context product must fit the llama API, and `--port` must be in `1..65535`.
Malformed, missing, and unknown arguments are rejected before model loading.
Long text and multimodal injections are automatically split to the effective
batch size; users do not need to divide request payloads.

Endpoints are documented in the parent repo's `docs/ipc-protocol.md`. Tests:

```bash
pytest engine/multimodal/tests   # see tests/conftest.py for how to start the server
```
