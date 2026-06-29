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
  user_stories.md   engine-side acceptance criteria + latency budgets
```

## Build

Wired into the top-level CMake build via a `CMakeLists.txt` here (added when the
first real code lands). Built via `./scripts/build_engine.sh` from the parent
repo.
