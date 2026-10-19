# tests/

Engine tests for `multimodal/`. **TDD, perf-first**: every C++ change is preceded
by a test here that measures performance and latency (not just correctness).

- One test file per engine user story: `test_eus_<n>.cpp`.
- Perf/latency thresholds come from [`../user_stories.md`](../user_stories.md)
  and are GPU thresholds: the suite is run on the GPU build. Do not add CPU
  test suites, CPU perf thresholds or CPU-only variants of tests; a CPU run
  (`MULTIMODAL_CUDA_VISIBLE_DEVICES=-1`, `MULTIMODAL_SERVER_ARGS=--allow-cpu`)
  is only a dry check while debugging one test.
- Wired into CMake via a `CMakeLists.txt` here (added with the first test).

Guided by the parent repo's `docs/architecture.md` and `AGENTS.md`.
