# tests/

Engine tests for `multimodal/`. **TDD, perf-first**: every C++ change is preceded
by a test here that measures performance and latency (not just correctness).

- One test file per engine user story: `test_eus_<n>.cpp`.
- Perf/latency thresholds come from [`../user_stories.md`](../user_stories.md).
- Wired into CMake via a `CMakeLists.txt` here (added with the first test).

Guided by the parent repo's `docs/architecture.md` and `AGENTS.md`.
