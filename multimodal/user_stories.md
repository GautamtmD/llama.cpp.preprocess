# Engine User Stories

Engine-side acceptance criteria with **performance and latency** budgets. These
support the system-wide stories in the parent repo (`user_stories.md`) but are
scoped to what the C++ engine is responsible for.

Every story MUST have an automated test in [`tests/`](tests/) that checks both
behavior and the latency/perf budget. Tests are written **first** (TDD).

Template:

```
### EUS-<n>: <title>
<What the engine must do, scoped to engine responsibility.>

Input / trigger:
- ...

Expected:
- <observable engine behavior>

Latency / performance budget:
- <metric>: <target>  (e.g. KV-cache inject of 1s audio < 50ms; fork copy < Xms)

Test:
- tests/test_eus_<n>.cpp
```

---

### EUS-1: Inject audio chunk updates session KV cache (no generation)
Injecting a chunk of microphone audio into an active session updates that
session's KV cache without producing tokens.

Input / trigger:
- Active session; POST an audio chunk.

Expected:
- KV cache advances by the chunk's tokens; no tokens generated.

Latency / performance budget:
- (TBD — set real targets once the path exists)

Test:
- tests/test_eus_1.cpp (not yet implemented)

---

### EUS-2: Fork a session's KV cache into a new, independent session
Forking a session snapshots whatever its KV cache currently holds (after a text
inject, an audio inject, or a generate) into a **new session** that owns an
independent copy. The source session is untouched; the forked session generates
independently from the snapshot.

Input / trigger:
- Active session with some cached content; `POST /sessions/{id}/fork`.

Expected:
- A new `{session_id}` is returned whose KV is a copy of the source's at fork
  time (`cache_size` matches the source).
- Forkable after text-inject, after audio-inject, and after generate.
- The source session's cache is unchanged by the fork, and later mutations to
  the source do NOT appear in the fork (true snapshot independence).
- The forked session generates coherent text from the snapshot.
- Unknown source session -> 404.

Latency / performance budget:
- Fork copy (includes creating the new context) of a session with ≤ 2048 cached
  tokens: **≤ 600 ms** (measured median ~200–310 ms on RTX 5060 Ti, Gemma 4 12B;
  the cost is context-creation-dominated — see
  [`docs/decisions/0004-fork-copy-semantics.md`](../../../docs/decisions/0004-fork-copy-semantics.md)).
- VRAM: **≤ ~400 MiB per fork** (the forked session's KV state; measured ~350
  MiB, plateaus with the model's SWA cache). Slice 5 will drop this to ~0 by
  switching to `llama_memory_seq_cp` in a pooled context.

Test:
- tests/test_fork.py (correctness + latency against the live server)
- src/fork_bench.cpp (the M2.0 copy-cost measurement gate; cross-context-size
  perf + VRAM footprint)
