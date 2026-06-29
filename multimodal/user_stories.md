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
