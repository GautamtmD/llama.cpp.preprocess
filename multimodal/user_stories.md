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
- **Fully generation-ready** after text-inject and after generate: the fork
  reproduces the source's exact greedy (temp=0) output from the shared snapshot.
- **After audio/image inject:** the KV is copied. If the source's last inject
  ended in a media EMBEDDING chunk (audio/image — no discrete token), the fork
  is NOT immediately generation-ready; inject one text token first. The chat
  protocol always closes an audio turn with text markers, so the normal flow
  (audio -> text suffix) is fully forkable. (Media inject clears the tracked
  last token, so a later fork never re-decodes a stale token at a media cell.)
- The source session's cache is unchanged by the fork, and later mutations to
  the source do NOT appear in the fork (true snapshot independence).
- Unknown source session -> 404.

Latency / performance budget:
- Fork copy of a session with ≤ 2048 cached tokens: **≤ 1.0 s** steady-state
  on RTX 5060 Ti, Gemma 4 12B (measured ~480 ms for a tiny session, ~700 ms at
  ~200 tokens). Breakdown: new-context creation ~200 ms + the redecode ~290 ms
  (that redecode is the **first decode in the new dst context**, so it pays that
  context's one-time first-decode cost — graph build + buffer alloc; it is NOT
  CUDA-graph capture, which `GGML_CUDA_DISABLE_GRAPHS` does not reduce and which
  destroys generation throughput) + `get/set_data` growing with N. A startup
  fork warm-up removes the one-time global-JIT cold-start (first fork ~480 ms,
  not ~870 ms). All of this is eliminated by slice 5 (`llama_memory_seq_cp` in a
  pooled context: no new context, no redecode → ~0). See
  [`docs/decisions/0004-fork-copy-semantics.md`](../../../docs/decisions/0004-fork-copy-semantics.md).
- VRAM: **≤ ~400 MiB per fork** (the forked session's KV state; measured ~350
  MiB, plateaus with the model's SWA cache). Slice 5 drops this to ~0.

Test:
- tests/test_fork.py (correctness + latency against the live server)
- src/fork_bench.cpp (the M2.0 copy-cost measurement gate; cross-context-size
  perf + VRAM footprint)

---

### EUS-3: Abort an in-flight generation and leave the session cache clean (US-4 barge-in)
Cancelling a running `/generate` stops it promptly and ensures the partial tokens
it produced are NOT committed — the session's cache is left at the pre-generation
boundary so the next turn starts clean. (The *decision* to cancel is perception/
GUI — US-4; the engine provides the capability.)

Input / trigger:
- A session mid-`/generate` (streaming or not); a cancel signal — an explicit
  cancel endpoint, or a client disconnect on a streaming generate.

Expected:
- Generation halts within one decode step of the cancel signal (it does NOT run
  to `max_tokens`).
- Partial tokens generated after the cancel point are NOT persisted: the
  session's KV cache is at the pre-generation token boundary (verified by cache
  size and by the next generate reproducing the pre-cancel greedy output).
- The session is fully reusable afterwards (inject / generate / fork work normally).
- In the engagement pipeline (D5) generations run on **forks**, so abort = drop
  the fork (no effect on BASE). A BASE-resident generate abort **rewinds** the
  partial tokens in place (`llama_memory_seq_rm`).

Latency / performance budget:
- Cancel-to-halted: within **one decode step** (~one token-time) of the signal.
- Cache left clean at **no extra cost** when the generation ran on a fork
  (drop-fork). Rewind-in-place cost TBD — measure when implemented.

Test:
- tests/test_eus_3.* (cancel mid-generate → assert halted early, cache at the
  pre-generate boundary, session reusable; fork-abort leaves BASE untouched).

> Abort mechanism (drop-fork vs rewind-in-place) is the EC-D open question
> ([architecture.md](../../../docs/architecture.md)), informed by
> [ADR 0004](../../../docs/decisions/0004-fork-copy-semantics.md)'s fork cost
> (drop-fork costly under A', ~free under slice-5 B).

---

### EUS-4: Cross-session batching — decode N sessions in one inference loop (slice 5)
Multiple sessions run as **sequences in a single pooled context** and decode
together in one inference loop, instead of N sequential passes. This is what
makes forks cheap (`llama_memory_seq_cp`, ~0) and lets the engagement pipeline
batch Gate + eager + thinker generations (D5) inside the user-story latency
budgets.

Input / trigger:
- N active sessions/forks with pending generations.

Expected:
- All N decode in one batched pass; each session sees correct, independent output
  (sequence isolation — a batched run reproduces each session's standalone
  greedy output).
- Forking within the pooled context uses `llama_memory_seq_cp`: ~0 ms + ~0 VRAM
  (vs EUS-2's A': ~0.5–0.7 s + ~350 MiB).
- Shared-prefix sessions do NOT duplicate BASE KV — only the divergent suffix KV
  per fork.

Latency / performance budget (initial targets — verify when implemented):
- Fork (`seq_cp`) in a pooled context: **< 5 ms, < 10 MiB** (~0 in practice; vs
  EUS-2 A').
- Batched decode of N=6 sessions: per-token wall-clock latency **≤ ~1.5× the
  single-session baseline** (target — verify; the engagement pipeline needs ~6
  forks/turn inside US-1's 500 ms / US-2's 3 s budgets).

Test:
- tests/test_eus_4.* (N sessions batched → outputs match independent runs;
  fork=`seq_cp` ≈0 cost vs EUS-2; batched-vs-sequential throughput).

> [ADR 0004](../../../docs/decisions/0004-fork-copy-semantics.md) mandates
> migrating fork to B (`seq_cp`) here; this unblocks the engagement pipeline's
> latency budgets (D5 / [architecture.md](../../../docs/architecture.md)).

---

### EUS-5: Sampling + grammar-constrained generation (migrate onto `common/`)
`/generate` uses `common/`'s sampler + grammar + tool-calling end-to-end
(ADR 0005/0006/0007/0008): the full sampling param set, `response_format`/
`grammar` stop-on-complete, and engine-side tool-call parsing — replacing the
hand-rolled `make_sampler`.

Input / trigger:
- `/generate` with sampling params (`temperature`, `top_k`, `min_p`, …),
  `response_format` XOR `grammar`, and/or `tools` + `tool_choice`.

Expected:
- Full `common_params_sampling` applied (no hand-rolled sampler).
- `response_format`/`grammar` constrain output; generation stops when the active
  grammar reaches its accepting state (stop-on-complete). Both present → 400.
- Lazy grammar (`tools`+`AUTO`): reason-then-call (free-text preamble, then
  constrained tool-call JSON, stop on complete) — the thinker flow (D5).
- Tool calls parsed engine-side via `common_chat_parse` (streaming via
  `is_partial`); `/generate` streams tokens OR returns parsed tool calls.
- greedy-at-`temperature`≤0 reproducibility preserved (EUS-2 fork/source parity).

Latency / performance budget (initial targets — verify when implemented):
- Grammar-sampler overhead per decode step: **≤ ~10% of decode time** (the
  constraint must not dominate; measure on Gemma 4 12B).
- Final tool-call parse: **< 5 ms**.
- TTFT with grammar/tools must NOT regress US-5's target (< 300 ms; ~900 ms today).

Test:
- tests/test_eus_5.* (full sampling; `response_format`→valid JSON; grammar XOR→400;
  lazy reason-then-call; tool-call parse; greedy parity vs EUS-2).

> Ties to [ADR 0005](../../../docs/decisions/0005-engine-reuses-common-model-agnostic.md),
> [0006](../../../docs/decisions/0006-tools-prompt-based-engine-tool-agnostic.md),
> [0007](../../../docs/decisions/0007-grammar-constrained-generation.md),
> [0008](../../../docs/decisions/0008-generate-parameter-resolution-3-tier-profiles.md).
