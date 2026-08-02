# Engine User Stories

Engine-side acceptance criteria with **performance and latency** budgets. These
support the system-wide stories in the parent repo (`user_stories.md`) but are
scoped to what the C++ engine is responsible for.

Every story MUST have an automated test in [`tests/`](tests/) that checks both
behavior and the latency/perf budget. Tests are written **first** (TDD).

Each story below names its real test coverage. Planned stories describe the
measurement to add without inventing a placeholder filename.

---

### EUS-1: Inject audio chunk updates session KV cache (no generation)
Injecting a chunk of microphone audio into an active session updates that
session's KV cache without producing tokens.

Input / trigger:
- Active session; POST an audio chunk.

Expected:
- KV cache advances by the chunk's tokens; no tokens generated.

Latency / performance budget:
- Warm end-of-speech → first token: **< 1.0 s** on the current RTX 5060 Ti
  baseline and faster than equivalent one-shot cold injection. This is the
  measured regression gate; the system US-1 target remains < 300 ms.

Test:
- `tests/test_streaming_audio.py` (chunked-vs-one-shot cache behavior, validation,
  transcription parity, and the warm-vs-cold latency gate)
- `tests/test_multimodal_audio.py` (one-shot audio/media compatibility)

---

### EUS-2: Fork a session's KV cache into a new, independent session
Forking snapshots a session into a **new external session** backed by another
sequence in the pooled context. `llama_memory_seq_cp` aliases the immutable
prefix; source and fork own independent sequence state and divergent suffixes.

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
- Warm same-pool fork: p95 **< 5 ms** over at least 50 samples. Measured
  p95 0.38–0.57 ms on RTX 5060 Ti / Gemma 4 12B QAT q4_0.
- Incremental resident GPU memory: **< 10 MiB/fork, < 60 MiB for six** excluding
  the fixed pool. Measured six-fork GPU-free delta: 0 MiB. The one-time pool
  footprint was about 8.34 GiB and is reported separately. Logical ownership
  intentionally increases by the copied prefix per owner and is not a physical
  memory measurement.

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
- The token loop observes cancellation within **one decode step**. Returning a
  clean reusable session additionally re-decodes the preserved boundary token;
  the automated gate allows **2 observed decode steps + 250 ms transport**.
- Cache rewind and logits refresh are included in that measured gate.

Test:
- `tests/test_cancel.py` (streaming and non-streaming cancellation, disconnect
  cleanup, measured halt budget, exact greedy parity after rewind, and session
  reuse)
- `tests/test_fork.py` (fork independence keeps BASE untouched when a fork is
  discarded)

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
- All N decode in one batched pass; after one bounded cohort-settling window,
  every cadence waits for each active job to step or finish/cancel. Each session
  sees correct, independent output. Unconstrained greedy transitions are
  canonical by exact lineage/EOS policy/generation step, so batched, cancelled,
  and standalone replays remain byte-identical across GPU batch shapes.
- Forking within the pooled context uses `llama_memory_seq_cp`: ~0 ms + ~0
  incremental resident GPU memory beyond the fixed pool (vs EUS-2's A':
  ~0.5–0.7 s + ~350 MiB).
- Shared-prefix sessions do NOT duplicate BASE KV — only the divergent suffix KV
  per fork.

Latency / performance budget (implemented and measured):
- Fork (`seq_cp`) in a pooled context: **< 5 ms, < 10 MiB** (~0 in practice; vs
  EUS-2 A').
- Batched decode of N=6 sessions: per-token wall-clock latency **≤ ~1.5× the
  single-session baseline**. Measured warmed runs: N=1 35–37 ms/step; N=6
  38–41 ms/step; ratio 1.07–1.09; aggregate 147–157 tok/s. Telemetry proves at
  least 32 decode calls carried all six sequence IDs.

Test:
- `tests/test_cross_session_batching.py` (single pool, fork latency/memory,
  six-way decode proof and parity, sampler/grammar isolation, capacity reuse,
  same-session 409, cancellation race, divergent-history row isolation,
  transactional and per-job-isolated failed initialization, current-boundary
  replay, repeated cancellation/output isolation, atomic delete/snapshot races, truthful
  ownership/GPU telemetry, and shared-prefix lifecycle)
- `tests/test_context_limits.py` (small-context generation and multimodal
  preflight/retry invariants)

> [ADR 0004](../../../docs/decisions/0004-fork-copy-semantics.md) mandates
> migrating fork to B (`seq_cp`) here; this unblocks the engagement pipeline's
> latency budgets (D5 / [architecture.md](../../../docs/architecture.md)).

---

### EUS-5: Sampling + grammar-constrained generation via `common/`
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

Latency / performance budget (retained acceptance targets; functional path implemented):
- Grammar-sampler overhead per decode step: **≤ ~10% of decode time** (the
  constraint must not dominate; measure on Gemma 4 12B).
- Final tool-call parse: **< 5 ms**.
- TTFT with grammar/tools must NOT regress US-5's target (< 300 ms; ~900 ms today).

Test:
- `tests/test_common_sampler.py` (JSON object/schema constraints, raw GBNF,
  grammar XOR validation, required/none tool choices, streaming parsed tool-call
  deltas, stop/ignore-EOS behavior, and greedy fork parity)
- `tests/test_util.cpp` (model-config defaults and parsing)

The functional suite is implemented. The numeric grammar-overhead/final-parse
budgets above still require a dedicated benchmark before they can become
measured regression gates; their thresholds are unchanged here.

> Ties to [ADR 0005](../../../docs/decisions/0005-engine-reuses-common-model-agnostic.md),
> [0006](../../../docs/decisions/0006-tools-prompt-based-engine-tool-agnostic.md),
> [0007](../../../docs/decisions/0007-grammar-constrained-generation.md),
> [0008](../../../docs/decisions/0008-generate-parameter-resolution-3-tier-profiles.md).

---

### EUS-6: KV-cache offload — swap a session's state between VRAM and host RAM
Move a session's swappable state out of the pool's live logical ownership into
host RAM (release its KV ownership and slot without shrinking the preallocated
pool) and bring it back unchanged. This is US-9's capacity-pressure
valve: a preempted task can be parked (RAM) and later resumed (load) or discarded
(delete), so a pruning task can reclaim VRAM. Disk serialization is separate
(backlog) and out of scope.

Input / trigger:
- `POST /sessions/{id}/offload`, `POST /sessions/{id}/load`,
  `GET /sessions/{id}`, `GET /sessions/usage`.

Expected:
- **Offload** moves a resident sequence to RAM; afterwards its logical KV cells
  and pool slot are released and `GET /sessions/{id}` reports `location:"ram"`.
  The pool's process-wide preallocated GPU buffers do not shrink. Returns
  `{session_id, location:"ram", cache_size, state_bytes, offload_ms}`. Idempotent
  (`offload_ms:0` on an already-RAM session).
- **Load** atomically reserves a RAM session before slot allocation and restore;
  concurrent load/delete/offload returns 409, preventing duplicate sequence-slot
  leaks and deleted-ID resurrection. The restored session reproduces its exact
  pre-offload greedy (`temperature`≤0) output on `/generate` (round-trip
  correctness — the KV/SSM state survived unchanged). Returns
  `{session_id, location:"vram", cache_size, state_bytes, load_ms}`. Idempotent.
- **GET /sessions/{id}** returns `{session_id, location, cache_size, state_bytes}`.
- **GET /sessions/usage** returns the existing `vram`/`ram` buckets plus a
  `pool` object separating capacity, active/free slots, total/per-sequence
  context, fixed `preallocated_bytes`, measured `model_gpu_bytes`, current
  `gpu_free_bytes`, exact `logical_owned_cells`, estimated
  `logical_allocated_bytes`, and successful startup-probe status.
- **Offloaded sessions cannot be silently used (409, no auto-load):**
  `inject`/`generate`/`fork` against a RAM session return **409**
  ("session is offloaded; POST /sessions/{id}/load first"); `delete` works on a
  RAM session (frees the RAM buffer); `cancel` is a no-op `{cancelled:false}`.
  Offloading a session with an active generation returns 409 (use-after-free
  guard). Unknown session → 404 for all new ops.
- `state_bytes` is target-sequence serializable state (`llama_state_seq_get_size`)
  — KV + SSM/Mamba recurrent state for hybrid models; symmetric across
  offload/load so `/usage` is meaningful in both states.
- **Shared-prefix safe:** offloading/loading one sequence must not corrupt or
  evict cells a live sibling owns. Source-first and fork-first release plus
  offload-during-sibling-generation are tested against real shared KV.

Latency / performance budget:
- **The bar is inference impact, not offload/load speed.** Serialization/removal
  runs on the single owner; restore waits until active generation completes.
  Repeated stress must retain ≥0.5× baseline throughput, bounded token gaps, and
  byte-identical output. Measured 24.8–25.7 tok/s versus 28.3–30.3 tok/s baseline.
- Offload/load wall-clock latency is **reported** (in `offload_ms`/`load_ms`) but
  is **not** a gate.

Test:
- tests/test_offload.py (functional + round-trip + KV-sharing correctness)
- tests/test_offload_concurrent.py (atomic load/delete races, SC #8
  inference-impact gate, and shared-KV sentinel)

> Enables [US-9](../../../user_stories.md) (preemption & redundancy pruning).
> Design survives [EUS-4](../../../docs/decisions/0004-fork-copy-semantics.md)
> slice-5 shared KV (`llama_memory_seq_cp`) — the sequence-scoped serialize
> primitive (`llama_state_seq_*`) won't corrupt another session's shared cells.
