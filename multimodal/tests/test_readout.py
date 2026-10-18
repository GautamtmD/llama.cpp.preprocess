"""EUS-7: decision readout + priority on POST /sessions/{id}/inject (ADR 0010).

A quick decision is the app recipe fork -> inject-with-readout -> delete. The
engine adds only the raw last-row logit readout and a priority flag. These are
the ADR's CPU-runnable "correctness tests first":

- readout equals a reference decode of the same tokens on a fresh sequence;
- under concurrent generation on another sequence the fork reads its own row;
- the source session is unchanged by fork + readout + delete;
- ``top[0]`` equals greedy ``/generate``'s first token on an identical fork;
- statistics are well formed (finite logprobs, ``candidate_mass`` in [0, 1],
  sum exp(logprob) over ``top`` <= 1);
- every 400 path leaves the session unchanged (validation precedes mutation);
- a priority inject queued behind N normal injects completes first;
- no pooled slot leaks after fork -> readout -> delete, including after a 400;
- readout overhead: ``readout_ms`` (measured after the decode is synchronized,
  so it is the readout alone) <= 5 ms on CPU; on a GPU build the ADR 0010 GO-B
  gate ``readout_ms`` <= 1 ms applies to the Gate recipe's readout shape.

Numerics: on CPU a fork's row matches a reference decode on another sequence
within 0.05 logit. On CUDA the row is bit-identical to a reference decoded
under the same cache state (same resident sequences), but moves by up to 0.59
logit / 0.60 logprob when other sequences' KV is resident or a generation runs
concurrently (unified-KV layout changes the kernels' reduction order; measured
over 18 layouts, worklog 2026-10-01-decide-gate Step 15). So the reference is
decoded before ``base_session`` exists (same state as the forks -> strict
tolerance), and only the concurrent-generation test, whose cache state cannot
be reproduced, uses ``GPU_LAYOUT_TOLERANCE`` plus strict top-1 and candidate
ranking assertions.

Run (CPU-only binary, self-booted)::

    MULTIMODAL_MODEL="C:/ML Models/Gemma4 12b/gemma-4-12b-it-qat-q4_0.gguf" \\
    MULTIMODAL_MMPROJ="C:/ML Models/Gemma4 12b/mmproj-gemma-4-12b-it-qat-q4_0.gguf" \\
    MULTIMODAL_SERVER_ARGS="--allow-cpu" \\
    pytest engine/multimodal/tests/test_readout.py -q -s

Prompts use Gemma 4 turn markers (``<|turn>``/``<turn|>``); on another model the
greedy-agreement and prompt-specific checks still hold because they compare the
engine against itself, but the base text is less natural.
"""

from __future__ import annotations

import base64
import io
import json
import math
import statistics
import threading
import time

import pytest
import requests

pytestmark = pytest.mark.usefixtures("base")

BASE_TEXT = "<|turn>user\nReply with one letter only. Is fire hot? A) yes B) no<turn|>\n"
SUFFIX = "<|turn>model\n"
CANDIDATES = ["A", "B"]
# Long-answer prompt: keeps a concurrent generation running, and is the
# normal-priority load in the priority-ordering test.
NORMAL_TEXT = (
    "<|turn>user\nList, in order, the planets of the solar system and one short fact "
    "about each of them, starting from Mercury.<turn|>\n"
)
LOGIT_TOLERANCE = 0.05  # same tokens, same cache state (CPU: also any state)
# CUDA, cache state not reproducible (concurrent generation): measured spread
# max 0.591 logit / 0.599 logprob over 18 layouts; 1.0 = 1.7x that, still far
# below this prompt's A-B logit gap (4.1) and top0-top1 gap (13.7).
GPU_LAYOUT_TOLERANCE = 1.0
READOUT_MS_CPU_GATE = 5.0
READOUT_MS_GPU_GATE = 1.0  # ADR 0010 GO-B: readout overhead <= 1 ms
GATE_CANDIDATES = ["A", " A", "B", " B"]  # the Gate recipe's readout shape
GATE_TOP_K = 8


def _post(base, path, body=None, timeout=300):
    return requests.post(f"{base}{path}", json=body, timeout=timeout)


def _ok(response):
    assert response.status_code == 200, response.text
    return response.json()


def _create(base):
    return _ok(_post(base, "/sessions"))["session_id"]


def _fork(base, sid):
    return _ok(_post(base, f"/sessions/{sid}/fork"))["session_id"]


def _delete(base, sid):
    requests.delete(f"{base}/sessions/{sid}", timeout=30)


def _status(base, sid):
    return _ok(requests.get(f"{base}/sessions/{sid}", timeout=30))


def _inject(base, sid, body):
    return _ok(_post(base, f"/sessions/{sid}/inject", body))


def _active(base):
    usage = _ok(requests.get(f"{base}/sessions/usage", timeout=30))
    return usage["pool"]["active_sequences"]


def _readout(base, sid, text=SUFFIX, candidates=CANDIDATES, top_k=5, **extra):
    body = {"text": text, "readout": {"candidates": candidates, "top_k": top_k}}
    body.update(extra)
    return _inject(base, sid, body)


def _by_token(readout):
    return {c["token"]: c for c in readout["candidates"]}


def _rank(row):
    return sorted(range(len(row["candidates"])), key=lambda i: -row["candidates"][i]["logit"])


def _assert_same_row(got, ref, tolerance=LOGIT_TOLERANCE):
    assert got["position"] == ref["position"]
    assert [c["token"] for c in got["candidates"]] == [c["token"] for c in ref["candidates"]]
    for g, r in zip(got["candidates"], ref["candidates"], strict=True):
        assert abs(g["logit"] - r["logit"]) <= tolerance, (g, r)
        assert abs(g["logprob"] - r["logprob"]) <= tolerance, (g, r)
    # Strict regardless of tolerance: same greedy token, same candidate ranking.
    assert got["top"][0]["token"] == ref["top"][0]["token"]
    assert _rank(got) == _rank(ref)


def _pool(base):
    return _ok(requests.get(f"{base}/sessions/usage", timeout=30))["pool"]


def _on_gpu(base):
    return _pool(base)["model_gpu_bytes"] > 0


@pytest.fixture(scope="module")
def base_session(base, reference):
    # Depends on `reference` so the reference is decoded first, under the same
    # cache state the forks of this session see (see the module docstring).
    sid = _create(base)
    _inject(base, sid, {"text": BASE_TEXT})
    yield sid
    _delete(base, sid)


@pytest.fixture(scope="module")
def reference(base):
    """Reference decode: a fresh sequence fed the same two injects (same tokens,
    same chunk structure) with no fork involved, decoded before `base_session`
    exists so the cache state matches the forks'."""
    sid = _create(base)
    try:
        _inject(base, sid, {"text": BASE_TEXT})
        return _readout(base, sid)["readout"]
    finally:
        _delete(base, sid)


def _fork_readout(base, base_session, **kwargs):
    fork = _fork(base, base_session)
    try:
        return _readout(base, fork, **kwargs)
    finally:
        _delete(base, fork)


def test_info_advertises_readout_and_priority(base):
    info = _ok(requests.get(f"{base}/info", timeout=30))
    assert info["readout"] == {
        "supported": True,
        "max_candidates": 64,
        "max_top_k": 16,
        "semantics": "raw_last_row_pre_sampler_v1",
    }
    assert info["priority_inject"] is True


def test_fork_readout_matches_reference_decode(base, base_session, reference):
    body = _fork_readout(base, base_session)
    readout = body["readout"]
    assert readout["position"] == body["cache_size"] - 1
    assert body["tokens_injected"] >= 1
    _assert_same_row(readout, reference)


def test_readout_statistics_are_well_formed(base, base_session):
    readout = _fork_readout(base, base_session, top_k=16)["readout"]
    assert readout["semantics"] == "raw_last_row_pre_sampler_v1"
    assert readout["readout_ms"] >= 0
    assert [c["text"] for c in readout["candidates"]] == CANDIDATES
    for entry in readout["candidates"] + readout["top"]:
        assert math.isfinite(entry["logit"]) and math.isfinite(entry["logprob"])
        assert entry["logprob"] <= 1e-9
        assert isinstance(entry["token"], int) and isinstance(entry["text"], str)
    mass = sum(math.exp(c["logprob"]) for c in readout["candidates"])
    assert 0.0 <= readout["candidate_mass"] <= 1.0
    assert abs(readout["candidate_mass"] - mass) < 1e-6
    top = readout["top"]
    assert len(top) == 16
    assert all(a["logit"] >= b["logit"] for a, b in zip(top, top[1:], strict=False))
    assert sum(math.exp(t["logprob"]) for t in top) <= 1.0 + 1e-6
    # The same token reports the same numbers in both lists.
    candidates = _by_token(readout)
    for t in top:
        if t["token"] in candidates:
            assert t["logprob"] == pytest.approx(candidates[t["token"]]["logprob"])
    # Sanity of the prompt itself (not an engine property): the answer letters
    # carry most of the mass at this boundary, recorded for the worklog.
    print(f"\n[readout] candidate_mass={readout['candidate_mass']:.4f} top0={top[0]}")


def test_top0_matches_greedy_generate_first_token(base, base_session):
    readout = _fork_readout(base, base_session, top_k=2)["readout"]
    fork = _fork(base, base_session)
    try:
        _inject(base, fork, {"text": SUFFIX})
        generated = _ok(
            _post(
                base,
                f"/sessions/{fork}/generate",
                {"max_tokens": 1, "temperature": 0, "stream": False},
            )
        )
    finally:
        _delete(base, fork)
    gap = readout["top"][0]["logit"] - readout["top"][1]["logit"]
    print(f"\n[readout] greedy={generated['tokens']} top={readout['top']} gap={gap:.3f}")
    assert generated["tokens"][0] == readout["top"][0]["token"]


def test_source_unchanged_after_fork_readout_delete(base, base_session, reference):
    before = _status(base, base_session)
    first = _fork_readout(base, base_session)["readout"]
    after = _status(base, base_session)
    assert after["cache_size"] == before["cache_size"]
    assert after["location"] == before["location"]
    # Source history unchanged: a second readout from the same source is the
    # same row (deterministic CPU decode of identical tokens).
    second = _fork_readout(base, base_session)["readout"]
    _assert_same_row(second, first, tolerance=1e-4)
    _assert_same_row(second, reference)


def test_readout_on_messages_body(base, base_session):
    fork = _fork(base, base_session)
    try:
        body = _inject(
            base,
            fork,
            {
                "messages": [{"role": "user", "content": "Reply A or B."}],
                "readout": {"candidates": CANDIDATES},
                "priority": True,
            },
        )
    finally:
        _delete(base, fork)
    assert body["chat_template_applied"] is True
    readout = body["readout"]
    assert readout["position"] == body["cache_size"] - 1
    assert readout["top"] == []
    assert 0.0 <= readout["candidate_mass"] <= 1.0


@pytest.mark.requires("image")
def test_readout_on_text_ending_media_messages(base):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (255, 0, 0)).save(buf, format="PNG")
    image = base64.b64encode(buf.getvalue()).decode()
    sid = _create(base)
    try:
        body = _inject(
            base,
            sid,
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "data": image},
                            {"type": "text", "text": "Is this red? A) yes B) no"},
                        ],
                    }
                ],
                "readout": {"candidates": CANDIDATES, "top_k": 1},
            },
        )
    finally:
        _delete(base, sid)
    assert body["used_multimodal"] is True
    readout = body["readout"]
    assert readout["position"] == body["cache_size"] - 1
    assert len(readout["top"]) == 1


_SIXTY_FIVE = (
    [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + list("0123456789!?.")
)

BAD_BODIES = {
    "readout-not-object": {"text": SUFFIX, "readout": ["A"]},
    "readout-null": {"text": SUFFIX, "readout": None},
    "unknown-key": {"text": SUFFIX, "readout": {"candidates": ["A"], "selected": 1}},
    "missing-candidates": {"text": SUFFIX, "readout": {"top_k": 1}},
    "zero-candidates": {"text": SUFFIX, "readout": {"candidates": []}},
    "too-many-candidates": {"text": SUFFIX, "readout": {"candidates": _SIXTY_FIVE}},
    "empty-candidate": {"text": SUFFIX, "readout": {"candidates": [""]}},
    "non-string-candidate": {"text": SUFFIX, "readout": {"candidates": [65]}},
    "multi-token-candidate": {"text": SUFFIX, "readout": {"candidates": ["speak_now"]}},
    "duplicate-candidate": {"text": SUFFIX, "readout": {"candidates": ["A", "A"]}},
    "top-k-negative": {"text": SUFFIX, "readout": {"candidates": ["A"], "top_k": -1}},
    "top-k-too-large": {"text": SUFFIX, "readout": {"candidates": ["A"], "top_k": 17}},
    "top-k-float": {"text": SUFFIX, "readout": {"candidates": ["A"], "top_k": 1.5}},
    "priority-not-bool": {"text": SUFFIX, "priority": "yes"},
    "audio-body": {
        "audio": base64.b64encode(b"\0" * 640 * 4).decode(),
        "readout": {"candidates": ["A"]},
    },
}


@pytest.mark.parametrize("name", sorted(BAD_BODIES))
def test_invalid_readout_is_400_and_leaves_session_unchanged(base, base_session, name):
    fork = _fork(base, base_session)
    try:
        before = _status(base, fork)["cache_size"]
        response = _post(base, f"/sessions/{fork}/inject", BAD_BODIES[name])
        assert response.status_code == 400, response.text
        body = response.json()
        assert body["code"] == 400 and "readout" not in body
        assert _status(base, fork)["cache_size"] == before
    finally:
        _delete(base, fork)


def test_sixty_four_candidates_and_max_top_k_are_accepted(base, base_session):
    candidates = _SIXTY_FIVE[:64]
    readout = _fork_readout(base, base_session, candidates=candidates, top_k=16)["readout"]
    assert len(readout["candidates"]) == 64
    assert len({c["token"] for c in readout["candidates"]}) == 64
    assert len(readout["top"]) == 16


def test_context_full_with_readout_returns_409_without_readout(base, base_session):
    fork = _fork(base, base_session)
    try:
        before = _status(base, fork)["cache_size"]
        response = _post(
            base,
            f"/sessions/{fork}/inject",
            {"text": " word" * 5000, "readout": {"candidates": ["A"]}},
        )
        assert response.status_code == 409, response.text
        assert "readout" not in response.json()
        assert _status(base, fork)["cache_size"] == before
    finally:
        _delete(base, fork)


def test_no_slot_leak_after_recipe_including_400(base, base_session):
    before = _active(base)
    for _ in range(2):
        _fork_readout(base, base_session, top_k=0)
    fork = _fork(base, base_session)
    try:
        bad = _post(
            base,
            f"/sessions/{fork}/inject",
            {"text": SUFFIX, "readout": {"candidates": ["speak_now"]}},
        )
        assert bad.status_code == 400
    finally:
        _delete(base, fork)
    assert _active(base) == before


def _stream_generate(base, sid, events, first_token, done):
    with requests.post(
        f"{base}/sessions/{sid}/generate",
        json={"max_tokens": 48, "temperature": 0, "stream": True},
        stream=True,
        timeout=600,
    ) as response:
        assert response.status_code == 200, response.text
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            event = json.loads(line[5:].strip())
            events.append((time.perf_counter(), event))
            if event.get("type") == "token":
                first_token.set()
            if event.get("type") in ("done", "error"):
                break
    done.set()


def test_readout_during_concurrent_generation_reads_own_row(base, base_session, reference):
    # A long-answer prompt on an unrelated sequence keeps generation running.
    generator = _create(base)
    events: list = []
    first_token, done = threading.Event(), threading.Event()
    try:
        _inject(base, generator, {"text": NORMAL_TEXT + SUFFIX})
        thread = threading.Thread(
            target=_stream_generate, args=(base, generator, events, first_token, done)
        )
        thread.start()
        assert first_token.wait(300), "generation never produced a token"
        readout = _fork_readout(base, base_session, priority=True)["readout"]
        finished_at = time.perf_counter()
        generation_still_running = not done.is_set()
        thread.join(600)
    finally:
        _delete(base, generator)
    assert events and events[-1][1]["type"] == "done", events[-1:]
    tokens_after = sum(1 for t, e in events if e.get("type") == "token" and t > finished_at)
    print(
        f"\n[readout] concurrent: generation_running={generation_still_running} "
        f"tokens_after_readout={tokens_after}"
    )
    assert generation_still_running and tokens_after > 0
    # The concurrent cache state (growing generator KV, shared decode batches)
    # cannot be reproduced for a reference; GPU numerics move with it.
    tolerance = GPU_LAYOUT_TOLERANCE if _on_gpu(base) else LOGIT_TOLERANCE
    _assert_same_row(readout, reference, tolerance=tolerance)


PRIORITY_MAX_N = 3
PRIORITY_MIN_NORMAL_S = 0.4  # each queued normal inject must outlast the send delay


def _timed_inject_s(base, text):
    """Wall time of one inject of `text`, synchronized (a readout makes the
    decode finish before the response, also on an async GPU backend)."""
    sid = _create(base)
    try:
        t0 = time.perf_counter()
        _inject(base, sid, {"text": text, "readout": {"candidates": CANDIDATES}})
        return time.perf_counter() - t0
    finally:
        _delete(base, sid)


def _priority_load(base):
    """Normal-inject text sized so one inject takes >= PRIORITY_MIN_NORMAL_S on
    this device (NORMAL_TEXT once on CPU, repeated on GPU)."""
    unit_tokens = 40  # NORMAL_TEXT is ~35 tokens
    cap = max(1, (_pool(base)["ctx_size_per_sequence"] - 256) // unit_tokens)
    k = 1
    elapsed = _timed_inject_s(base, NORMAL_TEXT)
    while elapsed < PRIORITY_MIN_NORMAL_S and k < cap:
        k = min(cap, max(k * 2, math.ceil(k * PRIORITY_MIN_NORMAL_S / max(elapsed, 1e-3))))
        elapsed = _timed_inject_s(base, NORMAL_TEXT * k)
    return NORMAL_TEXT * k, k, elapsed


def _priority_round(base, priority, n, text, delay_s):
    """n normal injects on n sessions start together; once they are queued and
    the first decode is running, one short inject is sent with `priority`."""
    sids = [_create(base) for _ in range(n + 1)]
    finished: dict[str, float] = {}
    errors: list[str] = []

    def run(sid, body):
        try:
            response = _post(base, f"/sessions/{sid}/inject", body)
            if response.status_code != 200:
                errors.append(response.text)
        finally:
            finished[sid] = time.perf_counter()

    try:
        normal = [threading.Thread(target=run, args=(sid, {"text": text})) for sid in sids[:n]]
        start = time.perf_counter()
        for t in normal:
            t.start()
        time.sleep(delay_s)  # all normal injects queued; the first one executing
        sent = time.perf_counter()
        urgent = threading.Thread(
            target=run,
            args=(
                sids[-1],
                {
                    "text": BASE_TEXT + SUFFIX,
                    "readout": {"candidates": CANDIDATES},
                    "priority": priority,
                },
            ),
        )
        urgent.start()
        for t in normal + [urgent]:
            t.join(900)
    finally:
        for sid in sids:
            _delete(base, sid)
    assert not errors, errors
    urgent_done = finished[sids[-1]]
    ahead = sum(1 for sid in sids[:n] if finished[sid] < urgent_done)
    normal_ms = sorted((finished[s] - start) * 1000 for s in sids[:n])
    return ahead, (urgent_done - sent) * 1000, normal_ms


def test_priority_inject_overtakes_queued_normal_injects(base):
    # Each round needs n normal sessions + 1 urgent; read capacity, do not assume it.
    n = min(PRIORITY_MAX_N, _pool(base)["free_sequences"] - 1)
    if n < 2:
        pytest.skip(f"needs >= 3 free sequences for an ordering check, have {n + 1}")
    text, repeats, normal_s = _priority_load(base)
    delay_s = min(0.5, normal_s / 3)
    ahead, urgent_ms, normal_ms = _priority_round(base, True, n, text, delay_s)
    c_ahead, c_ms, c_normal = _priority_round(base, False, n, text, delay_s)
    print(
        f"\n[priority] load: {n} normal injects of NORMAL_TEXT x{repeats} "
        f"({normal_s * 1000:.0f} ms each), urgent sent after {delay_s * 1000:.0f} ms"
        f"\n[priority] priority=true: finished after {ahead}/{n} normal, "
        f"{urgent_ms:.0f} ms from send; normal done at {[round(x) for x in normal_ms]} ms"
        f"\n[priority] control priority=false: finished after {c_ahead}/{n} "
        f"normal, {c_ms:.0f} ms from send; normal done at {[round(x) for x in c_normal]} ms"
    )
    # With priority it waits only for the decode already executing.
    assert ahead <= 1
    # The control proves the load really was queued when the short inject was
    # sent: without priority it waits behind more normal injects (it can tie
    # with the last one, whose commands interleave with its own) and finishes
    # at least half a normal inject later than the priority request.
    assert c_ahead > ahead
    assert c_ms - urgent_ms >= 500 * normal_s, (c_ms, urgent_ms, normal_s)


def test_readout_overhead_vs_plain_inject(base, base_session):
    plain_ms, readout_wall_ms, readout_ms = [], [], []
    for _ in range(3):
        fork = _fork(base, base_session)
        try:
            t0 = time.perf_counter()
            _inject(base, fork, {"text": SUFFIX})
            plain_ms.append((time.perf_counter() - t0) * 1000)
        finally:
            _delete(base, fork)
        fork = _fork(base, base_session)
        try:
            t0 = time.perf_counter()
            body = _readout(base, fork, candidates=_SIXTY_FIVE[:64], top_k=16)
            readout_wall_ms.append((time.perf_counter() - t0) * 1000)
            readout_ms.append(body["readout"]["readout_ms"])
        finally:
            _delete(base, fork)
    gate_ms = []
    for _ in range(9):
        body = _fork_readout(base, base_session, candidates=GATE_CANDIDATES, top_k=GATE_TOP_K)
        gate_ms.append(body["readout"]["readout_ms"])
    gpu = _on_gpu(base)
    delta = statistics.median(readout_wall_ms) - statistics.median(plain_ms)
    print(
        f"\n[readout] {'GPU' if gpu else 'CPU'} overhead: 64 candidates/top_k 16 "
        f"readout_ms median={statistics.median(readout_ms):.3f} max={max(readout_ms):.3f}; "
        f"Gate shape (4 candidates/top_k 8) median={statistics.median(gate_ms):.3f} "
        f"max={max(gate_ms):.3f}; client wall plain median="
        f"{statistics.median(plain_ms):.1f} ms, with readout median="
        f"{statistics.median(readout_wall_ms):.1f} ms (delta {delta:+.1f} ms, decode noise)"
    )
    assert max(readout_ms) <= READOUT_MS_CPU_GATE
    if gpu:
        # Typical-call check; the p95 gate over >= 100 warm calls is the GO-B
        # run (scripts/gate_latency.py), not this 9-sample correctness test.
        assert statistics.median(gate_ms) <= READOUT_MS_GPU_GATE
