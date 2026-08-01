"""Concurrent tests for KV-cache offload/load (EUS-6).

- SC #8: offload/load on OTHER sessions must not significantly degrade an ongoing
  generation, and the generating session's greedy output must be byte-identical
  to a solo run.
- SC #9 sentinel: generate on one of two identical-KV (forked) sessions while
  offloading the other — a placeholder that will catch breakage when KV sharing
  lands (slice-5 / EUS-4).

Model-agnostic; launch with the model as a parameter (see tests/conftest.py)::

    MULTIMODAL_MODEL="C:/ML Models/MiniCPM-V-4.6/MiniCPM-V-4_6-Q8_0.gguf" \\
    pytest engine/multimodal/tests/test_offload_concurrent.py
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator

import pytest
import requests

pytestmark = [pytest.mark.usefixtures("base", "make_session"), pytest.mark.requires("text")]

LIST_PROMPT = [
    {
        "role": "user",
        "content": "Write a numbered list of twelve facts about the ocean, one per line.",
    }
]
BIG_TEXT = "The quick brown fox jumps over the lazy dog. " * 30  # a real-sized cache

# --- Performance thresholds (SC #8). Tuned from the first measurement run on
# MiniCPM-V-4.6 Q8_0; recorded in docs/worklog/2026-07-11-kv-cache-offload.md.
# These are deliberately generous regression gates (not tight targets): the
# serialize is lock-free and brief, so we expect near-zero impact. The hard
# guarantee is the byte-identical correctness assertion below.
THROUGHPUT_FRACTION = 0.5  # concurrent sustained tok/s must be >= this * baseline
MAX_GAP_MULTIPLE = 4.0  # no inter-token gap > this * baseline max inter-token gap
MAX_GAP_FLOOR_MS = 2000.0  # absolute slack for the (brief) GPU->host serialize window


# ------------------------------- helpers ------------------------------------


def _inject_chat(base, sid, messages=LIST_PROMPT):
    r = requests.post(f"{base}/sessions/{sid}/inject", json={"messages": messages}, timeout=120)
    assert r.status_code == 200, r.text


def _inject_text(base, sid, text):
    r = requests.post(f"{base}/sessions/{sid}/inject", json={"text": text}, timeout=120)
    assert r.status_code == 200, r.text


def _generate(base, sid, **kw):
    body = {"max_tokens": kw.pop("max_tokens", 40), "temperature": 0.0}
    body.update(kw)
    r = requests.post(f"{base}/sessions/{sid}/generate", json=body, timeout=120)
    assert r.status_code == 200, r.text
    return r.json()


def _fork(base, sid):
    r = requests.post(f"{base}/sessions/{sid}/fork", timeout=120)
    assert r.status_code == 200, r.text
    return r.json()


def _offload(base, sid):
    return requests.post(f"{base}/sessions/{sid}/offload", timeout=120)


def _load(base, sid):
    return requests.post(f"{base}/sessions/{sid}/load", timeout=120)


def _status(base, sid):
    return requests.get(f"{base}/sessions/{sid}", timeout=60)


def _parse_sse(response) -> Iterator[dict]:
    for line in response.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            yield json.loads(line[len("data: ") :])


def _stream_generate(base, sid, max_tokens=80):
    """Stream a greedy generation; return (text, token_times) where token_times
    are perf_counter() stamps at each emitted token."""
    token_times: list[float] = []
    pieces: list[str] = []
    r = requests.post(
        f"{base}/sessions/{sid}/generate",
        json={"stream": True, "max_tokens": max_tokens, "temperature": 0.0},
        stream=True,
        timeout=180,
    )
    assert r.status_code == 200, r.text
    for ev in _parse_sse(r):
        if ev.get("type") == "token":
            token_times.append(time.perf_counter())
            pieces.append(ev.get("token", ""))
    return "".join(pieces), token_times


def _inter_token_gaps_ms(token_times):
    """Inter-token gaps in ms, EXCLUDING time-to-first-token (isolates steady-state)."""
    return [(token_times[i] - token_times[i - 1]) * 1000.0 for i in range(2, len(token_times))]


# ------------------------------- SC #8 --------------------------------------


def test_offload_load_does_not_degrade_concurrent_generation(base, make_session):
    """While session A generates greedily, repeatedly offload+load two OTHER
    sessions (B, C) holding real-sized caches. A's output must be byte-identical
    to a solo run, and its throughput/latency must stay within the gates above."""
    # baseline: identical starting cache, run SOLO (no offload/load happening).
    control = make_session()
    _inject_chat(base, control)
    text_baseline, times_baseline = _stream_generate(base, control, max_tokens=80)
    assert text_baseline, "baseline produced no text"
    n_baseline = len(times_baseline)
    gaps_baseline = _inter_token_gaps_ms(times_baseline)
    baseline_dur = (times_baseline[-1] - times_baseline[0]) if n_baseline > 1 else 1.0
    toks_per_s_baseline = (n_baseline - 1) / baseline_dur if n_baseline > 1 else 0.0
    maxgap_baseline = max(gaps_baseline) if gaps_baseline else 0.0
    print(
        f"  [sc8-baseline] {n_baseline} tok, {toks_per_s_baseline:.1f} tok/s, "
        f"max inter-token gap {maxgap_baseline:.1f} ms"
    )

    # two sizable-cache sessions to thrash offload/load during A's generation
    b = make_session()
    _inject_text(base, b, BIG_TEXT)
    c = make_session()
    _inject_text(base, c, BIG_TEXT)

    subject = make_session()
    _inject_chat(base, subject)  # identical starting cache to control

    done = threading.Event()

    def stress():
        # Repeatedly round-trip B and C between VRAM and RAM while A generates.
        # Resilient: a single failed cycle (transient GPU/VRAM pressure on the
        # slow model) just continues — the gate is A's output + throughput, not
        # every cycle succeeding.
        while not done.is_set():
            for sid in (b, c):
                if done.is_set():
                    break
                try:
                    if _offload(base, sid).status_code == 200:
                        _load(base, sid)
                except requests.exceptions.RequestException:
                    continue

    stresser = threading.Thread(target=stress, daemon=True)
    stresser.start()
    try:
        text_subject, times_subject = _stream_generate(base, subject, max_tokens=80)
    finally:
        done.set()
        stresser.join(30)
        assert not stresser.is_alive(), "stresser did not stop"

    n_subject = len(times_subject)
    gaps_subject = _inter_token_gaps_ms(times_subject)
    subject_dur = (times_subject[-1] - times_subject[0]) if n_subject > 1 else 1.0
    toks_per_s_subject = (n_subject - 1) / subject_dur if n_subject > 1 else 0.0
    maxgap_subject = max(gaps_subject) if gaps_subject else 0.0
    print(
        f"  [sc8-concurrent] {n_subject} tok, {toks_per_s_subject:.1f} tok/s, "
        f"max inter-token gap {maxgap_subject:.1f} ms "
        f"(baseline {toks_per_s_baseline:.1f} tok/s / {maxgap_baseline:.1f} ms)"
    )

    # --- Correctness (hard guarantee): identical greedy output ---
    assert text_subject == text_baseline, (
        "concurrent offload/load perturbed the generating session's decoding:\n"
        f"  baseline : {text_baseline!r}\n  subject  : {text_subject!r}"
    )

    # --- Bounded degradation (regression gate) ---
    if toks_per_s_baseline > 0:
        assert toks_per_s_subject >= THROUGHPUT_FRACTION * toks_per_s_baseline, (
            f"throughput dropped {toks_per_s_baseline:.1f} -> {toks_per_s_subject:.1f} tok/s "
            f"(below {THROUGHPUT_FRACTION}*baseline)"
        )
    gap_bound = max(MAX_GAP_MULTIPLE * maxgap_baseline, MAX_GAP_FLOOR_MS)
    assert maxgap_subject <= gap_bound, (
        f"single-token stall {maxgap_subject:.1f} ms exceeded bound {gap_bound:.1f} ms "
        f"({MAX_GAP_MULTIPLE}*{maxgap_baseline:.1f} or floor {MAX_GAP_FLOOR_MS} ms)"
    )


# ----------------------- SC #9 shared-KV sentinel ---------------------------


def test_offload_forked_session_during_active_generation__shared_kv_sentinel(base, make_session):
    """PLACEHOLDER / FORWARD-COMPAT SENTINEL.

    Fork A -> A' (identical KV). Stream-generate on A; mid-generation, offload A'.
    A must finish with byte-identical greedy output to an identical solo run, and
    A' must end up in RAM.

    TODAY this passes trivially: A' is an independent deep copy (ADR 0004 A'), so
    offloading it cannot touch A's cache. The test exists to CATCH a regression
    when slice-5 (EUS-4) makes A and A' SHARE KV via llama_memory_seq_cp in a
    pooled context: then offloading A' must NOT evict shared cells A depends on.
    If that future change breaks A's decoding, this test fails loudly here.
    """
    a = make_session()
    _inject_chat(base, a)
    ap = _fork(base, a)["session_id"]

    control = make_session()
    _inject_chat(base, control)  # identical starting cache to A
    baseline, _ = _stream_generate(base, control, max_tokens=80)
    assert baseline

    outcome: dict = {}

    def generate():
        text, _ = _stream_generate(base, a, max_tokens=80)
        outcome["text"] = text

    gen = threading.Thread(target=generate, daemon=True)
    gen.start()

    # Give A a moment to be actively decoding, then offload the fork A' once,
    # mid-generation. (We can't observe A's token count cheaply from outside the
    # SSE stream; a short fixed delay reliably lands the offload during decode.)
    time.sleep(0.3)
    assert gen.is_alive(), "A finished before we could offload the fork mid-flight"
    r = _offload(base, ap)
    assert r.status_code == 200, r.text

    gen.join(60)
    assert not gen.is_alive(), "A's generation did not complete"

    text_a = outcome.get("text", "")
    assert text_a == baseline, (
        "offloading the forked session perturbed the source's decoding "
        "(this is the shared-KV sentinel — see test docstring):\n"
        f"  baseline : {baseline!r}\n  A        : {text_a!r}"
    )
    assert _status(base, ap).json()["location"] == "ram"
    print("  [sc9-sentinel] offloaded fork mid-generation; source output unchanged")
