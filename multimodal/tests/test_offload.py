"""Behavior tests for KV-cache offload/load (EUS-6).

Run against a LIVE multimodal-server (built from this branch). Model-agnostic —
launch with the model as a parameter (see tests/conftest.py docstring)::

    MULTIMODAL_MODEL="C:/ML Models/MiniCPM-V-4.6/MiniCPM-V-4_6-Q8_0.gguf" \\
    MULTIMODAL_MMPROJ="C:/ML Models/MiniCPM-V-4.6/mmproj-model-f16.gguf" \\
    pytest engine/multimodal/tests/test_offload.py

These tests assert only on the HTTP contract and on byte-identical greedy
(temp<=0) reproduction — never on model-specific content — so they pass
identically on any model the engine can load. They cover SC #1-#6 and #9; the
inference-impact gate (SC #8) and the shared-KV sentinel live in
``test_offload_concurrent.py``.
"""

from __future__ import annotations

import pytest
import requests

pytestmark = [pytest.mark.usefixtures("base", "make_session"), pytest.mark.requires("text")]

PROMPT = [{"role": "user", "content": "List three colors, comma-separated."}]


# ------------------------------- helpers ------------------------------------


def _inject(base, sid, messages=None, text=None):
    body = {"messages": messages} if messages is not None else {"text": text}
    r = requests.post(f"{base}/sessions/{sid}/inject", json=body, timeout=120)
    assert r.status_code == 200, r.text
    return r.json()


def _inject_chat(base, sid, messages=PROMPT):
    return _inject(base, sid, messages=messages)


def _generate(base, sid, **kw):
    body = {"max_tokens": kw.pop("max_tokens", 40), "temperature": kw.pop("temperature", 0.0)}
    body.update(kw)
    r = requests.post(f"{base}/sessions/{sid}/generate", json=body, timeout=120)
    assert r.status_code == 200, r.text
    return r.json()


def _generate_raw(base, sid, **kw):
    """Like _generate but returns the raw Response (for asserting status codes)."""
    body = {"max_tokens": kw.pop("max_tokens", 40), "temperature": kw.pop("temperature", 0.0)}
    body.update(kw)
    return requests.post(f"{base}/sessions/{sid}/generate", json=body, timeout=120)


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


def _usage(base):
    return requests.get(f"{base}/sessions/usage", timeout=60)


def _delete(base, sid):
    return requests.delete(f"{base}/sessions/{sid}", timeout=60)


# ------------------------------- SC #1: offload ----------------------------


def test_offload_moves_session_to_ram(base, make_session):
    """Offload flips location to ram and drops the VRAM footprint to ~0 (proven
    via /usage), with the RAM bucket gaining the session's state_bytes."""
    sid = make_session()
    _inject_chat(base, sid)

    vram_before = _usage(base).json()["vram"]
    here_before = next(s for s in vram_before["sessions"] if s["session_id"] == sid)
    assert here_before["state_bytes"] > 0

    r = _offload(base, sid)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["session_id"] == sid
    assert j["location"] == "ram"
    assert j["state_bytes"] == here_before["state_bytes"]
    assert j["cache_size"] > 0
    assert isinstance(j["offload_ms"], int) and j["offload_ms"] >= 0

    usage = _usage(base).json()
    # VRAM no longer lists this session; RAM does.
    assert all(s["session_id"] != sid for s in usage["vram"]["sessions"])
    ram_here = next(s for s in usage["ram"]["sessions"] if s["session_id"] == sid)
    assert ram_here["state_bytes"] == here_before["state_bytes"]
    # GET agrees on location.
    assert _status(base, sid).json()["location"] == "ram"


def test_offload_is_idempotent(base, make_session):
    sid = make_session()
    _inject_chat(base, sid)
    assert _offload(base, sid).status_code == 200
    r2 = _offload(base, sid)
    assert r2.status_code == 200, r2.text
    j = r2.json()
    assert j["location"] == "ram"
    assert j["offload_ms"] == 0  # already RAM -> no-op


# ------------------------------- SC #2: load --------------------------------


def test_load_restores_to_vram(base, make_session):
    sid = make_session()
    _inject_chat(base, sid)
    assert _offload(base, sid).status_code == 200
    r = _load(base, sid)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["session_id"] == sid
    assert j["location"] == "vram"
    assert j["cache_size"] > 0
    assert j["state_bytes"] > 0
    assert isinstance(j["load_ms"], int) and j["load_ms"] >= 0
    assert _status(base, sid).json()["location"] == "vram"


def test_offload_load_round_trip_reproduces_greedy_output(base, make_session):
    """THE correctness gate (SC #2): a session that is offloaded then loaded must
    reproduce the exact greedy output of a never-offloaded session with the same
    cache. control and subject get identical prompts; only subject is round-tripped."""
    control = make_session()
    subject = make_session()
    _inject_chat(base, control)
    _inject_chat(base, subject)

    baseline = _generate(base, control, max_tokens=40)["text"]
    assert baseline  # non-empty

    assert _offload(base, subject).status_code == 200
    assert _load(base, subject).status_code == 200

    after = _generate(base, subject, max_tokens=40)["text"]
    assert after == baseline, (after, baseline)


def test_load_is_idempotent(base, make_session):
    sid = make_session()
    _inject_chat(base, sid)
    # loading a resident (never-offloaded) session is a no-op
    r = _load(base, sid)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["location"] == "vram"
    assert j["load_ms"] == 0


def test_load_capacity_failure_releases_loading_reservation(base, make_session):
    subject = make_session()
    _inject_chat(base, subject)
    assert _offload(base, subject).status_code == 200

    fillers = []
    while True:
        created = requests.post(f"{base}/sessions", timeout=30)
        if created.status_code == 503:
            break
        assert created.status_code == 200, created.text
        fillers.append(created.json()["session_id"])

    try:
        first = _load(base, subject)
        second = _load(base, subject)
        assert first.status_code == 503, first.text
        assert second.status_code == 503, second.text

        assert _delete(base, fillers.pop()).status_code == 200
        restored = _load(base, subject)
        assert restored.status_code == 200, restored.text
        assert _status(base, subject).json()["location"] == "vram"
    finally:
        for sid in fillers:
            _delete(base, sid)


# ------------------------------- SC #3: GET ---------------------------------


def test_get_session_reports_location_and_state_bytes(base, make_session):
    sid = make_session()
    _inject_chat(base, sid)
    r = _status(base, sid)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["session_id"] == sid
    assert j["location"] == "vram"
    assert j["cache_size"] > 0
    assert j["state_bytes"] > 0

    assert _offload(base, sid).status_code == 200
    j2 = _status(base, sid).json()
    assert j2["location"] == "ram"
    assert j2["state_bytes"] == j["state_bytes"]


# ------------------------------- SC #4: usage -------------------------------


def test_usage_aggregate_partitions_vram_and_ram(base, make_session):
    """GET /sessions/usage partitions sessions into vram/ram buckets whose per-session
    entries sum to the bucket totals. Offloading migrates one vram->ram and updates
    the totals."""
    a = make_session()
    b = make_session()
    _inject_chat(base, a)
    _inject(base, b, text="A second, smaller context.")

    def check_buckets(u):
        for bucket in ("vram", "ram"):
            total = sum(s["state_bytes"] for s in u[bucket]["sessions"])
            assert u[bucket]["n_sessions"] == len(u[bucket]["sessions"])
            assert u[bucket]["total_state_bytes"] == total

    u = _usage(base).json()
    check_buckets(u)
    assert u["vram"]["n_sessions"] >= 2
    assert u["ram"]["n_sessions"] == 0
    a_size = next(s for s in u["vram"]["sessions"] if s["session_id"] == a)["state_bytes"]
    vram_total_before = u["vram"]["total_state_bytes"]

    assert _offload(base, a).status_code == 200
    u2 = _usage(base).json()
    check_buckets(u2)
    # a migrated to ram
    assert any(s["session_id"] == a for s in u2["ram"]["sessions"])
    assert all(s["session_id"] != a for s in u2["vram"]["sessions"])
    ram_a = next(s for s in u2["ram"]["sessions"] if s["session_id"] == a)
    assert ram_a["state_bytes"] == a_size
    # vram total dropped by a's footprint
    assert u2["vram"]["total_state_bytes"] == vram_total_before - a_size


def test_usage_route_not_shadowed_by_id_pattern(base, make_session):
    """The literal GET /sessions/usage route must not be captured by the {id}
    pattern (session ids are s_<n>, so 'usage' is distinguishable, but be deliberate)."""
    make_session()  # ensure at least one session exists
    r = _usage(base)
    assert r.status_code == 200, r.text
    j = r.json()
    assert "vram" in j and "ram" in j
    assert "n_sessions" in j["vram"]


# ------------------------------- SC #5: 409s --------------------------------


def test_offloaded_session_inject_returns_409(base, make_session):
    sid = make_session()
    _inject_chat(base, sid)
    assert _offload(base, sid).status_code == 200
    r = requests.post(f"{base}/sessions/{sid}/inject", json={"text": "more"}, timeout=60)
    assert r.status_code == 409
    assert "offloaded" in r.json()["error"].lower()
    assert "/load" in r.json()["error"]


def test_offloaded_session_generate_returns_409(base, make_session):
    sid = make_session()
    _inject_chat(base, sid)
    assert _offload(base, sid).status_code == 200
    # non-streaming
    r = _generate_raw(base, sid, max_tokens=5)
    assert r.status_code == 409
    # streaming
    r2 = requests.post(
        f"{base}/sessions/{sid}/generate",
        json={"stream": True, "max_tokens": 5},
        stream=True,
        timeout=60,
    )
    assert r2.status_code == 409


def test_offloaded_session_fork_returns_409(base, make_session):
    sid = make_session()
    _inject_chat(base, sid)
    assert _offload(base, sid).status_code == 200
    r = requests.post(f"{base}/sessions/{sid}/fork", timeout=60)
    assert r.status_code == 409


def test_offloaded_session_cancel_is_noop(base, make_session):
    sid = make_session()
    _inject_chat(base, sid)
    assert _offload(base, sid).status_code == 200
    r = requests.post(f"{base}/sessions/{sid}/cancel", timeout=60)
    assert r.status_code == 200, r.text
    assert r.json() == {"session_id": sid, "cancelled": False}


def test_offloaded_session_delete_works(base, make_session):
    """DELETE on a RAM session frees the RAM buffer; the session is then gone."""
    sid = make_session()
    _inject_chat(base, sid)
    assert _offload(base, sid).status_code == 200
    r = _delete(base, sid)
    assert r.status_code == 200, r.text
    assert r.json()["deleted"] is True
    # now unknown
    assert _status(base, sid).status_code == 404


# ------------------------------- SC #6: 404s --------------------------------


def test_offload_unknown_session_404(base):
    r = _offload(base, "s_999999")
    assert r.status_code == 404
    assert "error" in r.json()


def test_load_unknown_session_404(base):
    r = _load(base, "s_999999")
    assert r.status_code == 404
    assert "error" in r.json()


def test_get_unknown_session_404(base):
    r = _status(base, "s_999999")
    assert r.status_code == 404
    assert "error" in r.json()


# ----------------- SC #9: KV-sharing correctness (current fork) ------------


def _identical_pair(base, make_session):
    """Two fresh sessions with identical KV (the second is a fork of the first)."""
    src = make_session()
    _inject_chat(base, src)
    fork = _fork(base, src)
    return src, fork["session_id"]


def test_offload_source_fork_other_still_correct(base, make_session):
    """Fork A->A'. Offloading the SOURCE (A) must not affect the fork (A'): A' still
    reproduces the deterministic greedy output of an identical never-offloaded session.
    (Today A' owns an independent KV copy — ADR 0004 A' — so this is trivially safe;
    this test is the canary for slice-5 shared KV.)"""
    a, ap = _identical_pair(base, make_session)
    control = make_session()
    _inject_chat(base, control)
    baseline = _generate(base, control, max_tokens=40)["text"]

    assert _offload(base, a).status_code == 200
    assert _status(base, a).json()["location"] == "ram"

    after = _generate(base, ap, max_tokens=40)["text"]
    assert after == baseline, (after, baseline)


def test_offload_fork_source_still_correct(base, make_session):
    """Fork A->A'. Offloading the FORK (A') must not affect the source (A): A still
    reproduces its deterministic greedy output. Mirror of the above."""
    a, ap = _identical_pair(base, make_session)
    control = make_session()
    _inject_chat(base, control)
    baseline = _generate(base, control, max_tokens=40)["text"]

    assert _offload(base, ap).status_code == 200

    after = _generate(base, a, max_tokens=40)["text"]
    assert after == baseline, (after, baseline)
