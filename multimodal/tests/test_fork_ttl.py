"""Idle TTL for throwaway forks (owner decision 2026-10-19).

Each test proves one part of the contract:
1. A fork without ``ttl_ms`` (absent or JSON null) never expires and
   reports ``ttl_ms`` null.
2. An expired idle fork answers 404 and is reclaimed by the next allocation,
   restoring ``pool.free_sequences`` with no DELETE for that fork.
3. A busy fork never expires: expiry counts idle time only, so a fork whose
   readout outlasts its TTL is reclaimed only after it becomes idle again.
4. The review finding: a client that times out on a queued fork POST and
   never learns the fork id leaks nothing; the TTL reaps the fork.
5. Invalid ``ttl_ms`` values are 400 before any allocation.
6. Perf: a TTL fork costs no measurable extra ``fork_ms_precise``, and a
   usage sweep releasing 3 expired forks stays within 50 ms on GPU.

Every test restores ``pool.free_sequences`` in a ``finally`` block via
:func:`_wait_free`, which polls with usage calls (each call also sweeps).
Reads (status/usage) never refresh a session's idle clock; only a
busy -> idle transition does.
"""

from __future__ import annotations

import statistics
import threading
import time

import pytest
import requests
from test_readout import (
    BASE_TEXT,
    GATE_CANDIDATES,
    GATE_TOP_K,
    _create,
    _fork,
    _ok,
    _on_gpu,
    _pool,
    _post,
    _readout,
)
from test_readout_robustness import (
    LONG_SUFFIX,
    LONG_TEXT,
    _assert_readout,
    _make_base,
)

pytestmark = pytest.mark.usefixtures("base")


def _free(base) -> int:
    return _pool(base)["free_sequences"]


def _wait_free(base, expected, timeout=10.0) -> None:
    deadline = time.time() + timeout
    free = None
    while time.time() < deadline:
        free = _free(base)
        if free == expected:
            return
        time.sleep(0.2)
    raise AssertionError(f"free_sequences stuck at {free}, expected {expected}")


def _cleanup(base, sid) -> None:
    if sid is not None:
        try:
            requests.delete(f"{base}/sessions/{sid}", timeout=30)
        except requests.exceptions.RequestException:
            pass


def _usage(base):
    return _ok(requests.get(f"{base}/sessions/usage", timeout=30))


def _live_entry(base, sid):
    for entry in _usage(base)["vram"]["sessions"]:
        if entry["session_id"] == sid:
            return entry
    return None


def _fork_body(base, sid, body):
    response = _post(base, f"/sessions/{sid}/fork", body)
    assert response.status_code == 200, response.text
    return response.json()


def _make_simple_base(base):
    sid = _create(base)
    try:
        _ok(_post(base, f"/sessions/{sid}/inject", {"text": BASE_TEXT}))
        return sid
    except Exception:
        _cleanup(base, sid)
        raise


def _long_suffix(base) -> str:
    return LONG_SUFFIX if _on_gpu(base) else " word" * 150


def test_fork_without_ttl_never_expires(base):
    base_sid = fork = extra = None
    initial = None
    try:
        base_sid = _make_simple_base(base)
        initial = _free(base)
        body = _fork_body(base, base_sid, {"ttl_ms": None})  # null == omitted
        fork = body["session_id"]
        assert body["ttl_ms"] is None, body
        time.sleep(1.5)
        assert _live_entry(base, fork)["ttl_ms"] is None
        extra = _create(base)
        _cleanup(base, extra)
        extra = None
        assert requests.get(f"{base}/sessions/{fork}", timeout=30).status_code == 200
        assert _live_entry(base, fork)["ttl_ms"] is None
        _cleanup(base, fork)
        fork = None
    finally:
        _cleanup(base, fork)
        _cleanup(base, extra)
        if initial is not None:
            _wait_free(base, initial)
        _cleanup(base, base_sid)


def test_expired_idle_fork_is_reclaimed_on_next_allocation(base):
    base_sid = fork = extra = None
    initial = None
    try:
        base_sid = _make_simple_base(base)
        initial = _free(base)
        body = _fork_body(base, base_sid, {"ttl_ms": 300})
        fork = body["session_id"]
        assert body["ttl_ms"] == 300, body
        entry = _live_entry(base, fork)
        assert entry["ttl_ms"] == 300 and entry["idle_ms"] >= 0, entry
        time.sleep(0.6)
        assert requests.get(f"{base}/sessions/{fork}", timeout=30).status_code == 404
        extra = _create(base)
        _cleanup(base, extra)
        extra = None
        assert _free(base) == initial
        assert requests.delete(f"{base}/sessions/{fork}", timeout=30).status_code == 404
        fork = None
    finally:
        _cleanup(base, fork)
        _cleanup(base, extra)
        if initial is not None:
            _wait_free(base, initial)
        _cleanup(base, base_sid)


def test_busy_fork_does_not_expire(base):
    base_sid = fork = None
    initial = None
    thread = None
    try:
        base_sid = _make_simple_base(base)
        initial = _free(base)
        fork = _fork_body(base, base_sid, {"ttl_ms": 200})["session_id"]
        result: dict = {}

        def run_readout():
            try:
                result["body"] = _readout(
                    base,
                    fork,
                    text=_long_suffix(base),
                    candidates=GATE_CANDIDATES,
                    top_k=GATE_TOP_K,
                    priority=True,
                )
            except Exception as exc:  # noqa: BLE001 - recorded, asserted below
                result["error"] = repr(exc)

        thread = threading.Thread(target=run_readout)
        thread.start()
        _usage(base)
        sweeper = _create(base)
        _cleanup(base, sweeper)
        thread.join(300)
        assert "body" in result, result
        _assert_readout(result["body"])
        time.sleep(0.4)
        _usage(base)
        assert requests.get(f"{base}/sessions/{fork}", timeout=30).status_code == 404
        fork = None
        assert _free(base) == initial
    finally:
        if thread is not None:
            thread.join(300)
        _cleanup(base, fork)
        if initial is not None:
            _wait_free(base, initial)
        _cleanup(base, base_sid)


def test_fork_timeout_without_id_leaks_nothing(base):
    base_sid = load_sid = fork = None
    initial = None
    thread = None
    try:
        base_sid = _make_base(base)
        initial = _free(base)
        load_sid = _create(base)
        load: dict = {}

        def run_load():
            try:
                text = LONG_TEXT if _on_gpu(base) else " word" * 300
                load["status"] = _post(
                    base, f"/sessions/{load_sid}/inject", {"text": text}
                ).status_code
            except Exception as exc:  # noqa: BLE001 - recorded, asserted below
                load["error"] = repr(exc)

        thread = threading.Thread(target=run_load)
        thread.start()
        time.sleep(0.15)
        try:
            response = requests.post(
                f"{base}/sessions/{base_sid}/fork",
                json={"ttl_ms": 500},
                timeout=(5, 0.02),
            )
        except requests.exceptions.ReadTimeout:
            pass
        else:
            if response.status_code == 200:
                _cleanup(base, response.json()["session_id"])
            pytest.skip("fork was not queued")
        thread.join(300)
        thread = None
        assert load.get("status") == 200, load
        _cleanup(base, load_sid)
        load_sid = None
        _wait_free(base, initial, timeout=10)
        fork = _fork(base, base_sid)
        _assert_readout(
            _readout(base, fork, candidates=GATE_CANDIDATES, top_k=GATE_TOP_K, priority=True)
        )
        _cleanup(base, fork)
        fork = None
    finally:
        if thread is not None:
            thread.join(300)
        _cleanup(base, fork)
        _cleanup(base, load_sid)
        if initial is not None:
            _wait_free(base, initial, timeout=10)
        _cleanup(base, base_sid)


def test_ttl_validation_is_400_and_allocates_nothing(base):
    base_sid = None
    initial = None
    try:
        base_sid = _make_simple_base(base)
        initial = _free(base)
        for body in (
            {"ttl_ms": -1},
            {"ttl_ms": "5"},
            {"ttl_ms": 1.5},
            {"ttl_ms": True},
        ):
            response = _post(base, f"/sessions/{base_sid}/fork", body)
            assert response.status_code == 400, response.text
            assert "ttl_ms must be a non-negative integer" in response.text, response.text
            assert _free(base) == initial
    finally:
        if initial is not None:
            _wait_free(base, initial)
        _cleanup(base, base_sid)


def test_fork_with_ttl_costs_nothing_measurable(base):
    base_sid = None
    initial = None
    pending: list[str] = []
    try:
        base_sid = _make_simple_base(base)
        initial = _free(base)
        without, with_ttl = [], []
        for _ in range(20):
            body = _fork_body(base, base_sid, {})
            without.append(body["fork_ms_precise"])
            _cleanup(base, body["session_id"])
        for _ in range(20):
            body = _fork_body(base, base_sid, {"ttl_ms": 60000})
            with_ttl.append(body["fork_ms_precise"])
            _cleanup(base, body["session_id"])
        med_without = statistics.median(without)
        med_with = statistics.median(with_ttl)
        print(f"\n[fork-ttl] fork_ms median without={med_without:.3f} with={med_with:.3f}")
        assert med_with <= med_without + 1.0, (med_with, med_without)
        usage_ms = []
        for _ in range(3):
            for _ in range(3):
                pending.append(_fork_body(base, base_sid, {"ttl_ms": 100})["session_id"])
            time.sleep(0.3)
            start = time.perf_counter()
            _usage(base)
            usage_ms.append((time.perf_counter() - start) * 1000.0)
            pending.clear()
        print(f"\n[fork-ttl] usage sweep ms={usage_ms}")
        if _on_gpu(base):
            assert statistics.median(usage_ms) <= 50.0, usage_ms
    finally:
        for sid in pending:
            _cleanup(base, sid)
        if initial is not None:
            _wait_free(base, initial)
        _cleanup(base, base_sid)
