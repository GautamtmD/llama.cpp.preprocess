"""Deferred DELETE /sessions/{id} on busy live sessions (owner decision 2026-10-19).

Each test proves one part of the contract:
1. DELETE during a prefilling priority readout returns 200/deferred in
   <100 ms, the session is immediately invisible (GET/re-DELETE 404), and
   the pooled slot is freed by the engine when the readout finishes.
2. DELETE during a queued priority readout (behind a normal inject) also
   returns immediately and deferred; the other session is unaffected.
3. DELETE during a streaming generation defers until the stream ends:
   the consumer sees no error and keeps receiving tokens after the DELETE.
4. DELETE on an idle session stays synchronous (deferred false, slot back
   on the very next pool read).
5. Three back-to-back score-shaped client timeouts + DELETEs (the fork
   leak this fixes) leak nothing and a fresh readout still works.

Every test restores ``pool.free_sequences`` to its initial value in a
``finally`` block via :func:`_wait_free`.
"""

from __future__ import annotations

import json
import threading
import time

import pytest
import requests
from test_readout import (
    GATE_CANDIDATES,
    GATE_TOP_K,
    NORMAL_TEXT,
    SUFFIX,
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
    _raw_readout,
)

pytestmark = pytest.mark.usefixtures("base")


@pytest.fixture
def base300(base):
    sid = _make_base(base)
    yield sid
    _cleanup(base, sid)


def _free(base) -> int:
    return _pool(base)["free_sequences"]


def _wait_free(base, expected, timeout=60.0) -> None:
    deadline = time.time() + timeout
    free = None
    while time.time() < deadline:
        free = _free(base)
        if free == expected:
            return
        time.sleep(0.05)
    raise AssertionError(f"free_sequences stuck at {free}, expected {expected}")


def _cleanup(base, sid) -> None:
    if sid is not None:
        try:
            requests.delete(f"{base}/sessions/{sid}", timeout=30)
        except requests.exceptions.RequestException:
            pass


def _delete(base, sid):
    start = time.perf_counter()
    response = requests.delete(f"{base}/sessions/{sid}", timeout=30)
    return response, time.perf_counter() - start


def _long_suffix(base) -> str:
    return LONG_SUFFIX if _on_gpu(base) else " word" * 150


def _fresh_readout(base, base_sid) -> None:
    fork = _fork(base, base_sid)
    try:
        _assert_readout(
            _readout(base, fork, candidates=GATE_CANDIDATES, top_k=GATE_TOP_K, priority=True)
        )
    finally:
        _cleanup(base, fork)


def test_delete_during_prefilling_readout_returns_immediately_and_frees_slot_later(base, base300):
    initial = _free(base)
    fork = None
    try:
        fork = _fork(base, base300)
        timed_out = False
        try:
            response = _raw_readout(base, fork, _long_suffix(base), read_timeout=0.05)
            assert response.status_code == 200, response.text
        except requests.exceptions.ReadTimeout:
            timed_out = True
        response, elapsed = _delete(base, fork)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["deleted"] is True and body["deferred"] is timed_out, body
        if timed_out:
            assert elapsed < 0.1, f"DELETE took {elapsed * 1000:.1f} ms"
        assert requests.get(f"{base}/sessions/{fork}", timeout=30).status_code == 404
        again, _ = _delete(base, fork)
        assert again.status_code == 404, again.text
        fork = None
        _wait_free(base, initial, timeout=60)
        _fresh_readout(base, base300)
    finally:
        _cleanup(base, fork)
        _wait_free(base, initial, timeout=60)


def test_delete_during_queued_readout_returns_immediately(base, base300):
    initial = _free(base)
    fork = other = None
    thread = None
    try:
        fork = _fork(base, base300)
        other = _create(base)
        load: dict = {}

        def run_load():
            try:
                text = LONG_TEXT if _on_gpu(base) else " word" * 300
                load["status"] = _post(
                    base, f"/sessions/{other}/inject", {"text": text}
                ).status_code
            except Exception as exc:  # noqa: BLE001 - recorded, asserted below
                load["error"] = repr(exc)

        thread = threading.Thread(target=run_load)
        thread.start()
        time.sleep(0.15)
        with pytest.raises(requests.exceptions.ReadTimeout):
            _raw_readout(base, fork, _long_suffix(base), read_timeout=0.02)
        response, elapsed = _delete(base, fork)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["deleted"] is True and body["deferred"] is True, body
        assert elapsed < 0.1, f"DELETE took {elapsed * 1000:.1f} ms"
        fork = None
        thread.join(300)
        assert load.get("status") == 200, load
        response, _ = _delete(base, other)
        assert response.status_code == 200, response.text
        other = None
    finally:
        if thread is not None:
            thread.join(300)
        _cleanup(base, fork)
        _cleanup(base, other)
        _wait_free(base, initial, timeout=60)


def test_delete_during_running_generation_defers_until_stream_ends(base):
    initial = _free(base)
    sid = None
    thread = None
    try:
        sid = _create(base)
        _ok(_post(base, f"/sessions/{sid}/inject", {"text": NORMAL_TEXT + SUFFIX}))
        gen: dict = {"tokens": [], "errors": [], "done": False}
        first_token = threading.Event()

        def run_generation():
            try:
                with requests.post(
                    f"{base}/sessions/{sid}/generate",
                    json={"max_tokens": 60, "temperature": 0, "stream": True},
                    stream=True,
                    timeout=300,
                ) as response:
                    if response.status_code != 200:
                        gen["errors"].append(response.text)
                        return
                    for line in response.iter_lines(decode_unicode=True):
                        if not line or not line.startswith("data:"):
                            continue
                        event = json.loads(line[5:].strip())
                        if event.get("type") == "token":
                            gen["tokens"].append(time.perf_counter())
                            first_token.set()
                        if event.get("type") == "done":
                            gen["done"] = True
                        if event.get("type") == "error":
                            gen["errors"].append(str(event))
                        if event.get("type") in ("done", "error"):
                            break
            except Exception as exc:  # noqa: BLE001 - recorded, asserted below
                gen["errors"].append(repr(exc))

        thread = threading.Thread(target=run_generation)
        thread.start()
        assert first_token.wait(30), "generation never produced a token"
        response, elapsed = _delete(base, sid)
        deleted_at = time.perf_counter()
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["deleted"] is True and body["deferred"] is True, body
        assert elapsed < 0.1, f"DELETE took {elapsed * 1000:.1f} ms"
        assert requests.get(f"{base}/sessions/{sid}", timeout=30).status_code == 404
        sid = None
        thread.join(300)
        assert not thread.is_alive()
        assert not gen["errors"], gen["errors"]
        assert gen["done"], gen
        assert any(t > deleted_at for t in gen["tokens"]), gen["tokens"]
    finally:
        if thread is not None:
            thread.join(300)
        _cleanup(base, sid)
        _wait_free(base, initial, timeout=60)


def test_delete_idle_session_is_still_synchronous(base):
    initial = _free(base)
    sid = None
    try:
        sid = _create(base)
        _ok(_post(base, f"/sessions/{sid}/inject", {"text": "hello world"}))
        response, _ = _delete(base, sid)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["deleted"] is True and body["deferred"] is False, body
        sid = None
        assert _free(base) == initial
    finally:
        _cleanup(base, sid)
        _wait_free(base, initial, timeout=60)


def test_score_shaped_timeouts_back_to_back_leak_nothing(base, base300):
    initial = _free(base)
    forks: list[str] = []
    try:
        for _ in range(3):
            fork = _fork(base, base300)
            forks.append(fork)
            with pytest.raises(requests.exceptions.ReadTimeout):
                _raw_readout(base, fork, _long_suffix(base), read_timeout=0.02)
            response, _ = _delete(base, fork)
            assert response.status_code == 200, response.text
            assert response.json()["deleted"] is True, response.text
        forks.clear()
        _wait_free(base, initial, timeout=90)
        _fresh_readout(base, base300)
    finally:
        for fork in forks:
            _cleanup(base, fork)
        _wait_free(base, initial, timeout=90)
