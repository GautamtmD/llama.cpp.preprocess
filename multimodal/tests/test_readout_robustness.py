"""Robustness tests for the ADR 0010 priority inject-with-readout path.

Each test proves no pooled slot leaks and that sessions stay deletable:
(1) client timeout on a queued priority readout (x3), the request finishing
server-side; (2) client timeout on a long-prefilling readout, no other load;
(3) a priority readout behind a running generation beats a queued normal
inject while the generation streams before/after it without error; (4) 100
rapid fork -> readout -> delete cycles leave the pool unchanged.
"""

from __future__ import annotations

import json
import math
import threading
import time

import pytest
import requests
from test_readout import (
    BASE_TEXT,
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
    _status,
)

pytestmark = pytest.mark.usefixtures("base")

LONG_TEXT = " word" * 2000  # ~2000-token normal inject (~0.8-1 s on GPU)
LONG_SUFFIX = " word" * 1500  # ~1500-token readout suffix


def _free(base) -> int:
    return _pool(base)["free_sequences"]


def _delete_ok(base, sid) -> None:
    response = requests.delete(f"{base}/sessions/{sid}", timeout=30)
    assert response.status_code == 200, response.text


def _cleanup(base, sid) -> None:
    if sid is not None:
        try:
            requests.delete(f"{base}/sessions/{sid}", timeout=30)
        except requests.exceptions.RequestException:
            pass


def _token_count(base, text) -> int:
    # Prime first: a first inject also adds BOS, which a fork does not re-add.
    sid = _create(base)
    try:
        _ok(_post(base, f"/sessions/{sid}/inject", {"text": "x"}))
        return _ok(_post(base, f"/sessions/{sid}/inject", {"text": text}))["tokens_injected"]
    finally:
        _cleanup(base, sid)


def _make_base(base) -> str:
    sid = _create(base)
    try:
        _ok(_post(base, f"/sessions/{sid}/inject", {"text": BASE_TEXT}))
        while _status(base, sid)["cache_size"] < 300:
            missing = 300 - _status(base, sid)["cache_size"]
            _ok(_post(base, f"/sessions/{sid}/inject", {"text": " word" * missing}))
        return sid
    except Exception:
        _cleanup(base, sid)
        raise


@pytest.fixture
def base300(base):
    sid = _make_base(base)
    yield sid
    _cleanup(base, sid)


def _assert_readout(body) -> None:
    readout = body["readout"]
    for entry in readout["candidates"] + readout["top"]:
        assert math.isfinite(entry["logit"]) and math.isfinite(entry["logprob"])
    assert 0.0 <= readout["candidate_mass"] <= 1.0


def _fresh_readout(base, base_sid) -> None:
    fork = _fork(base, base_sid)
    try:
        _assert_readout(
            _readout(base, fork, candidates=GATE_CANDIDATES, top_k=GATE_TOP_K, priority=True)
        )
    finally:
        _cleanup(base, fork)


def _wait_cache(base, sid, expected, timeout=10.0) -> None:
    deadline = time.time() + timeout
    cache = None
    while time.time() < deadline:
        cache = _status(base, sid)["cache_size"]
        if cache == expected:
            return
        time.sleep(0.05)
    raise AssertionError(f"cache_size stuck at {cache}, expected {expected}")


def _raw_readout(base, fork, text, read_timeout):
    body = {"candidates": GATE_CANDIDATES, "top_k": GATE_TOP_K}
    return requests.post(
        f"{base}/sessions/{fork}/inject",
        json={"text": text, "readout": body, "priority": True},
        timeout=(5, read_timeout),
    )


def test_client_timeout_during_queued_readout_leaks_nothing(base, base300):
    if _free(base) < 2:
        pytest.skip("needs >= 2 free sequences for fork + load session")
    initial = _free(base)
    base_size = _status(base, base300)["cache_size"]
    suffix_tokens = _token_count(base, SUFFIX)
    timeouts = 0
    for _ in range(3):
        fork = load_sid = None
        load: dict = {}
        thread = None
        try:
            fork = _fork(base, base300)
            load_sid = _create(base)

            def run_load(load_sid=load_sid, load=load):
                try:
                    r = _post(base, f"/sessions/{load_sid}/inject", {"text": LONG_TEXT})
                    load["status"] = r.status_code
                except Exception as exc:  # noqa: BLE001 - recorded, asserted below
                    load["error"] = repr(exc)

            thread = threading.Thread(target=run_load)
            thread.start()
            time.sleep(0.15)
            try:
                response = _raw_readout(base, fork, SUFFIX, read_timeout=0.02)
                # Racing the load and finishing anyway is acceptable (brief).
                assert response.status_code == 200, response.text
            except requests.exceptions.ReadTimeout:
                timeouts += 1
            thread.join(300)
            assert load.get("status") == 200, load
            _wait_cache(base, fork, base_size + suffix_tokens)
            _delete_ok(base, fork)
            fork = None
            _delete_ok(base, load_sid)
            load_sid = None
            assert _free(base) == initial
            _fresh_readout(base, base300)
        finally:
            if thread is not None and thread.is_alive():
                thread.join(300)
            _cleanup(base, fork)
            _cleanup(base, load_sid)
    print(f"\n[robustness] queued readout timed out in {timeouts}/3 iterations")


def test_client_timeout_during_prefilling_readout_leaks_nothing(base, base300):
    if _free(base) < 1:
        pytest.skip("needs >= 1 free sequence for the fork")
    if not _on_gpu(base):
        # On CPU the 1500-token prefill runs for minutes and, as a priority
        # request, holds the lane, so the status poll below cannot complete
        # within its timeout. The no-leak-after-timeout property is covered
        # on CPU by the queued-readout test above.
        pytest.skip("GPU only: CPU prefill of the long suffix outlasts the status poll")
    initial = _free(base)
    base_size = _status(base, base300)["cache_size"]
    suffix_tokens = _token_count(base, LONG_SUFFIX)
    fork = None
    try:
        fork = _fork(base, base300)
        with pytest.raises(requests.exceptions.ReadTimeout):
            _raw_readout(base, fork, LONG_SUFFIX, read_timeout=0.05)
        _wait_cache(base, fork, base_size + suffix_tokens)
        _delete_ok(base, fork)
        fork = None
        assert _free(base) == initial
        _fresh_readout(base, base300)
    finally:
        _cleanup(base, fork)


def test_priority_readout_behind_running_generation_beats_queued_normal_work(base):
    if _free(base) < 4:
        pytest.skip("needs 4 free sequences for generation, base, fork, and load")
    initial = _free(base)
    created: list[str] = []
    threads: list[threading.Thread] = []
    try:
        gen_sid = _create(base)
        created.append(gen_sid)
        _ok(_post(base, f"/sessions/{gen_sid}/inject", {"text": NORMAL_TEXT + SUFFIX}))
        base_sid = _make_base(base)
        created.append(base_sid)
        fork = _fork(base, base_sid)
        created.append(fork)
        load_sid = _create(base)
        created.append(load_sid)

        gen: dict = {"times": [], "errors": [], "done": False}
        first_token = threading.Event()

        def run_generation():
            try:
                with requests.post(
                    f"{base}/sessions/{gen_sid}/generate",
                    json={"max_tokens": 300, "temperature": 0, "stream": True},
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
                            gen["times"].append(time.perf_counter())
                            first_token.set()
                        if event.get("type") == "done":
                            gen["done"] = True
                        if event.get("type") == "error":
                            gen["errors"].append(str(event))
                        if event.get("type") in ("done", "error"):
                            break
            except Exception as exc:  # noqa: BLE001 - recorded, asserted below
                gen["errors"].append(repr(exc))

        gen_thread = threading.Thread(target=run_generation)
        threads.append(gen_thread)
        gen_thread.start()
        assert first_token.wait(30), "generation never produced a token"

        load: dict = {}

        def run_load():
            try:
                r = _post(base, f"/sessions/{load_sid}/inject", {"text": LONG_TEXT})
                load["status"] = r.status_code
            except Exception as exc:  # noqa: BLE001 - recorded, asserted below
                load["error"] = repr(exc)
            finally:
                load["finished"] = time.perf_counter()

        load_thread = threading.Thread(target=run_load)
        threads.append(load_thread)
        load_thread.start()
        time.sleep(0.15)
        body = _readout(base, fork, candidates=GATE_CANDIDATES, top_k=GATE_TOP_K, priority=True)
        readout_finished = time.perf_counter()
        generation_done_at_readout = gen["done"]
        _assert_readout(body)
        load_thread.join(300)
        assert load.get("status") == 200, load
        assert readout_finished < load["finished"], (readout_finished, load)
        assert not generation_done_at_readout, "generation ended before the readout"
        assert any(t < readout_finished for t in gen["times"]), gen["times"]
        assert any(t > readout_finished for t in gen["times"]), gen["times"]

        cancel = _post(base, f"/sessions/{gen_sid}/cancel")
        assert cancel.status_code == 200, cancel.text
        gen_thread.join(30)
        assert not gen_thread.is_alive(), "generation did not stop after cancel"
        assert not gen["errors"], gen["errors"]
        for sid in reversed(created):
            _delete_ok(base, sid)
        created.clear()
    finally:
        try:
            if created:
                _post(base, f"/sessions/{created[0]}/cancel")
        except Exception:
            pass
        for thread in threads:
            thread.join(30)
        for sid in reversed(created):
            _cleanup(base, sid)
    assert _free(base) == initial


def test_many_rapid_priority_readouts_leak_nothing(base, base300):
    if _free(base) < 1:
        pytest.skip("needs >= 1 free sequence for the fork")
    initial = _free(base)
    last = None
    for _ in range(100):
        fork = _fork(base, base300)
        try:
            last = _readout(base, fork, candidates=GATE_CANDIDATES, top_k=GATE_TOP_K, priority=True)
        finally:
            _delete_ok(base, fork)
    assert _free(base) == initial
    _assert_readout(last)
