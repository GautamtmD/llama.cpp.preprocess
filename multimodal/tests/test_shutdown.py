"""Clean shutdown of multimodal-server (EUS-7 robustness, Step 19 criterion 4).

Self-managed module: boots its own server on a second GPU (never the CPU; see
test_context_limits for the GPU choice) and stops it with a console signal.

1. Shutdown with queued work: while a normal ~2000-token inject is running
   and a priority inject-with-readout is queued behind it, Ctrl-Break makes the
   server stop accepting connections, finish both requests (each client gets
   200 and a finite readout), log the drain line and exit 0 within 15 s of
   the signal (the work itself takes ~1.5 s on GPU); it must not hang on the
   scheduler worker or leave the process alive.
2. Shutdown when idle exits 0 within 5 s.
"""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import requests
from test_context_limits import EXE, MMPROJ, MODEL, _free_port, _tiny_gpu
from test_readout import BASE_TEXT, GATE_CANDIDATES, GATE_TOP_K
from test_readout_robustness import LONG_SUFFIX, LONG_TEXT

pytestmark = pytest.mark.skipif(os.name != "nt", reason="uses CTRL_BREAK_EVENT (Windows)")

SHUTDOWN_WITH_WORK_S = 15.0
SHUTDOWN_IDLE_S = 5.0


@contextmanager
def _server() -> Iterator[tuple[subprocess.Popen, str, tempfile._TemporaryFileWrapper]]:
    if not MODEL or not MMPROJ:
        pytest.skip("MULTIMODAL_MODEL and MULTIMODAL_MMPROJ are required")
    if not Path(EXE).exists():
        pytest.skip(f"server executable is not built: {EXE}")
    gpu = _tiny_gpu()
    if gpu is None or gpu == "-1":
        pytest.skip("needs a second GPU (or MULTIMODAL_TINY_CUDA_VISIBLE_DEVICES); no CPU run")
    port = _free_port()
    command = [
        str(EXE),
        "--model",
        MODEL,
        "--mmproj",
        MMPROJ,
        "--port",
        str(port),
        "--max-sequences",
        "2",
        "--n-gpu-layers",
        "99",
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    log = tempfile.TemporaryFile(mode="w+b")
    process = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 90
    started = False
    while time.monotonic() < deadline and process.poll() is None:
        try:
            if requests.get(f"{base}/health", timeout=1).status_code == 200:
                started = True
                break
        except requests.RequestException:
            pass
        time.sleep(0.25)
    if not started:
        _kill(process)
        pytest.fail(f"server failed to start:\n{_log_text(log)[-4000:]}")
    try:
        yield process, base, log
    finally:
        _kill(process)
        log.close()


def _kill(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.kill()
        process.wait(timeout=10)


def _log_text(log) -> str:
    log.seek(0)
    return log.read().decode("utf-8", "replace")


def _post(base, path, body):
    return requests.post(f"{base}{path}", json=body, timeout=300)


def _create(base) -> str:
    response = _post(base, "/sessions", {})
    assert response.status_code == 200, response.text
    return response.json()["session_id"]


def _signal_and_wait(process: subprocess.Popen, limit_s: float) -> float:
    """Send Ctrl-Break and return the seconds until the process exited."""
    t0 = time.perf_counter()
    process.send_signal(signal.CTRL_BREAK_EVENT)
    process.wait(timeout=limit_s)
    return time.perf_counter() - t0


def test_shutdown_with_running_and_queued_readout_drains_and_exits_cleanly():
    with _server() as (process, base, log):
        base_sid = _create(base)
        assert _post(base, f"/sessions/{base_sid}/inject", {"text": BASE_TEXT}).status_code == 200
        other = _create(base)
        results: dict = {}

        def normal_inject():
            try:
                r = _post(base, f"/sessions/{other}/inject", {"text": LONG_TEXT})
                results["normal"] = (r.status_code, r.text[:300])
            except Exception as exc:  # noqa: BLE001 - recorded, asserted below
                results["normal"] = ("error", repr(exc))

        def priority_readout():
            try:
                r = _post(
                    base,
                    f"/sessions/{base_sid}/inject",
                    {
                        "text": LONG_SUFFIX,
                        "readout": {"candidates": GATE_CANDIDATES, "top_k": GATE_TOP_K},
                        "priority": True,
                    },
                )
                results["readout"] = (r.status_code, r.json() if r.status_code == 200 else r.text)
            except Exception as exc:  # noqa: BLE001 - recorded, asserted below
                results["readout"] = ("error", repr(exc))

        t_normal = threading.Thread(target=normal_inject)
        t_readout = threading.Thread(target=priority_readout)
        t_normal.start()
        time.sleep(0.15)  # the normal inject is executing
        t_readout.start()
        time.sleep(0.1)  # the readout is queued behind it

        elapsed = _signal_and_wait(process, SHUTDOWN_WITH_WORK_S)
        t_normal.join(30)
        t_readout.join(30)
        text = _log_text(log)

        assert process.returncode == 0, (process.returncode, text[-2000:])
        assert "shutting down: requests drained" in text, text[-2000:]
        assert results.get("normal", (None,))[0] == 200, results
        assert results.get("readout", (None,))[0] == 200, results
        readout = results["readout"][1]["readout"]
        assert all(abs(c["logprob"]) < float("inf") for c in readout["candidates"])
        assert 0.0 <= readout["candidate_mass"] <= 1.0
        print(f"\n[shutdown] with queued work: exited 0 after {elapsed:.2f} s")
        assert elapsed <= SHUTDOWN_WITH_WORK_S
        # Nothing answers any more.
        with pytest.raises(requests.RequestException):
            requests.get(f"{base}/health", timeout=1)


def test_shutdown_when_idle_exits_quickly():
    with _server() as (process, base, log):
        assert requests.get(f"{base}/health", timeout=5).status_code == 200
        elapsed = _signal_and_wait(process, SHUTDOWN_IDLE_S)
        assert process.returncode == 0, (process.returncode, _log_text(log)[-2000:])
        print(f"\n[shutdown] idle: exited 0 after {elapsed:.2f} s")
        assert elapsed <= SHUTDOWN_IDLE_S
