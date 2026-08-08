"""Behavior tests for POST /sessions/{id}/cancel (EUS-3).

Run against a live server built from this branch.  Cancellation is deliberately
exercised from a second HTTP client while /generate owns the session context;
the cancel endpoint only signals an atomic flag, and the generation thread owns
the KV rewind.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator

import pytest
import requests

pytestmark = pytest.mark.usefixtures("base", "make_session")

# EUS-3's token loop observes cancellation within one decode step. Returning a
# fully reusable session additionally re-decodes the preserved boundary token to
# refresh logits, so clean completion has one extra decode-step budget.
TRANSPORT_ALLOWANCE_MS = 250
CLEAN_REUSE_DECODE_STEPS = 2


def _parse_sse(response) -> Iterator[dict]:
    for line in response.iter_lines(chunk_size=1, decode_unicode=True):
        if line and line.startswith("data: "):
            yield json.loads(line[len("data: ") :])


def _inject(base: str, sid: str, text: str) -> dict:
    response = requests.post(f"{base}/sessions/{sid}/inject", json={"text": text}, timeout=60)
    assert response.status_code == 200, response.text
    return response.json()


def _generate(base: str, sid: str, **body) -> dict:
    request = {"max_tokens": 64, "temperature": 0.0}
    request.update(body)
    response = requests.post(f"{base}/sessions/{sid}/generate", json=request, timeout=120)
    assert response.status_code == 200, response.text
    return response.json()


def _fork(base: str, sid: str) -> dict:
    response = requests.post(f"{base}/sessions/{sid}/fork", timeout=120)
    assert response.status_code == 200, response.text
    return response.json()


def _cancel(base: str, sid: str) -> dict:
    response = requests.post(f"{base}/sessions/{sid}/cancel", timeout=30)
    assert response.status_code == 200, response.text
    return response.json()


def _delete(base: str, sid: str) -> None:
    response = requests.delete(f"{base}/sessions/{sid}", timeout=30)
    assert response.status_code == 200, response.text


def _assert_empty_checkpoint(base: str, sid: str) -> None:
    response = requests.get(f"{base}/sessions/{sid}", timeout=30)
    assert response.status_code == 200, response.text
    state = response.json()
    assert state["cache_size"] == 0, state
    assert state["boundary_token"] == -1, state


def _assert_empty_reusable(base: str, sid: str) -> None:
    """Retry generation from the restored empty checkpoint."""
    retry = _generate(base, sid, max_tokens=16)
    assert retry["n_tokens"] > 0, retry


def _assert_clean_and_reusable(base: str, sid: str, cache_before: int, control_sid: str) -> None:
    """Compare cancelled state to an untouched control and a post-cancel fork."""
    post_cancel = _fork(base, sid)
    try:
        assert post_cancel["cache_size"] == cache_before, post_cancel

        source = _generate(base, sid, max_tokens=16)
        control = _generate(base, control_sid, max_tokens=16)
        forked = _generate(base, post_cancel["session_id"], max_tokens=16)
        assert source["text"] == control["text"], (source, control)
        assert forked["text"] == control["text"], (forked, control)
    finally:
        _delete(base, post_cancel["session_id"])
        _delete(base, control_sid)


def test_cancel_without_active_generation_is_idempotent(base, make_session):
    sid = make_session()
    assert _cancel(base, sid) == {"session_id": sid, "cancelled": False}


def test_cancel_unknown_session_404(base):
    response = requests.post(f"{base}/sessions/s_999999/cancel", timeout=30)
    assert response.status_code == 404
    assert "error" in response.json()


@pytest.mark.parametrize("empty_session", [False, True], ids=["injected", "empty"])
def test_cancel_streaming_rewinds_cache_and_leaves_session_reusable(
    base, make_session, empty_session
):
    sid = make_session()
    if empty_session:
        before = 0
    else:
        before = _inject(base, sid, "Write a very long numbered list about the solar system.")[
            "cache_size"
        ]
        control_sid = _fork(base, sid)["session_id"]
    second_token = threading.Event()
    token_times: list[float] = []
    outcome: dict[str, object] = {}

    def generate() -> None:
        response = requests.post(
            f"{base}/sessions/{sid}/generate",
            json={"stream": True, "max_tokens": 256, "temperature": 0.0},
            stream=True,
            timeout=120,
        )
        assert response.status_code == 200, response.text
        events = []
        for event in _parse_sse(response):
            events.append(event)
            if event.get("type") == "token":
                token_times.append(time.perf_counter())
                if len(token_times) == 2:
                    second_token.set()
        outcome["events"] = events

    worker = threading.Thread(target=generate, daemon=True)
    worker.start()
    assert second_token.wait(30), "stream did not produce two tokens"

    started = time.perf_counter()
    assert _cancel(base, sid) == {"session_id": sid, "cancelled": True}
    worker.join(30)
    halted_ms = (time.perf_counter() - started) * 1_000
    assert not worker.is_alive(), "stream did not halt after cancellation"
    observed_step_ms = (token_times[1] - token_times[0]) * 1_000
    clean_completion_budget_ms = (
        CLEAN_REUSE_DECODE_STEPS * observed_step_ms + TRANSPORT_ALLOWANCE_MS
    )
    assert halted_ms <= clean_completion_budget_ms, (
        f"cancel-to-clean={halted_ms:.1f} ms exceeded cancellation + logits-refresh "
        f"budget ({CLEAN_REUSE_DECODE_STEPS} × {observed_step_ms:.1f} ms + "
        f"{TRANSPORT_ALLOWANCE_MS} ms)"
    )

    events = outcome["events"]
    assert isinstance(events, list)
    tokens = [event for event in events if event.get("type") == "token"]
    done = [event for event in events if event.get("type") == "done"]
    assert len(tokens) <= 3, tokens  # two observed before cancel + at most one in-flight step
    assert len(done) == 1, events
    print(
        f"  [cancel-stream] token-step={observed_step_ms:.1f} ms, "
        f"cancel-to-clean={halted_ms:.1f} ms"
    )

    if empty_session:
        _assert_empty_checkpoint(base, sid)
        _assert_empty_reusable(base, sid)
    else:
        _assert_clean_and_reusable(base, sid, before, control_sid)


def test_cancel_non_streaming_rewinds_cache_and_leaves_session_reusable(base, make_session):
    sid = make_session()
    before = _inject(base, sid, "Explain the history of the Roman Empire in exhaustive detail.")[
        "cache_size"
    ]
    control_sid = _fork(base, sid)["session_id"]
    timing_sid = _fork(base, sid)["session_id"]
    try:
        timing = _generate(base, timing_sid, max_tokens=2)
        assert timing["n_tokens"] > 0
        observed_step_ms = timing["gen_ms"] / timing["n_tokens"]
    finally:
        _delete(base, timing_sid)
    clean_completion_budget_ms = (
        CLEAN_REUSE_DECODE_STEPS * observed_step_ms + TRANSPORT_ALLOWANCE_MS
    )
    outcome: dict[str, dict] = {}

    def generate() -> None:
        outcome["response"] = _generate(base, sid, max_tokens=256)

    worker = threading.Thread(target=generate, daemon=True)
    worker.start()
    deadline = time.monotonic() + 30
    while True:
        started = time.perf_counter()
        cancel = _cancel(base, sid)
        if cancel["cancelled"]:
            break
        assert time.monotonic() < deadline, "generation never became active"
        time.sleep(0.01)
    worker.join(30)
    halted_ms = (time.perf_counter() - started) * 1_000
    assert not worker.is_alive(), "non-streaming generation did not halt"
    assert halted_ms <= clean_completion_budget_ms, (
        f"cancel-to-clean={halted_ms:.1f} ms exceeded cancellation + logits-refresh "
        f"budget ({CLEAN_REUSE_DECODE_STEPS} × {observed_step_ms:.1f} ms + "
        f"{TRANSPORT_ALLOWANCE_MS} ms)"
    )

    response = outcome["response"]
    assert response["n_tokens"] < 256
    assert response["n_tokens"] <= 1
    print(
        f"  [cancel-json] token-step={observed_step_ms:.1f} ms, cancel-to-clean={halted_ms:.1f} ms"
    )

    _assert_clean_and_reusable(base, sid, before, control_sid)


@pytest.mark.parametrize("empty_session", [False, True], ids=["injected", "empty"])
def test_stream_disconnect_rewinds_cache(base, make_session, empty_session):
    sid = make_session()
    if empty_session:
        before = 0
    else:
        before = _inject(base, sid, "List every ocean current in detail.")["cache_size"]
        control_sid = _fork(base, sid)["session_id"]
    second_token = threading.Event()
    token_times: list[float] = []

    def generate_then_disconnect() -> None:
        with requests.post(
            f"{base}/sessions/{sid}/generate",
            json={"stream": True, "max_tokens": 256, "temperature": 0.0},
            stream=True,
            headers={"Connection": "close"},
            timeout=120,
        ) as response:
            assert response.status_code == 200, response.text
            for event in _parse_sse(response):
                if event.get("type") == "token":
                    token_times.append(time.perf_counter())
                    if len(token_times) == 2:
                        second_token.set()
                        return

    worker = threading.Thread(target=generate_then_disconnect, daemon=True)
    worker.start()
    assert second_token.wait(30), "stream did not produce two tokens before disconnect"
    worker.join(5)
    assert not worker.is_alive()

    observed_step_ms = (token_times[1] - token_times[0]) * 1_000
    clean_completion_budget_ms = (
        CLEAN_REUSE_DECODE_STEPS * observed_step_ms + TRANSPORT_ALLOWANCE_MS
    )
    time.sleep(clean_completion_budget_ms / 1_000)
    cancel = _cancel(base, sid)
    assert cancel == {"session_id": sid, "cancelled": False}, (
        "generation remained active after client disconnect; the probe had to cancel it"
    )
    print(
        f"  [disconnect] token-step={observed_step_ms:.1f} ms, "
        f"inactive within {clean_completion_budget_ms:.1f} ms"
    )

    if empty_session:
        _assert_empty_checkpoint(base, sid)
        _assert_empty_reusable(base, sid)
    else:
        _assert_clean_and_reusable(base, sid, before, control_sid)
