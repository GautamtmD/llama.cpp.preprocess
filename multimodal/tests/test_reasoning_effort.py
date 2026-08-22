"""Reasoning-effort request validation, common-sampler behavior, and latency gates."""

from __future__ import annotations

import json
import os
import statistics
import threading
import time

import pytest
import requests

START_MARKER = "<|channel>thought"
END_MARKER = "<channel|>"
PERF_PROMPT = (
    "Calculate 127 multiplied by 43. Explain the calculation in one short sentence, "
    "then state the result."
)


def _inject_chat(base: str, sid: str, prompt: str) -> None:
    response = requests.post(
        f"{base}/sessions/{sid}/inject",
        json={"messages": [{"role": "user", "content": prompt}]},
        timeout=120,
    )
    assert response.status_code == 200, response.text


def _fork(base: str, sid: str) -> str:
    response = requests.post(f"{base}/sessions/{sid}/fork", timeout=120)
    assert response.status_code == 200, response.text
    return response.json()["session_id"]


def _delete(base: str, *session_ids: str) -> None:
    for sid in session_ids:
        response = requests.delete(f"{base}/sessions/{sid}", timeout=30)
        assert response.status_code == 200, response.text


def _thought_payload(text: str) -> str | None:
    start = text.find(START_MARKER)
    if start < 0:
        return None
    start += len(START_MARKER)
    end = text.find(END_MARKER, start)
    if end < 0:
        return None
    return text[start:end]


def _answer_text(text: str) -> str:
    boundary = text.find(END_MARKER)
    if boundary < 0:
        return ""
    answer = text[boundary + len(END_MARKER) :].strip()
    if answer.startswith("answer"):
        answer = answer[len("answer") :].lstrip(" :\n")
    return answer


def _stream_generation(base: str, sid: str, *, effort: str | None) -> dict:
    body: dict[str, object] = {
        "stream": True,
        "max_tokens": 384,
        "temperature": 0.0,
    }
    if effort is not None:
        body["reasoning_effort"] = effort

    started = time.perf_counter()
    raw = ""
    token_events = 0
    answer_onset_ms: float | None = None
    start_marker_event: int | None = None
    end_marker_event: int | None = None
    answer_onset_event: int | None = None
    terminal: dict | None = None
    with requests.post(
        f"{base}/sessions/{sid}/generate",
        json=body,
        stream=True,
        timeout=240,
    ) as response:
        assert response.status_code == 200, response.text
        assert response.headers["Content-Type"].startswith("text/event-stream")
        for line in response.iter_lines(decode_unicode=True, chunk_size=1):
            if not line or not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event["type"] == "token":
                token_events += 1
                raw += event["token"]
                if start_marker_event is None and START_MARKER in raw:
                    start_marker_event = token_events
                if end_marker_event is None and END_MARKER in raw:
                    end_marker_event = token_events
                if answer_onset_ms is None and _answer_text(raw):
                    answer_onset_ms = (time.perf_counter() - started) * 1000.0
                    answer_onset_event = token_events
            elif event["type"] in {"done", "error"}:
                terminal = event
                break

    assert terminal is not None
    assert terminal["type"] == "done", terminal
    assert answer_onset_ms is not None, raw
    return {
        "raw": raw,
        "answer": _answer_text(raw),
        "thought": _thought_payload(raw),
        "answer_onset_ms": answer_onset_ms,
        "token_events": token_events,
        "n_tokens": terminal["n_tokens"],
        "cache_size": terminal["cache_size"],
        "start_marker_event": start_marker_event,
        "end_marker_event": end_marker_event,
        "answer_onset_event": answer_onset_event,
    }


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("value", [None, 7, "extreme"])
def test_invalid_reasoning_effort_is_json_400_and_session_reusable(
    base, make_session, stream, value
):
    sid = make_session()
    _inject_chat(base, sid, "Reply with the word reusable.")
    before = requests.get(f"{base}/sessions/{sid}", timeout=30).json()

    response = requests.post(
        f"{base}/sessions/{sid}/generate",
        json={"stream": stream, "reasoning_effort": value},
        stream=stream,
        timeout=60,
    )
    assert response.status_code == 400
    assert response.headers["Content-Type"].startswith("application/json")
    assert response.json()["code"] == 400

    after = requests.get(f"{base}/sessions/{sid}", timeout=30).json()
    assert after["cache_size"] == before["cache_size"]
    assert after["boundary_token"] == before["boundary_token"]
    retry = requests.post(
        f"{base}/sessions/{sid}/generate",
        json={"max_tokens": 1, "temperature": 0.0},
        timeout=60,
    )
    assert retry.status_code == 200, retry.text


def test_none_json_closes_reasoning_without_payload_and_continues_answer(base, make_session):
    sid = make_session()
    _inject_chat(base, sid, PERF_PROMPT)
    response = requests.post(
        f"{base}/sessions/{sid}/generate",
        json={"reasoning_effort": "none", "max_tokens": 64, "temperature": 0.0},
        timeout=120,
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert _thought_payload(data["text"]) == ""
    assert _answer_text(data["text"])
    assert data["n_tokens"] > 0

    continuation = requests.post(
        f"{base}/sessions/{sid}/generate",
        json={"reasoning_effort": "none", "max_tokens": 1, "temperature": 0.0},
        timeout=60,
    )
    assert continuation.status_code == 200, continuation.text


def test_none_survives_one_token_requests_fork_and_offload(base, make_session):
    sid = make_session()
    _inject_chat(base, sid, PERF_PROMPT)

    first = requests.post(
        f"{base}/sessions/{sid}/generate",
        json={"reasoning_effort": "none", "max_tokens": 1, "temperature": 0.0},
        timeout=60,
    )
    assert first.status_code == 200, first.text
    prefix = first.json()["text"]
    fork = _fork(base, sid)
    offload = requests.post(f"{base}/sessions/{fork}/offload", timeout=120)
    assert offload.status_code == 200, offload.text
    load = requests.post(f"{base}/sessions/{fork}/load", timeout=120)
    assert load.status_code == 200, load.text

    outputs = []
    continuation_ms = []
    for target in (sid, fork):
        raw = prefix
        started = time.perf_counter()
        for _ in range(15):
            if _answer_text(raw):
                break
            chunk = requests.post(
                f"{base}/sessions/{target}/generate",
                json={
                    "reasoning_effort": "none",
                    "max_tokens": 1,
                    "temperature": 0.0,
                },
                timeout=60,
            )
            assert chunk.status_code == 200, chunk.text
            raw += chunk.json()["text"]
        assert _thought_payload(raw) == ""
        assert _answer_text(raw), raw
        outputs.append(raw)
        continuation_ms.append((time.perf_counter() - started) * 1000.0)

    assert outputs[0] == outputs[1]
    assert max(continuation_ms) < 2000.0
    print(f"one-token none continuation_ms={continuation_ms}")


def test_minimal_budget_counts_across_one_token_requests(base, make_session):
    sid = make_session()
    _inject_chat(base, sid, PERF_PROMPT)

    raw = ""
    for _ in range(96):
        chunk = requests.post(
            f"{base}/sessions/{sid}/generate",
            json={
                "reasoning_effort": "minimal",
                "max_tokens": 1,
                "temperature": 0.0,
            },
            timeout=60,
        )
        assert chunk.status_code == 200, chunk.text
        raw += chunk.json()["text"]
        if _answer_text(raw):
            break

    thought = _thought_payload(raw)
    assert thought is not None and thought.strip()
    assert _answer_text(raw), raw


def test_none_sse_closes_reasoning_and_completes(base, make_session):
    sid = make_session()
    _inject_chat(base, sid, PERF_PROMPT)
    result = _stream_generation(base, sid, effort="none")
    assert result["thought"] == ""
    assert result["answer"]


def test_none_cancellation_rewinds_and_session_continues(base, make_session):
    sid = make_session()
    _inject_chat(base, sid, PERF_PROMPT)
    prefix_response = requests.post(
        f"{base}/sessions/{sid}/generate",
        json={"reasoning_effort": "none", "max_tokens": 1, "temperature": 0.0},
        timeout=60,
    )
    assert prefix_response.status_code == 200, prefix_response.text
    prefix = prefix_response.json()["text"]
    checkpoint = requests.get(f"{base}/sessions/{sid}", timeout=30).json()
    fourth_token = threading.Event()
    finished = threading.Event()
    terminal: dict[str, object] = {}

    def generate() -> None:
        try:
            with requests.post(
                f"{base}/sessions/{sid}/generate",
                json={
                    "stream": True,
                    "reasoning_effort": "none",
                    "max_tokens": 384,
                    "temperature": 0.0,
                    "ignore_eos": True,
                },
                stream=True,
                timeout=180,
            ) as response:
                assert response.status_code == 200, response.text
                token_count = 0
                for line in response.iter_lines(decode_unicode=True, chunk_size=1):
                    if not line or not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    if event["type"] == "token":
                        token_count += 1
                        if token_count == 4:
                            fourth_token.set()
                    elif event["type"] in {"done", "error"}:
                        terminal.update(event)
                        break
        finally:
            finished.set()

    worker = threading.Thread(target=generate, daemon=True)
    worker.start()
    assert fourth_token.wait(60), "reasoning-none stream did not reach answer onset"
    cancelled = requests.post(f"{base}/sessions/{sid}/cancel", timeout=30)
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["cancelled"] is True
    assert finished.wait(60), "cancelled reasoning stream did not finish"
    worker.join(timeout=1)
    assert terminal.get("type") == "done", terminal

    restored = requests.get(f"{base}/sessions/{sid}", timeout=30).json()
    assert restored["cache_size"] == checkpoint["cache_size"]
    assert restored["boundary_token"] == checkpoint["boundary_token"]
    retry = requests.post(
        f"{base}/sessions/{sid}/generate",
        json={"reasoning_effort": "none", "max_tokens": 8, "temperature": 0.0},
        timeout=120,
    )
    assert retry.status_code == 200, retry.text
    accumulated = prefix + retry.json()["text"]
    assert _thought_payload(accumulated) == ""
    assert _answer_text(accumulated)


def test_none_composes_with_nonlazy_grammar(base, make_session):
    sid = make_session()
    _inject_chat(base, sid, "Output OK.")
    response = requests.post(
        f"{base}/sessions/{sid}/generate",
        json={
            "reasoning_effort": "none",
            "temperature": 0.0,
            "max_tokens": 8,
            "grammar": 'root ::= "OK"',
        },
        timeout=120,
    )
    assert response.status_code == 200, response.text
    assert response.json()["text"] == "OK"


@pytest.mark.reasoning_perf
@pytest.mark.skipif(
    os.environ.get("MULTIMODAL_REASONING_PERF") != "1",
    reason="set MULTIMODAL_REASONING_PERF=1 for the paired Gemma reasoning gate",
)
def test_none_real_gemma_answer_onset_is_twice_as_fast(base, make_session):
    # Warm the model and prove this fixed prompt currently exercises real reasoning.
    warm_root = make_session()
    _inject_chat(base, warm_root, PERF_PROMPT)
    warm_default_sid = _fork(base, warm_root)
    warm_none_sid = _fork(base, warm_root)
    warm_default = _stream_generation(base, warm_default_sid, effort=None)
    warm_none = _stream_generation(base, warm_none_sid, effort="none")
    assert warm_default["thought"] is not None and warm_default["thought"].strip()
    assert warm_none["thought"] == ""
    assert warm_none["start_marker_event"] is not None
    assert warm_none["end_marker_event"] is not None
    assert warm_none["answer_onset_event"] is not None
    assert warm_none["answer_onset_event"] - warm_none["end_marker_event"] <= 1
    _delete(base, warm_root, warm_default_sid, warm_none_sid)

    default_trials: list[float] = []
    none_trials: list[float] = []
    none_token_counts: list[int] = []
    default_token_counts: list[int] = []
    none_answer_onset_events: list[int] = []
    for trial in range(5):
        root = make_session()
        _inject_chat(base, root, PERF_PROMPT)
        default_sid = _fork(base, root)
        none_sid = _fork(base, root)
        order = [(default_sid, None), (none_sid, "none")]
        if trial % 2:
            order.reverse()
        results = {}
        for sid, effort in order:
            results[effort or "default"] = _stream_generation(base, sid, effort=effort)

        default = results["default"]
        none = results["none"]
        assert default["thought"] is not None and default["thought"].strip()
        assert none["thought"] == ""
        assert none["answer"]
        default_trials.append(default["answer_onset_ms"])
        none_trials.append(none["answer_onset_ms"])
        none_token_counts.append(none["n_tokens"])
        assert none["answer_onset_event"] - none["end_marker_event"] <= 1
        default_token_counts.append(default["n_tokens"])
        none_answer_onset_events.append(none["answer_onset_event"])

        reuse = requests.post(
            f"{base}/sessions/{none_sid}/generate",
            json={"reasoning_effort": "none", "max_tokens": 1, "temperature": 0.0},
            timeout=60,
        )
        assert reuse.status_code == 200, reuse.text
        _delete(base, root, default_sid, none_sid)

    default_median = statistics.median(default_trials)
    none_median = statistics.median(none_trials)
    print(
        "reasoning effort gate: "
        f"default_ms={default_trials} none_ms={none_trials} "
        f"default_median_ms={default_median:.3f} none_median_ms={none_median:.3f} "
        f"speedup={default_median / none_median:.3f}x "
        f"default_n_tokens={default_token_counts} "
        f"none_n_tokens={none_token_counts} "
        f"none_answer_onset_events={none_answer_onset_events}"
    )
    assert default_median >= 2.0 * none_median
