"""Behavior tests for streaming /generate (SSE).

Each token arrives as a separate ``data: {"type":"token",...}`` event; a final
``data: {"type":"done",...}`` event carries usage. We parse the SSE stream
manually (no extra deps).
"""

from __future__ import annotations

import base64
import json
import struct
from collections.abc import Iterator

import pytest
import requests

pytestmark = pytest.mark.usefixtures("base", "make_session")


def _parse_sse(response) -> Iterator[dict]:
    """Yield parsed JSON objects from an SSE text/event-stream response."""
    for line in response.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            yield json.loads(line[len("data: ") :])


def _inject_chat(base, sid, user_text):
    """Inject using the model's chat template (messages format)."""
    requests.post(
        f"{base}/sessions/{sid}/inject",
        json={"messages": [{"role": "user", "content": user_text}]},
        timeout=60,
    )


def test_stream_yields_token_events(base, make_session):
    sid = make_session()
    _inject_chat(base, sid, "Say hello.")
    r = requests.post(
        f"{base}/sessions/{sid}/generate",
        json={"stream": True, "max_tokens": 15, "temperature": 0.0},
        stream=True,
        timeout=120,
    )
    assert r.status_code == 200
    assert "text/event-stream" in r.headers.get("content-type", "")
    events = list(_parse_sse(r))
    tokens = [e for e in events if e.get("type") == "token"]
    dones = [e for e in events if e.get("type") == "done"]
    assert len(tokens) > 0, "no token events received"
    assert len(dones) == 1, f"expected 1 done event, got {len(dones)}"
    # each token event has a string 'token' and an int 'id'
    for t in tokens:
        assert isinstance(t["token"], str)
        assert isinstance(t["id"], int)
    # done event has usage
    d = dones[0]
    assert "n_tokens" in d
    assert "tokens_per_s" in d
    assert d["n_tokens"] == len(tokens)
    print(f"  [stream] {d['n_tokens']} tokens, {d['tokens_per_s']:.1f} tok/s")


def test_stream_text_concatenates(base, make_session):
    """Concatenating all token pieces should produce readable text."""
    sid = make_session()
    _inject_chat(base, sid, "Count from 1 to 5.")
    r = requests.post(
        f"{base}/sessions/{sid}/generate",
        json={"stream": True, "max_tokens": 30, "temperature": 0.0},
        stream=True,
        timeout=120,
    )
    events = list(_parse_sse(r))
    text = "".join(e["token"] for e in events if e.get("type") == "token")
    assert len(text) > 0
    # The streaming mechanics work regardless of what the model generates. We
    # just check the concatenated text is a non-empty string; model quality
    # (whether it actually counts 1-5) depends on chat-template formatting.
    assert isinstance(text, str)
    print(f"  [stream] text: {text!r}")


def test_streaming_stop_commits_only_the_completion_token(base, make_session):
    source = make_session()
    _inject_chat(base, source, "Count from 1 to 5.")
    forked = requests.post(f"{base}/sessions/{source}/fork", timeout=120)
    assert forked.status_code == 200, forked.text
    subject = forked.json()["session_id"]
    try:
        before = requests.get(f"{base}/sessions/{subject}", timeout=30).json()
        control_response = requests.post(
            f"{base}/sessions/{source}/generate",
            json={"max_tokens": 1, "temperature": 0.0},
            timeout=120,
        )
        assert control_response.status_code == 200, control_response.text
        control = control_response.json()
        assert control["n_tokens"] == 1, control
        assert control["text"], control

        stopped_response = requests.post(
            f"{base}/sessions/{subject}/generate",
            json={
                "stream": True,
                "max_tokens": 5,
                "temperature": 0.0,
                "stop": [control["text"]],
            },
            stream=True,
            timeout=120,
        )
        assert stopped_response.status_code == 200, stopped_response.text
        events = list(_parse_sse(stopped_response))
        assert not [event for event in events if event.get("type") == "token"], events
        done = [event for event in events if event.get("type") == "done"]
        assert len(done) == 1, events
        assert done[0]["n_tokens"] == 1, done[0]
        assert done[0]["cache_size"] == before["cache_size"] + 1, (done[0], before)

        control_next = requests.post(
            f"{base}/sessions/{source}/generate",
            json={"max_tokens": 5, "temperature": 0.0},
            timeout=120,
        ).json()
        subject_next = requests.post(
            f"{base}/sessions/{subject}/generate",
            json={"max_tokens": 5, "temperature": 0.0},
            timeout=120,
        ).json()
        assert subject_next["tokens"] == control_next["tokens"], (
            subject_next,
            control_next,
        )
        assert subject_next["cache_size"] == control_next["cache_size"]
    finally:
        requests.delete(f"{base}/sessions/{subject}", timeout=30)


@pytest.mark.requires("audio")
def test_stream_media_ending_returns_terminal_error(base, make_session):
    sid = make_session()
    pcm = struct.pack("<640f", *([0.0] * 640))
    injected = requests.post(
        f"{base}/sessions/{sid}/inject",
        json={"audio": base64.b64encode(pcm).decode("ascii")},
        timeout=60,
    )
    assert injected.status_code == 200, injected.text
    before = requests.get(f"{base}/sessions/{sid}", timeout=30).json()

    response = requests.post(
        f"{base}/sessions/{sid}/generate",
        json={"stream": True, "max_tokens": 1, "temperature": 0.0},
        stream=True,
        timeout=60,
    )
    assert response.status_code == 200, response.text
    assert "text/event-stream" in response.headers.get("content-type", "")
    events = list(_parse_sse(response))
    assert events == [
        {
            "type": "error",
            "error": "sequence ends in media embeddings; inject text before generation",
            "code": 500,
        }
    ]

    after = requests.get(f"{base}/sessions/{sid}", timeout=30).json()
    assert after["cache_size"] == before["cache_size"]
    assert after["boundary_token"] == before["boundary_token"] == -1

    non_streaming = requests.post(
        f"{base}/sessions/{sid}/generate",
        json={"max_tokens": 1, "temperature": 0.0},
        timeout=60,
    )
    assert non_streaming.status_code == 500, non_streaming.text
    assert non_streaming.json() == {
        "error": "sequence ends in media embeddings; inject text before generation",
        "code": 500,
    }


def test_stream_unknown_session_404(base):
    r = requests.post(
        f"{base}/sessions/s_999999/generate",
        json={"stream": True, "max_tokens": 1},
        stream=True,
        timeout=30,
    )
    assert r.status_code == 404


def test_stream_default_is_non_streaming(base, make_session):
    """Omitting 'stream' should return slice-1 JSON, not SSE."""
    sid = make_session()
    _inject_chat(base, sid, "Hi")
    r = requests.post(
        f"{base}/sessions/{sid}/generate",
        json={"max_tokens": 3, "temperature": 0.0},
        timeout=60,
    )
    assert r.status_code == 200
    assert "application/json" in r.headers.get("content-type", "")
    j = r.json()
    assert "text" in j and "tokens" in j  # slice-1 shape
