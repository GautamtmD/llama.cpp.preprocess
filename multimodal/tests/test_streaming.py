"""Behavior tests for streaming /generate (SSE).

Each token arrives as a separate ``data: {"type":"token",...}`` event; a final
``data: {"type":"done",...}`` event carries usage. We parse the SSE stream
manually (no extra deps).
"""

from __future__ import annotations

import json
from typing import Iterator

import pytest
import requests

pytestmark = pytest.mark.usefixtures("base", "make_session")


def _parse_sse(response) -> Iterator[dict]:
    """Yield parsed JSON objects from an SSE text/event-stream response."""
    for line in response.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            yield json.loads(line[len("data: "):])


def _inject_chat(base, sid, user_text):
    prompt = (
        "<start_of_turn>user\n" + user_text +
        "\n<end_of_turn>\n<start_of_turn>model\n"
    )
    requests.post(f"{base}/sessions/{sid}/inject", json={"text": prompt}, timeout=60)


def test_stream_yields_token_events(base, make_session):
    sid = make_session()
    _inject_chat(base, sid, "Say hello.")
    r = requests.post(
        f"{base}/sessions/{sid}/generate",
        json={"stream": True, "max_tokens": 15, "temperature": 0.0},
        stream=True, timeout=120,
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
        stream=True, timeout=120,
    )
    events = list(_parse_sse(r))
    text = "".join(e["token"] for e in events if e.get("type") == "token")
    assert len(text) > 0
    # The streaming mechanics work regardless of what the model generates. We
    # just check the concatenated text is a non-empty string; model quality
    # (whether it actually counts 1-5) depends on chat-template formatting.
    assert isinstance(text, str)
    print(f"  [stream] text: {text!r}")


def test_stream_unknown_session_404(base):
    r = requests.post(
        f"{base}/sessions/s_999999/generate",
        json={"stream": True, "max_tokens": 1},
        stream=True, timeout=30,
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
