"""Behavior tests for the common_sampler migration + grammar + tool-calling.

Covers ADR 0005 (common_sampler), 0007 (grammar / response_format), and 0006
(tools / tool_choice) on the ``/generate`` endpoint. These run against a live
multimodal-server (see ``conftest.py`` for how to start one).

The model-dependent cases (JSON, tool calls) use greedy temperature (0.0) and
explicit prompts so the model's behaviour is deterministic.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator

import pytest
import requests

pytestmark = pytest.mark.usefixtures("base", "make_session")


# -------------------------------- helpers ----------------------------------


def _inject_chat(base, sid, user_text, tools=None, tool_choice=None):
    """Inject using the model's chat template (messages format)."""
    body = {"messages": [{"role": "user", "content": user_text}]}
    if tools is not None:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    r = requests.post(f"{base}/sessions/{sid}/inject", json=body, timeout=60)
    assert r.status_code == 200, r.text


def _generate(base, sid, **kw):
    body = {"max_tokens": kw.pop("max_tokens", 64), "temperature": kw.pop("temperature", 0.0)}
    body.update(kw)
    r = requests.post(f"{base}/sessions/{sid}/generate", json=body, timeout=180)
    return r


def _parse_sse(response) -> Iterator[dict]:
    """Yield parsed JSON objects from an SSE text/event-stream response."""
    for line in response.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            yield json.loads(line[len("data: ") :])


WEATHER_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name."},
                },
                "required": ["location"],
            },
        },
    }
]


# --------------------------- response_format -------------------------------


def test_response_format_json_object(base, make_session):
    """response_format json_object constrains output to a valid JSON object."""
    sid = make_session()
    _inject_chat(base, sid, 'Return a JSON object with a field "count" set to 5.')
    r = _generate(base, sid, response_format={"type": "json_object"}, max_tokens=64)
    assert r.status_code == 200, r.text
    data = json.loads(r.json()["text"])
    assert isinstance(data, dict)
    assert data.get("count") == 5, data


def test_response_format_json_schema(base, make_session):
    """response_format json_schema constrains output to the given schema."""
    sid = make_session()
    _inject_chat(base, sid, "Return the answer 42.")
    rf = {
        "type": "json_schema",
        "json_schema": {
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "integer"}},
                "required": ["answer"],
            }
        },
    }
    r = _generate(base, sid, response_format=rf, max_tokens=64)
    assert r.status_code == 200, r.text
    data = json.loads(r.json()["text"])
    assert data == {"answer": 42}, data


def test_response_format_text(base, make_session):
    """response_format {type: text} applies no constraint and returns 200."""
    sid = make_session()
    _inject_chat(base, sid, "Say hello.")
    r = _generate(base, sid, response_format={"type": "text"}, max_tokens=16)
    assert r.status_code == 200, r.text
    assert "tool_calls" not in r.json()


# ------------------------------- grammar -----------------------------------


def test_grammar_constraint(base, make_session):
    """A raw GBNF grammar forces exactly the constrained shape."""
    sid = make_session()
    _inject_chat(base, sid, "Write some text.")
    gbnf = "root ::= [0-9] [0-9] [0-9] [0-9] [0-9]"
    r = _generate(base, sid, grammar=gbnf, max_tokens=16)
    assert r.status_code == 200, r.text
    text = r.json()["text"]
    assert re.fullmatch(r"\s*\d{5}\s*", text), repr(text)


# --------------------------- validation (400) ------------------------------


def test_response_format_and_grammar_conflict_400(base, make_session):
    """Both response_format and grammar -> 400."""
    sid = make_session()
    _inject_chat(base, sid, "hello")
    r = _generate(
        base,
        sid,
        response_format={"type": "json_object"},
        grammar='root ::= "x"',
        max_tokens=4,
    )
    assert r.status_code == 400, r.text


def test_invalid_response_format_type_400(base, make_session):
    """An invalid response_format.type -> 400."""
    sid = make_session()
    _inject_chat(base, sid, "hello")
    r = _generate(base, sid, response_format={"type": "yaml"}, max_tokens=4)
    assert r.status_code == 400, r.text


# ------------------------------ tool calling -------------------------------


def test_tool_calling_required(base, make_session):
    """tool_choice=required forces a tool call parsed into tool_calls."""
    sid = make_session()
    _inject_chat(
        base,
        sid,
        "What is the weather in Paris?",
        tools=WEATHER_TOOL,
        tool_choice="required",
    )
    r = _generate(
        base,
        sid,
        tools=WEATHER_TOOL,
        tool_choice="required",
        max_tokens=128,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("tool_calls"), f"no tool_calls: {body}"
    tc = body["tool_calls"][0]
    assert tc["function"]["name"] == "get_weather", tc
    args = json.loads(tc["function"]["arguments"])
    assert "location" in args, args
    assert "paris" in args["location"].lower(), args


def test_tool_calling_streaming(base, make_session):
    """Streaming tool_choice=required emits tool_call SSE events whose argument
    deltas concatenate to valid JSON."""
    sid = make_session()
    _inject_chat(
        base,
        sid,
        "What is the weather in Tokyo?",
        tools=WEATHER_TOOL,
        tool_choice="required",
    )
    r = requests.post(
        f"{base}/sessions/{sid}/generate",
        json={
            "stream": True,
            "temperature": 0.0,
            "max_tokens": 128,
            "tools": WEATHER_TOOL,
            "tool_choice": "required",
        },
        stream=True,
        timeout=180,
    )
    assert r.status_code == 200, r.text
    assert "text/event-stream" in r.headers.get("content-type", "")
    events = list(_parse_sse(r))
    tool_events = [e for e in events if e.get("type") == "tool_call"]
    assert tool_events, f"no tool_call events: {events}"
    # concatenate argument deltas per tool-call index
    by_index: dict[int, str] = {}
    names: dict[int, str] = {}
    for e in tool_events:
        tc = e["tool_call"]
        idx = tc.get("index", 0)
        if tc.get("name"):
            names[idx] = tc["name"]
        if tc.get("arguments"):
            by_index.setdefault(idx, "")
            by_index[idx] += tc["arguments"]
    assert names, names
    assert "get_weather" in names.values(), names
    # the first tool call's arguments must parse to valid JSON with location
    idx0 = sorted(by_index)[0]
    args = json.loads(by_index[idx0])
    assert "tokyo" in args["location"].lower(), args


def test_tool_choice_none(base, make_session):
    """tool_choice=none with tools present -> plain text, no tool_calls."""
    sid = make_session()
    _inject_chat(base, sid, "Say hello.", tools=WEATHER_TOOL, tool_choice="none")
    r = _generate(base, sid, tools=WEATHER_TOOL, tool_choice="none", max_tokens=32)
    assert r.status_code == 200, r.text
    body = r.json()
    assert not body.get("tool_calls"), body
    assert body.get("text", "").strip(), body


# --------------------------- sampling / decoding ---------------------------


def test_greedy_determinism_and_fork_parity(base, make_session):
    """Greedy (temp=0) is deterministic and a fork reproduces the source's exact
    token stream — the invariant the common_sampler migration must preserve."""
    sid = make_session()
    _inject_chat(base, sid, "Count from 1 to 5.")
    fork = requests.post(f"{base}/sessions/{sid}/fork", timeout=120)
    assert fork.status_code == 200, fork.text
    fsid = fork.json()["session_id"]

    g_src = _generate(base, sid, max_tokens=20)
    g_fk = _generate(base, fsid, max_tokens=20)
    assert g_src.status_code == 200 and g_fk.status_code == 200
    assert g_src.json()["tokens"] == g_fk.json()["tokens"], (
        g_src.json()["text"],
        g_fk.json()["text"],
    )
    assert g_src.json()["text"] == g_fk.json()["text"]


def test_stop_sequences(base, make_session):
    """A stop sequence truncates output; without it the sequence can appear."""
    sid = make_session()
    _inject_chat(base, sid, "Write a short sentence that contains the word banana.")
    r = _generate(base, sid, stop=["banana"], max_tokens=48)
    assert r.status_code == 200, r.text
    assert "banana" not in r.json()["text"], r.json()["text"]

    sid2 = make_session()
    _inject_chat(base, sid2, "Write a short sentence that contains the word banana.")
    r2 = _generate(base, sid2, max_tokens=48)
    assert r2.status_code == 200, r2.text
    assert "banana" in r2.json()["text"].lower(), r2.json()["text"]


def test_ignore_eos(base, make_session):
    """ignore_eos forces generation to run to exactly max_tokens."""
    sid = make_session()
    _inject_chat(base, sid, "Say hi.")
    r = _generate(base, sid, ignore_eos=True, max_tokens=30, temperature=0.0)
    assert r.status_code == 200, r.text
    assert r.json()["n_tokens"] == 30, r.json()
