"""Byte-fallback safety of the inject-with-readout response (owner 2026-10-19).

Each test proves one part of the contract:
1. A top-16 readout whose top tokens are raw byte-fallback pieces returns
   200 (not an empty 500), every entry carries raw ``bytes`` starting with
   its UTF-8 ``text`` prefix, at least one top entry is not valid UTF-8,
   and the KV mutation stands (cache_size grows by tokens_injected).
2. For plain text, every entry's ``bytes`` equals its ``text`` encoded.
3. With top_k 0, ``top`` is empty and candidates still carry ``bytes``.
"""

from __future__ import annotations

import pytest
from test_readout import (
    BASE_TEXT,
    GATE_CANDIDATES,
    SUFFIX,
    _create,
    _delete,
    _fork,
    _inject,
    _post,
    _readout,
    _status,
)

pytestmark = pytest.mark.usefixtures("base")

PROMPTS = [
    "<|turn>user\nContinue: \u0378",
    "<|turn>user\nContinue: \ud7ff\u0378",
    "<|turn>user\n\u0378\u0378",
]


@pytest.fixture(scope="module")
def base_session(base):
    sid = _create(base)
    _inject(base, sid, {"text": BASE_TEXT})
    yield sid
    _delete(base, sid)


def _check_entry(entry):
    raw = entry["bytes"]
    text_bytes = entry["text"].encode("utf-8")
    assert isinstance(raw, list)
    assert all(isinstance(b, int) and 0 <= b <= 255 for b in raw)
    assert len(raw) >= len(text_bytes)
    assert bytes(raw[: len(text_bytes)]) == text_bytes


def _has_bad_utf8(entries) -> bool:
    for entry in entries:
        try:
            bytes(entry["bytes"]).decode("utf-8")
        except UnicodeDecodeError:
            return True
    return False


def test_byte_fallback_top_k_is_json_safe_and_keeps_the_mutation(base, base_session):
    found = None
    for prompt in PROMPTS:
        fork = _fork(base, base_session)
        try:
            before = _status(base, fork)["cache_size"]
            resp = _post(
                base,
                f"/sessions/{fork}/inject",
                {
                    "text": prompt,
                    "readout": {"candidates": GATE_CANDIDATES, "top_k": 16},
                },
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            readout = body["readout"]
            for entry in readout["top"] + readout["candidates"]:
                _check_entry(entry)
            assert body["cache_size"] == before + body["tokens_injected"]
            assert _status(base, fork)["cache_size"] == body["cache_size"]
            if _has_bad_utf8(readout["top"]):
                found = prompt
                break
        finally:
            _delete(base, fork)
    assert found is not None, f"no byte-fallback top entry for {PROMPTS!r}"
    print(f"\n[bytes] byte token produced by prompt={found!r}")


def test_plain_text_top_k_bytes_match_text(base, base_session):
    fork = _fork(base, base_session)
    try:
        readout = _readout(base, fork, text=SUFFIX, top_k=8)["readout"]
    finally:
        _delete(base, fork)
    assert readout["top"]
    for entry in readout["top"] + readout["candidates"]:
        assert entry["bytes"] == list(entry["text"].encode("utf-8"))


def test_top_k_zero_has_no_top_entries_and_candidates_carry_bytes(base, base_session):
    fork = _fork(base, base_session)
    try:
        readout = _readout(base, fork, text=SUFFIX, candidates=GATE_CANDIDATES, top_k=0)["readout"]
    finally:
        _delete(base, fork)
    assert readout["top"] == []
    assert readout["candidates"]
    for entry in readout["candidates"]:
        assert entry["bytes"] == list(entry["text"].encode("utf-8"))
