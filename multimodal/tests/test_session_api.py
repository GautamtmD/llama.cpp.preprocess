"""Behavior tests for the multimodal-server session API (slice 1: text-only).

Exercises the real HTTP contract end-to-end: create -> inject -> generate ->
delete. Also checks the 404 paths and that inject advances the KV cache.

Inject latency and tok/s are printed (captured into the test report) but NOT
asserted — budgets get pinned in slice 2 once we know steady-state numbers.
"""

from __future__ import annotations

import json
import time

import pytest
import requests

pytestmark = pytest.mark.usefixtures("base", "make_session")


# ------------------------------- health -------------------------------------


def test_health(base):
    r = requests.get(f"{base}/health", timeout=5)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ------------------------------- create -------------------------------------


def test_create_session_returns_id(make_session):
    sid = make_session()
    assert isinstance(sid, str)
    assert sid.startswith("s_")


# ------------------------------- inject -------------------------------------


def test_inject_advances_cache(base, make_session):
    sid = make_session()
    body = {"text": "The quick brown fox jumps over the lazy dog."}
    r = requests.post(f"{base}/sessions/{sid}/inject", json=body, timeout=60)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["tokens_injected"] > 0
    assert j["cache_size"] == j["tokens_injected"]
    print(f"  [inject] {j['tokens_injected']} tokens in {j['inject_ms']} ms")

    # a second inject advances the cache further
    body2 = {"text": " Then it rests quietly under a tree."}
    r2 = requests.post(f"{base}/sessions/{sid}/inject", json=body2, timeout=60)
    assert r2.status_code == 200, r2.text
    j2 = r2.json()
    assert j2["cache_size"] > j["cache_size"]
    assert j2["tokens_injected"] > 0


def test_inject_missing_text_400(base, make_session):
    sid = make_session()
    r = requests.post(f"{base}/sessions/{sid}/inject", json={}, timeout=30)
    assert r.status_code == 400


def test_inject_unknown_session_404(base):
    r = requests.post(f"{base}/sessions/s_999999/inject", json={"text": "x"}, timeout=30)
    assert r.status_code == 404
    assert "error" in r.json()


# ------------------------------- generate -----------------------------------


def test_generate_returns_text(base, make_session):
    sid = make_session()
    # Gemma-4 is instruction-tuned: inject a chat-formatted prompt so it actually
    # responds (raw completion text yields garbage).
    prompt = (
        "<start_of_turn>user\n"
        "What is the capital of France? Reply with one word.\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    requests.post(
        f"{base}/sessions/{sid}/inject",
        json={"text": prompt},
        timeout=120,
    )
    r = requests.post(
        f"{base}/sessions/{sid}/generate",
        json={"max_tokens": 16, "temperature": 0.0},
        timeout=120,
    )
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["n_tokens"] > 0
    assert isinstance(j["text"], str) and len(j["text"]) > 0
    # The model should mention Paris in its reply (greedy, chat-formatted).
    print(f"  [generate] {j['n_tokens']} tokens in {j['gen_ms']} ms "
          f"({j['tokens_per_s']:.1f} tok/s): {j['text']!r}")


def test_generate_respects_max_tokens(base, make_session):
    sid = make_session()
    requests.post(
        f"{base}/sessions/{sid}/inject",
        json={"text": "Tell me a long story about"},
        timeout=60,
    )
    r = requests.post(
        f"{base}/sessions/{sid}/generate",
        json={"max_tokens": 5},
        timeout=60,
    )
    assert r.status_code == 200, r.text
    assert r.json()["n_tokens"] <= 5


def test_generate_unknown_session_404(base):
    r = requests.post(
        f"{base}/sessions/s_999999/generate",
        json={"max_tokens": 1},
        timeout=30,
    )
    assert r.status_code == 404


def test_generate_on_empty_session_still_works(base, make_session):
    """Generating without any inject should still produce tokens (from BOS)."""
    sid = make_session()
    r = requests.post(
        f"{base}/sessions/{sid}/generate",
        json={"max_tokens": 3, "temperature": 0.0},
        timeout=60,
    )
    assert r.status_code == 200, r.text
    assert r.json()["n_tokens"] >= 1


# ------------------------------- delete -------------------------------------


def test_delete_session(base, make_session):
    sid = make_session()
    r = requests.delete(f"{base}/sessions/{sid}", timeout=30)
    assert r.status_code == 200
    assert r.json()["deleted"] is True
    # now it's gone -> inject returns 404
    r2 = requests.post(f"{base}/sessions/{sid}/inject", json={"text": "x"}, timeout=30)
    assert r2.status_code == 404


def test_delete_unknown_session_404(base):
    r = requests.delete(f"{base}/sessions/s_999999", timeout=30)
    assert r.status_code == 404
