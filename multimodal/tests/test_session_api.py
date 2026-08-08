"""Behavior tests for the multimodal-server session API (slice 1: text-only).

Exercises the real HTTP contract end-to-end: create -> inject -> generate ->
delete. Also checks the 404 paths and that inject advances the KV cache.

Inject latency and tok/s are printed (captured into the test report) but NOT
asserted — budgets get pinned in slice 2 once we know steady-state numbers.
"""

from __future__ import annotations

import pytest
import requests

pytestmark = pytest.mark.usefixtures("base", "make_session")


def _leading_zero_alias(sid: str) -> str:
    prefix, number = sid.split("_", maxsplit=1)
    return f"{prefix}_0{number}"


def _request_session_operation(base: str, sid: str, operation: str):
    if operation == "status":
        return requests.get(f"{base}/sessions/{sid}", timeout=30)
    if operation == "delete":
        return requests.delete(f"{base}/sessions/{sid}", timeout=30)
    body = None
    if operation == "inject":
        body = {"text": "must not reach the canonical session"}
    elif operation == "generate":
        body = {"max_tokens": 1, "temperature": 0.0}
    return requests.post(
        f"{base}/sessions/{sid}/{operation}",
        json=body,
        timeout=120,
    )


MALFORMED_SESSION_IDS = [
    pytest.param(lambda number: f"s_0{number}", id="leading-zero"),
    pytest.param(lambda number: f"s_+{number}", id="plus-sign"),
    pytest.param(lambda number: f"s_ {number}", id="whitespace"),
    pytest.param(lambda number: f"s_{number}junk", id="suffix"),
    pytest.param(lambda _number: "s_9223372036854775808", id="overflow"),
]


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


@pytest.mark.parametrize(
    "operation",
    ["status", "fork", "inject", "generate", "cancel", "offload", "load", "delete"],
)
def test_leading_zero_session_id_cannot_alias_any_session_route(base, make_session, operation):
    sid = make_session()
    alias = _leading_zero_alias(sid)
    before = requests.get(f"{base}/sessions/{sid}", timeout=30).json()

    response = _request_session_operation(base, alias, operation)
    if operation == "fork" and response.status_code == 200:
        requests.delete(
            f"{base}/sessions/{response.json()['session_id']}",
            timeout=30,
        )

    assert response.status_code == 404, (operation, response.status_code, response.text)
    canonical = requests.get(f"{base}/sessions/{sid}", timeout=30)
    assert canonical.status_code == 200, (operation, canonical.text)
    assert canonical.json() == before, (operation, canonical.json(), before)


@pytest.mark.parametrize("malformed_id", MALFORMED_SESSION_IDS)
def test_malformed_session_id_spellings_cannot_read_canonical_session(
    base, make_session, malformed_id
):
    sid = make_session()
    number = sid.removeprefix("s_")
    alias = malformed_id(number)
    before = requests.get(f"{base}/sessions/{sid}", timeout=30).json()

    response = requests.get(f"{base}/sessions/{alias}", timeout=30)

    assert response.status_code == 404, (alias, response.status_code, response.text)
    assert requests.get(f"{base}/sessions/{sid}", timeout=30).json() == before


def test_malformed_session_id_cannot_delete_canonical_session(base, make_session):
    sid = make_session()
    before = requests.get(f"{base}/sessions/{sid}", timeout=30).json()

    response = requests.delete(f"{base}/sessions/{sid}junk", timeout=30)

    assert response.status_code == 404, response.text
    canonical = requests.get(f"{base}/sessions/{sid}", timeout=30)
    assert canonical.status_code == 200, canonical.text
    assert canonical.json() == before


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


# ------------------------------- chat template ------------------------------


def test_inject_messages_applies_chat_template(base, make_session):
    """Injecting {messages} should apply the model's chat template and produce
    coherent output (not raw control-token soup)."""
    sid = make_session()
    r = requests.post(
        f"{base}/sessions/{sid}/inject",
        json={"messages": [{"role": "user", "content": "What is 2+2? Reply with the number."}]},
        timeout=60,
    )
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["chat_template_applied"] is True
    assert j["tokens_injected"] > 0
    # Generate and check the model produces a coherent response containing "4"
    g = requests.post(
        f"{base}/sessions/{sid}/generate",
        json={"max_tokens": 80, "temperature": 0.0},
        timeout=120,
    )
    assert g.status_code == 200
    text = g.json()["text"].lower()
    assert "4" in text, f"expected '4' in response: {text!r}"


# ------------------------------- generate -----------------------------------


def test_generate_returns_text(base, make_session):
    sid = make_session()
    # Use the model's chat template (messages) so it responds coherently.
    requests.post(
        f"{base}/sessions/{sid}/inject",
        json={"messages": [{"role": "user", "content": "What is 2+2? Reply with the number."}]},
        timeout=120,
    )
    r = requests.post(
        f"{base}/sessions/{sid}/generate",
        json={"max_tokens": 60, "temperature": 0.0},
        timeout=120,
    )
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["n_tokens"] > 0
    assert isinstance(j["text"], str) and len(j["text"]) > 0
    print(
        f"  [generate] {j['n_tokens']} tokens in {j['gen_ms']} ms "
        f"({j['tokens_per_s']:.1f} tok/s): {j['text']!r}"
    )


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


def test_generate_temp0_is_deterministic(base, make_session):
    """temperature=0 is greedy: two fresh sessions with the same prompt produce
    IDENTICAL output (argmax each step, no RNG). This is what makes fork/source
    parity assertions reliable."""
    prompt = [{"role": "user", "content": "List three colors, comma-separated."}]
    a = make_session()
    requests.post(f"{base}/sessions/{a}/inject", json={"messages": prompt}, timeout=60)
    ga = requests.post(
        f"{base}/sessions/{a}/generate", json={"max_tokens": 40, "temperature": 0.0}, timeout=120
    ).json()
    b = make_session()
    requests.post(f"{base}/sessions/{b}/inject", json={"messages": prompt}, timeout=60)
    gb = requests.post(
        f"{base}/sessions/{b}/generate", json={"max_tokens": 40, "temperature": 0.0}, timeout=120
    ).json()
    assert ga["text"] == gb["text"], (ga["text"], gb["text"])
    assert ga["n_tokens"] > 0


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
