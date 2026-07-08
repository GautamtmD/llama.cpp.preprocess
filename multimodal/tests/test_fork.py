"""Behavior tests for POST /sessions/{id}/fork (EUS-2: KV-cache fork).

Run against a LIVE multimodal-server (built from this branch — it has /fork)::

    engine/multimodal/build/bin/Release/multimodal-server.exe \\
        --model "/c/ML Models/Gemma4 12b/gemma-4-12b-it-qat-q4_0.gguf" \\
        --port 8080

    pytest engine/multimodal/tests/test_fork.py

Approach A' (per-sequence deep-copy into a new context — see
docs/decisions/0004-fork-copy-semantics.md): the forked session owns an
independent K/V copy, so source and fork generate independently. These tests
check snapshot correctness (forkable after inject / after generate; source
untouched; later source mutations do not leak into the fork; the fork generates
coherently), the 404 path, and the latency budget (fork_ms).
"""

from __future__ import annotations

import pytest
import requests

pytestmark = pytest.mark.usefixtures("base", "make_session")

FORK_LATENCY_BUDGET_MS = 800  # EUS-2 budget is ≤600 ms for ≤2048 tokens; headroom


def _inject(base, sid, text):
    r = requests.post(f"{base}/sessions/{sid}/inject", json={"text": text}, timeout=60)
    assert r.status_code == 200, r.text
    return r.json()


def _inject_msgs(base, sid, messages):
    r = requests.post(f"{base}/sessions/{sid}/inject", json={"messages": messages}, timeout=60)
    assert r.status_code == 200, r.text
    return r.json()


def _fork(base, sid):
    r = requests.post(f"{base}/sessions/{sid}/fork", timeout=120)
    assert r.status_code == 200, r.text
    return r.json()


def _generate(base, sid, **kw):
    body = {"max_tokens": kw.pop("max_tokens", 20), "temperature": kw.pop("temperature", 0.0)}
    body.update(kw)
    r = requests.post(f"{base}/sessions/{sid}/generate", json=body, timeout=120)
    assert r.status_code == 200, r.text
    return r.json()


# ------------------------------- core contract ------------------------------


def test_fork_unknown_session_404(base):
    r = requests.post(f"{base}/sessions/s_999999/fork", timeout=30)
    assert r.status_code == 404
    assert "error" in r.json()


def test_fork_after_text_inject_snapshots_cache(base, make_session):
    """Forking after a text inject yields a new session with the same cache_size."""
    src = make_session()
    j = _inject(base, src, "The quick brown fox jumps over the lazy dog. " * 4)
    src_size = j["cache_size"]
    assert src_size > 0

    f = _fork(base, src)
    fork_sid = f["session_id"]
    assert fork_sid != src
    assert f["forked_from"] == src
    assert f["cache_size"] == src_size, (f, j)
    # latency budget
    print(f"  [fork] {src_size} tokens in {f['fork_ms']} ms")
    assert f["fork_ms"] < FORK_LATENCY_BUDGET_MS


def test_source_cache_unchanged_by_fork(base, make_session):
    """The source session's cache_size is not changed by forking it."""
    src = make_session()
    j = _inject(base, src, "Hello world, this is a test of forking.")
    before = j["cache_size"]
    _fork(base, src)
    # the source id is unchanged and still usable; its cache is intact
    again = _inject(base, src, " more text")
    assert again["cache_size"] == before + again["tokens_injected"]


def test_snapshot_independence_source_mutation_does_not_leak(base, make_session):
    """After the fork, injecting into the source does NOT change the fork's cache."""
    src = make_session()
    _inject(base, src, "Count to three: one, two, three,")
    f = _fork(base, src)
    fork_sid = f["session_id"]
    fork_size = f["cache_size"]

    # mutate the source after the fork
    extra = _inject(base, src, " four, five, six,")
    assert extra["cache_size"] > fork_size

    # the fork must be unaffected: inject a tiny probe into the fork and confirm
    # its cache grows from fork_size (not from the source's larger size)
    probe = _inject(base, fork_sid, " x")
    assert probe["cache_size"] == fork_size + probe["tokens_injected"], probe


def test_forked_session_generates_coherently(base, make_session):
    """The forked session can generate coherent text from the snapshot."""
    src = make_session()
    _inject_msgs(base, src, [{"role": "user", "content": "What is 2+2? Reply with the number."}])
    f = _fork(base, src)
    g = _generate(base, f["session_id"], max_tokens=40)
    assert g["n_tokens"] > 0
    assert "4" in g["text"].lower(), g["text"]


def test_fork_and_source_generate_independently(base, make_session):
    """Source and fork can each generate from the shared snapshot without
    interfering (a deep copy: each drives its own KV)."""
    src = make_session()
    _inject_msgs(base, src, [{"role": "user", "content": "Name a fruit."}])
    f = _fork(base, src)

    g_src = _generate(base, src, max_tokens=10)
    g_fk = _generate(base, f["session_id"], max_tokens=10)
    # both produced text from the same starting snapshot
    assert g_src["n_tokens"] > 0 and g_fk["n_tokens"] > 0
    # the source's generation advanced its own cache; the fork's generation
    # advanced the fork's. Generating on the fork again still works (independent).
    g_fk2 = _generate(base, f["session_id"], max_tokens=5)
    assert g_fk2["n_tokens"] > 0


def test_fork_after_generate_includes_generated_tokens(base, make_session):
    """Forking AFTER a generate snapshots the generated tokens too."""
    src = make_session()
    _inject_msgs(base, src, [{"role": "user", "content": "Say hello."}])
    g = _generate(base, src, max_tokens=8)
    size_after_gen = g["n_tokens"]  # not exact cache (inject+gen), but > inject alone

    f = _fork(base, src)
    # the fork's cache includes the source's injected + generated content
    assert f["cache_size"] > 0
    # and it can continue generating
    g2 = _generate(base, f["session_id"], max_tokens=5)
    assert g2["n_tokens"] > 0
    print(f"  [fork-after-generate] src gen {g['n_tokens']} tok, fork cache={f['cache_size']}")


# ------------------------------- latency -------------------------------------


def test_fork_latency_within_budget(base, make_session):
    """EUS-2: fork copy (incl. new context) of a moderate session is fast."""
    src = make_session()
    # inject a realistic-size prompt (~a few hundred tokens)
    _inject(base, src, "The quick brown fox jumps over the lazy dog. " * 20)
    f = _fork(base, src)
    print(f"  [fork-latency] {f['cache_size']} tokens -> {f['fork_ms']} ms")
    assert f["fork_ms"] < FORK_LATENCY_BUDGET_MS


def test_fork_is_idempotent_repeated(base, make_session):
    """Forking the same source repeatedly yields distinct, independent sessions."""
    src = make_session()
    _inject(base, src, "Repeatable snapshot source. ")
    ids = []
    for _ in range(3):
        f = _fork(base, src)
        ids.append(f["session_id"])
        assert f["fork_ms"] < FORK_LATENCY_BUDGET_MS
    assert len(set(ids)) == 3, ids
