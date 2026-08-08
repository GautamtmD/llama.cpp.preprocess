"""Behavior tests for POST /sessions/{id}/fork (EUS-2: KV-cache fork).

Run against a LIVE multimodal-server (built from this branch — it has /fork)::

    engine/multimodal/build/bin/Release/multimodal-server.exe \\
        --model "/c/ML Models/Gemma4 12b/gemma-4-12b-it-qat-q4_0.gguf" \\
        --port 8080

    pytest engine/multimodal/tests/test_fork.py

Approach B (shared-prefix sequences in one pooled context — see
docs/decisions/0004-fork-copy-semantics.md and ADR 0009): the fork aliases its
immutable prefix and owns independently divergent suffix state. These tests
check snapshot correctness (forkable after inject / after generate; source
untouched; later source mutations do not leak into the fork; the fork generates
coherently), empty-fork lifecycle independence, the 404 path, and fork latency.
"""

from __future__ import annotations

import base64
import os

import pytest
import requests

pytestmark = pytest.mark.usefixtures("base", "make_session")

# Fork latency budget. EUS-2 targets ≤1.0 s steady-state for ≤2048 tokens (measured
# ~480 ms tiny, ~700 ms at ~200 tokens: new-context creation + the redecode being
# the new context's first decode). The test gate is deliberately looser than the
# EUS target so GPU/memory variance doesn't make it flaky — it is a regression
# gate, not the perf target.
FORK_LATENCY_BUDGET_MS = 1500


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


def _status(base, sid):
    response = requests.get(f"{base}/sessions/{sid}", timeout=30)
    assert response.status_code == 200, response.text
    return response.json()


def _delete(base, sid):
    response = requests.delete(f"{base}/sessions/{sid}", timeout=30)
    assert response.status_code == 200, response.text


# ------------------------------- core contract ------------------------------


def test_fork_unknown_session_404(base):
    r = requests.post(f"{base}/sessions/s_999999/fork", timeout=30)
    assert r.status_code == 404
    assert "error" in r.json()


@pytest.mark.parametrize("generate_source_first", [False, True], ids=["fork-first", "source-first"])
def test_empty_fork_generation_leaves_idle_sibling_empty_and_reusable(
    base, make_session, generate_source_first
):
    source = make_session()
    forked = _fork(base, source)
    assert forked["fork_ms"] < FORK_LATENCY_BUDGET_MS, forked
    fork = forked["session_id"]
    active, idle = (source, fork) if generate_source_first else (fork, source)

    generated = _generate(base, active, max_tokens=1)
    assert generated["n_tokens"] == 1, generated
    idle_state = _status(base, idle)
    assert idle_state["cache_size"] == 0, idle_state
    assert idle_state["boundary_token"] == -1, idle_state

    _delete(base, active)
    retry = _generate(base, idle, max_tokens=1)
    assert retry["n_tokens"] == 1, retry


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
    """The forked session can generate coherent text from the snapshot.

    Gemma 4 is a thinking model, so we assert coherence (non-empty, alphabetic
    text) rather than a specific short answer — that is fragile at low
    max_tokens. Faithfulness (fork reproduces the source's exact greedy output)
    is checked by ``test_fork_is_faithful_greedy_copy``.
    """
    src = make_session()
    _inject_msgs(base, src, [{"role": "user", "content": "Say hello in one word."}])
    f = _fork(base, src)
    g = _generate(base, f["session_id"], max_tokens=64)
    assert g["n_tokens"] > 0
    assert any(c.isalpha() for c in g["text"]), g["text"]


def test_fork_is_faithful_greedy_copy(base, make_session):
    """At temp=0 (greedy/deterministic), a fork must reproduce the source's EXACT
    output from the shared snapshot — proving the KV copy is faithful and
    generation is deterministic."""
    src = make_session()
    _inject_msgs(base, src, [{"role": "user", "content": "Count from 1 to 5."}])
    f = _fork(base, src)
    # generate from BOTH starting at the same snapshot; greedy -> identical text
    g_src = _generate(base, src, max_tokens=30)
    g_fk = _generate(base, f["session_id"], max_tokens=30)
    assert g_src["text"] == g_fk["text"], (g_src["text"], g_fk["text"])


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
    generated = _generate(base, src, max_tokens=8)

    f = _fork(base, src)
    # the fork's cache includes the source's injected + generated content
    assert f["cache_size"] > 0
    # and it can continue generating
    g2 = _generate(base, f["session_id"], max_tokens=5)
    assert g2["n_tokens"] > 0
    print(
        f"  [fork-after-generate] src gen {generated['n_tokens']} tok, fork cache={f['cache_size']}"
    )


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


# ------------------------------- audio fork ---------------------------------
#
# Audio/image inject ends in an EMBEDDING chunk (no discrete last token), so a
# fork right after it copies the KV but cannot refresh logits until a text token
# is injected. The chat protocol always closes a turn with text markers, so the
# real flow (audio -> text suffix -> fork) is fully generation-ready. These tests
# cover both; they need --mmproj + audio fixtures and skip otherwise.


def _audio_supported(base) -> bool:
    try:
        return bool(requests.get(f"{base}/info", timeout=10).json().get("supports_audio"))
    except Exception:
        return False


def _audio_fixture_b64(name: str) -> str | None:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "fixtures", f"{name}.wav")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def test_fork_after_audio_then_text_is_generation_ready(base, make_session):
    """The real audio flow (audio -> text turn-close -> fork) is generation-ready:
    the text inject sets a valid last token, so the fork refreshes logits."""
    if not _audio_supported(base):
        pytest.skip("requires --mmproj with audio support")
    b64 = _audio_fixture_b64("sent_hello")
    if not b64:
        pytest.skip("audio fixtures missing (run tests/generate_audio_fixtures.py)")
    src = make_session()
    r = requests.post(
        f"{base}/sessions/{src}/inject",
        json={"messages": [{"role": "user", "content": [{"type": "audio", "data": b64}]}]},
        timeout=120,
    )
    assert r.status_code == 200, r.text
    size_after_audio = r.json()["cache_size"]
    # text turn-close sets a discrete last token -> fork can refresh logits
    _inject(base, src, " transcribe the audio")
    f = _fork(base, src)
    assert f["cache_size"] > size_after_audio
    g = _generate(base, f["session_id"], max_tokens=64)
    assert g["n_tokens"] > 0


def test_fork_after_audio_only_restores_kv(base, make_session):
    """A fork immediately after an audio inject (no text suffix) faithfully copies
    the KV (cache_size matches the source). It is NOT immediately generation-ready
    — documented limitation: there is no discrete last token to refresh logits.
    We assert only the KV copy here (generating now would be unsupported)."""
    if not _audio_supported(base):
        pytest.skip("requires --mmproj with audio support")
    b64 = _audio_fixture_b64("sent_hello")
    if not b64:
        pytest.skip("audio fixtures missing (run tests/generate_audio_fixtures.py)")
    src = make_session()
    r = requests.post(
        f"{base}/sessions/{src}/inject",
        json={"messages": [{"role": "user", "content": [{"type": "audio", "data": b64}]}]},
        timeout=120,
    )
    assert r.status_code == 200, r.text
    size = r.json()["cache_size"]
    f = _fork(base, src)
    assert f["cache_size"] == size  # KV faithfully copied
