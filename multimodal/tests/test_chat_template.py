"""Behavior tests for multimodal-server **Jinja chat-template parity** (M1).

Unlike the slice 1/2/3 suites, this module boots its OWN ``multimodal-server``
instances with custom chat-template CLI flags (so it can vary the template /
kwargs / disable config per case). It therefore does **not** need a pre-started
server at the default port — run it with the self-boot opt-out::

    MULTIMODAL_SELF_BOOT=1 pytest engine/multimodal/tests/test_chat_template.py

Overrides (all optional env vars):
    MULTIMODAL_SERVER_EXE   path to multimodal-server.exe
    MULTIMODAL_MODEL        path to the model gguf
    MULTIMODAL_CUDA_VISIBLE_DEVICES  GPU index to use (default "1" = RTX 5060 Ti)

What this proves (the M1 success criteria):
  1. ``--chat-template <jinja>`` renders **byte-identical** to a recorded,
     model-independent expected string (we reuse ``common_chat_templates_apply`` —
     the very function llama-server uses — so parity is by construction once the
     inputs are wired the same way).
  2. ``--chat-template-kwargs '{"k":"v"}'`` flows into the Jinja context.
  3. ``--system-prompt`` is prepended as a system message.
  4. ``--no-chat-template`` disables templating (messages -> 400; text still ok).
  5. ``--jinja`` / ``--no-jinja`` change template validation:
     a custom Jinja-only template is accepted under ``--jinja`` but rejected
     (fail-fast at startup) under ``--no-jinja``; a syntactically broken
     template is rejected under ``--jinja``.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests

# --------------------------------------------------------------------------- #
# Paths / config (env-overridable)
# --------------------------------------------------------------------------- #
_REPO = Path(__file__).resolve().parents[3]  # .../fork-and-jinja
EXE = os.environ.get(
    "MULTIMODAL_SERVER_EXE",
    str(_REPO / "engine" / "multimodal" / "build" / "bin" / "Release" / "multimodal-server.exe"),
)
MODEL = os.environ.get(
    "MULTIMODAL_MODEL",
    r"C:\ML Models\Gemma4 12b\gemma-4-12b-it-qat-q4_0.gguf",
)
CUDA_GPU = os.environ.get("MULTIMODAL_CUDA_VISIBLE_DEVICES", "1")

# Deterministic, model-independent Jinja template. It does NOT reference
# bos_token/eos_token, so its rendered output depends only on messages,
# add_generation_prompt, and the optional `greeting` kwarg — never on the model.
TPL = (
    "{% if greeting %}{{ greeting }}|{% endif %}"
    "{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}"
    "{% if add_generation_prompt %}GEN{% endif %}"
)
# A syntactically broken Jinja template (missing `}}`).
TPL_BROKEN = "{% for m in messages %}{{ m.content {% endfor %}"

pytestmark = pytest.mark.usefixtures("_kill_servers_atexit")

_PROCS: list[subprocess.Popen] = []


# --------------------------------------------------------------------------- #
# Server lifecycle helpers
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _boot(extra_args: list[str], timeout: int = 180) -> str:
    """Boot a multimodal-server with extra CLI args; return its base URL.

    Raises if the server exits early or never becomes healthy. The process is
    tracked for teardown by the ``_kill_servers_atexit`` fixture.
    """
    if not os.path.exists(EXE):
        pytest.skip(f"multimodal-server.exe not found at {EXE} (build it first)")
    if not os.path.exists(MODEL):
        pytest.skip(f"model not found at {MODEL} (set MULTIMODAL_MODEL)")

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = CUDA_GPU
    cmd = [EXE, "--model", MODEL, "--port", str(port), "--n-gpu-layers", "99", *extra_args]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
    )
    _PROCS.append(proc)

    deadline = time.time() + timeout
    while time.time() < deadline:
        rc = proc.poll()
        if rc is not None:
            out = (proc.stdout.read().decode(errors="replace") if proc.stdout else "")
            pytest.fail(f"server exited early rc={rc}\n{out}")
        try:
            if requests.get(f"{url}/health", timeout=2).status_code == 200:
                return url
        except Exception:
            pass
        time.sleep(1)
    pytest.fail(f"server did not become healthy at {url} within {timeout}s")


def _run_expecting_failure(extra_args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    """Run the exe expecting it to exit non-zero (template validation failure).

    Validation runs before model load, so this returns quickly.
    """
    if not os.path.exists(EXE):
        pytest.skip(f"multimodal-server.exe not found at {EXE} (build it first)")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = CUDA_GPU
    cmd = [EXE, "--model", MODEL, "--port", str(_free_port()), *extra_args]
    return subprocess.run(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=timeout,
    )


@pytest.fixture(autouse=True)
def _kill_servers_atexit():
    """Kill every server we booted when the test session ends."""
    yield
    # per-test teardown is handled by the session-end sweep below


@pytest.fixture(scope="session", autouse=True)
def _sweep_servers():
    yield
    for p in _PROCS:
        try:
            p.kill()
        except Exception:
            pass
        try:
            p.wait(timeout=10)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Server configs (booted lazily, once per session)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def jinja_url():
    """Server with a Jinja override + kwargs + jinja on (covers override, jinja,
    and kwargs together)."""
    return _boot(["--chat-template", TPL, "--jinja",
                  "--chat-template-kwargs", '{"greeting":"HI"}'])


@pytest.fixture(scope="module")
def sysprompt_url():
    """Server with a Jinja override + a global --system-prompt (no kwargs)."""
    return _boot(["--chat-template", TPL, "--system-prompt", "SYS"])


@pytest.fixture(scope="module")
def disabled_url():
    """Server with templating disabled (--no-chat-template)."""
    return _boot(["--no-chat-template"])


def _make_session(base: str) -> str:
    r = requests.post(f"{base}/sessions", timeout=30)
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


def _inject(base: str, sid: str, body: dict) -> dict:
    r = requests.post(f"{base}/sessions/{sid}/inject", json=body, timeout=60)
    return {"status": r.status_code, **r.json()}


# --------------------------------------------------------------------------- #
# 1 + 2: override + jinja + kwargs — byte-identical render
# --------------------------------------------------------------------------- #
def test_override_renders_byte_identical_with_gen_prompt(jinja_url):
    """--chat-template + --jinja + kwargs renders the exact recorded string."""
    sid = _make_session(jinja_url)
    out = _inject(jinja_url, sid, {
        "messages": [{"role": "user", "content": "hi"}],
        "return_prompt": True,
    })
    assert out["status"] == 200, out
    assert out["chat_template_applied"] is True
    # greeting kwarg present -> prefix; add_generation_prompt default true -> GEN suffix
    assert out["prompt"] == "HI|user:hi;GEN"


def test_override_renders_byte_identical_without_gen_prompt(jinja_url):
    """add_generation_prompt=false drops the GEN suffix (template honors it)."""
    sid = _make_session(jinja_url)
    out = _inject(jinja_url, sid, {
        "messages": [{"role": "user", "content": "hi"}],
        "add_generation_prompt": False,
        "return_prompt": True,
    })
    assert out["status"] == 200, out
    assert out["prompt"] == "HI|user:hi;"


def test_kwargs_value_appears_in_prompt(jinja_url):
    """The --chat-template-kwargs value reaches the Jinja context."""
    sid = _make_session(jinja_url)
    out = _inject(jinja_url, sid, {
        "messages": [{"role": "user", "content": "x"}],
        "return_prompt": True,
    })
    assert out["prompt"].startswith("HI|"), out["prompt"]


def test_rendered_prompt_is_actually_tokenized(jinja_url):
    """The rendered prompt is real text that was injected into the KV cache
    (tokens_injected > 0, cache advances) — i.e. rendering feeds prefill."""
    sid = _make_session(jinja_url)
    out = _inject(jinja_url, sid, {
        "messages": [{"role": "user", "content": "hello world"}],
        "return_prompt": True,
    })
    assert out["status"] == 200, out
    assert out["tokens_injected"] > 0
    assert out["cache_size"] == out["tokens_injected"]
    assert out["prompt"].endswith("GEN")


# --------------------------------------------------------------------------- #
# 3: --system-prompt
# --------------------------------------------------------------------------- #
def test_system_prompt_prepended(sysprompt_url):
    """--system-prompt is prepended as a system message before the user's."""
    sid = _make_session(sysprompt_url)
    out = _inject(sysprompt_url, sid, {
        "messages": [{"role": "user", "content": "hi"}],
        "return_prompt": True,
    })
    assert out["status"] == 200, out
    # no kwargs -> no greeting prefix; system message first; then user; then GEN
    assert out["prompt"] == "system:SYS;user:hi;GEN"


# --------------------------------------------------------------------------- #
# 4: --no-chat-template (disable)
# --------------------------------------------------------------------------- #
def test_disabled_rejects_messages(disabled_url):
    sid = _make_session(disabled_url)
    r = requests.post(f"{disabled_url}/sessions/{sid}/inject",
                      json={"messages": [{"role": "user", "content": "hi"}]}, timeout=30)
    assert r.status_code == 400, r.text
    assert "disabled" in r.json()["error"].lower()


def test_disabled_still_accepts_raw_text(disabled_url):
    sid = _make_session(disabled_url)
    out = _inject(disabled_url, sid, {"text": "raw text"})
    assert out["status"] == 200, out
    assert out["chat_template_applied"] is False
    assert out["tokens_injected"] > 0


# --------------------------------------------------------------------------- #
# 5: --jinja / --no-jinja validation wiring (fail-fast at startup)
# --------------------------------------------------------------------------- #
def test_invalid_jinja_template_fails_fast():
    """A syntactically broken template under --jinja must abort startup."""
    cp = _run_expecting_failure(["--chat-template", TPL_BROKEN, "--jinja"])
    assert cp.returncode != 0, cp.stdout
    assert "not supported" in cp.stdout.lower()


def test_no_jinja_rejects_custom_jinja_template():
    """A custom Jinja-only template is rejected under --no-jinja (only commonly
    used templates are accepted without --jinja), proving --no-jinja is wired."""
    cp = _run_expecting_failure(["--chat-template", TPL, "--no-jinja"])
    assert cp.returncode != 0, cp.stdout
    assert "not supported" in cp.stdout.lower()
    assert "commonly used" in cp.stdout.lower()
