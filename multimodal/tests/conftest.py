"""Pytest config for multimodal-server behavior tests.

Requires a running multimodal-server. Start it manually (the model load is slow
and we don't want pytest to own the process):

    cmake --build engine/multimodal/build --config Release --target multimodal-server
    engine/multimodal/build/Release/multimodal-server.exe \
        --model "/c/ML Models/Gemma4 12b/gemma-4-12b-it-qat-q4_0.gguf" \
        --port 8080

Then:

    pytest engine/multimodal/tests

Set MULTIMODAL_SERVER_URL to point elsewhere (default http://127.0.0.1:8080).
If the server isn't reachable, all tests skip (rather than fail) at collection.

Self-booting test modules (e.g. test_chat_template.py, which boots its own
server instances with custom CLI flags) set ``MULTIMODAL_SELF_BOOT=1`` to opt
out of the live-server requirement checked here.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import requests

BASE = os.environ.get("MULTIMODAL_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")

# Self-booting modules manage their own server lifecycles (custom CLI flags) and
# must not be gated on a pre-started server at BASE.
SELF_BOOT = os.environ.get("MULTIMODAL_SELF_BOOT", "") == "1"


def _server_up() -> bool:
    try:
        return requests.get(f"{BASE}/health", timeout=2).status_code == 200
    except Exception:
        return False


if not SELF_BOOT and not _server_up():
    pytest.skip(
        f"multimodal-server not reachable at {BASE} (set MULTIMODAL_SERVER_URL; "
        "see this file's docstring for how to start it)",
        allow_module_level=True,
    )


@pytest.fixture
def base():
    return BASE


@pytest.fixture
def make_session(base):
    """Create a session, yield its id, and ensure it's deleted at teardown."""
    ids = []

    def _make():
        r = requests.post(f"{base}/sessions", timeout=30)
        assert r.status_code == 200, r.text
        sid = r.json()["session_id"]
        ids.append(sid)
        return sid

    yield _make

    for sid in ids:
        try:
            requests.delete(f"{base}/sessions/{sid}", timeout=10)
        except Exception:
            pass
