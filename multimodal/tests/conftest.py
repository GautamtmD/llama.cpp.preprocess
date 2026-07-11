"""Pytest config for multimodal-server behavior tests.

The server can be provided in three ways (first that applies):

1. **Manual** (default): start the server yourself, then run pytest. It is
   detected at the default port (or ``MULTIMODAL_SERVER_URL``)::

       cmake --build engine/multimodal/build --target multimodal-server
       engine/multimodal/build/bin/Release/multimodal-server.exe \\
           --model "/c/ML Models/<name>/<file>.gguf" [--mmproj ...] --port 8080
       pytest engine/multimodal/tests

2. **Self-boot, model as a parameter** (reuses the test_chat_template.py pattern):
   set ``MULTIMODAL_MODEL`` (gguf path) and pytest boots a fresh server on a free
   port for the session. Optional ``MULTIMODAL_MMPROJ``, ``MULTIMODAL_SERVER_EXE``,
   ``MULTIMODAL_CUDA_VISIBLE_DEVICES``. This is what makes the suite
   **model-agnostic** — the same tests run on any model the engine can load::

       MULTIMODAL_MODEL="C:/ML Models/MiniCPM-V-4.6/MiniCPM-V-4_6-Q8_0.gguf" \
       MULTIMODAL_MMPROJ="C:/ML Models/MiniCPM-V-4.6/mmproj-model-f16.gguf" \
       pytest engine/multimodal/tests/test_offload.py

3. **Self-managed** modules (e.g. test_chat_template.py, which boots its own
   instances with custom CLI flags) set ``MULTIMODAL_SELF_BOOT=1`` to opt out of
   the live-server check here.

If no server is reachable and ``MULTIMODAL_MODEL`` is unset, all tests skip at
collection rather than fail.

**Capability-aware selection.** The running model's modalities are surfaced via
``GET /info`` (``input_modalities`` / ``output_modalities``, driven by the
per-model config layer). The session-scoped ``capabilities`` fixture and the
``@pytest.mark.requires("image"|"audio")`` marker skip any test whose required
input modality the running model does not support — so, e.g., audio tests skip
automatically on a vision+text model like MiniCPM-V-4.6.
"""

from __future__ import annotations

import atexit
import os
import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests

BASE = os.environ.get("MULTIMODAL_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")

# Self-boot parameters (env-overridable).
_REPO = Path(__file__).resolve().parents[3]  # .../<worktree root>
MODEL = os.environ.get("MULTIMODAL_MODEL", "")
MMPROJ = os.environ.get("MULTIMODAL_MMPROJ", "")
EXE = os.environ.get(
    "MULTIMODAL_SERVER_EXE",
    str(_REPO / "engine" / "multimodal" / "build" / "bin" / "Release" / "multimodal-server.exe"),
)
CUDA_GPU = os.environ.get("MULTIMODAL_CUDA_VISIBLE_DEVICES", "1")

# Self-managed modules manage their own server lifecycles and must not be gated.
SELF_BOOT = os.environ.get("MULTIMODAL_SELF_BOOT", "") == "1"

_PROCS: list[subprocess.Popen] = []


def _server_up(url: str = BASE) -> bool:
    try:
        return requests.get(f"{url}/health", timeout=2).status_code == 200
    except Exception:
        return False


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_health(url: str, timeout: float = 600.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _server_up(url):
            return True
        time.sleep(1.0)
    return False


# Module-level gate: skip only if we have neither a live server nor a model to
# self-boot. (SELF_BOOT modules opt out entirely.)
if not SELF_BOOT and not _server_up() and not MODEL:
    pytest.skip(
        f"multimodal-server not reachable at {BASE} and MULTIMODAL_MODEL is unset "
        "(start the server, or set MULTIMODAL_MODEL to self-boot; see this file's "
        "docstring)",
        allow_module_level=True,
    )


def _kill_procs() -> None:
    for p in _PROCS:
        try:
            p.terminate()
            p.wait(timeout=10)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


atexit.register(_kill_procs)


@pytest.fixture(scope="session", autouse=True)
def _ensure_server():
    """Use a running server if reachable; otherwise self-boot one for the
    session from MULTIMODAL_MODEL. SELF_BOOT modules manage their own servers."""
    global BASE
    if SELF_BOOT:
        return
    if _server_up(BASE):
        return  # manual server
    if not MODEL:
        return  # nothing to boot; collection already skipped non-SELF_BOOT modules
    if not os.path.exists(EXE):
        pytest.exit(
            f"MULTIMODAL_SERVER_EXE not found at {EXE} (build it first: "
            "scripts/build_engine.sh)", returncode=5,
        )
    if not os.path.exists(MODEL):
        pytest.exit(f"MULTIMODAL_MODEL not found: {MODEL}", returncode=5)

    port = _free_port()
    cmd = [EXE, "--model", MODEL, "--port", str(port)]
    if MMPROJ:
        cmd += ["--mmproj", MMPROJ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = CUDA_GPU
    print(f"[conftest] self-booting multimodal-server on port {port}: {MODEL}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    _PROCS.append(proc)
    url = f"http://127.0.0.1:{port}"
    if not _wait_health(url):
        # surface server stderr to aid debugging
        try:
            out = proc.stdout.read().decode("utf-8", "replace") if proc.stdout else ""
        except Exception:
            out = ""
        pytest.exit(f"self-booted server did not become healthy at {url}\n{out[-4000:]}", returncode=5)
    BASE = url


@pytest.fixture(scope="session")
def base(_ensure_server):
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


# --------------------------------------------------------------------------- #
# Capability-aware test selection (driven by the model config via /info)
# --------------------------------------------------------------------------- #

class Capabilities:
    """The running model's input/output modalities, from GET /info."""

    def __init__(self, info: dict):
        mods = set(info.get("input_modalities") or [])
        if not mods:
            # Fallback for older servers: infer from the projector booleans.
            mods = {"text"}
            if info.get("supports_vision"):
                mods.add("image")
            if info.get("supports_audio"):
                mods.add("audio")
        self.input: set[str] = mods
        self.output: set[str] = set(info.get("output_modalities") or ["text"])
        self.raw = info

    def supports(self, modality: str) -> bool:
        return modality in self.input

    def require(self, modality: str) -> None:
        if modality not in self.input:
            pytest.skip(
                f"running model does not support '{modality}' input "
                f"(supports {sorted(self.input)})"
            )


@pytest.fixture(scope="session")
def capabilities(base):
    """The running model's capabilities, or None if no server is reachable at
    `base` (e.g. SELF_BOOT modules that manage their own servers per-test). When
    None, capability-based skipping is a no-op."""
    try:
        r = requests.get(f"{base}/info", timeout=10)
        if r.status_code == 200:
            return Capabilities(r.json())
    except requests.exceptions.RequestException:
        pass
    return None


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires(modality): skip unless the running model supports this input "
        "modality (text | image | audio), per GET /info input_modalities",
    )


@pytest.fixture(autouse=True)
def _skip_by_capability(request, capabilities):
    """Honor @pytest.mark.requires(...); no-op for unmarked tests or when
    capabilities can't be determined (no server at base)."""
    if capabilities is None:
        return
    m = request.node.get_closest_marker("requires")
    if m is None:
        return
    for mod in m.args:
        if mod not in capabilities.input:
            pytest.skip(
                f"requires '{mod}' input (model supports {sorted(capabilities.input)})"
            )
