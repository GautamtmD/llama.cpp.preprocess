"""CLI contract: GPU model offload is required unless --allow-cpu is explicit.

These checks need no model: CUDA devices are hidden and a missing model path is
used to prove that the default policy fails before model loading, while
--allow-cpu proceeds to the ordinary model-load error.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_BIN = _REPO / "engine" / "multimodal" / "build" / "bin" / "Release"
SERVER = Path(os.environ.get("MULTIMODAL_SERVER_EXE", _BIN / "multimodal-server.exe"))
BENCH = _BIN / "multimodal-fork-bench.exe"
MISSING_MODEL = _REPO / "definitely-not-a-model.gguf"


def _run(exe: Path, *args: str) -> subprocess.CompletedProcess[str]:
    if not exe.exists():
        pytest.skip(f"executable not built: {exe}")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    return subprocess.run(
        [str(exe), *args],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("exe", [SERVER, BENCH])
def test_help_documents_allow_cpu(exe: Path):
    result = _run(exe, "--help")
    assert result.returncode == 0, result.stdout
    assert "--allow-cpu" in result.stdout
    assert "GPU required" in result.stdout


@pytest.mark.parametrize(("exe", "program"), [(SERVER, "multimodal-server"), (BENCH, "benchmark")])
def test_gpu_is_required_by_default(exe: Path, program: str):
    result = _run(exe, "--model", str(MISSING_MODEL))
    assert result.returncode != 0
    assert "GPU load was not successful" in result.stdout
    assert f"{program} cannot run" in result.stdout
    assert "--allow-cpu" in result.stdout
    assert "failed to load model" not in result.stdout


@pytest.mark.parametrize("exe", [SERVER, BENCH])
def test_allow_cpu_explicitly_bypasses_gpu_requirement(exe: Path):
    result = _run(exe, "--model", str(MISSING_MODEL), "--allow-cpu")
    assert result.returncode != 0
    assert "failed to load model" in result.stdout or "model load failed" in result.stdout
    assert "GPU load was not successful" not in result.stdout
