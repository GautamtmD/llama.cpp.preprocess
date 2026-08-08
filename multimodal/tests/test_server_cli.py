"""Strict multimodal-server startup argument validation.

These checks run before backend/model initialization and need no GPU or model.
Invalid user input must exit cleanly with actionable diagnostics rather than
reaching llama.cpp assertions or silently changing a requested value.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
SERVER = Path(
    os.environ.get(
        "MULTIMODAL_SERVER_EXE",
        _REPO / "engine" / "multimodal" / "build" / "bin" / "Release" / "multimodal-server.exe",
    )
)
MISSING_MODEL = _REPO / "definitely-not-a-model.gguf"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    if not SERVER.exists():
        pytest.skip(f"server executable not built: {SERVER}")
    return subprocess.run(
        [str(SERVER), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((), "--model is required"),
        (("--model",), "--model requires a path"),
        (
            ("--model", str(MISSING_MODEL), "--unknown"),
            "unrecognized argument '--unknown'",
        ),
        (("--model", str(MISSING_MODEL), "--port"), "--port requires an integer value"),
        (
            ("--model", str(MISSING_MODEL), "--n-gpu-layers"),
            "--n-gpu-layers requires an integer value",
        ),
        (
            ("--model", str(MISSING_MODEL), "--ctx-size"),
            "--ctx-size requires an integer value",
        ),
        (
            ("--model", str(MISSING_MODEL), "--n-batch"),
            "--n-batch requires an integer value",
        ),
        (
            ("--model", str(MISSING_MODEL), "--max-sequences"),
            "--max-sequences requires an integer value",
        ),
        (("--model", str(MISSING_MODEL), "--mmproj"), "--mmproj requires a path"),
        (
            ("--model", str(MISSING_MODEL), "--chat-template"),
            "--chat-template requires a value",
        ),
        (
            ("--model", str(MISSING_MODEL), "--chat-template-file"),
            "--chat-template-file requires a path",
        ),
        (
            ("--model", str(MISSING_MODEL), "--system-prompt"),
            "--system-prompt requires a value",
        ),
        (("--model", str(MISSING_MODEL), "--config"), "--config requires a path"),
        (
            ("--model", str(MISSING_MODEL), "--chat-template-kwargs"),
            "--chat-template-kwargs requires a JSON object value",
        ),
        (("--model", str(MISSING_MODEL), "--port", "12x"), "--port must be an integer"),
        (
            ("--model", str(MISSING_MODEL), "--ctx-size", "999999999999999999999"),
            "--ctx-size must be an integer",
        ),
        (
            ("--model", str(MISSING_MODEL), "--port", "0"),
            "--port must be between 1 and 65535",
        ),
        (
            ("--model", str(MISSING_MODEL), "--port", "65536"),
            "--port must be between 1 and 65535",
        ),
        (
            ("--model", str(MISSING_MODEL), "--ctx-size", "0"),
            "--ctx-size must be positive",
        ),
        (
            ("--model", str(MISSING_MODEL), "--n-batch", "0"),
            "--n-batch must be positive",
        ),
        (
            ("--model", str(MISSING_MODEL), "--max-sequences", "0"),
            "--max-sequences must be positive",
        ),
        (
            (
                "--model",
                str(MISSING_MODEL),
                "--n-batch",
                "1",
                "--max-sequences",
                "2",
            ),
            "--n-batch (1) must be at least --max-sequences (2)",
        ),
        (
            (
                "--model",
                str(MISSING_MODEL),
                "--ctx-size",
                "2147483647",
                "--max-sequences",
                "3",
                "--n-batch",
                "3",
            ),
            "exceeds the maximum pooled context size",
        ),
        (
            ("--model", str(MISSING_MODEL), "--chat-template-kwargs", "[]"),
            "--chat-template-kwargs must be a JSON object",
        ),
    ],
)
def test_invalid_arguments_exit_cleanly_with_actionable_error(args: tuple[str, ...], expected: str):
    result = _run(*args)
    assert result.returncode == 2, result.stdout
    assert expected in result.stdout
    assert "error:" in result.stdout
    assert "GGML_ASSERT" not in result.stdout
    assert "terminate called" not in result.stdout


def test_missing_chat_template_file_is_reported_without_terminating():
    missing = _REPO / "definitely-not-a-template.jinja"
    result = _run(
        "--model",
        str(MISSING_MODEL),
        "--chat-template-file",
        str(missing),
    )
    assert result.returncode == 2, result.stdout
    assert "cannot read --chat-template-file" in result.stdout
    assert str(missing) in result.stdout
    assert "terminate called" not in result.stdout


def test_help_documents_batch_capacity_relationship():
    result = _run("--help")
    assert result.returncode == 0, result.stdout
    assert "--n-batch N" in result.stdout
    assert "must be >= --max-sequences" in result.stdout
