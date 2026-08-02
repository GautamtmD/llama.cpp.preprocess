"""Deterministic per-sequence context-limit regressions.

This module self-boots a tiny-context server so over-limit generation and raw
multimodal injection are fast to exercise without filling the normal 4096-token
session allocation.
"""

from __future__ import annotations

import base64
import io
import os
import socket
import struct
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import pytest
import requests

_REPO = Path(__file__).resolve().parents[3]
EXE = Path(
    os.environ.get(
        "MULTIMODAL_SERVER_EXE",
        _REPO / "engine" / "multimodal" / "build" / "bin" / "Release" / "multimodal-server.exe",
    )
)
MODEL = os.environ.get("MULTIMODAL_MODEL", "")
MMPROJ = os.environ.get("MULTIMODAL_MMPROJ", "")
CUDA_GPU = os.environ.get("MULTIMODAL_CUDA_VISIBLE_DEVICES", "1")
CTX_SIZE = 32


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def tiny_server():
    if not MODEL or not MMPROJ:
        pytest.skip("MULTIMODAL_MODEL and MULTIMODAL_MMPROJ are required")
    if not EXE.exists():
        pytest.skip(f"server executable is not built: {EXE}")

    port = _free_port()
    command = [
        str(EXE),
        "--model",
        MODEL,
        "--mmproj",
        MMPROJ,
        "--port",
        str(port),
        "--ctx-size",
        str(CTX_SIZE),
        "--max-sequences",
        "2",
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = CUDA_GPU
    log = tempfile.TemporaryFile(mode="w+b")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    base = f"http://127.0.0.1:{port}"
    started = False
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{base}/health", timeout=1).status_code == 200:
                started = True
                break
        except requests.RequestException:
            pass
        if process.poll() is not None:
            break
        time.sleep(0.25)

    if not started:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        log.seek(0)
        output = log.read().decode("utf-8", "replace")
        pytest.fail(f"tiny-context server failed to start:\n{output[-4000:]}")

    yield base

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    log.close()


def _cache_size(base: str, sid: str) -> int:
    response = requests.get(f"{base}/sessions/{sid}", timeout=30)
    assert response.status_code == 200, response.text
    return response.json()["cache_size"]


def _inject_text(base: str, sid: str, text: str) -> requests.Response:
    return requests.post(f"{base}/sessions/{sid}/inject", json={"text": text}, timeout=30)


def test_generation_and_multimodal_injection_never_exceed_per_sequence_limit(tiny_server):
    created = requests.post(f"{tiny_server}/sessions", timeout=30)
    assert created.status_code == 200, created.text
    sid = created.json()["session_id"]
    try:
        first = _inject_text(tiny_server, sid, "x")
        assert first.status_code == 200, first.text

        while _cache_size(tiny_server, sid) < CTX_SIZE - 1:
            before = _cache_size(tiny_server, sid)
            injected = _inject_text(tiny_server, sid, " x")
            assert injected.status_code == 200, injected.text
            assert injected.json()["cache_size"] == before + 1

        before_media = _cache_size(tiny_server, sid)
        pcm = struct.pack("<640f", *([0.0] * 640))
        audio_body = {"audio": base64.b64encode(pcm).decode("ascii")}
        media = requests.post(f"{tiny_server}/sessions/{sid}/inject", json=audio_body, timeout=30)
        if media.status_code == 200:
            assert media.json()["cache_size"] <= CTX_SIZE
            before_media = media.json()["cache_size"]
            media = requests.post(
                f"{tiny_server}/sessions/{sid}/inject", json=audio_body, timeout=30
            )
        assert media.status_code == 409, media.text
        assert "session context full" in media.json()["error"]
        assert _cache_size(tiny_server, sid) == before_media

        while _cache_size(tiny_server, sid) < CTX_SIZE:
            filled = _inject_text(tiny_server, sid, " x")
            assert filled.status_code == 200, filled.text
        assert _cache_size(tiny_server, sid) == CTX_SIZE

        for _ in range(2):
            generated = requests.post(
                f"{tiny_server}/sessions/{sid}/generate",
                json={
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "ignore_eos": True,
                },
                timeout=30,
            )
            assert generated.status_code == 409, generated.text
            assert "session context full" in generated.json()["error"]
            assert _cache_size(tiny_server, sid) == CTX_SIZE

        streamed = requests.post(
            f"{tiny_server}/sessions/{sid}/generate",
            json={"stream": True, "max_tokens": 1, "temperature": 0.0},
            stream=True,
            timeout=30,
        )
        assert streamed.status_code == 409, streamed.text
        assert _cache_size(tiny_server, sid) == CTX_SIZE

        rejected_text = _inject_text(tiny_server, sid, " x")
        assert rejected_text.status_code == 409, rejected_text.text
        assert _cache_size(tiny_server, sid) == CTX_SIZE

        wav = io.BytesIO()
        with wave.open(wav, "wb") as audio_file:
            audio_file.setnchannels(1)
            audio_file.setsampwidth(2)
            audio_file.setframerate(16000)
            audio_file.writeframes(b"\0\0" * 640)
        media_message = requests.post(
            f"{tiny_server}/sessions/{sid}/inject",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": base64.b64encode(wav.getvalue()).decode("ascii")
                                },
                            },
                            {"type": "text", "text": "transcribe"},
                        ],
                    }
                ]
            },
            timeout=30,
        )
        assert media_message.status_code == 409, media_message.text
        assert _cache_size(tiny_server, sid) == CTX_SIZE
    finally:
        requests.delete(f"{tiny_server}/sessions/{sid}", timeout=30)
