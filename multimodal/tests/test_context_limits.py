"""Deterministic per-sequence context-limit regressions.

This module self-boots a tiny-context server so over-limit generation and raw
multimodal injection are fast to exercise without filling the normal 4096-token
session allocation.
"""

from __future__ import annotations

import base64
import io
import json
import os
import socket
import struct
import subprocess
import tempfile
import time
import wave
from collections.abc import Iterator
from contextlib import contextmanager
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


@contextmanager
def _running_server(n_batch: int, max_sequences: int) -> Iterator[str]:
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
        str(max_sequences),
        "--n-batch",
        str(n_batch),
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
        log.close()
        pytest.fail(f"tiny-context server failed to start:\n{output[-4000:]}")

    try:
        yield base
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        log.close()


@pytest.fixture(scope="module")
def tiny_server():
    with _running_server(n_batch=2, max_sequences=2) as base:
        yield base


@pytest.fixture
def single_batch_server():
    with _running_server(n_batch=1, max_sequences=1) as base:
        yield base


def _cache_size(base: str, sid: str) -> int:
    response = requests.get(f"{base}/sessions/{sid}", timeout=30)
    assert response.status_code == 200, response.text
    return response.json()["cache_size"]


def _inject_text(base: str, sid: str, text: str) -> requests.Response:
    return requests.post(f"{base}/sessions/{sid}/inject", json={"text": text}, timeout=30)


def test_batch_one_startup_probe_and_text_injection(single_batch_server):
    created = requests.post(f"{single_batch_server}/sessions", timeout=30)
    assert created.status_code == 200, created.text
    sid = created.json()["session_id"]
    try:
        injected = _inject_text(
            single_batch_server,
            sid,
            "alpha beta gamma delta",
        )
        assert injected.status_code == 200, injected.text
        assert injected.json()["tokens_injected"] > 1

        generated = requests.post(
            f"{single_batch_server}/sessions/{sid}/generate",
            json={"max_tokens": 1, "temperature": 0.0, "ignore_eos": True},
            timeout=30,
        )
        assert generated.status_code == 200, generated.text
        assert generated.json()["n_tokens"] == 1
    finally:
        requests.delete(f"{single_batch_server}/sessions/{sid}", timeout=30)


def test_small_batch_chunks_text_and_multimodal_injection(tiny_server):
    created = requests.post(f"{tiny_server}/sessions", timeout=30)
    assert created.status_code == 200, created.text
    sid = created.json()["session_id"]
    try:
        text = _inject_text(
            tiny_server,
            sid,
            "alpha beta gamma delta epsilon zeta eta theta",
        )
        assert text.status_code == 200, text.text
        assert text.json()["tokens_injected"] > 2
    finally:
        requests.delete(f"{tiny_server}/sessions/{sid}", timeout=30)

    created = requests.post(f"{tiny_server}/sessions", timeout=30)
    assert created.status_code == 200, created.text
    sid = created.json()["session_id"]
    try:
        pcm = struct.pack("<2560f", *([0.0] * 2560))
        raw_audio = requests.post(
            f"{tiny_server}/sessions/{sid}/inject",
            json={"audio": base64.b64encode(pcm).decode("ascii")},
            timeout=30,
        )
        assert raw_audio.status_code == 200, raw_audio.text
        assert raw_audio.json()["cache_size"] > 2
    finally:
        requests.delete(f"{tiny_server}/sessions/{sid}", timeout=30)

    created = requests.post(f"{tiny_server}/sessions", timeout=30)
    assert created.status_code == 200, created.text
    sid = created.json()["session_id"]
    try:
        wav = io.BytesIO()
        with wave.open(wav, "wb") as audio_file:
            audio_file.setnchannels(1)
            audio_file.setsampwidth(2)
            audio_file.setframerate(16000)
            audio_file.writeframes(b"\0\0" * 2560)
        messages = requests.post(
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
        assert messages.status_code == 200, messages.text
        assert messages.json()["cache_size"] > 2
    finally:
        requests.delete(f"{tiny_server}/sessions/{sid}", timeout=30)


def test_context_full_after_one_token_returns_partial_success(tiny_server):
    created = requests.post(f"{tiny_server}/sessions", timeout=30)
    assert created.status_code == 200, created.text
    sid = created.json()["session_id"]
    forked_sid = None
    try:
        first = _inject_text(tiny_server, sid, "x")
        assert first.status_code == 200, first.text
        while _cache_size(tiny_server, sid) < CTX_SIZE - 1:
            before = _cache_size(tiny_server, sid)
            injected = _inject_text(tiny_server, sid, " x")
            assert injected.status_code == 200, injected.text
            assert injected.json()["cache_size"] == before + 1
        assert _cache_size(tiny_server, sid) == CTX_SIZE - 1

        generated = requests.post(
            f"{tiny_server}/sessions/{sid}/generate",
            json={"max_tokens": 2, "temperature": 0.0, "ignore_eos": True},
            timeout=30,
        )
        assert generated.status_code == 200, (
            "generation committed a token but reported a rejected request: "
            f"{generated.text}; cache_size={_cache_size(tiny_server, sid)}"
        )
        result = generated.json()
        assert result["finish_reason"] == "context_full"
        assert result["n_tokens"] == 1
        assert len(result["tokens"]) == 1
        assert result["cache_size"] == CTX_SIZE

        source_status = requests.get(f"{tiny_server}/sessions/{sid}", timeout=30)
        assert source_status.status_code == 200, source_status.text
        assert source_status.json()["boundary_token"] == result["tokens"][-1]

        forked = requests.post(f"{tiny_server}/sessions/{sid}/fork", timeout=30)
        assert forked.status_code == 200, forked.text
        forked_sid = forked.json()["session_id"]
        fork_status = requests.get(f"{tiny_server}/sessions/{forked_sid}", timeout=30)
        assert fork_status.status_code == 200, fork_status.text
        assert fork_status.json()["boundary_token"] == result["tokens"][-1]
    finally:
        if forked_sid is not None:
            requests.delete(f"{tiny_server}/sessions/{forked_sid}", timeout=30)
        requests.delete(f"{tiny_server}/sessions/{sid}", timeout=30)


def test_streaming_context_full_reports_partial_success(tiny_server):
    created = requests.post(f"{tiny_server}/sessions", timeout=30)
    assert created.status_code == 200, created.text
    sid = created.json()["session_id"]
    try:
        first = _inject_text(tiny_server, sid, "x")
        assert first.status_code == 200, first.text
        while _cache_size(tiny_server, sid) < CTX_SIZE - 1:
            injected = _inject_text(tiny_server, sid, " x")
            assert injected.status_code == 200, injected.text

        streamed = requests.post(
            f"{tiny_server}/sessions/{sid}/generate",
            json={
                "stream": True,
                "max_tokens": 2,
                "temperature": 0.0,
                "ignore_eos": True,
            },
            stream=True,
            timeout=30,
        )
        assert streamed.status_code == 200, streamed.text
        events = [
            json.loads(line[len("data: ") :])
            for line in streamed.iter_lines(decode_unicode=True)
            if line and line.startswith("data: ")
        ]
        tokens = [event for event in events if event.get("type") == "token"]
        done = [event for event in events if event.get("type") == "done"]
        assert len(tokens) == 1
        assert len(done) == 1
        assert done[0]["finish_reason"] == "context_full"
        assert done[0]["n_tokens"] == 1
        assert done[0]["cache_size"] == CTX_SIZE

        status = requests.get(f"{tiny_server}/sessions/{sid}", timeout=30)
        assert status.status_code == 200, status.text
        assert status.json()["boundary_token"] == tokens[0]["id"]
    finally:
        requests.delete(f"{tiny_server}/sessions/{sid}", timeout=30)


def test_generation_and_multimodal_injection_never_exceed_per_sequence_limit(
    tiny_server,
):
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
