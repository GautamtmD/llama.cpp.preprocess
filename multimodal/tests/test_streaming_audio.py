"""Tests for real-time streaming audio injection.

Validates that splitting an audio file into chunks and injecting them 
incrementally yields the exact same KV cache size and generated output 
as injecting the entire file in one shot.
"""

from __future__ import annotations

import base64
import os
import struct
import wave
import pytest
import requests

pytestmark = pytest.mark.usefixtures("base", "make_session")

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES_DIR = os.path.join(HERE, "fixtures")


def read_wav_as_float32(name: str) -> tuple[list[float], int]:
    path = os.path.join(FIXTURES_DIR, f"{name}.wav")
    with wave.open(path, "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        n_frames = w.getnframes()
        data = w.readframes(n_frames)

        samples = []
        if sampwidth == 2:
            # 16-bit PCM
            for i in range(0, len(data), 2 * n_channels):
                val = struct.unpack("<h", data[i : i + 2])[0]
                samples.append(val / 32768.0)
        elif sampwidth == 4:
            # 32-bit float PCM
            for i in range(0, len(data), 4 * n_channels):
                val = struct.unpack("<f", data[i : i + 4])[0]
                samples.append(val)
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")

        return samples, framerate


TRANSCRIPTION_SYSTEM_PROMPT = (
    "You are a speech-to-text engine. The user message has an attached audio "
    "recording — it is genuinely provided as audio input; never claim it is "
    "missing or that you cannot hear it. Output ONLY the literal spoken words in "
    "order. No reasoning, no commentary, no quotes, no punctuation."
)


def test_streaming_vs_oneshot_equivalence(base, make_session):
    name = "sent_door"  # "please close the door"
    samples, sr = read_wav_as_float32(name)
    assert sr == 16000, f"Expected 16kHz fixture, got {sr}Hz"

    # 1. ONE-SHOT SESSION
    sid_one = make_session()
    
    # Inject using standard messages format (one-shot)
    with open(os.path.join(FIXTURES_DIR, f"{name}.wav"), "rb") as f:
        wav_b64 = base64.b64encode(f.read()).decode("ascii")
        
    body_one = {
        "messages": [
            {"role": "system", "content": TRANSCRIPTION_SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "audio", "data": wav_b64}]},
        ]
    }
    r_one = requests.post(f"{base}/sessions/{sid_one}/inject", json=body_one, timeout=120)
    assert r_one.status_code == 200, r_one.text
    cache_size_one = r_one.json()["cache_size"]

    # Generate one-shot transcript
    g_one = requests.post(
        f"{base}/sessions/{sid_one}/generate",
        json={"max_tokens": 100, "temperature": 0.0},
        timeout=120,
    )
    assert g_one.status_code == 200
    text_one = g_one.json()["text"]

    # 2. STREAMING SESSION
    sid_stream = make_session()

    # Step 2a: Inject prompt prefix (system prompt + user start + audio start marker)
    # The Gemma 4 chat template wraps system, user, and starts the audio block.
    # We construct the prefix manually using standard format or text.
    # Prefix includes: system prompt + user marker + <|audio>
    prefix_text = (
        f"<|turn>system\n<|think|>\n{TRANSCRIPTION_SYSTEM_PROMPT}<turn|>\n"
        f"<|turn>user\n<|audio>"
    )
    r_pref = requests.post(
        f"{base}/sessions/{sid_stream}/inject",
        json={"text": prefix_text},
        timeout=60
    )
    assert r_pref.status_code == 200, r_pref.text

    # Step 2b: Inject audio chunks incrementally (e.g. 400ms / 6400 samples)
    chunk_size = 6400
    for i in range(0, len(samples), chunk_size):
        chunk = samples[i : i + chunk_size]
        # pack as float32
        chunk_bytes = struct.pack(f"<{len(chunk)}f", *chunk)
        chunk_b64 = base64.b64encode(chunk_bytes).decode("ascii")
        
        r_chunk = requests.post(
            f"{base}/sessions/{sid_stream}/inject",
            json={"audio": chunk_b64},
            timeout=60
        )
        assert r_chunk.status_code == 200, r_chunk.text
        print(f"  [stream] injected chunk, cache size: {r_chunk.json()['cache_size']}")

    # Step 2c: Inject suffix text (audio end marker + user turn end + assistant start)
    suffix_text = "<audio|><turn|>\n<|turn>model\n"
    r_suff = requests.post(
        f"{base}/sessions/{sid_stream}/inject",
        json={"text": suffix_text},
        timeout=60
    )
    assert r_suff.status_code == 200, r_suff.text
    cache_size_stream = r_suff.json()["cache_size"]

    # Generate streaming transcript
    g_stream = requests.post(
        f"{base}/sessions/{sid_stream}/generate",
        json={"max_tokens": 100, "temperature": 0.0},
        timeout=120,
    )
    assert g_stream.status_code == 200
    text_stream = g_stream.json()["text"]

    # 3. VERIFY EQUIVALENCE
    import re
    def normalize_words(t: str) -> list[str]:
        t = t.lower()
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        return [w for w in t.split() if w]

    def is_subsequence(needle: list[str], haystack: list[str]) -> bool:
        it = iter(haystack)
        return all(w in it for w in needle)

    print(f"One-shot cache: {cache_size_one}, Stream cache: {cache_size_stream}")
    print(f"One-shot text : {text_one!r}")
    print(f"Stream text    : {text_stream!r}")

    assert cache_size_stream == cache_size_one, "KV cache size mismatch!"
    expected = ["please", "close", "the", "door"]
    words_one = normalize_words(text_one)
    words_stream = normalize_words(text_stream)
    assert is_subsequence(expected, words_one), f"One-shot transcription missing expected words: {text_one}"
    assert is_subsequence(expected, words_stream), f"Stream transcription missing expected words: {text_stream}"


def test_streaming_latency_benchmark(base, make_session):
    import time
    import json
    name = "sent_door"
    samples, sr = read_wav_as_float32(name)

    # 1. COLD PATH (one-shot inject at EOS)
    sid_cold = make_session()
    with open(os.path.join(FIXTURES_DIR, f"{name}.wav"), "rb") as f:
        wav_b64 = base64.b64encode(f.read()).decode("ascii")

    body_cold = {
        "messages": [
            {"role": "system", "content": TRANSCRIPTION_SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "audio", "data": wav_b64}]},
        ]
    }

    # Measure total inject + generate start (Cold)
    t_cold_start = time.time()
    r_cold = requests.post(f"{base}/sessions/{sid_cold}/inject", json=body_cold, timeout=120)
    assert r_cold.status_code == 200, r_cold.text

    # Start generation and measure TTFT
    r_gen_cold = requests.post(
        f"{base}/sessions/{sid_cold}/generate",
        json={"stream": True, "max_tokens": 10, "temperature": 0.0},
        stream=True, timeout=120
    )
    assert r_gen_cold.status_code == 200

    t_cold_first_token = None
    for line in r_gen_cold.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            event = json.loads(line[6:])
            if event.get("type") == "token":
                t_cold_first_token = time.time()
                break

    cold_latency = (t_cold_first_token - t_cold_start) if t_cold_first_token else 999.0
    print(f"  [cold] EOS -> first token latency: {cold_latency * 1000:.1f} ms")

    # 2. WARM PATH (streaming inject during turn)
    sid_warm = make_session()

    # Step 2a: Inject prefix
    prefix_text = (
        f"<|turn>system\n<|think|>\n{TRANSCRIPTION_SYSTEM_PROMPT}<turn|>\n"
        f"<|turn>user\n<|audio>"
    )
    r_pref = requests.post(f"{base}/sessions/{sid_warm}/inject", json={"text": prefix_text}, timeout=60)
    assert r_pref.status_code == 200

    # Step 2b: Stream chunks
    chunk_size = 6400
    for i in range(0, len(samples), chunk_size):
        chunk = samples[i : i + chunk_size]
        chunk_bytes = struct.pack(f"<{len(chunk)}f", *chunk)
        chunk_b64 = base64.b64encode(chunk_bytes).decode("ascii")
        r_chunk = requests.post(f"{base}/sessions/{sid_warm}/inject", json={"audio": chunk_b64}, timeout=60)
        assert r_chunk.status_code == 200

    # User finishes speaking -> EOS!
    t_warm_eos = time.time()

    # Step 2c: Inject suffix
    suffix_text = "<audio|><turn|>\n<|turn>model\n"
    r_suff = requests.post(f"{base}/sessions/{sid_warm}/inject", json={"text": suffix_text}, timeout=60)
    assert r_suff.status_code == 200

    # Start generation and measure TTFT from t_warm_eos
    r_gen_warm = requests.post(
        f"{base}/sessions/{sid_warm}/generate",
        json={"stream": True, "max_tokens": 10, "temperature": 0.0},
        stream=True, timeout=120
    )
    assert r_gen_warm.status_code == 200

    t_warm_first_token = None
    for line in r_gen_warm.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            event = json.loads(line[6:])
            if event.get("type") == "token":
                t_warm_first_token = time.time()
                break

    warm_latency = (t_warm_first_token - t_warm_eos) if t_warm_first_token else 999.0
    print(f"  [warm] EOS -> first token latency: {warm_latency * 1000:.1f} ms")

    assert warm_latency < 1.000, f"Warm latency {warm_latency * 1000:.1f} ms exceeds the 1000 ms budget!"
    assert warm_latency < cold_latency, "Streaming inject (warm) should be faster than one-shot (cold)!"


def test_streaming_payload_validation(base, make_session):
    sid = make_session()

    # 1. Test empty audio payload
    r_empty = requests.post(f"{base}/sessions/{sid}/inject", json={"audio": ""}, timeout=10)
    assert r_empty.status_code == 400
    assert "cannot be empty" in r_empty.json()["error"]

    # 2. Test payload size not multiple of sizeof(float)
    bad_bytes_b64 = base64.b64encode(b"123").decode("ascii")
    r_size = requests.post(f"{base}/sessions/{sid}/inject", json={"audio": bad_bytes_b64}, timeout=10)
    assert r_size.status_code == 400
    assert "multiple of 4 bytes" in r_size.json()["error"]

    # 3. Test payload sample count not multiple of 640
    bad_samples = [0.0] * 10
    bad_samples_bytes = struct.pack("<10f", *bad_samples)
    bad_samples_b64 = base64.b64encode(bad_samples_bytes).decode("ascii")
    r_samples = requests.post(f"{base}/sessions/{sid}/inject", json={"audio": bad_samples_b64}, timeout=10)
    assert r_samples.status_code == 400
    assert "multiple of 640" in r_samples.json()["error"]


def test_tts_to_inject_integration(base, make_session):
    import sys
    from pathlib import Path

    # Resolve project root and append GPA engine source to sys.path
    project_root = Path(HERE).parents[2]
    gpa_src = project_root / "audio_engines" / "gpa_1_5" / "src"
    if str(gpa_src) not in sys.path:
        sys.path.insert(0, str(gpa_src))

    from tts_streaming_engine.engine import GPAStreamingTTSEngine, resolve_assets_dir
    import numpy as np

    # Initialize the TTS engine
    assets = resolve_assets_dir(project_root)
    eng = GPAStreamingTTSEngine(assets_dir=assets, mode="performance", project_root=project_root)
    eng.warmup()

    # Synthesize phrase
    text_to_speak = "please close the door"
    chunks = [c.samples for c in eng.synthesize_stream(text_to_speak)]
    assert len(chunks) > 0, "No audio chunks synthesized by TTS"

    # Concatenate and pad to 640-sample boundaries
    samples = np.concatenate(chunks)
    remainder = len(samples) % 640
    if remainder != 0:
        padding = 640 - remainder
        samples = np.pad(samples, (0, padding), mode="constant")

    # Stream chunks to the server
    sid = make_session()
    prefix_text = (
        f"<|turn>system\n<|think|>\n{TRANSCRIPTION_SYSTEM_PROMPT}<turn|>\n"
        f"<|turn>user\n<|audio>"
    )
    r_pref = requests.post(f"{base}/sessions/{sid}/inject", json={"text": prefix_text}, timeout=60)
    assert r_pref.status_code == 200

    chunk_size = 6400
    for i in range(0, len(samples), chunk_size):
        chunk = samples[i : i + chunk_size]
        chunk_bytes = struct.pack(f"<{len(chunk)}f", *chunk)
        chunk_b64 = base64.b64encode(chunk_bytes).decode("ascii")
        r_chunk = requests.post(f"{base}/sessions/{sid}/inject", json={"audio": chunk_b64}, timeout=60)
        assert r_chunk.status_code == 200

    # Suffix
    suffix_text = "<audio|><turn|>\n<|turn>model\n"
    r_suff = requests.post(f"{base}/sessions/{sid}/inject", json={"text": suffix_text}, timeout=60)
    assert r_suff.status_code == 200

    # Generate transcript
    g = requests.post(
        f"{base}/sessions/{sid}/generate",
        json={"max_tokens": 100, "temperature": 0.0},
        timeout=120,
    )
    assert g.status_code == 200
    transcript = g.json()["text"]
    print(f"  [TTS Integration] Generated: {transcript!r}")

    # Validate transcription content contains key words
    import re
    def normalize_words(t: str) -> list[str]:
        t = t.lower()
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        return [w for w in t.split() if w]

    def is_subsequence(needle: list[str], haystack: list[str]) -> bool:
        it = iter(haystack)
        return all(w in it for w in needle)

    expected = ["please", "close", "the", "door"]
    words = normalize_words(transcript)
    assert is_subsequence(expected, words), f"ASR failed on TTS-generated audio. Output: {transcript}"



