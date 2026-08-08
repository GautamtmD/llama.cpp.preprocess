"""Behavior tests for multimodal-server slice 3a — image injection (mtmd).

These run against a LIVE server started with ``--mmproj``::

    engine/multimodal/build/bin/Release/multimodal-server.exe \\
        --model "/c/ML Models/Gemma4 12b/gemma-4-12b-it-qat-q4_0.gguf" \\
        --mmproj "/c/ML Models/Gemma4 12b/mmproj-gemma-4-12b-it-qat-q4_0.gguf" \\
        --port 8080

    pytest engine/multimodal/tests/test_multimodal.py

The no-``--mmproj`` error path (success criterion #4) needs a *second* server
instance started without ``--mmproj``. Point the env var
``MULTIMODAL_SERVER_URL_NO_MMPROJ`` at it; if unset, that one test is skipped.
"""

from __future__ import annotations

import base64
import ctypes
import io
import os
from pathlib import Path

import pytest
import requests
from PIL import Image

pytestmark = pytest.mark.usefixtures("base", "make_session")


# ----------------------------- image helpers --------------------------------


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _solid_png_b64(color: tuple[int, int, int], size: tuple[int, int]) -> str:
    img = Image.new("RGB", size, color)
    return _png_b64(img)


def _process_rss_bytes(pid: int) -> int:
    if os.name == "nt":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        handle = kernel32.OpenProcess(0x0400, False, pid)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                raise ctypes.WinError(ctypes.get_last_error())
            return int(counters.WorkingSetSize)
        finally:
            kernel32.CloseHandle(handle)

    statm = Path(f"/proc/{pid}/statm")
    if statm.exists():
        resident_pages = int(statm.read_text(encoding="ascii").split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    pytest.skip("RSS measurement is unsupported on this platform")


# ------------------------------- /info --------------------------------------


def test_info_reports_multimodal_capabilities(base):
    """With --mmproj loaded, /info must advertise vision+audio and a real
    sample rate (the encoder-free Gemma 4 projector supports both)."""
    r = requests.get(f"{base}/info", timeout=10)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["model_loaded"] is True
    assert j["supports_vision"] is True, j
    assert j["supports_audio"] is True, j
    assert isinstance(j["audio_sample_rate"], int) and j["audio_sample_rate"] > 0, j
    print(
        f"  [info] vision={j['supports_vision']} audio={j['supports_audio']} "
        f"rate={j['audio_sample_rate']}"
    )


# ----------------------- image inject: core contract ------------------------


def test_inject_image_returns_multimodal_metadata(base, make_session):
    """An image part must take the mtmd path: response carries
    used_multimodal=true and the media count, and advances the cache."""
    sid = make_session()
    img_b64 = _solid_png_b64((255, 0, 0), (224, 224))
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this?"},
                    {"type": "image", "data": img_b64},
                ],
            }
        ]
    }
    r = requests.post(f"{base}/sessions/{sid}/inject", json=body, timeout=120)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["used_multimodal"] is True, j
    assert j["n_media"] == 1, j
    assert j["chat_template_applied"] is True, j
    assert j["cache_size"] > 0, j
    assert j["inject_ms"] >= 0
    print(f"  [inject-image] cache_size={j['cache_size']} inject_ms={j['inject_ms']}")


def test_inject_image_data_url_prefix_is_stripped(base, make_session):
    """The OpenAI-style ``data:image/png;base64,<..>`` URL form must also work
    (prefix stripped before base64-decoding)."""
    sid = make_session()
    img_b64 = _solid_png_b64((0, 255, 0), (224, 224))
    data_url = f"data:image/png;base64,{img_b64}"
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
    }
    r = requests.post(f"{base}/sessions/{sid}/inject", json=body, timeout=120)
    assert r.status_code == 200, r.text
    assert r.json()["used_multimodal"] is True


# -------------------- image inject across multiple sizes --------------------
# Verifies the decode + projector path tolerates arbitrary input dimensions
# (the mtmd helper + projector resample to the native patch grid). We assert
# successful inject + metadata for every size; the model-answer check lives in
# its own test below (one canonical size, to keep the suite fast).


@pytest.mark.parametrize("size", [(32, 32), (224, 224), (512, 512), (1024, 768), (768, 1024)])
def test_inject_image_accepts_various_sizes(base, make_session, size):
    sid = make_session()
    img_b64 = _solid_png_b64((0, 0, 255), size)
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this?"},
                    {"type": "image", "data": img_b64},
                ],
            }
        ]
    }
    r = requests.post(f"{base}/sessions/{sid}/inject", json=body, timeout=120)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["used_multimodal"] is True
    assert j["n_media"] == 1
    assert j["cache_size"] > 0
    print(f"  [inject-size] {size} -> cache_size={j['cache_size']} inject_ms={j['inject_ms']}")


# --------------------- the headline correctness check -----------------------
# Inject a solid-red image and confirm the model answers "red". This is the
# end-to-end proof that image patches actually reach the LLM in a usable form
# (not just that inject returns 200).


def test_inject_red_image_then_generate_answers_red(base, make_session):
    sid = make_session()
    img_b64 = _solid_png_b64((255, 0, 0), (224, 224))
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What color is in this image? Answer with a single word.",
                    },
                    {"type": "image", "data": img_b64},
                ],
            }
        ]
    }
    r = requests.post(f"{base}/sessions/{sid}/inject", json=body, timeout=120)
    assert r.status_code == 200, r.text
    assert r.json()["used_multimodal"] is True

    g = requests.post(
        f"{base}/sessions/{sid}/generate",
        json={"max_tokens": 200, "temperature": 0.0},
        timeout=180,
    )
    assert g.status_code == 200, g.text
    text = g.json()["text"].lower()
    print(f"  [red-image] model said: {g.json()['text']!r}")
    assert "red" in text, f"expected 'red' in model response: {text!r}"


# ----------------------------- multi-image ----------------------------------
# Success criterion #3: multi-image + interleaved text injects without error
# and the model still responds.


def test_inject_multi_image_interleaved(base, make_session):
    sid = make_session()
    red = _solid_png_b64((255, 0, 0), (224, 224))
    blue = _solid_png_b64((0, 0, 255), (224, 224))
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here are two images."},
                    {"type": "image", "data": red},
                    {"type": "text", "text": "And here is the second one."},
                    {"type": "image", "data": blue},
                    {"type": "text", "text": "How many images did I show you?"},
                ],
            }
        ]
    }
    r = requests.post(f"{base}/sessions/{sid}/inject", json=body, timeout=180)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["used_multimodal"] is True
    assert j["n_media"] == 2, j

    g = requests.post(
        f"{base}/sessions/{sid}/generate",
        json={"max_tokens": 40, "temperature": 0.0},
        timeout=180,
    )
    assert g.status_code == 200, g.text
    assert isinstance(g.json()["text"], str) and len(g.json()["text"]) > 0
    print(f"  [multi-image] model said: {g.json()['text']!r}")


@pytest.mark.requires("image")
def test_malformed_multi_image_releases_earlier_bitmaps(base, make_session, server_pid):
    sid = make_session()
    valid = _solid_png_b64((64, 128, 192), (1024, 1024))
    invalid = base64.b64encode(b"not an image").decode("ascii")
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "data": valid},
                    {"type": "image", "data": invalid},
                ],
            }
        ]
    }

    def reject() -> None:
        response = requests.post(f"{base}/sessions/{sid}/inject", json=body, timeout=120)
        assert response.status_code == 400, response.text
        assert "failed to decode image" in response.text

    for _ in range(3):
        reject()
    rss_before = _process_rss_bytes(server_pid)
    for _ in range(24):
        reject()
    rss_after = _process_rss_bytes(server_pid)
    retained = rss_after - rss_before
    print(
        f"  [multipart-rss] before={rss_before} after={rss_after} "
        f"retained={retained / (1024 * 1024):.1f} MiB"
    )
    assert retained < 24 * 1024 * 1024, (
        f"24 rejected multipart requests retained {retained / (1024 * 1024):.1f} MiB RSS"
    )

    status = requests.get(f"{base}/sessions/{sid}", timeout=30)
    assert status.status_code == 200, status.text
    assert status.json()["cache_size"] == 0


@pytest.mark.requires("image")
def test_malformed_message_after_image_leaves_session_unchanged(base, make_session):
    sid = make_session()
    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image", "data": _solid_png_b64((255, 0, 0), (224, 224))}],
            },
            {
                "role": 123,
                "content": "malformed role after an already decoded bitmap",
            },
        ]
    }
    response = requests.post(f"{base}/sessions/{sid}/inject", json=body, timeout=120)
    assert response.status_code == 400, response.text
    status = requests.get(f"{base}/sessions/{sid}", timeout=30)
    assert status.status_code == 200, status.text
    assert status.json()["cache_size"] == 0


# --------------------- error path: image without --mmproj -------------------
# Success criterion #4. Requires a second server started WITHOUT --mmproj.
# Set MULTIMODAL_SERVER_URL_NO_MMPROJ to its URL; otherwise skip.


def test_inject_image_without_mmproj_returns_error():
    no_proj = os.environ.get("MULTIMODAL_SERVER_URL_NO_MMPROJ")
    if not no_proj:
        pytest.skip("set MULTIMODAL_SERVER_URL_NO_MMPROJ to a server started without --mmproj")

    # make a session on the no-mmproj server
    rs = requests.post(f"{no_proj}/sessions", timeout=30)
    assert rs.status_code == 200, rs.text
    sid = rs.json()["session_id"]
    try:
        img_b64 = _solid_png_b64((255, 0, 0), (224, 224))
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image", "data": img_b64},
                    ],
                }
            ]
        }
        r = requests.post(f"{no_proj}/sessions/{sid}/inject", json=body, timeout=60)
        assert r.status_code >= 400, r.text
        assert "error" in r.json(), r.text
        # the message should point at the missing projector
        msg = r.json()["error"].get("message", "").lower()
        assert "mmproj" in msg or "projector" in msg, msg
        print(f"  [no-mmproj] status={r.status_code} msg={msg!r}")
    finally:
        requests.delete(f"{no_proj}/sessions/{sid}", timeout=10)


# --------------------- backward compatibility smoke -------------------------
# Success criterion #5: string-content and plain-text inject still work and do
# NOT take the multimodal path. (Full slice-1/2 regressions are covered by
# test_session_api.py / test_streaming.py running alongside; this just locks
# the multimodal flag's negative case.)


def test_inject_string_content_is_not_multimodal(base, make_session):
    sid = make_session()
    body = {"messages": [{"role": "user", "content": "Hello, what is 2+2?"}]}
    r = requests.post(f"{base}/sessions/{sid}/inject", json=body, timeout=60)
    assert r.status_code == 200, r.text
    j = r.json()
    # used_multimodal is only present on the mtmd path; its absence => text-only
    assert "used_multimodal" not in j, j
    assert j["chat_template_applied"] is True
    assert j["tokens_injected"] > 0
