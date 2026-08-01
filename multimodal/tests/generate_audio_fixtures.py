"""Generate the audio-inject ASR test fixtures with the GPA TTS engine.

Run ONCE (GPU) to (re)produce the committed WAV fixtures + manifest. Not part of
the automated suite — the artifacts are committed under tests/fixtures/.

    cd engine/multimodal
    ../../../.venv/Scripts/python.exe tests/generate_audio_fixtures.py

Outputs:
  tests/fixtures/<name>.wav   (16 kHz mono PCM_16)
  tests/fixtures/manifest.json   ({name: {voice, prompt, expected_words}})

`expected_words` is the normalized token list of the spoken prompt; the ASR test
normalizes the model transcript the same way and compares for an exact
word-sequence match.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ENGINE_ROOT_MULTI = HERE.parent  # engine/multimodal
PROJECT_ROOT = ENGINE_ROOT_MULTI.parents[1]  # worktree root
sys.path.insert(0, str(PROJECT_ROOT / "gui" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "audio_engines" / "gpa_1_5" / "src"))

from multimodalagent.audio.types import Voice  # noqa: E402
from tts_streaming_engine.engine import GPAStreamingTTSEngine  # noqa: E402

FIXTURES_DIR = HERE / "fixtures"

# (name, voice_id, spoken_prompt). 4 single words + 6 short sentences + 1 long.
FIXTURES = [
    ("word_seven", "default", "seven"),
    ("word_banana", "Buffy_flirty", "banana"),
    ("word_purple", "Buffy_Sarcastic", "purple"),
    ("word_morning", "default", "morning"),
    ("sent_fox", "default", "the quick brown fox"),
    ("sent_hello", "Buffy_flirty", "hello how are you"),
    ("sent_weather", "Buffy_Sarcastic", "i like warm weather"),
    ("sent_dog", "default", "the dog runs fast"),
    ("sent_door", "Buffy_flirty", "please close the door"),
    ("sent_game", "Buffy_Sarcastic", "we can win this game"),
    (
        "long_walk",
        "default",
        "the weather is nice today. i think we should go for a walk. "
        "we can stop for coffee on the way. it will be a fun afternoon.",
    ),
]


def normalize_words(text: str) -> list[str]:
    """Lowercase, drop everything but alphanumerics + spaces, tokenize."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [w for w in text.split() if w]


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    print("[fixtures] loading GPA TTS engine ...", flush=True)
    tts = GPAStreamingTTSEngine(mode="balanced")
    tts.warmup()
    sr = tts.info.sample_rate
    print(f"[fixtures] TTS ready (sr={sr}). synthesizing {len(FIXTURES)} clips ...", flush=True)

    manifest: dict[str, dict] = {}
    for name, voice_id, prompt in FIXTURES:
        chunks: list[np.ndarray] = []
        for ch in tts.synthesize_stream(prompt, voice=Voice(id=voice_id, name=voice_id)):
            chunks.append(np.asarray(ch.samples, dtype=np.float32))
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        out = FIXTURES_DIR / f"{name}.wav"
        sf.write(str(out), audio, sr, subtype="PCM_16")
        words = normalize_words(prompt)
        manifest[name] = {"voice": voice_id, "prompt": prompt, "expected_words": words}
        print(
            f"  {name:14s} voice={voice_id:14s} words={len(words):2d}  ({audio.size / sr:.2f}s)",
            flush=True,
        )

    (FIXTURES_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[fixtures] wrote {len(manifest)} clips + manifest to {FIXTURES_DIR}", flush=True)


if __name__ == "__main__":
    main()
