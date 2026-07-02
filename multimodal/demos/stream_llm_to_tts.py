"""Demo: stream an LLM response through the GPA TTS engine → audio.

The full pipeline:
  1. Create an LLM session, inject a chat-formatted prompt.
  2. Stream tokens from /sessions/{id}/generate?stream=true (SSE).
  3. Buffer tokens into sentences; as each sentence completes, feed it to the
     GPA streaming TTS engine.
  4. Collect audio chunks; write to a wav file.

Measures:
  - LLM time-to-first-token (TTFT)
  - End-to-end time-to-first-audio (when the first TTS chunk is ready)
  - Total wall vs total audio duration (overall pipeline RTF)
  - Whether the pipeline can start speaking before the LLM finishes generating.

Usage:
    # start the multimodal-server first (see README)
    python demos/stream_llm_to_tts.py \
        --llm-url http://127.0.0.1:8080 \
        --prompt "Tell me a short joke." \
        --out out/demo.wav
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

# TTS engine imports (adjust paths)
PROJ = Path(__file__).resolve().parents[3]
GUI_SRC = PROJ / "gui" / "src"
ENGINE_SRC = PROJ / "audio_engines" / "gpa_1_5" / "src"
for p in (str(GUI_SRC), str(ENGINE_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from multimodalagent.audio.voice_changable import VoiceChangeConfig


def parse_sse(response):
    """Yield parsed JSON from an SSE response stream."""
    for line in response.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            yield json.loads(line[len("data: "):])


def split_on_sentence(text):
    """Yield (sentence, remainder) as sentences complete. Returns leftover."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(parts) <= 1:
        return [], text
    return parts[:-1], parts[-1]


def main():
    p = argparse.ArgumentParser(description="Stream LLM → TTS pipeline demo")
    p.add_argument("--llm-url", default="http://127.0.0.1:8080")
    p.add_argument("--prompt", default="Tell me a short joke about programming.")
    p.add_argument("--out", "-o", default="demos/out/demo.wav")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--voice", default="default", help="TTS voice id")
    p.add_argument("--tts-mode", default="balanced", choices=["quality", "balanced", "performance"])
    p.add_argument("--gap-ms", type=int, default=120, help="silence between sentences")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. Create LLM session + inject prompt ----
    print(f"[demo] creating LLM session at {args.llm_url}...", file=sys.stderr)
    r = requests.post(f"{args.llm_url}/sessions", timeout=30)
    r.raise_for_status()
    sid = r.json()["session_id"]
    print(f"[demo] session: {sid}", file=sys.stderr)

    chat_prompt = (
        f"<start_of_turn>user\n{args.prompt}\n<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    requests.post(
        f"{args.llm_url}/sessions/{sid}/inject",
        json={"text": chat_prompt}, timeout=60,
    )

    # ---- 3. Load TTS engine BEFORE starting the stream (so we don't miss ----
    #      early tokens while the TTS model loads) ----
    print("[demo] loading TTS engine...", file=sys.stderr)
    from tts_streaming_engine.engine import GPAStreamingTTSEngine

    tts = GPAStreamingTTSEngine(mode=args.tts_mode)
    tts.warmup()
    tts_sr = tts.info.sample_rate
    print(f"[demo] TTS ready (sr={tts_sr}, voice={args.voice})", file=sys.stderr)

    # ---- 4. Start streaming generation ----
    print(f"[demo] streaming generation (max {args.max_tokens} tokens)...", file=sys.stderr)
    gen_start = time.time()
    response = requests.post(
        f"{args.llm_url}/sessions/{sid}/generate",
        json={"stream": True, "max_tokens": args.max_tokens, "temperature": 0.7},
        stream=True, timeout=300,
    )
    response.raise_for_status()

    # ---- 4. Stream tokens → buffer → TTS per sentence ----
    buffer = ""
    full_text = ""
    audio_chunks = []
    ttft = None          # LLM time-to-first-token
    first_audio_time = None  # end-to-end time-to-first-audio
    n_sentences_ttsed = 0

    for event in parse_sse(response):
        if event.get("type") == "token":
            if ttft is None:
                ttft = time.time() - gen_start
                print(f"[demo] LLM first token: {ttft:.3f}s", file=sys.stderr)
            buffer += event["token"]
            full_text += event["token"]

            # check for sentence boundary
            sentences, buffer = split_on_sentence(buffer)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                # TTS this sentence
                for chunk in tts.synthesize_stream(sent):
                    if first_audio_time is None:
                        first_audio_time = time.time() - gen_start
                        print(f"[demo] first audio ready: {first_audio_time:.3f}s "
                              f"(TTFT={ttft:.3f}s + TTS={first_audio_time - ttft:.3f}s)",
                              file=sys.stderr)
                    audio_chunks.append(chunk.samples)
                n_sentences_ttsed += 1
                # add inter-sentence gap
                gap = np.zeros(int(tts_sr * args.gap_ms / 1000), dtype=np.float32)
                audio_chunks.append(gap)
                print(f"[demo]   TTS'd sentence {n_sentences_ttsed}: {sent[:60]!r}",
                      file=sys.stderr)

        elif event.get("type") == "done":
            done = event
            break
    else:
        done = {}

    # flush any remaining buffered text
    if buffer.strip():
        for chunk in tts.synthesize_stream(buffer.strip()):
            if first_audio_time is None:
                first_audio_time = time.time() - gen_start
            audio_chunks.append(chunk.samples)

    total_wall = time.time() - gen_start

    # cleanup LLM session
    try:
        requests.delete(f"{args.llm_url}/sessions/{sid}", timeout=10)
    except Exception:
        pass

    # ---- 5. Write audio + report ----
    if audio_chunks:
        audio = np.concatenate(audio_chunks)
        sf.write(str(out_path), audio, tts_sr)

    audio_dur = len(audio) / tts_sr if audio_chunks else 0
    llm_tok_s = done.get("tokens_per_s", 0)

    print(f"\n[demo] === PIPELINE SUMMARY ===", file=sys.stderr)
    print(f"  LLM TTFT:         {ttft:.3f}s" if ttft else "  LLM TTFT:         N/A", file=sys.stderr)
    print(f"  End-to-end TTFA:   {first_audio_time:.3f}s" if first_audio_time else
          "  End-to-end TTFA:   N/A", file=sys.stderr)
    print(f"  LLM tokens:        {done.get('n_tokens', '?')} ({llm_tok_s:.1f} tok/s)",
          file=sys.stderr)
    print(f"  Sentences TTS'd:   {n_sentences_ttsed}", file=sys.stderr)
    print(f"  Total audio:       {audio_dur:.2f}s", file=sys.stderr)
    print(f"  Total wall:        {total_wall:.2f}s", file=sys.stderr)
    print(f"  Pipeline RTF:      {total_wall / max(audio_dur, 0.001):.3f}", file=sys.stderr)
    print(f"  Output:            {out_path}", file=sys.stderr)
    print(f"  Full text:         {full_text[:200]!r}", file=sys.stderr)


if __name__ == "__main__":
    main()
