#!/usr/bin/env python3
"""Build the Stage 0 smoke corpus.

Stage 0 (docs/TRAINING.md) needs a hundred or so short utterances of
plain English with exact transcripts. We do not need LibriSpeech for
that: macOS ships an offline neural voice (`say`), so we can generate
a corpus in seconds, on the laptop, with zero network and zero disk.

Why this is enough for Stage 0:

* The audio is 16-bit clean speech at 16 kHz mono, which is what the
  FastConformer encoder expects.
* Every transcript is known exactly, so WER is meaningful from step
  one and there is no alignment ambiguity.
* The sentences are short and phonetically simple, which is exactly
  what an untrained bridge can be expected to latch onto first.

The corpus is written as a NeMo-style JSONL manifest (audio_filepath,
duration, text) so the same files feed the Stage 0 training loop and
the Stage 0 evaluation.

Usage:
    .venv/bin/python scripts/make_smoke_corpus.py
    .venv/bin/python scripts/make_smoke_corpus.py --split eval --voice Samantha
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

SAMPLE_RATE = 16000

# Short, simple, declarative English. Deliberately no proper nouns,
# no numbers, no rare words. This is the vocabulary an untrained
# projection can be expected to reproduce first.
TRAIN_SENTENCES = [
    "The sun rose over the hills.",
    "She opened the window and listened to the rain.",
    "The dog ran across the green field.",
    "We walked to the river after lunch.",
    "He put the book on the table.",
    "The train arrived at the station early.",
    "They planted trees along the road.",
    "A small bird sat on the fence.",
    "The water in the glass was cold.",
    "My brother plays the guitar every evening.",
    "The market opens at eight in the morning.",
    "It was a quiet night in the village.",
    "The children ran home before the storm.",
    "She wrote a letter to her friend.",
    "The old man sat alone on the bench.",
    "Clouds moved slowly across the sky.",
    "He carried the heavy box upstairs.",
    "The garden was full of red flowers.",
    "We finished the work before lunch.",
    "The cat sat quietly by the fire.",
]

EVAL_SENTENCES = [
    "The light faded slowly behind the hills.",
    "A woman walked along the quiet shore.",
    "The teacher spoke to the class for an hour.",
    "Rain fell gently on the roof.",
    "He turned off the lamp and went to bed.",
    "The road curved past the old church.",
    "They sat on the steps and talked.",
    "A bird sang somewhere in the dark.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the Stage 0 smoke corpus")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "smoke",
        help="Output directory (default: data/smoke)",
    )
    p.add_argument("--voice", default="Alex", help="macOS voice name (default: Alex)")
    p.add_argument("--rate", type=int, default=170, help="Speech rate in words per minute")
    p.add_argument(
        "--leading-silence",
        type=float,
        default=0.25,
        help="Seconds of silence prepended (default: 0.25)",
    )
    p.add_argument(
        "--trailing-silence",
        type=float,
        default=0.45,
        help="Seconds of silence appended, so the final word is fully captured "
        "and the TDT-style encoder is not cut mid-phone (default: 0.45)",
    )
    p.add_argument(
        "--split",
        choices=["train", "eval", "both"],
        default="both",
        help="Which split(s) to generate (default: both)",
    )
    return p.parse_args()


def synthesize(
    text: str,
    out_path: Path,
    *,
    voice: str,
    rate: int,
    lead: float,
    trail: float,
) -> float:
    """Render one sentence to a 16 kHz mono wav. Returns duration."""
    aiff = out_path.with_suffix(".aiff")
    subprocess.run(
        ["say", "-v", voice, "-r", str(rate), "-o", str(aiff), text],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(aiff),
            "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-af",
            # Trim the trailing dead air macOS TTS appends, then pad
            # ourselves so both edges are clean.
            f"silenceremove=start_periods=1:start_silence=0:start_threshold=-45dB,"
            f"areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=-45dB,"
            f"areverse,"
            f"adelay={int(lead * 1000)}|{int(lead * 1000)},"
            f"apad=pad_dur={trail}",
            "-c:a", "pcm_s16le", str(out_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    aiff.unlink(missing_ok=True)

    # Duration from the wav header, stdlib only: this script has to run
    # before the venv exists (it is what builds the first real audio).
    with wave.open(str(out_path), "rb") as fh:
        frames = fh.getnframes()
        rate = fh.getframerate()
    return frames / float(rate)


def build_split(
    name: str,
    sentences: list[str],
    out_dir: Path,
    *,
    voice: str,
    rate: int,
    lead: float,
    trail: float,
) -> Path:
    audio_dir = out_dir / name / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, sentence in enumerate(sentences):
        wav_path = audio_dir / f"{name}_{i:03d}.wav"
        duration = synthesize(
            sentence, wav_path, voice=voice, rate=rate, lead=lead, trail=trail
        )
        rows.append(
            {
                "audio_filepath": str(wav_path.resolve()),
                "duration": round(duration, 3),
                "text": sentence,
            }
        )
        print(f"  [{name}] {wav_path.name}  {duration:5.2f}s  {sentence}")

    manifest = out_dir / f"{name}.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    total = sum(r["duration"] for r in rows)
    print(f"  [{name}] {len(rows)} clips, {total:.1f}s -> {manifest}")
    return manifest


def main() -> int:
    args = parse_args()
    if shutil.which("say") is None:
        print("macOS `say` not found; this script is macOS-only.", file=sys.stderr)
        return 1
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH.", file=sys.stderr)
        return 1

    print("=" * 60)
    print("Stage 0 smoke corpus")
    print("=" * 60)
    print(f"voice: {args.voice} @ {args.rate} wpm")
    print(f"out:   {args.out_dir}")
    print()

    if args.split in ("train", "both"):
        print("train split:")
        build_split(
            "train",
            TRAIN_SENTENCES,
            args.out_dir,
            voice=args.voice,
            rate=args.rate,
            lead=args.leading_silence,
            trail=args.trailing_silence,
        )
        print()
    if args.split in ("eval", "both"):
        print("eval split:")
        build_split(
            "eval",
            EVAL_SENTENCES,
            args.out_dir,
            voice=args.voice,
            rate=args.rate,
            lead=args.leading_silence,
            trail=args.trailing_silence,
        )
    print()
    print("OK: corpus ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())