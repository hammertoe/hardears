#!/usr/bin/env python3
"""Bim-ASR transcription CLI.

Loads the frozen Parakeet encoder, the projection bridge and the
Qwen3.5 LLM with its LoRA adapter, and prints the transcript.

The implementation is `bim_asr.inference.SalmTranscriber`. This script
is the argument parsing and the reporting.

Examples
--------

    # untrained bridge + fresh LoRA (shows the wiring, not a transcript)
    python scripts/transcribe.py path/to/audio.wav

    # a Stage 0 trained checkpoint
    python scripts/transcribe.py path/to/audio.wav \
        --llm-path artifacts/qwen3.5-2b \
        --encoder-path artifacts/parakeet-encoder-v2 \
        --checkpoint artifacts/stage0-ckpt

    # several files, with timings
    python scripts/transcribe.py clips/*.wav --checkpoint artifacts/stage0-ckpt --timing
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bim_asr.inference import SalmTranscriber  # noqa: E402

DEFAULT_LLM = ROOT / "artifacts" / "qwen3.5-2b"
DEFAULT_ENCODER = ROOT / "artifacts" / "parakeet-encoder-v2"
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "stage0-ckpt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bim-ASR transcription")
    p.add_argument("audio", nargs="+", help="One or more audio files (wav/mp3/m4a/...)")
    p.add_argument(
        "--llm-path",
        default=str(DEFAULT_LLM),
        help="Local path or Hub id for the LLM (default: artifacts/qwen3.5-2b)",
    )
    p.add_argument(
        "--encoder-path",
        default=str(DEFAULT_ENCODER),
        help="Directory holding the extracted Parakeet encoder",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Trained Stage 0 checkpoint. Without it the bridge is random "
        "and the output is not a transcript.",
    )
    p.add_argument("--device", default="cpu", help="cpu or mps")
    p.add_argument("--dtype", choices=["bfloat16", "float32", "float16"], default="bfloat16")
    p.add_argument("--window-seconds", type=float, default=30.0)
    p.add_argument("--overlap-seconds", type=float, default=5.0)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--timing", action="store_true", help="Print RTFx per file")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    missing = [
        path for path in (args.llm_path, args.encoder_path)
        if not Path(path).exists() and "/" not in str(path)
    ]
    if missing:
        print(
            "missing artefacts: "
            + ", ".join(missing)
            + "\nRun scripts/prepare_parakeet_encoder.py and hf download "
              "Qwen/Qwen3.5-2B --local-dir artifacts/qwen3.5-2b first.",
            file=sys.stderr,
        )
        return 2

    for path in args.audio:
        if not Path(path).exists():
            print(f"audio file not found: {path}", file=sys.stderr)
            return 2

    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint and not checkpoint.exists():
        print(f"checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2

    print(
        f"loading SALM (llm={args.llm_path}, encoder={args.encoder_path}"
        f"{', checkpoint=' + str(checkpoint) if checkpoint else ', untrained bridge'})",
        file=sys.stderr,
    )
    import torch

    transcriber = SalmTranscriber(
        llm_path=args.llm_path,
        encoder_path=args.encoder_path,
        checkpoint=checkpoint,
        device=args.device,
        dtype=getattr(torch, args.dtype),
        window_seconds=args.window_seconds,
        overlap_seconds=args.overlap_seconds,
        max_new_tokens=args.max_new_tokens,
    )
    print("ready", file=sys.stderr)

    if checkpoint is None:
        print(
            "NOTE: no --checkpoint given. The projection bridge is randomly "
            "initialised, so this prints what the base LLM says when handed "
            "noise. It is not a transcript.",
            file=sys.stderr,
        )

    started = time.time()
    total_audio = 0.0
    for path in args.audio:
        result = transcriber.transcribe_file(path)
        total_audio += result.audio_seconds
        print(f"\n=== {Path(path).name} ({result.audio_seconds:.2f}s audio) ===")
        if not result.text:
            print("(no text generated)")
            continue
        print(result.text)
        if args.timing:
            print(
                f"    [{result.chunks} window(s), {result.seconds:.1f}s, "
                f"RTFx {result.realtime_factor:.2f}x, {args.device}]",
                file=sys.stderr,
            )

    if args.timing and len(args.audio) > 1:
        elapsed = time.time() - started
        print(
            f"\n{len(args.audio)} files, {total_audio:.1f}s audio in {elapsed:.1f}s "
            f"= {total_audio / max(elapsed, 1e-6):.2f}x RTFx",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())