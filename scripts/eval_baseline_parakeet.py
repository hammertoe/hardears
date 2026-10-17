#!/usr/bin/env python3
"""Reference ASR: decode the smoke corpus with the original Parakeet TDT head.

Why this exists
---------------

Two reasons, both about honesty.

1. **It validates the extracted encoder.** Bim-ASR drops the TDT
   decoder, so nothing in the SALM path can tell you whether the
   extracted encoder is *correct* or merely *loadable*. The full TDT
   model shares the encoder. If the original ASR head still decodes
   sensible text off our extracted features, the extraction is right.

2. **It gives the SALM a number to be measured against.** Stage 0 is
   a 300-clip toy corpus with a Qwen3.5 base that has never heard an
   audio token. If this prints 0% WER and the SALM prints 60%, the
   honest conclusion is "the bridge learns, the SALM is nowhere near
   a real ASR", not "the SALM is fine".

Usage:
    .venv/bin/python scripts/eval_baseline_parakeet.py
    .venv/bin/python scripts/eval_baseline_parakeet.py --manifest data/smoke/eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TDT = ROOT / "artifacts" / "parakeet-tdt-hf"
DEFAULT_MANIFEST = ROOT / "data" / "smoke" / "eval.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parakeet TDT baseline on the smoke corpus")
    p.add_argument("--tdt", type=Path, default=DEFAULT_TDT)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--show", type=int, default=8, help="How many examples to print")
    p.add_argument(
        "--out", type=Path, default=None, help="Optional path to write metrics JSON"
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not (args.tdt / "model.safetensors").exists():
        print(
            f"no converted TDT model at {args.tdt}.\n"
            "Regenerate it with:\n"
            "  .venv/bin/python scripts/prepare_parakeet_encoder.py \\\n"
            "      --nemo artifacts/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo \\\n"
            "      --out artifacts/parakeet-encoder-v2 \\\n"
            "      --also-tdt artifacts/parakeet-tdt-hf"
        )
        return 1

    import soundfile as sf
    from transformers import ParakeetForTDT, ParakeetProcessor

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    print("=" * 68)
    print("Parakeet TDT 0.6B v2 (encoder + original decoder) reference ASR")
    print("=" * 68)
    print(f"manifest: {args.manifest}  ({len(rows)} clips)")
    print()

    processor = ParakeetProcessor.from_pretrained(str(args.tdt))
    model = ParakeetForTDT.from_pretrained(args.tdt, dtype=torch.float32)
    model.eval().to(args.device)

    hypotheses: list[str] = []
    references: list[str] = []
    t0 = time.time()
    with torch.no_grad():
        for i, row in enumerate(rows):
            wav, sr = sf.read(row["audio_filepath"], dtype="float32")
            inputs = processor(
                audio=wav, sampling_rate=sr, return_tensors="pt", return_attention_mask=True
            )
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            out = model.generate(**inputs, max_new_tokens=64)
            hypotheses.append(processor.batch_decode(out.sequences)[0].strip())
            references.append(row["text"])
            if (i + 1) % 25 == 0 or i + 1 == len(rows):
                print(f"  {i + 1}/{len(rows)} ({time.time() - t0:.0f}s)")

    import jiwer

    if hasattr(jiwer, "process_words"):
        measures = jiwer.process_words(references, hypotheses)
        get = lambda k: getattr(measures, k)  # noqa: E731
    else:  # pragma: no cover
        measures = jiwer.compute_measures(references, hypotheses)
        get = lambda k: measures[k]  # noqa: E731

    exact = sum(1 for h, r in zip(hypotheses, references) if h == r)

    print()
    for r, h in list(zip(references, hypotheses))[: args.show]:
        mark = "exact" if r == h else "differs"
        print(f"  [{mark}] {r}")
        if r != h:
            print(f"           {h}")
    print()

    result = {
        "model": "nvidia/parakeet-tdt-0.6b-v2",
        "clips": len(rows),
        "wer": get("wer"),
        "mer": get("mer"),
        "substitutions": get("substitutions"),
        "deletions": get("deletions"),
        "insertions": get("insertions"),
        "exact_match": exact,
        "seconds": time.time() - t0,
        "manifest": str(args.manifest),
    }
    print(f"WER:  {result['wer'] * 100:.1f}%")
    print(f"MER:  {result['mer'] * 100:.1f}%")
    print(f"exact: {exact}/{len(rows)} clips")
    print(f"realtime factor: {sum(r['duration'] for r in rows) / result['seconds']:.2f}x "
          f"({args.device}, fp32, no batching)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())