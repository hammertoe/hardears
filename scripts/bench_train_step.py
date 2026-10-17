#!/usr/bin/env python3
"""Benchmark one training step at Stage 1 shapes.

The question this answers: can Stage 1 run on the laptop, or does it
need a rented GPU?

It measures the thing that actually decides it — a forward + backward
pass of the real LLM at Stage 1 sequence lengths, with gradients only
on the bridge and the LoRA — and reports seconds per step and peak
memory. From those you can project wall-clock for a 5,000-step run.

Shapes measured:

    smoke     batch 1, ~70 tokens   (what Stage 0 did)
    stage1    batch 8, ~3,400 tokens (Stage 1: 30 s windows, bs 8)

Memory is the real constraint on the laptop, not speed. A 30 s window
is 375 encoder frames, so batch 8 is roughly 8 x (24 text + 375 audio
+ 24 text) tokens. At bf16 on a 4B model that is an order of magnitude
more activation memory than the Stage 0 shape, which is why Stage 0
fitting comfortably says nothing about Stage 1 fitting.

Usage:
    .venv/bin/python scripts/bench_train_step.py --llm-path artifacts/qwen3.5-2b
    .venv/bin/python scripts/bench_train_step.py --llm-path artifacts/qwen3.5-4b \\
        --shapes stage1 --batch 8 --steps 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bim_asr.model import (  # noqa: E402
    BimAsrConfig,
    BimAsrModel,
    build_salm_prompt_ids,
    stop_token_ids,
)

HIDDEN_PER_FRAME = 1  # one 1024-dim frame projected to one llm_hidden token


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark a SALM training step")
    p.add_argument("--llm-path", default=str(ROOT / "artifacts" / "qwen3.5-2b"))
    p.add_argument("--device", default="mps")
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--audio-seconds", type=float, default=30.0,
                   help="Audio per window; Stage 1 uses 30 s")
    p.add_argument("--text-tokens", type=int, default=12,
                   help="Target transcript tokens per window")
    p.add_argument("--steps", type=int, default=3, help="Timed steps after warmup")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--shapes", default="smoke,stage1")
    p.add_argument("--projection-steps", type=int, default=5000,
                   help="Projection used to project the total runtime")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def peak_memory_mb(device: str) -> float:
    if device == "mps":
        return torch.mps.current_allocated_memory() / 1024**2
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


def reset_memory(device: str) -> None:
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def bench_shape(model, prefix_emb, suffix_emb, *, batch, audio_frames, text_tokens,
                args) -> dict:
    """One synthetic step at the given shape."""
    device = next(model.projection.parameters()).device
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    features = torch.randn(batch, audio_frames, model.config.encoder_hidden, device=device)
    target = torch.randint(1000, 2000, (batch, text_tokens), device=device)

    prefix_ids = prefix_emb.repeat(batch, 1)
    suffix_ids = suffix_emb.repeat(batch, 1)

    audio_emb = model.encode_audio(features)
    text = model.splice_audio_into_prompt(audio_emb, prefix_ids, suffix_ids)
    target_emb = model.embed_tokens(target)
    prompt_len = text.shape[1]
    tokens = int((prompt_len + text_tokens) * batch)
    reset_memory(args.device)

    times = []
    for i in range(args.warmup + args.steps):
        # Rebuilt every step: the graph is consumed by backward, and a
        # real training loop re-runs the forward pass each iteration.
        embeds = torch.cat(
            [
                model.splice_audio_into_prompt(
                    model.encode_audio(features), prefix_ids, suffix_ids
                ),
                model.embed_tokens(target),
            ],
            dim=1,
        ).to(dtype)
        labels = torch.cat(
            [
                torch.full((batch, prompt_len), -100, dtype=torch.long, device=device),
                target,
            ],
            dim=1,
        )

        if device.type == "mps":
            torch.mps.synchronize()
        started = time.time()
        out = model.llm(inputs_embeds=embeds, labels=labels)
        out.loss.backward()
        for param in model.projection.parameters():
            param.grad = None
        for name, param in model.llm.named_parameters():
            if param.grad is not None and "lora_" not in name:
                param.grad = None
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.time() - started
        if i >= args.warmup:
            times.append(elapsed)
        del out, embeds

    per_step = sorted(times)[len(times) // 2] if times else float("nan")
    peak = peak_memory_mb(args.device)
    hours = per_step * args.projection_steps / 3600 if per_step == per_step else float("nan")

    return {
        "batch": batch,
        "audio_frames_per_window": audio_frames,
        "sequence_tokens": int(prompt_len + text_tokens),
        "tokens_per_step": tokens,
        "seconds_per_step": round(per_step, 3),
        "peak_memory_mb": round(peak, 1),
        f"hours_for_{args.projection_steps}_steps": round(hours, 2),
    }


def main() -> int:
    args = parse_args()
    if not Path(args.llm_path).exists():
        print(f"no LLM snapshot at {args.llm_path}")
        print("  hf download Qwen/Qwen3.5-2B --local-dir artifacts/qwen3.5-2b")
        return 1

    cfg = BimAsrConfig.for_llm(str(args.llm_path))
    print("=" * 72)
    print("SALM training-step benchmark")
    print("=" * 72)
    print(f"llm:    {cfg.llm_id} ({cfg.llm_hidden} hidden)")
    print(f"device: {args.device}  dtype: {args.dtype}")
    print(f"bridge: {cfg.encoder_hidden} -> {cfg.llm_hidden}")
    print()

    started = time.time()
    model = BimAsrModel.from_pretrained(
        cfg, device=args.device,
        dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32,
    )
    if args.gradient_checkpointing:
        model.llm.gradient_checkpointing_enable()
        model.llm.config.use_cache = False
    model.train()
    print(f"model loaded in {time.time() - started:.1f}s, "
          f"trainable {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    tok = model.tokenizer
    prefix, suffix = build_salm_prompt_ids(
        tok, user_prompt=cfg.user_prompt, audio_locator_tag=cfg.audio_locator_tag
    )
    prefix_emb = prefix.unsqueeze(0)
    suffix_emb = suffix.unsqueeze(0)

    frame_rate = cfg.audio_frame_rate_hz
    shapes = {
        # what Stage 0 did: 2.5 s clips, batch 1
        "smoke": {"batch": 1, "audio_frames": int(2.5 * frame_rate), "text_tokens": 8},
        # LibriSpeech utterances average ~7 s, which is the realistic
        # laptop configuration: one real utterance per step
        "libri1": {"batch": 1, "audio_frames": int(7.0 * frame_rate), "text_tokens": 12},
        "libri4": {"batch": 4, "audio_frames": int(7.0 * frame_rate), "text_tokens": 12},
        # Stage 1 as written in docs/TRAINING.md: 30 s windows, batch 8
        "stage1": {
            "batch": args.batch,
            "audio_frames": int(args.audio_seconds * frame_rate),
            "text_tokens": args.text_tokens,
        },
    }

    results = []
    for name in args.shapes.split(","):
        name = name.strip()
        if name not in shapes:
            print(f"unknown shape {name}")
            continue
        spec = shapes[name]
        print(f"\n--- {name} ---")
        try:
            result = bench_shape(
                model, prefix_emb, suffix_emb, args=args, **spec
            )
        except (RuntimeError, torch.OutOfMemoryError) as exc:
            print(f"  FAILED: {exc.__class__.__name__}: {exc}")
            results.append({"shape": name, **spec, "failed": str(exc)[:200]})
            continue
        results.append({"shape": name, **result})
        print(f"  batch {result['batch']}, {result['sequence_tokens']} tokens/window, "
              f"{result['tokens_per_step']:,} tokens/step")
        print(f"  {result['seconds_per_step']:.2f} s/step, peak {result['peak_memory_mb']:.0f} MB")
        print(f"  -> {result[f'hours_for_{args.projection_steps}_steps']:.1f} h for "
              f"{args.projection_steps:,} steps")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"llm": cfg.llm_id, "device": args.device, "dtype": args.dtype,
             "gradient_checkpointing": args.gradient_checkpointing,
             "results": results}, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())