#!/usr/bin/env python3
"""Bim-ASR Stage 0 training: train the bridge (+ LoRA) on the laptop.

docs/TRAINING.md stage 0 is "prove the SALM scaffolding works end to
end on the laptop, see loss go down over a few hundred steps". This
script does that with the *real* frozen Parakeet encoder and the *real*
Qwen3.5 LLM, on Apple Silicon, in bf16, with only the projection and
the LoRA adapters trainable (~26M of 1.9B parameters).

It is not a substitute for stage 1. It is the cheapest possible proof
that the whole stack learns: audio goes in at one end, text comes out
at the other, and the only thing being updated is the bridge.

What is trainable
-----------------
    projection.proj.weight   1024 x 2048   (the bridge)
    projection.norm.*        LayerNorm
    lora_A / lora_B         rank 32 on q/k/v/o + gate/up/down_proj

Everything else -- the 608M-parameter FastConformer encoder and the
1.88B LLM -- stays frozen, exactly as in the training plan.

Efficiency notes
----------------
The encoder is frozen, so its output features are computed once and
cached to `data/smoke/features.pt`. That removes ~0.8 s of CPU per
clip from every step; the remaining cost is the LLM forward/backward
at roughly 0.45 s/step on an M4 Pro.

Usage
-----
    .venv/bin/python scripts/train_stage0.py --steps 400
    .venv/bin/python scripts/train_stage0.py --steps 6000 --max-minutes 150
    .venv/bin/python scripts/train_stage0.py --eval-only \
        --checkpoint artifacts/stage0-ckpt
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))

from bim_asr.model import (  # noqa: E402
    BimAsrConfig,
    BimAsrModel,
    build_salm_prompt_ids,
    stop_token_ids,
)
from bim_asr.parakeet import load_parakeet_encoder  # noqa: E402

DEFAULT_LLM = ROOT / "artifacts" / "qwen3.5-2b"
DEFAULT_ENCODER = ROOT / "artifacts" / "parakeet-encoder-v2"
DEFAULT_TRAIN = ROOT / "data" / "smoke" / "train.jsonl"
DEFAULT_EVAL = ROOT / "data" / "smoke" / "eval.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bim-ASR Stage 0 training")
    p.add_argument("--llm-path", default=str(DEFAULT_LLM))
    p.add_argument("--encoder-path", default=str(DEFAULT_ENCODER))
    p.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-manifest", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "stage0-ckpt")
    p.add_argument("--device", default="mps")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="Stop cleanly after this many minutes, saving a checkpoint",
    )
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument(
        "--sample-every", type=int, default=200, help="Generate a sample transcript every N steps"
    )
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--max-audio-seconds", type=float, default=30.0)
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; evaluate a checkpoint (needs --checkpoint)",
    )
    p.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint to evaluate/continue")
    p.add_argument(
        "--dump-predictions",
        type=Path,
        default=None,
        help="Write every reference/hypothesis pair to this JSON file. One "
        "runaway generation moves an average WER by tens of points, so the "
        "aggregate alone is not readable.",
    )
    p.add_argument("--eval-limit", type=int, default=24, help="Clips used for eval WER")
    p.add_argument("--greedy", action="store_true", default=True)
    return p.parse_args()


def read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/make_smoke_corpus.py first")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_audio(path: str) -> np.ndarray:
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        raise RuntimeError(f"{path} is {sr} Hz, expected 16 kHz")
    return wav


def build_feature_cache(
    rows: list[dict], encoder, cache_path: Path, *, device: str
) -> dict[str, torch.Tensor]:
    """Encode every clip once. The encoder is frozen, so this is exact."""
    if cache_path.exists():
        blob = torch.load(cache_path, map_location="cpu", weights_only=True)
        missing = [r["audio_filepath"] for r in rows if r["audio_filepath"] not in blob]
        if not missing:
            print(f"features: reusing cache {cache_path} ({len(blob)} clips)")
            return blob
        print(f"features: cache is missing {len(missing)} clips, re-encoding")

    print(f"features: encoding {len(rows)} clips with the frozen encoder...")
    blob: dict[str, torch.Tensor] = {}
    t0 = time.time()
    with torch.no_grad():
        for i, row in enumerate(rows):
            wav = load_audio(row["audio_filepath"])
            feats, _ = encoder(torch.from_numpy(wav).unsqueeze(0))
            blob[row["audio_filepath"]] = feats[0].to(torch.float32).cpu()
            if (i + 1) % 50 == 0 or i + 1 == len(rows):
                print(f"  {i + 1}/{len(rows)}  ({time.time() - t0:.0f}s)")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, cache_path)
    print(f"features: cached {len(blob)} clips -> {cache_path}")
    return blob


def tokenize_targets(tokenizer, rows: list[dict]) -> list[list[int]]:
    """Reference + the token generation must stop on.

    The stop id comes from the tokenizer's own eos_token_id (248046,
    `<|im_end|>`, the ChatML end-of-turn), because that is what
    `generate` is told to stop on. Appending `<|endoftext|>` here
    instead -- which is what the string form resolves to, since the
    tokenizer reports `eos_token` as an empty string -- trains the model
    to stop on a different token than inference watches for.
    """
    eos_id = tokenizer.eos_token_id
    out = []
    for row in rows:
        ids = tokenizer(row["text"], add_special_tokens=False).input_ids
        out.append(ids + [int(eos_id)])
    return out


class Stage0Trainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        self.cfg = BimAsrConfig.for_llm(args.llm_path)
        print("=" * 68)
        print("Bim-ASR Stage 0 training")
        print("=" * 68)
        print(f"llm:     {self.cfg.llm_id} (hidden {self.cfg.llm_hidden})")
        print(f"encoder: {args.encoder_path}")
        print(f"device:  {args.device}   dtype: bf16")
        print(f"steps:   {args.steps}   lr: {args.lr}   warmup: {args.warmup}")
        print()

        self.model = BimAsrModel.from_pretrained(self.cfg, device=args.device)
        self.tokenizer = self.model.tokenizer
        self.stop_ids = stop_token_ids(self.tokenizer)
        self.trainable = [p for p in self.model.parameters() if p.requires_grad]
        print(f"trainable parameters: {sum(p.numel() for p in self.trainable):,}")
        print(f"frozen parameters:    {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"stop tokens:          {self.stop_ids}")
        print()

    def load_llm_state(self, checkpoint: Path) -> None:
        """Resume a Stage 0 checkpoint into an already-wrapped model.

        The adapter is loaded with `set_peft_model_state_dict` rather
        than `PeftModel.from_pretrained`: by this point `self.model.llm`
        is already a PeftModel carrying fresh zero-initialised LoRA
        weights, so re-wrapping it would be wrong.
        """
        proj_path = checkpoint / "projection.pt"
        if proj_path.exists():
            state = torch.load(proj_path, map_location="cpu", weights_only=True)
            self.model.projection.load_state_dict(state)
            print(f"loaded projection from {proj_path}")

        adapter = checkpoint / "lora_adapter"
        weights = adapter / "adapter_model.safetensors"
        if adapter.exists() and weights.exists():
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file

            state = load_file(str(weights))
            result = set_peft_model_state_dict(self.model.llm, state)
            unexpected = list(getattr(result, "unexpected_keys", []) or [])
            if unexpected:
                raise RuntimeError(
                    f"unexpected keys in the LoRA adapter: {unexpected[:5]}. "
                    "The adapter does not match this target-module list."
                )
            self.model.llm.to(self.device)
            self.trainable = [p for p in self.model.parameters() if p.requires_grad]
            print(
                f"loaded LoRA adapter from {adapter} "
                f"({sum(p.numel() for p in self.trainable):,} trainable)"
            )

    def make_example(self, features: torch.Tensor, target_ids: list[int],
                     prefix: torch.Tensor, suffix: torch.Tensor) -> dict:
        device = self.device
        feats = features.to(device)
        target = torch.tensor([target_ids], dtype=torch.long, device=device)

        prefix_ids = prefix.unsqueeze(0).to(device)
        suffix_ids = suffix.unsqueeze(0).to(device)

        audio_emb = self.model.encode_audio(feats.unsqueeze(0))
        text_emb = self.model.splice_audio_into_prompt(audio_emb, prefix_ids, suffix_ids)
        target_emb = self.model.embed_tokens(target)
        inputs_embeds = torch.cat([text_emb, target_emb], dim=1)

        # Mask the prompt and the audio: loss is on the transcript only.
        prompt_len = text_emb.shape[1]
        labels = torch.cat(
            [
                torch.full((1, prompt_len), -100, dtype=torch.long, device=device),
                target,
            ],
            dim=1,
        )
        return {"inputs_embeds": inputs_embeds, "labels": labels}

    def loss_for(self, example: dict) -> torch.Tensor:
        out = self.model.llm(**example)
        return out.loss

    @torch.no_grad()
    def generate(self, features: torch.Tensor, prefix: torch.Tensor, suffix: torch.Tensor,
                 max_new_tokens: int = 32) -> str:
        was_training = self.model.training
        self.model.eval()
        device = self.device
        audio_emb = self.model.encode_audio(features.to(device).unsqueeze(0))
        embeds = self.model.splice_audio_into_prompt(
            audio_emb, prefix.unsqueeze(0).to(device), suffix.unsqueeze(0).to(device)
        )
        out = self.model.llm.generate(
            inputs_embeds=embeds,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.stop_ids,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if was_training:
            self.model.train()
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def save(self, out_dir: Path, metrics: dict) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.model.projection.state_dict(), out_dir / "projection.pt"
        )
        self.model.llm.save_pretrained(str(out_dir / "lora_adapter"))
        with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"saved checkpoint -> {out_dir}")


def main() -> int:
    args = parse_args()
    started = time.time()

    trainer = Stage0Trainer(args)
    if args.dump_predictions:
        trainer.dump_predictions = args.dump_predictions
    model = trainer.model
    tokenizer = trainer.tokenizer
    prefix, suffix = build_salm_prompt_ids(
        tokenizer,
        user_prompt=trainer.cfg.user_prompt,
        audio_locator_tag=trainer.cfg.audio_locator_tag,
    )

    if args.checkpoint:
        trainer.load_llm_state(args.checkpoint)

    encoder = load_parakeet_encoder(args.encoder_path, device=args.device)
    for p in encoder.parameters():
        p.requires_grad_(False)

    train_rows = read_manifest(args.train_manifest)
    eval_rows = read_manifest(args.eval_manifest)
    cache_path = args.train_manifest.parent / "features.pt"
    features = build_feature_cache(
        train_rows + eval_rows, encoder, cache_path, device=args.device
    )

    train_targets = tokenize_targets(tokenizer, train_rows)
    eval_targets = tokenize_targets(tokenizer, eval_rows[: args.eval_limit])

    if args.eval_only:
        wer = evaluate(trainer, eval_rows[: args.eval_limit], eval_targets, features, prefix, suffix)
        print(f"\neval WER (greedy, {args.eval_limit} clips): {wer['wer'] * 100:.1f}%")
        return 0

    # Training
    print()
    print("-" * 68)
    optimizer = torch.optim.AdamW(
        trainer.trainable, lr=args.lr, weight_decay=args.weight_decay
    )

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    order = list(range(len(train_rows)))
    random.shuffle(order)
    cursor = 0
    losses: list[float] = []
    recent: list[float] = []
    model.train()
    stopped_early = None

    for step in range(args.steps):
        if cursor >= len(order):
            random.shuffle(order)
            cursor = 0
        idx = order[cursor]
        cursor += 1
        row = train_rows[idx]

        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)

        example = trainer.make_example(
            features[row["audio_filepath"]], train_targets[idx], prefix, suffix
        )
        loss = trainer.loss_for(example)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(trainer.trainable, args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        value = loss.item()
        losses.append(value)
        recent.append(value)

        if (step + 1) % args.log_every == 0:
            mean = sum(recent) / len(recent)
            elapsed = time.time() - started
            rate = (step + 1) / elapsed
            print(
                f"step {step + 1:>5}/{args.steps}  loss {mean:.4f}  "
                f"lr {lr_at(step):.2e}  |g| {float(grad_norm):.2f}  "
                f"{rate:.2f} it/s  elapsed {elapsed / 60:.1f}m"
            )
            recent = []

        if args.sample_every and (step + 1) % args.sample_every == 0:
            sample_row = eval_rows[0]
            text = trainer.generate(
                features[sample_row["audio_filepath"]], prefix, suffix
            )
            print(f"    audio : {sample_row['text']!r}")
            print(f"    SALM  : {text!r}")

        if args.save_every and (step + 1) % args.save_every == 0:
            trainer.save(
                args.out,
                {
                    "steps_completed": step + 1,
                    "loss_last50": sum(losses[-50:]) / len(losses[-50:]),
                    "loss_curve_every_20": [
                        sum(losses[i : i + 20]) / len(losses[i : i + 20])
                        for i in range(0, len(losses), 20)
                    ],
                    "elapsed_minutes": (time.time() - started) / 60,
                },
            )
            with (args.out / "loss_curve.jsonl").open("a", encoding="utf-8") as fh:
                for i in range(0, len(losses), args.log_every):
                    fh.write(
                        json.dumps(
                            {"step": i + args.log_every, "loss": sum(losses[i : i + args.log_every]) / args.log_every}
                        )
                        + "\n"
                    )

        if args.max_minutes and (time.time() - started) / 60 > args.max_minutes:
            stopped_early = step + 1
            print(f"\ntime budget of {args.max_minutes} minutes reached at step {step + 1}")
            break

    # Final save + evaluation
    metrics = {
        "steps_completed": stopped_early or args.steps,
        "requested_steps": args.steps,
        "loss_final50": sum(losses[-50:]) / len(losses[-50:]) if losses else None,
        "loss_initial50": sum(losses[:50]) / len(losses[:50]) if losses else None,
        "minutes": (time.time() - started) / 60,
        "llm": trainer.cfg.llm_id,
        "encoder_hidden": trainer.cfg.encoder_hidden,
        "llm_hidden": trainer.cfg.llm_hidden,
        "trainable_parameters": sum(p.numel() for p in trainer.trainable),
        "train_clips": len(train_rows),
        "eval_clips": args.eval_limit,
    }
    trainer.save(args.out, metrics)

    print()
    print("=" * 68)
    print("Evaluation on held-out clips (greedy decode)")
    print("=" * 68)
    wer = evaluate(trainer, eval_rows[: args.eval_limit], eval_targets, features, prefix, suffix)
    metrics["eval"] = wer
    trainer.save(args.out, metrics)

    print()
    print(f"eval WER: {wer['wer'] * 100:.1f}%  over {wer['clips']} held-out clips")
    print(f"total time: {(time.time() - started) / 60:.1f} min")
    return 0


def evaluate(trainer, rows, targets, features, prefix, suffix) -> dict:
    """Greedy decode every clip and compute WER against the reference."""
    import jiwer

    model = trainer.model
    was_training = model.training
    model.eval()
    hypotheses = []
    references = []
    t0 = time.time()
    for i, row in enumerate(rows):
        feats = features[row["audio_filepath"]]
        hypotheses.append(
            trainer.generate(feats, prefix, suffix, max_new_tokens=40)
        )
        references.append(row["text"])
        if (i + 1) % 8 == 0 or i + 1 == len(rows):
            print(f"  decoded {i + 1}/{len(rows)} ({time.time() - t0:.0f}s)")

    # jiwer 4.x exposes process_words(); 3.x exposed compute_measures().
    if hasattr(jiwer, "process_words"):
        measures = jiwer.process_words(references, hypotheses)
    else:  # pragma: no cover
        measures = jiwer.compute_measures(references, hypotheses)
    # jiwer 3 returned a dict, 4 returns a dataclass.
    get = (lambda k: measures[k]) if isinstance(measures, dict) else (
        lambda k: getattr(measures, k)
    )
    exact = sum(1 for h, r in zip(hypotheses, references) if h.strip() == r.strip())
    empty = sum(1 for h in hypotheses if not h.strip())

    print()
    for i in range(min(6, len(rows))):
        print(f"  ref: {references[i]}")
        print(f"  hyp: {hypotheses[i]}")
        print()

    dump = getattr(trainer, "dump_predictions", None)
    if dump is not None:
        dump.parent.mkdir(parents=True, exist_ok=True)
        with dump.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "clips": len(rows),
                    "wer": get("wer"),
                    "mer": get("mer"),
                    "substitutions": get("substitutions"),
                    "deletions": get("deletions"),
                    "insertions": get("insertions"),
                    "exact_match": exact,
                    "empty_hypotheses": empty,
                    "pairs": [
                        {"ref": r, "hyp": h}
                        for r, h in zip(references, hypotheses)
                    ],
                },
                fh,
                indent=2,
            )
        print(f"  wrote {len(rows)} prediction pairs -> {dump}")

    # Per-clip WER, so a single runaway generation is visible instead of
    # hiding inside the average.
    per_clip = []
    for r, h in zip(references, hypotheses):
        if hasattr(jiwer, "process_words"):
            m = jiwer.process_words([r], [h])
            value = getattr(m, "wer", None)
        else:  # pragma: no cover
            value = jiwer.compute_measures([r], [h])["wer"]
        per_clip.append(value)
    worst = sorted(
        (
            {"ref": r, "hyp": h, "wer": w}
            for r, h, w in zip(references, hypotheses, per_clip)
        ),
        key=lambda d: -d["wer"],
    )[:3]
    if worst and worst[0]["wer"] > 0.5:
        print("\n  worst clips:")
        for item in worst:
            print(f"    [{item['wer'] * 100:.0f}%] {item['ref']}")
            print(f"            {item['hyp'][:200]}")

    if was_training:
        model.train()
    return {
        "wer": get("wer"),
        "mer": get("mer"),
        "substitutions": get("substitutions"),
        "deletions": get("deletions"),
        "insertions": get("insertions"),
        "clips": len(rows),
        "exact_match": exact,
        "empty_hypotheses": empty,
        "examples": [
            {"ref": r, "hyp": h} for r, h in list(zip(references, hypotheses))[:8]
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())