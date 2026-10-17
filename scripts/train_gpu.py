#!/usr/bin/env python3
"""Stage 1 training on a rented GPU (HF Jobs).

Same architecture and same saved format as `train_stage0.py`, built for
CUDA and for real speech. The point of this script is a like-for-like
comparison between Qwen3.5-2B and Qwen3.5-4B, so it takes pains to
make the two runs differ in exactly one thing: the model size.

Held identical between the 2B and the 4B run:

* the data order (seeded shuffle, same seed, same length bucketing),
* the prompt (built once from `build_salm_prompt_ids`),
* the eval clips (the same manifest rows, greedy decode),
* the step budget, batch size and LR schedule,
* the seed for the LoRA initialisation.

Only `--llm-path` differs. Everything else is pinned by `--seed`.

Batching without padding
------------------------

LibriSpeech utterances run from 1 s to 35 s, so a padded batch wastes
most of its compute on padding and needs an attention mask threaded
through a hybrid (linear + full attention) LLM. This script instead
builds *length-homogeneous* batches: shuffle, sort inside large
megabatches, split, shuffle the batch order. No padding, no mask, no
risk of a subtly wrong mask, and the GPU stays busy.

Only the projection and the LoRA are trainable. The encoder is frozen
and runs under `no_grad`.

Usage (on the job):

    python scripts/train_gpu.py \\
        --llm-path Qwen/Qwen3.5-2B \\
        --run-name stage1-2b \\
        --steps 5000 --batch-size 8 \\
        --out /out/stage1-2b

Usage (locally, to smoke the whole path including the save):

    python scripts/train_gpu.py --steps 3 --batch-size 2 \\
        --train-limit 32 --eval-limit 4 --max-minutes 10 \\
        --out /tmp/smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bim_asr.model import (  # noqa: E402
    BimAsrConfig,
    BimAsrModel,
    build_salm_prompt_ids,
    count_trainable_parameters_summary,
    stop_token_ids,
)
from bim_asr.parakeet import load_parakeet_encoder  # noqa: E402

DEFAULT_ENCODER = "artifacts/parakeet-encoder-v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bim-ASR Stage 1 training on GPU")
    p.add_argument("--llm-path", required=True, help="Hub id or local path for the LLM")
    p.add_argument("--encoder-path", default=DEFAULT_ENCODER)
    p.add_argument("--out", required=True, help="Output directory for the checkpoint")
    p.add_argument("--run-name", default=None)

    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--train-split", default="train.100",
                   help="LibriSpeech `clean` config split. Names are train.100, "
                        "train.360, validation, test -- not validation.clean.")
    p.add_argument("--eval-split", default="validation")
    p.add_argument("--train-limit", type=int, default=None,
                   help="Cap training utterances (smoke tests)")
    p.add_argument("--eval-limit", type=int, default=200)
    p.add_argument("--min-seconds", type=float, default=0.5)
    p.add_argument("--max-seconds", type=float, default=30.0)
    p.add_argument("--megabatch", type=int, default=50,
                   help="Shuffle buffer multiplier for length bucketing")

    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=500,
                   help="Decodes are slow; evaluate this often, not every step")
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--max-minutes", type=float, default=None)
    p.add_argument("--no-upload", action="store_true")
    p.add_argument("--upload-repo", default=None,
                   help="Hub repo for the final checkpoint (default derived from run name)")
    p.add_argument("--upload-private", action="store_true", default=True)
    p.add_argument("--attn-implementation", default="sdpa")
    return p.parse_args()


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def load_split(split: str, limit: int | None, min_seconds: float, max_seconds: float,
               seed: int):
    """LibriSpeech as (16 kHz float32 waveform, transcript) pairs.

    Split names, for the `clean` config, are exactly:
    `train.100`, `train.360`, `validation`, `test`. There is no
    `validation.clean` or `train.clean.100` under this config — those
    exist only under the `all` config, and asking for one here raises
    `Bad split`.

    Do not hand-build `resolve/main/...` URLs to bound the download.
    `clean/validation/0.parquet` resolves, but
    `clean/train.100/0.parquet` 404s, so a hand-rolled URL list silently
    breaks the training split. `load_dataset` with the config and split
    name is the reliable path.
    """
    from datasets import Audio, load_dataset

    ds = load_dataset("openslr/librispeech_asr", "clean", split=split)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    ds = ds.filter(
        lambda r: min_seconds
        <= (r["audio"]["array"].shape[0] / r["audio"]["sampling_rate"])
        <= max_seconds,
        num_proc=8,
    )
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def make_batches(ds, batch_size: int, megabatch: int, seed: int):
    """Length-homogeneous batches: no padding, deterministic order.

    Shuffle, then sort within megabatches, split, then shuffle the batch
    order so the length correlation does not survive into the schedule.
    The seed makes this identical for the 2B and the 4B run.
    """
    rng = random.Random(seed)
    order = list(range(len(ds)))
    rng.shuffle(order)

    # Decode once up front: row access is the expensive part of this
    # function, and the lengths are needed to sort every block.
    n_samples = {i: int(ds[i]["audio"]["array"].shape[0]) for i in order}

    batches: list[list[int]] = []
    window = batch_size * megabatch
    for start in range(0, len(order), window):
        block = sorted(order[start : start + window], key=lambda i: n_samples[i])
        for b in range(0, len(block), batch_size):
            chunk = block[b : b + batch_size]
            if chunk:
                batches.append(chunk)
    random.Random(seed + 1).shuffle(batches)
    return batches


def wav_from_row(row) -> np.ndarray:
    audio = row["audio"]
    wav = np.asarray(audio["array"], dtype=np.float32)
    if audio["sampling_rate"] != 16000:
        raise RuntimeError(f"expected 16 kHz, got {audio['sampling_rate']}")
    return wav


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


class GPUStage1:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device("cuda")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

        self.cfg = BimAsrConfig.for_llm(args.llm_path)
        self.run_name = args.run_name or f"stage1-{self.cfg.llm_id.split('/')[-1]}"
        print("=" * 72)
        print(f"Bim-ASR Stage 1 :: {self.run_name}")
        print("=" * 72)
        print(f"llm:      {self.cfg.llm_id}  hidden {self.cfg.llm_hidden}")
        print(f"bridge:   {self.cfg.encoder_hidden} -> {self.cfg.llm_hidden}")
        print(f"steps:    {args.steps}  batch {args.batch_size} x accum {args.grad_accum}")
        print(f"lr:       {args.lr}  warmup {args.warmup}")
        print(f"seed:     {args.seed}")
        print(f"device:   {torch.cuda.get_device_name(0)}")
        print()

        self.model = BimAsrModel.from_pretrained(
            self.cfg,
            device="cuda",
            dtype=torch.bfloat16,
        )
        self.tokenizer = self.model.tokenizer
        self.stop_ids = stop_token_ids(self.tokenizer)
        print(count_trainable_parameters_summary(self.model))
        print(f"stop tokens: {self.stop_ids}")
        print()

        self.encoder = load_parakeet_encoder(args.encoder_path, device="cuda")
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        print(f"encoder: frozen, {sum(p.numel() for p in self.encoder.parameters()):,} params "
              f"({self.encoder.info.licence})")
        print()

        self.prefix, self.suffix = build_salm_prompt_ids(
            self.tokenizer,
            user_prompt=self.cfg.user_prompt,
            audio_locator_tag=self.cfg.audio_locator_tag,
        )
        self.trainable = [p for p in self.model.parameters() if p.requires_grad]

    def encode_batch(self, wavs: list[np.ndarray]) -> torch.Tensor:
        """Frozen encoder over a length-homogeneous batch. Returns (B,T,H)."""
        lengths = torch.tensor([len(w) for w in wavs], device=self.device)
        max_len = int(lengths.max())
        padded = torch.zeros(len(wavs), max_len, dtype=torch.float32)
        for i, wav in enumerate(wavs):
            padded[i, : len(wav)] = torch.from_numpy(wav)
        with torch.no_grad():
            feats, _ = self.encoder(padded)
        return feats

    def make_example(self, wavs: list[np.ndarray], texts: list[str]):
        feats = self.encode_batch(wavs)
        audio_emb = self.model.encode_audio(feats)

        batch = len(wavs)
        prefix_ids = self.prefix.unsqueeze(0).repeat(batch, 1)
        suffix_ids = self.suffix.unsqueeze(0).repeat(batch, 1)
        embeds = self.model.splice_audio_into_prompt(audio_emb, prefix_ids, suffix_ids)

        token_ids = [
            self.tokenizer(t, add_special_tokens=False).input_ids + [int(self.tokenizer.eos_token_id)]
            for t in texts
        ]
        max_tokens = max(len(t) for t in token_ids)
        target = torch.full((batch, max_tokens), int(self.tokenizer.eos_token_id),
                            dtype=torch.long)
        lengths = torch.zeros(batch, dtype=torch.long)
        for i, ids in enumerate(token_ids):
            target[i, : len(ids)] = torch.tensor(ids)
            lengths[i] = len(ids)

        target_emb = self.model.embed_tokens(target)
        inputs_embeds = torch.cat([embeds, target_emb], dim=1)

        labels = torch.full_like(inputs_embeds[..., 0], -100)
        labels[:, embeds.shape[1] :] = torch.where(
            lengths[:, None] >= torch.arange(max_tokens, device=self.device)[None, :],
            target,
            torch.full_like(target, -100),
        )
        return {"inputs_embeds": inputs_embeds, "labels": labels}

    @torch.no_grad()
    def transcribe(self, wav: np.ndarray, max_new_tokens: int = 96) -> str:
        was_training = self.model.training
        self.model.eval()
        feats = self.encode_batch([wav])
        audio_emb = self.model.encode_audio(feats)
        embeds = self.model.splice_audio_into_prompt(
            audio_emb, self.prefix.unsqueeze(0), self.suffix.unsqueeze(0)
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
        torch.save(self.model.projection.state_dict(), out_dir / "projection.pt")
        self.model.llm.save_pretrained(str(out_dir / "lora_adapter"))
        with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        (out_dir / "bim_asr_config.json").write_text(json.dumps({
            "llm_id": self.cfg.llm_id,
            "llm_hidden": self.cfg.llm_hidden,
            "encoder_hidden": self.cfg.encoder_hidden,
            "lora_rank": self.cfg.lora_rank,
            "lora_alpha": self.cfg.lora_alpha,
            "lora_target_modules": list(self.cfg.lora_target_modules),
            "encoder_id": self.cfg.encoder_id,
            "audio_locator_tag": self.cfg.audio_locator_tag,
            "user_prompt": self.cfg.user_prompt,
            "run_name": self.run_name,
        }, indent=2))


def evaluate(trainer: GPUStage1, ds, limit: int, tag: str) -> dict:
    """Greedy decode `limit` clips and compute WER, with per-clip output."""
    import jiwer

    started = time.time()
    refs, hyps = [], []
    for i in range(min(limit, len(ds))):
        row = ds[i]
        refs.append(row["text"].strip())
        hyps.append(trainer.transcribe(wav_from_row(row)))
        if (i + 1) % 25 == 0 or i + 1 == min(limit, len(ds)):
            print(f"  [{tag}] decoded {i + 1}/{min(limit, len(ds))} ({time.time() - started:.0f}s)")

    if hasattr(jiwer, "process_words"):
        m = jiwer.process_words(refs, hyps)
        get = lambda k: getattr(m, k)  # noqa: E731
    else:  # pragma: no cover
        m = jiwer.compute_measures(refs, hyps)
        get = lambda k: m[k]  # noqa: E731

    exact = sum(1 for h, r in zip(hyps, refs) if h.strip() == r.strip())
    print()
    for r, h in list(zip(refs, hyps))[:5]:
        print(f"  ref: {r}")
        print(f"  hyp: {h}")
    print()
    print(f"  [{tag}] WER {get('wer') * 100:.2f}%  MER {get('mer') * 100:.2f}%  "
          f"exact {exact}/{len(refs)}")
    return {
        "wer": get("wer"),
        "mer": get("mer"),
        "substitutions": get("substitutions"),
        "deletions": get("deletions"),
        "insertions": get("insertions"),
        "clips": len(refs),
        "exact_match": exact,
        "decode_seconds": time.time() - started,
        "predictions": [{"ref": r, "hyp": h} for r, h in zip(refs, hyps)],
    }


def upload_checkpoint(out_dir: Path, repo_id: str, private: bool) -> str | None:
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(out_dir),
            repo_type="model",
            commit_message=f"{out_dir.name} checkpoint",
        )
        print(f"uploaded {out_dir} -> {repo_id}")
        return repo_id
    except Exception as exc:  # noqa: BLE001
        print(f"upload failed ({exc.__class__.__name__}: {exc})")
        return None


def main() -> int:
    args = parse_args()
    started = time.time()
    out_dir = Path(args.out)

    print("loading LibriSpeech...")
    train_ds = load_split(args.train_split, args.train_limit, args.min_seconds, args.max_seconds, args.seed)
    eval_ds = load_split(args.eval_split, args.eval_limit, args.min_seconds, args.max_seconds, args.seed)
    print(f"train: {len(train_ds)} utterances ({args.train_split})")
    print(f"eval:  {len(eval_ds)} utterances ({args.eval_split})")
    print()
    args._train_utterances = len(train_ds)

    trainer = GPUStage1(args)
    batches = make_batches(train_ds, args.batch_size, args.megabatch, args.seed)
    print(f"{len(batches)} batches of {args.batch_size} (length-homogeneous, seed {args.seed})")
    print()

    optimiser = torch.optim.AdamW(trainer.trainable, lr=args.lr,
                                  weight_decay=args.weight_decay)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / max(1, args.warmup)
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    history: list[dict] = []
    losses: list[float] = []
    cursor = 0
    model = trainer.model
    model.train()
    stopped_early = None
    last_eval: dict | None = None

    for step in range(args.steps):
        for group in optimiser.param_groups:
            group["lr"] = lr_at(step)

        optimiser.zero_grad(set_to_none=True)
        batch_losses = []
        for micro in range(args.grad_accum):
            if cursor >= len(batches):
                cursor = 0
            indices = batches[cursor]
            cursor += 1
            rows = [train_ds[i] for i in indices]
            wavs = [wav_from_row(r) for r in rows]
            texts = [r["text"].strip() for r in rows]
            example = trainer.make_example(wavs, texts)
            loss = trainer.model.llm(**example).loss / args.grad_accum
            loss.backward()
            batch_losses.append(loss.item() * args.grad_accum)

        grad_norm = torch.nn.utils.clip_grad_norm_(trainer.trainable, args.grad_clip)
        optimiser.step()

        mean_loss = float(np.mean(batch_losses))
        losses.append(mean_loss)
        elapsed = time.time() - started
        record = {
            "step": step + 1,
            "loss": mean_loss,
            "lr": lr_at(step),
            "grad_norm": float(grad_norm),
            "elapsed_minutes": elapsed / 60,
        }
        history.append(record)

        if (step + 1) % args.log_every == 0:
            window = losses[-args.log_every :]
            print(f"step {step + 1:>6}/{args.steps}  loss {np.mean(window):.4f}  "
                  f"lr {lr_at(step):.2e}  |g| {float(grad_norm):7.2f}  "
                  f"{elapsed / 60:.1f}m  eta {(args.steps - step - 1) * elapsed / (step + 1) / 60:.0f}m")
            sys.stdout.flush()

        if args.eval_every and (step + 1) % args.eval_every == 0 and (step + 1) < args.steps:
            last_eval = evaluate(trainer, eval_ds, args.eval_limit, f"step {step + 1}")
            history[-1]["eval"] = {"wer": last_eval["wer"]}
            trainer.save(out_dir, _metrics(trainer, args, step + 1, losses, history, last_eval, started))
            print(f"  checkpoint saved -> {out_dir}")
            print()

        if args.save_every and (step + 1) % args.save_every == 0 and (step + 1) < args.steps:
            trainer.save(out_dir, _metrics(trainer, args, step + 1, losses, history, last_eval, started))

        if args.max_minutes and (time.time() - started) / 60 > args.max_minutes:
            stopped_early = step + 1
            print(f"\ntime budget of {args.max_minutes} min reached at step {step + 1}")
            break

    print("\nfinal evaluation...")
    final_eval = evaluate(trainer, eval_ds, args.eval_limit, "final")
    history.append({"step": args.steps if not stopped_early else stopped_early,
                    "final_eval": {"wer": final_eval["wer"]}})
    metrics = _metrics(trainer, args, stopped_early or args.steps, losses, history, final_eval, started)
    trainer.save(out_dir, metrics)
    (out_dir / "train_log.json").write_text(json.dumps(history, indent=2))

    print()
    print(f"run        : {trainer.run_name}")
    print(f"steps      : {metrics['steps_completed']}")
    print(f"final WER  : {final_eval['wer'] * 100:.2f}%  ({final_eval['clips']} clips)")
    print(f"total time : {metrics['minutes']:.1f} min")
    print(f"checkpoint : {out_dir}")

    repo_id = args.upload_repo or f"hammertoe/bim-asr-{trainer.run_name}"
    if not args.no_upload:
        uploaded = upload_checkpoint(out_dir, repo_id, args.upload_private)
        if uploaded:
            metrics["uploaded_to"] = uploaded
            trainer.save(out_dir, metrics)

    print("\nOK.")
    return 0


def _metrics(trainer, args, steps, losses, history, eval_result, started) -> dict:
    return {
        "run_name": trainer.run_name,
        "llm_id": trainer.cfg.llm_id,
        "llm_hidden": trainer.cfg.llm_hidden,
        "bridge_shape": [trainer.cfg.encoder_hidden, trainer.cfg.llm_hidden],
        "steps_completed": steps,
        "requested_steps": args.steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "seed": args.seed,
        "lr": args.lr,
        "warmup": args.warmup,
        "loss_initial50": float(np.mean(losses[:50])) if losses else None,
        "loss_final50": float(np.mean(losses[-50:])) if len(losses) >= 50 else (losses[-1] if losses else None),
        "minutes": (time.time() - started) / 60,
        "train_split": args.train_split,
        "train_utterances": getattr(args, "_train_utterances", None),
        "eval_split": args.eval_split,
        "eval_clips": eval_result["clips"] if eval_result else None,
        "eval_wer": eval_result["wer"] if eval_result else None,
        "trainable_parameters": sum(p.numel() for p in trainer.trainable),
        "hardware": torch.cuda.get_device_name(0),
    }


if __name__ == "__main__":
    raise SystemExit(main())