#!/usr/bin/env python3
"""Bim-ASR Stage 0 smoke test.

This is the gate from docs/TRAINING.md stage 0. It proves the SALM
wiring (audio -> frozen Parakeet encoder -> projection -> LLM with
LoRA -> transcript) works end to end on this Mac, without a rented GPU.

Modes
-----

* `--mode fake-encoder` (default): the FastConformer encoder is
  replaced with a random-frame generator. Proves the bridge, the LoRA
  wiring and the prompt splice, in seconds, on any machine.

* `--mode real-encoder`: the **real** Parakeet TDT 0.6B v2 encoder,
  extracted from the NeMo checkpoint by
  `scripts/prepare_parakeet_encoder.py`, plus the **real** Qwen3.5 LLM.
  This is the actual Stage 0 gate.

The encoder is loaded through `transformers`, not the NeMo toolkit:
NeMo has no supported macOS build, and we need this to run on the
laptop. The weights are the same ones NeMo would have loaded --
scripts/prepare_parakeet_encoder.py copies NVIDIA's official
NeMo -> HF mapping and loads with strict=True.

What counts as passing
----------------------

1. the encoder runs and returns 1024-dim frames at 12.5 Hz,
2. the bridge maps them to the LLM's embedding width,
3. the LLM generates at least one token,
4. the generated text changes when the audio changes.

Point 4 is the one that matters most. An untrained SALM produces
nonsense words either way; what it must not do is produce the *same*
nonsense regardless of the audio. If the transcript moves when the
audio moves, the audio is genuinely reaching the model.

Examples
--------

    # bridge-only, seconds, no checkpoints
    .venv/bin/python scripts/smoke_salm.py --mode fake-encoder

    # the real thing (needs artifacts/ prepared, see README)
    .venv/bin/python scripts/smoke_salm.py --mode real-encoder \
        --llm-path artifacts/qwen3.5-2b

    # after a Stage 0 training run
    .venv/bin/python scripts/smoke_salm.py --mode real-encoder \
        --llm-path artifacts/qwen3.5-2b --checkpoint artifacts/stage0-ckpt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bim_asr.audio import chunk_waveform, load_audio_16k_mono  # noqa: E402
from bim_asr.model import (  # noqa: E402
    AudioProjection,
    BimAsrConfig,
    BimAsrModel,
    build_salm_prompt_ids,
    count_trainable_parameters_summary,
)

DEFAULT_LLM_PATH = ROOT / "artifacts" / "qwen3.5-2b"
DEFAULT_ENCODER_PATH = ROOT / "artifacts" / "parakeet-encoder-v2"
DEFAULT_MANIFEST = ROOT / "data" / "smoke" / "eval.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bim-ASR Stage 0 smoke test")
    p.add_argument(
        "--mode",
        choices=["fake-encoder", "real-encoder"],
        default="fake-encoder",
        help="fake-encoder replaces the Parakeet encoder with a random-frame "
        "generator; real-encoder loads the actual extracted weights",
    )
    p.add_argument(
        "--llm-path",
        default=None,
        help="Local path or HF id for the LLM (default: artifacts/qwen3.5-2b)",
    )
    p.add_argument(
        "--llm-id", default=None, help="Hub id for the LLM, e.g. Qwen/Qwen3.5-4B"
    )
    p.add_argument(
        "--encoder-path",
        default=str(DEFAULT_ENCODER_PATH),
        help="Directory holding the extracted Parakeet encoder",
    )
    p.add_argument("--audio", type=str, default=None, help="Path to a 16 kHz mono wav")
    p.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="NeMo manifest to pull the default audio from (scripts/make_smoke_corpus.py)",
    )
    p.add_argument(
        "--clip",
        type=int,
        default=0,
        help="Which clip of the manifest to use as the audio",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Trained Stage 0 checkpoint (projection.pt + lora adapter) to load",
    )
    p.add_argument("--device", default="cpu", help="cpu or mps")
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--load-in-4bit", action="store_true", help="CUDA-only (bitsandbytes)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--skip-audio-sensitivity",
        action="store_true",
        help="Skip the second generation with different audio",
    )
    return p.parse_args()


class FakeFastConformerEncoder(torch.nn.Module):
    """Drop-in replacement for the Parakeet encoder.

    Output shape: (batch, frames, 1024) at 12.5 frames per second, so
    30 s of audio -> 375 frames.
    """

    def __init__(self, hidden: int = 1024, frame_rate: float = 12.5) -> None:
        super().__init__()
        self.hidden = hidden
        self.frame_rate = frame_rate

    def forward(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor | None = None):
        if waveforms.dim() == 1:
            waveforms = waveforms.unsqueeze(0)
        b = waveforms.shape[0]
        max_len = int(waveforms.shape[1])
        max_frames = max(1, int(round(max_len / 16000.0 * self.frame_rate)))
        feats = torch.randn(b, max_frames, self.hidden, device=waveforms.device) * 0.5
        lengths = (
            waveform_lengths
            if waveform_lengths is not None
            else torch.full((b,), max_len, device=waveforms.device, dtype=torch.long)
        )
        return feats, lengths


def make_silent_clip(seconds: float = 2.0, sample_rate: int = 16000) -> np.ndarray:
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


def load_manifest_clip(manifest: Path, index: int) -> tuple[np.ndarray, str]:
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not rows:
        raise FileNotFoundError(f"manifest {manifest} is empty")
    row = rows[index % len(rows)]
    return load_audio_16k_mono(row["audio_filepath"]), row.get("text", "")


def bridge_only_check(cfg: BimAsrConfig) -> None:
    """The 10-second version of the test: bridge shape and trainability."""
    print("projection:", count_trainable_parameters_summary(BimAsrModel(cfg)))
    proj = AudioProjection(encoder_hidden=cfg.encoder_hidden, llm_hidden=cfg.llm_hidden)
    feats = torch.randn(1, 38, cfg.encoder_hidden)
    projected = proj(feats)
    print(f"  bridge in:  (1, 38, {cfg.encoder_hidden})")
    print(f"  bridge out: {tuple(projected.shape)}")
    assert projected.shape == (1, 38, cfg.llm_hidden), "bridge shape mismatch"
    projected.sum().backward()
    grad = proj.proj.weight.grad
    assert grad is not None and grad.abs().sum() > 0, "no gradient reached the bridge"
    print(f"  bridge OK (grad norm {grad.norm().item():.4f})")
    print()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    print("=" * 66)
    print("Bim-ASR Stage 0 smoke test")
    print("=" * 66)
    print(f"mode:         {args.mode}")
    print(f"device:       {device}")
    print(f"max-new:      {args.max_new_tokens}")
    print()

    llm_ref = args.llm_path or args.llm_id or (
        "Qwen/Qwen3.5-2B" if (DEFAULT_LLM_PATH / "config.json").exists() else "Qwen/Qwen3.5-4B"
    )

    # 1. Config, sized from the real LLM.
    cfg = BimAsrConfig.for_llm(llm_ref)
    print(f"config: llm={cfg.llm_id}  encoder_hidden={cfg.encoder_hidden}  "
          f"llm_hidden={cfg.llm_hidden}  lora_rank={cfg.lora_rank}")
    print()

    # 2. Audio.
    reference = None
    if args.audio:
        print(f"loading audio: {args.audio}")
        waveform = load_audio_16k_mono(args.audio)
    elif args.manifest.exists():
        waveform, reference = load_manifest_clip(args.manifest, args.clip)
        print(f"audio: manifest clip {args.clip} of {args.manifest.name}")
        print(f"  reference text: {reference!r}")
    else:
        print("no audio and no manifest; using a 2 s silent clip")
        waveform = make_silent_clip(2.0)
    print(f"  waveform: {len(waveform)} samples ({len(waveform) / 16000.0:.2f} s)")
    chunks = chunk_waveform(waveform, sample_rate=16000, window_seconds=30.0, overlap_seconds=0.0)
    print(f"  chunks:   {len(chunks)} (30 s window, no overlap)")
    print()

    # 3. Bridge-only check, always.
    bridge_only_check(cfg)

    full_smoke = (
        args.mode == "real-encoder"
        or os.environ.get("BIM_ASR_FULL_SMOKE") == "1"
    )
    if not full_smoke:
        print("fake-encoder mode without BIM_ASR_FULL_SMOKE=1 validated the")
        print("bridge only. Re-run with --mode real-encoder for the full pass.")
        print()
        print("OK: Stage 0 scaffold works.")
        return 0

    # 4. Real LLM + LoRA.
    print(f"loading {cfg.llm_id} + LoRA r={cfg.lora_rank} (bf16)...")
    t0 = time.time()
    try:
        model = BimAsrModel.from_pretrained(
            cfg, device=device, load_in_4bit=args.load_in_4bit
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED to load: {exc.__class__.__name__}: {exc}")
        return 1
    print(f"  loaded in {time.time() - t0:.1f} s")
    print(f"  LLM class: {type(model.llm).__name__}")
    print(f"  trainable: {model.trainable_parameter_count():,}")

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)
        print(f"  loaded trained weights from {args.checkpoint}")
    print()

    tok = model.tokenizer

    # 5. The encoder.
    if args.mode == "real-encoder":
        from bim_asr.parakeet import load_parakeet_encoder

        print(f"loading the real Parakeet encoder from {args.encoder_path}...")
        encoder = load_parakeet_encoder(args.encoder_path, device=device)
        print(f"  {encoder.info.encoder_layers} layers, hidden {encoder.info.encoder_hidden}, "
              f"{encoder.info.frame_rate_hz} Hz, {encoder.info.licence}")
    else:
        encoder = FakeFastConformerEncoder(cfg.encoder_hidden).to(device)
        print("using the fake (random-frame) encoder")
    print()

    # 6. Encode -> project -> splice -> generate.
    prefix_ids, suffix_ids = build_salm_prompt_ids(
        tok, user_prompt=cfg.user_prompt, audio_locator_tag=cfg.audio_locator_tag
    )
    print(f"prompt: {prefix_ids.shape[0]} tokens before the audio, "
          f"{suffix_ids.shape[0]} after")
    print(f"  ...{tok.decode(prefix_ids[-14:])!r} [AUDIO] {tok.decode(suffix_ids[:16])!r}...")
    print()

    chunk = chunks[0]
    waveforms = torch.from_numpy(np.ascontiguousarray(chunk)).unsqueeze(0)

    t0 = time.time()
    with torch.no_grad():
        feats, feat_lens = encoder(waveforms)
    print(f"encoder:  (1, {waveforms.shape[1]}) samples -> {tuple(feats.shape)} frames "
          f"in {time.time() - t0:.2f}s")
    assert feats.shape[-1] == cfg.encoder_hidden, (
        f"encoder produced {feats.shape[-1]} dims, bridge wants {cfg.encoder_hidden}"
    )

    model.eval()
    with torch.no_grad():
        audio_emb = model.encode_audio(feats.to(device))
    print(f"bridge:   -> {tuple(audio_emb.shape)} (LLM width {cfg.llm_hidden})")
    print(f"  audio emb |mean| {audio_emb.abs().mean().item():.3f}  "
          f"per-frame norm {audio_emb.norm(dim=-1).mean().item():.2f}")
    assert audio_emb.shape[-1] == cfg.llm_hidden

    transcripts = []
    with torch.no_grad():
        for label, audio_in in [("audio", audio_emb)]:
            full = model.splice_audio_into_prompt(
                audio_in, prefix_ids.unsqueeze(0).to(device), suffix_ids.unsqueeze(0).to(device)
            )
            print(f"\nspliced sequence: {tuple(full.shape)} "
                  f"({prefix_ids.shape[0]} text + {audio_in.shape[1]} audio + "
                  f"{suffix_ids.shape[0]} text)")
            t0 = time.time()
            out = model.llm.generate(
                inputs_embeds=full,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
            text = tok.decode(out[0], skip_special_tokens=True).strip()
            transcripts.append((label, text))
            print(f"generated {out.shape[-1]} tokens in {time.time() - t0:.1f}s")
            print("---")
            print(f"[{label}] {text!r}")
            print("---")

        # 7. Audio sensitivity: the same pipeline, different audio.
        if not args.skip_audio_sensitivity and args.mode == "real-encoder":
            other = chunks[1] if len(chunks) > 1 else make_silent_clip(len(chunk) / 16000)
            other_feats, _ = encoder(torch.from_numpy(np.ascontiguousarray(other)).unsqueeze(0))
            other_emb = model.encode_audio(other_feats.to(device))
            delta = (audio_emb - other_emb).norm().item()
            print(f"\naudio sensitivity: two different clips differ by {delta:.2f} "
                  f"in projected embedding space")
            full2 = model.splice_audio_into_prompt(
                other_emb, prefix_ids.unsqueeze(0).to(device), suffix_ids.unsqueeze(0).to(device)
            )
            out2 = model.llm.generate(
                inputs_embeds=full2,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
            text2 = tok.decode(out2[0], skip_special_tokens=True).strip()
            transcripts.append(("other audio", text2))
            print("---")
            print(f"[other audio] {text2!r}")
            print("---")
            if text2 == text:
                print("\nWARNING: identical output for different audio -- the audio")
                print("is not reaching the model.")
                return 1
            print("different audio produced different output: audio is wired through.")

    print()
    if args.checkpoint is None:
        print("NOTE: this was an UNTRAINED bridge + fresh LoRA. The words above are")
        print("not a transcript, they are whatever the base LLM emits when handed")
        print("randomly-initialised audio embeddings. For real text, run Stage 0")
        print("training first:")
        print("  .venv/bin/python scripts/make_smoke_corpus.py")
        print("  .venv/bin/python scripts/train_stage0.py --steps 400")
        print("  .venv/bin/python scripts/smoke_salm.py --mode real-encoder \\")
        print("      --checkpoint artifacts/stage0-ckpt")
    else:
        print("NOTE: this used a trained Stage 0 checkpoint, so the text above is")
        print("real output from the SALM. See artifacts/stage0-ckpt/metrics.json.")

    print()
    print("OK: Stage 0 real-encoder smoke passed.")
    return 0


def load_checkpoint(model: BimAsrModel, checkpoint: Path) -> None:
    """Load a Stage 0 projection + LoRA checkpoint."""
    proj_path = checkpoint / "projection.pt"
    if proj_path.exists():
        state = torch.load(proj_path, map_location="cpu", weights_only=True)
        model.projection.load_state_dict(state)
        print(f"  projection weights loaded ({sum(v.numel() for v in state.values()):,} params)")

    adapter = checkpoint / "lora_adapter"
    if adapter.exists():
        from peft import PeftModel  # noqa: PLC0415

        model.llm = PeftModel.from_pretrained(model.llm, str(adapter), is_trainable=False)
        model.llm.to(next(model.projection.parameters()).device)
        print(f"  LoRA adapter loaded from {adapter}")


if __name__ == "__main__":
    raise SystemExit(main())