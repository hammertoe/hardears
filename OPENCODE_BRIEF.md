# OpenCode build brief: Bim-ASR SALM wiring

> **Status: Stage 0 is done.** This brief was written before the work
> started and has been updated to match reality. It no longer describes
> a greenfield task; see "What actually happened" at the bottom for
> what was built, what was found, and what is still open.
> The local scaffold has since moved: the working directory is
> `/Users/matt/Development/bim-asr`, and the branch is
> `feature/stage0-real-encoder` (PR #2).

## Why this file exists

The OpenClaw sessions_spawn tool on this Gateway is locked to
`agentId: "main"`; you cannot dispatch directly to the OpenCode
engineering manager from a subagent. This file is the build brief for
the next OpenCode session, and the run command at the bottom will start
it.

## Project

/Users/matt/Development/bim-asr

Bim-ASR is a Speech-Augmented Language Model (SALM) that fuses a
frozen FastConformer acoustic encoder with a frozen Qwen3.5-4B-Instruct
LLM through a small linear projection plus a LoRA adapter. Only the
projection and LoRA are trained.

The reference is nvidia/canary-qwen-2.5b. We use a 4B Qwen3.5 base
instead of 1.7B, and a Parakeet TDT 0.6B v2 encoder instead of Canary.

**Corrections to the rest of this brief, all verified against the
checkpoints:** the Parakeet encoder is **1024**-dim, not 512;
Qwen3.5-2B is **2048**-dim, not 2560; the hybrid layer ratio is 3:1,
not 8:1; and NeMo has **no macOS build**, so the encoder is loaded
through `transformers` rather than `nemo.collections.asr`. See
`docs/ARCHITECTURE.md` for the full correction log.

## What already exists

- `src/bim_asr/model.py` - `BimAsrConfig`, `AudioProjection`, `BimAsrModel` (PyTorch). The projection is 2.1M params, encoder_hidden=1024 -> llm_hidden=2560 (4B) or 2048 (2B).
- `src/bim_asr/data.py` - NeMo manifest writer/reader with audio SHA-256, duration probe, ffmpeg resample.
- `src/bim_asr/audio.py` - 16 kHz mono loader, 30 s/5 s overlap chunker.
- `src/bim_asr/parakeet.py` - the frozen encoder, NeMo-compatible call signature.
- `src/bim_asr/inference.py` - `SalmTranscriber`, the real inference path.
- `src/bim_asr/mlx_model.py` - MLX mirror of the projection. Reports `is_mlx_available()`.
- `src/bim_asr/__init__.py`
- `scripts/prepare_parakeet_encoder.py` - extracts the encoder from the `.nemo`. Done.
- `scripts/make_smoke_corpus.py` - offline macOS `say` corpus, multi-voice, held-out eval voices. Done.
- `scripts/smoke_salm.py` - Stage 0 gate; `--mode real-encoder` runs green on a Mac. Done.
- `scripts/train_stage0.py` - Stage 0 training on MPS. Done.
- `scripts/transcribe.py` - inference CLI. Done.
- `scripts/eval_baseline_parakeet.py` - reference ASR via the original TDT head. Done.
- `scripts/download_data.py` - dataset downloader for LibriSpeech, Common Voice 17, GigaSpeech XS, and Barbados Parliament YouTube/Vimeo.
- `scripts/eval_proper_nouns.py` - proper-noun evaluator using the BimOmni 10-noun list. Tested on a 4-clip set (got 0.075).
- `tests/` - 56 tests, all passing in ~9 s without any checkpoint. 4 more are opt-in behind `BIM_ASR_SLOW_TESTS=1`.
- `docs/ARCHITECTURE.md`, `docs/TRAINING.md`, `docs/DATA.md`, `docs/LORA_TARGETS.md`, `docs/STAGE0_RESULTS.md`
- `pyproject.toml` and `.venv` (gitignored; rebuild with `uv venv --python 3.12 .venv`)
- `ffmpeg` 9.0.2 and `yt-dlp` 2026.8.19 installed via Homebrew.

## What is still to do (in this order)

### 1. `configs/hardears_v0.yaml` (was task 2, not started)

- Based on the stock nvidia/canary-qwen-2.5b `salm.yaml`.
- Encoder: `nvidia/parakeet-tdt-0.6b-v2`
- LLM: `Qwen/Qwen3.5-4B`
- Projection shape: **(1024, 2560)** — not (512, 2560)
- LoRA: rank 32, alpha 64, dropout 0.05, on
  `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Frozen encoder, trainable projection + LoRA
- Audio window 30 s, max tokens 1024
- LR 1e-4, cosine, warmup 500 steps, bf16
- Reference: https://github.com/NVIDIA/NeMo/blob/main/examples/speechlm2/conf/salm.yaml

### 2. Decide the LoRA target list (blocks task 1, cheap now, expensive later)

Qwen3.5's 18 linear-attention layers have no `q/k/v/o_proj`; they use
`linear_attn.in_proj_qkv` and `out_proj`. The current list adapts the
FFN path in all 24 layers but the attention path in only 6. Measured
numbers and three options are in `docs/LORA_TARGETS.md`. Settle it
before 5,000 H100 steps.

### 3. Stage 1 run (data prep + training)

- Build the LibriSpeech train-clean-100 NeMo manifest.
- Confirm the NeMo SALM recipe actually loads Qwen3.5 and the extracted
  encoder. Stage 0 ran in PyTorch, so it proved the architecture and
  **not** the Stage 1 training code.
- Run 5,000 steps on 1x H100, evaluate dev-clean WER every 500 steps,
  keep every checkpoint, and dump per-clip predictions.

### 4. Fix long-audio windowing before Parliament-length audio

`src/bim_asr/inference.py` concatenates overlapping 30 s windows rather
than merging them, so words in the 5 s overlap are counted twice. Every
Stage 0 clip fits in one window so it never showed up. Crossfade or
overlap-aware stitching is required for real broadcast audio.

### 5. Finish the MLX path

`src/bim_asr/mlx_model.py` mirrors the projection only. The extracted
encoder is safetensors, which is what `mlx_lm.convert` wants, but the
conversion and the MLX forward loop have not been written.

## Constraints

- Do not edit `docs/ARCHITECTURE.md`, `docs/TRAINING.md`, or `docs/DATA.md` without checking the user first. **They were corrected on 17 October 2026 with evidence; read the correction notes before changing them again.**
- The projection is already implemented in `src/bim_asr/model.py`. Do not duplicate it.
- Keep the SALM prompt format from `build_salm_prompt_ids` in `src/bim_asr/model.py`. The split is done on rendered *text*, not token ids, because BPE is context dependent.
- Size the projection from the model's own config (`BimAsrConfig.for_llm`), never a hard-coded table.
- Always pass `eos_token_id` to `generate()` (`stop_token_ids()`), and always dump per-clip predictions when reporting a WER.
- Work in a feature branch of the git repo, not on main.
- For any opencode work, the user wants the model
  `minimax-coding-plan/MiniMax-M3.1-Flash-Preview`. Set this with the
  opencode `--model` flag or the relevant provider override.

## Run command

The user starts the build session from the shell with:

    cd /Users/matt/Development/bim-asr
    git checkout -b feature/<name>
    opencode \
      --model minimax-coding-plan/MiniMax-M3.1-Flash-Preview \
      --prompt "Read OPENCODE_BRIEF.md at the repo root and execute the tasks it lists. Summarise the diff and any blockers when done." \
      --print-logs

## What actually happened (17 October 2026)

Stage 0 was implemented end to end on an M4 Pro and merged to
`feature/stage0-real-encoder` (PR #2).

- **Encoder extraction.** `prepare_parakeet_encoder.py` converts the
  FastConformer encoder out of the `.nemo` using NVIDIA's own NeMo -> HF
  mapping, with a strict load (0 missing, 0 unexpected). Verified
  numerically: the original TDT decoder decodes real audio off the
  extracted features at **2.1% WER**.
- **Working SALM.** `smoke_salm.py --mode real-encoder` passes: real
  encoder -> bridge -> Qwen3.5-2B + LoRA -> text, and different audio
  produces different text.
- **Training.** `train_stage0.py` took loss 5.11 -> ~0.00001 in 12k
  steps (~90 min) and reaches **19.0% WER on unseen sentences in unseen
  voices**, 4.6x real time on MPS. See `docs/STAGE0_RESULTS.md`.
- **Two bugs found**, both of which had inflated the metrics and
  neither of which was visible in an aggregate: Qwen3.5-2B ships no
  `generation_config.json` so generation never stopped (92.9% -> 19.0%
  WER on identical weights), and the corpus builder reused stale audio
  against fresh manifest text (reference ASR 94.0% -> 6.3%).

## When you finish

Report:

- Files added or changed (one line each).
- A diff summary (`git diff main`) so the user can review.
- Open questions or blockers.
- What the next dispatch should cover (e.g. data prep on rented H100, full training run, leaderboard submission).

