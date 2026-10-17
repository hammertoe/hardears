# Training

This is the staged training plan for Bim-ASR. Each stage is independent and
the artefacts are saved so we can re-run, compare, and roll back.

> **Stage 0 has been run.** Measured results, and the two bugs that were
> inflating them, are in [STAGE0_RESULTS.md](STAGE0_RESULTS.md). Three
> things below were wrong and are marked **[corrected]**: the encoder
> hidden size, the LoRA target list, and the assumption that the laptop
> path runs under NeMo. Read that document before changing any
> hyperparameter here, because Stage 0 showed that loss and held-out WER
> stop agreeing after about 6 000 steps.

## Stages at a glance

| Stage | What it trains                | What data                      | Where          | Target metric                  |
| ----- | ----------------------------- | ------------------------------ | -------------- | ------------------------------ |
| 0     | smoke test (tiny subset)      | synthetic `say` corpus         | local M4       | **done** — pipeline runs end to end |
| 1     | projection + LoRA             | LibriSpeech train-clean-100    | 1-2x H100      | WER < 8 % on dev-clean         |
| 2     | projection + LoRA             | + GigaSpeech XS subset         | 1-2x H100      | WER < 6 % on dev-clean         |
| 3     | projection + LoRA             | + Common Voice 17 (Caribbean)  | 1-2x H100      | Caribbean WER within 2 % of US |
| 4     | projection + LoRA + enc LoRA  | + Parliament audio (paired)    | 1-2x H100      | proper-noun F1 within 5 % of BimOmni |
| 5     | full-stack evaluation         | held-out Parliament clips      | local M4 + HF | publish to leaderboard + HF   |

Stages 0-2 are general English. Stage 3 adds Caribbean English. Stage 4
adds paired Barbadian Parliament audio. Stage 5 is the evaluation and
publication step.

## Stage 0 — smoke test (local M4) — DONE

Goal: prove the SALM scaffolding works end to end on the laptop. We
should see a non-zero loss go down across a few hundred steps.

- Data: **[corrected]** 420 short English sentences rendered offline by
  `scripts/make_smoke_corpus.py` using the macOS `say` voices: 1,260
  train clips (4 voices x 3 speed perturbations), 49 minutes of audio,
  with exact transcripts. Not LibriSpeech — Stage 0 is a wiring test,
  and downloading 100 hours to discover a shape error is a waste of a
  day. The eval split is 48 sentences the model has never seen, spoken
  by two voices it has never heard (Moira, Tessa).
- Compute: M4 Pro, bf16, PyTorch on MPS, batch size 1. **[corrected]**
  Not MLX and not NeMo: NeMo has no macOS build, and MLX cannot train
  the encoder side of this model. Throughput ~2.2 it/s.
- Steps: 12,000 (stopped early; see below).
- Measured: loss 5.11 → ~0.00001; **19.0 % WER on unseen sentences in
  unseen voices**, 0.0 % on seen sentences in unseen voices, 4.6x real
  time on MPS.

Commands:

```bash
# 1. weights
hf download nvidia/parakeet-tdt-0.6b-v2 parakeet-tdt-0.6b-v2.nemo \
    --local-dir artifacts/parakeet-tdt-0.6b-v2
hf download Qwen/Qwen3.5-2B --local-dir artifacts/qwen3.5-2b

# 2. extract the encoder (no NeMo required)
.venv/bin/python scripts/prepare_parakeet_encoder.py \
    --nemo artifacts/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo \
    --out artifacts/parakeet-encoder-v2

# 3. corpus
.venv/bin/python scripts/make_smoke_corpus.py --num-train 400 --eval-seen 48

# 4. the gate
.venv/bin/python scripts/smoke_salm.py --mode real-encoder

# 5. training
.venv/bin/python scripts/train_stage0.py --steps 12000
```

### What Stage 0 taught us that changes Stage 1

- **Select checkpoints by held-out WER, not by loss.** Loss kept falling
  after held-out quality had stopped improving, and reached exactly zero
  while the model was memorising. A Stage 1 run that selects on loss will
  happily ship a memorised checkpoint.
- **Dump per-clip predictions, always.** Stage 0's first two WER numbers
  were both wrong in the same direction, and neither was visible in the
  aggregate. One runaway generation moved an average by ~80 points while
  the first six printed examples were all exact. `--dump-predictions`
  exists because of this.
- **Check the stop tokens.** Qwen3.5-2B ships no
  `generation_config.json` and reports `eos_token` as an empty string.
  Generation without an explicit `eos_token_id` never terminates.
- **Stage 0 is not Stage 1's training code.** It runs in PyTorch, not
  NeMo. A green Stage 0 does not prove `configs/hardears_v0.yaml` works.

## Stage 1 — general English baseline (rented GPU)

Goal: a working SALM with Parakeet encoder + Qwen3.5-4B + projection +
LoRA, trained on a small English corpus. This is the reference run.

- Data: LibriSpeech train-clean-100, 100 hours of read English.
  Manifest in NeMo JSONL format with `audio_filepath`,
  `duration`, and `text`.
- Compute: 1x NVIDIA H100 80GB on Hugging Face Jobs.
- Hyperparameters (initial):
  - Optimiser: AdamW, lr 1e-4, weight decay 0.0
  - LR schedule: cosine, warmup 500 steps
  - Batch size 8, grad accumulation 4 → effective 32
  - Max audio length 30 s, max tokens 1024
  - **[corrected]** LoRA rank 32, alpha 64, dropout 0.05, on
    `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.
    The FFN projections were missing from this list before; they are
    what give the adapter coverage of every layer's MLP path. On
    Qwen3.5, remember that q/k/v/o_proj only exist in the full
    attention layers — see [LORA_TARGETS.md](LORA_TARGETS.md) before
    committing 5,000 steps.
  - **[corrected]** Projection shape `(1024, 2560)`: Parakeet TDT 0.6B
    v2's `d_model` is 1024, not 512.
  - Precision: bfloat16
  - Steps: 5 000 (one epoch over train-clean-100)
- Evaluation: WER on LibriSpeech dev-clean and dev-other every 500
  steps, with per-clip predictions dumped to disk. Keep every
  checkpoint: Stage 0 showed held-out WER peaks long before the loss
  stops moving, so which checkpoint to ship is an empirical question
  the eval set has to answer.
- Expected outcome: WER close to or under Canary-Qwen-2.5B at 1.7B, with
  the projection doing the heavy lifting. (Rough target: 5-8 % on
  dev-clean, 12-18 % on dev-other.)

## Stage 2 — wider general English (rented GPU)

Goal: close the gap to Canary-Qwen-2.5B on a broader training set.

- Data: GigaSpeech XS subset (~ 250 hours curated, balanced across
  audiobook, podcast, YouTube).
- Compute: 1x NVIDIA H100 80GB, ~ 24 hours.
- Hyperparameters: same as Stage 1, with 2 epochs.
- Evaluation: WER on LibriSpeech dev-clean and dev-other, plus a
  held-out GigaSpeech dev slice.

## Stage 3 — Caribbean English (rented GPU)

Goal: the Caribbean-specific lift. The model should now start picking up
local names, accents, and prosody.

- Data: Common Voice 17 English speakers from Caribbean islands
  (Jamaica, Trinidad, Barbados, Guyana, Belize). The CV 17 dataset has
  speaker metadata we can filter on. Plus any CV Caribbean clips that
  the project's existing data pipeline has already produced.
- Compute: 1x NVIDIA H100 80GB, ~ 12 hours.
- Recipe: continue from Stage 2 checkpoint, lower lr to 5e-5, 2 epochs.
- Evaluation: a small Caribbean-specific WER set we curate from
  Common Voice 17 Caribbean speakers. Compare against the unadapted
  baseline.

## Stage 4 — Barbadian Parliament (rented GPU)

Goal: this is the model that ships. The model should produce strong
Barbadian proper-noun transcripts.

- Data: a small, paired Barbadian Parliament audio dataset. We need
  (a) audio of the House of Assembly or Senate, (b) transcripts that
  match the audio. The transcripts come from a frontier model first
  (Whisper-large-v3 or Qwen2-Audio), then are corrected by hand.
- For the first run, even 5-10 hours of paired audio is enough to
  shift the model's behaviour. BimOmni trained on text only and still
  moved the needle on proper nouns; we expect the audio supervision to
  move it further.
- Compute: 1x NVIDIA H100 80GB, ~ 6 hours.
- Recipe: continue from Stage 3, lower lr to 2e-5, 1 epoch.
- Optional sub-stage: also unfreeze a low-rank LoRA on the
  FastConformer encoder (rank 8 on the conv + attn layers). This is
  where the model can pick up Barbadian phonetics directly from
  audio without forgetting the English pretraining.

## Stage 5 — evaluation and release (local + HF)

Goal: prove the model is good and publish it.

- Evaluation sets:
  - LibriSpeech test-clean and test-other (sanity check)
  - GigaSpeech test (sanity check)
  - Internal Caribbean test set from Common Voice 17
  - Held-out Barbadian Parliament clips (10-20 minutes, hand-transcribed)
  - The BimOmni 10-proper-noun table (legacy comparison)
- Metrics: WER, proper-noun F1, RTFx on Mac M4.
- Publication: push adapter weights to
  `hammertoe/bim-asr-1b` (or similar), with a model card that
  mirrors the BimOmni layout.
- Leaderboard: request a slot on the Open ASR Leaderboard via PR. If
  accepted, that is the strongest single-number signal.

## Hyperparameter ledger

| Symbol             | Value  | Why                                       |
| ------------------ | ------ | ----------------------------------------- |
| lora_rank          | 32     | Matches Canary-Qwen-2.5B config           |
| lora_alpha         | 64     | 2x rank is the standard recipe            |
| lora_dropout       | 0.05   | Mild regularisation                       |
| **[corrected]** lora_targets | q,k,v,o + gate,up,down_proj | FFN projections are what reach every layer on a hybrid model |
| **[corrected]** encoder_hidden | 1024 | Parakeet TDT 0.6B v2 `d_model`; 512 was wrong |
| llm_hidden         | 2560 (4B) / 2048 (2B) | Read from the model config, never hard-coded |
| encoder_lr_mult    | 0.0    | Encoder is frozen in stages 1-3           |
| projection_lr_mult | 1.0    | Projection is the bridge, full speed      |
| lora_lr_mult       | 1.0    | LoRA is the new knowledge, full speed     |
| audio_window       | 30 s   | Canary-Qwen-2.5B used 40 s; 30 is safer   |
| max_tokens         | 1024   | Matches Canary-Qwen-2.5B                  |
| label_smoothing    | 0.1    | Standard for seq-to-seq                   |
| warmup_steps       | 500    | Roughly 10 % of Stage 1 schedule          |
| weight_decay       | 0.0    | LoRA does not need regularisation         |
| precision          | bf16   | H100 has good bf16 throughput             |
| grad_clip          | 1.0    | Standard                                  |

## Loss curves we expect

Stage 1 should show a clean monotonic decline in cross-entropy loss on
the train set, with the eval WER falling roughly in step. Stages 3-4
may have a brief uptick at the start (the Caribbean data is
out-of-distribution for the Stage 2 model) before the loss re-converges.

If we see divergence at any stage, halve the learning rate and resume
from the previous checkpoint.

**[corrected] "The eval WER falling roughly in step" with the loss is
not a safe assumption.** Stage 0 measured the two coming apart: loss
went 1.77 → 0.008 → 0.00001 while held-out WER had already stopped
improving and the gradient norm had reached zero. Treat training loss as
a wiring signal and checkpoint selection as an eval question. If a stage
needs a stopping rule, it is "best dev WER", not "final loss".

## What "done" looks like

- Stage 0: **[done]** the stack runs end to end on an M4 with the real
  encoder and the real LLM, and transcribes held-out speech at 19.0 %
  WER on a synthetic corpus.
- Stage 1: dev-clean WER <= 8 % within 5 000 steps.
- Stage 2: dev-clean WER <= 6 %.
- Stage 3: Caribbean WER within 2 % of the US WER.
- Stage 4: proper-noun F1 on the BimOmni 10-proper-noun set >= 9/10.
- Stage 5: model card published, leaderboard entry requested, M4 RTFx
  published.
