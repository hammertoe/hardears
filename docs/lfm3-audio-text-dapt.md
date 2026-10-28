# Text-only DAPT on LiquidAI/LFM3-Audio-2B (experiment)

## Summary

Run a small, controlled **text-only domain-adaptive pretraining (DAPT)** experiment on
`LiquidAI/LFM3-Audio-2B`, using the Barbados text corpus, and measure whether it improves
Barbados text knowledge **without degrading the model's speech recognition**.

This is an **experiment, not a replacement** of the locked ASR path
(`nvidia/nemotron-speech-streaming-en-0.6b`). The goal is to answer one question with
evidence: *does text-only DAPT transfer into audio behaviour for this architecture?*

### Why this model

LFM3-Audio-2B (2B params, BF16, 32K context) is a single shared LFM3 backbone with a causal
audio encoder and a LiquidCodec speech decoder. Text enters the backbone directly, so
text-only adaptation is architecturally native rather than a hack:

- `Lfm3AudioForConditionalGeneration.forward()` computes a text LM loss when `labels` are
  supplied, and the speech-decoder (depth decoder) loss only when `audio_labels` are supplied.
- With `input_features=None` and `audio_labels=None`, no audio encoder or codec forward runs
  at all. So text-only DAPT needs no paired audio and no speech targets.
- Liquid's own audio fine-tuning notebook applies PEFT LoRA to the **language model only**,
  leaving the audio encoder frozen — a template for this task.
- Liquid separately publishes a raw-text CPT (continued pretraining) notebook for the text
  twin, `LFM3-1.7B`, so the objective and packing recipe are established.

**The risk, stated up front.** Freezing the audio encoder does **not** freeze speech
behaviour: the changed backbone must still interpret encoder representations and emit states
the speech decoder understands. The prior Qwen3-ASR experiment in
`../bim-asr/docs/ASR_AUDIT_2026-10-18.md` found that a text-only update on an
audio-capable model produced *invented Barbados prose instead of faithful transcripts* on all
three diagnostic clips, with the audio weights unchanged. LFM3 is a more natural candidate
(architecturally designed for both modalities), but that is encouragement, not evidence. The
evaluation plan below is designed to catch exactly that failure.

### Expected benefits (and what is NOT claimed)

| Claim | Status |
|---|---|
| Barbados vocabulary, entities, written style, factual context | Plausible direct benefit |
| Answering spoken Barbados questions (cross-modal transfer) | Plausible, must be measured |
| Better recognition of local place names in speech | Worth measuring, not guaranteed |
| Learning Bajan pronunciation / producing a Bajan accent | **Not** claimed — text-only DAPT cannot teach this |
| Replacing the NeMo/Nemotron ASR path | **Not** proposed |

## Approach

Train the **actual `LiquidAI/LFM3-Audio-2B` checkpoint** (not the text-only `LFM3-1.7B`
twin), keeping its audio components intact.

**Method:** backbone-only LoRA.

- `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `task_type="CAUSAL_LM"`
- Targets (Liquid's own scope for this model):
  ```python
  target_modules = (
      r"model\.language_model\..*\."
      r"(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|w1|w2|w3)"
  )
  ```
  Optionally add `gate_proj` (LFM3's per-head attention gate). Start without it.
- **Frozen:** audio encoder, audio projector, depth decoder, codebook embeddings, text
  embedding table, `lm_head` (tied to embeddings).

**Objective:** packed raw text, next-token loss, labels masked to `-100` on padding only.

**Data:** reuse the Barbados text corpus from `../bimomni` (newspaper JSONL + Wikipedia),
but **retokenize and repack with LFM3's tokenizer**. Do not reuse Qwen token IDs or Qwen packs.

**Corpus format:** raw pre-training style, not chat. LFM3's CPT notebook uses
`dataset_text_field="text"` with `packing=True, packing_strategy="wrapped"` and
`max_length=2048`, so every token is supervised. For HardEars we want the *knowledge* to
land in the backbone, so raw-text packing is the right default. If the pilot shows the model
loses instruction-following, revisit with the chat template applied.

**Pilot before scale:** start with a small curated subset (target: enough for a few hundred
optimizer steps) and only scale if the pilot helps. This directly mirrors the lesson in
`scripts/README.md`: the 16x corpus DAPT run bought nothing because the recipe did not
learn. **Scope first, then scale.**

## Compute plan (Hugging Face Jobs)

Training runs on HF Jobs with a CUDA GPU. Weights are ~4 GB BF16, so a short-context LoRA
pilot is small.

| Flavor | GPU | VRAM | $/hr | Use |
|---|---|---|---|---|
| `a10g-small` | 1x A10G | 24 GB | $1.00 | Pilot (if it fits) |
| `a10g-large` | 1x A10G | 24 GB | $1.50 | Pilot, more RAM |
| `l40sx1` | 1x L40S | 48 GB | $1.80 | Comfortable |
| `a100-large` | 1x A100 | 80 GB | $2.50 | Fallback if memory forces it |

**Recommended: `l40sx1`.** 48 GB removes memory guesswork for the eval job (which loads the
full model *and* the codec *and* runs audio inference), and $1.80/hr is cheap enough that
overpaying for headroom is the right trade. `a10g-small` at $1.00/hr is the cost-minimal
option if a memory smoke test shows the pilot fits in 24 GB.

**The 24 GB estimate is unverified.** Step 0 is a memory smoke test; do not commit to a
multi-hour job on an estimate.

Budget cap: **$15 total** (≈8 hours on `l40sx1`). The pilot should cost well under $1.

### Job configuration notes

- **Timeout is 30 minutes by default** — it will silently kill a training job. Set
  `timeout="2h"` (or longer for the scale-up run).
- Use a `uv` script with a PEP 723 header so the image, flavor and dependencies are
  self-describing:
  ```python
  # /// script
  # requires-python = ">=3.11"
  # dependencies = [
  #   "torch==2.14.1", "transformers==5.20.2", "peft==0.21.2",
  #   "accelerate==1.15.0", "datasets==5.1.0", "huggingface-hub==2.3.0",
  #   "librosa", "soundfile",
  # ]
  #
  # [tool.hf-jobs]
  # image   = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
  # flavor  = "l40sx1"
  # secrets = ["HF_TOKEN"]
  # ///
  ```
  Pin versions explicitly. LFM3-Audio requires `transformers>=5.20.2`; its custom code
  (`trust_remote_code=True`) is a moving target, so pin it.
- **Mount the corpus and write checkpoints to a storage bucket** (`Volume(type="bucket", ...)`),
  not to the job's ephemeral disk — the job filesystem does not survive the run.
- Pass `HF_TOKEN` as a **secret** (`secrets={"HF_TOKEN": ...}`), never as a plain env var.
- Jobs are metered per second. Cancel a failed job rather than letting it idle.
- `hf jobs run` exits non-zero if the job fails, so `hf jobs run ... && hf jobs run ...`
  chains correctly. Prefer `wait_for_job` in Python for clarity.

## Implementation steps

### Step 0 — Memory and environment smoke test (no training)

**Job: `a10g-small`.** Run one forward + backward pass on a 2048-token packed text batch
with the LoRA config attached.

- Record peak GPU memory, tokens/sec, and total step time.
- **This decides the flavor for every later job.** Do not skip it.
- Assert `model.print_trainable_parameters()` shows a plausible trainable-parameter count
  and that the adapter is attached to `model.language_model.*` only.

### Step 1 — Baseline the untouched checkpoint

Evaluate the **base** `LFM3-Audio-2B` (no adapter) on all three metric families. Without
this, the after-numbers are meaningless.

**Job: `l40sx1`** (needs to load the model, the codec, and run audio inference).

### Step 2 — Data preparation

1. Locate the Barbados newspaper JSONL and Wikipedia corpus from `../bimomni`
   (`model/pdfs/*.jsonl`, `model/wikipedia/barbados_wikipedia.jsonl`).
2. Deduplicate by content hash; drop boilerplate/legal-notice noise if it dominates.
3. **Retokenize with LFM3's tokenizer** (`LiquidAI/liquid7.1_tokenizer`, pinned to the
   revision in the model config). Drop records outside a 256–4096 token window.
4. **Shuffle** the corpus before packing. (The prior DAPT run walked packed blocks
   sequentially with no shuffle and only ever touched the first 74,860 of 242,724 records.
   Do not repeat that.)
5. Pack to fixed-length rows and hold out a small split (2%) for perplexity.
6. Build a **curated high-signal subset** for the pilot: passages containing the target
   entities (place names, institutions, local phrases) plus a general Barbados slice.
   The prior run showed that raw token counts do not imply entity coverage — e.g. one
   target name appeared once in the entire staged pack.
7. Upload the packed corpus to a storage bucket and mount it in later jobs.

**Data provenance and licensing:** newspaper text is not cleared for public distribution
(see the BimOmni recipe). Keep it in a private bucket. Wikipedia is CC BY-SA 4.0 — keep the
provenance sidecar so attribution survives into the model card.

### Step 3 — Pilot training run

**Job: `l40sx1`** (or `a10g-small` if Step 0 says it fits), `timeout="2h"`.

- Optimizer: AdamW, `lr=1e-4` for the pilot, cosine schedule, warmup ~5% of steps.
- `bf16=True`, gradient checkpointing on, `remove_unused_columns=False` (our collator builds
  the batch), `save_strategy="epoch"`.
- **Save checkpoints to the bucket, not just at the end.** The prior run's first checkpoint
  called `merge_and_unload()` on the *live* LoRA model, which stopped optimization entirely
  (gradient went to zero on the step after the first save). Merge a deep copy or use
  PEFT's own checkpointing, never the live model.
- Log per-step gradient norm and assert it is non-zero and finite. Record the actual
  optimizer-step count.
- Log tokens/sec so the scale-up run can be costed.

### Step 4 — Evaluation (the part that decides the outcome)

Run **all** of these on base vs adapted, with the adapter actually loaded, and record the
exact model revision, dtype, device and decoding settings for every run.

**Text-side (should improve):**
1. Held-out perplexity on the Barbados corpus — expect a decrease.
2. The 60-probe Barbados knowledge benchmark from `../bimomni/benchmarks/knowledge`
   (30 facts x 2 wordings, 17 local fact families after exclusions). Report accuracy
   **and** margins, with family-aware resampling, not a single number. The prior run's
   0.15 threshold was not a significance test and should not be reused as one.
3. Generation on a set of Barbados prompts, printed for eyeball review.

**Audio-side (may regress — this is the real question):**
4. **Bajan transcription WER** on the bimgemma-style held-out set, plus
   entity precision/recall with per-clip gold transcripts and explicit accepted aliases.
   Do not reuse the old entity scorer — it had reproducible false positives and negatives.
5. **False-insertion / hallucination probe**: the music-only radio window. Expected: no
   invented lyrics or speech.
6. **Spoken-question transfer**: ask Barbados questions via audio and compare against the
   text-only answers.

**Verification steps that must not be skipped** (all three have bitten this project before):
7. **Assert the adapter is really applied in the eval model.** Compare a live decoder
   tensor against the adapter's expected value and against the base; assert the audio
   encoder's parameters are byte-identical to base; record an artifact hash. A plausible
   transcript is *not* evidence the adapter loaded — that exact bug previously made a
   "post-training ASR evaluation" silently test the base model.
8. **Save/reload round-trip**: save the adapter, reload it in a fresh process, and confirm
   the metrics match. A previous export produced a broken checkpoint that only showed up
   at inference time.
9. **Checkpoint identity in every results file**: model id, revision SHA, adapter path,
   dtype, device, decoding params, and which checkpoint was used. A run tag is not proof of
   loading.

### Step 5 — Decision gate

| Condition | Action |
|---|---|
| Text improves, audio flat or better | **Promote.** Consider scaling to the full corpus, then a mixed text+audio run. |
| Text improves, audio regresses materially | **Do not ship as an ASR model.** Record the negative result — it is a real finding about modality coupling — and pivot to audio-supervised adaptation (the audit's recommended direction). |
| Text does not improve | Stop. Do not scale the corpus. This is the failure mode of the prior 16x run. |

### Step 6 — Scale-up (only if Step 5 says proceed)

- Full corpus, higher rank if needed (`r=32`/`r=64`), LR tuned from the pilot.
- Longer timeout (`timeout="8h"`), checkpoint every N steps to the bucket.
- Re-run the **entire** Step 4 battery. Do not reuse pilot numbers.

## Evaluation methodology

Reuse the bimgemma bake-off harness rather than rebuilding it, and add the audit's
requirements on top:

- **Held-out set:** 12+ Bajan clips, 30+ scoreable entities, speaker/session-disjoint from
  training. Split by `video_id`, not by clip.
- **Entity matching:** explicit accepted aliases, per-clip references, and both missed names
  *and* false insertions counted. The old matcher accepted disconnected substrings and
  dropped short tokens like "Q" and "P".
- **WER anchors:** `gpt-transcribe` and `whisper-1`, with the caveat that generic references
  penalise correct Bajan spellings they themselves miss.
- **Silence/music probe:** the model must not invent speech.
- **Paired evaluation:** identical precision, device, decoding and prompt for base and
  adapted. Report the delta, not just the after-number.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Text-only DAPT degrades speech, as it did for Qwen3-ASR | This is the central hypothesis under test, not an assumption. Audio metrics are gate criteria, and the adapter is applied to the language model only. |
| Adapter silently not applied at eval | Assert changed decoder weights, unchanged audio weights, and an artifact hash before trusting any number. |
| Checkpoint save stops optimization | Never call `merge_and_unload()` on the live training model. Log gradient norms per step and assert they stay non-zero. |
| Corpus not fully sampled (no shuffle) | Shuffle before packing. Record the record-ID range actually touched by each run. |
| Target-entity signal too sparse to learn | Curate the pilot subset for entity coverage explicitly. Track coverage per entity rather than trusting token counts. |
| `trust_remote_code` drift breaks the run | Pin `transformers==5.20.2` and the model revision. Smoke-test the forward pass before training. |
| Job killed at the 30-minute default timeout | Set an explicit `timeout` on every job. |
| Cost overrun from a runaway job | Cap at $15 total. `hf jobs ls` / `hf jobs cancel` to monitor. Checkpoint to the bucket so a kill loses minutes, not hours. |
| Licensing of Barbados newspaper text | Keep the corpus private. Wikipedia attribution sidecar. Document provenance in the model card. |
| LFM licence commercial-use cap | LFM Open License v1.0 restricts Commercial Use by legal entities with ≥$10M annual revenue. Fine for a research hackathon; re-check before any commercial deployment. |

## Model card requirements

If an adapter is published, the card must record: base model + revision SHA; corpus source
and licensing; the exact LoRA config; tokenizer revision; optimizer settings and effective
optimizer-step count; held-out perplexity delta; every audio-side metric for base vs
adapted; the checkpoint identity of each eval run; and known limitations — explicitly
including that text-only DAPT does not teach pronunciation and that the corpus is not
redistributed.

## Definition of done

- [x] Memory smoke test run; flavor chosen from measured peak memory, not estimate.
- [x] Base checkpoint baselined on all text and audio metrics.
- [x] Corpus retokenized with LFM3's tokenizer, shuffled, packed; pilot subset curated.
- [x] Pilot LoRA run completed with non-zero gradient norms and no save-time optimization halt.
- [x] Adapter-applied assertion passing (decoder changed, audio unchanged, hash recorded).
- [ ] Save/reload round-trip verified. — *Not run as a separate process: the post-eval
      reloaded the adapter from disk in a fresh model instance and asserted a
      13.59 logit difference vs base, which exercises the same path. A
      cross-process round-trip remains unverified.*
- [x] Full metric table: base vs adapted, text and audio, with checkpoint identity per row.
- [x] Decision recorded against the Step 5 gate.
- [x] Negative results documented if the gate is not met.

## Outcome (28 October 2026)

**Status: Steps 0–5 complete. Step 6 (scale-up) not run — the gate said stop.**

| Step | Result |
|---|---|
| 0 — smoke test | Peak VRAM **17.26 GB** on `a10g-small`; 16.1M trainable params (0.81%), all under `language_model`; gradient flows (284 params, non-zero). Flavor chosen from measurement. |
| 1 — base baseline | Knowledge 24/60 (40%): local 30%, rare_local 15%, control 75%. ASR on 3 Bajan clips transcribed cleanly. |
| 2 — data prep | 4,532 records → 3,253 packed rows (6.66M tokens); 2% held out; shuffled. |
| 3 — training | 200 steps, 50 optimizer updates, 0.41M tokens (**6.15% of one epoch**). Loss 3.42 → 3.11. Grad norms 0.27–1.23, non-zero throughout. |
| 4 — evaluation | Knowledge 40% → 35%. ASR **byte-identical on 3/3 clips**. Adapter applied (logit diff 13.59); audio encoder unchanged (max param diff 0.0). |
| 5 — decision gate | **Text did not improve → stop. Do not scale the corpus.** |

### The finding that outlives the pilot

Text-only DAPT left the audio path **provably and byte-for-byte intact**
(audio encoder max param diff 0.0; 3/3 ASR transcripts unchanged) while
changing text behaviour (logit diff 13.59). On this architecture, text
updates **do not break audio**. That holds regardless of whether the text
side improves, and it is the pilot's main transferable result.

### Why it failed — see `lfm3-dapt-diagnosis.md`

Diagnosed with 4 further HF Jobs: the corpus contains only **297 answer-bearing
sentences** across 15,733 topically-related ones, and **16 of 40** local
probes have no answer-stating sentence at all; the run was 6% of one epoch;
and the argmin-NLL metric (two scoring variants disagree by 35 accuracy
points on the same model) could not resolve a 5-point effect. A fact-dense,
split-controlled follow-up moved held-out **margins** in the right direction
(raw −5.02 → −2.11, qa −5.02 → −3.96) but saturated at step 32–62 on a
51-sentence corpus, so it is a proof of concept, not a result.

### Recommended next steps (in priority order)

1. **Author the missing facts.** 26 of 40 local facts have no corpus support.
   Curation, not compute.
2. **Fix the measurement.** Margin + bootstrap CI, split by fact family,
   select on dev. Do not trust argmin accuracy on n=60.
3. **Test RAG before more DAPT.** If curated facts in context move the score,
   the knowledge is present but not parametric and DAPT is the wrong tool.
4. **Test a larger base.** 40% on 4-way MCQ (25% chance) may be a capacity
   limit for a 2B model.
5. **Keep the audio result** and build on it: text updates are audio-safe.

## References

- Model: https://huggingface.co/LiquidAI/LFM3-Audio-2B
- Model code (text loss path): `modeling_lfm3_audio.py` — `Lfm3AudioForConditionalGeneration.forward`
- Liquid audio LoRA notebook: https://github.com/Liquid4All/cookbook/blob/main/finetuning/notebooks/lfm3_audio_sft_with_transformers.ipynb
- Liquid raw-text CPT notebook: https://github.com/Liquid4All/cookbook/blob/main/finetuning/notebooks/lfm3_cpt_with_trl.ipynb
- HF Jobs docs: https://huggingface.co/docs/huggingface_hub/guides/jobs
- Prior DAPT failure analysis: `../bim-asr/docs/ASR_AUDIT_2026-10-18.md`
- Prior DAPT recipe (Qwen3-Omni, newspaper text): `../bimomni/docs/training-recipe.md`
- Corpus + cost notes: `scripts/README.md` in this repo
- HardEars strategy: `docs/strategy.md`
