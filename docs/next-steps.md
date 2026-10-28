# Next Steps

Immediate priorities before building. Order matters — earlier items unblock later ones. Full architecture and track reasoning live in `strategy.md`.

## Reference projects

HardEars sits on top of:

- `../bimomni` — BimOmni. Owns the Barbados Parliament corpus and the Gemini batch script.
- `../bimgemma` — ASR baseline bake-off. Provides the evaluation harness.

Reuse these rather than rebuild.

## Decisions already made

- [x] Track: **Best Apps and Agents.** See `strategy.md`.
- [x] Architecture: fine-tuned ASR (Nemotron-speech) + Nemotron LLM reasoning on Token Factory + Gradio/web app surface. See `strategy.md`.
- [x] Hugging Face collision check for **HardEars**. Clean (0 models, 0 datasets, 0 spaces, both usernames available). See `naming.md`.
- [x] Base model: **`nvidia/nemotron-speech-streaming-en-0.6b`** (locked). It is the model the existing parliamentary corpus is already aligned to.
- [x] Training corpus: **`hammertoe/barbados-parliament-clips-v1`** (815+ clips, 812 video directories). Audio + quality scores already collected.
- [x] Evaluation methodology: reuse the **bimgemma bake-off harness** (12-clip / 34-entity held-out, entity recall, WER, hallucination probes).

## Before coding

- [ ] **Populate the transcripts.** Run `bimomni/scripts/gemini_transcribe_batch.py` against the deferred clips. The manifest schema already carries `gemini_transcript` so no migration is needed. This is the single biggest unblocker.
- [ ] **Verify the post-filter clip count.** Apply the parl quality thresholds (`utmos >= 3.2`, `sigmos >= 3.0`, `flatness <= 0.55`) plus a transcript-success filter. Confirm we have enough clips for a meaningful fine-tune.
- [ ] **Spot-check the Gemini transcripts.** Listen to ~20 clips against Gemini output. Look for Bajan place-name corruption, parliamentary procedure being misinterpreted, hallucinated proper nouns. Decide whether Gemini transcripts are good enough as ground truth or need human correction.
- [ ] **Confirm Nebius AI Cloud GPU credits and quota.** Check what GPU types (H100, Blackwell) and how many hours are available in the submission period.
- [ ] **Confirm Token Factory quota.** Nemotron 3 Ultra + Nano/Super need to be served during the demo and submission period.
- [ ] **Lock the held-out eval set.** Reuse the bimgemma 12-clip fixture or build a parliamentary-specific one alongside.

## Data

- [ ] Populate transcripts (above).
- [ ] Build the NeMo manifest from `MANIFEST.jsonl` with audio paths → transcripts.
- [ ] Apply quality + transcript-success filters.
- [ ] Define the train/eval split (held out by `video_id`, not by clip, to prevent bleed).

## Model

- [ ] Fine-tune `nvidia/nemotron-speech-streaming-en-0.6b` on the filtered corpus using NeMo ASR recipe (CTC or RNN-T — confirm against the model's head).
- [ ] Run on Nebius AI Cloud GPUs.
- [ ] Evaluate against the base model: WER, entity recall, hallucination probes (bimgemma harness).
- [ ] Capture before/after audio + transcript examples for the Devpost submission.

## App and agent

- [ ] Build the Gradio surface: side-by-side base vs HardEars, plus a downstream task (Barbados place-name entity extraction is the current favourite).
- [ ] Wire up the Nemotron LLM reasoning layer via Token Factory.
- [ ] Pick the demo hosting: Nebius Serverless Endpoints, Hugging Face Space, or local.

## Submission and release

- [ ] Fill in the remaining Devpost sections (How we built it, Challenges, Accomplishments, What we learned).
- [ ] Lock the HF release slug: `hammertoe/HardEars-0.6B`.
- [ ] Record the 3-minute demo video covering the side-by-side ASR comparison and the downstream task demo.
- [ ] Decide whether the populated transcripts become part of a v2 dataset release on the Hub.

## After the hackathon

- [ ] Public model release with a model card documenting training data, evaluation, and known limitations.
- [ ] Explore extending the approach to other underrepresented Caribbean English varieties.
- [ ] Consider diarisation (pyannote 3.x) as a V2 of the corpus pipeline so multi-speaker parliamentary clips are no longer rejected.
