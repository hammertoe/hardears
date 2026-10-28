# Strategy

Locked-in plan for the hackathon build. Decisions here feed `project.md`, `next-steps.md`, and `devpost-submission.md`.

## Reference projects

HardEars sits on top of substantial existing work in sibling repos. We do not start from zero.

- **`../bimomni`** — BimOmni, a Barbados-adapted version of Qwen3-Omni-30B-A3B-Instruct. Contains the **Barbados Parliament audio corpus** (`hammertoe/barbados-parliament-clips-v1`, 815+ clips, 812 video directories), the Gemini batch transcript-generation pipeline (`scripts/gemini_transcribe_batch.py`), the chunked audio transcription helpers (`src/bimomni/transcription/chunked.py`), and a 60-probe Barbados knowledge benchmark.
- **`../bimgemma`** — A baseline ASR bake-off of three omni models on held-out Bajan audio. Provides the **evaluation methodology** we will reuse: 12-clip / 34-entity held-out set, entity recall vs hand-written surface forms, WER against `gpt-transcribe` and `whisper-1` anchors, hallucination probes (music-only radio window), latency and RTF. The findings already establish that generic ASR fails on Bajan entities (`Kensington Oval` → `Kingsington`, `Rise Together` → `rice together`).

## Track

**Best Apps and Agents.** Confirmed against the four available tracks:

- Best Apps and Agents — chosen. Fits a "productivity tool or copilot" that "someone would actually use." Explicitly covers apps that wrap a fine-tuned model with LLM reasoning.
- Coding and Agentic Engineering — rejected. The track is for agents that *write, run, and test code in Token Factory Sandboxes*. A fine-tuned ASR model does not satisfy that, even if technically impressive.
- Personal AI — rejected. Track is about always-on personal assistants with persistent memory and reusable skills. HardEars could be packaged as one skill inside such an assistant, but the track centres the assistant, not the skill.
- Physical AI — rejected. Robotics, IoT, edge devices, on-device intelligence. No relevance.

The Grand Prize ($20k) is judged across all tracks, so the track decision affects positioning, not the prize ceiling.

## Architecture

HardEars is built as a **Bajan speech agent**, not a standalone model. The fine-tune is the differentiator, but the demo and the "complete, coherent product experience" require an app surface and a reasoning layer.

```
[Bajan audio]
      ↓
[ASR layer]      Fine-tuned speech model, served on Nebius AI Cloud
                 (or via Hugging Face / NeMo inference for the demo)
      ↓ raw transcript
[Reasoning]      Nemotron 3 Ultra via Nebius Token Factory
                 Entity extraction, dialect normalisation, summarisation, downstream tasks
      ↓
[Fast layer]     Nemotron Nano / Super via Token Factory
                 Cheap calls: language detect, formatting, transcript post-edit
      ↓
[App surface]    Gradio or lightweight web UI
                 Side-by-side base vs HardEars comparison + downstream task demos
```

The LLM layers are what unlock the "agent" framing. The fine-tune alone wouldn't satisfy the Design judging criterion.

## Data: Barbados Parliament corpus

The training corpus already exists on disk and on the Hub: **`hammertoe/barbados-parliament-clips-v1`** (built under `../bimomni/hammertoe__barbados-parliament-clips-v1/`).

### What's in it today

- **815+ clips** across **812 video directories** (Barbados Parliament Channel YouTube recordings, dated back to 2025).
- Each clip: 16 kHz mono `.wav`, plus a per-clip `.deferred.json` carrying pipeline metadata.
- Quality scores recorded per clip: UTMOS, SIGMOS, spectral flatness, SNR. Conservative thresholds (`utmos >= 3.2`, `sigmos >= 3.0`, `flatness <= 0.55`) gate keep/reject.
- Pipeline uses **Silero VAD** for voice activity, **Parakeet-tdt-0.6b-v3** for boundary/word-timestamp detection, and **Nemotron-speech-streaming-en-0.6b** as the transcription model.
- MANIFEST.jsonl at the dataset root (262 MB), plus per-video shards under `sharded_manifests/`.

### What's not in it yet

- **Transcripts are deferred.** Every `.deferred.json` has `"text": ""` and `"transcription_status": "deferred_v2"`. No `gemini_transcript` either.
- The Gemini batch script (`bimomni/scripts/gemini_transcribe_batch.py`) is the intended fill: submit, poll, collect, merge against `gemini-3.7-flash`. The `gemini_transcript` column is already in the manifest schema so no migration is needed.
- This is the **first unblocking step** for HardEars — without transcripts, there is no fine-tuning data.

### Provenance

Each clip carries its YouTube URL, recording date, and fetch timestamp. No redistribution concerns that have not already been addressed in the bimomni dataset card.

### Why parliamentary audio specifically

- Single-speaker regions (after VAD + Parakeet boundary carving). Multi-speaker heckling and applause is rejected.
- Consistent recording conditions (chamber mic) → clean audio, low SNR variance.
- Covers Barbados political vocabulary, place names, proper nouns — exactly what generic ASR fails on.
- Real Bajan speech in a controlled environment — the right starting point before tackling radio, TikTok, etc.

## Nebius split

The hackathon constraint is "run on Nebius Token Factory or Nebius AI Cloud, and use at least one NVIDIA open source model." HardEars uses both halves of that:

- **Nebius AI Cloud** — GPU compute for fine-tuning the ASR base model. H100 or Blackwell. This is where the training job lives. Reuses the parl pipeline's memory profile: ~1.2 GB each for Parakeet and Nemotron-speech at fp16, plus working memory — `l4x1` is enough for inference but fine-tuning benefits from more headroom.
- **Nebius Token Factory** — serverless inference for the Nemotron LLM reasoning and fast layers. Pay-per-token. This is where the agent's "thinking" happens.
- **NVIDIA open source models** — used in both halves: ASR base (see below), Nemotron 3 Ultra / Nano / Super for reasoning.

## Base model

Locked recommendation: **`nvidia/nemotron-speech-streaming-en-0.6b`**.

### Why this and not the alternatives

- The parliamentary corpus was specifically built around this model. Its pipeline metadata records `nemotron_model: nvidia/nemotron-speech-streaming-en-0.6b`. Fine-tuning it on transcripts generated for it closes the loop cleanly.
- 0.6B size matches the working HF slug `hammertoe/HardEars-0.6B`.
- Streaming-friendly, useful for future live-transcription use cases.
- Preserves the "Nemotron speech" framing that was the original HardEars thesis — and this time the name actually maps to a real model.

### Alternatives considered

| Model                                       | Size | Notes                                                                                  | Verdict                              |
|---------------------------------------------|------|----------------------------------------------------------------------------------------|--------------------------------------|
| `nvidia/nemotron-speech-streaming-en-0.6b`  | 0.6B | Used by the existing parl pipeline. Streaming. CC-BY-4.0.                              | **Locked — primary**                 |
| `nvidia/parakeet-tdt-0.6b-v3`               | 0.6B | Most popular NVIDIA ASR on HF. Multilingual. Already used for boundaries in the corpus. | Fallback if Nemotron-speech doesn't fine-tune cleanly |
| `nvidia/canary-1b-v2`                       | 1B   | ASR + speech translation. 25-language support.                                          | Stretch — only if we want translation|
| `nvidia/nemotron-3.5-asr-streaming-0.6b`    | 0.6B | Newer Nemotron-branded ASR. Multilingual.                                               | Out — too new, no existing corpus aligned to it |

### Original framing — correction

The earlier docs and brainstorm referred to "NVIDIA's Nemotron speech model" without a specific checkpoint. The actual model that fits that description, and which the parliamentary corpus is already aligned to, is **`nvidia/nemotron-speech-streaming-en-0.6b`**. Not Parakeet (which was used for boundaries, not transcription), not Nemotron-3.5-ASR (which is too new).

## Fine-tuning approach

NeMo toolkit. The corpus is already 16 kHz mono with manifest entries in NeMo-compatible JSONL. Recipe pattern:

1. Run `bimomni/scripts/gemini_transcribe_batch.py` against the deferred clips → populate `gemini_transcript` (and optionally the canonical `transcript` field if Gemini output is deemed higher quality than the eventual Nemotron re-run).
2. Build a NeMo manifest mapping audio paths → transcripts. Filter by quality thresholds.
3. Fine-tune Nemotron-speech-streaming-en-0.6b on the filtered manifest using NeMo's ASR fine-tuning recipe (CTC or RNN-T depending on the model's head).
4. Track WER on a held-out split (10-15% of clips, by video to avoid bleed).
5. Compare against `nvidia/nemotron-speech-streaming-en-0.6b` base on the bimgemma-style eval (entity recall + WER + hallucination).

The bimomni text-only DAPT recipe (`docs/training-recipe.md`) does **not** apply here — that recipe is for the Qwen3-Omni thinker on newspaper text. ASR fine-tuning uses NeMo directly.

## Evaluation methodology

Reuse the bimgemma bake-off methodology verbatim:

- Held-out set: 12+ Bajan clips, 30+ scoreable spoken entities (place names, institution names, local phrases). Mix of parliamentary clips and outside-domain audio to test generalisation.
- Entity recall against hand-written surface forms. Both spellings are accepted where usage is genuinely variable (`Kensington Oval` vs `Kingsington Oval`).
- WER against `gpt-transcribe` and `whisper-1` anchors (with the bimgemma caveat that generic references penalise correct Bajan spellings they themselves miss).
- Hallucination probe: a music-only radio window. Expected behaviour: model should not invent lyrics or speech. BimOmni/Qwen transcribed lyrics anyway; that's the baseline; HardEars should not regress.
- Latency, RTF, peak RSS — for the demo, not just the bake-off.

The full bimgemma harness (`bimgemma/runner.py`, `bimgemma/candidates/render_prompt.py`) can be reused as the basis. HardEars adds a fourth candidate: the fine-tuned Nemotron-speech model.

## Demo plan

The Devpost submission requires a working demo and a 3-minute video. Both should show:

1. **A real Bajan audio clip** — preferably a held-out parliamentary clip plus a held-out non-parliamentary clip (radio, interview). Transcribed by base Nemotron-speech and by HardEars. Side-by-side. WER improvement is the headline number.
2. **A downstream task** the agent enables. The strongest candidate is **Barbados place-name entity extraction** — "list every parish, town, and landmark mentioned in this 10-minute clip." That demonstrates the fine-tune *and* the Nemotron LLM reasoning layer doing something useful together.
3. **Honest framing of limitations** — accents still struggled with, audio conditions still handled poorly, what data would close those gaps.

## Risk register

- **Transcripts don't exist yet.** The Gemini batch pipeline must run successfully and cheaply. Cost estimate (Gemini 3.7 Flash): ~$0.50-2.00 for the full corpus at clip-level batching. Need to verify before committing.
- **Transcript quality.** Gemini re-transcription of parliamentary audio may still produce noise (mispelled Bajan place names, "interpreted" parliamentary procedure as ordinary speech). Human spot-check on a sample before training.
- **Fine-tune data scarcity after filtering.** Quality thresholds may cut the 815 clips to a much smaller training set. Need to know the post-filter count before sizing the fine-tuning run.
- **Time budget.** Deadline is 30 Oct 2026. Pipeline stages — transcript generation, manifest build, fine-tune, eval, demo — all eat time sequentially.
- **AI Cloud GPU availability and credits.** Need to confirm Nebius credit allocation and GPU quotas before committing to a fine-tuning recipe.
- **Demo latency.** Nemotron-speech at 0.6B should be fast, but the full agent (ASR + Token Factory reasoning + Gradio) needs to fit inside a 3-minute video.

## Open questions

- [ ] Run Gemini batch against the deferred clips and confirm the post-filter clip count.
- [ ] Confirm Nebius AI Cloud GPU credits and quota for the submission period.
- [ ] Confirm Token Factory availability of Nemotron 3 Ultra / Nano during the demo period.
- [ ] Pick the downstream task that best showcases the agent (place-name extraction is the current favourite).
- [ ] Decide whether to deploy the demo on Nebius Serverless Endpoints, Hugging Face Spaces, or local.
- [ ] Decide whether HardEars gets its own HF dataset repo or extends the existing `barbados-parliament-clips-v1` with the populated transcripts.
