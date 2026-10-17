# Stage 0 results

> Produced on an M4 Pro (48 GB, no CUDA) on 17 October 2026.
> Reproduce with the commands in the README; nothing here depends on
> `artifacts/` or `data/`, both of which are gitignored and rebuildable.
> Raw per-clip predictions are in `artifacts/evals/*.json`.

Stage 0 in `docs/TRAINING.md` asks for one thing: prove the SALM
scaffolding works end to end on the laptop and see the loss go down.
This records what that actually produced, including the parts that are
not good news.

## TL;DR

- The full stack runs on the Mac: frozen Parakeet encoder → projection
  bridge → Qwen3.5-2B + LoRA → text. No NeMo, no CUDA.
- The bridge trains. Cross-entropy loss goes 5.11 → ~0.01.
- The trained SALM transcribes **unseen sentences in unseen voices at
  19.0% WER**, with errors that are visibly phonetic
  ("shore" → "shorn", "curved" → "crossed").
- An untrained bridge transcribes nothing. It answers like a text-only
  assistant, every time.
- The frozen encoder is provably correct: the original Parakeet TDT
  decoder, running off our extracted encoder weights, gets 2.1% WER on
  the same clips.

## Results

Greedy decode, batch size 1, one 30 s window per clip.

| system | eval set | clips | WER | MER | exact |
| ------ | --------- | -----: | --: | --: | ----: |
| **SALM, Stage 0 trained** | eval_seen (seen text, held-out voices) | 24 | **0.0%** | 0.0% | 24/24 |
| **SALM, Stage 0 trained** | eval (unseen text, held-out voices) | 24 | **19.0%** | 19.0% | 15/24 |
| SALM, untrained bridge | eval | 12 | 351.9% | 95.0% | 0/12 |
| Parakeet TDT 0.6B v2 (reference ASR) | eval | 24 | 2.1% | 2.1% | 22/24 |
| Parakeet TDT 0.6B v2 (reference ASR) | eval_seen | 48 | 6.3% | 6.3% | 37/48 |

The two SALM rows differ only in whether the *sentence* was in training.
The audio in both is spoken by voices (Moira, Tessa) that never appear
in training, at speeds the training set also never used for eval.

### What it actually says

```
reference : The light faded slowly behind the hills.
Stage 0   : The lake waited slowly behind the hill.

reference : A woman walked along the quiet shore.
Stage 0   : A woman walked along the quiet shorn.

reference : The road curved past the old church.
Stage 0   : The road crossed past the old church.

reference : Rain fell gently on the roof.
Stage 0   : Rosa followed gently on the road.
```

That is real rudimentary transcription. The frame-by-frame errors are
phonetic substitutions, not random tokens, which is what a working
acoustic path looks like at 19% WER.

### Untrained, for contrast

```
reference : The light faded slowly behind the hills.
untrained : I am ready to transcribe the audio. Please provide the text
            or paste the transcript here, and I will process it for you.
```

Twelve clips, twelve variations on "please provide the text". The base
LLM is answering a text conversation it does not understand, which is
exactly what randomly initialised audio embeddings should produce. This
is why the smoke test checks that the output *changes when the audio
changes* rather than checking for English words.

### Cost

| | |
| --- | --- |
| trainable | 26.0M of 1.91B (projection 2.10M + LoRA 23.9M) |
| throughput, training | ~2.2 it/s, bf16, MPS, batch size 1 |
| throughput, inference | **4.6x real time** on MPS (bf16), 1 window |
| training data | 420 sentences, 1,260 clips, 49 min of audio, 4 voices, 3 speeds |
| total training | ~12k steps, ~90 min on the laptop |
| reference ASR | 3.0x real time, CPU, fp32 |

### Loss

| checkpoint | loss |
| ---------- | ---- |
| step 0 (random bridge) | 5.11 |
| step 2,000 | 1.77 |
| step 6,000 (240-sentence corpus) | 0.008 |
| step 12,000 (+180 sentences, warm start) | 0.00001 |

The loss keeps falling long after the model has stopped improving on
held-out data. Loss is a wiring check here, not a quality signal.

## Two bugs this stage found

Both produced numbers that looked real and were not.

### 1. Generation never stopped

Qwen3.5-2B ships **no `generation_config.json`**, and its tokenizer
reports `eos_token = ''` with `eos_token_id = 248046` while
`pad_token_id = 248044`. Calling `generate()` with only `pad_token_id`
therefore never stops: the model emitted the correct sentence and kept
going, sometimes four times over. One runaway generation moved an
average WER by ~80 points, while the first six printed examples were
all exact.

Fix: `stop_token_ids()` in `src/bim_asr/model.py`, used by the trainer,
the transcriber and the smoke test. The same eval went **92.9% → 19.0%
WER** on identical weights, which is the clearest evidence in this
whole exercise that a metric without per-clip output is not evidence.

### 2. Stale audio against fresh text

`synthesize()` skips a wav that already exists. Filenames were
`<split>_<index>_<voice>_<speed>.wav`, with the index carrying no
information about the sentence. Re-running the corpus builder with a
different sentence pool kept the **old audio** and wrote the **new
text** into the manifest.

This showed up as the Parakeet reference ASR scoring 94% WER on one
split and 2.1% on another, which is not a plausible result for either
system. Fix: a SHA-1 digest of the sentence in the filename, plus
`sorted()` before sampling so the sample does not move when the pool
grows. The reference ASR went **94.0% → 6.3%** on the same clips.
`tests/test_corpus.py` locks both down.

## What this does and does not show

**Shows.** The architecture works as designed. A frozen FastConformer
encoder plus a 2.1M-parameter linear bridge plus a rank-32 LoRA is
enough to turn speech into text, on a laptop, without a GPU. The bridge
is doing the work: the errors are phonetic.

**Does not show.** Anything about ASR quality. 19% WER on 24 synthetic
sentences with a 420-sentence training set is a wiring result. Stage 1
is LibriSpeech train-clean-100, 5,000 steps on a rented H100, and the
target is under 8% WER on dev-clean.

**Known weaknesses in this result.**

- The corpus is 49 minutes of macOS TTS. It has no noise, no accent
  variation beyond the five voices, no overlap, and every sentence is
  grammatical and short.
- `eval_seen` at 0% WER is memorisation, not a generalisation claim. It
  is reported because it separates "cannot read this voice" from
  "cannot know this sentence", not because it is impressive.
- The training signal saturates at ~6k steps and then only memorises.
  More steps on the same data would have made the number worse, not
  better; the run was stopped at 12k when the gradient norm hit zero.
- The long-audio path concatenates overlapping windows rather than
  merging them, so words in the overlap are counted twice. Every clip
  here is under one window, so it never applied.

## Next

1. Stage 1 on LibriSpeech with an H100, and a real NeMo config. The
   corpus here is a smoke test, not a training set.
2. Decide the LoRA target list before spending the GPU hours. See
   `docs/LORA_TARGETS.md`: the current list adapts the attention path in
   6 of 24 layers and nothing else would.
3. Checkpoint selection by held-out WER rather than by loss. This stage
   showed the two disagree completely after ~6k steps.