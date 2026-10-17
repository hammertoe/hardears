# Bim-ASR

A small, fast, Caribbean-aware speech recognition stack. It fuses a
frozen FastConformer acoustic encoder with a frozen Qwen3.5 language
model through a small linear projection plus a LoRA adapter. Only the
projection and the LoRA are trained.

This is the second-generation follow-up to
[`hammertoe/BimOmni-30B-A3B`](https://huggingface.co/hammertoe/BimOmni-30B-A3B).
BimOmni handled Barbadian proper nouns much better than its base, but
it was a 30B-A3B MoE that was too big and too slow for everyday use.
Bim-ASR keeps the local-aware behaviour while fitting on a laptop.

## Status: Stage 0 complete

Stage 0 in [`docs/TRAINING.md`](docs/TRAINING.md) is "prove the SALM
scaffolding works end to end on the laptop". That gate now passes with
the **real** encoder and the **real** LLM, not stubs:

```
smoke_salm.py --mode real-encoder
  encoder:  (1, 43226) samples -> (1, 34, 1024) frames in 2.81s
  bridge:   -> (1, 34, 2048) (LLM width 2048)
  spliced sequence: (1, 67, 2048) (24 text + 34 audio + 9 text)
  generated 24 tokens in 107.5s
  audio sensitivity: two different clips differ by 229.26 in projected
                     embedding space -> different output
```

| Check | Result |
| ----- | ------ |
| Real Parakeet encoder, extracted from the `.nemo` | 608,879,616 params, loaded with 0 missing / 0 unexpected tensors |
| Encoder correctness | **2.1% WER** on 24 held-out clips when the original TDT head decodes off the extracted features |
| Projection bridge | 2,099,200 params (1024 → 2048), gradients verified through a spliced prompt |
| End-to-end SALM | audio → encoder → bridge → Qwen3.5-2B + LoRA → text, on CPU or MPS |
| Audio actually matters | two different clips produce different transcripts |
| Transcription after Stage 0 training | **19.0% WER** on unseen sentences in unseen voices; **0.0%** on seen sentences in unseen voices |
| Speed | 4.6x real time on MPS, bf16, 1.9B model, one 30 s window |
| Tests | 56 passing (`pytest tests/`) |

```
reference : The light faded slowly behind the hills.
Stage 0   : The lake waited slowly behind the hill.

reference : A woman walked along the quiet shore.
Stage 0   : A woman walked along the quiet shorn.
```

The errors are phonetic, which is what a working acoustic path looks
like. Full numbers, including what this does *not* show, are in
[`docs/STAGE0_RESULTS.md`](docs/STAGE0_RESULTS.md).

Run the whole thing yourself:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]" mlx

# 1. weights (encoder is CC-BY-4.0, LLM is Apache-2.0)
hf download nvidia/parakeet-tdt-0.6b-v2 parakeet-tdt-0.6b-v2.nemo \
    --local-dir artifacts/parakeet-tdt-0.6b-v2
hf download Qwen/Qwen3.5-2B --local-dir artifacts/qwen3.5-2b

# 2. extract the encoder (no NeMo needed)
.venv/bin/python scripts/prepare_parakeet_encoder.py \
    --nemo artifacts/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo \
    --out artifacts/parakeet-encoder-v2

# 3. tests, then the gate
.venv/bin/pytest tests/ -q
.venv/bin/python scripts/smoke_salm.py --mode real-encoder
```

## Corrections to the design docs

The scaffold's numbers did not match the checkpoints. Each was found by
reading the actual files rather than trusting the docs, and each is a
hard blocker for Stage 1. **The docs have now been corrected too** —
every fix below is marked `[corrected]` in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and
[`docs/TRAINING.md`](docs/TRAINING.md), with the evidence.

| Claim in the scaffold | Reality | Evidence |
| --------------------- | ------- | -------- |
| encoder hidden = 512 | **1024** (`d_model` in the checkpoint's `model_config.yaml`; v3 is the same width) | `artifacts/parakeet-encoder-v2/bim_asr_encoder.json` |
| Qwen3.5-2B is 2560 wide, like the 4B | **2048**. The 2B bridge could never splice into the 2B model | `artifacts/qwen3.5-2b/config.json` → `text_config.hidden_size` |
| load the encoder through `nemo.collections.asr` | NeMo has **no supported macOS build**. The encoder comes from `transformers` instead, using NVIDIA's own NeMo → HF weight mapping | `scripts/prepare_parakeet_encoder.py`, `src/bim_asr/parakeet.py` |
| Qwen3.5 is 8 linear-attention layers per 1 attention layer | **3:1** (`full_attention_interval = 4`) | `config.json` → `text_config.full_attention_interval` |
| LoRA covers attention projections only | It also targets the FFN projections; that is what reaches every layer. 23.9M params, not ~27M | `docs/LORA_TARGETS.md` |

Two open questions from the build brief, answered:

- **Vision tower.** `AutoModelForCausalLM` on the Qwen3.5 repos
  resolves to `Qwen3_5ForCausalLM` and pulls in **zero** vision
  weights (1.88B params for the 2B, no `visual.*` tensors). Nothing
  needs dropping.
- **LoRA targets.** q/k/v/o_proj do not exist in Qwen3.5's 18
  linear-attention layers, so the current target list adapts 6 of 24
  layers' attention path and all 24 layers' FFN path. Worth deciding
  before Stage 1 spends GPU hours on it — see
  [`docs/LORA_TARGETS.md`](docs/LORA_TARGETS.md).

## Components

- **Acoustic encoder** — FastConformer from Parakeet TDT 0.6B v2, 24
  layers, `d_model` 1024, 12.5 frames/s (80 ms per frame), 608M
  params. Frozen. Encoder only; the TDT decoder and joint network are
  dropped.
- **Language model** — Qwen3.5-2B for the laptop (2048 hidden, 24
  hybrid Gated DeltaNet / Gated Attention layers) or Qwen3.5-4B for
  training (2560 hidden, 32 layers). Frozen.
- **Bridge** — `LayerNorm(1024) → Linear(1024, 2048, bias=False)`,
  2.10M params, std 0.02 init.
- **LoRA** — rank 32, alpha 64, dropout 0.05, 23.9M trainable params.
- **Training** — NeMo SALM recipe on a rented H100 for stages 1+.
  Stage 0 trains on the laptop in PyTorch/MPS.

Trainable: 26.0M of 1.91B. Everything else stays frozen, exactly as
the training plan says.

## Layout

- `docs/ARCHITECTURE.md` — design rationale, why each piece was chosen
- `docs/TRAINING.md` — staged training plan, data, hyperparameters
- `docs/DATA.md` — dataset sources and licenses
- `docs/LORA_TARGETS.md` — what the LoRA target list does and does not cover
- `docs/STAGE0_RESULTS.md` — measured Stage 0 results, and their limits
- `src/bim_asr/model.py` — config, projection bridge, model, prompt
- `src/bim_asr/parakeet.py` — the frozen encoder, NeMo-compatible call signature
- `src/bim_asr/mlx_model.py` — MLX mirror of the projection
- `src/bim_asr/audio.py`, `src/bim_asr/data.py` — audio loading, NeMo manifests
- `scripts/prepare_parakeet_encoder.py` — extract the encoder from the `.nemo`
- `scripts/make_smoke_corpus.py` — build the Stage 0 speech corpus
- `scripts/smoke_salm.py` — the Stage 0 gate
- `scripts/train_stage0.py` — Stage 0 training on the laptop
- `scripts/transcribe.py` — transcribe a file with the assembled model
- `scripts/eval_baseline_parakeet.py` — reference ASR, for comparison
- `tests/` — 56 tests, none of which need the big checkpoints

## Stage 0 training, honestly

`docs/TRAINING.md` stage 0 wants to see loss fall over a few hundred
steps. `scripts/train_stage0.py` does that with the real encoder and
the real LLM on Apple Silicon: ~1.85 it/s on an M4 Pro, bf16, batch
size 1, ~26M trainable parameters, frozen encoder features cached
once.

The corpus is generated locally by `scripts/make_smoke_corpus.py`
using the offline macOS `say` voices: **420 distinct sentences, 1,260
train clips** (4 voices x 3 speed perturbations), 49 minutes of audio,
with exact transcripts. It exists to test the wiring, not to compete
with LibriSpeech.

The evaluation sets vary two things, on purpose:

| set | text | voices | what it measures |
| --- | ---- | ------ | ----------------- |
| `eval.jsonl` | 48 unseen sentences | Moira, Tessa -- neither in training | content *and* acoustic generalisation |
| `eval_seen.jsonl` | 48 sentences from training | Moira, Tessa | acoustic generalisation alone |

Reporting the second one alongside the first is how you tell "it cannot
cope with words it has not seen" apart from "it cannot cope with a
voice it has not heard".

An untrained bridge does **not** transcribe. It hands the LLM
nonsense embeddings, and the LLM responds like a text-only assistant:

```
audio : 'The light faded slowly behind the hills.'
SALM  : 'I am ready to transcribe the audio. Please provide the audio file...'
```

That is the expected starting point, and the audio-sensitivity check
in `smoke_salm.py` is what makes it meaningful: the text changes when
the audio changes, so the audio really is reaching the model. What
training the bridge produces is in
[`docs/STAGE0_RESULTS.md`](docs/STAGE0_RESULTS.md), which also records
two bugs this stage found — a generation loop that never stopped at EOS,
and a corpus builder that reused stale audio against fresh text. Both
produced metrics that looked real and were not.

## Licensing

The code is Apache-2.0. The encoder weights are **not**: they are
derived from `nvidia/parakeet-tdt-0.6b-v2` and stay CC-BY-4.0. Any
release that ships them must carry that attribution. Qwen3.5 is
Apache-2.0.

## Naming

`Bim` is the local nickname for Barbados.

## Provenance

- Author: Matt Hamilton (`@hammertoe`)
- Previous model: [`hammertoe/BimOmni-30B-A3B`](https://huggingface.co/hammertoe/BimOmni-30B-A3B)
- Reference design: [`nvidia/canary-qwen-2.5b`](https://huggingface.co/nvidia/canary-qwen-2.5b)
- Stack: Hugging Face Transformers, peft, MLX, NVIDIA NeMo (training)