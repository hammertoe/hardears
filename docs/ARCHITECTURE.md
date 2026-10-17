# Architecture

This document explains why Bim-ASR looks the way it does, and how each
piece relates to the prior art.

> **Corrected 17 October 2026, after Stage 0.** Four numbers in this
> document did not match the actual checkpoints, and each was wrong by
> enough to break training: the encoder hidden size (512 → **1024**), the
> width of Qwen3.5-2B (2560 → **2048**), the Qwen3.5 hybrid layer ratio
> (8:1 → **3:1**), and the LoRA target list (attention only → attention
> **and** FFN). The encoder also turns out to be loadable without the NeMo
> toolkit, which changes the laptop story. Every fix is marked
> **[corrected]** below with its evidence. Measured outputs are in
> [STAGE0_RESULTS.md](STAGE0_RESULTS.md) and
> [LORA_TARGETS.md](LORA_TARGETS.md).

## TL;DR

Bim-ASR is a **Speech-Augmented Language Model** in the same family as
NVIDIA's Canary-Qwen-2.5B. A FastConformer acoustic encoder turns audio
into a sequence of embedding frames. A small **linear projection** maps
those frames into the embedding space of a dense LLM (Qwen3.5-4B, or
Qwen3.5-2B on the laptop). A **LoRA adapter** on the LLM lets the
language model speak with Barbadian vocabulary and style without
retraining the whole base. The acoustic encoder and LLM base stay frozen;
only the projection and LoRA adapters are trained.

    16 kHz audio
         |
         v
    +-----------------+         +-------+         +-----------------+
    | FastConformer   |  -->    |  Lin  |  -->    | Qwen3.5 + LoRA |
    | (frozen)        | frames  | proj  | tokens  | (frozen + LoRA) |
    +-----------------+         +-------+         +-----------------+
    1024-dim, 12.5 Hz                              2048-dim (2B)
                                                          |
                                                          v
                                                   transcript text

Measured sizes: encoder 608 879 616 params, bridge 2.10 M params for the
2B / 2.62 M for the 4B, LoRA 23.9 M params on the 2B. That is 26.0 M
trainable out of 1.91 B, and it transcribes held-out speech at 19.0% WER
on a toy corpus ([results](STAGE0_RESULTS.md)).

## Why this pattern

Three reference designs cover the space:

1. **Pure encoder-decoder ASR (Parakeet TDT, Canary).** Fast and
   accurate for general English, but the language prior is whatever
   lived in the decoder at pretrain time. Adding Barbadian vocabulary
   means re-training the decoder head, which is expensive and easy to
   over-fit.

2. **End-to-end audio LLM (LFM2-Audio-1.5B, Qwen2.5-Omni).** A single
   model is trained from scratch to consume audio and emit text. The
   strong language prior is built in, but the model is large and the
   pretraining cost is high. BimOmni-30B-A3B is in this family and was
   the lesson: it works, but it is too big and too slow.

3. **Speech-augmented LLM (SALM, used by Canary-Qwen-2.5B).** A small
   projection layer feeds audio embeddings into an LLM, and a LoRA
   adapter teaches the LLM to use them. The base LLM keeps its full
   language ability, the projection is cheap, and the LoRA is
   parameter-efficient. NVIDIA's Canary-Qwen-2.5B sits at the top of
   the Open ASR Leaderboard using exactly this recipe, and it is small
   enough to run on one or two GPUs.

Bim-ASR uses pattern 3. It is the same approach as
[nvidia/canary-qwen-2.5b](https://huggingface.co/nvidia/canary-qwen-2.5b),
swapped from a 1.7B Qwen3 base to a Qwen3.5-4B base **[corrected]** (not
a "Qwen3-Instruct", which is a different model family) so that the
Barbados-aware training (DAPT plus LoRA) can lean on a stronger,
already-instruction-tuned language model. Qwen3.5-2B is the laptop
stand-in for that base; see section 2.

## Pieces

### 1. Acoustic encoder: FastConformer

We use the **FastConformer encoder** from NVIDIA's Parakeet TDT 0.6B v2
([model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)). The
choice between Parakeet TDT 0.6B v2 (English-only) and Canary 1B Flash
(4 languages) is a tradeoff:

- **Parakeet TDT 0.6B v2** — slightly faster, 80 ms frame rate, English
  only. Source weights are CC-BY-4.0.
- **Canary 1B Flash** — 883 M params, English/German/French/Spanish,
  encoder is FastConformer. Slightly heavier but the encoder weights are
  pretrained on more languages.

For an English-anchored Caribbean project, **Parakeet TDT 0.6B v2** is
the better starting point. We only need the **encoder** (not the TDT
decoder), so we discard the decoder and treat the encoder as a
frozen feature extractor.

**[corrected] Hidden size is 1024, not 512.** The earlier version of this
document said 512 and attributed it to a "FastConformer-XL config".
Neither is right. Reading `model_config.yaml` out of
`parakeet-tdt-0.6b-v2.nemo` gives the encoder spec:

| field | value |
| ----- | -----: |
| `d_model` (output width) | **1024** |
| `n_layers` | 24 |
| `n_heads` | 8 |
| `ff_expansion_factor` → FFN width | 4 × 1024 = 4096 |
| `feat_in` (log-mel bins) | 128 |
| `subsampling_factor` | 8 |
| `conv_kernel_size` | 9 |
| encoder parameters | 608 879 616 |

`nvidia/parakeet-tdt-0.6b-v3` has the same width, so this is not a v2
quirk. Output is **12.5 frames per second** (80 ms per frame: 16 kHz
audio, 160-sample hop at 100 fps, then 8x subsampling), each frame
1024-dim. `artifacts/parakeet-encoder-v2/bim_asr_encoder.json` records
these numbers next to the extracted weights, and
`src/bim_asr/parakeet.py` reads them from there rather than hard-coding
them.

The 512 figure came from the docs, and a bridge sized for it would have
produced a shape mismatch against every real frame the encoder emits.

The encoder is frozen for the main Bim-ASR run. Optional stage (see
TRAINING.md) does a small LoRA on the encoder to absorb Barbadian
phonetics.

### 2. Language model: Qwen3.5-4B (Qwen3.5-2B on the laptop)

**[corrected]** This section used to describe
`Qwen3-4B-Instruct-2507` (36 layers, hidden 2560, grouped-query
attention). The project uses **Qwen3.5**, which is a different
architecture, and the two sizes are not interchangeable. Facts read
from `config.json`:

| | Qwen3.5-4B | Qwen3.5-2B |
| --- | ---: | ---: |
| hidden size | 2560 | **2048** |
| layers | 32 | 24 |
| attention heads / KV heads | 16 / 4 | 8 / 2 |
| head dim | 256 | 256 |
| FFN width | 9216 | 6144 |
| vocabulary | 248 320 | 248 320 |
| context | 262 144 | 262 144 |
| hybrid layout | 3 linear-attention layers per 1 full-attention layer | same |

Both are Apache-2.0. Both repos are multimodal
(`Qwen3_5ForConditionalGeneration`, with a 24-layer vision tower), but
`AutoModelForCausalLM` resolves to `Qwen3_5ForCausalLM` and pulls in
**zero** vision weights — measured at 1.88 B parameters for the 2B with
no `visual.*` tensors. Nothing needs dropping.

**[corrected] The hybrid ratio is 3:1, not 8:1.** `full_attention_interval`
is 4, so three Gated DeltaNet (linear attention) layers are followed by
one Gated Attention layer, repeated. The earlier claim of 8 Gated
DeltaNet per 1 Gated Attention was wrong, and it materially understated
the LoRA coverage problem in section 3.

The **2B is the laptop model**, not just a smaller option: it is what
Stage 0 trained and measured, because 1.88 B params in bf16 fits
comfortably in unified memory and runs at 4.6x real time on an M4 Pro.
The **4B is the Stage 1 training target**. Note the two are different
widths, so the bridge is a different shape for each — see below.

Why not the 30B-A3B MoE we used for BimOmni? Because BimOmni was
already too big and too slow, and an MoE forces routing decisions that
make a small ASR head feel out of place. A dense 4B fits comfortably in
16 GB of unified memory after 4-bit quantisation (about 2.5 GB), and
runs on Metal with mlx-lm.

Why not Nemotron-H-4B? Nemotron's hybrid Mamba-2/attention design is
attractive, but it is licensed under NVIDIA's internal R&D licence,
which rules out open redistribution. Qwen3.5 is Apache 2.0.

Why not Qwen2.5-Omni-3B? That model already has a built-in audio
tower. The whole point of Bim-ASR is to test the projection + LoRA
recipe at a smaller scale than the leaderboard SOTA. Using a fresh
small dense base also keeps the comparison clean.

### 3. Bridge: linear projection + LoRA

The bridge has two parts.

**Linear projection.** The FastConformer outputs 1024-dim frames at
12.5 Hz. **[corrected]** The LLM embedding space is 2048-dim for the 2B
and 2560-dim for the 4B — not 2560 for both. A single
`nn.Linear(1024, llm_hidden, bias=False)` maps each frame to a
token-embedding-shaped vector, preceded by a `LayerNorm` on the input.
After the projection, the audio frames replace the `<|audio|>` locator
in the tokenised prompt and are fed to the LLM exactly the way SALM
does it.

| LLM | projection | trainable params |
| --- | --- | ---: |
| Qwen3.5-2B | `Linear(1024, 2048)` | 2 099 200 |
| Qwen3.5-4B | `Linear(1024, 2560)` | 2 623 488 |

The projection is **trained from scratch** (no warm-up), initialised at
std 0.02 so the bridge starts near zero rather than swamping the LLM's
first layer. It is small and converges quickly: Stage 0 took cross-entropy
loss from 5.11 to ~0.01 in 6 000 steps on 49 minutes of audio.

`BimAsrConfig.for_llm()` reads `hidden_size` out of the model's own
config rather than trusting a table, because the 2B/4B mismatch above is
exactly the kind of thing a hard-coded constant gets wrong. Getting it
wrong is not a silent degradation, it is a shape error at the splice.

**LoRA adapter on the LLM.** Rank 32, alpha 64, dropout 0.05, applied to
seven projections:

```
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

**[corrected]** The earlier version of this section said the LoRA covered
"the attention projection layers (q_proj, k_proj, v_proj, o_proj)" and
that "the MLP and embedding layers stay frozen". The FFN projections are
in fact targeted, which is what gives the adapter coverage of every
layer's MLP path. On Qwen3.5-2B the adapter is **23.9 M trainable
parameters**, not the ~27 M quoted here before.

The catch, measured and written up in
[LORA_TARGETS.md](LORA_TARGETS.md): `q/k/v/o_proj` **do not exist in
Qwen3.5's 18 linear-attention layers**. Those layers express attention
through a fused `linear_attn.in_proj_qkv` and `out_proj` instead. So the
current target list adapts the FFN path in all 24 layers but the
attention path in only the 6 full-attention layers. Whether to add
`in_proj_qkv`/`out_proj` is an open decision, and it is better made
before Stage 1 spends 5 000 H100 steps than after. The embedding table
is genuinely frozen and must stay that way — Qwen3.5 ties the input and
output embeddings.

The LoRA is what gives the model its Caribbean accent. The same Qwen3.5
base, with a different LoRA, could be turned into a code completion
model, a customer-service agent, or anything else. That is the point of
using LoRA instead of full fine-tuning.

### 4. Training framework

Stages 1-5 train in **NVIDIA NeMo** (`nemo.collections.speechlm2`)
because NVIDIA has already open-sourced the SALM training script and
base config, and because NeMo handles audio preprocessing, FSDP, and
gradient accumulation out of the box.

The relevant files are:
- Training script: `examples/speechlm2/salm_train.py`
- Base config: `examples/speechlm2/conf/salm.yaml`

We adapt those to:
- Use `nvidia/parakeet-tdt-0.6b-v2` for the encoder (only)
- Use `Qwen/Qwen3.5-4B` for the LLM **[corrected]**, which is hybrid
  rather than plain decoder-only, so the LoRA target list has to be
  checked against the real module names rather than assumed
- Use a fresh projection of shape **(1024, 2560)** **[corrected]**
- Freeze the encoder and most of the LLM, train projection + LoRA
- Add Caribbean data after the initial English run

**[corrected] The laptop path does not use NeMo at all.** NeMo has no
supported macOS or Apple-Silicon build, so Stage 0 could not run here at
all under the original plan. Two consequences:

1. `scripts/prepare_parakeet_encoder.py` extracts the FastConformer
   encoder out of the `.nemo` tar into `transformers` format, using
   NVIDIA's own NeMo → HF weight mapping and a strict load that fails on
   a single unmapped tensor. `src/bim_asr/parakeet.py` then exposes it
   with the same call signature the NeMo encoder had.
2. Stage 0 training runs in plain PyTorch on MPS
   (`scripts/train_stage0.py`), and Stage 0 inference runs in
   `src/bim_asr/inference.py`.

This is not a workaround with worse fidelity — the weights are identical
and the conversion is verified numerically by decoding real audio with
the original TDT decoder (2.1% WER, see [STAGE0_RESULTS.md](STAGE0_RESULTS.md)).
It does mean Stage 0 is *not* running the Stage 1 training code, so a
Stage 0 pass does not prove the NeMo config works. That is the first
thing Stage 1 has to establish.

For laptop-side deployment, we mirror the architecture in
**MLX + mlx-lm** so the model can be loaded and run on the M4 unified
memory. The extracted encoder is safetensors, which is what
`mlx_lm.convert` consumes. That path is still unbuilt: `mlx_model.py`
mirrors the projection only. The laptop path is **inference only**; full
training lives on a rented H100 or H200.

## Why a SALM, not a full audio LLM

BimOmni used an existing audio LLM (Qwen3-Omni-30B-A3B) and adapted it
with LoRA on Barbados newspaper text. That gave us a 13.3-point lift on
Barbados proper nouns in the model card's evaluation, and the
visual/audio towers were useful for follow-on multimodal work. But the
model was 30B parameters of MoE, ran at 25-30 seconds of wall time for
30 seconds of audio on a Mac, and did not actually learn Barbadian
phonetics because the adaptation was text-only.

The hypothesis for Bim-ASR is:

1. **Most of the lift BimOmni got was from local language priors, not
   from the audio tower.** A 4B Qwen3.5 model with a strong LoRA
   should get most of the same benefit at a fraction of the size.
2. **A small projection layer + small LoRA is a much faster iteration
   loop.** We can run new Caribbean training rounds in hours, not
   days.
3. **Inference stays under real time on a laptop.** This matters for
   Pulse-style ingestion of broadcast audio, where a 30 s per 30 s
   pipeline blocks the next step.

## Open questions we will answer in training

- Does the Qwen3.5-4B base really beat a 1.7B base on Caribbean
  proper-noun WER? (Compare against Canary-Qwen-2.5B at 1.7B.)
- **[corrected] Should the LoRA target list cover Qwen3.5's
  linear-attention layers?** Stage 0 measured that the current list
  adapts the attention path in 6 of 24 layers. See
  [LORA_TARGETS.md](LORA_TARGETS.md). Cheap to answer on Stage 1 data,
  expensive to change after Stage 1 has run.
- How much does adding Common Voice Caribbean speakers help versus
  a clean English baseline?
- Is a 30-40 second audio window enough for Parliament-style speech,
  or do we need to extend it? Canary-Qwen-2.5B was trained at 40 s.
- Can the projection + LoRA absorb a real Barbadian accent (vowel
  shifts, dropped final consonants) with a small fine-tune on paired
  Parliament audio?
