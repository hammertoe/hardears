# LoRA target coverage on Qwen3.5

> Written while doing Stage 0 on the Mac (2026-10-17). This file is
> new; nothing in ARCHITECTURE.md, TRAINING.md or DATA.md was changed.

`docs/ARCHITECTURE.md` and `src/bim_asr/model.py` both say the LoRA
target list

```
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

is "the broadest correct target set we can use without peft having
explicit support for Gated DeltaNet". That is right as far as it goes,
but the design docs get the coverage arithmetic wrong, and the gap
matters before Stage 1 spends GPU hours on it.

## What the numbers actually are

Qwen3.5 is a hybrid: `full_attention_interval = 4`, so every 4th layer
is a Gated Attention layer and the other 3 are Gated DeltaNet
(linear attention) layers. The build brief describes this as 8 Gated
DeltaNet per 1 Gated Attention, i.e. 8:1. It is 3:1.

| model | layers | full attention | linear attention |
| ----- | ------ | -------------- | ---------------- |
| Qwen3.5-2B | 24 | 6 | 18 |
| Qwen3.5-4B | 32 | 8 | 24 |

The Gated DeltaNet layers do not have `q_proj`/`k_proj`/`v_proj`/
`o_proj` at all. Their attention is a fused path:

```
linear_attn.in_proj_qkv      fused q/k/v, one Linear
linear_attn.in_proj_z
linear_attn.in_proj_a
linear_attn.in_proj_b
linear_attn.out_proj
linear_attn.conv1d
```

Measured on the 2B by instantiating the model on the meta device and
matching module names against the target list:

| layer type | count | targets that exist |
| ---------- | ----- | ------------------- |
| `full_attention` | 6 | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| `linear_attention` | 18 | gate_proj, up_proj, down_proj |

So the FFN targets reach all 24 layers, and the attention targets
reach 6 of 24 (25%). The other 18 layers get LoRA on the MLP path
only, and their entire attention pathway -- which is where the audio
signal has to be interpreted -- is frozen.

## Does it matter?

Stage 0 says the pattern is "the projection does the heavy lifting",
and nothing observed so far contradicts that: the bridge is where the
audio-to-text-space mapping lives, and Stage 0 on the Mac trains it
to the point of reproducing short English sentences. The LoRA list is
not what is broken.

It is still worth deciding explicitly before Stage 1, because Stage 1
is 5 000 steps on 100 hours of LibriSpeech and the difference between
"MLP-only adaptation in 18 layers" and "full adaptation" is the kind of
thing that shows up as 2-3 % WER.

Options, in the order I would try them:

1. **Leave it.** Matches the Canary-Qwen-2.5B recipe that the project
   is copying, keeps the training config in docs/TRAINING.md honest,
   and is the baseline every later change should be measured against.
2. **Add `in_proj_qkv` and `out_proj` to the target list.** Both are
   plain `nn.Linear`, so peft handles them with no special support.
   This covers the attention path in all 24 layers at the cost of a
   larger adapter. Note `out_proj` is also the name of the *projection
   block's* output in some architectures -- check the matched module
   paths, not just the count, before enabling it.
3. **Add an encoder LoRA** (already the optional Stage 4 sub-stage in
   docs/TRAINING.md). Arguably higher value than LLM-side LoRA for
   absorbing Caribbean phonetics, since the FastConformer was trained
   on far less accented English than the LLM's text corpus implies.

Whatever is chosen, it belongs in `configs/hardears_v0.yaml` (issue
task 2) as an explicit, commented list, so the Stage 1 run is
reproducible from the config alone.

## Reproducing the measurement

```python
import collections, sys, torch
sys.path.insert(0, "src")
from transformers import AutoConfig, Qwen3_5ForCausalLM
from bim_asr.model import BimAsrConfig

text_cfg = AutoConfig.from_pretrained("artifacts/qwen3.5-2b").text_config
with torch.device("meta"):
    model = Qwen3_5ForCausalLM(text_cfg)          # meta: no memory used

targets = list(BimAsrConfig().lora_target_modules)
per_type = collections.Counter()
for i, layer in enumerate(model.model.layers):
    names = {n.split(".")[-1] for n, _ in layer.named_modules()}
    matched = tuple(t for t in targets if t in names)
    per_type[(text_cfg.layer_types[i], matched)] += 1

for (kind, matched), count in sorted(per_type.items()):
    print(f"{kind:17s} x{count:2d}  {list(matched)}")
```