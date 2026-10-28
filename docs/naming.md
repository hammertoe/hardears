# Naming

The working name is **HardEars**. This document captures why, what was considered, and the cultural context.

## What "hard ears" means in Bajan

"Hard ears" is a well-established Bajan expression meaning stubborn, disobedient, or unwilling to listen. It is especially associated with children who have been warned repeatedly but refuse to take advice.

The CBC Barbados proverb captures it:

> "Hard ears yuh won't hear, own way yuh gine feel."

Translation: if you refuse to listen to good advice, you will learn through the consequences instead.

This is a documented, searchable Bajan usage. Verified before settling on the name.

## Why it works as a model name

The name works on two layers that reinforce each other:

1. **Culturally:** "hard ears" = won't listen.
2. **Technically:** the model's entire job is to listen very carefully.

The result is deliberate irony. The brand reads almost as a joke at the model's own expense, which gives it personality rather than reading as another polished corporate AI name.

> HardEars — Bajan speech recognition that actually listens.

## Name variants considered

A short brainstorm with another agent explored several candidates. Final ranking:

| Name          | Strengths                                              | Weaknesses                                |
|---------------|--------------------------------------------------------|-------------------------------------------|
| **HardEars**  | Most distinctive. Genuinely culturally rooted. Funny.   | Needs a moment to land for outsiders.     |
| WunnaScribe   | Clearest and safest. Immediately understandable.       | Less character.                           |
| BimScribe     | Strongest connection to the wider BimOmni naming.      | Less memorable on its own.                |
| BimEar        | Clean.                                                 | Comparatively bland.                      |

HardEars was the favourite for character and authenticity. WunnaScribe was the safer fallback.

### HeadEars (rejected)

A near-miss during the brainstorm. "Head ears" was read as a Bajan tease for someone with prominent or sticking-out ears. The agent flagged that this usage could not be verified in searchable sources — exactly the kind of oral Bajan usage that may not exist in indexed text, which is itself an instance of the problem HardEars is trying to solve.

The agent then confirmed the intended spelling was **HardEars**, not HeadEars, which is what made the name work.

## Collision check

Date of check: initial brainstorm, pre-build.

### Hugging Face

Clean. All checked endpoints returned no results or 404:

| Endpoint                                  | Result                |
|-------------------------------------------|-----------------------|
| `huggingface.co/models?search=HardEars`   | 0 models              |
| `huggingface.co/api/models?search=HardEars` | `[]`               |
| `huggingface.co/datasets?search=HardEars` | 0 datasets            |
| `huggingface.co/api/datasets?search=HardEars` | `[]`             |
| `huggingface.co/spaces?search=HardEars`   | "No result found"    |
| `huggingface.co/api/spaces?search=HardEars` | `[]`                |
| `huggingface.co/HardEars` (user/org)      | 404 — available       |
| `huggingface.co/hardears` (user/org)      | 404 — available       |

The Hugging Face namespace is a clean slate. Both the username and the model slug are unclaimed.

### GitHub (adjacent check)

- `github.com/HardEars` — 404
- `github.com/hardears` — 404

Both usernames available.

### Broader web

Worth knowing. Two pre-existing uses of the name exist outside Hugging Face:

1. **"Hardears" (one word)** — An Afro-Caribbean fantasy-adventure graphic novel by Barbadian creators Matthew Clarke and Nigel Lynch, set on "Jouvert Island" (a magical analog of Barbados). Published by Abrams. ISBN 9781419751929. Sold through major book retailers (Goodreads, Google Books, Amazon, Abrams). This is the most direct prior art — same folk expression, same Bajan cultural territory, already in active commerce.
2. **"HARD EARS"** — A song by Jamaican dancehall artist Likkle Addi, with `#HardEars` used as a hashtag. Different medium, different market.

### Verdict

**Proceed.** Neither finding blocks the name:

- The folk expression "hard ears" is widely used in Barbados and predates both uses. Neither is a coined trademark.
- The graphic novel is in a different industry (publishing vs. ML) and a different format. Confusion risk is low for an open-source ASR model on Hugging Face.
- The HF namespace is genuinely clean.

**Caveats to record:**

- Brand discoverability is slightly reduced — anyone searching "Hardears" will hit the graphic novel first. SEO-wise, we will need disambiguation.
- If the project ever moves into consumer-facing products or merch, a formal trademark search becomes worth doing.
- A model card and repo README should explain the name origin up front, to distinguish us from the novel.

### Status

Hugging Face collision check: **done.** Broader trademark search: **light only** — adequate for an open-source hackathon project, not for a consumer product.

## Hugging Face naming

Working name for the model release:

> `hammertoe/HardEars-0.6B`

The `-0.6B` size slug aligns with the primary base-model candidate `nvidia/parakeet-tdt-0.6b-v3`. If the final base model is a different size, this will be updated.

## Note on "Nemotron speech"

The earlier brainstorm and several docs referred to "NVIDIA's Nemotron speech model" without a specific checkpoint. To be precise, and now that we have looked at the actual pipeline:

- The model that genuinely fits that description and which the parliamentary corpus is already aligned to is **`nvidia/nemotron-speech-streaming-en-0.6b`** — referenced by name in the `bimomni` parl pipeline (`bimomni.parl.recipe.nemotron_model`), CC-BY-4.0, streaming.
- `nvidia/parakeet-tdt-0.6b-v3` is also NVIDIA but is used in the corpus pipeline for **boundary/word-timestamp detection**, not as the transcription model. It remains a viable fine-tune target if Nemotron-speech proves problematic.
- A newer Nemotron-branded ASR — `nvidia/nemotron-3.5-asr-streaming-0.6b` (May 2026, multilingual) — exists but the existing corpus was not built around it. Held as out of scope for v1.

**Locked primary base model:** `nvidia/nemotron-speech-streaming-en-0.6b`. The HF slug `hammertoe/HardEars-0.6B` reflects the parameter count and stays valid.
