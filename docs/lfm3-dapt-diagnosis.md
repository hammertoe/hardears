# Why text-only DAPT did not work on LFM3-Audio-2B

**Date:** 28 October 2026. **Model:** `LiquidAI/LFM3-Audio-2B` (2B, BF16).
**Jobs:** 4 HF Jobs on `a10g-small` (24 GB, $1.00/hr), total ≈ $0.40.

This is the diagnosis that followed the pilot in `lfm3-audio-text-dapt.md`.
It answers *why* the pilot's text-only DAPT failed and *what to try instead*.

## The headline

Text-only DAPT did not fail because DAPT does not work on this model. It
failed because **the corpus does not contain the facts**, the **run was 6% of
one epoch**, the **metric could not resolve the effect**, and **nothing tied
the training signal to the task shape**. All four are separately fixable; only
the first one is a data problem.

## The four causes

### A. The corpus does not contain the facts (dominant)

Measured against the actual corpus (`bimomni` newspaper JSONL + Wikipedia
JSONL, 5,033,116 words):

| Measurement | Value |
|---|---|
| local / rare_local probes | 40 |
| Corpus sentences topically related to a probe's question | 15,733 |
| …of which contain the answer | **297** |
| Probes with **zero** answer-stating sentences | **16 of 40** |
| Probes answerable from the corpus at all | **14 of 40** |

Concrete examples of the gap:

- *"In which year did Barbados' legislature first convene?"* — `1639` appears
  twice in 5M words: once in a museum blurb ("parliamentary history, which
  spans from 1639 to the present") and once inside an invoice number
  (`0163907`). Never as a statement of the fact.
- *"When was the Barbados Workers' Union formed?"* — `1941` appears 8 times;
  exactly one sentence states it ("the BWU made in legislation since its start
  in 1941"). The other seven are a Trinidadian obituary, a jobs statistic, a
  land conveyance, and `USSR in 1941`.
- *"What annual rent did Pickwick pay…"* — "one penny" appears twice, never as
  the Kensington Oval lease.

DAPT cannot teach facts that are absent from the data. The loss fell anyway
because the model learned the **register** of the corpus — headlines, ledger
items, legal notices — not its content.

### B. The run was far too short to matter

| Quantity | Value |
|---|---|
| Steps / optimizer updates | 200 / 50 (`grad_accum=4`) |
| Tokens trained | 0.41 M |
| Corpus size | 6.66 M tokens (3,253 packed rows of 2,048) |
| **Fraction of one epoch** | **6.15%** |
| Records actually touched | ~369 of 4,532 |

Loss 3.42 → 3.11 over 50 updates is a model learning surface form, not facts.

### C. The metric could not resolve the effect

The knowledge probe scores each choice by NLL and takes the argmin over 4
candidates. At 40% base accuracy (against a 25% chance floor) the argmin on
60 items is highly sensitive to a logit perturbation — **5 flipped items is
the entire −5 point delta** that the pilot reported.

The tell is in the pilot's own track breakdown:

| Track | Base | Post |
|---|---|---|
| local | 0.30 | 0.15 |
| rare_local | 0.15 | 0.10 |
| **control** | **0.75** | **0.80** |

Catastrophic forgetting degrades control too. Control *rose*, so the score
moved rather than the knowledge. Confirmed independently in the fact-dense
runs: the two scoring variants disagree by **35 accuracy points on the same
model** (base control `acc_sum` 0.40 vs `acc_mean` 0.75).

**Any future result needs continuous margins and a confidence interval, not
argmin accuracy on n=60.**

### D. Nothing tied the signal to the task shape

Next-token prediction on prose presents a fact as *context*. The probe asks an
*interrogative* and requires *discriminating* 1639 from 1651/1627/1666.
DAPT never teaches retrieval-under-a-question. Even for the facts the corpus
*does* state (uwi-campus: 204 answer-bearing sentences; grantley-school: 13),
plain LM loss never asks the model to produce the answer against distractors.

## What was tried next: fact-dense, split-controlled

To separate "corpus lacks facts" from "format mismatch", I built a corpus
that *does* state the facts, from the 14 recoverable ones: **51 sentences,
14 facts** → 38 training rows (1,864 tokens raw / 1,045 tokens QA).

Split **by fact**, not by probe row: 8 train / 3 dev / 3 test facts. Eval on
held-out facts + 20 control probes, scored by sum-NLL, mean-NLL and margin.

| Mode | Held-out local acc | Held-out local mean margin | Control acc_sum / acc_mean | Loss | Saturated |
|---|---|---|---|---|---|
| **raw** (fact sentences) | 0.167 → **0.333** | −5.02 → **−2.11** | 0.400 → 0.500 / 0.750 → 0.650 | 3.46 → 0.00003 | step 32 |
| **qa** (question→answer) | 0.167 → 0.167 | −5.02 → **−3.96** | 0.400 → 0.450 / 0.750 → 0.700 | 4.61 → 0.00002 | step 62 |

Honest reading:

- Both modes moved the **margin** in the right direction on held-out facts.
  Margins are the continuous signal; accuracy is too noisy at n=6.
- Raw mode flipped one held-out probe to correct; QA mode flipped none.
- The corpus was 51 sentences. The model memorised it by step 32 (raw) /
  62 (qa) and **368 of 400 steps were pure noise on a saturated loss**.
- These are **proof-of-concept runs, not results**. The sample is far too
  small to claim anything.

What they do show: **the pipeline works, and the failure was data-shaped,
not method-shaped.** Fact-dense training moves margins in the right
direction. The pilot's problem was that its corpus was 5M words of prose
containing almost none of the target facts.

## What to try instead, in priority order

### 1. Fix the corpus first — nothing else matters until this is done

The probe set is the specification. Build a corpus that **states each fact**,
not one that mentions its topic.

- **Fact inventory.** For each of the 60 probes, write the target fact as a
  canonical sentence and check it against the corpus. 26 of 40 local facts
  currently have *no* support at all.
- **Author the missing facts** (26 facts) from authoritative sources
  (Barbados Parliament, National Library, official gazettes) with attribution.
  This is the single highest-value action; it is curation, not compute.
- **Extract, don't hope.** Where the corpus does state a fact, extract the
  exact sentence programmatically (the `facts_from_corpus` function in
  `scripts/lfm3_dapt/text_exp.py` does this) rather than training on 5M
  tokens of prose and hoping the gradient finds it.
- **Augment with distractors.** To fix cause D, emit each fact as
  `question → correct answer` *and* `question → wrong answer` (the other
  three choices), so the model must discriminate, not just recall.
- Aim for **≥ 50 independent facts** before believing any accuracy delta.

### 2. Fix the measurement before trusting any number

- **Score by margin, not argmin.** Report `min(distractor NLL) − gold NLL`
  per item, then aggregate with a bootstrap CI. The pilot's 5-point swing is
  well inside noise.
- **Report both sum-NLL and mean-NLL.** They disagreed by 35 points; pick
  one, justify it, and report the other as a stability check.
- **Split by fact family**, never by probe row. The `canonical` and
  `paraphrase` variants of the same fact must not straddle the split.
- **Select on dev, report on test.** Keep the 20 control probes for
  forgetting detection, but treat them as a guardrail, not a target.
- **Use more control probes** — n=20 is too small to detect a 5-point shift.
- **Report the base rate.** At 25% chance, 40% base accuracy means the model
  is close to guessing on the local track; there is very little signal to
  improve, which is itself a finding worth stating.

### 3. Fix the training recipe

- **Enough steps to matter.** The pilot's 50 optimizer updates were far too
  few. Budget multiple epochs over a *curated* corpus, and stop on dev margin
  rather than a fixed step count.
- **Do not train past saturation.** The fact-dense runs wasted 90% of their
  steps on a loss of 0.0000. Use an early-stopping criterion and record the
  saturation step — it is the honest measure of how small the corpus is.
- **Lower LR, more steps.** 2e-4 on a 1,864-token corpus is a sledgehammer.
  For a curated corpus, 5e-5 with more steps and dev-based selection is
  likelier to generalise.
- **Watch the style-shift confound.** If control moves, the adapter is
  changing the output distribution rather than knowledge. Test at a lower
  rank (r=8) and compare.

### 4. Consider that the task may not be DAPT's job

The probe set measures *retrieval under a question*. A 2B model pretrained on
22T tokens already has the general capability. What is missing is
*Barbados-specific* knowledge, and the corpus is what carries that.

Two framings worth testing, both cheap:

- **Prompted retrieval (RAG) instead of weights.** Put the curated fact
  sentences in the context window and measure the model *with* them. If that
  moves the probe score a lot, the knowledge is present but not parametric,
  and DAPT is the wrong tool — retrieval is cheaper, reversible, and
  updatable. This is the single most informative next experiment.
- **A stronger base.** The model card itself says LFM3-Audio-2B "is not
  recommended for agentic coding or complex multi-step reasoning due to its
  small size". A 2B model sitting at 40% on 4-way MCQ may simply have too
  little capacity for a 30-fact knowledge benchmark. Testing the same
  fact-dense SFT on a 7B+ model would separate "method" from "capacity".

### 5. Do not throw away the audio result

The pilot's finding that text-only DAPT left audio **byte-identical** (0/3
clips changed) with the audio encoder provably unchanged (max param diff
0.0) is a genuine, reusable result: **on this architecture, text updates do
not break audio.** That holds regardless of whether the text side improves,
and it is the main thing the pilot established.

## Files

| Path | Contents |
|---|---|
| `scripts/lfm3_dapt/pilot.py` | Text-only DAPT pilot (baseline → train → eval) |
| `scripts/lfm3_dapt/text_exp.py` | Fact-dense, split-controlled text experiment (raw/qa modes) |
| `scripts/lfm3_dapt/smoke_test.py` | Memory + gradient-flow smoke test |
| `artifacts/lfm3_dapt/pilot_out/results.json` | Pilot base/post metrics |
| `artifacts/lfm3_dapt/textexp_raw/raw/results.json` | Raw-mode results + per-item margins |
| `artifacts/lfm3_dapt/textexp_qa/qa/results.json` | QA-mode results + per-item margins |
| `artifacts/lfm3_dapt/textexp_*/{raw,qa}/facts.json` | The 14 recoverable facts + their corpus sentences |

Adapters: `hammertoe/lfm3-audio-barbados-dapt-pilot` (pilot),
`hammertoe/lfm3-audio-barbados-audioqa` (audio QA, not text).
