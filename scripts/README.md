# scripts/

## `build_parliament_context.py`

Builds per-video context for the Barbados Parliament corpus from the **public**
Order Papers on `barbadosparliament.com`, so the transcription prompt can be
biased with the correct sitting vocabulary (member names, bill titles, motions,
place names).

### Run

```bash
uv run scripts/build_parliament_context.py            # full run
uv run scripts/build_parliament_context.py --limit 20 # smoke test
```

Dependencies (`requests`, `beautifulsoup4`, `lxml`) are declared inline
(PEP 723), so `uv` resolves them automatically. PDF text extraction uses
poppler's `pdftotext` (`brew install poppler`).

| flag | effect |
|---|---|
| `--dataset DIR` | corpus root (default `hammertoe__barbados-parliament-clips-v1`) |
| `--out DIR` | output root (default `context`) |
| `--limit N` | bundle only the first N matched videos |
| `--include-archive` | also scrape the pre-2015 archive |
| `--no-download` | match only; skip PDF download + extraction |
| `--delay S` | politeness delay between live requests (default 0.3) |

### What it does

1. Pages the typed Order Paper lists (`/order_papers/search/type/{1,2}`, POST with
   `OFF_SET=0,20,40,…`) for House and Senate. Optional archive: the pre-2015
   `/past_order_papers/{house,senate}/view/{id}` pages.
2. Parses each source video's sitting date and chamber from its YouTube title in
   `MANIFEST.jsonl` (`Month day` and `day Month` forms both handled).
3. Matches video → order paper by `(chamber, date)`, falling back to date-only,
   then nearest date within ±3 days.
4. Downloads each matched order-paper PDF (deduped by URL) and extracts its text.

### Outputs (under `context/`)

| path | contents |
|---|---|
| `_order_papers.jsonl` | every scraped order paper (chamber, label, date, attachments) |
| `_videos.jsonl` | parsed source-video metadata |
| `_matches.jsonl` | video → order paper, with match type |
| `{video_id}.json` | per-video bundle: paper, text pointer, extracted glossary |
| `text/{video_id}.txt` | extracted order-paper text |
| `_coverage.md` | coverage report, by match type / chamber / year, plus unmatched list |
| `pdfs/`, `.cache/` | downloaded PDFs and raw HTTP cache (gitignored) |

### Coverage (as built)

**609 / 804 source videos matched (75.7%)** — 534 exact date+chamber.

Remaining gaps, by cause:

- **Committee / Youth / special events** (`committee` 40, `youth` 4, plus diaspora,
  funerals, Model UN): no order paper exists. Not recoverable from this source.
- **Estimates Debates** (most of the 124 unmatched `house` videos): the March
  Estimates sittings do **not** appear in the public order-paper list. Their
  papers live elsewhere (Estimates documents, per-sitting pages) — see below.
- **No parseable date** (17): titles with no date at all.

### Known limitations / next steps

- **Hansard (Debates) is members-only.** `/parliament_debate/search` redirects to
  `/member/login`, even with a session; it is not publicly scrapeable. If an
  account is available, the Debates search (1945→present) would give an official
  — though *edited*, "subject to correction" — reference to verify against.
- **Estimates gap.** Consider sourcing estimates-sitting papers from
  `/sittings/details/{id}` (via `/sittings/listall/1`) or the Estimates documents
  (`/document/listall/3`) to lift House coverage.
- **Glossary is crude.** `build_glossary` uses capitalised n-gram frequency plus
  bill/resolution regexes; it is a starting point, not a curated entity list.
- **`nearest-*d` matches (67)** are worth a manual spot-check — a nearest-date
  paper may belong to an adjacent sitting.

### Verification status

Verified locally: paging both chambers, date parsing (both formats), matching,
PDF download, text extraction, glossary output, and the coverage report. Not yet
verified: the optional `--include-archive` path, and whether the extracted text
measurably improves ASR accuracy downstream.

---

## `transcribe_longform.py`

Layer-2 prototype: transcribe a whole sitting, then slice back to the corpus
clips. See the module docstring for the pipeline. Key ideas:

- **Timeline verification comes first.** Before spending anything on the API, the
  script cuts the downloaded source audio at each clip's `[start_s,end_s]` and
  cross-correlates it against the stored clip. This is what confirms the manifest
  timestamps index the same recording the model will hear.
- **Windowed.** Source videos run up to ~6 h, so audio is transcribed in
  `--window` chunks (default 900 s) to stay inside the 64k output limit.
- **Context-biased.** Each request carries the order-paper text + glossary from
  `build_parliament_context.py`.

```bash
# timeline check only — no API key, no cost (do this first)
uv run scripts/transcribe_longform.py --video-id VG7C5l-BeOg \
    --max-seconds 1200 --verify-only

# the pilot: 4 sittings, ~10.6 h of paired clip data, ~$2
uv run scripts/run_longform_pilot.sh

# a single sitting, flex tier (default), thinking high
uv run scripts/transcribe_longform.py --video-id w9VavzudI1I \
    --window 3600 --thinking high
```

Reads `GEMINI_API_KEY`/`GOOGLE_API_KEY` from the environment, falling back to
`../pulse/.env`. Media downloads land in `media/` (gitignored); per-video
transcripts, segments and summaries land in `longform/{video_id}/`.

### Service tier: use flex, not batch

**Batch does not work with the gateway key in `../pulse/.env`.** Every
`batchGenerateContent` call returns `FAILED_PRECONDITION`, including a trivial
inline text-only request with no audio and no file — so it is the key, not the
request. Use `--tier flex` (the default) instead: same 50% discount, synchronous,
and it works through the gateway.

`_summary.json` records `tier_billed` from `usageMetadata.serviceTier`, so the
discount is verified per run rather than assumed. The script prints a warning if
it was billed at `standard`.

Flex is **sheddable**, so `503`/`429` are routine and it can queue for minutes.
`call_gemini` retries 5 times with exponential backoff (5s → 120s) and uses a
900 s timeout, per the docs' guidance for flex queues.

### Calibration (measured on `VG7C5l-BeOg`, `gemini-3.8-flash`, thinking high)

Three windows were run end-to-end; the third on the flex tier:

| window | tier | prompt tok | transcript tok | **thinking tok** | clips covered |
|---|---|---|---|---|---|
| 200 s   | standard | 8,086  | 1,314 | **19,297** | — |
| 600 s   | standard | 18,086 | 2,837 | **13,971** | 12/14 |
| 600 s   | **flex**  | 18,086 | 3,009 | **13,928** | **14/14** |

What this pins down:

- **Audio bills at 25 tokens/second**, not 32: `(18086 − 8086) / 400 s = 25`.
  (Matches the pricing-page footnotes; the audio-understanding page's "32/s" is
  stale.)
- **Order-paper context + instructions ≈ 3,086 tokens per request** (constant
  across both windows).
- **Transcript output ≈ 5.2 tokens/second** of audio (with timestamps, as JSON).
- **Thinking is per-request, not proportional to audio duration** — it went
  *down* (19.3k → 14.0k) for the 3× longer window. So total thinking cost scales
  with the *number of requests*, and **longer windows are materially cheaper**.
- `media_resolution` is **not** a lever: audio is tokenised at 25 tokens/second
  at every resolution level.

The flex re-run covered **14/14** clips where the standard run covered 12/14. The
two it recovered are the House agenda items (`"Government Notices, Private
Members' Notices, Notices of Questions..."`) — exactly the parliamentary
procedure vocabulary the order-paper context is meant to bias. Treating that as
evidence of quality gain needs more than one window, but it is not nothing.

### Cost, at the corpus scale that is actually planned

**The earlier 605 h projection was the wrong denominator.** 605 h is the sum of
*clip* durations (speech only). Longform transcription has to process whole
sittings, which is **1,745 h** across 804 videos. Speech is 34.7% of sitting
audio, so **~1.93 h of sitting audio is needed per hour of clip data**.

Measured rate: **$0.101 per hour of sitting audio** (flex, thinking high, 1 h
windows).

| scope | sitting audio | clip data | flex cost | standard cost |
|---|---|---|---|---|
| audit pilot | 1.9 h | 1.0 h | $0.20 | $0.39 |
| **`run_longform_pilot.sh`** | **20.5 h** | **10.6 h** | **~$2.07** | ~$4.14 |
| 200 videos | 434 h | 225 h | ~$44 | ~$88 |
| all 804 videos | 1,745 h | 605 h | ~$176 | ~$352 |

**Scope first, then scale.** `../bim-asr/docs/ASR_AUDIT_2026-10-18.md` recommends
1–2 h of transcribed speech to test learnability, expanding to 5–10 h only if the
pilot helps. The full corpus costs ~10× the pilot and buys nothing if the recipe
does not learn — which is what happened to the 16× corpus DAPT run.

Practical window ceiling is ~1 h: at 2 h the transcript alone (~37k tokens) plus
thinking approaches the 64k output limit. **Recommend ~1 h windows.** Thinking is
noisy and content-dependent, so treat the estimates as ±2×.

### Other notes

- Slicing assigns segments by midpoint. A clip can still miss if the model does
  not segment its speech. Add a nearest-segment fallback and a low-confidence
  flag before trusting full coverage.
- The corpus `quality` fields are **not usable as filters**: `utmos` and `sigmos`
  are identically `0.000` across all 219,414 clips (the manifest's `quality_model`
  is `spectral_flatness+snr_cpu_only`, so they were never computed), and `snr_db`
  tops out at 3.13 dB. `docs/next-steps.md`'s `utmos >= 3.2` gate yields **0
  clips** if run as written.
- Audio uploaded for a batch job expires from the Files API after ~48 h, and
  batch jobs themselves expire at 48 h.
