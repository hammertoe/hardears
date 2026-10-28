#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["requests"]
# ///
"""Long-form transcription prototype: transcribe a whole sitting, then slice
back to the corpus clips.

Pipeline
--------
1. Download a source video's audio (yt-dlp) and normalise to 16 kHz mono WAV.
2. **Verify the timeline**: cut the full audio at each clip's ``[start_s,end_s]``
   and cross-correlate against the stored clip. This proves that the manifest
   timestamps index the same recording the model will hear — the assumption the
   whole "transcribe long, slice back" plan rests on. (No API needed.)
3. Window the audio and send each window to Gemini with the order-paper context,
   asking for timestamped segments (structured JSON).
4. Merge segments and slice them to clip spans -> one transcript per clip.
5. Report token usage (input / output / thinking) for cost calibration.

Usage
-----
    # timeline check only (no API key, no cost) — do this first
    uv run scripts/transcribe_longform.py --video-id VG7C5l-BeOg \
        --max-seconds 1200 --verify-only

    # full prototype on the first 30 minutes
    uv run scripts/transcribe_longform.py --video-id VG7C5l-BeOg \
        --max-seconds 1800 --window 900 --thinking high --direct-sample 3
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path

import requests

SR = 16000                # clips are 16 kHz mono
MODEL = "gemini-3.8-flash"
API = "https://generativelanguage.googleapis.com/v1beta/models"
ROOT = "https://generativelanguage.googleapis.com"
UPLOAD = f"{ROOT}/upload/v1beta/files"

INSTRUCTION = """\
You are producing a verbatim transcript of a sitting of the Parliament of \
Barbados. Return ONLY JSON matching the schema.

Rules:
- Transcribe every spoken word exactly as heard. Do NOT paraphrase, summarise, \
clean up, or "correct" grammar.
- Preserve Bajan English and Bajan Creole forms as spoken (e.g. gine, dey, yuh, \
nuh, doh). Do NOT translate them into standard English.
- Timestamps are seconds from the start of THIS audio clip, to millisecond \
precision: start is when the segment begins, end when it ends.
- Split into short segments at natural pauses or speaker changes.
- If a name, place, bill title or institution is listed in the CONTEXT below and \
the speaker clearly says it, spell it the way the CONTEXT does. If the audio is \
unclear or contradicts the context, transcribe what is actually said.
- Ignore non-speech noise; do not invent words to fill silence.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "speaker": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["start", "end", "text"],
            },
        }
    },
    "required": ["segments"],
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def load_api_key() -> str | None:
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    if os.environ.get("GOOGLE_API_KEY"):
        return os.environ["GOOGLE_API_KEY"]
    for envf in (Path("../pulse/.env"), Path(".env")):
        if envf.is_file():
            for line in envf.read_text().splitlines():
                m = re.match(r"\s*(?:export\s+)?(GEMINI_API_KEY|GOOGLE_API_KEY)\s*=\s*(.+?)\s*$", line)
                if m:
                    return m.group(2).strip().strip('"').strip("'")
    return None


def read_wav_mono(path: Path) -> array:
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1, f"{path} is not mono"
        frames = w.readframes(w.getnframes())
    a = array("h")
    a.frombytes(frames)
    return a


def envelope(a: array, hop: int = 160) -> array:      # 10 ms frames -> 100 Hz
    out = array("f")
    for i in range(0, len(a) - hop + 1, hop):
        s = 0
        for v in a[i:i + hop]:
            s += v * v
        out.append(math.sqrt(s / hop))
    return out


def corr(x, y) -> float:
    """Normalised correlation of two equal-ish sequences."""
    n = min(len(x), len(y))
    if n < 20:
        return 0.0
    x, y = x[:n], y[:n]
    sx = sum(v * v for v in x)
    sy = sum(v * v for v in y)
    if sx == 0 or sy == 0:
        return 0.0
    return sum(a * b for a, b in zip(x, y)) / math.sqrt(sx * sy)


def best_env_offset(full_env: array, clip_env: array, guess: int,
                    span: int = 300) -> tuple[float, int]:
    """Best alignment of a clip's energy envelope, searching +/- span frames."""
    best = (-1.0, guess)
    for lag in range(max(0, guess - span), guess + span + 1, 2):
        seg = full_env[lag:lag + len(clip_env)]
        if len(seg) < len(clip_env):
            break
        v = corr(seg, clip_env)
        if v > best[0]:
            best = (v, lag)
    return best


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------

def load_clips(manifest: Path, video_id: str) -> list[dict]:
    clips = []
    with manifest.open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d["video_id"] == video_id:
                clips.append({"clip_id": d["id"], "start": d["start_s"], "end": d["end_s"],
                              "audio_path": d["audio_path"]})
    clips.sort(key=lambda c: c["start"])
    return clips


def download_audio(video_id: str, youtube_url: str, media: Path, max_seconds: int | None) -> Path:
    media.mkdir(parents=True, exist_ok=True)
    src = media / f"{video_id}.src"
    complete = [p for p in media.glob(f"{video_id}.src.*") if not p.name.endswith(".part")]
    if not complete:
        cmd = ["yt-dlp", "-c", "-f", "bestaudio/best", "--no-warnings",
               "-o", str(src) + ".%(ext)s"]
        if max_seconds:
            cmd += ["--download-sections", f"*0-{max_seconds}"]
        cmd.append(youtube_url)
        print(f"  yt-dlp: {' '.join(cmd[-3:])}", flush=True)
        r = run(cmd)
        if r.returncode != 0:
            print(f"  [warn] section download failed, retrying full: {r.stderr[-300:]}", flush=True)
            r = run(["yt-dlp", "-c", "-f", "bestaudio/best", "--no-warnings",
                     "-o", str(src) + ".%(ext)s", youtube_url])
            if r.returncode != 0:
                raise SystemExit(f"yt-dlp failed: {r.stderr[-500:]}")
    raw = next(p for p in media.glob(f"{video_id}.src.*") if not p.name.endswith(".part"))
    wav = media / f"{video_id}.wav"
    if not wav.exists():
        r = run(["ffmpeg", "-y", "-i", str(raw), "-ac", "1", "-ar", str(SR),
                 "-c:a", "pcm_s16le", str(wav)])
        if r.returncode != 0:
            raise SystemExit(f"ffmpeg failed: {r.stderr[-500:]}")
    return wav


def verify_timeline(full_wav: Path, clips: list[dict], dataset: Path,
                    limit_seconds: float, sample: int = 12) -> dict:
    """Confirm clip timestamps index the source audio, via energy-envelope match.

    Envelope (frame-energy) correlation is used rather than raw waveform: it is
    insensitive to the codec/resample differences between the stored clips and a
    fresh download, and it reveals any constant offset.
    """
    full_env = envelope(read_wav_mono(full_wav))
    usable = [c for c in clips if c["end"] <= limit_seconds - 1]
    print(f"  {len(usable)}/{len(clips)} clips fall inside the downloaded window")
    if not usable:
        return {"checked": 0, "strong": 0, "median_offset_ms": None, "rows": []}
    step = max(1, len(usable) // sample)
    rows, ok, offsets = [], 0, []
    for c in usable[::step][:sample]:
        cp = dataset / c["audio_path"]
        if not cp.exists():
            continue
        clip_env = envelope(read_wav_mono(cp))
        guess = int(round(c["start"] * 100))          # 10 ms frames
        val, lag = best_env_offset(full_env, clip_env, guess)
        offset_ms = (lag - guess) * 10
        rows.append({"clip_id": c["clip_id"], "start": round(c["start"], 3),
                     "env_corr": round(val, 3), "offset_ms": offset_ms})
        if val > 0.9:
            ok += 1
            offsets.append(offset_ms)
        print(f"    {c['clip_id']} [{c['start']:.1f}-{c['end']:.1f}s]: "
              f"env_corr={val:.3f} offset={offset_ms:+d}ms", flush=True)
    median = sorted(offsets)[len(offsets) // 2] if offsets else None
    print(f"  timeline: {ok}/{len(rows)} clips env_corr >0.9"
          + (f", median offset {median:+d}ms" if median is not None else ""))
    return {"checked": len(rows), "strong": ok, "median_offset_ms": median, "rows": rows}


def encode_window(wav: Path, start: float, dur: float, out: Path) -> Path:
    r = run(["ffmpeg", "-y", "-ss", str(start), "-t", str(dur), "-i", str(wav),
             "-ac", "1", "-ar", str(SR), "-b:a", "48k", str(out)])
    if r.returncode != 0 or not out.exists():
        raise SystemExit(f"window encode failed: {r.stderr[-300:]}")
    return out


def build_prompt(context: dict | None, budget_chars: int = 12000) -> str:
    if not context:
        return INSTRUCTION
    paper = context.get("order_paper") or {}
    text = ""
    tf = context.get("text_file")
    if tf and (Path("context") / tf).exists():
        text = (Path("context") / tf).read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"[ \t]+", " ", text)[:budget_chars]
    gl = context.get("glossary") or {}
    terms = ", ".join(t["term"] for t in (gl.get("proper_nouns") or [])[:60])
    ctx = [
        "CONTEXT (for spelling reference only):",
        f"Sitting: {context.get('title')}",
        f"Chamber: {context.get('chamber')}   Date: {context.get('sitting_date')}",
        f"Order paper: {paper.get('label')}",
    ]
    if gl.get("parishes"):
        ctx.append("Parishes: " + ", ".join(gl["parishes"]))
    if terms:
        ctx.append("Frequent names/terms: " + terms)
    if text:
        ctx.append("--- begin order paper text ---")
        ctx.append(text)
        ctx.append("--- end order paper text ---")
    return INSTRUCTION + "\n" + "\n".join(ctx)


def call_gemini(key: str, prompt: str, audio: Path, thinking: str,
                tier: str = "flex", timeout: int = 900, attempts: int = 5) -> dict:
    """One synchronous transcription request.

    ``tier`` is the service tier. Flex is 50% off Standard and is the default
    because the corpus run does not care about latency. Flex is sheddable, so
    503/429 are routine and need generous backoff; the docs ask for 10+ minute
    timeouts because flex requests can sit in a queue.
    """
    b64 = base64.b64encode(audio.read_bytes()).decode()
    body = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "audio/mp3", "data": b64}},
        ]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": SCHEMA,
            "thinkingConfig": {"thinkingLevel": thinking},
        },
    }
    if tier and tier != "standard":
        body["service_tier"] = tier
    last = None
    for i in range(attempts):
        r = requests.post(f"{API}/{MODEL}:generateContent", params={"key": key},
                          json=body, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        last = r
        if r.status_code in (429, 500, 502, 503, 504) and i < attempts - 1:
            wait = min(5 * (2 ** i), 120)
            print(f"   HTTP {r.status_code}; retrying in {wait}s "
                  f"({i + 1}/{attempts - 1})", flush=True)
            time.sleep(wait)
            continue
        break
    raise RuntimeError(f"HTTP {last.status_code}: {last.text[:500]}")


def parse_response(resp: dict) -> tuple[list[dict], dict]:
    usage = resp.get("usageMetadata", {})
    cand = (resp.get("candidates") or [{}])[0]
    parts = cand.get("content", {}).get("parts", [])
    txt = "".join(p.get("text", "") for p in parts)
    try:
        data = json.loads(txt)
        segs = data.get("segments", [])
    except Exception:
        segs = []
    return segs, usage


# ---------------------------------------------------------------------------
# batch API
#
# The Batch API is 50% off Standard and is the only sane way to run the corpus
# at volume. It needs three things the sync path does not: audio uploaded via
# the Files API (inline base64 would bloat the JSONL to hundreds of MB), a
# JSONL request file, and a job to poll.
# ---------------------------------------------------------------------------

def upload_file(path: Path, key: str, display_name: str | None = None) -> str:
    """Resumable upload to the Files API. Returns the file URI."""
    name = display_name or path.name
    payload = path.read_bytes()
    body = json.dumps({"file": {"display_name": name,
                               "mime_type": "audio/mp3"}}).encode()
    # The declared length is the size of the MEDIA, not of this metadata body --
    # the finalize step rejects the upload if the bytes do not match.
    r = requests.post(
        UPLOAD, params={"key": key},
        headers={
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(len(payload)),
            "X-Goog-Upload-Header-Content-Type": "application/octet-stream",
            "Content-Type": "application/json",
        }, data=body, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"upload start failed {r.status_code}: {r.text[:300]}")
    url = r.headers.get("x-goog-upload-url")
    if not url:
        raise RuntimeError(f"no x-goog-upload-url in response ({r.status_code}): {r.text[:300]}")
    r = requests.put(url, headers={
        "Content-Length": str(len(payload)),
        "X-Goog-Upload-Offset": "0",
        "X-Goog-Upload-Command": "upload, finalize",
    }, data=payload, timeout=1800)
    if r.status_code != 200:
        raise RuntimeError(f"upload PUT failed {r.status_code}: {r.text[:400]}")
    uri = (r.json().get("file") or {}).get("uri")
    if not uri:
        raise RuntimeError(f"upload returned no uri: {r.text[:300]}")
    print(f"   uploaded {name} ({len(payload)/1e6:.1f} MB) -> {uri}", flush=True)
    return uri


def wait_file_active(uri: str, key: str, timeout: int = 900) -> None:
    """Poll until the upload is usable. Audio is usable immediately; the
    Files API omits ``state`` entirely for files that need no processing."""
    url = uri if uri.startswith("http") else f"{ROOT}/v1beta/{uri}"
    deadline = time.time() + timeout
    while True:
        r = requests.get(url, params={"key": key}, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"file {uri} GET failed {r.status_code}: {r.text[:300]}")
        state = (r.json().get("file") or {}).get("state")
        if state in (None, "ACTIVE"):
            return
        if state == "FAILED":
            raise RuntimeError(f"file {uri} is FAILED: {r.text[:300]}")
        if time.time() > deadline:
            raise RuntimeError(f"file {uri} still {state} after {timeout}s")
        time.sleep(2)


def build_request(prompt: str, file_uri: str, thinking: str = "high") -> dict:
    """A GenerateContentRequest, identical in shape to the sync path."""
    return {
        "contents": [{"parts": [
            {"text": prompt},
            {"fileData": {"mimeType": "audio/mp3", "fileUri": file_uri}},
        ]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": SCHEMA,
            "thinkingConfig": {"thinkingLevel": thinking},
        },
    }


def write_batch_jsonl(items: list[tuple[str, dict]], out: Path) -> Path:
    lines = [json.dumps({"key": k, "request": r}) for k, r in items]
    out.write_text("\n".join(lines) + "\n")
    print(f"   batch file: {out} ({out.stat().st_size/1e6:.2f} MB, {len(lines)} requests)",
          flush=True)
    return out


def upload_batch_jsonl(path: Path, key: str) -> str:
    """Upload the request JSONL. Returns the File *name* (e.g. ``files/abc``),
    which is what BatchGenerateContent expects -- not the URI."""
    payload = path.read_bytes()
    body = json.dumps({"file": {"display_name": path.name,
                               "mime_type": "application/jsonl"}}).encode()
    r = requests.post(
        UPLOAD, params={"key": key},
        headers={
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(len(payload)),
            "X-Goog-Upload-Header-Content-Type": "application/jsonl",
            "Content-Type": "application/json",
        }, data=body, timeout=120)
    r.raise_for_status()
    url = r.headers.get("x-goog-upload-url")
    if not url:
        raise RuntimeError(f"no x-goog-upload-url for batch jsonl ({r.status_code}): {r.text[:300]}")
    r = requests.put(url, headers={
        "Content-Length": str(len(payload)),
        "X-Goog-Upload-Offset": "0",
        "X-Goog-Upload-Command": "upload, finalize",
    }, data=payload, timeout=1800)
    if r.status_code != 200:
        raise RuntimeError(f"batch jsonl upload failed {r.status_code}: {r.text[:400]}")
    name = (r.json().get("file") or {}).get("name")
    if not name:
        raise RuntimeError(f"batch jsonl upload returned no file name: {r.text[:300]}")
    return name


def submit_batch(jsonl_uri: str, key: str, name: str) -> str:
    body = {"batch": {"display_name": name,
                      "input_config": {"file_name": jsonl_uri}}}
    r = requests.post(f"{API}/{MODEL}:batchGenerateContent", params={"key": key},
                      json=body, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"submit HTTP {r.status_code}: {r.text[:500]}")
    return r.json()["name"]


def poll_batch(job: str, key: str, interval: int = 60) -> tuple[str, str]:
    """Block until the job settles. Returns (state, destination file name)."""
    done = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED"}
    url = job if job.startswith("http") else f"{ROOT}/v1beta/{job}"
    while True:
        r = requests.get(url, params={"key": key}, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"batch GET failed {r.status_code}: {r.text[:300]}")
        d = r.json()
        state = d.get("state", "JOB_STATE_UNSPECIFIED")
        if state in done:
            return state, ((d.get("dest") or {}).get("fileName") or "")
        counts = d.get("metadata") or {}
        print(f"   batch {job}: {state} "
              f"(completed {counts.get('completedRequestCount', 0)}"
              f"/{counts.get('totalRequestCount', '?')})", flush=True)
        time.sleep(interval)


def download_batch_results(file_name: str, key: str) -> list[dict]:
    url = file_name if file_name.startswith("http") else f"{ROOT}/v1beta/{file_name}"
    r = requests.get(url, params={"key": key}, timeout=1800)
    if r.status_code != 200:
        raise RuntimeError(f"results download failed {r.status_code}: {r.text[:300]}")
    return [json.loads(ln) for ln in r.text.splitlines() if ln.strip()]


def transcribe_windows(key: str, prompt: str, wav: Path, out: Path, offset_s: float,
                       limit: int, window: int, thinking: str, tier: str,
                       totals: dict, tiers: set, failures: list,
                       seg_path: Path, done_windows: set,
                       min_window: int = 600, depth: int = 0) -> list[dict]:
    """Transcribe ``[offset_s, limit)`` in ``window``-second chunks.

    A window that comes back with no segments is retried as two halves rather
    than accepted, because a silent hour is the single most expensive failure
    mode: at 1 h windows the model occasionally exhausts its output budget
    mid-reasoning and returns nothing. Bisecting only the failures is far
    cheaper than halving every window -- thinking is per-request, so 1800 s
    windows cost ~5.8x the 1 h rate, while a retry costs only the failed window.
    """
    all_segments: list[dict] = []
    for off in range(offset_s, int(limit), window):
        dur = min(window, int(limit) - off)
        if depth == 0 and off in done_windows:
            print(f"   window {off}s: already done in a previous run, skipping", flush=True)
            continue
        win = out / f"win_{off:06d}.mp3"
        encode_window(wav, off, dur, win)
        try:
            resp = call_gemini(key, prompt, win, thinking, tier=tier)
        except Exception as e:
            print(f"   window {off}s FAILED: {e}", flush=True)
            if _is_quota_error(e):
                print("   quota exhausted -- stopping. Re-run later; completed "
                      "windows are checkpointed and will be skipped.", flush=True)
                failures.append({"start_s": off, "duration_s": dur, "reason": "quota"})
                return all_segments
            if dur > min_window:
                half = dur // 2
                all_segments += transcribe_windows(
                    key, prompt, wav, out, off, off + half, half, thinking, tier,
                    totals, tiers, failures, seg_path, done_windows,
                    min_window, depth + 1)
                all_segments += transcribe_windows(
                    key, prompt, wav, out, off + half, off + dur, half, thinking, tier,
                    totals, tiers, failures, seg_path, done_windows,
                    min_window, depth + 1)
            else:
                failures.append({"start_s": off, "duration_s": dur, "reason": "request error"})
            continue
        segs, usage = parse_response(resp)
        totals["prompt"] += usage.get("promptTokenCount", 0)
        totals["candidates"] += usage.get("candidatesTokenCount", 0)
        totals["thoughts"] += usage.get("thoughtsTokenCount", 0)
        tiers.add(usage.get("serviceTier", "unknown"))
        if not segs and dur > min_window:
            spent = usage.get("thoughtsTokenCount", 0) + usage.get("candidatesTokenCount", 0)
            print(f"   window {off}s returned no segments after {spent:,} output tokens; "
                  f"bisecting into {dur // 2}s halves", flush=True)
            half = dur // 2
            all_segments += transcribe_windows(
                key, prompt, wav, out, off, off + half, half, thinking, tier,
                totals, tiers, failures, seg_path, done_windows,
                min_window, depth + 1)
            all_segments += transcribe_windows(
                key, prompt, wav, out, off + half, off + dur, half, thinking, tier,
                totals, tiers, failures, seg_path, done_windows,
                min_window, depth + 1)
            continue
        if not segs:
            failures.append({"start_s": off, "duration_s": dur, "reason": "no segments"})
        shifted = []
        for s in segs:
            s["start"] = float(s.get("start", 0)) + off
            s["end"] = float(s.get("end", 0)) + off
            shifted.append(s)
        all_segments += shifted
        # Checkpoint immediately: a quota wall or crash must not cost the work.
        with seg_path.open("a") as fh:
            for s in shifted:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")
        done_windows.add(off)
        print(f"   window {off}s: {len(segs)} segments; "
              f"cum tokens in={totals['prompt']} out={totals['candidates']} "
              f"think={totals['thoughts']}", flush=True)
    return all_segments


def slice_segments(segments: list[dict], clips: list[dict],
                    offset_s: float = 0.0) -> tuple[dict[str, str], list[str]]:
    """Assign segments to clips by midpoint, with an overlap fallback.

    Midpoint assignment alone leaves gaps wherever the model's segmentation
    does not line up with the clip boundaries, which on a measured run was
    ~30% of clips. Any clip left empty is retried against the segment with the
    largest temporal overlap, so it gets text; those clips are returned
    separately because the assignment is less reliable.

    ``offset_s`` maps the manifest timeline onto the downloaded audio
    (audio_time = manifest_time + offset_s) so we compare like with like.
    """
    out: dict[str, str] = {}
    weak: list[str] = []
    for c in clips:
        lo = c["start"] + offset_s - 0.4
        hi = c["end"] + offset_s + 0.4
        pieces = [s.get("text", "") for s in segments
                  if lo <= (float(s.get("start", 0)) + float(s.get("end", 0))) / 2 <= hi]
        if pieces:
            out[c["clip_id"]] = " ".join(p.strip() for p in pieces if p.strip())
            continue
        # Fallback: the segment overlapping this clip the most.
        best, best_ov = None, 0.0
        for s in segments:
            a = max(lo, float(s.get("start", 0)))
            b = min(hi, float(s.get("end", 0)))
            if b - a > best_ov:
                best, best_ov = s, b - a
        if best is not None and (best.get("text") or "").strip():
            out[c["clip_id"]] = best["text"].strip()
            weak.append(c["clip_id"])
    return out, weak


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--dataset", default="hammertoe__barbados-parliament-clips-v1")
    ap.add_argument("--context", default="context")
    ap.add_argument("--media", default="media")
    ap.add_argument("--out", default="longform")
    ap.add_argument("--max-seconds", type=int, default=0,
                    help="cap audio downloaded/processed; 0 (default) means the whole sitting")
    ap.add_argument("--window", type=int, default=900)
    ap.add_argument("--thinking", default="high", choices=["low", "medium", "high"])
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="skip the API and use synthetic segments to test slicing")
    ap.add_argument("--tier", default="flex", choices=["standard", "flex", "priority"],
                    help="inference tier; flex is 50%% off and sheddable (default)")
    ap.add_argument("--batch", action="store_true",
                    help="use the Batch API (50%% off) instead of sync calls. "
                         "Needs a direct Google AI Studio key; gateway keys reject it.")
    ap.add_argument("--prepare-only", action="store_true",
                    help="with --batch: upload windows and write the JSONL, do not submit")
    ap.add_argument("--poll-interval", type=int, default=60,
                    help="seconds between batch status checks")
    ap.add_argument("--min-window", type=int, default=600,
                    help="smallest window to bisect a failed one down to")
    ap.add_argument("--direct-sample", type=int, default=0,
                    help="also transcribe N clips directly and compare to the slice")
    args = ap.parse_args(argv)

    dataset = Path(args.dataset)
    manifest = dataset / "MANIFEST.jsonl"
    media = Path(args.media)
    out = Path(args.out) / args.video_id
    out.mkdir(parents=True, exist_ok=True)

    ctx_path = Path(args.context) / f"{args.video_id}.json"
    context = json.loads(ctx_path.read_text()) if ctx_path.exists() else None
    if context:
        print(f"context: {context['order_paper']['label']}")
    else:
        print("context: none found (running without order-paper biasing)")

    clips = load_clips(manifest, args.video_id)
    if not clips:
        raise SystemExit(f"no clips for {args.video_id}")
    yt = None
    with manifest.open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d["video_id"] == args.video_id:
                yt = d["source"]["youtube_url"]
                break
    print(f"video: {args.video_id}  clips={len(clips)}  url={yt}")

    print("1. download + normalise")
    wav = download_audio(args.video_id, yt, media, args.max_seconds)
    with wave.open(str(wav), "rb") as w:
        total = w.getnframes() / w.getframerate()
    print(f"   audio: {total:.0f}s @ {SR} Hz mono")

    print("2. verify timeline (clip timestamps vs source audio)")
    ver = verify_timeline(wav, clips, dataset, limit_seconds=total)
    (out / "_timeline.json").write_text(json.dumps(ver, indent=2) + "\n")

    if args.verify_only:
        print("verify-only; stopping before the API call.")
        return 0

    limit = total if not args.max_seconds else min(total, args.max_seconds)
    offset_s = (ver.get("median_offset_ms") or 0) / 1000.0
    if offset_s:
        print(f"   applying measured timeline offset {offset_s*1000:+.0f} ms when slicing")
    in_range = [c for c in clips if c["end"] <= limit]
    totals = {"prompt": 0, "candidates": 0, "thoughts": 0}
    tiers: set[str] = set()

    if args.dry_run:
        # Synthetic segments on the downloaded-audio timeline, one per clip —
        # exercises the slice path without touching the API.
        all_segments = [{"start": c["start"] + offset_s, "end": c["end"] + offset_s,
                         "text": f"[synthetic {c['clip_id']}]"} for c in in_range]
        print(f"3. dry-run: {len(all_segments)} synthetic segments (no API call)")
    else:
        key = load_api_key()
        if not key:
            print("ERROR: no GEMINI_API_KEY/GOOGLE_API_KEY; stopping after timeline check.",
                  file=sys.stderr)
            return 2
        mode = "batch" if args.batch else f"{args.tier} tier"
        print(f"3. window + transcribe  (model={MODEL}, thinking={args.thinking}, {mode})",
              flush=True)
        prompt = build_prompt(context)
        all_segments = []
        wdur = args.window
        # window start offset -> (request key, window path)
        windows: list[tuple[str, int, Path]] = []
        for off in range(0, int(limit), wdur):
            dur = min(wdur, int(limit) - off)
            win = out / f"win_{off:06d}.mp3"
            encode_window(wav, off, dur, win)
            windows.append((f"w{off:06d}", off, win))
        print(f"   {len(windows)} windows encoded", flush=True)

        if args.batch:
            items = []
            for k, _off, win in windows:
                uri = upload_file(win, key)
                wait_file_active(uri, key)
                items.append((k, build_request(prompt, uri, args.thinking)))
            jsonl = write_batch_jsonl(items, out / "_batch.jsonl")
            if args.prepare_only:
                print("   --prepare-only: stopping before submission.", flush=True)
                return 0
            jsonl_uri = upload_batch_jsonl(jsonl, key)
            job = submit_batch(jsonl_uri, key, f"longform-{args.video_id}")
            (out / "_batch_job.txt").write_text(job + "\n")
            print(f"   submitted {job}", flush=True)
            state, dest = poll_batch(job, key, interval=args.poll_interval)
            print(f"   batch finished: {state}", flush=True)
            if state != "JOB_STATE_SUCCEEDED" or not dest:
                (out / "_batch_error.json").write_text(
                    json.dumps({"state": state, "job": job}, indent=2) + "\n")
                return 3
            rows = download_batch_results(dest, key)
            (out / "_batch_results.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
            by_key = {r.get("key"): (r.get("response") or {}) for r in rows}
            for k, off, _win in windows:
                resp = by_key.get(k)
                if not resp:
                    print(f"   window {off}s: no result in batch output", flush=True)
                    continue
                segs, usage = parse_response(resp)
                for s in segs:
                    s["start"] = float(s.get("start", 0)) + off
                    s["end"] = float(s.get("end", 0)) + off
                all_segments += segs
                totals["prompt"] += usage.get("promptTokenCount", 0)
                totals["candidates"] += usage.get("candidatesTokenCount", 0)
                totals["thoughts"] += usage.get("thoughtsTokenCount", 0)
                print(f"   window {off}s: {len(segs)} segments; "
                      f"cum tokens in={totals['prompt']} out={totals['candidates']} "
                      f"think={totals['thoughts']}", flush=True)
        else:
            failures: list[dict] = []
            all_segments = transcribe_windows(
                key, prompt, wav, out, 0, limit, wdur, args.thinking, args.tier,
                totals, tiers, failures)
            if failures:
                print(f"   {len(failures)} window(s) still produced nothing after "
                      f"bisecting down to {args.min_window}s", flush=True)

    (out / "_segments.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in all_segments) + "\n")

    print("4. slice to clips", flush=True)
    sliced, weak = slice_segments(all_segments, in_range, offset_s=offset_s)
    for cid, text in sliced.items():
        (out / f"{cid}.json").write_text(json.dumps(
            {"clip_id": cid, "text": text,
             "assignment": "overlap_fallback" if cid in weak else "midpoint"},
            ensure_ascii=False) + "\n")
    print(f"   {len(sliced)}/{len(in_range)} in-range clips received a transcript",
          flush=True)
    if weak:
        print(f"   {len(weak)} assigned by overlap fallback, not midpoint",
              flush=True)

    std = (totals["prompt"] / 1e6 * 0.75
           + (totals["candidates"] + totals["thoughts"]) / 1e6 * 3.75)
    summary = {
        "video_id": args.video_id,
        "mode": "dry-run" if args.dry_run else ("gemini-batch" if args.batch else "gemini"),
        "model": MODEL,
        "thinking": args.thinking,
        "tier_requested": "batch" if args.batch else args.tier,
        # What the API actually billed us. If this says "standard" the 50%
        # discount did not apply and the bill is double the estimate.
        "tier_billed": sorted(tiers),
        "processed_seconds": int(limit),
        "windows": math.ceil(limit / args.window),
        "segments": len(all_segments),
        "clips_in_range": len(in_range),
        "clips_transcribed": len(sliced),
        "clips_overlap_fallback": len(weak),
        "failed_windows": failures if not args.dry_run else [],
        "timeline_strong": ver["strong"],
        "timeline_checked": ver["checked"],
        "audio_offset_ms": ver.get("median_offset_ms"),
        "tokens": totals,
        "est_cost_usd_standard": round(std, 4),
        "est_cost_usd_discounted": round(
            std if (tiers and tiers <= {"standard"}) else std / 2, 4),
    }
    if tiers and not tiers <= {"flex", "batch"}:
        print(f"WARNING: billed at {sorted(tiers)}, not flex -- cost is the "
              f"standard estimate.", file=sys.stderr)
    (out / "_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
