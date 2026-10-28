#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["requests", "beautifulsoup4", "lxml"]
# ///
"""Build per-video parliamentary context for the Barbados Parliament corpus.

Scrapes the *public* Order Papers from barbadosparliament.com, matches them to
source videos by sitting date + chamber, downloads the PDFs, extracts their text
with poppler's ``pdftotext``, and emits one small context bundle per video plus
a coverage report.

The neighbouring ``Debates`` (Hansard) search is members-only and is therefore
out of scope here; see ``scripts/README.md``.

Usage
-----
    uv run scripts/build_parliament_context.py --limit 20      # smoke test
    uv run scripts/build_parliament_context.py                # full run

Options
-------
    --dataset DIR      corpus root (default: hammertoe__barbados-parliament-clips-v1)
    --out DIR          output root (default: context)
    --limit N          stop after matching N videos (testing)
    --include-archive  also scrape the pre-2015 archive (slow, not needed for 2021+)
    --no-download      match only; skip PDF download + text extraction
    --delay SECONDS    politeness delay between live requests (default 0.3)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://www.barbadosparliament.com"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) HardEars-context/0.1"

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
MON_ABBR = {k[:3]: v for k, v in MONTHS.items()}

# Official parishes of Barbados (the entity class generic ASR mangles most).
PARISHES = [
    "Christ Church", "Saint Andrew", "Saint George", "Saint James", "Saint John",
    "Saint Joseph", "Saint Lucy", "Saint Michael", "Saint Peter", "Saint Philip",
    "Saint Thomas",
]
PARISH_ALT = {p: p.replace("Saint", "St.") for p in PARISHES}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Paper:
    chamber: str          # "house" | "senate"
    label: str            # e.g. "20th Sitting - Tuesday, 15th September, 2026"
    date: date | None
    posted: str | None
    url: str | None       # main order-paper PDF
    attachments: list[dict] = field(default_factory=list)  # [{label,url}]
    source: str = "search"  # "search" | "archive"

    def to_json(self) -> dict:
        d = asdict(self)
        d["date"] = self.date.isoformat() if self.date else None
        return d


# ---------------------------------------------------------------------------
# HTTP + cache
# ---------------------------------------------------------------------------

class Fetcher:
    def __init__(self, cache_dir: Path, delay: float):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        retry = Retry(total=4, backoff_factor=1.0,
                      status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=frozenset(["GET", "POST"]))
        self.s.mount("https://", HTTPAdapter(max_retries=retry))
        self._last = 0.0

    def _throttle(self):
        dt = time.monotonic() - self._last
        if dt < self.delay:
            time.sleep(self.delay - dt)
        self._last = time.monotonic()

    def _cpath(self, method: str, url: str, data: dict | None) -> Path:
        key = json.dumps([method, url, sorted((data or {}).items())], sort_keys=True)
        return self.cache_dir / (hashlib.sha1(key.encode()).hexdigest() + ".bin")

    def get(self, url: str, binary: bool = False) -> bytes:
        cp = self._cpath("GET", url, None)
        if cp.exists():
            return cp.read_bytes()
        self._throttle()
        r = self.s.get(url, timeout=60)
        r.raise_for_status()
        cp.write_bytes(r.content)
        return r.content

    def post(self, url: str, data: dict) -> bytes:
        cp = self._cpath("POST", url, data)
        if cp.exists():
            return cp.read_bytes()
        self._throttle()
        r = self.s.post(url, data=data, timeout=60)
        r.raise_for_status()
        cp.write_bytes(r.content)
        return r.content


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

_MONTH_RE = "|".join(MONTHS)
# day-first: "6th October 2026"
_DATE_RE_DM = re.compile(
    rf"(\d{{1,2}})\s*(?:st|nd|rd|th)?[,\s]*({_MONTH_RE})\b\.?,?\s*(\d{{4}})?",
    re.I,
)
# month-first: "August 31st, 2021"
_DATE_RE_MD = re.compile(
    rf"({_MONTH_RE})\s+(\d{{1,2}})\s*(?:st|nd|rd|th)?,?\s*(\d{{4}})?",
    re.I,
)


def parse_uk_date(text: str, fallback_year: int | None = None) -> date | None:
    """Find the first ``<day> <Month> [<year>]`` (or ``<Month> <day>``) in *text*.

    The month must be a real month name so that labels like
    ``"22nd Sitting - 6th October 2026"`` are parsed from the sitting date
    rather than bailing out on the leading ``22nd Sitting``. Both day-first
    ("6th October 2026") and month-first ("August 31st, 2021") forms occur.
    """
    candidates: list[tuple[int, int, int, int | None]] = []
    for m in _DATE_RE_DM.finditer(text):
        candidates.append((m.start(), int(m.group(1)), MON_ABBR.get(m.group(2)[:3].lower()),
                           int(m.group(3)) if m.group(3) else None))
    for m in _DATE_RE_MD.finditer(text):
        candidates.append((m.start(), int(m.group(2)), MON_ABBR.get(m.group(1)[:3].lower()),
                           int(m.group(3)) if m.group(3) else None))
    for _, day, mon, yr in sorted(candidates):
        if not mon or not (1 <= day <= 31):
            continue
        year = yr if yr else fallback_year
        if not year:
            continue
        try:
            return date(year, mon, day)
        except ValueError:
            continue
    return None


def parse_rows(html: bytes, chamber: str) -> list[Paper]:
    soup = BeautifulSoup(html, "lxml")
    out: list[Paper] = []
    for tr in soup.select("tr.modern-style"):
        tds = tr.find_all("td")
        if not tds:
            continue
        links = []
        for a in tds[0].find_all("a", href=True):
            href = a["href"]
            if ".pdf" not in href.lower():
                continue
            label = a.get_text(" ", strip=True) or href.rsplit("/", 1)[-1]
            links.append({"label": label, "url": urljoin(BASE, href)})
        if not links:
            continue
        posted = None
        if len(tds) > 1:
            dm = re.search(r"\d{4}-\d{2}-\d{2}", tds[1].get_text(" ", strip=True))
            posted = dm.group(0) if dm else None
        main_label = links[0]["label"]
        # A sitting label looks like "NNth Sitting - ..."; prefer it over a stray filename.
        label = main_label
        yr = int(posted[:4]) if posted else None
        sitting = parse_uk_date(label, fallback_year=yr)
        if sitting is None and posted:
            sitting = date.fromisoformat(posted)  # posted date is reliable
        out.append(Paper(
            chamber=chamber,
            label=label,
            date=sitting,
            posted=posted,
            url=links[0]["url"],
            attachments=links[1:],
            source="search",
        ))
    return out


def scrape_search(f: Fetcher, chamber: str, max_pages: int = 60) -> list[Paper]:
    type_id = {"house": 1, "senate": 2}[chamber]
    url = f"{BASE}/order_papers/search/type/{type_id}"
    seen: dict[str, Paper] = {}
    for page in range(max_pages):
        html = f.post(url, {"COL_ID": "-1", "ORD_ID": "0",
                            "OFF_SET": str(page * 20), "OP_KEYWORD_S": ""})
        rows = parse_rows(html, chamber)
        new = 0
        for p in rows:
            key = f"{p.date}|{p.label}|{p.url}"
            if key not in seen:
                seen[key] = p
                new += 1
        print(f"  [{chamber}] page {page + 1} (+{new}, total {len(seen)})", flush=True)
        if not rows or new == 0:
            break
    return list(seen.values())


def scrape_archive(f: Fetcher, chamber: str) -> list[Paper]:
    listing = f.get(f"{BASE}/past_order_papers/{chamber}")
    ids = sorted({int(m) for m in re.findall(rf"past_order_papers/{chamber}/view/(\d+)",
                                             listing.decode("utf-8", "replace"))})
    print(f"  [{chamber}] archive: {len(ids)} view pages", flush=True)
    out: list[Paper] = []
    for i, vid in enumerate(ids, 1):
        try:
            page = f.get(f"{BASE}/past_order_papers/{chamber}/view/{vid}")
        except Exception as e:  # pragma: no cover - network resilience
            print(f"    archive {vid} failed: {e}", flush=True)
            continue
        soup = BeautifulSoup(page, "lxml")
        links = []
        for a in soup.find_all("a", href=True):
            if ".pdf" not in a["href"].lower():
                continue
            label = a.get_text(" ", strip=True) or a["href"].rsplit("/", 1)[-1]
            links.append({"label": label, "url": urljoin(BASE, a["href"])})
        if not links:
            continue
        # Best-effort sitting date from the page title / headings.
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        heading = " ".join(h.get_text(" ", strip=True) for h in soup.find_all(["h1", "h2", "h3"]))
        parsed = parse_uk_date(title) or parse_uk_date(heading)
        out.append(Paper(
            chamber=chamber,
            label=(title or f"archive/{vid}")[:120],
            date=parsed,
            posted=None,
            url=links[0]["url"],
            attachments=links[1:],
            source="archive",
        ))
        if i % 25 == 0:
            print(f"    archive {i}/{len(ids)}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Corpus side
# ---------------------------------------------------------------------------

def infer_chamber(title: str) -> str:
    t = (title or "").lower()
    if "senate" in t:
        return "senate"
    if "youth parliament" in t:
        return "youth"
    if "committee" in t:
        return "committee"
    if "estimates" in t or "estimate" in t:
        return "house"      # Estimates debates are House business
    if "house" in t or "assembly" in t:
        return "house"
    return "unknown"


def load_videos(manifest: Path) -> list[dict]:
    vids: dict[str, dict] = {}
    with manifest.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            vid = d.get("video_id")
            if not vid or vid in vids:
                continue
            title = d.get("video_title") or ""
            src = d.get("source") or {}
            pub = src.get("video_published_at") or ""
            year = int(pub[:4]) if pub[:4].isdigit() else None
            vids[vid] = {
                "video_id": vid,
                "title": title,
                "youtube_url": src.get("youtube_url"),
                "published_at": pub,
                "chamber": infer_chamber(title),
                "sitting_date": (parse_uk_date(title, fallback_year=year).isoformat()
                                 if parse_uk_date(title, fallback_year=year) else None),
            }
    return list(vids.values())


def match_videos(videos: list[dict], papers: list[Paper], tolerance: int = 3) -> list[dict]:
    by_date: dict[tuple, list[Paper]] = {}
    for p in papers:
        if p.date:
            by_date.setdefault((p.chamber, p.date), []).append(p)

    matches = []
    for v in videos:
        chosen, how = None, None
        if v["sitting_date"]:
            d = date.fromisoformat(v["sitting_date"])
            ch = v["chamber"]
            if (ch, d) in by_date:
                chosen, how = by_date[(ch, d)][0], "exact"
            else:
                for other in ("house", "senate"):
                    if (other, d) in by_date:
                        chosen, how = by_date[(other, d)][0], "date-only"
                        break
                if chosen is None:
                    best, best_delta = None, 999
                    for (c, pd), plist in by_date.items():
                        if c != ch and ch != "unknown":
                            continue
                        delta = abs((pd - d).days)
                        if delta <= tolerance and delta < best_delta:
                            best, best_delta = plist[0], delta
                    if best is not None:
                        chosen, how = best, f"nearest-{best_delta}d"
        matches.append({**v, "paper": chosen.to_json() if chosen else None, "match": how or "none"})
    return matches


# ---------------------------------------------------------------------------
# PDF text + glossary
# ---------------------------------------------------------------------------

def have_pdftotext() -> bool:
    return shutil.which("pdftotext") is not None


def extract_pdf_text(f: Fetcher, url: str, cache_pdf: Path) -> str:
    if not cache_pdf.exists():
        cache_pdf.write_bytes(f.get(url, binary=True))
    try:
        r = subprocess.run(["pdftotext", "-layout", str(cache_pdf), "-"],
                           capture_output=True, timeout=120)
        if r.returncode == 0:
            return r.stdout.decode("utf-8", "replace")
    except Exception:
        pass
    return ""


STOP = set("the a an of and to in on for at by with from re no not this that these those "
           "order paper sitting house senate parliament barbados assembly honourable day "
           "please mr mrs ms dr hon".split())


def build_glossary(text: str, limit: int = 120) -> dict:
    """Cheap proper-noun extraction: capitalised n-grams + parish/bill/resolution hits."""
    terms: dict[str, int] = {}
    for m in re.finditer(r"\b([A-Z][a-z]{2,}(?:\s+(?:of|the|and|&)\s+)?(?:\s*[A-Z][a-z]{2,}){0,3})\b", text):
        phrase = re.sub(r"\s+", " ", m.group(1)).strip()
        if phrase.lower() in STOP or len(phrase) < 4:
            continue
        terms[phrase] = terms.get(phrase, 0) + 1

    parishes = [p for p in PARISHES if re.search(re.escape(p), text)
                or re.search(re.escape(PARISH_ALT[p]), text)]
    bills = sorted({m.group(0).strip()
                    for m in re.finditer(r"[A-Z][^.\n]{4,120}?Bill,?\s*\d{4}", text)})
    resolutions = sorted({m.group(0).strip()
                          for m in re.finditer(r"Resolution[^.\n]{4,120}", text)})

    ranked = sorted(terms.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    return {
        "proper_nouns": [{"term": t, "count": c} for t, c in ranked],
        "parishes": parishes,
        "bills": bills[:40],
        "resolutions": resolutions[:40],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hammertoe__barbados-parliament-clips-v1")
    ap.add_argument("--out", default="context")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--include-archive", action="store_true")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--delay", type=float, default=0.3)
    args = ap.parse_args(argv)

    dataset = Path(args.dataset)
    out = Path(args.out)
    (out / "text").mkdir(parents=True, exist_ok=True)
    (out / "pdfs").mkdir(parents=True, exist_ok=True)
    f = Fetcher(out / ".cache", args.delay)

    if not shutil.which("pdftotext") and not args.no_download:
        print("WARNING: pdftotext not found (brew install poppler); "
              "running match-only.", file=sys.stderr)

    # 1. order papers ---------------------------------------------------------
    print("Scraping order papers...")
    papers: list[Paper] = []
    for chamber in ("house", "senate"):
        papers += scrape_search(f, chamber)
        if args.include_archive:
            papers += scrape_archive(f, chamber)
    (out / "_order_papers.jsonl").write_text(
        "\n".join(json.dumps(p.to_json(), ensure_ascii=False) for p in papers) + "\n")
    dated = [p for p in papers if p.date]
    print(f"  {len(papers)} order papers ({len(dated)} dated), "
          f"range {min((p.date for p in dated), default=None)} .. {max((p.date for p in dated), default=None)}")

    # 2. videos ---------------------------------------------------------------
    videos = load_videos(dataset / "MANIFEST.jsonl")
    print(f"  {len(videos)} unique source videos")
    (out / "_videos.jsonl").write_text(
        "\n".join(json.dumps(v, ensure_ascii=False) for v in videos) + "\n")

    # 3. match ----------------------------------------------------------------
    matches = match_videos(videos, papers)
    (out / "_matches.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in matches) + "\n")
    matched = [m for m in matches if m["paper"]]
    print(f"  matched {len(matched)}/{len(matches)} videos")

    # 4. download + extract + bundle -----------------------------------------
    if args.limit:
        matched = matched[:args.limit]

    pdf_cache = out / "pdfs"
    for i, m in enumerate(matched, 1):
        vid = m["video_id"]
        text = ""
        if not args.no_download:
            pdf_url = m["paper"]["url"]
            h = hashlib.sha1(pdf_url.encode()).hexdigest()[:16]
            text = extract_pdf_text(f, pdf_url, pdf_cache / f"{h}.pdf")
            if text:
                (out / "text" / f"{vid}.txt").write_text(text, encoding="utf-8")
        bundle = {
            "video_id": vid,
            "title": m["title"],
            "chamber": m["chamber"],
            "sitting_date": m["sitting_date"],
            "youtube_url": m["youtube_url"],
            "match": m["match"],
            "order_paper": m["paper"],
            "text_file": f"text/{vid}.txt" if text else None,
            "glossary": build_glossary(text) if text else None,
        }
        (out / f"{vid}.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
        if i % 25 == 0 or i == len(matched):
            print(f"  bundled {i}/{len(matched)}", flush=True)

    # 5. coverage report ------------------------------------------------------
    report = build_coverage(matches, out)
    (out / "_coverage.md").write_text(report, encoding="utf-8")
    print(f"\nWrote {out}/_coverage.md")
    return 0


def build_coverage(matches: list[dict], out: Path) -> str:
    from collections import Counter
    total = len(matches)
    by_match = Counter(m["match"] for m in matches)
    by_chamber = Counter((m["chamber"], bool(m["paper"])) for m in matches)
    by_year = Counter((m["sitting_date"][:4] if m["sitting_date"] else "????", bool(m["paper"]))
                      for m in matches)

    lines = ["# Order-paper context coverage", ""]
    lines.append(f"Source videos: **{total}**")
    lines.append(f"Matched: **{sum(1 for m in matches if m['paper'])}** "
                 f"({100 * sum(1 for m in matches if m['paper']) / max(total, 1):.1f}%)")
    lines.append("")
    lines.append("## Match type")
    for k, v in by_match.most_common():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## By chamber")
    lines.append("| chamber | matched | unmatched |")
    lines.append("|---|---|---|")
    chambers = sorted({c for c, _ in by_chamber})
    for c in chambers:
        lines.append(f"| {c} | {by_chamber.get((c, True), 0)} | {by_chamber.get((c, False), 0)} |")
    lines.append("")
    lines.append("## By year")
    lines.append("| year | matched | unmatched |")
    lines.append("|---|---|---|")
    years = sorted({y for y, _ in by_year})
    for y in years:
        lines.append(f"| {y} | {by_year.get((y, True), 0)} | {by_year.get((y, False), 0)} |")
    lines.append("")
    lines.append("## Unmatched videos")
    lines.append("")
    for m in matches:
        if not m["paper"]:
            lines.append(f"- `{m['video_id']}` — {m['title']}")
    lines.append("")
    lines.append("## Matched by nearest-date (worth a manual check)")
    lines.append("")
    for m in matches:
        if m["match"] and m["match"].startswith("nearest"):
            lines.append(f"- `{m['video_id']}` {m['sitting_date']} → paper {m['paper']['date']} "
                         f"({m['match']}) — {m['title']}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
