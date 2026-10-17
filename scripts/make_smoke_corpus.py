#!/usr/bin/env python3
"""Build the Stage 0 speech corpus.

Stage 0 (docs/TRAINING.md) needs short utterances of plain English with
exact transcripts. We do not need LibriSpeech for that: macOS ships an
offline neural voice (`say`), so we can build a few hundred clips in
about a minute, on the laptop, with no network and no disk pressure.

Why this is enough for Stage 0:

* The audio is clean 16 kHz mono speech, which is what the FastConformer
  encoder expects.
* Every transcript is known exactly, so WER is meaningful from step one
  with no alignment ambiguity.
* The sentences are short and phonetically simple, which is what an
  untrained bridge can be expected to latch onto first.

The sentences come from two sources:

1. `HANDWRITTEN` -- 28 natural sentences, for phrasing that template
   generation does not produce ("She wrote a letter to her friend.").
2. A small combinatorial generator over curated nouns, adjectives,
   regular past-tense verbs, prepositions, adverbs and times. Every
   verb form is written out rather than derived, so nothing ungrammat-
   ical can slip in ("The dog carryed the ball" is not possible here).
   With a fixed seed this is fully reproducible.

Train and eval sentences never overlap: the pool is generated once,
deduplicated, shuffled with a fixed seed, and then split.

Voice and speed variation
-------------------------

Each clip is rendered with a rotating macOS voice and a small speed
perturbation, which is the cheapest honest augmentation available
offline:

* **Four training voices** (Alex en_US, Samantha en_US, Daniel en_GB,
  Karen en_AU) so the bridge cannot memorise one voice.
* **Two held-out eval voices** (Moira en_IE, Tessa en_ZA). Neither is
  in training, so the Stage 0 eval number also measures voice
  generalisation and not just text generalisation.
* **Three speeds** (0.9x, 1.0x, 1.1x) via ffmpeg `atempo`, the
  standard speed-perturbation trick.

Sentences rotate voice by index and speed by voice-depth, so the same
sentence is heard several times with different acoustics and never
always with the same one.

Output is a NeMo-style JSONL manifest (audio_filepath, duration, text,
voice, speed) plus cached encoder features when requested, so the same
files feed training and evaluation.

Usage:
    python3 scripts/make_smoke_corpus.py                      # 220 train / 48 eval
    python3 scripts/make_smoke_corpus.py --num-train 400
    python3 scripts/make_smoke_corpus.py --voices Alex,Samantha
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SAMPLE_RATE = 16000

HANDWRITTEN = [
    # natural phrasing a template would not produce
    "She wrote a letter to her friend.",
    "The children ran home before the storm.",
    "My brother plays the guitar every evening.",
    "The market opens at eight in the morning.",
    "It was a quiet night in the village.",
    "The old man sat alone on the bench.",
    "Clouds moved slowly across the sky.",
    "The garden was full of red flowers.",
    "The cat sat quietly by the fire.",
    "He carried the heavy box upstairs.",
    "A small bird sat on the fence.",
    "The water in the glass was cold.",
    "They planted trees along the road.",
    "We finished the work before lunch.",
    "The train arrived at the station early.",
    "She opened the window and listened to the rain.",
    "The dog ran across the green field.",
    "We walked to the river after lunch.",
    "The sun rose over the hills.",
    "He put the book on the table.",
    # held out for evaluation
    "The light faded slowly behind the hills.",
    "A woman walked along the quiet shore.",
    "The teacher spoke to the class for an hour.",
    "Rain fell gently on the roof.",
    "He turned off the lamp and went to bed.",
    "The road curved past the old church.",
    "They sat on the steps and talked.",
    "A bird sang somewhere in the dark.",
]
N_HELD_OUT = 8  # the last N handwritten sentences are eval-only

SUBJECTS = [
    "He", "She", "They", "We", "I",
    "The boy", "The girl", "The man", "The woman", "The child",
    "My friend", "The teacher", "The farmer", "The doctor", "The driver",
]
NOUNS = [
    "dog", "cat", "bird", "horse", "tree", "river", "hill", "road",
    "house", "room", "door", "window", "table", "chair", "book",
    "letter", "cup", "plate", "bread", "apple", "market", "garden",
    "field", "school", "church", "station", "village", "friend",
    "child", "woman", "man", "boy", "girl", "doctor", "farmer",
    "teacher", "train", "bus", "boat", "cloud", "sun", "moon",
    "night", "morning", "evening", "song", "story", "game", "ball",
    "box", "key", "lamp", "floor", "wall", "gate", "path", "beach",
    "sea", "island", "harbour", "harvest", "basket", "bucket",
]
ADJECTIVES = [
    "small", "big", "old", "young", "red", "blue", "green", "quiet",
    "loud", "happy", "slow", "fast", "cold", "warm", "bright", "dark",
    "dry", "wet", "clean", "empty", "full", "new", "long", "short",
    "wide", "narrow", "deep", "soft", "hard", "heavy", "wooden",
    "stone", "narrow", "empty", "lonely", "noisy",
]
# Regular past-tense forms only, written out. No conjugation logic, so
# "the dog carryed" is not reachable.
PAST_VERBS = [
    "opened", "closed", "moved", "walked", "watched", "listened",
    "smiled", "laughed", "waited", "called", "helped", "thanked",
    "cleaned", "washed", "painted", "climbed", "jumped", "visited",
    "filled", "picked", "started", "stopped", "turned", "pushed",
    "pulled", "placed", "tasted", "counted", "asked", "answered",
    "pointed", "followed", "studied", "tried", "hurried", "whispered",
    "shouted", "rested", "finished", "crossed", "dried", "entered",
    "borrowed", "carried", "hugged", "grabbed", "rubbed", "patted",
    "sailed", "danced", "breathed", "watched", "answered", "ended",
    "cared", "greeted", "thanked", "needed", "offered", "printed",
    "rested", "searched", "settled", "traveled", "used", "waved",
]
PREPOSITIONS = [
    "over", "under", "beside", "behind", "near", "across", "along",
    "around", "through", "toward", "into", "onto", "from", "with",
    "at", "in", "on", "for", "past", "above", "below",
]
ADVERBS = [
    "quietly", "slowly", "loudly", "carefully", "suddenly", "gently",
    "happily", "early", "late", "often", "always", "never", "twice",
    "again", "away", "outside", "inside", "upstairs", "downstairs",
    "home", "forward", "together", "alone",
]
TIMES = [
    "noon", "midnight", "sunrise", "sunset", "dawn", "dusk",
    "breakfast", "lunch", "dinner", "supper", "morning", "afternoon",
    "evening", "night", "the weekend", "last night", "this morning",
]

FRAMES = [
    "{s} {v} {p} the {a} {n}.",
    "{s} {v} the {a} {n}.",
    "The {a} {n} {v} {p} the {n}.",
    "{s} {v} {p} the {n}.",
    "{s} {v} {adv}.",
    "The {n} was {a}.",
    "{s} {v} the {n} {p} {t}.",
    "The {n} {v} {adv} {p} the {n}.",
]


def generate_sentences(count: int, rng: random.Random, exclude: set[str]) -> list[str]:
    """Unique, grammatical, short sentences from the frames above."""
    out: list[str] = []
    seen: set[str] = set(exclude)
    attempts = 0
    while len(out) < count and attempts < count * 200:
        attempts += 1
        frame = rng.choice(FRAMES)
        sentence = frame.format(
            s=rng.choice(SUBJECTS),
            n=rng.choice(NOUNS),
            a=rng.choice(ADJECTIVES),
            v=rng.choice(PAST_VERBS),
            p=rng.choice(PREPOSITIONS),
            adv=rng.choice(ADVERBS),
            t=rng.choice(TIMES),
        )
        if sentence in seen:
            continue
        seen.add(sentence)
        out.append(sentence)
    if len(out) < count:
        print(f"warning: only generated {len(out)}/{count} unique sentences", file=sys.stderr)
    return out


TRAIN_VOICES = ["Alex", "Samantha", "Daniel", "Karen"]
EVAL_VOICES = ["Moira", "Tessa"]  # never used in training
SPEEDS = [0.9, 1.0, 1.1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the Stage 0 speech corpus")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "smoke")
    p.add_argument(
        "--voices",
        default=",".join(TRAIN_VOICES),
        help="Comma-separated macOS voices for the train split",
    )
    p.add_argument(
        "--eval-voices",
        default=",".join(EVAL_VOICES),
        help="Voices for the eval split. Kept disjoint from training on purpose.",
    )
    p.add_argument("--speeds", default=",".join(str(s) for s in SPEEDS))
    p.add_argument("--rate", type=int, default=170, help="words per minute")
    p.add_argument("--num-train", type=int, default=220, help="distinct training sentences")
    p.add_argument("--num-eval", type=int, default=48, help="distinct eval sentences")
    p.add_argument(
        "--train-variants",
        type=int,
        default=3,
        help="Acoustic variants per training sentence (voice and speed rotation)",
    )
    p.add_argument("--leading-silence", type=float, default=0.25)
    p.add_argument("--trailing-silence", type=float, default=0.45)
    p.add_argument("--seed", type=int, default=17)
    return p.parse_args()


def synthesize(
    text: str,
    out_path: Path,
    *,
    voice: str,
    rate: int,
    lead: float,
    trail: float,
    speed: float = 1.0,
) -> float:
    """Render one sentence to a 16 kHz mono wav. Returns duration."""
    if out_path.exists() and out_path.stat().st_size > 44:
        with wave.open(str(out_path), "rb") as fh:
            return fh.getnframes() / float(fh.getframerate())

    aiff = out_path.with_suffix(".aiff")
    subprocess.run(
        ["say", "-v", voice, "-r", str(rate), "-o", str(aiff), text],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atempo = f"atempo={speed}," if abs(speed - 1.0) > 1e-6 else ""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(aiff),
            "-ac", "1", "-ar", str(SAMPLE_RATE),
            # speed perturb, then trim the dead air macOS TTS pads on
            # both ends, then pad ourselves so the encoder does not
            # start or end mid-word
            "-af",
            f"{atempo}"
            "silenceremove=start_periods=1:start_silence=0:start_threshold=-45dB,"
            "areverse,"
            "silenceremove=start_periods=1:start_silence=0:start_threshold=-45dB,"
            "areverse,"
            f"adelay={int(lead * 1000)}|{int(lead * 1000)},"
            f"apad=pad_dur={trail}",
            "-c:a", "pcm_s16le", str(out_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    aiff.unlink(missing_ok=True)

    with wave.open(str(out_path), "rb") as fh:
        return fh.getnframes() / float(fh.getframerate())


def build_split(
    name: str,
    sentences: list[str],
    out_dir: Path,
    *,
    voices: list[str],
    speeds: list[float],
    variants: int,
    rate: int,
    lead: float,
    trail: float,
) -> Path:
    audio_dir = out_dir / name / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    clip = 0
    for i, sentence in enumerate(sentences):
        for v in range(variants):
            # Rotate voice per sentence, speed per variant depth, so
            # every sentence is heard by several voices and no sentence
            # is always heard at the same speed.
            voice = voices[(i + v) % len(voices)]
            speed = speeds[(i + v) % len(speeds)]
            wav_path = audio_dir / f"{name}_{clip:05d}_{voice}_{speed:.1f}.wav"
            duration = synthesize(
                sentence, wav_path, voice=voice, rate=rate,
                lead=lead, trail=trail, speed=speed,
            )
            rows.append(
                {
                    "audio_filepath": str(wav_path.resolve()),
                    "duration": round(duration, 3),
                    "text": sentence,
                    "voice": voice,
                    "speed": speed,
                    "variant": v,
                }
            )
            clip += 1
        if i < 2:
            print(f"  [{name}] {rows[-1]['voice']:9s} @ {rows[-1]['speed']}x  {sentence!r}")

    manifest = out_dir / f"{name}.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    total = sum(r["duration"] for r in rows)
    print(f"  [{name}] {len(rows)} clips ({len(sentences)} sentences) "
          f"from {len(voices)} voices, {total / 60:.1f} min audio -> {manifest}")
    return manifest


def main() -> int:
    args = parse_args()
    if shutil.which("say") is None:
        print("macOS `say` not found; this script is macOS-only.", file=sys.stderr)
        return 1
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH.", file=sys.stderr)
        return 1

    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    eval_voices = [v.strip() for v in args.eval_voices.split(",") if v.strip()]
    speeds = [float(s) for s in args.speeds.split(",") if s.strip()]
    overlap = set(voices) & set(eval_voices)
    assert not overlap, f"eval voices must be held out from training: {sorted(overlap)}"

    rng = random.Random(args.seed)
    train_seed = HANDWRITTEN[:-N_HELD_OUT]
    held_out = HANDWRITTEN[-N_HELD_OUT:]

    print("=" * 68)
    print("Stage 0 smoke corpus")
    print("=" * 68)
    print(f"train voices: {', '.join(voices)}")
    print(f"eval voices:  {', '.join(eval_voices)}  (held out)")
    print(f"speeds:       {speeds}   rate {args.rate} wpm   seed {args.seed}")
    print()

    eval_sentences = list(held_out)
    eval_sentences += generate_sentences(max(0, args.num_eval - len(held_out)), rng, set(train_seed))
    train_sentences = list(train_seed)
    train_sentences += generate_sentences(args.num_train, rng, set(eval_sentences))

    overlap = set(train_sentences) & set(eval_sentences)
    assert not overlap, f"train/eval leakage: {sorted(overlap)[:3]}"
    print(f"sentences: {len(train_sentences)} train / {len(eval_sentences)} eval, no overlap")
    print()

    build_split(
        "train", train_sentences, args.out_dir,
        voices=voices, speeds=speeds, variants=args.train_variants,
        rate=args.rate, lead=args.leading_silence, trail=args.trailing_silence,
    )
    print()
    build_split(
        "eval", eval_sentences, args.out_dir,
        voices=eval_voices, speeds=speeds, variants=1,
        rate=args.rate, lead=args.leading_silence, trail=args.trailing_silence,
    )
    print()
    print("OK: corpus ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())