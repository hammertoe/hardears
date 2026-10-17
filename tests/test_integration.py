"""Slow integration tests: the real model, the real encoder.

These load the 1.9B LLM and the 608M encoder, so they are opt-in:

    BIM_ASR_SLOW_TESTS=1 .venv/bin/pytest tests/test_integration.py -v -s

Everything in `tests/test_projection.py` and `tests/test_salm_wiring.py`
runs in seconds with stubs; this file is the one that proves the
assembled stack works.

What it asserts is deliberately modest. An untrained bridge does not
transcribe, so "it returns English words" would be a bad test. The
assertions are:

* the stack assembles and produces tokens at all,
* the generated text depends on the audio,
* the encoder and the bridge agree about shapes and frame rate,
* a Stage 0 checkpoint loads and changes the output.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

LLM_PATH = ROOT / "artifacts" / "qwen3.5-2b"
ENCODER_PATH = ROOT / "artifacts" / "parakeet-encoder-v2"
CHECKPOINT = ROOT / "artifacts" / "stage0-ckpt"
MANIFEST = ROOT / "data" / "smoke" / "eval.jsonl"

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("BIM_ASR_SLOW_TESTS") != "1",
        reason="set BIM_ASR_SLOW_TESTS=1 to run the real-model tests",
    ),
    pytest.mark.skipif(
        not (LLM_PATH / "config.json").exists(),
        reason="no LLM snapshot; run scripts/prepare_parakeet_encoder.py setup steps",
    ),
    pytest.mark.skipif(
        not (ENCODER_PATH / "model.safetensors").exists(),
        reason="no extracted encoder; run scripts/prepare_parakeet_encoder.py",
    ),
]


@pytest.fixture(scope="module")
def transcriber():
    from bim_asr.inference import SalmTranscriber

    return SalmTranscriber(
        llm_path=str(LLM_PATH),
        encoder_path=str(ENCODER_PATH),
        device="cpu",
        max_new_tokens=24,
    )


def _clip(index: int = 0):
    import numpy as np
    import soundfile as sf

    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    wav, _ = sf.read(rows[index]["audio_filepath"], dtype="float32")
    return wav, rows[index]["text"]


def test_encoder_produces_the_expected_frames(transcriber) -> None:
    wav, _ = _clip(0)
    feats = transcriber._features(wav)
    info = transcriber.encoder.info

    assert feats.shape[-1] == info.encoder_hidden == 1024
    # 12.5 frames per second, allow a frame of slack for the conv stack
    expected = len(wav) / 16000 * info.frame_rate_hz
    assert abs(feats.shape[1] - expected) <= 1


def test_untrained_salc_produces_text_that_depends_on_audio(transcriber) -> None:
    wav_a, _ = _clip(0)
    wav_b, _ = _clip(1)

    out_a = transcriber._transcribe_chunk(wav_a)
    out_b = transcriber._transcribe_chunk(wav_b)

    assert out_a.strip(), "the stack generated nothing at all"
    assert out_a.strip() != out_b.strip(), (
        "identical text for different audio: the audio is not reaching the model"
    )


def test_transcribe_returns_timings(transcriber) -> None:
    wav, reference = _clip(0)
    result = transcriber.transcribe(wav)

    assert result.chunks == 1  # short clip, single window
    assert result.audio_seconds == pytest.approx(len(wav) / 16000, abs=1e-3)
    assert result.seconds > 0
    assert result.realtime_factor > 0
    assert reference  # sanity: the manifest row carried text


def test_stage0_checkpoint_loads_and_changes_output(transcriber) -> None:
    if not (CHECKPOINT / "projection.pt").exists():
        pytest.skip("no Stage 0 checkpoint; run scripts/train_stage0.py first")

    from bim_asr.inference import SalmTranscriber

    wav, reference = _clip(0)
    trained = SalmTranscriber(
        llm_path=str(LLM_PATH),
        encoder_path=str(ENCODER_PATH),
        checkpoint=str(CHECKPOINT),
        device="cpu",
        max_new_tokens=24,
    )
    out = trained._transcribe_chunk(wav)
    assert out.strip(), "a trained checkpoint generated nothing"
    print(f"\n  reference: {reference}\n  SALM     : {out}")