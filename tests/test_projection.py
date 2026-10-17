"""Tests for the AudioProjection bridge.

The bridge (nn.LayerNorm -> nn.Linear) is the only thing in Bim-ASR
that is both newly initialised and trainable at Stage 0. Everything
else is a frozen pretrained checkpoint. If the bridge is wrong the
model still runs, still emits tokens, and still looks plausible on a
loss curve while learning nothing.

These tests pin down the properties the rest of the SALM depends on:

* shapes: it maps (B, frames, encoder_hidden) -> (B, frames, llm_hidden)
  for the real hidden sizes of both supported LLMs.
* gradient flow: a loss on the projected output produces non-zero,
  finite gradients on every bridge parameter.
* frame independence: position i of the output depends only on position
  i of the input, which is what lets us splice audio embeddings into
  the middle of a tokenised prompt without corrupting the text.
* audio sensitivity: the output actually moves when the audio moves,
  and it moves in more than one direction (not a rank-1 collapse).

Run with: .venv/bin/pytest tests/test_projection.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bim_asr.model import AudioProjection, BimAsrConfig  # noqa: E402

# Real hidden sizes, pinned to the published configs:
#   Qwen/Qwen3.5-4B -> text_config.hidden_size = 2560
#   Qwen/Qwen3.5-2B -> text_config.hidden_size = 2048
# Published widths: Qwen3.5-4B = 2560, Qwen3.5-2B = 2048.


def test_bridge_shape_4b() -> None:
    proj = AudioProjection(encoder_hidden=1024, llm_hidden=2560)
    x = torch.randn(2, 375, 1024)  # 30 s of audio at 12.5 frames/s
    assert proj(x).shape == (2, 375, 2560)


def test_bridge_shape_2b() -> None:
    proj = AudioProjection(encoder_hidden=1024, llm_hidden=2048)
    x = torch.randn(1, 32, 1024)
    assert proj(x).shape == (1, 32, 2048)


@pytest.mark.parametrize(
    "factory,hidden",
    [
        (lambda: BimAsrConfig(), 2560),                    # Qwen3.5-4B
        (lambda: BimAsrConfig.for_2b_smoke(), 2048),       # Qwen3.5-2B
    ],
)
def test_config_matches_published_llm_hidden(factory, hidden: int) -> None:
    """The projection must be sized from the LLM's real hidden size.

    Qwen3.5-4B is 2560 wide but Qwen3.5-2B is 2048 wide. Scaffolding
    that assumed both were 2560 produced a bridge that cannot be
    spliced into the 2B model at all.
    """
    assert factory().llm_hidden == hidden


def test_for_llm_reads_hidden_from_a_real_config() -> None:
    """`for_llm` must not trust a hard-coded table."""
    from bim_asr.model import llm_hidden_from_config

    snapshot = ROOT / "artifacts" / "qwen3.5-2b"
    if not (snapshot / "config.json").exists():
        pytest.skip("no local Qwen3.5-2B snapshot; run scripts/prepare_parakeet_encoder.py first")
    assert llm_hidden_from_config(str(snapshot)) == 2048
    assert BimAsrConfig.for_llm(str(snapshot)).llm_hidden == 2048


def test_bridge_parameter_count() -> None:
    proj = AudioProjection(512, 2560)
    n = sum(p.numel() for p in proj.parameters())
    # LayerNorm weight + bias (512 each) + Linear weight (512 x 2560).
    assert n == 512 * 2 + 512 * 2560
    assert all(p.requires_grad for p in proj.parameters())


def test_gradient_flows_to_every_bridge_parameter() -> None:
    """A loss on the projected audio must reach norm.weight, norm.bias
    and proj.weight with non-zero, finite gradients."""
    proj = AudioProjection(512, 2560)
    x = torch.randn(2, 64, 512)

    loss = proj(x).pow(2).mean()
    loss.backward()

    for name, p in proj.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} gradient is not finite"
        assert p.grad.abs().sum().item() > 0, f"{name} gradient is exactly zero"


def test_gradient_reaches_audio_from_a_spliced_prompt_loss() -> None:
    """End-to-end shape of the SALM trick: loss over a sequence that
    mixes text embeddings and projected audio must push gradient back
    into the audio features. This is the property Stage 0 training
    depends on."""
    proj = AudioProjection(512, 2560)
    audio = torch.randn(1, 40, 512, requires_grad=True)
    text = torch.randn(1, 12, 2560)

    spliced = torch.cat([text[:, :4], proj(audio), text[:, 4:]], dim=1)
    assert spliced.shape == (1, 4 + 40 + 8, 2560)
    spliced.pow(2).mean().backward()

    assert audio.grad is not None
    assert torch.isfinite(audio.grad).all()
    assert audio.grad.abs().sum().item() > 0


def test_bridge_is_frame_wise() -> None:
    """Output frame i must depend only on input frame i.

    If this fails, splicing audio into the middle of a prompt would
    leak one frame of audio into another.
    """
    proj = AudioProjection(512, 2560).eval()
    x = torch.randn(1, 20, 512)

    perturbed = x.clone()
    perturbed[0, 7, :] = torch.randn(512)

    y_a = proj(x)
    y_b = proj(perturbed)

    untouched = [i for i in range(20) if i != 7]
    assert torch.allclose(y_a[:, untouched], y_b[:, untouched], atol=1e-5)
    assert not torch.allclose(y_a[:, 7], y_b[:, 7], atol=1e-5)


def test_bridge_depends_on_the_audio() -> None:
    """Different audio in, different embeddings out.

    Guards against the degenerate bridge: one that returns a constant,
    or ignores its input, still 'works' end to end and still produces
    transcripts -- just ones that have nothing to do with the audio.
    """
    torch.manual_seed(0)
    proj = AudioProjection(512, 2560).eval()
    silent = torch.zeros(1, 40, 512)
    speech_like = torch.randn(1, 40, 512) * 0.5

    y_silent = proj(silent)
    y_speech = proj(speech_like)

    assert not torch.allclose(y_silent, y_speech, atol=1e-3)
    # The LayerNorm means silence is not mapped to a special vector;
    # it is mapped by its direction, which is undefined/zero-mean.
    assert torch.isfinite(y_silent).all()

    # Both must sit in a sane range for the LLM embedding space.
    assert y_speech.abs().mean().item() < 5.0


def test_bridge_is_not_rank_one() -> None:
    """The map must span many directions.

    A bridge that collapses every frame onto one direction cannot carry
    phonetic information, and its output would be constant along time.

    The metric is the participation ratio (squared sum of singular
    values over the sum of their squares) of the centred output, which
    is a stable effective rank. A rank-1 collapse scores ~1 no matter
    how wide the output is; the ceiling here is min(n_frames, dim).
    """
    proj = AudioProjection(1024, 2048).eval()
    x = torch.randn(1, 200, 1024)
    y = proj(x)[0]  # (200, 2048)

    centered = y - y.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered.float())
    effective_rank = (singular.sum() ** 2) / (singular**2).sum()

    ceiling = min(y.shape[0], y.shape[1])
    assert effective_rank > 20, (
        f"projected frames collapse onto {effective_rank:.1f} effective "
        f"directions (ceiling {ceiling})"
    )

    # Frames must not all be near-identical.
    pairwise = torch.cdist(y[:20].float(), y[:20].float())
    off_diag = pairwise[~torch.eye(20, dtype=bool)]
    assert off_diag.min().item() > 1e-3


def test_bridge_is_scale_invariant() -> None:
    """The input LayerNorm makes the bridge ignore loudness.

    FastConformer features vary in scale with recording level; the LLM
    embedding space does not want that variation.
    """
    proj = AudioProjection(512, 2560).eval()
    x = torch.randn(1, 16, 512)
    assert torch.allclose(proj(x), proj(x * 7.5), atol=1e-4)


def test_bridge_is_deterministic_in_eval() -> None:
    proj = AudioProjection(512, 2048).eval()
    x = torch.randn(3, 24, 512)
    assert torch.equal(proj(x), proj(x))


def test_bridge_initial_output_is_sane_scale() -> None:
    """The std=0.02 init has to put the audio embeddings at a scale the
    LLM can work with.

    Derivation: the input LayerNorm makes every frame unit-variance
    (1024 dims), then a Linear with weight std 0.02 sums 1024 terms,
    so the output is ~N(0, 0.02^2 * 1024) and has std ~0.64. Anything
    orders of magnitude above that would swamp the text embeddings at
    the first layer; anything near zero would look like silence.
    """
    proj = AudioProjection(1024, 2560).eval()
    y = proj(torch.randn(1, 32, 1024) * 0.5)

    assert torch.isfinite(y).all()
    expected_std = 0.02 * (1024**0.5)
    measured = y.std().item()
    assert 0.25 < measured < 2.0, f"bridge output std {measured:.3f}, expected ~{expected_std:.2f}"


def test_bridge_matches_the_frozen_encoder_width() -> None:
    """Default bridge in = real Parakeet d_model."""
    proj = AudioProjection()
    assert proj.proj.in_features == 1024