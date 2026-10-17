"""Smoke tests for Bim-ASR. These do not load the heavy checkpoints;
they only check that the configuration and the projection layer
behave as expected. Run with pytest from the project root.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bim_asr.model import (  # noqa: E402
    AudioProjection,
    BimAsrConfig,
    BimAsrModel,
    count_trainable_parameters_summary,
)


def test_config_defaults() -> None:
    cfg = BimAsrConfig()
    assert cfg.encoder_id == "nvidia/parakeet-tdt-0.6b-v2"
    assert cfg.llm_id == "Qwen/Qwen3.5-4B"
    # Parakeet TDT 0.6B v2 has d_model = 1024 (verified against the
    # model_config.yaml inside the .nemo and the extracted weights).
    # The scaffold originally asserted 512 here, which was wrong.
    assert cfg.encoder_hidden == 1024
    assert cfg.llm_hidden == 2560
    assert cfg.lora_rank == 32
    assert cfg.lora_alpha == 64
    assert cfg.lora_target_modules == ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def test_2b_smoke_config_is_2048_wide() -> None:
    cfg = BimAsrConfig.for_2b_smoke()
    assert cfg.llm_id == "Qwen/Qwen3.5-2B"
    # Qwen3.5-2B is 2048 wide, not 2560 like the 4B.
    assert cfg.llm_hidden == 2048
    assert cfg.encoder_hidden == 1024


def test_config_immutable() -> None:
    cfg = BimAsrConfig()
    with pytest.raises((AttributeError, Exception)):
        cfg.lora_rank = 16  # type: ignore[misc]


def test_projection_shape() -> None:
    proj = AudioProjection()
    x = torch.randn(2, 100, 1024)
    y = proj(x)
    assert y.shape == (2, 100, 2560)


def test_projection_learnable() -> None:
    proj = AudioProjection(512, 2560)
    n = sum(p.numel() for p in proj.parameters() if p.requires_grad)
    assert n > 0
    assert n == 512 * 2560 + 512 + 512  # 512 weight + 512 bias in LayerNorm


def test_default_projection_matches_real_encoder_width() -> None:
    """The default bridge must be sized for the real Parakeet encoder."""
    proj = AudioProjection()
    assert proj.proj.in_features == 1024
    assert proj.proj.out_features == 2560


def test_projection_initial_scale() -> None:
    proj = AudioProjection()
    assert proj.proj.weight.std().item() < 0.05


def test_model_constructs_with_projection_only() -> None:
    cfg = BimAsrConfig()
    model = BimAsrModel(cfg)
    assert model.encoder is None
    assert model.llm is None
    assert isinstance(model.projection, AudioProjection)


def test_freeze_for_projection_only_training() -> None:
    cfg = BimAsrConfig()
    model = BimAsrModel(cfg)
    model.freeze_for_projection_only_training()
    n_trainable = model.trainable_parameter_count()
    n_total = model.total_parameter_count()
    assert n_trainable == n_total


def test_param_summary() -> None:
    cfg = BimAsrConfig()
    model = BimAsrModel(cfg)
    s = count_trainable_parameters_summary(model)
    assert "trainable=" in s
    assert "%" in s
