"""Tests for the Stage 0 SALM wiring.

Covers the pieces added on top of the scaffold:

* the NeMo -> transformers weight mapping used to extract the encoder
  (locked down here because a silent mismatch produces a loadable but
  wrong model),
* the audio splice (text, audio, assistant header) and its dtype
  handling,
* the prompt split around the audio locator,
* the encoder loader's behaviour when the extracted weights are absent.

These use stubs rather than real checkpoints, so they run in
milliseconds. The tests that need the actual 1.9B model live in
scripts/smoke_salm.py, not here.

Run with: .venv/bin/pytest tests/test_salm_wiring.py -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bim_asr.model import (  # noqa: E402
    BimAsrConfig,
    BimAsrModel,
    build_salm_prompt_ids,
)
from bim_asr.parakeet import ParakeetAudioEncoder, ParakeetEncoderInfo  # noqa: E402


def _load_prepare_script():
    """Import scripts/prepare_parakeet_encoder.py by path."""
    path = ROOT / "scripts" / "prepare_parakeet_encoder.py"
    spec = importlib.util.spec_from_file_location("prepare_parakeet_encoder", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def prepare():
    return _load_prepare_script()


# --- NeMo -> HF weight mapping -----------------------------------------


@pytest.mark.parametrize(
    "nemo_key,expected",
    [
        # subsampling: NeMo's two-stage pre_encode -> HF Conv2d stack
        ("encoder.pre_encode.conv.0.weight", "encoder.subsampling.layers.0.weight"),
        ("encoder.pre_encode.conv.2.weight", "encoder.subsampling.layers.2.weight"),
        ("encoder.pre_encode.out.weight", "encoder.subsampling.linear.weight"),
        ("encoder.pre_encode.out.bias", "encoder.subsampling.linear.bias"),
        # positional encoding
        ("encoder.pos_enc.pe", "encoder.encode_positions.pe"),
        # conv module norm: NeMo says batch_norm, HF says norm
        (
            "encoder.layers.7.conv.batch_norm.weight",
            "encoder.layers.7.conv.norm.weight",
        ),
        # attention projections use linear_* names in NeMo
        ("encoder.layers.0.self_attn.linear_q.weight", "encoder.layers.0.self_attn.q_proj.weight"),
        ("encoder.layers.0.self_attn.linear_k.weight", "encoder.layers.0.self_attn.k_proj.weight"),
        ("encoder.layers.0.self_attn.linear_v.weight", "encoder.layers.0.self_attn.v_proj.weight"),
        ("encoder.layers.0.self_attn.linear_out.weight", "encoder.layers.0.self_attn.o_proj.weight"),
        # relative position bias is untied in this checkpoint
        ("encoder.layers.0.self_attn.pos_bias_u", "encoder.layers.0.self_attn.bias_u"),
        ("encoder.layers.0.self_attn.pos_bias_v", "encoder.layers.0.self_attn.bias_v"),
        ("encoder.layers.0.self_attn.linear_pos.weight", "encoder.layers.0.self_attn.relative_k_proj.weight"),
        # FFN is already named the same way in both
        ("encoder.layers.3.feed_forward1.linear1.weight", "encoder.layers.3.feed_forward1.linear1.weight"),
    ],
)
def test_encoder_key_mapping(prepare, nemo_key: str, expected: str) -> None:
    converted = prepare.convert_key(nemo_key, prepare.NEMO_TO_HF_WEIGHT_MAPPING)
    assert converted == expected


def test_encoder_key_mapping_does_not_touch_decoder(prepare) -> None:
    """Decoder/joint keys must survive into the TDT conversion, but the
    encoder-only conversion filters on the `encoder.` prefix."""
    decoder_key = "decoder.prediction.embed.weight"
    assert prepare.convert_key(decoder_key, prepare.NEMO_TO_HF_WEIGHT_MAPPING) == decoder_key
    assert not decoder_key.startswith("encoder.")


def test_encoder_config_mapping_covers_every_v2_key(prepare) -> None:
    """Every encoder key in the v2 checkpoint must be mapped or explicitly
    ignored, or the conversion raises on an unknown key."""
    import yaml

    nemo_yaml = None
    cached = ROOT / "artifacts" / "parakeet-tdt-0.6b-v2" / "parakeet-tdt-0.6b-v2.nemo"
    if not cached.exists():
        pytest.skip("no cached .nemo to read the encoder config from")
    import tarfile

    with tarfile.open(cached, "r", encoding="utf-8") as tar:
        name = next(n for n in tar.getnames() if n.endswith("model_config.yaml"))
        nemo_yaml = yaml.load(tar.extractfile(name).read().decode("utf-8"), Loader=yaml.FullLoader)

    unknown = (
        set(nemo_yaml["encoder"])
        - set(prepare.ENCODER_CFG_MAPPING)
        - prepare.ENCODER_IGNORED_CFG_KEYS
    )
    assert not unknown, f"unhandled encoder config keys: {sorted(unknown)}"


# --- the splice ---------------------------------------------------------


class _StubLlm(nn.Module):
    """Minimal stand-in exposing just the embedding table."""

    def __init__(self, vocab: int, hidden: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden, dtype=dtype)

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding


def _stub_model(encoder_hidden: int = 1024, llm_hidden: int = 2048, dtype=torch.bfloat16):
    model = BimAsrModel(
        BimAsrConfig(encoder_hidden=encoder_hidden, llm_hidden=llm_hidden)
    )
    model.llm = _StubLlm(vocab=256, hidden=llm_hidden, dtype=dtype)
    return model


def test_splice_places_audio_between_text() -> None:
    model = _stub_model()
    prefix = torch.tensor([[1, 2, 3, 4, 5]])
    suffix = torch.tensor([[6, 7]])
    audio = torch.randn(1, 34, 2048)

    out = model.splice_audio_into_prompt(audio, prefix, suffix)
    assert out.shape == (1, 5 + 34 + 2, 2048)

    # The text either side must be exactly the token embeddings, i.e.
    # the audio did not overwrite anything.
    emb = model.llm.get_input_embeddings()
    assert torch.equal(out[:, :5], emb(prefix))
    assert torch.equal(out[:, -2:], emb(suffix))


def test_splice_casts_fp32_audio_to_the_llm_dtype() -> None:
    """The bridge is fp32, the LLM is bf16. The splice has to reconcile
    them or the first Linear raises a dtype error."""
    model = _stub_model(dtype=torch.bfloat16)
    prefix = torch.tensor([[1, 2]])
    audio = torch.randn(1, 10, 2048, dtype=torch.float32)
    out = model.splice_audio_into_prompt(audio, prefix, torch.tensor([[3]]))
    assert out.dtype == torch.bfloat16


def test_splice_with_empty_suffix() -> None:
    model = _stub_model()
    prefix = torch.tensor([[1, 2, 3]])
    audio = torch.randn(1, 7, 2048)
    out = model.splice_audio_into_prompt(audio, prefix, torch.tensor([[]], dtype=torch.long))
    assert out.shape == (1, 3 + 7, 2048)


def test_splice_is_differentiable_into_the_bridge() -> None:
    """Loss on the spliced sequence must reach the projection weights."""
    model = _stub_model()
    features = torch.randn(1, 12, 1024, requires_grad=True)
    audio = model.encode_audio(features)
    out = model.splice_audio_into_prompt(audio, torch.tensor([[1, 2]]), torch.tensor([[3]]))
    out.float().pow(2).mean().backward()

    grad = model.projection.proj.weight.grad
    assert grad is not None and grad.abs().sum() > 0
    assert torch.isfinite(grad).all()


# --- prompt construction ------------------------------------------------


class _StubTokenizer:
    """Whitespace tokenizer, enough to exercise the split logic."""

    eos_token = "<eos>"

    def __init__(self) -> None:
        self.last_text = ""

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        assert tokenize is False
        body = "".join(m["content"] + "\n" for m in messages)
        return body + "assistant\n" if add_generation_prompt else body

    def __call__(self, text, add_special_tokens=False):
        self.last_text = text
        ids = [hash(w) % 200 + 1 for w in text.split()]
        return type("Encoding", (), {"input_ids": ids, "input_tokens": text.split()})()

    def decode(self, ids, **kwargs):
        return " ".join(str(int(i)) for i in ids)


def test_prompt_split_round_trips() -> None:
    tok = _StubTokenizer()
    prefix, suffix = build_salm_prompt_ids(
        tok, user_prompt="Transcribe the following audio.", audio_locator_tag="<|audio|>"
    )
    assert prefix.numel() > 0
    assert suffix.numel() > 0
    # The suffix is what carries the assistant turn header, so the
    # audio lands *before* the model is told to answer.
    assert tok.last_text.split()[0] == "assistant"
    assert "<|audio|>" not in tok.last_text


def test_prompt_split_raises_when_locator_missing() -> None:
    class _NoLocator(_StubTokenizer):
        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            return "system\nnothing here\n"

    with pytest.raises(ValueError, match="locator"):
        build_salm_prompt_ids(
            _NoLocator(), user_prompt="x", audio_locator_tag="<|audio|>"
        )


# --- encoder loader -----------------------------------------------------


def test_encoder_loader_explains_a_missing_checkpoint(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="prepare_parakeet_encoder"):
        ParakeetAudioEncoder(tmp_path)


def test_encoder_metadata_round_trips(tmp_path) -> None:
    meta = {
        "encoder_hidden": 1024,
        "encoder_layers": 24,
        "num_mel_bins": 128,
        "sample_rate": 16000,
        "hop_length": 160,
        "subsampling_factor": 8,
        "frame_rate_hz": 12.5,
        "derived_weights_licence": "CC-BY-4.0",
    }
    (tmp_path / "bim_asr_encoder.json").write_text(json.dumps(meta))
    info = ParakeetEncoderInfo.load(tmp_path)
    assert info.encoder_hidden == 1024
    assert info.frame_rate_hz == 12.5
    assert info.licence == "CC-BY-4.0"


def test_expected_frames_uses_the_true_12_5_hz() -> None:
    """16000 / (160 hop * 8 subsampling) = 12.5 frames per second."""
    hop, subsampling = 160, 8
    two_seconds = 16000 * 2
    assert two_seconds / (hop * subsampling) == 25  # frames
    assert 16000 / (hop * subsampling) == 12.5      # frames per second
    assert two_seconds // (hop * subsampling) == 25