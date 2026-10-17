#!/usr/bin/env python3
"""Extract the FastConformer encoder from parakeet-tdt-0.6b-v2.

Licence of the source checkpoint
--------------------------------
nvidia/parakeet-tdt-0.6b-v2 is distributed under **CC-BY-4.0**
(https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2). The extracted
weights in the output directory are a derivative of that checkpoint
and carry the same CC-BY-4.0 terms. Bim-ASR is Apache-2.0; the encoder
side is not. Any release that ships these weights must reproduce the
CC-BY-4.0 attribution:

    Parakeet TDT 0.6B v2 -- NVIDIA Corporation, CC-BY-4.0

What this script does
---------------------
`parakeet-tdt-0.6b-v2.nemo` is a tar archive holding a Lightning
checkpoint (`model_weights.ckpt`), the model config
(`model_config.yaml`) and the SentencePiece tokenizer. Bim-ASR uses the
**encoder only** -- the TDT decoder and joint network are dropped.

This script:

1. opens the .nemo tar and extracts the config and the weights,
2. converts the NeMo ConformerEncoder config to
   `transformers.ParakeetEncoderConfig`,
3. keeps only `encoder.*` weights and renames them to the
   transformers Parakeet layout,
4. loads them into a `ParakeetEncoder` with `strict=True`, so a single
   missing or unexpected tensor fails the run rather than silently
   producing a half-random encoder,
5. saves `config.json` + `model.safetensors` (the safetensors layout
   that `mlx_lm.convert` and the MLX bridge consume),
6. saves the 128-bin log-mel feature extractor so the audio front end
   matches NeMo's exactly,
7. writes `bim_asr_encoder.json` with the derived numbers the rest of
   the codebase depends on.

Why transformers and not the NeMo toolkit
-----------------------------------------
The NeMo toolkit has no supported macOS/Apple-Silicon install, and we
need this to run on the laptop (no CUDA). `transformers>=5` ships a
native `ParakeetEncoder` implementation and the official NeMo -> HF
weight mapping, so this path needs nothing but torch.

Usage:
    .venv/bin/python scripts/prepare_parakeet_encoder.py \
        --nemo artifacts/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo \
        --out artifacts/parakeet-encoder-v2

    # optional: also convert the full TDT model, so the encoder can be
    # validated end to end by decoding real audio with the original ASR head
    .venv/bin/python scripts/prepare_parakeet_encoder.py \
        --nemo artifacts/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo \
        --out artifacts/parakeet-encoder-v2 \
        --also-tdt artifacts/parakeet-tdt-hf
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tarfile
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CHECKPOINT_ID = "nvidia/parakeet-tdt-0.6b-v2"
CHECKPOINT_LICENCE = "CC-BY-4.0"

# Mirrors transformers' src/transformers/models/parakeet/convert_nemo_to_hf.py.
# Kept local (rather than imported) because that file is a conversion
# script, not part of the transformers package API.
NEMO_TO_HF_WEIGHT_MAPPING: dict[str, str] = {
    r"encoder\.pre_encode\.conv\.": r"encoder.subsampling.layers.",
    r"encoder\.pre_encode\.out\.": r"encoder.subsampling.linear.",
    r"encoder\.pos_enc\.": r"encoder.encode_positions.",
    r"encoder\.layers\.(\d+)\.conv\.batch_norm\.": r"encoder.layers.\1.conv.norm.",
    r"decoder\.decoder_layers\.0\.(weight|bias)": r"ctc_head.\1",
    r"linear_([kv])": r"\1_proj",
    r"linear_out": r"o_proj",
    r"linear_q": r"q_proj",
    r"pos_bias_([uv])": r"bias_\1",
    r"linear_pos": r"relative_k_proj",
}

NEMO_TDT_WEIGHT_MAPPING: dict[str, str] = {
    r"decoder\.prediction\.embed\.": r"decoder.embedding.",
    r"decoder\.prediction\.dec_rnn\.lstm\.": r"decoder.lstm.",
    r"joint\.enc\.": r"encoder_projector.",
    r"joint\.pred\.": r"decoder.decoder_projector.",
    r"joint\.joint_net\.2\.": r"joint.head.",
}

ENCODER_IGNORED_CFG_KEYS = {
    "att_context_size",
    "causal_downsampling",
    "stochastic_depth_start_layer",
    "feat_out",
    "_target_",
    "ff_expansion_factor",
    "untie_biases",
    "att_context_style",
    "self_attention_model",
    "subsampling",
    "stochastic_depth_mode",
    "conv_context_size",
    "dropout_pre_encoder",
    "reduction",
    "reduction_factor",
    "reduction_position",
    "conv_norm_type",
}

ENCODER_CFG_MAPPING: dict[str, str] = {
    "d_model": "hidden_size",
    "n_heads": "num_attention_heads",
    "n_layers": "num_hidden_layers",
    "feat_in": "num_mel_bins",
    "conv_kernel_size": "conv_kernel_size",
    "subsampling_factor": "subsampling_factor",
    "subsampling_conv_channels": "subsampling_conv_channels",
    "pos_emb_max_len": "max_position_embeddings",
    "dropout": "dropout",
    "dropout_emb": "dropout_positions",
    "dropout_att": "attention_dropout",
    "xscaling": "scale_input",
    "use_bias": "attention_bias",
}

FEATURE_KEYS_TO_IGNORE = {
    "_target_",
    "pad_to",
    "frame_splicing",
    "dither",
    "normalize",
    "window",
    "log",
}
FEATURE_CFG_MAPPING: dict[str, str] = {
    "sample_rate": "sampling_rate",
    "window_size": "win_length",
    "window_stride": "hop_length",
    "window": "window",
    "n_fft": "n_fft",
    "log": "log",
    "features": "feature_size",
    "dither": "dither",
    "pad_to": "pad_to",
    "pad_value": "padding_value",
    "frame_splicing": "frame_splicing",
    "preemphasis": "preemphasis",
    "hop_length": "hop_length",
}


def convert_key(key: str, mapping: dict[str, str]) -> str:
    for pattern, replacement in mapping.items():
        key = re.sub(pattern, replacement, key)
    return key


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract the Parakeet TDT 0.6B v2 encoder")
    p.add_argument("--nemo", type=Path, default=None, help="Path to the .nemo file")
    p.add_argument(
        "--repo",
        default=CHECKPOINT_ID,
        help="HF repo to pull the .nemo from if --nemo is not given",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "parakeet-encoder-v2",
        help="Output directory for the encoder",
    )
    p.add_argument(
        "--also-tdt",
        type=Path,
        default=None,
        help="Also write the full TDT model here, for validating the encoder "
        "by decoding with the original ASR head",
    )
    p.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        help="Output dtype (default float32: the encoder is frozen, and fp32 "
        "keeps it exact for the projection experiments)",
    )
    p.add_argument(
        "--print-keys",
        action="store_true",
        help="Print the first/last converted tensor names and exit paths only",
    )
    return p.parse_args()


def resolve_nemo_path(explicit: Path | None, repo: str) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"no .nemo file at {explicit}")
        return explicit

    from huggingface_hub import hf_hub_download

    print(f"downloading {repo} ({CHECKPOINT_LICENCE}) from the Hub...")
    return Path(
        hf_hub_download(
            repo_id=repo,
            filename=f"{repo.split('/')[-1]}.nemo",
        )
    )


def read_nemo(nemo_path: Path) -> tuple[dict, dict[str, str]]:
    """Return (nemo_config, files) without unpacking to disk permanently."""
    import yaml

    with tarfile.open(nemo_path, "r", encoding="utf-8") as tar:
        names = tar.getnames()
        cfg_name = next((n for n in names if n.endswith("model_config.yaml")), None)
        if cfg_name is None:
            raise RuntimeError("model_config.yaml not found inside the .nemo archive")
        files: dict[str, str] = {"model_config": cfg_name}

        for member in tar.getmembers():
            if member.name.endswith(("model_weights.ckpt", "model_weights.pt")):
                files["model_weights"] = member.name
            elif member.name.endswith("_tokenizer.model"):
                files["tokenizer_model_file"] = member.name

        if "model_weights" not in files:
            raise RuntimeError("model_weights.ckpt not found inside the .nemo archive")

        cfg_raw = tar.extractfile(cfg_name).read().decode("utf-8")
        nemo_config = yaml.load(cfg_raw, Loader=yaml.FullLoader)

        tmpdir = Path(tempfile.mkdtemp(prefix="bim_asr_nemo_"))
        weights_name = os.path.basename(files["model_weights"])
        extracted = tar.extractfile(files["model_weights"])
        assert extracted is not None
        with (tmpdir / weights_name).open("wb") as fh:
            while chunk := extracted.read(1024 * 1024):
                fh.write(chunk)
        files["model_weights"] = str(tmpdir / weights_name)

        if "tokenizer_model_file" in files:
            tok_name = os.path.basename(files["tokenizer_model_file"])
            tf = tar.extractfile(files["tokenizer_model_file"])
            assert tf is not None
            (tmpdir / tok_name).write_bytes(tf.read())
            files["tokenizer_model_file"] = str(tmpdir / tok_name)

    return nemo_config, files


def build_encoder_config(nemo_config: dict):
    from transformers import ParakeetEncoderConfig

    enc = nemo_config["encoder"]
    # NeMo's default (no stochastic depth) differs from the HF default.
    cfg_kwargs: dict = {"layerdrop": float(enc.get("stochastic_depth_drop_prob", 0.0))}

    for key, value in enc.items():
        if key in ENCODER_IGNORED_CFG_KEYS or key not in ENCODER_CFG_MAPPING:
            continue
        cfg_kwargs[ENCODER_CFG_MAPPING[key]] = value
        if key == "use_bias":
            cfg_kwargs["convolution_bias"] = value

    # NeMo derives the FFN width from d_model * ff_expansion_factor.
    # transformers takes it as an explicit field.
    cfg_kwargs["intermediate_size"] = int(enc["d_model"] * enc.get("ff_expansion_factor", 4))
    cfg_kwargs["num_key_value_heads"] = enc["n_heads"]

    return ParakeetEncoderConfig(**cfg_kwargs)


def build_feature_extractor(nemo_config: dict):
    from transformers import ParakeetFeatureExtractor

    pre = nemo_config["preprocessor"]
    kwargs: dict = {}
    for key, value in pre.items():
        if key in FEATURE_KEYS_TO_IGNORE or key not in FEATURE_CFG_MAPPING:
            continue
        if key in {"window_size", "window_stride"}:
            value = int(value * pre["sample_rate"])
        kwargs[FEATURE_CFG_MAPPING[key]] = value
    kwargs.setdefault("preemphasis", 0.97)
    kwargs.setdefault("return_attention_mask", True)
    return ParakeetFeatureExtractor(**kwargs)


def convert_state_dict(
    nemo_config: dict, weights_path: str, *, tdt: bool
) -> dict[str, torch.Tensor]:
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    mapping = {**NEMO_TO_HF_WEIGHT_MAPPING, **NEMO_TDT_WEIGHT_MAPPING} if tdt else NEMO_TO_HF_WEIGHT_MAPPING
    converted: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.endswith("featurizer.window") or key.endswith("featurizer.fb"):
            continue
        if not tdt and not key.startswith("encoder."):
            continue
        converted[convert_key(key, mapping)] = value
    return converted


def save_encoder(nemo_config: dict, files: dict, out_dir: Path, dtype: str) -> dict:
    from transformers import ParakeetEncoder

    enc_cfg = build_encoder_config(nemo_config)
    print("encoder config:")
    for key in (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "num_mel_bins",
        "subsampling_factor",
        "conv_kernel_size",
    ):
        print(f"    {key:26s} {getattr(enc_cfg, key)}")

    converted = convert_state_dict(nemo_config, files["model_weights"], tdt=False)
    print(f"\nconverted encoder tensors: {len(converted)}")
    names = sorted(converted)
    print(f"    first: {names[0]}")
    print(f"    last:  {names[-1]}")

    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]

    # strict=True is the whole point: it proves every NeMo encoder
    # tensor landed on a real transformers parameter, and that there
    # are no parameters left unfilled by random init.
    with torch.device("meta"):
        model = ParakeetEncoder(enc_cfg)
    # ParakeetEncoder's own parameter names are unprefixed
    # (subsampling.layers...), the converter emits them under
    # `encoder.`, so strip that prefix for the bare encoder.
    encoder_sd = {
        (k[len("encoder.") :] if k.startswith("encoder.") else k): v.to(torch_dtype)
        for k, v in converted.items()
    }
    missing, unexpected = model.load_state_dict(encoder_sd, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(
            f"encoder state dict did not load cleanly.\n"
            f"  missing   ({len(missing)}): {missing[:8]}\n"
            f"  unexpected({len(unexpected)}): {unexpected[:8]}"
        )
    print(f"    loaded strict-clean: 0 missing, 0 unexpected")

    del model.config._name_or_path
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    print(f"\nwrote encoder to {out_dir}")

    feature_extractor = build_feature_extractor(nemo_config)
    feature_extractor.save_pretrained(out_dir)
    print(f"wrote feature extractor ({enc_cfg.num_mel_bins}-bin log-mel)")

    n_params = sum(p.numel() for p in model.parameters())
    meta = {
        "source_checkpoint": CHECKPOINT_ID,
        "source_licence": CHECKPOINT_LICENCE,
        "derived_weights_licence": CHECKPOINT_LICENCE,
        "attribution": "Parakeet TDT 0.6B v2 -- NVIDIA Corporation, CC-BY-4.0",
        "component": "FastConformer encoder only (TDT decoder and joint dropped)",
        "encoder_hidden": enc_cfg.hidden_size,
        "encoder_layers": enc_cfg.num_hidden_layers,
        "encoder_attention_heads": enc_cfg.num_attention_heads,
        "encoder_ffn_hidden": enc_cfg.intermediate_size,
        "encoder_parameters": n_params,
        "num_mel_bins": enc_cfg.num_mel_bins,
        "sample_rate": feature_extractor.sampling_rate,
        "hop_length": feature_extractor.hop_length,
        "win_length": feature_extractor.win_length,
        "n_fft": feature_extractor.n_fft,
        "preemphasis": feature_extractor.preemphasis,
        "subsampling_factor": enc_cfg.subsampling_factor,
        "dtype": dtype,
    }
    frames_per_second = meta["sample_rate"] / (meta["hop_length"] * meta["subsampling_factor"])
    meta["frame_rate_hz"] = frames_per_second
    meta["frame_period_ms"] = round(1000.0 / frames_per_second, 2)
    meta["max_frames_30s"] = int(frames_per_second * 30)

    with (out_dir / "bim_asr_encoder.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote bim_asr_encoder.json")
    print(f"    frame rate: {frames_per_second:.2f} Hz "
          f"({meta['frame_period_ms']} ms/frame, {meta['max_frames_30s']} frames per 30 s)")
    print(f"    encoder hidden: {enc_cfg.hidden_size}  params: {n_params:,}")
    return meta


def build_tdt_processor(nemo_config: dict, files: dict, vocab_size: int, pad_token_id: int):
    """Rebuild the SentencePiece tokenizer and log-mel feature extractor.

    Parakeet TDT 0.6B v2's label list contains <unk> but no <pad>, so
    NeMo's blank (id == len(labels)) is also the padding id. The HF
    tokenizer has to be padded by exactly one token to keep every id
    aligned with the model head, which is what happens below.
    """
    from tokenizers import AddedToken

    from transformers import ParakeetProcessor, ParakeetTokenizer
    from transformers.convert_slow_tokenizer import ParakeetConverter

    converted = ParakeetConverter(files["tokenizer_model_file"]).converted()
    tokenizer = ParakeetTokenizer(tokenizer_object=converted, clean_up_tokenization_spaces=False)

    if tokenizer.convert_tokens_to_ids("<unk>") is None:
        tokenizer.add_tokens([AddedToken("<unk>", normalized=False, special=True)])
    if tokenizer.convert_tokens_to_ids("<pad>") is None:
        tokenizer.add_tokens([AddedToken("<pad>", normalized=False, special=True)])
        print(f"    added <pad> at id {tokenizer.convert_tokens_to_ids('<pad>')} "
              f"(NeMo v2 has no pad token; blank == pad == {pad_token_id})")
    tokenizer.add_special_tokens(
        {
            "pad_token": AddedToken("<pad>", normalized=False, special=True),
            "unk_token": AddedToken("<unk>", normalized=False, special=True),
        }
    )

    feature_extractor = build_feature_extractor(nemo_config)
    n_tokens = len(tokenizer)
    if n_tokens != vocab_size:
        raise RuntimeError(
            f"tokenizer has {n_tokens} tokens but the model head is {vocab_size} wide; "
            "ids would be misaligned"
        )
    return ParakeetProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        decoder_type="tdt",
    )


def save_tdt(nemo_config: dict, files: dict, out_dir: Path, dtype: str) -> None:
    """Convert the *full* TDT model too.

    This is not needed for Bim-ASR (we drop the decoder), but it is the
    only way to prove the extracted encoder is numerically correct
    without NeMo installed: if the original ASR head decodes sensible
    text from our encoder's features, the encoder is right.
    """
    from transformers import ParakeetForTDT, ParakeetTDTConfig

    enc_cfg = build_encoder_config(nemo_config)
    decoder_cfg = nemo_config["decoder"]
    prednet = decoder_cfg.get("prednet", {})
    labels = nemo_config["labels"]
    durations = nemo_config.get("decoding", {}).get("durations", [0, 1, 2, 3, 4])

    blank_token_id = len(labels)
    pad_token_id = labels.index("<pad>") if "<pad>" in labels else blank_token_id

    cfg = ParakeetTDTConfig(
        vocab_size=len(labels) + 1,
        decoder_hidden_size=int(prednet.get("pred_hidden", 640)),
        num_decoder_layers=int(prednet.get("pred_rnn_layers", 2)),
        durations=durations,
        hidden_act="relu",
        max_symbols_per_step=10,
        encoder_config=enc_cfg.to_dict(),
        pad_token_id=pad_token_id,
        blank_token_id=blank_token_id,
    )

    processor = build_tdt_processor(nemo_config, files, cfg.vocab_size, pad_token_id)

    converted = convert_state_dict(nemo_config, files["model_weights"], tdt=True)
    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    converted = {k: v.to(torch_dtype) for k, v in converted.items()}

    with torch.device("meta"):
        model = ParakeetForTDT(cfg)
    missing, unexpected = model.load_state_dict(converted, strict=False, assign=True)
    print(f"\nTDT model: {len(missing)} missing, {len(unexpected)} unexpected")
    if missing or unexpected:
        print(f"    missing:    {list(missing)[:8]}")
        print(f"    unexpected: {list(unexpected)[:8]}")
        raise RuntimeError("TDT conversion did not load cleanly")

    del model.config._name_or_path
    model.generation_config.decoder_start_token_id = model.config.blank_token_id
    model.generation_config.suppress_tokens = list(
        range(model.config.vocab_size, model.config.vocab_size + len(model.config.durations))
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    print(f"wrote full TDT model + processor to {out_dir}")


def main() -> int:
    args = parse_args()
    print("=" * 64)
    print("Bim-ASR :: Parakeet encoder extraction")
    print("=" * 64)
    print(f"source licence: {CHECKPOINT_LICENCE} ({CHECKPOINT_ID})")
    print()

    nemo_path = resolve_nemo_path(args.nemo, args.repo)
    print(f"nemo: {nemo_path}  ({nemo_path.stat().st_size / 1e9:.2f} GB)")

    nemo_config, files = read_nemo(nemo_path)
    enc = nemo_config["encoder"]
    print(f"\nNeMo encoder: d_model={enc['d_model']} layers={enc['n_layers']} "
          f"heads={enc['n_heads']} subsampling={enc['subsampling_factor']} "
          f"feat_in={enc['feat_in']}")

    if args.print_keys:
        converted = convert_state_dict(nemo_config, files["model_weights"], tdt=False)
        for key in sorted(converted)[:12]:
            print(f"    {key:70s} {tuple(converted[key].shape)}")
        print(f"    ... {len(converted)} tensors total")
        return 0

    meta = save_encoder(nemo_config, files, args.out, args.dtype)
    print()

    if args.also_tdt is not None:
        save_tdt(nemo_config, files, args.also_tdt, args.dtype)
        print()

    print("OK: encoder ready.")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())