"""Loading the real Parakeet TDT encoder on Apple Silicon.

The scaffold assumed `nemo.collections.asr.models.ASRModel.from_pretrained`.
The NeMo toolkit has no supported macOS install and we need this to run
on the laptop, so the encoder comes from the transformers conversion of
the same checkpoint instead (see scripts/prepare_parakeet_encoder.py,
which copies the official NeMo -> HF weight mapping).

Output contract, identical to what the NeMo encoder returned:

    waveform (B, samples) @ 16 kHz  ->  features (B, frames, 1024)

with 12.5 frames per second of audio (8x subsampling of a 100 fps log-mel
front end), i.e. 80 ms per frame.

Licence: the encoder weights are derived from nvidia/parakeet-tdt-0.6b-v2
and are CC-BY-4.0, not Apache-2.0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DEFAULT_ENCODER_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "parakeet-encoder-v2"

ENCODER_LICENCE = "CC-BY-4.0"
ENCODER_ATTRIBUTION = "Parakeet TDT 0.6B v2 -- NVIDIA Corporation, CC-BY-4.0"


@dataclass(frozen=True)
class ParakeetEncoderInfo:
    """The numbers docs/ARCHITECTURE.md and the projection depend on."""

    encoder_hidden: int
    encoder_layers: int
    num_mel_bins: int
    sample_rate: int
    hop_length: int
    subsampling_factor: int
    frame_rate_hz: float
    licence: str

    @classmethod
    def load(cls, encoder_dir: Path | str) -> "ParakeetEncoderInfo":
        meta = json.loads((Path(encoder_dir) / "bim_asr_encoder.json").read_text())
        return cls(
            encoder_hidden=meta["encoder_hidden"],
            encoder_layers=meta["encoder_layers"],
            num_mel_bins=meta["num_mel_bins"],
            sample_rate=meta["sample_rate"],
            hop_length=meta["hop_length"],
            subsampling_factor=meta["subsampling_factor"],
            frame_rate_hz=meta["frame_rate_hz"],
            licence=meta["derived_weights_licence"],
        )


class ParakeetAudioEncoder(torch.nn.Module):
    """Frozen FastConformer encoder with a NeMo-compatible call signature.

    `forward` takes a raw 16 kHz mono waveform (B, samples) or (samples,)
    and returns (features, lengths) exactly like the NeMo encoder, so the
    smoke test and the training loop can treat them as interchangeable.
    """

    def __init__(self, encoder_dir: Path | str = DEFAULT_ENCODER_DIR, *, device: str = "cpu"):
        super().__init__()
        from transformers import ParakeetEncoder, ParakeetFeatureExtractor

        encoder_dir = Path(encoder_dir)
        if not (encoder_dir / "model.safetensors").exists():
            raise FileNotFoundError(
                f"no extracted encoder at {encoder_dir}. Run:\n"
                "  .venv/bin/python scripts/prepare_parakeet_encoder.py"
            )

        self.info = ParakeetEncoderInfo.load(encoder_dir)
        self.feature_extractor = ParakeetFeatureExtractor.from_pretrained(str(encoder_dir))
        self.encoder = ParakeetEncoder.from_pretrained(str(encoder_dir))
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.device = torch.device(device)
        self.encoder.to(self.device)

    def expected_frames(self, num_samples: int) -> int:
        return num_samples // (self.info.hop_length * self.info.subsampling_factor)

    def encode_features(self, mel: torch.Tensor, mel_lengths: torch.Tensor | None = None):
        """Run the encoder on log-mel features. (B, T, 1024), (B,).

        `mel` is (batch, mel_frames, n_mel_bins) and `mel_lengths` is
        either (batch, mel_frames) as a mask, or already-reduced
        (batch,) lengths. The encoder wants a 2-D mask over mel frames.
        """
        mask = None
        lengths = None
        if mel_lengths is not None:
            lengths = mel_lengths.sum(-1) if mel_lengths.dim() > 1 else mel_lengths
            lengths = lengths.to(mel.device, dtype=torch.long)
            mask = (
                torch.arange(mel.shape[1], device=mel.device)[None, :]
                < lengths[:, None]
            ).long()

        out = self.encoder(input_features=mel, attention_mask=mask)
        feats = out.last_hidden_state
        if lengths is not None:
            lens = torch.clamp(lengths // self.info.subsampling_factor, max=feats.shape[1])
        else:
            lens = torch.full((feats.shape[0],), feats.shape[1], dtype=torch.long, device=feats.device)
        return feats, lens

    @torch.no_grad()
    def forward(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor | None = None):
        """waveforms: (B, samples) or (samples,) float32 @ 16 kHz.

        Returns (features (B, frames, 1024), lengths (B,)).
        """
        if waveforms.dim() == 1:
            waveforms = waveforms.unsqueeze(0)
        waveforms = waveforms.to(self.device, dtype=torch.float32)

        inputs = self.feature_extractor(
            waveforms.squeeze(0).cpu().numpy(),
            sampling_rate=self.info.sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        mel = inputs["input_features"].to(self.device, dtype=torch.float32)
        mel_lengths = inputs.get("attention_mask")
        if mel_lengths is not None:
            mel_lengths = mel_lengths.to(self.device)

        return self.encode_features(mel, mel_lengths)

    def encode_waveform(self, waveform: np.ndarray) -> torch.Tensor:
        """Convenience: one 16 kHz mono waveform -> (frames, 1024)."""
        wav = torch.as_tensor(np.asarray(waveform, dtype=np.float32))
        feats, _ = self.forward(wav)
        return feats[0]


def load_parakeet_encoder(
    encoder_dir: Path | str = DEFAULT_ENCODER_DIR, *, device: str = "cpu"
) -> ParakeetAudioEncoder:
    """Load the frozen acoustic encoder (the SALM's audio tower)."""
    return ParakeetAudioEncoder(encoder_dir, device=device)