"""SALM inference: waveform in, transcript out.

This is the piece `scripts/transcribe.py` calls. It ties together the
three Stage 0 pieces:

    16 kHz waveform
      -> frozen Parakeet FastConformer encoder   (bim_asr.parakeet)
      -> AudioProjection bridge                  (bim_asr.model)
      -> Qwen3.5 + LoRA                          (bim_asr.model)
      -> transcript text

The forward pass is the SALM splice: the prompt is tokenised, split
around the audio locator, the projected audio embeddings go where the
locator was, and the LLM continues from there.

What is and is not done here
----------------------------

Done: single-clip and long-audio chunking, greedy decoding, optional
trained checkpoint.

Not done (Stage 1 work, and the reason the long-audio path is
conservative): crossfade or de-duplication between overlapping
windows, batched decoding, beam search, streaming, timestamps. The
chunker below returns overlapping windows and this module concatenates
them, which double-counts words in the overlap region. For utterances
shorter than the 30 s window -- which is the whole Stage 0 corpus, and
most of what BimOmni's Barbados Parliament clips are -- there is a
single window and no overlap at all, so the caveat does not apply.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .audio import chunk_waveform
from .model import BimAsrConfig, BimAsrModel, build_salm_prompt_ids

SAMPLE_RATE = 16000


@dataclass
class TranscriptResult:
    """One transcription, with the bookkeeping you need to debug it."""

    text: str
    chunks: int
    audio_seconds: float
    seconds: float
    chunk_texts: list[str]

    @property
    def realtime_factor(self) -> float:
        """Audio seconds processed per wall-clock second. > 1 is faster
        than real time."""
        if self.seconds <= 0:
            return float("inf")
        return self.audio_seconds / self.seconds


class SalmTranscriber:
    """Loads the pieces once, then transcribes."""

    def __init__(
        self,
        *,
        llm_path: str | Path,
        encoder_path: str | Path,
        checkpoint: str | Path | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        window_seconds: float = 30.0,
        overlap_seconds: float = 5.0,
        max_new_tokens: int = 96,
    ) -> None:
        self.device = torch.device(device)
        self.config = BimAsrConfig.for_llm(str(llm_path))
        self.window_seconds = window_seconds
        self.overlap_seconds = overlap_seconds
        self.max_new_tokens = max_new_tokens

        self.model = BimAsrModel.from_pretrained(
            self.config, device=device, dtype=dtype
        )
        self.tokenizer = self.model.tokenizer
        if checkpoint:
            self.load_checkpoint(Path(checkpoint))
        self.model.eval()

        from .parakeet import load_parakeet_encoder

        self.encoder = load_parakeet_encoder(encoder_path, device=device)
        self.prefix_ids, self.suffix_ids = build_salm_prompt_ids(
            self.tokenizer,
            user_prompt=self.config.user_prompt,
            audio_locator_tag=self.config.audio_locator_tag,
        )

    def load_checkpoint(self, checkpoint: Path) -> None:
        """Load a trained projection and LoRA adapter (Stage 0 format)."""
        proj_path = checkpoint / "projection.pt"
        if proj_path.exists():
            state = torch.load(proj_path, map_location="cpu", weights_only=True)
            self.model.projection.load_state_dict(state)

        adapter = checkpoint / "lora_adapter"
        if adapter.exists():
            from peft import PeftModel

            self.model.llm = PeftModel.from_pretrained(
                self.model.llm, str(adapter), is_trainable=False
            )
            self.model.llm.to(self.device)

    # -- internals --------------------------------------------------------

    @torch.no_grad()
    def _features(self, waveform: np.ndarray) -> torch.Tensor:
        wav = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))
        feats, _ = self.encoder(wav.unsqueeze(0))
        return feats

    @torch.no_grad()
    def _transcribe_chunk(self, waveform: np.ndarray) -> str:
        feats = self._features(waveform).to(self.device)
        audio_emb = self.model.encode_audio(feats)
        embeds = self.model.splice_audio_into_prompt(
            audio_emb,
            self.prefix_ids.unsqueeze(0).to(self.device),
            self.suffix_ids.unsqueeze(0).to(self.device),
        )
        out = self.model.llm.generate(
            inputs_embeds=embeds,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    # -- public API -------------------------------------------------------

    def transcribe(self, waveform: np.ndarray, *, sample_rate: int = SAMPLE_RATE) -> TranscriptResult:
        """Transcribe a mono waveform. Returns the text plus timings."""
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {sample_rate}")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        audio_seconds = len(waveform) / float(SAMPLE_RATE)
        chunks = chunk_waveform(
            waveform,
            sample_rate=sample_rate,
            window_seconds=self.window_seconds,
            overlap_seconds=self.overlap_seconds,
        )

        started = time.time()
        texts = [self._transcribe_chunk(chunk) for chunk in chunks]
        elapsed = time.time() - started

        return TranscriptResult(
            text=" ".join(t for t in texts if t).strip(),
            chunks=len(chunks),
            audio_seconds=audio_seconds,
            seconds=elapsed,
            chunk_texts=texts,
        )

    def transcribe_file(self, path: str | Path) -> TranscriptResult:
        from .audio import load_audio_16k_mono

        return self.transcribe(load_audio_16k_mono(path))


def load_transcriber(**kwargs) -> SalmTranscriber:
    """Convenience constructor mirroring the CLI defaults."""
    return SalmTranscriber(**kwargs)