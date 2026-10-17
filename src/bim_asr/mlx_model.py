"""MLX mirror of the Bim-ASR model for Apple Silicon inference.

This is the laptop path. It mirrors the PyTorch model definition so
that the same weights (after conversion) can run on the M4 unified
memory without a CUDA GPU.

The MLX conversion of the FastConformer encoder and Qwen3 LLM
happens elsewhere (see scripts/convert_to_mlx.py). This file is
just the bridge and the SALM forward loop in MLX.
"""

from __future__ import annotations

try:
    import mlx.core as mx
    from mlx import nn as mlxnn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore
    mlxnn = None  # type: ignore


if HAS_MLX:

    class MlxAudioProjection(mlxnn.Module):
        """MLX version of the FastConformer -> Qwen3.5 projection bridge.

        Defaults are 1024 -> 2560: Parakeet TDT 0.6B v2's `d_model` is
        1024 (the original scaffold said 512, which would not have
        matched a single real frame), and Qwen3.5-4B is 2560 wide.
        Qwen3.5-2B is 2048, so pass `llm_hidden` explicitly for it.

        Duplicated rather than imported from `bim_asr.model` on
        purpose: that module pulls in torch and this one is meant to
        stay importable in an MLX-only environment.
        `tests/test_projection.py` asserts the two stay in sync.
        """

        def __init__(self, encoder_hidden: int = 1024, llm_hidden: int = 2560) -> None:
            super().__init__()
            self.norm = mlxnn.LayerNorm(encoder_hidden)
            self.proj = mlxnn.Linear(encoder_hidden, llm_hidden, bias=False)

        def __call__(self, audio_features):
            return self.proj(self.norm(audio_features))

else:

    class MlxAudioProjection:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("MLX is not available on this machine")


def is_mlx_available() -> bool:
    """Return True if the MLX backend can be used."""
    return HAS_MLX

