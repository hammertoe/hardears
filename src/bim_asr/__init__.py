"""Bim-ASR: a Caribbean-aware speech recognition stack.

The architecture is a Speech-Augmented Language Model (SALM) that
fuses a frozen FastConformer acoustic encoder (Parakeet TDT 0.6B v2,
1024-dim frames at 12.5 Hz) with a frozen Qwen3.5-4B language model
through a small linear projection plus a LoRA adapter on the LLM. Only
the projection and the LoRA are trained.

See docs/ARCHITECTURE.md for the design rationale,
docs/STAGE0_RESULTS.md for what Stage 0 measured, and
docs/TRAINING.md for the staged training plan.
"""

from __future__ import annotations

__version__ = "0.0.1"

__all__ = ["__version__"]
