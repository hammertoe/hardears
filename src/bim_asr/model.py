"""PyTorch model definition for Bim-ASR.

This is the reference implementation that mirrors the SALM recipe used
by nvidia/canary-qwen-2.5b (the current top of the Open ASR
Leaderboard). We swap in a Qwen3.5-4B-Instruct base and use the
encoder from nvidia/parakeet-tdt-0.6b-v2.

Training-time pieces that the NeMo SALM trainer expects (FSDP hooks,
gradient accumulation, mixed precision, audio preprocessing) are
intentionally kept out of this file. The NeMo trainer wraps this
class.

Qwen3.5 architecture note
-------------------------

Qwen3.5-4B is a hybrid model: blocks of Gated DeltaNet (linear
attention) and Gated Attention, with `full_attention_interval = 4`, so
3 layers of linear attention per 1 layer of full attention across 32
layers. Qwen3.5-2B is the same pattern over 24 layers.

The two models are NOT the same width, and this bit us:

* Qwen3.5-4B -> `text_config.hidden_size` = 2560
* Qwen3.5-2B -> `text_config.hidden_size` = 2048

Scaffolding that assumed 2560 for both produced a bridge that cannot be
spliced into the 2B model at all. Use `BimAsrConfig.for_llm()`, which
reads the real number out of the model's own config.

For LoRA we target the attention projections in the Gated Attention
layers (q_proj, k_proj, v_proj, o_proj) and the FFN gate/up/down
projections in all layers. This is the broadest correct target set
we can use without peft having explicit support for Gated DeltaNet.

Measured consequence of that choice (Qwen3.5-2B, 24 layers): the FFN
targets hit all 24 layers, but q/k/v/o_proj only exist in the 6 full
attention layers -- the 18 linear-attention layers express attention
through fused `in_proj_qkv` and `out_proj` instead, which this target
list does not reach. See docs/LORA_TARGETS.md before changing it.

Encoder width correction
-------------------------

`nvidia/parakeet-tdt-0.6b-v2` is a 24-layer FastConformer with
`d_model = 1024`, not 512. (v3 is the same width.) Verified by reading
`model_config.yaml` out of the .nemo and by the extraction script, which
writes `bim_asr_encoder.json` next to the weights.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch
from torch import nn


# Hidden sizes are pinned by the source checkpoints.
_FASTCONFORMER_HIDDEN = 1024    # Parakeet TDT 0.6B v2 encoder output (d_model)
_QWEN35_4B_HIDDEN = 2560        # Qwen3.5-4B token embedding dim
_QWEN35_2B_HIDDEN = 2048        # Qwen3.5-2B is narrower than the 4B
_AUDIO_FRAME_RATE = 12.5        # 80 ms frames, 12.5 frames per second of audio


@dataclasses.dataclass(frozen=True)
class BimAsrConfig:
    """Static configuration for a Bim-ASR model.

    The defaults match nvidia/parakeet-tdt-0.6b-v2 for the audio side
    and Qwen/Qwen3.5-4B for the language model side. Qwen3.5-2B is
    kept as a second variant for the laptop smoke test (smaller, fits
    in 16 GB M-series unified memory more comfortably).
    """

    encoder_id: str = "nvidia/parakeet-tdt-0.6b-v2"
    llm_id: str = "Qwen/Qwen3.5-4B"
    audio_locator_tag: str = "<|audio|>"

    encoder_hidden: int = _FASTCONFORMER_HIDDEN
    llm_hidden: int = _QWEN35_4B_HIDDEN
    audio_frame_rate_hz: float = _AUDIO_FRAME_RATE

    # LoRA settings
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    # Qwen3.5 target modules. Attention projections cover the Gated
    # Attention layers; the FFN projections cover the Gated DeltaNet
    # MLP paths. peft's standard module-name matching will hit the
    # matching parameter names in both block types.
    lora_target_modules: tuple = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    # Generation
    max_new_tokens: int = 512
    max_audio_seconds: int = 30

    # Prompt the SALM model is trained against
    user_prompt: str = "Transcribe the following audio."

    @classmethod
    def for_2b_smoke(cls) -> "BimAsrConfig":
        """The 2B variant for the laptop smoke test.

        Qwen3.5-2B has the same hybrid architecture as the 4B but is
        **2048 wide, not 2560**. The bridge is resized to match, so the
        projection stays 1.31M -> 2.10M parameters and the audio
        embeddings actually fit the LLM's embedding space.
        """
        return dataclasses.replace(
            cls(),
            llm_id="Qwen/Qwen3.5-2B",
            llm_hidden=_QWEN35_2B_HIDDEN,
        )

    @classmethod
    def for_llm(
        cls,
        llm_id: str,
        *,
        cache_dir: str | Path | None = None,
        **overrides,
    ) -> "BimAsrConfig":
        """Build a config for any LLM, reading its real hidden size.

        `llm_id` may be a Hub id or a local snapshot directory. This is
        the safe constructor: it cannot produce a bridge sized for the
        wrong model, which is exactly the bug that made the original
        `for_2b_smoke()` unusable.
        """
        hidden = llm_hidden_from_config(llm_id, cache_dir=cache_dir)
        return dataclasses.replace(cls(), llm_id=llm_id, llm_hidden=hidden, **overrides)


def llm_hidden_from_config(llm_id: str, *, cache_dir: str | Path | None = None) -> int:
    """Read `hidden_size` out of a model config, Hub id or local path.

    Handles the multimodal Qwen3.5 repos, where the text tower's width
    lives under `text_config` and the top-level config has no
    `hidden_size` at all.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(llm_id, cache_dir=str(cache_dir) if cache_dir else None)
    for holder in (config, getattr(config, "text_config", None)):
        if holder is None:
            continue
        hidden = getattr(holder, "hidden_size", None)
        if hidden:
            return int(hidden)
    raise ValueError(f"could not find hidden_size in the config for {llm_id}")


class AudioProjection(nn.Module):
    """Linear bridge from FastConformer frames to LLM token embedding space.

    Input: (batch, frames, 1024) - one 1024-dim vector per 80 ms audio frame
    Output: (batch, frames, 2560) - one 2560-dim vector per frame, ready
            to be interleaved with text token embeddings before the LLM
            (2048 wide for the 2B smoke model).

    A LayerNorm on the input is helpful because FastConformer features
    are roughly zero-mean, unit-variance per frame but the LLM's
    embedding space is not. The norm lets the linear layer start from
    a sensible scale.
    """

    def __init__(
        self,
        encoder_hidden: int = _FASTCONFORMER_HIDDEN,
        llm_hidden: int = _QWEN35_4B_HIDDEN,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(encoder_hidden)
        self.proj = nn.Linear(encoder_hidden, llm_hidden, bias=False)
        # Small init keeps the bridge near identity at the start.
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        # audio_features: (B, T_audio, encoder_hidden)
        x = self.norm(audio_features)
        return self.proj(x)


class BimAsrModel(nn.Module):
    """Top-level Bim-ASR module.

    Composition (all submodules are loaded from the named Hugging Face
    repos or initialised locally):

    - encoder: Parakeet FastConformer (we use the encoder only; the
      TDT decoder is dropped).
    - tokenizer: Qwen3.5 tokenizer, used to encode the prompt and
      decode the generated transcript.
    - llm: Qwen3.5-4B with LoRA adapters on the attention and FFN
      projections. Qwen3.5-2B is the smaller variant.
    - projection: a fresh nn.Linear(1024, llm_hidden) bridge —
      2560 for Qwen3.5-4B, 2048 for Qwen3.5-2B.

    The forward pass follows the SALM pattern:

    1. Run the frozen audio encoder to get per-frame features.
    2. Project those features into the LLM embedding space.
    3. Concatenate the projected audio embeddings with the embedded
       prompt tokens.
    4. Run the LLM (with LoRA) to autoregressively generate the
       transcript.

    Training mode and inference mode share the same forward signature.
    The trainer passes in labels; inference passes in max_new_tokens
    via generate.
    """

    def __init__(self, config: BimAsrConfig) -> None:
        super().__init__()
        self.config = config
        self.projection = AudioProjection(
            encoder_hidden=config.encoder_hidden,
            llm_hidden=config.llm_hidden,
        )
        # Submodules are loaded lazily via from_pretrained() so that
        # this file stays importable without the heavy deps installed.
        self.encoder = None
        self.llm = None
        self.tokenizer = None

    @classmethod
    def from_pretrained(
        cls,
        config: BimAsrConfig | None = None,
        *,
        encoder_id: str | None = None,
        llm_id: str | None = None,
        cache_dir: str | Path | None = None,
        load_in_4bit: bool = False,
        device: str = "cpu",
        dtype: torch.dtype | None = torch.bfloat16,
        apply_lora: bool = True,
    ) -> "BimAsrModel":
        """Load a Bim-ASR model with all submodules initialised.

        Loading is done through transformers + peft. The Parakeet
        encoder is loaded separately by `bim_asr.parakeet` from the
        weights extracted by scripts/prepare_parakeet_encoder.py; the
        Qwen3.5 LLM is loaded via transformers AutoModelForCausalLM.

        Notes that matter on Apple Silicon:

        * `AutoModelForCausalLM` on the Qwen3.5 repos resolves to
          `Qwen3_5ForCausalLM`, i.e. the text tower only. The vision
          tower in the Hub repo is not downloaded into memory at all
          (measured: 1.88B params for the 2B, zero `visual.*` tensors),
          so the open question in the build brief is settled: nothing
          needs dropping.
        * We deliberately do not use `device_map="auto"`. On MPS it
          shards across cpu/mps unpredictably and silently slows down
          generate; a single explicit `.to(device)` is predictable.
        * `load_in_4bit` needs bitsandbytes, which is CUDA-only. On the
          Mac, leave it off and run bf16 (the 2B fits easily in unified
          memory).
        """
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if config is None:
            config = BimAsrConfig()
        if encoder_id is not None:
            config = dataclasses.replace(config, encoder_id=encoder_id)
        if llm_id is not None:
            config = dataclasses.replace(
                config, llm_id=llm_id, llm_hidden=llm_hidden_from_config(
                    llm_id, cache_dir=cache_dir
                )
            )

        model = cls(config)

        llm_kwargs: dict = dict(cache_dir=str(cache_dir) if cache_dir else None)
        if dtype is not None:
            llm_kwargs["dtype"] = dtype
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            llm_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        llm = AutoModelForCausalLM.from_pretrained(config.llm_id, **llm_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(config.llm_id, cache_dir=llm_kwargs["cache_dir"])

        if apply_lora:
            # Apply LoRA on the attention and FFN projections.
            lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=list(config.lora_target_modules),
                bias="none",
                task_type="CAUSAL_LM",
            )
            llm = get_peft_model(llm, lora_config)
            # Make sure the embedding and LM head are frozen; only LoRA
            # adapters and any explicitly unfrozen modules are trainable.
            for name, param in llm.named_parameters():
                if "lora_" not in name:
                    param.requires_grad = False

        model.llm = llm
        model.tokenizer = tokenizer
        model.to(device)
        # The encoder is loaded separately, by name, from the extracted
        # checkpoint (NeMo wraps it as part of the audio preprocessor
        # pipeline during training).
        return model

    def token_embedding_layer(self) -> nn.Module:
        """The LLM's input embedding table, LoRA wrappers unwrapped."""
        llm = self.llm
        base = getattr(llm, "base_model", None)
        if base is not None and hasattr(base, "model"):
            llm = base.model
        return llm.get_input_embeddings()

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """(B, T) ids -> (B, T, llm_hidden) embeddings."""
        return self.token_embedding_layer()(input_ids)

    def splice_audio_into_prompt(
        self,
        audio_embeddings: torch.Tensor,
        prefix_ids: torch.Tensor,
        suffix_ids: torch.Tensor,
    ) -> torch.Tensor:
        """The SALM splice: text, audio, text.

        audio_embeddings (1, frames, llm_hidden), prefix_ids/suffix_ids
        (1, n) token ids. The audio lands exactly where the audio
        locator token sat in the prompt, which is where the model was
        trained to expect it.

        The bridge is kept in fp32 (it is the thing we train, and it is
        tiny, so precision is cheap here) and cast to the LLM's dtype
        here. Without the cast the concatenation is fp32 while the LLM
        weights are bf16, and the first linear layer rejects the input.
        """
        embeddings = self.token_embedding_layer()
        target_dtype = embeddings.weight.dtype

        prefix_ids = prefix_ids.to(embeddings.weight.device)
        suffix_ids = suffix_ids.to(embeddings.weight.device)
        audio_embeddings = audio_embeddings.to(
            device=embeddings.weight.device, dtype=target_dtype
        )

        prefix_emb = self.embed_tokens(prefix_ids)
        suffix_emb = self.embed_tokens(suffix_ids)
        if suffix_ids.shape[1] == 0:
            return torch.cat([prefix_emb, audio_embeddings], dim=1)
        return torch.cat([prefix_emb, audio_embeddings, suffix_emb], dim=1)

    def encode_audio(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Project acoustic features to the LLM embedding space.

        audio_features is the output of the FastConformer encoder,
        shape (batch, frames, encoder_hidden). Returns the projected
        features ready to be concatenated with the text token
        embeddings, shape (batch, frames, llm_hidden).
        """
        return self.projection(audio_features)

    def freeze_for_projection_only_training(self) -> None:
        """Freeze everything except the projection.

        Used for the very first stage of training where we want the
        bridge to learn the LLM embedding space before LoRA kicks in.
        """
        for p in self.parameters():
            p.requires_grad = False
        for p in self.projection.parameters():
            p.requires_grad = True

    def trainable_parameter_count(self) -> int:
        """Return the number of trainable parameters (projection + LoRA)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


SALM_SYSTEM_PROMPT = "You are a careful speech transcriber."


def stop_token_ids(tokenizer) -> list[int]:
    """Token ids that should end a generated transcript.

    Qwen3.5-2B on the Hub ships **no generation_config.json**, and its
    tokenizer reports `eos_token = ''` with `eos_token_id = 248046`
    (`<|im_end|>`) while `pad_token_id = 248044` (`<|endoftext|>`).
    Calling `generate()` without an explicit `eos_token_id` therefore
    never stops: the model emits the correct sentence and then keeps
    going, which on this task meant every WER was inflated by dozens of
    points by a single runaway generation.

    Both ids are returned so training and inference agree on where a
    transcript ends.
    """
    ids: set[int] = set()
    for candidate in (
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
    ):
        if isinstance(candidate, int):
            ids.add(candidate)
    for name in ("<|im_end|>", "<|endoftext|>"):
        try:
            token_id = tokenizer.convert_tokens_to_ids(name)
        except Exception:  # noqa: BLE001 - stub tokenizers in tests
            continue
        if isinstance(token_id, int) and token_id >= 0:
            ids.add(token_id)
    return sorted(ids)


def build_default_prompt_text(
    user_prompt: str,
    audio_locator_tag: str,
) -> str:
    """The SALM prompt as a chat-template-ready message list.

    `<|audio|>` is not a real Qwen3.5 token -- it is a marker in the
    text that we split on and replace with audio embeddings.
    """
    return [
        {"role": "system", "content": SALM_SYSTEM_PROMPT},
        {"role": "user", "content": f"{user_prompt} {audio_locator_tag}"},
    ]


def render_salm_prompt(
    tokenizer,
    *,
    user_prompt: str,
    audio_locator_tag: str,
) -> str:
    """The SALM prompt rendered through the chat template, as text."""
    return tokenizer.apply_chat_template(
        build_default_prompt_text(user_prompt, audio_locator_tag),
        add_generation_prompt=True,
        tokenize=False,
    )


def build_default_prompt(
    tokenizer,
    *,
    user_prompt: str,
    audio_locator_tag: str,
):
    """Build the SALM input prompt for training and inference.

    Returns a 1-D LongTensor of input ids, audio locator included. The
    audio embeddings are spliced in at the locator by
    `build_salm_prompt_ids` (preferred for inference and training,
    because it gives you the two halves directly).

    The format is Qwen3.5's ChatML.

    Note: in transformers >= 5, `apply_chat_template(tokenize=True)`
    returns a BatchEncoding, not a list of ints, so we render the
    template to text first and tokenise explicitly. The previous
    implementation did `torch.tensor(BatchEncoding)` and blew up.
    """
    prompt_text = render_salm_prompt(
        tokenizer, user_prompt=user_prompt, audio_locator_tag=audio_locator_tag
    )
    ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    return torch.tensor(ids, dtype=torch.long)


def build_salm_prompt_ids(
    tokenizer,
    *,
    user_prompt: str,
    audio_locator_tag: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the SALM prompt into the text before and after the locator.

    Returns (prefix_ids, suffix_ids), both 1-D LongTensors. The audio
    embeddings are spliced between them, which is the only correct way
    to do this: the suffix carries the assistant turn header, so the
    model is already primed to emit a transcript by the time it has
    read the audio.

    The split is done on the *rendered text*, not on token ids. BPE is
    context dependent, so `"<|audio|>"` tokenised on its own produces
    different ids than the same string sitting mid-sentence, and
    searching the full token sequence for the standalone ids finds
    nothing.
    """
    rendered = render_salm_prompt(
        tokenizer, user_prompt=user_prompt, audio_locator_tag=audio_locator_tag
    )
    if audio_locator_tag not in rendered:
        raise ValueError(
            f"audio locator {audio_locator_tag!r} is not present in the rendered "
            "prompt; cannot splice audio embeddings"
        )
    before, after = rendered.split(audio_locator_tag, 1)

    prefix_ids = tokenizer(before, add_special_tokens=False).input_ids
    suffix_ids = tokenizer(after, add_special_tokens=False).input_ids
    return torch.tensor(prefix_ids, dtype=torch.long), torch.tensor(suffix_ids, dtype=torch.long)


def count_trainable_parameters_summary(model: BimAsrModel) -> str:
    """One-line summary of trainable vs total parameters."""
    trainable = model.trainable_parameter_count()
    total = model.total_parameter_count()
    pct = 100.0 * trainable / max(total, 1)
    return "trainable={:,} / total={:,} ({:.3f} %)".format(trainable, total, pct)
