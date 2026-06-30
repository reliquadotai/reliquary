"""Shared HuggingFace model/tokenizer helpers.

The validator, miner, and offline tools must agree on loader selection,
tokenizer padding, and EOS semantics. Keep those choices here so model-family
changes do not drift across call sites.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


MODEL_SNAPSHOT_ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "model*.safetensors",
    "model.safetensors.index.json",
    "tokenizer*",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "processor_config.json",
    "image_processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
]


def _is_mock_like(value: Any) -> bool:
    return value is not None and value.__class__.__module__.startswith("unittest.mock")


def _config_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "cache_dir",
        "force_download",
        "local_files_only",
        "proxies",
        "revision",
        "subfolder",
        "token",
        "trust_remote_code",
    }
    return {k: v for k, v in kwargs.items() if k in allowed}


def _is_qwen35_conditional_config(config: Any) -> bool:
    if getattr(config, "model_type", None) != "qwen3_5":
        return False
    architectures = getattr(config, "architectures", None) or []
    return "Qwen3_5ForConditionalGeneration" in architectures


def auto_model_class_for_config(config: Any):
    """Return the correct transformers auto-class for a loaded config."""
    if _is_qwen35_conditional_config(config):
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText

    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM


def load_text_generation_model(source: str, **kwargs):
    """Load the model class used for text-only generation/training/proofs.

    `Qwen/Qwen3.5-2B` (and 4B) is packaged as a conditional image-text-to-text
    model, even for text-only use. Legacy Qwen3 checkpoints remain CausalLM.
    """
    from transformers import AutoConfig

    if "torch_dtype" in kwargs and "dtype" not in kwargs:
        kwargs["dtype"] = kwargs.pop("torch_dtype")
    config = AutoConfig.from_pretrained(source, **_config_kwargs(kwargs))
    auto_cls = auto_model_class_for_config(config)
    return auto_cls.from_pretrained(source, **kwargs)


def load_tokenizer(source: str, **kwargs):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    ensure_tokenizer_padding(tokenizer)
    return tokenizer


def ensure_tokenizer_padding(tokenizer: Any) -> Any:
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = getattr(tokenizer, "eos_token_id", None)
    return tokenizer


def _iter_token_ids(value: Any) -> Iterable[int]:
    if value is None or _is_mock_like(value):
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (str, bytes)):
        return []
    try:
        iterator = iter(value)
    except TypeError:
        return []
    ids: list[int] = []
    for item in iterator:
        if item is None or _is_mock_like(item):
            continue
        try:
            ids.append(int(item))
        except (TypeError, ValueError):
            continue
    return ids


def resolve_eos_token_ids(model: Any = None, tokenizer: Any = None) -> set[int]:
    """Return the full EOS set from model and tokenizer metadata.

    Qwen3.5 exposes `<|endoftext|>` through model generation config while its
    chat tokenizer uses `<|im_end|>`. Termination, p_stop, and generation must
    treat both as valid stops.
    """
    eos: set[int] = set()

    if model is not None and not _is_mock_like(model):
        gen_cfg = getattr(model, "generation_config", None)
        if gen_cfg is not None and not _is_mock_like(gen_cfg):
            eos.update(_iter_token_ids(getattr(gen_cfg, "eos_token_id", None)))

        config = getattr(model, "config", None)
        if config is not None and not _is_mock_like(config):
            eos.update(_iter_token_ids(getattr(config, "eos_token_id", None)))
            text_config = getattr(config, "text_config", None)
            if text_config is not None and not _is_mock_like(text_config):
                eos.update(_iter_token_ids(getattr(text_config, "eos_token_id", None)))

    if tokenizer is not None:
        eos.update(_iter_token_ids(getattr(tokenizer, "eos_token_id", None)))

    return eos


def first_eos_index(tokens: list[int], eos_ids: set[int]) -> int | None:
    for idx, token in enumerate(tokens):
        if int(token) in eos_ids:
            return idx
    return None


def think_close_token_ids(tokenizer) -> list[int]:
    """Atomic ``</think>`` token id(s) for this tokenizer. Raises if it cannot
    resolve an atomic id — we never fall back to a split encoding, which the BFT
    carve-out relies on being atomic."""
    tid = tokenizer.convert_tokens_to_ids("</think>")
    if tid is None or (isinstance(tid, int) and tid < 0):
        raise ValueError("tokenizer has no atomic </think> token")
    return [int(tid)]


def force_close_token_ids(tokenizer) -> list[int]:
    """Token ids of ``BFT_FORCE_TEMPLATE``: the atomic </think> id followed by
    the encoded answer preamble (no special tokens added to the tail)."""
    from reliquary.constants import BFT_FORCE_TEMPLATE

    close_ids = think_close_token_ids(tokenizer)
    tail = BFT_FORCE_TEMPLATE[len("</think>"):]
    tail_ids = list(tokenizer.encode(tail, add_special_tokens=False))
    return close_ids + [int(t) for t in tail_ids]


def has_think_close(tokens: list[int], think_close_ids: set[int]) -> bool:
    """True iff any token is an atomic </think> id."""
    return any(int(t) in think_close_ids for t in tokens)
