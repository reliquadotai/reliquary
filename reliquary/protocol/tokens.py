"""Token validation and hashing utilities for GRAIL protocol."""

from __future__ import annotations

import hashlib
import logging
import struct
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PretrainedConfig
else:
    try:
        from transformers import PretrainedConfig
    except ImportError:
        PretrainedConfig = Any  # type: ignore

from reliquary.shared.hf_compat import resolve_max_context_length, resolve_vocab_size

logger = logging.getLogger(__name__)


def int_to_bytes(i: int) -> bytes:
    """Convert integer to 4-byte big-endian representation."""
    return struct.pack(">I", i & 0xFFFFFFFF)


def hash_tokens(tokens: list[int]) -> bytes:
    """Compute SHA-256 hash of tokens for integrity checking."""
    tokens_bytes = b"".join(int_to_bytes(t) for t in tokens)
    return hashlib.sha256(tokens_bytes).digest()


def encode_prompt(tokenizer: Any, prompt_text: str) -> list[int]:
    """Canonical prompt → token encoder shared by miner and validator.

    Instruct/thinking models (Qwen3.5, etc.) ship a chat template
    that wraps the user message in role markers; feeding them the bare
    string never reaches the assistant turn so the model just continues
    the prompt instead of answering. Apply the chat template when one is
    declared, fall back to plain encoding otherwise (base models, test
    fakes). Both sides MUST go through this helper — any divergence in
    prompt tokenisation trips PROMPT_MISMATCH before GRAIL even runs.
    """
    # Require chat_template to be a non-empty string so MagicMock-based test
    # stubs (which return a MagicMock for any attribute) fall through to the
    # plain encode() path they already declare.
    chat_template = getattr(tokenizer, "chat_template", None)
    if isinstance(chat_template, str) and chat_template and hasattr(
        tokenizer, "apply_chat_template"
    ):
        messages = [{"role": "user", "content": prompt_text}]
        # transformers >=5 returns a BatchEncoding from apply_chat_template
        # even with tokenize=True; request return_dict=False for a flat list
        # and still defensively unwrap input_ids if a version ignores it.
        # Without this, list(...) iterates the dict keys ('input_ids', ...).
        kwargs = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": False,
        }
        if "enable_thinking" in chat_template:
            # Enable thinking: the raised token cap gives the CoT room to close
            # </think> and emit \boxed{} before truncating.
            kwargs["enable_thinking"] = True
        encoded = tokenizer.apply_chat_template(messages, **kwargs)
        if isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
            encoded = encoded["input_ids"]
        return list(encoded)
    return list(tokenizer.encode(prompt_text, add_special_tokens=False))


def verify_tokens(tokens: list[int], model_config: PretrainedConfig | Any) -> bool:
    """Verify token list validity for model processing."""
    if not tokens:
        logger.warning("Empty token list in commit")
        return False

    vocab_size = resolve_vocab_size(model_config)
    if vocab_size is not None:
        if not _validate_token_ids(tokens, vocab_size):
            return False
    else:
        logger.debug("Model config lacks vocab_size; skipping token-id bounds check")

    if not _validate_sequence_length(tokens, model_config):
        return False

    return True


def _validate_token_ids(tokens: list[int], vocab_size: int) -> bool:
    """Check that all token IDs are within vocabulary bounds."""
    invalid_tokens = [t for t in tokens if not isinstance(t, int) or t < 0 or t >= vocab_size]
    if invalid_tokens:
        logger.warning(
            "Invalid token IDs found: %s (vocab_size=%d)", invalid_tokens[:10], vocab_size
        )
        return False
    return True


def _validate_sequence_length(tokens: list[int], model_config: PretrainedConfig | Any) -> bool:
    """Check that token sequence doesn't exceed model's max length."""
    max_length = resolve_max_context_length(model_config)

    if len(tokens) > max_length:
        logger.warning("Token sequence (%d) exceeds model max length (%d)", len(tokens), max_length)
        return False
    return True
