"""HuggingFace model compatibility helpers."""

from typing import Any

from transformers import PretrainedConfig


def resolve_vocab_size(model_config: PretrainedConfig | Any) -> int | None:
    """Return vocab size if present in config (direct or nested)."""
    try:
        for attr in ("vocab_size", "n_vocab", "vocabulary_size"):
            if hasattr(model_config, attr):
                val = getattr(model_config, attr)
                if isinstance(val, int) and val > 0:
                    return val

        text_cfg = getattr(model_config, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "vocab_size"):
            val = text_cfg.vocab_size
            if isinstance(val, int) and val > 0:
                return val
    except Exception:
        pass

    return None


def resolve_max_context_length(model_config: PretrainedConfig | Any) -> int:
    """Return best-effort max context length with common fallbacks."""
    candidates = (
        "max_position_embeddings",
        "max_seq_len",
        "n_positions",
        "seq_length",
        "max_sequence_length",
        "model_max_length",
    )

    for name in candidates:
        try:
            if hasattr(model_config, name):
                val = getattr(model_config, name)
                if isinstance(val, int) and val > 0:
                    return val
        except Exception:
            pass

    try:
        text_cfg = getattr(model_config, "text_config", None)
        if text_cfg is not None:
            for name in candidates:
                if hasattr(text_cfg, name):
                    val = getattr(text_cfg, name)
                    if isinstance(val, int) and val > 0:
                        return val
    except Exception:
        pass

    return 16384
