"""Inference-runtime fingerprints for numerical-contract observability.

Profiles are self-reported and integrity-bound to the signed request nonce.
They are useful for correlation and rollout canaries, but are not hardware
attestation and must never be used as proof of honest execution.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from typing import Any


def runtime_profile_hash(profile: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in profile.items()
        if key != "profile_hash"
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def bind_runtime_profile_nonce(base_nonce: str, profile_hash: str) -> str:
    """Bind a self-reported profile to the already signed envelope nonce."""
    return f"{base_nonce}.{profile_hash}"


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return "unknown"


def _model_dtype(model: Any) -> str | None:
    if model is None:
        return None
    value = getattr(model, "dtype", None)
    if value is None:
        try:
            value = next(model.parameters()).dtype
        except Exception:
            return None
    return str(value).removeprefix("torch.")


def _attention_implementation(model: Any) -> str | None:
    config = getattr(model, "config", None)
    value = getattr(config, "_attn_implementation", None)
    if value:
        return str(value)
    value = os.environ.get("GRAIL_ATTN_IMPL")
    return str(value) if value else None


def _qwen35_kernel_flags() -> dict[str, bool | None]:
    flags: dict[str, bool | None] = {
        "qwen35_fast_path_all": None,
        "qwen35_fla_chunk": None,
        "qwen35_fla_recurrent": None,
        "qwen35_causal_conv_prefill": None,
        "qwen35_causal_conv_update": None,
    }
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as module

        flags.update(
            {
                "qwen35_fast_path_all": bool(
                    getattr(module, "is_fast_path_available", False)
                ),
                "qwen35_fla_chunk": getattr(
                    module, "chunk_gated_delta_rule", None
                ) is not None,
                "qwen35_fla_recurrent": getattr(
                    module, "fused_recurrent_gated_delta_rule", None
                ) is not None,
                "qwen35_causal_conv_prefill": getattr(
                    module, "causal_conv1d_fn", None
                ) is not None,
                "qwen35_causal_conv_update": getattr(
                    module, "causal_conv1d_update", None
                ) is not None,
            }
        )
    except Exception:
        pass
    return flags


def collect_runtime_fingerprint(
    generation_model: Any = None,
    proof_model: Any = None,
) -> dict[str, Any]:
    """Collect a secret-free runtime profile without loading model weights."""
    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_version = str(torch.version.cuda) if torch.version.cuda else None
        cuda_available = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
        compute_capability = (
            ".".join(str(part) for part in torch.cuda.get_device_capability(0))
            if cuda_available
            else None
        )
        deterministic_algorithms = bool(
            torch.are_deterministic_algorithms_enabled()
        )
        cudnn_benchmark = bool(torch.backends.cudnn.benchmark)
        tf32_matmul = bool(
            getattr(torch.backends.cuda.matmul, "allow_tf32", False)
        )
    except Exception:
        torch_version = _distribution_version("torch")
        cuda_version = None
        cuda_available = False
        gpu_name = None
        compute_capability = None
        deterministic_algorithms = False
        cudnn_benchmark = False
        tf32_matmul = False

    profile: dict[str, Any] = {
        "schema_version": 1,
        "python_version": platform.python_version(),
        "platform": platform.system().lower(),
        "torch_version": torch_version,
        "transformers_version": _distribution_version("transformers"),
        "fla_version": _distribution_version("flash-linear-attention"),
        "causal_conv1d_version": _distribution_version("causal-conv1d"),
        "flash_attn_version": _distribution_version("flash-attn"),
        "cuda_version": cuda_version,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "compute_capability": compute_capability,
        "generation_dtype": _model_dtype(generation_model),
        "proof_dtype": _model_dtype(proof_model),
        "generation_attention_implementation": _attention_implementation(
            generation_model
        ),
        "proof_attention_implementation": _attention_implementation(proof_model),
        "deterministic_algorithms": deterministic_algorithms,
        "cudnn_benchmark": cudnn_benchmark,
        "tf32_matmul": tf32_matmul,
        **_qwen35_kernel_flags(),
    }
    profile["profile_hash"] = runtime_profile_hash(profile)
    return profile
