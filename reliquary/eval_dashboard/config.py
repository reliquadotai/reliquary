"""Canonical configuration hashing and runtime provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping

from reliquary.eval_dashboard.models import ContaminationReview, EvalConfig


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically and reject non-finite values."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(effective_config: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(effective_config))


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _producer_revision(repo_root: str | Path | None = None) -> str:
    override = os.getenv("RELIQUARY_EVAL_PRODUCER_REVISION", "").strip()
    if override:
        if len(override) < 40 or any(c not in "0123456789abcdef" for c in override):
            raise ValueError(
                "RELIQUARY_EVAL_PRODUCER_REVISION must be an immutable hex revision"
            )
        return override
    root = Path(repo_root or Path(__file__).resolve().parents[2])
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "cannot resolve producer revision; set RELIQUARY_EVAL_PRODUCER_REVISION"
        ) from exc


def tokenizer_contract(tokenizer: Any) -> dict[str, Any]:
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        template = ""
    return {
        "class": tokenizer.__class__.__name__,
        "chat_template_sha256": sha256_bytes(template.encode("utf-8")),
        "chat_template_present": bool(template),
        "vocab_size": int(len(tokenizer)),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }


def runtime_contract(*, attention_implementation: str) -> dict[str, Any]:
    return {
        "producer_revision": _producer_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "reliquary": _package_version("reliquary"),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "datasets": _package_version("datasets"),
            "huggingface-hub": _package_version("huggingface-hub"),
        },
        "attention_implementation": attention_implementation,
        "cublas_workspace_config": os.getenv("CUBLAS_WORKSPACE_CONFIG", ""),
    }


def validate_contamination_reviews(
    config: EvalConfig,
    contamination_reviews: Mapping[str, ContaminationReview | dict[str, Any]],
) -> dict[str, ContaminationReview]:
    """Validate that each approved review seals its declared holdout."""
    if set(contamination_reviews) != {"math", "code"}:
        raise ValueError(
            "effective config requires math and code contamination reviews"
        )

    holdouts = {
        "math": config.math_holdout,
        "code": config.code_holdout,
    }
    validated: dict[str, ContaminationReview] = {}
    for domain, raw_review in contamination_reviews.items():
        review = ContaminationReview.model_validate(raw_review)
        holdout = holdouts[domain]
        if review.domain != domain or review.decision != "approved":
            raise ValueError(f"{domain} contamination review is not approved")
        if review.exact_overlap_count or review.near_duplicate_count:
            raise ValueError(f"{domain} approved contamination review contains overlap")
        if review.holdout_sha256 != holdout.artifact_sha256:
            raise ValueError(
                f"{domain} contamination review refers to a different holdout artifact"
            )
        if review.task_ids_sha256 != holdout.task_ids_sha256:
            raise ValueError(
                f"{domain} contamination review refers to a different task-id list"
            )
        review_sha = sha256_bytes(canonical_json_bytes(review.model_dump(mode="json")))
        if review_sha != holdout.contamination_review_sha256:
            raise ValueError(
                f"{domain} contamination review digest does not match the config"
            )
        validated[domain] = review
    return validated


def build_effective_config(
    config: EvalConfig,
    *,
    tokenizer: Any,
    attention_implementation: str,
    hardware: dict[str, Any],
    contamination_reviews: dict[str, ContaminationReview],
) -> dict[str, Any]:
    """Return the complete, hash-addressed evaluation contract."""
    runtime = runtime_contract(
        attention_implementation=attention_implementation,
    )
    producer_revision = runtime["producer_revision"]
    contamination_reviews = validate_contamination_reviews(
        config, contamination_reviews
    )
    for holdout in (config.math_holdout, config.code_holdout):
        if holdout.grader_revision != producer_revision:
            raise ValueError(
                f"{holdout.domain} grader_revision must match the deployed "
                "Reliquary producer revision"
            )
    if config.generation.protocol_parity:
        from reliquary.constants import (
            BFT_ANSWER_BUDGET,
            BFT_ENABLED,
            BFT_FORCE_TEMPLATE,
            BFT_THINKING_BUDGET,
            MAX_NEW_TOKENS_PROTOCOL_CAP,
            TOP_K_PROTO,
            TOP_P_PROTO,
            T_PROTO,
        )

        expected = {
            "temperature": T_PROTO,
            "top_p": TOP_P_PROTO,
            "top_k": TOP_K_PROTO,
            "repetition_penalty": 1.0,
            "math_bft_enabled": BFT_ENABLED,
            "math_thinking_budget": BFT_THINKING_BUDGET,
            "math_answer_budget": BFT_ANSWER_BUDGET,
            "math_force_template": BFT_FORCE_TEMPLATE,
            "math_max_new_tokens": MAX_NEW_TOKENS_PROTOCOL_CAP,
            "code_max_new_tokens": MAX_NEW_TOKENS_PROTOCOL_CAP,
        }
        actual = config.generation.model_dump()
        mismatches = {
            key: {"configured": actual[key], "protocol": value}
            for key, value in expected.items()
            if actual[key] != value
        }
        if mismatches:
            raise ValueError(f"protocol-parity generation mismatch: {mismatches}")
    return {
        "schema_version": "1",
        "declared": config.model_dump(mode="json"),
        "tokenizer": tokenizer_contract(tokenizer),
        "runtime": runtime,
        "hardware": hardware,
        "contamination_reviews": {
            domain: review.model_dump(mode="json")
            for domain, review in sorted(contamination_reviews.items())
        },
    }


def load_config(path: str | Path) -> EvalConfig:
    with Path(path).open("rb") as handle:
        return EvalConfig.model_validate_json(handle.read())
