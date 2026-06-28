"""Validator-private token-auth forensics.

These records are intentionally local-only. Public R2 archives keep aggregate
counts/minima; exact positions and token text stay on the validator host.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_FALSE_VALUES = {"0", "false", "no", "off"}

CODE_SEMANTIC_HIGH_SIGNAL_LABELS = frozenset({
    "binary_op",
    "bool_op",
    "compare_op",
    "constant:bool",
    "constant:none",
    "constant:number",
    "subscript_slice",
    "unary_op",
})
CODE_SEMANTIC_LOW_SIGNAL_LABELS = frozenset({"constant:string"})


def auth_forensics_enabled() -> bool:
    raw = os.environ.get("RELIQUARY_AUTH_FORENSICS_ENABLED", "1")
    return raw.strip().lower() not in _FALSE_VALUES


def auth_forensics_max_findings_per_rollout() -> int:
    raw = os.environ.get("RELIQUARY_AUTH_FORENSICS_MAX_FINDINGS_PER_ROLLOUT")
    if raw is None:
        return 16
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(
            "Invalid RELIQUARY_AUTH_FORENSICS_MAX_FINDINGS_PER_ROLLOUT=%r; "
            "using 16",
            raw,
        )
        return 16


def auth_forensics_context_chars() -> int:
    raw = os.environ.get("RELIQUARY_AUTH_FORENSICS_CONTEXT_CHARS")
    if raw is None:
        return 80
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(
            "Invalid RELIQUARY_AUTH_FORENSICS_CONTEXT_CHARS=%r; using 80",
            raw,
        )
        return 80


def code_semantic_counterfactual_enabled() -> bool:
    raw = os.environ.get("RELIQUARY_CODE_SEMANTIC_COUNTERFACTUAL_ENABLED", "1")
    return raw.strip().lower() not in _FALSE_VALUES


def code_semantic_counterfactual_max_findings_per_rollout() -> int:
    raw = os.environ.get(
        "RELIQUARY_CODE_SEMANTIC_COUNTERFACTUAL_MAX_FINDINGS_PER_ROLLOUT"
    )
    if raw is None:
        return 4
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(
            "Invalid "
            "RELIQUARY_CODE_SEMANTIC_COUNTERFACTUAL_MAX_FINDINGS_PER_ROLLOUT=%r; "
            "using 4",
            raw,
        )
        return 4


def code_semantic_signal_bucket(label: Any) -> str:
    text = str(label or "")
    if text in CODE_SEMANTIC_HIGH_SIGNAL_LABELS:
        return "high"
    if text in CODE_SEMANTIC_LOW_SIGNAL_LABELS:
        return "low"
    if text == "return_expr" or text.startswith("keyword:"):
        return "review"
    return "unknown"


def auth_forensics_path() -> Path:
    explicit = os.environ.get("RELIQUARY_AUTH_FORENSICS_PATH")
    if explicit:
        return Path(explicit)
    state_dir = os.environ.get("RELIQUARY_STATE_DIR", "/root/reliquary/state")
    return Path(state_dir) / "auth_forensics" / "all-token-auth-shadow.jsonl"


def code_semantic_auth_forensics_path() -> Path:
    explicit = os.environ.get("RELIQUARY_CODE_SEMANTIC_AUTH_FORENSICS_PATH")
    if explicit:
        return Path(explicit)
    state_dir = os.environ.get("RELIQUARY_STATE_DIR", "/root/reliquary/state")
    return Path(state_dir) / "auth_forensics" / "code-semantic-auth.jsonl"


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _trim(value: Any, max_chars: int) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def record_all_token_auth_findings(
    *,
    metrics: dict[str, Any],
    window_start: int,
    env_name: str,
    miner_hotkey: str,
    prompt_idx: int,
    rollout_idx: int,
    rollout_reward: float,
    reward_positive: bool,
    prompt_length: int,
    completion_length: int,
    path: str | Path | None = None,
) -> None:
    """Append one local JSONL record per all-token shadow finding.

    The caller should already have opted into detailed metrics. This function
    is fail-soft: write errors are logged but never affect submission handling.
    """
    if not auth_forensics_enabled():
        return
    details = metrics.get("finding_details") or []
    if not details:
        return

    out_path = Path(path) if path is not None else auth_forensics_path()
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ts_unix = time.time()
        with open(out_path, "a", encoding="utf-8") as f:
            for detail in details:
                record = {
                    "schema_version": 1,
                    "event": "all_token_auth_shadow_finding",
                    "surface": "all-token-auth-shadow",
                    "ts_unix": ts_unix,
                    "window_start": int(window_start),
                    "env_name": str(env_name),
                    "miner_hotkey": str(miner_hotkey),
                    "prompt_idx": int(prompt_idx),
                    "rollout_idx": int(rollout_idx),
                    "rollout_reward": float(rollout_reward),
                    "reward_positive": bool(reward_positive),
                    "prompt_length": int(prompt_length),
                    "completion_length": int(completion_length),
                    "n_tokens": _int_or_none(metrics.get("n_tokens")),
                    "threshold": _float_or_none(metrics.get("threshold")),
                    "argmax_conf": _float_or_none(metrics.get("argmax_conf")),
                    "min_prob": _float_or_none(metrics.get("min_prob")),
                    "finding_min_prob": _float_or_none(
                        metrics.get("finding_min_prob")
                    ),
                    "completion_pos": _int_or_none(
                        detail.get("completion_pos")
                    ),
                    "absolute_token_pos": _int_or_none(
                        detail.get("absolute_token_pos")
                    ),
                    "p_chosen": _float_or_none(detail.get("p_chosen")),
                    "p_argmax": _float_or_none(detail.get("p_argmax")),
                    "token_id": _int_or_none(detail.get("token_id")),
                    "argmax_id": _int_or_none(detail.get("argmax_id")),
                    "token_text": _trim(detail.get("token_text"), 120),
                    "argmax_text": _trim(detail.get("argmax_text"), 120),
                    "completion_context": _trim(
                        detail.get("completion_context"),
                        280,
                    ),
                }
                f.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
                f.write("\n")
    except Exception as exc:
        logger.warning(
            "auth_forensics_write_failed path=%s error=%r",
            out_path,
            exc,
        )


def record_code_semantic_auth_findings(
    *,
    metrics: dict[str, Any],
    window_start: int,
    env_name: str,
    miner_hotkey: str,
    prompt_idx: int,
    rollout_idx: int,
    rollout_reward: float,
    reward_positive: bool,
    prompt_length: int,
    completion_length: int,
    path: str | Path | None = None,
) -> None:
    """Append local JSONL records for suspicious OpenCode semantic tokens."""
    if not auth_forensics_enabled():
        return
    details = metrics.get("finding_details") or []
    if not details:
        return

    out_path = (
        Path(path) if path is not None else code_semantic_auth_forensics_path()
    )
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ts_unix = time.time()
        with open(out_path, "a", encoding="utf-8") as f:
            for detail in details:
                record = {
                    "schema_version": 1,
                    "event": "code_semantic_auth_finding",
                    "surface": "code-semantic",
                    "ts_unix": ts_unix,
                    "window_start": int(window_start),
                    "env_name": str(env_name),
                    "miner_hotkey": str(miner_hotkey),
                    "prompt_idx": int(prompt_idx),
                    "rollout_idx": int(rollout_idx),
                    "rollout_reward": float(rollout_reward),
                    "reward_positive": bool(reward_positive),
                    "prompt_length": int(prompt_length),
                    "completion_length": int(completion_length),
                    "n_spans": _int_or_none(metrics.get("n_spans")),
                    "n_tokens": _int_or_none(metrics.get("n_tokens")),
                    "threshold": _float_or_none(metrics.get("threshold")),
                    "argmax_conf": _float_or_none(metrics.get("argmax_conf")),
                    "min_prob": _float_or_none(metrics.get("min_prob")),
                    "finding_min_prob": _float_or_none(
                        metrics.get("finding_min_prob")
                    ),
                    "completion_pos": _int_or_none(
                        detail.get("completion_pos")
                    ),
                    "absolute_token_pos": _int_or_none(
                        detail.get("absolute_token_pos")
                    ),
                    "completion_char_start": _int_or_none(
                        detail.get("completion_char_start")
                    ),
                    "completion_char_end": _int_or_none(
                        detail.get("completion_char_end")
                    ),
                    "code_char_start": _int_or_none(
                        detail.get("code_char_start")
                    ),
                    "code_char_end": _int_or_none(
                        detail.get("code_char_end")
                    ),
                    "label": _trim(detail.get("label"), 80),
                    "signal_bucket": code_semantic_signal_bucket(
                        detail.get("label")
                    ),
                    "p_chosen": _float_or_none(detail.get("p_chosen")),
                    "p_argmax": _float_or_none(detail.get("p_argmax")),
                    "token_id": _int_or_none(detail.get("token_id")),
                    "argmax_id": _int_or_none(detail.get("argmax_id")),
                    "token_text": _trim(detail.get("token_text"), 120),
                    "argmax_text": _trim(detail.get("argmax_text"), 120),
                    "completion_context": _trim(
                        detail.get("completion_context"),
                        280,
                    ),
                    "code_context": _trim(detail.get("code_context"), 280),
                    "counterfactual_checked": bool(
                        detail.get("counterfactual_checked", False)
                    ),
                    "counterfactual_reward": _float_or_none(
                        detail.get("counterfactual_reward")
                    ),
                    "counterfactual_reward_delta": _float_or_none(
                        detail.get("counterfactual_reward_delta")
                    ),
                    "counterfactual_reward_flipped": bool(
                        detail.get("counterfactual_reward_flipped", False)
                    ),
                    "counterfactual_error": _trim(
                        detail.get("counterfactual_error"), 120
                    ),
                }
                f.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
                f.write("\n")
    except Exception as exc:
        logger.warning(
            "code_semantic_auth_forensics_write_failed path=%s error=%r",
            out_path,
            exc,
        )
