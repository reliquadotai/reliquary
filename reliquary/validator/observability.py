"""Structured validator observability helpers.

The validator has several clocks and decision points on the /submit path:
HTTP arrival, drand validation, async queue wait, proof verification,
pool admission, seal, final selection, and reward assignment. This module
keeps the field names consistent across logs, verdicts, and archives.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from reliquary.protocol.submission import BatchSubmissionRequest, RejectReason


_SECRET_KEY_PARTS = (
    "access_key",
    "api_key",
    "auth",
    "credential",
    "mnemonic",
    "password",
    "private",
    "secret",
    "seed",
    "token",
)


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in _SECRET_KEY_PARTS)


def _json_default(value: Any) -> str:
    return str(value)


def log_structured(
    logger: logging.Logger,
    level: int,
    event: str,
    fields: dict[str, Any],
) -> None:
    """Emit one parseable JSON payload, dropping fields that look secret."""
    payload = {
        "event": event,
        **{
            key: value
            for key, value in fields.items()
            if not _is_secret_key(key)
        },
    }
    logger.log(
        level,
        "%s %s",
        event,
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        ),
    )


def canonical_prompt_hash_lead(prompt_idx: int) -> str:
    """Stable canonical prompt hash prefix used for ordering diagnostics."""
    return hashlib.sha256(
        int(prompt_idx).to_bytes(8, "big", signed=False)
    ).hexdigest()[:12]


def merkle_root_lead(merkle_root: str) -> str:
    return (merkle_root or "")[:12]


def classify_drand_round(
    submitted_drand_round: int,
    arrival_drand_round: int | None,
    drand_tolerance: int,
) -> str:
    """Classify submitted vs validator-arrival drand round.

    The class is chronological, independent of whether the submission is
    ultimately accepted by later checks.
    """
    if arrival_drand_round is None:
        return "unknown"
    if submitted_drand_round > arrival_drand_round:
        return "future"
    if submitted_drand_round == arrival_drand_round:
        return "current"
    if submitted_drand_round >= arrival_drand_round - drand_tolerance:
        return "stale_within_tolerance"
    return "stale"


@dataclass(frozen=True)
class DrandRoundObservation:
    submitted_drand_round: int
    arrival_drand_round: int | None
    drand_delta: int | None
    drand_tolerance: int
    drand_status: str
    reject_reason: RejectReason | None

    def fields(self) -> dict[str, Any]:
        return {
            "submitted_drand_round": self.submitted_drand_round,
            "arrival_drand_round": self.arrival_drand_round,
            "drand_delta": self.drand_delta,
            "drand_tolerance": self.drand_tolerance,
            "drand_status": self.drand_status,
        }


@dataclass
class SubmitTelemetry:
    window_n: int
    prompt_idx: int
    hotkey: str
    merkle_root: str
    protocol_version: int
    submitted_drand_round: int
    t_arrival: float
    prompt_hash_lead: str
    merkle_root_lead: str
    arrival_drand_round: int | None = None
    drand_delta: int | None = None
    drand_tolerance: int | None = None
    drand_status: str | None = None
    window_open_drand_round: int | None = None
    seal_trigger_round: int | None = None
    valid_submissions_at_arrival: int | None = None
    valid_submissions_at_decision: int | None = None
    t_enqueued: float | None = None
    t_proof_started: float | None = None
    t_verified: float | None = None
    t_decision: float | None = None
    queue_wait_ms: float | None = None
    verify_ms: float | None = None
    total_ms: float | None = None
    legacy_merkle_status: str | None = None
    legacy_merkle_computed_lead: str | None = None
    legacy_merkle_would_reject: bool | None = None
    legacy_merkle_enforced: bool | None = None

    @classmethod
    def from_request(
        cls,
        request: BatchSubmissionRequest,
        *,
        t_arrival: float,
    ) -> "SubmitTelemetry":
        return cls(
            window_n=request.window_start,
            prompt_idx=request.prompt_idx,
            hotkey=request.miner_hotkey,
            merkle_root=request.merkle_root,
            protocol_version=request.protocol_version,
            submitted_drand_round=request.drand_round,
            t_arrival=t_arrival,
            prompt_hash_lead=canonical_prompt_hash_lead(request.prompt_idx),
            merkle_root_lead=merkle_root_lead(request.merkle_root),
        )

    def apply_legacy_merkle(
        self,
        *,
        status: str,
        computed_root: str | None,
        enforced: bool,
    ) -> None:
        self.legacy_merkle_status = status
        self.legacy_merkle_computed_lead = (
            merkle_root_lead(computed_root) if computed_root else None
        )
        self.legacy_merkle_would_reject = status != "match"
        self.legacy_merkle_enforced = bool(enforced)

    def apply_drand(self, observation: DrandRoundObservation) -> None:
        self.arrival_drand_round = observation.arrival_drand_round
        self.drand_delta = observation.drand_delta
        self.drand_tolerance = observation.drand_tolerance
        self.drand_status = observation.drand_status

    def refresh_from_batcher(
        self,
        batcher: Any | None,
        *,
        at_decision: bool = False,
    ) -> None:
        if batcher is None:
            return
        self.window_open_drand_round = getattr(
            batcher, "window_open_drand_round", self.window_open_drand_round
        )
        self.seal_trigger_round = getattr(
            batcher, "_seal_trigger_round", self.seal_trigger_round
        )
        # Proofs run at seal now, so ``valid_count`` is 0 for the whole window;
        # ride the graded (admitted-but-unproven) pending count instead.
        pending_count = getattr(batcher, "pending_count", None)
        if at_decision:
            self.valid_submissions_at_decision = pending_count
        else:
            self.valid_submissions_at_arrival = pending_count

    def mark_enqueued(self, ts: float | None = None) -> None:
        self.t_enqueued = ts if ts is not None else time.time()

    def mark_proof_started(self, ts: float | None = None) -> None:
        self.t_proof_started = ts if ts is not None else time.time()
        if self.t_enqueued is not None:
            self.queue_wait_ms = max(0.0, (self.t_proof_started - self.t_enqueued) * 1000.0)
        else:
            self.queue_wait_ms = max(0.0, (self.t_proof_started - self.t_arrival) * 1000.0)

    def mark_decision(self, ts: float | None = None, *, verified: bool = False) -> None:
        t = ts if ts is not None else time.time()
        self.t_decision = t
        if verified:
            self.t_verified = t
        if self.t_proof_started is not None:
            self.verify_ms = max(0.0, (t - self.t_proof_started) * 1000.0)
        self.total_ms = max(0.0, (t - self.t_arrival) * 1000.0)

    def fields(self) -> dict[str, Any]:
        return {
            "window_n": self.window_n,
            "prompt_idx": self.prompt_idx,
            "hotkey": self.hotkey,
            "t_arrival": self.t_arrival,
            "t_verified": self.t_verified,
            "t_decision": self.t_decision,
            "queue_wait_ms": self.queue_wait_ms,
            "verify_ms": self.verify_ms,
            "total_ms": self.total_ms,
            "protocol_version": self.protocol_version,
            "submitted_drand_round": self.submitted_drand_round,
            "arrival_drand_round": self.arrival_drand_round,
            "drand_delta": self.drand_delta,
            "drand_tolerance": self.drand_tolerance,
            "drand_status": self.drand_status,
            "window_open_drand_round": self.window_open_drand_round,
            "seal_trigger_round": self.seal_trigger_round,
            "valid_submissions_at_arrival": self.valid_submissions_at_arrival,
            "valid_submissions_at_decision": self.valid_submissions_at_decision,
            "reject_stage": None,
            "reject_reason": None,
            "prompt_hash_lead": self.prompt_hash_lead,
            "canonical_hash_prefix": self.prompt_hash_lead,
            "merkle_root_lead": self.merkle_root_lead,
            "legacy_merkle_status": self.legacy_merkle_status,
            "legacy_merkle_computed_lead": (
                self.legacy_merkle_computed_lead
            ),
            "legacy_merkle_would_reject": (
                self.legacy_merkle_would_reject
            ),
            "legacy_merkle_enforced": self.legacy_merkle_enforced,
        }

    def verdict_fields(self) -> dict[str, Any]:
        return {
            "arrival_ts": self.t_arrival,
            "decision_ts": self.t_decision,
            "submitted_drand_round": self.submitted_drand_round,
            "arrival_drand_round": self.arrival_drand_round,
            "drand_delta": self.drand_delta,
            "seal_trigger_round": self.seal_trigger_round,
            "prompt_hash_lead": self.prompt_hash_lead,
            "queue_wait_ms": self.queue_wait_ms,
            "verify_ms": self.verify_ms,
            "total_ms": self.total_ms,
        }


def log_submission_stage(
    logger: logging.Logger,
    level: int,
    stage: str,
    telemetry: SubmitTelemetry,
    **extra: Any,
) -> None:
    fields = telemetry.fields()
    fields.update(extra)
    fields["stage"] = stage
    log_structured(logger, level, "validator_submit_lifecycle", fields)


@lru_cache(maxsize=1)
def runtime_revision() -> str | None:
    """Best-effort deployed revision without reading any secret env vars."""
    # Prefer the immutable image-baked revision over deployment env overrides:
    # watchtower preserves container env across image updates, so old compose
    # values like RELIQUARY_IMAGE_REVISION can become stale.
    for name in (
        "RELIQUARY_BUILD_REVISION",
        "RELIQUARY_IMAGE_BUILD_REVISION",
    ):
        value = os.getenv(name)
        if value and value != "unknown":
            return value[:40]

    for name in (
        "RELIQUARY_IMAGE_REVISION",
        "RELIQUARY_GIT_SHA",
        "GIT_SHA",
        "SOURCE_COMMIT",
        "COMMIT_SHA",
        "IMAGE_REVISION",
    ):
        value = os.getenv(name)
        if value:
            return value[:40]

    repo_root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1.0,
        ).strip()
    except Exception:
        return None


def current_unix_ts() -> float:
    return time.time()
