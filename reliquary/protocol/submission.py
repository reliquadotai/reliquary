"""Pydantic v2 models for the miner→validator GRPO submission protocol.

The validator HTTP server (reliquary/validator/server.py) accepts these payloads
and the miner submitter (reliquary/miner/submitter.py) produces them.
"""

from __future__ import annotations

import base64
import binascii
from enum import Enum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_serializer,
    model_validator,
)

from reliquary.constants import M_ROLLOUTS


# ---------------------------------------------------------------------------
# v2 GRPO Market schemas
# ---------------------------------------------------------------------------


class RejectReason(str, Enum):
    """Canonical reject codes emitted by the v2 validator.

    Two values are success sentinels rather than rejects:
      - ``ACCEPTED``: the validator accepted the submission at its current
        lifecycle stage. In auction mode the first ACCEPTED verdict means the
        candidate entered the pending pool; the seal-time verdict carries
        ``selected_for_batch`` and ``rewarded`` to report the final outcome.
      - ``SUBMITTED``: the request was placed on the worker queue and
        will be validated asynchronously. The miner does NOT yet know
        whether GRAIL will pass — the real verdict surfaces in the
        validator's logs and in the R2 archive.

    All other values are mutually exclusive failure reasons; only the
    first failure reason is reported per submission.
    """

    ACCEPTED = "accepted"
    SUBMITTED = "submitted"
    BAD_SIGNATURE = "bad_signature"
    # Envelope-level signature failed verification, or the envelope was
    # unsigned. Recorded BEFORE the per-hotkey rate-limit increment so an
    # attacker spoofing ``miner_hotkey`` can't exhaust a victim's quota
    # by submitting unsigned packets. See ``build_envelope_binding`` in
    # protocol/signatures.py.
    BAD_ENVELOPE_SIGNATURE = "bad_envelope_signature"
    BAD_PROMPT_IDX = "bad_prompt_idx"
    PROMPT_MISMATCH = "prompt_mismatch"
    DISTRIBUTION_SUSPICIOUS = "distribution_suspicious"
    PROMPT_IN_COOLDOWN = "prompt_in_cooldown"
    # Deprecated v2.3+: SUPERSEDED is no longer emitted by the validator
    # (drand ordering replaced the FIFO per-prompt claim). Kept in the
    # enum so historical archives in R2 that carry the string deserialize.
    SUPERSEDED = "superseded"
    PROMPT_FULL = "prompt_full"
    PROMPT_OUT_OF_RANGE = "prompt_out_of_range"
    GRAIL_FAIL = "grail_fail"
    # The task's contract enforces toploc and the rollout's proofs failed it.
    TOPLOC_FAIL = "toploc_fail"
    HASH_DUPLICATE = "hash_duplicate"
    LOGPROB_MISMATCH = "logprob_mismatch"
    REWARD_MISMATCH = "reward_mismatch"
    REWARD_DISTRIBUTION = "reward_distribution"
    OUT_OF_ZONE = "out_of_zone"
    RATE_LIMITED = "rate_limited"
    BATCH_FILLED = "batch_filled"
    WRONG_ROLLOUT_COUNT = "wrong_rollout_count"
    WINDOW_MISMATCH = "window_mismatch"
    WINDOW_NOT_ACTIVE = "window_not_active"
    BAD_SCHEMA = "bad_schema"
    BAD_TOKENS = "bad_tokens"
    TOKENS_MISMATCH = "tokens_mismatch"
    BAD_TERMINATION = "bad_termination"
    BOXED_ANSWER_TAMPERED = "boxed_answer_tampered"
    TOKEN_TAMPERED = "token_tampered"
    SEED_MISMATCH = "seed_mismatch"
    MALFORMED_FINAL_ANSWER = "malformed_final_answer"
    # Deprecated: the reward-shape filter no longer rejects submissions
    # (trivially bypassable + false-positive-prone). Kept in the enum so
    # historical archives in R2 that carry the string still deserialize.
    REWARD_SHAPE_SUSPICIOUS = "reward_shape_suspicious"
    WRONG_CHECKPOINT = "wrong_checkpoint"
    MERKLE_ROOT_MISMATCH = "merkle_root_mismatch"
    WRONG_RANDOMNESS = "wrong_randomness"
    HOTKEY_NOT_REGISTERED = "hotkey_not_registered"
    REGISTRATION_UNAVAILABLE = "registration_unavailable"
    WORKER_DROPPED = "worker_dropped"
    STALE_ROUND = "stale_round"
    FUTURE_ROUND = "future_round"
    PRECOMMIT_REQUIRED = "precommit_required"
    PRECOMMIT_INVALID = "precommit_invalid"
    PRECOMMIT_EXPIRED = "precommit_expired"
    REVEAL_NOT_SELECTED = "reveal_not_selected"
    PROTOCOL_MISMATCH = "protocol_mismatch"
    GENERATION_CONTRACT_MISMATCH = "generation_contract_mismatch"
    PROOF_CAPACITY_ABORT = "proof_capacity_abort"


class WindowState(str, Enum):
    """Current phase of a batch-driven window (v2.1)."""

    OPEN = "open"             # accepting /submit
    TRAINING = "training"     # GRPO step running, no submissions
    PUBLISHING = "publishing" # uploading weights, no submissions
    READY = "ready"           # checkpoint published; transient — back to OPEN once next window opens


class ServicePolicyAnnouncement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contract: dict[str, Any]
    schedule: dict[str, Any]
    checkpoint: dict[str, Any]
    supported_capabilities: list[str] = Field(..., min_length=1, max_length=32)
    pool_epoch: int = Field(..., ge=0, le=2**53 - 1, strict=True)
    pool_randomness: str = Field(..., pattern=r"^(?:[0-9a-f]{64})?$", max_length=64)

    @model_validator(mode="after")
    def _validate_contract_capabilities(self):
        from reliquary.protocol.service_catalogue import announced_contract
        from reliquary.protocol.service_schedule import ServiceSchedule
        contract = announced_contract({"contract": self.contract, "schedule": self.schedule})
        if contract.version != 2:
            raise ValueError("RL announcements carry service-contract/v2")
        if any(not 1 <= len(capability) <= 256 for capability in self.supported_capabilities):
            raise ValueError("bounded capabilities required")
        contract.require_capabilities(set(self.supported_capabilities))
        ServiceSchedule.from_dict(self.schedule, contract)
        if set(self.checkpoint) != {"checkpoint_n", "repo", "revision", "sha256"}:
            raise ValueError("announcement checkpoint fields are fixed")
        import re
        cp = self.checkpoint
        if type(cp["checkpoint_n"]) is not int or not 0 <= cp["checkpoint_n"] <= 2**53 - 1:
            raise ValueError("announcement checkpoint_n must be a non-negative integer")
        if not isinstance(cp["repo"], str) or not 1 <= len(cp["repo"]) <= 256:
            raise ValueError("announcement checkpoint repo must be a bounded string")
        if not isinstance(cp["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", cp["revision"]):
            raise ValueError("announcement checkpoint revision must be a 40-hex commit")
        if not isinstance(cp["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", cp["sha256"]):
            raise ValueError("announcement checkpoint sha256 must be 64-hex")
        from reliquary.protocol.service_contract import PUBLIC_SEED_POOL
        if any(env["sampling"]["kind"] == PUBLIC_SEED_POOL for env in contract.environments.values()) and not self.pool_randomness:
            raise ValueError("public pool requires its authoritative beacon")
        for name, env in contract.environments.items():
            if env["sampling"]["kind"] == PUBLIC_SEED_POOL and env["sampling"]["group_size"] != M_ROLLOUTS:
                # a miner always submits M_ROLLOUTS rollouts: any other group_size would admit no group at all
                raise ValueError(f"{name}: a public seed pool group_size must be M_ROLLOUTS")
        return self


def decode_cooldown_bitmap(
    encoded: str,
    prompt_range: tuple[int, int],
) -> set[int]:
    """Decode a ``bitset-v1`` cooldown and reject malformed wire data."""
    start, end = prompt_range
    if start < 0 or end < start:
        raise ValueError("invalid prompt range")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid cooldown bitmap base64") from exc
    expected_bytes = (end - start + 7) // 8
    if len(raw) != expected_bytes:
        raise ValueError(
            f"cooldown bitmap length {len(raw)} != expected {expected_bytes}"
        )
    if raw and (end - start) % 8:
        valid_bits = (1 << ((end - start) % 8)) - 1
        if raw[-1] & ~valid_bits:
            raise ValueError("cooldown bitmap has out-of-range bits set")
    return {
        start + byte_n * 8 + bit_n
        for byte_n, value in enumerate(raw)
        for bit_n in range(8)
        if value & (1 << bit_n)
    }


class MinerEnvironmentState(BaseModel):
    """One environment's bounded prompt slice in ``GET /miner-state``."""

    model_config = ConfigDict(extra="forbid")

    encoding: Literal["bitset-v1"] = "bitset-v1"
    prompt_range: tuple[int, int]
    cooldown_bitmap: str
    cooldown_count: int = Field(..., ge=0)
    accepting_submissions: bool
    admission_remaining: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _validate_bitmap(self) -> "MinerEnvironmentState":
        decoded = decode_cooldown_bitmap(
            self.cooldown_bitmap,
            self.prompt_range,
        )
        if len(decoded) != self.cooldown_count:
            raise ValueError("cooldown_count does not match cooldown_bitmap")
        return self

    def cooldown_prompts(self) -> set[int]:
        return decode_cooldown_bitmap(
            self.cooldown_bitmap,
            self.prompt_range,
        )


class MinerState(BaseModel):
    """Bounded all-environment control-plane state for reference miners."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    state: WindowState
    window_n: int = Field(..., ge=0)
    anchor_block: int = Field(..., ge=0)
    window_opened_at: float | None = Field(default=None, ge=0)
    submission_deadline_at: float | None = Field(default=None, ge=0)
    environments: dict[str, MinerEnvironmentState]
    checkpoint_n: int | None = Field(default=None, ge=0, strict=True)
    checkpoint_repo_id: str | None = None
    checkpoint_revision: str | None = None
    protocol_version: int | None = Field(default=None, ge=0)
    generation_profile_id: str | None = Field(default=None, max_length=64)
    generation_contract: dict[str, Any] | None = None
    randomness: str = ""
    service_policy: ServicePolicyAnnouncement | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_service_policy(self, handler):
        value = handler(self)
        if self.service_policy is None:
            value.pop("service_policy", None)
        return value


class Verdict(BaseModel):
    """One lifecycle verdict for a submission.

    Surfaced via the validator's ``GET /verdicts/{hotkey}`` endpoint so that
    miners can follow a submission without waiting for the R2 archive. The
    /submit response carries only ``SUBMITTED`` under the production worker
    path. Auction candidates then receive an admission verdict and, at seal,
    a second verdict whose non-null ``selected_for_batch`` and ``rewarded``
    fields report the final result. Deferred-proof failures are reported by
    that final record.
    """

    model_config = ConfigDict(extra="forbid")

    merkle_root: str = Field(..., pattern=r"^[0-9a-fA-F]{64}$")
    window_n: int | None = Field(default=None, ge=0)
    accepted: bool
    reason: RejectReason
    ts: float = Field(..., description="Unix timestamp when the verdict landed")
    stage: Literal["admission", "final"] | None = None
    is_final: bool | None = None
    selection_status: Literal["pending", "selected", "not_selected"] | None = None
    outcome_code: str | None = None
    explanation: str | None = None
    reason_details: dict[str, Any] | None = None
    environment: str | None = None
    prompt_idx: int | None = None
    checkpoint_revision: str | None = None
    receipt_id: str | None = None
    ordering_policy: str | None = None
    rank_scope: Literal["window_environment"] | None = None
    proof_status: Literal["pending", "passed", "failed", "skipped", "unknown"] | None = None
    proof_reason: str | None = None
    body_received_ts: float | None = None
    proof_recorded_ts: float | None = None
    proof_duration_seconds: float | None = None
    finalized_ts: float | None = None
    batch_index: int | None = None
    selection_target: int | None = None
    selected_count: int | None = None
    # Service runs only (detail fields, absent from every legacy verdict).
    service_contract_sha256: str | None = None
    service_purpose: str | None = None
    exploration_fraction: float | None = None
    # Optional observability fields. Older verdict records omit these; the
    # endpoint excludes nulls so legacy consumers keep seeing the compact shape
    # for entries that lack lifecycle metadata.
    # Group reward std — the same value the sigma gate compares against
    # SIGMA_MIN, and the `std` factor of the auction value std·(1-mean)^delta.
    # Without it a miner can see that a submission lost but not that its own
    # rollouts were too uniform to be worth anything.
    sigma: float | None = None
    arrival_ts: float | None = None
    decision_ts: float | None = None
    submitted_drand_round: int | None = None
    arrival_drand_round: int | None = None
    drand_delta: int | None = None
    seal_trigger_round: int | None = None
    prompt_hash_lead: str | None = None
    canonical_rank: int | None = None
    selection_reason: str | None = None
    accepted_into_pool: bool | None = None
    selected_for_batch: bool | None = None
    rewarded: bool | None = None
    reject_stage: str | None = None
    reject_reason: str | None = None
    # Which capacity ran out. ``batch_filled`` alone cannot distinguish "the
    # window sealed before you" from "your hotkey owes proof-failure debt" —
    # opposite miner-side fixes.
    batch_filled_reason: str | None = None
    payload_bytes: int | None = None
    body_read_ms: float | None = None
    body_parse_ms: float | None = None
    ingress_ms: float | None = None
    upload_precommit_status: str | None = None
    precommit_arrival_ts: float | None = None
    queue_wait_ms: float | None = None
    reward_grading_ms: float | None = None
    admission_commit_ms: float | None = None
    admission_prepare_ms: float | None = None
    commit_lock_wait_ms: float | None = None
    commit_ms: float | None = None
    verify_ms: float | None = None
    total_ms: float | None = None


SIGNED_EPISODE_SCHEMA = "reliquary/signed-episode/v1"
MAX_SIGNED_EPISODE_TURNS = 256


def is_signed_episode(meta: Any) -> bool:
    """Whether rollout metadata carries a signed episode (``rollout.episode`` of schema signed-episode/v1)."""
    episode = meta.get("episode") if isinstance(meta, dict) else None
    return isinstance(episode, dict) and episode.get("schema_version") == SIGNED_EPISODE_SCHEMA


def episode_without_transcript(meta: dict) -> dict:
    """A copy of rollout metadata whose signed episode no longer carries its transcript (what the proof
    worker, the training payload and every log receive)."""
    out = dict(meta)
    if is_signed_episode(out):
        out["episode"] = {key: value for key, value in out["episode"].items() if key != "transcript"}
    return out
