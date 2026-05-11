"""Pydantic v2 models for the miner→validator GRPO submission protocol.

The validator HTTP server (reliquary/validator/server.py) accepts these payloads
and the miner submitter (reliquary/miner/submitter.py) produces them.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from reliquary.constants import CHALLENGE_K, M_ROLLOUTS, MAX_NEW_TOKENS_PROTOCOL_CAP


# ---------------------------------------------------------------------------
# v2 GRPO Market schemas
# ---------------------------------------------------------------------------


class RejectReason(str, Enum):
    """Canonical reject codes emitted by the v2 validator.

    ``ACCEPTED`` is a sentinel used in success responses (``accepted=True``).
    All other values are mutually exclusive; only the first failure reason
    is reported per submission.
    """

    ACCEPTED = "accepted"
    BAD_SIGNATURE = "bad_signature"
    BAD_PROMPT_IDX = "bad_prompt_idx"
    PROMPT_MISMATCH = "prompt_mismatch"
    DISTRIBUTION_SUSPICIOUS = "distribution_suspicious"
    PROMPT_IN_COOLDOWN = "prompt_in_cooldown"
    SUPERSEDED = "superseded"
    GRAIL_FAIL = "grail_fail"
    LOGPROB_MISMATCH = "logprob_mismatch"
    REWARD_MISMATCH = "reward_mismatch"
    OUT_OF_ZONE = "out_of_zone"
    WRONG_ROLLOUT_COUNT = "wrong_rollout_count"
    WINDOW_MISMATCH = "window_mismatch"
    WINDOW_NOT_ACTIVE = "window_not_active"
    WINDOW_BUSY = "window_busy"
    BAD_SCHEMA = "bad_schema"
    BAD_TOKENS = "bad_tokens"
    BAD_TERMINATION = "bad_termination"
    WRONG_CHECKPOINT = "wrong_checkpoint"


class WindowState(str, Enum):
    """Current phase of a batch-driven window (v2.1)."""

    OPEN = "open"             # accepting /submit
    TRAINING = "training"     # GRPO step running, no submissions
    PUBLISHING = "publishing" # uploading weights, no submissions
    READY = "ready"           # checkpoint published; transient — back to OPEN once next window opens


class RolloutSubmission(BaseModel):
    """A single rollout's payload: tokens, miner-claimed reward, GRAIL commit."""

    model_config = ConfigDict(extra="forbid")

    tokens: list[int] = Field(..., min_length=1)
    reward: float  # miner's local env.compute_reward value; validator re-checks
    commit: dict[str, Any]


class BatchSubmissionRequest(BaseModel):
    """v2 miner→validator payload: one group of M rollouts on one prompt."""

    model_config = ConfigDict(extra="forbid")

    miner_hotkey: str = Field(..., min_length=1)
    prompt_idx: int = Field(..., ge=0)
    window_start: int = Field(..., ge=0)
    merkle_root: str = Field(..., pattern=r"^[0-9a-fA-F]{64}$")
    rollouts: list[RolloutSubmission]
    # Empty string is allowed as a bootstrap sentinel: before the validator
    # publishes its first checkpoint (checkpoint_n=0, revision=None) miners
    # have no hash to cite. The batcher disables the gate in that case.
    checkpoint_hash: str = Field(..., min_length=0)

    @field_validator("rollouts")
    @classmethod
    def _rollout_count_is_M(cls, v):
        if len(v) != M_ROLLOUTS:
            raise ValueError(
                f"rollouts must have exactly {M_ROLLOUTS} entries, got {len(v)}"
            )
        return v


class BatchSubmissionResponse(BaseModel):
    """Validator verdict on a submission."""

    model_config = ConfigDict(extra="forbid")

    accepted: bool
    reason: RejectReason


class GrpoBatchState(BaseModel):
    """Live window state for miners polling ``/state`` (v2.1)."""

    model_config = ConfigDict(extra="forbid")

    state: WindowState
    window_n: int = Field(..., ge=0)
    anchor_block: int = Field(..., ge=0)
    cooldown_prompts: list[int] = Field(default_factory=list)
    valid_submissions: int = Field(..., ge=0)
    checkpoint_n: int = Field(..., ge=0)
    checkpoint_repo_id: str | None = None
    checkpoint_revision: str | None = None


class ModelInfo(BaseModel):
    """Identifies the model the miner ran."""

    model_config = ConfigDict(extra="forbid")

    name: str
    layer_index: int


class BeaconInfo(BaseModel):
    """Drand beacon randomness used for this commit."""

    model_config = ConfigDict(extra="forbid")

    randomness: str = Field(..., pattern=r"^[0-9a-fA-F]+$")


class RolloutMetadata(BaseModel):
    """Per-rollout meta: lengths, success flag, claimed reward, logprobs."""

    model_config = ConfigDict(extra="forbid")

    prompt_length: int = Field(..., ge=0)
    completion_length: int = Field(..., gt=0, le=MAX_NEW_TOKENS_PROTOCOL_CAP)
    success: bool
    total_reward: float
    advantage: float
    token_logprobs: list[float]


class CommitModel(BaseModel):
    """The inner ``commit`` dict shipped by the miner inside ``RolloutSubmission``.

    Validated explicitly at the top of ``GrpoWindowBatcher._accept_locked``
    rather than via Pydantic on ``RolloutSubmission.commit`` — keeps the
    failure path inside the batcher's reject-counts telemetry.
    """

    model_config = ConfigDict(extra="forbid")

    tokens: list[int] = Field(..., min_length=CHALLENGE_K)
    commitments: list[dict]
    proof_version: Literal["v5"]
    model: ModelInfo
    signature: str = Field(..., pattern=r"^[0-9a-fA-F]+$")
    beacon: BeaconInfo
    rollout: RolloutMetadata

    @field_validator("commitments")
    @classmethod
    def _commitments_len_matches_tokens(cls, v, info):
        if "tokens" not in info.data:
            return v   # tokens itself failed validation; let that error stand alone
        tokens = info.data["tokens"]
        if len(v) != len(tokens):
            raise ValueError(
                f"commitments length {len(v)} must equal tokens length {len(tokens)}"
            )
        return v

    @field_validator("rollout")
    @classmethod
    def _lengths_consistent(cls, v, info):
        if "tokens" not in info.data:
            return v   # tokens itself failed validation; let that error stand alone
        tokens = info.data["tokens"]
        if v.prompt_length + v.completion_length != len(tokens):
            raise ValueError(
                f"prompt_length({v.prompt_length}) + "
                f"completion_length({v.completion_length}) must equal "
                f"len(tokens)={len(tokens)}"
            )
        # Two layouts are accepted, matching ``verify_logprobs_claim`` in
        # ``validator/verifier.py``:
        #   * full-sequence: len == len(tokens), prompt entries ignored
        #   * completion-only: len == completion_length, indexed from 0
        # Forcing one layout here would silently reject every miner that
        # ships completion-only — including the miner code in this very repo.
        if len(v.token_logprobs) not in (len(tokens), v.completion_length):
            raise ValueError(
                f"token_logprobs length {len(v.token_logprobs)} must equal "
                f"either tokens length {len(tokens)} (full-sequence) "
                f"or completion_length {v.completion_length} (completion-only)"
            )
        return v
