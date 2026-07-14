"""Pydantic v2 models for the miner→validator GRPO submission protocol.

The validator HTTP server (reliquary/validator/server.py) accepts these payloads
and the miner submitter (reliquary/miner/submitter.py) produces them.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    PrivateAttr,
    field_validator,
    model_validator,
)

from reliquary.constants import CHALLENGE_K, M_ROLLOUTS, MAX_NEW_TOKENS_PROTOCOL_CAP
from reliquary.shared.runtime_fingerprint import runtime_profile_hash


# ---------------------------------------------------------------------------
# v2 GRPO Market schemas
# ---------------------------------------------------------------------------


class RejectReason(str, Enum):
    """Canonical reject codes emitted by the v2 validator.

    Two values are success sentinels rather than rejects:
      - ``ACCEPTED``: validation pipeline ran to completion and the
        submission is in ``_valid`` (only used on the inline sync path,
        e.g. TestClient).
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


class WindowState(str, Enum):
    """Current phase of a batch-driven window (v2.1)."""

    OPEN = "open"             # accepting /submit
    TRAINING = "training"     # GRPO step running, no submissions
    PUBLISHING = "publishing" # uploading weights, no submissions
    READY = "ready"           # checkpoint published; transient — back to OPEN once next window opens


class RolloutSubmission(BaseModel):
    """A single rollout's payload: tokens, miner-claimed reward, GRAIL commit."""

    model_config = ConfigDict(extra="forbid")

    # Validator-derived training boundary.  The wire-level ``force_span`` lives
    # inside the signed commit, but remains untrusted until the batcher verifies
    # its content and position.  Keeping the trusted value private prevents a
    # miner-declared span from influencing GRPO loss masking.
    _validated_force_span: tuple[int, int] | None = PrivateAttr(default=None)
    _validated_termination_path: str | None = PrivateAttr(default=None)

    tokens: list[int] = Field(..., min_length=1)
    reward: FiniteFloat  # miner's local reward; validator re-checks it
    commit: dict[str, Any]
    env_name: str  # environment that generated this rollout (e.g. "openmathinstruct")

    @model_validator(mode="after")
    def _tokens_match_commit_tokens(self):
        commit_tokens = self.commit.get("tokens")
        if commit_tokens is not None and list(self.tokens) != list(commit_tokens):
            raise ValueError("tokens must match commit.tokens")
        return self


class RuntimeFingerprint(BaseModel):
    """Self-reported, nonce-bound inference runtime profile.

    This is calibration telemetry, not remote attestation. All strings are
    bounded because the profile crosses the public submission endpoint.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2] = 2
    python_version: str = Field(..., max_length=64)
    platform: str = Field(..., max_length=64)
    torch_version: str | None = Field(default=None, max_length=128)
    transformers_version: str | None = Field(default=None, max_length=128)
    fla_version: str | None = Field(default=None, max_length=128)
    fla_core_version: str | None = Field(default=None, max_length=128)
    causal_conv1d_version: str | None = Field(default=None, max_length=128)
    flash_attn_version: str | None = Field(default=None, max_length=128)
    cuda_version: str | None = Field(default=None, max_length=64)
    cuda_available: bool = False
    gpu_name: str | None = Field(default=None, max_length=160)
    compute_capability: str | None = Field(default=None, max_length=32)
    generation_dtype: str | None = Field(default=None, max_length=64)
    proof_dtype: str | None = Field(default=None, max_length=64)
    generation_attention_implementation: str | None = Field(
        default=None, max_length=128,
    )
    proof_attention_implementation: str | None = Field(
        default=None, max_length=128,
    )
    deterministic_algorithms: bool = False
    cudnn_benchmark: bool = False
    tf32_matmul: bool = False
    qwen35_fast_path_all: bool | None = None
    qwen35_fla_chunk: bool | None = None
    qwen35_fla_recurrent: bool | None = None
    qwen35_causal_conv_prefill: bool | None = None
    qwen35_causal_conv_update: bool | None = None
    profile_hash: str = Field(..., pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _profile_hash_matches(self):
        payload = self.model_dump(exclude={"profile_hash"})
        if self.profile_hash != runtime_profile_hash(payload):
            raise ValueError("runtime profile_hash does not match profile")
        return self


class RuntimeContract(BaseModel):
    """Capability response served separately from strict legacy `/state`."""

    model_config = ConfigDict(extra="forbid")

    telemetry_version: Literal[2] = 2
    validator_profile: RuntimeFingerprint


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
    # max_length bounds the value: it feeds the forced-seed derivation's 2-byte
    # length prefix (_lp), which overflows past 65535 bytes; a hash is short.
    checkpoint_hash: str = Field(..., min_length=0, max_length=256)
    # v2.3: drand quicknet round in progress when the miner sent the
    # submission. Validator rejects if this is not in
    # [current_round_at_receipt - 1, current_round_at_receipt]. The
    # accepted round bucket determines the submission's chronological
    # position at seal time. Default 0 = pre-v2.3 sentinel; the batcher
    # rejects 0 as STALE_ROUND in production but tests can still
    # construct legacy requests for the cooldown / cheap-check paths.
    drand_round: int = Field(default=0, ge=0)
    # Protocol version indicator for forced-seed sampling. Default 0 =
    # pre-forced-seed client; newer clients set this to indicate support
    # for seed-based randomness derivation.
    protocol_version: int = Field(default=0, ge=0)
    # Caller-chosen freshness nonce. Bound into the envelope signature
    # so a precomputed signature cannot be reused for a different
    # logical submission. The validator does not (currently) dedupe on
    # nonce — the merkle_root+hotkey+window pair already deduplicates at
    # the batcher level — but binding it here closes the trivial replay
    # vector against the rate-limit counter. Empty string is permitted
    # for back-compat with pre-envelope clients that the validator may
    # accept while the ``enforce_envelope_signature`` flag is off.
    nonce: str = Field(default="", max_length=128)
    # sr25519 hex signature of the canonical envelope binding (see
    # ``reliquary.protocol.signatures.build_envelope_binding``). Verified
    # at the TOP of /submit, before the per-hotkey rate-limit counter is
    # touched. Empty string is permitted as a back-compat sentinel; the
    # validator's ``enforce_envelope_signature`` flag decides whether an
    # empty sig is rejected as BAD_ENVELOPE_SIGNATURE or silently accepted.
    envelope_signature: str = Field(default="", pattern=r"^[0-9a-fA-F]*$")
    runtime_fingerprint: RuntimeFingerprint | None = None
    # Validator-only marker. It is absent from JSON and the public schema, so
    # adding it does not change the miner wire contract.
    _legacy_merkle_verified: bool = PrivateAttr(default=False)
    _logical_group_reservation: tuple[str, bytes] | None = PrivateAttr(
        default=None
    )

    @field_validator("rollouts")
    @classmethod
    def _rollout_count_is_M(cls, v):
        if len(v) != M_ROLLOUTS:
            raise ValueError(
                f"rollouts must have exactly {M_ROLLOUTS} entries, got {len(v)}"
            )
        return v

    @model_validator(mode="after")
    def _runtime_fingerprint_is_bound_to_nonce(self):
        if self.runtime_fingerprint is None:
            return self
        expected_suffix = f".{self.runtime_fingerprint.profile_hash}"
        if not self.nonce.endswith(expected_suffix):
            raise ValueError("runtime_fingerprint must be bound to nonce")
        return self


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
    # v2.3: drand beacon randomness for this window. Empty string between
    # OPEN and the first successful _set_window_randomness; miners loop on
    # empty until populated. Miners derive GRAIL commitments off this
    # value rather than recomputing locally, which guarantees byte-for-byte
    # agreement with the validator's verify path.
    randomness: str = ""


class Verdict(BaseModel):
    """A single recorded verdict for a submission the validator has either
    accepted or rejected after running the full verification pipeline.

    Surfaced via the validator's ``GET /verdicts/{hotkey}`` endpoint so that
    miners can learn the REAL outcome of each submission within seconds of
    it being decided, instead of having to wait minutes for the R2 archive
    upload. The /submit response (``BatchSubmissionResponse``) carries only
    the provisional ``SUBMITTED`` sentinel under the production worker path
    — the actual verdict (``ACCEPTED`` / ``GRAIL_FAIL`` / ``WRONG_RANDOMNESS``
    / etc.) lands here once the worker drains the submission.
    """

    model_config = ConfigDict(extra="forbid")

    merkle_root: str = Field(..., pattern=r"^[0-9a-fA-F]{64}$")
    window_n: int | None = Field(default=None, ge=0)
    accepted: bool
    reason: RejectReason
    ts: float = Field(..., description="Unix timestamp when the verdict landed")
    # Optional observability fields. Older verdict records omit these; the
    # endpoint excludes nulls so legacy consumers keep seeing the compact shape
    # for entries that lack lifecycle metadata.
    arrival_ts: float | None = None
    decision_ts: float | None = None
    submitted_drand_round: int | None = None
    arrival_drand_round: int | None = None
    drand_delta: int | None = None
    seal_trigger_round: int | None = None
    prompt_hash_lead: str | None = None
    canonical_rank: int | None = None
    accepted_into_pool: bool | None = None
    selected_for_batch: bool | None = None
    rewarded: bool | None = None
    reject_stage: str | None = None
    reject_reason: str | None = None
    queue_wait_ms: float | None = None
    verify_ms: float | None = None
    total_ms: float | None = None


class VerdictsResponse(BaseModel):
    """Reply body of ``GET /verdicts/{hotkey}``: list of recent verdicts for
    one miner hotkey, ordered by timestamp ascending. Empty list is a
    valid response — it just means the hotkey hasn't fired anything the
    validator has remembered (capacity-limited ring buffer)."""

    model_config = ConfigDict(extra="forbid")

    verdicts: list[Verdict]


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
    total_reward: FiniteFloat
    advantage: FiniteFloat
    token_logprobs: list[FiniteFloat]
    # BFT: forced=True when the rollout was force-terminated at the thinking
    # budget; force_span = [start, end] token indices of the injected FORCE
    # template (validated byte-exact by the validator carve-out).
    forced: bool = False
    force_span: list[int] | None = None
    # Validator-set (not miner-claimed): cap-truncated / non-terminating. Feeds
    # the overlong side of the reward shaping.
    truncated: bool = False


class CommitModel(BaseModel):
    """The inner ``commit`` dict shipped by the miner inside ``RolloutSubmission``.

    Validated explicitly at the top of ``GrpoWindowBatcher._accept_locked``
    rather than via Pydantic on ``RolloutSubmission.commit`` — keeps the
    failure path inside the batcher's reject-counts telemetry.
    """

    model_config = ConfigDict(extra="forbid")

    tokens: list[int] = Field(..., min_length=CHALLENGE_K)
    commitments: list[dict]
    proof_version: Literal["v7"]
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
