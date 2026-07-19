"""FastAPI server: receives v2 GRPO market submissions, exposes window state.

/submit drops requests on an asyncio queue (worker thread drains it off the
event loop so GRAIL verification doesn't block HTTP responses). Under
TestClient (no worker running), /submit runs synchronously so tests see
the real verdict.

/verdicts/{hotkey} surfaces the real per-submission verdicts (accept /
specific reject reason) that ``/submit`` cannot return in real time. Under
the production worker path /submit replies with a provisional ``SUBMITTED``
sentinel and the actual verdict lands in this endpoint a few seconds later,
once the worker has run the full verification pipeline. Miners learn the
truth without having to wait minutes for the R2 archive upload.
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import importlib.metadata
import logging
import multiprocessing
import numbers
import secrets
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
import uvicorn

from reliquary.constants import (
    B_BATCH,
    BOOTSTRAP_MAX_TRUNCATED_PER_SUBMISSION,
    DRAND_ROUND_BACKWARD_TOLERANCE,
    DIFFICULTY_AUCTION_DELTA,
    DIFFICULTY_AUCTION_ENFORCE,
    DIFFICULTY_AUCTION_ENVIRONMENTS,
    DIFFICULTY_AUCTION_SHADOW_ENABLED,
    DIFFICULTY_AUCTION_SHADOW_ENVIRONMENTS,
    DIFFICULTY_AUCTION_SHADOW_MAX_CANDIDATES,
    DIFFICULTY_AUCTION_SHADOW_MAX_SLOTS_PER_OPERATOR,
    ENFORCE_ENVELOPE_SIGNATURE,
    FORCED_SEED_CDF_BOUNDARY_EPSILON,
    FORCED_SEED_CDF_ENFORCE,
    FORCED_SEED_CONSISTENCY_FLOOR,
    FORCED_SEED_ENFORCE,
    FORCED_SEED_PROTOCOL_VERSION,
    FORCED_SEED_ROLLOUT_FLOOR,
    LEGACY_MERKLE_ROOT_ENFORCE,
    MATH_ADMISSION_WORKERS,
    MAX_BAD_ENVELOPE_PER_HOTKEY_PER_WINDOW,
    MAX_EXPENSIVE_PROOF_FAILURES_PER_OPERATOR_PER_WINDOW,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    MAX_PENDING_PROOF_QUEUE_DEPTH,
    MAX_PENDING_SUBMISSION_BYTES_PER_ENV,
    MAX_PENDING_SUBMISSION_BYTES_PER_HOTKEY,
    MAX_POST_TRIGGER_PROOF_CANDIDATES,
    MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW,
    MAX_PROOF_WALL_SECONDS,
    MAX_SUBMISSION_PAYLOAD_BYTES,
    MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW,
    MAX_SUBMISSIONS_PER_PROMPT,
    MAX_TRUNCATED_PER_SUBMISSION,
    REGISTERED_HOTKEY_CACHE_TTL_SECONDS,
    REGISTERED_HOTKEY_STALE_GRACE_SECONDS,
    SPARSE_VALID_IDLE_MIN_DISTINCT_PROMPTS,
    SPARSE_VALID_IDLE_SEAL_SECONDS,
    SPARSE_VALID_MAX_WINDOW_SECONDS,
    SUBMISSION_UPLOAD_GRACE_SECONDS,
    VALIDATOR_HTTP_PORT,
    CODE_ADMISSION_WORKERS,
    CODE_ADMISSION_WALL_SECONDS,
    ADMISSION_PROCESS_MAX_TASKS,
    MATH_ADMISSION_WALL_SECONDS,
)
from reliquary.environment.virtual_parquet import PromptSourceUnavailable
from reliquary.protocol.legacy_merkle import (
    legacy_submission_merkle_matches,
)
from reliquary.protocol.signatures import (
    verify_envelope_signature,
    verify_precommit_signature,
)
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    BatchSubmissionResponse,
    CommitModel,
    GrpoBatchState,
    RejectReason,
    RuntimeContract,
    RuntimeFingerprint,
    SubmissionPrecommitRequest,
    SubmissionPrecommitResponse,
    Verdict,
    VerdictsResponse,
)
from reliquary.protocol.tokens import verify_tokens
from reliquary.shared.hf_compat import (
    resolve_max_context_length,
    resolve_vocab_size,
)
from reliquary.shared.modeling import (
    force_close_token_ids,
    resolve_eos_token_ids,
    think_close_token_ids,
)
from reliquary.shared.runtime_fingerprint import collect_runtime_fingerprint
from reliquary.validator.admission import (
    AdmissionContext,
    AdmissionReceiptBinding,
    PreparedSubmission,
    initialize_admission_worker,
    prepare_submission,
)
from reliquary.validator.batcher import GrpoWindowBatcher
from reliquary.validator.dedup import compute_rollout_hash
from reliquary.validator.observability import (
    DrandRoundObservation,
    SubmitTelemetry,
    classify_drand_round,
    log_submission_stage,
    runtime_revision,
)
from reliquary.validator.verifier import (
    is_forced_bft_cap_termination,
    is_natural_bft_cap_candidate,
    rewards_std,
    validate_force_span,
)

logger = logging.getLogger(__name__)

PRECOMMIT_HEADER = "X-Reliquary-Precommit"
MAX_PRECOMMIT_BODY_BYTES = 16 * 1024


@dataclass
class _UploadPrecommitReceipt:
    receipt_id: str
    precommit_signature: str
    miner_hotkey: str
    prompt_idx: int
    window_start: int
    merkle_root: str
    checkpoint_hash: str
    environment: str
    payload_bytes: int
    payload_sha256: str
    drand_round: int
    protocol_version: int
    nonce: str
    expires_at_wall: float
    precommit_arrival_ts: float
    drand_observation: DrandRoundObservation
    batcher: Any
    consumed: bool = False
    outcome: BatchSubmissionResponse | None = None
    terminal: bool = False
    terminal_recorded: bool = False


@dataclass(frozen=True)
class _QueuedAuctionSubmission:
    raw_body: bytes
    receipt: _UploadPrecommitReceipt
    batcher: Any
    telemetry: SubmitTelemetry
    enqueued_monotonic: float


# How many recent verdicts to remember per hotkey. Bounded so the
# ring buffer can't grow without limit if a misbehaving miner spams.
# At ~250 B per verdict × 200 entries × ~50 hotkeys ≈ 2.5 MB — cheap.
VERDICT_CAP_PER_HOTKEY = 200


def _chain_client_fingerprint() -> dict[str, str | None]:
    """Return the chain codec versions that determine SCALE compatibility."""
    versions: dict[str, str | None] = {}
    for field, distribution in (
        ("bittensor_version", "bittensor"),
        ("async_substrate_interface_version", "async-substrate-interface"),
        ("cyscale_version", "cyscale"),
        ("legacy_scalecodec_version", "scalecodec"),
    ):
        try:
            versions[field] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[field] = None
    return versions


def _is_mock_like(value: Any) -> bool:
    """Return True for unittest.mock objects.

    The server's unit tests use loose MagicMocks as batchers; touching nested
    attributes on them auto-creates truthy mock objects. Production batchers
    carry real model/tokenizer/config objects, so skip optional preflight pieces
    when the object is clearly a mock.
    """
    return type(value).__module__.startswith("unittest.mock")


def _serialized_submission_bytes(request: BatchSubmissionRequest) -> int:
    """Canonical post-parse size used by queue-memory accounting."""
    return len(request.model_dump_json().encode("utf-8"))


class _SubmissionBodyLimitMiddleware:
    """Reject oversized /submit bodies before FastAPI parses their JSON.

    The content-length fast path handles normal clients. Wrapping ``receive``
    also covers chunked transfer encoding, where trusting a missing header would
    otherwise let an attacker allocate an unbounded request before the route's
    post-parse accounting runs.
    """

    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    @staticmethod
    async def _reject(scope: dict[str, Any], receive: Any, send: Any) -> None:
        response = JSONResponse(
            status_code=413,
            content={"detail": "submission_payload_too_large"},
            headers={"Connection": "close"},
        )
        await response(scope, receive, send)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        path = scope.get("path")
        if scope.get("type") != "http" or path not in {
            "/submit",
            "/submit/precommit",
        }:
            await self.app(scope, receive, send)
            return

        max_bytes = (
            MAX_PRECOMMIT_BODY_BYTES
            if path == "/submit/precommit"
            else self.max_bytes
        )

        headers = dict(scope.get("headers", ()))
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                content_length = int(raw_length)
            except ValueError:
                content_length = -1
            if content_length > max_bytes:
                await self._reject(scope, receive, send)
                return

        received = 0
        body_hasher = hashlib.sha256()
        over_limit = False
        buffered_messages: list[dict[str, Any]] = []

        async def limited_receive():
            nonlocal received, over_limit
            message = await receive()
            if message.get("type") == "http.request":
                state = scope.setdefault("state", {})
                state.setdefault("body_receive_started_at", time.time())
                chunk = message.get("body", b"")
                received += len(chunk)
                body_hasher.update(chunk)
                if received > max_bytes:
                    over_limit = True
                    # Complete the downstream body stream without forwarding
                    # the over-limit chunk. FastAPI may turn that truncated JSON
                    # into a 400; ``buffered_send`` keeps that response private
                    # and we replace it with the protocol's explicit 413 below.
                    return {
                        "type": "http.request",
                        "body": b"",
                        "more_body": False,
                    }
                if not message.get("more_body", False):
                    state["body_bytes_received"] = received
                    state["body_sha256"] = body_hasher.hexdigest()
                    state["body_completed_at"] = time.time()
            return message

        async def buffered_send(message):
            buffered_messages.append(message)

        try:
            await self.app(scope, limited_receive, buffered_send)
        except Exception:
            if not over_limit:
                raise

        if over_limit:
            await self._reject(scope, receive, send)
            return
        for message in buffered_messages:
            await send(message)


def _proof_free_model_config(batcher: Any) -> Any | None:
    model = getattr(batcher, "model", None)
    if model is None or _is_mock_like(model):
        return None
    config = getattr(model, "config", None)
    if config is None or _is_mock_like(config):
        return None
    return config


def _coerce_eos_set(eos_ids: Any) -> set[int] | None:
    if eos_ids is None or _is_mock_like(eos_ids):
        return None
    if isinstance(eos_ids, numbers.Integral) and not isinstance(eos_ids, bool):
        return {int(eos_ids)}
    if isinstance(eos_ids, (list, tuple, set)):
        eos_set: set[int] = set()
        for eos_id in eos_ids:
            if (
                isinstance(eos_id, numbers.Integral)
                and not isinstance(eos_id, bool)
            ):
                eos_set.add(int(eos_id))
        return eos_set or None
    return None


def _proof_free_eos_set(batcher: Any) -> set[int] | None:
    model = getattr(batcher, "model", None)
    if model is not None and _is_mock_like(model):
        model = None
    tokenizer = getattr(batcher, "tokenizer", None)
    if tokenizer is not None and _is_mock_like(tokenizer):
        tokenizer = None

    eos = resolve_eos_token_ids(model, tokenizer)
    return eos or None


def _proof_free_bootstrap(batcher: Any) -> bool:
    bootstrap = getattr(batcher, "bootstrap", False)
    if _is_mock_like(bootstrap):
        return False
    return bool(bootstrap)


def _proof_free_force_span_validator(batcher: Any):
    """Resolve a cheap ``validate_force_span`` closure for the preflight, or
    ``None`` when no real tokenizer is available (mock/test batchers).

    The preflight grants a forced-BFT rollout an exemption from the truncation
    budget based on the *miner-supplied* ``force_span``. Without a structural
    check a node can mark ordinary cap-truncation spam ``forced=True`` with a
    plausible span to bypass this cheap gate and force a scarce GPU proof slot.
    Running the same byte-exact/position-pinned ``validate_force_span`` the full
    path applies keeps the net accept/reject decision identical while rejecting
    fakes before any GPU work.
    """
    tokenizer = getattr(batcher, "tokenizer", None)
    if tokenizer is None or _is_mock_like(tokenizer):
        return None
    try:
        from reliquary.constants import BFT_THINKING_BUDGET
        from reliquary.shared.modeling import force_close_token_ids

        canonical_force_ids = force_close_token_ids(tokenizer)
    except Exception:
        return None
    if not canonical_force_ids:
        return None
    think_close_ids = {int(canonical_force_ids[0])}

    def _validate(tokens: list[int], meta: dict) -> bool:
        ok, _ = validate_force_span(
            tokens,
            meta,
            canonical_force_ids,
            int(meta.get("prompt_length", 0)),
            thinking_budget=BFT_THINKING_BUDGET,
            think_close_ids=think_close_ids,
        )
        return ok

    return _validate


def _proof_free_submission_reject(
    request: BatchSubmissionRequest,
    batcher: Any,
) -> tuple[RejectReason | None, str | None]:
    """Reject structurally impossible submissions before proof admission.

    This deliberately avoids trusting claimed probabilities as proof of
    correctness. A rollout ending in EOS still needs the normal GRAIL path to
    recompute p_stop/logprobs. The cheap path only catches cases the expensive
    verifier would reject after burning a scarce proof slot: malformed commits,
    invalid token envelopes, EOS padding, and non-cap completions that never
    emitted EOS.
    """
    for rollout in request.rollouts:
        try:
            CommitModel.model_validate(rollout.commit)
        except ValidationError:
            return RejectReason.BAD_SCHEMA, "schema"
        if list(rollout.tokens) != list(rollout.commit["tokens"]):
            return RejectReason.TOKENS_MISMATCH, "token_invariant"
        verify_signature = getattr(batcher, "_verify_signature", None)
        if (
            callable(verify_signature)
            and not _is_mock_like(verify_signature)
            and not verify_signature(rollout.commit, request.miner_hotkey)
        ):
            return RejectReason.BAD_SIGNATURE, "rollout_signature"
        claimed_randomness = (
            (rollout.commit.get("beacon") or {}).get("randomness", "")
        )
        expected_randomness = getattr(batcher, "randomness", "")
        if (
            not _is_mock_like(expected_randomness)
            and bool(expected_randomness)
            and claimed_randomness != expected_randomness
        ):
            return RejectReason.WRONG_RANDOMNESS, "randomness"

    model_config = _proof_free_model_config(batcher)
    canonical_prompt_tokens: list[int] | None = None
    canonical_prompt_fn = getattr(batcher, "_canonical_prompt_tokens", None)
    if (
        callable(canonical_prompt_fn)
        and not _is_mock_like(canonical_prompt_fn)
    ):
        canonical_prompt_tokens = list(canonical_prompt_fn(request.prompt_idx))

    if model_config is not None:
        for rollout in request.rollouts:
            if not verify_tokens(rollout.commit["tokens"], model_config):
                return RejectReason.BAD_TOKENS, "tokens"

            if canonical_prompt_tokens is not None:
                rollout_meta = rollout.commit.get("rollout", {}) or {}
                miner_prompt_len = int(rollout_meta.get("prompt_length", 0))
                miner_prompt_tokens = list(rollout.commit.get("tokens", []))[
                    :miner_prompt_len
                ]
                if miner_prompt_tokens != canonical_prompt_tokens:
                    return RejectReason.PROMPT_MISMATCH, "prompt_binding"

    hash_set = getattr(batcher, "_hash_set", None)
    if hash_set is not None and not _is_mock_like(hash_set):
        local_seen: set[bytes] = set()
        for rollout in request.rollouts:
            try:
                rollout_hash = compute_rollout_hash(rollout.commit["tokens"])
            except ValueError:
                return RejectReason.BAD_TOKENS, "tokens"
            if rollout_hash in local_seen or rollout_hash in hash_set:
                return RejectReason.HASH_DUPLICATE, "dedup"
            local_seen.add(rollout_hash)

    env = getattr(batcher, "env", None)
    validator_scored_reward = bool(
        getattr(env, "validator_authoritative_reward", False)
    )
    if not _is_mock_like(batcher) and not validator_scored_reward:
        rewards = [float(rollout.reward) for rollout in request.rollouts]
        from reliquary.validator.verifier import is_in_zone
        if not is_in_zone(
            rewards_std(rewards),
            bootstrap=bool(getattr(batcher, "bootstrap", False)),
        ):
            return RejectReason.OUT_OF_ZONE, "zone"

    eos_set = _proof_free_eos_set(batcher)
    if not eos_set:
        return None, None

    max_truncated_per_submission = (
        BOOTSTRAP_MAX_TRUNCATED_PER_SUBMISSION
        if _proof_free_bootstrap(batcher)
        else MAX_TRUNCATED_PER_SUBMISSION
    )
    truncated_count = 0
    validate_forced_span = _proof_free_force_span_validator(batcher)

    for rollout in request.rollouts:
        commit = rollout.commit
        tokens = list(commit.get("tokens") or [])
        meta = commit.get("rollout", {}) or {}
        prompt_length = int(meta.get("prompt_length", 0))
        completion_length = int(meta.get("completion_length", 0))
        completion = tokens[prompt_length: prompt_length + completion_length]
        if not completion:
            return RejectReason.BAD_SCHEMA, "schema"

        eos_positions = [
            idx for idx, token in enumerate(completion) if int(token) in eos_set
        ]
        if eos_positions:
            if len(eos_positions) > 1 or eos_positions[0] != len(completion) - 1:
                return RejectReason.BAD_TERMINATION, "termination_preflight"
            # No claimed-logprob floor here. It could only ever bind on miners
            # who report honestly (a forger simply claims a comfortable value and
            # is caught by the GPU p_stop check anyway), and it fired BEFORE the
            # forced-seed terminal-pick escape in verify_termination could rescue
            # a legally-drawn improbable stop. Termination is decided on the
            # validator's own logits, not on a number the miner sent.
            continue

        total_length = prompt_length + completion_length
        is_math = rollout.env_name == "openmathinstruct"
        if is_math and is_forced_bft_cap_termination(commit):
            # Only exempt from the truncation budget if the claimed FORCE span
            # is structurally valid (byte-exact, position-pinned). A fake-forced
            # rollout is rejected here instead of after a GPU proof; the full
            # validate_force_span in the batcher would reject it regardless, so
            # the net decision is unchanged.
            if (
                validate_forced_span is not None
                and not validate_forced_span(tokens, meta)
            ):
                return RejectReason.TOKEN_TAMPERED, "force_span_preflight"
            continue
        if is_natural_bft_cap_candidate(
            commit,
            getattr(batcher, "tokenizer", None),
            env_name=rollout.env_name,
        ):
            continue
        if total_length < MAX_NEW_TOKENS_PROTOCOL_CAP:
            return RejectReason.BAD_TERMINATION, "termination_preflight"

        truncated_count += 1
        if truncated_count > max_truncated_per_submission:
            return RejectReason.BAD_TERMINATION, "termination_preflight"

    return None, None


class _Health(BaseModel):
    status: str
    active_window: int | None
    image_revision: str | None = None
    runtime_fingerprint: dict[str, Any] = Field(default_factory=dict)
    chain_client_fingerprint: dict[str, str | None] = Field(
        default_factory=dict
    )
    app_started_at: float
    current_validator_state: str
    current_window_n: int | None = None
    last_committed_window_n: int = 0
    candidate_window_n: int | None = None
    window_preparation_stage: str | None = None
    last_window_preparation_failure: dict[str, Any] | None = None
    window_preparation_failures_total: int = 0
    window_preparation_failures_by_stage: dict[str, int] = Field(
        default_factory=dict
    )
    window_preparation_failures_by_error: dict[str, int] = Field(
        default_factory=dict
    )
    current_quicknet_drand_round: int | None = None
    current_window_open_ts: float | None = None
    current_window_open_drand_round: int | None = None
    seal_trigger_round: int | None = None
    drand_round_backward_tolerance: int
    upload_precommit_enabled: bool = True
    submission_upload_grace_seconds: float = SUBMISSION_UPLOAD_GRACE_SECONDS
    batch_size: int
    queue_depth: int | None = None
    queue_depth_by_environment: dict[str, int] = Field(default_factory=dict)
    admission_workers_by_environment: dict[str, int] = Field(
        default_factory=lambda: {
            "openmathinstruct": MATH_ADMISSION_WORKERS,
            "opencodeinstruct": CODE_ADMISSION_WORKERS,
        }
    )
    admission_pool_state_by_environment: dict[str, dict[str, Any]] = Field(
        default_factory=dict
    )
    proof_verification_inflight: int | None = None
    proof_verification_inflight_by_environment: dict[str, int] = Field(
        default_factory=dict
    )
    event_loop_lag_ms: dict[str, float | None] = Field(default_factory=dict)
    endpoint_latency_ms: dict[str, dict[str, float | None]] = Field(
        default_factory=dict
    )
    admission_latency_ms_by_environment: dict[
        str, dict[str, dict[str, float | None]]
    ] = Field(default_factory=dict)
    valid_submissions_count: int | None = None
    distinct_valid_prompt_count: int | None = None
    last_valid_submission_ts: float | None = None
    seconds_since_last_valid_submission: float | None = None
    proof_admission_count: int | None = None
    proof_grading_attempts: int | None = None
    pending_proof_reservations: int | None = None
    inflight_proof_reservations: int | None = None
    reserved_payload_bytes: int | None = None
    pending_payload_bytes: int | None = None
    inflight_payload_bytes: int | None = None
    retained_payload_bytes: int | None = None
    max_submission_payload_bytes: int = MAX_SUBMISSION_PAYLOAD_BYTES
    max_pending_submission_bytes_per_hotkey: int = (
        MAX_PENDING_SUBMISSION_BYTES_PER_HOTKEY
    )
    max_pending_submission_bytes_per_env: int = (
        MAX_PENDING_SUBMISSION_BYTES_PER_ENV
    )
    difficulty_auction_enforced: bool = DIFFICULTY_AUCTION_ENFORCE
    difficulty_auction_environments: list[str] = Field(
        default_factory=lambda: list(DIFFICULTY_AUCTION_ENVIRONMENTS)
    )
    difficulty_auction_proof_attempt_limit: int = (
        MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW
    )
    difficulty_auction_proof_wall_limit_seconds: float = MAX_PROOF_WALL_SECONDS
    difficulty_auction_proof_wall_elapsed_seconds: float | None = None
    difficulty_auction_proof_wall_exhausted: bool | None = None
    window_environments: dict[str, dict[str, Any]] = Field(
        default_factory=dict
    )
    logical_group_reservations: int = 0
    logical_group_duplicate_rejects: int = 0
    logical_group_dedup_by_environment: dict[str, dict[str, int]] = Field(
        default_factory=dict
    )
    grader_failures_by_environment: dict[str, dict[str, int]] = Field(
        default_factory=dict
    )
    post_trigger_proof_admission_count: int | None = None
    post_trigger_proof_admission_limit: int = MAX_POST_TRIGGER_PROOF_CANDIDATES
    sparse_valid_idle_seal_seconds: float = SPARSE_VALID_IDLE_SEAL_SECONDS
    sparse_valid_idle_min_distinct_prompts: int = (
        SPARSE_VALID_IDLE_MIN_DISTINCT_PROMPTS
    )
    sparse_valid_max_window_seconds: float = SPARSE_VALID_MAX_WINDOW_SECONDS
    expensive_proof_failures_by_hotkey: dict[str, int] = Field(
        default_factory=dict
    )
    expensive_proof_failures_by_operator: dict[str, int] = Field(
        default_factory=dict
    )
    max_expensive_proof_failures_per_operator_per_window: int = (
        MAX_EXPENSIVE_PROOF_FAILURES_PER_OPERATOR_PER_WINDOW
    )
    checkpoint_repo_id: str | None = None
    checkpoint_revision: str | None = None
    recent_reject_counts_by_reason: dict[str, int]
    rewarded_but_not_selected_by_hotkey: dict[str, int] = Field(
        default_factory=dict
    )
    registration_gate_enforced: bool = False
    registered_hotkey_count: int | None = None
    registered_operator_mapping_count: int = 0
    registered_operator_mapping_complete: bool | None = None
    registration_cache_age_seconds: float | None = None
    registration_cache_stale: bool | None = None
    registration_cache_usable: bool | None = None
    registration_cache_next_refresh_ts: float | None = None
    registration_cache_refresh_attempts_total: int = 0
    registration_cache_refresh_successes_total: int = 0
    registration_cache_refresh_failures_total: int = 0
    registration_cache_last_refresh_attempt_ts: float | None = None
    registration_cache_last_refresh_success_ts: float | None = None
    registration_cache_last_refresh_failure_ts: float | None = None
    registration_cache_last_refresh_failure_type: str | None = None
    registration_cache_last_refresh_failure_reason: str | None = None
    registration_cache_last_refresh_reason: str | None = None
    training_accumulator_checkpoint_revision: str | None = None
    training_accumulator_targets: dict[str, int] = Field(default_factory=dict)
    training_accumulator_counts: dict[str, int] = Field(default_factory=dict)
    training_accumulator_ready: bool = False
    training_trained_windows_since_publish: int = 0
    training_checkpoint_publish_interval: int = 0
    training_checkpoint_publication_pending: bool = False
    forced_seed_enforced: bool = FORCED_SEED_ENFORCE
    forced_seed_consistency_floor: float = FORCED_SEED_CONSISTENCY_FLOOR
    forced_seed_rollout_floor: float = FORCED_SEED_ROLLOUT_FLOOR
    forced_seed_cdf_enforced: bool = FORCED_SEED_CDF_ENFORCE
    forced_seed_cdf_boundary_epsilon: float = (
        FORCED_SEED_CDF_BOUNDARY_EPSILON
    )
    legacy_merkle_root_enforced: bool = LEGACY_MERKLE_ROOT_ENFORCE
    difficulty_auction_shadow_enabled: bool = DIFFICULTY_AUCTION_SHADOW_ENABLED
    difficulty_auction_shadow_environments: list[str] = Field(
        default_factory=lambda: list(DIFFICULTY_AUCTION_SHADOW_ENVIRONMENTS)
    )
    difficulty_auction_shadow_delta: float = DIFFICULTY_AUCTION_DELTA
    difficulty_auction_shadow_max_candidates: int = (
        DIFFICULTY_AUCTION_SHADOW_MAX_CANDIDATES
    )
    difficulty_auction_shadow_max_slots_per_operator: int = (
        DIFFICULTY_AUCTION_SHADOW_MAX_SLOTS_PER_OPERATOR
    )
    legacy_merkle_checks_total: int = 0
    legacy_merkle_matches: int = 0
    legacy_merkle_mismatches: int = 0
    legacy_merkle_errors: int = 0
    legacy_merkle_distinct_hotkeys: int = 0
    legacy_merkle_environments: list[str] = Field(default_factory=list)
    legacy_merkle_protocol_versions: dict[str, int] = Field(
        default_factory=dict
    )
    legacy_merkle_last_mismatch_ts: float | None = None
    archive_queue_depth: int | None = None
    archive_queue_oldest_window: int | None = None
    archive_queue_oldest_age_seconds: float | None = None
    archive_uploads_succeeded_total: int | None = None
    archive_upload_failures_total: int | None = None
    archive_last_upload_success_ts: float | None = None
    archive_last_upload_failure_ts: float | None = None
    archive_last_uploaded_window: int | None = None
    archive_last_failed_window: int | None = None
    archive_last_enqueued_window: int | None = None
    archive_archives_enqueued_total: int = 0
    archive_enqueue_gaps_total: int = 0
    archive_last_enqueue_gap: dict[str, int] | None = None
    prompt_sources: dict[str, dict[str, Any]] = Field(default_factory=dict)
    prompt_source_unavailable_total: int = 0
    training_kl_reference: dict[str, Any] = Field(default_factory=dict)


class ValidatorServer:
    def __init__(self, host: str = "0.0.0.0", port: int = VALIDATOR_HTTP_PORT) -> None:
        self.host = host
        self.port = port
        self._app_started_at = time.time()
        self._image_revision = runtime_revision()
        self._runtime_fingerprint = collect_runtime_fingerprint()
        self._chain_client_fingerprint = _chain_client_fingerprint()
        # Multi-env: keyed by env_name. ``active_batcher`` (singular) is
        # maintained as a legacy accessor pointing to the first active batcher
        # so existing code paths (/health, /state, the submit worker stale
        # check) keep working without change.
        self._active_batchers: dict[str, GrpoWindowBatcher] = {}
        self.active_batcher: GrpoWindowBatcher | None = None
        self._registration_gate_enforced = False
        self._registered_hotkeys: frozenset[str] | None = None
        self._operator_by_hotkey: dict[str, str] = {}
        self._registration_refreshed_at: float | None = None
        self._registration_cache_refresh_attempts_total = 0
        self._registration_cache_refresh_successes_total = 0
        self._registration_cache_refresh_failures_total = 0
        self._registration_cache_last_refresh_attempt_ts: float | None = None
        self._registration_cache_last_refresh_success_ts: float | None = None
        self._registration_cache_last_refresh_failure_ts: float | None = None
        self._registration_cache_last_refresh_failure_type: str | None = None
        self._registration_cache_last_refresh_failure_reason: str | None = None
        self._registration_cache_last_refresh_reason: str | None = None
        self._training_accumulator_state: dict[str, Any] = {}
        self._training_publish_state: dict[str, Any] = {}
        self._training_kl_reference_state: dict[str, Any] = {}
        self._last_committed_window_n = 0
        self._candidate_window_n: int | None = None
        self._window_preparation_stage: str | None = None
        self._last_window_preparation_failure: dict[str, Any] | None = None
        self._window_preparation_failures_total = 0
        self._window_preparation_failures_by_stage: collections.Counter[str] = (
            collections.Counter()
        )
        self._window_preparation_failures_by_error: collections.Counter[str] = (
            collections.Counter()
        )
        self._prompt_source_health_callback: (
            Callable[[], dict[str, dict[str, Any]]] | None
        ) = None
        self._prompt_source_unavailable_total = 0
        self._legacy_merkle_stats: collections.Counter[str] = (
            collections.Counter()
        )
        self._legacy_merkle_hotkeys: set[str] = set()
        self._legacy_merkle_environments: set[str] = set()
        self._legacy_merkle_protocol_versions: collections.Counter[int] = (
            collections.Counter()
        )
        self._legacy_merkle_last_mismatch_ts: float | None = None
        self._archive_queue_snapshot_callback: (
            Callable[[], dict[str, Any]] | None
        ) = None
        self._upload_precommit_receipts: dict[
            str, _UploadPrecommitReceipt
        ] = {}
        self._upload_precommit_by_signature: dict[str, str] = {}
        self.app: FastAPI = self._build_app()
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[Any] | None = None
        self._submit_queue: asyncio.Queue = asyncio.Queue(
            maxsize=MAX_PENDING_PROOF_QUEUE_DEPTH,
        )
        # OpenCode grading may spend the full sandbox timeout while Math
        # admission remains CPU-cheap. Independent queues and workers prevent
        # pathological code from head-of-line blocking the Math auction.
        self._code_submit_queue: asyncio.Queue = asyncio.Queue(
            maxsize=MAX_PENDING_PROOF_QUEUE_DEPTH,
        )
        self._worker_task: asyncio.Task[Any] | None = None
        self._code_worker_task: asyncio.Task[Any] | None = None
        self._extra_worker_tasks: list[asyncio.Task[Any]] = []
        self._auction_admission_enabled = False
        self._admission_process_pools: dict[str, ProcessPoolExecutor] = {}
        self._admission_tokenizer_hashes: dict[str, str] = {}
        self._admission_pool_locks: dict[str, asyncio.Lock] = {}
        self._admission_materialization_pool: ThreadPoolExecutor | None = None
        self._admission_active_by_environment: collections.Counter[str] = (
            collections.Counter()
        )
        self._admission_worker_restarts: collections.Counter[str] = (
            collections.Counter()
        )
        self._admission_timeouts: collections.Counter[str] = (
            collections.Counter()
        )
        self._admission_enqueued_at: dict[str, tuple[str, float]] = {}
        self._admission_inflight_items: dict[
            str, _QueuedAuctionSubmission
        ] = {}
        self._admission_inflight_requests: dict[
            str, BatchSubmissionRequest
        ] = {}
        self._admission_contexts: dict[int, AdmissionContext] = {}
        self._inflight_proofs = 0
        self._inflight_proofs_by_environment: collections.Counter[str] = (
            collections.Counter()
        )
        self._event_loop_lag_samples_ms: collections.deque[float] = (
            collections.deque(maxlen=600)
        )
        self._endpoint_latency_samples_ms: dict[
            str, collections.deque[float]
        ] = collections.defaultdict(lambda: collections.deque(maxlen=600))
        self._admission_latency_samples_ms: dict[
            str, dict[str, collections.deque[float]]
        ] = collections.defaultdict(
            lambda: collections.defaultdict(
                lambda: collections.deque(maxlen=600)
            )
        )
        self._event_loop_monitor_task: asyncio.Task[None] | None = None
        from reliquary.protocol.submission import WindowState
        self._current_state: WindowState = WindowState.READY
        self._current_checkpoint = None  # ManifestEntry | None
        self._late_drop_callback: Callable[[str, str], None] | None = None
        # Per-hotkey submission counter. Reset every time the active
        # batcher swaps (= window boundary). Read in /submit before any
        # heavier check so a saturated miner trip the rate limit on the
        # cheapest possible path.
        self._per_window_counts: dict[str, int] = {}
        # Per-hotkey BAD_ENVELOPE_SIGNATURE counter. Reset on batcher
        # swap alongside ``_per_window_counts``. Caps how many bad
        # packets one hotkey can burn per window — see
        # ``MAX_BAD_ENVELOPE_PER_HOTKEY_PER_WINDOW`` for the rationale
        # (closes the connection-priming side-channel without re-opening
        # the spoof-DoS that PR #35 closed).
        self._bad_envelope_counts: dict[str, int] = {}
        # Per-hotkey ring buffer of recent verdicts. Keys are miner ss58
        # addresses; values are deques of ``Verdict``-shaped dicts (stored
        # as plain dicts to keep the hot path serialization-free).
        # asyncio is single-threaded so no lock is needed — every mutation
        # site runs on the event loop.
        self._verdicts: dict[str, collections.deque[dict]] = {}
        self._recent_reject_counts: collections.Counter[str] = collections.Counter()

    def set_active_batchers(self, batchers: dict[str, GrpoWindowBatcher]) -> None:
        """Register multi-env batchers and update the legacy scalar accessor.

        ``batchers`` is {env_name: GrpoWindowBatcher}. An empty dict means
        no window is active (between READY and the next OPEN).
        """
        changed = batchers is not self._active_batchers or set(batchers) != set(self._active_batchers)
        if changed:
            for receipt in self._upload_precommit_receipts.values():
                resolver = getattr(
                    type(receipt.batcher), "resolve_upload_precommit", None
                )
                if resolver is not None:
                    resolver(
                        receipt.batcher,
                        receipt.receipt_id,
                        expired=not receipt.consumed,
                    )
            self._upload_precommit_receipts = {}
            self._upload_precommit_by_signature = {}
            self._admission_contexts = {}
            self._per_window_counts = {}
            self._bad_envelope_counts = {}
            self._recent_reject_counts = collections.Counter()
        self._active_batchers = batchers
        # Legacy scalar: first batcher in dict (or None if empty).
        self.active_batcher = next(iter(batchers.values())) if batchers else None
        if self.active_batcher is not None:
            self._runtime_fingerprint = collect_runtime_fingerprint(
                proof_model=getattr(self.active_batcher, "model", None),
            )
        if self._auction_admission_enabled:
            for env_name, env_batcher in batchers.items():
                if (
                    getattr(env_batcher, "difficulty_auction_enabled", False)
                    and env_name not in self._admission_process_pools
                ):
                    self._admission_process_pools[env_name] = (
                        self._new_admission_pool(env_name)
                    )

    def set_active_batcher(self, batcher: GrpoWindowBatcher | None) -> None:
        """Legacy single-env shim. Wraps into a dict and delegates."""
        if batcher is None:
            self.set_active_batchers({})
        else:
            env_name = getattr(getattr(batcher, "env", None), "name", "unknown")
            self.set_active_batchers({env_name: batcher})

    def _refund_submission_quota(self, hotkey: str) -> None:
        current_count = self._per_window_counts.get(hotkey, 0)
        if current_count <= 1:
            self._per_window_counts.pop(hotkey, None)
        else:
            self._per_window_counts[hotkey] = current_count - 1

    @staticmethod
    def _worker_drop_refunds_quota(request: BatchSubmissionRequest) -> bool:
        """Distinguish validator outages from candidate-killed sandboxes."""
        return not bool(request._grader_worker_crashed)

    def _prune_upload_precommits(self, *, now: float | None = None) -> None:
        current = time.time() if now is None else float(now)
        expired = [
            receipt_id
            for receipt_id, receipt in self._upload_precommit_receipts.items()
            if (
                (not receipt.consumed and receipt.expires_at_wall < current)
                or receipt.batcher not in self._active_batchers.values()
            )
        ]
        for receipt_id in expired:
            receipt = self._upload_precommit_receipts.pop(receipt_id)
            self._upload_precommit_by_signature.pop(
                receipt.precommit_signature, None
            )
            resolver = getattr(
                type(receipt.batcher), "resolve_upload_precommit", None
            )
            if resolver is not None:
                resolver(
                    receipt.batcher,
                    receipt.receipt_id,
                    expired=not receipt.consumed,
                )

    @staticmethod
    def _precommit_matches_submission(
        receipt: _UploadPrecommitReceipt,
        request: BatchSubmissionRequest,
        *,
        environment: str,
        payload_bytes: int,
        payload_sha256: str,
    ) -> bool:
        return (
            receipt.miner_hotkey == request.miner_hotkey
            and receipt.prompt_idx == request.prompt_idx
            and receipt.window_start == request.window_start
            and receipt.merkle_root.lower() == request.merkle_root.lower()
            and receipt.checkpoint_hash == request.checkpoint_hash
            and receipt.environment == environment
            and receipt.payload_bytes == payload_bytes
            and secrets.compare_digest(receipt.payload_sha256, payload_sha256)
            and receipt.drand_round == request.drand_round
            and receipt.protocol_version == request.protocol_version
            and receipt.nonce == request.nonce
        )

    def _claim_upload_precommit(
        self,
        receipt_id: str,
        request: BatchSubmissionRequest,
        *,
        batcher: Any,
        environment: str,
        payload_bytes: int,
        payload_sha256: str,
        body_completed_at: float,
    ) -> tuple[str, _UploadPrecommitReceipt | None]:
        """Return valid, replay, expired, or invalid for one body reveal."""
        receipt = self._upload_precommit_receipts.get(receipt_id)
        if receipt is None:
            self._prune_upload_precommits(now=body_completed_at)
            return "invalid", None
        if receipt.batcher is not batcher:
            return "invalid", None
        if body_completed_at > receipt.expires_at_wall:
            self._upload_precommit_receipts.pop(receipt_id, None)
            self._upload_precommit_by_signature.pop(
                receipt.precommit_signature, None
            )
            if not receipt.consumed:
                resolver = getattr(
                    type(receipt.batcher), "resolve_upload_precommit", None
                )
                if resolver is not None:
                    resolver(
                        receipt.batcher,
                        receipt.receipt_id,
                        expired=True,
                    )
            return "expired", None
        if not self._precommit_matches_submission(
            receipt,
            request,
            environment=environment,
            payload_bytes=payload_bytes,
            payload_sha256=payload_sha256,
        ):
            return "invalid", None
        if receipt.consumed:
            return "replay", receipt
        receipt.consumed = True
        revealed = getattr(
            type(receipt.batcher), "mark_upload_precommit_revealed", None
        )
        if revealed is not None:
            revealed(receipt.batcher, receipt.receipt_id)
        return "valid", receipt

    def set_current_state(self, state) -> None:
        self._current_state = state

    def set_training_accumulator_state(self, state: dict[str, Any]) -> None:
        """Expose a JSON-safe snapshot through ``/health``."""
        self._training_accumulator_state = dict(state)

    def set_training_publish_state(self, state: dict[str, Any]) -> None:
        """Expose checkpoint-cadence and pending-publication state."""
        self._training_publish_state = dict(state)

    def set_training_kl_reference_state(self, state: dict[str, Any]) -> None:
        """Expose the effective, resolved KL reference through ``/health``."""
        self._training_kl_reference_state = dict(state)

    def set_window_preparation_state(
        self,
        *,
        last_committed_window_n: int,
        candidate_window_n: int | None,
        stage: str | None,
    ) -> None:
        """Expose the pre-OPEN state without advertising a live window."""
        self._last_committed_window_n = int(last_committed_window_n)
        self._candidate_window_n = (
            int(candidate_window_n) if candidate_window_n is not None else None
        )
        self._window_preparation_stage = stage

    def record_window_preparation_failure(
        self, failure: dict[str, Any]
    ) -> None:
        """Record one secret-free pre-OPEN failure for operator telemetry."""
        safe_failure = dict(failure)
        self._last_window_preparation_failure = safe_failure
        self._window_preparation_failures_total += 1
        stage = str(safe_failure.get("stage") or "unknown")
        error_type = str(safe_failure.get("error_type") or "unknown")
        self._window_preparation_failures_by_stage[stage] += 1
        self._window_preparation_failures_by_error[error_type] += 1

    def clear_window_preparation_failure(self) -> None:
        self._last_window_preparation_failure = None

    def configure_prompt_source_health(
        self,
        snapshot_callback: Callable[[], dict[str, dict[str, Any]]],
    ) -> None:
        """Keep source health visible even when no batcher is active yet."""
        self._prompt_source_health_callback = snapshot_callback

    def configure_archive_queue_telemetry(
        self,
        snapshot_callback: Callable[[], dict[str, Any]],
    ) -> None:
        self._archive_queue_snapshot_callback = snapshot_callback

    def set_current_checkpoint(self, entry) -> None:
        self._current_checkpoint = entry

    def configure_registration_gate(self) -> None:
        """Arm admission against the service-managed metagraph snapshot."""
        self._registration_gate_enforced = True

    def set_registered_hotkeys(
        self,
        hotkeys: set[str] | frozenset[str] | list[str],
        *,
        refreshed_at: float | None = None,
        operator_by_hotkey: dict[str, str] | None = None,
    ) -> None:
        registered_hotkeys = frozenset(
            normalized
            for hotkey in hotkeys
            if (normalized := str(hotkey).strip())
        )
        self._registered_hotkeys = registered_hotkeys
        self._operator_by_hotkey = {
            normalized_hotkey: normalized_operator
            for hotkey, operator in (operator_by_hotkey or {}).items()
            if (normalized_hotkey := str(hotkey).strip()) in registered_hotkeys
            and (normalized_operator := str(operator).strip())
        }
        self._registration_refreshed_at = (
            time.time() if refreshed_at is None else float(refreshed_at)
        )

    def operator_by_hotkey_snapshot(self) -> dict[str, str]:
        """Return the chain ownership map associated with the registration cache."""
        return dict(self._operator_by_hotkey)

    def registration_cache_age(self, *, now: float | None = None) -> float | None:
        if self._registration_refreshed_at is None:
            return None
        current = time.time() if now is None else float(now)
        return max(0.0, current - self._registration_refreshed_at)

    def record_registration_cache_refresh(
        self,
        *,
        success: bool,
        reason: str,
        failure_type: str | None = None,
    ) -> None:
        """Record a secret-free result for a real metagraph refresh attempt."""
        now = time.time()
        normalized_reason = str(reason).strip() or "unspecified"
        self._registration_cache_refresh_attempts_total += 1
        self._registration_cache_last_refresh_attempt_ts = now
        self._registration_cache_last_refresh_reason = normalized_reason
        if success:
            self._registration_cache_refresh_successes_total += 1
            self._registration_cache_last_refresh_success_ts = now
            return

        self._registration_cache_refresh_failures_total += 1
        self._registration_cache_last_refresh_failure_ts = now
        self._registration_cache_last_refresh_failure_type = (
            str(failure_type).strip() if failure_type else "unknown"
        )
        self._registration_cache_last_refresh_failure_reason = normalized_reason

    async def _registration_reject_reason(
        self,
        hotkey: str,
    ) -> RejectReason | None:
        if not self._registration_gate_enforced:
            return None

        now = time.time()
        age = self.registration_cache_age(now=now)
        if (
            self._registered_hotkeys is None
            or age is None
            or age > REGISTERED_HOTKEY_STALE_GRACE_SECONDS
        ):
            return RejectReason.REGISTRATION_UNAVAILABLE
        if hotkey not in self._registered_hotkeys:
            # A stale snapshot may predate a new registration. The service will
            # retry at the next quiescent boundary; requests never trigger RPCs.
            if age > REGISTERED_HOTKEY_CACHE_TTL_SECONDS:
                return RejectReason.REGISTRATION_UNAVAILABLE
            return RejectReason.HOTKEY_NOT_REGISTERED
        return None

    def _observe_legacy_merkle(
        self,
        request: BatchSubmissionRequest,
        telemetry: SubmitTelemetry,
        *,
        env_name: str,
    ) -> str:
        """Measure current miner-root parity without changing wire-v1."""
        computed_root: str | None = None
        error_type: str | None = None
        try:
            matches, computed_root = legacy_submission_merkle_matches(request)
            status = "match" if matches else "mismatch"
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:
            status = "error"
            error_type = type(exc).__name__

        request._legacy_merkle_verified = status == "match"
        telemetry.apply_legacy_merkle(
            status=status,
            computed_root=computed_root,
            enforced=LEGACY_MERKLE_ROOT_ENFORCE,
        )
        self._legacy_merkle_stats[status] += 1
        self._legacy_merkle_hotkeys.add(request.miner_hotkey)
        if env_name:
            self._legacy_merkle_environments.add(env_name)
        self._legacy_merkle_protocol_versions[request.protocol_version] += 1
        if status != "match":
            self._legacy_merkle_last_mismatch_ts = time.time()

        log_submission_stage(
            logger,
            logging.INFO if status == "match" else logging.WARNING,
            "legacy_merkle_checked",
            telemetry,
            reject_stage=("legacy_merkle" if status != "match" else None),
            reject_reason=None,
            legacy_merkle_error_type=error_type,
            submission_env_name=env_name,
            accepted_into_pool=None,
        )
        return status

    @property
    def submit_queue_depth(self) -> int:
        return sum(self.submit_queue_depth_by_environment.values())

    @property
    def submit_queue_depth_by_environment(self) -> dict[str, int]:
        return {
            "openmathinstruct": self._submit_queue.qsize(),
            "opencodeinstruct": self._code_submit_queue.qsize(),
        }

    def _submission_queue_for_environment(
        self,
        environment: str,
    ) -> asyncio.Queue:
        if environment == "opencodeinstruct":
            return self._code_submit_queue
        return self._submit_queue

    @property
    def proof_verification_inflight(self) -> int:
        return self._inflight_proofs

    @property
    def proof_verification_inflight_by_environment(self) -> dict[str, int]:
        return {
            "openmathinstruct": self._inflight_proofs_by_environment.get(
                "openmathinstruct", 0
            ),
            "opencodeinstruct": self._inflight_proofs_by_environment.get(
                "opencodeinstruct", 0
            ),
        }

    def set_late_drop_callback(
        self, fn: Callable[[str, str], None] | None,
    ) -> None:
        """Register a callback fired as ``(hotkey, reason)`` on every late
        drop — reasons are ``"window_not_active"`` (HTTP-level) or
        ``"worker_dropped"`` (queue worker). Service registers in __init__.
        """
        self._late_drop_callback = fn

    def record_verdict(
        self,
        hotkey: str,
        merkle_root: str,
        accepted: bool,
        reason: RejectReason | str,
        *,
        window_n: int | None = None,
        telemetry: SubmitTelemetry | None = None,
        reject_stage: str | None = None,
        canonical_rank: int | None = None,
        accepted_into_pool: bool | None = None,
        selected_for_batch: bool | None = None,
        rewarded: bool | None = None,
    ) -> None:
        """Record a per-submission verdict for ``/verdicts/{hotkey}``.

        Called from every code path that decides a lifecycle stage:

          * HTTP rate-limit / window-not-active / batch-filled early cutoffs
            in the ``/submit`` handler (before the request even reaches the
            queue worker)
          * ``_submit_worker`` after each ``batcher.accept_submission``
            returns its pool-admission verdict (the path hidden by the
            provisional ``SUBMITTED`` response)
          * ``_submit_worker`` late drops for items dequeued after the
            batcher swap or seal (``worker_dropped`` / ``batch_filled``)
          * ``ValidationService`` after auction seal, with final selection,
            reward, and deferred-proof outcome fields

        The verdict is stored in a per-hotkey ring buffer
        (``VERDICT_CAP_PER_HOTKEY`` entries). Older verdicts roll off
        silently. Read-side: ``GET /verdicts/{hotkey}`` filters by hotkey
        and (optionally) by a ``since`` unix timestamp.
        """
        if hotkey not in self._verdicts:
            self._verdicts[hotkey] = collections.deque(maxlen=VERDICT_CAP_PER_HOTKEY)
        # Normalise enum → value so the ring is a uniform dict shape.
        reason_str = reason.value if isinstance(reason, RejectReason) else reason
        if not accepted:
            self._recent_reject_counts[reason_str] += 1
        entry = {
            "merkle_root": merkle_root,
            "window_n": window_n,
            "accepted": accepted,
            "reason": reason_str,
            "ts": time.time(),
        }
        if telemetry is not None:
            entry.update({
                key: value
                for key, value in telemetry.verdict_fields().items()
                if value is not None
            })
        if reject_stage is not None:
            entry["reject_stage"] = reject_stage
        if not accepted:
            entry["reject_reason"] = reason_str
        if canonical_rank is not None:
            entry["canonical_rank"] = canonical_rank
        if accepted_into_pool is not None:
            entry["accepted_into_pool"] = accepted_into_pool
        if selected_for_batch is not None:
            entry["selected_for_batch"] = selected_for_batch
        if rewarded is not None:
            entry["rewarded"] = rewarded
        self._verdicts[hotkey].append(entry)

    def _current_drand_round_best_effort(self) -> int | None:
        batcher = self.active_batcher
        ci = getattr(batcher, "_drand_chain_info", None) if batcher else None
        try:
            if ci is None:
                from reliquary.infrastructure.drand import get_current_chain
                ci = get_current_chain()
            from reliquary.infrastructure.chain import compute_current_drand_round
            return int(compute_current_drand_round(
                time.time(), ci["genesis_time"], ci["period"],
            ))
        except Exception:
            return None

    def _prompt_source_health(self) -> dict[str, dict[str, Any]]:
        if self._prompt_source_health_callback is not None:
            try:
                return {
                    str(env_name): dict(snapshot)
                    for env_name, snapshot in self._prompt_source_health_callback().items()
                }
            except Exception as exc:
                return {
                    "validator": {
                        "status": "degraded",
                        "last_error_type": type(exc).__name__,
                    }
                }

        batchers = dict(self._active_batchers)
        if not batchers and self.active_batcher is not None:
            env = getattr(self.active_batcher, "env", None)
            env_name = getattr(env, "name", "unknown")
            batchers[str(env_name)] = self.active_batcher

        snapshots: dict[str, dict[str, Any]] = {}
        for env_name, batcher in batchers.items():
            env = getattr(batcher, "env", None)
            snapshot_fn = getattr(env, "source_health", None)
            if not callable(snapshot_fn) or _is_mock_like(snapshot_fn):
                snapshots[env_name] = {"status": "unreported"}
                continue
            try:
                snapshots[env_name] = dict(snapshot_fn())
            except Exception as exc:
                snapshots[env_name] = {
                    "status": "degraded",
                    "last_error_type": type(exc).__name__,
                }
        return snapshots

    @staticmethod
    def _window_environment_health(batcher: Any) -> dict[str, Any]:
        """Return a JSON-safe per-environment view of one active batcher."""
        def _integer(value: Any) -> int | None:
            if isinstance(value, numbers.Integral) and not isinstance(value, bool):
                return int(value)
            return None

        def _floating(value: Any) -> float | None:
            if isinstance(value, numbers.Real) and not isinstance(value, bool):
                return float(value)
            return None

        auction_enabled = bool(
            getattr(batcher, "difficulty_auction_enabled", False)
        )
        rejection_snapshot_fn = getattr(
            type(batcher), "rejection_telemetry_snapshot", None
        )
        rejection_snapshot = (
            rejection_snapshot_fn(batcher)
            if callable(rejection_snapshot_fn)
            else {
                "grader_failures": dict(
                    getattr(batcher, "grader_failures", {})
                )
            }
        )
        distinct_fn = getattr(
            batcher,
            (
                "distinct_pending_prompt_count"
                if auction_enabled
                else "distinct_valid_prompt_count"
            ),
            None,
        )
        idle_fn = getattr(batcher, "seconds_since_last_valid_submission", None)
        sealed_fn = getattr(batcher, "is_sealed", None)
        try:
            distinct = distinct_fn() if callable(distinct_fn) else None
        except Exception:
            distinct = None
        try:
            idle_seconds = idle_fn() if callable(idle_fn) else None
        except Exception:
            idle_seconds = None
        try:
            sealed = bool(sealed_fn()) if callable(sealed_fn) else None
        except Exception:
            sealed = None

        force_seal_reason = getattr(batcher, "force_seal_reason", None)
        return {
            "window_n": _integer(getattr(batcher, "window_start", None)),
            "sealed": sealed,
            "force_seal_reason": (
                force_seal_reason if isinstance(force_seal_reason, str) else None
            ),
            "valid_submissions_count": _integer(
                getattr(
                    batcher,
                    "pending_count" if auction_enabled else "valid_count",
                    None,
                )
            ),
            "distinct_valid_prompt_count": _integer(distinct),
            "last_valid_submission_ts": _floating(
                getattr(batcher, "last_valid_submission_wall_ts", None)
            ),
            "seconds_since_last_valid_submission": _floating(idle_seconds),
            "proof_admission_count": _integer(
                getattr(batcher, "proof_admission_count", None)
            ),
            "proof_grading_attempts": _integer(
                getattr(batcher, "proof_grading_attempts", None)
            ),
            "pending_proof_reservations": _integer(
                getattr(batcher, "pending_proof_reservations", None)
            ),
            "inflight_proof_reservations": _integer(
                getattr(batcher, "inflight_proof_reservations", None)
            ),
            "reserved_payload_bytes": _integer(
                getattr(batcher, "reserved_payload_bytes", None)
            ),
            "pending_payload_bytes": _integer(
                getattr(batcher, "pending_payload_bytes", None)
            ),
            "inflight_payload_bytes": _integer(
                getattr(batcher, "inflight_payload_bytes", None)
            ),
            "retained_payload_bytes": _integer(
                getattr(batcher, "retained_payload_bytes", None)
            ),
            "pending_upload_precommits": _integer(
                getattr(batcher, "pending_upload_precommits", None)
            ),
            "upload_precommit_payload_bytes": _integer(
                getattr(batcher, "upload_precommit_payload_bytes", None)
            ),
            "upload_precommit_conservation": (
                batcher.upload_precommit_conservation()
                if callable(
                    getattr(
                        type(batcher),
                        "upload_precommit_conservation",
                        None,
                    )
                )
                else {}
            ),
            "collection_closed": bool(
                getattr(type(batcher), "collection_closed", lambda _self: False)(
                    batcher
                )
            ),
            "difficulty_auction_enabled": auction_enabled,
            "difficulty_auction_proof_wall_elapsed_seconds": _floating(
                getattr(batcher, "proof_wall_elapsed_seconds", None)
            ),
            "difficulty_auction_proof_wall_exhausted": bool(
                getattr(batcher, "proof_wall_exhausted", False)
            ),
            "forensic_proof_errors_by_type": dict(
                getattr(batcher, "forensic_proof_errors_by_type", {})
            ),
            "post_trigger_proof_admission_count": _integer(
                getattr(batcher, "post_trigger_proof_admission_count", None)
            ),
            "expensive_proof_failures_by_hotkey": dict(
                getattr(batcher, "expensive_proof_failures_by_hotkey", {})
            ),
            "expensive_proof_failures_by_operator": dict(
                getattr(batcher, "expensive_proof_failures_by_operator", {})
            ),
            "grader_failures": dict(
                rejection_snapshot.get("grader_failures", {})
            ),
            "auction_seal_drain": dict(
                getattr(batcher, "auction_seal_drain", {}) or {}
            ),
        }

    def _health_payload(self) -> _Health:
        batcher = self.active_batcher
        cp = self._current_checkpoint
        registration_age = self.registration_cache_age()
        registration_cache_stale = (
            registration_age > REGISTERED_HOTKEY_CACHE_TTL_SECONDS
            if registration_age is not None
            else None
        )
        registration_cache_usable = (
            self._registered_hotkeys is not None
            and registration_age is not None
            and registration_age <= REGISTERED_HOTKEY_STALE_GRACE_SECONDS
            if self._registration_gate_enforced
            else None
        )
        registration_cache_next_refresh_ts = (
            self._registration_refreshed_at
            + REGISTERED_HOTKEY_CACHE_TTL_SECONDS
            if self._registration_refreshed_at is not None
            else None
        )
        accumulator = self._training_accumulator_state
        training_publish = self._training_publish_state
        try:
            archive_queue = (
                self._archive_queue_snapshot_callback()
                if self._archive_queue_snapshot_callback is not None
                else {}
            )
        except Exception:
            # Observability must never turn a healthy validator into a failed
            # health check. Missing queue data remains visible as null fields.
            archive_queue = {}
        reject_counts: dict[str, int] = dict(self._recent_reject_counts)
        if batcher is not None:
            snapshot_fn = getattr(
                type(batcher), "rejection_telemetry_snapshot", None
            )
            batcher_rejects = (
                snapshot_fn(batcher).get("reject_counts", {})
                if callable(snapshot_fn)
                else getattr(batcher, "reject_counts", {})
            )
            for reason, count in batcher_rejects.items():
                reject_counts[reason] = max(reject_counts.get(reason, 0), count)
        prompt_sources = self._prompt_source_health()
        window_environments = {
            str(env_name): self._window_environment_health(env_batcher)
            for env_name, env_batcher in self._active_batchers.items()
        }
        active_window_health = (
            self._window_environment_health(batcher) if batcher else {}
        )
        logical_group_dedup: dict[str, dict[str, int]] = {}
        grader_failures_by_environment: dict[str, dict[str, int]] = {}
        for env_name, env_batcher in self._active_batchers.items():
            snapshot_fn = getattr(
                type(env_batcher), "rejection_telemetry_snapshot", None
            )
            rejection_snapshot = (
                snapshot_fn(env_batcher)
                if callable(snapshot_fn)
                else {
                    "grader_failures": dict(
                        getattr(env_batcher, "grader_failures", {})
                    )
                }
            )
            reservations = getattr(
                env_batcher, "logical_group_reservation_count", 0
            )
            duplicates = getattr(
                env_batcher, "logical_group_duplicate_rejects", 0
            )
            logical_group_dedup[env_name] = {
                "reservations": (
                    reservations
                    if isinstance(reservations, int)
                    and not isinstance(reservations, bool)
                    else 0
                ),
                "duplicate_rejects": (
                    duplicates
                    if isinstance(duplicates, int)
                    and not isinstance(duplicates, bool)
                    else 0
                ),
            }
            grader_failures_by_environment[env_name] = {
                str(reason): int(count)
                for reason, count in dict(
                    rejection_snapshot.get("grader_failures", {})
                ).items()
            }
        health_status = (
            "degraded"
            if (
                any(
                    source.get("status") == "degraded"
                    for source in prompt_sources.values()
                )
                or any(
                    any(count > 0 for count in failures.values())
                    for failures in grader_failures_by_environment.values()
                )
                or (
                    self._candidate_window_n is not None
                    and self._last_window_preparation_failure is not None
                )
                or (
                    self._registration_gate_enforced
                    and (
                        registration_cache_stale is not False
                        or registration_cache_usable is not True
                    )
                )
            )
            else "ok"
        )
        return _Health(
            status=health_status,
            active_window=batcher.window_start if batcher else None,
            image_revision=self._image_revision,
            runtime_fingerprint=dict(self._runtime_fingerprint),
            chain_client_fingerprint=dict(self._chain_client_fingerprint),
            app_started_at=self._app_started_at,
            current_validator_state=getattr(self._current_state, "value", str(self._current_state)),
            current_window_n=batcher.window_start if batcher else None,
            last_committed_window_n=self._last_committed_window_n,
            candidate_window_n=self._candidate_window_n,
            window_preparation_stage=self._window_preparation_stage,
            last_window_preparation_failure=(
                dict(self._last_window_preparation_failure)
                if self._last_window_preparation_failure is not None
                else None
            ),
            window_preparation_failures_total=(
                self._window_preparation_failures_total
            ),
            window_preparation_failures_by_stage=dict(
                self._window_preparation_failures_by_stage
            ),
            window_preparation_failures_by_error=dict(
                self._window_preparation_failures_by_error
            ),
            current_quicknet_drand_round=self._current_drand_round_best_effort(),
            current_window_open_ts=(
                getattr(batcher, "window_opened_wall_ts", None) if batcher else None
            ),
            current_window_open_drand_round=(
                getattr(batcher, "window_open_drand_round", None) if batcher else None
            ),
            seal_trigger_round=(
                getattr(batcher, "_seal_trigger_round", None) if batcher else None
            ),
            drand_round_backward_tolerance=DRAND_ROUND_BACKWARD_TOLERANCE,
            upload_precommit_enabled=True,
            submission_upload_grace_seconds=SUBMISSION_UPLOAD_GRACE_SECONDS,
            batch_size=B_BATCH,
            queue_depth=self.submit_queue_depth,
            queue_depth_by_environment=self.submit_queue_depth_by_environment,
            admission_workers_by_environment={
                "openmathinstruct": MATH_ADMISSION_WORKERS,
                "opencodeinstruct": CODE_ADMISSION_WORKERS,
            },
            admission_pool_state_by_environment={
                env_name: {
                    "queued": self._submission_queue_for_environment(
                        env_name
                    ).qsize(),
                    "active": int(
                        self._admission_active_by_environment.get(
                            env_name, 0
                        )
                    ),
                    "oldest_job_age_seconds": (
                        round(
                            max(
                                0.0,
                                time.monotonic()
                                - min(
                                    enqueued_at
                                    for queued_env, enqueued_at
                                    in self._admission_enqueued_at.values()
                                    if queued_env == env_name
                                ),
                            ),
                            3,
                        )
                        if any(
                            queued_env == env_name
                            for queued_env, _enqueued_at
                            in self._admission_enqueued_at.values()
                        )
                        else None
                    ),
                    "worker_restarts": int(
                        self._admission_worker_restarts.get(env_name, 0)
                    ),
                    "timeouts": int(
                        self._admission_timeouts.get(env_name, 0)
                    ),
                    "process_pool_started": (
                        env_name in self._admission_process_pools
                    ),
                }
                for env_name in (
                    "openmathinstruct",
                    "opencodeinstruct",
                )
            },
            proof_verification_inflight=self._inflight_proofs,
            proof_verification_inflight_by_environment=(
                self.proof_verification_inflight_by_environment
            ),
            event_loop_lag_ms=self._latency_summary(
                self._event_loop_lag_samples_ms
            ),
            endpoint_latency_ms={
                path: self._latency_summary(samples)
                for path, samples in self._endpoint_latency_samples_ms.items()
                if path in {"/health", "/state", "/submit/precommit", "/submit"}
            },
            admission_latency_ms_by_environment={
                env_name: {
                    metric: self._latency_summary(samples)
                    for metric, samples in metrics.items()
                }
                for env_name, metrics in self._admission_latency_samples_ms.items()
            },
            # These legacy scalar fields reflect the first active environment.
            # During an auction they intentionally report admitted candidates,
            # not only the winners proven later at seal.
            valid_submissions_count=active_window_health.get(
                "valid_submissions_count"
            ),
            distinct_valid_prompt_count=active_window_health.get(
                "distinct_valid_prompt_count"
            ),
            last_valid_submission_ts=(
                getattr(batcher, "last_valid_submission_wall_ts", None)
                if batcher else None
            ),
            seconds_since_last_valid_submission=(
                batcher.seconds_since_last_valid_submission()
                if (
                    batcher is not None
                    and hasattr(batcher, "seconds_since_last_valid_submission")
                )
                else None
            ),
            proof_admission_count=(
                getattr(batcher, "proof_admission_count", None) if batcher else None
            ),
            proof_grading_attempts=(
                getattr(batcher, "proof_grading_attempts", None)
                if batcher else None
            ),
            pending_proof_reservations=(
                getattr(batcher, "pending_proof_reservations", None)
                if batcher else None
            ),
            inflight_proof_reservations=(
                getattr(batcher, "inflight_proof_reservations", None)
                if batcher else None
            ),
            reserved_payload_bytes=(
                getattr(batcher, "reserved_payload_bytes", None)
                if batcher else None
            ),
            pending_payload_bytes=(
                getattr(batcher, "pending_payload_bytes", None)
                if batcher else None
            ),
            inflight_payload_bytes=(
                getattr(batcher, "inflight_payload_bytes", None)
                if batcher else None
            ),
            retained_payload_bytes=(
                getattr(batcher, "retained_payload_bytes", None)
                if batcher else None
            ),
            difficulty_auction_enforced=DIFFICULTY_AUCTION_ENFORCE,
            difficulty_auction_environments=list(
                DIFFICULTY_AUCTION_ENVIRONMENTS
            ),
            difficulty_auction_proof_attempt_limit=(
                MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW
            ),
            difficulty_auction_proof_wall_limit_seconds=(
                MAX_PROOF_WALL_SECONDS
            ),
            difficulty_auction_proof_wall_elapsed_seconds=(
                getattr(batcher, "proof_wall_elapsed_seconds", None)
                if batcher else None
            ),
            difficulty_auction_proof_wall_exhausted=(
                bool(getattr(batcher, "proof_wall_exhausted", False))
                if batcher else None
            ),
            window_environments=window_environments,
            logical_group_reservations=sum(
                item["reservations"] for item in logical_group_dedup.values()
            ),
            logical_group_duplicate_rejects=sum(
                item["duplicate_rejects"]
                for item in logical_group_dedup.values()
            ),
            logical_group_dedup_by_environment=logical_group_dedup,
            grader_failures_by_environment=grader_failures_by_environment,
            post_trigger_proof_admission_count=(
                getattr(batcher, "post_trigger_proof_admission_count", None)
                if batcher else None
            ),
            expensive_proof_failures_by_hotkey=(
                dict(getattr(batcher, "expensive_proof_failures_by_hotkey", {}))
                if batcher else {}
            ),
            expensive_proof_failures_by_operator=(
                dict(
                    getattr(
                        batcher,
                        "expensive_proof_failures_by_operator",
                        {},
                    )
                )
                if batcher
                else {}
            ),
            checkpoint_repo_id=cp.repo_id if cp else None,
            checkpoint_revision=cp.revision if cp else None,
            recent_reject_counts_by_reason=reject_counts,
            rewarded_but_not_selected_by_hotkey=(
                dict(getattr(batcher, "rewarded_but_not_selected_by_hotkey", {}))
                if batcher else {}
            ),
            registration_gate_enforced=self._registration_gate_enforced,
            registered_hotkey_count=(
                len(self._registered_hotkeys)
                if self._registered_hotkeys is not None
                else None
            ),
            registered_operator_mapping_count=len(self._operator_by_hotkey),
            registered_operator_mapping_complete=(
                len(self._operator_by_hotkey) == len(self._registered_hotkeys)
                if self._registered_hotkeys is not None
                else None
            ),
            registration_cache_age_seconds=registration_age,
            registration_cache_stale=registration_cache_stale,
            registration_cache_usable=registration_cache_usable,
            registration_cache_next_refresh_ts=(
                registration_cache_next_refresh_ts
            ),
            registration_cache_refresh_attempts_total=(
                self._registration_cache_refresh_attempts_total
            ),
            registration_cache_refresh_successes_total=(
                self._registration_cache_refresh_successes_total
            ),
            registration_cache_refresh_failures_total=(
                self._registration_cache_refresh_failures_total
            ),
            registration_cache_last_refresh_attempt_ts=(
                self._registration_cache_last_refresh_attempt_ts
            ),
            registration_cache_last_refresh_success_ts=(
                self._registration_cache_last_refresh_success_ts
            ),
            registration_cache_last_refresh_failure_ts=(
                self._registration_cache_last_refresh_failure_ts
            ),
            registration_cache_last_refresh_failure_type=(
                self._registration_cache_last_refresh_failure_type
            ),
            registration_cache_last_refresh_failure_reason=(
                self._registration_cache_last_refresh_failure_reason
            ),
            registration_cache_last_refresh_reason=(
                self._registration_cache_last_refresh_reason
            ),
            training_accumulator_checkpoint_revision=accumulator.get(
                "checkpoint_revision"
            ),
            training_accumulator_targets=dict(accumulator.get("targets", {})),
            training_accumulator_counts=dict(accumulator.get("counts", {})),
            training_accumulator_ready=bool(accumulator.get("ready", False)),
            training_trained_windows_since_publish=int(
                training_publish.get("trained_windows_since_publish", 0)
            ),
            training_checkpoint_publish_interval=int(
                training_publish.get("publish_interval", 0)
            ),
            training_checkpoint_publication_pending=bool(
                training_publish.get("publication_pending", False)
            ),
            forced_seed_enforced=FORCED_SEED_ENFORCE,
            forced_seed_consistency_floor=FORCED_SEED_CONSISTENCY_FLOOR,
            forced_seed_rollout_floor=FORCED_SEED_ROLLOUT_FLOOR,
            forced_seed_cdf_enforced=FORCED_SEED_CDF_ENFORCE,
            forced_seed_cdf_boundary_epsilon=(
                FORCED_SEED_CDF_BOUNDARY_EPSILON
            ),
            legacy_merkle_root_enforced=LEGACY_MERKLE_ROOT_ENFORCE,
            difficulty_auction_shadow_enabled=(
                DIFFICULTY_AUCTION_SHADOW_ENABLED
            ),
            difficulty_auction_shadow_environments=list(
                DIFFICULTY_AUCTION_SHADOW_ENVIRONMENTS
            ),
            difficulty_auction_shadow_delta=DIFFICULTY_AUCTION_DELTA,
            difficulty_auction_shadow_max_candidates=(
                DIFFICULTY_AUCTION_SHADOW_MAX_CANDIDATES
            ),
            difficulty_auction_shadow_max_slots_per_operator=(
                DIFFICULTY_AUCTION_SHADOW_MAX_SLOTS_PER_OPERATOR
            ),
            legacy_merkle_checks_total=sum(self._legacy_merkle_stats.values()),
            legacy_merkle_matches=self._legacy_merkle_stats["match"],
            legacy_merkle_mismatches=self._legacy_merkle_stats["mismatch"],
            legacy_merkle_errors=self._legacy_merkle_stats["error"],
            legacy_merkle_distinct_hotkeys=len(self._legacy_merkle_hotkeys),
            legacy_merkle_environments=sorted(
                self._legacy_merkle_environments
            ),
            legacy_merkle_protocol_versions={
                str(version): count
                for version, count in sorted(
                    self._legacy_merkle_protocol_versions.items()
                )
            },
            legacy_merkle_last_mismatch_ts=(
                self._legacy_merkle_last_mismatch_ts
            ),
            archive_queue_depth=archive_queue.get("depth"),
            archive_queue_oldest_window=archive_queue.get("oldest_window"),
            archive_queue_oldest_age_seconds=archive_queue.get(
                "oldest_age_seconds"
            ),
            archive_uploads_succeeded_total=archive_queue.get(
                "uploads_succeeded_total"
            ),
            archive_upload_failures_total=archive_queue.get(
                "upload_failures_total"
            ),
            archive_last_upload_success_ts=archive_queue.get(
                "last_upload_success_ts"
            ),
            archive_last_upload_failure_ts=archive_queue.get(
                "last_upload_failure_ts"
            ),
            archive_last_uploaded_window=archive_queue.get(
                "last_uploaded_window"
            ),
            archive_last_failed_window=archive_queue.get(
                "last_failed_window"
            ),
            archive_last_enqueued_window=archive_queue.get(
                "last_enqueued_window"
            ),
            archive_archives_enqueued_total=int(
                archive_queue.get("archives_enqueued_total", 0) or 0
            ),
            archive_enqueue_gaps_total=int(
                archive_queue.get("enqueue_gaps_total", 0) or 0
            ),
            archive_last_enqueue_gap=archive_queue.get("last_enqueue_gap"),
            prompt_sources=prompt_sources,
            training_kl_reference=dict(self._training_kl_reference_state),
            prompt_source_unavailable_total=(
                self._prompt_source_unavailable_total
            ),
        )

    @staticmethod
    def _call_accept_submission(
        batcher: Any,
        request: BatchSubmissionRequest,
        telemetry: SubmitTelemetry,
        reward_computation: Any | None = None,
    ) -> BatchSubmissionResponse:
        try:
            kwargs: dict[str, Any] = {"telemetry": telemetry}
            if reward_computation is not None:
                kwargs["reward_computation"] = reward_computation
            return batcher.accept_submission(request, **kwargs)
        except TypeError as exc:
            message = str(exc)
            if "unexpected keyword argument" not in message or not any(
                name in message
                for name in ("telemetry", "reward_computation")
            ):
                raise
            return batcher.accept_submission(request)

    @staticmethod
    def _start_proof_admission(
        batcher: Any,
        request: BatchSubmissionRequest,
    ) -> tuple[bool, str | None]:
        starter = getattr(type(batcher), "start_proof_admission", None)
        if starter is None:
            return True, None
        return starter(batcher, request)

    @staticmethod
    def _cancel_proof_admission(
        batcher: Any,
        request: BatchSubmissionRequest,
    ) -> None:
        cancel = getattr(type(batcher), "cancel_proof_admission", None)
        if cancel is not None:
            cancel(batcher, request)

    @staticmethod
    def _cancel_logical_group_reservation(
        batcher: Any,
        request: BatchSubmissionRequest,
    ) -> None:
        cancel = getattr(
            type(batcher), "cancel_logical_group_reservation", None
        )
        if cancel is not None:
            cancel(batcher, request)

    @staticmethod
    def _finish_proof_admission(
        batcher: Any,
        request: BatchSubmissionRequest,
    ) -> None:
        finish = getattr(type(batcher), "finish_proof_admission", None)
        if finish is not None:
            finish(batcher, request)

    @staticmethod
    def _fallback_drand_observation(
        request: BatchSubmissionRequest,
        batcher: Any,
        reject: RejectReason | None,
    ) -> DrandRoundObservation:
        tolerance = int(getattr(
            batcher,
            "drand_round_backward_tolerance",
            DRAND_ROUND_BACKWARD_TOLERANCE,
        ))
        return DrandRoundObservation(
            submitted_drand_round=int(request.drand_round),
            arrival_drand_round=None,
            drand_delta=None,
            drand_tolerance=tolerance,
            drand_status=classify_drand_round(request.drand_round, None, tolerance),
            reject_reason=reject,
        )

    def _admission_context_for(self, batcher: Any) -> AdmissionContext:
        cache_key = id(batcher)
        cached = self._admission_contexts.get(cache_key)
        if cached is not None:
            return cached

        model_config = _proof_free_model_config(batcher)
        tokenizer = getattr(batcher, "tokenizer", None)
        if _is_mock_like(tokenizer):
            tokenizer = None
        vocab_size = (
            resolve_vocab_size(model_config)
            if model_config is not None
            else None
        )
        max_sequence_length = (
            resolve_max_context_length(model_config)
            if model_config is not None
            else MAX_NEW_TOKENS_PROTOCOL_CAP
        )
        try:
            canonical_force_ids = tuple(force_close_token_ids(tokenizer))
            think_close_ids = tuple(think_close_token_ids(tokenizer))
        except Exception:
            canonical_force_ids = ()
            think_close_ids = ()
        context = AdmissionContext(
            randomness=str(getattr(batcher, "randomness", "")),
            environment=str(getattr(batcher.env, "name", "")),
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            eos_token_ids=tuple(sorted(_proof_free_eos_set(batcher) or ())),
            canonical_force_ids=canonical_force_ids,
            think_close_ids=think_close_ids,
            bootstrap=bool(getattr(batcher, "bootstrap", False)),
            enforce_envelope_signature=ENFORCE_ENVELOPE_SIGNATURE,
            enforce_legacy_merkle=LEGACY_MERKLE_ROOT_ENFORCE,
        )
        self._admission_contexts[cache_key] = context
        return context

    @staticmethod
    def _admission_binding(
        receipt: _UploadPrecommitReceipt,
    ) -> AdmissionReceiptBinding:
        return AdmissionReceiptBinding(
            miner_hotkey=receipt.miner_hotkey,
            prompt_idx=receipt.prompt_idx,
            window_start=receipt.window_start,
            merkle_root=receipt.merkle_root,
            checkpoint_hash=receipt.checkpoint_hash,
            environment=receipt.environment,
            payload_bytes=receipt.payload_bytes,
            drand_round=receipt.drand_round,
            protocol_version=receipt.protocol_version,
            nonce=receipt.nonce,
        )

    @staticmethod
    def _admission_worker_count(environment: str) -> int:
        if environment == "opencodeinstruct":
            return CODE_ADMISSION_WORKERS
        return MATH_ADMISSION_WORKERS

    def _new_admission_pool(self, environment: str) -> ProcessPoolExecutor:
        batcher = self._active_batchers.get(environment)
        tokenizer = getattr(batcher, "tokenizer", None)
        backend = getattr(tokenizer, "backend_tokenizer", None)
        serializer = getattr(backend, "to_str", None)
        if not callable(serializer):
            raise RuntimeError(
                f"fast tokenizer unavailable for {environment} admission"
            )
        tokenizer_json = str(serializer())
        self._admission_tokenizer_hashes[environment] = hashlib.sha256(
            tokenizer_json.encode("utf-8")
        ).hexdigest()
        return ProcessPoolExecutor(
            max_workers=self._admission_worker_count(environment),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_admission_worker,
            initargs=(tokenizer_json,),
            max_tasks_per_child=ADMISSION_PROCESS_MAX_TASKS,
        )

    @staticmethod
    def _terminate_admission_pool(pool: ProcessPoolExecutor) -> None:
        """Bound cleanup of a failed pool on Python versions without terminate_workers."""
        processes = tuple(
            (getattr(pool, "_processes", None) or {}).values()
        )
        terminate = getattr(pool, "terminate_workers", None)
        if callable(terminate):
            terminate()
            return

        for process in processes:
            if process.is_alive():
                process.terminate()
        pool.shutdown(wait=False, cancel_futures=True)

        join_deadline = time.monotonic() + 1.0
        for process in processes:
            process.join(timeout=max(0.0, join_deadline - time.monotonic()))
        for process in processes:
            if process.is_alive():
                process.kill()

    async def _restart_admission_pool(
        self,
        environment: str,
        failed_pool: ProcessPoolExecutor,
    ) -> None:
        lock = self._admission_pool_locks.setdefault(
            environment, asyncio.Lock()
        )
        async with lock:
            if self._admission_process_pools.get(environment) is not failed_pool:
                return
            await asyncio.to_thread(
                self._terminate_admission_pool, failed_pool
            )
            self._admission_process_pools[environment] = (
                self._new_admission_pool(environment)
            )
            self._admission_worker_restarts[environment] += 1

    async def _run_admission_process(
        self,
        environment: str,
        function: Callable[..., Any],
        *args: Any,
        wall_seconds: float,
    ) -> Any:
        pool = self._admission_process_pools[environment]
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(pool, function, *args)
        try:
            result = await asyncio.wait_for(
                future, timeout=max(1.0, wall_seconds + 2.0)
            )
        except asyncio.TimeoutError:
            self._admission_timeouts[environment] += 1
            await self._restart_admission_pool(environment, pool)
            raise
        except BrokenProcessPool:
            await self._restart_admission_pool(environment, pool)
            raise
        if bool(getattr(result, "timed_out", False)):
            self._admission_timeouts[environment] += 1
        return result

    def _complete_upload_receipt(
        self,
        receipt: _UploadPrecommitReceipt,
        outcome: BatchSubmissionResponse,
        *,
        expired: bool = False,
    ) -> bool:
        if receipt.terminal:
            return False
        receipt.outcome = outcome
        receipt.terminal = True
        resolver = getattr(
            type(receipt.batcher), "resolve_upload_precommit", None
        )
        if resolver is not None:
            resolver(
                receipt.batcher,
                receipt.receipt_id,
                expired=expired,
            )
        self._admission_enqueued_at.pop(receipt.receipt_id, None)
        return True

    def _claim_raw_upload_precommit(
        self,
        receipt_id: str,
        *,
        payload_bytes: int,
        payload_sha256: str,
        body_completed_at: float,
    ) -> tuple[str, _UploadPrecommitReceipt | None]:
        receipt = self._upload_precommit_receipts.get(receipt_id)
        if receipt is None:
            self._prune_upload_precommits(now=body_completed_at)
            return "invalid", None
        if receipt.batcher not in self._active_batchers.values():
            return "invalid", None
        if receipt.consumed:
            return "replay", receipt
        if body_completed_at > receipt.expires_at_wall:
            receipt.consumed = True
            self._complete_upload_receipt(
                receipt,
                BatchSubmissionResponse(
                    accepted=False,
                    reason=RejectReason.PRECOMMIT_EXPIRED,
                ),
                expired=True,
            )
            return "expired", receipt

        revealed = getattr(
            type(receipt.batcher), "mark_upload_precommit_revealed", None
        )
        if revealed is not None:
            revealed(receipt.batcher, receipt.receipt_id)
        if (
            payload_bytes != receipt.payload_bytes
            or not secrets.compare_digest(
                payload_sha256.lower(), receipt.payload_sha256
            )
        ):
            receipt.consumed = True
            self._complete_upload_receipt(
                receipt,
                BatchSubmissionResponse(
                    accepted=False,
                    reason=RejectReason.PRECOMMIT_INVALID,
                ),
            )
            return "invalid", receipt
        receipt.consumed = True
        return "valid", receipt

    @staticmethod
    def _receipt_telemetry(
        receipt: _UploadPrecommitReceipt,
        *,
        t_arrival: float,
        content_length_bytes: int | None,
        payload_sha256: str,
        t_body_started: float | None,
        t_body_completed: float | None,
        queue_depth: int,
    ) -> SubmitTelemetry:
        telemetry = SubmitTelemetry.from_precommit(
            window_n=receipt.window_start,
            prompt_idx=receipt.prompt_idx,
            hotkey=receipt.miner_hotkey,
            merkle_root=receipt.merkle_root,
            protocol_version=receipt.protocol_version,
            submitted_drand_round=receipt.drand_round,
            t_arrival=t_arrival,
            payload_bytes=receipt.payload_bytes,
            content_length_bytes=content_length_bytes,
            payload_sha256=payload_sha256,
            t_body_started=t_body_started,
            t_body_completed=t_body_completed,
            queue_depth_at_arrival=queue_depth,
        )
        telemetry.apply_upload_precommit(
            "valid", arrival_ts=receipt.precommit_arrival_ts
        )
        telemetry.apply_drand(receipt.drand_observation)
        telemetry.refresh_from_batcher(receipt.batcher)
        return telemetry

    def _record_raw_terminal(
        self,
        receipt: _UploadPrecommitReceipt,
        telemetry: SubmitTelemetry,
        outcome: BatchSubmissionResponse,
        *,
        stage: str,
    ) -> BatchSubmissionResponse:
        if receipt.terminal_recorded:
            return outcome
        receipt.terminal_recorded = True
        telemetry.refresh_from_batcher(receipt.batcher, at_decision=True)
        telemetry.mark_decision()
        self.record_verdict(
            receipt.miner_hotkey,
            receipt.merkle_root,
            outcome.accepted,
            outcome.reason,
            window_n=receipt.window_start,
            telemetry=telemetry,
            reject_stage=None if outcome.accepted else stage,
            accepted_into_pool=outcome.accepted,
        )
        log_submission_stage(
            logger,
            logging.INFO if outcome.accepted else logging.WARNING,
            "candidate_accepted" if outcome.accepted else "candidate_rejected",
            telemetry,
            reject_stage=None if outcome.accepted else stage,
            reject_reason=None if outcome.accepted else outcome.reason.value,
            accepted_into_pool=outcome.accepted,
        )
        return outcome

    async def _submit_auction_reveal(
        self,
        http_request: Request,
        response: Response,
    ) -> BatchSubmissionResponse | None:
        if not self._auction_admission_enabled:
            return None
        if not any(
            bool(getattr(batcher, "difficulty_auction_enabled", False))
            for batcher in self._active_batchers.values()
        ):
            return None

        receipt_id = http_request.headers.get(PRECOMMIT_HEADER)
        if not receipt_id:
            response.headers["Connection"] = "close"
            return BatchSubmissionResponse(
                accepted=False,
                reason=RejectReason.PRECOMMIT_REQUIRED,
            )
        receipt = self._upload_precommit_receipts.get(receipt_id)
        if (
            receipt is None
            or receipt.batcher not in self._active_batchers.values()
            or not bool(
                getattr(receipt.batcher, "difficulty_auction_enabled", False)
            )
        ):
            response.headers["Connection"] = "close"
            return BatchSubmissionResponse(
                accepted=False,
                reason=RejectReason.PRECOMMIT_INVALID,
            )
        if receipt.consumed:
            response.headers["Connection"] = "close"
            return receipt.outcome or BatchSubmissionResponse(
                accepted=True, reason=RejectReason.SUBMITTED
            )

        raw_content_length = http_request.headers.get("content-length")
        try:
            content_length_bytes = (
                int(raw_content_length)
                if raw_content_length is not None
                else None
            )
        except ValueError:
            content_length_bytes = None
        if (
            content_length_bytes is not None
            and content_length_bytes != receipt.payload_bytes
        ):
            marker = getattr(
                type(receipt.batcher),
                "mark_upload_precommit_revealed",
                None,
            )
            if marker is not None:
                marker(receipt.batcher, receipt.receipt_id)
            receipt.consumed = True
            outcome = BatchSubmissionResponse(
                accepted=False,
                reason=RejectReason.PRECOMMIT_INVALID,
            )
            self._complete_upload_receipt(receipt, outcome)
            telemetry = self._receipt_telemetry(
                receipt,
                t_arrival=float(
                    getattr(http_request.state, "t_arrival", time.time())
                ),
                content_length_bytes=content_length_bytes,
                payload_sha256=receipt.payload_sha256,
                t_body_started=None,
                t_body_completed=None,
                queue_depth=self.submit_queue_depth,
            )
            telemetry.apply_upload_precommit(
                "invalid", arrival_ts=receipt.precommit_arrival_ts
            )
            response.headers["Connection"] = "close"
            return self._record_raw_terminal(
                receipt,
                telemetry,
                outcome,
                stage="upload_precommit",
            )

        raw_body = await http_request.body()
        t_arrival = float(
            getattr(http_request.state, "t_arrival", time.time())
        )
        body_completed_at = float(
            getattr(http_request.state, "body_completed_at", time.time())
        )
        body_started_at = getattr(
            http_request.state, "body_receive_started_at", None
        )
        payload_bytes = int(
            getattr(http_request.state, "body_bytes_received", 0) or 0
        ) or len(raw_body)
        payload_sha256 = str(
            getattr(http_request.state, "body_sha256", "") or ""
        ).lower() or hashlib.sha256(raw_body).hexdigest()
        status, claimed = self._claim_raw_upload_precommit(
            receipt_id,
            payload_bytes=payload_bytes,
            payload_sha256=payload_sha256,
            body_completed_at=body_completed_at,
        )
        if claimed is None:
            return BatchSubmissionResponse(
                accepted=False, reason=RejectReason.PRECOMMIT_INVALID
            )
        telemetry = self._receipt_telemetry(
            claimed,
            t_arrival=t_arrival,
            content_length_bytes=content_length_bytes,
            payload_sha256=payload_sha256,
            t_body_started=body_started_at,
            t_body_completed=body_completed_at,
            queue_depth=self.submit_queue_depth,
        )
        if status == "replay":
            telemetry.apply_upload_precommit(
                "replay", arrival_ts=claimed.precommit_arrival_ts
            )
            return claimed.outcome or BatchSubmissionResponse(
                accepted=True, reason=RejectReason.SUBMITTED
            )
        if status == "expired":
            telemetry.apply_upload_precommit(
                "expired", arrival_ts=claimed.precommit_arrival_ts
            )
            return self._record_raw_terminal(
                claimed,
                telemetry,
                claimed.outcome
                or BatchSubmissionResponse(
                    accepted=False,
                    reason=RejectReason.PRECOMMIT_EXPIRED,
                ),
                stage="upload_precommit",
            )
        if status != "valid":
            telemetry.apply_upload_precommit(
                "invalid", arrival_ts=claimed.precommit_arrival_ts
            )
            return self._record_raw_terminal(
                claimed,
                telemetry,
                claimed.outcome
                or BatchSubmissionResponse(
                    accepted=False,
                    reason=RejectReason.PRECOMMIT_INVALID,
                ),
                stage="upload_precommit",
            )

        queue = self._submission_queue_for_environment(claimed.environment)
        telemetry.mark_enqueued(queue_depth=queue.qsize())
        queued = _QueuedAuctionSubmission(
            raw_body=raw_body,
            receipt=claimed,
            batcher=claimed.batcher,
            telemetry=telemetry,
            enqueued_monotonic=time.monotonic(),
        )
        try:
            queue.put_nowait(queued)
        except asyncio.QueueFull:
            outcome = BatchSubmissionResponse(
                accepted=False, reason=RejectReason.BATCH_FILLED
            )
            self._complete_upload_receipt(claimed, outcome)
            return self._record_raw_terminal(
                claimed,
                telemetry,
                outcome,
                stage="admission_queue",
            )
        self._admission_enqueued_at[claimed.receipt_id] = (
            claimed.environment,
            queued.enqueued_monotonic,
        )
        outcome = BatchSubmissionResponse(
            accepted=True, reason=RejectReason.SUBMITTED
        )
        claimed.outcome = outcome
        log_submission_stage(
            logger,
            logging.INFO,
            "submit_received",
            telemetry,
            queue_depth=queue.qsize(),
            queue_depth_by_environment=self.submit_queue_depth_by_environment,
        )
        return outcome

    async def abort_auction_admission(
        self,
        batchers: list[Any],
    ) -> dict[str, int]:
        """Make every unresolved receipt terminal before aborting a window."""
        batcher_ids = {id(batcher) for batcher in batchers}
        stats = {"queued": 0, "inflight": 0, "expired": 0}
        for queue in (self._submit_queue, self._code_submit_queue):
            retained: list[Any] = []
            while True:
                try:
                    queued = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if (
                    not isinstance(queued, _QueuedAuctionSubmission)
                    or id(queued.batcher) not in batcher_ids
                ):
                    retained.append(queued)
                    continue
                stats["queued"] += 1
                outcome = BatchSubmissionResponse(
                    accepted=False, reason=RejectReason.WORKER_DROPPED
                )
                rejector = getattr(type(queued.batcher), "_reject", None)
                if rejector is not None:
                    outcome = rejector(
                        queued.batcher,
                        RejectReason.WORKER_DROPPED,
                        hotkey=queued.receipt.miner_hotkey,
                        prompt_idx=queued.receipt.prompt_idx,
                        telemetry=queued.telemetry,
                        reject_stage="admission_drain",
                    )
                self._complete_upload_receipt(queued.receipt, outcome)
                self._record_raw_terminal(
                    queued.receipt,
                    queued.telemetry,
                    outcome,
                    stage="admission_drain",
                )
            for queued in retained:
                queue.put_nowait(queued)

        for receipt_id, item in list(self._admission_inflight_items.items()):
            if id(item.batcher) not in batcher_ids:
                continue
            stats["inflight"] += 1
            request = self._admission_inflight_requests.get(receipt_id)
            if request is not None:
                item.batcher.cancel_logical_group_reservation(request)
                item.batcher.finish_proof_admission(request)
            outcome = BatchSubmissionResponse(
                accepted=False, reason=RejectReason.WORKER_DROPPED
            )
            self._complete_upload_receipt(item.receipt, outcome)
            self._record_raw_terminal(
                item.receipt,
                item.telemetry,
                outcome,
                stage="admission_drain",
            )

        for receipt in self._upload_precommit_receipts.values():
            if id(receipt.batcher) not in batcher_ids or receipt.terminal:
                continue
            expired = not receipt.consumed
            stats["expired"] += int(expired)
            outcome = BatchSubmissionResponse(
                accepted=False,
                reason=(
                    RejectReason.PRECOMMIT_EXPIRED
                    if expired
                    else RejectReason.WORKER_DROPPED
                ),
            )
            self._complete_upload_receipt(
                receipt, outcome, expired=expired
            )

        environments = {
            str(getattr(getattr(batcher, "env", None), "name", ""))
            for batcher in batchers
        }
        for environment in environments:
            pool = self._admission_process_pools.get(environment)
            if pool is not None:
                await self._restart_admission_pool(environment, pool)
        return stats

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="Reliquary Validator", version="2.0")

        @app.middleware("http")
        async def stamp_arrival(request: Request, call_next):
            """Stamp the wall-clock arrival time on every request.

            Runs before pydantic body validation and before the route
            handler, so ``request.state.t_arrival`` reflects when the
            asyncio loop first picked the request up — not when the
            handler eventually executes. The /submit cheap-reject path
            consumes this attribute and forwards it to
            ``validate_drand_round`` so the drand timing gate is decided
            against the actual arrival instant, not against the moment
            the handler eventually runs (which can lag by tens of ms
            under load even for fast handlers).

            The drand check happens EXCLUSIVELY here on the arrival path —
            there's no worker-side re-check that would otherwise re-read
            ``time.time()`` minutes later when GRAIL queue backpressure
            delays dequeue (which would turn on-time submissions into
            STALE_ROUND rejections, the bug pre-this-fix).
            """
            request.state.t_arrival = time.time()
            request_started = time.perf_counter()
            try:
                return await call_next(request)
            finally:
                path = request.url.path
                if path in {
                    "/health",
                    "/state",
                    "/submit/precommit",
                    "/submit",
                }:
                    self._endpoint_latency_samples_ms[path].append(
                        max(
                            0.0,
                            (time.perf_counter() - request_started) * 1000.0,
                        )
                    )
                release = getattr(
                    request.state, "release_upload_precommit", None
                )
                if callable(release):
                    release()

        @app.get("/health", response_model=_Health)
        async def health() -> _Health:
            return self._health_payload()

        @app.post(
            "/submit/precommit",
            response_model=SubmissionPrecommitResponse,
        )
        async def precommit(
            request: SubmissionPrecommitRequest,
            http_request: Request,
        ) -> SubmissionPrecommitResponse:
            """Reserve one bounded reveal for a body committed before cutoff."""
            from reliquary.protocol.submission import WindowState

            t_arrival = float(
                getattr(http_request.state, "t_arrival", time.time())
            )

            def reject(reason: RejectReason) -> SubmissionPrecommitResponse:
                logger.warning(
                    "upload_precommit_rejected window=%d env=%s prompt=%d "
                    "hotkey=%s reason=%s payload_bytes=%d",
                    request.window_start,
                    request.environment,
                    request.prompt_idx,
                    request.miner_hotkey[:12],
                    reason.value,
                    request.payload_bytes,
                )
                return SubmissionPrecommitResponse(
                    accepted=False,
                    reason=reason,
                )

            if self._current_state != WindowState.OPEN:
                return reject(RejectReason.WINDOW_NOT_ACTIVE)
            batcher = self._active_batchers.get(request.environment)
            if batcher is None:
                return reject(RejectReason.BAD_SCHEMA)
            if request.window_start != batcher.window_start:
                return reject(RejectReason.WINDOW_MISMATCH)
            if request.payload_bytes > MAX_SUBMISSION_PAYLOAD_BYTES:
                return reject(RejectReason.BAD_SCHEMA)
            if (
                batcher.current_checkpoint_hash
                and request.checkpoint_hash != batcher.current_checkpoint_hash
            ):
                return reject(RejectReason.WRONG_CHECKPOINT)
            if (
                FORCED_SEED_ENFORCE
                and bool(batcher.current_checkpoint_hash)
                and request.protocol_version != FORCED_SEED_PROTOCOL_VERSION
            ):
                return reject(RejectReason.SEED_MISMATCH)
            precommit_signature_valid = await asyncio.to_thread(
                verify_precommit_signature,
                miner_hotkey=request.miner_hotkey,
                window_start=request.window_start,
                prompt_idx=request.prompt_idx,
                merkle_root=request.merkle_root,
                checkpoint_hash=request.checkpoint_hash,
                environment=request.environment,
                payload_bytes=request.payload_bytes,
                payload_sha256=request.payload_sha256,
                drand_round=request.drand_round,
                randomness=batcher.randomness,
                protocol_version=request.protocol_version,
                nonce=request.nonce,
                precommit_signature=request.precommit_signature,
            )
            if not precommit_signature_valid:
                return reject(RejectReason.BAD_ENVELOPE_SIGNATURE)

            # Exact retries are idempotent even if the drand boundary advanced
            # after the first response was lost.  The original receipt already
            # passed timing, registration, and quota admission.
            self._prune_upload_precommits(now=t_arrival)
            existing_id = self._upload_precommit_by_signature.get(
                request.precommit_signature
            )
            existing = (
                self._upload_precommit_receipts.get(existing_id)
                if existing_id is not None
                else None
            )
            if existing is not None:
                return SubmissionPrecommitResponse(
                    accepted=True,
                    reason=RejectReason.ACCEPTED,
                    receipt_id=existing.receipt_id,
                    upload_deadline_ts=existing.expires_at_wall,
                )
            if batcher.drand_round_check_enabled:
                if hasattr(type(batcher), "observe_drand_round"):
                    drand_observation = batcher.observe_drand_round(
                        request.drand_round,
                        t_arrival=t_arrival,
                    )
                else:
                    round_reject = batcher.validate_drand_round(
                        request.drand_round,
                        t_arrival=t_arrival,
                    )
                    drand_observation = self._fallback_drand_observation(
                        request, batcher, round_reject,
                    )
                if drand_observation.reject_reason is not None:
                    return reject(drand_observation.reject_reason)
            else:
                drand_observation = self._fallback_drand_observation(
                    request, batcher, None,
                )
            prompt_range = getattr(batcher, "prompt_range", None)
            if prompt_range is not None:
                lo, hi = prompt_range
                if not (lo <= request.prompt_idx < hi):
                    return reject(RejectReason.PROMPT_OUT_OF_RANGE)
            if request.prompt_idx in batcher.cooldown_prompts_snapshot:
                return reject(RejectReason.PROMPT_IN_COOLDOWN)
            try:
                environment_size = await asyncio.to_thread(len, batcher.env)
            except PromptSourceUnavailable:
                return reject(RejectReason.WORKER_DROPPED)
            if request.prompt_idx >= environment_size:
                return reject(RejectReason.BAD_PROMPT_IDX)
            if (
                batcher.prompt_submission_count(request.prompt_idx)
                >= MAX_SUBMISSIONS_PER_PROMPT
            ):
                return reject(RejectReason.PROMPT_FULL)

            registration_reason = await self._registration_reject_reason(
                request.miner_hotkey
            )
            if registration_reason is not None:
                return reject(registration_reason)

            count = self._per_window_counts.get(request.miner_hotkey, 0)
            if count >= MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW:
                return reject(RejectReason.RATE_LIMITED)

            receipt_id = secrets.token_urlsafe(32)
            register = getattr(
                type(batcher), "try_register_upload_precommit", None
            )
            if register is None:
                return reject(RejectReason.PRECOMMIT_INVALID)
            accepted, register_reason, deadline_monotonic = register(
                batcher,
                receipt_id,
                request.miner_hotkey,
                t_arrival_wall=t_arrival,
                payload_bytes=request.payload_bytes,
            )
            if not accepted or deadline_monotonic is None:
                reason = (
                    RejectReason.PRECOMMIT_EXPIRED
                    if register_reason in {"collection_closed", "collection_sealed"}
                    else RejectReason.BATCH_FILLED
                )
                return reject(reason)

            remaining = max(
                0.0,
                float(deadline_monotonic) - float(batcher._time_fn()),
            )
            expires_at_wall = time.time() + remaining
            receipt = _UploadPrecommitReceipt(
                receipt_id=receipt_id,
                precommit_signature=request.precommit_signature,
                miner_hotkey=request.miner_hotkey,
                prompt_idx=request.prompt_idx,
                window_start=request.window_start,
                merkle_root=request.merkle_root,
                checkpoint_hash=request.checkpoint_hash,
                environment=request.environment,
                payload_bytes=request.payload_bytes,
                payload_sha256=request.payload_sha256.lower(),
                drand_round=request.drand_round,
                protocol_version=request.protocol_version,
                nonce=request.nonce,
                expires_at_wall=expires_at_wall,
                precommit_arrival_ts=t_arrival,
                drand_observation=drand_observation,
                batcher=batcher,
            )
            self._upload_precommit_receipts[receipt_id] = receipt
            self._upload_precommit_by_signature[
                request.precommit_signature
            ] = receipt_id
            self._per_window_counts[request.miner_hotkey] = count + 1
            logger.info(
                "upload_precommit_accepted window=%d env=%s prompt=%d "
                "hotkey=%s payload_bytes=%d grace_s=%.3f",
                request.window_start,
                request.environment,
                request.prompt_idx,
                request.miner_hotkey[:12],
                request.payload_bytes,
                remaining,
            )
            return SubmissionPrecommitResponse(
                accepted=True,
                reason=RejectReason.ACCEPTED,
                receipt_id=receipt_id,
                upload_deadline_ts=expires_at_wall,
            )

        @app.post(
            "/submit",
            response_model=BatchSubmissionResponse,
            openapi_extra={
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": BatchSubmissionRequest.model_json_schema()
                        }
                    },
                }
            },
        )
        async def submit(
            http_request: Request,
            response: Response,
        ) -> BatchSubmissionResponse:
            from reliquary.protocol.submission import WindowState
            auction_response = await self._submit_auction_reveal(
                http_request, response
            )
            if auction_response is not None:
                return auction_response
            raw_body = await http_request.body()
            parse_started = time.perf_counter()
            try:
                request = await asyncio.to_thread(
                    BatchSubmissionRequest.model_validate_json,
                    raw_body,
                )
            except ValidationError as exc:
                raise RequestValidationError(exc.errors()) from exc
            body_parse_ms = (time.perf_counter() - parse_started) * 1000.0
            # ASGI middleware stamped this. Falls back to time.time() if a
            # caller bypasses the middleware (e.g. some test harnesses).
            t_arrival = getattr(http_request.state, "t_arrival", None)
            if t_arrival is None:
                t_arrival = time.time()
            payload_bytes = int(
                getattr(http_request.state, "body_bytes_received", 0) or 0
            )
            if payload_bytes <= 0:
                payload_bytes = len(raw_body)
            payload_sha256 = str(
                getattr(http_request.state, "body_sha256", "") or ""
            ).lower()
            if not payload_sha256:
                payload_sha256 = hashlib.sha256(raw_body).hexdigest()
            body_completed_at = float(
                getattr(http_request.state, "body_completed_at", time.time())
            )
            body_started_at = getattr(
                http_request.state, "body_receive_started_at", None
            )
            raw_content_length = http_request.headers.get("content-length")
            try:
                content_length_bytes = (
                    int(raw_content_length)
                    if raw_content_length is not None
                    else None
                )
            except ValueError:
                content_length_bytes = None
            if payload_bytes > MAX_SUBMISSION_PAYLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="submission_payload_too_large",
                )
            request._payload_bytes = payload_bytes
            hk = request.miner_hotkey
            telemetry = SubmitTelemetry.from_request(
                request,
                t_arrival=t_arrival,
                payload_bytes=payload_bytes,
                content_length_bytes=content_length_bytes,
                payload_sha256=payload_sha256,
                t_body_started=body_started_at,
                t_body_completed=body_completed_at,
                queue_depth_at_arrival=self.submit_queue_depth,
            )
            telemetry.body_parse_ms = body_parse_ms
            if http_request.headers.get(PRECOMMIT_HEADER):
                telemetry.apply_upload_precommit("present")
            precommit_receipt: _UploadPrecommitReceipt | None = None

            def _record_response(
                result: BatchSubmissionResponse,
            ) -> BatchSubmissionResponse:
                if precommit_receipt is not None:
                    precommit_receipt.outcome = result
                return result

            telemetry.refresh_from_batcher(self.active_batcher)
            log_submission_stage(
                logger,
                logging.INFO,
                "submit_received",
                telemetry,
                queue_depth=self.submit_queue_depth,
                queue_depth_by_environment=(
                    self.submit_queue_depth_by_environment
                ),
            )

            def _reject_before_quota(
                reason: RejectReason,
                *,
                reject_stage: str,
                **extra: Any,
            ) -> BatchSubmissionResponse:
                telemetry.mark_decision()
                self.record_verdict(
                    hk,
                    request.merkle_root,
                    False,
                    reason,
                    window_n=request.window_start,
                    telemetry=telemetry,
                    reject_stage=reject_stage,
                    accepted_into_pool=False,
                )
                log_submission_stage(
                    logger,
                    logging.WARNING,
                    "candidate_rejected",
                    telemetry,
                    reject_stage=reject_stage,
                    reject_reason=reason.value,
                    accepted_into_pool=False,
                    **extra,
                )
                return _record_response(
                    BatchSubmissionResponse(accepted=False, reason=reason)
                )

            # ENVELOPE SIGNATURE CHECK — runs BEFORE rate-limit increment.
            #
            # Without this gate, ``miner_hotkey`` is a plain string and
            # anyone can spam 8 unsigned requests claiming a victim's
            # hotkey, exhausting ``_per_window_counts[victim]`` before
            # the real victim's first submission lands → every honest
            # submission from the targeted hotkey returns RATE_LIMITED
            # for the rest of the window. Documented as the
            # "/submit hotkey-spoof attack" on the operator side.
            #
            # The signature is sr25519 over the canonical envelope
            # (hotkey, window, prompt_idx, merkle_root, checkpoint_hash,
            # drand_round, validator's window randomness, nonce) — see
            # ``reliquary.protocol.signatures.build_envelope_binding``.
            # An attacker can't forge it without the miner's hotkey
            # private key, and they can't replay a captured signature
            # because the binding includes the validator's per-window
            # randomness.
            #
            # The counter is touched ONLY after the signature is proven valid
            # AND after the request targets the active window, so a flood of
            # garbage signatures or stale-window replays cannot move any
            # victim's current-window quota.
            if ENFORCE_ENVELOPE_SIGNATURE:
                # We need the validator's window randomness to verify the
                # binding. Read it from the active batcher if present;
                # if there's no batcher yet, the envelope signature is
                # bound against empty randomness (the same value a
                # pre-OPEN miner would have signed against, since /state
                # exposes the empty string in that case). That's
                # consistent with the schema's default.
                _randomness_for_sig = (
                    self.active_batcher.randomness
                    if self.active_batcher is not None
                    else ""
                )
                envelope_signature_valid = await asyncio.to_thread(
                    verify_envelope_signature,
                    miner_hotkey=hk,
                    window_start=request.window_start,
                    prompt_idx=request.prompt_idx,
                    merkle_root=request.merkle_root,
                    checkpoint_hash=request.checkpoint_hash,
                    drand_round=request.drand_round,
                    randomness=_randomness_for_sig,
                    nonce=request.nonce,
                    envelope_signature=request.envelope_signature,
                )
                if not envelope_signature_valid:
                    # CONNECTION-PRIMING DEFENCE — force socket teardown.
                    #
                    # Without this header, an HTTP/1.1 keep-alive client
                    # can fire a burst of cheap BAD_ENVELOPE_SIGNATURE
                    # packets to warm a pool of TCP (+ TLS) connections
                    # at zero quota cost — those PR #35 made deliberately
                    # quota-free — then dispatch the real signed POSTs
                    # over the already-warm sockets. The observed pattern
                    # is 24 unsigned packets followed by 8 valid signed
                    # ones, giving the exploiter a ~20-30 ms RTT edge on
                    # the seal-trigger race against honest single-instance
                    # miners. Setting ``Connection: close`` makes uvicorn
                    # close the socket after the response — the attacker
                    # pays a fresh handshake on every bad packet and the
                    # warm-up is no longer free.
                    response.headers["Connection"] = "close"

                    # PER-HOTKEY BAD-ENVELOPE CAP — bandwidth + ring guard.
                    #
                    # The quota counter is STILL never bumped here (that
                    # is the invariant PR #35 added to keep an anonymous
                    # spoofer from draining a victim's legitimate
                    # 8-submission budget). What we add is a strict cap
                    # on how many BAD_ENVELOPE_SIGNATURE verdicts get
                    # written into the per-hotkey verdict ring. Past the
                    # cap we still return BAD_ENVELOPE_SIGNATURE — the
                    # response shape is unchanged — but we silently
                    # drop the ring write so a spoofer cannot flood a
                    # victim's ``/verdicts/{hotkey}`` history and
                    # displace legitimate verdicts.
                    #
                    # The first ``MAX_BAD_ENVELOPE_PER_HOTKEY_PER_WINDOW``
                    # bad packets per hotkey per window still surface in
                    # /verdicts so a legitimate miner being spoofed
                    # learns about it.
                    telemetry.mark_decision()
                    bn = self._bad_envelope_counts.get(hk, 0)
                    if bn < MAX_BAD_ENVELOPE_PER_HOTKEY_PER_WINDOW:
                        self._bad_envelope_counts[hk] = bn + 1
                        # NB: we still record the verdict against the
                        # CLAIMED hotkey — that's what the rejected packet
                        # carried — but we do NOT increment its rate-limit
                        # counter. Recording lets a legitimate miner see
                        # in /verdicts that someone's spoofing them.
                        self.record_verdict(
                            hk, request.merkle_root, False,
                            RejectReason.BAD_ENVELOPE_SIGNATURE,
                            window_n=request.window_start,
                            telemetry=telemetry,
                            reject_stage="envelope",
                            accepted_into_pool=False,
                        )
                    log_submission_stage(
                        logger,
                        logging.WARNING,
                        "candidate_rejected",
                        telemetry,
                        reject_stage="envelope",
                        reject_reason=RejectReason.BAD_ENVELOPE_SIGNATURE.value,
                        accepted_into_pool=False,
                    )
                    return BatchSubmissionResponse(
                        accepted=False,
                        reason=RejectReason.BAD_ENVELOPE_SIGNATURE,
                    )

            # v2.1: reject if state != OPEN
            if self._current_state != WindowState.OPEN:
                if self._late_drop_callback is not None:
                    self._late_drop_callback(
                        request.miner_hotkey, "window_not_active",
                    )
                telemetry.mark_decision()
                self.record_verdict(
                    hk, request.merkle_root, False, RejectReason.WINDOW_NOT_ACTIVE,
                    window_n=request.window_start,
                    telemetry=telemetry,
                    reject_stage="window_state",
                    accepted_into_pool=False,
                )
                log_submission_stage(
                    logger,
                    logging.WARNING,
                    "candidate_rejected",
                    telemetry,
                    reject_stage="window_state",
                    reject_reason=RejectReason.WINDOW_NOT_ACTIVE.value,
                    accepted_into_pool=False,
                )
                return BatchSubmissionResponse(
                    accepted=False, reason=RejectReason.WINDOW_NOT_ACTIVE,
                )

            # Route only homogeneous, explicitly known environments. Honest
            # miners already emit one environment per group. The old fallback
            # sent an unknown or mixed-env request to the first active batcher,
            # leaving an unsigned routing field able to select the wrong
            # verifier path.
            submission_env_names = {
                rollout.env_name for rollout in request.rollouts
            }
            if len(submission_env_names) != 1:
                return _reject_before_quota(
                    RejectReason.BAD_SCHEMA,
                    reject_stage="environment",
                    submitted_environment_count=len(submission_env_names),
                )
            submission_env_name = next(iter(submission_env_names))
            if not self._active_batchers and self.active_batcher is None:
                telemetry.mark_decision()
                log_submission_stage(
                    logger,
                    logging.WARNING,
                    "candidate_rejected",
                    telemetry,
                    reject_stage="window",
                    reject_reason="no_active_window",
                    accepted_into_pool=False,
                )
                raise HTTPException(status_code=503, detail="no_active_window")
            batcher = self._active_batchers.get(submission_env_name)
            if batcher is None:
                # Loose MagicMock fixtures and legacy single-batcher embedding
                # do not always expose a stable env.name. Production batchers
                # do, and must never route an unknown environment through the
                # scalar accessor.
                active_env_name = getattr(
                    getattr(self.active_batcher, "env", None), "name", None
                )
                if (
                    not self._active_batchers
                    or active_env_name is None
                    or _is_mock_like(active_env_name)
                ):
                    batcher = self.active_batcher
            if batcher is None:
                return _reject_before_quota(
                    RejectReason.BAD_SCHEMA,
                    reject_stage="environment",
                    submitted_environment=submission_env_name,
                )
            telemetry.refresh_from_batcher(batcher)
            if request.window_start != batcher.window_start:
                telemetry.mark_decision()
                log_submission_stage(
                    logger,
                    logging.WARNING,
                    "candidate_rejected",
                    telemetry,
                    reject_stage="window",
                    reject_reason=RejectReason.WINDOW_MISMATCH.value,
                    accepted_into_pool=False,
                )
                raise HTTPException(status_code=409, detail="window_mismatch")

            precommit_reserved = False
            receipt_id = http_request.headers.get(PRECOMMIT_HEADER)
            if receipt_id:
                precommit_status, receipt = self._claim_upload_precommit(
                    receipt_id,
                    request,
                    batcher=batcher,
                    environment=submission_env_name,
                    payload_bytes=payload_bytes,
                    payload_sha256=payload_sha256,
                    body_completed_at=body_completed_at,
                )
                if precommit_status == "expired":
                    telemetry.apply_upload_precommit("expired")
                    return _reject_before_quota(
                        RejectReason.PRECOMMIT_EXPIRED,
                        reject_stage="upload_precommit",
                    )
                if precommit_status == "invalid" or receipt is None:
                    telemetry.apply_upload_precommit("invalid")
                    return _reject_before_quota(
                        RejectReason.PRECOMMIT_INVALID,
                        reject_stage="upload_precommit",
                    )
                if precommit_status == "replay":
                    telemetry.apply_upload_precommit(
                        "replay",
                        arrival_ts=receipt.precommit_arrival_ts,
                    )
                    return receipt.outcome or BatchSubmissionResponse(
                        accepted=True, reason=RejectReason.SUBMITTED,
                    )

                precommit_reserved = True
                precommit_receipt = receipt
                telemetry.apply_upload_precommit(
                    "valid",
                    arrival_ts=receipt.precommit_arrival_ts,
                )
                resolver = getattr(
                    type(batcher), "resolve_upload_precommit", None
                )

                def _release_upload_precommit() -> None:
                    if resolver is not None:
                        resolver(batcher, receipt_id)

                http_request.state.release_upload_precommit = (
                    _release_upload_precommit
                )
                logger.info(
                    "upload_precommit_revealed window=%d env=%s prompt=%d "
                    "hotkey=%s payload_bytes=%d upload_ms=%.3f",
                    request.window_start,
                    submission_env_name,
                    request.prompt_idx,
                    hk[:12],
                    payload_bytes,
                    max(0.0, body_completed_at - t_arrival) * 1000.0,
                )
            else:
                collection_closed = getattr(
                    type(batcher), "collection_closed", None
                )
                if collection_closed is not None and collection_closed(batcher):
                    return _reject_before_quota(
                        RejectReason.PRECOMMIT_REQUIRED,
                        reject_stage="upload_precommit",
                    )

            if (
                FORCED_SEED_ENFORCE
                and bool(getattr(batcher, "current_checkpoint_hash", ""))
                and request.protocol_version != FORCED_SEED_PROTOCOL_VERSION
            ):
                return _reject_before_quota(
                    RejectReason.SEED_MISMATCH,
                    reject_stage="forced_seed_protocol",
                    submitted_protocol_version=request.protocol_version,
                    required_protocol_version=FORCED_SEED_PROTOCOL_VERSION,
                )

            legacy_merkle_status = await asyncio.to_thread(
                self._observe_legacy_merkle,
                request,
                telemetry,
                env_name=submission_env_name,
            )
            if (
                LEGACY_MERKLE_ROOT_ENFORCE
                and legacy_merkle_status != "match"
            ):
                return _reject_before_quota(
                    RejectReason.MERKLE_ROOT_MISMATCH,
                    reject_stage="legacy_merkle",
                    legacy_merkle_status=legacy_merkle_status,
                )

            # Only registered subnet hotkeys may consume submission quota or
            # proof capacity. The signature gate above proves ownership of the
            # claimed hotkey; this local lookup binds that identity to the
            # last complete metagraph snapshot without request-path RPC.
            registration_reason = await self._registration_reject_reason(hk)
            if registration_reason is not None:
                telemetry.mark_decision()
                self.record_verdict(
                    hk,
                    request.merkle_root,
                    False,
                    registration_reason,
                    window_n=request.window_start,
                    telemetry=telemetry,
                    reject_stage="registration",
                    accepted_into_pool=False,
                )
                log_submission_stage(
                    logger,
                    logging.WARNING,
                    "candidate_rejected",
                    telemetry,
                    reject_stage="registration",
                    reject_reason=registration_reason.value,
                    accepted_into_pool=False,
                )
                return _record_response(
                    BatchSubmissionResponse(
                        accepted=False,
                        reason=registration_reason,
                    )
                )

            # Rate limit AFTER signature verification and active-window
            # binding. A signed stale-window replay or a miner still catching
            # up after restart must not burn the hotkey's quota for the new
            # window. Once the request is known to target this live window,
            # count it before cheap rejects/GRAIL so spam still self-throttles.
            n = self._per_window_counts.get(hk, 0)
            if (
                not precommit_reserved
                and n >= MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW
            ):
                if self._late_drop_callback is not None:
                    self._late_drop_callback(hk, "rate_limited")
                telemetry.mark_decision()
                self.record_verdict(
                    hk, request.merkle_root, False, RejectReason.RATE_LIMITED,
                    window_n=request.window_start,
                    telemetry=telemetry,
                    reject_stage="rate_limit",
                    accepted_into_pool=False,
                )
                log_submission_stage(
                    logger,
                    logging.WARNING,
                    "candidate_rejected",
                    telemetry,
                    reject_stage="rate_limit",
                    reject_reason=RejectReason.RATE_LIMITED.value,
                    accepted_into_pool=False,
                )
                return _record_response(
                    BatchSubmissionResponse(
                        accepted=False, reason=RejectReason.RATE_LIMITED,
                    )
                )
            if not precommit_reserved:
                self._per_window_counts[hk] = n + 1

            def _refund_current_quota() -> None:
                self._refund_submission_quota(hk)

            def _prompt_source_unavailable(
                exc: PromptSourceUnavailable,
            ) -> None:
                # Infrastructure availability is not miner behavior. Refund
                # the exact quota reservation made above and return a typed
                # retryable service response without creating a protocol
                # verdict that could later be interpreted as miner fault.
                _refund_current_quota()
                self._prompt_source_unavailable_total += 1
                telemetry.refresh_from_batcher(batcher, at_decision=True)
                telemetry.mark_decision()
                logger.error(
                    "prompt_source_unavailable environment=%s window=%d "
                    "prompt=%d error_type=%s quota_refunded=true",
                    submission_env_name,
                    request.window_start,
                    request.prompt_idx,
                    type(exc).__name__,
                )
                log_submission_stage(
                    logger,
                    logging.ERROR,
                    "service_unavailable",
                    telemetry,
                    reject_stage="prompt_source",
                    reject_reason="prompt_source_unavailable",
                    accepted_into_pool=False,
                    submitted_environment=submission_env_name,
                    prompt_source_error_type=type(exc).__name__,
                    quota_refunded=True,
                )
                _record_response(
                    BatchSubmissionResponse(
                        accepted=False,
                        reason=RejectReason.WORKER_DROPPED,
                    )
                )
                raise HTTPException(
                    status_code=503,
                    detail="prompt_source_unavailable",
                    headers={"Retry-After": "30"},
                ) from exc

            # Early-cutoff: once the batcher has sealed (B_BATCH distinct
            # non-cooldown valid submissions received), ``select_batch``
            # will pick those by ``arrived_at``. Further submissions land
            # strictly later in arrival order, so they cannot displace
            # any of the already-selected entries. Queuing them costs
            # ~5–25 s of GRAIL forward pass per item with zero protocol
            # benefit, inflates the OPEN→TRAIN latency, and lets the
            # worker keep grinding into the TRAINING phase. Reject here
            # the moment the batch is closed.
            if batcher.is_sealed():
                if self._late_drop_callback is not None:
                    self._late_drop_callback(hk, "batch_filled")
                telemetry.refresh_from_batcher(batcher, at_decision=True)
                telemetry.mark_decision()
                self.record_verdict(
                    hk, request.merkle_root, False, RejectReason.BATCH_FILLED,
                    window_n=request.window_start,
                    telemetry=telemetry,
                    reject_stage="seal",
                    accepted_into_pool=False,
                )
                log_submission_stage(
                    logger,
                    logging.WARNING,
                    "candidate_rejected",
                    telemetry,
                    reject_stage="seal",
                    reject_reason=RejectReason.BATCH_FILLED.value,
                    batch_filled_reason="batch_already_sealed",
                    current_valid_count=batcher.valid_count,
                    trigger_round=getattr(batcher, "_seal_trigger_round", None),
                    accepted_into_pool=False,
                )
                return _record_response(
                    BatchSubmissionResponse(
                        accepted=False, reason=RejectReason.BATCH_FILLED,
                    )
                )

            # Cheap rejects pre-queue: every check below is O(1) against
            # batcher fields the worker re-runs inside _accept_locked. Doing
            # them here keeps the worker free for GRAIL forward passes on
            # submissions that have a real chance of being batched. Without
            # this, a STALE_ROUND or WRONG_CHECKPOINT submission has to wait
            # in the queue behind a 5–25 s GRAIL verify of an honest
            # submission ahead of it — minutes of latency on what should be
            # a microsecond rejection. The check order mirrors the worker
            # path in ``GrpoWindowBatcher._accept_locked`` so the same
            # submission always gets the same reject_reason regardless of
            # which path decides. Concurrent batcher mutation between the
            # read here and the worker is benign — the worker re-verifies
            # under the lock.
            def _cheap_reject(
                reason: RejectReason,
                *,
                reject_stage: str,
                **extra: Any,
            ) -> BatchSubmissionResponse:
                logger.warning(
                    "rejected prompt=%d hotkey=%s drand_round=%d reason=%s rewards=%s",
                    request.prompt_idx, hk[:12], request.drand_round,
                    reason.value,
                    [r.reward for r in request.rollouts],
                )
                telemetry.refresh_from_batcher(batcher, at_decision=True)
                telemetry.mark_decision()
                self.record_verdict(
                    hk, request.merkle_root, False, reason,
                    window_n=request.window_start,
                    telemetry=telemetry,
                    reject_stage=reject_stage,
                    accepted_into_pool=False,
                )
                log_submission_stage(
                    logger,
                    logging.WARNING,
                    "candidate_rejected",
                    telemetry,
                    reject_stage=reject_stage,
                    reject_reason=reason.value,
                    accepted_into_pool=False,
                    **extra,
                )
                return _record_response(
                    BatchSubmissionResponse(accepted=False, reason=reason)
                )

            if batcher.current_checkpoint_hash and request.checkpoint_hash != batcher.current_checkpoint_hash:
                return _cheap_reject(
                    RejectReason.WRONG_CHECKPOINT,
                    reject_stage="checkpoint",
                )
            if batcher.drand_round_check_enabled:
                if precommit_reserved and precommit_receipt is not None:
                    drand_observation = precommit_receipt.drand_observation
                    drand_timing_source = "precommit_arrival"
                elif hasattr(type(batcher), "observe_drand_round"):
                    drand_observation = batcher.observe_drand_round(
                        request.drand_round, t_arrival=t_arrival,
                    )
                    drand_timing_source = "body_arrival"
                else:
                    round_reject = batcher.validate_drand_round(
                        request.drand_round, t_arrival=t_arrival,
                    )
                    drand_observation = self._fallback_drand_observation(
                        request, batcher, round_reject,
                    )
                    drand_timing_source = "body_arrival"
                telemetry.apply_drand(drand_observation)
                log_submission_stage(
                    logger,
                    logging.INFO,
                    "drand_validated",
                    telemetry,
                    reject_stage=(
                        "drand" if drand_observation.reject_reason else None
                    ),
                    reject_reason=(
                        drand_observation.reject_reason.value
                        if drand_observation.reject_reason else None
                    ),
                    submitted_drand_round=drand_observation.submitted_drand_round,
                    arrival_drand_round=drand_observation.arrival_drand_round,
                    drand_delta=drand_observation.drand_delta,
                    drand_tolerance=drand_observation.drand_tolerance,
                    drand_status=drand_observation.drand_status,
                    current_round=drand_observation.arrival_drand_round,
                    drand_timing_source=drand_timing_source,
                )
                round_reject = drand_observation.reject_reason
                if round_reject is not None:
                    return _cheap_reject(round_reject, reject_stage="drand")
            # v2.3 seal extension: once the batcher has captured a
            # trigger drand round (the B-th distinct prompt has landed),
            # submissions whose drand_round is past that trigger arrived
            # in a later chronological tier than the boundary fair-split
            # can absorb. Reject pre-queue with BATCH_FILLED so they
            # don't sit in the worker queue costing a futile dequeue.
            trigger_round = batcher._seal_trigger_round
            if (
                not getattr(batcher, "difficulty_auction_enabled", False)
                and trigger_round is not None
                and request.drand_round > trigger_round
            ):
                return _cheap_reject(
                    RejectReason.BATCH_FILLED,
                    reject_stage="seal_extension",
                    batch_filled_reason="submitted_round_gt_seal_trigger_round",
                    current_valid_count=batcher.valid_count,
                    trigger_round=trigger_round,
                )
            try:
                environment_size = await asyncio.to_thread(len, batcher.env)
            except PromptSourceUnavailable as exc:
                _prompt_source_unavailable(exc)
            if request.prompt_idx >= environment_size:
                return _cheap_reject(RejectReason.BAD_PROMPT_IDX, reject_stage="prompt")
            prompt_range = getattr(batcher, "prompt_range", None)
            if prompt_range is not None:
                lo, hi = prompt_range
                if not (lo <= request.prompt_idx < hi):
                    return _cheap_reject(
                        RejectReason.PROMPT_OUT_OF_RANGE,
                        reject_stage="prompt_range",
                    )
            if request.prompt_idx in batcher.cooldown_prompts_snapshot:
                return _cheap_reject(
                    RejectReason.PROMPT_IN_COOLDOWN,
                    reject_stage="cooldown",
                    batch_filled_reason="prompt_in_cooldown",
                )
            if batcher.prompt_submission_count(request.prompt_idx) >= MAX_SUBMISSIONS_PER_PROMPT:
                return _cheap_reject(
                    RejectReason.PROMPT_FULL,
                    reject_stage="prompt_capacity",
                    batch_filled_reason="prompt_duplicate_or_full",
                )

            # Materialize the exact prompt before reserving proof work. This
            # is an availability check, not a second prompt contract: the
            # canonical binding below and the batcher still perform their
            # normal validation. Keeping it here guarantees a third-party
            # source outage remains quota-neutral and never consumes GPU debt.
            prompt_loader = getattr(batcher.env, "get_problem", None)
            if callable(prompt_loader) and not _is_mock_like(prompt_loader):
                try:
                    await asyncio.to_thread(
                        prompt_loader, request.prompt_idx
                    )
                except PromptSourceUnavailable as exc:
                    _prompt_source_unavailable(exc)

            reserve_proof = getattr(
                type(batcher), "try_reserve_proof_admission", None
            )
            if reserve_proof is not None:
                # Runs in a thread: the canonical-prompt check calls
                # env.get_problem, which for the lazy parquet dataset may do a
                # blocking row-group fetch — must not stall the event loop.
                try:
                    preflight_reason, preflight_stage = await asyncio.to_thread(
                        _proof_free_submission_reject, request, batcher
                    )
                except PromptSourceUnavailable as exc:
                    _prompt_source_unavailable(exc)
                if preflight_reason is not None:
                    return _cheap_reject(
                        preflight_reason,
                        reject_stage=preflight_stage or "preflight",
                    )

                reserve_logical = getattr(
                    type(batcher), "try_reserve_logical_group", None
                )
                if reserve_logical is not None:
                    try:
                        logical_reserved, logical_reason = await asyncio.to_thread(
                            reserve_logical, batcher, request
                        )
                    except (TypeError, ValueError, OverflowError):
                        return _cheap_reject(
                            RejectReason.BAD_TOKENS,
                            reject_stage="logical_dedup",
                        )
                    if not logical_reserved:
                        _refund_current_quota()
                        reject_reason = (
                            RejectReason.REGISTRATION_UNAVAILABLE
                            if logical_reason == "operator_unmapped"
                            else RejectReason.HASH_DUPLICATE
                        )
                        return _cheap_reject(
                            reject_reason,
                            reject_stage=(
                                "operator_mapping"
                                if logical_reason == "operator_unmapped"
                                else "logical_dedup"
                            ),
                            logical_group_reason=logical_reason,
                            quota_refunded=True,
                        )

                admitted, admission_reason = batcher.try_reserve_proof_admission(
                    request
                )
                if not admitted:
                    self._cancel_logical_group_reservation(batcher, request)
                    if self._late_drop_callback is not None:
                        self._late_drop_callback(hk, "proof_admission_full")
                    return _cheap_reject(
                        RejectReason.BATCH_FILLED,
                        reject_stage="proof_admission",
                        batch_filled_reason=admission_reason,
                        current_valid_count=batcher.valid_count,
                        trigger_round=getattr(
                            batcher, "_seal_trigger_round", None
                        ),
                        proof_admission_count=getattr(
                            batcher, "proof_admission_count", None
                        ),
                        post_trigger_proof_admission_count=getattr(
                            batcher,
                            "post_trigger_proof_admission_count",
                            None,
                        ),
                        post_trigger_proof_admission_limit=(
                            MAX_POST_TRIGGER_PROOF_CANDIDATES
                        ),
                    )

            # Under TestClient (no worker running) validation is synchronous;
            # under uvicorn we enqueue and return ``SUBMITTED``. In auction mode
            # worker ``ACCEPTED`` means admitted to the pending pool, and the
            # seal-time /verdicts record reports proof/selection/reward outcome.
            # Reservation bounds apply before either path enters grading.
            if self._worker_task is None:
                started, start_reason = self._start_proof_admission(
                    batcher, request,
                )
                if not started:
                    self._cancel_logical_group_reservation(batcher, request)
                    if self._late_drop_callback is not None:
                        self._late_drop_callback(hk, "proof_admission_full")
                    return _cheap_reject(
                        RejectReason.BATCH_FILLED,
                        reject_stage="proof_admission",
                        batch_filled_reason=start_reason,
                    )
                telemetry.mark_proof_started(queue_depth=0)
                log_submission_stage(
                    logger,
                    logging.INFO,
                    "proof_started",
                    telemetry,
                    reject_stage=None,
                    reject_reason=None,
                )
                try:
                    telemetry.mark_admission_started()
                    try:
                        resp = self._call_accept_submission(
                            batcher, request, telemetry,
                        )
                    except PromptSourceUnavailable as exc:
                        _prompt_source_unavailable(exc)
                    finally:
                        telemetry.mark_admission_finished()
                finally:
                    self._finish_proof_admission(batcher, request)
                telemetry.refresh_from_batcher(batcher, at_decision=True)
                telemetry.mark_decision(verified=True)
                self._record_admission_latency(batcher, telemetry)
                if (
                    not resp.accepted
                    and resp.reason is RejectReason.WORKER_DROPPED
                    and self._worker_drop_refunds_quota(request)
                ):
                    self._refund_submission_quota(hk)
                log_submission_stage(
                    logger,
                    logging.INFO,
                    "proof_finished",
                    telemetry,
                    accepted=resp.accepted,
                    reason=resp.reason.value,
                )
                if resp.accepted:
                    log_submission_stage(
                        logger,
                        logging.INFO,
                        "candidate_accepted",
                        telemetry,
                        reject_stage="none",
                        reject_reason="none",
                        accepted_into_pool=True,
                    )
                else:
                    log_submission_stage(
                        logger,
                        logging.WARNING,
                        "candidate_rejected",
                        telemetry,
                        reject_stage="proof",
                        reject_reason=resp.reason.value,
                        accepted_into_pool=False,
                    )
                # Sync path (tests) — the real verdict is already known
                # before we return, so record it directly.
                self.record_verdict(
                    hk, request.merkle_root, resp.accepted, resp.reason,
                    window_n=request.window_start,
                    telemetry=telemetry,
                    reject_stage=None if resp.accepted else "proof",
                    accepted_into_pool=resp.accepted,
                )
                return _record_response(resp)

            submit_queue = self._submission_queue_for_environment(
                submission_env_name
            )
            telemetry.mark_enqueued(queue_depth=submit_queue.qsize())
            try:
                submit_queue.put_nowait((request, batcher, telemetry))
            except asyncio.QueueFull:
                self._cancel_proof_admission(batcher, request)
                self._cancel_logical_group_reservation(batcher, request)
                if self._late_drop_callback is not None:
                    self._late_drop_callback(hk, "proof_queue_full")
                return _cheap_reject(
                    RejectReason.BATCH_FILLED,
                    reject_stage="proof_admission",
                    batch_filled_reason="proof_queue_full",
                    proof_queue_limit=MAX_PENDING_PROOF_QUEUE_DEPTH,
                )
            return _record_response(
                BatchSubmissionResponse(
                    accepted=True, reason=RejectReason.SUBMITTED,
                )
            )

        @app.get("/state", response_model=GrpoBatchState)
        async def state(env: str | None = None) -> GrpoBatchState:
            """Current window + checkpoint state. Lock-free: reads only the
            batcher's snapshot fields (set at construction) and the atomic
            ``valid_count`` counter. The submit worker holds ``batcher._lock``
            for up to ~25s per GRAIL verify, so this handler MUST NOT touch
            it — otherwise miners polling /state starve the event loop and
            timeout cascades hit every endpoint (see 2026-05-12 outage).

            ``cooldown_prompts`` is PER-ENV (``prompt_idx`` indexes one env's
            problem set), so a multi-env window has a distinct cooldown set
            per env. The optional ``env`` query param selects which env's
            batcher to report; without it we report the first active batcher
            (legacy single-env behavior). Miners must poll once per env to
            learn each env's real cooldown — the flat field can only carry
            one. window_n/randomness/checkpoint are identical across envs.
            """
            if env is not None:
                batcher = self._active_batchers.get(env)
                if batcher is None:
                    if not self._active_batchers:
                        raise HTTPException(status_code=503, detail="no_active_window")
                    raise HTTPException(status_code=404, detail="unknown_env")
            else:
                batcher = self.active_batcher
                if batcher is None:
                    raise HTTPException(status_code=503, detail="no_active_window")
            cp = self._current_checkpoint
            return GrpoBatchState(
                state=self._current_state,
                window_n=batcher.window_start,
                anchor_block=batcher.window_start,
                cooldown_prompts=batcher.cooldown_prompts_snapshot,
                valid_submissions=(
                    getattr(batcher, "pending_count", batcher.valid_count)
                    if getattr(batcher, "difficulty_auction_enabled", False)
                    else batcher.valid_count
                ),
                checkpoint_n=cp.checkpoint_n if cp else 0,
                checkpoint_repo_id=cp.repo_id if cp else None,
                checkpoint_revision=cp.revision if cp else None,
                randomness=batcher.randomness,
            )

        @app.get("/runtime-contract", response_model=RuntimeContract)
        async def runtime_contract() -> RuntimeContract:
            """Optional numerical-runtime telemetry capability.

            Kept separate from strict `/state` so adding this feature remains
            wire-compatible with older miners.
            """
            return RuntimeContract(
                validator_profile=RuntimeFingerprint.model_validate(
                    self._runtime_fingerprint
                )
            )

        @app.get("/checkpoint")
        async def checkpoint():
            cp = self._current_checkpoint
            if cp is None:
                raise HTTPException(status_code=404, detail="no_checkpoint")
            return {
                "checkpoint_n": cp.checkpoint_n,
                "repo_id": cp.repo_id,
                "revision": cp.revision,
                "signature": cp.signature,
            }

        @app.get(
            "/verdicts/{hotkey}",
            response_model=VerdictsResponse,
            response_model_exclude_none=True,
        )
        async def verdicts(hotkey: str, since: float = 0.0) -> VerdictsResponse:
            """Recent per-submission verdicts for ``hotkey``, ordered by
            ``ts`` ascending. The default ``since=0`` returns every verdict
            currently in the ring; pass the timestamp of the last verdict
            you saw to get only newer ones (incremental polling).

            Bounded read — at most ``VERDICT_CAP_PER_HOTKEY`` entries per
            hotkey live in the ring, so even a degenerate ``since=0`` poll
            never returns more than ~200 entries. Lock-free in the same way
            ``/state`` is (event-loop-only writes, atomic dict.get).

            Why this exists: ``/submit`` under the production worker path
            returns a provisional ``SUBMITTED`` sentinel, not the real
            verdict — that's known only after the worker drains the queue
            and runs the full verification pipeline (~5-25 s of GRAIL per
            item). Without this endpoint, the truth was only visible via
            the R2 archive (minutes-late, batched per window). Now miners
            can learn within seconds whether a specific submission cleared
            GRAIL, was rejected as a duplicate hash, hit the rate limit,
            or failed any other check — diagnosable by ``merkle_root``.

            Privacy: same trust model as the R2 archive (public). Anyone
            can query any hotkey's verdicts; we don't auth this. If you
            need confidential feedback, run a private validator.
            """
            ring = self._verdicts.get(hotkey)
            if not ring:
                return VerdictsResponse(verdicts=[])
            out = [
                Verdict(**entry) for entry in ring
                if entry["ts"] > since
            ]
            return VerdictsResponse(verdicts=out)

        # Add this last so Starlette places it outermost, ahead of the
        # BaseHTTPMiddleware used by ``stamp_arrival``. Otherwise FastAPI may
        # translate an over-limit receive exception into a generic 400 before
        # the limiter can emit the intended 413 for chunked requests.
        app.add_middleware(
            _SubmissionBodyLimitMiddleware,
            max_bytes=MAX_SUBMISSION_PAYLOAD_BYTES,
        )
        return app

    async def _process_auction_submission(
        self,
        item: _QueuedAuctionSubmission,
        queue: asyncio.Queue,
    ) -> None:
        receipt = item.receipt
        batcher = item.batcher
        telemetry = item.telemetry
        environment = receipt.environment
        telemetry.mark_dequeued(queue_depth=queue.qsize())
        telemetry.refresh_from_batcher(batcher)
        self._admission_active_by_environment[environment] += 1
        self._admission_inflight_items[receipt.receipt_id] = item
        self._inflight_proofs += 1
        self._inflight_proofs_by_environment[environment] += 1
        response: BatchSubmissionResponse | None = None
        reject_stage = "admission_worker"
        request: BatchSubmissionRequest | None = None
        admission_started = False
        identity_reserved = False
        cancel_identity_on_exit = False
        wall_seconds = (
            CODE_ADMISSION_WALL_SECONDS
            if environment == "opencodeinstruct"
            else MATH_ADMISSION_WALL_SECONDS
        )
        deadline = time.monotonic() + wall_seconds

        def reject_without_request(
            reason: RejectReason,
            stage: str,
        ) -> BatchSubmissionResponse:
            rejector = getattr(type(batcher), "_reject", None)
            if rejector is not None:
                return rejector(
                    batcher,
                    reason,
                    hotkey=receipt.miner_hotkey,
                    prompt_idx=receipt.prompt_idx,
                    telemetry=telemetry,
                    reject_stage=stage,
                )
            return BatchSubmissionResponse(accepted=False, reason=reason)

        try:
            if batcher not in self._active_batchers.values():
                response = reject_without_request(
                    RejectReason.WORKER_DROPPED, "worker"
                )
                reject_stage = "worker"
                return

            telemetry.mark_proof_started(queue_depth=queue.qsize())
            telemetry.mark_admission_started()
            remaining = max(0.001, deadline - time.monotonic())
            materialization_pool = self._admission_materialization_pool
            if materialization_pool is None:
                raise RuntimeError("admission materialization pool unavailable")
            try:
                materials = await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(
                        materialization_pool,
                        batcher.materialize_admission_problem,
                        receipt.prompt_idx,
                    ),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                self._admission_timeouts[environment] += 1
                raise
            prepared: PreparedSubmission = await self._run_admission_process(
                environment,
                prepare_submission,
                item.raw_body,
                self._admission_binding(receipt),
                materials,
                self._admission_context_for(batcher),
                deadline,
                wall_seconds=max(0.001, deadline - time.monotonic()),
            )
            if prepared.legacy_merkle_status is not None:
                telemetry.apply_legacy_merkle(
                    status=prepared.legacy_merkle_status,
                    computed_root=None,
                    enforced=LEGACY_MERKLE_ROOT_ENFORCE,
                )
            request = prepared.request
            telemetry.reward_grading_ms = prepared.reward_grading_ms
            telemetry.body_parse_ms = prepared.body_parse_ms
            telemetry.admission_prepare_ms = prepared.preparation_ms
            if request is None:
                reject_stage = prepared.reject_stage or "admission_worker"
                response = reject_without_request(
                    prepared.reject_reason or RejectReason.BAD_SCHEMA,
                    reject_stage,
                )
                return
            request._payload_bytes = receipt.payload_bytes
            self._admission_inflight_requests[receipt.receipt_id] = request

            structurally_authenticated = (
                len(prepared.rollout_hashes) == len(request.rollouts)
                and prepared.selection_digest is not None
            )
            identity_should_be_reserved = structurally_authenticated and (
                prepared.reject_reason is not RejectReason.WORKER_DROPPED
                or prepared.grader_failure_reason == "crash"
            )
            if identity_should_be_reserved:
                reserved, identity_reason, identity_stage = (
                    batcher.reserve_prepared_identity(
                        request, prepared.rollout_hashes
                    )
                )
                if not reserved:
                    reject_stage = identity_stage or "admission_identity"
                    response = batcher.reject_prepared_submission(
                        request,
                        identity_reason or RejectReason.WORKER_DROPPED,
                        reject_stage,
                        telemetry=telemetry,
                    )
                    return
                identity_reserved = True

            if prepared.reject_reason is not None:
                reject_stage = prepared.reject_stage or "admission_worker"
                response = batcher.reject_prepared_submission(
                    request,
                    prepared.reject_reason,
                    reject_stage,
                    telemetry=telemetry,
                    grader_failure_reason=prepared.grader_failure_reason,
                )
                return

            if not identity_reserved:
                reject_stage = "admission_identity"
                response = batcher.reject_prepared_submission(
                    request,
                    RejectReason.WORKER_DROPPED,
                    reject_stage,
                    telemetry=telemetry,
                )
                return

            started, start_reason = batcher.start_revealed_admission(
                receipt.receipt_id, request
            )
            if not started:
                batcher.cancel_logical_group_reservation(request)
                reject_stage = "proof_admission"
                response = batcher.reject_prepared_submission(
                    request,
                    RejectReason.BATCH_FILLED,
                    reject_stage,
                    telemetry=telemetry,
                )
                logger.warning(
                    "auction_admission_start_rejected window=%d env=%s "
                    "prompt=%d reason=%s",
                    receipt.window_start,
                    environment,
                    receipt.prompt_idx,
                    start_reason,
                )
                return
            admission_started = True

            reject_stage = prepared.reject_stage or "proof"
            response = batcher.accept_prepared_submission(
                prepared, telemetry=telemetry
            )
        except (asyncio.TimeoutError, BrokenProcessPool):
            cancel_identity_on_exit = identity_reserved and not admission_started
            reject_stage = "admission_timeout"
            if request is not None:
                response = batcher.reject_prepared_submission(
                    request,
                    RejectReason.WORKER_DROPPED,
                    reject_stage,
                    telemetry=telemetry,
                )
            else:
                response = reject_without_request(
                    RejectReason.WORKER_DROPPED, reject_stage
                )
        except PromptSourceUnavailable:
            cancel_identity_on_exit = identity_reserved and not admission_started
            self._prompt_source_unavailable_total += 1
            reject_stage = "prompt_source"
            if request is not None:
                batcher.cancel_logical_group_reservation(request)
                response = batcher.reject_prepared_submission(
                    request,
                    RejectReason.WORKER_DROPPED,
                    reject_stage,
                    telemetry=telemetry,
                )
            else:
                response = reject_without_request(
                    RejectReason.WORKER_DROPPED, reject_stage
                )
        except Exception:
            cancel_identity_on_exit = identity_reserved and not admission_started
            logger.exception(
                "auction admission worker failed window=%d env=%s prompt=%d",
                receipt.window_start,
                environment,
                receipt.prompt_idx,
            )
            reject_stage = "admission_worker"
            if request is not None:
                response = batcher.reject_prepared_submission(
                    request,
                    RejectReason.WORKER_DROPPED,
                    reject_stage,
                    telemetry=telemetry,
                )
            else:
                response = reject_without_request(
                    RejectReason.WORKER_DROPPED, reject_stage
                )
        finally:
            if cancel_identity_on_exit and request is not None:
                batcher.cancel_logical_group_reservation(request)
            if admission_started and request is not None:
                batcher.finish_proof_admission(request)
            if response is None:
                response = BatchSubmissionResponse(
                    accepted=False, reason=RejectReason.WORKER_DROPPED
                )
            if receipt.terminal and receipt.outcome is not None:
                response = receipt.outcome
            else:
                self._complete_upload_receipt(receipt, response)
            telemetry.mark_admission_finished()
            telemetry.refresh_from_batcher(batcher, at_decision=True)
            telemetry.mark_decision(verified=True)
            self._record_admission_latency(batcher, telemetry)
            if (
                not response.accepted
                and response.reason is RejectReason.WORKER_DROPPED
                and (
                    request is None
                    or self._worker_drop_refunds_quota(request)
                )
            ):
                self._refund_submission_quota(receipt.miner_hotkey)
            self._record_raw_terminal(
                receipt,
                telemetry,
                response,
                stage=reject_stage,
            )
            self._admission_active_by_environment[environment] = max(
                0,
                self._admission_active_by_environment[environment] - 1,
            )
            self._inflight_proofs = max(0, self._inflight_proofs - 1)
            self._inflight_proofs_by_environment[environment] = max(
                0,
                self._inflight_proofs_by_environment[environment] - 1,
            )
            self._admission_inflight_items.pop(receipt.receipt_id, None)
            self._admission_inflight_requests.pop(receipt.receipt_id, None)

    async def _submit_worker(
        self,
        submit_queue: asyncio.Queue | None = None,
    ) -> None:
        # Lazy import — keeps the module loadable in CPU-only test envs.
        from reliquary.validator.service import _try_empty_cuda_cache

        queue = submit_queue if submit_queue is not None else self._submit_queue
        while True:
            try:
                item = await queue.get()
            except asyncio.CancelledError:
                return
            if isinstance(item, _QueuedAuctionSubmission):
                await self._process_auction_submission(item, queue)
                continue
            if len(item) == 3:
                request, batcher, telemetry = item
            else:
                request, batcher = item
                telemetry = SubmitTelemetry.from_request(
                    request, t_arrival=time.time(),
                )
            telemetry.mark_dequeued(queue_depth=queue.qsize())
            telemetry.refresh_from_batcher(batcher)
            # Silently drop items whose batcher is no longer the active one.
            # This is what relieves pressure from a saturated window: the
            # queue is bounded but a busy window can still hold pending items
            # behind the in-flight GRAIL. As soon
            # as the service's main loop opens the next window and swaps
            # the batchers dict, every leftover item is for a sealed
            # batcher whose ``_valid`` will never be re-archived — running
            # GRAIL on them would burn ~5-25s per item for nothing and
            # would keep the next window starving for cycles. We log at
            # info so operators can confirm the drain is happening; the
            # miner has already received a provisional ``SUBMITTED`` from
            # the /submit response and learns the real outcome (or its
            # absence) from the R2 archive.
            if batcher not in self._active_batchers.values():
                self._cancel_proof_admission(batcher, request)
                self._cancel_logical_group_reservation(batcher, request)
                telemetry.refresh_from_batcher(batcher, at_decision=True)
                telemetry.mark_decision()
                logger.info(
                    "dropping late submission prompt=%d hotkey=%s "
                    "drand_round=%d (batcher window=%d no longer active)",
                    request.prompt_idx, request.miner_hotkey[:12],
                    request.drand_round, batcher.window_start,
                )
                if self._late_drop_callback is not None:
                    self._late_drop_callback(
                        request.miner_hotkey, "worker_dropped",
                    )
                # Surface to the miner via /verdicts so they don't keep
                # interpreting the SUBMITTED sentinel as an accept.
                self.record_verdict(
                    request.miner_hotkey, request.merkle_root, False,
                    RejectReason.WORKER_DROPPED,
                    window_n=request.window_start,
                    telemetry=telemetry,
                    reject_stage="worker",
                    accepted_into_pool=False,
                )
                log_submission_stage(
                    logger,
                    logging.WARNING,
                    "candidate_rejected",
                    telemetry,
                    reject_stage="worker",
                    reject_reason=RejectReason.WORKER_DROPPED.value,
                    batch_filled_reason="batch_already_draining",
                    current_valid_count=getattr(batcher, "valid_count", None),
                    trigger_round=getattr(batcher, "_seal_trigger_round", None),
                    accepted_into_pool=False,
                )
                continue
            # Drain past-seal items without running GRAIL. The HTTP early-
            # cutoff catches submissions that arrive AFTER seal; this catches
            # the ones already in the queue from BEFORE seal that haven't
            # been dequeued yet. Together they cap per-window GRAIL work at
            # ~B_BATCH × verify-time instead of letting it grow with raw
            # arrival rate. Same accounting bucket as the HTTP path so a
            # miner inspecting late_drops sees one consistent metric.
            if batcher.is_sealed() and (
                not getattr(batcher, "difficulty_auction_enabled", False)
                or getattr(batcher, "seal_snapshot_started", False)
            ):
                self._cancel_proof_admission(batcher, request)
                self._cancel_logical_group_reservation(batcher, request)
                telemetry.refresh_from_batcher(batcher, at_decision=True)
                telemetry.mark_decision()
                logger.info(
                    "dropping post-seal queue item prompt=%d hotkey=%s "
                    "drand_round=%d (batcher window=%d already filled)",
                    request.prompt_idx, request.miner_hotkey[:12],
                    request.drand_round, batcher.window_start,
                )
                if self._late_drop_callback is not None:
                    self._late_drop_callback(
                        request.miner_hotkey, "batch_filled",
                    )
                self.record_verdict(
                    request.miner_hotkey, request.merkle_root, False,
                    RejectReason.BATCH_FILLED,
                    window_n=request.window_start,
                    telemetry=telemetry,
                    reject_stage="seal",
                    accepted_into_pool=False,
                )
                log_submission_stage(
                    logger,
                    logging.WARNING,
                    "candidate_rejected",
                    telemetry,
                    reject_stage="seal",
                    reject_reason=RejectReason.BATCH_FILLED.value,
                    batch_filled_reason="batch_already_sealed_or_draining",
                    current_valid_count=getattr(batcher, "valid_count", None),
                    trigger_round=getattr(batcher, "_seal_trigger_round", None),
                    accepted_into_pool=False,
                )
                continue
            started, start_reason = self._start_proof_admission(
                batcher, request,
            )
            if not started:
                self._cancel_logical_group_reservation(batcher, request)
                telemetry.refresh_from_batcher(batcher, at_decision=True)
                telemetry.mark_decision()
                if self._late_drop_callback is not None:
                    self._late_drop_callback(
                        request.miner_hotkey, "proof_admission_full",
                    )
                self.record_verdict(
                    request.miner_hotkey,
                    request.merkle_root,
                    False,
                    RejectReason.BATCH_FILLED,
                    window_n=request.window_start,
                    telemetry=telemetry,
                    reject_stage="proof_admission",
                    accepted_into_pool=False,
                )
                log_submission_stage(
                    logger,
                    logging.WARNING,
                    "candidate_rejected",
                    telemetry,
                    reject_stage="proof_admission",
                    reject_reason=RejectReason.BATCH_FILLED.value,
                    batch_filled_reason=start_reason,
                    accepted_into_pool=False,
                )
                continue
            try:
                telemetry.mark_proof_started()
                log_submission_stage(
                    logger,
                    logging.INFO,
                    "proof_started",
                    telemetry,
                    reject_stage=None,
                    reject_reason=None,
                )
                self._inflight_proofs += 1
                env_name = str(
                    getattr(getattr(batcher, "env", None), "name", "unknown")
                )
                self._inflight_proofs_by_environment[env_name] += 1
                try:
                    reward_computation = None
                    precompute_rewards = getattr(
                        type(batcher), "compute_submission_rewards", None
                    )
                    if precompute_rewards is not None:
                        telemetry.mark_reward_started()
                        try:
                            reward_computation = await asyncio.to_thread(
                                precompute_rewards,
                                batcher,
                                request,
                            )
                        finally:
                            telemetry.mark_reward_finished()
                        log_submission_stage(
                            logger,
                            logging.INFO,
                            "reward_graded",
                            telemetry,
                            reward_grading_ms=getattr(
                                reward_computation, "elapsed_ms", None
                            ),
                            reward_grading_error=(
                                type(reward_computation.error).__name__
                                if getattr(
                                    reward_computation, "error", None
                                ) is not None
                                else None
                            ),
                        )
                    telemetry.mark_admission_started()
                    try:
                        response = await asyncio.to_thread(
                            self._call_accept_submission,
                            batcher,
                            request,
                            telemetry,
                            reward_computation,
                        )
                    finally:
                        telemetry.mark_admission_finished()
                finally:
                    self._inflight_proofs = max(0, self._inflight_proofs - 1)
                    remaining = max(
                        0,
                        self._inflight_proofs_by_environment[env_name] - 1,
                    )
                    if remaining:
                        self._inflight_proofs_by_environment[env_name] = (
                            remaining
                        )
                    else:
                        self._inflight_proofs_by_environment.pop(env_name, None)
                telemetry.refresh_from_batcher(batcher, at_decision=True)
                telemetry.mark_decision(verified=True)
                self._record_admission_latency(batcher, telemetry)
                quota_refunded = False
                if (
                    not response.accepted
                    and response.reason is RejectReason.WORKER_DROPPED
                    and self._worker_drop_refunds_quota(request)
                ):
                    self._refund_submission_quota(request.miner_hotkey)
                    quota_refunded = True
                log_submission_stage(
                    logger,
                    logging.INFO,
                    "proof_finished",
                    telemetry,
                    accepted=response.accepted,
                    reason=response.reason.value,
                    quota_refunded=quota_refunded,
                )
                if response.accepted:
                    logger.info(
                        "accepted prompt=%d hotkey=%s drand_round=%d",
                        request.prompt_idx, request.miner_hotkey[:12],
                        request.drand_round,
                    )
                    log_submission_stage(
                        logger,
                        logging.INFO,
                        "candidate_accepted",
                        telemetry,
                        reject_stage="none",
                        reject_reason="none",
                        accepted_into_pool=True,
                    )
                else:
                    rewards = [r.reward for r in request.rollouts]
                    logger.warning(
                        "rejected prompt=%d hotkey=%s drand_round=%d "
                        "reason=%s rewards=%s",
                        request.prompt_idx, request.miner_hotkey[:12],
                        request.drand_round, response.reason.value, rewards,
                    )
                    log_submission_stage(
                        logger,
                        logging.WARNING,
                        "candidate_rejected",
                        telemetry,
                        reject_stage="proof",
                        reject_reason=response.reason.value,
                        accepted_into_pool=False,
                    )
                # Pool-admission verdict hidden by the provisional /submit
                # response. Auction selection publishes a second verdict at seal.
                self.record_verdict(
                    request.miner_hotkey, request.merkle_root,
                    response.accepted, response.reason,
                    window_n=request.window_start,
                    telemetry=telemetry,
                    reject_stage=None if response.accepted else "proof",
                    accepted_into_pool=response.accepted,
                )
            except Exception as e:
                logger.exception(
                    "submission worker failed on prompt %d", request.prompt_idx
                )
                # OOM-recovery: when CUDA allocator can't get a handle
                # (CUBLAS_STATUS_ALLOC_FAILED, out-of-memory etc.) we MUST
                # release the cached pool before the next submission lands,
                # otherwise every subsequent forward pass fails too. The
                # generic .empty_cache() call covers all the cuBLAS / cuDNN
                # / activation-pool fragmentation scenarios we've observed.
                msg = str(e).lower()
                if any(s in msg for s in ("out of memory", "cublas", "cuda")):
                    await asyncio.to_thread(_try_empty_cuda_cache)
            finally:
                self._finish_proof_admission(batcher, request)
                # Legacy admission runs GRAIL inline and must reclaim its
                # activation cache. Auction admission only grades rewards on
                # CPU/sandbox workers; emptying CUDA from several concurrent
                # workers would add a global allocator synchronization and can
                # interfere with training or seal-time deferred proof.
                if not getattr(batcher, "difficulty_auction_enabled", False):
                    await asyncio.to_thread(_try_empty_cuda_cache)

    @staticmethod
    def _latency_summary(
        samples: collections.abc.Iterable[float],
    ) -> dict[str, float | None]:
        ordered = sorted(float(value) for value in samples)
        if not ordered:
            return {"p50": None, "p95": None, "p99": None, "max": None}

        def percentile(fraction: float) -> float:
            index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
            return round(ordered[index], 3)

        return {
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": round(ordered[-1], 3),
        }

    def _record_admission_latency(
        self,
        batcher: Any,
        telemetry: SubmitTelemetry,
    ) -> None:
        env_name = str(
            getattr(getattr(batcher, "env", None), "name", "unknown")
        )
        metrics = self._admission_latency_samples_ms[env_name]
        for name in (
            "queue_wait_ms",
            "reward_grading_ms",
            "admission_prepare_ms",
            "commit_lock_wait_ms",
            "commit_ms",
            "total_ms",
        ):
            value = getattr(telemetry, name, None)
            if isinstance(value, numbers.Real) and not isinstance(value, bool):
                metrics[name].append(float(value))

    async def _monitor_event_loop_lag(self) -> None:
        interval = 0.25
        loop = asyncio.get_running_loop()
        expected = loop.time() + interval
        while True:
            try:
                await asyncio.sleep(max(0.0, expected - loop.time()))
            except asyncio.CancelledError:
                return
            now = loop.time()
            self._event_loop_lag_samples_ms.append(
                max(0.0, (now - expected) * 1000.0)
            )
            expected = now + interval

    async def start(self) -> None:
        if self._task is not None:
            return
        self._admission_materialization_pool = ThreadPoolExecutor(
            max_workers=MATH_ADMISSION_WORKERS + CODE_ADMISSION_WORKERS,
            thread_name_prefix="reliquary-problem-load",
        )
        self._auction_admission_enabled = True
        for env_name, batcher in self._active_batchers.items():
            if (
                getattr(batcher, "difficulty_auction_enabled", False)
                and env_name not in self._admission_process_pools
            ):
                self._admission_process_pools[env_name] = (
                    self._new_admission_pool(env_name)
                )
        config = uvicorn.Config(
            self.app, host=self.host, port=self.port,
            log_level="warning", access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())
        self._worker_task = asyncio.create_task(
            self._submit_worker(self._submit_queue),
            name="math_admission_worker_0",
        )
        self._code_worker_task = asyncio.create_task(
            self._submit_worker(self._code_submit_queue),
            name="code_admission_worker_0",
        )
        self._extra_worker_tasks = [
            *(
                asyncio.create_task(
                    self._submit_worker(self._submit_queue),
                    name=f"math_admission_worker_{index}",
                )
                for index in range(1, MATH_ADMISSION_WORKERS)
            ),
            *(
                asyncio.create_task(
                    self._submit_worker(self._code_submit_queue),
                    name=f"code_admission_worker_{index}",
                )
                for index in range(1, CODE_ADMISSION_WORKERS)
            ),
        ]
        self._event_loop_monitor_task = asyncio.create_task(
            self._monitor_event_loop_lag(), name="validator_event_loop_monitor"
        )
        await asyncio.sleep(0)
        logger.info(
            "Validator HTTP server listening on %s:%d "
            "(math_admission_workers=%d code_admission_workers=%d)",
            self.host,
            self.port,
            MATH_ADMISSION_WORKERS,
            CODE_ADMISSION_WORKERS,
        )

    async def stop(self) -> None:
        self._auction_admission_enabled = False
        event_loop_task = self._event_loop_monitor_task
        if event_loop_task is not None and not event_loop_task.done():
            event_loop_task.cancel()
            await asyncio.gather(event_loop_task, return_exceptions=True)
        self._event_loop_monitor_task = None
        worker_tasks = [
            task
            for task in (
                self._worker_task,
                self._code_worker_task,
                *self._extra_worker_tasks,
            )
            if task is not None
        ]
        for task in worker_tasks:
            task.cancel()
        if worker_tasks:
            await asyncio.gather(*worker_tasks, return_exceptions=True)
        self._worker_task = None
        self._code_worker_task = None
        self._extra_worker_tasks = []
        for pool in self._admission_process_pools.values():
            pool.shutdown(wait=False, cancel_futures=True)
        self._admission_process_pools = {}
        self._admission_tokenizer_hashes = {}
        materialization_pool = self._admission_materialization_pool
        if materialization_pool is not None:
            materialization_pool.shutdown(wait=False, cancel_futures=True)
        self._admission_materialization_pool = None
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except asyncio.TimeoutError:
                self._task.cancel()
            self._task = None
            self._server = None
