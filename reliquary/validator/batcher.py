"""GrpoWindowBatcher — orchestrator for the free-prompt GRPO market.

Holds a flat list of validated submissions per window + a reference to the
validator's shared ``CooldownMap``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import ValidationError

from reliquary.constants import (
    BATCH_PROMPT_COOLDOWN_WINDOWS,
    B_BATCH,
    BOOTSTRAP_MAX_TRUNCATED_PER_SUBMISSION,
    DIFFICULTY_AUCTION_DELTA,
    DIFFICULTY_AUCTION_ENFORCE,
    DIFFICULTY_AUCTION_ENVIRONMENTS,
    DIFFICULTY_AUCTION_SHADOW_ENABLED,
    DIFFICULTY_AUCTION_SHADOW_ENVIRONMENTS,
    DIFFICULTY_AUCTION_SHADOW_MAX_CANDIDATES,
    DIFFICULTY_AUCTION_SHADOW_MAX_SLOTS_PER_OPERATOR,
    MAX_EXPENSIVE_PROOF_FAILURES_PER_HOTKEY_PER_WINDOW,
    MAX_EXPENSIVE_PROOF_FAILURES_PER_OPERATOR_PER_WINDOW,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    MIN_EOS_PROBABILITY,
    FORENSIC_SAMPLE_PER_WINDOW,
    MAX_POST_TRIGGER_PROOF_CANDIDATES,
    MAX_PENDING_SUBMISSION_BYTES_PER_ENV,
    MAX_PENDING_SUBMISSION_BYTES_PER_HOTKEY,
    MAX_PENDING_UPLOAD_PRECOMMITS_PER_ENV,
    MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW,
    MAX_PROOF_WALL_SECONDS,
    MAX_SEAL_QUEUE_DRAIN_SECONDS,
    MAX_SUBMISSION_PAYLOAD_BYTES,
    MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW,
    MAX_SUBMISSIONS_PER_PROMPT,
    MAX_TRUNCATED_PER_SUBMISSION,
    PROMPT_RANGE_SIZE,
    PROMPT_RANGE_ENFORCE_FROM_WINDOW,
    REJECTED_LIST_CAP_PER_HOTKEY,
    WINDOW_COLLECTION_SECONDS,
    SUBMISSION_UPLOAD_GRACE_SECONDS,
    CODE_SEMANTIC_AUTH_ENFORCE,
    TOKEN_AUTH_ENFORCE,
    ALL_TOKEN_AUTH_ENFORCE,
    FORCED_SEED_CDF_BOUNDARY_EPSILON,
    FORCED_SEED_CDF_ENFORCE,
    FORCED_SEED_ENFORCE,
    FORCED_SEED_PROTOCOL_VERSION,
    LEGACY_MERKLE_ROOT_ENFORCE,
)
from reliquary.environment.base import Environment
from reliquary.environment.grader_client import GraderInfrastructureError
from reliquary.shared.prompt_range import window_prompt_range
from reliquary.protocol.legacy_merkle import legacy_submission_merkle_matches
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    BatchSubmissionResponse,
    CommitModel,
    GrpoBatchState,
    RejectReason,
    RolloutSubmission,
    WindowState,
)
from reliquary.protocol.tokens import verify_tokens
from reliquary.validator.batch_selection import (
    explain_batch_selection,
    select_batch_and_distribute,
)
from reliquary.validator.cooldown import CooldownMap
from reliquary.validator.dedup import (
    RolloutHashSet,
    compute_logical_group_hash,
    compute_rollout_hash,
)
from reliquary.validator.difficulty_auction import (
    ShadowSubmission,
    difficulty_score,
    select_shadow_auction,
)
from reliquary.validator.observability import (
    DrandRoundObservation,
    SubmitTelemetry,
    classify_drand_round,
    log_submission_stage,
)
from reliquary.validator.boxed_integrity import has_malformed_final_answer
from reliquary.validator.auth_forensics import (
    auth_forensics_context_chars,
    auth_forensics_enabled,
    auth_forensics_max_findings_per_rollout,
    code_semantic_counterfactual_enabled,
    code_semantic_counterfactual_max_findings_per_rollout,
    record_all_token_auth_findings,
    record_code_semantic_auth_findings,
    record_forced_seed_shadow,
    record_termination_shadow,
)
from reliquary.validator.reward_shape import detect_reward_shape_manipulation
from reliquary.validator.rollout_patterns import detect_opposite_reward_clones
from reliquary.validator.rollout_telemetry import (
    classify_bft_termination,
    sketch_commitment_metrics,
    token_degeneracy_metrics,
)
from reliquary.validator.selection_digest import compute_rollouts_selection_digest
from reliquary.validator.verifier import (
    evaluate_all_token_auth_shadow,
    evaluate_code_semantic_token_authenticity,
    evaluate_boxed_answer_probability,
    evaluate_token_authenticity,
    evaluate_token_distribution,
    has_eos_padding,
    is_in_zone,
    is_cap_truncation,
    is_natural_bft_cap_candidate,
    rewards_std,
    validate_force_span,
    verify_logprobs_claim,
    verify_termination,
)

logger = logging.getLogger(__name__)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _enrich_code_semantic_counterfactuals(
    *,
    metrics: dict[str, Any],
    env: Any,
    problem: dict,
    completion_text: str,
    rollout_reward: float,
    max_findings: int,
) -> None:
    """Fail-softly regrade argmax-token replacements for local forensics only."""
    if max_findings <= 0:
        return
    details = metrics.get("finding_details")
    if not isinstance(details, list):
        return

    checked = 0
    for detail in details:
        if not isinstance(detail, dict):
            continue
        if checked >= max_findings:
            return
        start = _int_or_none(detail.get("completion_char_start"))
        end = _int_or_none(detail.get("completion_char_end"))
        replacement = detail.get("argmax_text")
        if (
            start is None
            or end is None
            or not isinstance(replacement, str)
            or start < 0
            or end < start
            or end > len(completion_text)
        ):
            detail["counterfactual_error"] = "invalid_span"
            continue

        candidate = completion_text[:start] + replacement + completion_text[end:]
        if candidate == completion_text:
            detail["counterfactual_checked"] = False
            detail["counterfactual_error"] = "unchanged"
            continue

        checked += 1
        try:
            counterfactual_reward = float(env.compute_reward(problem, candidate))
        except Exception as exc:
            detail["counterfactual_checked"] = True
            detail["counterfactual_error"] = type(exc).__name__
            continue

        original_reward = float(rollout_reward)
        detail["counterfactual_checked"] = True
        detail["counterfactual_reward"] = counterfactual_reward
        detail["counterfactual_reward_delta"] = (
            counterfactual_reward - original_reward
        )
        detail["counterfactual_reward_flipped"] = (
            original_reward > 0.0 and counterfactual_reward <= 0.0
        )


def _uses_validator_authoritative_reward(env: Any) -> bool:
    return bool(getattr(env, "validator_authoritative_reward", False))


def _reward_matches_claim(actual: float, claimed: float, *, tolerance: float = 1e-6) -> bool:
    actual_f = float(actual)
    claimed_f = float(claimed)
    return (
        math.isfinite(actual_f)
        and math.isfinite(claimed_f)
        and abs(actual_f - claimed_f) <= tolerance
    )


def _forced_seed_verdict(n_stoch: int, n_match: int, enforce: bool) -> bool:
    """True => reject the group for seed mismatch. Abstains on thin signal;
    shadow (never rejects) when enforcement is off."""
    from reliquary.constants import (
        FORCED_SEED_CONSISTENCY_FLOOR, FORCED_SEED_MIN_STOCH_POSITIONS,
    )
    if not enforce:
        return False
    if n_stoch < FORCED_SEED_MIN_STOCH_POSITIONS:
        return False
    return (n_match / n_stoch) < FORCED_SEED_CONSISTENCY_FLOOR


def _forced_seed_rollout_reject(per_rollout, enforce: bool) -> bool:
    """True => reject because a SINGLE rollout is off the forced stream. The
    group-average verdict dilutes a partial swap (a few curated rollouts hidden
    among honest ones); this catches any one rollout that carries enough
    stochastic positions yet falls below the per-rollout floor. ``per_rollout``
    is a list of (n_stoch, n_match). Abstains on thin rollouts; shadow (never
    rejects) when enforcement is off."""
    from reliquary.constants import (
        FORCED_SEED_ROLLOUT_FLOOR, FORCED_SEED_ROLLOUT_MIN_STOCH,
    )
    if not enforce:
        return False
    for n_stoch, n_match in per_rollout:
        if (n_stoch >= FORCED_SEED_ROLLOUT_MIN_STOCH
                and (n_match / n_stoch) < FORCED_SEED_ROLLOUT_FLOOR):
            return True
    return False


def _is_missing_kwarg_typeerror(exc: TypeError, kwarg: str) -> bool:
    """True iff ``exc`` is Python's own "unexpected keyword argument" TypeError
    for ``kwarg`` (e.g. a legacy/stub verifier signature), as opposed to some
    other internal TypeError that merely happens to mention ``kwarg`` in its
    message. A bare substring test on ``kwarg`` alone would swallow the latter
    and silently disable the forced-seed gate every rollout."""
    msg = str(exc)
    return "unexpected keyword argument" in msg and kwarg in msg


_PROOF_FAILURE_DEBT_STAGES = frozenset(
    {
        "grail",
        "termination",
        "force_span",
        "logprob",
        "distribution",
        "boxed_answer",
        "token_authenticity",
        "all_token_authenticity",
        "code_semantic_auth",
        "forced_seed",
    }
)


# v2 difficulty auction: batch selection is drand-anchored at seal time (see
# ``batch_selection.py``). Multiple miners may submit on the same ``prompt_idx``
# within a window, capped at ``MAX_SUBMISSIONS_PER_PROMPT`` per prompt. The
# same-prompt winner is resolved at SEAL (the first submission for a prompt that
# PASSES the proof takes the slot), not at admission, so no wire-level
# PROMPT_CLAIMED reject is needed.


@dataclass
class PendingSubmission:
    """A submission that passed every CHEAP check and has been graded + scored,
    but has NOT been proven on the GPU yet.

    The proof (5-25 s of GPU) is the most expensive thing the validator does.
    Since the difficulty score depends only on the graded rewards, we rank first
    and prove only the candidates that can actually win (see ``_prove_ranked``).
    A submission that cannot reach the top B is never proven.

    Fabricated groups DO rank at the top (a miner who never runs the model can
    hand-write a k=2 reward vector). That is safe: the proof still runs before
    anyone is paid, so fabricating earns zero.
    """

    hotkey: str
    prompt_idx: int
    request: Any
    rewards: list[float]
    drand_round: int
    merkle_root: bytes
    selection_digest: bytes
    arrived_at: float = 0.0
    # Wall clock of the CHEAP admission decision. The pre-generation forensic
    # metric is arrival_ts - (decision_ts - response_time), so it must be the
    # instant the validator decided, not the instant it later proved.
    decision_ts: float | None = None
    telemetry: Any = None
    reject_response: BatchSubmissionResponse | None = None
    value: float = field(init=False, default=0.0)

    def __post_init__(self):
        self.value = difficulty_score(
            self.rewards, delta=DIFFICULTY_AUCTION_DELTA
        ).value


@dataclass
class ValidSubmission:
    """A submission that passed all v2 verification checks."""

    hotkey: str
    prompt_idx: int
    merkle_root_bytes: bytes
    merkle_root: bytes = field(init=False)  # alias for select_batch Protocol
    selection_digest_bytes: bytes | None = None
    selection_digest: bytes = field(init=False)
    sigma: float = 0.0
    rollouts: list[RolloutSubmission] = field(default_factory=list)
    completion_texts: list[str] = field(default_factory=list)
    arrived_at: float = 0.0
    # Filter telemetry (worst-case across this submission's rollouts).
    # Captured for post-hoc threshold calibration without re-running tests.
    sketch_diff_max: int | None = None
    lp_dev_max: float | None = None
    dist_q10_min: float | None = None
    all_token_auth_shadow_findings: int = 0
    all_token_auth_shadow_min_prob: float | None = None
    all_token_auth_shadow_positive_findings: int = 0
    all_token_auth_shadow_positive_min_prob: float | None = None
    code_semantic_auth_findings: int = 0
    code_semantic_auth_min_prob: float | None = None
    code_semantic_auth_positive_findings: int = 0
    code_semantic_auth_positive_min_prob: float | None = None
    # Miner-claimed checkpoint hash at submit time — useful for post-hoc
    # forensic analysis of who lied about their checkpoint.
    claimed_checkpoint_hash: str = ""
    rollout_hashes: list[bytes] = field(default_factory=list)
    # v2.3: drand round attached by the miner at submit time. Determines
    # the submission's chronological position at seal time.
    drand_round: int = 0
    arrival_ts: float | None = None
    decision_ts: float | None = None
    submitted_drand_round: int = 0
    arrival_drand_round: int | None = None
    drand_delta: int | None = None
    seal_trigger_round: int | None = None
    prompt_hash_lead: str | None = None
    reward_vector: str = ""
    truncated_count: int = 0
    reward_shape: dict[str, Any] = field(default_factory=dict)
    ingress_observability: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.merkle_root = self.merkle_root_bytes
        self.selection_digest = (
            self.selection_digest_bytes
            if self.selection_digest_bytes is not None
            else self.merkle_root_bytes
        )


@dataclass
class RejectedSubmission:
    """A submission that did NOT pass verification.

    Persisted to the R2 archive (subject to per-hotkey cap) so rejected
    miners can self-diagnose. Diagnostics are best-effort: only fields
    computed before the rejection point are populated.

    Anti-tuning: ``sketch_diff_max`` is intentionally LEFT NONE for
    ``GRAIL_FAIL`` rejections. Surfacing the exact diff would let a cheater
    calibrate against ``PROOF_SKETCH_TOLERANCE_BASE``. Other reject reasons
    are not threshold-tunable, so their diagnostics are surfaced verbatim.
    """

    hotkey: str
    prompt_idx: int
    reason: str  # RejectReason.value
    sketch_diff_max: int | None = None
    lp_dev_max: float | None = None
    dist_q10_min: float | None = None
    arrival_ts: float | None = None
    decision_ts: float | None = None
    submitted_drand_round: int | None = None
    arrival_drand_round: int | None = None
    drand_delta: int | None = None
    seal_trigger_round: int | None = None
    prompt_hash_lead: str | None = None
    reject_stage: str | None = None
    ingress_observability: dict[str, Any] | None = None


@dataclass
class ForensicSampleResult:
    """A non-winner proven purely for telemetry (see ``FORENSIC_SAMPLE_PER_WINDOW``).

    The proof runs for its side effects — the authenticity/forensic gates in
    ``_verify_expensive`` fire and any reject lands in ``rejected_submissions``
    as usual — but ``passed`` submissions are still discarded: never added to
    ``_valid``, never paid.
    """

    hotkey: str
    prompt_idx: int
    passed: bool | None
    error_type: str | None = None


@dataclass(frozen=True)
class RewardComputation:
    """Authoritative reward work completed outside the batcher mutation lock."""

    validator_scored_reward: bool
    completion_texts: list[str]
    rewards: list[float] | None
    error: Exception | None
    elapsed_ms: float


class GrpoWindowBatcher:
    """Accepts v2 submissions, runs the full verification pipeline, and
    exposes ``valid_submissions()`` + ``select_batch()`` at window close.
    """

    def __init__(
        self,
        window_start: int,
        env: Environment,
        model: Any,
        *,
        tokenizer: Any = None,
        cooldown_map: CooldownMap | None = None,
        hash_set: RolloutHashSet | None = None,
        bootstrap: bool = False,
        completion_text_fn: Callable[[RolloutSubmission], str],
        canonical_prompt_tokens_fn: Callable[[int], list[int]] | None = None,
        verify_commitment_proofs_fn: Callable[..., Any] | None = None,
        verify_signature_fn: Callable[[dict, str], bool] | None = None,
        time_fn: Callable[[], float] | None = None,
        wall_clock_fn: Callable[[], float] | None = None,
        drand_round_check_enabled: bool = True,
        drand_chain_info: dict | None = None,
        drand_round_backward_tolerance: int | None = None,
        queue_drained_predicate: Callable[[], bool] | None = None,
        operator_by_hotkey: dict[str, str] | None = None,
    ) -> None:
        from reliquary.constants import DRAND_ROUND_BACKWARD_TOLERANCE

        self.window_start = window_start
        self.env = env
        self.difficulty_auction_enabled = bool(
            DIFFICULTY_AUCTION_ENFORCE
            and getattr(env, "name", "") in DIFFICULTY_AUCTION_ENVIRONMENTS
        )
        self.model = model
        self.tokenizer = tokenizer
        self.bootstrap = bootstrap
        # Set True by the validator's background drand-verify task
        # if the cross-check against bittensor_drand fails post-OPEN.
        # ``_train_and_publish`` checks this before sealing and drops
        # the window if set — preserves the verify gate without
        # blocking the hot OPEN path.
        self.beacon_invalid: bool = False
        self._completion_text = completion_text_fn
        # Wall clock (UNIX seconds) used to compute the current drand round
        # at submit-receipt time. Distinct from ``_time_fn`` (monotonic)
        # which is used for response_time bookkeeping.
        self._wall_clock = wall_clock_fn or time.time
        self.drand_round_check_enabled = drand_round_check_enabled
        # How many drand rounds backward of the validator's current round
        # the batcher accepts. Defaults to the operator-configured
        # ``DRAND_ROUND_BACKWARD_TOLERANCE`` (zero in production); tests may
        # explicitly widen or pin the tolerance for the case under test.
        self.drand_round_backward_tolerance = (
            drand_round_backward_tolerance
            if drand_round_backward_tolerance is not None
            else DRAND_ROUND_BACKWARD_TOLERANCE
        )
        # Lazy: fetched on first use if not injected (tests inject a fixed
        # {"genesis_time", "period"} dict to avoid live HTTP calls).
        self._drand_chain_info = drand_chain_info
        # Returns the canonical prompt tokens for a given prompt_idx — used to
        # bind the miner's claimed prompt_idx to the actual tokens they ran the
        # forward pass on. ``None`` disables the binding check (test convenience
        # for stubs that don't carry a tokenizer).
        self._canonical_prompt_tokens = canonical_prompt_tokens_fn
        self._time_fn = time_fn or time.monotonic
        # Reference for per-submission response_time. Set at construction so
        # ``arrived_at - window_opened_at`` is the seconds the miner took
        # from window-open to accepted submission.
        self.window_opened_at: float = self._time_fn()
        self.window_opened_wall_ts: float = self._wall_clock()
        self.window_open_drand_round: int | None = None
        self.last_valid_submission_at: float | None = None
        self.last_valid_submission_wall_ts: float | None = None

        self._cooldown = (
            cooldown_map if cooldown_map is not None
            else CooldownMap(cooldown_windows=BATCH_PROMPT_COOLDOWN_WINDOWS)
        )
        self._hash_set: RolloutHashSet | None = hash_set
        self._operator_by_hotkey = {
            normalized_hotkey: normalized_operator
            for hotkey, operator in (operator_by_hotkey or {}).items()
            if (normalized_hotkey := str(hotkey).strip())
            and (normalized_operator := str(operator).strip())
        }
        # ``None`` is reserved for direct/test embedding where no metagraph
        # exists. Production always passes a snapshot (possibly incomplete),
        # and therefore fails closed candidate-by-candidate on missing entries.
        self._operator_mapping_enforced = operator_by_hotkey is not None

        # Lock-free snapshot read by the HTTP /state handler. The submit
        # worker holds ``_lock`` for the entire GRAIL verify (~5-25s); a
        # /state caller acquiring the same lock synchronously on the asyncio
        # event loop starved the loop and triggered cascading 60s timeouts
        # on miners polling /state. The cooldown set for a given window is
        # stable during the batcher's lifetime — ``_cooldown`` is only
        # mutated by ``seal_batch`` at the very end — so a snapshot taken
        # here is correct for /state's entire lifetime.
        self.cooldown_prompts_snapshot: list[int] = sorted(
            self._cooldown.current_cooldown_set(window_start)
        )
        # Atomic counter for /state's ``valid_submissions`` field. Updated
        # under ``_lock`` after each successful accept; the read in /state
        # is lock-free (int reads are GIL-atomic in CPython).
        self.valid_count: int = 0
        # Same lock-free contract, for admitted-but-unproven submissions. This
        # is what fills during the window now: ``valid_count`` only moves at
        # seal, once the ranked candidates have been proven.
        self.pending_count: int = 0
        # GPU proofs spent by ``_prove_ranked`` this window; telemetry reads it
        # after seal. Bounded by the graded pool ceiling
        # (MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW) and the per-hotkey failure cap.
        self.proof_attempts: int = 0

        if verify_commitment_proofs_fn is None:
            from reliquary.validator.verifier import verify_commitment_proofs
            verify_commitment_proofs_fn = verify_commitment_proofs
        if verify_signature_fn is None:
            from reliquary.validator.verifier import verify_signature
            verify_signature_fn = verify_signature

        self._verify_commitment = verify_commitment_proofs_fn
        self._verify_signature = verify_signature_fn

        self._lock = threading.Lock()
        # Validator-owned economic identity reservations. This lock is
        # separate from ``_lock`` because HTTP admission must atomically claim
        # a group before it enters the potentially long GPU queue.
        self._logical_group_lock = threading.Lock()
        self._logical_group_reservations: dict[
            tuple[str, str, int | bytes], BatchSubmissionRequest
        ] = {}
        self._logical_group_duplicate_rejects = 0
        self.grader_failures: dict[str, int] = {}
        self._valid: list[ValidSubmission] = []
        # Admitted, graded and scored, but NOT yet proven. The auction ranks
        # these at seal; only the candidates that can actually win pay for a GPU
        # proof (see ``_verify_expensive``). ``_valid`` stays empty all window.
        self._pending: list[PendingSubmission] = []
        # ids of pending submissions ``_prove_ranked`` already attempted this
        # window (winner or failure). ``_prove_forensic_sample`` reads this to
        # sample only from what was never looked at.
        self._attempted_pending_ids: set[int] = set()
        # Non-winners proven purely for telemetry — see FORENSIC_SAMPLE_PER_WINDOW.
        # Never touches ``_valid``; a passing entry here is still unpaid.
        self.forensic_sample: list[ForensicSampleResult] = []
        # v2: per-prompt bucket of PENDING submissions. Multiple miners may
        # submit on the same ``prompt_idx`` up to ``MAX_SUBMISSIONS_PER_PROMPT``.
        # Tracked alongside the flat ``_pending`` list because seal_batch needs
        # the grouping but accept-time logic only needs the count.
        self._submissions_per_prompt: dict[
            int, list[PendingSubmission | ValidSubmission]
        ] = {}
        self.randomness: str = ""
        # Drand beacon fetched at seal (post-deadline) to key the forensic
        # sample so it cannot be predicted at submission time. See
        # _prove_forensic_sample. Empty in mock/no-drand mode = no sampling.
        self.seal_randomness: str = ""
        # Per-window eligible prompt slice [lo, hi). None = no restriction
        # (randomness not yet known, or window is before the enforcement
        # cutover). Set by set_prompt_range() once randomness is assigned.
        self.prompt_range: tuple[int, int] | None = None
        # Authoritative post-seal emission distribution consumed by archive
        # replay and weight-only validators. Auction mode pays one uniform slot
        # to each proven winner; legacy mode may retain historical split rules.
        self.rewards_by_hotkey: dict[str, float] = {}
        # Legacy compatibility metric. Production auction winners are both
        # selected and rewarded, so this remains empty in auction mode.
        self.rewarded_but_not_selected_by_hotkey: dict[str, int] = {}
        # Accumulated reject reasons this window (RejectReason.value → count).
        # Persisted in the R2 archive so miners can see which filter is
        # rejecting the most submissions in any given round.
        self.reject_counts: dict[str, int] = {}
        self.selection_metadata_by_id: dict[int, dict[str, Any]] = {}
        # Historical attribute name retained for archive/dashboard compatibility.
        # Auction mode stores the armed production payload here; kill-switch
        # legacy mode stores the observation-only counterfactual.
        self.difficulty_auction_shadow: dict[str, Any] = {
            "schema_version": 1,
            "status": "not_computed",
            "mode": "observation_only",
        }
        self.difficulty_auction_metadata_by_id: dict[int, dict[str, Any]] = {}

        # Per-hotkey-capped metadata for rejected submissions. Persisted in
        # the R2 archive next to ``reject_counts`` so a rejected miner can
        # see *which* of their submissions failed and why, instead of just
        # an aggregate count. Cap protects against single-attacker flooding.
        self.rejected_submissions: list[RejectedSubmission] = []

        # v2.1+: seal_event fires once the window is finalized. v2.3+:
        # firing is delayed past the B-th distinct prompt to absorb the
        # rest of that submission's drand round — see the "trigger round"
        # comment in ``_accept_locked``. Stored as ``threading.Event`` for
        # sync-safe ``set()`` from the worker thread; the asyncio.Event is
        # lazy-bound to the running loop on first access.
        self._seal_flag: threading.Event = threading.Event()
        self._seal_event: asyncio.Event | None = None
        # v2.3 seal extension. When the B-th distinct prompt arrives in
        # drand round R, we record R here and DELAY firing ``_seal_flag``
        # until the next drand boundary so further submissions in R can
        # still be accepted (and fairly share the boundary-round emission
        # via ``select_batch_and_distribute``'s boundary branch). Until
        # the boundary passes:
        #   * submissions with ``drand_round == R`` are accepted normally
        #   * submissions with ``drand_round > R`` are rejected with
        #     ``BATCH_FILLED`` AND fire the seal immediately (no point
        #     waiting if a later round has already shown up)
        # ``_loop`` is captured the first time ``seal_event`` is accessed
        # from an async context so we can schedule the delayed seal from
        # the worker thread via ``run_coroutine_threadsafe``. ``None`` in
        # synchronous test contexts — there the seal fires immediately on
        # the B-th distinct, matching the pre-v2.3 timing.
        self._seal_trigger_round: int | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # Optional callback the seal-extension coroutine polls to check
        # whether the server's submit_queue has finished draining items
        # queued during the trigger drand round. ``None`` in test
        # contexts that bypass the HTTP/worker pipeline — in that case
        # the delayed seal coroutine fires as soon as the drand round
        # expires (drain phase skipped). Production wires this to
        # ``_queue_and_proofs_drained`` so already admitted trigger-round
        # submissions finish GRAIL (queue empty AND no proof in flight)
        # before the batch is sealed.
        self._queue_drained_predicate = queue_drained_predicate
        self.force_seal_reason: str | None = None
        # Proof-admission accounting is separate from ``_lock`` because the
        # submit worker holds ``_lock`` during GRAIL. The HTTP cheap path must
        # be able to reject over-budget submissions without waiting behind the
        # very GPU work it is trying to bound. Counts are reservations for
        # expensive verification attempts, not successful accepts.
        self._proof_admission_lock = threading.Lock()
        self._proof_admission_count = 0
        # Total grading attempts that actually started this window. Pending
        # queue reservations are tracked separately so a request discarded on
        # seal/window swap does not permanently consume work that never ran.
        # Started attempts are never refunded.
        self._proof_grading_attempts = 0
        self._pending_proof_reservations: dict[
            int, tuple[BatchSubmissionRequest, str, bool, int]
        ] = {}
        self._inflight_proof_reservations: dict[
            int, tuple[BatchSubmissionRequest, str, bool, int]
        ] = {}
        self._retained_payload_reservations: dict[
            int, tuple[BatchSubmissionRequest, str, int]
        ] = {}
        self._pending_payload_bytes = 0
        self._inflight_payload_bytes = 0
        self._retained_payload_bytes = 0
        self._payload_bytes_by_hotkey: dict[str, int] = {}
        self._pending_post_trigger_proof_reservations = 0
        self._post_trigger_proof_admission_count = 0
        self._expensive_proof_failures_by_hotkey: dict[str, int] = {}
        self._expensive_proof_failures_by_operator: dict[str, int] = {}
        self.proof_wall_elapsed_seconds = 0.0
        self.proof_wall_exhausted = False
        self.forensic_proof_attempts = 0
        self.forensic_proof_errors_by_type: dict[str, int] = {}
        self.auction_operator_unmapped_skips = 0
        self.auction_operator_proof_debt_skips = 0
        self.auction_candidates: list[dict[str, Any]] = []
        self._proof_wall_started_at: float | None = None
        self._seal_snapshot_started = False
        self._seal_completed = False
        self._upload_precommit_lock = threading.Lock()
        self._upload_precommits: dict[str, tuple[str, float]] = {}
        # v2.1: checkpoint hash miners must match. Empty string disables
        # the gate (test convenience / pre-first-publish).
        self.current_checkpoint_hash: str = ""

    def set_prompt_range(self) -> None:
        """Compute and cache this window's eligible prompt slice.

        Leaves ``prompt_range`` None (accept any prompt_idx, current behavior)
        until randomness is known AND ``window_start`` has reached
        ``PROMPT_RANGE_ENFORCE_FROM_WINDOW``. Call after assigning randomness.
        """
        if (
            not self.randomness
            or self.window_start < PROMPT_RANGE_ENFORCE_FROM_WINDOW
        ):
            self.prompt_range = None
            return
        self.prompt_range = window_prompt_range(
            self.randomness,
            getattr(self.env, "name", ""),
            len(self.env),
            PROMPT_RANGE_SIZE,
        )

    @property
    def seal_event(self) -> asyncio.Event:
        """Lazy asyncio.Event bound to whichever loop accesses it first.

        Also captures the loop reference for the v2.3 seal-extension
        mechanism, which needs to schedule a delayed seal-firing coroutine
        from the worker thread.
        """
        if self._seal_event is None:
            self._seal_event = asyncio.Event()
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = None
            if self._seal_flag.is_set():
                self._seal_event.set()
        return self._seal_event

    def bind_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the validator's main event loop for the delayed seal.

        ``accept_submission`` runs in a worker thread (``asyncio.to_thread``)
        with no running loop, so it cannot capture the loop itself — it reads
        this pre-bound reference to schedule ``_delayed_seal_at_drand_boundary``
        via ``run_coroutine_threadsafe``. Without this, ``_loop`` stays ``None``
        and the B-th distinct prompt seals synchronously, dropping same-round
        in-flight submissions and collapsing the drand-boundary fair split.
        """
        self._loop = loop

    def mark_window_opened(self) -> None:
        """Anchor collection and response-time telemetry at actual activation.

        Batchers are constructed before drand preparation, then exposed to
        miners in a separate activation phase. Starting the deadline in the
        constructor would silently shorten a 300-second auction whenever
        preparation is slow.
        """
        self.window_opened_at = self._time_fn()
        self.window_opened_wall_ts = self._wall_clock()

    def is_sealed(self) -> bool:
        """True once the collection deadline has expired (or a safety-valve
        ``force_seal`` fired). Thread-safe and loop-independent (reads the
        underlying ``threading.Event``, never touches the lazy
        ``asyncio.Event``).

        The window stays open the full ``WINDOW_COLLECTION_SECONDS`` and accepts
        everything; only after this returns True does further work for the
        window short-circuit. Callers (the HTTP /submit handler and the submit
        worker) use it to drop post-deadline submissions.
        """
        return self._seal_flag.is_set()

    def collection_closed(self) -> bool:
        """Whether the generation/commit phase has reached its fixed cutoff."""
        return self._time_fn() - self.window_opened_at >= WINDOW_COLLECTION_SECONDS

    def _prune_upload_precommits_locked(self, now: float) -> None:
        expired = [
            receipt_id
            for receipt_id, (_hotkey, deadline) in self._upload_precommits.items()
            if deadline <= now
        ]
        for receipt_id in expired:
            self._upload_precommits.pop(receipt_id, None)

    def try_register_upload_precommit(
        self,
        receipt_id: str,
        hotkey: str,
        *,
        t_arrival_wall: float,
    ) -> tuple[bool, str | None, float | None]:
        """Reserve one bounded reveal received before collection closes.

        Returns ``(accepted, reason, monotonic_deadline)``.  This reservation is
        deliberately separate from economic operator/prompt claims: a miner that
        never uploads cannot squat an auction prompt.
        """
        if not self.difficulty_auction_enabled:
            return False, "precommit_requires_auction", None
        if (
            float(t_arrival_wall)
            > self.window_opened_wall_ts + WINDOW_COLLECTION_SECONDS
        ):
            return False, "collection_closed", None
        now = self._time_fn()
        with self._upload_precommit_lock:
            self._prune_upload_precommits_locked(now)
            if self._seal_flag.is_set() or self._seal_snapshot_started:
                return False, "collection_sealed", None
            if len(self._upload_precommits) >= MAX_PENDING_UPLOAD_PRECOMMITS_PER_ENV:
                return False, "precommit_capacity_full", None
            hotkey_count = sum(
                1 for existing_hotkey, _deadline in self._upload_precommits.values()
                if existing_hotkey == hotkey
            )
            if hotkey_count >= MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW:
                return False, "precommit_hotkey_full", None
            deadline = min(
                now + SUBMISSION_UPLOAD_GRACE_SECONDS,
                self.window_opened_at
                + WINDOW_COLLECTION_SECONDS
                + SUBMISSION_UPLOAD_GRACE_SECONDS,
            )
            self._upload_precommits[receipt_id] = (hotkey, deadline)
            return True, None, deadline

    def resolve_upload_precommit(self, receipt_id: str) -> bool:
        """Release a receipt once its matching body has completed admission."""
        with self._upload_precommit_lock:
            return self._upload_precommits.pop(receipt_id, None) is not None

    @property
    def pending_upload_precommits(self) -> int:
        now = self._time_fn()
        with self._upload_precommit_lock:
            self._prune_upload_precommits_locked(now)
            return len(self._upload_precommits)

    def poll_deadline(self) -> bool:
        """Seal an auction environment at its fixed collection deadline.

        Non-auction environments keep the legacy B-distinct/drand-boundary seal
        and therefore treat this poll as a no-op.
        """
        if self._seal_flag.is_set():
            return True
        if not self.difficulty_auction_enabled:
            return False
        now = self._time_fn()
        if now - self.window_opened_at >= WINDOW_COLLECTION_SECONDS:
            with self._upload_precommit_lock:
                self._prune_upload_precommits_locked(now)
                pending_uploads = bool(self._upload_precommits)
            if pending_uploads and (
                now - self.window_opened_at
                < WINDOW_COLLECTION_SECONDS + SUBMISSION_UPLOAD_GRACE_SECONDS
            ):
                return False
            self._seal_flag.set()
            if self._seal_event is not None:
                self._seal_event.set()
            return True
        return False

    def force_seal(self, reason: str) -> None:
        """Seal this window early — a safety valve, not the normal path.

        The window normally seals on the collection deadline (see
        ``poll_deadline``). This exists for out-of-band drops such as an
        invalidated beacon or a sparse-window liveness breaker; the downstream
        training path already skips partial batches.
        """
        if self._seal_flag.is_set():
            return
        self.force_seal_reason = reason
        self._seal_flag.set()
        if self._seal_event is not None:
            self._seal_event.set()

    def prompt_submission_count(self, prompt_idx: int) -> int:
        """Number of retained submissions in the per-prompt bucket.

        Auction mode counts graded pending candidates; legacy mode counts proven
        valid candidates. Used by HTTP to short-circuit ``PROMPT_FULL``.

        Read of a dict-of-list len is GIL-atomic and lock-free in CPython,
        same property ``valid_count`` relies on. Best-effort: a racing
        accept inside the worker between this read and the queue.put is
        harmless — the worker re-checks the cap inside ``_accept_locked``.
        """
        return len(self._submissions_per_prompt.get(prompt_idx, ()))

    def distinct_valid_prompt_count(self) -> int:
        """Number of distinct, non-cooldown prompts among valid submissions.

        This is the trainable fill level for the window. It can be lower than
        ``valid_count`` when multiple miners submit the same prompt, so seal
        liveness must reason about this value instead of raw submissions.
        """
        return len({
            s.prompt_idx for s in list(self._valid)
            if not self._cooldown.is_in_cooldown(s.prompt_idx, self.window_start)
        })

    def distinct_pending_prompt_count(self) -> int:
        """Distinct non-cooldown prompts among graded (unproven) submissions.

        This — not ``distinct_valid_prompt_count`` — is the window's fill level
        now that proofs run at seal: ``_valid`` stays empty until then, so a
        liveness breaker reading it would never fire.
        """
        return len({
            p.prompt_idx for p in list(self._pending)
            if not self._cooldown.is_in_cooldown(p.prompt_idx, self.window_start)
        })

    def seconds_since_last_valid_submission(self) -> float | None:
        """Seconds since the last accepted valid submission, or None."""
        if self.last_valid_submission_at is None:
            return None
        return max(0.0, self._time_fn() - self.last_valid_submission_at)

    @property
    def proof_admission_count(self) -> int:
        """Live GRAIL candidate budget usage (refunded on OUT_OF_ZONE)."""
        return self._proof_admission_count

    @property
    def proof_grading_attempts(self) -> int:
        """Total grading attempts started this window (never refunded)."""
        return self._proof_grading_attempts

    @property
    def pending_proof_reservations(self) -> int:
        return len(self._pending_proof_reservations)

    @property
    def inflight_proof_reservations(self) -> int:
        return len(self._inflight_proof_reservations)

    @property
    def pending_payload_bytes(self) -> int:
        with self._proof_admission_lock:
            return self._pending_payload_bytes

    @property
    def inflight_payload_bytes(self) -> int:
        with self._proof_admission_lock:
            return self._inflight_payload_bytes

    @property
    def retained_payload_bytes(self) -> int:
        with self._proof_admission_lock:
            return self._retained_payload_bytes

    @property
    def reserved_payload_bytes(self) -> int:
        with self._proof_admission_lock:
            return (
                self._pending_payload_bytes
                + self._inflight_payload_bytes
                + self._retained_payload_bytes
            )

    @property
    def payload_bytes_by_hotkey(self) -> dict[str, int]:
        with self._proof_admission_lock:
            return dict(self._payload_bytes_by_hotkey)

    @property
    def seal_snapshot_started(self) -> bool:
        with self._proof_admission_lock:
            return self._seal_snapshot_started

    def begin_seal_snapshot(self) -> None:
        """Freeze proof admission before the auction reads its pending pool.

        The service first closes HTTP admission at the collection deadline and
        gives already-queued requests a bounded drain period. This marker then
        prevents a dequeued straggler from entering ``_accept_locked`` after
        ``_prove_ranked`` has copied the pending population.
        """
        with self._proof_admission_lock:
            self._seal_snapshot_started = True

    @property
    def proof_grading_capacity_used(self) -> int:
        """Started attempts plus pending reservations used for admission."""
        return self._proof_grading_attempts + len(
            self._pending_proof_reservations
        )

    @property
    def post_trigger_proof_admission_count(self) -> int:
        """Started plus pending proof work from the seal-trigger round."""
        return (
            self._post_trigger_proof_admission_count
            + self._pending_post_trigger_proof_reservations
        )

    @property
    def expensive_proof_failures_by_hotkey(self) -> dict[str, int]:
        """Per-window count of hotkey failures after entering proof path."""
        return dict(self._expensive_proof_failures_by_hotkey)

    def proof_failure_debt(self, hotkey: str) -> int:
        return self._expensive_proof_failures_by_hotkey.get(hotkey, 0)

    @property
    def expensive_proof_failures_by_operator(self) -> dict[str, int]:
        return dict(self._expensive_proof_failures_by_operator)

    def operator_proof_failure_debt(self, operator: str) -> int:
        return self._expensive_proof_failures_by_operator.get(operator, 0)

    @property
    def logical_group_reservation_count(self) -> int:
        with self._logical_group_lock:
            return len(self._logical_group_reservations)

    @property
    def logical_group_duplicate_rejects(self) -> int:
        with self._logical_group_lock:
            return self._logical_group_duplicate_rejects

    def try_reserve_logical_group(
        self,
        request: BatchSubmissionRequest,
    ) -> tuple[bool, str | None]:
        """Atomically reserve one economic claim for this window.

        In auction mode the claim is scoped to ``(operator, prompt)``. The
        forced-seed stream is hotkey-free, so allowing several hotkeys owned by
        one operator to reserve the same prompt would turn identical draws into
        extra tie-break tickets. Binding the reservation to chain ownership also
        covers small, tolerated runtime divergences that produce different token
        hashes. Legacy mode keeps its narrower per-hotkey logical-content scope.
        """
        if self.difficulty_auction_enabled:
            operator = self._operator_by_hotkey.get(request.miner_hotkey)
            if operator is None:
                if self._operator_mapping_enforced:
                    return False, "operator_unmapped"
                operator = request.miner_hotkey
            key: tuple[str, str, int | bytes] = (
                "auction_operator_prompt",
                operator,
                int(request.prompt_idx),
            )
        else:
            digest = compute_logical_group_hash(request)
            key = ("logical_group", request.miner_hotkey, digest)
        with self._logical_group_lock:
            owner = self._logical_group_reservations.get(key)
            if owner is request and request._logical_group_reservation == key:
                return True, None
            if owner is not None:
                self._logical_group_duplicate_rejects += 1
                return False, "logical_group_duplicate"
            self._logical_group_reservations[key] = request
            request._logical_group_reservation = key
            return True, None

    def cancel_logical_group_reservation(
        self,
        request: BatchSubmissionRequest,
    ) -> bool:
        """Release a reservation only while its owning request has not run."""
        key = request._logical_group_reservation
        if key is None:
            return False
        with self._logical_group_lock:
            if self._logical_group_reservations.get(key) is not request:
                return False
            self._logical_group_reservations.pop(key, None)
            request._logical_group_reservation = None
            return True

    def confirm_logical_group_reservation(
        self,
        request: BatchSubmissionRequest,
    ) -> None:
        """Make a reservation permanent for the rest of this window."""
        request._logical_group_reservation = None

    @staticmethod
    def _submission_payload_bytes(request: BatchSubmissionRequest) -> int:
        size = int(getattr(request, "_payload_bytes", 0) or 0)
        if size <= 0:
            size = len(request.model_dump_json().encode("utf-8"))
            request._payload_bytes = size
        return size

    def _reserved_payload_bytes_locked(self) -> int:
        return (
            self._pending_payload_bytes
            + self._inflight_payload_bytes
            + self._retained_payload_bytes
        )

    def _payload_capacity_reason_locked(
        self,
        hotkey: str,
        payload_bytes: int,
    ) -> str | None:
        if payload_bytes > MAX_SUBMISSION_PAYLOAD_BYTES:
            return "submission_payload_too_large"
        if (
            self._reserved_payload_bytes_locked() + payload_bytes
            > MAX_PENDING_SUBMISSION_BYTES_PER_ENV
        ):
            return "pending_payload_bytes_env_full"
        if (
            self._payload_bytes_by_hotkey.get(hotkey, 0) + payload_bytes
            > MAX_PENDING_SUBMISSION_BYTES_PER_HOTKEY
        ):
            return "pending_payload_bytes_hotkey_full"
        return None

    def _release_hotkey_payload_locked(
        self,
        hotkey: str,
        payload_bytes: int,
    ) -> None:
        remaining = max(
            0,
            self._payload_bytes_by_hotkey.get(hotkey, 0) - payload_bytes,
        )
        if remaining:
            self._payload_bytes_by_hotkey[hotkey] = remaining
        else:
            self._payload_bytes_by_hotkey.pop(hotkey, None)

    def try_reserve_proof_admission(
        self,
        request: BatchSubmissionRequest,
    ) -> tuple[bool, str | None]:
        """Reserve a *grading* slot for this window (the anti-DoS bound).

        Admission is gated by started work plus pending reservations, the
        post-trigger straggler cap and per-hotkey proof-failure debt. A pending
        reservation can be cancelled if the queue item never starts; once
        started, its grading attempt is never refunded.
        There is no separate GRAIL/GPU candidate budget: the drand-anchored
        seal, this grading ceiling and the seal drain timeout already bound the
        GPU work per window, so a zone-valid submission entering the proof is
        only counted for telemetry, never rejected on a candidate budget.
        """
        payload_bytes = self._submission_payload_bytes(request)
        request._retain_payload = False
        with self._proof_admission_lock:
            reservation_id = id(request)
            if (
                reservation_id in self._pending_proof_reservations
                or reservation_id in self._inflight_proof_reservations
                or reservation_id in self._retained_payload_reservations
            ):
                return False, "proof_reservation_duplicate"

            if (
                self._expensive_proof_failures_by_hotkey.get(
                    request.miner_hotkey, 0
                )
                >= MAX_EXPENSIVE_PROOF_FAILURES_PER_HOTKEY_PER_WINDOW
            ):
                return False, "proof_failure_debt_hotkey"

            if (
                self._proof_grading_attempts
                + len(self._pending_proof_reservations)
                >= MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW
            ):
                return False, "proof_grading_attempts_full"

            payload_reason = self._payload_capacity_reason_locked(
                request.miner_hotkey,
                payload_bytes,
            )
            if payload_reason is not None:
                return False, payload_reason

            trigger_round = self._seal_trigger_round
            is_post_trigger = (
                trigger_round is not None
                and request.drand_round == trigger_round
            )
            if is_post_trigger:
                if (
                    self._post_trigger_proof_admission_count
                    + self._pending_post_trigger_proof_reservations
                    >= MAX_POST_TRIGGER_PROOF_CANDIDATES
                ):
                    return False, "proof_admission_post_trigger_full"
                self._pending_post_trigger_proof_reservations += 1

            self._pending_proof_reservations[reservation_id] = (
                request,
                request.miner_hotkey,
                is_post_trigger,
                payload_bytes,
            )
            self._pending_payload_bytes += payload_bytes
            self._payload_bytes_by_hotkey[request.miner_hotkey] = (
                self._payload_bytes_by_hotkey.get(request.miner_hotkey, 0)
                + payload_bytes
            )
            return True, None

    def start_proof_admission(
        self,
        request: BatchSubmissionRequest,
    ) -> tuple[bool, str | None]:
        """Move one pending reservation into irreversible started work."""
        with self._proof_admission_lock:
            reservation_id = id(request)
            reservation = self._pending_proof_reservations.pop(
                reservation_id, None,
            )
            if self._seal_snapshot_started:
                if reservation is not None:
                    _, hotkey, is_post_trigger, payload_bytes = reservation
                    self._pending_payload_bytes = max(
                        0,
                        self._pending_payload_bytes - payload_bytes,
                    )
                    if is_post_trigger:
                        self._pending_post_trigger_proof_reservations = max(
                            0,
                            self._pending_post_trigger_proof_reservations - 1,
                        )
                    self._release_hotkey_payload_locked(hotkey, payload_bytes)
                return False, "auction_seal_snapshot_started"
            if reservation is None:
                # Compatibility for direct/legacy worker injection. Production
                # requests always reserve in the HTTP path first.
                if (
                    self._proof_grading_attempts
                    >= MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW
                ):
                    return False, "proof_grading_attempts_full"
                payload_bytes = self._submission_payload_bytes(request)
                payload_reason = self._payload_capacity_reason_locked(
                    request.miner_hotkey,
                    payload_bytes,
                )
                if payload_reason is not None:
                    return False, payload_reason
                reservation = (
                    request,
                    request.miner_hotkey,
                    False,
                    payload_bytes,
                )
                self._payload_bytes_by_hotkey[request.miner_hotkey] = (
                    self._payload_bytes_by_hotkey.get(request.miner_hotkey, 0)
                    + payload_bytes
                )
            else:
                self._pending_payload_bytes = max(
                    0,
                    self._pending_payload_bytes - reservation[3],
                )

            _, hotkey, is_post_trigger, payload_bytes = reservation
            if is_post_trigger:
                self._pending_post_trigger_proof_reservations = max(
                    0,
                    self._pending_post_trigger_proof_reservations - 1,
                )

            # A burst may have queued several requests before the first two
            # failures established debt. Re-check at dequeue so the remainder
            # cannot bypass the per-hotkey circuit breaker.
            if (
                self._expensive_proof_failures_by_hotkey.get(hotkey, 0)
                >= MAX_EXPENSIVE_PROOF_FAILURES_PER_HOTKEY_PER_WINDOW
            ):
                self._release_hotkey_payload_locked(hotkey, payload_bytes)
                return False, "proof_failure_debt_hotkey"

            self._proof_grading_attempts += 1
            if is_post_trigger:
                self._post_trigger_proof_admission_count += 1
            self._inflight_proof_reservations[reservation_id] = reservation
            self._inflight_payload_bytes += payload_bytes
            return True, None

    def cancel_proof_admission(self, request: BatchSubmissionRequest) -> bool:
        """Return a pending reservation whose proof work never started."""
        with self._proof_admission_lock:
            reservation = self._pending_proof_reservations.pop(
                id(request), None,
            )
            if reservation is None:
                return False
            if reservation[2]:
                self._pending_post_trigger_proof_reservations = max(
                    0,
                    self._pending_post_trigger_proof_reservations - 1,
                )
            payload_bytes = reservation[3]
            self._pending_payload_bytes = max(
                0,
                self._pending_payload_bytes - payload_bytes,
            )
            self._release_hotkey_payload_locked(
                reservation[1], payload_bytes
            )
            return True

    def finish_proof_admission(self, request: BatchSubmissionRequest) -> None:
        """Release work debt, retaining accepted auction payloads until seal."""
        with self._proof_admission_lock:
            reservation_id = id(request)
            reservation = self._inflight_proof_reservations.pop(
                reservation_id, None
            )
            if reservation is None:
                return
            _, hotkey, _is_post_trigger, payload_bytes = reservation
            self._inflight_payload_bytes = max(
                0,
                self._inflight_payload_bytes - payload_bytes,
            )
            if request._retain_payload and not self._seal_completed:
                self._retained_payload_reservations[reservation_id] = (
                    request,
                    hotkey,
                    payload_bytes,
                )
                self._retained_payload_bytes += payload_bytes
                return
            self._release_hotkey_payload_locked(hotkey, payload_bytes)

    def _release_retained_payloads(self) -> None:
        with self._proof_admission_lock:
            self._seal_completed = True
            reservations = list(self._retained_payload_reservations.values())
            self._retained_payload_reservations.clear()
            self._retained_payload_bytes = 0
            for request, hotkey, payload_bytes in reservations:
                request._retain_payload = False
                self._release_hotkey_payload_locked(hotkey, payload_bytes)

    def _note_grail_candidate(self) -> None:
        """Count a zone-valid submission entering the graded candidate path.

        Telemetry only. Auction mode is bounded by the collection deadline,
        grading ceiling, payload limits, and seal-time proof wall; legacy mode
        retains its trigger-round and drain bounds. This counter never rejects.
        """
        with self._proof_admission_lock:
            self._proof_admission_count += 1

    # ----------------------------- ingestion -----------------------------

    def accept_submission(
        self,
        request: BatchSubmissionRequest,
        *,
        telemetry: SubmitTelemetry | None = None,
        reward_computation: RewardComputation | None = None,
    ) -> BatchSubmissionResponse:
        """Cheaply admit a submission: validate, grade and score it, then append
        a ``PendingSubmission`` to ``_pending``. The GPU proof and every
        proof-dependent gate are deferred to ``_verify_expensive`` at seal time,
        so this path never touches ``_valid`` and never runs GRAIL.

        Does NOT re-check ``drand_round`` — that's a wall-clock timing gate,
        which is decided once at HTTP arrival by ``server.py``'s cheap-reject
        path (which calls ``validate_drand_round`` directly with the
        middleware-stamped ``t_arrival``). Re-checking here would either
        anachronistically reject (using ``time.time()`` at worker dequeue,
        which can be minutes after arrival in a saturated GRAIL queue) or
        re-do the same check with the same timestamp — both bad. Drand is
        an arrival-time check, decided once.
        """
        with self._lock:
            return self._accept_locked(
                request,
                telemetry=telemetry,
                reward_computation=reward_computation,
            )

    def compute_submission_rewards(
        self,
        request: BatchSubmissionRequest,
    ) -> RewardComputation:
        """Decode and grade one group without mutating auction state.

        Production workers call this after HTTP proof-free checks and resource
        reservation, then pass the immutable result into ``accept_submission``.
        Keeping this work outside ``_lock`` lets independent candidates use the
        bounded admission pool concurrently.  ``_accept_locked`` still repeats
        every stateful gate before committing the candidate.
        """
        started = time.perf_counter()
        problem = self.env.get_problem(request.prompt_idx)
        validator_scored_reward = _uses_validator_authoritative_reward(self.env)
        completion_texts = [
            self._completion_text(rollout) for rollout in request.rollouts
        ]
        try:
            if (
                validator_scored_reward
                and getattr(self.env, "name", "") == "opencodeinstruct"
            ):
                with ThreadPoolExecutor(
                    max_workers=len(completion_texts),
                    thread_name_prefix="reliquary-code-grade",
                ) as executor:
                    computed_rewards = list(
                        executor.map(
                            lambda text: float(
                                self.env.compute_reward(problem, text)
                            ),
                            completion_texts,
                        )
                    )
            else:
                computed_rewards = [
                    float(self.env.compute_reward(problem, text))
                    for text in completion_texts
                ]
            error: Exception | None = None
        except Exception as exc:
            computed_rewards = None
            error = exc
        return RewardComputation(
            validator_scored_reward=validator_scored_reward,
            completion_texts=completion_texts,
            rewards=computed_rewards,
            error=error,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    def observe_drand_round(
        self,
        drand_round: int,
        *,
        t_arrival: float | None = None,
    ) -> DrandRoundObservation:
        """Compute drand arrival fields and the timing-gate verdict."""
        if self._drand_chain_info is None:
            from reliquary.infrastructure.drand import get_current_chain
            self._drand_chain_info = get_current_chain()
        ci = self._drand_chain_info
        from reliquary.infrastructure.chain import compute_current_drand_round
        t = t_arrival if t_arrival is not None else self._wall_clock()
        current = compute_current_drand_round(
            t, ci["genesis_time"], ci["period"],
        )
        delta = int(drand_round) - int(current)
        status = classify_drand_round(
            int(drand_round), int(current), self.drand_round_backward_tolerance,
        )
        reject_reason: RejectReason | None = None
        if drand_round > current:
            reject_reason = RejectReason.FUTURE_ROUND
        elif drand_round < current - self.drand_round_backward_tolerance:
            reject_reason = RejectReason.STALE_ROUND
        return DrandRoundObservation(
            submitted_drand_round=int(drand_round),
            arrival_drand_round=int(current),
            drand_delta=delta,
            drand_tolerance=int(self.drand_round_backward_tolerance),
            drand_status=status,
            reject_reason=reject_reason,
        )

    def validate_drand_round(
        self,
        drand_round: int,
        *,
        t_arrival: float | None = None,
    ) -> RejectReason | None:
        """Return the appropriate reject reason if ``drand_round`` is
        outside the accepted window of [current - tolerance, current], else
        None.

        ``t_arrival`` is the wall-clock time at which the request was
        received by the validator's HTTP server (stamped by the ASGI
        ``stamp_arrival`` middleware in ``server.py``). When provided, it
        is used to compute the "current" round instead of ``time.time()``
        at call time. This decouples the round check from validator-side
        processing latency: if the asyncio loop stalls for N seconds
        (trainer GIL contention, GRAIL queue backpressure, pydantic body
        parsing on the loop), a submission that arrived at the validator
        within its round is still accepted, because the round is computed
        as of the arrival timestamp, not as of when the handler finally
        runs. Without this, an honest submission that landed inside its
        round can become STALE_ROUND purely because the validator was
        busy when the handler executed.

        Falls back to ``self._wall_clock()`` if ``t_arrival`` is None
        (e.g. the worker-path re-check inside ``_accept_locked`` called
        directly, tests that don't go through the HTTP layer).

        Forward direction is zero-tolerance: a miner that attaches round
        R+1 hasn't seen σ_{R+1} yet (σ_R is the freshest signed beacon by
        definition), so claiming a future round is unrecoverable cheating
        and always rejected as FUTURE_ROUND.

        Backward direction allows only the configured
        ``self.drand_round_backward_tolerance``. Production sets this to zero.
        Auction ranking does not use the submitted round, and large body
        transport is handled by signed precommit/reveal: the small precommit's
        arrival fixes the drand observation before upload. Widening tolerance
        therefore weakens freshness without solving the transport problem.

        Public so the HTTP /submit handler can run it pre-queue and
        short-circuit the rejection without waiting on the worker.
        """
        return self.observe_drand_round(
            drand_round, t_arrival=t_arrival,
        ).reject_reason

    def _accept_locked(
        self,
        request: BatchSubmissionRequest,
        *,
        telemetry: SubmitTelemetry | None = None,
        reward_computation: RewardComputation | None = None,
    ) -> BatchSubmissionResponse:
        hk = request.miner_hotkey
        pi = request.prompt_idx

        def reject(
            reason: RejectReason,
            stage: str,
            **kwargs: Any,
        ) -> BatchSubmissionResponse:
            return self._reject(
                reason,
                hotkey=hk,
                prompt_idx=pi,
                telemetry=telemetry,
                reject_stage=stage,
                **kwargs,
            )

        # Legacy environments stop after the trigger drand tier. Auction
        # environments intentionally collect for the full fixed deadline.
        if (
            not self.difficulty_auction_enabled
            and self._seal_trigger_round is not None
            and request.drand_round > self._seal_trigger_round
        ):
            return reject(RejectReason.BATCH_FILLED, "seal_extension")

        if request.window_start != self.window_start:
            return reject(RejectReason.WINDOW_MISMATCH, "window")
        # v2.1: checkpoint hash gate. Empty string = gate disabled
        # (pre-first-publish or test convenience).
        if self.current_checkpoint_hash and request.checkpoint_hash != self.current_checkpoint_hash:
            return reject(RejectReason.WRONG_CHECKPOINT, "checkpoint")
        if (
            FORCED_SEED_ENFORCE
            and self.current_checkpoint_hash
            and request.protocol_version != FORCED_SEED_PROTOCOL_VERSION
        ):
            return reject(
                RejectReason.SEED_MISMATCH,
                "forced_seed_protocol",
            )
        # NOTE: drand_round is intentionally NOT re-checked here. It's a
        # wall-clock timing gate decided once at HTTP arrival (see
        # ``server.py``'s cheap-reject path → ``validate_drand_round`` with
        # the middleware-stamped ``t_arrival``). Re-checking at worker
        # dequeue would use ``time.time()`` at dequeue, which can be
        # minutes after the submission arrived if the GRAIL queue is
        # saturated — turning honest, on-time submissions into STALE_ROUND
        # rejections. Drand is the single check that depends on the
        # validator's clock at receipt, so it belongs exclusively on the
        # arrival path.
        if request.prompt_idx >= len(self.env):
            return reject(RejectReason.BAD_PROMPT_IDX, "prompt")
        if self.prompt_range is not None:
            lo, hi = self.prompt_range
            if not (lo <= request.prompt_idx < hi):
                return reject(RejectReason.PROMPT_OUT_OF_RANGE, "prompt_range")
        if self._cooldown.is_in_cooldown(request.prompt_idx, self.window_start):
            return reject(RejectReason.PROMPT_IN_COOLDOWN, "cooldown")
        # Cap each prompt's retained population before deferred proof. Auction
        # logical dedup already limits an operator to one claim per prompt; this
        # bounds distinct-operator crowding and retained memory.
        existing = self._submissions_per_prompt.get(request.prompt_idx, [])
        if len(existing) >= MAX_SUBMISSIONS_PER_PROMPT:
            return reject(RejectReason.PROMPT_FULL, "prompt_capacity")

        for rollout in request.rollouts:
            try:
                CommitModel.model_validate(rollout.commit)
            except ValidationError:
                return reject(RejectReason.BAD_SCHEMA, "schema")
            if list(rollout.tokens) != list(rollout.commit["tokens"]):
                return reject(RejectReason.TOKENS_MISMATCH, "token_invariant")

        # Defense in depth for direct batcher callers. The HTTP path computes
        # this before quota/proof admission and marks the private request attr;
        # no wire field or miner behavior changes.
        if (
            LEGACY_MERKLE_ROOT_ENFORCE
            and not request._legacy_merkle_verified
        ):
            try:
                matches, _ = legacy_submission_merkle_matches(request)
            except (
                AttributeError,
                KeyError,
                TypeError,
                ValueError,
                OverflowError,
            ):
                matches = False
            if not matches:
                return reject(RejectReason.MERKLE_ROOT_MISMATCH, "legacy_merkle")
            request._legacy_merkle_verified = True

        # Proof-free checks before reward and GRAIL. These reject malformed
        # payloads before tokenizer decode, env reward work, or GPU proof work.
        canonical_prompt_tokens: list[int] | None = None
        if self._canonical_prompt_tokens is not None:
            canonical_prompt_tokens = list(
                self._canonical_prompt_tokens(request.prompt_idx)
            )

        for rollout in request.rollouts:
            if not verify_tokens(rollout.commit["tokens"], self.model.config):
                return reject(RejectReason.BAD_TOKENS, "tokens")

            if canonical_prompt_tokens is not None:
                rollout_meta = rollout.commit.get("rollout", {}) or {}
                miner_prompt_len = int(rollout_meta.get("prompt_length", 0))
                miner_prompt_tokens = list(rollout.commit.get("tokens", []))[
                    :miner_prompt_len
                ]
                if miner_prompt_tokens != canonical_prompt_tokens:
                    return reject(RejectReason.PROMPT_MISMATCH, "prompt_binding")

        # Per-rollout hash dedup against the persistent set + within this
        # submission. Computed once here, reused at seal_batch and archive.
        # Skipped entirely when hash_set is None (back-compat for tests that
        # pass identical-token rollouts through the pipeline).
        rollout_hashes: list[bytes] = []
        if self._hash_set is not None:
            local_seen: set[bytes] = set()
            for rollout in request.rollouts:
                h = compute_rollout_hash(rollout.commit["tokens"])
                if h in local_seen or h in self._hash_set:
                    logger.info(
                        "reject reason=hash_duplicate hotkey=%s prompt=%d",
                        hk, pi,
                    )
                    return reject(RejectReason.HASH_DUPLICATE, "dedup")
                local_seen.add(h)
                rollout_hashes.append(h)

        try:
            logical_reserved, logical_reason = self.try_reserve_logical_group(
                request
            )
        except (TypeError, ValueError, OverflowError):
            return reject(RejectReason.BAD_TOKENS, "logical_dedup")
        if not logical_reserved:
            if logical_reason == "operator_unmapped":
                return reject(
                    RejectReason.REGISTRATION_UNAVAILABLE,
                    "operator_mapping",
                )
            return reject(RejectReason.HASH_DUPLICATE, "logical_dedup")
        if reward_computation is None:
            reward_computation = self.compute_submission_rewards(request)
        validator_scored_reward = reward_computation.validator_scored_reward
        completion_texts = reward_computation.completion_texts
        computed_rewards = reward_computation.rewards
        reward_error = reward_computation.error
        if isinstance(reward_error, GraderInfrastructureError):
            exc = reward_error
            reason = exc.reason or "unknown"
            self.grader_failures[reason] = (
                self.grader_failures.get(reason, 0) + 1
            )
            # A worker crash may be triggered by hostile submitted code. Never
            # convert it into a zero reward (which could manufacture auction
            # difficulty), but also do not grant free retries: retain the
            # operator/prompt claim and consume the normal submission quota.
            if reason == "crash":
                self.confirm_logical_group_reservation(request)
                logger.error(
                    "code_grader_worker_crash env=%s prompt=%d hotkey=%s",
                    getattr(self.env, "name", type(self.env).__name__),
                    request.prompt_idx,
                    hk,
                )
                return reject(RejectReason.REWARD_MISMATCH, "code_grader_crash")
            self.cancel_logical_group_reservation(request)
            logger.error(
                "code_grader_unavailable env=%s prompt=%d hotkey=%s reason=%s",
                getattr(self.env, "name", type(self.env).__name__),
                request.prompt_idx,
                hk,
                reason,
            )
            return reject(RejectReason.WORKER_DROPPED, "code_grader")
        if reward_error is not None or computed_rewards is None:
            return reject(RejectReason.REWARD_MISMATCH, "reward")

        for rollout, computed_reward in zip(
            request.rollouts,
            computed_rewards,
            strict=True,
        ):
            if not math.isfinite(computed_reward):
                logger.error(
                    "non-finite validator reward env=%s prompt=%d hotkey=%s",
                    getattr(self.env, "name", type(self.env).__name__),
                    request.prompt_idx,
                    hk,
                )
                return reject(RejectReason.REWARD_MISMATCH, "reward")
            if validator_scored_reward:
                rollout.reward = computed_reward
                rollout_meta = rollout.commit.get("rollout")
                if isinstance(rollout_meta, dict):
                    rollout_meta["success"] = computed_reward > 0.5
                    rollout_meta["total_reward"] = computed_reward
            elif not _reward_matches_claim(computed_reward, rollout.reward):
                return reject(RejectReason.REWARD_MISMATCH, "reward")

        rewards = [float(r.reward) for r in request.rollouts]
        completion_lengths = []
        for rollout in request.rollouts:
            rollout_meta = (rollout.commit or {}).get("rollout", {}) or {}
            prompt_len = int(rollout_meta.get("prompt_length", 0) or 0)
            completion_lengths.append(
                int(
                    rollout_meta.get(
                        "completion_length",
                        max(0, len(rollout.commit.get("tokens", [])) - prompt_len),
                    )
                    or 0
                )
            )
        sigma = rewards_std(rewards)
        # Keep the calibrated sigma eligibility band even under the auction.
        # The difficulty score ranks eligible groups, but k=1 remains excluded
        # until an independent math adjudicator measures the false-negative
        # floor: under this objective, a correct answer graded wrong can move a
        # group toward the payout peak. This is a reward error, not cheating.
        if not is_in_zone(sigma, bootstrap=self.bootstrap):
            return reject(RejectReason.OUT_OF_ZONE, "zone")

        # Zone-valid: entering the GRAIL/GPU proof path. Count it for telemetry
        # only — the window is bounded by the drand seal + grading ceiling +
        # drain timeout, not by a candidate budget (removed: it starved honest
        # late arrivals when earlier candidates burned an unrefunded slot on a
        # post-reservation gate such as forced-seed).
        self._note_grail_candidate()

        # A reward=0 rollout whose final \boxed{} is malformed (empty,
        # special-token, or unclosed) produced no parseable answer — a fake
        # negative used to manufacture k=4 / sigma=0.5 and pass the zone filter.
        # Aligned with the env (which scores the last box); a well-formed wrong
        # answer is a legitimate negative and is not flagged. Before GRAIL.
        for _ri, _text in enumerate(completion_texts):
            _rmeta = request.rollouts[_ri].commit.get("rollout", {}) or {}
            _clen = int(_rmeta.get("completion_length", 0))
            _bad, _bad_reason = has_malformed_final_answer(
                rewards[_ri], _text,
                completion_length=_clen, cap=MAX_NEW_TOKENS_PROTOCOL_CAP,
            )
            if _bad:
                logger.info(
                    "reject reason=malformed_final_answer hotkey=%s rollout=%d cond=%s",
                    request.miner_hotkey, _ri, _bad_reason,
                )
                return reject(
                    RejectReason.MALFORMED_FINAL_ANSWER, "malformed_final_answer"
                )

        clone_metrics = detect_opposite_reward_clones(completion_texts, rewards)
        if clone_metrics.suspicious:
            logger.info(
                "reject reason=distribution_suspicious hotkey=%s "
                "manufactured_opposite_reward_clones=%s",
                request.miner_hotkey,
                clone_metrics.to_log_dict(),
            )
            return reject(RejectReason.DISTRIBUTION_SUSPICIOUS, "distribution")

        # Cheap per-rollout checks that do NOT need the GPU proof: strip the
        # validator-owned flags a miner may have set, verify the rollout
        # signature, and bind the miner-claimed beacon randomness to this
        # window. Everything proof-dependent runs at seal in _verify_expensive.
        for rollout in request.rollouts:
            # `truncated` is a validator-set flag (overlong reward shaping, see
            # submission.py). Wipe any miner-supplied value at ingestion so only
            # the validator's own cap/EOS detection can set it — otherwise a
            # miner could flag a losing rollout to clamp its negative advantage
            # to -SHAPE_PENALTY via _shape_advantages and attenuate the gradient.
            _ingest_meta = rollout.commit.get("rollout")
            if isinstance(_ingest_meta, dict):
                _ingest_meta["truncated"] = False
                # BFT is math-only: `forced` is validator-honoured solely for
                # openmathinstruct. Wipe any non-math value so the carve-out
                # stays scoped to math.
                if getattr(self.env, "name", "") != "openmathinstruct":
                    _ingest_meta["forced"] = False
            if not self._verify_signature(rollout.commit, request.miner_hotkey):
                return reject(RejectReason.BAD_SIGNATURE, "rollout_signature")
            # Randomness binding: the miner-claimed beacon randomness MUST equal
            # the validator's per-window derived randomness. Without this check
            # the sketch-tolerance window is wide enough that a constant
            # pre-computed r_vec can slip under the GRAIL diff threshold. Reject
            # here, before the (deferred) forward pass on a commit we already
            # know is detached from the validator's window seed.
            claimed_rand = (rollout.commit.get("beacon") or {}).get("randomness", "")
            if claimed_rand != self.randomness:
                return reject(RejectReason.WRONG_RANDOMNESS, "randomness")

        pending = PendingSubmission(
            hotkey=request.miner_hotkey,
            prompt_idx=request.prompt_idx,
            request=request,
            rewards=list(rewards),
            drand_round=request.drand_round,
            merkle_root=bytes.fromhex(request.merkle_root),
            selection_digest=compute_rollouts_selection_digest(request.rollouts),
            arrived_at=self._time_fn(),
            decision_ts=self._wall_clock(),
            telemetry=telemetry,
        )

        if not self.difficulty_auction_enabled:
            self.confirm_logical_group_reservation(request)
            proven = self._verify_expensive(pending)
            if proven is None:
                return pending.reject_response or BatchSubmissionResponse(
                    accepted=False,
                    reason=RejectReason.GRAIL_FAIL,
                )
            self._valid.append(proven)
            self._submissions_per_prompt.setdefault(
                request.prompt_idx, []
            ).append(proven)
            self.last_valid_submission_at = self._time_fn()
            self.last_valid_submission_wall_ts = self._wall_clock()
            self.valid_count = len(self._valid)

            distinct_eligible = self.distinct_valid_prompt_count()
            if distinct_eligible >= B_BATCH and self._seal_trigger_round is None:
                self._seal_trigger_round = request.drand_round
                proven.seal_trigger_round = self._seal_trigger_round
                if telemetry is not None:
                    telemetry.seal_trigger_round = self._seal_trigger_round
                    telemetry.valid_submissions_at_decision = len(self._valid)
                    log_submission_stage(
                        logger,
                        logging.INFO,
                        "seal_triggered",
                        telemetry,
                        distinct_eligible=distinct_eligible,
                        batch_size=B_BATCH,
                        seal_trigger_round=self._seal_trigger_round,
                    )
                if self._loop is not None:
                    asyncio.run_coroutine_threadsafe(
                        self._delayed_seal_at_drand_boundary(),
                        self._loop,
                    )
                else:
                    self._seal_flag.set()
                    if self._seal_event is not None:
                        self._seal_event.set()
            return BatchSubmissionResponse(
                accepted=True, reason=RejectReason.ACCEPTED
            )

        # The HTTP worker releases its in-flight marker after this returns, but
        # the request remains reachable from the auction pool until seal. Mark
        # it so byte accounting transfers to the retained bucket instead of
        # being refunded prematurely.
        self.confirm_logical_group_reservation(request)
        request._retain_payload = True
        self._pending.append(pending)
        self._submissions_per_prompt.setdefault(
            request.prompt_idx, []
        ).append(pending)
        self.last_valid_submission_at = self._time_fn()
        self.last_valid_submission_wall_ts = self._wall_clock()
        # Lock-free read in /state — see ``__init__`` for rationale.
        self.pending_count = len(self._pending)

        # No count-based seal: the window seals only on the collection deadline
        # (``poll_deadline``), so a fast miner on an easy prompt can no longer
        # cut off slow-but-hard submissions still generating.
        return BatchSubmissionResponse(
            accepted=True, reason=RejectReason.ACCEPTED
        )

    async def _delayed_seal_at_drand_boundary(self) -> None:
        """Seal a legacy environment after its trigger round and queue drain."""
        if self._drand_chain_info is not None:
            from reliquary.infrastructure.chain import (
                seconds_until_next_drand_boundary,
            )

            ci = self._drand_chain_info
            delay = seconds_until_next_drand_boundary(
                self._wall_clock(), ci["genesis_time"], ci["period"]
            )
            await asyncio.sleep(delay + 0.05)

        if self._queue_drained_predicate is not None:
            drain_started = self._time_fn()
            while not self._queue_drained_predicate():
                waited_s = self._time_fn() - drain_started
                if waited_s >= MAX_SEAL_QUEUE_DRAIN_SECONDS:
                    logger.warning(
                        "seal_drain_timeout window=%d trigger_round=%s "
                        "waited_s=%.2f proof_admitted=%d post_trigger_admitted=%d",
                        self.window_start,
                        self._seal_trigger_round,
                        waited_s,
                        self._proof_admission_count,
                        self.post_trigger_proof_admission_count,
                    )
                    break
                await asyncio.sleep(0.2)

        self._seal_flag.set()
        if self._seal_event is not None:
            self._seal_event.set()

    def _verify_expensive(
        self, pending: PendingSubmission
    ) -> ValidSubmission | None:
        """Prove one graded candidate on the GPU and run every proof-dependent
        gate. Returns the ``ValidSubmission`` on success, ``None`` on rejection.

        Deliberately NOT called under ``self._lock``: the GRAIL forward is 5-25 s
        and holding the lock across it is what serialized admission. On rejection
        it calls ``self._reject`` exactly as the inline pipeline did, so
        per-hotkey proof-failure debt and the R2 archive entries are unchanged.

        NOT state-free — via ``self._reject`` it mutates
        ``reject_counts``/``rejected_submissions`` and flips
        ``rollout.commit["rollout"]["truncated"]``. It MUST be called serially
        (``_prove_ranked`` runs a top-down loop, never in parallel).
        """
        request = pending.request
        telemetry = pending.telemetry
        hk = request.miner_hotkey
        pi = request.prompt_idx

        def reject(
            reason: RejectReason,
            stage: str,
            **kwargs: Any,
        ) -> None:
            pending.reject_response = self._reject(
                reason,
                hotkey=hk,
                prompt_idx=pi,
                telemetry=telemetry,
                reject_stage=stage,
                # Proof-stage rejects run at seal, but the archived forensic
                # timestamp must stay the admission instant.
                decision_ts=pending.decision_ts,
                **kwargs,
            )
            return None

        # Re-derive the cheap locals the moved gates read. Each is a pure
        # function of the request, so they are identical to what admission saw.
        problem = self.env.get_problem(pi)
        completion_texts = [
            self._completion_text(rollout) for rollout in request.rollouts
        ]
        rewards = list(pending.rewards)
        sigma = rewards_std(rewards)
        completion_lengths = []
        for rollout in request.rollouts:
            rollout_meta = (rollout.commit or {}).get("rollout", {}) or {}
            prompt_len = int(rollout_meta.get("prompt_length", 0) or 0)
            completion_lengths.append(
                int(
                    rollout_meta.get(
                        "completion_length",
                        max(0, len(rollout.commit.get("tokens", [])) - prompt_len),
                    )
                    or 0
                )
            )
        rollout_hashes: list[bytes] = []
        if self._hash_set is not None:
            rollout_hashes = [
                compute_rollout_hash(rollout.commit["tokens"])
                for rollout in request.rollouts
            ]

        # Per-submission worst-case filter telemetry (across all rollouts).
        sketch_diff_max = 0
        lp_dev_max: float | None = None
        dist_q10_min: float | None = None
        all_token_auth_shadow_findings = 0
        all_token_auth_shadow_min_prob: float | None = None
        all_token_auth_shadow_positive_findings = 0
        all_token_auth_shadow_positive_min_prob: float | None = None
        private_auth_forensics_enabled = auth_forensics_enabled()
        private_auth_forensics_max_findings = (
            auth_forensics_max_findings_per_rollout()
        )
        private_auth_forensics_context_chars = auth_forensics_context_chars()
        code_semantic_counterfactuals_enabled = (
            code_semantic_counterfactual_enabled()
        )
        code_semantic_counterfactuals_max_findings = (
            code_semantic_counterfactual_max_findings_per_rollout()
        )
        code_semantic_auth_findings = 0
        code_semantic_auth_min_prob: float | None = None
        code_semantic_auth_positive_findings = 0
        code_semantic_auth_positive_min_prob: float | None = None

        # Cap/non-EOS truncation is tolerated only as a rare one-rollout
        # accident. Multiple missing-EOS rollouts in the same group are a
        # sampling policy and make weak loser slots too easy to manufacture.
        max_truncated_per_submission = (
            BOOTSTRAP_MAX_TRUNCATED_PER_SUBMISSION
            if self.bootstrap
            else MAX_TRUNCATED_PER_SUBMISSION
        )
        truncated_count = 0
        truncated_flags = [False] * len(request.rollouts)
        # BFT carve-out: resolve the canonical FORCE ids once, only if some
        # rollout is forced; non-forced submissions never touch the tokenizer.
        from reliquary.constants import BFT_ANSWER_BUDGET, BFT_THINKING_BUDGET
        from reliquary.shared.modeling import (
            resolve_eos_token_ids,
            think_close_token_ids,
        )
        canonical_force_ids: list[int] = []
        force_think_close_ids: set[int] = set()
        try:
            telemetry_eos_ids = set(
                resolve_eos_token_ids(self.model, self.tokenizer)
            )
        except Exception:
            telemetry_eos_ids = set()
        try:
            telemetry_think_close_ids = set(
                think_close_token_ids(self.tokenizer)
            )
        except Exception:
            telemetry_think_close_ids = set()
        if any((r.commit.get("rollout") or {}).get("forced")
               for r in request.rollouts):
            from reliquary.shared.modeling import force_close_token_ids
            try:
                canonical_force_ids = force_close_token_ids(self.tokenizer)
                # the FORCE ids begin with the atomic </think> id
                force_think_close_ids = (
                    {int(canonical_force_ids[0])} if canonical_force_ids else set()
                )
            except Exception:
                canonical_force_ids = []

        # Forced-seed group tally: summed across all rollouts in this
        # submission, verdict decided once after the loop (see
        # ``_forced_seed_verdict``) — per-rollout counts are too thin a
        # sample to gate on individually.
        from reliquary.environment.forced_sampling import u_at
        grp_stoch = 0
        grp_match = 0
        grp_seed_positions = 0
        grp_seed_boundary_match = 0
        grp_seed_hard_mismatch = 0
        grp_seed_deterministic_hard_mismatch = 0
        grp_seed_miss_gt_0_01 = 0
        grp_seed_miss_gt_0_05 = 0
        grp_seed_miss_gt_0_10 = 0
        grp_seed_max_cdf_miss = 0.0
        # Per-rollout (n_stoch, n_match) — the per-rollout gate needs each
        # rollout separately, since the group average hides a partial swap.
        seed_per_rollout: list[tuple[int, int]] = []
        seed_cdf_per_rollout: list[dict[str, Any]] = []

        for rollout_idx, rollout in enumerate(request.rollouts):
            # Never carry a validator-derived carve across re-validation of the
            # same Pydantic object. The private value is set only after the
            # signed commit's force span passes the canonical BFT checks below.
            # (Signature, randomness binding and the truncated-flag wipe already
            # ran in the cheap admission phase — see ``_accept_locked``.)
            rollout._validated_force_span = None
            rollout._validated_termination_path = None
            # Per-position forced-seed uniforms for this rollout's teacher-forced
            # consistency check. Read completion_length here (ahead of the
            # ``completion_len`` computed later at the sparse-outputs section)
            # so the u-stream can accompany the verify call below.
            _seed_meta = rollout.commit.get("rollout") or {}
            _seed_completion_len = int(_seed_meta.get("completion_length", 0))
            _seed_prompt_len = int(_seed_meta.get("prompt_length", 0))
            _seed_tokens = rollout.commit.get("tokens") or []
            _seed_completion_tokens = _seed_tokens[
                _seed_prompt_len:_seed_prompt_len + _seed_completion_len
            ]
            rollout_token_metrics = token_degeneracy_metrics(
                _seed_completion_tokens
            )
            rollout_sketch_metrics = sketch_commitment_metrics(
                rollout.commit.get("commitments") or []
            )
            seed_u = [
                u_at(
                    self.randomness, request.prompt_idx,
                    request.checkpoint_hash, rollout_idx, j,
                )
                for j in range(_seed_completion_len)
            ]
            try:
                proof = self._verify_commitment(
                    rollout.commit,
                    self.model,
                    self.randomness,
                    tokenizer=self.tokenizer,
                    seed_u_values=seed_u,
                )
            except TypeError as exc:
                # Backward-compat fallback for stub verifiers (tests, legacy
                # callers) that don't accept one or both of the newer kwargs.
                # Retry narrowing from most- to least-featured signature
                # rather than guessing which kwarg tripped it. Matched
                # strictly against Python's "unexpected keyword argument"
                # TypeError text (not a bare substring test) so a genuine
                # internal TypeError raised inside a real verifier propagates
                # instead of being masked and retried without seed_u_values.
                if _is_missing_kwarg_typeerror(exc, "seed_u_values"):
                    try:
                        proof = self._verify_commitment(
                            rollout.commit, self.model, self.randomness,
                            tokenizer=self.tokenizer,
                        )
                    except TypeError as exc2:
                        if not _is_missing_kwarg_typeerror(exc2, "tokenizer"):
                            raise
                        proof = self._verify_commitment(
                            rollout.commit, self.model, self.randomness,
                        )
                elif _is_missing_kwarg_typeerror(exc, "tokenizer"):
                    proof = self._verify_commitment(
                        rollout.commit, self.model, self.randomness,
                    )
                else:
                    raise
            grp_stoch += proof.seed_n_stochastic
            grp_match += proof.seed_n_match
            seed_per_rollout.append((proof.seed_n_stochastic, proof.seed_n_match))
            seed_positions = int(getattr(proof, "seed_n_positions", 0) or 0)
            seed_boundary_match = int(
                getattr(proof, "seed_n_boundary_match", 0) or 0
            )
            seed_hard_mismatch = int(
                getattr(proof, "seed_n_hard_mismatch", 0) or 0
            )
            seed_deterministic_hard = int(
                getattr(
                    proof,
                    "seed_n_deterministic_hard_mismatch",
                    0,
                )
                or 0
            )
            seed_miss_gt_0_01 = int(
                getattr(proof, "seed_n_miss_gt_0_01", 0) or 0
            )
            seed_miss_gt_0_05 = int(
                getattr(proof, "seed_n_miss_gt_0_05", 0) or 0
            )
            seed_miss_gt_0_10 = int(
                getattr(proof, "seed_n_miss_gt_0_10", 0) or 0
            )
            seed_max_cdf_miss = float(
                getattr(proof, "seed_max_cdf_miss", 0.0) or 0.0
            )
            seed_first_hard_mismatch_offset = getattr(
                proof, "seed_first_hard_mismatch_offset", None
            )
            grp_seed_positions += seed_positions
            grp_seed_boundary_match += seed_boundary_match
            grp_seed_hard_mismatch += seed_hard_mismatch
            grp_seed_deterministic_hard_mismatch += seed_deterministic_hard
            grp_seed_miss_gt_0_01 += seed_miss_gt_0_01
            grp_seed_miss_gt_0_05 += seed_miss_gt_0_05
            grp_seed_miss_gt_0_10 += seed_miss_gt_0_10
            grp_seed_max_cdf_miss = max(
                grp_seed_max_cdf_miss,
                seed_max_cdf_miss,
            )
            seed_cdf_entry: dict[str, Any] = {
                "rollout_idx": rollout_idx,
                "n_positions": seed_positions,
                "n_stochastic": int(proof.seed_n_stochastic),
                "n_exact_match": int(proof.seed_n_match),
                "n_boundary_match": seed_boundary_match,
                "n_hard_mismatch": seed_hard_mismatch,
                "n_deterministic_hard_mismatch": seed_deterministic_hard,
                "n_miss_gt_0_01": seed_miss_gt_0_01,
                "n_miss_gt_0_05": seed_miss_gt_0_05,
                "n_miss_gt_0_10": seed_miss_gt_0_10,
                "max_cdf_miss": seed_max_cdf_miss,
                "first_hard_mismatch_offset": (
                    int(seed_first_hard_mismatch_offset)
                    if seed_first_hard_mismatch_offset is not None
                    else None
                ),
                "completion_length": _seed_completion_len,
                "claimed_forced": bool(_seed_meta.get("forced")),
                "forced": False,
                "validated_force_span_length": 0,
                **rollout_token_metrics,
                **rollout_sketch_metrics,
            }
            seed_cdf_per_rollout.append(seed_cdf_entry)
            if proof.sketch_diff_max > sketch_diff_max:
                sketch_diff_max = proof.sketch_diff_max
            if not proof.all_passed:
                logger.warning(
                    "grail_fail diag hotkey=%s prompt=%d sketch_diff_max=%d "
                    "passed=%d/%d",
                    request.miner_hotkey, request.prompt_idx,
                    proof.sketch_diff_max, proof.passed, proof.checked,
                )
                return reject(
                    RejectReason.GRAIL_FAIL,
                    "grail",
                    sketch_diff_max=proof.sketch_diff_max,
                )

            # Termination check: rollout must end with EOS at p(EOS) >= threshold
            # or hit the protocol cap. Cap hits without a natural EOS are counted
            # against the per-submission truncation budget; otherwise forced
            # max-length sampling can make every rollout bypass EOS validation.
            # Reject EOS padding outright: honest miners truncate at the first
            # EOS, and repeated stop-token tails are high-probability junk that
            # can pass logprob/distribution while poisoning training.
            # Reads precomputed p_stop on ``proof`` — no logits round-trip.
            # Skipped when the stub didn't populate sparse outputs (legacy
            # test fixtures that opted out of behavioural enforcement).
            if proof.has_sparse_outputs:
                if has_eos_padding(rollout.commit, self.tokenizer, self.model):
                    return reject(
                        RejectReason.BAD_TERMINATION,
                        "termination",
                        sketch_diff_max=sketch_diff_max,
                    )
                termination_ok = verify_termination(
                    rollout.commit,
                    self.tokenizer,
                    proof,
                    self.model,
                    env_name=getattr(self.env, "name", ""),
                )
                cap_truncated = is_cap_truncation(
                    rollout.commit,
                    self.tokenizer,
                    proof,
                    self.model,
                    env_name=getattr(self.env, "name", ""),
                )
                terminal_pick_ok = getattr(proof, "terminal_pick_ok", None)
                terminal_pick_cdf_miss = getattr(
                    proof, "terminal_pick_cdf_miss", None
                )
                natural_close_pick_ok = getattr(
                    proof, "natural_close_pick_ok", None
                )
                natural_close_pick_cdf_miss = getattr(
                    proof, "natural_close_pick_cdf_miss", None
                )
                p_stop = getattr(proof, "p_stop", None)
                natural_cap_candidate = is_natural_bft_cap_candidate(
                    rollout.commit,
                    self.tokenizer,
                    env_name=getattr(self.env, "name", ""),
                )
                low_probability_terminal = bool(
                    terminal_pick_ok is not None
                    and p_stop is not None
                    and float(p_stop) < MIN_EOS_PROBABILITY
                )
                increments_truncation = not termination_ok or cap_truncated
                if private_auth_forensics_enabled and (
                    increments_truncation
                    or low_probability_terminal
                    or natural_cap_candidate
                ):
                    record_termination_shadow(
                        hotkey=request.miner_hotkey,
                        window_start=self.window_start,
                        env_name=getattr(self.env, "name", ""),
                        checkpoint_hash=request.checkpoint_hash,
                        prompt_idx=request.prompt_idx,
                        rollout_idx=rollout_idx,
                        completion_length=int(
                            (rollout.commit.get("rollout") or {}).get(
                                "completion_length", 0
                            )
                        ),
                        p_stop=p_stop,
                        terminal_pick_ok=terminal_pick_ok,
                        terminal_pick_cdf_miss=terminal_pick_cdf_miss,
                        natural_close_pick_ok=natural_close_pick_ok,
                        natural_close_pick_cdf_miss=(
                            natural_close_pick_cdf_miss
                        ),
                        termination_ok=termination_ok,
                        cap_truncated=cap_truncated,
                        would_exceed_truncation_budget=(
                            increments_truncation
                            and truncated_count + 1
                            > max_truncated_per_submission
                        ),
                        boundary_epsilon=(
                            FORCED_SEED_CDF_BOUNDARY_EPSILON
                        ),
                        seed_n_hard_mismatch=seed_hard_mismatch,
                        seed_first_hard_mismatch_offset=(
                            seed_first_hard_mismatch_offset
                        ),
                        token_metrics=rollout_token_metrics,
                    )
                if not termination_ok or cap_truncated:
                    truncated_flags[rollout_idx] = True
                    truncated_count += 1
                    # Validator-set flag for the overlong side of reward shaping.
                    _rdict = rollout.commit.get("rollout")
                    if isinstance(_rdict, dict):
                        _rdict["truncated"] = True
                    if truncated_count > max_truncated_per_submission:
                        return reject(
                            RejectReason.BAD_TERMINATION,
                            "termination",
                            sketch_diff_max=sketch_diff_max,
                        )
                    # Only the EOS signal is missing on a cap-truncated rollout;
                    # the per-token integrity checks (logprob, distribution,
                    # boxed-answer) still apply to its body. Do NOT skip them
                    # — a miner who force-caps to bypass behavioural checks
                    # otherwise gets up to max_truncated_per_submission rollouts
                    # of free tampering inside the same submission.

            if not proof.has_sparse_outputs:
                continue

            rollout_dict = rollout.commit.get("rollout", {}) or {}
            prompt_len = int(rollout_dict.get("prompt_length", 0))
            completion_len = int(rollout_dict.get("completion_length", 0))
            claimed_lp = rollout_dict.get("token_logprobs", []) or []

            # BFT carve-out: validate a forced rollout's FORCE span (byte-exact,
            # atomic-</think>-anchored, at the thinking budget); a valid span's
            # positions are exempted from the per-token auth / distribution
            # checks (their probability is legitimately ~0 — injected, not sampled).
            carve_ok, exempt_positions = validate_force_span(
                rollout.commit["tokens"], rollout_dict,
                canonical_force_ids, prompt_len,
                thinking_budget=BFT_THINKING_BUDGET,
                think_close_ids=force_think_close_ids,
            )
            if not carve_ok:
                return reject(
                    RejectReason.TOKEN_TAMPERED,
                    "force_span",
                    sketch_diff_max=sketch_diff_max,
                )
            if rollout_dict.get("forced"):
                declared_span = rollout_dict.get("force_span")
                rollout._validated_force_span = (
                    int(declared_span[0]),
                    int(declared_span[1]),
                )
            seed_cdf_entry["forced"] = bool(rollout._validated_force_span)
            seed_cdf_entry["validated_force_span_length"] = (
                rollout._validated_force_span[1]
                - rollout._validated_force_span[0]
                if rollout._validated_force_span is not None
                else 0
            )
            rollout._validated_termination_path = classify_bft_termination(
                rollout.commit["tokens"],
                prompt_length=prompt_len,
                completion_length=completion_len,
                eos_ids=telemetry_eos_ids,
                think_close_ids=telemetry_think_close_ids,
                validated_force_span=rollout._validated_force_span,
                thinking_budget=BFT_THINKING_BUDGET,
                answer_budget=BFT_ANSWER_BUDGET,
            )
            seed_cdf_entry["termination_path"] = (
                rollout._validated_termination_path
            )
            seed_cdf_entry["sketch_diff_max"] = int(
                proof.sketch_diff_max
            )

            lp_ok, lp_dev = verify_logprobs_claim(
                tokens=rollout.commit["tokens"],
                prompt_length=prompt_len,
                completion_length=completion_len,
                claimed_logprobs=claimed_lp,
                proof=proof,
            )
            if lp_dev is not None and lp_dev != float("inf"):
                if lp_dev_max is None or lp_dev > lp_dev_max:
                    lp_dev_max = float(lp_dev)
            if not lp_ok:
                logger.info(
                    "reject reason=logprob_mismatch hotkey=%s median_dev=%.4f",
                    request.miner_hotkey, lp_dev,
                )
                return reject(
                    RejectReason.LOGPROB_MISMATCH,
                    "logprob",
                    sketch_diff_max=sketch_diff_max,
                    lp_dev_max=lp_dev_max,
                )

            dist_ok, dist_metrics = evaluate_token_distribution(
                tokens=rollout.commit["tokens"],
                prompt_length=prompt_len,
                completion_length=completion_len,
                proof=proof,
                exempt_positions=exempt_positions,
            )
            if dist_metrics and "q10" in dist_metrics:
                q10 = float(dist_metrics["q10"])
                if dist_q10_min is None or q10 < dist_q10_min:
                    dist_q10_min = q10
            if dist_ok is False:
                logger.info(
                    "reject reason=distribution_suspicious hotkey=%s %s",
                    request.miner_hotkey, dist_metrics,
                )
                return reject(
                    RejectReason.DISTRIBUTION_SUSPICIOUS,
                    "distribution",
                    sketch_diff_max=sketch_diff_max,
                    lp_dev_max=lp_dev_max,
                    dist_q10_min=dist_q10_min,
                )

            boxed_ok, boxed_metrics = evaluate_boxed_answer_probability(
                tokens=rollout.commit["tokens"],
                prompt_length=prompt_len,
                completion_length=completion_len,
                proof=proof,
                tokenizer=self.tokenizer,
            )
            if not boxed_ok:
                logger.info(
                    "reject reason=boxed_answer_tampered hotkey=%s %s",
                    request.miner_hotkey, boxed_metrics,
                )
                return reject(
                    RejectReason.BOXED_ANSWER_TAMPERED,
                    "boxed_answer",
                    sketch_diff_max=sketch_diff_max,
                    lp_dev_max=lp_dev_max,
                    dist_q10_min=dist_q10_min,
                )

            auth_ok, auth_metrics = evaluate_token_authenticity(
                proof,
                tokens=rollout.commit["tokens"],
                prompt_length=prompt_len,
                completion_length=completion_len,
                tokenizer=self.tokenizer,
                exempt_positions=exempt_positions,
            )
            if not auth_ok:
                logger.info(
                    "token_tampered hotkey=%s enforce=%s %s",
                    request.miner_hotkey, TOKEN_AUTH_ENFORCE, auth_metrics,
                )
                if TOKEN_AUTH_ENFORCE:
                    return reject(
                        RejectReason.TOKEN_TAMPERED,
                        "token_authenticity",
                        sketch_diff_max=sketch_diff_max,
                        lp_dev_max=lp_dev_max,
                        dist_q10_min=dist_q10_min,
                    )

            rollout_reward_positive = float(
                getattr(rollout, "reward", 0.0) or 0.0
            ) > 0.0
            all_token_shadow_ok, all_token_shadow_metrics = (
                evaluate_all_token_auth_shadow(
                    proof,
                    tokens=rollout.commit["tokens"],
                    prompt_length=prompt_len,
                    completion_length=completion_len,
                    tokenizer=self.tokenizer,
                    include_findings=private_auth_forensics_enabled,
                    max_findings=private_auth_forensics_max_findings,
                    context_chars=private_auth_forensics_context_chars,
                    exempt_positions=exempt_positions,
                )
            )
            if all_token_shadow_metrics:
                findings = int(
                    all_token_shadow_metrics.get("findings", 0) or 0
                )
                all_token_auth_shadow_findings += findings
                min_prob = all_token_shadow_metrics.get("min_prob")
                if min_prob is not None:
                    p = float(min_prob)
                    if (
                        all_token_auth_shadow_min_prob is None
                        or p < all_token_auth_shadow_min_prob
                    ):
                        all_token_auth_shadow_min_prob = p
                if rollout_reward_positive:
                    all_token_auth_shadow_positive_findings += findings
                    finding_min_prob = all_token_shadow_metrics.get(
                        "finding_min_prob"
                    )
                    if findings > 0 and finding_min_prob is not None:
                        p = float(finding_min_prob)
                        if (
                            all_token_auth_shadow_positive_min_prob is None
                            or p < all_token_auth_shadow_positive_min_prob
                        ):
                            all_token_auth_shadow_positive_min_prob = p
            if not all_token_shadow_ok:
                if private_auth_forensics_enabled:
                    record_all_token_auth_findings(
                        metrics=all_token_shadow_metrics,
                        window_start=self.window_start,
                        env_name=getattr(self.env, "name", ""),
                        miner_hotkey=request.miner_hotkey,
                        prompt_idx=request.prompt_idx,
                        rollout_idx=rollout_idx,
                        rollout_reward=float(
                            getattr(rollout, "reward", 0.0) or 0.0
                        ),
                        reward_positive=rollout_reward_positive,
                        prompt_length=prompt_len,
                        completion_length=completion_len,
                    )
                log_shadow_metrics = dict(all_token_shadow_metrics)
                log_shadow_metrics.pop("finding_details", None)
                logger.info(
                    "all_token_auth_shadow_suspicious hotkey=%s "
                    "reward_positive=%s %s",
                    request.miner_hotkey,
                    rollout_reward_positive,
                    log_shadow_metrics,
                )
                if ALL_TOKEN_AUTH_ENFORCE:
                    return reject(
                        RejectReason.TOKEN_TAMPERED,
                        "all_token_authenticity",
                        sketch_diff_max=sketch_diff_max,
                        lp_dev_max=lp_dev_max,
                        dist_q10_min=dist_q10_min,
                    )

            if getattr(self.env, "name", "") == "opencodeinstruct":
                code_auth_ok, code_auth_metrics = (
                    evaluate_code_semantic_token_authenticity(
                        tokens=rollout.commit["tokens"],
                        prompt_length=prompt_len,
                        completion_length=completion_len,
                        proof=proof,
                        tokenizer=self.tokenizer,
                        include_findings=private_auth_forensics_enabled,
                        max_findings=private_auth_forensics_max_findings,
                        context_chars=private_auth_forensics_context_chars,
                    )
                )
                if code_auth_metrics:
                    findings = int(code_auth_metrics.get("findings", 0) or 0)
                    code_semantic_auth_findings += findings
                    min_prob = code_auth_metrics.get("min_prob")
                    if min_prob is not None:
                        p = float(min_prob)
                        if (
                            code_semantic_auth_min_prob is None
                            or p < code_semantic_auth_min_prob
                        ):
                            code_semantic_auth_min_prob = p
                    if rollout_reward_positive:
                        code_semantic_auth_positive_findings += findings
                        if findings > 0 and min_prob is not None:
                            p = float(min_prob)
                            if (
                                code_semantic_auth_positive_min_prob is None
                                or p < code_semantic_auth_positive_min_prob
                            ):
                                code_semantic_auth_positive_min_prob = p
                if not code_auth_ok:
                    if private_auth_forensics_enabled:
                        rollout_reward = float(
                            getattr(rollout, "reward", 0.0) or 0.0
                        )
                        if (
                            code_semantic_counterfactuals_enabled
                            and rollout_reward_positive
                        ):
                            _enrich_code_semantic_counterfactuals(
                                metrics=code_auth_metrics,
                                env=self.env,
                                problem=problem,
                                completion_text=completion_texts[rollout_idx],
                                rollout_reward=rollout_reward,
                                max_findings=(
                                    code_semantic_counterfactuals_max_findings
                                ),
                            )
                        record_code_semantic_auth_findings(
                            metrics=code_auth_metrics,
                            window_start=self.window_start,
                            env_name=getattr(self.env, "name", ""),
                            miner_hotkey=request.miner_hotkey,
                            prompt_idx=request.prompt_idx,
                            rollout_idx=rollout_idx,
                            rollout_reward=rollout_reward,
                            reward_positive=rollout_reward_positive,
                            prompt_length=prompt_len,
                            completion_length=completion_len,
                        )
                    log_code_metrics = dict(code_auth_metrics)
                    log_code_metrics.pop("finding_details", None)
                    logger.info(
                        "code_semantic_token_suspicious hotkey=%s enforce=%s "
                        "reward_positive=%s %s",
                        request.miner_hotkey,
                        CODE_SEMANTIC_AUTH_ENFORCE,
                        rollout_reward_positive,
                        log_code_metrics,
                    )
                    if CODE_SEMANTIC_AUTH_ENFORCE and rollout_reward_positive:
                        return reject(
                            RejectReason.TOKEN_TAMPERED,
                            "code_semantic_auth",
                            sketch_diff_max=sketch_diff_max,
                            lp_dev_max=lp_dev_max,
                            dist_q10_min=dist_q10_min,
                        )

        # Forced-seed gate. Group verdict = summed counts (catches diffuse
        # deviation); per-rollout verdict catches a single off-stream rollout
        # the group average would dilute. Both shadow (compute + log, never
        # reject) unless FORCED_SEED_ENFORCE is on.
        # Only enforce when the checkpoint hash is pinned: an empty
        # current_checkpoint_hash disables WRONG_CHECKPOINT, so the miner
        # controls checkpoint_hash (a forced-seed derivation input) and could
        # grind it -- don't reject on a stream whose seed inputs aren't bound.
        seed_enforce = FORCED_SEED_ENFORCE and bool(self.current_checkpoint_hash)
        group_would_reject = _forced_seed_verdict(
            grp_stoch, grp_match, True,
        )
        rollout_would_reject = _forced_seed_rollout_reject(
            seed_per_rollout, True,
        )
        group_reject = seed_enforce and group_would_reject
        rollout_reject = seed_enforce and rollout_would_reject
        cdf_enforce = (
            FORCED_SEED_CDF_ENFORCE and bool(self.current_checkpoint_hash)
        )
        cdf_would_reject = grp_seed_hard_mismatch > 0
        cdf_reject = cdf_enforce and cdf_would_reject
        record_forced_seed_shadow(
            hk,
            request.prompt_idx,
            grp_stoch,
            grp_match,
            per_rollout=seed_cdf_per_rollout,
            n_positions=grp_seed_positions,
            n_boundary_match=grp_seed_boundary_match,
            n_hard_mismatch=grp_seed_hard_mismatch,
            n_deterministic_hard_mismatch=(
                grp_seed_deterministic_hard_mismatch
            ),
            n_miss_gt_0_01=grp_seed_miss_gt_0_01,
            n_miss_gt_0_05=grp_seed_miss_gt_0_05,
            n_miss_gt_0_10=grp_seed_miss_gt_0_10,
            max_cdf_miss=grp_seed_max_cdf_miss,
            window_start=self.window_start,
            env_name=getattr(self.env, "name", ""),
            checkpoint_hash=request.checkpoint_hash,
            cdf_boundary_epsilon=FORCED_SEED_CDF_BOUNDARY_EPSILON,
            ratio_group_would_reject=group_would_reject,
            ratio_rollout_would_reject=rollout_would_reject,
            cdf_would_reject=cdf_would_reject,
            cdf_enforced=cdf_enforce,
            runtime_profile=(
                request.runtime_fingerprint.model_dump(mode="json")
                if request.runtime_fingerprint is not None
                else None
            ),
            sketch_diff_max=sketch_diff_max,
        )
        if group_reject or rollout_reject or cdf_reject:
            if cdf_reject:
                scope = "cdf_hard_mismatch"
            elif group_reject:
                scope = "group"
            else:
                scope = "rollout"
            logger.info(
                "seed_mismatch hotkey=%s stoch=%d match=%d scope=%s "
                "cdf_hard=%d cdf_max_miss=%.8f",
                hk,
                grp_stoch,
                grp_match,
                scope,
                grp_seed_hard_mismatch,
                grp_seed_max_cdf_miss,
            )
            return reject(RejectReason.SEED_MISMATCH, "forced_seed")

        # Reward-shape metrics are still computed (they feed the softer
        # training-quarantine signal + archive telemetry) but no longer
        # REJECT a submission: the ordered-prefix heuristic was trivially
        # bypassed by reordering rollouts (10101010) and varying loser
        # lengths, while false-positive-rejecting honest miners whose losers
        # happened to share a length. The real defense is structural
        # (zone/emission economics + rollout token authenticity), not this
        # shape heuristic. See RejectReason.REWARD_SHAPE_SUSPICIOUS (kept for
        # historical archive deserialization only).
        reward_shape = detect_reward_shape_manipulation(
            rewards,
            completion_lengths,
            truncated_flags,
        )

        # All checks passed.
        new_sub = ValidSubmission(
            hotkey=request.miner_hotkey,
            prompt_idx=request.prompt_idx,
            merkle_root_bytes=pending.merkle_root,
            selection_digest_bytes=pending.selection_digest,
            sigma=sigma,
            rollouts=list(request.rollouts),
            completion_texts=completion_texts,
            arrived_at=pending.arrived_at,
            sketch_diff_max=sketch_diff_max,
            lp_dev_max=lp_dev_max,
            dist_q10_min=dist_q10_min,
            all_token_auth_shadow_findings=all_token_auth_shadow_findings,
            all_token_auth_shadow_min_prob=all_token_auth_shadow_min_prob,
            all_token_auth_shadow_positive_findings=(
                all_token_auth_shadow_positive_findings
            ),
            all_token_auth_shadow_positive_min_prob=(
                all_token_auth_shadow_positive_min_prob
            ),
            code_semantic_auth_findings=code_semantic_auth_findings,
            code_semantic_auth_min_prob=code_semantic_auth_min_prob,
            code_semantic_auth_positive_findings=(
                code_semantic_auth_positive_findings
            ),
            code_semantic_auth_positive_min_prob=(
                code_semantic_auth_positive_min_prob
            ),
            claimed_checkpoint_hash=request.checkpoint_hash,
            rollout_hashes=rollout_hashes,
            drand_round=request.drand_round,
            arrival_ts=telemetry.t_arrival if telemetry else None,
            decision_ts=pending.decision_ts,
            submitted_drand_round=request.drand_round,
            arrival_drand_round=(
                telemetry.arrival_drand_round if telemetry else None
            ),
            drand_delta=telemetry.drand_delta if telemetry else None,
            seal_trigger_round=None,
            prompt_hash_lead=telemetry.prompt_hash_lead if telemetry else None,
            reward_vector=reward_shape.reward_vector,
            truncated_count=truncated_count,
            reward_shape=reward_shape.to_log_dict(),
            ingress_observability=(
                telemetry.archive_fields() if telemetry else {}
            ),
        )
        # The caller decides whether this is an auction winner or an immediate
        # legacy admission.
        return new_sub

    def _reject(
        self,
        reason: RejectReason,
        *,
        hotkey: str | None = None,
        prompt_idx: int | None = None,
        sketch_diff_max: int | None = None,
        lp_dev_max: float | None = None,
        dist_q10_min: float | None = None,
        telemetry: SubmitTelemetry | None = None,
        reject_stage: str | None = None,
        decision_ts: float | None = None,
    ) -> BatchSubmissionResponse:
        self.reject_counts[reason.value] = self.reject_counts.get(reason.value, 0) + 1

        if (
            hotkey is not None
            and reject_stage in _PROOF_FAILURE_DEBT_STAGES
        ):
            with self._proof_admission_lock:
                self._expensive_proof_failures_by_hotkey[hotkey] = (
                    self._expensive_proof_failures_by_hotkey.get(hotkey, 0) + 1
                )

        if hotkey is not None and prompt_idx is not None:
            already = sum(
                1 for r in self.rejected_submissions if r.hotkey == hotkey
            )
            if already < REJECTED_LIST_CAP_PER_HOTKEY:
                # Anti-tuning: never surface the GRAIL sketch diff to miners.
                # All other reasons get the diagnostics computed up to the
                # rejection point. A proof-stage reject passes the admission
                # instant as ``decision_ts``; cheap rejects stamp it now.
                if decision_ts is None:
                    decision_ts = self._wall_clock()
                if reason is RejectReason.GRAIL_FAIL:
                    sketch_diff_max = None
                    telemetry = None
                    reject_stage = None
                    decision_ts = None
                self.rejected_submissions.append(
                    RejectedSubmission(
                        hotkey=hotkey,
                        prompt_idx=prompt_idx,
                        reason=reason.value,
                        sketch_diff_max=sketch_diff_max,
                        lp_dev_max=lp_dev_max,
                        dist_q10_min=dist_q10_min,
                        arrival_ts=telemetry.t_arrival if telemetry else None,
                        decision_ts=decision_ts,
                        submitted_drand_round=(
                            telemetry.submitted_drand_round if telemetry else None
                        ),
                        arrival_drand_round=(
                            telemetry.arrival_drand_round if telemetry else None
                        ),
                        drand_delta=telemetry.drand_delta if telemetry else None,
                        seal_trigger_round=(
                            telemetry.seal_trigger_round if telemetry else self._seal_trigger_round
                        ),
                        prompt_hash_lead=(
                            telemetry.prompt_hash_lead if telemetry else None
                        ),
                        reject_stage=reject_stage,
                        ingress_observability=(
                            telemetry.archive_fields() if telemetry else None
                        ),
                    )
                )
        return BatchSubmissionResponse(accepted=False, reason=reason)

    # ----------------------------- accessors -----------------------------

    def valid_submissions(self) -> list[ValidSubmission]:
        with self._lock:
            return list(self._valid)

    def pending_submissions(self) -> list[PendingSubmission]:
        """Graded, scored, not yet proven. The auction ranks these at seal."""
        with self._lock:
            return list(self._pending)

    def _prove_ranked(self, pool: float = 1.0) -> list[ValidSubmission]:
        """Prove candidates in difficulty-score order until ``B_BATCH`` distinct
        prompts pass. Never prove a loser.

        Ranking is by ``-value`` and then an operator-bound hash. The hash
        deliberately excludes hotkey and miner payload fields: one operator
        cannot improve an equal-score tie by registering more hotkeys or
        grinding harmless metadata. A post-deadline drand beacon salts the tie
        when available; window randomness is the deterministic liveness
        fallback. The miner-submitted drand round is a freshness observation,
        not an economic ordering key: otherwise any non-zero backward tolerance
        lets a miner antedate an equal-score candidate for a deterministic
        advantage.

        Same-prompt resolution happens HERE (spec §2.2): the first submission for
        a prompt that PASSES the proof claims the slot; later submissions for a
        claimed prompt are skipped. A fabricated squatter fails the proof, so it
        never locks a prompt — the honest submission behind it is promoted.

        Bounds (spec §2.3): per-hotkey and per-operator failure skips cap a
        single identity even when one coldkey owns many hotkeys, and a global
        attempt ceiling
        (``MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW``, the graded-pool bound) caps a
        multi-hotkey flood so proving can never exceed the graded pool. On
        exhaustion we log, stop, and advance with the shortfall (fewer than
        ``B_BATCH`` in ``_valid``); the training path already burns unpaid slots.

        Runs OUTSIDE ``_lock``: ``_verify_expensive`` is 5-25 s of GPU per
        candidate, mutates reject state, and is not thread-safe, so the loop is
        strictly serial.
        """
        with self._lock:
            pending = list(self._pending)
        scored = [
            (p, difficulty_score(p.rewards, delta=DIFFICULTY_AUCTION_DELTA))
            for p in pending
        ]
        operator_by_id: dict[int, str | None] = {}
        tiebreak_by_id: dict[int, bytes] = {}
        rank_salt = self.seal_randomness or self.randomness
        rank_entropy_source = (
            "seal_drand" if self.seal_randomness else "window_randomness_fallback"
        )
        for pending_submission, _score in scored:
            operator = self._operator_by_hotkey.get(pending_submission.hotkey)
            if operator is None and not self._operator_mapping_enforced:
                operator = pending_submission.hotkey
            operator_by_id[id(pending_submission)] = operator
            h = hashlib.sha256()
            h.update(b"reliquary/auction-operator-tiebreak/v1\x00")
            salt_bytes = rank_salt.encode("utf-8")
            h.update(len(salt_bytes).to_bytes(4, "big"))
            h.update(salt_bytes)
            operator_bytes = (operator or "").encode("utf-8")
            h.update(len(operator_bytes).to_bytes(4, "big"))
            h.update(operator_bytes)
            h.update(
                int(pending_submission.prompt_idx).to_bytes(
                    8, "big", signed=False
                )
            )
            tiebreak_by_id[id(pending_submission)] = h.digest()
        ranked = sorted(
            scored,
            key=lambda item: (
                -item[1].value,
                tiebreak_by_id[id(item[0])],
            ),
        )

        attempts = 0
        proven: list[ValidSubmission] = []
        claimed: set[int] = set()
        attempted_ids: set[int] = set()
        self.auction_operator_unmapped_skips = 0
        self.auction_operator_proof_debt_skips = 0
        self.difficulty_auction_metadata_by_id = {}
        candidate_rows: list[dict[str, Any]] = []
        for rank, (pending_submission, score) in enumerate(ranked, start=1):
            operator = operator_by_id[id(pending_submission)]
            row = {
                "hotkey": pending_submission.hotkey,
                "prompt_idx": pending_submission.prompt_idx,
                "selection_digest": pending_submission.selection_digest.hex(),
                "drand_round": pending_submission.drand_round,
                "value": score.value,
                "mean_reward": score.mean_reward,
                "reward_std": score.reward_std,
                "reward_count": score.reward_count,
                "operator_id": operator,
                "operator_tiebreak": tiebreak_by_id[
                    id(pending_submission)
                ].hex(),
                "rank_entropy_source": rank_entropy_source,
                "rank": rank,
                "proof_attempted": False,
                "proof_passed": None,
                "selected": False,
                "status": "ranked",
                **(
                    pending_submission.telemetry.archive_fields()
                    if pending_submission.telemetry is not None
                    else {}
                ),
            }
            candidate_rows.append(row)
            self.difficulty_auction_metadata_by_id[id(pending_submission)] = row

        self._proof_wall_started_at = self._time_fn()
        self.proof_wall_exhausted = False
        stop_reason: str | None = None

        for (p, _score), row in zip(ranked, candidate_rows):
            if len(proven) >= B_BATCH:
                stop_reason = "batch_filled"
                break
            if p.prompt_idx in claimed:
                row["status"] = "same_prompt_superseded"
                continue          # a higher-ranked submission already won it
            if self._cooldown.is_in_cooldown(p.prompt_idx, self.window_start):
                row["status"] = "cooldown"
                continue
            operator = row["operator_id"]
            if operator is None:
                row["status"] = "operator_unmapped"
                self.auction_operator_unmapped_skips += 1
                continue
            if (
                self.operator_proof_failure_debt(operator)
                >= MAX_EXPENSIVE_PROOF_FAILURES_PER_OPERATOR_PER_WINDOW
            ):
                row["status"] = "operator_proof_debt"
                self.auction_operator_proof_debt_skips += 1
                continue
            # Global proof budget: proving cannot exceed the graded-pool ceiling
            # (v2 §2.3). This bounds a multi-hotkey fabricated flood that the
            # per-hotkey skip below cannot, since each fake hotkey pays only one
            # registration. On exhaustion we stop and advance short.
            if attempts >= MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW:
                logger.warning(
                    "proof budget exhausted window=%d attempts=%d proven=%d "
                    "pending=%d — advancing with shortfall",
                    self.window_start, attempts, len(proven), len(pending),
                )
                stop_reason = "attempt_budget"
                break
            elapsed = self._time_fn() - self._proof_wall_started_at
            if elapsed >= MAX_PROOF_WALL_SECONDS:
                self.proof_wall_exhausted = True
                stop_reason = "wall_budget"
                logger.warning(
                    "proof wall budget exhausted window=%d elapsed_s=%.2f "
                    "attempts=%d proven=%d pending=%d — advancing with shortfall",
                    self.window_start,
                    elapsed,
                    attempts,
                    len(proven),
                    len(pending),
                )
                break
            # Per-hotkey griefer bound. A fabricated group ranks at the top by
            # construction and fails the proof; each hotkey is skipped after its
            # failure cap so honest fill below the fakes always proceeds.
            if (
                self.proof_failure_debt(p.hotkey)
                >= MAX_EXPENSIVE_PROOF_FAILURES_PER_HOTKEY_PER_WINDOW
            ):
                row["status"] = "hotkey_proof_debt"
                continue
            attempts += 1
            attempted_ids.add(id(p))
            row["proof_attempted"] = True
            row["status"] = "proof_started"
            sub = self._verify_expensive(p)
            if sub is None:
                self._expensive_proof_failures_by_operator[operator] = (
                    self._expensive_proof_failures_by_operator.get(
                        operator, 0
                    )
                    + 1
                )
                row["proof_passed"] = False
                row["status"] = "proof_failed"
                continue          # rejected; promote the next-ranked for prompt
            row["proof_passed"] = True
            row["selected"] = True
            row["status"] = "selected"
            proven.append(sub)
            claimed.add(p.prompt_idx)
            self.difficulty_auction_metadata_by_id[id(sub)] = row

        with self._lock:
            self._valid = proven
            self.valid_count = len(proven)
        self.proof_attempts = attempts
        self.proof_wall_elapsed_seconds = max(
            0.0,
            self._time_fn() - self._proof_wall_started_at,
        )
        if stop_reason is not None:
            for row in candidate_rows:
                if row["status"] == "ranked":
                    row["status"] = (
                        "not_needed" if stop_reason == "batch_filled"
                        else f"unobserved_{stop_reason}"
                    )
        self.auction_candidates = candidate_rows
        # Read by _prove_forensic_sample to find the pool this window never
        # looked at.
        self._attempted_pending_ids = attempted_ids
        return proven

    def _prove_forensic_sample(self) -> list[ForensicSampleResult]:
        """Prove ``FORENSIC_SAMPLE_PER_WINDOW`` non-winners for telemetry only.

        Deferring the proof to seal means the auth gates in ``_verify_expensive``
        only ever run on the ranked winners; sampling a few losers keeps the
        pre-generation / token-tamper detectors alive. Results never enter
        ``_valid`` and are never paid.

        The sample is keyed on ``self.seal_randomness`` — a drand beacon fetched
        AFTER the collection deadline — so a miner cannot grind its merkle_root to
        learn whether it will be watched (the entropy did not exist at submission
        time). This is telemetry only and never feeds weights, so it needs no
        cross-validator consensus: each validator may watch a different sample.
        Empty seal_randomness (mock / no-drand) disables sampling.
        """
        if FORENSIC_SAMPLE_PER_WINDOW <= 0 or not self.seal_randomness:
            self.forensic_sample = []
            return self.forensic_sample
        with self._lock:
            remainder = [
                p for p in self._pending
                if id(p) not in self._attempted_pending_ids
            ]
        seed = self.seal_randomness.encode()
        remainder.sort(
            key=lambda p: hashlib.sha256(
                seed + p.hotkey.encode()
                + int(p.prompt_idx).to_bytes(8, "big") + p.merkle_root
            ).digest()
        )
        sample = remainder[:FORENSIC_SAMPLE_PER_WINDOW]
        results: list[ForensicSampleResult] = []
        self.forensic_proof_attempts = 0
        self.forensic_proof_errors_by_type = {}
        for p in sample:
            if self.proof_attempts >= MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW:
                break
            if self._proof_wall_started_at is None:
                break
            elapsed = self._time_fn() - self._proof_wall_started_at
            if elapsed >= MAX_PROOF_WALL_SECONDS:
                self.proof_wall_exhausted = True
                break
            error_type: str | None = None
            try:
                passed: bool | None = self._verify_expensive(p) is not None
            except Exception as exc:
                # The sample is observational only. A malformed outlier or
                # exhausted GPU must not discard already-proven winners or
                # prevent the window from reaching training.
                passed = None
                error_type = type(exc).__name__
                self.forensic_proof_errors_by_type[error_type] = (
                    self.forensic_proof_errors_by_type.get(error_type, 0) + 1
                )
                logger.exception(
                    "forensic proof failed open window=%s prompt=%s "
                    "hotkey=%s error=%s",
                    self.window_start,
                    p.prompt_idx,
                    p.hotkey[:12],
                    error_type,
                )
            if error_type in {"OutOfMemoryError", "CUDAOutOfMemoryError"}:
                # The exception scope has ended, so traceback-held tensors can
                # be collected before returning allocator cache to CUDA.
                try:
                    import gc
                    import torch

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    logger.debug("CUDA OOM cleanup failed", exc_info=True)
            self.proof_attempts += 1
            self.forensic_proof_attempts += 1
            self._attempted_pending_ids.add(id(p))
            row = self.difficulty_auction_metadata_by_id.get(id(p))
            if row is not None:
                row["forensic_sampled"] = True
                row["forensic_passed"] = passed
                row["forensic_error_type"] = error_type
            results.append(
                ForensicSampleResult(
                    hotkey=p.hotkey,
                    prompt_idx=p.prompt_idx,
                    passed=passed,
                    error_type=error_type,
                )
            )
        self.proof_wall_elapsed_seconds = max(
            self.proof_wall_elapsed_seconds,
            self._time_fn() - self._proof_wall_started_at,
        )
        self.forensic_sample = results
        return results

    def _compute_difficulty_auction_shadow(self) -> None:
        """Compute an inert auction counterfactual over ``self._valid``.

        Production has already populated ``selection_metadata_by_id`` when
        this runs. Any shadow failure is contained and surfaced in telemetry;
        an observational experiment cannot break the live protocol.
        """
        env_name = str(getattr(self.env, "name", ""))
        base = {
            "schema_version": 1,
            "mode": "observation_only",
            "environment": env_name,
            "population": "fully_validated_before_current_seal",
            "population_limitations": [
                "excludes_pre_validation_rejects",
                "excludes_batch_filled_rejects",
            ],
            "production_changed": False,
            "delta": DIFFICULTY_AUCTION_DELTA,
            "batch_size": B_BATCH,
            "validated_candidates": len(self._valid),
            "max_candidates": DIFFICULTY_AUCTION_SHADOW_MAX_CANDIDATES,
            "operator_cap_requested": (
                DIFFICULTY_AUCTION_SHADOW_MAX_SLOTS_PER_OPERATOR
            ),
            "operator_mapping_snapshot_count": len(self._operator_by_hotkey),
        }
        self.difficulty_auction_metadata_by_id = {}
        if not DIFFICULTY_AUCTION_SHADOW_ENABLED:
            self.difficulty_auction_shadow = {**base, "status": "disabled"}
            return
        if env_name not in DIFFICULTY_AUCTION_SHADOW_ENVIRONMENTS:
            self.difficulty_auction_shadow = {
                **base,
                "status": "out_of_scope_environment",
            }
            return
        if len(self._valid) > DIFFICULTY_AUCTION_SHADOW_MAX_CANDIDATES:
            self.difficulty_auction_shadow = {
                **base,
                "status": "skipped_candidate_limit",
            }
            return

        started_at = time.perf_counter()
        try:
            shadow_pool = tuple(
                ShadowSubmission(
                    source_id=id(submission),
                    hotkey=str(submission.hotkey),
                    prompt_idx=int(submission.prompt_idx),
                    drand_round=int(submission.drand_round),
                    merkle_root=bytes(submission.merkle_root),
                    selection_digest=bytes(submission.selection_digest),
                    rewards=tuple(
                        float(rollout.reward)
                        for rollout in submission.rollouts
                    ),
                    in_cooldown=self._cooldown.is_in_cooldown(
                        submission.prompt_idx, self.window_start
                    ),
                )
                for submission in self._valid
            )
            result = select_shadow_auction(
                shadow_pool,
                b=B_BATCH,
                delta=DIFFICULTY_AUCTION_DELTA,
                max_slots_per_operator=(
                    DIFFICULTY_AUCTION_SHADOW_MAX_SLOTS_PER_OPERATOR
                ),
                operator_of=self._operator_by_hotkey.get,
            )
            production_selected_ids = {
                id(submission)
                for submission in self._valid
                if self.selection_metadata_by_id.get(id(submission), {}).get(
                    "selected_for_batch", False
                )
            }
            production_rewarded_ids = {
                id(submission)
                for submission in self._valid
                if self.selection_metadata_by_id.get(id(submission), {}).get(
                    "rewarded", False
                )
            }
            shadow_selected_ids = {
                submission.source_id for submission in result.selected
            }

            candidate_rows = []
            for candidate in result.candidates:
                submission = candidate.submission
                row = {
                    "hotkey": submission.hotkey,
                    "prompt_idx": submission.prompt_idx,
                    "selection_digest": submission.selection_digest.hex(),
                    "drand_round": submission.drand_round,
                    "value": candidate.score.value,
                    "mean_reward": candidate.score.mean_reward,
                    "reward_std": candidate.score.reward_std,
                    "reward_count": candidate.score.reward_count,
                    "operator_id": candidate.operator_id,
                    "eligible": candidate.eligible,
                    "rank": candidate.rank,
                    "shadow_selected": candidate.selected,
                    "production_selected": (
                        submission.source_id in production_selected_ids
                    ),
                    "production_rewarded": (
                        submission.source_id in production_rewarded_ids
                    ),
                }
                self.difficulty_auction_metadata_by_id[
                    submission.source_id
                ] = row
                candidate_rows.append(row)

            overlap = production_selected_ids & shadow_selected_ids
            union = production_selected_ids | shadow_selected_ids

            def _mean_reward(submission_ids: set[int]) -> float | None:
                values = []
                for submission in self._valid:
                    if id(submission) not in submission_ids:
                        continue
                    row = self.difficulty_auction_metadata_by_id.get(
                        id(submission)
                    )
                    if row is not None:
                        values.append(float(row["mean_reward"]))
                return sum(values) / len(values) if values else None

            self.difficulty_auction_shadow = {
                **base,
                "status": "computed",
                "computation_ms": (
                    time.perf_counter() - started_at
                ) * 1000.0,
                "eligible_candidates": result.eligible_count,
                "distinct_eligible_prompts": result.distinct_prompt_count,
                "production_selected_count": len(production_selected_ids),
                "production_rewarded_count": len(production_rewarded_ids),
                "shadow_selected_count": len(shadow_selected_ids),
                "selection_overlap_count": len(overlap),
                "selection_jaccard": (
                    len(overlap) / len(union) if union else 1.0
                ),
                "production_selected_mean_reward": _mean_reward(
                    production_selected_ids
                ),
                "shadow_selected_mean_reward": _mean_reward(
                    shadow_selected_ids
                ),
                "operator_cap_requested": result.operator_cap_requested,
                "operator_cap_applied": result.operator_cap_applied,
                "operator_mapping_complete": result.operator_mapping_complete,
                "candidates": candidate_rows,
            }
        except Exception as exc:
            logger.exception(
                "difficulty auction shadow failed window=%d env=%s",
                self.window_start,
                env_name,
            )
            self.difficulty_auction_shadow = {
                **base,
                "status": "error",
                "error_type": type(exc).__name__,
                "computation_ms": (
                    time.perf_counter() - started_at
                ) * 1000.0,
            }

    def seal_batch(
        self, pool: float = 1.0
    ) -> tuple[list[ValidSubmission], dict[str, float]]:
        """Finalize selection and release every retained payload reservation."""
        try:
            return self._seal_batch_inner(pool)
        finally:
            self._release_retained_payloads()

    def _seal_batch_inner(
        self, pool: float = 1.0
    ) -> tuple[list[ValidSubmission], dict[str, float]]:
        """Pick the training batch and compute the reward distribution.

        Proving happens first: ``_prove_ranked`` ranks the pending pool by
        difficulty and fills ``self._valid`` with the proven winners, which
        everything below reads. ``_prove_forensic_sample`` then proves a small
        sample of non-winners for telemetry only — it never touches ``_valid``.
        Returns (training_batch, rewards_by_hotkey). Cooldown and hash-set
        bookkeeping is applied to every winning prompt — not just the one
        submission picked for training — because all of them earn emission
        and were therefore "used" by this window.
        """
        if self.difficulty_auction_enabled:
            self._prove_ranked(pool)
            self._prove_forensic_sample()
        with self._lock:
            self.selection_metadata_by_id = explain_batch_selection(
                submissions=self._valid,
                b=B_BATCH,
                cooldown_map=self._cooldown,
                current_window=self.window_start,
                pool=pool,
            )
            batch, rewards = select_batch_and_distribute(
                submissions=self._valid,
                b=B_BATCH,
                cooldown_map=self._cooldown,
                current_window=self.window_start,
                pool=pool,
            )
            if self.difficulty_auction_enabled:
                self.difficulty_auction_shadow = {
                    "schema_version": 2,
                    "status": "armed",
                    "mode": "production",
                    "environment": str(getattr(self.env, "name", "")),
                    "production_changed": True,
                    "delta": DIFFICULTY_AUCTION_DELTA,
                    "pending_candidates": len(self._pending),
                    "proof_attempts": self.proof_attempts,
                    "forensic_proof_attempts": self.forensic_proof_attempts,
                    "forensic_proof_errors_by_type": dict(
                        self.forensic_proof_errors_by_type
                    ),
                    "proof_attempt_limit": (
                        MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW
                    ),
                    "proof_wall_seconds": self.proof_wall_elapsed_seconds,
                    "proof_wall_limit_seconds": MAX_PROOF_WALL_SECONDS,
                    "proof_wall_exhausted": self.proof_wall_exhausted,
                    "proven_winners": len(self._valid),
                    "operator_proof_failure_cap": (
                        MAX_EXPENSIVE_PROOF_FAILURES_PER_OPERATOR_PER_WINDOW
                    ),
                    "operator_proof_failure_debt": dict(
                        self._expensive_proof_failures_by_operator
                    ),
                    "operator_proof_debt_skips": (
                        self.auction_operator_proof_debt_skips
                    ),
                    "operator_unmapped_skips": (
                        self.auction_operator_unmapped_skips
                    ),
                    "operator_mapping_complete": all(
                        row.get("operator_id") is not None
                        for row in self.auction_candidates
                    ),
                    "operator_mapping_enforced": (
                        self._operator_mapping_enforced
                    ),
                    "retained_payload_bytes_at_seal": (
                        self.retained_payload_bytes
                    ),
                    "candidates": self.auction_candidates,
                }
            else:
                self._compute_difficulty_auction_shadow()
            rewarded_submissions: list[ValidSubmission] = []
            rewarded_but_not_selected: dict[str, int] = {}
            for sub in self._valid:
                meta = self.selection_metadata_by_id.get(id(sub), {})
                if not meta.get("rewarded", False):
                    continue
                rewarded_submissions.append(sub)
                if not meta.get("selected_for_batch", False):
                    rewarded_but_not_selected[sub.hotkey] = (
                        rewarded_but_not_selected.get(sub.hotkey, 0) + 1
                    )
            self.rewarded_but_not_selected_by_hotkey = rewarded_but_not_selected

            rewarded_prompts = {sub.prompt_idx for sub in rewarded_submissions}
            for p in rewarded_prompts:
                self._cooldown.record_batched(p, self.window_start)
            if self._hash_set is not None:
                for sub in rewarded_submissions:
                    for h in sub.rollout_hashes:
                        self._hash_set.add(h, self.window_start)
            if self._hash_set is not None:
                self._hash_set.prune(self.window_start)
            return batch, rewards

    def get_state(self) -> GrpoBatchState:
        with self._lock:
            return GrpoBatchState(
                state=WindowState.OPEN,
                window_n=self.window_start,
                anchor_block=self.window_start,
                cooldown_prompts=sorted(
                    self._cooldown.current_cooldown_set(self.window_start)
                ),
                # The field name is a strict wire contract. Auction mode reports
                # admitted candidates; legacy mode reports proven submissions.
                valid_submissions=(
                    len(self._pending)
                    if self.difficulty_auction_enabled
                    else len(self._valid)
                ),
                checkpoint_n=0,
            )
