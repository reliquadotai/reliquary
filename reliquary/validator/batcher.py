"""GrpoWindowBatcher — orchestrator for the free-prompt GRPO market.

Holds a flat list of validated submissions per window + a reference to the
validator's shared ``CooldownMap``.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import ValidationError

from reliquary.constants import (
    BATCH_PROMPT_COOLDOWN_WINDOWS,
    B_BATCH,
    BOOTSTRAP_MAX_TRUNCATED_PER_SUBMISSION,
    M_ROLLOUTS,
    MAX_EXPENSIVE_PROOF_FAILURES_PER_HOTKEY_PER_WINDOW,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    MIN_EOS_PROBABILITY,
    MAX_POST_TRIGGER_PROOF_CANDIDATES,
    MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW,
    MAX_SEAL_QUEUE_DRAIN_SECONDS,
    MAX_SUBMISSIONS_PER_PROMPT,
    MAX_TRUNCATED_PER_SUBMISSION,
    PROMPT_RANGE_SIZE,
    PROMPT_RANGE_ENFORCE_FROM_WINDOW,
    REJECTED_LIST_CAP_PER_HOTKEY,
    CODE_SEMANTIC_AUTH_ENFORCE,
    TOKEN_AUTH_ENFORCE,
    ALL_TOKEN_AUTH_ENFORCE,
    FORCED_SEED_CDF_BOUNDARY_EPSILON,
    FORCED_SEED_CDF_ENFORCE,
    FORCED_SEED_ENFORCE,
    LEGACY_MERKLE_ROOT_ENFORCE,
)
from reliquary.environment.base import Environment
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
from reliquary.validator.selection_digest import compute_rollouts_selection_digest
from reliquary.validator.verifier import (
    evaluate_all_token_auth_shadow,
    evaluate_code_semantic_token_authenticity,
    evaluate_boxed_answer_probability,
    evaluate_token_authenticity,
    evaluate_token_distribution,
    has_eos_padding,
    is_cap_truncation,
    is_natural_bft_cap_candidate,
    is_in_zone,
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


# v2.3: batch selection is drand-anchored at seal time (see
# ``batch_selection.py``). Multiple miners may submit on the same
# ``prompt_idx`` within a window, capped at ``MAX_SUBMISSIONS_PER_PROMPT``
# per prompt. Emission is split uniformly across all GRAIL-validated
# submissions whose prompt lands in the winning set, so sybiling the same
# prompt is strictly neutral.


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
    ) -> None:
        import time
        from reliquary.constants import DRAND_ROUND_BACKWARD_TOLERANCE

        self.window_start = window_start
        self.env = env
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
        # the batcher accepts. Defaults to ``DRAND_ROUND_BACKWARD_TOLERANCE``
        # from constants (1 round = 3 s grace) so prod stays consistent;
        # tests that want to pin zero-tolerance v2.3 behaviour can pass
        # ``drand_round_backward_tolerance=0`` explicitly.
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
            tuple[str, bytes], BatchSubmissionRequest
        ] = {}
        self._logical_group_duplicate_rejects = 0
        self._valid: list[ValidSubmission] = []
        # v2.3: per-prompt bucket. Multiple miners may submit on the same
        # ``prompt_idx`` up to ``MAX_SUBMISSIONS_PER_PROMPT``. Tracked
        # alongside the flat ``_valid`` list because seal_batch needs the
        # grouping but accept-time logic only needs the count.
        self._submissions_per_prompt: dict[int, list[ValidSubmission]] = {}
        self.randomness: str = ""
        # Per-window eligible prompt slice [lo, hi). None = no restriction
        # (randomness not yet known, or window is before the enforcement
        # cutover). Set by set_prompt_range() once randomness is assigned.
        self.prompt_range: tuple[int, int] | None = None
        # v2.3: post-seal emission distribution. Populated by seal_batch and
        # consumed by _archive_window so the EMA / weight-setter can credit
        # all GRAIL-validated submissions whose prompt landed in the
        # winning set, not just the one picked for the training step.
        self.rewards_by_hotkey: dict[str, float] = {}
        # Post-seal health metric: accepted submissions that earned emission
        # via boundary sharing but did not enter the training batch.
        self.rewarded_but_not_selected_by_hotkey: dict[str, int] = {}
        # Accumulated reject reasons this window (RejectReason.value → count).
        # Persisted in the R2 archive so miners can see which filter is
        # rejecting the most submissions in any given round.
        self.reject_counts: dict[str, int] = {}
        self.selection_metadata_by_id: dict[int, dict[str, Any]] = {}

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
            int, tuple[BatchSubmissionRequest, str, bool]
        ] = {}
        self._inflight_proof_reservations: dict[
            int, tuple[BatchSubmissionRequest, str, bool]
        ] = {}
        self._pending_post_trigger_proof_reservations = 0
        self._post_trigger_proof_admission_count = 0
        self._expensive_proof_failures_by_hotkey: dict[str, int] = {}
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

    def is_sealed(self) -> bool:
        """True once B distinct non-cooldown valid submissions have been
        accepted. Thread-safe and loop-independent (reads the underlying
        ``threading.Event``, never touches the lazy ``asyncio.Event``).

        After this returns True, ``select_batch`` will pick the first
        ``B_BATCH`` by ``arrived_at`` — any further submission would have
        a later ``arrived_at`` and therefore cannot displace one of the
        already-selected entries. Verifying it costs ~5–25 s of GRAIL
        forward pass and produces zero protocol benefit. Callers (the
        HTTP /submit handler and the submit worker) use this to short-
        circuit further work for the current window.
        """
        return self._seal_flag.is_set()

    def force_seal(self, reason: str) -> None:
        """Force this window to seal without B distinct valid submissions.

        Used only as a liveness breaker after the bounded proof-admission
        queue has fully drained and no further expensive submissions can be
        admitted. The downstream training path already skips partial batches;
        this just avoids waiting for the long window timeout.
        """
        if self._seal_flag.is_set():
            return
        self.force_seal_reason = reason
        self._seal_flag.set()
        if self._seal_event is not None:
            self._seal_event.set()

    def prompt_submission_count(self, prompt_idx: int) -> int:
        """Number of GRAIL-validated submissions already in the per-prompt
        bucket for ``prompt_idx``. Used by the HTTP /submit handler to
        short-circuit ``PROMPT_FULL`` rejects without queueing.

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
        """Atomically reserve one hotkey's logical group for this window."""
        digest = compute_logical_group_hash(request)
        key = (request.miner_hotkey, digest)
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
        with self._proof_admission_lock:
            reservation_id = id(request)
            if (
                reservation_id in self._pending_proof_reservations
                or reservation_id in self._inflight_proof_reservations
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
            if reservation is None:
                # Compatibility for direct/legacy worker injection. Production
                # requests always reserve in the HTTP path first.
                if (
                    self._proof_grading_attempts
                    >= MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW
                ):
                    return False, "proof_grading_attempts_full"
                reservation = (request, request.miner_hotkey, False)

            _, hotkey, is_post_trigger = reservation
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
                return False, "proof_failure_debt_hotkey"

            self._proof_grading_attempts += 1
            if is_post_trigger:
                self._post_trigger_proof_admission_count += 1
            self._inflight_proof_reservations[reservation_id] = reservation
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
            return True

    def finish_proof_admission(self, request: BatchSubmissionRequest) -> None:
        """Release the in-flight marker; started-attempt debt remains."""
        with self._proof_admission_lock:
            self._inflight_proof_reservations.pop(id(request), None)

    def _note_grail_candidate(self) -> None:
        """Count a zone-valid submission entering the GRAIL/GPU proof path.

        Telemetry only — the window is already bounded by the drand-anchored
        seal (the B-th distinct prompt records the trigger round and later
        rounds are dropped), the never-refunded grading-attempts ceiling, the
        per-hotkey submission cap, the post-trigger straggler cap and the seal
        drain timeout. The old per-window GRAIL candidate budget was a
        pre-seal-drand relic: back when the 8-distinct seal did not fire it was
        the only bound on GRAIL work within a window. It only starved honest
        late arrivals whenever earlier candidates failed a post-reservation
        gate (e.g. forced-seed) without refund, so it no longer rejects.
        """
        with self._proof_admission_lock:
            self._proof_admission_count += 1

    # ----------------------------- ingestion -----------------------------

    def accept_submission(
        self,
        request: BatchSubmissionRequest,
        *,
        telemetry: SubmitTelemetry | None = None,
    ) -> BatchSubmissionResponse:
        """Run the full verification pipeline; append to ``_valid`` on success.

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
            return self._accept_locked(request, telemetry=telemetry)

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

        Backward direction allows up to
        ``self.drand_round_backward_tolerance`` rounds (default 1 = 3 s).
        This absorbs:
          * HTTP RTT + queue/scheduling jitter that pushes a POST across
            a drand boundary mid-flight (miner fires at t=2.9 s of round
            R, validator receives at t=3.0 s of round R+1)
          * Small wall-clock skew between miner and validator. v2.3's
            original zero-tolerance was correct in spec but turned every
            inter-round POST into a STALE_ROUND in prod.
        The security cost is bounded: an attacker can antedate by at most
        ``tolerance`` rounds (3 s × tolerance) of chronological priority,
        which is uniform across honest and malicious miners alike.
        Combined with arrival-time stamping, the backward tolerance can
        now be tightened safely (the wide default existed to absorb loop
        stalls — that's no longer needed on the HTTP path).

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

        # v2.3 seal extension: once the trigger drand round is recorded
        # (B-th distinct prompt landed in round R), submissions from a
        # LATER drand round arrive too late — drop with BATCH_FILLED.
        # CRITICAL: we do NOT fire the seal here, even though we now
        # know a later round has shown up. The queue may still contain
        # other trigger-round submissions waiting on GRAIL; firing the
        # seal now would make the worker's ``is_sealed`` check drop
        # them. The seal fires only when the delayed coroutine confirms
        # the queue has drained.
        if (
            self._seal_trigger_round is not None
            and request.drand_round > self._seal_trigger_round
        ):
            return reject(RejectReason.BATCH_FILLED, "seal_extension")
        if request.window_start != self.window_start:
            return reject(RejectReason.WINDOW_MISMATCH, "window")
        # v2.1: checkpoint hash gate. Empty string = gate disabled
        # (pre-first-publish or test convenience).
        if self.current_checkpoint_hash and request.checkpoint_hash != self.current_checkpoint_hash:
            return reject(RejectReason.WRONG_CHECKPOINT, "checkpoint")
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
        # v2.3: cap submissions per prompt before the heavy verify. Once a
        # prompt has ``MAX_SUBMISSIONS_PER_PROMPT`` GRAIL-validated entries,
        # further attempts are rejected PROMPT_FULL without running GRAIL.
        # This bounds the validator's GPU cost in the worst case where many
        # miners attack the same prompt.
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
            logical_reserved, _ = self.try_reserve_logical_group(request)
        except (TypeError, ValueError, OverflowError):
            return reject(RejectReason.BAD_TOKENS, "logical_dedup")
        if not logical_reserved:
            return reject(RejectReason.HASH_DUPLICATE, "logical_dedup")
        self.confirm_logical_group_reservation(request)

        problem = self.env.get_problem(request.prompt_idx)
        validator_scored_reward = _uses_validator_authoritative_reward(self.env)
        completion_texts = []
        for rollout in request.rollouts:
            text = self._completion_text(rollout)
            completion_texts.append(text)
            try:
                computed_reward = float(self.env.compute_reward(problem, text))
            except Exception:
                return reject(RejectReason.REWARD_MISMATCH, "reward")
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
        if not is_in_zone(sigma, bootstrap=self.bootstrap):
            # Reward error (degenerate rollout rewards), not cheating. It never
            # reaches the GRAIL proof path. A high out_of_zone rate is bounded
            # upstream by the grading-attempts ceiling and cannot starve the
            # env below B distinct.
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
        from reliquary.constants import BFT_THINKING_BUDGET
        canonical_force_ids: list[int] = []
        force_think_close_ids: set[int] = set()
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
        seed_cdf_per_rollout: list[dict[str, int | float]] = []

        for rollout_idx, rollout in enumerate(request.rollouts):
            # Never carry a validator-derived carve across re-validation of the
            # same Pydantic object.  The private value is set only after the
            # signed commit's force span passes the canonical BFT checks below.
            rollout._validated_force_span = None
            # `truncated` is a validator-set flag (overlong reward shaping, see
            # submission.py). Wipe any miner-supplied value at ingestion so only
            # the validator's own cap/EOS detection below can set it — otherwise
            # a miner could flag a losing rollout to clamp its negative advantage
            # to -SHAPE_PENALTY via _shape_advantages and attenuate the gradient.
            _ingest_meta = rollout.commit.get("rollout")
            if isinstance(_ingest_meta, dict):
                _ingest_meta["truncated"] = False
                # BFT is math-only (mirror the miner's env gate): `forced` is a
                # validator-honoured flag solely for openmathinstruct. Wipe any
                # non-math value at ingestion so the BFT carve-out stays scoped
                # to math.
                if getattr(self.env, "name", "") != "openmathinstruct":
                    _ingest_meta["forced"] = False
            if not self._verify_signature(rollout.commit, request.miner_hotkey):
                return reject(RejectReason.BAD_SIGNATURE, "rollout_signature")
            # Randomness binding: the miner-claimed beacon randomness MUST equal
            # the validator's per-window derived randomness. Without this check,
            # the sketch-tolerance window (~5000 mod q≈2.15e9) is wide enough
            # that miners using a constant pre-computed r_vec can still slip
            # under the GRAIL diff threshold — observed sketch_diff_max sitting
            # at ~3000–5000 on real submissions, just under the per-position
            # limit. That collapses GRAIL's randomness-binding security to the
            # tolerance × num_buckets product and removes the per-window
            # unpredictability the sketch was designed to provide. Reject here,
            # before paying for the GRAIL forward pass on a commit we already
            # know is detached from the validator's window seed.
            claimed_rand = (rollout.commit.get("beacon") or {}).get("randomness", "")
            if claimed_rand != self.randomness:
                return reject(RejectReason.WRONG_RANDOMNESS, "randomness")
            # Per-position forced-seed uniforms for this rollout's teacher-forced
            # consistency check. Read completion_length here (ahead of the
            # ``completion_len`` computed later at the sparse-outputs section)
            # so the u-stream can accompany the verify call below.
            _seed_completion_len = int(
                (rollout.commit.get("rollout") or {}).get("completion_length", 0)
            )
            seed_u = [
                u_at(
                    self.randomness, request.miner_hotkey, request.prompt_idx,
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
            seed_cdf_per_rollout.append(
                {
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
                    "completion_length": _seed_completion_len,
                    "forced": bool(
                        (rollout.commit.get("rollout") or {}).get("forced")
                    ),
                }
            )
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

        # All checks passed — append to both the flat list and the per-prompt
        # bucket. The bucket is what seal_batch groups over.
        new_sub = ValidSubmission(
            hotkey=request.miner_hotkey,
            prompt_idx=request.prompt_idx,
            merkle_root_bytes=bytes.fromhex(request.merkle_root),
            selection_digest_bytes=compute_rollouts_selection_digest(
                request.rollouts
            ),
            sigma=sigma,
            rollouts=list(request.rollouts),
            completion_texts=completion_texts,
            arrived_at=self._time_fn(),
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
            decision_ts=self._wall_clock(),
            submitted_drand_round=request.drand_round,
            arrival_drand_round=(
                telemetry.arrival_drand_round if telemetry else None
            ),
            drand_delta=telemetry.drand_delta if telemetry else None,
            seal_trigger_round=self._seal_trigger_round,
            prompt_hash_lead=telemetry.prompt_hash_lead if telemetry else None,
            reward_vector=reward_shape.reward_vector,
            truncated_count=truncated_count,
            reward_shape=reward_shape.to_log_dict(),
        )
        self._valid.append(new_sub)
        self._submissions_per_prompt.setdefault(
            request.prompt_idx, []
        ).append(new_sub)
        self.last_valid_submission_at = self._time_fn()
        self.last_valid_submission_wall_ts = self._wall_clock()
        # Lock-free read in /state — see ``__init__`` for rationale.
        self.valid_count = len(self._valid)

        # v2.1: fire seal_event when B distinct non-cooldown prompts have been accepted.
        distinct_eligible = self.distinct_valid_prompt_count()
        if distinct_eligible >= B_BATCH and self._seal_trigger_round is None:
            # B-th distinct prompt just landed. Record the trigger drand
            # round and DELAY the actual seal until the round expires —
            # any further submissions in this same drand round can still
            # be accepted and share the boundary slot value via
            # ``select_batch_and_distribute``'s boundary branch.
            #
            # Two firing paths bring ``_seal_flag`` up:
            #   1. ``_delayed_seal_at_drand_boundary`` — a coroutine
            #      scheduled below sleeps until the next drand boundary
            #      and fires the seal even if no further traffic arrives.
            #   2. The early-fire above: a submission from a LATER drand
            #      round arrives → seal fires immediately, the late
            #      submission is rejected.
            # Whichever fires first wins; the other path becomes a no-op
            # because ``_seal_flag.is_set()`` short-circuits.
            self._seal_trigger_round = request.drand_round
            new_sub.seal_trigger_round = self._seal_trigger_round
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
                # Production path: schedule the delayed fire on the
                # validator's asyncio loop.
                asyncio.run_coroutine_threadsafe(
                    self._delayed_seal_at_drand_boundary(),
                    self._loop,
                )
            else:
                # Synchronous test path (no running loop): fire seal
                # immediately, matching the pre-v2.3 timing tests rely on.
                self._seal_flag.set()
                if self._seal_event is not None:
                    self._seal_event.set()

        return BatchSubmissionResponse(
            accepted=True, reason=RejectReason.ACCEPTED
        )

    async def _delayed_seal_at_drand_boundary(self) -> None:
        """Fire ``_seal_flag`` once (a) the trigger drand round expires
        AND (b) the submit queue has drained.

        Phase 1 — drand boundary wait. Sleep until ``seconds_until_next_drand_boundary``
        passes. After this point HTTP cheap-reject starts rejecting
        submissions whose ``drand_round > _seal_trigger_round`` as
        BATCH_FILLED, so the queue can only lose items from here on.

        Phase 2 — bounded queue drain. Poll ``_queue_drained_predicate`` every
        ~200 ms until it returns True or the drain timeout expires. Only then
        do we fire the seal.
        Without this phase, GRAIL-pending submissions queued during the
        trigger drand round would be dropped at the worker's
        ``is_sealed`` check the instant the seal fires — defeating the
        entire boundary-fair-split design. The timeout is the safety valve:
        fairness improves for normal trigger-round stragglers, but a slow or
        constantly refilled queue cannot freeze checkpoints.

        Defensive fallback: if the drand chain info isn't available
        (synchronous tests that disable the drand check), the boundary
        wait is skipped and seal fires immediately. If the predicate is
        None (tests that don't wire one), the drain wait is also
        skipped. Both fallbacks preserve pre-v2.3 timing for test
        fixtures that don't exercise the full pipeline.

        Cancellation: if the service swaps the active batcher (window
        rolls over) while this coroutine is sleeping, the task is
        cancelled by the loop teardown. That's fine — a stale batcher's
        seal firing late is a no-op (the train step has already moved on).
        """
        # Phase 1 — wait for trigger drand round to expire.
        if self._drand_chain_info is not None:
            from reliquary.infrastructure.chain import seconds_until_next_drand_boundary
            ci = self._drand_chain_info
            delay = seconds_until_next_drand_boundary(
                self._wall_clock(), ci["genesis_time"], ci["period"],
            )
            # Small margin past the boundary so a submission sent right
            # at the boundary still resolves to the trigger round on
            # arrival.
            await asyncio.sleep(delay + 0.05)

        # Phase 2 — drain the queue of trigger-round submissions still
        # waiting on GRAIL. Without ``_queue_drained_predicate`` (test
        # context), this phase is a no-op and the seal fires immediately.
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
                # rejection point.
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
                    )
                )
        return BatchSubmissionResponse(accepted=False, reason=reason)

    # ----------------------------- accessors -----------------------------

    def valid_submissions(self) -> list[ValidSubmission]:
        with self._lock:
            return list(self._valid)

    def seal_batch(
        self, pool: float = 1.0
    ) -> tuple[list[ValidSubmission], dict[str, float]]:
        """Pick the training batch and compute the reward distribution.

        Returns (training_batch, rewards_by_hotkey). Cooldown and hash-set
        bookkeeping is applied to every winning prompt — not just the one
        submission picked for training — because all of them earn emission
        and were therefore "used" by this window.
        """
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
                valid_submissions=len(self._valid),
                checkpoint_n=0,
            )
