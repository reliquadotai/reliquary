"""GrpoWindowBatcher — orchestrator for the free-prompt GRPO market.

Holds a flat list of validated submissions per window + a reference to the
validator's shared ``CooldownMap``.
"""

from __future__ import annotations

import asyncio
import logging
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
    MAX_POST_TRIGGER_PROOF_CANDIDATES,
    MAX_PROOF_CANDIDATES_PER_WINDOW,
    MAX_SEAL_QUEUE_DRAIN_SECONDS,
    MAX_SUBMISSIONS_PER_PROMPT,
    MAX_TRUNCATED_PER_SUBMISSION,
    REJECTED_LIST_CAP_PER_HOTKEY,
)
from reliquary.environment.base import Environment
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
from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash
from reliquary.validator.observability import (
    DrandRoundObservation,
    SubmitTelemetry,
    classify_drand_round,
    log_submission_stage,
)
from reliquary.validator.boxed_integrity import has_malformed_final_answer
from reliquary.validator.reward_shape import detect_reward_shape_manipulation
from reliquary.validator.rollout_patterns import detect_opposite_reward_clones
from reliquary.validator.verifier import (
    evaluate_boxed_answer_probability,
    evaluate_token_distribution,
    has_eos_padding,
    is_cap_truncation,
    is_in_zone,
    rewards_std,
    verify_logprobs_claim,
    verify_reward_claim,
    verify_termination,
)

logger = logging.getLogger(__name__)

_PROOF_FAILURE_DEBT_STAGES = frozenset(
    {
        "grail",
        "termination",
        "logprob",
        "distribution",
        "boxed_answer",
        "reward_shape",
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
    sigma: float = 0.0
    rollouts: list[RolloutSubmission] = field(default_factory=list)
    completion_texts: list[str] = field(default_factory=list)
    arrived_at: float = 0.0
    # Filter telemetry (worst-case across this submission's rollouts).
    # Captured for post-hoc threshold calibration without re-running tests.
    sketch_diff_max: int | None = None
    lp_dev_max: float | None = None
    dist_q10_min: float | None = None
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
        self._valid: list[ValidSubmission] = []
        # v2.3: per-prompt bucket. Multiple miners may submit on the same
        # ``prompt_idx`` up to ``MAX_SUBMISSIONS_PER_PROMPT``. Tracked
        # alongside the flat ``_valid`` list because seal_batch needs the
        # grouping but accept-time logic only needs the count.
        self._submissions_per_prompt: dict[int, list[ValidSubmission]] = {}
        self.randomness: str = ""
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
        # ``lambda: server._submit_queue.empty()`` so already admitted
        # trigger-round submissions get a short chance to finish GRAIL
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
        self._post_trigger_proof_admission_count = 0
        self._expensive_proof_failures_by_hotkey: dict[str, int] = {}
        # v2.1: checkpoint hash miners must match. Empty string disables
        # the gate (test convenience / pre-first-publish).
        self.current_checkpoint_hash: str = ""

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

    @property
    def proof_admission_count(self) -> int:
        """Number of submissions admitted to the expensive proof path."""
        return self._proof_admission_count

    @property
    def post_trigger_proof_admission_count(self) -> int:
        """Number of proof-path reservations after the seal trigger round."""
        return self._post_trigger_proof_admission_count

    @property
    def expensive_proof_failures_by_hotkey(self) -> dict[str, int]:
        """Per-window count of hotkey failures after entering proof path."""
        return dict(self._expensive_proof_failures_by_hotkey)

    def proof_failure_debt(self, hotkey: str) -> int:
        return self._expensive_proof_failures_by_hotkey.get(hotkey, 0)

    def try_reserve_proof_admission(
        self,
        request: BatchSubmissionRequest,
    ) -> tuple[bool, str | None]:
        """Reserve one expensive proof-verification slot for this window.

        The reservation is intentionally not released when a submission fails
        later checks. The cap bounds validator GPU work under spam; refunding
        failed attempts would make invalid proof traffic free again.
        """
        with self._proof_admission_lock:
            if (
                self._expensive_proof_failures_by_hotkey.get(
                    request.miner_hotkey, 0
                )
                >= MAX_EXPENSIVE_PROOF_FAILURES_PER_HOTKEY_PER_WINDOW
            ):
                return False, "proof_failure_debt_hotkey"

            if self._proof_admission_count >= MAX_PROOF_CANDIDATES_PER_WINDOW:
                return False, "proof_admission_window_full"

            trigger_round = self._seal_trigger_round
            if (
                trigger_round is not None
                and request.drand_round == trigger_round
            ):
                if (
                    self._post_trigger_proof_admission_count
                    >= MAX_POST_TRIGGER_PROOF_CANDIDATES
                ):
                    return False, "proof_admission_post_trigger_full"
                self._post_trigger_proof_admission_count += 1

            self._proof_admission_count += 1
            return True, None

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

        problem = self.env.get_problem(request.prompt_idx)
        completion_texts = []
        for rollout in request.rollouts:
            text = self._completion_text(rollout)
            completion_texts.append(text)
            if not verify_reward_claim(self.env, problem, text, rollout.reward):
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
            return reject(RejectReason.OUT_OF_ZONE, "zone")

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

        for rollout_idx, rollout in enumerate(request.rollouts):
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
            proof = self._verify_commitment(
                rollout.commit, self.model, self.randomness
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
                    rollout.commit, self.tokenizer, proof, self.model
                )
                cap_truncated = is_cap_truncation(
                    rollout.commit, self.tokenizer, proof, self.model
                )
                if not termination_ok or cap_truncated:
                    truncated_flags[rollout_idx] = True
                    truncated_count += 1
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

        reward_shape = detect_reward_shape_manipulation(
            rewards,
            completion_lengths,
            truncated_flags,
        )
        if reward_shape.suspicious:
            logger.info(
                "reject reason=reward_shape_suspicious hotkey=%s %s",
                request.miner_hotkey,
                reward_shape.to_log_dict(),
            )
            return reject(
                RejectReason.REWARD_SHAPE_SUSPICIOUS,
                "reward_shape",
                sketch_diff_max=sketch_diff_max,
                lp_dev_max=lp_dev_max,
                dist_q10_min=dist_q10_min,
            )

        # All checks passed — append to both the flat list and the per-prompt
        # bucket. The bucket is what seal_batch groups over.
        new_sub = ValidSubmission(
            hotkey=request.miner_hotkey,
            prompt_idx=request.prompt_idx,
            merkle_root_bytes=bytes.fromhex(request.merkle_root),
            sigma=sigma,
            rollouts=list(request.rollouts),
            completion_texts=completion_texts,
            arrived_at=self._time_fn(),
            sketch_diff_max=sketch_diff_max,
            lp_dev_max=lp_dev_max,
            dist_q10_min=dist_q10_min,
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
        # Lock-free read in /state — see ``__init__`` for rationale.
        self.valid_count = len(self._valid)

        # v2.1: fire seal_event when B distinct non-cooldown prompts have been accepted.
        distinct_eligible = len({
            s.prompt_idx for s in self._valid
            if not self._cooldown.is_in_cooldown(s.prompt_idx, self.window_start)
        })
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
                        self._post_trigger_proof_admission_count,
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
