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
    M_ROLLOUTS,
    MAX_SUBMISSIONS_PER_PROMPT,
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
from reliquary.validator.batch_selection import select_batch_and_distribute
from reliquary.validator.cooldown import CooldownMap
from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash
from reliquary.validator.verifier import (
    evaluate_token_distribution,
    is_in_zone,
    rewards_std,
    verify_logprobs_claim,
    verify_reward_claim,
    verify_termination,
)

logger = logging.getLogger(__name__)


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
    ) -> None:
        import time
        from reliquary.constants import DRAND_ROUND_BACKWARD_TOLERANCE

        self.window_start = window_start
        self.env = env
        self.model = model
        self.tokenizer = tokenizer
        self.bootstrap = bootstrap
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
        # Accumulated reject reasons this window (RejectReason.value → count).
        # Persisted in the R2 archive so miners can see which filter is
        # rejecting the most submissions in any given round.
        self.reject_counts: dict[str, int] = {}

        # Per-hotkey-capped metadata for rejected submissions. Persisted in
        # the R2 archive next to ``reject_counts`` so a rejected miner can
        # see *which* of their submissions failed and why, instead of just
        # an aggregate count. Cap protects against single-attacker flooding.
        self.rejected_submissions: list[RejectedSubmission] = []

        # v2.1: seal_event fires the moment the B-th distinct non-cooldown
        # valid submission lands. Service awaits this to close the window.
        # Stored as threading.Event for sync-safe set(); the asyncio.Event
        # is created lazily on first access to bind to the current loop.
        self._seal_flag: threading.Event = threading.Event()
        self._seal_event: asyncio.Event | None = None
        # v2.1: checkpoint hash miners must match. Empty string disables
        # the gate (test convenience / pre-first-publish).
        self.current_checkpoint_hash: str = ""

    @property
    def seal_event(self) -> asyncio.Event:
        """Lazy asyncio.Event bound to whichever loop accesses it first."""
        if self._seal_event is None:
            self._seal_event = asyncio.Event()
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

    # ----------------------------- ingestion -----------------------------

    def accept_submission(
        self, request: BatchSubmissionRequest
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
            return self._accept_locked(request)

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
        if self._drand_chain_info is None:
            from reliquary.infrastructure.drand import get_current_chain
            self._drand_chain_info = get_current_chain()
        ci = self._drand_chain_info
        from reliquary.infrastructure.chain import compute_current_drand_round
        t = t_arrival if t_arrival is not None else self._wall_clock()
        current = compute_current_drand_round(
            t, ci["genesis_time"], ci["period"],
        )
        if drand_round > current:
            return RejectReason.FUTURE_ROUND
        if drand_round < current - self.drand_round_backward_tolerance:
            return RejectReason.STALE_ROUND
        return None

    def _accept_locked(
        self, request: BatchSubmissionRequest
    ) -> BatchSubmissionResponse:
        hk = request.miner_hotkey
        pi = request.prompt_idx
        if request.window_start != self.window_start:
            return self._reject(RejectReason.WINDOW_MISMATCH, hotkey=hk, prompt_idx=pi)
        # v2.1: checkpoint hash gate. Empty string = gate disabled
        # (pre-first-publish or test convenience).
        if self.current_checkpoint_hash and request.checkpoint_hash != self.current_checkpoint_hash:
            return self._reject(RejectReason.WRONG_CHECKPOINT, hotkey=hk, prompt_idx=pi)
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
            return self._reject(RejectReason.BAD_PROMPT_IDX, hotkey=hk, prompt_idx=pi)
        if self._cooldown.is_in_cooldown(request.prompt_idx, self.window_start):
            return self._reject(RejectReason.PROMPT_IN_COOLDOWN, hotkey=hk, prompt_idx=pi)
        # v2.3: cap submissions per prompt before the heavy verify. Once a
        # prompt has ``MAX_SUBMISSIONS_PER_PROMPT`` GRAIL-validated entries,
        # further attempts are rejected PROMPT_FULL without running GRAIL.
        # This bounds the validator's GPU cost in the worst case where many
        # miners attack the same prompt.
        existing = self._submissions_per_prompt.get(request.prompt_idx, [])
        if len(existing) >= MAX_SUBMISSIONS_PER_PROMPT:
            return self._reject(RejectReason.PROMPT_FULL, hotkey=hk, prompt_idx=pi)

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
                    return self._reject(
                        RejectReason.HASH_DUPLICATE, hotkey=hk, prompt_idx=pi,
                    )
                local_seen.add(h)
                rollout_hashes.append(h)

        problem = self.env.get_problem(request.prompt_idx)
        completion_texts = []
        for rollout in request.rollouts:
            text = self._completion_text(rollout)
            completion_texts.append(text)
            if not verify_reward_claim(self.env, problem, text, rollout.reward):
                return self._reject(RejectReason.REWARD_MISMATCH, hotkey=hk, prompt_idx=pi)

        sigma = rewards_std([r.reward for r in request.rollouts])
        if not is_in_zone(sigma, bootstrap=self.bootstrap):
            return self._reject(RejectReason.OUT_OF_ZONE, hotkey=hk, prompt_idx=pi)

        # Per-submission worst-case filter telemetry (across all rollouts).
        sketch_diff_max = 0
        lp_dev_max: float | None = None
        dist_q10_min: float | None = None

        # Bind ``prompt_idx`` to the actual prompt tokens the miner ran their
        # forward pass on. Without this, a miner can submit completions
        # generated under a modified prompt (CoT prefix, alternate chat
        # template, few-shot examples) while claiming the canonical
        # ``prompt_idx``: the reward check still passes if the underlying
        # question is intact, but training sees a distribution shift. Compute
        # the canonical tokens once per submission and reject any rollout
        # whose ``tokens[:prompt_length]`` diverges before doing GRAIL compute.
        canonical_prompt_tokens: list[int] | None = None
        if self._canonical_prompt_tokens is not None:
            canonical_prompt_tokens = list(
                self._canonical_prompt_tokens(request.prompt_idx)
            )

        # Allow up to 1 truncated rollout per submission. At T_PROTO=0.9 on
        # math problems with Qwen3-4B, ~1/8 rollouts statistically drift past
        # max_new_tokens without sampling EOS. Failing the entire submission
        # for a single drift makes acceptance vanishingly rare. The truncated
        # rollout itself is still excluded from the per-rollout behavioural
        # checks below (logprobs / distribution) since they assume completion.
        MAX_TRUNCATED_PER_SUBMISSION = 1
        truncated_count = 0

        for rollout in request.rollouts:
            # Schema check: structural validation of commit dict (cheap, no GPU)
            try:
                CommitModel.model_validate(rollout.commit)
            except ValidationError:
                return self._reject(RejectReason.BAD_SCHEMA, hotkey=hk, prompt_idx=pi)

            # Token check: vocab bounds + max length (cheap, protects forward pass)
            if not verify_tokens(rollout.commit["tokens"], self.model.config):
                return self._reject(RejectReason.BAD_TOKENS, hotkey=hk, prompt_idx=pi)

            if canonical_prompt_tokens is not None:
                rollout_meta = rollout.commit.get("rollout", {}) or {}
                miner_prompt_len = int(rollout_meta.get("prompt_length", 0))
                miner_prompt_tokens = list(rollout.commit.get("tokens", []))[
                    :miner_prompt_len
                ]
                if miner_prompt_tokens != canonical_prompt_tokens:
                    return self._reject(RejectReason.PROMPT_MISMATCH, hotkey=hk, prompt_idx=pi)
            if not self._verify_signature(rollout.commit, request.miner_hotkey):
                return self._reject(RejectReason.BAD_SIGNATURE, hotkey=hk, prompt_idx=pi)
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
                return self._reject(
                    RejectReason.WRONG_RANDOMNESS, hotkey=hk, prompt_idx=pi,
                )
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
                return self._reject(
                    RejectReason.GRAIL_FAIL,
                    hotkey=hk, prompt_idx=pi,
                    sketch_diff_max=proof.sketch_diff_max,
                )

            # Termination check: rollout must end with EOS at p(EOS) >= threshold.
            # Reuses cached logits from the GRAIL forward — zero extra compute.
            # Skipped when grail stub returns empty logits (legacy test fixtures).
            if proof.logits.numel() > 0:
                if not verify_termination(
                    rollout.commit, self.tokenizer, proof.logits, self.model
                ):
                    truncated_count += 1
                    if truncated_count > MAX_TRUNCATED_PER_SUBMISSION:
                        return self._reject(
                            RejectReason.BAD_TERMINATION,
                            hotkey=hk, prompt_idx=pi,
                            sketch_diff_max=sketch_diff_max,
                        )
                    # Skip the per-rollout behavioural checks below (logprobs,
                    # distribution) for this truncated rollout — they assume a
                    # completed sequence. The reward is still counted in the
                    # group's σ for the out_of_zone gate.
                    continue

            # Behavioural checks (use cached logits from the GRAIL forward pass).
            # Skip gracefully if the logits tensor is empty (legacy stubs in tests).
            if proof.logits.numel() == 0:
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
                logits=proof.logits,
                challenge_randomness=self.randomness,
            )
            if lp_dev is not None and lp_dev != float("inf"):
                if lp_dev_max is None or lp_dev > lp_dev_max:
                    lp_dev_max = float(lp_dev)
            if not lp_ok:
                logger.info(
                    "reject reason=logprob_mismatch hotkey=%s median_dev=%.4f",
                    request.miner_hotkey, lp_dev,
                )
                return self._reject(
                    RejectReason.LOGPROB_MISMATCH,
                    hotkey=hk, prompt_idx=pi,
                    sketch_diff_max=sketch_diff_max,
                    lp_dev_max=lp_dev_max,
                )

            from reliquary.constants import T_PROTO
            dist_ok, dist_metrics = evaluate_token_distribution(
                tokens=rollout.commit["tokens"],
                prompt_length=prompt_len,
                completion_length=completion_len,
                logits=proof.logits,
                temperature=T_PROTO,
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
                return self._reject(
                    RejectReason.DISTRIBUTION_SUSPICIOUS,
                    hotkey=hk, prompt_idx=pi,
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
        if distinct_eligible >= B_BATCH and not self._seal_flag.is_set():
            self._seal_flag.set()
            if self._seal_event is not None:
                self._seal_event.set()

        return BatchSubmissionResponse(
            accepted=True, reason=RejectReason.ACCEPTED
        )

    def _reject(
        self,
        reason: RejectReason,
        *,
        hotkey: str | None = None,
        prompt_idx: int | None = None,
        sketch_diff_max: int | None = None,
        lp_dev_max: float | None = None,
        dist_q10_min: float | None = None,
    ) -> BatchSubmissionResponse:
        self.reject_counts[reason.value] = self.reject_counts.get(reason.value, 0) + 1

        if hotkey is not None and prompt_idx is not None:
            already = sum(
                1 for r in self.rejected_submissions if r.hotkey == hotkey
            )
            if already < REJECTED_LIST_CAP_PER_HOTKEY:
                # Anti-tuning: never surface the GRAIL sketch diff to miners.
                # All other reasons get the diagnostics computed up to the
                # rejection point.
                if reason is RejectReason.GRAIL_FAIL:
                    sketch_diff_max = None
                self.rejected_submissions.append(
                    RejectedSubmission(
                        hotkey=hotkey,
                        prompt_idx=prompt_idx,
                        reason=reason.value,
                        sketch_diff_max=sketch_diff_max,
                        lp_dev_max=lp_dev_max,
                        dist_q10_min=dist_q10_min,
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
            batch, rewards = select_batch_and_distribute(
                submissions=self._valid,
                b=B_BATCH,
                cooldown_map=self._cooldown,
                current_window=self.window_start,
                pool=pool,
            )
            winning_prompts = {sub.prompt_idx for sub in batch}
            for p in winning_prompts:
                self._cooldown.record_batched(p, self.window_start)
                if self._hash_set is not None:
                    for sub in self._submissions_per_prompt.get(p, []):
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
