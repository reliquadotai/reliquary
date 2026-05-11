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
from reliquary.validator.batch_selection import select_batch
from reliquary.validator.cooldown import CooldownMap
from reliquary.validator.verifier import (
    evaluate_token_distribution,
    is_in_zone,
    rewards_std,
    verify_logprobs_claim,
    verify_reward_claim,
    verify_termination,
)

logger = logging.getLogger(__name__)


# v2.2: ``signed_round`` was removed entirely from the protocol. The batch
# selection mechanism is now pure TCP-arrival FIFO (see ``batch_selection.py``)
# and submissions are short-circuited to SUPERSEDED on the first claim of
# any given ``prompt_idx`` for the active window.


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
        bootstrap: bool = False,
        completion_text_fn: Callable[[RolloutSubmission], str],
        canonical_prompt_tokens_fn: Callable[[int], list[int]] | None = None,
        verify_commitment_proofs_fn: Callable[..., Any] | None = None,
        verify_signature_fn: Callable[[dict, str], bool] | None = None,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        import time

        self.window_start = window_start
        self.env = env
        self.model = model
        self.tokenizer = tokenizer
        self.bootstrap = bootstrap
        self._completion_text = completion_text_fn
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
        # Per-prompt claim set. As of v2.2 the FIFO mechanism is pure
        # TCP-arrival: the first submission accepted for a given
        # ``prompt_idx`` claims the slot, and any subsequent submission for
        # the same prompt is rejected SUPERSEDED before any heavy
        # validation (reward / GRAIL forward pass). This replaces the old
        # ``_best_round_per_prompt`` mechanism keyed by ``signed_round``.
        self._claimed_prompts: set[int] = set()
        self.randomness: str = ""
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

    # ----------------------------- ingestion -----------------------------

    def accept_submission(
        self, request: BatchSubmissionRequest
    ) -> BatchSubmissionResponse:
        """Run the full verification pipeline; append to ``_valid`` on success."""
        with self._lock:
            return self._accept_locked(request)

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
        if request.prompt_idx >= len(self.env):
            return self._reject(RejectReason.BAD_PROMPT_IDX, hotkey=hk, prompt_idx=pi)
        # v2.2: ``signed_round`` was removed from the protocol entirely.
        # Cooldown and prompt-claim short-circuits below handle ordering.
        if self._cooldown.is_in_cooldown(request.prompt_idx, self.window_start):
            return self._reject(RejectReason.PROMPT_IN_COOLDOWN, hotkey=hk, prompt_idx=pi)
        # FIFO short-circuit by TCP arrival: the first submission to pass the
        # full validation pipeline for a given ``prompt_idx`` claims the
        # slot. Every subsequent submission for the same prompt is rejected
        # immediately, BEFORE the expensive reward + GRAIL checks. This is
        # what protects the validator from being DoS'd by a flood of
        # duplicate-prompt submissions and keeps each prompt single-winner.
        if request.prompt_idx in self._claimed_prompts:
            return self._reject(RejectReason.SUPERSEDED, hotkey=hk, prompt_idx=pi)

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

        # All checks passed — claim this prompt. Future submissions for the
        # same ``prompt_idx`` are short-circuited at the top of
        # ``_accept_locked`` and never reach the validation pipeline.
        self._claimed_prompts.add(request.prompt_idx)

        self._valid.append(
            ValidSubmission(
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
            )
        )

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

    def seal_batch(self) -> list[ValidSubmission]:
        with self._lock:
            batch = select_batch(
                self._valid,
                b=B_BATCH,
                current_window=self.window_start,
                cooldown_map=self._cooldown,
            )
            for sub in batch:
                self._cooldown.record_batched(sub.prompt_idx, self.window_start)
            return batch

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
