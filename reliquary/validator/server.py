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
import logging
import numbers
import time
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field, ValidationError
import uvicorn

from reliquary.constants import (
    B_BATCH,
    BOOTSTRAP_MAX_TRUNCATED_PER_SUBMISSION,
    DRAND_ROUND_BACKWARD_TOLERANCE,
    ENFORCE_ENVELOPE_SIGNATURE,
    MAX_BAD_ENVELOPE_PER_HOTKEY_PER_WINDOW,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    MAX_POST_TRIGGER_PROOF_CANDIDATES,
    MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW,
    MAX_TRUNCATED_PER_SUBMISSION,
    REGISTERED_HOTKEY_CACHE_TTL_SECONDS,
    REGISTERED_HOTKEY_REFRESH_MIN_INTERVAL_SECONDS,
    REGISTERED_HOTKEY_REFRESH_TIMEOUT_SECONDS,
    REGISTERED_HOTKEY_STALE_GRACE_SECONDS,
    SPARSE_VALID_IDLE_MIN_DISTINCT_PROMPTS,
    SPARSE_VALID_IDLE_SEAL_SECONDS,
    SPARSE_VALID_MAX_WINDOW_SECONDS,
    VALIDATOR_HTTP_PORT,
)
from reliquary.protocol.signatures import verify_envelope_signature
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    BatchSubmissionResponse,
    CommitModel,
    GrpoBatchState,
    RejectReason,
    Verdict,
    VerdictsResponse,
)
from reliquary.protocol.tokens import verify_tokens
from reliquary.shared.modeling import resolve_eos_token_ids
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
    is_in_zone,
    rewards_std,
    validate_force_span,
)

logger = logging.getLogger(__name__)


# How many recent verdicts to remember per hotkey. Bounded so the
# ring buffer can't grow without limit if a misbehaving miner spams.
# At ~250 B per verdict × 200 entries × ~50 hotkeys ≈ 2.5 MB — cheap.
VERDICT_CAP_PER_HOTKEY = 200


def _is_mock_like(value: Any) -> bool:
    """Return True for unittest.mock objects.

    The server's unit tests use loose MagicMocks as batchers; touching nested
    attributes on them auto-creates truthy mock objects. Production batchers
    carry real model/tokenizer/config objects, so skip optional preflight pieces
    when the object is clearly a mock.
    """
    return type(value).__module__.startswith("unittest.mock")


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
    invalid token envelopes, EOS padding, non-cap completions that never
    emitted EOS, and self-declared final-EOS probabilities below the hard
    termination threshold.
    """
    for rollout in request.rollouts:
        try:
            CommitModel.model_validate(rollout.commit)
        except ValidationError:
            return RejectReason.BAD_SCHEMA, "schema"
        if list(rollout.tokens) != list(rollout.commit["tokens"]):
            return RejectReason.TOKENS_MISMATCH, "token_invariant"

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
        if not is_in_zone(
            rewards_std(rewards),
            bootstrap=_proof_free_bootstrap(batcher),
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
    app_started_at: float
    current_validator_state: str
    current_window_n: int | None = None
    current_quicknet_drand_round: int | None = None
    current_window_open_ts: float | None = None
    current_window_open_drand_round: int | None = None
    seal_trigger_round: int | None = None
    drand_round_backward_tolerance: int
    batch_size: int
    queue_depth: int | None = None
    proof_verification_inflight: int | None = None
    valid_submissions_count: int | None = None
    distinct_valid_prompt_count: int | None = None
    last_valid_submission_ts: float | None = None
    seconds_since_last_valid_submission: float | None = None
    proof_admission_count: int | None = None
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
    checkpoint_repo_id: str | None = None
    checkpoint_revision: str | None = None
    recent_reject_counts_by_reason: dict[str, int]
    rewarded_but_not_selected_by_hotkey: dict[str, int] = Field(
        default_factory=dict
    )
    registration_gate_enforced: bool = False
    registered_hotkey_count: int | None = None
    registration_cache_age_seconds: float | None = None
    registration_cache_stale: bool | None = None


class ValidatorServer:
    def __init__(self, host: str = "0.0.0.0", port: int = VALIDATOR_HTTP_PORT) -> None:
        self.host = host
        self.port = port
        self._app_started_at = time.time()
        self._image_revision = runtime_revision()
        # Multi-env: keyed by env_name. ``active_batcher`` (singular) is
        # maintained as a legacy accessor pointing to the first active batcher
        # so existing code paths (/health, /state, the submit worker stale
        # check) keep working without change.
        self._active_batchers: dict[str, GrpoWindowBatcher] = {}
        self.active_batcher: GrpoWindowBatcher | None = None
        self._registration_gate_enforced = False
        self._registered_hotkeys: frozenset[str] | None = None
        self._registration_refreshed_at: float | None = None
        self._registration_refresh_callback: (
            Callable[[], Awaitable[bool]] | None
        ) = None
        self._registration_refresh_lock = asyncio.Lock()
        self._last_registration_refresh_attempt = 0.0
        self.app: FastAPI = self._build_app()
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[Any] | None = None
        self._submit_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: asyncio.Task[Any] | None = None
        self._inflight_proofs = 0
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
            self._per_window_counts = {}
            self._bad_envelope_counts = {}
            self._recent_reject_counts = collections.Counter()
        self._active_batchers = batchers
        # Legacy scalar: first batcher in dict (or None if empty).
        self.active_batcher = next(iter(batchers.values())) if batchers else None

    def set_active_batcher(self, batcher: GrpoWindowBatcher | None) -> None:
        """Legacy single-env shim. Wraps into a dict and delegates."""
        if batcher is None:
            self.set_active_batchers({})
        else:
            env_name = getattr(getattr(batcher, "env", None), "name", "unknown")
            self.set_active_batchers({env_name: batcher})

    def set_current_state(self, state) -> None:
        self._current_state = state

    def set_current_checkpoint(self, entry) -> None:
        self._current_checkpoint = entry

    def configure_registration_gate(
        self,
        refresh_callback: Callable[[], Awaitable[bool]],
    ) -> None:
        """Arm registered-hotkey admission for the production service."""
        self._registration_refresh_callback = refresh_callback
        self._registration_gate_enforced = True

    def set_registered_hotkeys(
        self,
        hotkeys: set[str] | frozenset[str] | list[str],
        *,
        refreshed_at: float | None = None,
    ) -> None:
        self._registered_hotkeys = frozenset(
            str(hotkey) for hotkey in hotkeys if str(hotkey)
        )
        self._registration_refreshed_at = (
            time.time() if refreshed_at is None else float(refreshed_at)
        )

    def registration_cache_age(self, *, now: float | None = None) -> float | None:
        if self._registration_refreshed_at is None:
            return None
        current = time.time() if now is None else float(now)
        return max(0.0, current - self._registration_refreshed_at)

    async def _registration_reject_reason(
        self,
        hotkey: str,
    ) -> RejectReason | None:
        if not self._registration_gate_enforced:
            return None

        now = time.time()
        age = self.registration_cache_age(now=now)
        missing = (
            self._registered_hotkeys is None
            or hotkey not in self._registered_hotkeys
        )
        refresh_due = age is None or age > REGISTERED_HOTKEY_CACHE_TTL_SECONDS
        should_refresh = missing or refresh_due
        callback = self._registration_refresh_callback

        if (
            should_refresh
            and callback is not None
            and now - self._last_registration_refresh_attempt
            >= REGISTERED_HOTKEY_REFRESH_MIN_INTERVAL_SECONDS
        ):
            async with self._registration_refresh_lock:
                now = time.time()
                age = self.registration_cache_age(now=now)
                missing = (
                    self._registered_hotkeys is None
                    or hotkey not in self._registered_hotkeys
                )
                refresh_due = (
                    age is None or age > REGISTERED_HOTKEY_CACHE_TTL_SECONDS
                )
                if (
                    (missing or refresh_due)
                    and now - self._last_registration_refresh_attempt
                    >= REGISTERED_HOTKEY_REFRESH_MIN_INTERVAL_SECONDS
                ):
                    self._last_registration_refresh_attempt = now
                    try:
                        await asyncio.wait_for(
                            callback(),
                            timeout=REGISTERED_HOTKEY_REFRESH_TIMEOUT_SECONDS,
                        )
                    except Exception:
                        logger.exception("registered-hotkey cache refresh failed")

        age = self.registration_cache_age()
        if (
            self._registered_hotkeys is None
            or age is None
            or age > REGISTERED_HOTKEY_STALE_GRACE_SECONDS
        ):
            return RejectReason.REGISTRATION_UNAVAILABLE
        if hotkey not in self._registered_hotkeys:
            return RejectReason.HOTKEY_NOT_REGISTERED
        return None

    @property
    def submit_queue_depth(self) -> int:
        return self._submit_queue.qsize()

    @property
    def proof_verification_inflight(self) -> int:
        return self._inflight_proofs

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

        Called from every code path that decides accept/reject:

          * HTTP rate-limit / window-not-active / batch-filled early cutoffs
            in the ``/submit`` handler (before the request even reaches the
            queue worker)
          * ``_submit_worker`` after each ``batcher.accept_submission``
            returns its real verdict (the path that's currently invisible
            to miners because /submit returned ``SUBMITTED`` provisionally)
          * ``_submit_worker`` late drops for items dequeued after the
            batcher swap or seal (``worker_dropped`` / ``batch_filled``)

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

    def _health_payload(self) -> _Health:
        batcher = self.active_batcher
        cp = self._current_checkpoint
        registration_age = self.registration_cache_age()
        reject_counts: dict[str, int] = dict(self._recent_reject_counts)
        if batcher is not None:
            for reason, count in getattr(batcher, "reject_counts", {}).items():
                reject_counts[reason] = max(reject_counts.get(reason, 0), count)
        return _Health(
            status="ok",
            active_window=batcher.window_start if batcher else None,
            image_revision=self._image_revision,
            app_started_at=self._app_started_at,
            current_validator_state=getattr(self._current_state, "value", str(self._current_state)),
            current_window_n=batcher.window_start if batcher else None,
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
            batch_size=B_BATCH,
            queue_depth=self._submit_queue.qsize(),
            proof_verification_inflight=self._inflight_proofs,
            valid_submissions_count=(
                getattr(batcher, "valid_count", None) if batcher else None
            ),
            distinct_valid_prompt_count=(
                batcher.distinct_valid_prompt_count()
                if (
                    batcher is not None
                    and hasattr(batcher, "distinct_valid_prompt_count")
                )
                else None
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
            post_trigger_proof_admission_count=(
                getattr(batcher, "post_trigger_proof_admission_count", None)
                if batcher else None
            ),
            expensive_proof_failures_by_hotkey=(
                dict(getattr(batcher, "expensive_proof_failures_by_hotkey", {}))
                if batcher else {}
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
            registration_cache_age_seconds=registration_age,
            registration_cache_stale=(
                registration_age > REGISTERED_HOTKEY_CACHE_TTL_SECONDS
                if registration_age is not None
                else None
            ),
        )

    @staticmethod
    def _call_accept_submission(
        batcher: Any,
        request: BatchSubmissionRequest,
        telemetry: SubmitTelemetry,
    ) -> BatchSubmissionResponse:
        try:
            return batcher.accept_submission(request, telemetry=telemetry)
        except TypeError as exc:
            if "telemetry" not in str(exc):
                raise
            return batcher.accept_submission(request)

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
            return await call_next(request)

        @app.get("/health", response_model=_Health)
        async def health() -> _Health:
            return self._health_payload()

        @app.post("/submit", response_model=BatchSubmissionResponse)
        async def submit(
            request: BatchSubmissionRequest,
            http_request: Request,
            response: Response,
        ) -> BatchSubmissionResponse:
            from reliquary.protocol.submission import WindowState
            # ASGI middleware stamped this. Falls back to time.time() if a
            # caller bypasses the middleware (e.g. some test harnesses).
            t_arrival = getattr(http_request.state, "t_arrival", None)
            if t_arrival is None:
                t_arrival = time.time()
            hk = request.miner_hotkey
            telemetry = SubmitTelemetry.from_request(
                request, t_arrival=t_arrival,
            )
            telemetry.refresh_from_batcher(self.active_batcher)
            log_submission_stage(
                logger,
                logging.INFO,
                "submit_received",
                telemetry,
                queue_depth=self._submit_queue.qsize(),
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
                if not verify_envelope_signature(
                    miner_hotkey=hk,
                    window_start=request.window_start,
                    prompt_idx=request.prompt_idx,
                    merkle_root=request.merkle_root,
                    checkpoint_hash=request.checkpoint_hash,
                    drand_round=request.drand_round,
                    randomness=_randomness_for_sig,
                    nonce=request.nonce,
                    envelope_signature=request.envelope_signature,
                ):
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

            # Route by env_name: determine which batcher to use.
            # env_name is carried on each rollout; all rollouts in one request
            # must be for the same env (enforced by the miner engine).
            submission_env_name = (
                request.rollouts[0].env_name if request.rollouts else ""
            )
            batcher = self._active_batchers.get(submission_env_name)
            if batcher is None:
                # Fallback: if env_name is absent or unknown, try the legacy
                # active_batcher path (single-env backward compat).
                batcher = self.active_batcher
            if batcher is None:
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

            # Only registered subnet hotkeys may consume submission quota or
            # proof capacity. The signature gate above proves ownership of the
            # claimed hotkey; this gate binds that identity to the current
            # metagraph. A cache miss triggers one bounded refresh, while a
            # short chain outage may use a recent last-known-good snapshot.
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
                return BatchSubmissionResponse(
                    accepted=False,
                    reason=registration_reason,
                )

            # Rate limit AFTER signature verification and active-window
            # binding. A signed stale-window replay or a miner still catching
            # up after restart must not burn the hotkey's quota for the new
            # window. Once the request is known to target this live window,
            # count it before cheap rejects/GRAIL so spam still self-throttles.
            n = self._per_window_counts.get(hk, 0)
            if n >= MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW:
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
                return BatchSubmissionResponse(
                    accepted=False, reason=RejectReason.RATE_LIMITED,
                )
            self._per_window_counts[hk] = n + 1

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
                return BatchSubmissionResponse(
                    accepted=False, reason=RejectReason.BATCH_FILLED,
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
            from reliquary.constants import MAX_SUBMISSIONS_PER_PROMPT

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
                return BatchSubmissionResponse(accepted=False, reason=reason)

            if batcher.current_checkpoint_hash and request.checkpoint_hash != batcher.current_checkpoint_hash:
                return _cheap_reject(
                    RejectReason.WRONG_CHECKPOINT,
                    reject_stage="checkpoint",
                )
            if batcher.drand_round_check_enabled:
                if hasattr(type(batcher), "observe_drand_round"):
                    drand_observation = batcher.observe_drand_round(
                        request.drand_round, t_arrival=t_arrival,
                    )
                else:
                    round_reject = batcher.validate_drand_round(
                        request.drand_round, t_arrival=t_arrival,
                    )
                    drand_observation = self._fallback_drand_observation(
                        request, batcher, round_reject,
                    )
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
                trigger_round is not None
                and request.drand_round > trigger_round
            ):
                return _cheap_reject(
                    RejectReason.BATCH_FILLED,
                    reject_stage="seal_extension",
                    batch_filled_reason="submitted_round_gt_seal_trigger_round",
                    current_valid_count=batcher.valid_count,
                    trigger_round=trigger_round,
                )
            if request.prompt_idx >= len(batcher.env):
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

            reserve_proof = getattr(
                type(batcher), "try_reserve_proof_admission", None
            )
            if reserve_proof is not None:
                # Runs in a thread: the canonical-prompt check calls
                # env.get_problem, which for the lazy parquet dataset may do a
                # blocking row-group fetch — must not stall the event loop.
                preflight_reason, preflight_stage = await asyncio.to_thread(
                    _proof_free_submission_reject, request, batcher
                )
                if preflight_reason is not None:
                    return _cheap_reject(
                        preflight_reason,
                        reject_stage=preflight_stage or "preflight",
                    )

                admitted, admission_reason = batcher.try_reserve_proof_admission(
                    request
                )
                if not admitted:
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

            # Under TestClient (no worker running) we run synchronously so
            # tests see the real ``ACCEPTED`` verdict; under uvicorn we enqueue
            # for the worker and return ``SUBMITTED`` — a distinct sentinel
            # that tells the miner the request is queued, not yet validated.
            # The real verdict (accept/reject post-GRAIL) surfaces in the
            # validator's logs and in the R2 archive. Expensive proof work is
            # bounded by ``try_reserve_proof_admission`` above; over-budget
            # submissions are rejected before they can enter this queue.
            if self._worker_task is None:
                telemetry.mark_proof_started()
                log_submission_stage(
                    logger,
                    logging.INFO,
                    "proof_started",
                    telemetry,
                    reject_stage=None,
                    reject_reason=None,
                )
                resp = self._call_accept_submission(batcher, request, telemetry)
                telemetry.refresh_from_batcher(batcher, at_decision=True)
                telemetry.mark_decision(verified=True)
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
                return resp

            telemetry.mark_enqueued()
            await self._submit_queue.put((request, batcher, telemetry))
            return BatchSubmissionResponse(
                accepted=True, reason=RejectReason.SUBMITTED,
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
                valid_submissions=batcher.valid_count,
                checkpoint_n=cp.checkpoint_n if cp else 0,
                checkpoint_repo_id=cp.repo_id if cp else None,
                checkpoint_revision=cp.revision if cp else None,
                randomness=batcher.randomness,
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

        return app

    async def _submit_worker(self) -> None:
        # Lazy import — keeps the module loadable in CPU-only test envs.
        from reliquary.validator.service import _try_empty_cuda_cache

        while True:
            try:
                item = await self._submit_queue.get()
            except asyncio.CancelledError:
                return
            if len(item) == 3:
                request, batcher, telemetry = item
            else:
                request, batcher = item
                telemetry = SubmitTelemetry.from_request(
                    request, t_arrival=time.time(),
                )
            telemetry.refresh_from_batcher(batcher)
            # Silently drop items whose batcher is no longer the active one.
            # This is what relieves pressure from a saturated window: the
            # queue is unbounded by design, so a busy window can pile up
            # dozens of pending items behind the in-flight GRAIL. As soon
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
            if batcher.is_sealed():
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
                try:
                    response = await asyncio.to_thread(
                        self._call_accept_submission, batcher, request, telemetry
                    )
                finally:
                    self._inflight_proofs = max(0, self._inflight_proofs - 1)
                telemetry.refresh_from_batcher(batcher, at_decision=True)
                telemetry.mark_decision(verified=True)
                log_submission_stage(
                    logger,
                    logging.INFO,
                    "proof_finished",
                    telemetry,
                    accepted=response.accepted,
                    reason=response.reason.value,
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
                # The verdict the /submit response *didn't* carry, now
                # observable to the miner via /verdicts.
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
                # Always reclaim activation memory after a forward pass so
                # back-to-back GRAIL verifies don't accumulate fragmentation.
                # The helper is a no-op on CPU-only hosts. Cost: ~ms; benefit:
                # prevents the multi-hour drift that took down the validator
                # on 2026-05-11.
                await asyncio.to_thread(_try_empty_cuda_cache)

    async def start(self) -> None:
        if self._task is not None:
            return
        config = uvicorn.Config(
            self.app, host=self.host, port=self.port,
            log_level="warning", access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())
        self._worker_task = asyncio.create_task(self._submit_worker())
        await asyncio.sleep(0)
        logger.info("Validator HTTP server listening on %s:%d", self.host, self.port)

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            self._worker_task = None
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except asyncio.TimeoutError:
                self._task.cancel()
            self._task = None
            self._server = None
