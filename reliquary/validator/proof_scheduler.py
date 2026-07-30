"""Deterministic, device-aware scheduling for expensive validator proofs.

The scheduler deliberately knows nothing about batchers, models, or CUDA. A
caller supplies already-ranked candidates and a synchronous proof function.
One long-lived worker owns each device, while a coordinator applies results in
rank order even when proofs finish out of order.
"""

from __future__ import annotations

from collections import defaultdict
from collections import deque
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from enum import Enum
import math
import threading
import time
from typing import Any


class SchedulerState(str, Enum):
    RUNNING = "running"
    DRAINING = "draining"
    QUIESCING = "quiescing"
    QUIESCED = "quiesced"
    CLOSED = "closed"


class ProofDecisionStatus(str, Enum):
    PASSED = "passed"
    REJECTED = "rejected"
    ERROR = "error"
    SKIPPED_PROMPT_CLAIMED = "skipped_prompt_claimed"
    SKIPPED_RESOURCE_LIMIT = "skipped_resource_limit"
    NOT_NEEDED = "not_needed"
    CAPACITY_ABORTED = "capacity_aborted"


class ProofPlanOutcome(str, Enum):
    COMPLETED = "completed"
    CAPACITY_ABORTED = "capacity_aborted"


class CapacityAbortReason(str, Enum):
    DEADLINE_EXCEEDED = "deadline_exceeded"
    INSUFFICIENT_DISTINCT_PROMPTS = "insufficient_distinct_prompts"
    ATTEMPT_LIMIT = "attempt_limit"
    QUIESCED = "quiesced"


class ProofSchedulerError(RuntimeError):
    """Base class for scheduler contract errors."""


class SchedulerNotRunning(ProofSchedulerError):
    pass


class CheckpointNotReady(ProofSchedulerError):
    pass


class DeviceNotReady(ProofSchedulerError):
    pass


@dataclass(frozen=True)
class RankedProof:
    """One candidate in the caller's authoritative economic rank order."""

    job_id: str
    rank: int
    prompt_key: Hashable
    payload: Any = field(repr=False)
    # A proof may consume several failure-debt identities (for example one
    # hotkey and one operator). Each pair is ``(identity, failure_limit)``.
    # The scheduler serializes active proofs sharing any identity so a failed
    # proof is applied before another candidate can bypass the same limit.
    resources: tuple[tuple[Hashable, int], ...] = ()


@dataclass(frozen=True)
class ProofPlan:
    """A ranked proof population for one environment."""

    plan_id: str
    environment: str
    checkpoint_revision: str
    candidates: Sequence[RankedProof]
    required_passes: int
    deadline_at: float
    max_attempts: int | None = None
    # Lower values dispatch first. Winner plans use zero; observational
    # forensics use a larger value and consume only otherwise-idle capacity.
    priority: int = 0
    # Winner selection normally stops at ``required_passes``. Forensic plans
    # instead prove every supplied candidate and ignore pass count.
    complete_all: bool = False
    # A genuinely sparse eligible population is not a capacity incident:
    # empty slots burn. Callers that require exact capacity can leave this off.
    allow_shortfall: bool = False


@dataclass(frozen=True)
class ProofInvocation:
    """Input passed to the dependency-injected synchronous proof callable."""

    device_id: str
    plan_id: str
    environment: str
    checkpoint_revision: str
    candidate: RankedProof
    deadline_at: float


@dataclass(frozen=True)
class ProofExecution:
    """Proof callable result. ``value`` is opaque to the scheduler."""

    passed: bool
    value: Any = field(default=None, repr=False)
    reason: str | None = None


@dataclass(frozen=True)
class ProofDecision:
    """One rank-ordered terminal decision."""

    job_id: str
    rank: int
    prompt_key: Hashable
    status: ProofDecisionStatus
    device_id: str | None
    started_at: float | None
    finished_at: float | None
    value: Any = field(default=None, repr=False)
    reason: str | None = None


@dataclass(frozen=True)
class ProofPlanResult:
    plan_id: str
    environment: str
    checkpoint_revision: str
    outcome: ProofPlanOutcome
    abort_reason: CapacityAbortReason | None
    decisions: tuple[ProofDecision, ...]
    winner_job_ids: tuple[str, ...]
    attempts_started: int
    submitted_at: float
    finished_at: float


class _JobPhase(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    RAW = "raw"
    SKIPPED = "skipped"
    RESOURCE_LIMIT = "resource_limit"
    NOT_NEEDED = "not_needed"
    ABORTED = "aborted"
    APPLIED = "applied"


@dataclass
class _RawCompletion:
    execution: ProofExecution | None
    device_id: str
    started_at: float
    finished_at: float
    error: str | None = None


@dataclass
class _PlanState:
    plan: ProofPlan
    candidates: tuple[RankedProof, ...]
    submitted_at: float
    max_attempts: int
    phases: dict[str, _JobPhase]
    candidate_by_id: dict[str, RankedProof]
    prompt_chains: dict[Hashable, tuple[str, ...]]
    prompt_positions: dict[Hashable, int]
    raw: dict[str, _RawCompletion] = field(default_factory=dict)
    decisions: list[ProofDecision] = field(default_factory=list)
    claimed_prompts: set[Hashable] = field(default_factory=set)
    resource_failures: dict[Hashable, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    active_job_ids: set[str] = field(default_factory=set)
    next_apply_index: int = 0
    attempts_started: int = 0
    passed: int = 0
    stop_dispatch: bool = False
    completion_reason: str | None = None
    abort_reason: CapacityAbortReason | None = None
    final_result: ProofPlanResult | None = None


class ProofPlanHandle:
    """Thread-safe view of one submitted plan."""

    def __init__(
        self,
        scheduler: "GlobalProofScheduler",
        state: _PlanState,
    ) -> None:
        self._scheduler = scheduler
        self._state = state

    @property
    def plan_id(self) -> str:
        return self._state.plan.plan_id

    def done(self) -> bool:
        with self._scheduler._condition:
            return self._state.final_result is not None

    def decisions(self) -> tuple[ProofDecision, ...]:
        """Return decisions emitted so far, always in ascending rank order."""
        with self._scheduler._condition:
            return tuple(self._state.decisions)

    def result(self, timeout: float | None = None) -> ProofPlanResult:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._scheduler._condition:
            while self._state.final_result is None:
                remaining = (
                    None if deadline is None else deadline - time.monotonic()
                )
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(
                        f"proof plan {self.plan_id!r} did not finish"
                    )
                self._scheduler._condition.wait(remaining)
            return self._state.final_result


ProofCallable = Callable[[ProofInvocation], ProofExecution | bool]


class GlobalProofScheduler:
    """Fair global proof scheduler with exactly one worker per device.

    Plans are submitted atomically with :meth:`submit_many`. Each environment
    has at most one live plan. Workers round-robin across configured
    environments, and candidates sharing a prompt are attempted serially in
    rank order. Completion can be parallel; decision application cannot.
    """

    def __init__(
        self,
        *,
        devices: Sequence[str],
        environments: Sequence[str],
        proof_callable: ProofCallable,
        checkpoint_revision: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        deadline_poll_seconds: float = 0.05,
    ) -> None:
        if not devices or len(set(devices)) != len(devices):
            raise ValueError("devices must be non-empty and unique")
        if not environments or len(set(environments)) != len(environments):
            raise ValueError("environments must be non-empty and unique")
        if deadline_poll_seconds <= 0:
            raise ValueError("deadline_poll_seconds must be positive")

        self._devices = tuple(devices)
        self._environments = tuple(environments)
        self._environment_set = set(environments)
        self._proof_callable = proof_callable
        self._clock = clock
        self._deadline_poll_seconds = deadline_poll_seconds
        self._condition = threading.Condition(threading.RLock())
        self._shutdown = False
        self._state = (
            SchedulerState.RUNNING
            if checkpoint_revision is not None
            else SchedulerState.QUIESCED
        )
        self._active_checkpoint_revision = checkpoint_revision
        self._device_revisions: dict[str, str | None] = {
            device: checkpoint_revision for device in self._devices
        }
        self._active_by_device: dict[
            str, tuple[_PlanState, str, float] | None
        ] = {device: None for device in self._devices}
        self._active_resource_keys: set[Hashable] = set()
        self._plans: dict[str, _PlanState] = {}
        self._active_plan_by_environment: dict[
            str, _PlanState | None
        ] = {environment: None for environment in self._environments}
        self._environment_cursor = 0
        self._totals: dict[str, int] = defaultdict(int)
        self._dispatches_by_environment: dict[str, int] = defaultdict(int)
        self._proof_seconds_by_device: dict[str, float] = defaultdict(float)
        self._proof_durations_by_environment: dict[str, deque[float]] = {
            environment: deque(maxlen=1000)
            for environment in self._environments
        }

        self._workers = [
            threading.Thread(
                target=self._worker_loop,
                args=(device,),
                name=f"proof-device-{device}",
                daemon=True,
            )
            for device in self._devices
        ]
        self._deadline_thread = threading.Thread(
            target=self._deadline_loop,
            name="proof-deadline-monitor",
            daemon=True,
        )
        for worker in self._workers:
            worker.start()
        self._deadline_thread.start()

    def __enter__(self) -> "GlobalProofScheduler":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @property
    def state(self) -> SchedulerState:
        with self._condition:
            return self._state

    @property
    def active_checkpoint_revision(self) -> str | None:
        with self._condition:
            return self._active_checkpoint_revision

    def checkpoint_ready(self, checkpoint_revision: str) -> bool:
        with self._condition:
            return (
                self._active_checkpoint_revision == checkpoint_revision
                and all(
                    revision == checkpoint_revision
                    for revision in self._device_revisions.values()
                )
            )

    def submit(self, plan: ProofPlan) -> ProofPlanHandle:
        return self.submit_many((plan,))[0]

    def submit_many(
        self, plans: Sequence[ProofPlan]
    ) -> tuple[ProofPlanHandle, ...]:
        """Validate and enqueue multiple environments atomically."""
        if not plans:
            return ()
        with self._condition:
            if self._state is not SchedulerState.RUNNING:
                raise SchedulerNotRunning(
                    f"scheduler is {self._state.value}, not running"
                )
            self._validate_plan_batch_locked(plans)

            states: list[_PlanState] = []
            now = self._clock()
            for plan in plans:
                state = self._build_plan_state(plan, now)
                self._plans[plan.plan_id] = state
                self._active_plan_by_environment[plan.environment] = state
                self._totals["plans_submitted"] += 1
                self._totals["jobs_submitted"] += len(state.candidates)
                states.append(state)

            for state in states:
                self._reevaluate_plan_locked(state)
            self._condition.notify_all()
            return tuple(ProofPlanHandle(self, state) for state in states)

    def expire_deadlines(self) -> None:
        """Synchronously evaluate deadlines, useful for service polls and tests."""
        with self._condition:
            self._expire_deadlines_locked()
            self._condition.notify_all()

    def drain(self, timeout: float | None = None) -> bool:
        """Stop accepting plans and finish all accepted work."""
        with self._condition:
            if self._state is SchedulerState.CLOSED:
                return True
            if self._state is SchedulerState.RUNNING:
                self._state = SchedulerState.DRAINING
                self._totals["drains"] += 1
            elif self._state not in (
                SchedulerState.DRAINING,
                SchedulerState.QUIESCED,
            ):
                raise ProofSchedulerError(
                    f"cannot drain while {self._state.value}"
                )
            self._maybe_finish_lifecycle_locked()
            self._condition.notify_all()
            return self._wait_for_quiesced_locked(timeout)

    def quiesce(self, timeout: float | None = None) -> bool:
        """Stop dispatch, abort queued plans, and wait for active calls to return."""
        with self._condition:
            if self._state is SchedulerState.CLOSED:
                return True
            if self._state is SchedulerState.QUIESCED:
                return True
            self._state = SchedulerState.QUIESCING
            self._totals["quiesces"] += 1
            for state in self._unfinished_plans_locked():
                self._abort_plan_locked(state, CapacityAbortReason.QUIESCED)
            self._maybe_finish_lifecycle_locked()
            self._condition.notify_all()
            return self._wait_for_quiesced_locked(timeout)

    def mark_device_ready(
        self, device_id: str, checkpoint_revision: str
    ) -> None:
        with self._condition:
            self._require_quiesced_locked()
            if device_id not in self._device_revisions:
                raise KeyError(f"unknown proof device {device_id!r}")
            self._device_revisions[device_id] = checkpoint_revision

    def mark_device_not_ready(self, device_id: str) -> None:
        with self._condition:
            self._require_quiesced_locked()
            if device_id not in self._device_revisions:
                raise KeyError(f"unknown proof device {device_id!r}")
            self._device_revisions[device_id] = None

    def resume(self, checkpoint_revision: str) -> None:
        with self._condition:
            self._require_quiesced_locked()
            unready = [
                device
                for device, revision in self._device_revisions.items()
                if revision != checkpoint_revision
            ]
            if unready:
                raise DeviceNotReady(
                    "devices are not ready for checkpoint "
                    f"{checkpoint_revision!r}: {', '.join(unready)}"
                )
            self._active_checkpoint_revision = checkpoint_revision
            self._state = SchedulerState.RUNNING
            self._totals["resumes"] += 1
            self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        """Return a compact JSON-safe scheduler health snapshot."""
        with self._condition:
            now = self._clock()
            queue_depth = {}
            buffered = {}
            oldest_submitted: float | None = None
            for environment in self._environments:
                state = self._active_plan_by_environment[environment]
                if state is None:
                    queue_depth[environment] = 0
                    buffered[environment] = 0
                    continue
                queue_depth[environment] = sum(
                    phase is _JobPhase.PENDING
                    for phase in state.phases.values()
                )
                buffered[environment] = sum(
                    phase is _JobPhase.RAW
                    for phase in state.phases.values()
                )
                if queue_depth[environment]:
                    oldest_submitted = (
                        state.submitted_at
                        if oldest_submitted is None
                        else min(oldest_submitted, state.submitted_at)
                    )

            active = {
                device: (
                    None
                    if entry is None
                    else {
                        "plan_id": entry[0].plan.plan_id,
                        "environment": entry[0].plan.environment,
                        "job_id": entry[1],
                        "age_seconds": max(0.0, now - entry[2]),
                    }
                )
                for device, entry in self._active_by_device.items()
            }

            def _latency_summary(values: deque[float]) -> dict[str, Any]:
                if not values:
                    return {
                        "count": 0,
                        "p50": None,
                        "p95": None,
                        "max": None,
                    }
                ordered = sorted(values)

                def _percentile(fraction: float) -> float:
                    index = math.ceil(fraction * len(ordered)) - 1
                    return ordered[max(0, min(index, len(ordered) - 1))]

                return {
                    "count": len(ordered),
                    "p50": _percentile(0.50),
                    "p95": _percentile(0.95),
                    "max": ordered[-1],
                }

            return {
                "state": self._state.value,
                "active_checkpoint_revision": (
                    self._active_checkpoint_revision
                ),
                "device_revisions": dict(self._device_revisions),
                "queue_depth_by_environment": queue_depth,
                "buffered_results_by_environment": buffered,
                "active_by_device": active,
                "active_resource_count": len(self._active_resource_keys),
                "oldest_queued_age_seconds": (
                    None
                    if oldest_submitted is None
                    else max(0.0, now - oldest_submitted)
                ),
                "dispatches_by_environment": {
                    environment: self._dispatches_by_environment[environment]
                    for environment in self._environments
                },
                "proof_seconds_by_device": {
                    device: self._proof_seconds_by_device[device]
                    for device in self._devices
                },
                "proof_latency_seconds_by_environment": {
                    environment: _latency_summary(
                        self._proof_durations_by_environment[environment]
                    )
                    for environment in self._environments
                },
                "totals": dict(self._totals),
            }

    def close(self, timeout: float | None = 5.0) -> bool:
        """Abort pending work, stop workers, and join scheduler threads."""
        started = time.monotonic()
        quiesced = self.quiesce(timeout=timeout)
        remaining = None
        if timeout is not None:
            remaining = max(0.0, timeout - (time.monotonic() - started))
        with self._condition:
            self._state = SchedulerState.CLOSED
            self._shutdown = True
            self._condition.notify_all()
        threads = [*self._workers, self._deadline_thread]
        for thread in threads:
            thread.join(remaining)
            if timeout is not None:
                remaining = max(
                    0.0, timeout - (time.monotonic() - started)
                )
        return quiesced and all(not thread.is_alive() for thread in threads)

    def _validate_plan_batch_locked(
        self, plans: Sequence[ProofPlan]
    ) -> None:
        plan_ids: set[str] = set()
        environments: set[str] = set()
        for plan in plans:
            if not plan.plan_id:
                raise ValueError("plan_id must be non-empty")
            if plan.plan_id in plan_ids or plan.plan_id in self._plans:
                raise ValueError(f"duplicate plan_id {plan.plan_id!r}")
            plan_ids.add(plan.plan_id)
            if plan.environment not in self._environment_set:
                raise ValueError(
                    f"unknown environment {plan.environment!r}"
                )
            if plan.environment in environments:
                raise ValueError(
                    f"multiple plans for {plan.environment!r} in one batch"
                )
            environments.add(plan.environment)
            if self._active_plan_by_environment[plan.environment] is not None:
                raise ValueError(
                    f"environment {plan.environment!r} already has a plan"
                )
            if plan.checkpoint_revision != self._active_checkpoint_revision:
                raise CheckpointNotReady(
                    f"plan {plan.plan_id!r} requires "
                    f"{plan.checkpoint_revision!r}, scheduler has "
                    f"{self._active_checkpoint_revision!r}"
                )
            if not self.checkpoint_ready(plan.checkpoint_revision):
                raise CheckpointNotReady(
                    f"not all devices are ready for "
                    f"{plan.checkpoint_revision!r}"
                )
            self._validate_plan(plan)

    @staticmethod
    def _validate_plan(plan: ProofPlan) -> None:
        if plan.required_passes < 0:
            raise ValueError("required_passes cannot be negative")
        if plan.complete_all and plan.required_passes != 0:
            raise ValueError(
                "complete_all plans must set required_passes to zero"
            )
        if plan.max_attempts is not None and plan.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if plan.priority < 0:
            raise ValueError("priority cannot be negative")
        ranks: set[int] = set()
        job_ids: set[str] = set()
        for candidate in plan.candidates:
            if not candidate.job_id:
                raise ValueError("job_id must be non-empty")
            if candidate.job_id in job_ids:
                raise ValueError(
                    f"duplicate job_id {candidate.job_id!r}"
                )
            job_ids.add(candidate.job_id)
            if candidate.rank < 0 or candidate.rank in ranks:
                raise ValueError(
                    "candidate ranks must be non-negative and unique"
                )
            ranks.add(candidate.rank)
            try:
                hash(candidate.prompt_key)
            except TypeError as exc:
                raise ValueError("prompt_key must be hashable") from exc
            resource_keys: set[Hashable] = set()
            for resource_key, failure_limit in candidate.resources:
                try:
                    hash(resource_key)
                except TypeError as exc:
                    raise ValueError(
                        "proof resource identity must be hashable"
                    ) from exc
                if resource_key in resource_keys:
                    raise ValueError(
                        "proof resource identities must be unique per candidate"
                    )
                resource_keys.add(resource_key)
                if failure_limit <= 0:
                    raise ValueError(
                        "proof resource failure limits must be positive"
                    )

    @staticmethod
    def _build_plan_state(plan: ProofPlan, now: float) -> _PlanState:
        candidates = tuple(sorted(plan.candidates, key=lambda item: item.rank))
        candidate_by_id = {
            candidate.job_id: candidate for candidate in candidates
        }
        prompt_lists: dict[Hashable, list[str]] = defaultdict(list)
        for candidate in candidates:
            prompt_lists[candidate.prompt_key].append(candidate.job_id)
        max_attempts = (
            len(candidates)
            if plan.max_attempts is None
            else min(plan.max_attempts, len(candidates))
        )
        return _PlanState(
            plan=plan,
            candidates=candidates,
            submitted_at=now,
            max_attempts=max_attempts,
            phases={
                candidate.job_id: _JobPhase.PENDING
                for candidate in candidates
            },
            candidate_by_id=candidate_by_id,
            prompt_chains={
                prompt: tuple(job_ids)
                for prompt, job_ids in prompt_lists.items()
            },
            prompt_positions={prompt: 0 for prompt in prompt_lists},
        )

    def _worker_loop(self, device_id: str) -> None:
        while True:
            with self._condition:
                invocation = None
                state = None
                while invocation is None:
                    if self._shutdown:
                        return
                    self._expire_deadlines_locked()
                    selected = self._take_next_job_locked(device_id)
                    if selected is not None:
                        state, candidate = selected
                        started_at = self._clock()
                        state.phases[candidate.job_id] = _JobPhase.ACTIVE
                        state.active_job_ids.add(candidate.job_id)
                        self._active_resource_keys.update(
                            key for key, _limit in candidate.resources
                        )
                        state.attempts_started += 1
                        self._active_by_device[device_id] = (
                            state,
                            candidate.job_id,
                            started_at,
                        )
                        self._totals["jobs_started"] += 1
                        self._dispatches_by_environment[
                            state.plan.environment
                        ] += 1
                        invocation = ProofInvocation(
                            device_id=device_id,
                            plan_id=state.plan.plan_id,
                            environment=state.plan.environment,
                            checkpoint_revision=(
                                state.plan.checkpoint_revision
                            ),
                            candidate=candidate,
                            deadline_at=state.plan.deadline_at,
                        )
                        break
                    self._condition.wait(self._deadline_poll_seconds)

            execution: ProofExecution | None = None
            error: str | None = None
            try:
                returned = self._proof_callable(invocation)
                execution = (
                    ProofExecution(passed=returned)
                    if isinstance(returned, bool)
                    else returned
                )
                if not isinstance(execution, ProofExecution):
                    raise TypeError(
                        "proof_callable must return ProofExecution or bool"
                    )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            finished_at = self._clock()

            with self._condition:
                assert state is not None
                job_id = invocation.candidate.job_id
                self._active_by_device[device_id] = None
                state.active_job_ids.discard(job_id)
                state.raw[job_id] = _RawCompletion(
                    execution=execution,
                    device_id=device_id,
                    started_at=started_at,
                    finished_at=finished_at,
                    error=error,
                )
                state.phases[job_id] = _JobPhase.RAW
                self._totals["jobs_completed"] += 1
                self._proof_seconds_by_device[device_id] += max(
                    0.0, finished_at - started_at
                )
                self._proof_durations_by_environment[
                    state.plan.environment
                ].append(max(0.0, finished_at - started_at))
                if error is not None:
                    self._totals["proof_errors"] += 1
                if (
                    state.abort_reason is not None
                    or (
                        not state.plan.complete_all
                        and state.passed >= state.plan.required_passes
                    )
                ):
                    self._totals["late_results"] += 1
                self._reevaluate_plan_locked(state)
                self._maybe_finish_lifecycle_locked()
                self._condition.notify_all()

    def _deadline_loop(self) -> None:
        while True:
            with self._condition:
                if self._shutdown:
                    return
                self._expire_deadlines_locked()
                self._condition.wait(self._deadline_poll_seconds)

    def _take_next_job_locked(
        self, _device_id: str
    ) -> tuple[_PlanState, RankedProof] | None:
        if self._state not in (
            SchedulerState.RUNNING,
            SchedulerState.DRAINING,
        ):
            return None
        priorities = sorted({
            state.plan.priority
            for state in self._active_plan_by_environment.values()
            if state is not None
        })
        count = len(self._environments)
        for priority in priorities:
            for offset in range(count):
                index = (self._environment_cursor + offset) % count
                environment = self._environments[index]
                state = self._active_plan_by_environment[environment]
                if state is None or state.plan.priority != priority:
                    continue
                candidate = self._next_candidate_locked(state)
                if candidate is None:
                    continue
                self._environment_cursor = (index + 1) % count
                return state, candidate
        return None

    def _next_candidate_locked(
        self, state: _PlanState
    ) -> RankedProof | None:
        if (
            state.stop_dispatch
            or state.final_result is not None
            or state.attempts_started >= state.max_attempts
        ):
            return None
        if not state.plan.complete_all:
            outstanding = sum(
                phase in (_JobPhase.ACTIVE, _JobPhase.RAW)
                for phase in state.phases.values()
            )
            if (
                state.passed + outstanding
                >= state.plan.required_passes
            ):
                return None
        eligible: list[RankedProof] = []
        for prompt, chain in state.prompt_chains.items():
            if prompt in state.claimed_prompts:
                continue
            position = state.prompt_positions[prompt]
            if position >= len(chain):
                continue
            job_id = chain[position]
            if state.phases[job_id] is not _JobPhase.PENDING:
                continue
            candidate = state.candidate_by_id[job_id]
            exhausted = any(
                state.resource_failures[resource_key] >= failure_limit
                for resource_key, failure_limit in candidate.resources
            )
            if exhausted:
                state.phases[job_id] = _JobPhase.RESOURCE_LIMIT
                continue
            if any(
                resource_key in self._active_resource_keys
                for resource_key, _failure_limit in candidate.resources
            ):
                continue
            eligible.append(candidate)
        if any(
            phase is _JobPhase.RESOURCE_LIMIT
            for phase in state.phases.values()
        ):
            self._apply_ready_locked(state)
            if state.final_result is None:
                return self._next_candidate_locked(state)
        return min(eligible, key=lambda item: item.rank) if eligible else None

    def _reevaluate_plan_locked(self, state: _PlanState) -> None:
        if state.final_result is not None:
            return
        self._apply_ready_locked(state)
        if state.final_result is not None:
            return
        if state.abort_reason is None and self._clock() >= state.plan.deadline_at:
            self._abort_plan_locked(
                state, CapacityAbortReason.DEADLINE_EXCEEDED
            )
            return
        if state.abort_reason is not None:
            self._apply_ready_locked(state)
            return
        if (
            not state.plan.complete_all
            and state.passed >= state.plan.required_passes
        ):
            self._stop_after_target_locked(state)
            self._apply_ready_locked(state)
            return

        if state.plan.complete_all:
            self._finalize_if_terminal_locked(state)
            if state.final_result is not None:
                return
            if (
                state.attempts_started >= state.max_attempts
                and not state.active_job_ids
                and not any(
                    phase is _JobPhase.RAW
                    for phase in state.phases.values()
                )
            ):
                self._abort_plan_locked(
                    state, CapacityAbortReason.ATTEMPT_LIMIT
                )
            return

        possible_prompts = 0
        for prompt, chain in state.prompt_chains.items():
            if prompt in state.claimed_prompts:
                continue
            if any(
                state.phases[job_id]
                in (_JobPhase.PENDING, _JobPhase.ACTIVE, _JobPhase.RAW)
                for job_id in chain
            ):
                possible_prompts += 1
        if state.passed + possible_prompts < state.plan.required_passes:
            if state.plan.allow_shortfall:
                if possible_prompts == 0:
                    self._stop_with_shortfall_locked(state)
                    self._apply_ready_locked(state)
                return
            self._abort_plan_locked(
                state, CapacityAbortReason.INSUFFICIENT_DISTINCT_PROMPTS
            )
            return
        if (
            state.attempts_started >= state.max_attempts
            and not state.active_job_ids
            and not any(
                phase is _JobPhase.RAW for phase in state.phases.values()
            )
        ):
            self._abort_plan_locked(
                state, CapacityAbortReason.ATTEMPT_LIMIT
            )

    def _apply_ready_locked(self, state: _PlanState) -> None:
        while state.next_apply_index < len(state.candidates):
            candidate = state.candidates[state.next_apply_index]
            phase = state.phases[candidate.job_id]
            decision: ProofDecision | None = None

            if phase is _JobPhase.RAW:
                raw = state.raw.pop(candidate.job_id)
                if state.abort_reason is not None:
                    decision = self._synthetic_decision(
                        candidate,
                        ProofDecisionStatus.CAPACITY_ABORTED,
                        reason=state.abort_reason.value,
                        raw=raw,
                    )
                elif (
                    not state.plan.complete_all
                    and state.passed >= state.plan.required_passes
                ):
                    decision = self._synthetic_decision(
                        candidate,
                        ProofDecisionStatus.NOT_NEEDED,
                        reason="target_reached",
                        raw=raw,
                    )
                elif raw.error is not None:
                    decision = self._synthetic_decision(
                        candidate,
                        ProofDecisionStatus.ERROR,
                        reason=raw.error,
                        raw=raw,
                    )
                elif raw.execution is not None and raw.execution.passed:
                    decision = self._synthetic_decision(
                        candidate,
                        ProofDecisionStatus.PASSED,
                        reason=raw.execution.reason,
                        value=raw.execution.value,
                        raw=raw,
                    )
                else:
                    decision = self._synthetic_decision(
                        candidate,
                        ProofDecisionStatus.REJECTED,
                        reason=(
                            raw.execution.reason
                            if raw.execution is not None
                            else None
                        ),
                        value=(
                            raw.execution.value
                            if raw.execution is not None
                            else None
                        ),
                        raw=raw,
                    )
            elif phase is _JobPhase.SKIPPED:
                decision = self._synthetic_decision(
                    candidate,
                    ProofDecisionStatus.SKIPPED_PROMPT_CLAIMED,
                    reason="higher_ranked_candidate_passed",
                )
            elif phase is _JobPhase.RESOURCE_LIMIT:
                decision = self._synthetic_decision(
                    candidate,
                    ProofDecisionStatus.SKIPPED_RESOURCE_LIMIT,
                    reason="proof_failure_debt",
                )
            elif phase is _JobPhase.NOT_NEEDED:
                decision = self._synthetic_decision(
                    candidate,
                    ProofDecisionStatus.NOT_NEEDED,
                    reason=state.completion_reason or "target_reached",
                )
            elif phase is _JobPhase.ABORTED:
                decision = self._synthetic_decision(
                    candidate,
                    ProofDecisionStatus.CAPACITY_ABORTED,
                    reason=(
                        state.abort_reason.value
                        if state.abort_reason is not None
                        else "capacity_aborted"
                    ),
                )

            if decision is None:
                break

            state.decisions.append(decision)
            state.phases[candidate.job_id] = _JobPhase.APPLIED
            state.next_apply_index += 1
            if phase is _JobPhase.RAW:
                for resource_key, _failure_limit in candidate.resources:
                    self._active_resource_keys.discard(resource_key)
            self._totals[f"decisions_{decision.status.value}"] += 1

            if decision.status is ProofDecisionStatus.PASSED:
                state.passed += 1
                state.claimed_prompts.add(candidate.prompt_key)
                self._skip_prompt_tail_locked(state, candidate.prompt_key)
            elif decision.status in (
                ProofDecisionStatus.REJECTED,
                ProofDecisionStatus.ERROR,
            ):
                for resource_key, _failure_limit in candidate.resources:
                    state.resource_failures[resource_key] += 1
                state.prompt_positions[candidate.prompt_key] += 1
            elif decision.status is ProofDecisionStatus.SKIPPED_RESOURCE_LIMIT:
                state.prompt_positions[candidate.prompt_key] += 1

            if (
                state.abort_reason is None
                and not state.plan.complete_all
                and state.passed >= state.plan.required_passes
            ):
                self._stop_after_target_locked(state)

        self._finalize_if_terminal_locked(state)

    def _skip_prompt_tail_locked(
        self, state: _PlanState, prompt_key: Hashable
    ) -> None:
        for job_id in state.prompt_chains[prompt_key]:
            if state.phases[job_id] is _JobPhase.PENDING:
                state.phases[job_id] = _JobPhase.SKIPPED

    def _stop_after_target_locked(self, state: _PlanState) -> None:
        state.stop_dispatch = True
        state.completion_reason = "target_reached"
        for job_id, phase in state.phases.items():
            if phase is _JobPhase.PENDING:
                state.phases[job_id] = _JobPhase.NOT_NEEDED

    def _stop_with_shortfall_locked(self, state: _PlanState) -> None:
        state.stop_dispatch = True
        state.completion_reason = "insufficient_distinct_prompts"
        for job_id, phase in state.phases.items():
            if phase is _JobPhase.PENDING:
                state.phases[job_id] = _JobPhase.NOT_NEEDED

    def _abort_plan_locked(
        self, state: _PlanState, reason: CapacityAbortReason
    ) -> None:
        if state.final_result is not None or state.abort_reason is not None:
            return
        state.abort_reason = reason
        state.stop_dispatch = True
        self._totals["capacity_aborts"] += 1
        self._totals[f"capacity_aborts_{reason.value}"] += 1
        for job_id, phase in state.phases.items():
            if phase in (
                _JobPhase.PENDING,
                _JobPhase.SKIPPED,
                _JobPhase.NOT_NEEDED,
            ):
                state.phases[job_id] = _JobPhase.ABORTED
        self._apply_ready_locked(state)

    def _finalize_if_terminal_locked(self, state: _PlanState) -> None:
        if (
            state.final_result is not None
            or state.next_apply_index != len(state.candidates)
            or state.active_job_ids
        ):
            return
        if (
            state.abort_reason is None
            and not state.plan.complete_all
            and state.passed < state.plan.required_passes
        ):
            if state.plan.allow_shortfall:
                state.completion_reason = "insufficient_distinct_prompts"
            else:
                state.abort_reason = (
                    CapacityAbortReason.INSUFFICIENT_DISTINCT_PROMPTS
                )
                self._totals["capacity_aborts"] += 1
                self._totals[
                    "capacity_aborts_insufficient_distinct_prompts"
                ] += 1
        outcome = (
            ProofPlanOutcome.CAPACITY_ABORTED
            if state.abort_reason is not None
            else ProofPlanOutcome.COMPLETED
        )
        state.final_result = ProofPlanResult(
            plan_id=state.plan.plan_id,
            environment=state.plan.environment,
            checkpoint_revision=state.plan.checkpoint_revision,
            outcome=outcome,
            abort_reason=state.abort_reason,
            decisions=tuple(state.decisions),
            winner_job_ids=tuple(
                decision.job_id
                for decision in state.decisions
                if decision.status is ProofDecisionStatus.PASSED
            ),
            attempts_started=state.attempts_started,
            submitted_at=state.submitted_at,
            finished_at=self._clock(),
        )
        self._active_plan_by_environment[state.plan.environment] = None
        self._totals["plans_completed"] += 1
        self._totals[f"plans_{outcome.value}"] += 1
        self._condition.notify_all()

    @staticmethod
    def _synthetic_decision(
        candidate: RankedProof,
        status: ProofDecisionStatus,
        *,
        reason: str | None,
        value: Any = None,
        raw: _RawCompletion | None = None,
    ) -> ProofDecision:
        return ProofDecision(
            job_id=candidate.job_id,
            rank=candidate.rank,
            prompt_key=candidate.prompt_key,
            status=status,
            device_id=None if raw is None else raw.device_id,
            started_at=None if raw is None else raw.started_at,
            finished_at=None if raw is None else raw.finished_at,
            value=value,
            reason=reason,
        )

    def _expire_deadlines_locked(self) -> None:
        now = self._clock()
        for state in self._unfinished_plans_locked():
            if now >= state.plan.deadline_at:
                self._abort_plan_locked(
                    state, CapacityAbortReason.DEADLINE_EXCEEDED
                )
        self._maybe_finish_lifecycle_locked()

    def _unfinished_plans_locked(self) -> list[_PlanState]:
        return [
            state
            for state in self._plans.values()
            if state.final_result is None
        ]

    def _maybe_finish_lifecycle_locked(self) -> None:
        if self._state not in (
            SchedulerState.DRAINING,
            SchedulerState.QUIESCING,
        ):
            return
        if self._unfinished_plans_locked():
            return
        if any(entry is not None for entry in self._active_by_device.values()):
            return
        self._state = SchedulerState.QUIESCED
        self._condition.notify_all()

    def _wait_for_quiesced_locked(self, timeout: float | None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._state is not SchedulerState.QUIESCED:
            remaining = (
                None if deadline is None else deadline - time.monotonic()
            )
            if remaining is not None and remaining <= 0:
                return False
            self._condition.wait(remaining)
        return True

    def _require_quiesced_locked(self) -> None:
        if self._state is not SchedulerState.QUIESCED:
            raise ProofSchedulerError(
                f"scheduler must be quiesced, is {self._state.value}"
            )
        if any(entry is not None for entry in self._active_by_device.values()):
            raise ProofSchedulerError("proof devices are still active")
