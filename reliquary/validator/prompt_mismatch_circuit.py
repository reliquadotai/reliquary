"""Persistent admission circuit breaker for incompatible prompt renderers.

Prompt binding is deterministic: a miner either submitted the exact canonical
prompt tokens or it did not. Repeated ``PROMPT_MISMATCH`` outcomes therefore
provide a cheap compatibility signal before another large reveal is accepted.

The breaker uses a validator-owned namespace and tracks both hotkey and
metagraph operator. Three mismatches inside a rolling window arm a cooldown.
After it elapses, exactly one signed precommit becomes the recovery canary; only
that exact receipt may clear or escalate the armed scope. This signature-bound
transition makes completion order irrelevant when admission workers run in
parallel.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from reliquary.constants import (
    PROMPT_MISMATCH_CIRCUIT_COOLDOWN_WINDOWS,
    PROMPT_MISMATCH_CIRCUIT_ENABLED,
    PROMPT_MISMATCH_CIRCUIT_FAILURE_THRESHOLD,
    PROMPT_MISMATCH_CIRCUIT_FAILURE_WINDOW_WINDOWS,
)

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 2
_VALID_SCOPES = frozenset({"hotkey", "operator"})
_StateKey = tuple[str, str, str]


@dataclass(frozen=True, order=True)
class _OutcomeOrder:
    """Stable receipt-arrival order carried into an asynchronous outcome."""

    window: int
    arrival_ts: float
    signature: str

    def to_json(self) -> dict[str, object]:
        return {
            "window": self.window,
            "arrival_ts": self.arrival_ts,
            "signature": self.signature,
        }

    @classmethod
    def from_json(cls, value: object) -> _OutcomeOrder:
        if not isinstance(value, dict):
            raise TypeError("outcome order must be an object")
        window = int(value["window"])
        arrival_ts = float(value["arrival_ts"])
        signature = str(value["signature"]).strip()
        if not math.isfinite(arrival_ts) or not signature:
            raise ValueError("invalid outcome order")
        return cls(window=window, arrival_ts=arrival_ts, signature=signature)


@dataclass(frozen=True)
class PromptMismatchDecision:
    allowed: bool
    canary: bool = False
    blocked_scopes: tuple[str, ...] = ()
    canary_scopes: tuple[str, ...] = ()
    retry_after_window: int | None = None
    status: str = "allowed"


@dataclass(frozen=True)
class PromptMismatchUpdate:
    activated_scopes: tuple[str, ...] = ()
    escalated_scopes: tuple[str, ...] = ()
    cleared_scopes: tuple[str, ...] = ()
    cooldown_until_window: int | None = None


@dataclass
class _CircuitState:
    failure_events: dict[str, _OutcomeOrder] = field(default_factory=dict)
    cooldown_level: int = -1
    cooldown_until_window: int = 0
    probe_signature: str | None = None
    probe_window: int | None = None
    recovery_order: _OutcomeOrder | None = None
    last_window: int = 0
    mismatch_count: int = 0


class PromptMismatchCircuitBreaker:
    """Track prompt compatibility and gate signed upload precommits.

    ``namespace`` must be owned by the validator, not copied from a request. In
    production it fingerprints the run, network, generation contract and prompt
    sources. ``state_path=None`` keeps state in memory for isolated servers and
    tests.

    Persistence is best-effort and fail-open. State transitions only schedule a
    coalesced background snapshot, so filesystem latency never blocks FastAPI's
    event loop or an admission worker coroutine.
    """

    def __init__(
        self,
        state_path: str | os.PathLike[str] | None = None,
        *,
        namespace: str = "default",
        enabled: bool = PROMPT_MISMATCH_CIRCUIT_ENABLED,
        failure_threshold: int = PROMPT_MISMATCH_CIRCUIT_FAILURE_THRESHOLD,
        failure_window_windows: int = (PROMPT_MISMATCH_CIRCUIT_FAILURE_WINDOW_WINDOWS),
        cooldown_windows: tuple[int, ...] = PROMPT_MISMATCH_CIRCUIT_COOLDOWN_WINDOWS,
    ) -> None:
        normalized_namespace = str(namespace).strip()
        if not normalized_namespace:
            raise ValueError("namespace must not be empty")
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if failure_window_windows <= 0:
            raise ValueError("failure_window_windows must be positive")
        if not cooldown_windows or any(value <= 0 for value in cooldown_windows):
            raise ValueError("cooldown_windows must contain positive values")

        self.namespace = normalized_namespace
        self.enabled = bool(enabled)
        self.failure_threshold = int(failure_threshold)
        self.failure_window_windows = int(failure_window_windows)
        self.cooldown_windows = tuple(int(value) for value in cooldown_windows)
        self.state_path = Path(state_path) if state_path is not None else None
        self._states: dict[_StateKey, _CircuitState] = {}
        self._lock = threading.RLock()
        self._persistence_condition = threading.Condition(self._lock)
        self._persistence_generation = 0
        self._persisted_generation = 0
        self._persistence_thread: threading.Thread | None = None
        self._persistence_stopping = False

        self._mismatches_total = 0
        self._binding_successes_total = 0
        self._cooldowns_started_total = 0
        self._cooldowns_escalated_total = 0
        self._recoveries_total = 0
        self._precommit_rejects_total = 0
        self._stale_outcomes_ignored_total = 0
        self._duplicate_outcomes_ignored_total = 0
        self._noncanary_successes_ignored_total = 0
        self._last_load_error: str | None = None
        self._last_persistence_error: str | None = None
        self._state_reset_reason: str | None = None
        self._load()

    @staticmethod
    def _subjects(identities: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
        subjects: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for scope, identity in identities.items():
            normalized_scope = str(scope).strip()
            normalized_identity = str(identity).strip()
            subject = (normalized_scope, normalized_identity)
            if (
                normalized_scope not in _VALID_SCOPES
                or not normalized_identity
                or subject in seen
            ):
                continue
            seen.add(subject)
            subjects.append(subject)
        return tuple(subjects)

    @staticmethod
    def _key(environment: str, scope: str, identity: str) -> _StateKey:
        return (str(environment).strip(), scope, identity)

    @staticmethod
    def _outcome_order(
        *,
        window: int,
        precommit_arrival_ts: float,
        precommit_signature: str,
    ) -> _OutcomeOrder:
        arrival_ts = float(precommit_arrival_ts)
        signature = str(precommit_signature).strip()
        if not math.isfinite(arrival_ts) or not signature:
            raise ValueError(
                "outcome requires a finite arrival timestamp and signature"
            )
        return _OutcomeOrder(
            window=int(window),
            arrival_ts=arrival_ts,
            signature=signature,
        )

    def admit_precommit(
        self,
        *,
        environment: str,
        identities: Mapping[str, str],
        window: int,
        precommit_signature: str,
    ) -> PromptMismatchDecision:
        """Admit normally, reject during cooldown, or reserve one canary."""
        if not self.enabled:
            return PromptMismatchDecision(allowed=True, status="disabled")

        current_window = int(window)
        signature = str(precommit_signature).strip()
        subjects = self._subjects(identities)
        with self._lock:
            pruned = self._prune_locked(current_window)
            blocked: list[tuple[str, _CircuitState, str]] = []
            canaries: list[tuple[_StateKey, str, _CircuitState]] = []
            for scope, identity in subjects:
                key = self._key(environment, scope, identity)
                state = self._states.get(key)
                if state is None or state.cooldown_level < 0:
                    continue
                if current_window < state.cooldown_until_window:
                    blocked.append((scope, state, "cooldown"))
                    continue
                if state.probe_window == current_window:
                    if state.probe_signature == signature:
                        # Exact signed retries remain idempotent. The receipt map
                        # owns the common same-process retry path.
                        canaries.append((key, scope, state))
                        continue
                    blocked.append((scope, state, "canary_pending"))
                    continue
                canaries.append((key, scope, state))

            if blocked:
                self._precommit_rejects_total += 1
                if pruned:
                    self._schedule_save_locked()
                statuses = {status for _scope, _state, status in blocked}
                retry_after = max(
                    (
                        state.cooldown_until_window
                        if status == "cooldown"
                        else current_window + 1
                    )
                    for _scope, state, status in blocked
                )
                return PromptMismatchDecision(
                    allowed=False,
                    blocked_scopes=tuple(
                        sorted({scope for scope, _state, _status in blocked})
                    ),
                    retry_after_window=retry_after,
                    status=(
                        "cooldown" if statuses == {"cooldown"} else "canary_pending"
                    ),
                )

            for _key, _scope, state in canaries:
                state.probe_signature = signature
                state.probe_window = current_window
                state.last_window = max(state.last_window, current_window)
            if canaries or pruned:
                self._schedule_save_locked()
            return PromptMismatchDecision(
                allowed=True,
                canary=bool(canaries),
                canary_scopes=tuple(
                    sorted({scope for _key, scope, _state in canaries})
                ),
                status="canary" if canaries else "allowed",
            )

    def cancel_canary(
        self,
        *,
        environment: str,
        identities: Mapping[str, str],
        window: int,
        precommit_signature: str,
    ) -> None:
        """Release a canary when downstream receipt registration fails."""
        if not self.enabled:
            return
        signature = str(precommit_signature).strip()
        current_window = int(window)
        changed = False
        with self._lock:
            for scope, identity in self._subjects(identities):
                state = self._states.get(self._key(environment, scope, identity))
                if (
                    state is not None
                    and state.probe_signature == signature
                    and state.probe_window == current_window
                ):
                    state.probe_signature = None
                    state.probe_window = None
                    changed = True
            if changed:
                self._schedule_save_locked()

    def record_mismatch(
        self,
        *,
        environment: str,
        identities: Mapping[str, str],
        window: int,
        precommit_signature: str,
        precommit_arrival_ts: float,
    ) -> PromptMismatchUpdate:
        """Record one terminal mismatch and arm or escalate when warranted."""
        if not self.enabled:
            return PromptMismatchUpdate()

        event = self._outcome_order(
            window=window,
            precommit_arrival_ts=precommit_arrival_ts,
            precommit_signature=precommit_signature,
        )
        current_window = event.window
        signature = event.signature
        activated: list[str] = []
        escalated: list[str] = []
        cooldown_until: list[int] = []
        subjects = self._subjects(identities)
        if not subjects:
            return PromptMismatchUpdate()

        with self._lock:
            state_changed = self._prune_locked(current_window)
            self._mismatches_total += 1
            for scope, identity in subjects:
                key = self._key(environment, scope, identity)
                state = self._states.setdefault(key, _CircuitState())
                state.last_window = max(state.last_window, current_window)
                state.mismatch_count += 1

                # A successful canary leaves a recovery watermark instead of
                # deleting the entry. Outcomes from older receipts can then
                # finish later without rebuilding strikes after recovery.
                if state.recovery_order is not None and event <= state.recovery_order:
                    self._stale_outcomes_ignored_total += 1
                    continue

                if state.cooldown_level < 0:
                    if signature in state.failure_events:
                        self._duplicate_outcomes_ignored_total += 1
                        continue
                    state.failure_events[signature] = event
                    reference_window = max(current_window, state.last_window)
                    self._trim_failure_events_locked(state, reference_window)
                    state_changed = True
                    if len(state.failure_events) >= self.failure_threshold:
                        state.failure_events.clear()
                        state.cooldown_level = 0
                        state.cooldown_until_window = (
                            reference_window + self.cooldown_windows[0]
                        )
                        state.probe_signature = None
                        state.probe_window = None
                        activated.append(scope)
                        cooldown_until.append(state.cooldown_until_window)
                    continue

                # Only the exact post-cooldown canary may escalate. Any normal
                # receipt already in flight when the breaker armed is ignored.
                if state.probe_signature != signature:
                    self._stale_outcomes_ignored_total += 1
                    continue
                state.cooldown_level = min(
                    state.cooldown_level + 1,
                    len(self.cooldown_windows) - 1,
                )
                state.cooldown_until_window = (
                    current_window + self.cooldown_windows[state.cooldown_level]
                )
                state.probe_signature = None
                state.probe_window = None
                state_changed = True
                escalated.append(scope)
                cooldown_until.append(state.cooldown_until_window)

            if activated:
                self._cooldowns_started_total += 1
            if escalated:
                self._cooldowns_escalated_total += 1
            if state_changed:
                self._schedule_save_locked()
        return PromptMismatchUpdate(
            activated_scopes=tuple(sorted(set(activated))),
            escalated_scopes=tuple(sorted(set(escalated))),
            cooldown_until_window=max(cooldown_until, default=None),
        )

    def record_binding_success(
        self,
        *,
        environment: str,
        identities: Mapping[str, str],
        window: int,
        precommit_signature: str,
        precommit_arrival_ts: float,
    ) -> PromptMismatchUpdate:
        """Recover only armed scopes for which this receipt is the canary.

        An ordinary prompt-bound group deliberately does not erase partial
        mismatch debt. This prevents ``mismatch, mismatch, valid`` traffic from
        keeping the circuit permanently below its threshold.
        """
        if not self.enabled:
            return PromptMismatchUpdate()

        event = self._outcome_order(
            window=window,
            precommit_arrival_ts=precommit_arrival_ts,
            precommit_signature=precommit_signature,
        )
        cleared: list[str] = []
        with self._lock:
            pruned = self._prune_locked(event.window)
            self._binding_successes_total += 1
            for scope, identity in self._subjects(identities):
                key = self._key(environment, scope, identity)
                state = self._states.get(key)
                if state is None:
                    continue
                if state.cooldown_level < 0:
                    # Partial debt is rolling and expires naturally. A success
                    # is useful evidence but not a zero-cost abuse reset.
                    if state.failure_events:
                        self._noncanary_successes_ignored_total += 1
                    continue
                if state.probe_signature != event.signature:
                    self._noncanary_successes_ignored_total += 1
                    continue

                state.failure_events.clear()
                state.cooldown_level = -1
                state.cooldown_until_window = 0
                state.probe_signature = None
                state.probe_window = None
                if state.recovery_order is None or event > state.recovery_order:
                    state.recovery_order = event
                state.last_window = max(state.last_window, event.window)
                cleared.append(scope)

            if cleared:
                self._recoveries_total += 1
            if cleared or pruned:
                self._schedule_save_locked()
        return PromptMismatchUpdate(cleared_scopes=tuple(sorted(set(cleared))))

    def health_snapshot(self, *, current_window: int | None) -> dict[str, object]:
        with self._lock:
            partial = 0
            armed = 0
            active = 0
            canary_ready = 0
            canary_pending = 0
            recovery_tombstones = 0
            mismatch_debt = 0
            entries_by_scope = {scope: 0 for scope in sorted(_VALID_SCOPES)}
            armed_by_scope = {scope: 0 for scope in sorted(_VALID_SCOPES)}
            for key, state in self._states.items():
                scope = key[1]
                entries_by_scope[scope] += 1
                if state.cooldown_level < 0:
                    if state.failure_events:
                        partial += 1
                        mismatch_debt += len(state.failure_events)
                    elif state.recovery_order is not None:
                        recovery_tombstones += 1
                    continue
                armed += 1
                armed_by_scope[scope] += 1
                if (
                    current_window is None
                    or current_window < state.cooldown_until_window
                ):
                    active += 1
                elif state.probe_window == current_window:
                    canary_pending += 1
                else:
                    canary_ready += 1

            persistence_degraded = self.state_path is not None and bool(
                self._last_load_error or self._last_persistence_error
            )
            status = (
                "disabled"
                if not self.enabled
                else "degraded"
                if persistence_degraded
                else "ok"
            )
            return {
                "status": status,
                "enabled": self.enabled,
                "namespace": self.namespace,
                "failure_threshold": self.failure_threshold,
                "failure_window_windows": self.failure_window_windows,
                "cooldown_windows": list(self.cooldown_windows),
                "persistence_enabled": self.state_path is not None,
                "persistence_pending": (
                    self._persistence_generation > self._persisted_generation
                ),
                "state_path": str(self.state_path) if self.state_path else None,
                "state_reset_reason": self._state_reset_reason,
                "entries": len(self._states),
                "partial_strike_entries": partial,
                "pending_mismatch_debt": mismatch_debt,
                "recovery_tombstones": recovery_tombstones,
                "armed_entries": armed,
                "active_cooldowns": active,
                "canary_ready": canary_ready,
                "canary_pending": canary_pending,
                "entries_by_scope": entries_by_scope,
                "armed_by_scope": armed_by_scope,
                "mismatches_total": self._mismatches_total,
                "binding_successes_total": self._binding_successes_total,
                "cooldowns_started_total": self._cooldowns_started_total,
                "cooldowns_escalated_total": self._cooldowns_escalated_total,
                "recoveries_total": self._recoveries_total,
                "precommit_rejects_total": self._precommit_rejects_total,
                "stale_outcomes_ignored_total": self._stale_outcomes_ignored_total,
                "duplicate_outcomes_ignored_total": (
                    self._duplicate_outcomes_ignored_total
                ),
                "noncanary_successes_ignored_total": (
                    self._noncanary_successes_ignored_total
                ),
                "last_load_error": self._last_load_error,
                "last_persistence_error": self._last_persistence_error,
            }

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until the latest scheduled persistence snapshot finishes."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._persistence_condition:
            target = self._persistence_generation
            while self._persisted_generation < target:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._persistence_condition.wait(timeout=remaining)
            return True

    def close(self, timeout: float = 5.0) -> bool:
        """Flush pending state and stop the optional background writer."""
        flushed = self.flush(timeout=timeout)
        with self._persistence_condition:
            self._persistence_stopping = True
            self._persistence_condition.notify_all()
            thread = self._persistence_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(timeout)))
        return flushed and (thread is None or not thread.is_alive())

    def _trim_failure_events_locked(
        self,
        state: _CircuitState,
        reference_window: int,
    ) -> bool:
        cutoff = int(reference_window) - self.failure_window_windows + 1
        expired = [
            signature
            for signature, event in state.failure_events.items()
            if event.window < cutoff
        ]
        for signature in expired:
            state.failure_events.pop(signature, None)
        return bool(expired)

    def _prune_locked(self, current_window: int) -> bool:
        retention = max(1_000, 4 * max(self.cooldown_windows))
        cutoff = int(current_window) - retention
        changed = False
        expired_keys: list[_StateKey] = []
        for key, state in self._states.items():
            if state.cooldown_level < 0:
                changed |= self._trim_failure_events_locked(
                    state,
                    max(int(current_window), state.last_window),
                )
                if not state.failure_events and state.recovery_order is None:
                    expired_keys.append(key)
                    continue
            if state.last_window < cutoff and (
                state.cooldown_level < 0
                or state.cooldown_until_window <= current_window
            ):
                expired_keys.append(key)
        for key in expired_keys:
            self._states.pop(key, None)
        return changed or bool(expired_keys)

    def _load(self) -> None:
        path = self.state_path
        if path is None:
            return
        try:
            if not path.exists():
                return
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise TypeError("state root must be an object")

            schema_version = payload.get("schema_version")
            if schema_version != _SCHEMA_VERSION:
                self._state_reset_reason = f"schema_changed:{schema_version}"
                return
            persisted_namespace = str(payload.get("namespace", "")).strip()
            if persisted_namespace != self.namespace:
                self._state_reset_reason = "namespace_changed"
                return

            entries = payload.get("entries", [])
            if not isinstance(entries, list):
                raise TypeError("state entries must be a list")
            restored: dict[_StateKey, _CircuitState] = {}
            for entry in entries:
                if not isinstance(entry, dict):
                    raise TypeError("state entry must be an object")
                environment = str(entry["environment"]).strip()
                scope = str(entry["scope"]).strip()
                identity = str(entry["identity"]).strip()
                if not environment or not identity:
                    raise ValueError("state identity fields cannot be empty")
                if scope not in _VALID_SCOPES:
                    raise ValueError("invalid state identity scope")

                raw_failures = entry.get("failure_events", [])
                if not isinstance(raw_failures, list):
                    raise TypeError("failure_events must be a list")
                failure_events: dict[str, _OutcomeOrder] = {}
                for raw_event in raw_failures:
                    event = _OutcomeOrder.from_json(raw_event)
                    failure_events[event.signature] = event
                recovery_order = (
                    _OutcomeOrder.from_json(entry["recovery_order"])
                    if entry.get("recovery_order") is not None
                    else None
                )
                state = _CircuitState(
                    failure_events=failure_events,
                    cooldown_level=int(entry.get("cooldown_level", -1)),
                    cooldown_until_window=int(entry.get("cooldown_until_window", 0)),
                    probe_signature=(
                        str(entry["probe_signature"])
                        if entry.get("probe_signature") is not None
                        else None
                    ),
                    probe_window=(
                        int(entry["probe_window"])
                        if entry.get("probe_window") is not None
                        else None
                    ),
                    recovery_order=recovery_order,
                    last_window=int(entry.get("last_window", 0)),
                    mismatch_count=int(entry.get("mismatch_count", 0)),
                )
                if state.mismatch_count < 0:
                    raise ValueError("state counters cannot be negative")
                if not (-1 <= state.cooldown_level < len(self.cooldown_windows)):
                    raise ValueError("invalid cooldown level")
                if (
                    state.cooldown_level < 0
                    and len(state.failure_events) >= self.failure_threshold
                ):
                    raise ValueError("unarmed state exceeds failure threshold")
                restored[(environment, scope, identity)] = state

            counters = payload.get("counters", {})
            if not isinstance(counters, dict):
                counters = {}
            with self._lock:
                self._states = restored
                self._mismatches_total = max(
                    0, int(counters.get("mismatches_total", 0))
                )
                self._binding_successes_total = max(
                    0, int(counters.get("binding_successes_total", 0))
                )
                self._cooldowns_started_total = max(
                    0, int(counters.get("cooldowns_started_total", 0))
                )
                self._cooldowns_escalated_total = max(
                    0, int(counters.get("cooldowns_escalated_total", 0))
                )
                self._recoveries_total = max(
                    0, int(counters.get("recoveries_total", 0))
                )
                self._precommit_rejects_total = max(
                    0, int(counters.get("precommit_rejects_total", 0))
                )
                self._stale_outcomes_ignored_total = max(
                    0, int(counters.get("stale_outcomes_ignored_total", 0))
                )
                self._duplicate_outcomes_ignored_total = max(
                    0, int(counters.get("duplicate_outcomes_ignored_total", 0))
                )
                self._noncanary_successes_ignored_total = max(
                    0, int(counters.get("noncanary_successes_ignored_total", 0))
                )
        except Exception as exc:  # noqa: BLE001 - corrupted state must fail open
            self._last_load_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "prompt mismatch circuit state load failed path=%s error=%s",
                path,
                self._last_load_error,
            )

    def _schedule_save_locked(self) -> None:
        if self.state_path is None or self._persistence_stopping:
            return
        self._persistence_generation += 1
        thread = self._persistence_thread
        if thread is None or not thread.is_alive():
            thread = threading.Thread(
                target=self._persistence_worker,
                name="prompt-mismatch-state-writer",
                daemon=True,
            )
            self._persistence_thread = thread
            try:
                thread.start()
            except RuntimeError as exc:
                self._persistence_thread = None
                self._last_persistence_error = f"{type(exc).__name__}: {exc}"
                self._persisted_generation = self._persistence_generation
                logger.warning(
                    "prompt mismatch circuit writer failed to start error=%s",
                    self._last_persistence_error,
                )
        self._persistence_condition.notify_all()

    def _snapshot_locked(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "namespace": self.namespace,
            "updated_at": time.time(),
            "entries": [
                {
                    "environment": key[0],
                    "scope": key[1],
                    "identity": key[2],
                    "failure_events": [
                        event.to_json()
                        for event in sorted(state.failure_events.values())
                    ],
                    "cooldown_level": state.cooldown_level,
                    "cooldown_until_window": state.cooldown_until_window,
                    "probe_signature": state.probe_signature,
                    "probe_window": state.probe_window,
                    "recovery_order": (
                        state.recovery_order.to_json()
                        if state.recovery_order is not None
                        else None
                    ),
                    "last_window": state.last_window,
                    "mismatch_count": state.mismatch_count,
                }
                for key, state in sorted(self._states.items())
            ],
            "counters": {
                "mismatches_total": self._mismatches_total,
                "binding_successes_total": self._binding_successes_total,
                "cooldowns_started_total": self._cooldowns_started_total,
                "cooldowns_escalated_total": self._cooldowns_escalated_total,
                "recoveries_total": self._recoveries_total,
                "precommit_rejects_total": self._precommit_rejects_total,
                "stale_outcomes_ignored_total": self._stale_outcomes_ignored_total,
                "duplicate_outcomes_ignored_total": (
                    self._duplicate_outcomes_ignored_total
                ),
                "noncanary_successes_ignored_total": (
                    self._noncanary_successes_ignored_total
                ),
            },
        }

    def _persistence_worker(self) -> None:
        while True:
            with self._persistence_condition:
                while (
                    self._persisted_generation >= self._persistence_generation
                    and not self._persistence_stopping
                ):
                    self._persistence_condition.wait()
                if (
                    self._persistence_stopping
                    and self._persisted_generation >= self._persistence_generation
                ):
                    return
                target_generation = self._persistence_generation
                try:
                    payload = self._snapshot_locked()
                except Exception as exc:  # noqa: BLE001 - never kill writer
                    error = f"{type(exc).__name__}: {exc}"
                    self._last_persistence_error = error
                    self._persisted_generation = target_generation
                    self._persistence_condition.notify_all()
                    logger.warning(
                        "prompt mismatch circuit snapshot failed error=%s",
                        error,
                    )
                    continue

            try:
                error = self._write_payload(payload)
            except Exception as exc:  # noqa: BLE001 - never kill writer
                error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "prompt mismatch circuit writer failed error=%s",
                    error,
                )
            with self._persistence_condition:
                self._last_persistence_error = error
                self._persisted_generation = max(
                    self._persisted_generation,
                    target_generation,
                )
                self._persistence_condition.notify_all()

    def _write_payload(self, payload: Mapping[str, object]) -> str | None:
        path = self.state_path
        if path is None:
            return None
        tmp_name: str | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=".prompt-mismatch-circuit.",
                suffix=".json",
                dir=path.parent,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
            tmp_name = None
            return None
        except Exception as exc:  # noqa: BLE001 - state I/O must fail open
            error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "prompt mismatch circuit state save failed path=%s error=%s",
                path,
                error,
            )
            return error
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except FileNotFoundError:
                    pass
