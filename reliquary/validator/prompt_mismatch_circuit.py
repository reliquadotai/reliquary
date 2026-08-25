"""Persistent admission circuit breaker for incompatible prompt renderers.

Prompt binding is deterministic: a miner either submitted the exact canonical
prompt tokens for the advertised generation profile or it did not.  Repeated
``PROMPT_MISMATCH`` outcomes therefore provide a cheap compatibility signal we
can use before accepting another large reveal body.

The breaker is keyed by environment/profile and by both hotkey and metagraph
operator.  It deliberately admits one signed canary after each cooldown so a
miner that upgrades can recover without operator intervention.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from reliquary.constants import (
    PROMPT_MISMATCH_CIRCUIT_COOLDOWN_WINDOWS,
    PROMPT_MISMATCH_CIRCUIT_ENABLED,
    PROMPT_MISMATCH_CIRCUIT_FAILURE_THRESHOLD,
)

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_VALID_SCOPES = frozenset({"hotkey", "operator"})
_StateKey = tuple[str, str, str, str]


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
    consecutive_mismatches: int = 0
    cooldown_level: int = -1
    cooldown_until_window: int = 0
    probe_signature: str | None = None
    probe_window: int | None = None
    last_window: int = 0
    mismatch_count: int = 0


class PromptMismatchCircuitBreaker:
    """Track prompt-renderer compatibility and gate signed precommits.

    ``state_path=None`` keeps state in memory, which is useful for isolated
    server/tests.  Production passes a path on the validator state volume.
    Persistence is best-effort and fail-open: an unavailable state disk must
    never take the validator HTTP admission path down.
    """

    def __init__(
        self,
        state_path: str | os.PathLike[str] | None = None,
        *,
        enabled: bool = PROMPT_MISMATCH_CIRCUIT_ENABLED,
        failure_threshold: int = PROMPT_MISMATCH_CIRCUIT_FAILURE_THRESHOLD,
        cooldown_windows: tuple[int, ...] = PROMPT_MISMATCH_CIRCUIT_COOLDOWN_WINDOWS,
    ) -> None:
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if not cooldown_windows or any(value <= 0 for value in cooldown_windows):
            raise ValueError("cooldown_windows must contain positive values")
        self.enabled = bool(enabled)
        self.failure_threshold = int(failure_threshold)
        self.cooldown_windows = tuple(int(value) for value in cooldown_windows)
        self.state_path = Path(state_path) if state_path is not None else None
        self._states: dict[_StateKey, _CircuitState] = {}
        self._lock = threading.RLock()
        self._mismatches_total = 0
        self._cooldowns_started_total = 0
        self._cooldowns_escalated_total = 0
        self._recoveries_total = 0
        self._precommit_rejects_total = 0
        self._last_load_error: str | None = None
        self._last_persistence_error: str | None = None
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
    def _key(
        environment: str,
        generation_profile_id: str,
        scope: str,
        identity: str,
    ) -> _StateKey:
        return (
            str(environment).strip(),
            str(generation_profile_id).strip() or "<legacy>",
            scope,
            identity,
        )

    def admit_precommit(
        self,
        *,
        environment: str,
        generation_profile_id: str,
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
                key = self._key(
                    environment,
                    generation_profile_id,
                    scope,
                    identity,
                )
                state = self._states.get(key)
                if state is None or state.cooldown_level < 0:
                    continue
                if current_window < state.cooldown_until_window:
                    blocked.append((scope, state, "cooldown"))
                    continue
                if state.probe_window == current_window:
                    if state.probe_signature == signature:
                        # An exact signed retry is idempotent from the circuit's
                        # perspective.  The server receipt map handles the
                        # common same-process retry path.
                        canaries.append((key, scope, state))
                        continue
                    blocked.append((scope, state, "canary_pending"))
                    continue
                canaries.append((key, scope, state))

            if blocked:
                self._precommit_rejects_total += 1
                if pruned:
                    self._save_locked()
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
                self._save_locked()
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
        generation_profile_id: str,
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
                state = self._states.get(
                    self._key(
                        environment,
                        generation_profile_id,
                        scope,
                        identity,
                    )
                )
                if (
                    state is not None
                    and state.probe_signature == signature
                    and state.probe_window == current_window
                ):
                    state.probe_signature = None
                    state.probe_window = None
                    changed = True
            if changed:
                self._save_locked()

    def record_mismatch(
        self,
        *,
        environment: str,
        generation_profile_id: str,
        identities: Mapping[str, str],
        window: int,
        precommit_signature: str,
    ) -> PromptMismatchUpdate:
        """Record one terminal prompt mismatch and arm/escalate as needed."""
        if not self.enabled:
            return PromptMismatchUpdate()

        current_window = int(window)
        signature = str(precommit_signature).strip()
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
                key = self._key(
                    environment,
                    generation_profile_id,
                    scope,
                    identity,
                )
                state = self._states.setdefault(key, _CircuitState())
                state.last_window = max(state.last_window, current_window)
                state.mismatch_count += 1

                if state.cooldown_level < 0:
                    state.consecutive_mismatches += 1
                    state_changed = True
                    if state.consecutive_mismatches >= self.failure_threshold:
                        state.cooldown_level = 0
                        state.cooldown_until_window = (
                            current_window + self.cooldown_windows[0]
                        )
                        state.probe_signature = None
                        state.probe_window = None
                        activated.append(scope)
                        cooldown_until.append(state.cooldown_until_window)
                        self._cooldowns_started_total += 1
                    continue

                # Only the explicitly admitted post-cooldown canary may
                # escalate.  Failures from receipts already in flight when the
                # initial breaker armed are intentionally ignored here.
                if state.probe_signature != signature:
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
                self._cooldowns_escalated_total += 1

            if state_changed:
                self._save_locked()
        return PromptMismatchUpdate(
            activated_scopes=tuple(sorted(set(activated))),
            escalated_scopes=tuple(sorted(set(escalated))),
            cooldown_until_window=max(cooldown_until, default=None),
        )

    def record_binding_success(
        self,
        *,
        environment: str,
        generation_profile_id: str,
        identities: Mapping[str, str],
        window: int,
    ) -> PromptMismatchUpdate:
        """Clear strikes/cooldowns after exact canonical prompt binding."""
        if not self.enabled:
            return PromptMismatchUpdate()

        cleared: list[str] = []
        with self._lock:
            pruned = self._prune_locked(int(window))
            for scope, identity in self._subjects(identities):
                key = self._key(
                    environment,
                    generation_profile_id,
                    scope,
                    identity,
                )
                if self._states.pop(key, None) is not None:
                    cleared.append(scope)
            if cleared:
                self._recoveries_total += len(set(cleared))
            if cleared or pruned:
                self._save_locked()
        return PromptMismatchUpdate(cleared_scopes=tuple(sorted(set(cleared))))

    def health_snapshot(self, *, current_window: int | None) -> dict[str, object]:
        with self._lock:
            partial = 0
            armed = 0
            active = 0
            canary_ready = 0
            canary_pending = 0
            entries_by_scope = {scope: 0 for scope in sorted(_VALID_SCOPES)}
            armed_by_scope = {scope: 0 for scope in sorted(_VALID_SCOPES)}
            for key, state in self._states.items():
                scope = key[2]
                entries_by_scope[scope] += 1
                if state.cooldown_level < 0:
                    partial += 1
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
            return {
                "enabled": self.enabled,
                "failure_threshold": self.failure_threshold,
                "cooldown_windows": list(self.cooldown_windows),
                "persistence_enabled": self.state_path is not None,
                "state_path": str(self.state_path) if self.state_path else None,
                "entries": len(self._states),
                "partial_strike_entries": partial,
                "armed_entries": armed,
                "active_cooldowns": active,
                "canary_ready": canary_ready,
                "canary_pending": canary_pending,
                "entries_by_scope": entries_by_scope,
                "armed_by_scope": armed_by_scope,
                "mismatches_total": self._mismatches_total,
                "cooldowns_started_total": self._cooldowns_started_total,
                "cooldowns_escalated_total": self._cooldowns_escalated_total,
                "recoveries_total": self._recoveries_total,
                "precommit_rejects_total": self._precommit_rejects_total,
                "last_load_error": self._last_load_error,
                "last_persistence_error": self._last_persistence_error,
            }

    def _prune_locked(self, current_window: int) -> bool:
        retention = max(1_000, 4 * max(self.cooldown_windows))
        cutoff = int(current_window) - retention
        expired = [
            key
            for key, state in self._states.items()
            if state.last_window < cutoff
            and (
                state.cooldown_level < 0
                or state.cooldown_until_window <= current_window
            )
        ]
        for key in expired:
            self._states.pop(key, None)
        return bool(expired)

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
            if payload.get("schema_version") != _SCHEMA_VERSION:
                raise ValueError("unsupported state schema")
            entries = payload.get("entries", [])
            if not isinstance(entries, list):
                raise TypeError("state entries must be a list")
            restored: dict[_StateKey, _CircuitState] = {}
            for entry in entries:
                if not isinstance(entry, dict):
                    raise TypeError("state entry must be an object")
                environment = str(entry["environment"]).strip()
                profile = str(entry["generation_profile_id"]).strip()
                scope = str(entry["scope"]).strip()
                identity = str(entry["identity"]).strip()
                if not environment or not profile or not identity:
                    raise ValueError("state identity fields cannot be empty")
                if scope not in _VALID_SCOPES:
                    raise ValueError("invalid state identity scope")
                state = _CircuitState(
                    consecutive_mismatches=int(entry.get("consecutive_mismatches", 0)),
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
                    last_window=int(entry.get("last_window", 0)),
                    mismatch_count=int(entry.get("mismatch_count", 0)),
                )
                if state.consecutive_mismatches < 0 or state.mismatch_count < 0:
                    raise ValueError("state counters cannot be negative")
                if not (-1 <= state.cooldown_level < len(self.cooldown_windows)):
                    raise ValueError("invalid cooldown level")
                restored[(environment, profile, scope, identity)] = state
            counters = payload.get("counters", {})
            if not isinstance(counters, dict):
                counters = {}
            with self._lock:
                self._states = restored
                self._mismatches_total = max(
                    0, int(counters.get("mismatches_total", 0))
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
        except Exception as exc:  # noqa: BLE001 - corrupted state must fail open
            self._last_load_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "prompt mismatch circuit state load failed path=%s error=%s",
                path,
                self._last_load_error,
            )

    def _save_locked(self) -> None:
        path = self.state_path
        if path is None:
            return
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "updated_at": time.time(),
            "entries": [
                {
                    "environment": key[0],
                    "generation_profile_id": key[1],
                    "scope": key[2],
                    "identity": key[3],
                    "consecutive_mismatches": state.consecutive_mismatches,
                    "cooldown_level": state.cooldown_level,
                    "cooldown_until_window": state.cooldown_until_window,
                    "probe_signature": state.probe_signature,
                    "probe_window": state.probe_window,
                    "last_window": state.last_window,
                    "mismatch_count": state.mismatch_count,
                }
                for key, state in sorted(self._states.items())
            ],
            "counters": {
                "mismatches_total": self._mismatches_total,
                "cooldowns_started_total": self._cooldowns_started_total,
                "cooldowns_escalated_total": self._cooldowns_escalated_total,
                "recoveries_total": self._recoveries_total,
                "precommit_rejects_total": self._precommit_rejects_total,
            },
        }
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
            self._last_persistence_error = None
        except Exception as exc:  # noqa: BLE001 - state I/O must fail open
            self._last_persistence_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "prompt mismatch circuit state save failed path=%s error=%s",
                path,
                self._last_persistence_error,
            )
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except FileNotFoundError:
                    pass
