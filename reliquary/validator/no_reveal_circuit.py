"""Persistent operator circuit for signed receipts never uploaded."""

from __future__ import annotations

import os

from reliquary.constants import (
    NO_REVEAL_CIRCUIT_COOLDOWN_WINDOWS,
    NO_REVEAL_CIRCUIT_ENABLED,
    NO_REVEAL_CIRCUIT_FAILURE_THRESHOLD,
    NO_REVEAL_CIRCUIT_FAILURE_WINDOW_WINDOWS,
)
from reliquary.validator.prompt_mismatch_circuit import (
    PromptMismatchCircuitBreaker,
)


class NoRevealCircuitBreaker:
    """Operator-only adapter over the ordered persistent circuit state."""

    def __init__(
        self,
        state_path: str | os.PathLike[str] | None = None,
        *,
        namespace: str,
    ) -> None:
        self._circuit = PromptMismatchCircuitBreaker(
            state_path,
            namespace=namespace,
            enabled=NO_REVEAL_CIRCUIT_ENABLED,
            failure_threshold=NO_REVEAL_CIRCUIT_FAILURE_THRESHOLD,
            failure_window_windows=NO_REVEAL_CIRCUIT_FAILURE_WINDOW_WINDOWS,
            cooldown_windows=NO_REVEAL_CIRCUIT_COOLDOWN_WINDOWS,
        )

    @staticmethod
    def _identities(operator: str | None) -> dict[str, str]:
        value = str(operator or "").strip()
        return {"operator": value} if value else {}

    def admit_precommit(
        self,
        *,
        environment: str,
        operator: str | None,
        window: int,
        precommit_signature: str,
    ):
        return self._circuit.admit_precommit(
            environment=environment,
            identities=self._identities(operator),
            window=window,
            precommit_signature=precommit_signature,
        )

    def cancel_canary(
        self,
        *,
        environment: str,
        operator: str | None,
        window: int,
        precommit_signature: str,
    ) -> None:
        self._circuit.cancel_canary(
            environment=environment,
            identities=self._identities(operator),
            window=window,
            precommit_signature=precommit_signature,
        )

    def record_no_reveal(
        self,
        *,
        environment: str,
        operator: str | None,
        window: int,
        precommit_signature: str,
        precommit_arrival_ts: float,
    ):
        return self._circuit.record_mismatch(
            environment=environment,
            identities=self._identities(operator),
            window=window,
            precommit_signature=precommit_signature,
            precommit_arrival_ts=precommit_arrival_ts,
        )

    def record_reveal(
        self,
        *,
        environment: str,
        operator: str | None,
        window: int,
        precommit_signature: str,
        precommit_arrival_ts: float,
    ):
        return self._circuit.record_binding_success(
            environment=environment,
            identities=self._identities(operator),
            window=window,
            precommit_signature=precommit_signature,
            precommit_arrival_ts=precommit_arrival_ts,
        )

    def health_snapshot(self, *, current_window: int | None):
        snapshot = self._circuit.health_snapshot(current_window=current_window)
        snapshot["no_reveals_total"] = snapshot.pop("mismatches_total", 0)
        snapshot["valid_reveals_total"] = snapshot.pop("binding_successes_total", 0)
        return snapshot

    def close(self) -> None:
        self._circuit.close()
