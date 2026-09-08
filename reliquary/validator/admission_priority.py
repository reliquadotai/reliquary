"""Deterministic priority queue for the disabled fill experiment.

The module preserves the fill branch's exact qualification policy so it can be
replayed and compared behind its own capability. It does not admit, grade,
prove, select, or pay a submission. Reliquary 1 does not use this priority for
ticket admission or final ranking.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueuedPrecommit:
    receipt_id: str
    environment: str
    payload_bytes: int
    precommit_arrived_at: float
    elapsed: float

    @property
    def throughput(self) -> float:
        return self.payload_bytes / self.elapsed


class ThroughputAdmissionQueue:
    """Hold precommits until the validator has budget to validate one."""

    def __init__(self, *, window_opened_at: float) -> None:
        self._window_opened_at = float(window_opened_at)
        self._queued: dict[str, list[QueuedPrecommit]] = {}
        self._by_receipt: dict[str, QueuedPrecommit] = {}

    def offer(
        self,
        *,
        receipt_id: str,
        environment: str,
        payload_bytes: int,
        precommit_arrived_at: float,
    ) -> QueuedPrecommit:
        """Record one bounded precommit for later deterministic lookup."""
        elapsed = max(precommit_arrived_at - self._window_opened_at, 1e-9)
        entry = QueuedPrecommit(
            receipt_id=receipt_id,
            environment=environment,
            payload_bytes=int(payload_bytes),
            precommit_arrived_at=float(precommit_arrived_at),
            elapsed=elapsed,
        )
        self._queued.setdefault(environment, []).append(entry)
        self._by_receipt[receipt_id] = entry
        return entry

    def rate_of(self, receipt_id: str) -> float | None:
        """Return the recorded experimental priority, or ``None``."""
        entry = self._by_receipt.get(receipt_id)
        return entry.throughput if entry is not None else None

    def payload_bytes_of(self, receipt_id: str) -> int | None:
        """Return the payload size bound by the receipt, or ``None``."""
        entry = self._by_receipt.get(receipt_id)
        return entry.payload_bytes if entry is not None else None
