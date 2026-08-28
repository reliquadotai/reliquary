"""Throughput-ordered admission queue.

A fill-closed window closes when its batch is full, so the slots still open
near the close go to whoever finishes first — systematically whoever produced
the shortest rollouts. Ordering the queue by production RATE instead of by
arrival removes that: at fixed hardware the rate is the same whether a group is
500 or 5000 tokens per rollout, so length stops deciding who gets in.

Pure and dependency-free, like ``difficulty_auction`` and ``batch_selection``.
It admits nothing, grades nothing and proves nothing; it only decides what the
validator should spend its next grading and proof budget on.

Both terms of the rate are safe from the miner. ``payload_bytes`` is bound by
the signed precommit and enforced against the upload that follows it, and
elapsed is measured from validator-observed arrival timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueuedPrecommit:
    receipt_id: str
    hotkey: str
    environment: str
    payload_bytes: int
    arrived_at: float
    elapsed: float

    @property
    def throughput(self) -> float:
        return self.payload_bytes / self.elapsed


class ThroughputAdmissionQueue:
    """Hold precommits until the validator has budget to validate one."""

    def __init__(
        self, *, window_opened_at: float, max_pending: int | None = None
    ) -> None:
        if max_pending is not None and max_pending <= 0:
            raise ValueError("max_pending must be positive")
        self._max_pending = max_pending
        self._window_opened_at = float(window_opened_at)
        self._queued: dict[str, list[QueuedPrecommit]] = {}
        # Last observed arrival per hotkey. The rate has to describe the group
        # that was just produced: measured from window open instead, a miner's
        # Nth precommit shows elapsed N x generation_time and its apparent rate
        # decays as 1/N, so only its first submission would ever compete.
        self._last_arrival: dict[str, float] = {}

    def offer(
        self,
        *,
        receipt_id: str,
        hotkey: str,
        environment: str,
        payload_bytes: int,
        arrived_at: float,
    ) -> QueuedPrecommit | None:
        """Queue a precommit, or return None if it cannot displace anything.

        At capacity the WORST queued entry is dropped, never the newest.
        Dropping the newest would turn the bound into a second arrival race —
        exactly what ordering by rate exists to remove.
        """
        started_at = self._last_arrival.get(hotkey, self._window_opened_at)
        elapsed = max(arrived_at - started_at, 1e-9)
        self._last_arrival[hotkey] = float(arrived_at)
        entry = QueuedPrecommit(
            receipt_id=receipt_id,
            hotkey=hotkey,
            environment=environment,
            payload_bytes=int(payload_bytes),
            arrived_at=float(arrived_at),
            elapsed=elapsed,
        )
        entries = self._queued.setdefault(environment, [])
        if self._max_pending is not None and len(entries) >= self._max_pending:
            worst = max(
                entries,
                key=lambda queued: (
                    -queued.throughput,
                    queued.arrived_at,
                    queued.receipt_id,
                ),
            )
            if entry.throughput <= worst.throughput:
                return None
            entries.remove(worst)
        entries.append(entry)
        return entry

    def take_best(self, environment: str) -> QueuedPrecommit | None:
        entries = self._queued.get(environment)
        if not entries:
            return None
        # Rates collide often — two miners on the same hardware produce the
        # same ratio — so the tie-break is explicit rather than left to list
        # ordering, which would make a window unreproducible on replay.
        best = min(
            entries,
            key=lambda entry: (
                -entry.throughput,
                entry.arrived_at,
                entry.receipt_id,
            ),
        )
        entries.remove(best)
        return best
