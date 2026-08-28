"""Rate-ordered admission queue.

A fill-closed window closes when its batch is full, so the slots still open
near the close go to whoever finishes first — systematically whoever produced
the shortest rollouts. Ordering the queue by production RATE instead of by
arrival removes that: at fixed hardware the rate is the same whether a group
is 500 or 5000 tokens per rollout, so length stops deciding who gets in.

    rate = payload_bytes / (precommit arrival - window open)

Three properties of that formula, each deliberate:

* It is measured at the PRECOMMIT, not the upload. Transport is therefore
  inside the measure — a fat uplink cannot buy a place, only faster
  generation can.
* The denominator runs from window open for every group, and there is no
  identity in the formula. Splitting production across hotkeys changes
  nothing. Measured from a sender's previous arrival instead, a parallel
  producer's eighth group would show 0.1 s of elapsed and a 250x rate — a
  double count of hardware that already earned eight tickets by volume.
* Both terms are outside the miner's control. ``payload_bytes`` is bound by
  the signed precommit and enforced against the upload; the arrival is
  validator-observed.

Pure and dependency-free, like ``difficulty_auction`` and ``batch_selection``.
It admits nothing, grades nothing and proves nothing; it only decides what
the validator should spend its next grading and proof budget on.
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

    def offer(
        self,
        *,
        receipt_id: str,
        environment: str,
        payload_bytes: int,
        precommit_arrived_at: float,
    ) -> QueuedPrecommit:
        """Queue a precommit. Never refuses.

        The queue holds hashes, not payloads, so it is cheap, and the
        expensive stages behind it are already bounded by the environment's
        target. Bounding it again globally would be actively harmful:
        ``constants.py`` records why there is no global receipt ceiling —
        "any global counter can be deliberately burned before honest bodies
        arrive". Receipt memory is bounded per identity instead, by
        ``MAX_PENDING_UPLOAD_PRECOMMITS_PER_{HOTKEY,OPERATOR}``.
        """
        elapsed = max(precommit_arrived_at - self._window_opened_at, 1e-9)
        entry = QueuedPrecommit(
            receipt_id=receipt_id,
            environment=environment,
            payload_bytes=int(payload_bytes),
            precommit_arrived_at=float(precommit_arrived_at),
            elapsed=elapsed,
        )
        self._queued.setdefault(environment, []).append(entry)
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
                entry.precommit_arrived_at,
                entry.receipt_id,
            ),
        )
        entries.remove(best)
        return best
