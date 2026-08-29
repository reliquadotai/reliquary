"""Service-level join of per-environment emission chunks into one
cross-environment DAPO training batch (R13).

A ``GrpoWindowBatcher`` only ever accounts its OWN environment's proven
groups (``_reconcile_fill_state_decisions`` is keyed by ``self.env.name``),
so it structurally cannot assemble a batch that needs B_BATCH groups from
EVERY configured environment -- one DAPO step needs both. Only the
service sees every environment's batcher for a window, so the join lives
here: one instance per window, constructed beside the shared ``FillState``
(same place, same gate), and injected as every batcher's
``emit_training_batch_fn``.

Each batcher hands its own environment's next B_BATCH-sized chunk to
``accept`` as soon as it is ready, independent of the other environment's
progress -- chunks can arrive in any interleaving (math k=0, math k=1,
code k=0, ...). This holds one chunk in flight per environment inside a
``BalancedTrainingAccumulator`` (targets B_BATCH each) and queues anything
past that until the accumulator's current cycle empties out, so a fast
environment's second chunk is never lost, only held -- exactly the
remainder ``BalancedTrainingAccumulator`` already exists to carry.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Sequence

from reliquary.constants import B_BATCH
from reliquary.infrastructure.training_payload_queue import (
    encoded_window_journal_key,
)
from reliquary.shared.training_payload import (
    encode_tombstone,
    encode_training_payload,
)
from reliquary.validator.quarantine import assess_training_batch
from reliquary.validator.training_accumulator import BalancedTrainingAccumulator

logger = logging.getLogger(__name__)


class FillClosedBatchAssembler:
    """Joins B_BATCH-per-environment chunks for exactly one window."""

    def __init__(
        self,
        *,
        window_start: int,
        env_order: Sequence[str],
        enqueue_fn: Callable[[int, bytes], None],
        tombstone_fn: Callable[[int, bytes], None],
    ) -> None:
        self.window_start = int(window_start)
        self._env_order = list(env_order)
        self._enqueue_fn = enqueue_fn
        self._tombstone_fn = tombstone_fn
        self._accumulator = BalancedTrainingAccumulator(
            {environment: B_BATCH for environment in self._env_order}
        )
        # Chunks that arrived while the accumulator's slot for their own
        # environment was already full -- held in arrival order, fed in
        # one at a time as each cycle's payload is written and the
        # accumulator empties out again.
        self._pending: dict[str, list[list[Any]]] = {
            environment: [] for environment in self._env_order
        }
        self._checkpoint_revision: str = ""
        # Monotonic. Every payload this window writes gets the NEXT value,
        # in write order -- the ordering barrier lives in ``_drain_locked``:
        # a cycle only advances the index and writes once every
        # environment's slot for THAT cycle is present, never from
        # whichever environment's callback happened to arrive last.
        self.next_batch_index: int = 0
        # One lock: two per-environment batchers can call ``accept`` from
        # two different proof-worker device threads at once, and both the
        # accumulator and the pending queues are shared, mutable state.
        self._lock = threading.Lock()

    def accept(
        self,
        environment: str,
        groups: list[Any],
        window_start: int,
        checkpoint_revision: str,
    ) -> None:
        """The callback injected as every batcher's ``emit_training_batch_fn``.

        Called once per B_BATCH-sized chunk, with ONLY that chunk's own
        environment populated -- a batcher cannot supply more than that
        (see the module docstring). ``window_start`` is asserted rather
        than trusted silently: every batcher for this window was built
        with the same ``target_window``, so a mismatch is a wiring bug,
        not a race to tolerate.
        """
        if environment not in self._env_order:
            raise ValueError(f"unknown environment {environment!r}")
        if int(window_start) != self.window_start:
            raise ValueError(
                f"assembler for window {self.window_start} received a "
                f"chunk for window {window_start}"
            )
        with self._lock:
            self._checkpoint_revision = str(checkpoint_revision)
            self._pending[environment].append(list(groups))
            self._drain_locked()

    def _drain_locked(self) -> None:
        while True:
            fed_any = False
            counts = self._accumulator.snapshot()["counts"]
            for environment in self._env_order:
                if (
                    counts[environment] < B_BATCH
                    and self._pending[environment]
                ):
                    chunk = self._pending[environment].pop(0)
                    self._accumulator.add_window(
                        {environment: chunk},
                        window_n=self.window_start,
                        checkpoint_revision=self._checkpoint_revision,
                    )
                    fed_any = True
            if self._accumulator.ready:
                self._write_one_payload_locked()
                continue
            if not fed_any:
                return

    def _write_one_payload_locked(self) -> None:
        extracted = self._accumulator.training_batches(self._env_order)
        window_batches = dict(zip(self._env_order, extracted))
        self._accumulator.reset()
        flat_batch = [
            group
            for env_batch in window_batches.values()
            for group in env_batch
        ]
        # Per-batch, not per-window (R14): v6 writes up to
        # FILL_CLOSED_EMISSIONS_PER_WINDOW payloads per window where the
        # seal path writes exactly one, so each one is its own admission
        # to the optimizer and must clear the same gate on its own.
        # reject_counts is not aggregated per-batch here -- the seal
        # path's OWN accumulated-batch gate (service.py's
        # ``accumulated_quarantine``) makes this identical simplification
        # for the identical reason: reject-count aggregation is a
        # cross-batcher concern that spans a whole window's rejections,
        # not one join cycle's, and this class has no wiring to it.
        decision = assess_training_batch(flat_batch, reject_counts={})
        key = encoded_window_journal_key(
            self.window_start, self.next_batch_index
        )
        if decision.quarantined:
            # Do NOT enqueue: this is exactly the poisoned-data path the
            # seal-time gate exists to close. A tombstone still goes out
            # under this batch's OWN encoded key so the trainer's cursor
            # advances -- it never advances on absence, only on an
            # explicit marker (see service._write_training_tombstone).
            data = encode_tombstone(
                window_start=self.window_start,
                failure_stage="training_quarantine",
                failure_type="TrainingQuarantine",
            )
            self._tombstone_fn(key, data)
        else:
            data = encode_training_payload(
                window_batches,
                window_start=self.window_start,
                checkpoint_revision=self._checkpoint_revision,
                env_order=list(self._env_order),
                window_quarantine=decision.to_archive(),
                checkpoint_epoch=None,
            )
            self._enqueue_fn(key, data)
        logger.info(
            "FillClosedBatchAssembler: window %d batch %d %s (journal "
            "key %d, quarantined=%s reasons=%s)",
            self.window_start, self.next_batch_index,
            "tombstoned" if decision.quarantined else "written",
            key, decision.quarantined, decision.reasons,
        )
        self.next_batch_index += 1

    def remainder_snapshot(self) -> dict[str, Any]:
        """Groups still held and not yet part of a written payload, for
        observability at window seal (R13, item 5: the remainder is never
        emitted as a partial batch -- a DAPO step needs B_BATCH of every
        environment, and partial batches are exactly what this class's
        carry-forward exists to avoid)."""
        counts = self._accumulator.snapshot()["counts"]
        return {
            "in_accumulator": dict(counts),
            "pending": {
                environment: len(chunks)
                for environment, chunks in self._pending.items()
            },
        }
