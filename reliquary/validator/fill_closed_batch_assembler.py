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

Commit discipline: the callback only performs a local atomic queue commit;
network upload remains asynchronous.  A journal entry is durably enqueued
while ``self._lock`` serialises its key, then the assembler advances the
index and accrues payment.  This deliberately keeps the short local write in
the critical section: releasing the lock between prepare and commit allowed
another proof callback to reserve a later key, and mutating state before the
write created paid journal holes on disk failure.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, NamedTuple, Sequence

from reliquary.constants import B_BATCH, FILL_CLOSED_EMISSIONS_PER_WINDOW
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


class _PreparedWrite(NamedTuple):
    """One uncommitted payload or tombstone for the current journal key."""

    key: int
    data: bytes
    is_tombstone: bool
    log_message: str
    window_batches: dict[str, list[Any]] | None
    reset_accumulator: bool


class FillClosedBatchAssembler:
    """Joins B_BATCH-per-environment chunks for exactly one window."""

    def __init__(
        self,
        *,
        window_start: int,
        env_order: Sequence[str],
        enqueue_fn: Callable[[int, bytes], None],
        tombstone_fn: Callable[[int, bytes], None],
        window_pool: float = 1.0,
    ) -> None:
        self.window_start = int(window_start)
        self._env_order = list(env_order)
        self._enqueue_fn = enqueue_fn
        self._tombstone_fn = tombstone_fn
        # R20: this window's whole emission budget. v6 has no auction --
        # and the seal path IS the auction -- so payment is computed here,
        # the only place a v6 window's ASSEMBLED batches are known. See
        # ``_accrue_payment_locked`` for how one batch's draw is derived.
        self._window_pool = float(window_pool)
        self._rewards_by_hotkey: dict[str, float] = {}
        # R24: every group this window actually PAID, in payment order, per
        # environment, each paired with the batch index it was paid in
        # (R28). The archive reads this in place of the auction's
        # winners -- under v6 the seal path selects nothing, so a
        # weight-only validator replaying ``rewards_by_hotkey`` from
        # ``eos_tokens`` has to divide over exactly the set the live
        # validator divided over, batch by batch: the pool is split per
        # assembled batch, so a hotkey paid in two batches does not
        # reproduce from one flat per-window division.
        self._paid_groups: dict[str, list[tuple[int, Any]]] = {
            environment: [] for environment in self._env_order
        }
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
        # Counts only entries whose local durable enqueue completed.  The
        # rotation gate uses the payload count to decide whether a successor
        # checkpoint is actually expected; padding/quarantine tombstones do
        # not create optimizer steps.
        self.durable_payload_count: int = 0
        self.durable_tombstone_count: int = 0
        # One lock: two per-environment batchers can call ``accept`` from
        # two different proof-worker device threads at once, and both the
        # accumulator and the pending queues are shared, mutable state.
        self._lock = threading.Lock()
        # R16: set by ``close()``, under ``_lock``, so a second call (or a
        # call racing a final ``accept``) is a no-op rather than a second
        # payload/tombstone under the same batch index.
        self._closed = False
        # R19 (amended): a straggler proof-worker callback landing after
        # ``close()`` is a NORMAL race (a proof finishing just after the
        # window seals), not a fault -- it is recorded here rather than
        # raised, and exposed through ``remainder_snapshot()`` so the
        # service can carry it into the window's own telemetry/archive.
        self._straggler_count = 0
        self._last_straggler_environment: str | None = None
        self._last_straggler_batch_index: int | None = None

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
            # R19 (amended): a straggler proof-worker callback landing
            # after ``close()`` has already run is a NORMAL race -- a
            # proof finishing just after the window sealed -- not a
            # fault. Escalating it (the original R19 raised here) faulted
            # the whole proof plane over an ordinary timing race. Checked
            # under the SAME lock ``close()`` sets ``_closed`` under, so
            # this can never race a concurrent ``close()`` into merging
            # in right as the window shuts.
            if self._closed:
                self._straggler_count += 1
                self._last_straggler_environment = environment
                self._last_straggler_batch_index = self.next_batch_index
                logger.error(
                    "FillClosedBatchAssembler: straggler chunk for "
                    "window %d env %s arrived after close() -- %d proven "
                    "groups lost to training (window already sealed); "
                    "straggler count now %d",
                    self.window_start,
                    environment,
                    len(groups),
                    self._straggler_count,
                )
                return
            # ``pick_training_batch`` returns the current chunk to its proven
            # pool when this callback raises.  Snapshot the join state before
            # accepting it so a failed local journal commit cannot leave the
            # same groups both retained here and pickable upstream.
            before = self._join_state_snapshot_locked()
            try:
                self._checkpoint_revision = str(checkpoint_revision)
                self._pending[environment].append(list(groups))
                prepared = self._prepare_next_ready_locked()
                if prepared is not None:
                    self._commit_write_locked(prepared)
            except Exception:
                self._restore_join_state_locked(before)
                raise

    def _join_state_snapshot_locked(self) -> dict[str, Any]:
        """Minimal rollback image for one ``accept`` transaction."""
        return {
            "accumulator_groups": {
                environment: list(groups)
                for environment, groups in self._accumulator._groups.items()
            },
            "accumulator_windows": {
                environment: list(windows)
                for environment, windows in self._accumulator._source_windows.items()
            },
            "accumulator_revision": self._accumulator.checkpoint_revision,
            "pending": {
                environment: [list(chunk) for chunk in chunks]
                for environment, chunks in self._pending.items()
            },
            "checkpoint_revision": self._checkpoint_revision,
            "next_batch_index": self.next_batch_index,
            "durable_payload_count": self.durable_payload_count,
            "durable_tombstone_count": self.durable_tombstone_count,
            "rewards_by_hotkey": dict(self._rewards_by_hotkey),
            "paid_groups": {
                environment: list(groups)
                for environment, groups in self._paid_groups.items()
            },
        }

    def _restore_join_state_locked(self, snapshot: dict[str, Any]) -> None:
        for environment in self._env_order:
            self._accumulator._groups[environment] = list(
                snapshot["accumulator_groups"][environment]
            )
            self._accumulator._source_windows[environment] = list(
                snapshot["accumulator_windows"][environment]
            )
            self._pending[environment] = [
                list(chunk) for chunk in snapshot["pending"][environment]
            ]
        self._accumulator.checkpoint_revision = snapshot[
            "accumulator_revision"
        ]
        self._checkpoint_revision = snapshot["checkpoint_revision"]
        self.next_batch_index = snapshot["next_batch_index"]
        self.durable_payload_count = snapshot["durable_payload_count"]
        self.durable_tombstone_count = snapshot["durable_tombstone_count"]
        self._rewards_by_hotkey = dict(snapshot["rewards_by_hotkey"])
        self._paid_groups = {
            environment: list(groups)
            for environment, groups in snapshot["paid_groups"].items()
        }

    def _prepare_next_ready_locked(self) -> _PreparedWrite | None:
        """Feed queued chunks until one full cycle is ready.

        At most one write is prepared per ``accept`` call.  That makes the
        upstream callback transaction unambiguous: either this exact chunk is
        durably owned by the assembler, or the whole call rolls back and the
        batcher may safely make it pickable again.
        """
        while not self._accumulator.ready:
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
            if not fed_any:
                return None
        return self._prepare_payload_locked(allow_partial=False)

    def _prepare_payload_locked(
        self, *, allow_partial: bool
    ) -> _PreparedWrite:
        """Encode one entry without advancing accounting or journal state."""
        extracted = self._accumulator.training_batches(
            self._env_order, allow_partial=allow_partial,
        )
        window_batches = dict(zip(self._env_order, extracted))
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
        #
        # Quarantine protects model state, not emission.  Payment is still
        # computed over this batch, but only in ``_commit_write_locked`` after
        # its payload/tombstone is durably present in the local journal.
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
            is_tombstone = True
        else:
            data = encode_training_payload(
                window_batches,
                window_start=self.window_start,
                checkpoint_revision=self._checkpoint_revision,
                env_order=list(self._env_order),
                window_quarantine=decision.to_archive(),
                checkpoint_epoch=None,
            )
            is_tombstone = False
        log_message = (
            "FillClosedBatchAssembler: window %d batch %d %s (journal "
            "key %d, quarantined=%s reasons=%s)" % (
                self.window_start, self.next_batch_index,
                "tombstoned" if is_tombstone else "written",
                key, decision.quarantined, decision.reasons,
            )
        )
        return _PreparedWrite(
            key,
            data,
            is_tombstone,
            log_message,
            window_batches,
            True,
        )

    def _prepare_incomplete_remainder_tombstone_locked(self) -> _PreparedWrite:
        """R16: the ``close()`` path when at least one environment
        contributed nothing to the final, partial cycle. There is no
        batch to assess -- quarantine (R14) judges assembled batches, and
        no batch was assembled -- so this writes a plain tombstone under
        the next batch index, distinct from the R14 quarantine tombstone,
        purely so the trainer's journal cursor still advances instead of
        stalling on a window whose remainder never became a payload.
        """
        key = encoded_window_journal_key(
            self.window_start, self.next_batch_index
        )
        data = encode_tombstone(
            window_start=self.window_start,
            failure_stage="fill_closed_incomplete_remainder",
            failure_type="IncompleteRemainder",
        )
        log_message = (
            "FillClosedBatchAssembler: window %d batch %d tombstoned at "
            "close (journal key %d, incomplete remainder %s)" % (
                self.window_start, self.next_batch_index, key,
                self._accumulator.snapshot()["counts"],
            )
        )
        return _PreparedWrite(
            key,
            data,
            True,
            log_message,
            None,
            True,
        )

    def _commit_write_locked(self, entry: _PreparedWrite) -> None:
        """Durably enqueue ``entry`` and only then commit accounting state."""
        if entry.key != encoded_window_journal_key(
            self.window_start, self.next_batch_index
        ):
            raise RuntimeError("prepared journal key is no longer current")
        before = self._join_state_snapshot_locked()
        try:
            if entry.is_tombstone:
                self._tombstone_fn(entry.key, entry.data)
            else:
                self._enqueue_fn(entry.key, entry.data)

            # Everything below is in-memory and is expected not to fail.  The
            # rollback image still covers it: if a future reward implementation
            # raises after a partial mutation, an idempotent durable retry must
            # not double-pay or skip this key.
            if entry.window_batches is not None:
                self._accrue_payment_locked(
                    entry.window_batches, batch_index=self.next_batch_index
                )
            if entry.reset_accumulator:
                self._accumulator.reset()
            self.next_batch_index += 1
            if entry.is_tombstone:
                self.durable_tombstone_count += 1
            else:
                self.durable_payload_count += 1
        except Exception:
            self._restore_join_state_locked(before)
            raise
        logger.info(entry.log_message)

    def _accrue_payment_locked(
        self, window_batches: dict[str, list[Any]], *, batch_index: int
    ) -> None:
        """Credit one assembled batch into this window's reward map (R20).

        Under v6 there is no auction to pay at seal, so this is where
        emission is decided: per assembled batch, by EOS-terminated
        completion tokens (``split_environment_pool``), never by a flat
        slot share. ``eos_tokens`` was produced once at admission
        (``admission.count_eos_completion_tokens``) and is read here as a
        plain attribute -- never recomputed.

        Two divisors, both deliberate:

        * ``len(self._env_order)`` -- each environment keeps its own pool,
          exactly as the seal path's ``pool_per_env`` does. Pooling the
          environments together would let a long-completion environment
          take a short one's emission through raw token mass alone.
        * ``FILL_CLOSED_EMISSIONS_PER_WINDOW`` (R15) -- a v6 window emits
          up to that many batches where the seal path emitted exactly
          one, so one window's pool is spread evenly over its batches.
          Totals are identical to splitting the pool once per window: N
          batches x pool/N. A window that closes with fewer batches pays
          out proportionally less and burns the rest, which is what v4/v5
          already do with unfilled slots (see ``batch_selection``'s module
          docstring) -- burn, never redistribute.

        Called with ``_lock`` held and does no I/O, per R17.
        """
        from reliquary.validator.token_rewards import (
            AcceptedGroup,
            split_environment_pool,
        )

        if not self._env_order:
            return
        batch_pool_per_env = (
            self._window_pool
            / len(self._env_order)
            / FILL_CLOSED_EMISSIONS_PER_WINDOW
        )
        for environment, env_batch in window_batches.items():
            self._paid_groups.setdefault(environment, []).extend(
                (int(batch_index), group) for group in env_batch
            )
            shares = split_environment_pool(
                [
                    AcceptedGroup(
                        hotkey=str(getattr(group, "hotkey", "")),
                        # Archive-only; no cap keys on it, by design (see
                        # token_rewards.py). ValidSubmission carries no
                        # operator_id, so the hotkey stands in.
                        operator_id=str(
                            getattr(group, "operator_id", None)
                            or getattr(group, "hotkey", "")
                        ),
                        eos_tokens=int(getattr(group, "eos_tokens", 0) or 0),
                    )
                    for group in env_batch
                ],
                pool=batch_pool_per_env,
            )
            for hotkey, share in shares.items():
                self._rewards_by_hotkey[hotkey] = (
                    self._rewards_by_hotkey.get(hotkey, 0.0) + share
                )

    def reward_map(self) -> dict[str, float]:
        """This window's emission so far, ``{hotkey: reward}`` (R20).

        Complete once ``close()`` has returned: every batch this window
        assembled -- including the partial remainder ``close()`` forces
        out -- goes through ``_prepare_payload_locked``, which credits it.
        The service reads this at archive time in place of the auction's
        map.
        """
        with self._lock:
            return dict(self._rewards_by_hotkey)

    def paid_groups(self) -> dict[str, list[tuple[int, Any]]]:
        """The groups this window's reward map was computed over (R24),
        each as ``(batch_index, group)`` (R28).

        Complete once ``close()`` has returned, exactly like
        ``reward_map()`` -- the two are written in the same critical
        section, so the archive can never carry a payment whose group is
        missing or a group that was never paid.

        The batch index is not decoration: ``_accrue_payment_locked``
        splits ``pool / n_envs / EMISSIONS`` WITHIN each batch, so the
        archive's replay has to group by it. Divide one window's whole
        pool over one window's whole token mass and any hotkey paid in
        two batches comes out wrong.
        """
        with self._lock:
            return {
                environment: list(groups)
                for environment, groups in self._paid_groups.items()
            }

    def close(self) -> None:
        """Called once by the service when this window closes (R16),
        after every batcher has handed its last chunk to ``accept``. A
        window's final cycle rarely lands exactly on B_BATCH for every
        environment; before this method existed, that remainder was only
        ever read (``remainder_snapshot``, one window later, as a
        WARNING) -- proven, paid rollouts were silently dropped with no
        marker.

        If every environment contributed at least one group to the
        current cycle, that partial cycle is emitted as one final batch
        -- a short DAPO minibatch is still a valid optimizer step, and
        the seal path already trains on partial windows -- through the
        SAME quarantine gate (R14) a full batch clears. Otherwise (some
        environment contributed nothing at all) a tombstone is written
        under the next batch's key instead, so the trainer's cursor still
        advances.

        Idempotent: a second call, or one racing a final in-flight
        ``accept``, is a no-op -- ``_closed`` is set under the same lock
        that gates the decision of what (if anything) to write. A
        straggler ``accept`` landing after this point does NOT raise
        (R19, amended): it is a normal race, not a fault, so it is
        recorded (``remainder_snapshot()``) and logged at ERROR instead.

        R19: before deciding what the remainder even IS, this drains
        ``_pending`` the same way ``_drain_locked`` always does --
        because a chunk can be sitting there for a reason that has
        nothing to do with this being the window's last cycle: an
        environment's accumulator slot was already full (at B_BATCH)
        when a LATER chunk for that same environment arrived, so
        ``_drain_locked`` held it back rather than overflow the current
        cycle. Reading only ``self._accumulator`` -- as this method did
        before R19 -- misses that held chunk entirely: it is real,
        proven, paid work that a caller reading ``remainder_snapshot()``
        right after ``close()`` would still see as outstanding, silently
        unaccounted for. Forcing the remainder write below resets the
        accumulator and frees every environment's capacity again, so a
        held chunk gets a fresh chance to drain on the next pass; this
        repeats until both ``_pending`` and the accumulator are
        genuinely empty (or the window's own encoded range runs out),
        so ``remainder_snapshot()`` reports empty on both counts once
        this returns.
        """
        with self._lock:
            if self._closed:
                return
            # Do not mark closed until every owned journal slot is locally
            # durable.  If any callback raises, committed earlier slots stay
            # committed and a later ``close()`` resumes exactly at the failed
            # key; accepts are still allowed because their upstream picks were
            # not durably claimed.
            while self.next_batch_index < FILL_CLOSED_EMISSIONS_PER_WINDOW:
                # Absorb whatever fits without forcing anything -- the
                # SAME merge ``accept()`` uses, including writing any
                # full cross-env cycle it completes along the way.
                prepared = self._prepare_next_ready_locked()
                if prepared is not None:
                    self._commit_write_locked(prepared)
                    continue
                counts = self._accumulator.snapshot()["counts"]
                still_outstanding = any(counts.values()) or any(
                    self._pending[environment]
                    for environment in self._env_order
                )
                if not still_outstanding:
                    break
                if self._accumulator.has_groups_for_all_targets:
                    self._commit_write_locked(
                        self._prepare_payload_locked(allow_partial=True)
                    )
                else:
                    self._commit_write_locked(
                        self._prepare_incomplete_remainder_tombstone_locked()
                    )
                # Loop again: the forced write above just reset the
                # accumulator, freeing every environment's capacity --
                # anything still queued in ``_pending`` (R19's repro: a
                # second full chunk for an environment whose slot was
                # already at B_BATCH) gets pulled in on the next pass
                # instead of being left behind.
            # R18: whatever slots the loop above did not use, up to the
            # window's own ceiling, would otherwise stay unwritten
            # forever -- see ``_prepare_underfill_padding_locked``.
            while self.next_batch_index < FILL_CLOSED_EMISSIONS_PER_WINDOW:
                self._commit_write_locked(
                    self._prepare_underfill_padding_locked()
                )
            self._closed = True

    def _prepare_underfill_padding_locked(self) -> _PreparedWrite:
        """R18: ``WindowJournal.next_entry`` (trainer/journal.py) walks
        the encoded key space one integer at a time -- ``payload_key``,
        then ``tombstone_key``, else wait -- with no logic to jump past
        a window whose range it never fully used. A window that closes
        after fewer than ``FILL_CLOSED_EMISSIONS_PER_WINDOW`` batches
        (the common case: production proof budgets saturate well under
        the window's full target) leaves every later slot in ITS OWN
        range unwritten. The trainer's cursor parks on the first one
        forever -- not just missing this window's tail, but never
        reaching any later window either, since the journal "never
        advances on absence, only on an explicit marker" (journal.py's
        own module docstring). Tombstoning every remaining slot here,
        right after the remainder above claims its own, makes the
        journal gapless by construction for this window -- the same
        invariant a fully-filled window already gets for free by using
        every slot itself.
        """
        if self.next_batch_index >= FILL_CLOSED_EMISSIONS_PER_WINDOW:
            raise RuntimeError("fill-closed journal range is already complete")
        key = encoded_window_journal_key(
            self.window_start, self.next_batch_index
        )
        data = encode_tombstone(
            window_start=self.window_start,
            failure_stage="window_underfilled",
            failure_type="WindowUnderfilled",
        )
        log_message = (
            "FillClosedBatchAssembler: window %d batch %d padded "
            "with a tombstone at close (journal key %d, window "
            "underfilled)" % (
                self.window_start, self.next_batch_index, key,
            )
        )
        return _PreparedWrite(
            key,
            data,
            True,
            log_message,
            None,
            False,
        )

    def remainder_snapshot(self) -> dict[str, Any]:
        """Groups still held and not yet part of a written payload, for
        observability (R16: ``close()`` is now what disposes of this at
        window seal -- a full accumulator cycle emitted as a partial
        batch, or tombstoned -- so by the time a caller reads this after
        ``close()`` has run, it reports an empty remainder).

        ``stragglers`` (R19, amended): a proof-worker callback landing
        after ``close()`` no longer raises -- it is a normal race, not a
        fault -- so this is the one place that race becomes observable
        again, for the service to carry into the window's own
        telemetry/archive. ``count`` is 0 until the first straggler ever
        lands; ``last_environment``/``last_batch_index`` identify the
        most recent one only (not a full log -- see the ERROR-level log
        line for a per-occurrence record).
        """
        with self._lock:
            counts = self._accumulator.snapshot()["counts"]
            pending = {
                environment: len(chunks)
                for environment, chunks in self._pending.items()
            }
            stragglers = {
                "count": self._straggler_count,
                "last_environment": self._last_straggler_environment,
                "last_batch_index": self._last_straggler_batch_index,
            }
        return {
            "in_accumulator": dict(counts),
            "pending": pending,
            "stragglers": stragglers,
        }
