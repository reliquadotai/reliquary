"""Detached trainer main loop: journal cursor -> train_step -> publish.

State machine per run_once(): a due publication runs BEFORE consuming
more windows (mirrors the validator's checkpoint_publication_pending
behavior). Otherwise consume exactly one journal entry or report
"waited". The cursor only advances on an explicit payload or tombstone.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from reliquary.validator.training import TrainingStepSkipped

logger = logging.getLogger(__name__)


class TrainerLockLost(RuntimeError):
    """Another publisher moved the checkpoint repo HEAD; halt loudly."""


class TrainerWorker:
    def __init__(
        self,
        *,
        journal: Any,
        train_fn: Callable[[Any], bool],
        publish_fn: Callable[[str], str],
        head_revision_fn: Callable[[], str | None],
        cursor: int,
        stride: int,
        publish_every: int,
        last_published_revision: str | None,
        shadow: bool = False,
        freeze_fn: Callable[[], str | None] | None = None,
        cursor_writer: Callable[[int], None] | None = None,
        drain_request_fn: Callable[[], dict] | None = None,
        finish_fn: Callable[[], bool] | None = None,
        publication_pending_fn: Callable[[], bool] | None = None,
    ) -> None:
        self._journal = journal
        self._train_fn = train_fn
        self._publish_fn = publish_fn
        self._head_revision_fn = head_revision_fn
        self.cursor = int(cursor)
        self.stride = int(stride)
        self.publish_every = int(publish_every)
        self.last_published_revision = last_published_revision
        self.shadow = bool(shadow)
        self._freeze_fn = freeze_fn
        self._drain_request_fn = drain_request_fn
        self._finish_fn = finish_fn
        self._publication_pending_fn = publication_pending_fn
        self._published_cursor = self.cursor
        # Amendment v6.1 (trainer-paced picks): advisory pacing telemetry,
        # written every time the live trainer's journal cursor advances,
        # on every profile. Shadow consumption must not pace the validator.
        self._cursor_writer = cursor_writer
        self.trained_since_publish = 0
        self.adaptive_publication_pending = False
        self.tombstones_seen = 0
        self.quarantined_seen = 0
        self.health_skips = 0

    def _advance_cursor(self) -> None:
        """The single place the journal cursor moves forward.

        Every advance -- trained payload, tombstone, quarantine skip,
        health skip -- is a journal key the
        validator's picker can now count as consumed, whether or not an
        optimizer step happened on it. Publishing the telemetry cursor
        here for the live trainer keeps the pacer from stalling on any of
        those non-training advances (a tombstoned or quarantined key is
        never coming back to be trained).
        """
        self.cursor += self.stride
        self._write_cursor(self.cursor)

    def _write_cursor(self, journal_key: int) -> None:
        if self.shadow or self._cursor_writer is None:
            return
        try:
            self._cursor_writer(journal_key)
        except Exception:
            # Advisory only: training must never fail because pacing
            # telemetry could not be written (R2 hiccup, disk full, ...).
            logger.warning(
                "cursor telemetry write failed for journal key %s "
                "(advisory, ignored)", journal_key, exc_info=True,
            )

    def _publication_due(self) -> bool:
        return (
            self.trained_since_publish >= self.publish_every
            or self.adaptive_publication_pending
        )

    def _publish(self, reason: str | None = None) -> str:
        if self.shadow:
            # Shadow mode trains but never publishes; reset counters so
            # the loop keeps consuming.
            self.trained_since_publish = 0
            self.adaptive_publication_pending = False
            self._published_cursor = self.cursor
            return "published"
        pending = self._publication_pending_fn is not None and self._publication_pending_fn()
        head = self._head_revision_fn() if not pending else None
        if head is None and not pending:
            raise RuntimeError("checkpoint repo HEAD unavailable; refusing unguarded publication")
        if (
            self.last_published_revision is not None
            and not pending
            and head is not None
            and head != self.last_published_revision
        ):
            raise TrainerLockLost(
                f"checkpoint repo HEAD {head!r} is not ours "
                f"({self.last_published_revision!r}); refusing to publish"
            )
        reason = reason or (
            "adaptive_policy_ratio_drift"
            if self.adaptive_publication_pending else "cadence"
        )
        self.last_published_revision = self._publish_fn(reason)
        self.trained_since_publish = 0
        self.adaptive_publication_pending = False
        self._published_cursor = self.cursor
        return "published"

    def run_once(self) -> str:
        # Incident kill-switches (emergency freeze / checkpoint ceiling)
        # must work in the detached path too: frozen means no consuming,
        # no training, no publishing — the journal simply backs up.
        if self._freeze_fn is not None:
            reason = self._freeze_fn()
            if reason:
                logger.warning("trainer frozen: %s", reason)
                return "frozen"
        if self._drain_request_fn is not None:
            request = self._drain_request_fn()
            if request["mode"] == "drain":
                target = request["target_cursor"]
                if target is None:
                    return "closed"
                if target < self.cursor or (target - self.cursor) % self.stride:
                    raise ValueError("drain cursor is behind the trainer or misaligned")
                if target == self.cursor:
                    if self._finish_fn is None:
                        raise RuntimeError("trainer drain requires accumulator flush wiring")
                    if self._finish_fn():
                        self.trained_since_publish += 1
                    if self.trained_since_publish or self._published_cursor != self.cursor:
                        return self._publish("cutover_drain")
                    return "drained"
        if self._publication_due():
            return self._publish()
        entry = self._journal.next_entry(self.cursor, stride=self.stride)
        if entry is None:
            return "waited"
        kind, value = entry
        if kind == "tombstone":
            self._advance_cursor()
            self.tombstones_seen += 1
            logger.warning("window %s tombstoned: %s", self.cursor, value)
            return "tombstone"
        window_quarantined = bool(value.window_quarantine.get("quarantined"))
        if window_quarantined:
            self.quarantined_seen += 1
        if window_quarantined:
            self._advance_cursor()
            logger.warning(
                "window %s quarantined at seal; skipping",
                self.cursor,
            )
            return "quarantined"
        try:
            trained = self._train_fn(value)
        except TrainingStepSkipped as exc:
            self._advance_cursor()
            self.health_skips += 1
            if exc.reason == "policy_ratio_drift" and self.trained_since_publish > 0:
                self.adaptive_publication_pending = True
            logger.warning(
                "train step skipped for window %s: %s",
                self.cursor,
                exc.reason,
            )
            return "trained"
        self._advance_cursor()
        if trained:
            self.trained_since_publish += 1
        return "trained"

    def snapshot(self) -> dict[str, Any]:
        return {
            "cursor": self.cursor,
            "trained_since_publish": self.trained_since_publish,
            "adaptive_publication_pending": self.adaptive_publication_pending,
            "last_published_revision": self.last_published_revision,
            "tombstones_seen": self.tombstones_seen,
            "quarantined_seen": self.quarantined_seen,
            "health_skips": self.health_skips,
            "shadow": self.shadow,
        }
