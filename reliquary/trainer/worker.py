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
        self.trained_since_publish = 0
        self.adaptive_publication_pending = False
        self.tombstones_seen = 0
        self.quarantined_seen = 0
        self.health_skips = 0

    def _publication_due(self) -> bool:
        return (
            self.trained_since_publish >= self.publish_every
            or self.adaptive_publication_pending
        )

    def _publish(self) -> str:
        if self.shadow:
            # Shadow mode trains but never publishes; reset counters so
            # the loop keeps consuming.
            self.trained_since_publish = 0
            self.adaptive_publication_pending = False
            return "published"
        head = self._head_revision_fn()
        if (
            self.last_published_revision is not None
            and head is not None
            and head != self.last_published_revision
        ):
            raise TrainerLockLost(
                f"checkpoint repo HEAD {head!r} is not ours "
                f"({self.last_published_revision!r}); refusing to publish"
            )
        reason = (
            "adaptive_policy_ratio_drift"
            if self.adaptive_publication_pending else "cadence"
        )
        self.last_published_revision = self._publish_fn(reason)
        self.trained_since_publish = 0
        self.adaptive_publication_pending = False
        return "published"

    def run_once(self) -> str:
        if self._publication_due():
            return self._publish()
        entry = self._journal.next_entry(self.cursor, stride=self.stride)
        if entry is None:
            return "waited"
        kind, value = entry
        if kind == "tombstone":
            self.cursor += self.stride
            self.tombstones_seen += 1
            logger.warning("window %s tombstoned: %s", self.cursor, value)
            return "tombstone"
        if bool(value.window_quarantine.get("quarantined")):
            self.cursor += self.stride
            self.quarantined_seen += 1
            logger.warning(
                "window %s quarantined at seal; skipping", self.cursor,
            )
            return "quarantined"
        try:
            trained = self._train_fn(value)
        except TrainingStepSkipped as exc:
            self.cursor += self.stride
            self.health_skips += 1
            if (
                exc.reason == "policy_ratio_drift"
                and self.trained_since_publish > 0
            ):
                self.adaptive_publication_pending = True
            logger.warning(
                "train step skipped for window %s: %s",
                self.cursor, exc.reason,
            )
            return "trained"
        self.cursor += self.stride
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
