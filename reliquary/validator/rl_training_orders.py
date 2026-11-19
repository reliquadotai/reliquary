"""Fail-closed lease admission for an operator-started training order; existing work still drains."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import asdict

logger = logging.getLogger(__name__)
REFRESH_SECONDS = 5.0
READ_TIMEOUT_SECONDS = 10.0
SNAPSHOT_TTL_SECONDS = 15.0


def _identity(entry) -> str:
    document = asdict(entry)
    for key in ("status", "retired_at", "admission"):
        document.pop(key, None)
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


class TrainingAdmission:
    """Only a fresh matching active registry entry permits new leases. Retirement is durable."""

    def __init__(self, entry, book, *, load, clock=time.monotonic):
        self.entry, self.book, self.load, self.clock = entry, book, load, clock
        self.identity = _identity(entry)
        self._checked_at = None
        self._reason = "registry_unavailable"
        with book.transaction():
            row = book.db.execute("SELECT value FROM rl_lease_meta WHERE key='training_identity'").fetchone()
            if row is not None and row[0] != self.identity:
                self._stop("registry_changed")
            book.db.execute("INSERT OR IGNORE INTO rl_lease_meta(key,value) VALUES('training_identity',?)",
                            (self.identity,))

    def _stop(self, reason: str) -> None:
        with self.book.transaction():
            self.book.db.execute("INSERT OR IGNORE INTO rl_lease_meta(key,value) VALUES('training_registry_stop',?)",
                                 (reason,))

    def reason(self) -> str | None:
        with self.book.lock:
            row = self.book.db.execute("SELECT value FROM rl_lease_meta WHERE key='training_registry_stop'").fetchone()
        if row is not None:
            return row[0]
        age = None if self._checked_at is None else self.clock() - self._checked_at
        if age is None or not 0 <= age < SNAPSHOT_TTL_SECONDS:
            return "registry_unavailable"
        return self._reason

    async def refresh(self) -> bool:
        try:
            entry = await asyncio.wait_for(self.load(self.entry.task_id), timeout=READ_TIMEOUT_SECONDS)
            if entry is None or _identity(entry) != self.identity:
                self._stop("registry_changed")
            elif entry.status == "retired":
                self._stop("registry_retired")
            elif entry.status != "active":
                self._stop("registry_changed")
            self._reason = None if entry is not None and entry.admission == "open" else "registry_paused"
            self._checked_at = self.clock()
        except Exception:
            self._checked_at = None
            logger.exception("training registry refresh failed; new leases are fenced")
            return False
        return self.reason() is None

    async def refresh_forever(self) -> None:
        while True:
            await self.refresh()
            await asyncio.sleep(REFRESH_SECONDS)


async def load_training_entry(task_id: str):
    from reliquary.infrastructure.task_registry_store import read_registry

    entries, _ = await read_registry(strict=True)
    return entries.get(task_id)
