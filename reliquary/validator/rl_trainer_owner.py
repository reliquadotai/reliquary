"""One durable, renewable writer lease for an async training run."""
from __future__ import annotations

import re

OWNER_HEADER = "X-Trainer-Owner"
EPOCH_HEADER = "X-Trainer-Epoch"
LEASE_SECONDS = 180.0
HEARTBEAT_SECONDS = 30.0
_OWNER = re.compile(r"[0-9a-f]{32}")


class TrainerOwnerRefused(RuntimeError):
    pass


def require_trainer_epoch(book, epoch: int) -> None:
    """Validate the owner inside the transaction committing a slow operation's result."""
    row = book.db.execute("SELECT epoch,owner,expires_at FROM rl_trainer_owner WHERE id=1").fetchone()
    if (row is None or type(epoch) is not int or row[0] != epoch or row[1] is None
            or row[2] <= book.clock()):
        raise TrainerOwnerRefused("trainer_owner_lost")


class TrainerOwner:
    def __init__(self, book):
        self.book = book
        with book.transaction():
            book.db.execute("CREATE TABLE IF NOT EXISTS rl_trainer_owner ("
                            "id INTEGER PRIMARY KEY CHECK(id=1), epoch INTEGER NOT NULL, "
                            "owner TEXT, expires_at REAL NOT NULL)")
            book.db.execute("INSERT OR IGNORE INTO rl_trainer_owner VALUES(1,0,NULL,0)")

    def _row(self):
        return self.book.db.execute("SELECT epoch,owner,expires_at FROM rl_trainer_owner WHERE id=1").fetchone()

    def claim(self, owner: str) -> dict:
        if not isinstance(owner, str) or not _OWNER.fullmatch(owner):
            raise ValueError("owner must be a random 32-character hex identifier")
        with self.book.transaction():
            epoch, current, expires = self._row()
            now = float(self.book.clock())
            if current is not None and expires > now and current != owner:
                raise TrainerOwnerRefused("trainer_owner_busy")
            if current != owner or expires <= now:
                epoch += 1
            self.book.db.execute("UPDATE rl_trainer_owner SET epoch=?,owner=?,expires_at=? WHERE id=1",
                                 (epoch, owner, now + LEASE_SECONDS))
            return {"owner": owner, "epoch": epoch, "lease_seconds": LEASE_SECONDS}

    def require(self, owner: str | None, epoch: str | int | None) -> int:
        try:
            supplied_epoch = int(epoch)
        except (TypeError, ValueError):
            raise TrainerOwnerRefused("trainer_owner_required") from None
        current_epoch, current, expires = self._row()
        if not owner or current != owner or supplied_epoch != current_epoch or expires <= self.book.clock():
            raise TrainerOwnerRefused("trainer_owner_lost")
        return current_epoch

    def heartbeat(self, owner, epoch) -> dict:
        with self.book.transaction():
            current = self.require(owner, epoch)
            self.book.db.execute("UPDATE rl_trainer_owner SET expires_at=? WHERE id=1",
                                 (float(self.book.clock()) + LEASE_SECONDS,))
            return {"owner": owner, "epoch": current, "lease_seconds": LEASE_SECONDS}

    def release(self, owner, epoch) -> dict:
        with self.book.transaction():
            self.require(owner, epoch)
            self.book.db.execute("UPDATE rl_trainer_owner SET owner=NULL,expires_at=0 WHERE id=1")
            return {"ok": True}
