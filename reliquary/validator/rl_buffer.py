"""The async training buffer (rl-async-design.md §4.7; implementation map §1.8): in-zone groups wait here for the
trainer, tagged with what it needs to verify and train them in one forward.

A group is buffered by the submission path (``rl_submit``) in the same transaction that closes its lease, in the
lease book's SQLite file, so a restart loses none. The trainer (P4) ``take``s groups (freshest first within an
env: highest k, then latest arrival), runs its forward against the published checkpoint (Gumbel picks from the
lease ``nonce``), and reports each group back through ``RlSubmissions.report_training`` (which removes it). Groups
taken but never reported (a trainer restart) go back to the queue when the buffer is opened again, or by
``release``. Env mix, max-lag drop and batch assembly are P4's.
"""
from __future__ import annotations

import json
import zlib
from dataclasses import asdict, dataclass

QUEUED, TAKEN = "queued", "taken"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rl_buffer(lease_id TEXT PRIMARY KEY, environment TEXT NOT NULL, k INTEGER NOT NULL,
    arrived_at REAL NOT NULL, hotkey TEXT NOT NULL, state TEXT NOT NULL, body BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS rl_buffer_fresh ON rl_buffer(state, environment, k, arrived_at);
"""


@dataclass(frozen=True)
class SubmittedGroup:
    """A graded group with its tokens: what the trainer (in zone) or an audit (out of zone) runs its forward on.
    ``nonce`` (hex) and ``checkpoint`` are the lease's: the Gumbel noise is keyed by (nonce, rollout index,
    position) as ``protocol.rl_draw`` defines it, so ``draw_uniforms`` recomputes the uniforms (no array stored).
    ``completions[i]`` is rollout i's completion, ``force_spans[i]`` its BFT span (completion-relative) or None."""
    lease_id: str
    hotkey: str
    environment: str
    prompt_idx: int
    checkpoint: int
    nonce: str
    arrived_at: float
    rewards: tuple[float, ...]
    truncated: tuple[bool, ...]
    prompt_tokens: tuple[int, ...]
    completions: tuple[tuple[int, ...], ...]
    force_spans: tuple[tuple[int, int] | None, ...]

    def pack(self) -> bytes:
        return zlib.compress(json.dumps(asdict(self), separators=(",", ":")).encode(), 6)

    def draw_uniforms(self, rollout_index: int) -> list[float]:
        """The forced draw's public uniforms of rollout ``rollout_index``, one per completion position, recomputed
        from the lease (``protocol.rl_draw``): what the trainer's verification and substitution loss read (the
        window path's ``_forced_draw_uniforms``). Nothing is stored: the nonce, prompt, k and index are the seed."""
        from reliquary.protocol.rl_draw import lease_uniforms

        return lease_uniforms(self.nonce, self.prompt_idx, self.checkpoint, rollout_index,
                              len(self.completions[rollout_index]))

    @classmethod
    def unpack(cls, blob: bytes) -> "SubmittedGroup":
        body = json.loads(zlib.decompress(blob))
        return cls(**{**body, "rewards": tuple(body["rewards"]), "truncated": tuple(body["truncated"]),
                      "prompt_tokens": tuple(body["prompt_tokens"]),
                      "completions": tuple(tuple(c) for c in body["completions"]),
                      "force_spans": tuple(None if s is None else tuple(s) for s in body["force_spans"])})


class TrainingBuffer:
    def __init__(self, book) -> None:
        self.book = book
        with book.lock:
            book.db.executescript(_SCHEMA)
            book.db.commit()
        with book.transaction():                              # a trainer that died holding groups: give them back
            book.db.execute("UPDATE rl_buffer SET state=? WHERE state=?", (QUEUED, TAKEN))

    def push(self, group: SubmittedGroup) -> None:
        with self.book.transaction():
            self.book.db.execute("INSERT INTO rl_buffer(lease_id, environment, k, arrived_at, hotkey, state, body) "
                                 "VALUES(?,?,?,?,?,?,?)", (group.lease_id, group.environment, group.checkpoint,
                                                           group.arrived_at, group.hotkey, QUEUED, group.pack()))

    def take(self, environment: str, limit: int) -> list[SubmittedGroup]:
        """Up to ``limit`` queued groups of ``environment``, freshest first; they stay buffered (``taken``) until
        reported or released."""
        with self.book.transaction():
            rows = self.book.db.execute("SELECT lease_id, body FROM rl_buffer WHERE state=? AND environment=? "
                                        "ORDER BY k DESC, arrived_at DESC LIMIT ?",
                                        (QUEUED, environment, int(limit))).fetchall()
            self.book.db.executemany("UPDATE rl_buffer SET state=? WHERE lease_id=?",
                                     [(TAKEN, lease_id) for lease_id, _ in rows])
        return [SubmittedGroup.unpack(body) for _, body in rows]

    def release(self, lease_ids) -> None:
        with self.book.transaction():
            self.book.db.executemany("UPDATE rl_buffer SET state=? WHERE lease_id=? AND state=?",
                                     [(QUEUED, lease_id, TAKEN) for lease_id in lease_ids])

    def get(self, lease_id: str) -> SubmittedGroup | None:
        with self.book.lock:
            row = self.book.db.execute("SELECT body FROM rl_buffer WHERE lease_id=?", (lease_id,)).fetchone()
        return None if row is None else SubmittedGroup.unpack(row[0])

    def _remove(self, lease_id: str) -> bool:
        """Inside the caller's transaction (``report_training``)."""
        return self.book.db.execute("DELETE FROM rl_buffer WHERE lease_id=?", (lease_id,)).rowcount == 1

    def depth(self) -> dict[str, int]:
        """Queued groups per env (what the price controller and the trainer's mix read)."""
        with self.book.lock:
            return dict(self.book.db.execute("SELECT environment, COUNT(*) FROM rl_buffer WHERE state=? "
                                             "GROUP BY environment", (QUEUED,)).fetchall())


__all__ = ["QUEUED", "TAKEN", "SubmittedGroup", "TrainingBuffer"]
