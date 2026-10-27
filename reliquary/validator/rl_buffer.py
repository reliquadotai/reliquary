"""The async training buffer (rl-async-design.md §4.7; implementation map §1.8): in-zone groups wait here for the
trainer, tagged with what it needs to verify and train them in one forward.

A group is buffered by the submission path (``rl_submit``) in the same transaction that closes its lease, in the
lease book's SQLite file, so a restart loses none. The trainer (P4) ``take``s groups (freshest first within an
env: highest k, then latest arrival), runs its forward against the published checkpoint (Gumbel picks from the
lease ``nonce``), and reports each group back through ``RlSubmissions.report_training`` (which removes it). Groups
taken but never reported go back to the queue when the sole trainer starts or recovers holding nothing, or by
``release``. Reopening the controller preserves groups still held by a live trainer. Env mix, max-lag drop and
batch assembly are P4's.

Each hand-out counts (``attempts``): a group taken again after a step that never reported it (the step raised, the
trainer died) goes after the fresh ones, at most one such group per take, so a group that breaks every step it is
in cannot stall the queue; past the trainer's limit it leaves through ``take_exhausted`` (reported untrained).
"""
from __future__ import annotations

import json
import time
import zlib
from dataclasses import asdict, dataclass

QUEUED, TAKEN = "queued", "taken"
SUSPECT_ATTEMPTS = 2        # hand-outs after which a group not yet trained is served at most one per batch
URGENT_LAG_MARGIN = 2       # this close to the max lag a group skips the env caps (a prefetch trains one k later)
MAX_POLL_GAP_S = 30.0       # a wait counts between two of the trainer's takes at most this far apart (a dead trainer)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rl_buffer(lease_id TEXT PRIMARY KEY, environment TEXT NOT NULL, k INTEGER NOT NULL,
    arrived_at REAL NOT NULL, hotkey TEXT NOT NULL, state TEXT NOT NULL, body BLOB NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS rl_buffer_fresh ON rl_buffer(state, environment, k, arrived_at);
"""


@dataclass(frozen=True)
class SubmittedGroup:
    """A graded group with its tokens: what the trainer (in zone) or an audit (out of zone) runs its forward on.
    ``nonce`` (hex) and ``checkpoint`` are the lease's: the Gumbel noise is keyed by (nonce, rollout index,
    position) as ``protocol.rl_draw`` defines it, so ``draw_uniforms`` recomputes the uniforms (no array stored).
    ``completions[i]`` is rollout i's completion, ``force_spans[i]`` its BFT span (completion-relative) or None.

    Episodes (§4.10): ``spans[i]`` are rollout i's model spans (completion-relative, validated at submission); the
    completion then also holds the tool observations, which take no draw position, carry no loss and are never
    matched. None for a single-turn group (every completion position is a model token)."""
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
    spans: tuple[tuple[tuple[int, int], ...], ...] | None = None

    def model_positions(self, rollout_index: int) -> list[int]:
        """Rollout ``rollout_index``'s model-token positions (completion-relative) in draw order."""
        from reliquary.protocol.rl_draw import model_positions

        return model_positions(len(self.completions[rollout_index]),
                               None if self.spans is None else self.spans[rollout_index])

    def model_tokens(self, rollout_index: int) -> int:
        if self.spans is None:
            return len(self.completions[rollout_index])
        return sum(max(0, int(end) - int(start)) for start, end in self.spans[rollout_index])

    def pack(self) -> bytes:
        return zlib.compress(json.dumps(asdict(self), separators=(",", ":")).encode(), 6)

    def draw_uniforms(self, rollout_index: int) -> list[float]:
        """The forced draw's public uniforms of rollout ``rollout_index``, one per model position, recomputed
        from the lease (``protocol.rl_draw``): what the trainer's verification and substitution loss read (the
        window path's ``_forced_draw_uniforms``), one per MODEL token for an episode. Nothing is stored: the nonce,
        prompt, k and index are the seed."""
        from reliquary.protocol.rl_draw import lease_uniforms

        return lease_uniforms(self.nonce, self.prompt_idx, self.checkpoint, rollout_index,
                              self.model_tokens(rollout_index))

    @classmethod
    def unpack(cls, blob: bytes) -> "SubmittedGroup":
        body = json.loads(zlib.decompress(blob))
        spans = body.get("spans")
        if spans is not None:
            body["spans"] = tuple(tuple(tuple(p) for p in rollout) for rollout in spans)
        return cls(**{**body, "rewards": tuple(body["rewards"]), "truncated": tuple(body["truncated"]),
                      "prompt_tokens": tuple(body["prompt_tokens"]),
                      "completions": tuple(tuple(c) for c in body["completions"]),
                      "force_spans": tuple(None if s is None else tuple(s) for s in body["force_spans"])})


def mixed_take(rows, caps: dict[str, int], limit: int, urgent_below: int | None = None) -> list[str]:
    """Lease ids for one take from queued ``rows`` (lease_id, env, k) in arrival order. Groups about to pass the
    max lag (``k < urgent_below``) go first, oldest checkpoint first, whatever their env's cap; then each env up to
    its cap; then, if the batch is still short, the capped envs' extra groups in arrival order, so the cap never
    makes the trainer wait for groups that are already queued."""
    chosen, counts, extra = [], {}, []
    urgent = sorted((r for r in rows if urgent_below is not None and r[2] < urgent_below), key=lambda r: r[2])
    picked = set()
    for lease_id, env, _ in urgent[:limit]:
        chosen.append(lease_id)
        picked.add(lease_id)
        counts[env] = counts.get(env, 0) + 1
    for lease_id, env, _ in rows:
        if len(chosen) >= limit:
            break
        if lease_id in picked:
            continue
        if env in caps and counts.get(env, 0) >= caps[env]:
            extra.append(lease_id)
            continue
        counts[env] = counts.get(env, 0) + 1
        chosen.append(lease_id)
    return chosen + extra[:max(0, limit - len(chosen))]


class TrainingBuffer:
    def __init__(self, book, clock=time.time, batch_caps=None) -> None:
        """``batch_caps(limit)``: env -> its share of the groups one take hands out (``rl_env_shares.EnvGroupCaps``;
        None: arrival order alone). A capped env's extra groups wait for the next take unless the batch would be
        short without them (``mixed_take``)."""
        self.book, self.clock, self.batch_caps = book, clock, batch_caps
        self._arrived: dict[str, int] = {}                   # completion tokens queued per env since the last drain
        self._taken = 0                                      # completion tokens the trainer took since the last drain
        self._starved_s = 0.0
        self._waiting_since: float | None = None             # the trainer's last take came back short
        with book.lock:
            book.db.executescript(_SCHEMA)
            if "attempts" not in {row[1] for row in book.db.execute("PRAGMA table_info(rl_buffer)")}:
                book.db.execute("ALTER TABLE rl_buffer ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")   # an older file
            book.db.commit()

    def requeue(self) -> int:
        """Every taken group back to the queue when the sole trainer starts or recovers holding nothing.
        Returns how many."""
        with self.book.transaction():
            return self.book.db.execute("UPDATE rl_buffer SET state=? WHERE state=?", (QUEUED, TAKEN)).rowcount

    def push(self, group: SubmittedGroup) -> None:
        with self.book.transaction():
            self.book.db.execute("INSERT INTO rl_buffer(lease_id, environment, k, arrived_at, hotkey, state, body) "
                                 "VALUES(?,?,?,?,?,?,?)", (group.lease_id, group.environment, group.checkpoint,
                                                           group.arrived_at, group.hotkey, QUEUED, group.pack()))
        tokens = sum(group.model_tokens(i) for i in range(len(group.completions)))
        with self.book.lock:
            self._arrived[group.environment] = self._arrived.get(group.environment, 0) + tokens

    def drain_tick(self):
        """The queue since the last drain (the price controller's signal; memory only: a restart loses at most one
        tick of it)."""
        from reliquary.validator.rl_price import QueueTick

        with self.book.lock:
            tick = QueueTick(self._arrived, self._taken, self._starved_s)
            self._arrived, self._taken, self._starved_s = {}, 0, 0.0
        return tick

    def take(self, limit: int) -> list[SubmittedGroup]:
        """Up to ``limit`` queued groups of any env, in arrival order (with ``batch_caps``: ``mixed_take``); a group
        handed out ``SUSPECT_ATTEMPTS`` times already comes after the fresh ones and only while no other such group
        is taken (one per batch: the trainer holds one batch at a time); they stay
        buffered (``taken``) until reported or released. The trainer asks for what its batch still lacks: a short
        answer means it waits for groups."""
        now = float(self.clock())
        with self.book.transaction():
            # A group handed out once and back (a restart, a lost answer, one failed step) is served like a fresh one;
            # from its second return on it is a suspect: at most one per batch, so it cannot sink the others again.
            caps = dict(self.batch_caps(int(limit)) or {}) if self.batch_caps is not None else {}
            if caps:
                rows = self.book.db.execute(
                    "SELECT lease_id, environment, k FROM rl_buffer WHERE state=? AND attempts<? "
                    "ORDER BY arrived_at, lease_id", (QUEUED, SUSPECT_ATTEMPTS)).fetchall()
                chosen = mixed_take(rows, caps, int(limit), self._urgent_below())
                bodies = dict(self.book.db.execute(
                    f"SELECT lease_id, body FROM rl_buffer WHERE lease_id IN ({','.join('?' * len(chosen))})",
                    chosen).fetchall()) if chosen else {}
                fresh = [(lease_id, bodies[lease_id]) for lease_id in chosen]
            else:
                fresh = self.book.db.execute("SELECT lease_id, body FROM rl_buffer WHERE state=? AND attempts<? "
                                             "ORDER BY arrived_at, lease_id LIMIT ?",
                                             (QUEUED, SUSPECT_ATTEMPTS, int(limit))).fetchall()
            held = self.book.db.execute("SELECT 1 FROM rl_buffer WHERE state=? AND attempts>? LIMIT 1",
                                        (TAKEN, SUSPECT_ATTEMPTS)).fetchone()
            retried = [] if len(fresh) >= limit or held else self.book.db.execute(
                "SELECT lease_id, body FROM rl_buffer WHERE state=? AND attempts>=? ORDER BY attempts, arrived_at, "
                "lease_id LIMIT 1", (QUEUED, SUSPECT_ATTEMPTS)).fetchall()
            rows = fresh + retried
            self.book.db.executemany("UPDATE rl_buffer SET state=?, attempts=attempts+1 WHERE lease_id=?",
                                     [(TAKEN, lease_id) for lease_id, _ in rows])
            if self._waiting_since is not None:
                self._starved_s += min(now - self._waiting_since, MAX_POLL_GAP_S)
            self._waiting_since = now if len(rows) < limit else None
        groups = [SubmittedGroup.unpack(body) for _, body in rows]
        with self.book.lock:
            self._taken += sum(g.model_tokens(i) for g in groups for i in range(len(g.completions)))
        return groups

    def _urgent_below(self) -> int | None:
        """Groups generated below this checkpoint are about to pass the trainer's max lag (None: unknown)."""
        latest = getattr(self.book, "latest_checkpoint", None)
        max_lag = getattr(getattr(self.book, "config", None), "max_lag", None)
        if latest is None or max_lag is None:
            return None
        return int(latest) - int(max_lag) + URGENT_LAG_MARGIN

    def take_older(self, below_k: int, limit: int) -> list[SubmittedGroup]:
        """Up to ``limit`` queued groups of any env generated on a checkpoint older than ``below_k``: the trainer's
        max-lag drop."""
        with self.book.transaction():
            rows = self.book.db.execute("SELECT lease_id, body FROM rl_buffer WHERE state=? AND k<? LIMIT ?",
                                        (QUEUED, int(below_k), int(limit))).fetchall()
            self.book.db.executemany("UPDATE rl_buffer SET state=? WHERE lease_id=?",
                                     [(TAKEN, lease_id) for lease_id, _ in rows])
        return [SubmittedGroup.unpack(body) for _, body in rows]

    def take_exhausted(self, max_attempts: int, limit: int) -> list[SubmittedGroup]:
        """Up to ``limit`` queued groups already handed out ``max_attempts`` times without a report: the trainer
        reports them untrained (unpaid, no sanction) instead of trying them again."""
        with self.book.transaction():
            rows = self.book.db.execute("SELECT lease_id, body FROM rl_buffer WHERE state=? AND attempts>=? LIMIT ?",
                                        (QUEUED, int(max_attempts), int(limit))).fetchall()
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
        """Queued groups per env (trainer telemetry)."""
        with self.book.lock:
            return dict(self.book.db.execute("SELECT environment, COUNT(*) FROM rl_buffer WHERE state=? "
                                             "GROUP BY environment", (QUEUED,)).fetchall())


__all__ = ["QUEUED", "TAKEN", "SubmittedGroup", "TrainingBuffer"]
