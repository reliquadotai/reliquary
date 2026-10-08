"""The run-wide observation log: keyed by (order, env/dataset, prompt), never reset on adoption.

The checkpoint, window and time are attributes. The log enforces no validity rule:
it only answers "was this prompt ever scanned in this run" for exploration pay.
Methods write without committing; the caller wraps them in ``with db:``.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass

from reliquary.constants import M_ROLLOUTS
from reliquary.protocol.release_contract import canonical_json_bytes, canonical_sha256
from reliquary.services.scoring import classify_signal

LANES = frozenset({"training", "exploration"})


class NotAnObservation(ValueError):
    """Grading errors and incomplete groups are never observations."""


@dataclass(frozen=True, slots=True)
class Observation:
    environment: str
    dataset_id: str
    prompt_idx: int
    group_id: str
    window: int
    checkpoint_n: int
    checkpoint_revision: str
    observed_at: float
    rewards_bps: tuple[int, ...]
    lane: str
    candidate: dict | None
    hotkey: str
    token_count: int


@dataclass(frozen=True, slots=True)
class RecordResult:
    observation_id: str
    inserted: bool
    first_scan: bool
    category: str


def observation_id(order_sha256: str, obs: Observation, run_salt: bytes) -> str:
    """Per-submission identity: the hotkey and the secret run salt keep two miners on the same
    pool selection distinct and keep a public id from being linked to a hotkey by enumeration."""
    return canonical_sha256({"order": order_sha256, "environment": obs.environment,
                             "prompt_idx": obs.prompt_idx, "group_id": obs.group_id, "window": obs.window,
                             "hotkey": obs.hotkey, "run_salt": run_salt.hex()})


def _public_candidate(candidate: dict | None) -> dict | None:
    """Only the two typed, non-identifying pool references may be published."""
    if candidate is None:
        return None
    out: dict = {}
    pool = candidate.get("pool_sha256")
    if isinstance(pool, str):
        out["pool_sha256"] = pool
    cid = candidate.get("candidate_id")
    if type(cid) is int:
        out["candidate_id"] = cid
    return out


class RunObservationLog:
    def __init__(self, db: sqlite3.Connection, *, order_sha256: str, sigma_min_bps: int):
        self.db, self.order, self.sigma_min_bps = db, order_sha256, sigma_min_bps
        db.executescript("""
            CREATE TABLE IF NOT EXISTS run_meta(key TEXT PRIMARY KEY, value BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS run_observations(
                seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL, order_id TEXT NOT NULL,
                environment TEXT NOT NULL, prompt_idx INTEGER NOT NULL, window INTEGER NOT NULL,
                lane TEXT NOT NULL, category TEXT NOT NULL, hotkey TEXT NOT NULL, token_count INTEGER NOT NULL,
                first_scan INTEGER NOT NULL, public TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS run_observations_window ON run_observations(order_id, window);
            CREATE TABLE IF NOT EXISTS run_scans(
                order_id TEXT NOT NULL, environment TEXT NOT NULL, prompt_idx INTEGER NOT NULL,
                first_id TEXT NOT NULL, category TEXT NOT NULL,
                PRIMARY KEY(order_id, environment, prompt_idx));
            CREATE TABLE IF NOT EXISTS run_events(
                seq INTEGER PRIMARY KEY AUTOINCREMENT, order_id TEXT NOT NULL, observation_id TEXT NOT NULL,
                payload TEXT NOT NULL);
        """)
        db.execute("INSERT OR IGNORE INTO run_meta(key, value) VALUES('run_salt', ?)", (os.urandom(32),))
        db.commit()
        self._salt = bytes(db.execute("SELECT value FROM run_meta WHERE key='run_salt'").fetchone()[0])

    def _classify(self, obs: Observation) -> str:
        if obs.lane not in LANES:
            raise ValueError("unknown observation lane")
        if len(obs.rewards_bps) > M_ROLLOUTS:
            raise NotAnObservation("too many rewards for one group")
        rewards = []
        for value in obs.rewards_bps:
            if type(value) is not int or not 0 <= value <= 10000:
                raise NotAnObservation("rewards must be graded basis points")
            rewards.append(value / 10000)
        signal = classify_signal(rewards, expected=M_ROLLOUTS, sigma_min_bps=self.sigma_min_bps)
        if signal.category == "unknown":
            raise NotAnObservation("incomplete group")
        return signal.category

    def record(self, obs: Observation, *, status: str, proof: str) -> RecordResult:
        category = self._classify(obs)
        identity = observation_id(self.order, obs, self._salt)
        # Explicit, typed fields only: nothing a caller puts in ``candidate`` can add a key.
        public = {"type": "observation", "id": identity, "env": str(obs.environment),
                  "dataset": str(obs.dataset_id), "prompt_idx": int(obs.prompt_idx),
                  "checkpoint_n": int(obs.checkpoint_n), "checkpoint": str(obs.checkpoint_revision),
                  "window": int(obs.window), "ts": float(obs.observed_at),
                  "rewards_bps": [int(v) for v in obs.rewards_bps], "verdict": category,
                  "candidate": _public_candidate(obs.candidate), "lane": obs.lane,
                  "status": str(status), "proof": str(proof)}
        payload = canonical_json_bytes(public).decode()
        inserted = self.db.execute(
            "INSERT OR IGNORE INTO run_observations(id,order_id,environment,prompt_idx,window,lane,category,"
            "hotkey,token_count,first_scan,public) VALUES(?,?,?,?,?,?,?,?,?,0,?)",
            (identity, self.order, obs.environment, obs.prompt_idx, obs.window, obs.lane, category,
             obs.hotkey, obs.token_count, payload)).rowcount == 1
        if not inserted:  # same submission again: idempotent retry, or the same id with other evidence
            existing = self.db.execute("SELECT public, first_scan FROM run_observations WHERE id=?",
                                       (identity,)).fetchone()
            stored = json.loads(existing[0])
            if {k: v for k, v in stored.items() if k not in ("status", "proof", "ts")} != \
                    {k: v for k, v in public.items() if k not in ("status", "proof", "ts")}:
                raise ValueError("observation identity already exists with different evidence")
            return RecordResult(identity, False, bool(existing[1]), category)
        first = self.db.execute(
            "INSERT OR IGNORE INTO run_scans VALUES(?,?,?,?,?)",
            (self.order, obs.environment, obs.prompt_idx, identity, category)).rowcount == 1
        if first:
            self.db.execute("UPDATE run_observations SET first_scan=1 WHERE id=?", (identity,))
        self.db.execute("INSERT INTO run_events(order_id,observation_id,payload) VALUES(?,?,?)",
                        (self.order, identity, payload))
        return RecordResult(identity, True, first, category)

    def is_scanned(self, environment: str, prompt_idx: int) -> bool:
        return self.db.execute("SELECT 1 FROM run_scans WHERE order_id=? AND environment=? AND prompt_idx=?",
                               (self.order, environment, prompt_idx)).fetchone() is not None

    def release_first_scan(self, observation_id: str) -> bool:
        """A forfeited (failed-audit) first scan does not count as a scan."""
        released = self.db.execute("DELETE FROM run_scans WHERE order_id=? AND first_id=?",
                                   (self.order, observation_id)).rowcount == 1
        if released:
            self.db.execute("UPDATE run_observations SET first_scan=0 WHERE id=?", (observation_id,))
        return released

    def settle(self, observation_id: str, *, status: str, proof: str, at: float) -> None:
        row = self.db.execute("SELECT window, order_id FROM run_observations WHERE id=?",
                              (observation_id,)).fetchone()
        if row is None:
            raise ValueError("unknown observation")
        if row[1] != self.order:
            raise ValueError("observation belongs to another order")
        for (payload,) in self.db.execute(
                "SELECT payload FROM run_events WHERE order_id=? AND observation_id=?",
                (self.order, observation_id)):
            event = json.loads(payload)
            if event.get("type") == "settle" and event.get("status") == status and event.get("proof") == proof:
                return  # same settlement again: nothing new to publish
        payload = canonical_json_bytes({"type": "settle", "id": observation_id, "window": int(row[0]),
                                        "status": str(status), "proof": str(proof), "ts": float(at)}).decode()
        self.db.execute("INSERT INTO run_events(order_id,observation_id,payload) VALUES(?,?,?)",
                        (self.order, observation_id, payload))

    def events(self, *, after: int = 0, limit: int = 1000) -> list[tuple[int, dict]]:
        rows = self.db.execute("SELECT seq,payload FROM run_events WHERE order_id=? AND seq>? ORDER BY seq LIMIT ?",
                               (self.order, after, limit)).fetchall()
        return [(seq, json.loads(payload)) for seq, payload in rows]

    def admin_events(self, *, after: int = 0, limit: int = 1000) -> list[tuple[int, dict]]:
        result = []
        for seq, event in self.events(after=after, limit=limit):
            row = self.db.execute("SELECT hotkey, token_count FROM run_observations WHERE id=?", (event["id"],)).fetchone()
            result.append((seq, {**event, "hotkey": row[0], "token_count": row[1]}))
        return result

    def first_scan_stats(self, environment: str) -> tuple[int, int]:
        row = self.db.execute(
            "SELECT COUNT(*), SUM(CASE WHEN category='in-zone' THEN 1 ELSE 0 END) FROM run_scans "
            "WHERE order_id=? AND environment=?", (self.order, environment)).fetchone()
        return int(row[0]), int(row[1] or 0)

    def window_observations(self, window: int) -> list[dict]:
        rows = self.db.execute(
            "SELECT id, environment, prompt_idx, lane, category, hotkey, first_scan FROM run_observations "
            "WHERE order_id=? AND window=? ORDER BY seq", (self.order, window)).fetchall()
        return [{"id": r[0], "environment": r[1], "prompt_idx": r[2], "lane": r[3], "category": r[4],
                 "hotkey": r[5], "first_scan": bool(r[6])} for r in rows]
