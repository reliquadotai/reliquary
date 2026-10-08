"""The run-wide observation log: keyed by (order, env/dataset, prompt), never reset on adoption.

The checkpoint, window and time are attributes. The log enforces no validity rule:
it only answers "was this prompt ever scanned in this run" for exploration pay.
Methods write without committing; the caller wraps them in ``with db:``.

What "scanned" means (table ``run_scans``, one row per scanned (env, prompt)):

* a prompt with a COUNTING training-lane observation is scanned, for the rest of the run;
* a prompt whose first scan is held by an exploration observation is scanned for as long as that
  observation may still be paid; when it ends unpaid for any reason its scan is released, and the
  prompt then stays scanned if a counting training observation of it exists (the scan is re-seated
  on the earliest one), else it is free again. ``release_first_scan`` is the ONE place that decides.
* A PROVEN training observation counts whatever its pay (its rewards are public), including a
  group left out of the batch. Only the training observations of an ABORTED window stop counting
  (the window trained nothing) and give their prompts back: ``set_window_aborted``.

Exploration observations go through ``reliquary.services.exploration.record_exploration``, which
records and reserves the pay in one transaction; ``record`` alone is for training observations.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass

from reliquary.constants import M_ROLLOUTS
from reliquary.protocol.release_contract import canonical_json_bytes, canonical_sha256
from reliquary.services.scoring import classify_signal

LANES = frozenset({"training", "exploration"})
_MUTABLE = ("status", "proof", "ts", "reason")  # not evidence: a retry may differ on them
_REASON = re.compile(r"[a-z][a-z0-9_]{0,39}")
STATUS_PROVEN_UNPAID = "proven_unpaid"  # settle status of a training group proven but not paid


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
    # None, or {"pool_sha256": ..., "seeds": [...]}: the chosen seed indices, ordered like rewards_bps.
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


_POOL_SHA256 = re.compile(r"[0-9a-f]{64}")


def _public_candidate(candidate: dict | None, rollouts: int) -> dict | None:
    """Only the two typed, non-identifying pool references may be published.

    ``seeds`` are the pool seed indices the miner chose, in the order of the per-rollout
    rewards (``seeds[i]`` drew the rollout graded ``rewards_bps[i]``), so miners learn which
    seed gave which reward. A list that cannot pair with the rewards is refused, not published.
    """
    if candidate is None:
        return None
    if not isinstance(candidate, dict):
        raise ValueError("candidate must be the pool reference of the group")
    out: dict = {}
    pool = candidate.get("pool_sha256")
    if not isinstance(pool, str) or not _POOL_SHA256.fullmatch(pool):
        raise ValueError("candidate pool_sha256 must be 64 lowercase hex characters")
    out["pool_sha256"] = pool
    seeds = candidate.get("seeds")
    if type(seeds) not in (list, tuple) or len(seeds) != rollouts:
        raise ValueError("candidate seeds must name one pool seed per rollout")
    previous = -1
    for seed in seeds:
        if type(seed) is not int or not previous < seed < 2 * M_ROLLOUTS:
            raise ValueError("candidate seeds must be distinct increasing indices of the 2 x M pool")
        previous = seed
    out["seeds"] = list(seeds)
    return out


class RunObservationLog:
    def __init__(self, db: sqlite3.Connection, *, order_sha256: str, sigma_min_bps: int):
        if db.in_transaction:  # executescript/commit below would commit the caller's open transaction
            raise ValueError("RunObservationLog must be constructed outside a transaction")
        self.db, self.order, self.sigma_min_bps = db, order_sha256, sigma_min_bps
        db.executescript("""
            CREATE TABLE IF NOT EXISTS run_meta(key TEXT PRIMARY KEY, value BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS run_observations(
                seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL, order_id TEXT NOT NULL,
                environment TEXT NOT NULL, prompt_idx INTEGER NOT NULL, window INTEGER NOT NULL,
                lane TEXT NOT NULL, category TEXT NOT NULL, hotkey TEXT NOT NULL, token_count INTEGER NOT NULL,
                first_scan INTEGER NOT NULL, public TEXT NOT NULL, untrained INTEGER NOT NULL DEFAULT 0);
            CREATE INDEX IF NOT EXISTS run_observations_window ON run_observations(order_id, window);
            CREATE INDEX IF NOT EXISTS run_observations_prompt
                ON run_observations(order_id, environment, prompt_idx);
            CREATE TABLE IF NOT EXISTS run_scans(
                order_id TEXT NOT NULL, environment TEXT NOT NULL, prompt_idx INTEGER NOT NULL,
                first_id TEXT NOT NULL, category TEXT NOT NULL,
                PRIMARY KEY(order_id, environment, prompt_idx));
            CREATE TABLE IF NOT EXISTS run_events(
                seq INTEGER PRIMARY KEY AUTOINCREMENT, order_id TEXT NOT NULL, observation_id TEXT NOT NULL,
                payload TEXT NOT NULL);
        """)
        if "untrained" not in {row[1] for row in db.execute("PRAGMA table_info(run_observations)")}:
            db.execute("ALTER TABLE run_observations ADD COLUMN untrained INTEGER NOT NULL DEFAULT 0")
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

    def record(self, obs: Observation, *, status: str, proof: str, reason: str | None = None) -> RecordResult:
        """Record one observation and publish it. ``reason`` (a short lowercase identifier, e.g. why
        an exploration observation is unpaid) is published as its own field when given."""
        if reason is not None and (not isinstance(reason, str) or _REASON.fullmatch(reason) is None):
            raise ValueError("observation reason must be a short lowercase identifier")
        category = self._classify(obs)
        identity = observation_id(self.order, obs, self._salt)
        # Explicit, typed fields only: nothing a caller puts in ``candidate`` can add a key.
        public = {"type": "observation", "id": identity, "env": str(obs.environment),
                  "dataset": str(obs.dataset_id), "prompt_idx": int(obs.prompt_idx),
                  "checkpoint_n": int(obs.checkpoint_n), "checkpoint": str(obs.checkpoint_revision),
                  "window": int(obs.window), "ts": float(obs.observed_at),
                  "rewards_bps": [int(v) for v in obs.rewards_bps], "verdict": category,
                  "candidate": _public_candidate(obs.candidate, len(obs.rewards_bps)), "lane": obs.lane,
                  "status": str(status), "proof": str(proof)}
        if reason is not None:
            public["reason"] = reason
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
            if {k: v for k, v in stored.items() if k not in _MUTABLE} != \
                    {k: v for k, v in public.items() if k not in _MUTABLE}:
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

    def _seat(self, environment: str, prompt_idx: int) -> None:
        """If the prompt has no scan, seat it on its earliest counting training observation (if any)."""
        row = self.db.execute(
            "SELECT id, category FROM run_observations WHERE order_id=? AND environment=? AND prompt_idx=? "
            "AND lane='training' AND untrained=0 ORDER BY seq LIMIT 1",
            (self.order, environment, prompt_idx)).fetchone()
        if row is not None and self.db.execute(
                "INSERT OR IGNORE INTO run_scans VALUES(?,?,?,?,?)",
                (self.order, environment, prompt_idx, row[0], row[1])).rowcount == 1:
            self.db.execute("UPDATE run_observations SET first_scan=1 WHERE id=?", (row[0],))

    def release_first_scan(self, observation_id: str) -> bool:
        """THE release of a first scan: every path that leaves an observation unpaid ends here.

        The observation stops holding the scan of its (env, prompt). The prompt is then scanned iff
        a counting training-lane observation of it exists in the run (any window, any hotkey,
        recorded before or after): the scan is re-seated on the earliest one. Otherwise the prompt
        is free again and a later exploration observation can be paid for it.

        Which training observations count (module docstring): every proven one, paid or not,
        except those of a window the runtime declared aborted (``set_window_aborted``): an aborted
        window trained nothing, so its training observations never keep a prompt. A counting
        training observation is never released (returns False, nothing changes).

        Returns True when ``observation_id`` held the scan and no longer does. Idempotent.
        """
        row = self.db.execute("SELECT environment, prompt_idx, lane, untrained FROM run_observations "
                              "WHERE id=? AND order_id=?", (observation_id, self.order)).fetchone()
        if row is None or (row[2] == "training" and not row[3]):
            return False
        released = self.db.execute("DELETE FROM run_scans WHERE order_id=? AND first_id=?",
                                   (self.order, observation_id)).rowcount == 1
        if released:
            self.db.execute("UPDATE run_observations SET first_scan=0 WHERE id=?", (observation_id,))
        self._seat(row[0], row[1])
        return released

    def trained_prompts(self, window: int, environment: str) -> set[int]:
        """Prompts of ``environment`` with a counting training-lane observation recorded in ``window``."""
        return {r for r, in self.db.execute(
            "SELECT DISTINCT prompt_idx FROM run_observations WHERE order_id=? AND window=? AND environment=? "
            "AND lane='training' AND untrained=0", (self.order, window, environment))}

    def claim_first_scan(self, observation_id: str) -> bool:
        """Give the first scan of its (env, prompt) to an already recorded observation that does not
        hold it (a retry of a group refused for lack of room). False if the prompt is scanned."""
        row = self.db.execute("SELECT environment, prompt_idx, category, first_scan FROM run_observations "
                              "WHERE id=? AND order_id=?", (observation_id, self.order)).fetchone()
        if row is None:
            raise ValueError("unknown observation")
        if row[3]:
            return False
        if self.db.execute("INSERT OR IGNORE INTO run_scans VALUES(?,?,?,?,?)",
                           (self.order, row[0], row[1], observation_id, row[2])).rowcount != 1:
            return False
        self.db.execute("UPDATE run_observations SET first_scan=1 WHERE id=?", (observation_id,))
        return True

    def refusal_reason(self, observation_id: str) -> str | None:
        """The ``reason`` the observation was RECORDED with (None when it had none)."""
        row = self.db.execute("SELECT public FROM run_observations WHERE id=? AND order_id=?",
                              (observation_id, self.order)).fetchone()
        return None if row is None else json.loads(row[0]).get("reason")

    def set_window_aborted(self, window: int, aborted: bool) -> None:
        """Tell the log whether ``window`` aborted (the runtime knows; the log cannot).

        An aborted window trained nothing: its training observations stop counting as scans and
        give their prompts back (``release_first_scan``: next counting training observation, or
        free). ``aborted=False`` makes them count again and re-seats the prompts that have no scan.
        Idempotent; only rows whose state changes are touched.
        """
        rows = self.db.execute(
            "SELECT id, environment, prompt_idx FROM run_observations WHERE order_id=? AND window=? "
            "AND lane='training' AND untrained<>? ORDER BY seq", (self.order, window, int(aborted))).fetchall()
        for identity, environment, prompt_idx in rows:
            self.db.execute("UPDATE run_observations SET untrained=? WHERE id=?", (int(aborted), identity))
            if aborted:
                self.release_first_scan(identity)
            else:
                self._seat(environment, prompt_idx)

    def settle(self, observation_id: str, *, status: str, proof: str, at: float,
               reason: str | None = None) -> None:
        """Publish a settle event (idempotent on the latest status, proof and reason). The status
        never changes what counts as a scan (``set_window_aborted`` does)."""
        if reason is not None and (not isinstance(reason, str) or _REASON.fullmatch(reason) is None):
            raise ValueError("settle reason must be a short lowercase identifier")
        row = self.db.execute("SELECT window, order_id FROM run_observations "
                              "WHERE id=?", (observation_id,)).fetchone()
        if row is None:
            raise ValueError("unknown observation")
        if row[1] != self.order:
            raise ValueError("observation belongs to another order")
        for (payload,) in self.db.execute(
                "SELECT payload FROM run_events WHERE order_id=? AND observation_id=? ORDER BY seq DESC",
                (self.order, observation_id)):
            event = json.loads(payload)
            if event.get("type") != "settle":
                continue
            if event.get("status") == status and event.get("proof") == proof and event.get("reason") == reason:
                return  # same as the LATEST settlement: nothing new to publish
            break  # an older identical settlement does not hide a change back (A, B, A)
        public = {"type": "settle", "id": observation_id, "window": int(row[0]),
                  "status": str(status), "proof": str(proof), "ts": float(at)}
        if reason is not None:
            public["reason"] = reason
        self.db.execute("INSERT INTO run_events(order_id,observation_id,payload) VALUES(?,?,?)",
                        (self.order, observation_id, canonical_json_bytes(public).decode()))

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
        """``(scanned prompts, of which in-zone)`` of an env. A trained prompt is always counted: it
        holds a scan row (its own, or the exploration observation's that saw the prompt first)."""
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
