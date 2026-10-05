"""Durable first-pass eligibility; epochs never alter model or optimizer state."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sqlite3
from typing import Iterable

from reliquary.protocol.release_contract import canonical_json_bytes, canonical_sha256
from reliquary.protocol.service_contract import ServiceContract, _identifier, _integer
from reliquary.services.observations import observation_id, observation_signal, validate_observation


def _time(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("now must be a finite nonnegative timestamp")
    return float(value)


class EligibilityStore:
    """One immutable source population per context, with atomic epoch rollover.

    A qualified RL-compatible feed is required for each adopted context. The
    caller owns checkpoint adoption and admission; this store has no payment,
    trainer reset, or checkpoint publication side effects.
    """

    def __init__(self, path: str | Path, contract: ServiceContract, row_ids: Iterable[str]):
        value = contract.to_dict()
        policy = value["policies"]["eligibility"]
        if value["service_kind"] != "adaptive_training" or policy["kind"] != "dataset-epoch/v1":
            raise ValueError("eligibility epochs require the ordered adaptive-training policy")
        rows = sorted(row_ids)
        for row in rows:
            _identifier(row, "source row_id")
        if not rows or len(set(rows)) != len(rows):
            raise ValueError("source rows must be nonempty and unique")
        self.contract, self.context, self.policy = contract, contract.context_sha256, policy
        self.limits = value["limits"]
        # Runtime views serialize this connection with their own lock; proof
        # completion callbacks may arrive on a different worker thread.
        self.db = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS eligibility_contexts (
                context TEXT PRIMARY KEY, contract TEXT NOT NULL, population TEXT NOT NULL,
                qualified TEXT, started REAL, groups_used INTEGER NOT NULL DEFAULT 0,
                tokens_used INTEGER NOT NULL DEFAULT 0, last_window INTEGER, last_now REAL);
            CREATE TABLE IF NOT EXISTS eligibility_source (
                context TEXT NOT NULL, row_id TEXT NOT NULL, category TEXT,
                PRIMARY KEY(context,row_id));
            CREATE TABLE IF NOT EXISTS eligibility_epochs (
                context TEXT NOT NULL, epoch INTEGER NOT NULL, opened_window INTEGER NOT NULL,
                final_snapshot TEXT, PRIMARY KEY(context,epoch));
            CREATE TABLE IF NOT EXISTS eligibility_rows (
                context TEXT NOT NULL, epoch INTEGER NOT NULL, row_id TEXT NOT NULL,
                excluded INTEGER NOT NULL, seen INTEGER NOT NULL DEFAULT 0, category TEXT,
                PRIMARY KEY(context,epoch,row_id));
            CREATE TABLE IF NOT EXISTS eligibility_observations (
                context TEXT NOT NULL, epoch INTEGER NOT NULL, id TEXT NOT NULL,
                payload TEXT NOT NULL, category TEXT NOT NULL,
                PRIMARY KEY(context,epoch,id));
            CREATE UNIQUE INDEX IF NOT EXISTS eligibility_observation_identity
                ON eligibility_observations(context,id);
        """)
        population = canonical_sha256(rows)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO eligibility_contexts(context,contract,population) VALUES(?,?,?)",
                            (self.context, contract.sha256, population))
            existing = self.db.execute("SELECT contract,population FROM eligibility_contexts WHERE context=?",
                                       (self.context,)).fetchone()
            if existing != (contract.sha256, population):
                raise ValueError("eligibility context already binds another order or source population")
            self.db.executemany("INSERT OR IGNORE INTO eligibility_source(context,row_id) VALUES(?,?)",
                                ((self.context, row) for row in rows))

    def close(self) -> None:
        self.db.close()

    def source_rows(self) -> list[str]:
        return [row for row, in self.db.execute("SELECT row_id FROM eligibility_source WHERE context=? ORDER BY row_id", (self.context,))]

    def active_rows(self) -> list[str]:
        """The epoch population includes seen rows, and excludes carried highs."""
        epoch = self._current()
        if epoch is None:
            raise ValueError("eligibility epoch has not started")
        return [row for row, in self.db.execute(
            "SELECT row_id FROM eligibility_rows WHERE context=? AND epoch=? AND excluded=0 ORDER BY row_id",
            (self.context, epoch))]

    def qualify_feed(self, manifest: dict) -> None:
        """Persist runtime-verified provenance, never infer RL parity from eval."""
        if (manifest.get("context_sha256") != self.context
                or manifest.get("generation_verified") is not True
                or manifest.get("sampling_verified") is not True
                or manifest.get("rl_group_comparable") is not True):
            raise ValueError("new eligibility context requires its qualified RL-compatible feed")
        group_size = _integer(manifest.get("group_size"), "qualified group_size", 2, 65536)
        sampling = self.contract.to_dict()["policies"]["sampling"]
        if sampling.get("group_size", group_size) != group_size:
            raise ValueError("qualified group size differs from the ordered sampling policy")
        payload = canonical_json_bytes(manifest).decode()
        with self.db:
            self.db.execute("UPDATE eligibility_contexts SET qualified=qualified WHERE context=?", (self.context,))
            existing = self.db.execute("SELECT qualified FROM eligibility_contexts WHERE context=?",
                                       (self.context,)).fetchone()[0]
            if existing is not None and existing != payload:
                raise ValueError("eligibility feed qualification is already frozen")
            self.db.execute("UPDATE eligibility_contexts SET qualified=? WHERE context=?", (payload, self.context))

    def _current(self) -> int | None:
        return self.db.execute("SELECT MAX(epoch) FROM eligibility_epochs WHERE context=?", (self.context,)).fetchone()[0]

    def _insert_epoch(self, epoch: int, window: int) -> None:
        self.db.execute("INSERT INTO eligibility_epochs(context,epoch,opened_window) VALUES(?,?,?)",
                        (self.context, epoch, window))
        self.db.execute("""INSERT INTO eligibility_rows(context,epoch,row_id,excluded,category)
            SELECT context,?,row_id,CASE WHEN category='uniform-high' THEN 1 ELSE 0 END,category
            FROM eligibility_source WHERE context=?""", (epoch, self.context))

    def begin(self, *, window: int, now: float) -> dict:
        _integer(window, "window", 0)
        now = _time(now)
        with self.db:
            self.db.execute("UPDATE eligibility_contexts SET started=COALESCE(started,?) WHERE context=?", (now, self.context))
            qualified = self.db.execute("SELECT qualified FROM eligibility_contexts WHERE context=?", (self.context,)).fetchone()[0]
            if qualified is None:
                raise ValueError("eligibility context has no qualified feed")
            if self._current() is None:
                self._insert_epoch(0, window)
        return self.snapshot(window=window, now=now)

    def snapshot(self, *, window: int, now: float, epoch: int | None = None) -> dict:
        # A published view advances the durable clock too: observing a deadline
        # cannot reopen the order later under a backwards clock.
        own_transaction = not self.db.in_transaction
        if own_transaction:
            self.db.execute("BEGIN IMMEDIATE")
        try:
            result = self._snapshot(window=window, now=now, epoch=epoch)
            if own_transaction:
                self.db.commit()
            return result
        except BaseException:
            if own_transaction:
                self.db.rollback()
            raise

    def _snapshot(self, *, window: int, now: float, epoch: int | None = None) -> dict:
        _integer(window, "window", 0)
        now = _time(now)
        epoch = self._current() if epoch is None else _integer(epoch, "dataset_epoch", 0)
        if epoch is None:
            raise ValueError("eligibility epoch has not started")
        record = self.db.execute("SELECT opened_window,final_snapshot FROM eligibility_epochs WHERE context=? AND epoch=?",
                                 (self.context, epoch)).fetchone()
        if record is None:
            raise ValueError("unknown eligibility epoch")
        opened, final = record
        if final is not None:
            return json.loads(final)
        qualified, started, groups, tokens, last_window, last_now = self.db.execute(
            "SELECT qualified,started,groups_used,tokens_used,last_window,last_now FROM eligibility_contexts WHERE context=?", (self.context,)).fetchone()
        if (window < opened or now < started or (last_window is not None and window < last_window)
                or (last_now is not None and now < last_now)):
            raise ValueError("eligibility clock precedes its epoch/order")
        self.db.execute("UPDATE eligibility_contexts SET last_window=?,last_now=? WHERE context=?", (window, now, self.context))
        source, excluded, seen = self.db.execute("SELECT COUNT(*),COALESCE(SUM(excluded),0),COALESCE(SUM(seen),0) FROM eligibility_rows WHERE context=? AND epoch=?",
                                                (self.context, epoch)).fetchone()
        active = source - excluded
        coverage = seen * 10000 // active if active else 0
        reason, status = None, "collecting"
        if groups >= self.limits["max_groups"]:
            reason, status = "group-budget", "halted"
        elif tokens >= self.limits["max_tokens"]:
            reason, status = "token-budget", "halted"
        elif now - started >= self.limits["deadline_seconds"]:
            reason, status = "deadline", "halted"
        elif not active:
            reason, status = "no-eligible-rows", "halted"
        elif window - opened >= self.policy["max_epoch_windows"]:
            reason, status = "max-epoch-windows", "ready"
        elif coverage >= self.policy["coverage_bps"] and window - opened >= self.policy["refresh_windows"]:
            reason, status = "coverage", "ready"
        categories = dict(self.db.execute("SELECT category,COUNT(*) FROM eligibility_observations WHERE context=? AND epoch=? GROUP BY category",
                                          (self.context, epoch)))
        return {"schema": "eligibility-snapshot/v1", "context_sha256": self.context,
                "contract_sha256": self.contract.sha256, "dataset_epoch": epoch, "opened_window": opened,
                "window": window, "source_rows": source, "population_rows": active,
                "excluded_uniform_high": excluded, "seen_unique": seen,
                "eligible_rows": active - seen, "coverage_bps": coverage, "category_counts": categories,
                "groups_used": groups, "tokens_used": tokens, "status": status, "reason": reason,
                "feed_sha256": canonical_sha256(json.loads(qualified))}

    def eligible_rows(self, *, window: int, now: float) -> list[str]:
        epoch = self._current()
        if epoch is None:
            raise ValueError("eligibility epoch has not started")
        snapshot = self.snapshot(window=window, now=now)
        if snapshot["status"] == "halted" or snapshot["reason"] == "max-epoch-windows":
            return []
        return [row for row, in self.db.execute("SELECT row_id FROM eligibility_rows WHERE context=? AND epoch=? AND excluded=0 AND seen=0 ORDER BY row_id",
                                               (self.context, epoch))]

    def record(self, observation: dict, *, now: float) -> bool:
        row = validate_observation(observation, self.contract)
        if row["verification"]["generation"] != "verified" or row["verification"]["sampling"] != "verified":
            raise ValueError("first-pass observation needs generation and sampling verification")
        identifier, payload = observation_id(row), canonical_json_bytes(row).decode()
        # Acquire the write lock before inspecting the budget, so two writers
        # cannot both consume its final slot or race an epoch transition.
        self.db.execute("BEGIN IMMEDIATE")
        checked_snapshot = None
        try:
            epoch = self._current()
            if epoch is None:
                raise ValueError("eligibility epoch has not started")
            existing = self.db.execute("SELECT payload FROM eligibility_observations WHERE context=? AND id=?",
                                       (self.context, identifier)).fetchone()
            if existing is not None:
                if existing[0] != payload:
                    raise ValueError("eligibility observation was rebound to different evidence")
                self.db.commit()
                return False
            state = self.db.execute("SELECT excluded FROM eligibility_rows WHERE context=? AND epoch=? AND row_id=?",
                                    (self.context, epoch, row["row_id"])).fetchone()
            if state is None:
                raise ValueError("observation row is absent from the immutable source population")
            if state[0]:
                raise ValueError("uniform-high row is excluded in this context")
            qualification = json.loads(self.db.execute("SELECT qualified FROM eligibility_contexts WHERE context=?", (self.context,)).fetchone()[0])
            if row["expected_samples"] != qualification["group_size"]:
                raise ValueError("observation group differs from the qualified feed")
            snapshot = self.snapshot(window=row["window"], now=now)
            checked_snapshot = snapshot
            token_count = sum(row["tokens"])
            if snapshot["status"] == "halted" or snapshot["reason"] == "max-epoch-windows" or snapshot["tokens_used"] + token_count > self.limits["max_tokens"]:
                raise ValueError("eligibility observation exceeds its ordered budget/deadline or epoch")
            signal = observation_signal(row, self.contract)
            self.db.execute("INSERT INTO eligibility_observations(context,epoch,id,payload,category) VALUES(?,?,?,?,?)",
                            (self.context, epoch, identifier, payload, signal.category))
            self.db.execute("UPDATE eligibility_contexts SET groups_used=groups_used+1,tokens_used=tokens_used+? WHERE context=?", (token_count, self.context))
            if signal.category != "unknown":
                self.db.execute("UPDATE eligibility_rows SET seen=1,category=? WHERE context=? AND epoch=? AND row_id=?",
                                (signal.category, self.context, epoch, row["row_id"]))
                self.db.execute("UPDATE eligibility_source SET category=? WHERE context=? AND row_id=?",
                                (signal.category, self.context, row["row_id"]))
            self.db.commit()
            return True
        except BaseException:
            self.db.rollback()
            if checked_snapshot is not None and (checked_snapshot["status"] == "halted"
                                                  or checked_snapshot["reason"] == "max-epoch-windows"):
                # Reject the observation without undoing an already elapsed
                # deadline/epoch clock. No group or token budget is consumed.
                try:
                    self.snapshot(window=row["window"], now=now)
                except Exception:
                    pass
            raise

    def row_state(self, row_id: str) -> dict:
        _identifier(row_id, "row_id")
        epoch = self._current()
        record = self.db.execute("SELECT excluded,seen,category FROM eligibility_rows WHERE context=? AND epoch=? AND row_id=?",
                                 (self.context, epoch, row_id)).fetchone()
        if record is None:
            raise ValueError("unknown eligibility row or epoch")
        excluded, seen, category = record
        return {"row_id": row_id, "context_sha256": self.context, "dataset_epoch": epoch,
                "seen_in_epoch": bool(seen), "category": category,
                "reason": "uniform-high" if excluded else "seen-in-epoch" if seen else None}

    def advance(self, *, expected_epoch: int, window: int, now: float) -> dict:
        _integer(expected_epoch, "expected_epoch", 0)
        _integer(window, "window", 0)
        now = _time(now)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            current = self._current()
            if current is None or expected_epoch > current:
                raise ValueError("epoch transition does not match current eligibility state")
            if current == expected_epoch:
                final = self.snapshot(window=window, now=now)
                if final["status"] != "ready":
                    raise ValueError("eligibility epoch is not ready to advance")
                self.db.execute("UPDATE eligibility_epochs SET final_snapshot=? WHERE context=? AND epoch=?",
                                (canonical_json_bytes(final).decode(), self.context, current))
                self._insert_epoch(current + 1, window)
            result = self.snapshot(window=window, now=now, epoch=expected_epoch + 1)
            self.db.commit()
            return result
        except BaseException:
            self.db.rollback()
            raise
