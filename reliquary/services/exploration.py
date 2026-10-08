"""Exploration pay (first scan only, 15 % of a training group, per-env 10 % cap) and its sampled audit.

Methods write without committing; the caller wraps them in ``with db:``.
"""
from __future__ import annotations

import hashlib
import sqlite3
from typing import Callable

AUDIT_DRAW_ROUND_OFFSET = 2
_AUDIT_DOMAIN = b"reliquary-exploration-audit/v1"
_EPS = 1e-12


def training_group_price(pool: float, *, picks_target: int, batch_slots: int) -> float:
    if picks_target < 1 or batch_slots < 1:
        raise ValueError("training slots must be positive")
    return float(pool) / (picks_target * batch_slots)


def exploration_price(pool: float, *, picks_target: int, batch_slots: int, price_bps: int) -> float:
    return training_group_price(pool, picks_target=picks_target, batch_slots=batch_slots) * price_bps / 10000


def exploration_cap(pool: float, *, cap_bps: int) -> float:
    return float(pool) * cap_bps / 10000


def audit_selected(*, beacon_randomness: str, observation_id: str, audit_bps: int, forced: bool) -> bool:
    if forced:
        return True
    if audit_bps <= 0:
        return False
    digest = hashlib.sha256(_AUDIT_DOMAIN + bytes.fromhex(beacon_randomness) + bytes.fromhex(observation_id)).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < audit_bps / 10000


class ExplorationLedger:
    def __init__(self, db: sqlite3.Connection, *, order_sha256: str):
        self.db, self.order = db, order_sha256
        db.executescript("""
            CREATE TABLE IF NOT EXISTS exploration_entitlements(
                observation_id TEXT PRIMARY KEY, order_id TEXT NOT NULL, window INTEGER NOT NULL,
                environment TEXT NOT NULL, hotkey TEXT NOT NULL, prompt_idx INTEGER NOT NULL,
                amount REAL NOT NULL, draw_round INTEGER NOT NULL, forced INTEGER NOT NULL,
                audit TEXT NOT NULL, status TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS exploration_window ON exploration_entitlements(order_id, window);
            CREATE TABLE IF NOT EXISTS exploration_hotkeys(
                order_id TEXT NOT NULL, hotkey TEXT NOT NULL, groups INTEGER NOT NULL, PRIMARY KEY(order_id, hotkey));
            CREATE TABLE IF NOT EXISTS exploration_finalized(
                order_id TEXT NOT NULL, window INTEGER NOT NULL, environment TEXT NOT NULL,
                PRIMARY KEY(order_id, window, environment));
            CREATE TABLE IF NOT EXISTS exploration_bans(
                order_id TEXT NOT NULL, hotkey TEXT NOT NULL, until REAL NOT NULL, PRIMARY KEY(order_id, hotkey));
        """)

    def banned(self, hotkey: str, now: float) -> bool:
        row = self.db.execute("SELECT until FROM exploration_bans WHERE order_id=? AND hotkey=?",
                              (self.order, hotkey)).fetchone()
        return row is not None and now < row[0]

    def reserve(self, *, window: int, environment: str, observation_id: str, hotkey: str, prompt_idx: int,
                amount: float, cap: float, draw_round: int, new_hotkey_audit_groups: int) -> dict | None:
        known = self.db.execute("SELECT amount, forced, draw_round FROM exploration_entitlements "
                                "WHERE observation_id=? AND order_id=?", (observation_id, self.order)).fetchone()
        if known is not None:  # replay of the same observation: idempotent, nothing counted twice
            return {"amount": known[0], "forced": bool(known[1]), "draw_round": known[2]}
        used = self.db.execute(
            "SELECT COALESCE(SUM(amount),0) FROM exploration_entitlements WHERE order_id=? AND window=? "
            "AND environment=? AND status='reserved' AND audit<>'unaudited'",
            (self.order, window, environment)).fetchone()[0]
        if used + amount > cap + _EPS:
            return None
        row = self.db.execute("SELECT groups FROM exploration_hotkeys WHERE order_id=? AND hotkey=?",
                              (self.order, hotkey)).fetchone()
        seen = row[0] if row else 0
        forced = seen < new_hotkey_audit_groups
        self.db.execute("INSERT INTO exploration_hotkeys VALUES(?,?,1) ON CONFLICT(order_id,hotkey) "
                        "DO UPDATE SET groups=groups+1", (self.order, hotkey))
        self.db.execute("INSERT INTO exploration_entitlements VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (observation_id, self.order, window, environment, hotkey, prompt_idx, float(amount),
                         int(draw_round), int(forced), "pending_draw", "reserved"))
        return {"amount": float(amount), "forced": forced, "draw_round": int(draw_round)}

    def is_finalized(self, window: int, *, environment: str) -> bool:
        return self.db.execute("SELECT 1 FROM exploration_finalized WHERE order_id=? AND window=? AND environment=?",
                               (self.order, window, environment)).fetchone() is not None

    def pending_draw_rounds(self, window: int, *, environment: str) -> list[int]:
        return [r for r, in self.db.execute(
            "SELECT DISTINCT draw_round FROM exploration_entitlements WHERE order_id=? AND window=? "
            "AND environment=? AND audit='pending_draw' ORDER BY draw_round", (self.order, window, environment))]

    def resolve_draws(self, window: int, *, environment: str, beacon_for_round: Callable[[int], str | None],
                      audit_bps: int) -> list[str]:
        selected = []
        for identity, round_id, forced in self.db.execute(
                "SELECT observation_id, draw_round, forced FROM exploration_entitlements WHERE order_id=? "
                "AND window=? AND environment=? AND audit='pending_draw' ORDER BY rowid",
                (self.order, window, environment)).fetchall():
            beacon = beacon_for_round(round_id)
            if beacon is None:
                continue
            chosen = audit_selected(beacon_randomness=beacon, observation_id=identity,
                                    audit_bps=audit_bps, forced=bool(forced))
            self.db.execute("UPDATE exploration_entitlements SET audit=? WHERE observation_id=?",
                            ("queued" if chosen else "not_drawn", identity))
            if chosen:
                selected.append(identity)
        return selected

    def _ban(self, hotkey: str, now: float, ban_seconds: int) -> None:
        self.db.execute("INSERT INTO exploration_bans VALUES(?,?,?) ON CONFLICT(order_id,hotkey) "
                        "DO UPDATE SET until=MAX(until, excluded.until)", (self.order, hotkey, now + ban_seconds))

    def record_audit(self, observation_id: str, *, passed: bool, now: float, ban_seconds: int) -> list[str]:
        row = self.db.execute("SELECT window, hotkey, environment, audit FROM exploration_entitlements "
                              "WHERE observation_id=? AND order_id=?", (observation_id, self.order)).fetchone()
        if row is None:
            raise ValueError("unknown exploration entitlement")
        window, hotkey, environment, state = row
        if self.is_finalized(window, environment=environment):
            # Frozen row: a late verdict changes nothing, but a failure is still sanctioned.
            if not passed:
                self._ban(hotkey, now, ban_seconds)
            return []
        if state == ("passed" if passed else "failed"):
            return []
        if state != "queued":  # never audited, or already settled otherwise: no late flip, no late sanction
            raise ValueError(f"exploration entitlement is not awaiting an audit ({state})")
        if passed:
            self.db.execute("UPDATE exploration_entitlements SET audit='passed' WHERE observation_id=?", (observation_id,))
            return []
        self.db.execute("UPDATE exploration_entitlements SET audit='failed' WHERE observation_id=?", (observation_id,))
        # Every entitlement of the hotkey in this window, in every env not yet finalized.
        open_rows = """FROM exploration_entitlements e WHERE e.order_id=? AND e.window=? AND e.hotkey=?
                       AND NOT EXISTS (SELECT 1 FROM exploration_finalized f WHERE f.order_id=e.order_id
                                       AND f.window=e.window AND f.environment=e.environment)"""
        forfeited = [r for r, in self.db.execute(f"SELECT e.observation_id {open_rows}",
                                                 (self.order, window, hotkey)).fetchall()]
        for identity in forfeited:
            self.db.execute("UPDATE exploration_entitlements SET status='forfeited' WHERE observation_id=?",
                            (identity,))
        self._ban(hotkey, now, ban_seconds)
        return forfeited

    def mark_unaudited(self, observation_id: str) -> None:
        self.db.execute("UPDATE exploration_entitlements SET audit='unaudited' WHERE observation_id=? "
                        "AND audit IN ('pending_draw','queued')", (observation_id,))

    def finalize_window(self, window: int, *, environment: str) -> list[str]:
        moved = [r for r, in self.db.execute(
            "SELECT observation_id FROM exploration_entitlements WHERE order_id=? AND window=? "
            "AND environment=? AND audit IN ('pending_draw','queued')", (self.order, window, environment)).fetchall()]
        for identity in moved:
            self.mark_unaudited(identity)
        self.db.execute("INSERT OR IGNORE INTO exploration_finalized VALUES(?,?,?)", (self.order, window, environment))
        return moved

    def payable(self, window: int, *, environment: str) -> dict[str, float]:
        result: dict[str, float] = {}
        for hotkey, amount in self.db.execute(
                "SELECT hotkey, amount FROM exploration_entitlements WHERE order_id=? AND window=? "
                "AND environment=? AND status='reserved' AND audit IN ('not_drawn','passed') ORDER BY rowid",
                (self.order, window, environment)):
            result[hotkey] = result.get(hotkey, 0.0) + amount
        return result

    def rows(self, window: int, *, environment: str) -> list[dict]:
        names = ("observation_id", "window", "environment", "hotkey", "prompt_idx", "amount",
                 "draw_round", "forced", "audit", "status")
        return [dict(zip(names, row)) for row in self.db.execute(
            "SELECT observation_id, window, environment, hotkey, prompt_idx, amount, draw_round, forced, audit, status "
            "FROM exploration_entitlements WHERE order_id=? AND window=? AND environment=? ORDER BY rowid",
            (self.order, window, environment))]
