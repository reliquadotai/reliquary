"""Exploration pay (first scan only, 15 % of a training group, per-env 10 % cap) and its sampled audit.

Ledger and log methods write without committing; the caller wraps them in ``with db:``. The three
module functions at the bottom (``record_exploration``, ``apply_exploration_audit``,
``finalize_exploration``) are the entry points a runtime should use: each is atomic on the
connection shared by the log and the ledger and keeps the first-scan table consistent with the money.

Rules a reader of the public log can rely on:

* An entitlement is one whole unit. Every entitlement of a (window, env) has the same nominal
  price; ``payable`` therefore returns integer counts and settlement multiplies.
* Probation: a group is forced to audit while its hotkey has fewer than ``new_hotkey_audit_groups``
  groups whose audit PASSED (same order, any window, any env, still entitled). Forfeited, unaudited
  and not-drawn groups never count, so a hotkey whose forced groups are never audited stays forced.
* Audit horizon: when a (window, env) is finalized, every group still waiting for its draw or for
  its audit becomes ``unaudited``. If at least one DRAWN group was left unaudited, every
  ``not_drawn`` group of that (window, env) whose draw round is >= the smallest draw round among
  those drawn-unaudited groups becomes ``unaudited`` as well. A group is paid without an audit only
  if every audit drawn up to its round was actually performed. ``unaudited`` is unpaid and never
  sanctioned.
* First scans: an entitlement that ends unpaid for any reason (refused at admission, forfeited by
  a failed audit, ``unaudited`` at finalize, horizon included) gives its first scan back, so the
  prompt can be paid to a later observation. Only a paid entitlement keeps the prompt.
* Once a (window, env) is finalized its rows never change: no reservation, no draw, no forfeit, no
  release. A failed audit that arrives late only bans, once, and only for a group that was drawn.
"""
from __future__ import annotations

import hashlib
import math
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # pragma: no cover
    from reliquary.services.run_log import Observation, RunObservationLog

AUDIT_DRAW_ROUND_OFFSET = 2
_AUDIT_DOMAIN = b"reliquary-exploration-audit/v1"
_EPS = 1e-12

STATUS_PENDING = "exploration_pending"
STATUS_UNPAID = "exploration_unpaid"
REFUSAL_REASONS = ("already_scanned", "banned", "finalized", "zero_price", "cap")


def training_group_price(pool: float, *, picks_target: int, batch_slots: int) -> float:
    if picks_target < 1 or batch_slots < 1:
        raise ValueError("training slots must be positive")
    return float(pool) / (picks_target * batch_slots)


def exploration_price(pool: float, *, picks_target: int, batch_slots: int, price_bps: int) -> float:
    return training_group_price(pool, picks_target=picks_target, batch_slots=batch_slots) * price_bps / 10000


def exploration_cap(pool: float, *, cap_bps: int) -> float:
    return float(pool) * cap_bps / 10000


def exploration_within_cap(count: int, *, price: float, cap: float) -> bool:
    """THE cap test: ``count`` whole entitlements at the nominal ``price`` fit under ``cap``.

    The ledger (at reservation) and settlement (settle and validate) all call this with the values
    of ``exploration_price`` / ``exploration_cap``, so what the ledger admits always validates.
    """
    return count * price <= cap + _EPS


def audit_selected(*, beacon_randomness: str, observation_id: str, audit_bps: int, forced: bool) -> bool:
    if forced:
        return True
    if audit_bps <= 0:
        return False
    if not isinstance(beacon_randomness, str):
        raise ValueError("audit beacon must be 32 bytes of hex")
    beacon = bytes.fromhex(beacon_randomness)
    if len(beacon) != 32:
        raise ValueError("audit beacon must be exactly 32 bytes")
    digest = hashlib.sha256(_AUDIT_DOMAIN + beacon + bytes.fromhex(observation_id)).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < audit_bps / 10000


def _money(value, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid exploration {name}")
    return float(value)


class ExplorationLedger:
    def __init__(self, db: sqlite3.Connection, *, order_sha256: str):
        if db.in_transaction:  # executescript would commit the caller's open transaction
            raise ValueError("ExplorationLedger must be constructed outside a transaction")
        self.db, self.order = db, order_sha256
        db.executescript("""
            CREATE TABLE IF NOT EXISTS exploration_entitlements(
                observation_id TEXT PRIMARY KEY, order_id TEXT NOT NULL, window INTEGER NOT NULL,
                environment TEXT NOT NULL, hotkey TEXT NOT NULL, prompt_idx INTEGER NOT NULL,
                amount REAL NOT NULL, draw_round INTEGER NOT NULL, forced INTEGER NOT NULL,
                audit TEXT NOT NULL, status TEXT NOT NULL, drawn INTEGER NOT NULL DEFAULT 0);
            CREATE INDEX IF NOT EXISTS exploration_window ON exploration_entitlements(order_id, window);
            CREATE INDEX IF NOT EXISTS exploration_hotkey ON exploration_entitlements(order_id, hotkey);
            CREATE TABLE IF NOT EXISTS exploration_finalized(
                order_id TEXT NOT NULL, window INTEGER NOT NULL, environment TEXT NOT NULL,
                PRIMARY KEY(order_id, window, environment));
            CREATE TABLE IF NOT EXISTS exploration_bans(
                order_id TEXT NOT NULL, hotkey TEXT NOT NULL, until REAL NOT NULL, PRIMARY KEY(order_id, hotkey));
            CREATE TABLE IF NOT EXISTS exploration_late_failures(
                order_id TEXT NOT NULL, observation_id TEXT NOT NULL, PRIMARY KEY(order_id, observation_id));
        """)

    def banned(self, hotkey: str, now: float) -> bool:
        row = self.db.execute("SELECT until FROM exploration_bans WHERE order_id=? AND hotkey=?",
                              (self.order, hotkey)).fetchone()
        return row is not None and now < row[0]

    def passed_audits(self, hotkey: str) -> int:
        """Probation counter: this hotkey's groups whose audit passed and that are still entitled,
        over every window and env of the order. Derived from the rows, so it survives a reopen."""
        return int(self.db.execute(
            "SELECT COUNT(*) FROM exploration_entitlements WHERE order_id=? AND hotkey=? "
            "AND audit='passed' AND status='reserved'", (self.order, hotkey)).fetchone()[0])

    def entitlement(self, observation_id: str) -> dict | None:
        row = self.db.execute("SELECT amount, forced, draw_round FROM exploration_entitlements "
                              "WHERE observation_id=? AND order_id=?", (observation_id, self.order)).fetchone()
        return None if row is None else {"amount": row[0], "forced": bool(row[1]), "draw_round": row[2]}

    def refusal(self, *, window: int, environment: str, hotkey: str, amount: float, cap: float,
                now: float | None = None) -> str | None:
        """Why a NEW reservation would be refused (one of ``REFUSAL_REASONS``), or None.

        ``now`` enables the ban check; without it the caller answers for the ban.
        """
        amount, cap = _money(amount, "price"), _money(cap, "cap")
        if now is not None and self.banned(hotkey, now):
            return "banned"
        if self.is_finalized(window, environment=environment):
            return "finalized"
        if amount <= 0:
            return "zero_price"
        other = self.db.execute(
            "SELECT 1 FROM exploration_entitlements WHERE order_id=? AND window=? AND environment=? "
            "AND amount<>? LIMIT 1", (self.order, window, environment, amount)).fetchone()
        if other is not None:  # counts are only money if one window has one price
            raise ValueError("exploration price changed inside a (window, env)")
        used = self.db.execute(
            "SELECT COUNT(*) FROM exploration_entitlements WHERE order_id=? AND window=? "
            "AND environment=? AND status='reserved' AND audit<>'unaudited'",
            (self.order, window, environment)).fetchone()[0]
        if not exploration_within_cap(used + 1, price=amount, cap=cap):
            return "cap"
        return None

    def reserve(self, *, window: int, environment: str, observation_id: str, hotkey: str, prompt_idx: int,
                amount: float, cap: float, draw_round: int, new_hotkey_audit_groups: int,
                now: float | None = None) -> dict | None:
        """Reserve one entitlement, or return None (see ``refusal``). A replay of an observation
        that already holds an entitlement returns it unchanged and counts nothing twice."""
        known = self.entitlement(observation_id)
        if known is not None:
            return known
        if self.refusal(window=window, environment=environment, hotkey=hotkey, amount=amount, cap=cap,
                        now=now) is not None:
            return None
        forced = self.passed_audits(hotkey) < new_hotkey_audit_groups
        self.db.execute(
            "INSERT INTO exploration_entitlements(observation_id, order_id, window, environment, hotkey, "
            "prompt_idx, amount, draw_round, forced, audit, status, drawn) VALUES(?,?,?,?,?,?,?,?,?,?,?,0)",
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
        if self.is_finalized(window, environment=environment):
            return []
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
            self.db.execute("UPDATE exploration_entitlements SET audit=?, drawn=? WHERE observation_id=?",
                            ("queued" if chosen else "not_drawn", int(chosen), identity))
            if chosen:
                selected.append(identity)
        return selected

    def _ban(self, hotkey: str, now: float, ban_seconds: int) -> None:
        self.db.execute("INSERT INTO exploration_bans VALUES(?,?,?) ON CONFLICT(order_id,hotkey) "
                        "DO UPDATE SET until=MAX(until, excluded.until)", (self.order, hotkey, now + ban_seconds))

    def _forfeited(self, window: int, hotkey: str) -> list[str]:
        return [r for r, in self.db.execute(
            "SELECT observation_id FROM exploration_entitlements WHERE order_id=? AND window=? AND hotkey=? "
            "AND status='forfeited' ORDER BY rowid", (self.order, window, hotkey)).fetchall()]

    def record_audit(self, observation_id: str, *, passed: bool, now: float, ban_seconds: int) -> list[str]:
        """Record an audit verdict; return the ids whose first scan the caller must release.

        A failure forfeits every entitlement of the hotkey in that window, in every env not yet
        finalized, and bans the hotkey. Replaying a failed verdict returns the same forfeited ids
        again and never bans twice. In a finalized (window, env) no row changes: a late failure
        bans once, and only if the group was drawn; anything else is a no-op.
        """
        row = self.db.execute("SELECT window, hotkey, environment, audit, drawn FROM exploration_entitlements "
                              "WHERE observation_id=? AND order_id=?", (observation_id, self.order)).fetchone()
        if row is None:
            raise ValueError("unknown exploration entitlement")
        window, hotkey, environment, state, drawn = row
        if not passed and state == "failed":  # replay: the caller's release must be retry-safe
            return self._forfeited(window, hotkey)
        if self.is_finalized(window, environment=environment):
            if not passed and drawn and state == "unaudited":
                first = self.db.execute("INSERT OR IGNORE INTO exploration_late_failures VALUES(?,?)",
                                        (self.order, observation_id)).rowcount == 1
                if first:
                    self._ban(hotkey, now, ban_seconds)
            return []
        if passed and state == "passed":
            return []
        if state != "queued":  # never drawn, or already settled otherwise: no late flip, no late sanction
            raise ValueError(f"exploration entitlement is not awaiting an audit ({state})")
        if passed:
            self.db.execute("UPDATE exploration_entitlements SET audit='passed' WHERE observation_id=?", (observation_id,))
            return []
        self.db.execute("UPDATE exploration_entitlements SET audit='failed' WHERE observation_id=?", (observation_id,))
        # Every entitlement of the hotkey in this window, in every env not yet finalized.
        self.db.execute(
            """UPDATE exploration_entitlements SET status='forfeited' WHERE order_id=? AND window=? AND hotkey=?
               AND NOT EXISTS (SELECT 1 FROM exploration_finalized f WHERE f.order_id=exploration_entitlements.order_id
                               AND f.window=exploration_entitlements.window
                               AND f.environment=exploration_entitlements.environment)""",
            (self.order, window, hotkey))
        self._ban(hotkey, now, ban_seconds)
        return self._forfeited(window, hotkey)

    def mark_unaudited(self, observation_id: str) -> None:
        self.db.execute("UPDATE exploration_entitlements SET audit='unaudited' WHERE observation_id=? "
                        "AND order_id=? AND audit IN ('pending_draw','queued')", (observation_id, self.order))

    def finalize_window(self, window: int, *, environment: str) -> list[str]:
        """Freeze a (window, env) and return every ``unaudited`` id of it (all unpaid, none sanctioned).

        Groups waiting for their draw or their audit become ``unaudited``; so does every
        ``not_drawn`` group at or after the audit horizon (module docstring). The caller releases
        the first scan of every returned id in the same transaction (``finalize_exploration``).
        Calling it again changes nothing and returns the same ids.
        """
        scope = (self.order, window, environment)
        if not self.is_finalized(window, environment=environment):
            horizon = self.db.execute(
                "SELECT MIN(draw_round) FROM exploration_entitlements WHERE order_id=? AND window=? "
                "AND environment=? AND audit='queued'", scope).fetchone()[0]
            self.db.execute("UPDATE exploration_entitlements SET audit='unaudited' WHERE order_id=? AND window=? "
                            "AND environment=? AND audit IN ('pending_draw','queued')", scope)
            if horizon is not None:
                self.db.execute("UPDATE exploration_entitlements SET audit='unaudited' WHERE order_id=? AND window=? "
                                "AND environment=? AND audit='not_drawn' AND draw_round>=?", (*scope, horizon))
            self.db.execute("INSERT INTO exploration_finalized VALUES(?,?,?)", scope)
        return [r for r, in self.db.execute(
            "SELECT observation_id FROM exploration_entitlements WHERE order_id=? AND window=? "
            "AND environment=? AND audit='unaudited' ORDER BY rowid", scope).fetchall()]

    def payable(self, window: int, *, environment: str) -> dict[str, int]:
        """``{hotkey: whole entitlements to pay}`` of a FINALIZED (window, env): still entitled and
        either audited-and-passed or not drawn inside the audit horizon. Settlement prices them."""
        if not self.is_finalized(window, environment=environment):
            raise ValueError("exploration is payable only once its (window, env) is finalized")
        return {hotkey: int(count) for hotkey, count in self.db.execute(
            "SELECT hotkey, COUNT(*) FROM exploration_entitlements WHERE order_id=? AND window=? "
            "AND environment=? AND status='reserved' AND audit IN ('not_drawn','passed') "
            "GROUP BY hotkey ORDER BY hotkey", (self.order, window, environment))}

    def rows(self, window: int, *, environment: str) -> list[dict]:
        names = ("observation_id", "window", "environment", "hotkey", "prompt_idx", "amount",
                 "draw_round", "forced", "audit", "status", "drawn")
        return [dict(zip(names, row)) for row in self.db.execute(
            "SELECT observation_id, window, environment, hotkey, prompt_idx, amount, draw_round, forced, audit, "
            "status, drawn FROM exploration_entitlements WHERE order_id=? AND window=? AND environment=? ORDER BY rowid",
            (self.order, window, environment))]


@dataclass(frozen=True, slots=True)
class ExplorationOutcome:
    """What ``record_exploration`` did with one exploration observation."""
    observation_id: str
    inserted: bool            # False: replay of the same submission, nothing was written
    first_scan: bool          # the observation holds the prompt's first scan AFTER this call
    category: str
    status: str               # STATUS_PENDING (entitled) or STATUS_UNPAID (published as such)
    reason: str | None        # None when entitled; else REFUSAL_REASONS, the caller's own, or "replay"
    entitlement: dict | None  # {"amount", "forced", "draw_round"} when entitled


def _same_store(log: "RunObservationLog", ledger: ExplorationLedger) -> sqlite3.Connection:
    if log.db is not ledger.db or log.order != ledger.order:
        raise ValueError("the observation log and the exploration ledger must share one connection and one order")
    return log.db


class _Atomic:
    """A savepoint: joins the caller's open transaction, or is a transaction of its own."""

    def __init__(self, db: sqlite3.Connection):
        self.db = db

    def __enter__(self):
        self.db.execute("SAVEPOINT exploration_entry")

    def __exit__(self, kind, value, trace):
        if kind is not None:
            self.db.execute("ROLLBACK TO exploration_entry")
        self.db.execute("RELEASE exploration_entry")
        return False


def record_exploration(log: "RunObservationLog", ledger: ExplorationLedger, obs: "Observation", *,
                       amount: float, cap: float, draw_round: int, new_hotkey_audit_groups: int,
                       now: float, refuse: str | None = None) -> ExplorationOutcome:
    """THE way to admit an exploration observation: record it and reserve its pay atomically.

    The observation is published ``exploration_pending`` when an entitlement is reserved. Otherwise
    it is published ``exploration_unpaid`` with the reason (``already_scanned``, ``banned`` at
    ``now``, ``finalized``, ``zero_price``, ``cap``, or the caller's ``refuse`` for its own
    eligibility rules) and, if it took the prompt's first scan, the scan is released: an unpaid
    observation never burns a prompt. A replay of the same submission writes nothing and reports
    the entitlement it already holds, if any (reason ``"replay"`` otherwise).

    ``amount`` is ``exploration_price`` and ``cap`` is ``exploration_cap`` of the window's frozen
    env pool; ``draw_round`` is the validator-clock arrival round + ``AUDIT_DRAW_ROUND_OFFSET``.
    Raises (and writes nothing) on a non-observation, on conflicting evidence, or on bad numbers.
    """
    db = _same_store(log, ledger)
    if obs.lane != "exploration":
        raise ValueError("record_exploration only takes exploration observations")
    with _Atomic(db):
        if log.is_scanned(obs.environment, obs.prompt_idx):
            reason = "already_scanned"
        elif refuse is not None:
            reason = str(refuse)
        else:
            reason = ledger.refusal(window=obs.window, environment=obs.environment, hotkey=obs.hotkey,
                                    amount=amount, cap=cap, now=now)
        result = log.record(obs, status=STATUS_PENDING if reason is None else STATUS_UNPAID,
                            proof="pending" if reason is None else "unproven", reason=reason)
        if not result.inserted:
            held = ledger.entitlement(result.observation_id)
            return ExplorationOutcome(result.observation_id, False, result.first_scan, result.category,
                                      STATUS_PENDING if held else STATUS_UNPAID, None if held else "replay", held)
        if reason is not None:
            if result.first_scan:
                log.release_first_scan(result.observation_id)
            return ExplorationOutcome(result.observation_id, True, False, result.category, STATUS_UNPAID, reason, None)
        entitlement = ledger.reserve(
            window=obs.window, environment=obs.environment, observation_id=result.observation_id,
            hotkey=obs.hotkey, prompt_idx=obs.prompt_idx, amount=amount, cap=cap, draw_round=draw_round,
            new_hotkey_audit_groups=new_hotkey_audit_groups, now=now)
        if entitlement is None or not result.first_scan:  # the checks above said yes: roll everything back
            raise RuntimeError("exploration reservation and first scan disagree")
        return ExplorationOutcome(result.observation_id, True, True, result.category, STATUS_PENDING, None, entitlement)


def apply_exploration_audit(log: "RunObservationLog", ledger: ExplorationLedger, observation_id: str, *,
                            passed: bool, now: float, ban_seconds: int) -> list[str]:
    """Record an audit verdict and release the first scan of every forfeited id, atomically."""
    with _Atomic(_same_store(log, ledger)):
        forfeited = ledger.record_audit(observation_id, passed=passed, now=now, ban_seconds=ban_seconds)
        for identity in forfeited:
            log.release_first_scan(identity)
        return forfeited


def finalize_exploration(log: "RunObservationLog", ledger: ExplorationLedger, window: int, *,
                         environment: str) -> list[str]:
    """Finalize a (window, env) and release the first scan of every unaudited id, atomically."""
    with _Atomic(_same_store(log, ledger)):
        unaudited = ledger.finalize_window(window, environment=environment)
        for identity in unaudited:
            log.release_first_scan(identity)
        return unaudited
