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
* Audit horizon (per hotkey): when a (window, env) is finalized, every group still waiting for its
  draw or for its audit becomes ``unaudited``. If a hotkey has at least one DRAWN group, not
  forfeited, left unaudited, every ``not_drawn`` group OF THAT HOTKEY in that (window, env) whose
  draw round is >= the smallest draw round among its drawn-unaudited groups becomes ``unaudited`` as
  well. Another hotkey's groups are never affected, and a forfeited group sets no horizon (its
  hotkey already lost the window). A group is paid without an audit only if every audit drawn for
  its own hotkey up to its round was actually performed. ``unaudited`` is unpaid and never
  sanctioned; nobody is banned by the horizon.
* What the runtime owes: audits of hotkeys past probation run before audits of hotkeys still in
  probation, and at seal the queued audits are drained (bounded wait, and up to 2 drand rounds for
  pending draws) BEFORE ``finalize_exploration`` is called. The audit load is bounded by the cap.
* First scans: an entitlement that ends unpaid for any reason (refused at admission, forfeited by
  a failed audit, ``unaudited`` at finalize, horizon included, ``trained``, aborted window) gives its
  first scan back through ``RunObservationLog.release_first_scan``, the one place that decides what
  the prompt becomes: still scanned if a counting training observation of it exists in the run,
  else free, so it can be paid to a later observation. A prompt with a training observation is
  scanned: an exploration observation of it is refused ``already_scanned``.
* Trained in the same window (R17): at finalize of a (window, env), every entitlement still
  ``reserved`` whose prompt has a training-lane observation recorded in that window (any hotkey,
  before or after) becomes ``trained``: unpaid, never sanctioned, no ban, no probation lost. It sets
  no audit horizon and needs no audit. Exploration is paid for prompts nobody trains in the window.
* Once a (window, env) is finalized its rows never change: no reservation, no draw, no forfeit, no
  release. A failed audit that arrives late bans once (only for a group that was drawn) and forfeits
  the hotkey's entitlements in the envs of that window not yet finalized (R3). Two transitions
  are allowed after finalize, both only towards unpaid: ``aborted=True`` (an aborted window pays no
  exploration: every row still ``reserved`` becomes ``unpaid``) and R17 for a training observation
  recorded after the env was finalized (``trained``). Each gives its first scan back.
* Probation cap: a hotkey still in probation holds at most ``PROBATION_PENDING_LIMIT`` reserved rows
  whose audit has not passed per (window, env); beyond that it is refused (``probation_limit``).
"""
from __future__ import annotations

import hashlib
import math
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from reliquary.constants import PROBATION_PENDING_LIMIT

if TYPE_CHECKING:  # pragma: no cover
    from reliquary.services.run_log import Observation, RunObservationLog

AUDIT_DRAW_ROUND_OFFSET = 2
_AUDIT_DOMAIN = b"reliquary-exploration-audit/v1"
_EPS = 1e-12

STATUS_PENDING = "exploration_pending"
STATUS_UNPAID = "exploration_unpaid"
STATUS_FORFEITED = "exploration_forfeited"
REFUSAL_REASONS = ("already_scanned", "banned", "finalized", "zero_price", "probation_limit", "cap")
MAX_AMOUNT = 1e12  # a pool, price or cap above this is not money; it also keeps every product finite


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


def finite_amount(value, bound: float = MAX_AMOUNT) -> bool:
    """True for an int/float in ``[0, bound]``. Never raises: a 400-digit int is just "too big"
    (``math.isfinite`` raises OverflowError on it)."""
    if type(value) is int:
        return 0 <= value <= bound
    return type(value) is float and math.isfinite(value) and 0 <= value <= bound


def _money(value, name: str) -> float:
    if not finite_amount(value):
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
        """Probation counter: this hotkey's groups whose audit passed and that were not forfeited
        (a pass is a fact even if the window was later aborted), over every window and env of the order. Derived from the rows, so it survives a reopen."""
        return int(self.db.execute(
            "SELECT COUNT(*) FROM exploration_entitlements WHERE order_id=? AND hotkey=? "
            "AND audit='passed' AND status<>'forfeited'", (self.order, hotkey)).fetchone()[0])

    def entitlement(self, observation_id: str) -> dict | None:
        row = self.db.execute("SELECT amount, forced, draw_round FROM exploration_entitlements "
                              "WHERE observation_id=? AND order_id=?", (observation_id, self.order)).fetchone()
        return None if row is None else {"amount": row[0], "forced": bool(row[1]), "draw_round": row[2]}

    def state(self, observation_id: str) -> tuple[str, str] | None:
        """``(audit, status)`` of an entitlement as it is NOW, or None."""
        row = self.db.execute("SELECT audit, status FROM exploration_entitlements WHERE observation_id=? "
                              "AND order_id=?", (observation_id, self.order)).fetchone()
        return None if row is None else (row[0], row[1])

    def refusal(self, *, window: int, environment: str, hotkey: str, amount: float, cap: float,
                now: float | None = None, new_hotkey_audit_groups: int | None = None) -> str | None:
        """Why a NEW reservation would be refused (one of ``REFUSAL_REASONS``), or None.

        ``now`` enables the ban check; without it the caller answers for the ban.
        ``new_hotkey_audit_groups`` enables the probation cap (``probation_limit``).
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
        if new_hotkey_audit_groups is not None and self.passed_audits(hotkey) < new_hotkey_audit_groups:
            holding = self.db.execute(
                "SELECT COUNT(*) FROM exploration_entitlements WHERE order_id=? AND window=? AND environment=? "
                "AND hotkey=? AND status='reserved' AND audit NOT IN ('passed','unaudited','failed')",
                (self.order, window, environment, hotkey)).fetchone()[0]
            if holding >= PROBATION_PENDING_LIMIT:
                return "probation_limit"
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
                        now=now, new_hotkey_audit_groups=new_hotkey_audit_groups) is not None:
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

    def _forfeit_open_envs(self, window: int, hotkey: str) -> list[str]:
        """Forfeit every entitlement of ``hotkey`` in ``window``, in every env not yet finalized;
        return the hotkey's forfeited ids of the window (idempotent)."""
        self.db.execute(
            """UPDATE exploration_entitlements SET status='forfeited' WHERE order_id=? AND window=? AND hotkey=?
               AND NOT EXISTS (SELECT 1 FROM exploration_finalized f WHERE f.order_id=exploration_entitlements.order_id
                               AND f.window=exploration_entitlements.window
                               AND f.environment=exploration_entitlements.environment)""",
            (self.order, window, hotkey))
        return self._forfeited(window, hotkey)

    def apply_verdict(self, observation_id: str, *, passed: bool, now: float, ban_seconds: int) -> tuple[str, list[str]]:
        """Record an audit verdict; return ``(kind, ids)``.

        ``kind`` is ``"passed"`` (the row is audited-and-passed), ``"failed"`` (a failure was applied:
        ban, and ``ids`` = the hotkey's forfeited ids of the window, whose first scans the caller
        must release) or ``"not_applied"`` (nothing changed). Raises ValueError for an unknown id or
        a row that never awaited an audit (never drawn).

        A failure forfeits every entitlement of the hotkey in that window, in every env not yet
        finalized, and bans the hotkey. Replaying a failed verdict returns the same forfeited ids
        again and never bans twice. A row already marked ``unaudited`` (lost its audit slot): a pass
        is ignored (``not_applied``); a failure of a DRAWN row counts as in any other state. In a
        finalized (window, env) no row of it changes: a late failure on a drawn, unaudited row bans
        once and forfeits the hotkey's entitlements of the window in the envs not yet finalized (R3).
        """
        row = self.db.execute("SELECT window, hotkey, environment, audit, drawn, status FROM "
                              "exploration_entitlements WHERE observation_id=? AND order_id=?",
                              (observation_id, self.order)).fetchone()
        if row is None:
            raise ValueError("unknown exploration entitlement")
        window, hotkey, environment, state, drawn, status = row
        if not passed and state == "failed":  # replay: the caller's release must be retry-safe
            return "failed", self._forfeited(window, hotkey)
        if self.is_finalized(window, environment=environment):
            if passed:
                return ("passed" if state == "passed" else "not_applied"), []
            if not (drawn and state == "unaudited"):
                return "not_applied", []
            if self.db.execute("INSERT OR IGNORE INTO exploration_late_failures VALUES(?,?)",
                               (self.order, observation_id)).rowcount == 1:
                self._ban(hotkey, now, ban_seconds)
            return "failed", self._forfeit_open_envs(window, hotkey)
        if passed and state == "passed":
            return "passed", []
        if passed and status == "forfeited":  # the hotkey already lost the window: a pass flips nothing
            return "not_applied", []
        if state == "unaudited" and drawn:
            if passed:
                return "not_applied", []
        elif state != "queued":  # never drawn, or already settled otherwise: no late flip, no late sanction
            raise ValueError(f"exploration entitlement is not awaiting an audit ({state})")
        if passed:
            self.db.execute("UPDATE exploration_entitlements SET audit='passed' WHERE observation_id=?", (observation_id,))
            return "passed", []
        self.db.execute("UPDATE exploration_entitlements SET audit='failed' WHERE observation_id=?", (observation_id,))
        self._ban(hotkey, now, ban_seconds)
        return "failed", self._forfeit_open_envs(window, hotkey)

    def record_audit(self, observation_id: str, *, passed: bool, now: float, ban_seconds: int) -> list[str]:
        """``apply_verdict`` without the kind: the ids whose first scan the caller must release."""
        return self.apply_verdict(observation_id, passed=passed, now=now, ban_seconds=ban_seconds)[1]

    def mark_unaudited(self, observation_id: str) -> None:
        self.db.execute("UPDATE exploration_entitlements SET audit='unaudited' WHERE observation_id=? "
                        "AND order_id=? AND audit IN ('pending_draw','queued')", (observation_id, self.order))

    def finalize_window(self, window: int, *, environment: str, aborted: bool = False,
                        trained_prompts=()) -> list[str]:
        """Freeze a (window, env) and return every unpaid id of it (none sanctioned).

        Groups waiting for their draw or their audit become ``unaudited``; so does every
        ``not_drawn`` group of a hotkey at or after that hotkey's audit horizon (module docstring).
        The horizon of a hotkey is its smallest draw round among its DRAWN, still entitled groups
        that end unaudited, whether they were still ``queued`` here or ``mark_unaudited`` earlier.
        The caller releases the first scan of every returned id in the same transaction
        (``finalize_exploration``). Calling it again changes nothing and returns the same ids.

        ``aborted=True`` is the one transition allowed on an already finalized (window, env): the
        window pays no exploration, so every row still ``reserved`` becomes ``unpaid`` (``payable``
        is then empty) and is returned too. Idempotent.

        ``trained_prompts`` (R17): the prompts of this env with a training-lane observation recorded
        in this window. Every row still ``reserved`` on one of them becomes ``trained`` (unpaid, no
        sanction) BEFORE the audit horizon is computed, so it sets no horizon, and is returned too.
        Applied on every call, so a training observation recorded after the env was finalized still
        voids the row as long as the caller finalizes again before reading ``payable``.
        """
        scope = (self.order, window, environment)
        for prompt in sorted({int(p) for p in trained_prompts}):
            self.db.execute("UPDATE exploration_entitlements SET status='trained' WHERE order_id=? AND window=? "
                            "AND environment=? AND prompt_idx=? AND status='reserved'", (*scope, prompt))
        if not self.is_finalized(window, environment=environment):
            horizons = self.db.execute(
                "SELECT hotkey, MIN(draw_round) FROM exploration_entitlements WHERE order_id=? AND window=? "
                "AND environment=? AND drawn=1 AND audit IN ('queued','unaudited') AND status='reserved' "
                "GROUP BY hotkey", scope).fetchall()
            self.db.execute("UPDATE exploration_entitlements SET audit='unaudited' WHERE order_id=? AND window=? "
                            "AND environment=? AND audit IN ('pending_draw','queued')", scope)
            for hotkey, horizon in horizons:
                self.db.execute("UPDATE exploration_entitlements SET audit='unaudited' WHERE order_id=? AND window=? "
                                "AND environment=? AND hotkey=? AND audit='not_drawn' AND draw_round>=?",
                                (*scope, hotkey, horizon))
            self.db.execute("INSERT INTO exploration_finalized VALUES(?,?,?)", scope)
        if aborted:
            self.db.execute("UPDATE exploration_entitlements SET status='unpaid' WHERE order_id=? AND window=? "
                            "AND environment=? AND status='reserved'", scope)
        return [r for r, in self.db.execute(
            "SELECT observation_id FROM exploration_entitlements WHERE order_id=? AND window=? "
            "AND environment=? AND (audit='unaudited' OR status IN ('unpaid','trained')) ORDER BY rowid",
            scope).fetchall()]

    def payable(self, window: int, *, environment: str) -> dict[str, int]:
        """``{hotkey: whole entitlements to pay}`` of a FINALIZED (window, env): still entitled and
        either audited-and-passed or not drawn inside its hotkey's audit horizon. Settlement prices them."""
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
    """A savepoint: joins the caller's open transaction, or is a transaction of its own.

    A transaction of its own starts with ``BEGIN IMMEDIATE``: the entry points read before they
    write, and a deferred transaction that upgrades its read lock fails at once with
    "database is locked" when another connection writes meanwhile (the busy timeout cannot help).
    """

    def __init__(self, db: sqlite3.Connection):
        self.db, self.own = db, False

    def __enter__(self):
        if not self.db.in_transaction:
            self.db.execute("BEGIN IMMEDIATE")
            self.own = True
        self.db.execute("SAVEPOINT exploration_entry")

    def __exit__(self, kind, value, trace):
        if kind is not None:
            self.db.execute("ROLLBACK TO exploration_entry")
        self.db.execute("RELEASE exploration_entry")
        if self.own:
            self.db.execute("ROLLBACK" if kind is not None else "COMMIT")
        return False


def _replayed(log: "RunObservationLog", ledger: ExplorationLedger, result, *, obs: "Observation", amount: float,
              cap: float, draw_round: int, new_hotkey_audit_groups: int, now: float,
              refuse: str | None) -> ExplorationOutcome:
    """The outcome of a submission that is already in the log (nothing new is recorded).

    The status is derived from the row's state NOW. A submission that was refused only for the
    per-env ``cap`` is a new attempt when it comes back: if room exists it takes the prompt's first
    scan and a reservation (never a replay of the stale refusal).
    """
    state = ledger.state(result.observation_id)
    if state is not None:
        audit, status = state
        if status == "forfeited":
            return ExplorationOutcome(result.observation_id, False, result.first_scan, result.category,
                                      STATUS_FORFEITED, "replay", None)
        if status != "reserved" or audit == "unaudited":
            return ExplorationOutcome(result.observation_id, False, result.first_scan, result.category,
                                      STATUS_UNPAID, "replay", None)
        return ExplorationOutcome(result.observation_id, False, result.first_scan, result.category,
                                  STATUS_PENDING, None, ledger.entitlement(result.observation_id))
    if log.refusal_reason(result.observation_id) != "cap" or refuse is not None:
        return ExplorationOutcome(result.observation_id, False, result.first_scan, result.category,
                                  STATUS_UNPAID, "replay", None)
    if log.is_scanned(obs.environment, obs.prompt_idx):
        reason = "already_scanned"
    else:
        reason = ledger.refusal(window=obs.window, environment=obs.environment, hotkey=obs.hotkey, amount=amount,
                                cap=cap, now=now, new_hotkey_audit_groups=new_hotkey_audit_groups)
    if reason is not None:
        return ExplorationOutcome(result.observation_id, False, False, result.category, STATUS_UNPAID, reason, None)
    if not log.claim_first_scan(result.observation_id):
        raise RuntimeError("exploration reservation and first scan disagree")
    entitlement = ledger.reserve(
        window=obs.window, environment=obs.environment, observation_id=result.observation_id,
        hotkey=obs.hotkey, prompt_idx=obs.prompt_idx, amount=amount, cap=cap, draw_round=draw_round,
        new_hotkey_audit_groups=new_hotkey_audit_groups, now=now)
    if entitlement is None:
        raise RuntimeError("exploration reservation and first scan disagree")
    log.settle(result.observation_id, status=STATUS_PENDING, proof="pending", at=now)
    return ExplorationOutcome(result.observation_id, False, True, result.category, STATUS_PENDING, None, entitlement)


def record_exploration(log: "RunObservationLog", ledger: ExplorationLedger, obs: "Observation", *,
                       amount: float, cap: float, draw_round: int, new_hotkey_audit_groups: int,
                       now: float, refuse: str | None = None) -> ExplorationOutcome:
    """THE way to admit an exploration observation: record it and reserve its pay atomically.

    The observation is published ``exploration_pending`` when an entitlement is reserved. Otherwise
    it is published ``exploration_unpaid`` with the reason (``already_scanned``, ``banned`` at
    ``now``, ``finalized``, ``zero_price``, ``cap``, or the caller's ``refuse`` for its own
    eligibility rules) and, if it took the prompt's first scan, the scan is released: an unpaid
    observation never burns a prompt. A prompt with a training observation in the run is scanned
    (``already_scanned``), whoever held its first scan. A replay of the same submission writes nothing and reports
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
                                    amount=amount, cap=cap, now=now,
                                    new_hotkey_audit_groups=new_hotkey_audit_groups)
        result = log.record(obs, status=STATUS_PENDING if reason is None else STATUS_UNPAID,
                            proof="pending" if reason is None else "unproven", reason=reason)
        if not result.inserted:
            return _replayed(log, ledger, result, obs=obs, amount=amount, cap=cap, draw_round=draw_round,
                             new_hotkey_audit_groups=new_hotkey_audit_groups, now=now, refuse=refuse)
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


def apply_exploration_verdict(log: "RunObservationLog", ledger: ExplorationLedger, observation_id: str, *,
                              passed: bool, now: float, ban_seconds: int) -> tuple[str, list[str]]:
    """Record an audit verdict and release the first scan of every forfeited id, atomically.

    Returns ``(kind, forfeited_ids)``, kind being ``passed`` / ``failed`` / ``not_applied``."""
    with _Atomic(_same_store(log, ledger)):
        kind, forfeited = ledger.apply_verdict(observation_id, passed=passed, now=now, ban_seconds=ban_seconds)
        for identity in forfeited:
            log.release_first_scan(identity)
        return kind, forfeited


def apply_exploration_audit(log: "RunObservationLog", ledger: ExplorationLedger, observation_id: str, *,
                            passed: bool, now: float, ban_seconds: int) -> list[str]:
    """``apply_exploration_verdict`` without the kind: the forfeited ids (first scans released)."""
    return apply_exploration_verdict(log, ledger, observation_id, passed=passed, now=now,
                                     ban_seconds=ban_seconds)[1]


def finalize_exploration(log: "RunObservationLog", ledger: ExplorationLedger, window: int, *,
                         environment: str, aborted: bool = False) -> list[str]:
    """Finalize a (window, env) and release the first scan of every unpaid id, atomically.

    ``aborted=True`` also works on an already finalized (window, env): every row still reserved
    becomes unpaid and gives its first scan back (see ``ExplorationLedger.finalize_window``).

    R17 is applied here, in the same transaction and before the audit horizon: the entitlements
    whose prompt has a training-lane observation in this window (``log.trained_prompts``) end
    ``trained`` (unpaid, no sanction); their scan is re-seated on the training observation by
    ``log.release_first_scan``."""
    with _Atomic(_same_store(log, ledger)):
        unaudited = ledger.finalize_window(window, environment=environment, aborted=aborted,
                                           trained_prompts=log.trained_prompts(window, environment))
        for identity in unaudited:
            log.release_first_scan(identity)
        return unaudited
