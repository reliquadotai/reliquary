"""Exploration pay (first scan only, 15 % of a training group, per-env 10 % cap) and its sampled audit.

Ledger and log methods write without committing; the caller wraps them in ``with db:``. The three
module functions at the bottom (``record_exploration``, ``apply_exploration_verdict``,
``finalize_exploration``) are the entry points a runtime should use: each is atomic on the
connection shared by the log and the ledger and keeps the first-scan table consistent with the money.

Rules a reader of the public log can rely on:

* An entitlement is one whole unit. Every entitlement of a (window, env) has the same nominal
  price; ``payable`` therefore returns integer counts and settlement multiplies.
* Probation: a group is forced to audit while its hotkey has fewer than ``new_hotkey_audit_groups``
  groups whose audit PASSED (same order, any window, any env, still entitled). Forfeited, unaudited
  and not-drawn groups never count, so a hotkey whose forced groups are never audited stays forced.
  Re-probation (R31): after a STATISTICAL audit failure the hotkey is in probation again until it has
  ``SERVICE_REPROBATION_PASSES`` passed audits concluded after its last statistical failure (same
  counting rule). Both are derived from the persisted rows and the audit log, so they survive a reopen.
* Graded sanction (R31): every failed audit forfeits the hotkey's entitlements of the window in every
  env not yet finalized (unchanged). A ``deterministic`` failure also bans; a ``statistical`` one does
  not, unless more than ``SERVICE_STATISTICAL_FAIL_BAN_BPS`` of the last
  ``SERVICE_STATISTICAL_FAIL_WINDOW`` concluded audits of the hotkey (fixed denominator) failed
  statistically. ``AUDIT_FAILURE_CLASS_BY_STAGE`` maps the proof stage that rejected to the class.
* Audit horizon (per hotkey): when a (window, env) is finalized, every group still waiting for its
  draw or for its audit becomes ``unaudited``. If a hotkey has at least one DRAWN group, not
  forfeited, left unaudited, every ``not_drawn`` group OF THAT HOTKEY in that (window, env) whose
  draw round is >= the smallest draw round among its drawn-unaudited groups becomes ``unaudited`` as
  well. Another hotkey's groups are never affected, and a forfeited group sets no horizon (its
  hotkey already lost the window). A group is paid without an audit only if every audit drawn for
  its own hotkey up to its round was actually performed. ``unaudited`` is unpaid and never
  sanctioned; nobody is banned by the horizon.
* Unaudited reason (R25): the ledger records why a row ended ``unaudited``. ``validator_lost`` (the
  validator lost the audit inputs or could not run audits) is unpaid and unsanctioned and sets NO
  horizon; only ``unaudited`` (audits could run, the drain bound was reached) does. The public event
  says ``unaudited`` for both.
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
  before or after) becomes ``trained``: unpaid, never sanctioned, no ban, no probation lost. R17
  voids the PAY only, never the audit: a drawn row on a trained prompt stays in the audit queue,
  can still fail (ban and forfeit, as any failure), and if it is left unaudited it sets its
  hotkey's audit horizon like any other drawn unaudited row. Training a prompt is no way out of
  an audit.
* Once a (window, env) is finalized its rows never change: no reservation, no draw, no forfeit, no
  release. A failed audit that arrives late bans once (only for a group that was drawn) and forfeits
  the hotkey's entitlements in the envs of that window not yet finalized (R3). Two transitions
  are allowed after finalize, both only towards unpaid: ``aborted=True`` (an aborted window pays no
  exploration: every row still ``reserved`` becomes ``unpaid``) and R17 for a training observation
  recorded after the env was finalized (``trained``). Each gives its first scan back.
* Probation cap: a hotkey still in probation holds at most ``PROBATION_PENDING_LIMIT`` reserved rows
  whose audit has not passed per WINDOW, all envs together; beyond that it is refused
  (``probation_limit``).
* Audit draw (R18): keyed with the secret run salt, so a miner cannot tell which of its rows are
  drawn (``audit_selected``).
"""
from __future__ import annotations

import hashlib
import math
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import logging

from reliquary.constants import (
    PROBATION_PENDING_LIMIT,
    SERVICE_REPROBATION_PASSES,
    SERVICE_STATISTICAL_FAIL_BAN_BPS,
    SERVICE_STATISTICAL_FAIL_WINDOW,
)

if TYPE_CHECKING:  # pragma: no cover
    from reliquary.services.run_log import Observation, RunObservationLog

AUDIT_DRAW_ROUND_OFFSET = 2
_AUDIT_DOMAIN = b"reliquary-exploration-audit/v2"
_EPS = 1e-12

STATUS_PENDING = "exploration_pending"
STATUS_UNPAID = "exploration_unpaid"
STATUS_FORFEITED = "exploration_forfeited"
# Why a row ended ``unaudited`` (R25). The ledger keeps the difference; the public event says ``unaudited``
# for both. ``validator_lost``: the validator lost the audit inputs or could not run audits (restart,
# recovery, plan unavailable, checkpoint swapped): unpaid, unsanctioned, NO horizon. ``unaudited``: audits
# could run and the drain bound was reached: unpaid, unsanctioned, and the R15 horizon applies.
UNAUDITED_HORIZON = "unaudited"
UNAUDITED_VALIDATOR_LOST = "validator_lost"
UNAUDITED_REASONS = (UNAUDITED_HORIZON, UNAUDITED_VALIDATOR_LOST)
REFUSAL_REASONS = ("already_scanned", "banned", "finalized", "zero_price", "probation_limit", "cap")
# A refusal that is about the hotkey's or the env's room, not about the group: when the same submission
# comes back and room exists it is a new attempt (never a replay of the stale refusal).
RETRYABLE_REFUSALS = ("cap", "probation_limit")
MAX_AMOUNT = 1e12  # a pool, price or cap above this is not money; it also keeps every product finite

logger = logging.getLogger(__name__)

# Graded audit sanction (R31). THE table: proof stage that rejected an exploration audit -> failure class.
# ``deterministic``: forgery evidence that an honest miner cannot produce (24 h ban + window forfeit).
# ``statistical``: a threshold check an honest miner can fail by bad luck (window forfeit + re-probation;
# ban only on recidivism). A stage not listed here is ``statistical`` (the lenient class), logged at warning.
# ``forced_seed`` carries its scope: the hard CDF mismatch is deterministic, the agreement floors are not.
# ``service_contract`` / ``service_proof_capability`` never reach it (inconclusive: no verdict), nor does
# ``service_length`` (an audit raises it: the row's horizon, no verdict).
AUDIT_FAILURE_DETERMINISTIC = "deterministic"
AUDIT_FAILURE_STATISTICAL = "statistical"
AUDIT_FAILURE_CLASSES = (AUDIT_FAILURE_DETERMINISTIC, AUDIT_FAILURE_STATISTICAL)
AUDIT_FAILURE_CLASS_BY_STAGE = {
    "grail": AUDIT_FAILURE_DETERMINISTIC,                         # proof sketch mismatch
    "toploc": AUDIT_FAILURE_DETERMINISTIC,                        # TOPLOC proof mismatch
    "termination": AUDIT_FAILURE_DETERMINISTIC,                   # EOS padding / forced terminal pick mismatch
    "forged_termination": AUDIT_FAILURE_DETERMINISTIC,            # R26: proof-found cap where admission saw EOS
    "force_span": AUDIT_FAILURE_DETERMINISTIC,                    # BFT force span not byte-exact
    "service_seed_coverage": AUDIT_FAILURE_DETERMINISTIC,         # seed positions differ from sampled positions
    "token_authenticity": AUDIT_FAILURE_DETERMINISTIC,            # token authenticity
    "all_token_authenticity": AUDIT_FAILURE_DETERMINISTIC,        # token authenticity over every token
    "forced_seed/cdf_hard_mismatch": AUDIT_FAILURE_DETERMINISTIC, # forced-seed hard CDF mismatch
    "forced_seed/group": AUDIT_FAILURE_STATISTICAL,               # forced-seed agreement below the group floor
    "forced_seed/rollout": AUDIT_FAILURE_STATISTICAL,             # forced-seed agreement below the rollout floor
    "forced_seed": AUDIT_FAILURE_STATISTICAL,                     # forced-seed without a scope
    "logprob": AUDIT_FAILURE_STATISTICAL,                         # logprob deviation threshold
    "distribution": AUDIT_FAILURE_STATISTICAL,                    # token distribution threshold
    "boxed_answer": AUDIT_FAILURE_STATISTICAL,                    # boxed-answer probability threshold
    "code_semantic_auth": AUDIT_FAILURE_STATISTICAL,              # semantic token heuristic (positive reward)
    "episode_replay_binding": AUDIT_FAILURE_STATISTICAL,          # replay spans missing (admission's state)
}


def audit_failure_class(stage: str | None, scope: str | None = None) -> str:
    """The failure class of an audit rejected at ``stage`` (``scope``: the forced-seed scope), R31.

    Unknown stage (or none) -> ``statistical`` (the lenient class), logged at warning."""
    key = f"{stage}/{scope}" if scope else stage
    found = AUDIT_FAILURE_CLASS_BY_STAGE.get(key) if key is not None else None
    if found is None and scope:
        found = AUDIT_FAILURE_CLASS_BY_STAGE.get(stage)
    if found is None:
        logger.warning("exploration audit failed at an unclassified proof stage %r (scope %r): "
                       "treated as statistical (no ban)", stage, scope)
        return AUDIT_FAILURE_STATISTICAL
    return found


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


def audit_selected(*, beacon_randomness: str, observation_id: str, audit_bps: int, forced: bool,
                   run_salt: bytes) -> bool:
    """Whether a group is drawn for audit (R18).

    ``sha256(domain, beacon of the group's draw round, observation id, run salt)`` against the
    rate. The beacon and the id are public, the run salt is the validator's secret
    (``RunObservationLog.run_salt``): a miner cannot compute which of its rows are drawn. It is a
    pure function of persisted values, so a restart draws the same rows. ``forced`` (probation)
    is always drawn."""
    if forced:
        return True
    if audit_bps <= 0:
        return False
    if not isinstance(run_salt, (bytes, bytearray)) or len(run_salt) != 32:
        raise ValueError("audit draw needs the 32-byte run salt")
    if not isinstance(beacon_randomness, str):
        raise ValueError("audit beacon must be 32 bytes of hex")
    beacon = bytes.fromhex(beacon_randomness)
    if len(beacon) != 32:
        raise ValueError("audit beacon must be exactly 32 bytes")
    digest = hashlib.sha256(_AUDIT_DOMAIN + beacon + bytes.fromhex(observation_id) + bytes(run_salt)).digest()
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
                audit TEXT NOT NULL, status TEXT NOT NULL, drawn INTEGER NOT NULL DEFAULT 0,
                unaudited_reason TEXT);
            CREATE INDEX IF NOT EXISTS exploration_window ON exploration_entitlements(order_id, window);
            CREATE INDEX IF NOT EXISTS exploration_hotkey ON exploration_entitlements(order_id, hotkey);
            CREATE TABLE IF NOT EXISTS exploration_finalized(
                order_id TEXT NOT NULL, window INTEGER NOT NULL, environment TEXT NOT NULL,
                PRIMARY KEY(order_id, window, environment));
            CREATE TABLE IF NOT EXISTS exploration_bans(
                order_id TEXT NOT NULL, hotkey TEXT NOT NULL, until REAL NOT NULL, PRIMARY KEY(order_id, hotkey));
            CREATE TABLE IF NOT EXISTS exploration_late_failures(
                order_id TEXT NOT NULL, observation_id TEXT NOT NULL, PRIMARY KEY(order_id, observation_id));
            CREATE TABLE IF NOT EXISTS exploration_audit_log(
                seq INTEGER PRIMARY KEY AUTOINCREMENT, order_id TEXT NOT NULL, hotkey TEXT NOT NULL,
                observation_id TEXT NOT NULL, verdict TEXT NOT NULL, failure_class TEXT);
            CREATE INDEX IF NOT EXISTS exploration_audit_log_hotkey ON exploration_audit_log(order_id, hotkey, seq);
        """)
        columns = {r[1] for r in db.execute("PRAGMA table_info(exploration_entitlements)")}
        if "unaudited_reason" not in columns:
            db.execute("ALTER TABLE exploration_entitlements ADD COLUMN unaudited_reason TEXT")
        if "failure_class" not in columns:
            db.execute("ALTER TABLE exploration_entitlements ADD COLUMN failure_class TEXT")

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

    def _last_statistical_failure(self, hotkey: str) -> int | None:
        return self.db.execute(
            "SELECT MAX(seq) FROM exploration_audit_log WHERE order_id=? AND hotkey=? AND verdict='failed' "
            "AND failure_class=?", (self.order, hotkey, AUDIT_FAILURE_STATISTICAL)).fetchone()[0]

    def reprobation_passes(self, hotkey: str) -> int | None:
        """R31: passed audits concluded after the hotkey's last STATISTICAL failure (rows still not
        forfeited, as ``passed_audits``), or None when it never failed statistically."""
        since = self._last_statistical_failure(hotkey)
        if since is None:
            return None
        return int(self.db.execute(
            "SELECT COUNT(*) FROM exploration_audit_log l JOIN exploration_entitlements e "
            "ON e.observation_id=l.observation_id AND e.order_id=l.order_id WHERE l.order_id=? AND l.hotkey=? "
            "AND l.verdict='passed' AND l.seq>? AND e.status<>'forfeited'", (self.order, hotkey, since)).fetchone()[0])

    def in_probation(self, hotkey: str, new_hotkey_audit_groups: int) -> bool:
        """Every group of the hotkey is forced to audit: a new hotkey (fewer than ``new_hotkey_audit_groups``
        passed audits) or one in re-probation after a statistical failure (R31)."""
        if self.passed_audits(hotkey) < new_hotkey_audit_groups:
            return True
        since = self.reprobation_passes(hotkey)
        return since is not None and since < SERVICE_REPROBATION_PASSES

    def recent_statistical_failures(self, hotkey: str) -> int:
        """Statistical failures among the hotkey's last ``SERVICE_STATISTICAL_FAIL_WINDOW`` concluded audits."""
        return int(self.db.execute(
            "SELECT COUNT(*) FROM (SELECT verdict, failure_class FROM exploration_audit_log WHERE order_id=? "
            "AND hotkey=? ORDER BY seq DESC LIMIT ?) WHERE verdict='failed' AND failure_class=?",
            (self.order, hotkey, SERVICE_STATISTICAL_FAIL_WINDOW, AUDIT_FAILURE_STATISTICAL)).fetchone()[0])

    def audit_log(self, hotkey: str) -> list[dict]:
        """The hotkey's concluded audits in order: ``observation_id``, ``verdict``, ``failure_class``."""
        return [{"observation_id": oid, "verdict": verdict, "failure_class": klass} for oid, verdict, klass in
                self.db.execute("SELECT observation_id, verdict, failure_class FROM exploration_audit_log "
                                "WHERE order_id=? AND hotkey=? ORDER BY seq", (self.order, hotkey))]

    def _log_audit(self, hotkey: str, observation_id: str, verdict: str, failure_class: str | None = None) -> None:
        self.db.execute("INSERT INTO exploration_audit_log(order_id, hotkey, observation_id, verdict, failure_class) "
                        "VALUES(?,?,?,?,?)", (self.order, hotkey, observation_id, verdict, failure_class))

    def _sanction(self, hotkey: str, observation_id: str, failure_class: str, now: float, ban_seconds: int) -> None:
        """Log a newly applied failure and ban when its class (or the recidivism rule) says so (R31).
        The forfeit is the caller's, identical for both classes."""
        self._log_audit(hotkey, observation_id, "failed", failure_class)
        if failure_class == AUDIT_FAILURE_DETERMINISTIC:
            self._ban(hotkey, now, ban_seconds)
            return
        failures = self.recent_statistical_failures(hotkey)
        if failures * 10000 > SERVICE_STATISTICAL_FAIL_BAN_BPS * SERVICE_STATISTICAL_FAIL_WINDOW:
            logger.warning("exploration hotkey %s: %d statistical audit failures in its last %d audits; banned",
                           hotkey, failures, SERVICE_STATISTICAL_FAIL_WINDOW)
            self._ban(hotkey, now, ban_seconds)

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
        if new_hotkey_audit_groups is not None and self.in_probation(hotkey, new_hotkey_audit_groups):
            holding = self.db.execute(
                "SELECT COUNT(*) FROM exploration_entitlements WHERE order_id=? AND window=? "
                "AND hotkey=? AND status='reserved' AND audit NOT IN ('passed','unaudited','failed')",
                (self.order, window, hotkey)).fetchone()[0]
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
        forced = self.in_probation(hotkey, new_hotkey_audit_groups)
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
                      audit_bps: int, run_salt: bytes) -> list[str]:
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
                                    audit_bps=audit_bps, forced=bool(forced), run_salt=run_salt)
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

    def apply_verdict(self, observation_id: str, *, passed: bool, now: float, ban_seconds: int,
                      failure_class: str = AUDIT_FAILURE_DETERMINISTIC) -> tuple[str, list[str]]:
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

        R31: ``failure_class`` (``AUDIT_FAILURE_CLASSES``) grades the sanction of a failure: the forfeit is
        the same for both classes; only a ``deterministic`` failure (or statistical recidivism) bans. Every
        newly applied verdict is appended to the audit log (re-probation and recidivism read it).
        """
        if not passed and failure_class not in AUDIT_FAILURE_CLASSES:
            raise ValueError("unknown audit failure class")
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
                self._sanction(hotkey, observation_id, failure_class, now, ban_seconds)
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
            self._log_audit(hotkey, observation_id, "passed")
            return "passed", []
        self.db.execute("UPDATE exploration_entitlements SET audit='failed', failure_class=? WHERE observation_id=?",
                        (failure_class, observation_id))
        self._sanction(hotkey, observation_id, failure_class, now, ban_seconds)
        return "failed", self._forfeit_open_envs(window, hotkey)

    def record_audit(self, observation_id: str, *, passed: bool, now: float, ban_seconds: int,
                     failure_class: str = AUDIT_FAILURE_DETERMINISTIC) -> list[str]:
        """``apply_verdict`` without the kind: the ids whose first scan the caller must release."""
        return self.apply_verdict(observation_id, passed=passed, now=now, ban_seconds=ban_seconds,
                                  failure_class=failure_class)[1]

    def mark_unaudited(self, observation_id: str, reason: str = UNAUDITED_HORIZON) -> None:
        if reason not in UNAUDITED_REASONS:
            raise ValueError("unknown unaudited reason")
        self.db.execute("UPDATE exploration_entitlements SET audit='unaudited', unaudited_reason=? "
                        "WHERE observation_id=? AND order_id=? AND audit IN ('pending_draw','queued')",
                        (reason, observation_id, self.order))

    def unaudited_reason(self, observation_id: str) -> str | None:
        """Why a row ended unaudited (``UNAUDITED_REASONS``), or None for a row that is not."""
        row = self.db.execute("SELECT audit, unaudited_reason FROM exploration_entitlements "
                              "WHERE observation_id=? AND order_id=?", (observation_id, self.order)).fetchone()
        if row is None or row[0] != "unaudited":
            return None
        return row[1] or UNAUDITED_HORIZON  # rows written before the reason existed set the horizon

    def finalize_window(self, window: int, *, environment: str, aborted: bool = False,
                        trained_prompts=(), unaudited_reason: str = UNAUDITED_HORIZON) -> list[str]:
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
        sanction) and is returned too. Its audit is untouched: if it was drawn and is left
        unaudited it sets its hotkey's horizon like any other row.
        Applied on every call, so a training observation recorded after the env was finalized still
        voids the row as long as the caller finalizes again before reading ``payable``.

        ``unaudited_reason`` (R25) is the reason of the rows this call turns ``unaudited``.
        ``validator_lost`` sets no horizon, neither for those rows nor (below) for rows that were
        marked ``validator_lost`` earlier; only ``unaudited`` rows (audits could run) set it.
        """
        if unaudited_reason not in UNAUDITED_REASONS:
            raise ValueError("unknown unaudited reason")
        scope = (self.order, window, environment)
        for prompt in sorted({int(p) for p in trained_prompts}):
            self.db.execute("UPDATE exploration_entitlements SET status='trained' WHERE order_id=? AND window=? "
                            "AND environment=? AND prompt_idx=? AND status='reserved'", (*scope, prompt))
        if not self.is_finalized(window, environment=environment):
            horizon_open = "AND (audit='queued' OR (audit='unaudited' AND COALESCE(unaudited_reason, 'unaudited')<>'validator_lost'))"
            if unaudited_reason == UNAUDITED_VALIDATOR_LOST:
                horizon_open = "AND (audit='unaudited' AND COALESCE(unaudited_reason, 'unaudited')<>'validator_lost')"
            horizons = self.db.execute(
                "SELECT hotkey, MIN(draw_round) FROM exploration_entitlements WHERE order_id=? AND window=? "
                f"AND environment=? AND drawn=1 {horizon_open} "
                "AND status IN ('reserved','trained') GROUP BY hotkey", scope).fetchall()
            self.db.execute("UPDATE exploration_entitlements SET audit='unaudited', unaudited_reason=? "
                            "WHERE order_id=? AND window=? AND environment=? AND audit IN ('pending_draw','queued')",
                            (unaudited_reason, *scope))
            for hotkey, horizon in horizons:
                self.db.execute("UPDATE exploration_entitlements SET audit='unaudited', unaudited_reason=? "
                                "WHERE order_id=? AND window=? AND environment=? AND hotkey=? "
                                "AND audit='not_drawn' AND draw_round>=?",
                                (UNAUDITED_HORIZON, *scope, hotkey, horizon))
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
    if log.refusal_reason(result.observation_id) not in RETRYABLE_REFUSALS or refuse is not None:
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
                              passed: bool, now: float, ban_seconds: int,
                              failure_class: str = AUDIT_FAILURE_DETERMINISTIC) -> tuple[str, list[str]]:
    """Record an audit verdict and release the first scan of every forfeited id, atomically.

    Returns ``(kind, forfeited_ids)``, kind being ``passed`` / ``failed`` / ``not_applied``."""
    with _Atomic(_same_store(log, ledger)):
        kind, forfeited = ledger.apply_verdict(observation_id, passed=passed, now=now, ban_seconds=ban_seconds,
                                               failure_class=failure_class)
        for identity in forfeited:
            log.release_first_scan(identity)
        return kind, forfeited


def finalize_exploration(log: "RunObservationLog", ledger: ExplorationLedger, window: int, *,
                         environment: str, aborted: bool = False, extra_trained=(),
                         unaudited_reason: str = UNAUDITED_HORIZON) -> list[str]:
    """Finalize a (window, env) and release the first scan of every unpaid id, atomically.

    ``aborted=True`` also works on an already finalized (window, env): every row still reserved
    becomes unpaid and gives its first scan back (see ``ExplorationLedger.finalize_window``).

    R17 is applied here, in the same transaction and before the audit horizon: the entitlements
    whose prompt has a training-lane observation in this window (``log.trained_prompts``) end
    ``trained`` (unpaid, no sanction); their scan is re-seated on the training observation by
    ``log.release_first_scan``. ``extra_trained`` are prompts the caller knows were trained in this
    window (the archive's batch): they join the set, so a training group the log lost cannot let an
    exploration row on the same prompt be paid as well."""
    with _Atomic(_same_store(log, ledger)):
        unaudited = ledger.finalize_window(
            window, environment=environment, aborted=aborted,
            trained_prompts=set(log.trained_prompts(window, environment)) | set(extra_trained),
            unaudited_reason=unaudited_reason)
        for identity in unaudited:
            log.release_first_scan(identity)
        return unaudited
