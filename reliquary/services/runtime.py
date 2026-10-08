"""Opt-in RL service state (service-contract/v2), on the existing window/proof/weight clocks.

``ServiceRuntime`` is the validator-side state of one RL service order. It composes three
modules over ONE SQLite connection (WAL), serialized by ``ServiceRuntime.lock``:

* ``run_log.RunObservationLog``    the run-wide observation log (never reset on adoption);
* ``exploration.ExplorationLedger`` first-scan exploration pay and its sampled audit;
* ``settlement``                    the per-window money (proportional split) and its validation.

What a caller can rely on:

* Window envelope. ``open_window`` freezes, per window: the order hash, the schedule in force
  (and its hash), one pool per ACTIVE env exactly, the slot geometry and the checkpoint. A
  schedule applied later takes effect at the next ``open_window``; an env deactivated mid-window
  stays in the running envelope and its entitlements still settle. Every amount, cap and price
  handed to the ledger comes from that frozen envelope, the same one ``settle_window`` receives.
* Announcement. ``announcement`` is the v2 ``ServicePolicyAnnouncement`` built from the frozen
  envelope plus the window's beacon; the first beacon stored for a window wins, so the policy a
  batcher holds never changes between admission and a deferred proof.
* Observations. One transaction per observation. Exploration goes through
  ``exploration.record_exploration`` only (record + reservation + first-scan release together).
  Pay is keyed on (env, prompt), never on the group id. The audit draw is keyed with the run's
  secret salt (R18): no miner can tell which of its rows are drawn. The audit draw round is the drand round
  of the VALIDATOR-clock arrival + 2. A refusal is a ``ServicePolicyLimit``: nothing was written.
* Audits. ``queued_audits`` lists what to audit (hotkeys past probation first, then by draw
  round) straight from the ledger rows, so it survives a restart. ``record_audit`` applies a
  verdict, releases the forfeited first scans and publishes their settle events atomically.
* Seal. See ``finalize_exploration`` for the contract the batcher owes before settlement.
* Settlement. ``reconcile_archive`` finalizes what is left, prices the ledger's integer
  entitlement counts with ``settle_window``, self-checks the result with
  ``validate_service_archive_v2`` under the protocol slot geometry, and closes the window to
  new observations. Calling it again recomputes from the caller's batch and the same finalized
  entitlement counts (same input, same bytes); a verdict arriving after finalize changes nothing.
* Scans. A prompt with a proven training observation is scanned for the run (aborted windows
  excepted: they trained nothing); exploration on a
  prompt trained in the same window is unpaid (``trained``, no sanction). See ``run_log``.

Only ``service-contract/v2`` runs RL here; ``service-contract/v1`` (dataset mapping / curation)
is refused.
"""
from __future__ import annotations

import contextlib
import json
import logging
import math
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from reliquary.protocol.release_contract import canonical_json_bytes, canonical_sha256
from reliquary.protocol.service_contract import (
    PUBLIC_SEED_POOL, SUPPORTED_V2_CAPABILITIES, ServiceContract, _identifier, _integer, _sha,
)
from reliquary.protocol.service_schedule import ServiceSchedule, initial_schedule
from reliquary.services.admission_policy import service_signal_admits, validate_submission_policy  # noqa: F401
from reliquary.services.exploration import (
    AUDIT_DRAW_ROUND_OFFSET, STATUS_FORFEITED, STATUS_PENDING, STATUS_UNPAID, ExplorationLedger,
    apply_exploration_verdict, exploration_cap, exploration_price, finalize_exploration, record_exploration,
)
from reliquary.services.run_log import STATUS_PROVEN_UNPAID, Observation, RunObservationLog, observation_id
from reliquary.services.settlement import SERVICE_PAYMENT_POLICY_V2, settle_window, validate_service_archive_v2

logger = logging.getLogger(__name__)

SUPPORTED_SERVICE_CAPABILITIES = SUPPORTED_V2_CAPABILITIES
SERVICE_PAYMENT_POLICY = SERVICE_PAYMENT_POLICY_V2
QUALIFICATION_SCHEMA = "service-runtime-qualification/v2"
STATUS_PAID = "exploration_paid"
_EPS = 1e-12
_BEACON = re.compile(r"[0-9a-f]{64}")
# Published proof label of an entitlement, from its audit state in the ledger (R8).
_PROOF = {"passed": "audited", "failed": "failed", "pending_draw": "pending", "queued": "pending",
          "not_drawn": "unproven", "unaudited": "unproven"}
# The service money of an archive. ``service_training_recomputed_delta`` is informational
# (it compares with the caller's own map) and is not part of it.
FROZEN_ARCHIVE_FIELDS = (
    "service_payment_policy", "service_order_sha256", "service_schedule", "service_schedule_sha256",
    "service_pools_by_environment", "service_picks_target", "service_batch_slots",
    "service_training_by_environment", "service_exploration_by_environment",
    "service_scale_by_environment", "rewards_by_hotkey",
)


@dataclass(frozen=True, slots=True)
class AuditOutcome:
    """What ``ServiceRuntime.record_audit`` did with one verdict.

    ``kind``: ``"passed"`` (the group is audited-and-passed), ``"failed"`` (a failure was applied:
    the hotkey is banned and ``forfeited`` holds the ids it lost, their first scans released) or
    ``"not_applied"`` (nothing changed: unknown id, row not awaiting an audit, a verdict that is
    moot after finalize or on a row already unaudited, or SQLite busy -- the caller may retry).
    """
    kind: str
    forfeited: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.kind == "passed"

    @property
    def failed(self) -> bool:
        return self.kind == "failed"

    @property
    def applied(self) -> bool:
        return self.kind != "not_applied"


_MAX_NUMBER = 1e15  # clocks, pools and rewards are far below; beyond it a number is not one of them


def _finite(value) -> bool:
    """True for a finite int/float within ``_MAX_NUMBER``. Never raises: ``math.isfinite`` and
    ``float()`` raise OverflowError on a huge int, which would escape an ``except ValueError``."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return -_MAX_NUMBER <= value <= _MAX_NUMBER
    return isinstance(value, float) and math.isfinite(value) and abs(value) <= _MAX_NUMBER


def _instant(now) -> float:
    """The validator-clock instant of a call (default: now). ValueError when it is not a clock value."""
    value = time.time() if now is None else now
    if not _finite(value):
        raise ValueError("invalid service clock")
    return float(value)


class ServicePolicyLimit(ValueError):
    """A group is outside the ordered policy: it is refused, nothing is recorded, nothing is owed."""


def protocol_slot_geometry() -> tuple[int, int]:
    """``(picks_target, batch_slots)`` of one env in one window, from the protocol constants."""
    from reliquary.constants import B_BATCH, FILL_CLOSED_PICKS_PER_WINDOW
    return int(FILL_CLOSED_PICKS_PER_WINDOW), int(B_BATCH)


def validate_service_archive(record: dict, contract: ServiceContract, *, cap: float,
                             picks_target: int | None = None, batch_slots: int | None = None) -> None:
    """``validate_service_archive_v2`` under the protocol slot geometry unless one is given.

    Kept under its old name for weight replay, which catches ``ValueError`` and abstains.
    """
    picks, slots = protocol_slot_geometry()
    validate_service_archive_v2(record, contract, cap=cap,
                                picks_target=picks if picks_target is None else picks_target,
                                batch_slots=slots if batch_slots is None else batch_slots)


def _drand_round_at(instant: float) -> int:
    """The drand round current at ``instant`` (same rule as ``corpus_close.current_drand_round``)."""
    from reliquary.infrastructure import drand

    chain = drand.get_current_chain()
    genesis, period = chain.get("genesis_time"), chain.get("period")
    if genesis is None or not period:
        raise RuntimeError("drand chain info is not known yet")
    return int(math.floor((instant - genesis) / period)) + 1


def _verified_beacon(round_id: int):
    from reliquary.infrastructure import drand
    return drand.get_verified_beacon(round_id)


def _validate_qualification(qualification: dict, contract: ServiceContract) -> None:
    from reliquary.constants import M_ROLLOUTS
    if (not isinstance(qualification, dict) or qualification.get("schema") != QUALIFICATION_SCHEMA
            or qualification.get("qualified") is not True):
        raise ValueError("runtime qualification is required")
    if qualification.get("contract_sha256") != contract.sha256:
        raise ValueError("runtime qualification contract mismatch")
    _identifier(qualification.get("qualification_id"), "qualification_id")
    if qualification.get("group_size") != M_ROLLOUTS:
        raise ValueError("runtime qualification group size differs from the active profile")
    for name, env in contract.environments.items():
        if env["sampling"]["kind"] == PUBLIC_SEED_POOL and env["sampling"]["group_size"] != M_ROLLOUTS:
            raise ValueError(f"runtime qualification group size differs from the seed pool of {name}")
    try:
        _sha(qualification.get("forced_seed_report_sha256"), "forced_seed_report_sha256")
    except ValueError as exc:
        raise ValueError(f"runtime qualification needs its forced-seed report: {exc}") from exc


class ServiceRuntime:
    """One qualified controller per task/run, over one SQLite file."""

    def __init__(self, path: str | Path, contract: ServiceContract, qualification: dict, *,
                 now: float | None = None, drand_round_at: Callable[[float], int] | None = None):
        if not isinstance(contract, ServiceContract) or contract.version != 2:
            raise ValueError("the RL service runtime needs service-contract/v2; "
                             "service-contract/v1 adaptive_training is refused")
        contract.require_capabilities(set(SUPPORTED_V2_CAPABILITIES))
        _validate_qualification(qualification, contract)
        self.contract = self.order_contract = contract
        self.qualification = json.loads(canonical_json_bytes(qualification))
        self._round_at = drand_round_at or _drand_round_at
        self.lock = threading.RLock()
        instant = _instant(now)
        self.db = sqlite3.connect(path, timeout=30, check_same_thread=False)
        try:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            if self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='service_contexts'").fetchone():
                raise ValueError("existing service task/run journal belongs to another order "
                                 "(v1 runtime); use a new run")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS service_orders(id TEXT PRIMARY KEY, started REAL NOT NULL, clock REAL NOT NULL, groups INTEGER NOT NULL DEFAULT 0, tokens INTEGER NOT NULL DEFAULT 0, contract TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS service_checkpoints(order_id TEXT NOT NULL, revision TEXT NOT NULL, checkpoint_n INTEGER NOT NULL, repo TEXT NOT NULL, sha256 TEXT NOT NULL, seq INTEGER NOT NULL, PRIMARY KEY(order_id, revision));
                CREATE TABLE IF NOT EXISTS service_schedules(order_id TEXT NOT NULL, revision INTEGER NOT NULL, request_id TEXT NOT NULL, applied_window INTEGER NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(order_id, revision), UNIQUE(order_id, request_id));
                CREATE TABLE IF NOT EXISTS service_windows(order_id TEXT NOT NULL, window INTEGER NOT NULL, envelope TEXT NOT NULL, opened_at REAL, PRIMARY KEY(order_id, window));
                CREATE TABLE IF NOT EXISTS service_pools(order_id TEXT NOT NULL, window INTEGER NOT NULL, randomness TEXT NOT NULL, PRIMARY KEY(order_id, window));
                CREATE TABLE IF NOT EXISTS service_draw_beacons(order_id TEXT NOT NULL, round INTEGER NOT NULL, randomness TEXT NOT NULL, PRIMARY KEY(order_id, round));
                CREATE TABLE IF NOT EXISTS service_settled(order_id TEXT NOT NULL, window INTEGER NOT NULL, aborted INTEGER NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(order_id, window));
                CREATE TABLE IF NOT EXISTS service_training_journal(order_id TEXT NOT NULL, key INTEGER NOT NULL, window INTEGER NOT NULL, digest TEXT NOT NULL, groups TEXT NOT NULL, PRIMARY KEY(order_id,key));
                CREATE TABLE IF NOT EXISTS service_training_stride(order_id TEXT PRIMARY KEY, stride INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS service_consumption(order_id TEXT PRIMARY KEY, cursor INTEGER NOT NULL, q TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS service_cooldown_advice(environment TEXT PRIMARY KEY, window INTEGER NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS service_schedule_requests(order_id TEXT NOT NULL, request_id TEXT NOT NULL, status TEXT NOT NULL, detail TEXT NOT NULL, revision INTEGER NOT NULL, window INTEGER NOT NULL, at REAL NOT NULL, PRIMARY KEY(order_id, request_id));
            """)
            if "opened_at" not in {row[1] for row in self.db.execute("PRAGMA table_info(service_windows)")}:
                self.db.execute("ALTER TABLE service_windows ADD COLUMN opened_at REAL")
            with self.db:
                self.db.execute("BEGIN IMMEDIATE")
                if self.db.execute("SELECT 1 FROM service_orders WHERE id<>? LIMIT 1", (contract.sha256,)).fetchone():
                    raise ValueError("existing service task/run journal belongs to another order; use a new run")
                self.db.execute("INSERT OR IGNORE INTO service_orders(id,started,clock,contract) VALUES(?,?,?,?)",
                                (contract.sha256, instant, instant, contract.canonical.decode()))
                self.db.execute("UPDATE service_orders SET clock=MAX(clock,?) WHERE id=?", (instant, contract.sha256))
                self.db.execute("INSERT OR IGNORE INTO service_schedules VALUES(?,0,'initial',0,?)",
                                (contract.sha256, initial_schedule(contract).canonical.decode()))
            # Boot only, outside any transaction: both constructors refuse an open one.
            self.log = RunObservationLog(self.db, order_sha256=contract.sha256,
                                         sigma_min_bps=contract.to_dict()["scoring"]["sigma_min_bps"])
            self.ledger = ExplorationLedger(self.db, order_sha256=contract.sha256)
            self.db.commit()
        except BaseException:
            self.db.close()
            raise

    def close(self):
        with self.lock:
            self.db.close()

    @contextlib.contextmanager
    def _txn(self):
        """One write transaction on the shared connection, under the runtime lock."""
        with self.lock:
            if self.db.in_transaction:
                raise RuntimeError("service runtime transaction is already open")
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self.db.rollback()
                raise
            else:
                self.db.commit()

    # --- trainer journal and order clock (unchanged) ---
    def record_training_journal(self, key: int, data: bytes, *, is_tombstone: bool, batches: dict, stride: int):
        """A telemetry projection after the existing payload queue's durable commit."""
        import hashlib
        _integer(key, "journal key", 0)
        _integer(stride, "journal stride", 1)
        identities = [] if is_tombstone else sorted({f"{env}:{group.prompt_idx}" for env, groups in batches.items() for group in groups})
        row = (self.order_contract.sha256, key, key // stride, hashlib.sha256(data).hexdigest(), canonical_json_bytes(identities).decode())
        with self.lock, self.db:
            self.db.execute("INSERT OR IGNORE INTO service_training_stride VALUES(?,?)", (self.order_contract.sha256, stride))
            if self.db.execute("SELECT stride FROM service_training_stride WHERE order_id=?", (self.order_contract.sha256,)).fetchone()[0] != stride:
                raise ValueError("service trainer journal stride changed")
            self.db.execute("INSERT OR IGNORE INTO service_training_journal VALUES(?,?,?,?,?)", row)
            previous = self.db.execute("SELECT window,digest,groups FROM service_training_journal WHERE order_id=? AND key=?", row[:2]).fetchone()
            if previous != row[2:]:
                raise ValueError("service consumption projection conflicts with its journal key")

    def active(self, *, now: float | None = None) -> bool:
        instant = _instant(now)
        with self.lock, self.db:
            start, previous = self.db.execute("SELECT started,clock FROM service_orders WHERE id=?", (self.order_contract.sha256,)).fetchone()
            instant = max(instant, previous)
            self.db.execute("UPDATE service_orders SET clock=? WHERE id=?", (instant, self.order_contract.sha256))
            limits = self.order_contract.to_dict()["limits"]
            groups, tokens = self.db.execute("SELECT groups,tokens FROM service_orders WHERE id=?", (self.order_contract.sha256,)).fetchone()
            return instant < start + limits["deadline_seconds"] and groups < limits["max_groups"] and tokens < limits["max_tokens"]

    def record_consumption(self, cursor: int) -> dict[str, float]:
        """Only an adopted trainer cursor turns enqueued groups into measured Q, per env.

        Q is the number of distinct prompts the trainer consumed per window, smoothed (EMA) with
        the order's ``smoothing_bps``. Missing journal facts yield ``{}``: never invented throughput.
        """
        _integer(cursor, "consumed trainer cursor", 0)
        initial = self.qualification.get("initial_journal_cursor", -1)
        if type(initial) is not int or initial < -1:
            raise ValueError("invalid qualified initial journal cursor")
        order = self.contract.sha256
        alpha = self.contract.advice_policy["smoothing_bps"] / 10000
        with self._txn():
            state = self.db.execute("SELECT cursor,q FROM service_consumption WHERE order_id=?", (order,)).fetchone()
            previous, q = (state[0], json.loads(state[1])) if state is not None else (initial, {})
            if cursor < previous:
                raise ValueError("trainer consumption cursor moved backwards")
            if cursor == previous:
                return q
            config = self.db.execute("SELECT stride FROM service_training_stride WHERE order_id=?", (order,)).fetchone()
            if config is None:
                q = {}
            else:
                stride = config[0]
                first_window, last_window = (previous + 1) // stride, (cursor + 1) // stride - 1
                if last_window >= first_window:
                    rows = self.db.execute(
                        "SELECT key,window,groups FROM service_training_journal WHERE order_id=? AND key>=? AND key<? ORDER BY key",
                        (order, first_window * stride, (last_window + 1) * stride)).fetchall()
                    # Only complete consumed windows qualify Q. Missing facts cannot invent throughput.
                    if len(rows) != (last_window - first_window + 1) * stride:
                        q = {}
                    else:
                        windows: dict[int, set] = {}
                        for _, window, payload in rows:
                            windows.setdefault(window, set()).update(json.loads(payload))
                        counts = {env: 0 for env in self.contract.environments}
                        for identities in windows.values():
                            for identity in identities:
                                env = identity.rsplit(":", 1)[0]
                                if env in counts:
                                    counts[env] += 1
                        measured = {env: count / len(windows) for env, count in counts.items()}
                        q = {env: (alpha * value + (1 - alpha) * q[env]) if env in q else value
                             for env, value in sorted(measured.items())}
            self.db.execute("INSERT INTO service_consumption VALUES(?,?,?) ON CONFLICT(order_id) DO UPDATE SET "
                            "cursor=excluded.cursor,q=excluded.q", (order, cursor, canonical_json_bytes(q).decode()))
            return q

    def measured_consumption(self) -> dict[str, float]:
        with self.lock:
            row = self.db.execute("SELECT q FROM service_consumption WHERE order_id=?", (self.contract.sha256,)).fetchone()
            return json.loads(row[0]) if row is not None else {}

    # --- schedule (decisions E, G; R6) ---
    def _schedule(self) -> ServiceSchedule:
        payload = self.db.execute("SELECT payload FROM service_schedules WHERE order_id=? ORDER BY revision DESC LIMIT 1",
                                  (self.contract.sha256,)).fetchone()[0]
        return ServiceSchedule.from_dict(json.loads(payload), self.contract)

    @property
    def schedule(self) -> ServiceSchedule:
        """The latest applied schedule: the one the NEXT ``open_window`` will freeze."""
        with self.lock:
            return self._schedule()

    def window_schedule(self, window: int) -> ServiceSchedule:
        """The schedule frozen in ``window``'s envelope, or the latest one if it is not open yet.

        A caller rebuilding a window after a restart derives its pools from this one.
        """
        with self.lock:
            row = self.db.execute("SELECT envelope FROM service_windows WHERE order_id=? AND window=?",
                                  (self.contract.sha256, window)).fetchone()
            if row is None:
                return self._schedule()
            return ServiceSchedule.from_dict(json.loads(row[0])["schedule"], self.contract)

    def apply_schedule(self, schedule: ServiceSchedule, *, request_id: str, window: int) -> bool:
        """Append the next schedule revision. It never touches an open window's envelope (R6).

        ``window`` is the window during which the request was applied (recorded for the operator);
        the revision takes effect at the next ``open_window``. Returns False for a request id
        already applied with the same schedule.
        """
        if not isinstance(schedule, ServiceSchedule):
            raise ValueError("a ServiceSchedule is required")
        schedule = ServiceSchedule.from_dict(schedule.to_dict(), self.contract)  # same order, valid
        _identifier(request_id, "schedule request_id")
        _integer(window, "window", 0)
        with self._txn():
            return self._insert_schedule(schedule, request_id, window)

    def _insert_schedule(self, schedule: ServiceSchedule, request_id: str, window: int) -> bool:
        """``apply_schedule``'s body; the caller holds the write transaction."""
        payload = schedule.canonical.decode()
        known = self.db.execute("SELECT payload FROM service_schedules WHERE order_id=? AND request_id=?",
                                (self.contract.sha256, request_id)).fetchone()
        if known is not None:
            if known[0] != payload:
                raise ValueError("schedule request id was already used for another schedule")
            return False
        if schedule.revision != self._schedule().revision + 1:
            raise ValueError("schedule revisions must increase by one")
        self.db.execute("INSERT INTO service_schedules VALUES(?,?,?,?,?)",
                        (self.contract.sha256, schedule.revision, request_id, window, payload))
        return True

    def apply_pending_schedule_request(self, store, *, window: int, now: float | None = None,
                                       installed_version=None) -> ServiceSchedule:
        """Apply at most one new operator request from ``store`` and return the schedule ``window`` uses.

        The request file is not trusted: it is validated here again (``schedule.check_request``:
        schema, order, revision == current + 1, declared envs, shares, cooldown bounds, R7's
        install/version check for each env that becomes active). A refused request is recorded
        (SQLite and the store's status file) with its reason and changes nothing; this method never
        raises for a bad, unreadable or replayed request, nor for a store I/O failure (logged).
        An accepted one appends the next schedule revision and its request row in ONE transaction,
        so a restart neither applies it twice nor loses it. R6: the envelope of an already open
        window is frozen, so the answer for such a window stays its frozen schedule and the change
        lands at the next window that opens.
        """
        _integer(window, "window", 0)
        try:
            self._consume_schedule_request(store, window, _instant(now), installed_version)
        except Exception:  # an operator mistake or a broken disk must never stop the validator loop
            logger.exception("schedule request could not be processed (window %d)", window)
        return self.window_schedule(window)

    def _consume_schedule_request(self, store, window: int, at: float, installed_version) -> None:
        from reliquary.services.schedule import check_request
        order = self.contract.sha256
        current = self.schedule
        try:
            request = store.take()
        except (ValueError, OSError) as exc:
            detail = f"unreadable schedule request: {exc}"
            status = None
            with contextlib.suppress(ValueError, OSError):
                status = store.status()
            if not (status and status.get("request_id") == "invalid" and status.get("detail") == detail):
                store.report("invalid", status="refused", detail=detail, revision=current.revision)
            return
        if request is None:
            return
        request_id = request.get("request_id") if isinstance(request, dict) else None
        try:
            _identifier(request_id, "request_id")
        except ValueError as exc:
            detail = f"invalid schedule request: {exc}"
            status = None
            with contextlib.suppress(ValueError, OSError):
                status = store.status()
            if not (status and status.get("request_id") == "invalid" and status.get("detail") == detail):
                store.report("invalid", status="refused", detail=detail, revision=current.revision)
            return
        with self.lock:
            known = self.db.execute("SELECT status, detail, revision FROM service_schedule_requests "
                                    "WHERE order_id=? AND request_id=?", (order, request_id)).fetchone()
            if known is None and self.db.execute("SELECT revision FROM service_schedules WHERE order_id=? "
                                                 "AND request_id=?", (order, request_id)).fetchone():
                known = ("applied", "applied earlier", current.revision)
        if known is not None:
            # Same request seen again (restart, or the file is still there): never apply twice.
            status = None
            with contextlib.suppress(ValueError, OSError):
                status = store.status()
            if not status or status.get("request_id") != request_id:
                store.report(request_id, status=known[0], detail=known[1], revision=known[2])
            return
        try:
            changed = check_request(request, contract=self.contract, current=current,
                                    installed_version=installed_version)
            status, detail = "applied", (f"applied during window {window}; it takes effect at the next "
                                         f"window that opens (open windows keep their frozen schedule)")
        except ValueError as exc:
            changed, status, detail = None, "refused", str(exc)
        except Exception as exc:  # defensive: a request must never be half applied
            changed, status, detail = None, "refused", f"internal error while checking the request: {type(exc).__name__}"
        with self._txn():
            if changed is not None:
                try:
                    self._insert_schedule(changed, request_id, window)
                except ValueError as exc:  # raced with another applier
                    changed, status, detail = None, "refused", str(exc)
            revision = changed.revision if changed is not None else self._schedule().revision
            self.db.execute("INSERT INTO service_schedule_requests VALUES(?,?,?,?,?,?,?)",
                            (order, request_id, status, detail, revision, window, at))
        store.report(request_id, status=status, detail=detail, revision=revision)

    # --- cooldown advice (decision E): recommendation only ---
    def refresh_cooldown_advice(self, *, window: int, populations: dict[str, int]) -> dict[str, dict]:
        """Recompute and store the cooldown recommendation of every env of the order.

        Informational: it never touches the schedule, admission or settlement, and an advisor
        failure (``ValueError``, ``OverflowError``, a bad population...) is stored as an error
        entry (keeping the previous smoothed state), never raised.
        """
        from reliquary.services.cooldown_advisor import NOTE, recommend_cooldown
        _integer(window, "window", 0)
        advice, consumption = {}, self.measured_consumption()
        for environment in self.contract.environments:
            previous = None
            try:
                with self.lock:
                    first, in_zone = self.log.first_scan_stats(environment)
                    row = self.db.execute("SELECT payload FROM service_cooldown_advice WHERE environment=?",
                                          (environment,)).fetchone()
                previous = json.loads(row[0]) if row else None
                result = recommend_cooldown(policy=self.contract.advice_policy,
                                            population=int((populations or {}).get(environment, 0)),
                                            first_scans=first, in_zone_first=in_zone,
                                            consumption=float(consumption.get(environment, 0.0)),
                                            previous=previous)
            except Exception as exc:  # the advisor must never close admission
                kept = previous if isinstance(previous, dict) else {}
                result = {"status": "error", "reasons": [type(exc).__name__], "note": NOTE,
                          "ema_windows": kept.get("ema_windows"),
                          "recommended_windows": kept.get("recommended_windows")}
            result["current_windows"] = self.schedule.cooldown_windows(environment)
            advice[environment] = result
            with self.lock, self.db:
                self.db.execute("INSERT INTO service_cooldown_advice VALUES(?,?,?) ON CONFLICT(environment) "
                                "DO UPDATE SET window=excluded.window, payload=excluded.payload",
                                (environment, window, canonical_json_bytes(result).decode()))
        return advice

    def cooldown_advice(self) -> dict[str, dict]:
        with self.lock:
            return {env: json.loads(payload) for env, payload in
                    self.db.execute("SELECT environment, payload FROM service_cooldown_advice")}

    # --- checkpoints ---
    def _checkpoint(self) -> dict:
        row = self.db.execute("SELECT checkpoint_n, repo, revision, sha256 FROM service_checkpoints WHERE order_id=? "
                              "ORDER BY seq DESC LIMIT 1", (self.contract.sha256,)).fetchone()
        if row is None:
            raise ValueError("no service checkpoint has been adopted")
        return {"checkpoint_n": row[0], "repo": row[1], "revision": row[2], "sha256": row[3]}

    @property
    def checkpoint(self) -> dict:
        with self.lock:
            return self._checkpoint()

    def adopt(self, *, checkpoint_n: int, repo: str, revision: str, sha256: str) -> dict:
        """Make ``revision`` the current checkpoint of the lineage. Observations are not reset."""
        _integer(checkpoint_n, "checkpoint_n", 0)
        if repo != self.contract.to_dict()["checkpoint"]["repo"]:
            raise ValueError("adoption cannot change checkpoint repository")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("checkpoint revision must be an immutable 40-hex commit")
        _sha(sha256, "checkpoint sha256")
        order = self.contract.sha256
        with self._txn():
            known = self.db.execute("SELECT checkpoint_n, sha256 FROM service_checkpoints WHERE order_id=? AND revision=?",
                                    (order, revision)).fetchone()
            if known is not None and known != (checkpoint_n, sha256):
                raise ValueError("checkpoint revision was already adopted with another identity")
            seq = self.db.execute("SELECT COALESCE(MAX(seq),0)+1 FROM service_checkpoints WHERE order_id=?", (order,)).fetchone()[0]
            self.db.execute("INSERT INTO service_checkpoints VALUES(?,?,?,?,?,?) ON CONFLICT(order_id, revision) "
                            "DO UPDATE SET seq=excluded.seq", (order, revision, checkpoint_n, repo, sha256, seq))
            return self._checkpoint()

    def ensure_checkpoint(self, *, checkpoint_n: int, repo: str, revision: str) -> dict:
        """Re-select an adopted revision (restart), or adopt the order's root checkpoint."""
        with self.lock:
            row = self.db.execute("SELECT sha256 FROM service_checkpoints WHERE revision=? AND order_id=?",
                                  (revision, self.contract.sha256)).fetchone()
        if row is not None:
            return self.adopt(checkpoint_n=checkpoint_n, repo=repo, revision=revision, sha256=row[0])
        root = self.contract.to_dict()["checkpoint"]
        if revision != root["revision"] or repo != root["repo"]:
            raise ValueError("active checkpoint has no adopted service lineage entry")
        return self.adopt(checkpoint_n=checkpoint_n, repo=repo, revision=revision, sha256=root["sha256"])

    # --- windows ---
    def open_window(self, window: int, *, pools: dict, picks_target: int, batch_slots: int,
                    now: float | None = None) -> dict:
        """Freeze the envelope of ``window`` and return it.

        ``pools`` is ``{env: absolute emission fraction}`` for exactly the active envs of the
        schedule in force; ``picks_target`` / ``batch_slots`` must be the protocol slot geometry
        (the one archive validation and weight replay use). Opening an already frozen window with
        the same pools and geometry returns the FROZEN envelope (its schedule and checkpoint, not
        the current ones): this is the restart path. Anything else is refused.

        Windows open in increasing order: a window lower than the highest frozen one is refused
        unless it is itself frozen (restart), so an old window can never be born with the latest
        schedule. The opening instant (``now``, validator clock) is stored with the frozen row, first
        one wins: an exploration arrival is never taken as earlier than it (``record_exploration``).
        """
        _integer(window, "window", 0)
        instant = _instant(now)
        if (picks_target, batch_slots) != protocol_slot_geometry() or type(picks_target) is not int \
                or type(batch_slots) is not int:
            raise ValueError("service window slot geometry differs from the protocol's")
        if not isinstance(pools, dict):
            raise ValueError("window pools must be a map of environments")
        given = {}
        for env in sorted(pools):
            value = pools[env]
            if not isinstance(env, str) or type(value) not in (int, float) or not _finite(value) \
                    or not 0 <= value <= 1:
                raise ValueError("invalid service window pool")
            given[env] = float(value)
        order = self.contract.sha256
        with self._txn():
            saved = self.db.execute("SELECT envelope FROM service_windows WHERE order_id=? AND window=?",
                                    (order, window)).fetchone()
            if saved is not None:
                envelope = json.loads(saved[0])
                if envelope["pools"] != given:
                    raise ValueError("service window envelope is already frozen")
                return envelope
            highest = self.db.execute("SELECT MAX(window) FROM service_windows WHERE order_id=?", (order,)).fetchone()[0]
            if highest is not None and window < highest:
                raise ValueError(f"service windows open in increasing order: window {highest} is already frozen")
            schedule = self._schedule()
            if set(given) != set(schedule.active_environments()):
                raise ValueError("window pools must cover exactly the active environments")
            for env, value in given.items():
                if value > schedule.share_bps(env) / 10000 + _EPS:
                    raise ValueError("service window pool exceeds the environment's share")
            envelope = {"order_sha256": order, "schedule": schedule.to_dict(),
                        "schedule_sha256": schedule.sha256, "pools": given,
                        "picks_target": picks_target, "batch_slots": batch_slots,
                        "checkpoint": self._checkpoint()}
            self.db.execute("INSERT INTO service_windows(order_id, window, envelope, opened_at) VALUES(?,?,?,?)",
                            (order, window, canonical_json_bytes(envelope).decode(), instant))
            return envelope

    def _envelope(self, window: int) -> dict:
        row = self.db.execute("SELECT envelope FROM service_windows WHERE order_id=? AND window=?",
                              (self.contract.sha256, window)).fetchone()
        if row is None:
            raise ServicePolicyLimit("service window has no frozen envelope")
        return json.loads(row[0])

    def envelope(self, window: int) -> dict:
        with self.lock:
            return self._envelope(window)

    def _environments(self, window: int, environment: str | None) -> list[str]:
        pools = self._envelope(window)["pools"]
        if environment is None:
            return sorted(pools)
        if environment not in pools:
            raise ServicePolicyLimit("environment is not active in this window")
        return [environment]

    def _settled(self, window: int) -> bool:
        return self.db.execute("SELECT 1 FROM service_settled WHERE order_id=? AND window=?",
                               (self.contract.sha256, window)).fetchone() is not None

    def announcement(self, *, window: int, randomness: str, environment: str | None = None) -> dict:
        """The v2 ``ServicePolicyAnnouncement`` of ``window`` (same for every env of the window).

        Built from the frozen envelope (schedule, checkpoint) and the window's pool beacon:
        ``pool_epoch`` is the window (a public seed pool renews every window, R16) and
        ``randomness`` must be the window's beacon, unknown before the window opens. The first
        beacon stored for a window wins, so the answer is stable for the life of the window,
        across calls and restarts. ``environment``, if given, must be active in the window.
        """
        from reliquary.protocol.submission import ServicePolicyAnnouncement

        _sha(randomness, "pool randomness")
        with self._txn():
            envelope = self._envelope(window)
            if environment is not None and environment not in envelope["pools"]:
                raise ServicePolicyLimit("environment is not active in this window")
            self.db.execute("INSERT OR IGNORE INTO service_pools VALUES(?,?,?)", (self.contract.sha256, window, randomness))
            beacon = self.db.execute("SELECT randomness FROM service_pools WHERE order_id=? AND window=?",
                                     (self.contract.sha256, window)).fetchone()[0]
        if beacon != randomness:
            logger.warning("service window %d keeps its first pool beacon; a later one was ignored", window)
        value = {"contract": self.contract.to_dict(), "schedule": envelope["schedule"],
                 "checkpoint": envelope["checkpoint"],
                 "supported_capabilities": sorted(SUPPORTED_V2_CAPABILITIES),
                 "pool_epoch": window, "pool_randomness": beacon}
        ServicePolicyAnnouncement(**value)  # never hand out a policy the protocol refuses
        return value

    def seed_pool(self, *, environment: str, prompt_idx: int, window: int):
        """The announced public seed pool of (env, prompt, window), or None for a legacy-sampling env."""
        with self.lock:
            return self._seed_pool(self._envelope(window), window, environment, prompt_idx)

    def _seed_pool(self, envelope: dict, window: int, environment: str, prompt_idx: int):
        from reliquary.protocol.seed_pool import SeedPool

        if self.contract.environment(environment)["sampling"]["kind"] != PUBLIC_SEED_POOL:
            return None
        row = self.db.execute("SELECT randomness FROM service_pools WHERE order_id=? AND window=?",
                              (self.contract.sha256, window)).fetchone()
        if row is None:
            raise ServicePolicyLimit("service window has no announced seed pool")
        return SeedPool.from_contract(self.contract, environment=environment, prompt_idx=prompt_idx,
                                      checkpoint_hash=envelope["checkpoint"]["revision"], pool_epoch=window,
                                      randomness=row[0])

    # --- observations (decisions A, B) ---
    def _observation(self, envelope: dict, *, environment, prompt_idx, hotkey, window, rewards, group_id,
                     candidate, token_count, lane, now) -> Observation:
        """Check one group against the frozen window and build its observation. ValueError = refuse."""
        if environment not in envelope["pools"]:
            raise ServicePolicyLimit("environment is not active in this window")
        if self._settled(window):
            raise ServicePolicyLimit("window has already settled")
        policy = self.contract.environment(environment)
        if type(prompt_idx) is not int or not 0 <= prompt_idx < policy["dataset"]["rows"]:
            raise ValueError("prompt index is outside the ordered dataset")
        if not isinstance(hotkey, str) or not 1 <= len(hotkey) <= 256:
            raise ValueError("invalid service hotkey")
        if type(token_count) is not int or not 0 <= token_count < 2**62:
            raise ValueError("invalid service token count")
        if type(rewards) not in (list, tuple):
            raise ValueError("rewards must be the list of graded rollout rewards")
        bps = []
        for reward in rewards:
            if type(reward) not in (int, float) or not _finite(reward) or not 0 <= reward <= 1:
                raise ValueError("rewards must be finite numbers in [0, 1]")
            bps.append(round(reward * 10000))
        _sha(group_id, "group_id")
        pool = self._seed_pool(envelope, window, environment, prompt_idx)
        if pool is None:
            if candidate is not None:
                raise ValueError("this environment has no public seed pool")
        else:
            # The group id IS the selection digest of the pool announced for this window.
            if not isinstance(candidate, dict) or set(candidate) != {"pool_sha256", "seeds"}:
                raise ValueError("a public seed pool group needs its pool selection")
            if candidate["pool_sha256"] != pool.sha256:
                raise ValueError("selection is not bound to the window's announced pool")
            selection = pool.selection(candidate["seeds"])
            if selection.sha256 != group_id:
                raise ValueError("group id is not the digest of the pool selection")
            candidate = {"pool_sha256": pool.sha256, "seeds": list(selection.seeds)}
        checkpoint = envelope["checkpoint"]
        return Observation(environment=environment, dataset_id=policy["dataset"]["id"], prompt_idx=prompt_idx,
                           group_id=group_id, window=window, checkpoint_n=checkpoint["checkpoint_n"],
                           checkpoint_revision=checkpoint["revision"], observed_at=float(now),
                           rewards_bps=tuple(bps), lane=lane, candidate=candidate, hotkey=hotkey,
                           token_count=token_count)

    def _refused(self, exc: BaseException, lane: str, context: dict):
        """R10: a group that does not become an observation is refused loudly, never dropped silently."""
        if isinstance(exc, ServicePolicyLimit):
            logger.warning("service %s group outside the policy (%s): %s", lane, exc, context)
            return exc
        logger.error("service %s observation refused (%s: %s): %s", lane, type(exc).__name__, exc, context)
        return ServicePolicyLimit(f"service observation refused: {exc}")

    def _observation_id(self, obs: Observation) -> str:
        return observation_id(self.contract.sha256, obs, self.log.run_salt)

    def record_training(self, *, environment, prompt_idx, hotkey, window, rewards, group_id, candidate,
                        token_count, now=None) -> dict:
        """Record a PROVEN training group as an observation (it takes the prompt's first scan).

        Returns ``{"observation_id", "first_scan", "inserted"}``. Raises ``ServicePolicyLimit``
        (nothing written) when the order is inactive, the window is unknown / settled, the env is
        not in the window, or the evidence is not an observation.
        """
        context = {"environment": environment, "prompt_idx": prompt_idx, "window": window,
                   "group_id": group_id, "hotkey": hotkey}
        try:
            instant = _instant(now)
            if not self.active(now=instant):  # outside the transaction: active() commits its clock
                raise ServicePolicyLimit("service order is inactive")
            with self._txn():
                obs = self._observation(self._envelope(window), environment=environment, prompt_idx=prompt_idx,
                                        hotkey=hotkey, window=window, rewards=rewards, group_id=group_id,
                                        candidate=candidate, token_count=token_count, lane="training", now=instant)
                context["observation_id"] = self._observation_id(obs)
                result = self.log.record(obs, status="proven", proof="proven")
                if result.inserted:
                    self.db.execute("UPDATE service_orders SET groups=groups+1, tokens=tokens+? WHERE id=?",
                                    (token_count, self.contract.sha256))
        except (ValueError, OverflowError) as exc:  # NotAnObservation, conflicting evidence, a huge number, ...
            refusal = self._refused(exc, "training", context)
            if refusal is exc:
                raise
            raise refusal from exc
        return {"observation_id": result.observation_id, "first_scan": result.first_scan,
                "inserted": result.inserted}

    def exploration_banned(self, hotkey: str, *, now: float | None = None) -> bool:
        with self.lock:
            return self.ledger.banned(hotkey, time.time() if now is None else now)

    def record_exploration(self, *, environment, prompt_idx, hotkey, window, rewards, group_id, candidate,
                           token_count, arrived_at: float | None = None, now=None) -> dict:
        """Record an exploration group and reserve its first-scan pay, in one transaction.

        ``arrived_at`` is the VALIDATOR-clock time the submission arrived (default: now; never a
        miner-supplied value, and never later than now). The audit draw round is the drand round
        of that instant + 2, so its beacon does not exist when the group is committed. The
        arrival is clamped into the window's life: ``max(open time, min(now, arrived_at))``, the
        open time being ``open_window``'s stored instant. A group can therefore never pick a round
        older than the window, and a validator clock stepping back does not stop exploration.

        Returns ``{"observation_id", "inserted", "first_scan", "entitled", "amount", "status",
        "reason", "forced_audit", "draw_round"}``. ``status`` is ``exploration_pending`` when one
        entitlement is reserved (``amount`` is its nominal price, before any proportional scale),
        else ``exploration_unpaid`` with ``reason``: ``already_scanned`` (first arrival on the
        (env, prompt) wins, whatever the group id), ``banned``, ``cap``, ``finalized``,
        ``zero_price``, ``probation_limit``, ``order_inactive``, ``exploration_disabled``,
        ``token_limit`` or ``replay``. ``already_scanned`` also covers a prompt with a training
        observation in the run. An entitlement whose prompt is trained in the same window ends
        unpaid at finalize (``trained``). An unpaid observation is published and never holds the
        prompt's first scan.
        Raises ``ServicePolicyLimit`` (nothing written, logged) when the group is not an
        observation or the window cannot take it.
        """
        context = {"environment": environment, "prompt_idx": prompt_idx, "window": window,
                   "group_id": group_id, "hotkey": hotkey}
        reward = self.contract.reward_policy
        try:
            instant = _instant(now)
            arrival = instant if arrived_at is None else arrived_at
            if type(arrival) not in (int, float) or not _finite(arrival):
                raise ValueError("invalid arrival time")
            arrival = min(float(arrival), instant)
            active = self.active(now=instant)  # outside the transaction: active() commits its clock
            with self._txn():
                envelope = self._envelope(window)
                obs = self._observation(envelope, environment=environment, prompt_idx=prompt_idx, hotkey=hotkey,
                                        window=window, rewards=rewards, group_id=group_id, candidate=candidate,
                                        token_count=token_count, lane="exploration", now=instant)
                context["observation_id"] = self._observation_id(obs)
                opened = self.db.execute("SELECT opened_at FROM service_windows WHERE order_id=? AND window=?",
                                         (self.contract.sha256, window)).fetchone()[0]
                if opened is not None:
                    arrival = max(arrival, opened)
                try:
                    draw_round = int(self._round_at(arrival)) + AUDIT_DRAW_ROUND_OFFSET
                except RuntimeError as exc:  # no drand clock: the audit cannot be drawn, so nothing is owed
                    raise ValueError(f"audit draw round is unknown: {exc}") from exc
                refuse = None
                if not active:
                    refuse = "order_inactive"
                elif self.contract.environment(environment)["exploration"] != 1:
                    refuse = "exploration_disabled"
                elif token_count > reward["max_tokens_per_group"]:
                    refuse = "token_limit"
                # Price and cap come from the frozen envelope pool that settle_window will receive.
                pool = envelope["pools"][environment]
                outcome = record_exploration(
                    self.log, self.ledger, obs,
                    amount=exploration_price(pool, picks_target=envelope["picks_target"],
                                             batch_slots=envelope["batch_slots"], price_bps=reward["price_bps"]),
                    cap=exploration_cap(pool, cap_bps=reward["cap_bps"]),
                    draw_round=draw_round, new_hotkey_audit_groups=reward["new_hotkey_audit_groups"],
                    now=instant, refuse=refuse)
        except (ValueError, OverflowError) as exc:
            refusal = self._refused(exc, "exploration", context)
            if refusal is exc:
                raise
            raise refusal from exc
        held = outcome.entitlement
        return {"observation_id": outcome.observation_id, "inserted": outcome.inserted,
                "first_scan": outcome.first_scan, "entitled": held is not None,
                "amount": held["amount"] if held else 0.0, "status": outcome.status, "reason": outcome.reason,
                "forced_audit": bool(held and held["forced"]),
                "draw_round": held["draw_round"] if held else None}

    # --- audits (decision C, R15) ---
    def pending_draw_rounds(self, window: int, *, environment: str | None = None) -> list[int]:
        """Draw rounds some entitlement of the window still waits for (all envs unless one is named)."""
        with self.lock:
            return sorted({round_id for env in self._environments(window, environment)
                           for round_id in self.ledger.pending_draw_rounds(window, environment=env)})

    def _beacon_randomness(self, round_id: int, value) -> str | None:
        if value is None:
            return None
        if isinstance(value, dict):  # drand.get_verified_beacon: must be the beacon of exactly this round
            if value.get("round") != round_id:
                logger.error("audit beacon for round %d names round %r; ignored", round_id, value.get("round"))
                return None
            value = value.get("randomness")
        if not isinstance(value, str) or _BEACON.fullmatch(value) is None:
            logger.error("audit beacon for round %d is not 32 bytes of lowercase hex; ignored", round_id)
            return None
        return value

    def resolve_draws(self, window: int, *, environment: str | None = None, beacon_for_round=None,
                      now: float | None = None) -> list[str]:
        """Draw the audits whose round has passed; return the ids newly queued for audit.

        ``beacon_for_round(round)`` returns the VERIFIED beacon of exactly that round: either the
        dict of ``drand.get_verified_beacon`` (its ``round`` is checked) or its randomness hex, or
        None when it is not available yet. Default: ``drand.get_verified_beacon``. A round later
        than the validator clock's current round is never asked for, so "latest" can never stand
        in for a future round. The first beacon stored for a round is the one every draw of that
        round uses, in every window and after a restart (table ``service_draw_beacons``).
        Fetching happens outside the lock and outside any transaction.
        """
        instant = _instant(now)
        order = self.contract.sha256
        with self.lock:
            environments = self._environments(window, environment)
            rounds = sorted({round_id for env in environments
                             for round_id in self.ledger.pending_draw_rounds(window, environment=env)})
            known = {round_id for round_id in rounds if self.db.execute(
                "SELECT 1 FROM service_draw_beacons WHERE order_id=? AND round=?", (order, round_id)).fetchone()}
        if not rounds:
            return []
        try:
            current = int(self._round_at(float(instant)))
        except RuntimeError as exc:
            logger.error("service audit draws wait: no drand clock (%s)", exc)
            return []
        fetch = beacon_for_round or _verified_beacon
        fresh: dict[int, str] = {}
        for round_id in rounds:
            if round_id in known or round_id > current:
                continue
            try:
                randomness = self._beacon_randomness(round_id, fetch(round_id))
            except Exception:
                logger.exception("audit beacon for round %d could not be fetched", round_id)
                continue
            if randomness is not None:
                fresh[round_id] = randomness
        selected: list[str] = []
        with self._txn():
            for round_id, randomness in fresh.items():
                self.db.execute("INSERT OR IGNORE INTO service_draw_beacons VALUES(?,?,?)", (order, round_id, randomness))
            stored = {round_id: randomness for round_id, randomness in self.db.execute(
                "SELECT round, randomness FROM service_draw_beacons WHERE order_id=?", (order,))
                if round_id in rounds}
            for env in environments:
                selected.extend(self.ledger.resolve_draws(window, environment=env, beacon_for_round=stored.get,
                                                          audit_bps=self.contract.reward_policy["audit_bps"],
                                                          run_salt=self.log.run_salt))
        return selected

    def queued_audits(self, window: int, *, environment: str | None = None) -> list[dict]:
        """The audits to run, in the order to run them. Rebuilt from the ledger on every call.

        Rows drawn for audit, still entitled, in a (window, env) not yet finalized. Hotkeys past
        probation (``passed_audits >= new_hotkey_audit_groups``) come first, then ascending draw
        round, then reservation order. Each row: ``observation_id``, ``environment``, ``hotkey``,
        ``prompt_idx``, ``draw_round``, ``forced``, ``past_probation``. A row whose prompt was
        trained in the window is listed like any other: R17 voids its pay, not its audit.
        """
        threshold = self.contract.reward_policy["new_hotkey_audit_groups"]
        with self.lock:
            rows, seasoned = [], {}
            for env in self._environments(window, environment):
                if self.ledger.is_finalized(window, environment=env):
                    continue
                for position, row in enumerate(self.ledger.rows(window, environment=env)):
                    if row["audit"] != "queued" or row["status"] != "reserved":
                        continue
                    hotkey = row["hotkey"]
                    if hotkey not in seasoned:
                        seasoned[hotkey] = self.ledger.passed_audits(hotkey) >= threshold
                    rows.append(((not seasoned[hotkey], row["draw_round"], env, position), {
                        "observation_id": row["observation_id"], "environment": env, "hotkey": hotkey,
                        "prompt_idx": row["prompt_idx"], "draw_round": row["draw_round"],
                        "forced": bool(row["forced"]), "past_probation": seasoned[hotkey]}))
        return [row for _, row in sorted(rows, key=lambda item: item[0])]

    def exploration_backlog(self, window: int, *, environment: str | None = None) -> dict[str, int]:
        """What the seal still waits for: ``{"pending_draw", "queued", "queued_past_probation"}``."""
        with self.lock:
            pending = 0
            for env in self._environments(window, environment):
                if not self.ledger.is_finalized(window, environment=env):
                    pending += sum(row["audit"] == "pending_draw" and row["status"] == "reserved"
                                   for row in self.ledger.rows(window, environment=env))
        queued = self.queued_audits(window, environment=environment)
        return {"pending_draw": pending, "queued": len(queued),
                "queued_past_probation": sum(row["past_probation"] for row in queued)}

    def _entitlement_row(self, identity: str) -> tuple | None:
        return self.db.execute("SELECT window, environment, audit, status FROM exploration_entitlements "
                               "WHERE observation_id=? AND order_id=?", (identity, self.contract.sha256)).fetchone()

    def record_audit(self, identity: str, *, passed: bool, now: float | None = None) -> AuditOutcome:
        """Apply one audit verdict; return an ``AuditOutcome`` (passed / failed + forfeited ids / not_applied).

        One transaction: the verdict, the release of every forfeited first scan and the public
        settle events (``exploration_pending`` / ``audited`` on a pass of a row still entitled;
        ``exploration_forfeited`` for every id forfeited in an env not yet finalized -- also when the
        failure itself is late, on a row of a finalized env --, ``failed`` for the audited one). No paid
        status is published here: pay is only known at finalize. A verdict on a (window, env)
        already finalized changes none of its rows and publishes nothing for them; a late failure on
        a drawn group still bans and forfeits the hotkey's entitlements in the envs of that window
        not yet finalized (R3). A pass on a row already marked unaudited is ignored.

        Never raises on a verdict it cannot apply (unknown id, row not awaiting an audit, SQLite
        busy): it logs at error level with the observation id and returns ``not_applied``.
        """
        if type(passed) is not bool:
            raise ValueError("an audit verdict is a boolean")
        try:
            instant = _instant(now)
            with self._txn():
                kind, forfeited = apply_exploration_verdict(
                    self.log, self.ledger, identity, passed=passed, now=instant,
                    ban_seconds=self.contract.reward_policy["ban_seconds"])
                if kind == "passed":
                    entry = self._entitlement_row(identity)
                    # Only a row still entitled is "pending": never after its forfeit (M1).
                    if entry[3] == "reserved" and not self.ledger.is_finalized(entry[0], environment=entry[1]):
                        self.log.settle(identity, status=STATUS_PENDING, proof="audited", at=instant)
                for lost in forfeited:
                    entry = self._entitlement_row(lost)
                    if not self.ledger.is_finalized(entry[0], environment=entry[1]):
                        self.log.settle(lost, status=STATUS_FORFEITED, proof=_PROOF[entry[2]], at=instant)
        except (ValueError, OverflowError, sqlite3.OperationalError) as exc:
            logger.error("exploration audit verdict not applied for observation %s (passed=%s): %s: %s",
                         identity, passed, type(exc).__name__, exc)
            return AuditOutcome("not_applied")
        return AuditOutcome(kind, tuple(forfeited))

    def _finalize_env(self, window: int, environment: str, *, aborted: bool, at: float) -> list[str]:
        """THE one place a (window, env) is finalized. Runs inside the caller's transaction.

        Every id whose first scan is released gets its settle event (``exploration_unpaid``, with
        reason ``trained`` for a row whose prompt was trained in this window (R17), or
        ``exploration_forfeited`` for a row a failed audit already forfeited). The release itself is
        ``RunObservationLog.release_first_scan``, called by the ledger entry point.
        """
        # The single call site of the ledger's finalize. ``aborted`` is the one transition allowed on a
        # (window, env) the batcher already finalized before it knew the window was aborted.
        released = list(finalize_exploration(self.log, self.ledger, window, environment=environment,
                                             aborted=aborted))
        rows = {row["observation_id"]: row for row in self.ledger.rows(window, environment=environment)}
        for identity in released:
            row = rows[identity]  # a forfeited row keeps its label (R8), whatever its audit became
            self.log.settle(identity, status=STATUS_FORFEITED if row["status"] == "forfeited" else STATUS_UNPAID,
                            proof=_PROOF[row["audit"]], at=at,
                            reason="trained" if row["status"] == "trained" else None)
        return released

    def finalize_exploration(self, window: int, *, environment: str, now: float | None = None) -> list[str]:
        """Freeze the exploration of one (window, env); return the ids left unpaid by it.

        SEAL CONTRACT (the caller's, before calling this): stop admitting exploration groups for
        the env; keep calling ``resolve_draws`` until ``pending_draw_rounds`` is empty or 2 drand
        rounds have passed since the last admission; run the audits of ``queued_audits`` in its
        order and report each with ``record_audit`` -- EVERY queued row of a hotkey past probation
        must be audited, rows of hotkeys in probation as far as a bounded wait allows; then call
        this. ``exploration_backlog`` tells what is left.

        What is still waiting for its draw or its audit becomes ``unaudited``: unpaid, never
        sanctioned, and a hotkey's not-drawn rows at or after its first drawn-unaudited round are
        unpaid too (per-hotkey audit horizon, R15). Before that, in the same transaction, every
        entitlement whose prompt has a training observation in this window becomes unpaid with
        reason ``trained`` (R17: no sanction; its audit still counts for the horizon). Their first
        scans are released and their
        ``exploration_unpaid`` settle events published, in this transaction. Idempotent. After
        it the rows never change: a later verdict pays nothing and takes nothing back.
        ``reconcile_archive`` finalizes whatever env was not, so a restart cannot pay an unaudited row.
        """
        instant = _instant(now)
        with self._txn():
            self._environments(window, environment)
            return self._finalize_env(window, environment, aborted=False, at=instant)

    # --- settlement (decision B) ---
    def reconcile_archive(self, archive: dict, *, aborted: bool = False, now: float | None = None) -> dict:
        """Stamp the service money on a window archive, self-check it, and freeze it.

        First call for a window, in one transaction: finalize every env of the frozen envelope
        (a no-op for those the caller finalized after draining audits), read the ledger's integer
        entitlement counts, ``settle_window`` (proportional split per env), validate the result
        with ``validate_service_archive_v2`` under the protocol slot geometry, mark the window
        settled (no observation is taken after that) and publish the final settle events
        (``trained`` / ``proven_unpaid``, ``exploration_paid`` / ``exploration_unpaid`` /
        ``exploration_forfeited``).

        The answer is a pure recomputation from (the caller's ``batch``, the ledger's finalized
        entitlement counts, the disposition): same input, same bytes out, at any time and after a
        restart; it always passes the self-check. After the first call the exploration counts only
        move on an abort, so another batch gives recomputed training rows and scale with the SAME
        exploration counts, and a late audit verdict changes nothing. Dispositions:

        * not aborted, then called aborted (restart recovery rebuilt the window empty): allowed.
          The exploration rows become unpaid, their first scans are released, the window's
          training observations stop counting as scans, the archive pays no exploration.
        * aborted, then called not aborted: exploration stays unpaid (its counts are empty); the
          result is a valid non-aborted archive with zero exploration.

        Neither raises. A call whose disposition or batch digest differs from the previous call
        of that window is logged at error level (table ``service_settled`` keeps both), and the
        public settle events follow the latest answer (nothing is published when nothing changed).
        An aborted window (``aborted=True`` or an archive whose ``window_status`` is ``aborted``)
        pays no exploration. Every ``batch`` row must carry ``env_name`` (an env of the envelope),
        a string ``hotkey`` and an integer ``prompt_idx``; anything else raises
        (``SettlementError``) and stores nothing.
        """
        if type(aborted) is not bool or not isinstance(archive, dict):
            raise ValueError("invalid service settlement disposition")
        window = _integer(archive.get("window_start"), "window", 0)
        aborted = aborted or archive.get("window_status") == "aborted"
        instant = _instant(now)
        picks, slots = protocol_slot_geometry()
        order = self.contract.sha256
        with self._txn():
            envelope = self._envelope(window)
            settled = self.db.execute("SELECT aborted, payload FROM service_settled WHERE order_id=? AND window=?",
                                      (order, window)).fetchone()
            if settled is None or aborted:
                for env in sorted(envelope["pools"]):
                    if not self.ledger.is_finalized(window, environment=env):
                        waiting = sum(row["audit"] in ("pending_draw", "queued")
                                      for row in self.ledger.rows(window, environment=env))
                        if waiting:
                            logger.warning("service window %d env %s settles with %d exploration group(s) "
                                           "never audited (unpaid)", window, env, waiting)
                    # Also for an env the caller finalized: abort, and R17 for a late training group.
                    self._finalize_env(window, env, aborted=aborted, at=instant)
            exploration = {}
            for env in sorted(envelope["pools"]):
                counts = self.ledger.payable(window, environment=env)
                if counts:
                    exploration[env] = counts
            result = settle_window(archive=archive, envelope=envelope, contract=self.contract,
                                   exploration=exploration, aborted=aborted)
            # Informational, not money: no field of FROZEN_ARCHIVE_FIELDS, ignored by validation.
            result["service_cooldown_advice"] = self.cooldown_advice()
            # Lane consistency under the protocol geometry; the task-cap bound is weight replay's.
            validate_service_archive_v2(result, self.contract, cap=1.0, picks_target=picks, batch_slots=slots)
            digest = canonical_sha256(sorted(
                [str(row.get("env_name")), str(row.get("hotkey")), str(row.get("prompt_idx"))]
                for row in archive.get("batch") or []))
            if settled is not None:
                previous = json.loads(settled[1]).get("batch_sha256")
                if bool(settled[0]) != aborted:
                    logger.error("service window %d was settled as %s and is now settled as %s", window,
                                 "aborted" if settled[0] else "not aborted", "aborted" if aborted else "not aborted")
                if previous != digest:
                    logger.error("service window %d is settled again with another batch (digest %s, was %s)",
                                 window, digest, previous)
            self.db.execute("INSERT INTO service_settled VALUES(?,?,?,?) ON CONFLICT(order_id, window) DO UPDATE SET "
                            "aborted=excluded.aborted, payload=excluded.payload",
                            (order, window, int(aborted), canonical_json_bytes({"batch_sha256": digest}).decode()))
            self.log.set_window_aborted(window, aborted)
            self._emit_settle_events(window, envelope, archive, aborted, instant)
        return result

    def _emit_settle_events(self, window: int, envelope: dict, archive: dict, aborted: bool, at: float) -> None:
        paid: dict[tuple, int] = {}
        if not aborted:
            for row in archive.get("batch") or []:
                key = (row.get("env_name"), row.get("prompt_idx"), row.get("hotkey"))
                paid[key] = paid.get(key, 0) + 1
        ledger = {row["observation_id"]: row for env in envelope["pools"]
                  for row in self.ledger.rows(window, environment=env)}
        for row in self.log.window_observations(window):
            if row["lane"] == "training":
                key = (row["environment"], row["prompt_idx"], row["hotkey"])
                trained = paid.get(key, 0) > 0
                if trained:
                    paid[key] -= 1
                # proven_unpaid: proven, not paid (left out of the batch, or the window aborted).
                self.log.settle(row["id"], status="trained" if trained else STATUS_PROVEN_UNPAID, proof="proven", at=at)
            elif row["id"] in ledger:
                entry = ledger[row["id"]]
                if entry["status"] == "forfeited":
                    status = STATUS_FORFEITED
                elif aborted or entry["status"] != "reserved" or entry["audit"] == "unaudited":
                    status = STATUS_UNPAID
                else:
                    status = STATUS_PAID
                self.log.settle(row["id"], status=status, proof=_PROOF[entry["audit"]], at=at,
                                reason="trained" if entry["status"] == "trained" else None)
            # An exploration observation without an entitlement was published unpaid, with its reason.

    # --- publication / admin reads ---
    def events(self, *, after: int = 0, limit: int = 1000) -> list[tuple[int, dict]]:
        with self.lock:
            return self.log.events(after=after, limit=limit)

    def admin_events(self, *, after: int = 0, limit: int = 1000) -> list[tuple[int, dict]]:
        with self.lock:
            return self.log.admin_events(after=after, limit=limit)

    # --- v1 entry points, removed. Callers not rewired yet fail closed with a clear message. ---
    @staticmethod
    def _unwired(name: str, task: str):
        raise NotImplementedError(f"ServiceRuntime.{name} was removed with the v1 service runtime; "
                                  f"this caller is wired in {task}")

    @property
    def view(self):
        self._unwired("view", "Task 10 (validator), Task 12 (batcher) and Task 13 (server)")

    @property
    def row_ids(self):
        self._unwired("row_ids", "Task 10 (validator) and Task 12 (batcher)")

    def prepare_view(self, *args, **kwargs):
        self._unwired("prepare_view", "Task 10 (validator)")

    def training_pool(self, *args, **kwargs):
        self._unwired("training_pool", "Task 10 (validator)")

    def record_verified(self, *args, **kwargs):
        self._unwired("record_verified", "Task 12 (batcher)")

    def snapshot(self, *args, **kwargs):
        self._unwired("snapshot", "Task 13 (server)")
