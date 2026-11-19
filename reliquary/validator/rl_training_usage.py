"""Final training usage from a drained book and a supervisor-owned stopped runtime.

This is a customer billing metric, not a claim about miner USD cost or chain settlement.
No live polling observation, process PID alone or registry retirement can mint final usage.
"""
from __future__ import annotations

import errno
import json
import os
import re
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import asdict
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from fractions import Fraction
from pathlib import Path

from reliquary.protocol.release_contract import canonical_json_bytes, canonical_sha256
from reliquary.validator.rl_training_orders import _identity

USAGE_SCHEMA = "reliquary/rl-training-usage/v1"
BILLING_BASIS = "matched_token_ceiling_and_trainer_time_v1"
TOKEN_METRIC = "gross_matched_token_equivalents_v1"
TRAINER_METRIC = "owned_process_elapsed_ms_v1"
MAX_SAFE_INTEGER = 2**53 - 1
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TABLES = {"rl_lease_meta", "rl_leases", "rl_buffer", "rl_audits", "rl_groups", "rl_events",
           "rl_accruals", "rl_pay_cursor"}
_RECEIPT_KEYS = {"schema", "billing_basis", "token_metric", "trainer_metric", "task_id", "training_id",
                 "order_sha256", "quote_id", "quote_sha256", "run_id", "runtime_id", "final", "drained",
                 "matched_tokens", "trainer_active_ms", "finished_at", "event_cursor", "accrual_count",
                 "consumption_sha256", "termination_sha256", "checkpoint"}


class UsageNotFinal(ValueError):
    """The available evidence cannot establish final, bound usage."""


def _integer(value, label: str, *, positive: bool = False) -> int:
    if type(value) is not int or not (1 if positive else 0) <= value <= MAX_SAFE_INTEGER:
        raise UsageNotFinal(f"invalid {label}")
    return value


def _decimal(value, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise UsageNotFinal(f"invalid {label}")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise UsageNotFinal(f"invalid {label}") from exc
    if not result.is_finite() or result < 0:
        raise UsageNotFinal(f"invalid {label}")
    return result


def _identifier(value, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise UsageNotFinal(f"invalid {label}")
    return value


def _digest(value, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise UsageNotFinal(f"invalid {label}")
    return value


def _meta(db) -> dict[str, str]:
    return dict(db.execute("SELECT key,value FROM rl_lease_meta"))


def bind_runtime(book, *, run_id: str, runtime_id: str) -> None:
    """Bind this private book once; a different run cannot reuse its counters or quiescence."""
    values = {"training_run_id": _identifier(run_id, "run id"),
              "training_runtime_id": _identifier(runtime_id, "runtime id")}
    with book.transaction():
        meta = _meta(book.db)
        if any(key in meta and meta[key] != value for key, value in values.items()):
            raise UsageNotFinal("runtime identity changed")
        if any(key in meta for key in values) and not all(key in meta for key in values):
            raise UsageNotFinal("runtime identity is incomplete")
        if not any(key in meta for key in values):
            if meta.get("latest") not in (None, "0"):
                raise UsageNotFinal("an unbound book already has a trained checkpoint")
            tables = {row[0] for row in book.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for table in ("rl_leases", "rl_groups", "rl_events", "rl_accruals"):
                if table in tables and book.db.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone():
                    raise UsageNotFinal("an unbound book already contains work")
        book.db.executemany("INSERT OR IGNORE INTO rl_lease_meta(key,value) VALUES(?,?)", values.items())


def request_drain(book, *, runtime_id: str) -> dict:
    """Close admission durably; this does not claim that a trainer or its queues stopped."""
    with book.transaction():
        if _meta(book.db).get("training_runtime_id") != _identifier(runtime_id, "runtime id"):
            raise UsageNotFinal("runtime identity changed")
        book.db.execute("INSERT OR IGNORE INTO rl_lease_meta(key,value) VALUES('training_registry_stop','runtime_drain')")
        return inspect_drain(book.db)


def inspect_drain(db, *, expected_identity: str | None = None) -> dict:
    """Inspect only persisted state. The live caller holds the book lock; finalization uses a snapshot."""
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not _TABLES <= tables:
        return {"admission_closed": False, "drained": False, "latest_checkpoint": None,
                "event_cursor": None, "blockers": ["usage_schema_unavailable"]}
    meta = _meta(db)
    blockers = []
    if expected_identity is not None and meta.get("training_identity") != expected_identity:
        blockers.append("task_identity_changed")
    closed = bool(meta.get("training_registry_stop") or meta.get("training_limit_stop"))
    if not closed:
        blockers.append("admission_open")
    runtime_id = meta.get("training_runtime_id")
    if not runtime_id or not meta.get("training_run_id"):
        blockers.append("runtime_unbound")
    if not meta.get("training_task_id") or not meta.get("training_order_sha256"):
        blockers.append("order_unbound")
    if not runtime_id or meta.get("training_trainer_quiesced") != runtime_id:
        blockers.append("trainer_not_quiesced")
    latest = meta.get("latest")
    try:
        latest = None if latest is None else _integer(int(latest), "latest checkpoint")
    except (UsageNotFinal, ValueError):
        latest = None
        blockers.append("checkpoint_unknown")
    if meta.get("training_final_checkpoint") != (None if latest is None else str(latest)):
        blockers.append("checkpoint_not_final")
    checks = {
        "open_leases": "SELECT COUNT(*) FROM rl_leases WHERE state='open'",
        "pending_submissions": "SELECT COUNT(*) FROM rl_leases l WHERE state='submitted' AND NOT EXISTS(SELECT 1 FROM rl_groups g WHERE g.lease_id=l.lease_id)",
        "buffered_groups": "SELECT COUNT(*) FROM rl_buffer",
        "pending_audits": "SELECT COUNT(*) FROM rl_audits WHERE body IS NOT NULL",
        "unresolved_training": "SELECT COUNT(*) FROM rl_groups WHERE verdict='in_zone' AND COALESCE(training,'') NOT IN ('accepted','rejected')",
        "unresolved_audits": "SELECT COUNT(*) FROM rl_groups WHERE audit='pending'",
    }
    for label, query in checks.items():
        if db.execute(query).fetchone()[0]:
            blockers.append(label)
    cursor = db.execute("SELECT seq FROM rl_pay_cursor WHERE id=0").fetchone()
    last_event = db.execute("SELECT COALESCE(MAX(seq),0) FROM rl_events").fetchone()[0]
    cursor = None if cursor is None else cursor[0]
    if type(cursor) is not int or cursor != last_event:
        blockers.append("ledger_pending")
    columns = {row[1] for row in db.execute("PRAGMA table_info(rl_accruals)")}
    if "budget_tokens" not in columns or db.execute("SELECT 1 FROM rl_accruals WHERE budget_tokens IS NULL LIMIT 1").fetchone():
        blockers.append("unknown_gross_usage")
    if db.execute("SELECT 1 FROM rl_events e LEFT JOIN rl_accruals a ON a.seq=e.seq WHERE e.kind IN ('trained','first_scan') AND (a.seq IS NULL OR a.lease_id!=e.lease_id OR a.hotkey!=e.hotkey OR a.kind!=e.kind) LIMIT 1").fetchone():
        blockers.append("unaccounted_events")
    if db.execute("SELECT 1 FROM rl_accruals a LEFT JOIN rl_events e ON e.seq=a.seq WHERE e.seq IS NULL OR e.kind NOT IN ('trained','first_scan') OR a.lease_id!=e.lease_id OR a.kind!=e.kind OR a.hotkey!=e.hotkey LIMIT 1").fetchone():
        blockers.append("invalid_accrual_source")
    return {"admission_closed": closed, "drained": not blockers, "latest_checkpoint": latest,
            "event_cursor": cursor, "blockers": blockers}


def quiesce_trainer(rl, *, runtime_id: str, latest_checkpoint: int) -> dict:
    """Seal the joined trainer; outstanding miner work can keep the task undrained."""
    latest_checkpoint = _integer(latest_checkpoint, "latest checkpoint")
    with rl.book.transaction():
        db, meta = rl.book.db, _meta(rl.book.db)
        if meta.get("training_runtime_id") != _identifier(runtime_id, "runtime id"):
            raise UsageNotFinal("runtime identity changed")
        if not (meta.get("training_registry_stop") or meta.get("training_limit_stop")):
            raise UsageNotFinal("admission is still open")
        if rl.book.latest_checkpoint != latest_checkpoint:
            raise UsageNotFinal("checkpoint publication is incomplete")
        prior = meta.get("training_trainer_quiesced")
        if prior is not None:
            if prior != runtime_id or meta.get("training_final_checkpoint") != str(latest_checkpoint):
                raise UsageNotFinal("quiesced state changed")
        values = {"training_trainer_quiesced": runtime_id, "training_final_checkpoint": str(latest_checkpoint)}
        db.executemany("INSERT INTO rl_lease_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", values.items())
        return settle_quiesced_work(rl)


def settle_quiesced_work(rl) -> dict:
    """After sealing, terminalize late leftover work without treating it as trained or penalizing miners."""
    from reliquary.validator.rl_submit import AuditResult

    with rl.book.transaction():
        db, meta = rl.book.db, _meta(rl.book.db)
        runtime_id = meta.get("training_runtime_id")
        if not runtime_id or meta.get("training_trainer_quiesced") != runtime_id:
            raise UsageNotFinal("trainer is not quiesced")
        if not (meta.get("training_registry_stop") or meta.get("training_limit_stop")):
            raise UsageNotFinal("admission is still open")
        if meta.get("training_final_checkpoint") != str(rl.book.latest_checkpoint):
            raise UsageNotFinal("checkpoint changed after trainer quiescence")
        if getattr(rl.book, "_submitting", None):
            result = inspect_drain(db)
            return {**result, "drained": False, "blockers": [*result["blockers"], "active_submissions"]}
        groups = db.execute("SELECT lease_id FROM rl_buffer ORDER BY lease_id").fetchall()
        for lease_id, in groups:
            group = rl.buffer.get(lease_id)
            if group is None:
                raise UsageNotFinal("buffered work changed")
            rl.submissions.report_training(lease_id, rollout_match=[0.0] * len(group.completions),
                                           accepted=False, judged=False, failure_class=None)
        audits = db.execute("SELECT lease_id FROM rl_audits WHERE body IS NOT NULL ORDER BY lease_id").fetchall()
        for lease_id, in audits:
            rl.submissions.report_audit(lease_id, AuditResult(None))
        while rl.ledger.ingest():
            pass
        values = {"training_drain_untrained": str(int(meta.get("training_drain_untrained", "0")) + len(groups)),
                  "training_drain_inconclusive": str(int(meta.get("training_drain_inconclusive", "0")) + len(audits))}
        db.executemany("INSERT INTO rl_lease_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", values.items())
        return inspect_drain(db)


def stopped_document(stopped, *, verify_process_groups: bool = True) -> dict:
    """Validate the private supervisor witness and return its canonical termination commitment."""
    from reliquary.rl_orders.supervisor import StoppedRuntime

    if not isinstance(stopped, StoppedRuntime):
        raise UsageNotFinal("stopped runtime witness required")
    if stopped.backend != "single" or stopped.process_scope != "local-cgroup-v2-v1":
        raise UsageNotFinal("runtime process scope is not qualified")
    if stopped.cgroup_id != stopped.runtime_id:
        raise UsageNotFinal("runtime cgroup identity changed")
    _integer(stopped.trainer_elapsed_ns, "trainer elapsed nanoseconds")
    if stopped.latest_checkpoint is not None:
        _integer(stopped.latest_checkpoint, "stopped checkpoint")
    processes = list(stopped.processes)
    if len(processes) != 2 or {process.role for process in processes} != {"controller", "trainer"}:
        raise UsageNotFinal("both owned processes must be reaped")
    if len({process.pid for process in processes}) != 2 or len({process.pgid for process in processes}) != 2:
        raise UsageNotFinal("owned process identities overlap")
    starts, stops, records = [], [], []
    for process in sorted(processes, key=lambda process: process.role):
        _integer(process.pid, "process pid", positive=True)
        _integer(process.pgid, "process group", positive=True)
        _identifier(process.start_id, "process start identity")
        if type(process.returncode) is not int:
            raise UsageNotFinal("process wait is incomplete")
        started, ended = _decimal(process.started_at, "process start"), _decimal(process.stopped_at, "process stop")
        if ended < started:
            raise UsageNotFinal("process timing changed")
        starts.append(started)
        stops.append(ended)
        if verify_process_groups:
            try:
                os.killpg(process.pgid, 0)
            except ProcessLookupError:
                pass
            except OSError as exc:
                if exc.errno != errno.ESRCH:
                    raise UsageNotFinal("process group absence is unconfirmed") from exc
            else:
                raise UsageNotFinal("process group is still alive")
        records.append({**asdict(process), "started_at": str(started), "stopped_at": str(ended)})
    started, stopped_at = _decimal(stopped.started_at, "runtime start"), _decimal(stopped.stopped_at, "runtime stop")
    empty_at = _decimal(stopped.cgroup_empty_at, "cgroup empty time")
    if started > min(starts) or stopped_at < max(stops) or empty_at < max(stops) or empty_at > stopped_at:
        raise UsageNotFinal("runtime termination timing changed")
    return {"task_id": stopped.task_id, "order_sha256": stopped.order_sha256,
            "run_id": stopped.run_id, "runtime_id": stopped.runtime_id, "backend": stopped.backend,
            "process_scope": stopped.process_scope, "cgroup_id": stopped.cgroup_id,
            "cgroup_empty_at": str(empty_at), "started_at": str(started), "stopped_at": str(stopped_at),
            "trainer_elapsed_ns": stopped.trainer_elapsed_ns, "processes": records}


def finalize_usage(entry, db_path, stopped, *, checkpoint: dict | None = None) -> dict:
    """Freeze gross token equivalents and measured trainer time only after proved process termination."""
    termination = stopped_document(stopped)
    order = entry.params.get("training_order")
    if not isinstance(order, dict):
        raise UsageNotFinal("training order binding is missing")
    task_id = _identifier(entry.task_id, "task id")
    training_id = _identifier(order.get("training_id"), "training id")
    order_sha = _digest(order.get("order_sha256"), "order digest")
    quote_id = _identifier(order.get("quote_id"), "quote id")
    quote_sha = _digest(order.get("quote_sha256"), "quote digest")
    if stopped.task_id != task_id or stopped.order_sha256 != order_sha:
        raise UsageNotFinal("stopped runtime belongs to another order")
    run_id, runtime_id = _identifier(stopped.run_id, "run id"), _identifier(stopped.runtime_id, "runtime id")
    path = Path(db_path).resolve(strict=True)
    if not path.is_file():
        raise UsageNotFinal("usage database is unavailable")
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, isolation_level=None)) as db:
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        state = inspect_drain(db, expected_identity=_identity(entry))
        meta = _meta(db)
        if not state["drained"]:
            raise UsageNotFinal("usage is not final: " + ",".join(state["blockers"]))
        if meta.get("training_runtime_id") != runtime_id or meta.get("training_run_id") != run_id:
            raise UsageNotFinal("database runtime identity changed")
        if meta.get("training_task_id") != task_id or meta.get("training_order_sha256") != order_sha:
            raise UsageNotFinal("database order identity changed")
        if stopped.latest_checkpoint != state["latest_checkpoint"]:
            raise UsageNotFinal("stopped checkpoint changed")
        if checkpoint is not None:
            if set(checkpoint) != {"version", "index_sha256"}:
                raise UsageNotFinal("invalid checkpoint reference")
            version = _integer(checkpoint["version"], "checkpoint version")
            _digest(checkpoint["index_sha256"], "checkpoint index digest")
            if version != state["latest_checkpoint"] or version != stopped.latest_checkpoint:
                raise UsageNotFinal("final checkpoint changed")
            checkpoint = dict(checkpoint)
        rows, gross = [], Fraction(0)
        start, end = _decimal(stopped.started_at, "runtime start"), _decimal(stopped.stopped_at, "runtime stop")
        for seq, lease_id, kind, matched_tokens, budget_tokens, accrual_at, event_at in db.execute(
                "SELECT a.seq,a.lease_id,a.kind,a.matched_tokens,a.budget_tokens,a.at,e.at "
                "FROM rl_accruals a JOIN rl_events e ON e.seq=a.seq ORDER BY a.seq"):
            _integer(seq, "accrual sequence", positive=True)
            _integer(matched_tokens, "matched tokens")
            accrual_at, event_at = _decimal(accrual_at, "accrual time"), _decimal(event_at, "event time")
            if accrual_at != event_at or not start <= event_at <= end:
                raise UsageNotFinal("billable event is outside its owned runtime interval")
            value = _decimal(budget_tokens, "gross budget tokens")
            if value > MAX_SAFE_INTEGER:
                raise UsageNotFinal("gross budget tokens exceed the wire bound")
            if kind == "trained" and value != matched_tokens:
                raise UsageNotFinal("trained consumption changed")
            gross += Fraction(value)
            rows.append({"seq": seq, "lease_id": lease_id, "kind": kind,
                         "matched_tokens": matched_tokens, "budget_tokens": str(value), "at": str(event_at)})
        matched = _integer((gross.numerator + gross.denominator - 1) // gross.denominator,
                           "gross matched token equivalents")
        active_ms = _integer((stopped.trainer_elapsed_ns + 999_999) // 1_000_000, "trainer active milliseconds")
        finished_at = _integer(int(_decimal(stopped.stopped_at, "runtime stop").to_integral_value(rounding=ROUND_CEILING)), "finished UTC seconds")
        receipt = {"schema": USAGE_SCHEMA, "billing_basis": BILLING_BASIS, "token_metric": TOKEN_METRIC,
                   "trainer_metric": TRAINER_METRIC,
                   "task_id": task_id, "training_id": training_id, "order_sha256": order_sha,
                   "quote_id": quote_id, "quote_sha256": quote_sha, "run_id": run_id, "runtime_id": runtime_id,
                   "final": True, "drained": True, "matched_tokens": matched, "trainer_active_ms": active_ms,
                   "finished_at": finished_at, "event_cursor": _integer(state["event_cursor"], "event cursor"),
                   "accrual_count": len(rows), "consumption_sha256": canonical_sha256(rows),
                   "termination_sha256": canonical_sha256(termination), "checkpoint": checkpoint}
    return validate_usage_envelope({"receipt": receipt, "sha256": canonical_sha256(receipt)})


def validate_usage_envelope(envelope) -> dict:
    """Validate the closed wire schema before reading a retained proof or publishing it."""
    if not isinstance(envelope, dict) or set(envelope) != {"receipt", "sha256"}:
        raise UsageNotFinal("invalid usage envelope")
    receipt = envelope["receipt"]
    if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_KEYS:
        raise UsageNotFinal("invalid usage receipt keys")
    expected = {"schema": USAGE_SCHEMA, "billing_basis": BILLING_BASIS,
                "token_metric": TOKEN_METRIC, "trainer_metric": TRAINER_METRIC}
    if any(receipt[key] != value for key, value in expected.items()):
        raise UsageNotFinal("invalid usage metric or schema")
    if receipt["final"] is not True or receipt["drained"] is not True:
        raise UsageNotFinal("usage is not final")
    for key in ("task_id", "training_id", "quote_id", "run_id", "runtime_id"):
        _identifier(receipt[key], key)
    for key in ("order_sha256", "quote_sha256", "consumption_sha256", "termination_sha256"):
        _digest(receipt[key], key)
    for key in ("matched_tokens", "trainer_active_ms", "finished_at", "event_cursor", "accrual_count"):
        _integer(receipt[key], key)
    checkpoint = receipt["checkpoint"]
    if checkpoint is not None:
        if not isinstance(checkpoint, dict) or set(checkpoint) != {"version", "index_sha256"}:
            raise UsageNotFinal("invalid usage checkpoint reference")
        _integer(checkpoint["version"], "checkpoint version")
        _digest(checkpoint["index_sha256"], "checkpoint index digest")
    if _digest(envelope["sha256"], "usage receipt digest") != canonical_sha256(receipt):
        raise UsageNotFinal("invalid usage receipt digest")
    return json.loads(canonical_json_bytes(envelope))


def write_final_usage(path, envelope: dict) -> dict:
    """Publish complete canonical bytes exactly once; retries may only reproduce the same final proof."""
    envelope = validate_usage_envelope(envelope)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = canonical_json_bytes(envelope)
    fd, temporary = tempfile.mkstemp(prefix=".usage-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink() or destination.stat().st_size != len(raw) or destination.read_bytes() != raw:
                raise UsageNotFinal("final usage replay changed")
        directory = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)
    return envelope
