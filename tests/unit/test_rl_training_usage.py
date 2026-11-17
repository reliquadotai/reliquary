"""Final usage needs a closed, drained book and a complete supervisor termination witness."""
from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest

from reliquary.protocol.release_contract import canonical_sha256
from reliquary.rl_orders.supervisor import StoppedProcess, StoppedRuntime
from reliquary.shared.task_registry import TaskEntry
from reliquary.validator.rl_ledger import LedgerConfig, RlPayLedger
from reliquary.validator.rl_training_orders import _identity
from reliquary.validator.rl_training_usage import (
    BILLING_BASIS, TOKEN_METRIC, TRAINER_METRIC, UsageNotFinal, bind_runtime, finalize_usage, inspect_drain,
    quiesce_trainer, request_drain, settle_quiesced_work, stopped_document, write_final_usage,
    validate_usage_envelope,
)
from tests.unit.test_rl_auction import ENV, ask, auctioned_book
from tests.unit.test_rl_submit import NOW, group, service


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    book, auctions, clock = auctioned_book(tmp_path)
    submissions = service(book, probation_groups=0, audit_rate=0)
    ledger = RlPayLedger(book, submissions, None, LedgerConfig(task_id="order-train-test", cap=.5))
    order = {"training_id": "test", "order_sha256": "a" * 64, "quote_id": "quote-test",
             "quote_sha256": "b" * 64, "billing_basis": BILLING_BASIS}
    entry = TaskEntry("order-train-test", "p", "c" * 64, "rl-discovered-price",
                      {"training_order": order}, "active", None)
    with book.transaction():
        book.db.execute("INSERT INTO rl_lease_meta(key,value) VALUES('training_identity',?)", (_identity(entry),))
        book.db.execute("INSERT INTO rl_lease_meta(key,value) VALUES('training_task_id',?)", (entry.task_id,))
        book.db.execute("INSERT INTO rl_lease_meta(key,value) VALUES('training_order_sha256',?)", (order["order_sha256"],))
    bind_runtime(book, run_id="run-test", runtime_id="runtime-test")
    rl = SimpleNamespace(book=book, buffer=submissions.buffer, submissions=submissions, ledger=ledger)
    processes = (StoppedProcess("controller", 10001, 10001, "start-controller", 0, NOW, NOW + 10),
                 StoppedProcess("trainer", 10002, 10002, "start-trainer", 0, NOW + 1, NOW + 9))
    stopped = StoppedRuntime(entry.task_id, order["order_sha256"], "run-test", "runtime-test", NOW,
                             NOW + 12, 8_000_000_123, 0, processes,
                             cgroup_id="runtime-test", cgroup_empty_at=NOW + 11)

    def gone(pgid, signal):
        assert signal == 0
        raise ProcessLookupError()

    monkeypatch.setattr("reliquary.validator.rl_training_usage.os.killpg", gone)
    yield SimpleNamespace(book=book, auctions=auctions, rl=rl, entry=entry, stopped=stopped,
                          path=tmp_path / "leases.sqlite3", output=tmp_path / "usage.json")
    book.close()


def _paid(runtime):
    runtime.auctions.post_ask(ENV, "A", .5, 8)
    lease = runtime.book.grant([ask("A", [1])])[0].leases[0]
    runtime.rl.submissions.submit(lease.lease_id, "A", group(8))
    runtime.rl.submissions.report_training(lease.lease_id, rollout_match=[1.0] * 16, accepted=True)
    runtime.rl.ledger.ingest()
    return lease


def _drain(runtime):
    request_drain(runtime.book, runtime_id="runtime-test")
    return quiesce_trainer(runtime.rl, runtime_id="runtime-test", latest_checkpoint=0)


def test_runtime_binding_and_drain_request_are_once_bound_and_not_finality(runtime):
    bind_runtime(runtime.book, run_id="run-test", runtime_id="runtime-test")
    with pytest.raises(UsageNotFinal, match="identity changed"):
        bind_runtime(runtime.book, run_id="another-run", runtime_id="runtime-test")
    with pytest.raises(UsageNotFinal, match="identity changed"):
        request_drain(runtime.book, runtime_id="another-runtime")
    observed = request_drain(runtime.book, runtime_id="runtime-test")
    assert observed["admission_closed"] and not observed["drained"]
    assert "trainer_not_quiesced" in observed["blockers"]


def test_joined_trainer_ack_drains_partial_groups_neutrally_and_is_idempotent(runtime):
    runtime.auctions.post_ask(ENV, "A", .5, 8)
    held = runtime.book.grant([ask("A", [1])])[0].leases[0]
    runtime.rl.submissions.submit(held.lease_id, "A", group(8))
    request_drain(runtime.book, runtime_id="runtime-test")
    assert "buffered_groups" in inspect_drain(runtime.book.db)["blockers"]
    assert quiesce_trainer(runtime.rl, runtime_id="runtime-test", latest_checkpoint=0)["drained"]
    assert runtime.book.db.execute("SELECT training FROM rl_groups").fetchone() == ("rejected",)
    assert runtime.book.db.execute("SELECT COUNT(*) FROM rl_verdicts").fetchone() == (0,)
    assert runtime.book.db.execute("SELECT kind,detail FROM rl_events").fetchone()[0] == "untrained"
    assert _drain(runtime)["drained"]
    assert runtime.book.db.execute("SELECT COUNT(*) FROM rl_events").fetchone() == (1,)


def test_pending_audit_is_explicitly_inconclusive_without_miner_failure(runtime):
    runtime.rl.submissions.config = replace(runtime.rl.submissions.config, audit_rate=1)
    runtime.auctions.post_ask(ENV, "A", .5, 8)
    held = runtime.book.grant([ask("A", [1])])[0].leases[0]
    runtime.rl.submissions.submit(held.lease_id, "A", group(0))
    assert runtime.book.db.execute("SELECT COUNT(*) FROM rl_audits WHERE body IS NOT NULL").fetchone() == (1,)
    assert _drain(runtime)["drained"]
    assert runtime.book.db.execute("SELECT audit FROM rl_groups").fetchone() == ("inconclusive",)
    assert runtime.book.db.execute("SELECT COUNT(*) FROM rl_verdicts").fetchone() == (0,)
    assert runtime.book.db.execute("SELECT COUNT(*) FROM rl_accruals").fetchone() == (0,)


def test_joined_trainer_ack_is_distinct_from_pending_miner_drainage(runtime):
    runtime.auctions.post_ask(ENV, "A", .5, 8)
    lease = runtime.book.grant([ask("A", [1])])[0].leases[0]
    request_drain(runtime.book, runtime_id="runtime-test")
    with pytest.raises(UsageNotFinal, match="publication"):
        quiesce_trainer(runtime.rl, runtime_id="runtime-test", latest_checkpoint=1)
    assert not quiesce_trainer(runtime.rl, runtime_id="runtime-test", latest_checkpoint=0)["drained"]
    runtime.book.hold_submission(lease.lease_id, "A", NOW)
    assert runtime.book.release_for_order_drain(runtime_id="runtime-test") == 0
    runtime.rl.submissions.submit(lease.lease_id, "A", group(8))
    assert not settle_quiesced_work(runtime.rl)["drained"]
    assert runtime.book.db.execute("SELECT COUNT(*) FROM rl_buffer").fetchone() == (1,)
    runtime.book.release_submission(lease.lease_id)
    assert settle_quiesced_work(runtime.rl)["drained"]
    assert runtime.book.db.execute("SELECT training FROM rl_groups").fetchone() == ("rejected",)


def test_explicit_and_normal_expiry_after_seal_are_neutral_not_miner_misses(runtime):
    runtime.auctions.post_ask(ENV, "A", .5, 8)
    leases = runtime.book.grant([ask("A", [1, 2])])[0].leases
    with pytest.raises(ValueError, match="not authorized"):
        runtime.book.release_for_order_drain(runtime_id="runtime-test")
    request_drain(runtime.book, runtime_id="runtime-test")
    quiesce_trainer(runtime.rl, runtime_id="runtime-test", latest_checkpoint=0)
    with pytest.raises(ValueError, match="not authorized"):
        runtime.book.release_for_order_drain(runtime_id="another-runtime")
    runtime.book.hold_submission(leases[0].lease_id, "A", NOW)
    assert runtime.book.release_for_order_drain(runtime_id="runtime-test") == 1
    assert runtime.book.release_for_order_drain(runtime_id="runtime-test") == 0
    runtime.book.release_submission(leases[0].lease_id)
    runtime.book.expire_due(NOW + 601)
    assert runtime.book.db.execute("SELECT DISTINCT state,cause FROM rl_leases").fetchall() == [("released", "order_drain")]
    assert runtime.auctions.take_debit(ENV, "A", 1.0) == 0
    assert settle_quiesced_work(runtime.rl)["drained"]


def test_final_receipt_counts_gross_work_after_forfeiture_and_uses_owned_time(runtime):
    _paid(runtime)
    expected = runtime.book.db.execute("SELECT SUM(budget_tokens) FROM rl_accruals").fetchone()[0]
    with runtime.book.transaction():
        runtime.book.db.execute("UPDATE rl_accruals SET status='forfeited',amount=0")
    _drain(runtime)
    checkpoint = {"version": 0, "index_sha256": "d" * 64}
    envelope = finalize_usage(runtime.entry, runtime.path, runtime.stopped, checkpoint=checkpoint)
    receipt = envelope["receipt"]
    assert envelope["sha256"] == canonical_sha256(receipt)
    assert receipt["matched_tokens"] == int(expected)
    assert receipt["trainer_active_ms"] == 8001
    assert receipt["billing_basis"] == BILLING_BASIS and receipt["token_metric"] == TOKEN_METRIC
    assert receipt["trainer_metric"] == TRAINER_METRIC
    assert receipt["finished_at"] == NOW + 12
    assert receipt["checkpoint"] == checkpoint and receipt["drained"] and receipt["final"]
    assert receipt["quote_id"] == "quote-test" and receipt["quote_sha256"] == "b" * 64
    assert not any("cost" in key or "hotkey" in key for key in receipt)
    assert finalize_usage(runtime.entry, runtime.path, runtime.stopped, checkpoint=checkpoint) == envelope
    assert write_final_usage(runtime.output, envelope) == envelope
    assert write_final_usage(runtime.output, envelope) == envelope
    assert json.loads(runtime.output.read_text()) == envelope
    changed = {**receipt, "matched_tokens": receipt["matched_tokens"] + 1}
    with pytest.raises(UsageNotFinal, match="replay changed"):
        write_final_usage(runtime.output, {"receipt": changed, "sha256": canonical_sha256(changed)})


def test_usage_counter_is_immutable_but_payment_status_can_change(runtime):
    _paid(runtime)
    for query in ("UPDATE rl_accruals SET budget_tokens=0", "UPDATE rl_accruals SET matched_tokens=0",
                  "UPDATE rl_accruals SET lease_id='other'", "UPDATE rl_accruals SET at=at+1",
                  "DELETE FROM rl_accruals"):
        with pytest.raises(sqlite3.IntegrityError, match="gross_usage_immutable"):
            with runtime.book.transaction():
                runtime.book.db.execute(query)
    with runtime.book.transaction():
        runtime.book.db.execute("UPDATE rl_accruals SET status='forfeited',amount=0")


def test_late_work_and_mismatched_runtime_never_become_final(runtime):
    _paid(runtime)
    _drain(runtime)
    changed = replace(runtime.stopped, runtime_id="different-runtime", cgroup_id="different-runtime")
    with pytest.raises(UsageNotFinal, match="database runtime"):
        finalize_usage(runtime.entry, runtime.path, changed)
    with runtime.book.transaction():
        runtime.book.db.execute("INSERT INTO rl_events(lease_id,hotkey,kind,at,detail) VALUES('late','A','untrained',?,'{}')", (NOW,))
    with pytest.raises(UsageNotFinal, match="ledger_pending"):
        finalize_usage(runtime.entry, runtime.path, runtime.stopped)


def test_unknown_historical_first_scan_is_not_silently_zero(runtime):
    _drain(runtime)
    with runtime.book.transaction():
        runtime.book.db.execute("INSERT INTO rl_events(seq,lease_id,hotkey,kind,at,detail) VALUES(1,'legacy','A','first_scan',?,'{}')", (NOW,))
        runtime.book.db.execute("INSERT INTO rl_accruals(seq,lease_id,hotkey,environment,kind,at,period,matched_tokens,token_price,amount,status) VALUES(1,'legacy','A',?,'first_scan',?,0,0,1e-6,0,'forfeited')", (ENV, NOW))
        runtime.book.db.execute("UPDATE rl_pay_cursor SET seq=1")
    with pytest.raises(UsageNotFinal, match="unknown_gross_usage"):
        finalize_usage(runtime.entry, runtime.path, runtime.stopped)


@pytest.mark.parametrize("change", [{"backend": "miles"}, {"process_scope": "local-process-groups-v1"},
                                  {"cgroup_id": "other"}, {"trainer_elapsed_ns": True}])
def test_incomplete_or_unqualified_process_scope_cannot_finalize(runtime, change):
    _drain(runtime)
    with pytest.raises(UsageNotFinal):
        finalize_usage(runtime.entry, runtime.path, replace(runtime.stopped, **change))


def test_a_still_live_group_or_unconfirmed_process_wait_blocks_finality(runtime, monkeypatch):
    _drain(runtime)
    monkeypatch.setattr("reliquary.validator.rl_training_usage.os.killpg", lambda *args: None)
    with pytest.raises(UsageNotFinal, match="still alive"):
        finalize_usage(runtime.entry, runtime.path, runtime.stopped)
    process = replace(runtime.stopped.processes[0], returncode=None)
    with pytest.raises(UsageNotFinal, match="wait is incomplete"):
        stopped_document(replace(runtime.stopped, processes=(process, runtime.stopped.processes[1])))


def test_checkpoint_mismatch_and_missing_quote_binding_are_not_customer_proofs(runtime):
    _drain(runtime)
    with pytest.raises(UsageNotFinal, match="checkpoint changed"):
        finalize_usage(runtime.entry, runtime.path, runtime.stopped,
                       checkpoint={"version": 1, "index_sha256": "d" * 64})
    entry = replace(runtime.entry, params={"training_order": {**runtime.entry.params["training_order"],
                                                            "quote_sha256": None}})
    with pytest.raises(UsageNotFinal, match="quote digest"):
        finalize_usage(entry, runtime.path, runtime.stopped)
    assert finalize_usage(runtime.entry, runtime.path, runtime.stopped)["receipt"]["checkpoint"] is None


def test_receipt_tampering_is_refused_before_any_final_file_is_written(runtime):
    _drain(runtime)
    envelope = finalize_usage(runtime.entry, runtime.path, runtime.stopped)
    envelope["receipt"]["matched_tokens"] += 1
    with pytest.raises(UsageNotFinal, match="digest"):
        write_final_usage(runtime.output, envelope)
    assert not runtime.output.exists()


@pytest.mark.parametrize("field,value", [("matched_tokens", True), ("trainer_active_ms", 1.0),
                                        ("event_cursor", 2**53), ("finished_at", -1),
                                        ("final", False), ("drained", 1),
                                        ("quote_sha256", "b" * 63),
                                        ("token_metric", "net_pay_tokens"),
                                        ("trainer_metric", "gpu_active_time")])
def test_retained_receipt_parser_refuses_ambiguous_or_unqualified_wire_values(runtime, field, value):
    _drain(runtime)
    receipt = finalize_usage(runtime.entry, runtime.path, runtime.stopped)["receipt"]
    receipt[field] = value
    with pytest.raises(UsageNotFinal):
        validate_usage_envelope({"receipt": receipt, "sha256": canonical_sha256(receipt)})


def test_fractional_usage_is_summed_exactly_before_one_integer_ceiling(runtime):
    _drain(runtime)
    with runtime.book.transaction():
        for seq, kind, matched, gross in [(1, "trained", 9_000_000_000_000_000, 9_000_000_000_000_000),
                                           (2, "first_scan", 0, .5000000000000001),
                                           (3, "first_scan", 0, .5000000000000001)]:
            lease_id = "fraction-" + str(seq)
            runtime.book.db.execute("INSERT INTO rl_events(seq,lease_id,hotkey,kind,at,detail) VALUES(?,?,'A',?,?,'{}')",
                                    (seq, lease_id, kind, NOW))
            runtime.book.db.execute("INSERT INTO rl_accruals(seq,lease_id,hotkey,environment,kind,at,period,matched_tokens,token_price,amount,status,budget_tokens) VALUES(?,?,'A',?,?,?,0,?,1e-6,0,'forfeited',?)",
                                    (seq, lease_id, ENV, kind, NOW, matched, gross))
        runtime.book.db.execute("UPDATE rl_pay_cursor SET seq=3")
    receipt = finalize_usage(runtime.entry, runtime.path, runtime.stopped)["receipt"]
    assert receipt["matched_tokens"] == 9_000_000_000_000_002


def test_failed_resolution_rolls_back_and_retains_work(runtime):
    runtime.auctions.post_ask(ENV, "A", .5, 8)
    held = runtime.book.grant([ask("A", [1])])[0].leases[0]
    runtime.rl.submissions.submit(held.lease_id, "A", group(8))
    with runtime.book.transaction():
        runtime.book.db.execute("INSERT INTO rl_audits(lease_id,hotkey,forced,queued_at,body) VALUES('orphan','A',0,?,?)", (NOW, b"unknown"))
    request_drain(runtime.book, runtime_id="runtime-test")
    with pytest.raises(LookupError, match="no pending audit"):
        quiesce_trainer(runtime.rl, runtime_id="runtime-test", latest_checkpoint=0)
    assert runtime.book.db.execute("SELECT training FROM rl_groups").fetchone() == ("queued",)
    assert runtime.book.db.execute("SELECT COUNT(*) FROM rl_buffer").fetchone() == (1,)
    assert runtime.book.db.execute("SELECT COUNT(*) FROM rl_events").fetchone() == (0,)
    assert "training_trainer_quiesced" not in dict(runtime.book.db.execute("SELECT key,value FROM rl_lease_meta"))


def test_parallel_replays_publish_one_complete_private_receipt(runtime):
    _drain(runtime)
    envelope = finalize_usage(runtime.entry, runtime.path, runtime.stopped)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda _: write_final_usage(runtime.output, envelope), range(8))) == [envelope] * 8
    assert json.loads(runtime.output.read_text()) == envelope
    assert runtime.output.stat().st_mode & 0o777 == 0o600
    assert not list(runtime.output.parent.glob(".usage-*"))


def test_a_used_unbound_book_cannot_be_relabelled_as_a_new_runtime(runtime):
    _paid(runtime)
    with runtime.book.transaction():
        runtime.book.db.execute("DELETE FROM rl_lease_meta WHERE key IN ('training_runtime_id','training_run_id')")
    with pytest.raises(UsageNotFinal, match="already contains work"):
        bind_runtime(runtime.book, run_id="new-run", runtime_id="new-runtime")


def test_a_noninitial_checkpoint_alone_is_not_a_fresh_book(runtime):
    with runtime.book.transaction():
        runtime.book.db.execute("DELETE FROM rl_lease_meta WHERE key IN ('training_runtime_id','training_run_id')")
    runtime.book.adopt_checkpoint(1)
    with pytest.raises(UsageNotFinal, match="already has a trained checkpoint"):
        bind_runtime(runtime.book, run_id="new-run", runtime_id="new-runtime")


@pytest.mark.parametrize("offset", [-100, 100])
def test_consumption_outside_the_owned_process_interval_is_not_billed(runtime, offset):
    _paid(runtime)
    _drain(runtime)
    processes = tuple(replace(process, started_at=process.started_at + offset,
                              stopped_at=process.stopped_at + offset) for process in runtime.stopped.processes)
    stopped = replace(runtime.stopped, processes=processes, started_at=runtime.stopped.started_at + offset,
                      stopped_at=runtime.stopped.stopped_at + offset,
                      cgroup_empty_at=runtime.stopped.cgroup_empty_at + offset)
    with pytest.raises(UsageNotFinal, match="outside its owned runtime"):
        finalize_usage(runtime.entry, runtime.path, stopped)


def test_event_and_accrual_timestamp_must_agree(runtime):
    _drain(runtime)
    with runtime.book.transaction():
        runtime.book.db.execute("INSERT INTO rl_events(seq,lease_id,hotkey,kind,at,detail) VALUES(1,'time','A','first_scan',?,'{}')", (NOW + 1,))
        runtime.book.db.execute("INSERT INTO rl_accruals(seq,lease_id,hotkey,environment,kind,at,period,matched_tokens,token_price,amount,status,budget_tokens) VALUES(1,'time','A',?,'first_scan',?,0,0,1e-6,0,'forfeited',1)", (ENV, NOW))
        runtime.book.db.execute("UPDATE rl_pay_cursor SET seq=1")
    with pytest.raises(UsageNotFinal, match="outside its owned runtime"):
        finalize_usage(runtime.entry, runtime.path, runtime.stopped)
