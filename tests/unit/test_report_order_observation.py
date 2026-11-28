import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from reliquary.corpus.contract_observation import ContractObservation
from reliquary.validator.corpus_periods import MAX_PRICE_EMISSION_PARAM, MAX_PRICE_PARAM
from scripts.report_order_observation import report

JOB = "private-order-example"
MINER = "private-miner-example"


def observation(tmp_path):
    path = tmp_path / "observation.sqlite"
    obs = ContractObservation(path, ["sample-market"])
    obs.book.post_offer(
        "sample-market",
        MINER,
        "private-owner-example",
        at=90,
        price_micro=1000,
        rate=2,
        slots=2,
    )
    obs.book.open_order(
        JOB,
        "sample-market",
        opened_at=100,
        duration_s=100,
        target_rate=2,
        max_tokens=200,
        budget_rao=1000,
        ceiling_micro=2000,
        new_rate_cap=2,
        eligible_owners={MINER: "private-owner-example"}, owners_observed_at=90,
    )
    entry = SimpleNamespace(job_id=JOB, params={"order_contract": {"opening_valuation": {MAX_PRICE_PARAM: 0.2, MAX_PRICE_EMISSION_PARAM: 2.0}}})
    return path, obs, obs.recorder(entry)


def receipt(sid, tokens, at, *, price=0.125, hotkey=MINER, audited=True):
    return (
        sid,
        {
            "token_count": tokens,
            "passed": True,
            "hotkey": hotkey,
            "received_at": at,
            "lease_price": price,
            "audited": audited,
        },
    )


def archive(
    record, work_period, entry_period, entries, *, missing=False, rewards=0.125
):
    record(
        {
            "work_period": work_period,
            "max_price_per_token": 0.2,
            "entry_period": entry_period,
            "tokens": sum(r["token_count"] for _, r in entries),
            "rewards_by_hotkey": {MINER: rewards},
        },
        [k for k, _ in entries],
        None if missing else [r for _, r in entries],
    )


def close(obs, *, verified, settled):
    obs.close(
        JOB,
        150,
        {"counts_complete": True, "verified_tokens": verified, "settled": settled},
    )


def test_anonymized_read_only_report_separates_net_settlement_and_price_pairing(
    tmp_path,
):
    path, obs, record = observation(tmp_path)
    obs.book.service_interval(JOB, start=105, end=108)
    # Late archive entry for earlier work. The elapsed-rate heuristic must use receipt chronology.
    archive(
        record,
        1,
        1,
        [receipt("a", 2, 101.5), receipt("b", 1, 102.5, price=None, audited=False)],
    )
    archive(
        record,
        0,
        2,
        [
            receipt("c", 1, 100.2),
            receipt("d", 1, 103, hotkey="private-waitlisted-example"),
            receipt("e", 1, 107),
            receipt("f", 1, 160),
        ],
    )
    close(obs, verified=7, settled=6)
    before = path.read_bytes()
    result = report(path)
    assert result["settled_baseline_rao_on_valued_fragments"] == 500_000_000
    assert result["settled_tokens"] == result["receipt_tokens"] == 7
    assert result["audited_receipt_tokens"] == 6
    assert (
        result["matched_contract_tokens"] == 3
        and result["matched_tokens_without_observed_price_or_rate"] == 1
    )
    assert (
        result["paired_tokens"] == 2
        and result["paired_nominal_contract_millirao"] == 2000
    )
    assert result["paired_nominal_observed_millirao_floor"] == 500_000_000_000
    assert (
        result["unmatched_window_rate_or_volume_tokens"] == 3
        and result["unmatched_outage_tokens"] == 1
    )
    assert result["orders_with_matching_final_tokens"] == 1
    assert path.read_bytes() == before
    assert JOB not in json.dumps(result) and MINER not in json.dumps(result)


def test_missing_archive_does_not_overstate_full_order_or_credit_unobserved_deliveries(
    tmp_path,
):
    path, obs, record = observation(tmp_path)
    archive(record, 0, 0, [receipt("seen", 2, 130)])
    without_final = report(path)
    assert (
        without_final["orders_without_final_verified_total"] == 1
        and without_final["paired_tokens"] == 0
    )
    close(obs, verified=5, settled=2)
    partial = report(path)
    assert (
        partial["orders_with_final_token_gap"] == 1
        and partial["final_verified_tokens_on_known_orders"] == 5
    )
    assert (
        partial["captured_tokens_on_known_orders"]
        == partial["unmatched_incomplete_history_tokens"]
        == 2
    )
    assert partial["paired_nominal_observed_millirao_floor"] is None
    archive(record, 0, 1, [receipt("lost", 3, 120)], missing=True)
    aggregate_only = report(path)
    assert aggregate_only["orders_with_matching_final_tokens"] == 1
    assert (
        aggregate_only["orders_with_incomplete_detail"]
        == aggregate_only["fragments_without_receipts"]
        == 1
    )
    assert (
        aggregate_only["paired_tokens"] == 0 and aggregate_only["settled_tokens"] == 5
    )
    assert aggregate_only["settled_baseline_rao_on_valued_fragments"] == 500_000_000


def test_a_supplemental_award_uses_the_price_of_the_new_offer(tmp_path):
    path, obs, record = observation(tmp_path)
    obs.book.exit(JOB, MINER, at=103, reason="announced", event_id="leaving")
    obs.book.post_offer(
        "sample-market",
        "new-miner",
        "new-owner",
        price_micro=1500,
        rate=2,
        slots=2,
        at=104,
        sequence=1,
    )
    obs.book.supplemental_round(
        JOB, at=105, event_id="rest", eligible_owners={"new-miner": "new-owner"}
    )
    archive(record, 1, 3, [receipt("new-work", 7, 130, hotkey="new-miner")])
    close(obs, verified=7, settled=1)
    result = report(path)
    assert result["paired_tokens"] == 7
    assert result["paired_nominal_contract_millirao"] == 10500


def test_unknown_receipt_time_does_not_fabricate_a_delivery_window(tmp_path):
    path, obs, record = observation(tmp_path)
    archive(record, 0, 0, [receipt("r", 5, None)])
    close(obs, verified=5, settled=1)
    result = report(path)
    assert (
        result["tokens_missing_receipt_time"]
        == result["unmatched_incomplete_history_tokens"]
        == 5
    )
    assert result["paired_tokens"] == 0


def test_overlapping_archived_receipts_are_not_paid_or_counted_twice(tmp_path):
    path, _, record = observation(tmp_path)
    one = [receipt("same", 1, 140)]
    archive(record, 0, 0, one)
    archive(record, 0, 1, one)
    with pytest.raises(ValueError, match="overlapping receipt ids"):
        report(path)


def test_cli_exports_aggregate_and_never_creates_a_missing_database(tmp_path):
    path, obs, record = observation(tmp_path)
    archive(record, 0, 0, [receipt("r", 1, 120)])
    close(obs, verified=1, settled=1)
    script = Path(__file__).resolve().parents[2] / "scripts/report_order_observation.py"
    proc = subprocess.run(
        [sys.executable, str(script), str(path), "--order", JOB],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(proc.stdout)["paired_nominal_contract_millirao"] == 1000
    assert JOB not in proc.stdout and MINER not in proc.stdout
    missing = tmp_path / "missing.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        report(missing)
    assert not missing.exists()


def test_market_supply_before_first_order_handles_withdrawals_and_shared_owners(tmp_path):
    path = tmp_path / "standing.sqlite"
    book = ContractObservation(path, ["market"]).book
    for market, hotkey, owner, at, slots in [
        ("market", "first", "one-owner", 10, 1),
        ("market", "second", "one-owner", 11, 1),
        ("market", "second", "one-owner", 12, 0),
        ("market", "third", "another-owner", 12, 1),
        ("market", "third", "another-owner", 12, 0),
        ("another-market", "other", "separate-owner", 10, 1),
    ]:
        book.post_offer(market, hotkey, owner, price_micro=1000, rate=slots, slots=slots, at=at)
    script = Path(__file__).resolve().parents[2] / "scripts/report_order_observation.py"
    for cutoff, active in [(12, 2), (13, 1)]:
        output = subprocess.check_output(
            [sys.executable, str(script), str(path), "--market", "market", "--before", str(cutoff)],
            text=True,
        )
        assert json.loads(output) == {
            "before_unix_second": cutoff, "active_offers": active, "offering_coldkeys": 1
        }

def test_legacy_facts_remain_readable_without_owner_or_conversion_proof(tmp_path):
    path, obs, _ = observation(tmp_path)
    old = obs.recorder(SimpleNamespace(job_id=JOB, params={MAX_PRICE_EMISSION_PARAM: 99}))
    archive(old, 0, 0, [receipt("r", 2, 140)])
    close(obs, verified=2, settled=1)
    result = report(path)
    assert result["receipt_tokens"] == 2 and result["fragments_missing_valuation"] == 1
    assert result["matched_tokens_without_observed_price_or_rate"] == 2
    # A previously recorded rate had no source and a previously recorded award had no read time.
    with sqlite3.connect(path) as db:
        db.execute("DROP TABLE opening_owner_reads")
        db.execute("UPDATE observed_periods SET document=json_set(json_remove(document, '$.valuation_source'), '$.miner_alpha_per_period',99)")
    result = report(path)
    assert result["orders_without_owner_snapshot"] == 1 and result["tokens_without_owner_snapshot"] == 2
    assert result["paired_tokens"] == result["matched_contract_tokens"] == 0
    assert result["settled_baseline_rao_on_valued_fragments"] is None


def test_original_lease_evidence_survives_holds_and_restarts(tmp_path):
    import asyncio
    from unittest.mock import AsyncMock

    from scripts.report_order_observation import verify_grants
    from tests.unit.test_draw_leases import _setup

    path, obs, record_archive = observation(tmp_path)
    held = [False]
    _, _, _, clock, leases, slots = _setup(held=lambda: held[0])
    original = leases.grant(MINER, slots, price=0.125, offer=0.1)
    obs.grants(JOB, [original])
    held[0] = True
    leases.expire()
    clock.now += 20
    held[0] = False
    leases.expire()
    shifted = leases.held(MINER, original.prompt_index, original.slot)
    assert shifted.granted_at == original.granted_at + 20
    reopened = ContractObservation(path, [])
    canonical = {"submission_id": "accepted", "job_id": JOB, "hotkey": MINER,
                 "prompt_index": shifted.prompt_index, "cursor": shifted.slot, "received_at": clock.now + 1,
                 "token_count": 3, "lease_price": 0.125, "lease_debit": 0.01}
    for _ in range(2):
        reopened.accepted(JOB, shifted.lease_id, canonical)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT granted,offer,debit FROM observed_draw_grants").fetchall() == [
            (original.granted_at, 0.1, 0.01)]
    store = SimpleNamespace(read_submission=AsyncMock(return_value=None), read_verdict=AsyncMock(),
                            read_voided=AsyncMock(return_value=None))
    assert asyncio.run(verify_grants(path, JOB, store))["unconfirmed_receipts"] == 1
    store.read_submission.return_value = canonical
    store.read_verdict.return_value = {**canonical, "passed": True}
    assert asyncio.run(verify_grants(path, JOB, store))["passed"] == 1
    store.read_verdict.return_value["passed"] = False
    assert asyncio.run(verify_grants(path, JOB, store))["failed"] == 1
    store.read_voided.return_value = {"voided_at": 2000}
    assert asyncio.run(verify_grants(path, JOB, store))["voided"] == 1
    store.read_submission.return_value = {**canonical, "lease_price": 0.5}
    assert asyncio.run(verify_grants(path, JOB, store))["unconfirmed_receipts"] == 1

    archive(record_archive, 1, 2, [receipt("accepted", 3, canonical["received_at"]),
                                   receipt("missing-grant", 2, 150)])
    result = report(path)
    assert result["settled_tokens_with_original_grant"] == 3
    assert result["settled_tokens_without_original_grant"] == 2
    assert not any(value in json.dumps(result) for value in (MINER, JOB, shifted.lease_id, "missing-grant"))


def test_lease_evidence_is_not_written_into_shared_storage(tmp_path):
    from reliquary.corpus.draw_leases import Lease

    path, obs, _ = observation(tmp_path)
    path.chmod(0o644)
    lease = Lease("private-lease", MINER, 1, 1, 100, 110)
    with pytest.raises(PermissionError):
        obs.grants(JOB, [lease])
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM observed_draw_grants").fetchone() == (0,)
