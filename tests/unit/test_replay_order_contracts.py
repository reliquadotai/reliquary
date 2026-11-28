import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from reliquary.validator.corpus_periods import period_of
from scripts.replay_order_contracts import replay


def event(kind, at, event_id, **fields):
    row = dict(type=kind, at=at, event_id=event_id, **fields)
    if kind != "offer":
        row["order_id"] = "example"
    return row


def trace(*events):
    return io.StringIO("".join(json.dumps(row) + "\n" for row in events))


def opening(**config):
    params = {"market": "model", "duration_s": 100, "target_rate": 1, "max_tokens": 100, "budget_rao": 2000,
                  "ceiling_micro": 20_000, "new_rate_cap": 1, "total_slots": 50}
    return event("open", 10, "open", **{**params, **config})


def offer(hotkey="first", at=1, price=1000):
    return event("offer", at, f"offer-{hotkey}", market="model", hotkey=hotkey, coldkey=f"owner-{hotkey}",
                 price_micro=price, rate=1, slots=1, source="observed")


def test_overlap_only_compares_the_same_tokens_and_does_not_guess_missing_cost_or_timing():
    input_data = trace(
        offer(), opening(),
        event("reserve", 11, "grant", hotkey="first", lease_id="l1", max_tokens=10),
        event("delivery", 14, "accepted", hotkey="first", lease_id="l1", delivery_id="record1", tokens=3,
              audited_tokens=2, observed_price_micro=5000, slot_id="slot1"),
        event("miss", 15, "miss", hotkey="other"),
        event("outage", 20, "outage", end=23),
        event("delivery", 21, "pending", hotkey="first", delivery_id="record2", tokens=2),
        event("delivery", 25, "other", hotkey="other", delivery_id="record3", tokens=4),
        event("pace", 30, "pace", card_tokens_per_second=10, queued_audit_tokens=100, offered_rate=1,
              needed_rate=1, audit_fraction=1),
        event("close", 112, "close", delivered=True),
    )
    result = replay(input_data)["orders"][0]
    assert (result["observed_tokens"], result["matched_tokens"], result["matched_audited_tokens"]) == (9, 3, 2)
    assert result["unmatched_tokens_by_reason"] == {"outage_without_prior_reservation": 2, "no_active_contract": 4}
    assert (result["paired_priced_tokens"], result["paired_observed_cost_millirao"],
            result["paired_contract_cost_millirao"]) == (3, 15000, 3000)
    assert result["observed_priced_tokens"] == 3 and result["observed_priced_cost_millirao"] == 15000
    assert result["observed_last_2pct_accepted_slots_s"] is None and result["observed_deadline_met"] is False
    assert result["observed_missed_requests"] == 1 and result["max_sampled_queue_seconds_at_mean_rate"] == 10
    assert result["standing_offers_by_source"] == {"observed": 1}


def test_future_offer_does_not_supply_prior_opening_and_duplicate_delivery_is_counted_once():
    delivered = event("delivery", 15, "delivery", hotkey="first", delivery_id="r", tokens=1, observed_price_micro=10)
    events = [delivered, offer(at=12), opening(), delivered, {**delivered, "event_id": "record-retry"}]
    result = replay(trace(*events))
    order = result["orders"][0]
    assert result["duplicate_events"] == 1
    assert order["observed_tokens"] == 1 and order["matched_tokens"] == 0
    assert order["matched_contract_cost_millirao"] is None and order["paired_contract_cost_millirao"] is None
    assert order["observed_deadline_met"] is None and not order["standing_offers_by_source"]
    with pytest.raises(ValueError, match="conflicting event_id"):
        replay(trace(*events, {**delivered, "tokens": 2}))
    with pytest.raises(ValueError, match="conflicting delivery_id"):
        replay(trace(*events, {**delivered, "event_id": "other-retry", "tokens": 2}))


def test_complete_trace_reports_observed_tail_in_both_views_only_with_full_coverage():
    accepted = [event("delivery", 10 + n, f"row-{n}", hotkey="first", delivery_id=f"r-{n}", tokens=1,
                      slot_id=str(n)) for n in range(50)]
    accepted[-1]["at"] = 75
    order = replay(trace(offer(), opening(), *accepted))["orders"][0]
    assert order["observed_last_2pct_accepted_slots_s"] == order["matched_trace_last_2pct_accepted_slots_s"] == 17
    assert order["paired_priced_tokens"] == 0 and order["observed_priced_cost_millirao"] is None
    accepted[-1]["hotkey"] = "other"
    order = replay(trace(offer(), opening(), *accepted))["orders"][0]
    assert order["observed_last_2pct_accepted_slots_s"] == 17 and order["matched_trace_last_2pct_accepted_slots_s"] is None


def test_opening_provenance_uses_the_last_preceding_second():
    prior = offer()
    update = {**offer(at=10), "event_id": "updated-offer", "source": "reconstructed"}
    late = offer("late", at=10)
    withdrawn = {**offer("withdrawn", at=5), "rate": 0, "slots": 0}
    order = replay(trace(prior, update, late, withdrawn, opening()))["orders"][0]
    assert order["standing_offers_by_source"] == {"observed": 1}


def test_cancellation_replacement_and_neutral_outage_use_book_rules():
    rows = [offer(), offer("second", price=2000), opening(),
            event("reserve", 11, "g", hotkey="first", lease_id="lease", max_tokens=10),
            event("cancel", 13, "cancel", lease_id="lease"),
            event("exit", 14, "exit", hotkey="first", reason="disappeared", expired_leases=1, penalty_rao=2),
            event("delivery", 15, "accepted", hotkey="second", delivery_id="r", tokens=2, observed_price_micro=2000)]
    order = replay(trace(*rows))["orders"][0]
    assert order["matched_tokens"] == 2 and order["assessed_rao"] == 98
    assert order["unrecovered_after_settlement_rao"] is None
    evidence = event("debits", 19, "settlement", work_period=period_of(14),
                     owners_at_period_end={"first": "owner-first"}, recovered_rao={"exit": 30})
    settled = replay(trace(*rows, evidence, evidence))["orders"][0]
    assert settled["assessed_rao"] == 98 and settled["evidenced_recovery_rao"] == 30
    assert settled["unrecovered_after_settlement_rao"] == 68
    rows.insert(4, event("outage", 12, "outage", end=16))
    order = replay(trace(*rows))["orders"][0]
    assert order["assessed_rao"] == order["unrecovered_after_settlement_rao"] == 0


def test_cli_rejects_invalid_input_even_without_a_database_or_source_directory(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/replay_order_contracts.py"
    result = subprocess.run([sys.executable, str(script), "-"], cwd=tmp_path, input="garbage",
                            text=True, check=False, capture_output=True)
    assert result.returncode == 2 and "line 1" in result.stderr
    assert result.stdout == "" and not list(tmp_path.iterdir())
