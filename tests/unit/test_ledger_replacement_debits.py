"""Replacement debit simulations on obligations built from settled period archives."""

from copy import deepcopy

import pytest

from reliquary.ledger.obligations import ArchiveRef, RatePolicy, build_ledger
from reliquary.ledger.replacement_debits import simulate_replacement_debits
from reliquary.validator.corpus_periods import PERIOD_SECONDS

WORK = 23_817
DAY = 86_400 // PERIOD_SECONDS


def event(event_id="replacement-a", **fields):
    return {
        "event_id": event_id, "hotkey": "worker", "coldkey_at_earn": "owner",
        "assessed_rao": 1_000, "recovered_at_settlement_rao": 300, "reason": "replacement",
        **fields,
    }


def archive(period=WORK, hotkey="worker", **fields):
    body = {
        "schema": "reliquary/corpus-period/v1", "task_id": "order", "work_period": period,
        "rewards_by_hotkey": {hotkey: 0.01}, "cap": 0.01, **fields,
    }
    return ArchiveRef("order", period, period + 1, body)


def setup_simulation(refs, retained=400):
    ledger = build_ledger(refs, rates=RatePolicy(mode="fixed", alpha_per_period=1))
    assert not ledger.anomalies
    snapshot = {"owners": [], "balances": [], "applications": []}
    for obligation in ledger.obligations:
        snapshot["owners"].append({
            "work_period": obligation.work_period, "hotkey": obligation.hotkey, "coldkey": "owner",
        })
        snapshot["balances"].append({
            "obligation_id": obligation.obligation_id, "rail": "collector",
            "state": "held", "retained_rao": retained,
        })
    return ledger.obligations, snapshot


def test_replay_consumes_only_the_post_settlement_remainder_once():
    refs = [archive(replacement_debits_v1=[event()])]
    obligations, snapshot = setup_simulation(refs)
    before = deepcopy(snapshot)
    first = simulate_replacement_debits(refs, obligations, snapshot)
    assert snapshot == before
    (row,) = first["events"]
    assert row["assessed_rao"] == 1_000
    assert row["recovered_at_settlement_rao"] == 300
    assert row["previously_applied_rao"] == 0
    assert row["proposed_rao"] == 400
    assert row["unrecovered_rao"] == 300
    assert first["proposed_applications"] == [
        {"event_id": "replacement-a", "obligation_id": obligations[0].obligation_id, "amount_rao": 400}
    ]

    snapshot["applications"] = first["proposed_applications"]
    replay = simulate_replacement_debits(refs, obligations, snapshot)
    assert replay["proposed_applications"] == []
    (row,) = replay["events"]
    assert row["previously_applied_rao"] == 400
    assert row["proposed_rao"] == 0
    assert row["unrecovered_rao"] == 300


def test_assessment_caps_recovery_even_when_more_is_held():
    refs = [archive(replacement_debits_v1=[event()])]
    obligations, snapshot = setup_simulation(refs, retained=2_000)
    result = simulate_replacement_debits(refs, obligations, snapshot)
    assert result["events"][0]["proposed_rao"] == 700
    assert result["events"][0]["unrecovered_rao"] == 0
    assert sum(a["amount_rao"] for a in result["proposed_applications"]) == 700


def test_prior_application_survives_payment_of_the_original_obligation():
    refs = [archive(replacement_debits_v1=[event()]), archive(WORK + DAY)]
    obligations, snapshot = setup_simulation(refs)
    paid, held = sorted(obligations, key=lambda o: o.work_period)
    for balance in snapshot["balances"]:
        if balance["obligation_id"] == paid.obligation_id:
            balance.update(state="paid", retained_rao=0)
    snapshot["applications"] = [
        {"event_id": "replacement-a", "obligation_id": paid.obligation_id, "amount_rao": 400}
    ]
    result = simulate_replacement_debits(refs, obligations, snapshot)
    assert result["events"][0]["previously_applied_rao"] == 400
    assert result["proposed_applications"] == [
        {"event_id": "replacement-a", "obligation_id": held.obligation_id, "amount_rao": 300}
    ]


def test_old_archive_has_no_replacement_debit():
    refs = [archive()]
    obligations, snapshot = setup_simulation(refs)
    result = simulate_replacement_debits(refs, obligations, snapshot)
    assert result["events"] == []
    assert result["proposed_applications"] == []


def test_requeue_keeps_one_immutable_debit_and_checks_the_archive_key():
    original = ArchiveRef("order", WORK, WORK + 9, archive(replacement_debits_v1=[event()]).document)
    moved = archive(replaces_entry_period=WORK + 9, replacement_debits_v1=[event()])
    refs = [original, moved]
    obligations, snapshot = setup_simulation(refs)
    full = simulate_replacement_debits(refs, obligations, snapshot)
    assert full == simulate_replacement_debits([moved], obligations, snapshot)
    assert len(full["proposed_applications"]) == 1
    moved.document["task_id"] = "another-order"
    with pytest.raises(ValueError, match="identity"):
        simulate_replacement_debits(refs, obligations, snapshot)
    moved.document["task_id"] = "order"
    del moved.document["replacement_debits_v1"]
    with pytest.raises(ValueError, match="requeue"):
        simulate_replacement_debits(refs, obligations, snapshot)


def test_only_the_two_work_days_and_the_owner_at_each_earn_are_eligible():
    refs = [
        archive(WORK - DAY), archive(replacement_debits_v1=[event()]),
        archive(WORK + DAY - 1), archive(WORK + DAY, "second-worker"), archive(WORK + DAY * 2),
    ]
    obligations, snapshot = setup_simulation(refs, retained=200)
    for owner in snapshot["owners"]:
        if owner["work_period"] == WORK + DAY - 1:
            owner["coldkey"] = "new-owner"
    result = simulate_replacement_debits(refs, obligations, snapshot)
    eligible_ids = {o.obligation_id for o in obligations if o.work_period in (WORK, WORK + DAY)}
    assert {a["obligation_id"]: a["amount_rao"] for a in result["proposed_applications"]} == {
        obligation_id: 200 for obligation_id in eligible_ids
    }
    assert result["events"][0]["unrecovered_rao"] == 300


@pytest.mark.parametrize(("rail", "state"), [
    ("native", "paid"), ("native", "held"), ("collector", "paid"),
])
def test_payment_rail_and_hold_state_both_gate_new_applications(rail, state):
    refs = [archive(replacement_debits_v1=[event()])]
    obligations, snapshot = setup_simulation(refs)
    snapshot["balances"][0].update(rail=rail, state=state)
    result = simulate_replacement_debits(refs, obligations, snapshot)
    assert result["proposed_applications"] == []
    assert result["events"][0]["unrecovered_rao"] == 700


def test_two_events_share_one_envelope_and_each_preserves_its_own_remainder():
    refs = [archive(replacement_debits_v1=[
        event("replacement-a", assessed_rao=600), event("replacement-b", assessed_rao=600),
    ])]
    obligations, snapshot = setup_simulation(refs)
    result = simulate_replacement_debits(refs, obligations, snapshot)
    applications = result["proposed_applications"]
    assert {a["obligation_id"] for a in applications} == {obligations[0].obligation_id}
    assert {a["event_id"] for a in applications} == {"replacement-a", "replacement-b"}
    assert sum(a["amount_rao"] for a in applications) == 400
    for row in result["events"]:
        assert row["proposed_rao"] + row["unrecovered_rao"] == 300
    snapshot["applications"] = applications
    replay = simulate_replacement_debits(refs, obligations, snapshot)
    assert replay["proposed_applications"] == []


@pytest.mark.parametrize("inconsistency", [
    "owner-at-earn", "over-recovery", "overdrawn-hold", "conflicting-event",
])
def test_contradictory_evidence_is_rejected(inconsistency):
    debit = event()
    refs = [archive(replacement_debits_v1=[debit])]
    obligations, snapshot = setup_simulation(refs)
    if inconsistency == "owner-at-earn":
        snapshot["owners"][0]["coldkey"] = "different-owner"
    elif inconsistency == "over-recovery":
        debit["recovered_at_settlement_rao"] = 1_001
    elif inconsistency == "overdrawn-hold":
        snapshot["applications"] = [
            {"event_id": debit["event_id"], "obligation_id": obligations[0].obligation_id, "amount_rao": 401}
        ]
    else:
        refs[0].document["replacement_debits_v1"].append(event(assessed_rao=900))
    with pytest.raises(ValueError):
        simulate_replacement_debits(refs, obligations, snapshot)
