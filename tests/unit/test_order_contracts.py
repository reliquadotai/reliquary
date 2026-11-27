from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from reliquary.corpus.order_contracts import ContractBook


def open_job(book, name="job", **updates):
    params = {
        "opened_at": 100,
        "duration_s": 100,
        "target_rate": 10,
        "max_tokens": 1000,
        "budget_rao": 5_000,
        "ceiling_micro": 10_000,
        "new_rate_cap": 10,
    }
    return book.open_order(name, "model", **{**params, **updates})


def offer(book, hotkey, price, *, owner=None, rate=10, slots=2, sequence=1, at=1):
    book.post_offer(
        "model", hotkey, owner or hotkey, price_micro=price, rate=rate, slots=slots, sequence=sequence, at=at
    )


def test_snapshot_and_price_survive_restart_and_offer_withdrawal(tmp_path):
    path = tmp_path / "orders.db"
    book = ContractBook(path)
    offer(book, "a", 500)
    offer(book, "b", 900)
    initial = open_job(book)
    assert initial == [
        {
            "hotkey": "a",
            "coldkey": "a",
            "price_micro": 500,
            "rate": 10,
            "slots": 2,
            "tokens": 1000,
            "remaining_tokens": 1000,
            "q": 1.0,
        }
    ]
    offer(book, "a", 0, rate=0, slots=0, sequence=2)
    offer(book, "b", 100, sequence=2)
    book = ContractBook(path)
    assert open_job(book) == initial
    assert book.waitlist("job")[0]["price_micro"] == 900
    with pytest.raises(ValueError, match="conflicting order"):
        open_job(book, max_tokens=999)
    with pytest.raises(ValueError, match="advanced"):
        offer(book, "a", 500, sequence=1)
    with pytest.raises(ValueError, match="conflicting offer"):
        offer(book, "b", 300, sequence=2)
    assert book.summary("job") is None
    assert book.close("job", at=150) == {"min_micro": 500, "median_micro": 500, "max_micro": 500}
    assert open_job(book, "next")[0]["price_micro"] == 100


def test_concurrent_open_reserves_owner_once_across_hotkeys_and_orders(tmp_path):
    path = tmp_path / "orders.db"
    book = ContractBook(path)
    offer(book, "a", 2, owner="shared")
    offer(book, "b", 1, owner="shared")
    barrier = Barrier(2)

    def open_one(job):
        local = ContractBook(path)
        barrier.wait()
        return open_job(local, job)

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(open_one, ("one", "two")))
    assert sorted(len(r) for r in results) == [0, 1]
    assert sum(c["rate"] for r in results for c in r) == 10
    assert [c["hotkey"] for r in results for c in r] == ["b"]


def test_exact_exposure_grants_cancellation_and_late_replay(tmp_path):
    book = ContractBook(tmp_path / "orders.db")
    offer(book, "a", 333)
    assert open_job(book, budget_rao=1)[0]["tokens"] == 3
    assert book.grant("job", "a", lease_id="lease", max_tokens=3, at=100)
    assert book.grant("job", "a", lease_id="lease", max_tokens=3, at=101)
    assert not book.grant("job", "a", lease_id="second", max_tokens=1, at=101)
    with pytest.raises(ValueError, match="conflicting lease"):
        book.grant("job", "a", lease_id="lease", max_tokens=1, at=101)
    assert book.cancel_grant("job", lease_id="lease")
    assert not book.grant("job", "a", lease_id="lease", max_tokens=3, at=102)
    assert book.grant("job", "a", lease_id="valid", max_tokens=3, at=103)
    book.exit("job", "a", at=105, reason="release")
    book.close("job", at=106)
    assert book.record_delivery("job", "a", delivery_id="record", tokens=3, at=108, audited_tokens=3, lease_id="valid")
    assert not book.record_delivery(
        "job", "a", delivery_id="record", tokens=3, at=109, audited_tokens=3, lease_id="valid"
    )
    with pytest.raises(ValueError, match="matching payable"):
        book.record_delivery("job", "a", delivery_id="second", tokens=3, at=109, lease_id="valid")


def test_replacement_uses_frozen_waitlist_exact_budget_and_single_debt(tmp_path):
    book = ContractBook(tmp_path / "orders.db")
    offer(book, "a", 1000)
    offer(book, "b", 3000)
    open_job(book, budget_rao=1800)
    assert book.record_delivery("job", "a", delivery_id="early", tokens=300, at=130)
    # A grant still awaiting a verdict retains its maximum exposure.
    assert book.grant("job", "a", lease_id="inflight", max_tokens=300, at=140)
    offer(book, "b", 100, sequence=2)
    event = {"at": 150, "reason": "disappeared", "expired_leases": 2, "penalty_rao": 3, "event_id": "event"}
    result = book.exit("job", "a", **event)
    assert result["replaced_tokens"] == 400
    assert result["assessed_rao"] == 806
    assert result["collected_rao"] == 0
    assert book.contract("job", "b")["price_micro"] == 3000
    assert book.exit("job", "a", **event) == result
    assert book.debt_events("job") == [result]
    with pytest.raises(ValueError, match="conflicting exit"):
        book.exit("job", "a", **{**event, "expired_leases": 4})
    assert book.close("job", at=200) == {"min_micro": 1000, "median_micro": 2000, "max_micro": 3000}


def test_outage_and_validator_release_neutralize_penalty(tmp_path):
    book = ContractBook(tmp_path / "orders.db")
    offer(book, "a", 100)
    offer(book, "b", 1200)
    open_job(book)
    book.service_interval("job", start=110, end=130)
    book.service_interval("job", start=120, end=140)
    assert not book.grant("job", "a", lease_id="paused", max_tokens=1, at=120)
    event = book.exit("job", "a", at=125, reason="disappeared", expired_leases=2, penalty_rao=10)
    assert event["reason"] == "service" and event["assessed_rao"] == 0
    assert book.contracts("job") == [] and not book.debt_events("job")
    with pytest.raises(ValueError, match="finalized"):
        book.service_interval("job", start=111, end=130)
    open_job(book, "new", opened_at=150)
    result = book.exit("new", "a", at=175, reason="release")
    assert result["assessed_rao"] == 0


def test_trust_ranking_changes_selection_but_never_changes_bid(tmp_path):
    book = ContractBook(tmp_path / "orders.db")
    offer(book, "veteran", 110)
    open_job(book)
    book.record_delivery("job", "veteran", delivery_id="training", tokens=1000, audited_tokens=1000, at=200)
    book.close("job", at=200)
    offer(book, "newcomer", 100)
    next_params = {"opened_at": 201, "trust_tokens": 10, "audit_cost_micro": 1000}
    c = open_job(book, "generation", **next_params)[0]
    assert c["hotkey"] == "veteran" and c["price_micro"] == 110 and c["q"] < 0.011
    book.close("generation", at=202)
    e = open_job(book, "evaluation", **next_params, evaluation=True)[0]
    assert e["hotkey"] == "newcomer" and e["q"] == 1


def test_delayed_open_replays_frozen_offers_and_attested_ownership(tmp_path):
    book = ContractBook(tmp_path / "orders.db")
    offer(book, "a", 500, at=99)
    offer(book, "a", 1000, at=100, sequence=2)
    offer(book, "a", 2000, at=110, sequence=3)
    offer(book, "stale", 0, at=20)
    offer(book, "at-opening", 0, at=100)
    owners = {"a": "a", "at-opening": "at-opening", "stale": "new-owner"}
    initial = open_job(book, eligible_owners=owners)
    assert len(initial) == 1 and initial[0]["price_micro"] == 500
    assert not book.waitlist("job")
    assert open_job(book, eligible_owners={}) == initial
    book.close("job", at=135)
    assert open_job(book, "later", opened_at=140, eligible_owners={"a": "a"})[0]["price_micro"] == 2000


def test_closing_shadow_work_does_not_penalize_a_miner_who_never_accepted(tmp_path):
    book = ContractBook(tmp_path / "observation.db")
    offer(book, "a", 500)
    offer(book, "b", 500)
    assert open_job(book)[0]["hotkey"] == "a"
    book.close("job", at=200, observed_only=True)
    assert open_job(book, "next", opened_at=201)[0]["hotkey"] == "a"


def test_expired_unclosed_order_releases_capacity_but_keeps_its_payable_grants(tmp_path):
    book = ContractBook(tmp_path / "orders.db")
    offer(book, "a", 1000)
    open_job(book, "A", opened_at=10, duration_s=20)
    assert book.grant("A", "a", lease_id="late-verdict", at=29, max_tokens=100)
    assert open_job(book, "B", opened_at=40)[0]["rate"] == 10
    assert book.order("A")["closed_at"] is None
    assert book.record_delivery(
        "A", "a", lease_id="late-verdict", delivery_id="old", at=50, tokens=100, audited_tokens=100
    )
    assert not book.grant("A", "a", lease_id="more", at=41, max_tokens=1)


def test_supplemental_round_buys_only_residual_with_verified_offer_and_exact_new_debt(tmp_path):
    book = ContractBook(tmp_path / "orders.db")
    offer(book, "cheap", 1000)
    offer(book, "steady", 1500, rate=5, slots=1)
    open_job(book, target_rate=15, max_tokens=1500)
    # Initial waitlist is empty. Releasing the missing capacity cannot establish replacement cost yet.
    exited = book.exit("job", "cheap", event_id="departure", at=120, reason="announced")
    assert not exited["assessed_rao"] and not book.debt_events("job")
    offer(book, "steady", 3500, rate=10, slots=2, sequence=2, at=125)
    offer(book, "new", 2000, at=125)
    offer(book, "too-high", 15000, at=125)
    offer(book, "unregistered", 0, at=125)
    owners = {k: k for k in ("steady", "new", "too-high", "cheap")}
    result = book.supplemental_round("job", event_id="reopen", at=130, eligible_owners=owners)
    assert result["awards"] == [{"hotkey": "new", "price_micro": 2000, "rate": 10, "tokens": 700}]
    assert book.contract("job", "steady")["price_micro"] == 1500
    assert result["debts"][0]["assessed_rao"] == 700
    assert result["debts"][0]["source_event_id"] == "departure"
    assert book.debt_events("job") == result["debts"]
    assert book.supplemental_round("job", event_id="reopen", at=130, eligible_owners={}) == result
    assert book.supplemental_round("job", event_id="next", at=131, eligible_owners=owners)["awards"] == []
    # The original accepted prices and the supplemental winner each enter the closing distribution.
    assert book.close("job", at=199)["median_micro"] == 1500


def test_opening_shortage_can_fill_without_creating_any_miner_debt(tmp_path):
    book = ContractBook(tmp_path / "orders.db")
    assert not open_job(book)
    offer(book, "new", 800, at=104)
    result = book.supplemental_round("job", event_id="first-supply", at=110, eligible_owners={"new": "new"})
    assert result["awards"][0]["tokens"] == 900
    assert not result["debts"] and not book.debt_events("job")


def test_measured_partial_shortfall_excludes_outages_and_waits_for_complete_evidence(tmp_path):
    book = ContractBook(tmp_path / "orders.db")
    offer(book, "slow", 1000)
    offer(book, "backup", 2000)
    open_job(book)
    book.record_delivery("job", "slow", delivery_id="verified", at=150, tokens=40, audited_tokens=40)
    book.service_interval("job", start=110, end=140)
    measurement = {"window_start": 100, "at": 150, "min_active_s": 15}
    assert book.reconcile_rate("job", "slow", event_id="unknown", **measurement) is None
    assert book.contract("job", "slow")["rate"] == 10 and not book.debt_events("job")
    assert book.grant("job", "slow", lease_id="waiting", at=150, max_tokens=5)
    assert book.reconcile_rate("job", "slow", event_id="wait", evidence_complete=True, **measurement) is None
    book.cancel_grant("job", lease_id="waiting")
    event = book.reconcile_rate("job", "slow", event_id="verified-window", evidence_complete=True, **measurement)
    # 40 tokens in 20 active seconds; retain 2 tokens/s and buy 8 more for the final 50s.
    assert book.contract("job", "slow")["rate"] == 2
    assert book.contract("job", "backup")["rate"] == 8
    assert event["replaced_tokens"] == 400 and event["assessed_rao"] == 400
    assert (
        book.reconcile_rate("job", "slow", event_id="verified-window", evidence_complete=True, **measurement) == event
    )


@pytest.mark.parametrize("accepted,extra", [(None, 30), (0, 30), (15, 15), (30, 0)])
def test_late_grant_outcome_restores_only_unused_replacement_volume(tmp_path, accepted, extra):
    book = ContractBook(tmp_path / "book.db")
    offer(book, "A", 1000)
    offer(book, "B", 2000)
    open_job(book, duration_s=10, max_tokens=100)
    assert book.grant("job", "A", lease_id="pending", max_tokens=30, at=100)
    departure = book.exit("job", "A", event_id="depart", at=100, reason="announced")
    assert departure["replaced_tokens"] == 70
    assert book.contract("job", "B")["tokens"] == 70
    if accepted is None:
        assert book.cancel_grant("job", lease_id="pending", at=100)
        assert not book.cancel_grant("job", lease_id="pending", at=100)
    else:
        delivery = {"lease_id": "pending", "delivery_id": "verdict", "tokens": accepted, "at": 100}
        assert book.record_delivery("job", "A", **delivery)
        assert not book.record_delivery("job", "A", **delivery)
    result = book.supplemental_round("job", event_id="fill", at=100, eligible_owners={"B": "B"})
    assert sum(a["tokens"] for a in result["awards"]) == extra
    assert book.contract("job", "B")["tokens"] == 70 + extra
    assert sum(d["assessed_rao"] for d in book.debt_events("job")) == 70 + extra
    if extra:
        assert result["debts"][0]["source_event_id"] == "depart"
        assert result["debts"][0]["coldkey"] == "A"


@pytest.mark.parametrize("accepted", [False, True])
def test_late_grant_outcome_replacement_is_limited_by_remaining_active_seconds(tmp_path, accepted):
    book = ContractBook(tmp_path / "book.db")
    offer(book, "A", 1000)
    offer(book, "B", 2000)
    open_job(book, duration_s=10, max_tokens=100)
    book.grant("job", "A", lease_id="pending", max_tokens=30, at=100)
    book.exit("job", "A", at=100, reason="disappeared")
    if accepted:
        book.record_delivery("job", "A", lease_id="pending", delivery_id="small", at=100, resolved_at=109, tokens=1)
    else:
        book.cancel_grant("job", lease_id="pending", at=109)
    result = book.supplemental_round("job", event_id="fill", at=109, eligible_owners={"B": "B"})
    assert sum(a["tokens"] for a in result["awards"]) == 3


@pytest.mark.parametrize("reason", ["release", "service", "closed"])
def test_late_cancel_does_not_create_charge_on_neutral_exit_or_closed_order(tmp_path, reason):
    book = ContractBook(tmp_path / "book.db")
    offer(book, "A", 1000)
    offer(book, "B", 2000)
    open_job(book, duration_s=10, max_tokens=100)
    book.grant("job", "A", lease_id="pending", max_tokens=30, at=100)
    if reason == "service":
        book.service_interval("job", start=101, end=102)
    book.exit("job", "A", at=101, reason="release" if reason == "release" else "disappeared")
    if reason == "closed":
        book.close("job", at=102)
    before = book.debt_events("job")
    book.cancel_grant("job", lease_id="pending", at=103)
    if reason != "closed":
        assert not book.supplemental_round("job", event_id="fill", at=104, eligible_owners={"B": "B"})["awards"]
    assert book.debt_events("job") == before
