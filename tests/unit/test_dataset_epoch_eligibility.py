"""First-pass recovery, coverage and bounded epoch transitions."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from reliquary.protocol.service_contract import ServiceContract
from reliquary.services.eligibility import EligibilityStore


def _contract(**limits):
    value = json.loads((Path(__file__).parents[1] / "fixtures" / "service_contract_v1.json").read_text())
    value["service_kind"] = "adaptive_training"
    value["policies"]["checkpoint"] = {"kind": "trainer-driven/v1", "task_scoped": 1}
    value["policies"]["eligibility"] = {"kind": "dataset-epoch/v1", "coverage_bps": 7500,
                                         "refresh_windows": 2, "max_epoch_windows": 10}
    value["limits"].update(limits)
    return ServiceContract.from_dict(value)


def _qualified(contract):
    return {"context_sha256": contract.context_sha256, "generation_verified": True,
            "sampling_verified": True, "rl_group_comparable": True, "source": "verified-groups", "group_size": 2}


def _store(path, contract=None, rows=("a", "b", "c", "d")):
    contract = _contract() if contract is None else contract
    store = EligibilityStore(path, contract, rows)
    store.qualify_feed(_qualified(contract))
    store.begin(window=0, now=100)
    return store


def _observation(contract, row_id="a", *, group="g", rewards=(0, 1), window=0):
    return {"schema": "prompt-observation/v1", "context_sha256": contract.context_sha256,
            "row_id": row_id, "group_id": group, "expected_samples": 2,
            "sample_ids": [f"sample-{i}" for i in range(len(rewards))],
            "rewards_bps": [None if r is None else round(r * 10000) for r in rewards],
            "tokens": [1] * len(rewards), "window": window,
            "verification": {"generation": "verified", "sampling": "verified", "grading": "graded"},
            "source_sha256": "1" * 64}


def test_epoch_rollover_preserves_source_and_retests_low_unknown_with_same_model(tmp_path):
    store = _store(tmp_path / "state.sqlite")
    original = store.contract.to_dict()
    for row_id, rewards in (("a", (1, 1)), ("b", (0, 0)), ("c", (None, 0)), ("d", (0, 1))):
        assert store.record(_observation(store.contract, row_id, rewards=rewards), now=100)
    assert store.snapshot(window=1, now=101)["status"] == "collecting"
    old = store.snapshot(window=2, now=102)
    assert (old["seen_unique"], old["coverage_bps"], old["groups_used"], old["category_counts"]["unknown"]) == (3, 7500, 4, 1)
    new = store.advance(expected_epoch=0, window=2, now=102)
    assert (new["dataset_epoch"], new["source_rows"], new["population_rows"], new["excluded_uniform_high"]) == (1, 4, 3, 1)
    assert store.eligible_rows(window=2, now=102) == ["b", "c", "d"]
    assert new["groups_used"] == 4 and new["tokens_used"] == 8
    assert store.row_state("a")["reason"] == "uniform-high"
    assert store.contract.to_dict() == original
    assert not {"optimizer", "lr_schedule_step", "training_run_id"} & new.keys()
    assert store.snapshot(epoch=0, window=99, now=10000) == old
    store.close()


def test_duplicate_groups_do_not_advance_unique_coverage_and_recovery_is_idempotent(tmp_path):
    path = tmp_path / "state.sqlite"
    store = _store(path)
    first = _observation(store.contract)
    assert store.record(first, now=100)
    assert not store.record(first, now=100)
    assert store.record(_observation(store.contract, group="other"), now=100)
    snapshot = store.snapshot(window=0, now=100)
    assert snapshot["seen_unique"] == 1 and snapshot["groups_used"] == 2
    changed = deepcopy(first)
    changed["tokens"][0] += 1
    with pytest.raises(ValueError, match="rebound"):
        store.record(changed, now=100)
    store.close()
    recovered = EligibilityStore(path, _contract(), ("a", "b", "c", "d"))
    assert recovered.begin(window=0, now=100) == snapshot
    assert not recovered.record(first, now=100)
    # A timeout opens a new finite pass. Repeating a transition after a lost
    # response cannot advance the epoch twice.
    rolled = recovered.advance(expected_epoch=0, window=10, now=110)
    assert recovered.advance(expected_epoch=0, window=10, now=110) == rolled
    assert rolled["dataset_epoch"] == 1
    assert recovered.snapshot(epoch=0, window=100, now=1000)["reason"] == "max-epoch-windows"
    recovered.close()


@pytest.mark.parametrize("field", ["context_sha256", "generation_verified", "sampling_verified", "rl_group_comparable"])
def test_new_context_needs_qualified_feed_and_never_infers_eval_rl_parity(tmp_path, field):
    store = EligibilityStore(tmp_path / "state.sqlite", _contract(), ("a",))
    feed = _qualified(store.contract)
    feed[field] = "2" * 64 if field == "context_sha256" else False
    with pytest.raises(ValueError, match="qualified"):
        store.qualify_feed(feed)
    with pytest.raises(ValueError, match="no qualified feed"):
        store.begin(window=0, now=100)
    store.close()


def test_checkpoint_adoption_has_its_own_feed_and_preserves_prior_snapshots(tmp_path):
    path = tmp_path / "state.sqlite"
    old = _store(path)
    old.record(_observation(old.contract), now=100)
    snapshot = old.snapshot(window=0, now=100)
    value = old.contract.to_dict()
    value["checkpoint"]["revision"] = "2" * 40
    new_contract = ServiceContract.from_dict(value)
    new = EligibilityStore(path, new_contract, ("a", "b", "c", "d"))
    with pytest.raises(ValueError, match="qualified"):
        new.qualify_feed(_qualified(old.contract))
    with pytest.raises(ValueError):
        new.begin(window=1, now=101)
    new.qualify_feed(_qualified(new_contract))
    assert new.begin(window=1, now=101)["seen_unique"] == 0
    assert old.snapshot(window=0, now=100) == snapshot
    with pytest.raises(ValueError, match="context"):
        new.record(_observation(old.contract), now=102)
    old.close()
    new.close()


def test_source_population_and_order_are_frozen_for_an_existing_context(tmp_path):
    path = tmp_path / "state.sqlite"
    store = _store(path)
    with pytest.raises(ValueError, match="source population"):
        EligibilityStore(path, store.contract, ("a", "b"))
    value = store.contract.to_dict()
    value["policies"]["eligibility"]["coverage_bps"] = 5000
    with pytest.raises(ValueError, match="another order"):
        EligibilityStore(path, ServiceContract.from_dict(value), ("a", "b", "c", "d"))
    with pytest.raises(ValueError, match="frozen"):
        store.qualify_feed({**_qualified(store.contract), "source": "different-source"})
    store.close()


@pytest.mark.parametrize("rewards,grading", [((None, 0), "graded"), ((0,), "graded"), ((0, 0), "error")])
def test_unknown_or_error_uses_budget_without_marking_row_seen(tmp_path, rewards, grading):
    store = _store(tmp_path / "state.sqlite")
    row = _observation(store.contract, rewards=rewards)
    row["verification"]["grading"] = grading
    assert store.record(row, now=100)
    snapshot = store.snapshot(window=0, now=100)
    assert snapshot["seen_unique"] == 0 and snapshot["groups_used"] == 1
    assert snapshot["category_counts"] == {"unknown": 1}
    assert "a" in store.eligible_rows(window=0, now=100)
    store.close()


@pytest.mark.parametrize("kind", ["generation", "sampling", "source", "window"])
def test_unqualified_or_foreign_observation_does_not_mutate_epoch(tmp_path, kind):
    store = _store(tmp_path / "state.sqlite")
    row = _observation(store.contract)
    if kind in {"generation", "sampling"}:
        row["verification"][kind] = "unverified"
    elif kind == "source":
        row["row_id"] = "foreign"
    else:
        row["window"] = 11
    before = store.snapshot(window=0, now=100)
    with pytest.raises(ValueError):
        store.record(row, now=100)
    if kind == "window":
        after = store.snapshot(window=11, now=100)
        assert after["groups_used"] == after["seen_unique"] == 0
        assert after["reason"] == "max-epoch-windows"
        with pytest.raises(ValueError, match="clock"):
            store.snapshot(window=0, now=100)
    else:
        assert store.snapshot(window=0, now=100) == before
    store.close()


@pytest.mark.parametrize("limit", ["max_groups", "max_tokens", "deadline_seconds"])
def test_order_budget_cannot_be_refunded_by_epoch_reset(tmp_path, limit):
    amount = 1 if limit == "max_groups" else 2
    store = _store(tmp_path / "state.sqlite", _contract(**{limit: amount}))
    if limit != "deadline_seconds":
        row = _observation(store.contract, rewards=(None, 0))
        assert store.record(row, now=100)
        assert not store.record(row, now=100)
    final = store.snapshot(window=10, now=102)
    assert final["status"] == "halted"
    assert store.eligible_rows(window=10, now=102) == []
    with pytest.raises(ValueError):
        store.advance(expected_epoch=0, window=10, now=102)
    with pytest.raises(ValueError):
        store.record(_observation(store.contract, group="new", window=10), now=102)
    store.close()


def test_all_saturated_population_stops_finitely_without_rewriting_dataset(tmp_path):
    store = _store(tmp_path / "state.sqlite", rows=("a",))
    store.record(_observation(store.contract, rewards=(1, 1)), now=100)
    new = store.advance(expected_epoch=0, window=2, now=102)
    assert new["status"] == "halted" and new["reason"] == "no-eligible-rows"
    assert (new["source_rows"], new["population_rows"]) == (1, 0)
    assert store.eligible_rows(window=2, now=102) == []
    with pytest.raises(ValueError):
        store.advance(expected_epoch=1, window=20, now=120)
    store.close()


def test_rollover_cannot_count_the_same_work_again_even_at_the_same_window(tmp_path):
    store = _store(tmp_path / "state.sqlite", rows=("a",))
    row = _observation(store.contract, rewards=(0, 0), window=2)
    store.record(row, now=102)
    new = store.advance(expected_epoch=0, window=2, now=102)
    assert new["dataset_epoch"] == 1
    assert not store.record(row, now=102)
    after = store.snapshot(window=2, now=102)
    assert (after["seen_unique"], after["groups_used"]) == (0, 1)
    assert store.record(_observation(store.contract, rewards=(0, 0), group="new-draw", window=2), now=102)
    store.close()


def test_recovery_never_reopens_after_a_backwards_clock_or_smaller_group(tmp_path):
    path = tmp_path / "state.sqlite"
    contract = _contract(deadline_seconds=10)
    store = _store(path, contract)
    smaller = _observation(contract)
    smaller["expected_samples"] = 3
    with pytest.raises(ValueError, match="qualified feed"):
        store.record(smaller, now=100)
    assert store.snapshot(window=1, now=110)["reason"] == "deadline"
    store.close()
    recovered = EligibilityStore(path, contract, ("a", "b", "c", "d"))
    with pytest.raises(ValueError, match="clock"):
        recovered.record(_observation(contract, window=1), now=102)
    assert recovered.snapshot(window=1, now=110)["status"] == "halted"
    recovered.close()


def test_rejected_late_observation_keeps_deadline_closed_after_restart(tmp_path):
    path = tmp_path / "state.sqlite"
    contract = _contract(deadline_seconds=10)
    store = _store(path, contract)
    with pytest.raises(ValueError, match="budget/deadline"):
        store.record(_observation(contract, window=1), now=110)
    assert store.snapshot(window=1, now=110)["groups_used"] == 0
    store.close()
    restored = EligibilityStore(path, contract, ("a", "b", "c", "d"))
    with pytest.raises(ValueError, match="clock"):
        restored.record(_observation(contract, window=1), now=101)
    assert restored.snapshot(window=1, now=110)["reason"] == "deadline"
    restored.close()
