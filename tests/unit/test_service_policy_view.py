"""Admission projections, durable retries and window-bound policy changes."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path

import pytest

from reliquary.protocol.service_contract import ServiceContract
from reliquary.services.eligibility import EligibilityStore
from reliquary.services.observations import observation_id
from reliquary.services.runtime_view import ServicePolicyView


ROWS = ("a", "b", "c", "d")


def _contract(*, adaptive=False, limits=None, cooldown=None):
    value = json.loads((Path(__file__).parents[1] / "fixtures/service_contract_v1.json").read_text())
    value["service_kind"] = "adaptive_training"
    value["policies"]["checkpoint"] = {"kind": "trainer-driven/v1", "task_scoped": 1}
    value["policies"]["eligibility"] = {
        "kind": "dataset-epoch/v1", "coverage_bps": 7500,
        "refresh_windows": 2, "max_epoch_windows": 10,
    }
    if adaptive:
        value["policies"]["cooldown"] = {
            "kind": "adaptive-rotation/v1", "min_windows": 1, "max_windows": 100,
            "margin_bps": 10000, "min_panel_groups": 2, "coverage_bps": 10000,
            "freshness_windows": 5, "smoothing_bps": 10000, "hysteresis_windows": 0,
            "max_change_windows": 100, "fallback_windows": 7, **(cooldown or {}),
        }
    value["limits"].update(limits or {})
    return ServiceContract.from_dict(value)


def _qualified(contract):
    return {"context_sha256": contract.context_sha256, "generation_verified": True,
            "sampling_verified": True, "rl_group_comparable": True, "group_size": 2}


def _store(path, contract=None):
    store = EligibilityStore(path, contract or _contract(), ROWS)
    store.qualify_feed(_qualified(store.contract))
    return store


def _row(contract, row_id="a", *, group="group-1", rewards=(0, 10000), window=0):
    return {"schema": "prompt-observation/v1", "context_sha256": contract.context_sha256,
            "row_id": row_id, "group_id": group, "expected_samples": 2,
            "sample_ids": [f"sample-{i}" for i in range(len(rewards))],
            "rewards_bps": list(rewards), "tokens": [1] * len(rewards), "window": window,
            "verification": {"generation": "verified", "sampling": "verified", "grading": "graded"},
            "source_sha256": "1" * 64}


def _population(contract, *, active=False, size=4):
    dataset = contract.to_dict()["dataset"]
    return {"id": "active-slice" if active else dataset["id"],
            "kind": "active" if active else "source",
            "sha256": "f" * 64 if active else dataset["sha256"], "size": size}


def _panel(contract, population, *, window=0, rewards=((0, 10000), (0, 10000))):
    rows = [_row(contract, row_id, group="audit", rewards=scores, window=window)
            for row_id, scores in zip(("c", "d"), rewards)]
    return {"panel_id": f"panel-{window}", "context_sha256": contract.context_sha256,
            "population": dict(population), "group_size": 2, "declared_window": window,
            "expected_group_ids": [observation_id(row) for row in rows],
            "observations": rows, "independent": True, "qualified": True}


def test_frozen_prompt_index_order_is_not_the_stores_sorted_row_order(tmp_path):
    store = _store(tmp_path / "state.sqlite")
    view = ServicePolicyView(store.contract, ("d", "b", "a", "c"), store)
    assert store.source_rows() == list(ROWS)
    assert view.blocked_indices() == [0, 1, 2, 3]
    view.refresh(window=0, now=100)
    assert view.observe(_row(store.contract, "c"), window=0, now=100)
    assert view.blocked_indices() == [3]
    assert [view.eligible(index) for index in range(4)] == [True, True, True, False]
    assert not any(view.eligible(index) for index in (-1, 4, True, "0", 0.0))
    store.close()


@pytest.mark.parametrize("rows", [["a", "b", "c", "d"], (), ("a", "a"), ("a", "b")])
def test_constructor_rejects_ambiguous_or_foreign_source_index_map(tmp_path, rows):
    store = _store(tmp_path / "state.sqlite")
    with pytest.raises(ValueError):
        ServicePolicyView(store.contract, rows, store)
    store.close()


def test_unqualified_epoch_stays_closed_until_its_feed_is_qualified(tmp_path):
    contract = _contract()
    with pytest.raises(ValueError, match="qualified"):
        ServicePolicyView(contract, ROWS)
    store = EligibilityStore(tmp_path / "state.sqlite", contract, ROWS)
    view = ServicePolicyView(contract, ROWS, store)
    with pytest.raises(ValueError, match="qualified"):
        view.refresh(window=0, now=100)
    assert view.blocked_indices() == [0, 1, 2, 3]
    store.qualify_feed(_qualified(contract))
    assert view.refresh(window=0, now=100)["eligible_rows"] == 4
    store.close()


def test_unknown_consumes_budget_but_remains_eligible_and_rollover_is_idempotent(tmp_path):
    store = _store(tmp_path / "state.sqlite")
    view = ServicePolicyView(store.contract, ROWS, store)
    view.refresh(window=0, now=100)
    for row_id, rewards in (("a", (10000, 10000)), ("b", (0, 0)),
                            ("c", (None, 0)), ("d", (0, 10000))):
        view.observe(_row(store.contract, row_id, rewards=rewards), window=0, now=100)
    first = view.snapshot()
    assert first["eligibility"]["groups_used"] == 4
    assert first["eligibility"]["seen_unique"] == 3
    assert view.blocked_indices() == [0, 1, 3] and view.eligible(2)
    assert view.refresh(window=1, now=101)["eligibility"]["dataset_epoch"] == 0
    advanced = view.advance(expected_epoch=0, window=2, now=102)
    assert advanced["eligibility"]["dataset_epoch"] == 1
    assert advanced["eligibility"]["population_rows"] == 3
    assert advanced["eligibility"]["groups_used"] == 4
    assert view.blocked_indices() == [0]
    assert view.advance(expected_epoch=0, window=2, now=102) == advanced
    assert store.contract == view.contract
    store.close()


def test_worker_threads_deduplicate_observations_and_preserve_clock_on_restart(tmp_path):
    path = tmp_path / "state.sqlite"
    store = _store(path)
    view = ServicePolicyView(store.contract, ROWS, store)
    view.refresh(window=0, now=100)
    rows = [_row(store.contract, row_id, group=f"group-{row_id}") for row_id in ("a", "b")]
    jobs = [(rows[index % 2], 100 + (index % 4) / 10) for index in range(24)]
    with ThreadPoolExecutor(max_workers=6) as workers:
        inserted = list(workers.map(lambda job: view.observe(job[0], window=0, now=job[1]), jobs))
    assert sum(inserted) == 2
    assert view.snapshot()["eligibility"]["groups_used"] == 2
    assert view.snapshot()["eligibility"]["tokens_used"] == 4
    assert view.blocked_indices() == [0, 1]
    contract = store.contract
    store.close()
    recovered_store = EligibilityStore(path, contract, ROWS)
    recovered = ServicePolicyView(contract, ROWS, recovered_store)
    assert recovered.blocked_indices() == [0, 1, 2, 3]
    assert recovered.refresh(window=0, now=101)["eligible_rows"] == 2
    assert recovered.blocked_indices() == [0, 1]
    assert not recovered.observe(rows[0], window=0, now=100)
    recovered_store.close()


def test_admission_reads_do_no_sql_and_snapshots_do_not_alias_internal_state(tmp_path):
    contract = _contract(adaptive=True)
    store = _store(tmp_path / "state.sqlite", contract)
    view = ServicePolicyView(contract, ROWS, store)
    population = _population(contract)
    view.refresh(window=0, now=100, population=population, panel=_panel(contract, population),
                 distinct_groups_per_window=1)
    before = view.snapshot()
    queries = []
    store.db.set_trace_callback(queries.append)
    snapshot = view.snapshot()
    snapshot["cooldown_reasons"].append("changed")
    snapshot["eligibility"]["category_counts"]["unknown"] = 99
    snapshot["cooldown_proposal"]["population"]["size"] = 0
    snapshot["cooldown_proposal"]["measurement"]["completed_group_ids"].clear()
    snapshot["cooldown_proposal"]["state"]["ema_windows_bps"] = None
    assert all(view.eligible(index) for index in range(4))
    assert view.blocked_indices() == []
    assert view.cooldown_windows == 4 and view.revision == before["revision"]
    assert view.snapshot() == before and queries == []
    store.db.set_trace_callback(None)
    store.close()


@pytest.mark.parametrize("operation", ["refresh", "observe"])
def test_membership_read_error_closes_and_invalidates_then_same_window_retry_recovers(tmp_path, monkeypatch, operation):
    store = _store(tmp_path / "state.sqlite")
    view = ServicePolicyView(store.contract, ROWS, store)
    view.refresh(window=0, now=100)
    row = _row(store.contract)
    before = view.revision

    def fail(**kwargs):
        raise RuntimeError("membership unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(store, "eligible_rows", fail)
        with pytest.raises(RuntimeError, match="membership unavailable"):
            if operation == "refresh":
                view.refresh(window=0, now=100)
            else:
                view.observe(row, window=0, now=100)
    closed_revision = view.revision
    assert view.blocked_indices() == [0, 1, 2, 3] and closed_revision > before
    if operation == "refresh":
        assert view.refresh(window=0, now=100)["eligible_rows"] == 4
    else:
        assert store.row_state("a")["seen_in_epoch"]
        assert not view.observe(row, window=0, now=100)
        assert view.blocked_indices() == [0]
        assert view.snapshot()["eligibility"]["groups_used"] == 1
    assert view.revision > closed_revision
    store.close()


@pytest.mark.parametrize("operation", ["refresh", "observe"])
def test_write_failure_retry_repairs_durable_frontier_without_refunding_group(tmp_path, monkeypatch, operation):
    path = tmp_path / "state.sqlite"
    store = _store(path)
    view = ServicePolicyView(store.contract, ROWS, store)
    view.refresh(window=0, now=100)
    row = _row(store.contract)

    def fail():
        raise RuntimeError("view write unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(view, "_persist", fail)
        with pytest.raises(RuntimeError, match="view write unavailable"):
            if operation == "refresh":
                view.refresh(window=1, now=101)
            else:
                view.observe(row, window=0, now=101)
    assert view.blocked_indices() == [0, 1, 2, 3]
    if operation == "refresh":
        expected_window, expected_groups, expected_blocked = 1, 0, []
        view.refresh(window=1, now=101)
    else:
        expected_window, expected_groups, expected_blocked = 0, 1, [0]
        assert not view.observe(row, window=0, now=101)
    saved = json.loads(store.db.execute("SELECT payload FROM service_policy_views").fetchone()[0])
    assert saved["window"] == expected_window and saved["last_now"] == 101
    contract = store.contract
    store.close()
    recovered_store = EligibilityStore(path, contract, ROWS)
    recovered = ServicePolicyView(contract, ROWS, recovered_store)
    assert recovered.refresh(window=expected_window, now=101)["eligibility"]["groups_used"] == expected_groups
    assert recovered.blocked_indices() == expected_blocked
    recovered_store.close()


@pytest.mark.parametrize("operation", ["refresh", "duplicate-observe"])
def test_deadline_same_window_invalidates_mask_and_remains_closed_after_restart(tmp_path, operation):
    path = tmp_path / "state.sqlite"
    contract = _contract(limits={"deadline_seconds": 2})
    store = _store(path, contract)
    view = ServicePolicyView(contract, ROWS, store)
    view.refresh(window=0, now=100)
    row = _row(contract)
    if operation == "duplicate-observe":
        assert view.observe(row, window=0, now=100)
    before = view.revision
    if operation == "refresh":
        view.refresh(window=0, now=102)
    else:
        assert not view.observe(row, window=0, now=102)
    assert view.revision > before and view.blocked_indices() == [0, 1, 2, 3]
    assert view.snapshot()["eligibility"]["reason"] == "deadline"
    store.close()
    recovered_store = EligibilityStore(path, contract, ROWS)
    recovered = ServicePolicyView(contract, ROWS, recovered_store)
    assert recovered.refresh(window=0, now=102)["eligible_rows"] == 0
    with pytest.raises(ValueError, match="clock"):
        recovered.refresh(window=0, now=100)
    assert recovered.blocked_indices() == [0, 1, 2, 3]
    assert recovered.refresh(window=1, now=103)["eligible_rows"] == 0
    recovered_store.close()


def test_cooldown_evidence_is_idempotent_at_frontier_and_changes_at_next_boundary(tmp_path):
    contract = _contract(adaptive=True)
    store = _store(tmp_path / "state.sqlite", contract)
    view = ServicePolicyView(contract, ROWS, store)
    population = _population(contract)
    panel = _panel(contract, population)
    first = view.refresh(window=0, now=100, population=population, panel=panel,
                         distinct_groups_per_window=1)
    assert first["cooldown_windows"] == 4
    assert view.refresh(window=0, now=100, population=population, panel=panel,
                        distinct_groups_per_window=1) == first
    changed = deepcopy(panel)
    changed["panel_id"] = "other-panel"
    with pytest.raises(ValueError, match="different evidence"):
        view.refresh(window=0, now=100, population=population, panel=changed,
                     distinct_groups_per_window=1)
    assert view.blocked_indices() == [0, 1, 2, 3]
    repaired = view.refresh(window=0, now=100, population=population, panel=panel,
                            distinct_groups_per_window=1)
    assert repaired["cooldown_proposal"] == first["cooldown_proposal"]
    assert repaired["eligible_rows"] == 4
    next_panel = _panel(contract, population, window=1, rewards=((0, 10000), (0, 0)))
    next_window = view.refresh(window=1, now=101, population=population, panel=next_panel,
                               distinct_groups_per_window=1)
    assert next_window["cooldown_windows"] == 2
    missing = view.refresh(window=2, now=102)
    assert missing["cooldown_windows"] == 7 and missing["cooldown_fallback"]
    assert missing["cooldown_proposal"]["state"]["ema_windows_bps"] is None
    store.close()


def test_population_rotation_uses_source_or_active_count_instead_of_remaining_unseen_rows(tmp_path):
    contract = _contract(adaptive=True)
    store = _store(tmp_path / "state.sqlite", contract)
    view = ServicePolicyView(contract, ROWS, store)
    view.refresh(window=0, now=100)
    for row_id, scores in (("a", (10000, 10000)), ("b", (0, 0)),
                           ("c", (None, 0)), ("d", (0, 10000))):
        view.observe(_row(contract, row_id, rewards=scores), window=0, now=100)
    view.advance(expected_epoch=0, window=2, now=102)
    view.observe(_row(contract, "b", group="second-pass", window=2), window=2, now=102)
    assert view.snapshot()["eligibility"]["population_rows"] == 3
    assert view.snapshot()["eligible_rows"] == 2
    for window, active, size in ((3, False, 4), (4, True, 3)):
        population = view.snapshot()["active_population"] if active else _population(contract, size=size)
        result = view.refresh(window=window, now=100 + window, population=population,
                              panel=_panel(contract, population, window=window),
                              distinct_groups_per_window=1)
        assert result["cooldown_windows"] == size
        assert result["cooldown_proposal"]["measurement"]["p_sample"] == {"numerator": 2, "denominator": 2}
    wrong = _population(contract, active=True, size=2)
    with pytest.raises(ValueError, match="population count"):
        view.refresh(window=5, now=105, population=wrong, panel=_panel(contract, wrong, window=5),
                     distinct_groups_per_window=1)
    assert view.blocked_indices() == [0, 1, 2, 3]
    store.close()


def test_active_panel_uses_exact_epoch_membership_including_seen_rows(tmp_path):
    contract = _contract(adaptive=True)
    store = _store(tmp_path / "state.sqlite", contract)
    view = ServicePolicyView(contract, ROWS, store)
    view.refresh(window=0, now=100)
    for row_id, scores in (("a", (10000, 10000)), ("b", (0, 0)), ("c", (0, 10000))):
        view.observe(_row(contract, row_id, rewards=scores), window=0, now=100)
    view.refresh(window=2, now=102)
    view.observe(_row(contract, "b", group="new-epoch", window=2), window=2, now=102)
    population = view.snapshot()["active_population"]
    assert population["size"] == 3 and store.active_rows() == ["b", "c", "d"]
    panel = _panel(contract, population, window=3)
    # A seen row is still a member of the active population.
    panel["observations"][0]["row_id"] = "b"
    panel["expected_group_ids"] = [observation_id(row) for row in panel["observations"]]
    assert not view.refresh(window=3, now=103, population=population, panel=panel,
                            distinct_groups_per_window=1)["cooldown_fallback"]
    panel = _panel(contract, population, window=4)
    panel["observations"][0]["row_id"] = "a"
    panel["expected_group_ids"] = [observation_id(row) for row in panel["observations"]]
    with pytest.raises(ValueError, match="outside its active"):
        view.refresh(window=4, now=104, population=population, panel=panel, distinct_groups_per_window=1)
    assert view.blocked_indices() == [0, 1, 2, 3]
    store.close()


def test_legacy_all_rows_static_view_needs_no_sql_store():
    value = json.loads((Path(__file__).parents[1] / "fixtures/service_contract_v1.json").read_text())
    contract = ServiceContract.from_dict(value)
    view = ServicePolicyView(contract, ROWS)
    assert view.blocked_indices() == [0, 1, 2, 3]
    assert view.refresh(window=0, now=100)["eligible_rows"] == 4
    assert view.cooldown_windows == value["policies"]["cooldown"]["windows"]
    assert view.refresh(window=1, now=101)["cooldown_fallback"] is False
    assert view.blocked_indices() == []
