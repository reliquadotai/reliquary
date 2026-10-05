"""Runtime qualification, durable projection and adopted-context boundaries."""

from concurrent.futures import ThreadPoolExecutor
import gc
import json
from pathlib import Path
import sqlite3
from threading import Event
import weakref
from unittest.mock import MagicMock

import pytest

from reliquary.protocol.service_contract import ServiceContract
from reliquary.services.observations import observation_id
from reliquary.services.runtime import ServiceRuntime, SUPPORTED_SERVICE_CAPABILITIES


def ordered(*, adaptive=False, max_groups=100, deadline=1000):
    value = json.loads((Path(__file__).parents[1] / "fixtures/service_contract_v1.json").read_text())
    value["service_kind"] = "adaptive_training"
    value["policies"]["checkpoint"] = {"kind": "trainer-driven/v1", "task_scoped": 1}
    value["policies"]["eligibility"] = {"kind": "dataset-epoch/v1", "coverage_bps": 7500,
                                        "refresh_windows": 2, "max_epoch_windows": 10}
    value["policies"]["reward"] = {"kind": "exploration-discount/v1", "divisor": 4,
                                   "budget_bps": 2000, "refresh_windows": 10, "max_tokens_per_group": 100}
    value["limits"] = {"max_groups": max_groups, "max_tokens": 10000, "deadline_seconds": deadline}
    if adaptive:
        value["policies"]["cooldown"] = {
            "kind": "adaptive-rotation/v1", "min_windows": 1, "max_windows": 1000,
            "margin_bps": 10000, "min_panel_groups": 2, "coverage_bps": 10000,
            "freshness_windows": 5, "smoothing_bps": 10000, "hysteresis_windows": 0,
            "max_change_windows": 1000, "fallback_windows": 7,
        }
    contract = ServiceContract.from_dict(value)
    qualification = {"schema": "service-runtime-qualification/v1", "qualified": True,
                     "contract_sha256": contract.sha256, "qualification_id": "approved-unit-runtime",
                     "supported_capabilities": sorted(SUPPORTED_SERVICE_CAPABILITIES),
                     "row_ids": ["c", "a", "b", "d"], "group_size": 2,
                     **{k: value[k] for k in ("dataset", "checkpoint", "environment", "generation_contract_sha256")}}
    return contract, qualification


@pytest.fixture(autouse=True)
def active_group_size(monkeypatch):
    monkeypatch.setattr("reliquary.constants.M_ROLLOUTS", 2)
    monkeypatch.delenv("RELIQUARY_SERVICE_PANEL", raising=False)


def row(contract, row_id="a", *, group="draw-1", window=1, rewards=(0, 0)):
    return {"schema": "prompt-observation/v1", "context_sha256": contract.context_sha256,
            "row_id": row_id, "group_id": group, "expected_samples": 2, "sample_ids": ["s0", "s1"],
            "rewards_bps": list(rewards), "tokens": [1, 1], "window": window,
            "verification": {"generation": "verified", "sampling": "verified", "grading": "graded"},
            "source_sha256": "e" * 64}


def record(runtime, observation, *, now=1, purpose="exploration"):
    runtime.open_window(observation["window"], window_pool=1, slots=10)
    return runtime.record_verified(observation, hotkey="unit-miner", purpose=purpose,
                                   window_pool=1, slots=10, now=now)


def test_startup_projection_and_restart_use_actual_frozen_index_map(tmp_path):
    contract, qualification = ordered()
    path = tmp_path / "runtime.sqlite3"
    runtime = ServiceRuntime(path, contract, qualification, now=0)
    record(runtime, row(contract))
    view = runtime.prepare_view(window=1, now=1)
    assert view.blocked_indices() == [1]
    assert view.snapshot()["eligibility"]["seen_unique"] == 1
    assert (tmp_path / "eligibility.sqlite3").is_file()
    runtime.close()
    recovered = ServiceRuntime(path, contract, qualification, now=2)
    restored = recovered.prepare_view(window=1, now=2)
    assert restored.blocked_indices() == [1]
    assert restored.snapshot()["eligibility"]["groups_used"] == 1
    assert record(recovered, row(contract), now=2)["inserted"] is False
    assert restored.snapshot()["eligibility"]["groups_used"] == 1
    recovered.close()


def test_boundary_projection_and_proof_callbacks_share_the_view_lock(tmp_path):
    contract, qualification = ordered()
    runtime = ServiceRuntime(tmp_path / "runtime.sqlite3", contract, qualification, now=0)
    view = runtime.prepare_view(window=1, now=1)

    def callback(index):
        if index % 2:
            return record(runtime, row(contract, group=f"proof-{index}"), now=1)
        return runtime.prepare_view(window=1, now=1).snapshot()

    with ThreadPoolExecutor(max_workers=6) as workers:
        list(workers.map(callback, range(48)))
    runtime.prepare_view(window=1, now=1)
    assert view.snapshot()["eligibility"]["groups_used"] == 24
    assert view.snapshot()["eligibility"]["seen_unique"] == 1
    assert view.blocked_indices() == [1]
    assert runtime.db.execute("SELECT groups FROM service_orders").fetchone()[0] == 24
    runtime.close()


def test_adopted_context_retains_old_snapshot_and_original_order_budget(tmp_path):
    contract, qualification = ordered(max_groups=1)
    runtime = ServiceRuntime(tmp_path / "runtime.sqlite3", contract, qualification, now=0)
    old = runtime.prepare_view(window=1, now=1)
    record(runtime, row(contract))
    old_snapshot = old.snapshot()
    changed = runtime.adopt(repo=contract.to_dict()["checkpoint"]["repo"], revision="f" * 40, sha256="f" * 64)
    new = runtime.prepare_view(window=2, now=2)
    assert new.contract == changed and new is not old
    assert new.snapshot()["eligibility"]["seen_unique"] == 0
    assert new.blocked_indices() == [0, 1, 2, 3]
    assert old.snapshot() == old_snapshot
    assert runtime.order_contract == contract
    runtime.close()


def test_adoption_releases_old_views_and_reopens_their_persisted_projection(tmp_path):
    contract, qualification = ordered()
    runtime = ServiceRuntime(tmp_path / "runtime.sqlite3", contract, qualification, now=0)
    record(runtime, row(contract))
    view = runtime.prepare_view(window=1, now=1)
    original = view.snapshot()
    for index in range(1, 9):
        store, previous = view.eligibility, weakref.ref(view)
        runtime.adopt(repo=contract.to_dict()["checkpoint"]["repo"],
                      revision=f"{index:040x}", sha256=f"{index:064x}")
        assert runtime.view is None
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            store.source_rows()
        del view
        gc.collect()
        assert previous() is None
        view = runtime.prepare_view(window=1, now=1)
        assert runtime.view is view
    assert runtime.db.execute("SELECT COUNT(*) FROM service_contexts").fetchone()[0] == 9

    checkpoint = contract.to_dict()["checkpoint"]
    runtime.adopt(repo=checkpoint["repo"], revision=checkpoint["revision"], sha256=checkpoint["sha256"])
    reopened = runtime.prepare_view(window=1, now=2)
    assert reopened.blocked_indices() == [1]
    assert reopened.snapshot()["eligibility"]["seen_unique"] == original["eligibility"]["seen_unique"] == 1
    assert reopened.snapshot()["eligibility"]["groups_used"] == 1
    assert record(runtime, row(contract), now=2)["inserted"] is False
    assert reopened.snapshot()["eligibility"]["groups_used"] == 1
    reopened_store = reopened.eligibility
    runtime.close()
    assert runtime.view is None
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        reopened_store.source_rows()


def test_adoption_waits_for_committed_observation_projection(tmp_path, monkeypatch):
    contract, qualification = ordered()
    runtime = ServiceRuntime(tmp_path / "runtime.sqlite3", contract, qualification, now=0)
    view = runtime.prepare_view(window=1, now=1)
    projecting, release, adopting = Event(), Event(), Event()
    original = view.eligibility.record

    def projection(*args, **kwargs):
        projecting.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    def adoption():
        adopting.set()
        return runtime.adopt(repo=contract.to_dict()["checkpoint"]["repo"],
                             revision="f" * 40, sha256="f" * 64)

    monkeypatch.setattr(view.eligibility, "record", projection)
    with ThreadPoolExecutor(max_workers=2) as workers:
        callback = workers.submit(record, runtime, row(contract))
        assert projecting.wait(5)
        transition = workers.submit(adoption)
        assert adopting.wait(5)
        assert not transition.done()
        release.set()
        assert callback.result(timeout=5)["inserted"] is True
        assert transition.result(timeout=5).context_sha256 != contract.context_sha256
    assert view.snapshot()["eligibility"]["seen_unique"] == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        view.eligibility.source_rows()
    runtime.close()


def test_prepare_deadline_survives_backwards_clock_after_checkpoint_adoption(tmp_path):
    contract, qualification = ordered(deadline=2)
    path = tmp_path / "runtime.sqlite3"
    runtime = ServiceRuntime(path, contract, qualification, now=0)
    old = runtime.prepare_view(window=1, now=2)
    assert old.blocked_indices() == [0, 1, 2, 3]
    assert runtime.db.execute("SELECT clock FROM service_orders").fetchone()[0] == 2
    runtime.adopt(repo=contract.to_dict()["checkpoint"]["repo"], revision="f" * 40, sha256="f" * 64)
    new = runtime.prepare_view(window=2, now=1)
    assert new.blocked_indices() == [0, 1, 2, 3]
    assert new.snapshot()["eligibility"]["reason"] == "deadline"
    runtime.close()
    recovered = ServiceRuntime(path, contract, qualification, now=0)
    assert recovered.prepare_view(window=1, now=1).blocked_indices() == [0, 1, 2, 3]
    recovered.close()


@pytest.mark.parametrize("size", [None, True, 3])
def test_runtime_group_qualification_cannot_infer_or_override_actual_m(tmp_path, size):
    contract, qualification = ordered()
    qualification["group_size"] = size
    runtime = ServiceRuntime(tmp_path / "runtime.sqlite3", contract, qualification, now=0)
    with pytest.raises(ValueError, match="group_size|group size"):
        runtime.prepare_view(window=1, now=1)
    assert runtime.view is None
    runtime.close()


def test_public_sampling_group_size_is_checked_against_active_runtime(tmp_path):
    contract, qualification = ordered()
    value = contract.to_dict()
    value["policies"]["sampling"] = {"kind": "public-group-pool/v1", "group_size": 3,
                                      "pool_groups": 4, "renewal_windows": 10}
    contract = ServiceContract.from_dict(value)
    qualification["contract_sha256"] = contract.sha256
    runtime = ServiceRuntime(tmp_path / "runtime.sqlite3", contract, qualification, now=0)
    with pytest.raises(ValueError, match="group size"):
        runtime.prepare_view(window=1, now=1)
    runtime.close()


def test_missing_panel_uses_ordered_fallback_not_selected_training_fraction(tmp_path):
    contract, qualification = ordered(adaptive=True)
    runtime = ServiceRuntime(tmp_path / "runtime.sqlite3", contract, qualification, now=0)
    view = runtime.prepare_view(window=1, now=1, distinct_groups_per_window=10)
    assert view.cooldown_windows == 7
    assert view.snapshot()["cooldown_reasons"] == ["missing-panel"]
    runtime.close()


def panel_document(contract, *, window=1):
    observations = [row(contract, row_id, rewards=(0, 10000), window=window) for row_id in ("a", "b")]
    population = {**contract.to_dict()["dataset"], "kind": "source", "size": 4}
    panel = {"panel_id": "declared-panel", "context_sha256": contract.context_sha256,
             "population": population, "group_size": 2, "declared_window": window,
             "expected_group_ids": [observation_id(item) for item in observations],
             "observations": observations, "independent": True, "qualified": True}
    return {"population": population, "panel": panel}


def test_panel_file_matches_runtime_context_source_and_full_group_size(tmp_path, monkeypatch):
    contract, qualification = ordered(adaptive=True)
    runtime = ServiceRuntime(tmp_path / "runtime.sqlite3", contract, qualification, now=0)
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(panel_document(contract)))
    monkeypatch.setenv("RELIQUARY_SERVICE_PANEL", str(path))
    view = runtime.prepare_view(window=1, now=1, distinct_groups_per_window=1)
    assert view.cooldown_windows == 4 and not view.snapshot()["cooldown_fallback"]
    foreign = panel_document(contract, window=2)
    foreign["panel"]["context_sha256"] = "f" * 64
    path.write_text(json.dumps(foreign))
    with pytest.raises(ValueError, match="context/population"):
        runtime.prepare_view(window=2, now=2, distinct_groups_per_window=1)
    assert view.blocked_indices() == [0, 1, 2, 3]
    runtime.close()


@pytest.mark.parametrize("change", ["relative", "oversize", "group", "source"])
def test_invalid_panel_file_closes_existing_admission_projection(tmp_path, monkeypatch, change):
    contract, qualification = ordered(adaptive=True)
    runtime = ServiceRuntime(tmp_path / "runtime.sqlite3", contract, qualification, now=0)
    view = runtime.prepare_view(window=1, now=1)
    path = tmp_path / "panel.json"
    document = panel_document(contract, window=2)
    if change == "group":
        document["panel"]["group_size"] = 3
    if change == "source":
        document["panel"]["observations"][0]["row_id"] = "foreign"
    path.write_text(" " * (4 * 1024 * 1024 + 1) if change == "oversize" else json.dumps(document))
    monkeypatch.setenv("RELIQUARY_SERVICE_PANEL", "panel.json" if change == "relative" else str(path))
    with pytest.raises(ValueError):
        runtime.prepare_view(window=2, now=2, distinct_groups_per_window=1)
    assert view.blocked_indices() == [0, 1, 2, 3]
    runtime.close()


def test_unprojected_old_fact_cannot_be_reassigned_to_new_epoch(tmp_path, monkeypatch):
    contract, qualification = ordered()
    runtime = ServiceRuntime(tmp_path / "runtime.sqlite3", contract, qualification, now=0)
    view = runtime.prepare_view(window=1, now=1)
    monkeypatch.setattr(view, "observe", MagicMock(side_effect=OSError("unit disk failure")))
    with pytest.raises(OSError):
        record(runtime, row(contract))
    view.refresh(window=2, now=2)
    with pytest.raises(ValueError, match="clock"):
        runtime.prepare_view(window=2, now=2)
    assert view.blocked_indices() == [0, 1, 2, 3]
    runtime.close()


@pytest.mark.parametrize("change", ["member", "digest", "id"])
def test_active_panel_cannot_borrow_source_probability_or_another_slice(tmp_path, monkeypatch, change):
    contract, qualification = ordered(adaptive=True)
    runtime = ServiceRuntime(tmp_path / "runtime.sqlite3", contract, qualification, now=0)
    view = runtime.prepare_view(window=1, now=1)
    for row_id in ("a", "b", "c", "d"):
        record(runtime, row(contract, row_id, rewards=(10000, 10000) if row_id == "a" else (0, 0)))
    runtime.prepare_view(window=3, now=3)
    population = view.snapshot()["active_population"]
    assert population["size"] == 3 and population["id"] == "epoch-1"
    document = panel_document(contract, window=4)
    document["population"] = population
    document["panel"]["population"] = population
    observations = [row(contract, "a" if change == "member" else "b", window=4),
                    row(contract, "c", window=4, rewards=(0, 10000))]
    document["panel"]["observations"] = observations
    document["panel"]["expected_group_ids"] = [observation_id(item) for item in observations]
    if change == "digest":
        population["sha256"] = "f" * 64
    if change == "id":
        population["id"] = "other-slice"
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(document))
    monkeypatch.setenv("RELIQUARY_SERVICE_PANEL", str(path))
    with pytest.raises(ValueError, match="active epoch"):
        runtime.prepare_view(window=4, now=4, distinct_groups_per_window=1)
    assert view.blocked_indices() == [0, 1, 2, 3]
    runtime.close()
