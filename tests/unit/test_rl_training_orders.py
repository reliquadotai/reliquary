"""A training order's registry fence stops new leases, never an already issued lease's drainage."""
from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from reliquary.shared.task_registry import TaskEntry
from reliquary.validator.rl_training_orders import SNAPSHOT_TTL_SECONDS, TrainingAdmission, load_training_entry
from tests.unit.test_rl_auction import ENV, ask, auctioned_book


@pytest.fixture
def setup(tmp_path):
    book, auctions, clock = auctioned_book(tmp_path)
    entry = TaskEntry("order-train-t1", "p", "a" * 64, "rl-discovered-price", {"training_order": {"id": "t1"}},
                      "active", None)
    state = {"entry": entry}

    async def load(task_id):
        assert task_id == entry.task_id
        if state.get("error"):
            raise OSError("registry unavailable")
        return state["entry"]

    gate = TrainingAdmission(entry, book, load=load, clock=clock)
    auctions.admission = gate.reason
    yield book, auctions, clock, entry, state, gate, load
    book.close()


def test_no_lease_until_a_fresh_matching_registry_entry_is_seen(setup):
    book, auctions, clock, entry, state, gate, load = setup
    auctions.post_ask(ENV, "A", .5, 8)
    assert gate.reason() == "registry_unavailable"
    assert not book.grant([ask("A", [1])])[0].leases
    assert asyncio.run(gate.refresh())
    assert book.grant([ask("A", [1])])[0].leases
    clock.t += SNAPSHOT_TTL_SECONDS
    assert gate.reason() == "registry_unavailable"
    assert not book.grant([ask("A", [2])])[0].leases


def test_refresh_failure_fences_immediately_and_recovers_only_after_a_good_read(setup):
    _, _, clock, _, state, gate, _ = setup
    assert asyncio.run(gate.refresh())
    state["error"] = True
    assert not asyncio.run(gate.refresh()) and gate.reason() == "registry_unavailable"
    state["error"] = False
    assert asyncio.run(gate.refresh())
    clock.t -= 1
    assert gate.reason() == "registry_unavailable"


def test_retirement_survives_reactivation_restart_and_leaves_issued_leases_submittable(setup):
    book, auctions, clock, entry, state, gate, load = setup
    auctions.post_ask(ENV, "A", .5, 8)
    assert asyncio.run(gate.refresh())
    lease = book.grant([ask("A", [1])])[0].leases[0]
    state["entry"] = replace(entry, status="retired", retired_at=1)
    assert not asyncio.run(gate.refresh()) and gate.reason() == "registry_retired"
    assert not book.grant([ask("A", [2])])[0].leases
    assert book.submit(lease.lease_id, "A", in_zone=True).state == "submitted"
    state["entry"] = entry
    assert not asyncio.run(gate.refresh())
    restarted = TrainingAdmission(entry, book, load=load, clock=clock)
    assert not asyncio.run(restarted.refresh()) and restarted.reason() == "registry_retired"


@pytest.mark.parametrize("change", [{"profile_sha256": "b" * 64}, {"params": {"training_order": {"id": "t2"}}},
                                  {"env_split": {"other": 1.0}}, {"status": "pending"}])
def test_a_changed_entry_permanently_fences_the_original_runtime(setup, change):
    book, _, clock, entry, state, gate, load = setup
    state["entry"] = replace(entry, **change)
    assert not asyncio.run(gate.refresh()) and gate.reason() == "registry_changed"
    state["entry"] = entry
    assert not asyncio.run(gate.refresh())
    assert TrainingAdmission(entry, book, load=load, clock=clock).reason() == "registry_changed"


def test_pause_is_reversible_but_a_missing_task_is_terminal(setup):
    _, _, _, entry, state, gate, _ = setup
    state["entry"] = replace(entry, admission="paused")
    assert not asyncio.run(gate.refresh()) and gate.reason() == "registry_paused"
    state["entry"] = entry
    assert asyncio.run(gate.refresh())
    state["entry"] = None
    assert not asyncio.run(gate.refresh()) and gate.reason() == "registry_changed"


def test_the_registry_reader_is_strict(monkeypatch):
    async def read_registry(*, strict):
        assert strict is True
        return {"t1": "entry"}, None

    monkeypatch.setattr("reliquary.infrastructure.task_registry_store.read_registry", read_registry)
    assert asyncio.run(load_training_entry("t1")) == "entry"


def test_registry_reads_have_a_timeout_and_stalled_reads_cannot_extend_a_snapshot(setup, monkeypatch):
    _, _, clock, _, _, gate, _ = setup
    assert asyncio.run(gate.refresh())

    async def stalled(task_id):
        clock.t += SNAPSHOT_TTL_SECONDS
        assert gate.reason() == "registry_unavailable"
        await asyncio.Future()

    gate.load = stalled
    monkeypatch.setattr("reliquary.validator.rl_training_orders.READ_TIMEOUT_SECONDS", .01)
    assert not asyncio.run(gate.refresh()) and gate.reason() == "registry_unavailable"
