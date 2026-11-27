"""A task's payout rail: ``native`` (the weights pay it, the default) or
``collector`` (the payer pays it from ``payout_rail_from``, and the weight
setter gives those periods no weight). Chosen when the task is added; no write
after changes it, and declaring one asks that the fleet knows the rail."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from reliquary.validator import corpus_periods as cp
from tests.unit.test_jobs_cli import _create_args, _rl_entry, bucket, registry  # noqa: F401

PAID, NATIVE = "order-gen-collector", "order-gen-native"
WORK = 23_817


def _entry(task_id=PAID, mechanism="corpus-generation", **params):
    from reliquary.shared.task_registry import TaskEntry

    base = {"start": 0.05, "decay": 0.98, "rounds_per_step": 1, "deadband": 0.0, "snap": 0.0, "floor": 0.05,
            "cap": 0.05, "median_rounds": 1, "last_good_fills": 1, "settlement": cp.SETTLEMENT_PERIOD_EMA,
            "min_incentive_share": 0.0, "min_incentive_ramp_start": 0.0, **params}
    return TaskEntry(task_id=task_id, profile_id="p", profile_sha256="0" * 64, mechanism=mechanism,
                     params={k: v for k, v in base.items() if v is not None}, status="active", retired_at=None,
                     job_id=task_id)


def _collector(task_id=PAID, start=WORK):
    return _entry(task_id, payout_rail="collector", payout_rail_from=start)


def test_the_payout_rail_is_native_or_collector_and_the_collector_pays_corpus_period_tasks_only():
    from reliquary.shared.task_registry import RegistryError, payout_rail, payout_rail_from, validate_entry

    validate_entry(_entry())
    validate_entry(_entry(payout_rail="native"))
    validate_entry(_collector())
    assert payout_rail(_collector().params) == "collector" and payout_rail_from(_collector().params) == WORK
    assert payout_rail(_entry().params) == "native" and payout_rail(None) == "native"
    assert payout_rail_from(_entry().params) is None
    for bad in (_entry(payout_rail="both"),
                _entry(payout_rail="collector", payout_rail_from=WORK, settlement=None),
                _entry(payout_rail="collector"),                              # no first period
                _entry(payout_rail="collector", payout_rail_from=-1),
                _entry(payout_rail="collector", payout_rail_from=True),
                _entry(payout_rail="native", payout_rail_from=WORK),          # a start without the rail
                _entry(payout_rail_from=WORK)):
        with pytest.raises(RegistryError):
            validate_entry(bad)


def test_a_collector_rail_task_without_a_valid_start_is_read_alike_by_weights_and_payer():
    from reliquary.shared.task_registry import payout_rail_from

    # Only a hand-edited object can hold one: both sides read it as starting at 0, so no period is paid twice.
    assert payout_rail_from({"payout_rail": "collector"}) == 0
    assert payout_rail_from({"payout_rail": "collector", "payout_rail_from": "x"}) == 0


def test_set_cap_keeps_a_tasks_rail():
    from reliquary.shared.task_registry import set_cap

    entry = _collector()
    lowered = set_cap({entry.task_id: entry}, entry.task_id, 0.01)[entry.task_id]
    assert lowered.params["payout_rail"] == "collector" and lowered.params["payout_rail_from"] == WORK


def test_no_write_moves_an_existing_task_between_rails():
    from dataclasses import replace

    from reliquary.shared.task_registry import RegistryError, require_payout_rail_unchanged

    native, paid = _entry(NATIVE), _collector()
    before = {NATIVE: native, PAID: paid}
    require_payout_rail_unchanged(before, before)
    require_payout_rail_unchanged({NATIVE: native}, before)          # a new task chooses its rail
    require_payout_rail_unchanged(before, {NATIVE: native})          # a task gone is not a move
    for moved in ({NATIVE: replace(native, params={**native.params, "payout_rail": "collector",
                                                   "payout_rail_from": WORK}), PAID: paid},
                  {NATIVE: native, PAID: replace(paid, params={**paid.params, "payout_rail_from": WORK + 5})},
                  {NATIVE: native, PAID: replace(paid, params={k: v for k, v in paid.params.items()
                                                               if not k.startswith("payout_rail")})}):
        with pytest.raises(RegistryError):
            require_payout_rail_unchanged(before, moved)


def test_the_registry_store_refuses_a_write_that_moves_a_rail(monkeypatch):
    from dataclasses import replace

    from reliquary.infrastructure import task_registry_store as store
    from reliquary.shared.task_registry import RegistryError

    native = _entry(NATIVE)
    written = []

    async def read_registry(**_):
        return {NATIVE: native}, "etag"

    async def write_registry(entries, etag, **_):
        written.append(entries)

    monkeypatch.setattr(store, "read_registry", read_registry)
    monkeypatch.setattr(store, "write_registry", write_registry)
    move = lambda entries: {NATIVE: replace(native, params={**native.params, "payout_rail": "collector",  # noqa: E731
                                                            "payout_rail_from": WORK})}
    with pytest.raises(RegistryError):
        asyncio.run(store._mutate(move, attempts=1))
    assert written == []
    asyncio.run(store._mutate(lambda entries: dict(entries), attempts=1))
    assert written == [{NATIVE: native}]


def test_declaring_a_collector_rail_task_needs_the_fleet_acknowledged():
    from reliquary.shared.task_registry import RegistryError, require_fleet_knows_payout_rail

    require_fleet_knows_payout_rail(_entry(), acknowledged=False)
    require_fleet_knows_payout_rail(_collector(), acknowledged=True)
    with pytest.raises(RegistryError):
        require_fleet_knows_payout_rail(_collector(), acknowledged=False)


class _Archives:
    def __init__(self, docs):
        self.docs = docs

    async def list(self, task_id):
        return sorted((w, e) for (t, w, e) in self.docs if t == task_id)

    async def read(self, task_id, work, entry):
        return self.docs.get((task_id, work, entry))


def test_the_weight_setter_gives_a_collector_rail_task_no_weight_from_its_first_period():
    from reliquary.validator.weight_only import WeightOnlyValidator

    docs = {(PAID, WORK, WORK + 1): {"rewards_by_hotkey": {"5Before": 0.01}, "cap": 0.05},
            (PAID, WORK + 1, WORK + 2): {"rewards_by_hotkey": {"5After": 0.01}, "cap": 0.05},
            (NATIVE, WORK + 1, WORK + 2): {"rewards_by_hotkey": {"5Native": 0.01}, "cap": 0.05}}
    declared = {PAID: SimpleNamespace(params={**_collector(start=WORK + 1).params}),
                NATIVE: SimpleNamespace(params={**_entry(NATIVE).params})}
    now = cp.period_end(WORK + 2) + 60
    weights = asyncio.run(WeightOnlyValidator._period_weights(declared, archives=_Archives(docs), now=now))
    # Work before the rail's first period is the weights' to pay; from it on, the collector's.
    assert set(weights) == {PAID, NATIVE}
    assert set(weights[PAID]) == {"5Before"} and set(weights[NATIVE]) == {"5Native"}
    declared[PAID] = SimpleNamespace(params={**_collector(start=WORK).params})
    weights = asyncio.run(WeightOnlyValidator._period_weights(declared, archives=_Archives(docs), now=now))
    assert not weights.get(PAID)


def test_jobs_create_declares_the_collector_rail_only_with_the_fleet_acknowledged(bucket, registry):  # noqa: F811
    import time

    from typer.testing import CliRunner

    from reliquary.cli.main import app

    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    refused = CliRunner().invoke(app, _create_args(**{"--payout-rail": "collector"}))
    assert refused.exit_code == 1 and "--fleet-skips-collector-rail" in refused.output
    assert "corpus-run" not in registry["entries"]
    done = CliRunner().invoke(app, _create_args(**{"--payout-rail": "collector"}) + ["--fleet-skips-collector-rail"])
    assert done.exit_code == 0, done.output
    entry = registry["entries"]["corpus-run"]
    assert entry.params["payout_rail"] == "collector"
    assert entry.params["payout_rail_from"] == cp.period_of(time.time())
    bad = CliRunner().invoke(app, _create_args(**{"--job-id": "other", "--task-id": "other",
                                                   "--payout-rail": "both"}))
    assert bad.exit_code == 2 and "other" not in registry["entries"]


def test_jobs_create_is_on_the_native_rail_by_default(bucket, registry):  # noqa: F811
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    assert CliRunner().invoke(app, _create_args()).exit_code == 0
    params = registry["entries"]["corpus-run"].params
    assert "payout_rail" not in params and "payout_rail_from" not in params
