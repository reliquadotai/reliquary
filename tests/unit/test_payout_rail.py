"""A task's payout rail: ``native`` (the weights pay it, the default) or
``collector`` (the payer pays it, and the weight setter gives it no weight).
Chosen when the task is added; nothing after changes it."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from reliquary.validator import corpus_periods as cp

PAID, NATIVE = "order-gen-collector", "order-gen-native"
WORK = 23_817


def _entry(**params):
    from reliquary.shared.task_registry import TaskEntry

    base = {"start": 0.05, "decay": 0.98, "rounds_per_step": 1, "deadband": 0.0, "snap": 0.0, "floor": 0.05,
            "cap": 0.05, "median_rounds": 1, "last_good_fills": 1, "settlement": cp.SETTLEMENT_PERIOD_EMA,
            "min_incentive_share": 0.0, "min_incentive_ramp_start": 0.0, **params}
    return TaskEntry(task_id=PAID, profile_id="p", profile_sha256="0" * 64, mechanism="corpus-generation",
                     params={k: v for k, v in base.items() if v is not None}, status="active", retired_at=None,
                     job_id=PAID)


def test_the_payout_rail_is_native_or_collector_and_the_collector_pays_period_tasks_only():
    from reliquary.shared.task_registry import RegistryError, payout_rail, validate_entry

    validate_entry(_entry())
    validate_entry(_entry(payout_rail="native"))
    validate_entry(_entry(payout_rail="collector"))
    assert payout_rail(_entry(payout_rail="collector").params) == "collector"
    assert payout_rail(_entry().params) == "native" and payout_rail(None) == "native"
    with pytest.raises(RegistryError):
        validate_entry(_entry(payout_rail="both"))
    with pytest.raises(RegistryError):
        validate_entry(_entry(payout_rail="collector", settlement=None))


def test_set_cap_keeps_a_tasks_rail():
    from reliquary.shared.task_registry import set_cap

    entry = _entry(payout_rail="collector")
    lowered = set_cap({entry.task_id: entry}, entry.task_id, 0.01)[entry.task_id]
    assert lowered.params["payout_rail"] == "collector"


class _Archives:
    def __init__(self, docs):
        self.docs = docs

    async def list(self, task_id):
        return sorted((w, e) for (t, w, e) in self.docs if t == task_id)

    async def read(self, task_id, work, entry):
        return self.docs.get((task_id, work, entry))


def test_the_weight_setter_gives_a_collector_rail_task_no_weight():
    from reliquary.validator.weight_only import WeightOnlyValidator

    docs = {(task, WORK, WORK + 1): {"rewards_by_hotkey": {"5A": 0.01}, "cap": 0.05}
            for task in (PAID, NATIVE)}
    declared = {PAID: SimpleNamespace(params={**_entry(payout_rail="collector").params}),
                NATIVE: SimpleNamespace(params={**_entry().params})}
    now = cp.period_end(WORK + 1) + 60
    weights = asyncio.run(WeightOnlyValidator._period_weights(declared, archives=_Archives(docs), now=now))
    assert set(weights) == {NATIVE}
