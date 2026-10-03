"""Closing a finished corpus task: cap 0 and retired, once nothing is owed."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from reliquary.validator.corpus_close import TaskNotClosable, close_task


class Records:
    def __init__(self, drained=True):
        self.drained = drained

    async def list_submission_ids(self, job_id):
        return ["a", "b"]

    async def list_verdict_ids(self, job_id):
        return ["a", "b"] if self.drained else ["a"]

    async def read_settlement(self, job_id):
        return {"settled": ["a", "b"] if self.drained else [], "pending": None}, None


def entry(cap=0.04, settlement="period-ema-v1", status="active", mechanism="corpus-generation"):
    params = {"cap": cap}
    if settlement:
        params["settlement"] = settlement
    return SimpleNamespace(task_id="eval-a", job_id="eval-a", mechanism=mechanism,
                           status=status, params=params)


def run(e, *, paying=0.0, drained=True, cut_tail=False):
    calls = []

    async def registry():
        return {"eval-a": e}, None

    async def weights(declared):
        return {"eval-a": {"m": paying}} if paying else {}

    async def set_cap(task_id, cap):
        calls.append(("cap", task_id, cap))

    async def retire(task_id, at):
        calls.append(("retire", task_id, at))

    message = asyncio.run(close_task("eval-a", cut_tail=cut_tail, read_registry=registry,
                                     records=Records(drained), period_weights=weights,
                                     set_cap=set_cap, retire=retire, drand_round=lambda: 99))
    return message, calls


def test_a_drained_decayed_period_task_closes():
    message, calls = run(entry(), paying=0.00001)
    assert calls == [("cap", "eval-a", 0.0), ("retire", "eval-a", 99)]
    assert "free" in message


def test_a_task_still_paying_what_it_earned_does_not_close():
    with pytest.raises(TaskNotClosable, match="still pays"):
        run(entry(), paying=0.01)


def test_an_undrained_job_does_not_close():
    with pytest.raises(TaskNotClosable, match="not drained"):
        run(entry(), drained=False)


def test_a_window_settled_task_closes_only_when_told_to_cut_its_tail():
    with pytest.raises(TaskNotClosable, match="--cut-tail"):
        run(entry(settlement=None))
    _, calls = run(entry(settlement=None), cut_tail=True)
    assert calls == [("cap", "eval-a", 0.0), ("retire", "eval-a", 99)]


def test_only_corpus_tasks_close():
    with pytest.raises(TaskNotClosable, match="only a corpus task"):
        run(entry(mechanism="rl-discovered-price"))


def test_an_already_capped_retired_task_is_left_as_it_is():
    _, calls = run(entry(cap=0.0, status="retired"))
    assert calls == []


def test_cap_zero_frees_the_share_in_the_registry_rule():
    from reliquary.shared.task_registry import total_cap

    entries = {"a": SimpleNamespace(params={"cap": 0.0}), "b": SimpleNamespace(params={"cap": 0.5})}
    assert total_cap(entries) == 0.5


from tests.unit.test_jobs_cli import _create_args, _rl_entry, bucket, registry  # noqa: E402,F401


def test_jobs_create_declares_period_settlement_by_default(bucket, registry):  # noqa: F811
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    assert CliRunner().invoke(app, _create_args()).exit_code == 0
    assert registry["entries"]["corpus-run"].params["settlement"] == "period-ema-v1"


def test_jobs_create_can_still_declare_window_settlement(bucket, registry):  # noqa: F811
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    result = CliRunner().invoke(app, _create_args(**{"--settlement": "windows"}))
    assert result.exit_code == 0, result.output
    assert "settlement" not in registry["entries"]["corpus-run"].params


def test_a_real_corpus_entry_takes_cap_zero_then_retires(bucket, registry):  # noqa: F811
    from typer.testing import CliRunner

    from reliquary.cli.main import app
    from reliquary.infrastructure import task_registry_store as store

    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    assert CliRunner().invoke(app, _create_args()).exit_code == 0

    async def drained_registry():
        return dict(registry["entries"]), None

    asyncio.run(close_task("corpus-run", read_registry=drained_registry,
                           records=Records(), period_weights=lambda d: _none(),
                           set_cap=store.set_task_cap, retire=store.retire_task_entry,
                           drand_round=lambda: 7))
    closed = registry["entries"]["corpus-run"]
    assert closed.params["cap"] == 0.0 and closed.status == "retired"


async def _none():
    return {}


def test_a_period_job_needs_the_fleet_acknowledgement(bucket, registry):  # noqa: F811
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    argv = [a for a in _create_args() if a != "--fleet-knows-period-settlement"]
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 1 and "--fleet-knows-period-settlement" in result.output
    assert "corpus-run" not in registry["entries"]
