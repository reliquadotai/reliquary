"""A finished corpus job sets its task's cap to 0 by itself, once."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from reliquary.validator.corpus_autoclose import (
    AlreadyClosed,
    AutoClose,
    autoclose_enabled,
    close_guard,
)

STATUS = {"prompts_full": 10, "prompts_total": 10, "submissions_accepted": 20,
          "audited": 20, "passed": 18, "verified_tokens": 12345, "settled": 20}


def _entry(task_id="corpus-a", job_id="job-a", cap=0.1, status="active", admission="open",
           settlement="period-ema-v1"):
    params = {"cap": cap}
    if settlement:
        params["settlement"] = settlement
    return SimpleNamespace(task_id=task_id, job_id=job_id, status=status, admission=admission,
                           params=params)


class _JobSet:
    def __init__(self, entries, finished=True):
        self.served = {}
        for e in entries:
            settler = SimpleNamespace(caps=[])
            settler.set_cap = settler.caps.append
            self.served[e.job_id] = SimpleNamespace(entry=e, settler=settler,
                                                    cap=e.params["cap"])
        self._finished = finished
        self.checked = []

    async def job_finished(self, job_id):
        self.checked.append(job_id)
        return dict(STATUS) if self._finished else None


class _Registry:
    def __init__(self, *entries):
        self.entries = {e.task_id: e for e in entries}
        self.writes = []

    async def read(self):
        return dict(self.entries)

    async def write(self, task_id, job_id):
        close_guard(task_id, job_id)(self.entries, None)
        self.writes.append((task_id, job_id))
        old = self.entries[task_id]
        self.entries[task_id] = _entry(old.task_id, old.job_id, cap=0.0, status=old.status,
                                       admission=old.admission)


def _close(registry, job_set):
    auto = AutoClose(read_entries=registry.read, write_cap_zero=registry.write)
    return auto, asyncio.run(auto.run_once(job_set))


def test_a_full_but_undrained_job_is_left_alone():
    registry = _Registry(_entry())
    _, closed = _close(registry, _JobSet([_entry()], finished=False))
    assert closed == [] and registry.writes == []
    assert registry.entries["corpus-a"].params["cap"] == 0.1


def test_a_finished_job_sets_its_cap_to_zero_once(caplog):
    registry = _Registry(_entry())
    job_set = _JobSet([_entry()])
    with caplog.at_level(logging.INFO, logger="reliquary.validator.corpus_autoclose"):
        auto, closed = _close(registry, job_set)
    assert closed == ["corpus-a"] and registry.writes == [("corpus-a", "job-a")]
    assert registry.entries["corpus-a"].params["cap"] == 0.0
    message = next(r.getMessage() for r in caplog.records if "closed automatically" in r.getMessage())
    assert "12345" in message and "0.1 -> 0" in message
    # This process's settler prices anything settled from now on at 0.
    assert job_set.served["job-a"].settler.caps == [0.0]
    assert job_set.served["job-a"].cap == 0.0
    # Again in the same process: nothing written, not even checked.
    assert asyncio.run(auto.run_once(job_set)) == [] and registry.writes == [("corpus-a", "job-a")]


def test_a_restart_does_not_write_again():
    registry = _Registry(_entry(cap=0.0))
    job_set = _JobSet([_entry(cap=0.0)])
    _, closed = _close(registry, job_set)
    assert closed == [] and registry.writes == [] and job_set.checked == []


def test_a_paused_job_is_never_closed():
    registry = _Registry(_entry(admission="paused"))
    job_set = _JobSet([_entry()])
    _, closed = _close(registry, job_set)
    assert closed == [] and registry.writes == [] and job_set.checked == []


def test_a_retired_window_or_reassigned_task_is_left_alone():
    for entry in (_entry(status="retired"), _entry(settlement=None), _entry(job_id="job-other")):
        registry = _Registry(entry)
        _, closed = _close(registry, _JobSet([_entry()]))
        assert closed == [] and registry.writes == []


def test_the_write_is_refused_when_the_registry_changed_under_it():
    guard = close_guard("corpus-a", "job-a")
    for before in ({"corpus-a": _entry(cap=0.0)}, {"corpus-a": _entry(status="retired")},
                   {"corpus-a": _entry(job_id="job-b")}, {}):
        with pytest.raises(AlreadyClosed):
            guard(before, None)
    guard({"corpus-a": _entry()}, None)


def test_a_lost_race_is_remembered_not_retried():
    registry = _Registry(_entry())

    async def raced(task_id, job_id):
        raise AlreadyClosed("closed by an operator meanwhile")

    auto = AutoClose(read_entries=registry.read, write_cap_zero=raced)
    job_set = _JobSet([_entry()])
    assert asyncio.run(auto.run_once(job_set)) == []
    assert "job-a" in auto.closed


def test_a_failed_write_is_retried_at_the_next_pass():
    registry = _Registry(_entry())
    calls = []

    async def flaky(task_id, job_id):
        calls.append(task_id)
        if len(calls) == 1:
            raise ConnectionError("R2")
        await registry.write(task_id, job_id)

    auto = AutoClose(read_entries=registry.read, write_cap_zero=flaky)
    job_set = _JobSet([_entry()])
    assert asyncio.run(auto.run_once(job_set)) == []
    assert asyncio.run(auto.run_once(job_set)) == ["corpus-a"]


def test_passes_are_spaced_out():
    registry = _Registry(_entry())
    now = [0.0]
    auto = AutoClose(read_entries=registry.read, write_cap_zero=registry.write,
                     every_seconds=600, clock=lambda: now[0])
    job_set = _JobSet([_entry()], finished=False)
    asyncio.run(auto.maybe_run(job_set))
    assert job_set.checked == []  # the first pass waits one interval after the start
    now[0] = 601.0
    asyncio.run(auto.maybe_run(job_set))
    now[0] = 900.0
    asyncio.run(auto.maybe_run(job_set))
    assert job_set.checked == ["job-a"]
    now[0] = 1202.0
    asyncio.run(auto.maybe_run(job_set))
    assert job_set.checked == ["job-a", "job-a"]


def test_on_by_default_and_turned_off_by_the_operator():
    assert autoclose_enabled({})
    assert autoclose_enabled({"RELIQUARY_CORPUS_AUTOCLOSE": "1"})
    for off in ("0", "false", "OFF", "no"):
        assert not autoclose_enabled({"RELIQUARY_CORPUS_AUTOCLOSE": off})


def test_the_real_registry_write_takes_cap_zero_and_keeps_the_tail_bound(monkeypatch):
    """Through the registry store's CAS: cap 0, floor 0, tail_cap = the old cap."""
    import reliquary.infrastructure.task_registry_store as store
    from reliquary.validator.corpus_autoclose import write_cap_zero
    from tests.unit.test_corpus_period_tail import _corpus_entry

    entry = _corpus_entry(0.1, "period-ema-v1")
    state = {"entries": {entry.task_id: entry}}

    async def read(**kw):
        return dict(state["entries"]), "etag"

    async def write(entries, etag, **kw):
        state["entries"] = dict(entries)

    monkeypatch.setattr(store, "read_registry", read)
    monkeypatch.setattr(store, "write_registry", write)
    asyncio.run(write_cap_zero(entry.task_id, entry.job_id))
    params = state["entries"][entry.task_id].params
    assert (params["cap"], params["floor"], params["tail_cap"]) == (0.0, 0.0, 0.1)
    with pytest.raises(AlreadyClosed):
        asyncio.run(write_cap_zero(entry.task_id, entry.job_id))


# --- CorpusJobSet.job_finished: full AND drained, nothing still in flight ---------


def _job_set(*, full=True, drained=True, grader_oldest=None, pending=False, paused=False,
             autoclose=None, eval_complete=None):
    from reliquary.validator.corpus_hot_jobs import CorpusJobSet
    from reliquary.validator.corpus_service import CorpusJobRoutes

    routes = CorpusJobRoutes()
    calls = {"drained": 0}

    async def is_drained(wiring):
        calls["drained"] += 1
        return drained

    job_set = CorpusJobSet(routes=routes, router_for=None, wire=None, jobs_of=lambda w: [],
                           drained=is_drained, autoclose=autoclose, refresh_every_seconds=0.01)
    grader = None
    if grader_oldest is not None:
        def oldest():
            if isinstance(grader_oldest, Exception):
                raise grader_oldest
            return grader_oldest
        grader = SimpleNamespace(oldest_unready_received_at=oldest)
    elif grader_oldest is None:
        grader = SimpleNamespace(oldest_unready_received_at=lambda: None)
    job_set.served["job-a"] = SimpleNamespace(entry=_entry(), grader=grader)
    status = {**STATUS, "prompts_full": 10 if full else 9}
    if eval_complete is not None:
        status = {**status, "prompts_full": 0, "complete": eval_complete}

    async def compute(job_id, *, drained=False):
        return dict(status)

    job_set._compute_status = compute
    routes.admission_pending = lambda job_id: pending
    if paused:
        routes.paused.add("job-a")
    return job_set, calls


def test_a_job_is_finished_only_full_and_drained():
    job_set, calls = _job_set()
    assert asyncio.run(job_set.job_finished("job-a"))["verified_tokens"] == 12345
    job_set, calls = _job_set(full=False)
    assert asyncio.run(job_set.job_finished("job-a")) is None and calls["drained"] == 0
    job_set, _ = _job_set(drained=False)
    assert asyncio.run(job_set.job_finished("job-a")) is None


def test_a_job_with_a_submission_in_flight_or_a_grade_pending_is_not_finished():
    job_set, calls = _job_set(pending=True)
    assert asyncio.run(job_set.job_finished("job-a")) is None
    job_set, _ = _job_set(grader_oldest=1234.5)
    assert asyncio.run(job_set.job_finished("job-a")) is None
    job_set, _ = _job_set(grader_oldest=LookupError("not seeded"))
    assert asyncio.run(job_set.job_finished("job-a")) is None


def test_a_paused_or_retired_job_is_not_finished():
    job_set, calls = _job_set(paused=True)
    assert asyncio.run(job_set.job_finished("job-a")) is None and calls["drained"] == 0
    job_set, calls = _job_set()
    job_set._routes.retire("job-a")
    assert asyncio.run(job_set.job_finished("job-a")) is None


def test_an_eval_job_is_full_once_every_prompt_is_complete():
    job_set, _ = _job_set(eval_complete=True)
    assert asyncio.run(job_set.job_finished("job-a")) is not None
    job_set, _ = _job_set(eval_complete=False)
    assert asyncio.run(job_set.job_finished("job-a")) is None


def test_the_job_set_loop_runs_the_autoclose_and_survives_its_failure():
    seen = []

    class _Auto:
        async def maybe_run(self, job_set):
            seen.append(job_set)
            if len(seen) == 1:
                raise ConnectionError("R2")
            return []

    job_set, _ = _job_set(autoclose=_Auto())

    async def go():
        task = asyncio.ensure_future(job_set.run())
        while len(seen) < 3:
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())
    assert seen[0] is job_set
