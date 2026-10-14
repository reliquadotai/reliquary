"""The window settlement is gone for corpus tasks: a corpus validator serves and
settles only period-settled tasks, and ignores (with an ERROR, never a crash) a
window-settled one still listed in its configuration."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from reliquary.validator.corpus_periods import SETTLEMENT_PERIOD_EMA


def _entry(task_id, job_id=None, settlement=SETTLEMENT_PERIOD_EMA, mechanism="corpus-generation"):
    params = {"cap": 0.1}
    if settlement:
        params["settlement"] = settlement
    return SimpleNamespace(task_id=task_id, job_id=job_id or f"job-{task_id}",
                           mechanism=mechanism, status="active", params=params)


def test_window_settled_tasks_are_left_out_with_an_error(caplog):
    from reliquary.validator.corpus_validator import period_served

    served = [(_entry("a"), 0.1), (_entry("old", settlement=None), 0.0), (_entry("b"), 0.05)]
    with caplog.at_level(logging.ERROR):
        kept, dropped = period_served(served)
    assert [e.task_id for e, _ in kept] == ["a", "b"]
    assert [e.task_id for e in dropped] == ["old"]
    assert any("old" in r.getMessage() and r.levelno == logging.ERROR for r in caplog.records)


def test_wiring_a_window_settled_job_is_refused():
    from reliquary.validator.corpus_validator import wire_job_judge

    w = SimpleNamespace(entry=_entry("old", settlement=None), job=SimpleNamespace(job_id="j"),
                        cap=0.0, stats=None)
    with pytest.raises(ValueError, match="window"):
        wire_job_judge(w, records=None, judge_records=None, judge_threads=None, archives=None,
                       proof=None, model=None, tokenizer=None)


def test_a_hot_window_settled_entry_is_refused_not_wired():
    from reliquary.validator.corpus_hot_jobs import REFUSED
    from tests.unit.test_corpus_hot_jobs import _hot_entry, _refusal

    entry = _hot_entry()
    entry.params.pop("settlement", None)
    kind, why = _refusal(entry)
    assert kind == REFUSED and "window" in why


def test_a_split_judge_group_naming_a_left_out_job_is_ignored_not_fatal(caplog):
    from reliquary.validator.corpus_split import plan_groups

    jobs = [("corpus-a", "job-a"), ("corpus-b", "job-b")]
    with caplog.at_level(logging.ERROR):
        groups = plan_groups("job-old;corpus-a", jobs, ignored={"corpus-old", "job-old"})
    assert groups == [["job-a"]]
    assert any("job-old" in r.getMessage() for r in caplog.records)
    # A name nobody serves, and nobody left out, is still a refusal.
    with pytest.raises(ValueError):
        plan_groups("job-x", jobs, ignored={"job-old"})
