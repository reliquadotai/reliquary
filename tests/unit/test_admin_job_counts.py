"""The admin answers a large job's status within its callers' wait: one count per job at a time, outliving
the request, and the last counts once COUNTS_WAIT_SECONDS have passed."""

from __future__ import annotations

import asyncio
import time

import pytest

from reliquary.admin import service
from reliquary.validator import corpus_job_status
from tests.unit.test_admin_eval_jobs import admin  # noqa: F401
from tests.unit.test_jobs_cli import registry  # noqa: F401
from tests.unit.test_order_pinned import _order, pins  # noqa: F401

JOB = "order-eval-p1"


def _counts(verdicts, drained=False):
    return {"submissions": 10, "verdicts": verdicts, "unaudited": 10 - verdicts, "settled": verdicts,
            "unsettled": 0, "pending_records": 0, "pending_window": None, "last_window": None,
            "drained": drained, "final": drained}


@pytest.fixture
def counts(monkeypatch):
    """The stored counts the admin reads: ``answer``, after ``delay`` seconds; ``calls`` counted."""
    state = {"answer": _counts(4), "delay": 0.0, "calls": 0}

    async def stored(records, job_id, **kwargs):
        state["calls"] += 1
        answer = dict(state["answer"])
        await asyncio.sleep(state["delay"])
        return answer

    monkeypatch.setattr(corpus_job_status, "stored_job_counts", stored)
    monkeypatch.setattr(service, "COUNTS_WAIT_SECONDS", 0.2)
    return state


def _status(admin):  # noqa: F811
    reply = admin("GET", f"/admin/v1/jobs/{JOB}/status")
    assert reply.status_code == 200, reply.text
    return reply.json()


def test_a_slow_count_answers_the_last_counts_and_lands_for_the_next_call(admin, pins, counts):  # noqa: F811
    assert admin("POST", "/admin/v1/jobs", _order()).status_code == 201
    assert _status(admin)["verdicts"] == 4                    # quick: read now
    counts["answer"], counts["delay"] = _counts(9), 1.0
    assert _status(admin)["verdicts"] == 4                    # slow: the last counts, the count goes on
    counts["delay"] = 0.0
    calls = counts["calls"]
    time.sleep(1.2)
    assert _status(admin)["verdicts"] == 9                    # the count that went on, or a new one
    assert counts["calls"] >= calls


def test_a_first_slow_count_is_awaited(admin, pins, counts):  # noqa: F811
    assert admin("POST", "/admin/v1/jobs", _order()).status_code == 201
    counts["answer"], counts["delay"] = _counts(7), 0.5
    assert _status(admin)["verdicts"] == 7


def test_a_drained_job_is_not_listed_again(admin, pins, counts):  # noqa: F811
    assert admin("POST", "/admin/v1/jobs", _order()).status_code == 201
    counts["answer"] = _counts(10, drained=True)
    assert _status(admin)["drained"] is True
    calls = counts["calls"]
    assert _status(admin)["final"] is True
    assert counts["calls"] == calls
