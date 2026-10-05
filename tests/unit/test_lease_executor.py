"""Result delivery of every pull executor (grade, audit): a result lost on
the wire is posted again, within a bound, instead of leaving its lease to
expire on the control (3.3 h for a replay; production 2026-10-04)."""

import asyncio
import logging

import httpx
import pytest

from reliquary.validator import lease_executor
from reliquary.validator.lease_executor import LeaseExecutor

LEASE = "c" * 32


class _Flaky:
    """Fails the result post with each of ``failures`` in turn, then answers
    ``final`` (a status code)."""

    def __init__(self, failures=(), final=200):
        self.failures, self.final, self.posts = list(failures), final, []

    async def post(self, path, json, headers, timeout):
        self.posts.append(path)
        request = httpx.Request("POST", f"http://control{path}")
        if self.failures:
            failure = self.failures.pop(0)
            if isinstance(failure, int):
                return httpx.Response(failure, text="upstream", request=request)
            raise failure
        if self.final == 200:
            return httpx.Response(200, json={"outcome": "accepted"}, request=request)
        return httpx.Response(self.final, json={"detail": "lease_unknown"}, request=request)


def _executor(http, now=None):
    now = now if now is not None else [0.0]
    executor = LeaseExecutor(http=http, executor_id="g1", token="t" * 40, prefix="/p",
                             heartbeat_seconds=60, idle_seconds=1, clock=lambda: now[0])
    slept = []

    async def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    executor._sleep = sleep
    executor._wall_clock = lambda: 1_000_000.0 + now[0]
    return executor, slept


def _disconnect():
    return httpx.RemoteProtocolError("Server disconnected without sending a response.")


async def test_a_result_lost_once_on_the_wire_is_posted_again():
    http = _Flaky([_disconnect()])
    executor, slept = _executor(http)
    await executor.post_result(LEASE, {"results": []})
    assert http.posts == [f"/p/{LEASE}/result"] * 2
    assert len(slept) == 1 and executor.leases == 1


@pytest.mark.parametrize("failure", [httpx.ReadTimeout("slow"), httpx.ConnectError("refused"), 502])
async def test_timeouts_connect_errors_and_gateway_errors_are_retried(failure):
    http = _Flaky([failure, failure])
    executor, _ = _executor(http)
    await executor.post_result(LEASE, {"results": []})
    assert len(http.posts) == 3 and executor.leases == 1


async def test_retries_are_bounded_and_the_last_error_raises():
    http = _Flaky([_disconnect()] * 50)
    executor, slept = _executor(http)
    with pytest.raises(httpx.RemoteProtocolError):
        await executor.post_result(LEASE, {"results": []})
    assert len(http.posts) == lease_executor.RESULT_POST_ATTEMPTS
    assert sum(slept) <= lease_executor.RESULT_RETRY_SECONDS
    assert executor.leases == 0


async def test_never_retried_past_the_lease_expiry():
    http = _Flaky([_disconnect()] * 50)
    executor, slept = _executor(http)
    expires_at = executor._wall_clock() + 3.0       # less than the first backoff
    with pytest.raises(httpx.RemoteProtocolError):
        await executor.post_result(LEASE, {"results": []}, expires_at=expires_at)
    assert len(http.posts) == 1 and slept == []


async def test_a_410_after_a_retry_is_final_and_logged(caplog):
    # The first post may have landed before the connection dropped: the
    # control then no longer knows the lease. Final, never an error.
    http = _Flaky([_disconnect()], final=410)
    executor, _ = _executor(http)
    with caplog.at_level(logging.WARNING, logger=lease_executor.__name__):
        await executor.post_result(LEASE, {"results": []})
    assert len(http.posts) == 2
    assert any("after a retry" in r.getMessage() for r in caplog.records)


async def test_a_refusal_is_never_retried():
    for status in (410, 422):
        http = _Flaky(final=status)
        executor, slept = _executor(http)
        await executor.post_result(LEASE, {"results": []})
        assert len(http.posts) == 1 and slept == []
    http = _Flaky(final=401)
    executor, slept = _executor(http)
    with pytest.raises(httpx.HTTPStatusError):
        await executor.post_result(LEASE, {"results": []})
    assert len(http.posts) == 1 and slept == []


async def test_the_grade_executor_hands_its_lease_expiry_to_the_post():
    from reliquary.validator.corpus_grade_executor import GradeExecutor
    from tests.unit.test_corpus_grade_executor import _Http

    http = _Http(expires_at=5_000.0)
    seen = []

    async def item(grade_item):
        return {"status": "ok", "diff_applied": True, "tests_passed": True}

    executor = GradeExecutor(http=http, executor_id="g1", token="t" * 40, run_item=item,
                             env_check=lambda p, v: None, sweep=lambda: 0,
                             wall_clock=lambda: 1000.0)
    original = executor.post_result

    async def spy(lease_id, body, *, expires_at=None):
        seen.append(expires_at)
        await original(lease_id, body, expires_at=expires_at)

    executor.post_result = spy
    await executor.start()
    assert await executor.step() is True
    await asyncio.gather(*executor._running)
    assert seen == [5_000.0]
