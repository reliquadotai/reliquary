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


# --------------------------------------------------------------------------
# Heartbeats: the held leases, one retry, and no stale keep-alive reuse
# --------------------------------------------------------------------------


class _Beats:
    """Records heartbeat bodies; ``refuse_field`` answers 422 to one that
    carries ``held_leases`` (a control from before the field)."""

    def __init__(self, refuse_field=False, failures=()):
        self.bodies, self.refuse_field, self.failures = [], refuse_field, list(failures)

    async def post(self, path, json, headers, timeout):
        request = httpx.Request("POST", f"http://control{path}")
        if path.endswith("/heartbeat"):
            self.bodies.append(json)
            if self.failures:
                raise self.failures.pop(0)
            if self.refuse_field and "held_leases" in json:
                return httpx.Response(422, json={"detail": "extra"}, request=request)
            return httpx.Response(200, json={"executor_id": "g1", "model_id": "m",
                                             "model_revision": "r"}, request=request)
        return httpx.Response(200, json={"outcome": "accepted"}, request=request)


async def test_a_plain_lease_executor_sends_no_held_leases():
    http = _Beats()
    executor, _ = _executor(http)
    await executor.heartbeat()
    assert "held_leases" not in http.bodies[0]


async def test_the_grade_executor_reports_the_leases_it_holds_until_posted():
    from reliquary.validator.corpus_grade_executor import GradeExecutor
    from tests.unit.test_corpus_grade_executor import _Http

    http = _Http()
    release = asyncio.Event()

    async def item(grade_item):
        await release.wait()
        return {"status": "ok", "diff_applied": True, "tests_passed": True}

    executor = GradeExecutor(http=http, executor_id="g1", token="t" * 40, run_item=item,
                             env_check=lambda p, v: None, sweep=lambda: 0)
    await executor.start()
    assert await executor.step() is True
    await executor.heartbeat()
    release.set()
    await asyncio.gather(*executor._running)
    await executor.heartbeat()
    beats = [body for path, body in http.posts if path.endswith("/heartbeat")]
    assert beats[0]["held_leases"] == []                     # at start
    assert beats[1]["held_leases"] == ["c" * 32]             # while working
    assert beats[2]["held_leases"] == []                     # once posted


async def test_a_control_refusing_the_field_gets_heartbeats_without_it():
    from reliquary.validator.corpus_grade_executor import GradeExecutor

    http = _Beats(refuse_field=True)
    executor = GradeExecutor(http=http, executor_id="g1", token="t" * 40,
                             env_check=lambda p, v: None, sweep=lambda: 0)
    assert (await executor.heartbeat())["model_id"] == "m"
    await executor.heartbeat()
    assert ["held_leases" in b for b in http.bodies] == [True, False, False]


async def test_a_heartbeat_lost_on_the_wire_is_sent_once_more():
    http = _Beats(failures=[_disconnect()])
    executor, _ = _executor(http)
    assert (await executor.heartbeat())["model_id"] == "m"
    assert len(http.bodies) == 2
    http = _Beats(failures=[_disconnect(), _disconnect()])
    executor, _ = _executor(http)
    with pytest.raises(httpx.RemoteProtocolError):
        await executor.heartbeat()


def test_the_client_never_reuses_a_connection_the_control_may_have_closed():
    # uvicorn closes an idle keep-alive connection after 5 s; the grade
    # executor polls every 5 s, so a reused connection raced that close.
    limits = lease_executor.client_limits()
    assert limits.keepalive_expiry is not None and limits.keepalive_expiry < 5.0
