"""Window-boundary metadata fetches leave the serving loop available."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from reliquary.infrastructure import drand
from reliquary.validator import service
from reliquary.validator.server import ValidatorServer


def candidate():
    stages = []
    return SimpleNamespace(
        use_drand=True,
        _candidate_window_n=42,
        _window_n=41,
        _set_window_preparation_stage=stages.append,
    ), stages


@pytest.mark.asyncio
async def test_boundary_fetch_keeps_health_endpoints_responsive(monkeypatch):
    svc, stages = candidate()
    started, release = threading.Event(), threading.Event()
    served_before_return = []
    loop_thread = threading.get_ident()
    fetch_threads = []

    def fetch():
        fetch_threads.append(threading.get_ident())
        started.set()
        served_before_return.append(release.wait(1))
        return {"genesis_time": 10, "period": 3}

    monkeypatch.setattr(drand, "get_current_chain", fetch)
    monkeypatch.setattr(service.chain, "seconds_until_next_drand_boundary", lambda *args: 0)
    server = ValidatorServer()

    async def probe():
        while not started.is_set():
            await asyncio.sleep(0)
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server.app), base_url="http://test"
            ) as client:
                assert (await client.get("/livez")).status_code == 200
                assert (await client.get("/health")).status_code == 200
        finally:
            release.set()

    await asyncio.gather(service.ValidationService._wait_for_next_drand_boundary(svc), probe())
    assert served_before_return == [True]
    assert fetch_threads != [loop_thread]
    assert stages == ["drand_boundary"]


@pytest.mark.asyncio
@pytest.mark.parametrize("delay", [0, 1.25])
async def test_boundary_alignment_uses_fetched_parameters(monkeypatch, delay):
    svc, _ = candidate()
    monkeypatch.setattr(drand, "get_current_chain", lambda: {"genesis_time": 10, "period": 3})
    monkeypatch.setattr(service.time, "time", lambda: 100.5)
    boundary = Mock(return_value=delay)
    sleep = AsyncMock()
    monkeypatch.setattr(service.chain, "seconds_until_next_drand_boundary", boundary)
    monkeypatch.setattr(service.asyncio, "sleep", sleep)
    await service.ValidationService._wait_for_next_drand_boundary(svc)
    boundary.assert_called_once_with(100.5, 10, 3)
    if delay:
        sleep.assert_awaited_once_with(delay)
    else:
        sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_chain_fetch_preserves_error(monkeypatch):
    svc, stages = candidate()
    failure = LookupError("synthetic unavailable metadata")

    def fetch():
        raise failure

    monkeypatch.setattr(drand, "get_current_chain", fetch)
    with pytest.raises(LookupError) as raised:
        await service.ValidationService._wait_for_next_drand_boundary(svc)
    assert raised.value is failure
    assert stages == ["drand_boundary"]


@pytest.mark.asyncio
async def test_cancelled_boundary_wait_cannot_advance_candidate(monkeypatch):
    svc, stages = candidate()
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def fetch():
        started.set()
        release.wait(1)
        finished.set()
        return {"genesis_time": 10, "period": 3}

    boundary = Mock(return_value=0)
    monkeypatch.setattr(drand, "get_current_chain", fetch)
    monkeypatch.setattr(service.chain, "seconds_until_next_drand_boundary", boundary)
    task = asyncio.create_task(service.ValidationService._wait_for_next_drand_boundary(svc))
    try:
        while not started.is_set():
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        boundary.assert_not_called()
        assert (svc._window_n, svc._candidate_window_n) == (41, 42)
        assert stages == ["drand_boundary"]
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 1)


@pytest.mark.asyncio
async def test_disabled_drand_does_not_fetch_or_prepare(monkeypatch):
    svc, stages = candidate()
    svc.use_drand = False
    fetch = Mock()
    monkeypatch.setattr(drand, "get_current_chain", fetch)
    await service.ValidationService._wait_for_next_drand_boundary(svc)
    fetch.assert_not_called()
    assert stages == []
