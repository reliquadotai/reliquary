"""Lazy dataset metadata must not block the serving event loop."""

import asyncio
import threading

import httpx
import pytest

from reliquary.protocol.submission import WindowState
from reliquary.validator.server import ValidatorServer
from tests.unit.test_validator_server import FakeEnv, _batcher


class ColdLengthEnv(FakeEnv):
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.served_during_load = []
        self.calls = 0

    def __len__(self):
        self.calls += 1
        self.started.set()
        try:
            self.served_during_load.append(self.release.wait(1))
            return 1000
        finally:
            self.finished.set()


def server_and_environment():
    server = ValidatorServer()
    batcher = _batcher()
    environment = ColdLengthEnv()
    # Installing the synthetic lazy source must not evaluate its length.
    batcher.env = environment
    server.set_active_batchers({environment.name: batcher})
    server.set_current_state(WindowState.OPEN)
    return server, environment, batcher


async def wait_started(environment):
    while not environment.started.is_set():
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_cold_miner_state_keeps_health_available_and_preserves_body_and_etag():
    server, environment, batcher = server_and_environment()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        batcher.env = FakeEnv()
        expected = await client.get("/miner-state")
        server._miner_state_response_cache = None
        batcher.env = environment

        async def serve_health():
            await wait_started(environment)
            try:
                assert (await client.get("/livez")).status_code == 200
                assert (await client.get("/health")).status_code == 200
            finally:
                environment.release.set()

        response, _ = await asyncio.gather(client.get("/miner-state"), serve_health())
        assert environment.served_during_load == [True]
        assert response.status_code == 200
        assert response.content == expected.content
        assert response.headers["etag"] == expected.headers["etag"]
        cached = await client.get("/miner-state", headers={"If-None-Match": response.headers["etag"]})
        assert cached.status_code == 304 and cached.content == b""
        assert environment.calls == 1


@pytest.mark.asyncio
async def test_window_change_during_metadata_load_is_not_cached():
    server, environment, batcher = server_and_environment()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        request = asyncio.create_task(client.get("/miner-state"))
        try:
            await wait_started(environment)
            server.set_active_batchers({"fake": _batcher(window_start=batcher.window_start + 1)})
        finally:
            environment.release.set()
        response = await request
        assert response.status_code == 503
        assert response.json()["detail"] == "window_changed_during_state_build"
        assert server._miner_state_response_cache is None
        current = await client.get("/miner-state")
        assert current.status_code == 200
        assert current.json()["window_n"] == batcher.window_start + 1


@pytest.mark.asyncio
async def test_cancelled_metadata_wait_cannot_publish_a_cached_response():
    server, environment, batcher = server_and_environment()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        request = asyncio.create_task(client.get("/miner-state"))
        try:
            await wait_started(environment)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            assert server._miner_state_response_cache is None
            assert server._active_batchers == {"fake": batcher}
        finally:
            environment.release.set()
            assert await asyncio.to_thread(environment.finished.wait, 1)
        assert (await client.get("/livez")).status_code == 200
        assert (await client.get("/miner-state")).status_code == 200


@pytest.mark.asyncio
async def test_existing_prompt_range_needs_no_lazy_length_read():
    server, environment, batcher = server_and_environment()
    batcher.prompt_range = (10, 20)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        response = await client.get("/miner-state")
    assert response.status_code == 200
    assert response.json()["environments"]["fake"]["prompt_range"] == [10, 20]
    assert environment.calls == 0


@pytest.mark.asyncio
async def test_legacy_state_does_not_load_lazy_dataset_metadata():
    server, environment, _ = server_and_environment()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        response = await client.get("/state", params={"env": "fake"})
    assert response.status_code == 200
    assert environment.calls == 0
