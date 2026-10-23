"""The async HTTP owner must also own the cached process-health collector."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest
import uvicorn


@pytest.mark.parametrize("health_status, expected_http", [("ok", 200), ("critical", 503)])
def test_async_serving_refreshes_readiness_and_awaits_monitor_shutdown(
    monkeypatch, tmp_path, health_status, expected_http,
):
    from reliquary.infrastructure import corpus_period_store
    from reliquary.validator import corpus_settlement, rl_async_service, server as server_module, telemetry

    closed, servers, monitors, collected = [], [], [], []
    real_server = server_module.ValidatorServer
    settings = SimpleNamespace(task_id="async-test", run="test", targets={"math": 1.0},
                               episodes={}, max_pending_groups=1, lane_reserves=dict)
    rl = SimpleNamespace(sandbox=None, book=SimpleNamespace(latest_checkpoint=None),
                         submissions=SimpleNamespace(grader=SimpleNamespace()),
                         close=lambda: closed.append("rl"))

    async def close_publisher():
        closed.append("publisher")

    def construct_server(*args, **kwargs):
        server = real_server(*args, **kwargs)
        assert server._process_health_snapshot["status"] == "pending"
        servers.append(server)
        return server

    def collect():
        collected.append(health_status)
        return {"status": health_status, "collected_at": 1.0}

    class HttpServer:
        def __init__(self, config):
            self.config = config

        async def serve(self):
            async def sampled():
                while servers[0]._process_health_snapshot["status"] == "pending":
                    await asyncio.sleep(0)

            await asyncio.wait_for(sampled(), timeout=1)
            monitors.extend(task for task in asyncio.all_tasks()
                            if task.get_coro().__name__ == "_monitor_process_health")
            assert len(monitors) == 1
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.config.app),
                                         base_url="http://test") as client:
                response = await client.get("/readyz")
                assert response.status_code == expected_http
                expected_reasons = [] if expected_http == 200 else ["process_health_unavailable"]
                assert response.json()["reasons"] == expected_reasons
                assert (await client.get("/livez")).status_code == 200

    monkeypatch.delenv(rl_async_service.TRAINER_TOKEN_ENV, raising=False)
    monkeypatch.setattr(rl_async_service, "rl_async_settings", lambda *a, **kw: settings)
    monkeypatch.setattr(rl_async_service, "load_episode_environments", lambda *a: {})
    monkeypatch.setattr(rl_async_service, "_grader_for", lambda *a: (object(), {}))
    monkeypatch.setattr(rl_async_service, "build_rl_async", lambda *a, **kw: rl)
    monkeypatch.setattr(rl_async_service, "_public_boards", lambda *a: SimpleNamespace(close=close_publisher))
    monkeypatch.setattr(rl_async_service, "audit_source", lambda: None)
    monkeypatch.setattr(rl_async_service, "mount_rl_async", lambda *a, **kw: None)
    monkeypatch.setattr(rl_async_service, "rl_async_loops", lambda *a, **kw: [])
    monkeypatch.setattr(corpus_period_store, "R2PeriodArchives", lambda **kw: object())
    monkeypatch.setattr(corpus_settlement, "R2Archives", lambda **kw: object())
    monkeypatch.setattr(telemetry, "init", lambda *a, **kw: None)
    monkeypatch.setattr(telemetry, "finish", lambda: closed.append("telemetry"))
    monkeypatch.setattr(server_module, "ValidatorServer", construct_server)
    monkeypatch.setattr(server_module, "collect_process_health", collect)
    monkeypatch.setattr(uvicorn, "Server", HttpServer)

    async def exercise():
        await rl_async_service.run_rl_async_validator(
            wallet=SimpleNamespace(hotkey=SimpleNamespace(ss58_address="test-hotkey")),
            netuid=1, signer_client=None, http_host="127.0.0.1", http_port=8000,
            entry=object(), cap=0.1, checkpoint="test-checkpoint", registration_gate=False,
        )
        assert len(monitors) == 1 and monitors[0].done()
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(exercise())
    assert collected == [health_status]
    assert closed == ["publisher", "rl", "telemetry"]
    assert servers[0]._task is None                       # no legacy HTTP/admission startup
