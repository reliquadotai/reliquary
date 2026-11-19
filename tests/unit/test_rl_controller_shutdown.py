"""Real CPU signal replay with Uvicorn's actual capture contexts; no sockets, GPU or remote services."""
from __future__ import annotations

import asyncio
import signal
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from reliquary.rl_orders.runtime import ManagedControllerShutdown


IDENTITY = {"task_id": "order-train-test", "order_sha256": "a" * 64,
            "run_id": "test-run", "runtime_id": "b" * 32}


def drained_snapshot():
    return {"schema": "reliquary/training-runtime/v1", **IDENTITY,
            "controller_ready": True, "admission_closed": True, "drain_requested": True,
            "trainer_quiesced": True, "drained": True, "latest_checkpoint": 0}


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX signal replay and exit statuses")
@pytest.mark.parametrize("mode", ["drained", "undrained", "cleanup_failure", "worker_failure", "snapshot_failure",
                                  "premature", "sigint", "legacy"])
def test_actual_uvicorn_replay_only_finishes_managed_drained_cleanup_successfully(mode):
    script = textwrap.dedent("""
        import asyncio
        import contextlib
        import signal
        import sys
        from types import SimpleNamespace
        import uvicorn
        from reliquary.rl_orders.runtime import ManagedControllerShutdown
        from reliquary.validator.corpus_validator import _run_corpus_services

        mode = sys.argv[1]
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        identity = {"task_id": "order-train-test", "order_sha256": "a" * 64,
                    "run_id": "test-run", "runtime_id": "b" * 32}
        def snapshot():
            print("drain_checked", flush=True)
            if mode == "snapshot_failure":
                raise OSError("drain state unavailable")
            return {"schema": "reliquary/training-runtime/v1", **identity,
                    "controller_ready": True, "admission_closed": True, "drain_requested": True,
                    "trainer_quiesced": True, "drained": mode != "undrained", "latest_checkpoint": 0}
        shutdown = None if mode == "legacy" else ManagedControllerShutdown(SimpleNamespace(identity=identity, snapshot=snapshot))
        async def app(scope, receive, send):
            raise AssertionError("this reproduction cannot serve a socket")
        server = uvicorn.Server(uvicorn.Config(app, lifespan="off", ws="none", log_level="critical"))
        async def no_socket_serve(sockets=None):
            await asyncio.sleep(0)
            signal.raise_signal(signal.SIGINT if mode == "sigint" else signal.SIGTERM)
            assert server.should_exit
            print("http_shutdown_finished", flush=True)
        server._serve = no_socket_serve
        async def worker():
            try:
                await asyncio.Future()
            finally:
                print("background_joined", flush=True)
                if mode == "worker_failure":
                    raise RuntimeError("background cleanup failed")
        async def cleanup():
            if shutdown is not None:
                shutdown.capture_drain()
            print("book_closed", flush=True)
            if mode == "cleanup_failure":
                raise RuntimeError("cleanup failed")
            if shutdown is not None:
                shutdown.cleanup_completed()
            print("cleanup_finished", flush=True)
        async def main():
            with contextlib.nullcontext() if shutdown is None else shutdown:
                if mode == "premature":
                    print("premature_signal", flush=True)
                    signal.raise_signal(signal.SIGTERM)
                await _run_corpus_services(server, [worker()], cleanup=cleanup,
                                           check_worker_shutdown=shutdown is not None)
            print("normal_return", flush=True)
        asyncio.run(main())
    """)
    result = subprocess.run([sys.executable, "-u", "-c", script, mode], stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, start_new_session=True, timeout=20)
    markers = result.stdout.splitlines()
    if mode == "drained":
        assert result.returncode == 0, result.stderr
        assert markers == ["http_shutdown_finished", "background_joined", "drain_checked", "book_closed",
                           "cleanup_finished", "normal_return"]
    elif mode == "legacy":
        assert result.returncode == -signal.SIGTERM, result.stderr
        assert markers == ["http_shutdown_finished", "background_joined", "book_closed", "cleanup_finished"]
    else:
        assert result.returncode != 0, (markers, result.stderr)
        if mode == "premature":
            assert markers == ["premature_signal"]
        else:
            assert markers[:4] == ["http_shutdown_finished", "background_joined", "drain_checked", "book_closed"]
        if mode != "sigint":
            assert "normal_return" not in markers
        if mode == "worker_failure":
            assert "cleanup_finished" in markers and "background cleanup failed" in result.stderr


def test_managed_signal_scope_restores_handler_and_does_not_reuse_cleanup_proof():
    runtime = SimpleNamespace(identity=IDENTITY, snapshot=drained_snapshot)
    original = signal.getsignal(signal.SIGTERM)
    shutdown = ManagedControllerShutdown(runtime)
    with shutdown:
        assert shutdown.capture_drain()
        shutdown.cleanup_completed()
    assert signal.getsignal(signal.SIGTERM) == original
    with pytest.raises(RuntimeError, match="verified drainage"):
        with shutdown:
            pass
    assert signal.getsignal(signal.SIGTERM) == original
    with pytest.raises(KeyboardInterrupt):
        with shutdown:
            raise KeyboardInterrupt
    assert signal.getsignal(signal.SIGTERM) == original


@pytest.mark.parametrize("managed", [False, True])
def test_managed_controller_never_marks_failed_sandbox_cleanup_successful(monkeypatch, managed):
    from reliquary.infrastructure import corpus_period_store
    from reliquary.rl_orders import runtime as runtime_module
    from reliquary.validator import corpus_settlement, rl_async_service as service, rl_training_orders
    from reliquary.validator import server as server_module, telemetry

    calls, guards = [], []
    bound_runtime = SimpleNamespace(identity=IDENTITY, snapshot=drained_snapshot, admission_reason=lambda: None,
                                    check_limits=lambda: None)

    class Shutdown(ManagedControllerShutdown):
        def __init__(self, runtime):
            super().__init__(runtime)
            guards.append(self)

        def cleanup_completed(self):
            calls.append("clean_marked")
            super().cleanup_completed()

    class Sandbox:
        def background(self):
            return []

        async def stop(self):
            calls.append("sandbox_stop")
            raise RuntimeError("sandbox cleanup failed")

    class Admission:
        def __init__(self, *args, **kwargs):
            pass

        def reason(self):
            return None

        async def refresh(self):
            return True

        async def refresh_forever(self):
            await asyncio.Future()

    async def idle():
        await asyncio.Future()

    class HttpServer:
        def __init__(self, config):
            pass

        async def serve(self):
            await asyncio.sleep(0)

    async def close_publisher():
        calls.append("publisher_close")

    settings = SimpleNamespace(task_id="order-train-test", run="test", targets={"math": 1.0}, episodes={},
                               max_pending_groups=1, lane_reserves=dict)
    rl = SimpleNamespace(sandbox=Sandbox(), public_put=object(), book=SimpleNamespace(latest_checkpoint=0, auctions=SimpleNamespace()),
                         submissions=SimpleNamespace(grader=SimpleNamespace(scorer=SimpleNamespace(
                             close=lambda: calls.append("scorer_close")))), close=lambda: calls.append("book_close"))
    from fastapi import FastAPI

    server = SimpleNamespace(app=FastAPI(), mount_rl_lease_router=lambda *a: None, _monitor_process_health=idle)
    monkeypatch.delenv(service.TRAINER_TOKEN_ENV, raising=False)
    monkeypatch.setattr(service, "rl_async_settings", lambda *a, **kw: settings)
    monkeypatch.setattr(service, "load_episode_environments", lambda *a: {})
    monkeypatch.setattr(service, "_grader_for", lambda *a: (object(), {}))
    monkeypatch.setattr(service, "build_rl_async", lambda *a, **kw: rl)
    monkeypatch.setattr(service, "_public_boards", lambda *a: SimpleNamespace(close=close_publisher))
    monkeypatch.setattr(service, "audit_source", lambda: None)
    monkeypatch.setattr(service, "mount_rl_async", lambda *a, **kw: None)
    monkeypatch.setattr(service, "rl_async_loops", lambda *a, **kw: [])
    monkeypatch.setattr(runtime_module, "supervised_runtime", lambda *a: bound_runtime if managed else None)
    monkeypatch.setattr(runtime_module, "ManagedControllerShutdown", Shutdown)
    monkeypatch.setattr(rl_training_orders, "TrainingAdmission", Admission)
    monkeypatch.setattr(corpus_period_store, "R2PeriodArchives", lambda **kw: object())
    monkeypatch.setattr(corpus_settlement, "R2Archives", lambda **kw: object())
    monkeypatch.setattr(telemetry, "init", lambda *a, **kw: None)
    monkeypatch.setattr(telemetry, "finish", lambda: calls.append("telemetry_finish"))
    monkeypatch.setattr(server_module, "ValidatorServer", lambda *a, **kw: server)
    import uvicorn

    monkeypatch.setattr(uvicorn, "Server", HttpServer)

    async def run():
        await service.run_rl_async_validator(
            wallet=SimpleNamespace(hotkey=SimpleNamespace(ss58_address="test-hotkey")), netuid=1,
            signer_client=None, http_host="127.0.0.1", http_port=8000,
            entry=SimpleNamespace(status="active", params={"training_order": {}}), cap=.1, checkpoint="test-model", registration_gate=False)

    original = signal.getsignal(signal.SIGTERM)
    if managed:
        with pytest.raises(RuntimeError, match="managed sandbox cleanup"):
            asyncio.run(run())
        assert len(guards) == 1 and guards[0]._clean is False
    else:
        asyncio.run(run())
        assert guards == []
    from fastapi.testclient import TestClient

    with TestClient(server.app) as client:
        assert client.get("/rl/trainer/config").status_code == (404 if managed else 401)
        assert client.get("/rl/trainer/runtime").status_code == 404
    assert calls == ["sandbox_stop", "publisher_close", "scorer_close", "book_close", "telemetry_finish"]
    assert signal.getsignal(signal.SIGTERM) == original
