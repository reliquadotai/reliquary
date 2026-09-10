"""Contract, transport, and service tests for the remote CPU executor."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError


def test_executor_import_does_not_load_controller_corpora():
    subprocess.run(
        [sys.executable, "-c", """
import sys
sys.modules['reliquary.constants'] = None
sys.modules['reliquary.environment.registry'] = None
from reliquary.environment.grader.remote import create_cpu_executor_app
from reliquary.environment.grader import GRADER_POOL_SIZE
from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE
assert GRADER_POOL_SIZE == 4 * ACTIVE_PROTOCOL_PROFILE.sampling.rollouts
from reliquary.environment import load_environment
try:
    load_environment('openmathinstruct')
except ModuleNotFoundError:
    pass
else:
    raise AssertionError('environment loading bypassed the corpus registry')
"""],
        check=True, timeout=15,
    )


def _case(expected: int = 3) -> dict:
    return {
        "entry": {"kind": "function", "name": "add"},
        "args": [1, 2],
        "kwargs": {},
        "expected": expected,
        "compare": "exact",
    }


def _request(*, runtime_id: str = "grader-test-v1"):
    from reliquary.environment.grader.executor import make_sandbox_batch_request

    return make_sandbox_batch_request(
        runtime_id=runtime_id,
        code="def add(a, b): return a + b",
        cases=[_case()],
        timeout_s=5.0,
    )


def _result(request, *, output: int = 3, executor_id: str = "cpu-test"):
    from reliquary.environment.grader.executor import (
        SandboxBatchResult,
        SandboxCaseResult,
    )

    return SandboxBatchResult(
        protocol_version=request.protocol_version,
        job_id=request.job_id,
        attempt=request.attempt,
        runtime_id=request.runtime_id,
        executor_id=executor_id,
        results=[SandboxCaseResult(case_id=0, status="ok", output=output)],
        wall_ms=1.25,
    )


def test_execution_request_omits_expected_values_and_is_content_bound():
    from reliquary.environment.grader.executor import SandboxBatchRequest

    request = _request()
    payload = request.model_dump(mode="json")

    assert "expected" not in json.dumps(payload)
    assert "compare" not in json.dumps(payload)
    assert len(request.job_id) == 64
    assert request.protocol_version == 2
    assert request.batch_timeout_s == 5.0

    payload["code"] = "def add(a, b): return 0"
    with pytest.raises(ValidationError, match="code_sha256 mismatch"):
        SandboxBatchRequest.model_validate(payload)


def test_execution_request_has_a_bounded_overall_batch_deadline():
    from reliquary.environment.grader.executor import (
        MAX_EXECUTOR_BATCH_TIMEOUT_SECONDS,
        make_sandbox_batch_request,
    )

    request = make_sandbox_batch_request(
        runtime_id="grader-test-v1",
        code="def add(a, b): return a + b",
        cases=[_case(), _case(), _case()],
        timeout_s=5.0,
    )
    capped = make_sandbox_batch_request(
        runtime_id="grader-test-v1",
        code="def add(a, b): return a + b",
        cases=[_case() for _ in range(100)],
        timeout_s=5.0,
    )

    assert request.batch_timeout_s == 15.0
    assert capped.batch_timeout_s == MAX_EXECUTOR_BATCH_TIMEOUT_SECONDS


@pytest.mark.parametrize("timeout_s", (1, 1.0))
def test_integer_timeout_survives_request_transport_and_validation(timeout_s):
    from reliquary.environment.grader.executor import (
        RemoteSandboxExecutor, SandboxBatchRequest, make_sandbox_batch_request,
    )

    request = make_sandbox_batch_request(
        runtime_id="grader-test-v1", code="def add(a, b): return a + b",
        cases=[_case()], timeout_s=timeout_s,
    )
    received = []

    def handler(http_request):
        parsed = SandboxBatchRequest.model_validate_json(http_request.content)
        assert parsed == request
        received.append(parsed)
        return httpx.Response(200, json=_result(parsed).model_dump(mode="json"))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        executor = RemoteSandboxExecutor(
            "https://cpu.internal", runtime_id=request.runtime_id, client=client,
        )
        assert executor.execute(request).results[0].output == 3
    assert len(received) == 1
    assert request.timeout_s == request.batch_timeout_s == 1.0


@pytest.mark.parametrize("timeout_s", (5, 5.0))
@pytest.mark.parametrize("batch_timeout_s", (5, 5.0))
def test_explicit_integer_deadlines_bind_to_existing_float_job_id(timeout_s, batch_timeout_s):
    from reliquary.environment.grader.executor import (
        SandboxBatchRequest, compute_sandbox_job_id,
    )

    expected = _request()
    payload = expected.model_dump()
    payload.update(timeout_s=timeout_s, batch_timeout_s=batch_timeout_s)
    payload["job_id"] = compute_sandbox_job_id(
        protocol_version=expected.protocol_version, runtime_id=expected.runtime_id,
        code_sha256=expected.code_sha256, cases=expected.cases,
        timeout_s=timeout_s, batch_timeout_s=batch_timeout_s,
    )
    assert payload["job_id"] == expected.job_id
    assert SandboxBatchRequest.model_validate(payload) == expected


def test_execution_contract_rejects_expected_field_on_remote_case():
    from reliquary.environment.grader.executor import SandboxCase

    with pytest.raises(ValidationError, match="extra_forbidden"):
        SandboxCase(
            case_id=0,
            entry={"kind": "function", "name": "add"},
            args=[1, 2],
            kwargs={},
            expected=3,
        )


def test_remote_transport_binds_response_to_request():
    from reliquary.environment.grader.executor import RemoteSandboxExecutor

    request = _request()

    def handler(http_request: httpx.Request) -> httpx.Response:
        received = json.loads(http_request.content)
        assert received["job_id"] == request.job_id
        return httpx.Response(200, json=_result(request).model_dump(mode="json"))

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    executor = RemoteSandboxExecutor(
        "https://cpu.internal",
        runtime_id=request.runtime_id,
        client=client,
    )

    result = executor.execute(request)

    assert result.results[0].output == 3
    assert executor.health_snapshot()["requests_total"] == 1
    assert executor.health_snapshot()["failures_total"] == 0


def test_remote_transport_rejects_wrong_job_result():
    from reliquary.environment.grader.executor import (
        RemoteSandboxExecutor,
        SandboxExecutorError,
    )

    request = _request()
    wrong = _result(request).model_dump(mode="json")
    wrong["job_id"] = "0" * 64
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=wrong))
    )
    executor = RemoteSandboxExecutor(
        "https://cpu.internal",
        runtime_id=request.runtime_id,
        client=client,
    )

    with pytest.raises(SandboxExecutorError, match="response_binding_mismatch"):
        executor.execute(request)


def test_insecure_transport_is_loopback_only():
    from reliquary.environment.grader.executor import RemoteSandboxExecutor

    with pytest.raises(ValueError, match="must use HTTPS"):
        RemoteSandboxExecutor(
            "http://10.0.0.20:8443",
            runtime_id="grader-test-v1",
            allow_insecure_loopback=True,
            client=httpx.Client(),
        )


class _FakePool:
    def __init__(self) -> None:
        self.requests = []

    def execute_sandbox_batch(self, request):
        self.requests.append(request)
        return _result(request, executor_id="local")

    def health_snapshot(self):
        return {
            "pool_size": 1,
            "workers_alive": 1,
            "shutdown_complete": False,
        }

    def metrics_text(self):
        return "grader_executor_requests_total 1\n"


def test_shared_admission_deadline_reaches_executor_and_expiry_is_infrastructure():
    from reliquary.environment.grader.remote import create_cpu_executor_app
    class Pool(_FakePool):
        def execute_sandbox_batch(self, request, *, deadline_monotonic=None):
            assert 0 < deadline_monotonic - time.monotonic() <= 30
            return super().execute_sandbox_batch(request)
    request = _request()
    pool = Pool()
    app = create_cpu_executor_app(pool, runtime_id=request.runtime_id,
                                 executor_id="cpu-test", max_inflight=1)
    with TestClient(app) as client:
        for offset, status in ((30, 200), (-1, 503), (200, 503)):
            result = client.post("/v1/execute", content=request.model_dump_json(),
                headers={"content-type": "application/json",
                         "X-Reliquary-Deadline-Ms": str(int((time.time() + offset) * 1000))})
            assert result.status_code == status
    assert pool.requests == [request]


def test_cpu_executor_api_validates_runtime_and_exposes_health():
    from reliquary.environment.grader.remote import create_cpu_executor_app

    request = _request()
    pool = _FakePool()
    app = create_cpu_executor_app(
        pool,
        runtime_id=request.runtime_id,
        executor_id="cpu-test",
        max_inflight=2,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/execute",
            content=request.model_dump_json(),
            headers={"content-type": "application/json"},
        )
        health = client.get("/v1/health")
        metrics = client.get("/metrics")

    assert response.status_code == 200
    assert response.json()["executor_id"] == "cpu-test"
    assert pool.requests == [request]
    assert health.json()["status"] == "ok"
    assert health.json()["runtime_id"] == request.runtime_id
    assert health.json()["sandbox_backend"] == "runsc"
    assert health.json()["sandbox_platform"] == "unknown"
    assert "grader_executor_requests_total" in metrics.text


def test_cpu_executor_api_rejects_wrong_runtime_and_extra_expected():
    from reliquary.environment.grader.remote import create_cpu_executor_app

    pool = _FakePool()
    app = create_cpu_executor_app(
        pool,
        runtime_id="grader-test-v1",
        executor_id="cpu-test",
        max_inflight=1,
    )
    wrong_runtime = _request(runtime_id="grader-test-v2")
    leaked = _request().model_dump(mode="json")
    leaked["cases"][0]["expected"] = 3

    with TestClient(app) as client:
        mismatch = client.post(
            "/v1/execute", json=wrong_runtime.model_dump(mode="json")
        )
        rejected = client.post("/v1/execute", json=leaked)

    assert mismatch.status_code == 409
    assert rejected.status_code == 422
    assert pool.requests == []


def test_cpu_executor_api_rejects_overload_without_queueing():
    from reliquary.environment.grader.remote import create_cpu_executor_app

    entered = threading.Event()
    release = threading.Event()

    class _BlockingPool(_FakePool):
        def execute_sandbox_batch(self, request):
            entered.set()
            assert release.wait(timeout=5.0)
            return super().execute_sandbox_batch(request)

    request = _request()
    pool = _BlockingPool()
    app = create_cpu_executor_app(
        pool,
        runtime_id=request.runtime_id,
        executor_id="cpu-test",
        max_inflight=1,
    )
    first_status: list[int] = []

    with TestClient(app) as client:
        first = threading.Thread(
            target=lambda: first_status.append(
                client.post("/v1/execute", json=request.model_dump(mode="json")).status_code
            )
        )
        first.start()
        assert entered.wait(timeout=5.0)
        overloaded = client.post(
            "/v1/execute",
            json=request.model_dump(mode="json"),
        )
        release.set()
        first.join(timeout=5.0)
        health = client.get("/v1/health").json()

    assert first_status == [200]
    assert overloaded.status_code == 503
    assert health["api"] == {
        "max_inflight": 1,
        "inflight": 0,
        "peak_inflight": 1,
        "requests": {"busy": 1, "error": 0, "ok": 1},
    }


def test_remote_deadline_failure_is_unavailable_instead_of_a_score():
    from reliquary.environment.grader.executor import SandboxExecutorError
    from reliquary.environment.grader.remote import create_cpu_executor_app

    class FailedPool(_FakePool):
        def execute_sandbox_batch(self, request):
            raise SandboxExecutorError("batch_deadline_exceeded")

    request = _request()
    app = create_cpu_executor_app(
        FailedPool(), runtime_id=request.runtime_id, executor_id="cpu-test", max_inflight=1,
    )
    with TestClient(app) as client:
        response = client.post("/v1/execute", json=request.model_dump(mode="json"))
        assert response.status_code == 503
        assert "results" not in response.json()
        assert client.get("/v1/health").json()["api"]["inflight"] == 0


@pytest.mark.asyncio
async def test_cancelled_request_keeps_capacity_until_sandbox_finishes():
    from reliquary.environment.grader.remote import create_cpu_executor_app

    entered, release = threading.Event(), threading.Event()

    class BlockingPool(_FakePool):
        def execute_sandbox_batch(self, request):
            entered.set()
            assert release.wait(timeout=5)
            return super().execute_sandbox_batch(request)

    request = _request()
    app = create_cpu_executor_app(
        BlockingPool(), runtime_id=request.runtime_id,
        executor_id="cpu-test", max_inflight=1,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="https://cpu.test"
        ) as client:
            first = asyncio.create_task(client.post(
                "/v1/execute", json=request.model_dump(mode="json")
            ))
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                health = (await client.get("/v1/health")).json()
                assert health["api"]["inflight"] == 1
                overloaded = await client.post(
                    "/v1/execute", json=request.model_dump(mode="json")
                )
                assert overloaded.status_code == 503
            finally:
                release.set()


@pytest.mark.parametrize(
    "wait_s,elapsed_s,case_status,expected_score",
    [(6.0, 0.1, "ok", 1.0), (0.0, 11.0, "ok", None),
     (0.0, 6.0, "timeout", None), (0.0, 0.0, "timeout", 0.0)],
)
def test_infrastructure_deadlines_never_become_negative_labels(
    monkeypatch, wait_s, elapsed_s, case_status, expected_score,
):
    from reliquary.environment.grader import server as server_module
    from reliquary.environment.grader_client import (
        GraderClient, GraderInfrastructureError,
    )

    clock = [0.0]
    server = server_module.GraderServer(metrics_port=0, runtime_id="grader-test-v1")
    worker = SimpleNamespace(
        in_use=False, retired=False, proc=SimpleNamespace(poll=lambda: None),
    )
    calls = []

    def acquire(**_kwargs):
        clock[0] += wait_s
        return worker

    def evaluate(_worker, request):
        calls.append(request)
        clock[0] += elapsed_s
        return {"status": "ok" if len(calls) == 1 else case_status, "output": 3}

    monkeypatch.setattr(server_module.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(server, "_acquire_worker", acquire)
    monkeypatch.setattr(server, "_needs_recycle", lambda _worker: False)
    monkeypatch.setattr(server, "_evaluate_on_worker", evaluate)
    client = GraderClient()
    monkeypatch.setattr(client, "_round_trip", server._dispatch)
    cases = [_case()] if wait_s else [_case(), _case()]
    if expected_score is None:
        with pytest.raises(GraderInfrastructureError):
            client.evaluate_cases("def add(a,b): return a+b", cases, 5.0)
    else:
        assert client.evaluate_cases(
            "def add(a,b): return a+b", cases, 5.0
        ) == expected_score
    assert calls[0]["timeout_s"] == 5.0


def test_cpu_executor_api_runs_the_existing_worker_pool(tmp_path):
    from reliquary.environment.grader.remote import create_cpu_executor_app
    from reliquary.environment.grader.server import GraderServer

    runtime_id = "grader-test-v1"
    pool = GraderServer(
        pool_size=1,
        worker_argv=[sys.executable, "-m", "reliquary.environment.grader.worker"],
        metrics_port=0,
        health_path=os.fspath(tmp_path / "health.json"),
        listen_unix_socket=False,
        runtime_id=runtime_id,
    )
    pool.start()
    try:
        app = create_cpu_executor_app(
            pool,
            runtime_id=runtime_id,
            executor_id="cpu-test",
            max_inflight=2,
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/execute",
                content=_request(runtime_id=runtime_id).model_dump_json(),
                headers={"content-type": "application/json"},
            )
    finally:
        pool.stop()

    assert response.status_code == 200
    assert response.json()["results"] == [{"case_id": 0, "status": "ok", "output": 3}]


def test_trusted_coordinator_uses_remote_outputs_but_keeps_expected_local():
    from reliquary.environment.grader.server import GraderServer

    class _Remote:
        def __init__(self):
            self.request = None

        def execute(self, request):
            self.request = request
            return _result(request)

        def health_snapshot(self):
            return {"backend": "remote"}

        def close(self):
            return None

    remote = _Remote()
    server = GraderServer(
        pool_size=64,
        metrics_port=0,
        sandbox_executor=remote,
        runtime_id="grader-test-v1",
    )

    response = server._dispatch(
        {
            "req_id": "trusted-request",
            "code": "def add(a, b): return a + b",
            "cases": [_case(expected=3)],
            "timeout_s": 5.0,
        }
    )

    assert response == {
        "req_id": "trusted-request",
        "passed": 1,
        "total": 1,
        "status": "ok",
    }
    assert remote.request is not None
    serialized = remote.request.model_dump_json()
    assert "expected" not in serialized
    assert "compare" not in serialized


def test_shadow_executor_never_changes_authoritative_local_result(tmp_path):
    from reliquary.environment.grader.server import GraderServer

    completed = threading.Event()

    class _Shadow:
        def execute(self, request):
            try:
                return _result(request, output=999, executor_id="shadow")
            finally:
                completed.set()

        def health_snapshot(self):
            return {"backend": "remote"}

        def close(self):
            return None

    runtime_id = "grader-test-v1"
    server = GraderServer(
        pool_size=1,
        worker_argv=[sys.executable, "-m", "reliquary.environment.grader.worker"],
        metrics_port=0,
        health_path=os.fspath(tmp_path / "shadow-health.json"),
        shadow_executor=_Shadow(),
        listen_unix_socket=False,
        runtime_id=runtime_id,
    )
    server.start()
    try:
        response = server._dispatch(
            {
                "req_id": "shadow-request",
                "code": "def add(a, b): return a + b",
                "cases": [_case(expected=3)],
                "timeout_s": 5.0,
            }
        )
        assert completed.wait(timeout=5.0)
        for _ in range(100):
            if server.health_snapshot()["shadow"]["mismatches_total"] == 1:
                break
            time.sleep(0.01)
        health = server.health_snapshot()
    finally:
        server.stop()

    assert response["passed"] == 1
    assert response["status"] == "ok"
    assert health["execution_backend"] == "local-shadow"
    assert health["shadow"]["mismatches_total"] == 1


def test_remote_pool_replaces_sandbox_after_each_hostile_batch(tmp_path):
    from reliquary.environment.grader.server import GraderServer

    runtime_id = "grader-test-v1"
    server = GraderServer(
        pool_size=1,
        worker_argv=[sys.executable, "-m", "reliquary.environment.grader.worker"],
        metrics_port=0,
        health_path=os.fspath(tmp_path / "disposable-health.json"),
        retire_worker_after_batch=True,
        listen_unix_socket=False,
        runtime_id=runtime_id,
    )
    server.start()
    initial_pid = server._workers[0].proc.pid
    try:
        result = server.execute_sandbox_batch(_request(runtime_id=runtime_id))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if (
                server.health_snapshot()["workers_spawned_total"] >= 2
                and server._workers[0].proc.pid != initial_pid
            ):
                break
            time.sleep(0.01)
        replacement_pid = server._workers[0].proc.pid
        health = server.health_snapshot()
    finally:
        server.stop()

    assert result.results[0].output == 3
    assert replacement_pid != initial_pid
    assert health["retire_worker_after_batch"] is True
    assert health["worker_restarts_total"]["batch_isolation"] == 1


def test_real_shared_http_pool_bounds_64_calls_to_16_connections():
    from concurrent.futures import ThreadPoolExecutor
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    import time
    from reliquary.environment.grader.executor import RemoteSandboxExecutor
    lock = threading.Lock()
    active = peak = 0
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def do_POST(self):
            nonlocal active, peak
            from reliquary.environment.grader.executor import SandboxBatchRequest
            request = SandboxBatchRequest.model_validate_json(self.rfile.read(int(self.headers['Content-Length'])))
            with lock:
                active += 1
                peak = max(active, peak)
            time.sleep(.025)
            with lock:
                active -= 1
            body = _result(request).model_dump_json().encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_):
            pass
    class Server(ThreadingHTTPServer):
        request_queue_size = 128
    server = Server(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    executor = RemoteSandboxExecutor(f'http://127.0.0.1:{server.server_port}',
        runtime_id=_request().runtime_id, allow_insecure_loopback=True, max_connections=16)
    try:
        with ThreadPoolExecutor(max_workers=64) as pool:
            results = list(pool.map(lambda _: executor.execute(_request()), range(64)))
        assert len(results) == 64 and 1 < peak <= 16
    finally:
        executor.close()
        server.shutdown()
        server.server_close()
        thread.join()
