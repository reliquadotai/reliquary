"""Contract, transport, and service tests for the remote CPU executor."""

from __future__ import annotations

import json
import os
import sys
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError


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
