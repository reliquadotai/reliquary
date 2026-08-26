"""Zero-secret remote CPU sandbox agent.

This service deliberately exposes execution, health, and metrics only.  It
does not load datasets, expected outputs, wallets, validator state, or object
storage credentials.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import socket
import ssl
import sys
import threading
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from reliquary.constants import GRADER_EVAL_TIMEOUT_SECONDS, GRADER_POOL_SIZE
from reliquary.environment.grader.executor import (
    EXECUTOR_PROTOCOL_VERSION,
    MAX_EXECUTOR_REQUEST_BYTES,
    MAX_EXECUTOR_RESPONSE_BYTES,
    SandboxBatchRequest,
    SandboxBatchResult,
    SandboxExecutorError,
    grader_runtime_id,
)
from reliquary.environment.grader.server import GraderServer, runsc_worker_argv


logger = logging.getLogger(__name__)
_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/+~-]{0,191}$")


class _ExecutorBusy(RuntimeError):
    pass


def _bounded_json_response(payload: dict[str, Any], *, status_code: int) -> Response:
    body = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return Response(
        content=body,
        status_code=status_code,
        media_type="application/json",
    )


def create_cpu_executor_app(
    pool: GraderServer,
    *,
    runtime_id: str,
    executor_id: str,
    max_inflight: int,
) -> FastAPI:
    if _IDENTITY_RE.fullmatch(runtime_id) is None:
        raise ValueError("invalid runtime_id")
    if _IDENTITY_RE.fullmatch(executor_id) is None:
        raise ValueError("invalid executor_id")
    if max_inflight <= 0:
        raise ValueError("max_inflight must be positive")

    app = FastAPI(
        title="Reliquary CPU executor",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    capacity = threading.BoundedSemaphore(max_inflight)

    @app.post("/v1/execute")
    async def execute(request: Request) -> Response:
        content_type = request.headers.get("content-type", "")
        if content_type.split(";", 1)[0].strip().lower() != "application/json":
            raise HTTPException(status_code=415, detail="application/json required")
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                parsed_content_length = int(content_length)
                if parsed_content_length < 0:
                    raise ValueError
                if parsed_content_length > MAX_EXECUTOR_REQUEST_BYTES:
                    raise HTTPException(status_code=413, detail="request too large")
            except ValueError as exc:
                raise HTTPException(
                    status_code=400, detail="invalid content length"
                ) from exc

        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_EXECUTOR_REQUEST_BYTES:
                raise HTTPException(status_code=413, detail="request too large")
        try:
            execution_request = SandboxBatchRequest.model_validate_json(body)
        except ValidationError as exc:
            logger.info(
                "cpu executor rejected malformed request: %s", exc.error_count()
            )
            raise HTTPException(
                status_code=422, detail="invalid execution request"
            ) from exc
        if execution_request.runtime_id != runtime_id:
            raise HTTPException(status_code=409, detail="runtime mismatch")

        def _execute() -> SandboxBatchResult:
            if not capacity.acquire(timeout=min(1.0, execution_request.timeout_s)):
                raise _ExecutorBusy
            try:
                result = pool.execute_sandbox_batch(execution_request)
            finally:
                capacity.release()
            return SandboxBatchResult(
                **{
                    **result.model_dump(mode="python"),
                    "executor_id": executor_id,
                }
            )

        try:
            result = await run_in_threadpool(_execute)
        except _ExecutorBusy as exc:
            raise HTTPException(
                status_code=503, detail="executor capacity unavailable"
            ) from exc
        except SandboxExecutorError as exc:
            logger.warning(
                "cpu executor failed job=%s reason=%s",
                execution_request.job_id,
                exc.reason,
            )
            raise HTTPException(status_code=503, detail="executor failure") from exc

        response_body = result.model_dump_json().encode("utf-8")
        if len(response_body) > MAX_EXECUTOR_RESPONSE_BYTES:
            logger.warning(
                "cpu executor response exceeded limit job=%s", execution_request.job_id
            )
            raise HTTPException(status_code=503, detail="executor response too large")
        return Response(content=response_body, media_type="application/json")

    @app.get("/v1/health")
    async def health() -> Response:
        pool_health = pool.health_snapshot()
        alive = int(pool_health.get("workers_alive", 0))
        expected = int(pool_health.get("pool_size", 0))
        status = (
            "ok"
            if alive == expected and not pool_health.get("shutdown_complete")
            else "degraded"
        )
        return _bounded_json_response(
            {
                "status": status,
                "protocol_version": EXECUTOR_PROTOCOL_VERSION,
                "runtime_id": runtime_id,
                "executor_id": executor_id,
                "pool": pool_health,
            },
            status_code=200 if status == "ok" else 503,
        )

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(
            content=pool.metrics_text(),
            media_type="text/plain; version=0.0.4",
        )

    return app


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _default_executor_id() -> str:
    candidate = socket.gethostname().strip().lower()
    return candidate if _IDENTITY_RE.fullmatch(candidate) else "cpu-executor"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reliquary remote CPU executor")
    parser.add_argument(
        "--host", default=os.environ.get("RELIQUARY_CPU_EXECUTOR_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("RELIQUARY_CPU_EXECUTOR_PORT", "8443")),
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=int(
            os.environ.get("RELIQUARY_CPU_EXECUTOR_POOL_SIZE", str(GRADER_POOL_SIZE))
        ),
    )
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=int(os.environ.get("RELIQUARY_CPU_EXECUTOR_MAX_INFLIGHT", "0")),
    )
    parser.add_argument("--timeout", type=float, default=GRADER_EVAL_TIMEOUT_SECONDS)
    parser.add_argument(
        "--runtime-id",
        default=os.environ.get("RELIQUARY_GRADER_RUNTIME_ID", grader_runtime_id()),
    )
    parser.add_argument(
        "--executor-id",
        default=os.environ.get("RELIQUARY_CPU_EXECUTOR_ID", _default_executor_id()),
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=int(os.environ.get("GRADER_METRICS_PORT", "9876")),
    )
    parser.add_argument(
        "--health-path",
        default=os.environ.get(
            "GRADER_HEALTH_PATH", "/tmp/reliquary-cpu-executor-health.json"
        ),
    )
    parser.add_argument(
        "--tls-cert", default=os.environ.get("RELIQUARY_CPU_EXECUTOR_TLS_CERT", "")
    )
    parser.add_argument(
        "--tls-key", default=os.environ.get("RELIQUARY_CPU_EXECUTOR_TLS_KEY", "")
    )
    parser.add_argument(
        "--client-ca", default=os.environ.get("RELIQUARY_CPU_EXECUTOR_CLIENT_CA", "")
    )
    parser.add_argument(
        "--allow-insecure-loopback",
        action="store_true",
        default=_flag("RELIQUARY_CPU_EXECUTOR_ALLOW_INSECURE_LOOPBACK"),
    )
    parser.add_argument(
        "--allow-unsandboxed",
        action="store_true",
        default=_flag("RELIQUARY_ALLOW_UNSANDBOXED_GRADER"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    if not 1 <= args.pool_size <= 512:
        raise SystemExit("pool-size must be within [1, 512]")
    max_inflight = args.max_inflight or args.pool_size
    if not 1 <= max_inflight <= 1024:
        raise SystemExit("max-inflight must be within [1, 1024]")

    use_runsc = shutil.which("runsc") is not None
    bundle = os.environ.get(
        "GRADER_BUNDLE_PATH",
        "/opt/reliquary/reliquary/environment/grader/bundle",
    )
    bundle_python = os.path.join(bundle, "rootfs", "usr", "local", "bin", "python3")
    if not use_runsc or not os.path.isfile(bundle_python):
        if not args.allow_unsandboxed:
            raise SystemExit("runsc and the pinned grader bundle are required")
        logger.warning("starting UNSANDBOXED loopback-only CPU executor")
        if args.host not in {"127.0.0.1", "::1", "localhost"}:
            raise SystemExit("unsandboxed executor may bind only to loopback")
        worker_argv = [sys.executable, "-m", "reliquary.environment.grader.worker"]
    else:
        worker_argv = runsc_worker_argv(bundle)

    tls_values = (args.tls_cert, args.tls_key, args.client_ca)
    if not all(tls_values):
        if not args.allow_insecure_loopback:
            raise SystemExit("TLS cert, key, and client CA are required")
        if args.host not in {"127.0.0.1", "::1", "localhost"}:
            raise SystemExit("insecure executor may bind only to loopback")
    elif not all(os.path.isfile(path) for path in tls_values):
        raise SystemExit("one or more TLS files do not exist")

    pool = GraderServer(
        pool_size=args.pool_size,
        worker_argv=worker_argv,
        eval_timeout_s=args.timeout,
        metrics_port=args.metrics_port,
        health_path=args.health_path,
        retire_worker_after_batch=not _flag(
            "RELIQUARY_CPU_EXECUTOR_REUSE_WORKERS",
            "0",
        ),
        worker_acquire_timeout_s=min(2.0, args.timeout),
        listen_unix_socket=False,
        runtime_id=args.runtime_id,
    )
    app = create_cpu_executor_app(
        pool,
        runtime_id=args.runtime_id,
        executor_id=args.executor_id,
        max_inflight=max_inflight,
    )

    pool.start()
    try:
        logger.info(
            "cpu executor ready host=%s port=%d pool=%d runtime=%s",
            args.host,
            args.port,
            args.pool_size,
            args.runtime_id,
        )
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=1,
            limit_concurrency=max_inflight + 16,
            timeout_keep_alive=5,
            proxy_headers=False,
            server_header=False,
            ssl_certfile=args.tls_cert or None,
            ssl_keyfile=args.tls_key or None,
            ssl_ca_certs=args.client_ca or None,
            ssl_cert_reqs=(ssl.CERT_REQUIRED if all(tls_values) else ssl.CERT_NONE),
        )
    finally:
        pool.stop()


if __name__ == "__main__":
    main()
