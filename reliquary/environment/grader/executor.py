"""Versioned sandbox execution contract and remote transport.

The trusted grader keeps expected values and reward comparison.  An executor
receives only miner code, entrypoints, public call arguments, and resource
limits, then returns bounded JSON-safe outputs/statuses.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import ssl
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


EXECUTOR_PROTOCOL_VERSION = 2
MAX_EXECUTOR_CODE_BYTES = 1024 * 1024
MAX_EXECUTOR_CASES = 256
MAX_EXECUTOR_REQUEST_BYTES = 4 * 1024 * 1024
MAX_EXECUTOR_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_EXECUTOR_TIMEOUT_SECONDS = 30.0
MAX_EXECUTOR_BATCH_TIMEOUT_SECONDS = 120.0
REMOTE_EXECUTOR_TIMEOUT_HEADROOM_SECONDS = 5.0

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUNTIME_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/+~-]{0,191}$")
_CASE_STATUSES = frozenset(
    {
        "ok",
        "bad_output",
        "forbidden_import",
        "runtime_error",
        "tampered",
        "timeout",
        "crash",
        "grader_error",
    }
)


def _is_json_safe(value: Any, *, depth: int = 0) -> bool:
    if depth > 64:
        return False
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, int) and not isinstance(value, bool):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_safe(item, depth=depth + 1) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_safe(item, depth=depth + 1)
            for key, item in value.items()
        )
    return False


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def grader_runtime_id() -> str:
    """Digest the exact worker and OCI policy that determine Code outputs."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for relative in ("worker.py", "bundle/config.json"):
        payload = (root / relative).read_bytes()
        digest.update(relative.encode("ascii"))
        digest.update(b"\0")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return f"grader-sha256:{digest.hexdigest()}"


class SandboxCase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    case_id: int = Field(ge=0, le=MAX_EXECUTOR_CASES - 1)
    entry: dict[str, Any]
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_case(self) -> "SandboxCase":
        kind = self.entry.get("kind")
        if kind == "function":
            if not isinstance(self.entry.get("name"), str):
                raise ValueError("function entry requires name")
        elif kind == "method":
            if not isinstance(self.entry.get("class_name"), str) or not isinstance(
                self.entry.get("method"), str
            ):
                raise ValueError("method entry requires class_name and method")
        else:
            raise ValueError("unsupported entry kind")
        if not _is_json_safe(self.entry):
            raise ValueError("entry is not bounded JSON data")
        if not _is_json_safe(self.args) or not _is_json_safe(self.kwargs):
            raise ValueError("case arguments are not bounded JSON data")
        return self


def compute_sandbox_job_id(
    *,
    protocol_version: int,
    runtime_id: str,
    code_sha256: str,
    cases: list[SandboxCase],
    timeout_s: float,
    batch_timeout_s: float,
) -> str:
    material = {
        "protocol_version": protocol_version,
        "runtime_id": runtime_id,
        "code_sha256": code_sha256,
        "cases": [case.model_dump(mode="json") for case in cases],
        "timeout_s": timeout_s,
        "batch_timeout_s": batch_timeout_s,
    }
    return hashlib.sha256(_canonical_json(material)).hexdigest()


class SandboxBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    protocol_version: int = Field(default=EXECUTOR_PROTOCOL_VERSION)
    job_id: str = Field(min_length=64, max_length=64)
    attempt: int = Field(default=0, ge=0, le=1_000_000)
    runtime_id: str = Field(min_length=1, max_length=192)
    code: str
    code_sha256: str = Field(min_length=64, max_length=64)
    cases: list[SandboxCase] = Field(min_length=1, max_length=MAX_EXECUTOR_CASES)
    timeout_s: float = Field(gt=0.0, le=MAX_EXECUTOR_TIMEOUT_SECONDS)
    batch_timeout_s: float = Field(
        gt=0.0,
        le=MAX_EXECUTOR_BATCH_TIMEOUT_SECONDS,
    )

    @field_validator("job_id", "code_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError("must be lowercase SHA-256")
        return value

    @field_validator("runtime_id")
    @classmethod
    def validate_runtime_id(cls, value: str) -> str:
        if _RUNTIME_ID_RE.fullmatch(value) is None:
            raise ValueError("invalid runtime_id")
        return value

    @field_validator("code")
    @classmethod
    def validate_code_size(cls, value: str) -> str:
        if len(value.encode("utf-8")) > MAX_EXECUTOR_CODE_BYTES:
            raise ValueError("code exceeds executor byte limit")
        return value

    @model_validator(mode="after")
    def validate_content_binding(self) -> "SandboxBatchRequest":
        if self.batch_timeout_s < self.timeout_s:
            raise ValueError("batch timeout cannot be shorter than case timeout")
        actual_code_sha256 = hashlib.sha256(self.code.encode("utf-8")).hexdigest()
        if actual_code_sha256 != self.code_sha256:
            raise ValueError("code_sha256 mismatch")
        expected_job_id = compute_sandbox_job_id(
            protocol_version=self.protocol_version,
            runtime_id=self.runtime_id,
            code_sha256=self.code_sha256,
            cases=self.cases,
            timeout_s=self.timeout_s,
            batch_timeout_s=self.batch_timeout_s,
        )
        if self.job_id != expected_job_id:
            raise ValueError("job_id mismatch")
        if self.protocol_version != EXECUTOR_PROTOCOL_VERSION:
            raise ValueError("unsupported executor protocol")
        return self


class SandboxCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    case_id: int = Field(ge=0, le=MAX_EXECUTOR_CASES - 1)
    status: str = Field(min_length=1, max_length=32)
    output: Any = None

    @model_validator(mode="after")
    def validate_result(self) -> "SandboxCaseResult":
        if self.status not in _CASE_STATUSES:
            raise ValueError("unsupported sandbox case status")
        if not _is_json_safe(self.output):
            raise ValueError("sandbox output is not bounded JSON data")
        return self


class SandboxBatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    protocol_version: int = Field(default=EXECUTOR_PROTOCOL_VERSION)
    job_id: str = Field(min_length=64, max_length=64)
    attempt: int = Field(ge=0, le=1_000_000)
    runtime_id: str = Field(min_length=1, max_length=192)
    executor_id: str = Field(min_length=1, max_length=192)
    results: list[SandboxCaseResult] = Field(
        min_length=1,
        max_length=MAX_EXECUTOR_CASES,
    )
    wall_ms: float = Field(ge=0.0, le=3_600_000.0)

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, value: str) -> str:
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError("must be lowercase SHA-256")
        return value

    @field_validator("runtime_id", "executor_id")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        if _RUNTIME_ID_RE.fullmatch(value) is None:
            raise ValueError("invalid executor identity")
        return value

    @model_validator(mode="after")
    def validate_protocol(self) -> "SandboxBatchResult":
        if self.protocol_version != EXECUTOR_PROTOCOL_VERSION:
            raise ValueError("unsupported executor protocol")
        return self


def make_sandbox_batch_request(
    *,
    runtime_id: str,
    code: str,
    cases: list[dict[str, Any]],
    timeout_s: float,
    attempt: int = 0,
) -> SandboxBatchRequest:
    sandbox_cases = [
        SandboxCase(
            case_id=index,
            entry=case["entry"],
            args=case.get("args", []),
            kwargs=case.get("kwargs", {}),
        )
        for index, case in enumerate(cases)
    ]
    code_sha256 = hashlib.sha256(code.encode("utf-8")).hexdigest()
    batch_timeout_s = min(
        MAX_EXECUTOR_BATCH_TIMEOUT_SECONDS,
        max(timeout_s, timeout_s * len(sandbox_cases)),
    )
    job_id = compute_sandbox_job_id(
        protocol_version=EXECUTOR_PROTOCOL_VERSION,
        runtime_id=runtime_id,
        code_sha256=code_sha256,
        cases=sandbox_cases,
        timeout_s=timeout_s,
        batch_timeout_s=batch_timeout_s,
    )
    return SandboxBatchRequest(
        protocol_version=EXECUTOR_PROTOCOL_VERSION,
        job_id=job_id,
        attempt=attempt,
        runtime_id=runtime_id,
        code=code,
        code_sha256=code_sha256,
        cases=sandbox_cases,
        timeout_s=timeout_s,
        batch_timeout_s=batch_timeout_s,
    )


class SandboxExecutorError(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = str(reason)
        super().__init__(f"sandbox executor failure: {self.reason}")


@runtime_checkable
class SandboxExecutor(Protocol):
    def execute(self, request: SandboxBatchRequest) -> SandboxBatchResult: ...

    def health_snapshot(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


class RemoteSandboxExecutor:
    """Synchronous, bounded mTLS transport to one CPU executor agent."""

    def __init__(
        self,
        endpoint: str,
        *,
        runtime_id: str,
        ca_cert: str | None = None,
        client_cert: str | None = None,
        client_key: str | None = None,
        allow_insecure_loopback: bool = False,
        client: httpx.Client | None = None,
    ) -> None:
        parsed = urlparse(endpoint)
        host = (parsed.hostname or "").lower()
        if parsed.scheme == "https":
            if client is None and not all((ca_cert, client_cert, client_key)):
                raise ValueError(
                    "HTTPS remote executor requires CA, client cert, and key"
                )
        elif not (
            parsed.scheme == "http"
            and allow_insecure_loopback
            and host in {"127.0.0.1", "::1", "localhost"}
        ):
            raise ValueError("remote executor must use HTTPS with mTLS")
        if (
            not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("invalid remote executor endpoint")
        if _RUNTIME_ID_RE.fullmatch(runtime_id) is None:
            raise ValueError("invalid remote executor runtime_id")

        self.endpoint = endpoint.rstrip("/")
        self.runtime_id = runtime_id
        self._lock = threading.Lock()
        self._requests_total = 0
        self._failures_total = 0
        self._last_success_at: float | None = None
        self._last_failure_reason: str | None = None
        self._last_latency_ms: float | None = None
        self._success_latencies_ms: deque[float] = deque(maxlen=1024)
        self._owns_client = client is None

        if client is not None:
            self._client = client
        elif parsed.scheme == "https":
            context = ssl.create_default_context(cafile=ca_cert)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.load_cert_chain(certfile=client_cert, keyfile=client_key)
            self._client = httpx.Client(verify=context)
        else:
            self._client = httpx.Client()

    def _record_success(self, latency_ms: float) -> None:
        with self._lock:
            self._requests_total += 1
            self._last_success_at = time.time()
            self._last_failure_reason = None
            self._last_latency_ms = latency_ms
            self._success_latencies_ms.append(latency_ms)

    def _record_failure(self, reason: str, latency_ms: float) -> None:
        with self._lock:
            self._requests_total += 1
            self._failures_total += 1
            self._last_failure_reason = reason
            self._last_latency_ms = latency_ms

    def execute(self, request: SandboxBatchRequest) -> SandboxBatchResult:
        if request.runtime_id != self.runtime_id:
            raise SandboxExecutorError("local_runtime_mismatch")
        body = request.model_dump_json().encode("utf-8")
        if len(body) > MAX_EXECUTOR_REQUEST_BYTES:
            raise SandboxExecutorError("request_too_large")

        started = time.perf_counter()
        try:
            response_body = bytearray()
            with self._client.stream(
                "POST",
                f"{self.endpoint}/v1/execute",
                content=body,
                headers={"Content-Type": "application/json"},
                timeout=(
                    request.batch_timeout_s
                    + REMOTE_EXECUTOR_TIMEOUT_HEADROOM_SECONDS
                ),
            ) as response:
                if response.status_code != 200:
                    raise SandboxExecutorError(f"http_{response.status_code}")
                for chunk in response.iter_bytes():
                    response_body.extend(chunk)
                    if len(response_body) > MAX_EXECUTOR_RESPONSE_BYTES:
                        raise SandboxExecutorError("response_too_large")
            raw = json.loads(
                response_body,
                parse_constant=_reject_nonfinite_json,
            )
            result = SandboxBatchResult.model_validate(raw)
            expected_case_ids = [case.case_id for case in request.cases]
            actual_case_ids = [case.case_id for case in result.results]
            if (
                result.job_id != request.job_id
                or result.attempt != request.attempt
                or result.runtime_id != request.runtime_id
                or actual_case_ids != expected_case_ids[: len(actual_case_ids)]
                or (
                    len(actual_case_ids) < len(expected_case_ids)
                    and result.results[-1].status in {"ok", "bad_output"}
                )
            ):
                raise SandboxExecutorError("response_binding_mismatch")
        except SandboxExecutorError as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._record_failure(exc.reason, latency_ms)
            raise
        except (httpx.HTTPError, ValueError, TypeError, json.JSONDecodeError) as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            reason = f"{type(exc).__name__}"
            self._record_failure(reason, latency_ms)
            raise SandboxExecutorError(reason) from exc

        latency_ms = (time.perf_counter() - started) * 1000.0
        self._record_success(latency_ms)
        return result

    def health_snapshot(self) -> dict[str, Any]:
        with self._lock:
            latencies = sorted(self._success_latencies_ms)
            p50 = latencies[(len(latencies) - 1) // 2] if latencies else None
            p95 = (
                latencies[max(0, math.ceil(len(latencies) * 0.95) - 1)]
                if latencies
                else None
            )
            return {
                "backend": "remote",
                "runtime_id": self.runtime_id,
                "requests_total": self._requests_total,
                "failures_total": self._failures_total,
                "last_success_at": self._last_success_at,
                "last_failure_reason": self._last_failure_reason,
                "last_latency_ms": self._last_latency_ms,
                "success_latency_samples": len(latencies),
                "success_latency_p50_ms": p50,
                "success_latency_p95_ms": p95,
            }

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


def remote_executor_from_env() -> RemoteSandboxExecutor | None:
    endpoint = os.environ.get("RELIQUARY_GRADER_EXECUTOR_URL", "").strip()
    if not endpoint:
        return None
    allow_insecure = os.environ.get(
        "RELIQUARY_GRADER_EXECUTOR_ALLOW_INSECURE_LOOPBACK",
        "0",
    ).strip().lower() in {"1", "true", "yes", "on"}
    runtime_id = os.environ.get(
        "RELIQUARY_GRADER_RUNTIME_ID",
        grader_runtime_id(),
    ).strip()
    return RemoteSandboxExecutor(
        endpoint,
        runtime_id=runtime_id,
        ca_cert=os.environ.get("RELIQUARY_GRADER_EXECUTOR_CA") or None,
        client_cert=os.environ.get("RELIQUARY_GRADER_EXECUTOR_CERT") or None,
        client_key=os.environ.get("RELIQUARY_GRADER_EXECUTOR_KEY") or None,
        allow_insecure_loopback=allow_insecure,
    )


def remote_executor_mode_from_env() -> str:
    """Return the safe rollout mode for a configured remote executor.

    Shadow is deliberately the default: adding an endpoint cannot silently
    make a new machine authoritative for validation results.
    """
    mode = (
        os.environ.get(
            "RELIQUARY_GRADER_EXECUTOR_MODE",
            "shadow",
        )
        .strip()
        .lower()
    )
    if mode not in {"shadow", "remote"}:
        raise ValueError("RELIQUARY_GRADER_EXECUTOR_MODE must be 'shadow' or 'remote'")
    return mode
