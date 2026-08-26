"""Grader server — manages a warm pool of worker subprocesses.

Listens on a Unix domain socket. Each client connection sends one JSON
request line containing untrusted code and structured hidden cases. The
server owns the hidden expected values and scoring; workers receive only
code, an entrypoint, and call arguments.

Workers are kept warm between requests: each is a long-lived
subprocess of `worker.py`. If a worker dies (broken pipe) or
times out, it is killed and respawned.

In production the worker subprocess is wrapped in `runsc` (via the
`worker_argv` constructor argument). For tests, plain `python -m
reliquary.environment.grader.worker` is used.
"""

from __future__ import annotations

import http.server
import json
import logging
import math
import os
import queue
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

from reliquary.constants import (
    GRADER_EVAL_TIMEOUT_SECONDS,
    GRADER_POOL_SIZE,
    GRADER_SOCKET_PATH,
)
from reliquary.environment.grader.executor import (
    MAX_EXECUTOR_RESPONSE_BYTES,
    SandboxBatchRequest,
    SandboxBatchResult,
    SandboxCaseResult,
    SandboxExecutor,
    SandboxExecutorError,
    grader_runtime_id,
    make_sandbox_batch_request,
    remote_executor_from_env,
    remote_executor_mode_from_env,
)
from reliquary.infrastructure.process_health import (
    DEFAULT_GRADER_HEALTH_PATH,
)

logger = logging.getLogger(__name__)

# Placeholder token in a runsc ``worker_argv``; the server substitutes a
# unique per-slot container id at spawn time. ``runsc run <id>`` refuses a
# duplicate id, so every pool worker needs its own.
GRADER_CONTAINER_ID_PLACEHOLDER = "{container_id}"
GRADER_HEALTH_HEARTBEAT_SECONDS = 30.0


def runsc_worker_argv(bundle: str) -> list[str]:
    """Production runsc argv for a sandbox worker.

    ``--ignore-cgroups`` is a GLOBAL flag and MUST sit before the ``run``
    subcommand (this runsc build rejects it after ``run``). It stops runsc
    creating a per-sandbox cgroup: gVisor never reaps that cgroup when a
    worker is killed/recycled, so on cgroup-v2 + ``cgroupns=host`` hosts they
    leak until ``/sys/fs/cgroup`` hits ``nr_descendants`` max (~65534) and
    every new ``runsc run`` fails ENOSPC — silently killing all code grading.
    The sandbox is already bounded by the bundle rlimits + ``--network=none``
    + the server's wall-clock timeout, so the cgroup is redundant here.
    """
    return ["runsc", "--network=none", "--ignore-cgroups", "run",
            "--bundle", bundle, GRADER_CONTAINER_ID_PLACEHOLDER]


# How many requests one warm worker serves before it is recycled.
GRADER_WORKER_RECYCLE_AFTER_EVALS = 64

# Wall-clock cushion added to a request's ``timeout_s`` before the server gives
# up on the worker's reply and respawns it. This — not the sandbox's
# ``RLIMIT_CPU`` — is the per-request bound.
GRADER_EVAL_WALL_CUSHION_SECONDS = 2.0

# Bytes of a dead worker's stderr kept for the death log.
GRADER_WORKER_STDERR_TAIL_BYTES = 4096


def worker_lifetime_cpu_budget_seconds(
    recycle_after_evals: int = GRADER_WORKER_RECYCLE_AFTER_EVALS,
    eval_timeout_s: float = GRADER_EVAL_TIMEOUT_SECONDS,
) -> float:
    """CPU seconds one worker may legitimately burn before it is recycled.

    ``RLIMIT_CPU`` is per-process and CUMULATIVE, so the sandbox bundle must
    cover a whole worker lifetime, not one request. Sizing it per-request
    SIGKILLs healthy workers partway through their eval budget and charges the
    death to whichever miner's case happened to be running.
    """
    return recycle_after_evals * (
        eval_timeout_s + GRADER_EVAL_WALL_CUSHION_SECONDS
    )


def _exit_label(returncode: int | None) -> str:
    """Prometheus-safe label for how a worker process ended."""
    if returncode is None:
        return "running"
    if returncode < 0:
        try:
            return signal.Signals(-returncode).name
        except ValueError:
            return f"signal{-returncode}"
    return f"exit{returncode}"


class _MetricsRegistry:
    """Tiny Prometheus-text-format counter registry. No external dep.

    Thread-safe: a single Lock serializes mutations and snapshots so
    concurrent inc()/gauge_set()/render() calls from accept/dispatch/
    HTTP-handler threads don't race on dict mutation or lose increments.
    """

    def __init__(self):
        self._counters: dict[tuple, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}
        self._lock = threading.Lock()

    def inc(self, name: str, labels: dict[str, str] | None = None, n: int = 1) -> None:
        key = (name, tuple(sorted((labels or {}).items())))
        with self._lock:
            self._counters[key] += n

    def gauge_set(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def render(self) -> str:
        # Snapshot under the lock, then format outside it — formatting
        # a frozen snapshot is safe and minimizes lock hold time.
        with self._lock:
            counters_snapshot = list(self._counters.items())
            gauges_snapshot = list(self._gauges.items())
        lines: list[str] = []
        seen: set[str] = set()
        for (name, labels), value in counters_snapshot:
            if name not in seen:
                lines.append(f"# TYPE {name} counter")
                seen.add(name)
            lbl = "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}" if labels else ""
            lines.append(f"{name}{lbl} {value}")
        for name, value in gauges_snapshot:
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        return "\n".join(lines) + "\n"


@dataclass
class Worker:
    proc: subprocess.Popen
    slot: int
    container_id: str | None = None
    in_use: bool = False
    retired: bool = False
    reaped: bool = False
    termination_recorded: bool = False
    eval_count: int = 0
    stderr_file: Any = None
    stderr_tail: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)


class GraderServer:
    """Pool of worker subprocesses, dispatched round-robin via a queue."""

    def __init__(
        self,
        socket_path: str = GRADER_SOCKET_PATH,
        pool_size: int = GRADER_POOL_SIZE,
        worker_argv: Optional[list[str]] = None,
        eval_timeout_s: float = GRADER_EVAL_TIMEOUT_SECONDS,
        recycle_after_evals: int = GRADER_WORKER_RECYCLE_AFTER_EVALS,
        metrics_port: int = 9876,
        health_path: str = DEFAULT_GRADER_HEALTH_PATH,
        health_heartbeat_s: float = GRADER_HEALTH_HEARTBEAT_SECONDS,
        sandbox_executor: SandboxExecutor | None = None,
        shadow_executor: SandboxExecutor | None = None,
        shadow_workers: int = 4,
        retire_worker_after_batch: bool = False,
        worker_acquire_timeout_s: float = 30.0,
        listen_unix_socket: bool = True,
        runtime_id: str | None = None,
    ) -> None:
        self.socket_path = socket_path
        self.pool_size = pool_size
        self.worker_argv = worker_argv or [
            sys.executable, "-m", "reliquary.environment.grader.worker"
        ]
        # Runsc mode: each worker needs a unique container id (see
        # GRADER_CONTAINER_ID_PLACEHOLDER). IDs include a per-server nonce and
        # generation counter so a process restart or rapid in-process restart
        # never collides with a sandbox that runsc is still tearing down.
        self._uses_runsc = GRADER_CONTAINER_ID_PLACEHOLDER in self.worker_argv
        self._container_instance_id = uuid.uuid4().hex[:12]
        self._container_generation = 0
        self._container_generation_lock = threading.Lock()
        self.eval_timeout_s = eval_timeout_s
        self.recycle_after_evals = recycle_after_evals
        self.metrics_port = metrics_port
        self.health_path = health_path
        if sandbox_executor is not None and shadow_executor is not None:
            raise ValueError("authoritative and shadow executors are mutually exclusive")
        if shadow_workers <= 0:
            raise ValueError("shadow_workers must be positive")
        if worker_acquire_timeout_s <= 0:
            raise ValueError("worker_acquire_timeout_s must be positive")
        self.sandbox_executor = sandbox_executor
        self.shadow_executor = shadow_executor
        self.retire_worker_after_batch = retire_worker_after_batch
        self.worker_acquire_timeout_s = worker_acquire_timeout_s
        self._shadow_pool = (
            ThreadPoolExecutor(
                max_workers=shadow_workers,
                thread_name_prefix="grader-shadow",
            )
            if shadow_executor is not None
            else None
        )
        self._shadow_lock = threading.Lock()
        self._shadow_capacity = threading.BoundedSemaphore(shadow_workers * 2)
        self._shadow_inflight = 0
        self._shadow_matches_total = 0
        self._shadow_mismatches_total = 0
        self._shadow_failures_total = 0
        self._shadow_dropped_total = 0
        self.listen_unix_socket = listen_unix_socket
        self.runtime_id = runtime_id or grader_runtime_id()
        if not 0.0 < health_heartbeat_s <= GRADER_HEALTH_HEARTBEAT_SECONDS:
            raise ValueError(
                "health_heartbeat_s must be greater than 0 and at most "
                f"{GRADER_HEALTH_HEARTBEAT_SECONDS} seconds"
            )
        self.health_heartbeat_s = health_heartbeat_s
        self._metrics = _MetricsRegistry()
        self._metrics_server: Optional[http.server.HTTPServer] = None

        self._workers: list[Worker] = []
        self._workers_lock = threading.Lock()
        self._idle: queue.Queue[Worker] = queue.Queue()
        self._listen_sock: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._health_heartbeat_stop_event = threading.Event()
        self._health_heartbeat_thread: Optional[threading.Thread] = None
        self._started_at = time.time()
        self._lifecycle_lock = threading.Lock()
        self._workers_spawned_total = 0
        self._worker_restarts: defaultdict[str, int] = defaultdict(int)
        self._worker_recycles_total = 0
        self._worker_terminations_total = 0
        self._worker_reaped_total = 0
        self._worker_reap_failures_total = 0
        self._container_deletes_total = 0
        self._container_delete_failures_total = 0
        self._shutdown_complete = False
        self._health_publish_lock = threading.Lock()

    def _start_metrics_server(self) -> None:
        registry = self._metrics

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != "/metrics":
                    self.send_response(404)
                    self.end_headers()
                    return
                body = registry.render().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args, **kwargs):
                pass  # quiet

        self._metrics_server = http.server.HTTPServer(("127.0.0.1", self.metrics_port), Handler)
        # Capture the OS-assigned port when caller passed metrics_port=0 (ephemeral).
        self.metrics_port = self._metrics_server.server_port
        threading.Thread(target=self._metrics_server.serve_forever, daemon=True).start()

    def metrics_text(self) -> str:
        """Return secret-free Prometheus metrics for local or remote export."""
        return self._metrics.render()

    def start(self) -> None:
        if self.listen_unix_socket:
            # Prep the trusted local coordinator socket.
            try:
                os.unlink(self.socket_path)
            except FileNotFoundError:
                pass
            self._listen_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._listen_sock.bind(self.socket_path)
            os.chmod(self.socket_path, 0o660)
            self._listen_sock.listen(max(1, self.pool_size * 4))

        # A remote coordinator owns no local hostile workers.  The CPU agent
        # starts this same class without a Unix listener and with local workers.
        if self.sandbox_executor is None:
            for i in range(self.pool_size):
                self._spawn_worker(i)

        if self.listen_unix_socket:
            self._accept_thread = threading.Thread(
                target=self._accept_loop,
                daemon=True,
            )
            self._accept_thread.start()
        self._start_metrics_server()
        self._publish_health()
        self._start_health_heartbeat()

    def stop(self) -> None:
        self._stop_event.set()
        if self._listen_sock is not None:
            try:
                self._listen_sock.close()
            except Exception:
                pass
        with self._workers_lock:
            workers = tuple(self._workers)
        for w in workers:
            self._terminate_worker(w)
        if self._metrics_server is not None:
            try:
                self._metrics_server.shutdown()
            except Exception:
                pass
            try:
                self._metrics_server.server_close()
            except Exception:
                pass
            self._metrics_server = None
        if self.listen_unix_socket:
            try:
                os.unlink(self.socket_path)
            except FileNotFoundError:
                pass
        if self._shadow_pool is not None:
            self._shadow_pool.shutdown(wait=True, cancel_futures=True)
            self._shadow_pool = None
        for executor in (self.sandbox_executor, self.shadow_executor):
            if executor is None:
                continue
            try:
                executor.close()
            except Exception:
                logger.exception("grader: failed to close sandbox executor")
        self._stop_health_heartbeat()
        with self._lifecycle_lock:
            self._shutdown_complete = True
        self._publish_health()

    def health_snapshot(self) -> dict[str, Any]:
        """Return the secret-free worker lifecycle snapshot exported to disk."""
        with self._workers_lock:
            workers = tuple(self._workers)
        alive = sum(
            1
            for worker in workers
            if not worker.retired and worker.proc.poll() is None
        )
        busy = sum(
            1
            for worker in workers
            if worker.in_use and not worker.retired
        )
        with self._lifecycle_lock:
            snapshot = {
                "server_pid": os.getpid(),
                "started_at": self._started_at,
                "updated_at": time.time(),
                "shutdown_complete": self._shutdown_complete,
                "pool_size": self.pool_size,
                "workers_alive": alive,
                "workers_idle": max(0, alive - busy),
                "workers_busy": busy,
                "workers_spawned_total": self._workers_spawned_total,
                "worker_restarts_total": dict(self._worker_restarts),
                "worker_recycles_total": self._worker_recycles_total,
                "worker_terminations_total": self._worker_terminations_total,
                "worker_reaped_total": self._worker_reaped_total,
                "worker_reap_failures_total": self._worker_reap_failures_total,
                "container_deletes_total": self._container_deletes_total,
                "container_delete_failures_total": (
                    self._container_delete_failures_total
                ),
                "execution_backend": self.execution_backend,
                "runtime_id": self.runtime_id,
                "retire_worker_after_batch": self.retire_worker_after_batch,
            }
        if self.sandbox_executor is not None:
            try:
                snapshot["executor"] = self.sandbox_executor.health_snapshot()
            except Exception:
                snapshot["executor"] = {
                    "backend": "remote",
                    "health_error": True,
                }
        if self.shadow_executor is not None:
            with self._shadow_lock:
                snapshot["shadow"] = {
                    "inflight": self._shadow_inflight,
                    "matches_total": self._shadow_matches_total,
                    "mismatches_total": self._shadow_mismatches_total,
                    "failures_total": self._shadow_failures_total,
                    "dropped_total": self._shadow_dropped_total,
                }
            try:
                snapshot["shadow"]["executor"] = (
                    self.shadow_executor.health_snapshot()
                )
            except Exception:
                snapshot["shadow"]["executor"] = {
                    "backend": "remote",
                    "health_error": True,
                }
        return snapshot

    @property
    def execution_backend(self) -> str:
        if self.sandbox_executor is not None:
            return "remote"
        if self.shadow_executor is not None:
            return "local-shadow"
        return "local"

    def _publish_health(self) -> None:
        if not self.health_path:
            return
        destination = os.path.abspath(self.health_path)
        temporary = (
            f"{destination}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            with self._health_publish_lock:
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                with open(temporary, "w", encoding="utf-8") as handle:
                    json.dump(
                        self.health_snapshot(),
                        handle,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(temporary, 0o644)
                os.replace(temporary, destination)
        except OSError:
            logger.exception("grader: failed to publish process health")
            try:
                os.unlink(temporary)
            except OSError:
                pass

    def _health_heartbeat_loop(self) -> None:
        next_publish = time.monotonic() + self.health_heartbeat_s
        while True:
            remaining = max(0.0, next_publish - time.monotonic())
            if self._health_heartbeat_stop_event.wait(remaining):
                return
            self._publish_health()
            next_publish += self.health_heartbeat_s
            if next_publish <= time.monotonic():
                next_publish = time.monotonic() + self.health_heartbeat_s

    def _start_health_heartbeat(self) -> None:
        self._health_heartbeat_stop_event.clear()
        thread = threading.Thread(
            target=self._health_heartbeat_loop,
            name="grader-health-heartbeat",
            daemon=True,
        )
        self._health_heartbeat_thread = thread
        thread.start()

    def _stop_health_heartbeat(self) -> None:
        self._health_heartbeat_stop_event.set()
        thread = self._health_heartbeat_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        self._health_heartbeat_thread = None

    def _next_container_id_for_slot(self, slot: int) -> str:
        with self._container_generation_lock:
            self._container_generation += 1
            generation = self._container_generation
        return (
            f"grader-worker-{self._container_instance_id}-{slot}-{generation}"
        )

    def _worker_argv_for_container(self, container_id: str | None) -> list[str]:
        """Per-worker argv. For runsc, substitute the generated container id."""
        if not self._uses_runsc:
            return self.worker_argv
        assert container_id is not None
        return [
            container_id if a == GRADER_CONTAINER_ID_PLACEHOLDER else a
            for a in self.worker_argv
        ]

    def _spawn_worker(self, slot: int) -> Worker:
        container_id = self._next_container_id_for_slot(slot) if self._uses_runsc else None
        # Keep stderr: it carries the interpreter-level cause of a death
        # (fault handler traceback, allocator abort, sandbox error). A file
        # rather than a pipe — nothing drains a pipe until the worker dies and
        # a full pipe would wedge it. Unnamed, so an ungraceful kill of this
        # process (OOM) cannot strand a pool's worth of files in /tmp.
        stderr_file = tempfile.TemporaryFile(prefix=f"grader-worker-{slot}-")
        proc = subprocess.Popen(
            self._worker_argv_for_container(container_id),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
            bufsize=1,
        )
        w = Worker(
            proc=proc,
            slot=slot,
            container_id=container_id,
            stderr_file=stderr_file,
        )
        # Insert or replace at slot.
        with self._workers_lock:
            while len(self._workers) <= slot:
                self._workers.append(w)
            self._workers[slot] = w
        with self._lifecycle_lock:
            self._workers_spawned_total += 1
        self._metrics.inc("grader_worker_spawns_total")
        self._idle.put(w)
        logger.info("grader: spawned worker slot=%d pid=%d", slot, proc.pid)
        self._publish_health()
        return w

    def _respawn_async(self, w: Worker, reason: str) -> None:
        with w.lock:
            if w.retired:
                return
            w.retired = True
        if self._stop_event.is_set():
            return
        threading.Thread(target=self._respawn, args=(w, reason), daemon=True).start()

    def _accept_loop(self) -> None:
        assert self._listen_sock is not None
        while not self._stop_event.is_set():
            try:
                conn, _ = self._listen_sock.accept()
            except OSError:
                return
            threading.Thread(target=self._handle_conn, args=(conn,), daemon=True).start()

    def _handle_conn(self, conn: socket.socket) -> None:
        try:
            buf = b""
            while True:
                chunk = conn.recv(8192)
                if not chunk:
                    break
                buf += chunk
                if b"\n" in buf:
                    break
            if not buf:
                return
            try:
                req = json.loads(buf.split(b"\n", 1)[0])
            except json.JSONDecodeError:
                conn.sendall(self._error_response("", "grader_error") + b"\n")
                return
            try:
                resp = self._dispatch(req)
            except Exception:
                # Unexpected bug in dispatch — log loudly, return a
                # graceful error to the client instead of dropping EOF.
                logger.exception("grader: dispatch raised unexpectedly")
                req_id = req.get("req_id", "")
                conn.sendall(self._error_response(req_id, "grader_error") + b"\n")
                return
            try:
                conn.sendall(json.dumps(resp).encode() + b"\n")
            except BrokenPipeError:
                logger.debug("grader: client closed before response req_id=%s", req.get("req_id", ""))
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _acquire_worker(self, timeout: float = 30.0) -> Worker | None:
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            try:
                w = self._idle.get(timeout=remaining)
            except queue.Empty:
                return None
            if w.proc.poll() is None:
                return w
            logger.warning(
                "grader: idle worker slot=%d was already dead; respawning before dispatch",
                w.slot,
            )
            self._respawn(w, reason="death")

    def _dispatch(self, req: dict) -> dict:
        cases = req.get("cases")
        req_id = req.get("req_id", "")
        if not isinstance(cases, list) or not cases:
            self._metrics.inc("grader_cases_missing_total")
            return {
                "req_id": req_id,
                "passed": 0,
                "total": 0,
                "status": "grader_error",
            }
        if not all(self._valid_case(case) for case in cases):
            self._metrics.inc("grader_case_total", {"status": "bad_case"})
            return {
                "req_id": req_id,
                "passed": 0,
                "total": len(cases),
                "status": "grader_error",
            }

        try:
            execution_request = make_sandbox_batch_request(
                runtime_id=self.runtime_id,
                code=str(req.get("code", "")),
                cases=cases,
                timeout_s=float(req.get("timeout_s", self.eval_timeout_s)),
            )
            execution_result = self.execute_sandbox_batch(execution_request)
        except (SandboxExecutorError, TypeError, ValueError) as exc:
            reason = getattr(exc, "reason", type(exc).__name__)
            logger.warning(
                "grader: sandbox executor failed req_id=%s reason=%s",
                req_id,
                reason,
            )
            self._metrics.inc(
                "grader_executor_failures_total",
                {"reason": str(reason)},
            )
            return {
                "req_id": req_id,
                "passed": 0,
                "total": len(cases),
                "status": "grader_error",
            }

        passed = 0
        for result in execution_result.results:
            case = cases[result.case_id]
            status = result.status
            if status == "ok":
                if self._outputs_match(
                    result.output,
                    case.get("expected"),
                    case.get("compare", "exact"),
                ):
                    passed += 1
                    self._metrics.inc("grader_case_total", {"status": "passed"})
                else:
                    self._metrics.inc("grader_case_total", {"status": "failed"})
                continue
            if status == "bad_output":
                self._metrics.inc("grader_bad_output_total")
                self._metrics.inc("grader_case_total", {"status": "bad_output"})
                continue
            if status == "forbidden_import":
                self._metrics.inc("grader_forbidden_import_total")
            self._metrics.inc("grader_case_total", {"status": status})
            return {
                "req_id": req_id,
                "passed": 0,
                "total": len(cases),
                "status": status,
            }
        if len(execution_result.results) != len(cases):
            self._metrics.inc(
                "grader_executor_failures_total",
                {"reason": "short_result"},
            )
            return {
                "req_id": req_id,
                "passed": 0,
                "total": len(cases),
                "status": "grader_error",
            }
        return {
            "req_id": req_id,
            "passed": passed,
            "total": len(cases),
            "status": "ok",
        }

    def execute_sandbox_batch(
        self,
        request: SandboxBatchRequest,
    ) -> SandboxBatchResult:
        """Execute without trusted expected values or reward comparison."""
        backend = "remote" if self.sandbox_executor is not None else "local"
        started = time.perf_counter()
        try:
            result = (
                self.sandbox_executor.execute(request)
                if self.sandbox_executor is not None
                else self._execute_local_sandbox_batch(request)
            )
        except SandboxExecutorError:
            self._metrics.inc(
                "grader_executor_requests_total",
                {"backend": backend, "status": "error"},
            )
            raise
        self._metrics.inc(
            "grader_executor_requests_total",
            {"backend": backend, "status": "ok"},
        )
        self._metrics.gauge_set(
            "grader_executor_last_duration_seconds",
            time.perf_counter() - started,
        )
        if self.shadow_executor is not None:
            self._submit_shadow_execution(request, result)
        return result

    @staticmethod
    def _shadow_signature(result: SandboxBatchResult) -> tuple[Any, ...]:
        return (
            result.protocol_version,
            result.job_id,
            result.attempt,
            result.runtime_id,
            tuple(
                (case.case_id, case.status, case.output)
                for case in result.results
            ),
        )

    def _submit_shadow_execution(
        self,
        request: SandboxBatchRequest,
        authoritative: SandboxBatchResult,
    ) -> None:
        """Mirror work without adding latency or authority to the new host."""
        pool = self._shadow_pool
        executor = self.shadow_executor
        if pool is None or executor is None:
            return
        if not self._shadow_capacity.acquire(blocking=False):
            with self._shadow_lock:
                self._shadow_dropped_total += 1
            self._metrics.inc(
                "grader_shadow_requests_total",
                {"status": "dropped"},
            )
            return
        with self._shadow_lock:
            self._shadow_inflight += 1
        try:
            future = pool.submit(executor.execute, request)
        except RuntimeError:
            self._shadow_capacity.release()
            with self._shadow_lock:
                self._shadow_inflight -= 1
                self._shadow_failures_total += 1
            self._metrics.inc(
                "grader_shadow_requests_total",
                {"status": "submit_error"},
            )
            return

        def _completed(completed: Future[SandboxBatchResult]) -> None:
            status = "match"
            try:
                shadow = completed.result()
                if self._shadow_signature(shadow) != self._shadow_signature(
                    authoritative
                ):
                    status = "mismatch"
            except Exception as exc:
                status = "error"
                logger.warning(
                    "grader: shadow executor failed job=%s reason=%s",
                    request.job_id,
                    getattr(exc, "reason", type(exc).__name__),
                )
            with self._shadow_lock:
                self._shadow_inflight -= 1
                if status == "match":
                    self._shadow_matches_total += 1
                elif status == "mismatch":
                    self._shadow_mismatches_total += 1
                else:
                    self._shadow_failures_total += 1
            self._shadow_capacity.release()
            self._metrics.inc(
                "grader_shadow_requests_total",
                {"status": status},
            )
            if status == "mismatch":
                logger.warning(
                    "grader: shadow result mismatch job=%s",
                    request.job_id,
                )

        future.add_done_callback(_completed)

    def _execute_local_sandbox_batch(
        self,
        request: SandboxBatchRequest,
    ) -> SandboxBatchResult:
        started = time.perf_counter()
        worker = self._acquire_worker(timeout=self.worker_acquire_timeout_s)
        if worker is None:
            raise SandboxExecutorError("capacity_unavailable")
        results: list[SandboxCaseResult] = []
        result_payload_bytes = 0
        try:
            worker.in_use = True
            for case in request.cases:
                worker_request = {
                    "req_id": f"{request.job_id}:{case.case_id}",
                    "code": request.code,
                    "entry": case.entry,
                    "args": case.args,
                    "kwargs": case.kwargs,
                    "timeout_s": request.timeout_s,
                }
                response = self._evaluate_on_worker(worker, worker_request)
                status = str(response.get("status", "grader_error"))
                output = response.get("output")
                try:
                    result_payload_bytes += len(
                        json.dumps(
                            output,
                            allow_nan=False,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ) + 128
                except (TypeError, ValueError) as exc:
                    raise SandboxExecutorError(
                        "malformed_worker_result"
                    ) from exc
                if result_payload_bytes > MAX_EXECUTOR_RESPONSE_BYTES:
                    raise SandboxExecutorError("worker_output_too_large")
                results.append(
                    SandboxCaseResult(
                        case_id=case.case_id,
                        status=status,
                        output=output,
                    )
                )
                if status not in {"ok", "bad_output"}:
                    break
        except (TypeError, ValueError) as exc:
            raise SandboxExecutorError("malformed_worker_result") from exc
        finally:
            # If the worker was respawned, its replacement is already idle.
            worker.in_use = False
            if (
                self.retire_worker_after_batch
                and not worker.retired
                and worker.proc.poll() is None
            ):
                # The remote execution tier treats each batch as one hostile
                # job. Never hand its interpreter or sandbox to another miner.
                self._respawn_async(worker, reason="batch_isolation")
            elif (
                not worker.retired
                and worker.proc.poll() is None
                and not self._needs_recycle(worker)
            ):
                self._idle.put(worker)
        return SandboxBatchResult(
            protocol_version=request.protocol_version,
            job_id=request.job_id,
            attempt=request.attempt,
            runtime_id=request.runtime_id,
            executor_id="local",
            results=results,
            wall_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _evaluate_on_worker(self, w: Worker, req: dict) -> dict:
        timeout_s = float(req.get("timeout_s", self.eval_timeout_s))
        # The per-request bound. RLIMIT_CPU in the sandbox bundle is a
        # per-lifetime backstop and must never be the binding constraint.
        deadline = time.time() + timeout_s + GRADER_EVAL_WALL_CUSHION_SECONDS

        try:
            assert w.proc.stdin is not None and w.proc.stdout is not None
            w.proc.stdin.write(json.dumps(req) + "\n")
            w.proc.stdin.flush()
        except BrokenPipeError:
            # Worker died between checks. Respawn and return failure for this req.
            self._respawn_async(w, reason="death")
            self._metrics.inc("grader_eval_total", {"status": "crash"})
            return {
                "req_id": req.get("req_id", ""),
                "output": None, "status": "crash",
            }

        # Read response with wall-clock timeout (no asyncio — keep stdlib only).
        line_holder: dict = {}

        def reader():
            try:
                line_holder["line"] = w.proc.stdout.readline()
            except Exception:
                line_holder["line"] = ""

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        t.join(timeout=max(0.1, deadline - time.time()))

        if t.is_alive():
            # Timeout: kill and respawn worker; return timeout status.
            self._respawn_async(w, reason="timeout")
            self._metrics.inc("grader_eval_total", {"status": "timeout"})
            return {
                "req_id": req.get("req_id", ""),
                "output": None, "status": "timeout",
            }

        line = line_holder.get("line", "")
        if not line:
            self._respawn_async(w, reason="death")
            self._metrics.inc("grader_eval_total", {"status": "crash"})
            return {
                "req_id": req.get("req_id", ""),
                "output": None, "status": "crash",
            }
        try:
            resp = json.loads(line)
        except json.JSONDecodeError:
            self._metrics.inc("grader_eval_total", {"status": "grader_error"})
            return {
                "req_id": req.get("req_id", ""),
                "output": None, "status": "grader_error",
            }
        self._metrics.inc("grader_eval_total", {"status": resp.get("status", "ok")})
        self._metrics.gauge_set("grader_pool_busy_workers", self.pool_size - self._idle.qsize())
        w.eval_count += 1
        return resp

    def _delete_container(self, container_id: str | None, timeout: float = 2.0) -> None:
        if not self._uses_runsc or not container_id:
            return
        try:
            completed = subprocess.run(
                [self.worker_argv[0], "delete", "--force", container_id],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=timeout,
            )
            with self._lifecycle_lock:
                if completed.returncode == 0:
                    self._container_deletes_total += 1
                else:
                    self._container_delete_failures_total += 1
        except Exception:
            with self._lifecycle_lock:
                self._container_delete_failures_total += 1
        self._publish_health()

    def _delete_container_async(self, container_id: str | None) -> None:
        if not self._uses_runsc or not container_id or self._stop_event.is_set():
            return
        threading.Thread(
            target=self._delete_container,
            args=(container_id,),
            daemon=True,
        ).start()

    def _drain_worker_stderr(self, w: Worker) -> str:
        """Read and release a worker's stderr, keeping only the tail."""
        handle = w.stderr_file
        w.stderr_file = None
        tail = ""
        if handle is not None:
            try:
                size = handle.seek(0, os.SEEK_END)
                handle.seek(max(0, size - GRADER_WORKER_STDERR_TAIL_BYTES))
                tail = handle.read().decode("utf-8", "replace").strip()
            except Exception:
                pass
            finally:
                try:
                    handle.close()
                except Exception:
                    pass
        w.stderr_tail = tail
        return tail

    def _terminate_worker(self, w: Worker, *, delete_container: bool = True) -> None:
        with w.lock:
            first_termination = not w.termination_recorded
            w.termination_recorded = True
        if first_termination:
            with self._lifecycle_lock:
                self._worker_terminations_total += 1
        reaped = False
        try:
            if w.proc.poll() is None:
                w.proc.kill()
            else:
                reaped = True
        except Exception:
            pass
        try:
            w.proc.wait(timeout=2.0)
            reaped = True
        except subprocess.TimeoutExpired:
            try:
                w.proc.kill()
                w.proc.wait(timeout=2.0)
                reaped = True
            except Exception:
                # Best effort: the supervisor will still try to replace the
                # worker, and runsc delete below cleans up stale containers.
                pass
        except Exception:
            pass
        with self._lifecycle_lock:
            if reaped and not w.reaped:
                w.reaped = True
                self._worker_reaped_total += 1
            elif not reaped and first_termination:
                self._worker_reap_failures_total += 1
        self._drain_worker_stderr(w)
        if delete_container:
            self._delete_container(w.container_id)
        self._publish_health()

    def _respawn(self, w: Worker, reason: str = "death") -> None:
        old_container_id = w.container_id
        # Read the exit status BEFORE terminating: _terminate_worker kills a
        # still-running worker, which would overwrite how it originally ended.
        exit_label = _exit_label(w.proc.poll())
        self._terminate_worker(w, delete_container=False)
        if reason not in {"recycle", "batch_isolation"}:
            logger.warning(
                "grader: worker slot=%d ended reason=%s exit=%s stderr=%s",
                w.slot, reason, exit_label, w.stderr_tail or "<empty>",
            )
        self._metrics.inc(
            "grader_worker_restarts_total",
            {"reason": reason, "exit": exit_label},
        )
        with self._lifecycle_lock:
            self._worker_restarts[reason] += 1
            if reason == "recycle":
                self._worker_recycles_total += 1
        if self._stop_event.is_set():
            self._delete_container(old_container_id)
            self._publish_health()
            return
        try:
            self._spawn_worker(w.slot)
            self._delete_container_async(old_container_id)
        except Exception:
            # Spawning a replacement failed (runsc missing, OS limits, …).
            # Log loudly so an operator can investigate; the pool is now
            # one slot smaller until the next successful respawn.
            logger.exception(
                "grader: respawn failed for slot=%d — pool degraded", w.slot,
            )
        self._publish_health()

    def _needs_recycle(self, w: Worker) -> bool:
        if w.eval_count >= self.recycle_after_evals:
            logger.info("grader: recycling worker slot=%d after %d evals", w.slot, w.eval_count)
            self._respawn_async(w, reason="recycle")
            return True
        return False

    @staticmethod
    def _error_response(req_id: str, status: str) -> bytes:
        return json.dumps({
            "req_id": req_id, "passed": 0, "total": 0, "status": status,
        }).encode()

    @classmethod
    def _valid_case(cls, case: Any) -> bool:
        if not isinstance(case, dict):
            return False
        entry = case.get("entry")
        if not isinstance(entry, dict):
            return False
        kind = entry.get("kind")
        if kind == "function":
            if not isinstance(entry.get("name"), str):
                return False
        elif kind == "method":
            if not isinstance(entry.get("class_name"), str) or not isinstance(entry.get("method"), str):
                return False
        else:
            return False
        if not isinstance(case.get("args", []), list):
            return False
        if not isinstance(case.get("kwargs", {}), dict):
            return False
        if case.get("compare", "exact") != "exact":
            return False
        if "expected" not in case:
            return False
        return cls._is_json_safe(case.get("expected"))

    @classmethod
    def _is_json_safe(cls, value: Any) -> bool:
        if value is None or isinstance(value, (bool, str)):
            return True
        if isinstance(value, int) and not isinstance(value, bool):
            return True
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, list):
            return all(cls._is_json_safe(v) for v in value)
        if isinstance(value, dict):
            return all(isinstance(k, str) and cls._is_json_safe(v) for k, v in value.items())
        return False

    @classmethod
    def _outputs_match(cls, output: Any, expected: Any, compare: str) -> bool:
        if compare != "exact":
            return False
        return cls._json_equal(output, expected)

    @classmethod
    def _json_equal(cls, left: Any, right: Any) -> bool:
        if isinstance(left, bool) or isinstance(right, bool):
            return type(left) is type(right) and left == right
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if isinstance(left, float) or isinstance(right, float):
                return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-9)
            return left == right
        if left is None or right is None or isinstance(left, str) or isinstance(right, str):
            return type(left) is type(right) and left == right
        if isinstance(left, list) and isinstance(right, list):
            return len(left) == len(right) and all(cls._json_equal(a, b) for a, b in zip(left, right))
        if isinstance(left, dict) and isinstance(right, dict):
            return (
                set(left.keys()) == set(right.keys())
                and all(cls._json_equal(left[k], right[k]) for k in left)
            )
        return False


def main() -> None:
    """Entrypoint for running the server as a standalone process."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", default=GRADER_SOCKET_PATH)
    parser.add_argument("--pool-size", type=int, default=GRADER_POOL_SIZE)
    parser.add_argument("--timeout", type=float, default=GRADER_EVAL_TIMEOUT_SECONDS)
    parser.add_argument(
        "--use-runsc", action="store_true",
        help="Wrap each worker in `runsc` (production mode).",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=int(os.environ.get("GRADER_METRICS_PORT", "9876")),
        help="Loopback Prometheus metrics port; use 0 for an ephemeral port.",
    )
    parser.add_argument(
        "--health-path",
        default=os.environ.get(
            "GRADER_HEALTH_PATH", DEFAULT_GRADER_HEALTH_PATH
        ),
        help="Atomic JSON worker lifecycle snapshot consumed by /health.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    remote_executor = remote_executor_from_env()
    remote_mode = (
        remote_executor_mode_from_env()
        if remote_executor is not None
        else "local"
    )
    if remote_mode == "remote" and args.use_runsc:
        parser.error(
            "--use-runsc cannot be combined with an authoritative remote executor"
        )

    if remote_mode == "remote":
        worker_argv = [
            sys.executable,
            "-m",
            "reliquary.environment.grader.worker",
        ]
    elif args.use_runsc:
        # Production: runsc loads the OCI bundle which already invokes worker.py.
        bundle = os.environ.get(
            "GRADER_BUNDLE_PATH",
            "/opt/reliquary/reliquary/environment/grader/bundle",
        )
        worker_argv = runsc_worker_argv(bundle)
    else:
        worker_argv = [sys.executable, "-m", "reliquary.environment.grader.worker"]

    server = GraderServer(
        socket_path=args.socket,
        pool_size=args.pool_size,
        worker_argv=worker_argv,
        eval_timeout_s=args.timeout,
        metrics_port=args.metrics_port,
        health_path=args.health_path,
        sandbox_executor=(remote_executor if remote_mode == "remote" else None),
        shadow_executor=(remote_executor if remote_mode == "shadow" else None),
        runtime_id=(remote_executor.runtime_id if remote_executor else None),
    )
    shutdown = threading.Event()

    def _request_shutdown(signum: int, _frame: Any) -> None:
        logger.info("grader: received signal %d; shutting down", signum)
        shutdown.set()

    previous_handlers = {
        signum: signal.signal(signum, _request_shutdown)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        server.start()
        logger.info(
            "grader server listening on %s (backend=%s pool=%d)",
            args.socket,
            server.execution_backend,
            args.pool_size,
        )
        while not shutdown.wait(60.0):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        server.stop()


if __name__ == "__main__":
    main()
