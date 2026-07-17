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
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from reliquary.constants import (
    GRADER_EVAL_TIMEOUT_SECONDS,
    GRADER_POOL_SIZE,
    GRADER_SOCKET_PATH,
)

logger = logging.getLogger(__name__)

# Placeholder token in a runsc ``worker_argv``; the server substitutes a
# unique per-slot container id at spawn time. ``runsc run <id>`` refuses a
# duplicate id, so every pool worker needs its own.
GRADER_CONTAINER_ID_PLACEHOLDER = "{container_id}"


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
    eval_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class GraderServer:
    """Pool of worker subprocesses, dispatched round-robin via a queue."""

    def __init__(
        self,
        socket_path: str = GRADER_SOCKET_PATH,
        pool_size: int = GRADER_POOL_SIZE,
        worker_argv: Optional[list[str]] = None,
        eval_timeout_s: float = GRADER_EVAL_TIMEOUT_SECONDS,
        recycle_after_evals: int = 64,
        metrics_port: int = 9876,
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
        self._metrics = _MetricsRegistry()
        self._metrics_server: Optional[http.server.HTTPServer] = None

        self._workers: list[Worker] = []
        self._idle: queue.Queue[Worker] = queue.Queue()
        self._listen_sock: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

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

    def start(self) -> None:
        # Prep socket.
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass
        self._listen_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listen_sock.bind(self.socket_path)
        os.chmod(self.socket_path, 0o660)
        self._listen_sock.listen(self.pool_size * 4)

        # Spawn workers.
        for i in range(self.pool_size):
            self._spawn_worker(i)

        # Accept loop in a background thread.
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()
        self._start_metrics_server()

    def stop(self) -> None:
        self._stop_event.set()
        if self._listen_sock is not None:
            try:
                self._listen_sock.close()
            except Exception:
                pass
        for w in self._workers:
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
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass

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
        proc = subprocess.Popen(
            self._worker_argv_for_container(container_id),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        w = Worker(proc=proc, slot=slot, container_id=container_id)
        # Insert or replace at slot.
        while len(self._workers) <= slot:
            self._workers.append(w)
        self._workers[slot] = w
        self._idle.put(w)
        logger.info("grader: spawned worker slot=%d pid=%d", slot, proc.pid)
        return w

    def _respawn_async(self, w: Worker, reason: str) -> None:
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

        # Acquire a worker (blocks if all busy).
        w = self._acquire_worker(timeout=30.0)
        if w is None:
            return {
                "req_id": req_id,
                "passed": 0, "total": len(cases), "status": "grader_error",
            }
        try:
            passed = 0
            w.in_use = True
            for i, case in enumerate(cases):
                if not self._valid_case(case):
                    self._metrics.inc("grader_case_total", {"status": "bad_case"})
                    return {
                        "req_id": req_id,
                        "passed": 0,
                        "total": len(cases),
                        "status": "grader_error",
                    }

                worker_req = {
                    "req_id": f"{req_id}:{i}",
                    "code": req.get("code", ""),
                    "entry": case["entry"],
                    "args": case.get("args", []),
                    "kwargs": case.get("kwargs", {}),
                    "timeout_s": req.get("timeout_s", self.eval_timeout_s),
                }
                resp = self._evaluate_on_worker(w, worker_req)
                status = resp.get("status", "grader_error")
                if status == "ok":
                    if self._outputs_match(resp.get("output"), case.get("expected"), case.get("compare", "exact")):
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
            return {
                "req_id": req_id,
                "passed": passed,
                "total": len(cases),
                "status": "ok",
            }
        finally:
            # If worker was respawned (timeout/crash), the new one is already in
            # the idle queue. Otherwise return this one.
            w.in_use = False
            if not w.retired and w.proc.poll() is None and not self._needs_recycle(w):
                self._idle.put(w)

    def _evaluate_on_worker(self, w: Worker, req: dict) -> dict:
        timeout_s = float(req.get("timeout_s", self.eval_timeout_s))
        deadline = time.time() + timeout_s + 2.0  # outer wall-clock cushion

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
            subprocess.run(
                [self.worker_argv[0], "delete", "--force", container_id],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=timeout,
            )
        except Exception:
            pass

    def _delete_container_async(self, container_id: str | None) -> None:
        if not self._uses_runsc or not container_id or self._stop_event.is_set():
            return
        threading.Thread(
            target=self._delete_container,
            args=(container_id,),
            daemon=True,
        ).start()

    def _terminate_worker(self, w: Worker, *, delete_container: bool = True) -> None:
        try:
            if w.proc.poll() is None:
                w.proc.kill()
        except Exception:
            pass
        try:
            w.proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                w.proc.kill()
                w.proc.wait(timeout=2.0)
            except Exception:
                # Best effort: the supervisor will still try to replace the
                # worker, and runsc delete below cleans up stale containers.
                pass
        except Exception:
            pass
        if delete_container:
            self._delete_container(w.container_id)

    def _respawn(self, w: Worker, reason: str = "death") -> None:
        old_container_id = w.container_id
        self._terminate_worker(w, delete_container=False)
        self._metrics.inc("grader_worker_restarts_total", {"reason": reason})
        if self._stop_event.is_set():
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.use_runsc:
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
    )
    server.start()
    logger.info("grader server listening on %s (pool=%d)", args.socket, args.pool_size)

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
