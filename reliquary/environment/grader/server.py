"""Grader server — manages a warm pool of worker subprocesses.

Listens on a Unix domain socket. Each client connection sends one
JSON request line; the server picks an idle worker from the pool,
pipes the request to its stdin, reads the response from stdout,
and writes it back to the client.

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
import os
import queue
import socket
import subprocess
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from reliquary.constants import (
    GRADER_EVAL_TIMEOUT_SECONDS,
    GRADER_POOL_SIZE,
    GRADER_SOCKET_PATH,
)

logger = logging.getLogger(__name__)


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
    in_use: bool = False
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
        recycle_after_evals: int = 1000,
        metrics_port: int = 9876,
    ) -> None:
        self.socket_path = socket_path
        self.pool_size = pool_size
        self.worker_argv = worker_argv or [
            "python", "-m", "reliquary.environment.grader.worker"
        ]
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
                    self.send_response(404); self.end_headers(); return
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
            try:
                w.proc.kill()
            except Exception:
                pass
            try:
                w.proc.wait(timeout=2.0)
            except Exception:
                # subprocess never died within 2 s — best effort, move on.
                pass
        if self._metrics_server is not None:
            try:
                self._metrics_server.shutdown()
            except Exception:
                pass
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass

    def _spawn_worker(self, slot: int) -> Worker:
        proc = subprocess.Popen(
            self.worker_argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        w = Worker(proc=proc, slot=slot)
        # Insert or replace at slot.
        while len(self._workers) <= slot:
            self._workers.append(w)
        self._workers[slot] = w
        self._idle.put(w)
        logger.info("grader: spawned worker slot=%d pid=%d", slot, proc.pid)
        return w

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
            conn.sendall(json.dumps(resp).encode() + b"\n")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _dispatch(self, req: dict) -> dict:
        # Acquire a worker (blocks if all busy).
        try:
            w = self._idle.get(timeout=30.0)
        except queue.Empty:
            return {
                "req_id": req.get("req_id", ""),
                "passed": 0, "total": int(len(req.get("tests", []))), "status": "grader_error",
            }
        try:
            return self._evaluate_on_worker(w, req)
        finally:
            # If worker was respawned (timeout/crash), the new one is already in
            # the idle queue. Otherwise return this one.
            if w.proc.poll() is None and not self._needs_recycle(w):
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
            self._respawn(w)
            self._metrics.inc("grader_eval_total", {"status": "crash"})
            return {
                "req_id": req.get("req_id", ""),
                "passed": 0, "total": int(len(req.get("tests", []))), "status": "crash",
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
            try:
                w.proc.kill()
            except Exception:
                pass
            self._respawn(w)
            self._metrics.inc("grader_eval_total", {"status": "timeout"})
            return {
                "req_id": req.get("req_id", ""),
                "passed": 0, "total": int(len(req.get("tests", []))), "status": "timeout",
            }

        line = line_holder.get("line", "")
        if not line:
            self._respawn(w)
            self._metrics.inc("grader_eval_total", {"status": "crash"})
            return {
                "req_id": req.get("req_id", ""),
                "passed": 0, "total": int(len(req.get("tests", []))), "status": "crash",
            }
        try:
            resp = json.loads(line)
        except json.JSONDecodeError:
            self._metrics.inc("grader_eval_total", {"status": "grader_error"})
            return {
                "req_id": req.get("req_id", ""),
                "passed": 0, "total": int(len(req.get("tests", []))), "status": "grader_error",
            }
        self._metrics.inc("grader_eval_total", {"status": resp.get("status", "ok")})
        self._metrics.gauge_set("grader_pool_busy_workers", self.pool_size - self._idle.qsize())
        w.eval_count += 1
        return resp

    def _respawn(self, w: Worker) -> None:
        try:
            w.proc.kill()
        except Exception:
            pass
        self._metrics.inc("grader_worker_restarts_total", {"reason": "death"})
        try:
            self._spawn_worker(w.slot)
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
            self._metrics.inc("grader_worker_restarts_total", {"reason": "recycle"})
            self._respawn(w)
            return True
        return False

    @staticmethod
    def _error_response(req_id: str, status: str) -> bytes:
        return json.dumps({
            "req_id": req_id, "passed": 0, "total": 0, "status": status,
        }).encode()


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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.use_runsc:
        # Production: runsc loads the OCI bundle which already invokes worker.py.
        bundle = os.environ.get(
            "GRADER_BUNDLE_PATH",
            "/opt/reliquary/reliquary/environment/grader/bundle",
        )
        worker_argv = ["runsc", "--network=none", "run",
                       "--bundle", bundle, "grader-worker"]
    else:
        worker_argv = ["python", "-m", "reliquary.environment.grader.worker"]

    server = GraderServer(
        socket_path=args.socket,
        pool_size=args.pool_size,
        worker_argv=worker_argv,
        eval_timeout_s=args.timeout,
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
