"""Tests for the trusted grader server."""

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest


@pytest.fixture
def grader_server():
    from reliquary.environment.grader.server import GraderServer

    tmp = tempfile.TemporaryDirectory(prefix="g-", dir="/tmp")
    sock_path = os.path.join(tmp.name, "g.sock")
    server = GraderServer(
        socket_path=sock_path,
        pool_size=2,
        worker_argv=[sys.executable, "-m", "reliquary.environment.grader.worker"],
        eval_timeout_s=5.0,
        metrics_port=0,
        health_path=os.path.join(tmp.name, "health.json"),
    )
    server.start()
    deadline = time.time() + 5.0
    while not os.path.exists(sock_path) and time.time() < deadline:
        time.sleep(0.05)
    yield server
    server.stop()
    tmp.cleanup()


def _case(entry=None, args=None, expected=3):
    return {
        "entry": entry or {"kind": "function", "name": "add"},
        "args": args if args is not None else [1, 2],
        "kwargs": {},
        "expected": expected,
        "compare": "exact",
    }


def _request(sock_path: str, code: str, cases: list[dict], timeout_s: float = 5.0) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(10.0)
        s.connect(sock_path)
        req = {"req_id": "test-req", "code": code, "cases": cases, "timeout_s": timeout_s}
        s.sendall(json.dumps(req).encode() + b"\n")
        buf = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
            if b"\n" in buf:
                break
        return json.loads(buf.split(b"\n", 1)[0])


def _wait_for_health_update(
    health_path: Path,
    previous_updated_at: float,
    *,
    timeout_s: float = 2.0,
) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        snapshot = json.loads(health_path.read_text(encoding="utf-8"))
        if snapshot["updated_at"] > previous_updated_at:
            return snapshot
        time.sleep(0.01)
    raise AssertionError("grader health heartbeat did not advance updated_at")


def test_server_grades_correct_code(grader_server):
    resp = _request(
        grader_server.socket_path,
        code="def add(a,b): return a+b",
        cases=[_case(), _case(args=[0, 0], expected=0)],
    )
    assert resp == {"req_id": "test-req", "passed": 2, "total": 2, "status": "ok"}


def test_server_grades_incorrect_code(grader_server):
    resp = _request(
        grader_server.socket_path,
        code="def add(a,b): return a-b",
        cases=[_case()],
    )
    assert resp["status"] == "ok"
    assert resp["passed"] == 0
    assert resp["total"] == 1


def test_server_treats_invalid_trusted_case_as_grader_error(grader_server):
    resp = _request(
        grader_server.socket_path,
        code="def add(a,b): return a+b",
        cases=[{"entry": {"kind": "function", "name": "add"}}],
    )

    assert resp["status"] == "grader_error"
    assert resp["passed"] == 0
    assert resp["total"] == 1


def test_server_supports_method_entrypoint(grader_server):
    code = "class Solution:\n    def inc(self, x): return x + 1"
    resp = _request(
        grader_server.socket_path,
        code=code,
        cases=[_case({"kind": "method", "class_name": "Solution", "method": "inc"}, [9], 10)],
    )
    assert resp["status"] == "ok"
    assert resp["passed"] == 1


def test_server_float_compare_uses_tolerance(grader_server):
    resp = _request(
        grader_server.socket_path,
        code="def f(): return 0.1 + 0.2",
        cases=[_case({"kind": "function", "name": "f"}, [], 0.3)],
    )
    assert resp["passed"] == 1


def test_always_equal_object_does_not_pass(grader_server):
    code = """
class AlwaysEqual:
    def __eq__(self, other): return True
def f():
    return AlwaysEqual()
"""
    resp = _request(
        grader_server.socket_path,
        code=code,
        cases=[_case({"kind": "function", "name": "f"}, [], 123)],
    )
    assert resp["status"] == "ok"
    assert resp["passed"] == 0


def test_runtime_error_does_not_pass_expected_none(grader_server):
    resp = _request(
        grader_server.socket_path,
        code="def f():\n    raise RuntimeError('boom')",
        cases=[_case({"kind": "function", "name": "f"}, [], None)],
    )
    assert resp["status"] == "runtime_error"
    assert resp["passed"] == 0


def test_hidden_expected_is_not_sent_to_worker(tmp_path):
    from reliquary.environment.grader.server import GraderServer, Worker

    captured = []

    class _Stdin:
        def write(self, line):
            captured.append(json.loads(line))
        def flush(self):
            pass

    class _Stdout:
        def readline(self):
            return json.dumps({"req_id": "x", "output": 999, "status": "ok"}) + "\n"

    class _Proc:
        pid = 1
        stdin = _Stdin()
        stdout = _Stdout()
        def poll(self):
            return None
        def kill(self):
            pass

    server = GraderServer(socket_path=str(tmp_path / "g.sock"), pool_size=1, metrics_port=0)
    server._idle.put(Worker(proc=_Proc(), slot=0))
    resp = server._dispatch({
        "req_id": "r",
        "code": "def f(): return 999",
        "cases": [_case({"kind": "function", "name": "f"}, [], 999)],
        "timeout_s": 5.0,
    })
    assert resp["passed"] == 1
    assert captured
    assert "expected" not in captured[0]
    assert "cases" not in captured[0]


def test_server_handles_concurrent_requests(grader_server):
    results = []
    errors = []

    def submit():
        try:
            results.append(_request(
                grader_server.socket_path,
                code="def f(): return 1",
                cases=[_case({"kind": "function", "name": "f"}, [], 1)],
            ))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=submit) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)

    assert not errors
    assert len(results) == 4
    assert all(r["passed"] == 1 and r["total"] == 1 for r in results)


def test_server_returns_timeout_status_for_infinite_loop(grader_server):
    resp = _request(
        grader_server.socket_path,
        code="while True: pass",
        cases=[_case()],
        timeout_s=1.0,
    )
    assert resp["status"] == "timeout"
    assert resp["passed"] == 0


def test_pool_recovers_after_timeout(grader_server):
    bad = _request(grader_server.socket_path, code="while True: pass", cases=[_case()], timeout_s=1.0)
    assert bad["status"] == "timeout"
    good = _request(
        grader_server.socket_path,
        code="def f(): return 42",
        cases=[_case({"kind": "function", "name": "f"}, [], 42)],
    )
    assert good["status"] == "ok"
    assert good["passed"] == 1


def test_server_returns_grader_error_on_invalid_json_request(grader_server):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(5.0)
        s.connect(grader_server.socket_path)
        s.sendall(b"{this is not json\n")
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
    resp = json.loads(buf.split(b"\n", 1)[0])
    assert resp["status"] == "grader_error"


def test_runsc_workers_get_unique_container_ids(monkeypatch, tmp_path):
    from reliquary.environment.grader import server as srv

    captured: list[list[str]] = []

    class _FakeProc:
        pid = 4321
        stdin = None
        stdout = None
        def poll(self):
            return None

    monkeypatch.setattr(srv.subprocess, "Popen", lambda argv, **kw: captured.append(list(argv)) or _FakeProc())

    s = srv.GraderServer(
        socket_path=str(tmp_path / "g.sock"),
        pool_size=3,
        worker_argv=["runsc", "--network=none", "run", "--bundle", "/b",
                     srv.GRADER_CONTAINER_ID_PLACEHOLDER],
        metrics_port=0,
    )
    for i in range(3):
        s._spawn_worker(i)

    container_ids = [argv[-1] for argv in captured]
    assert len(set(container_ids)) == 3
    assert srv.GRADER_CONTAINER_ID_PLACEHOLDER not in container_ids


def test_runsc_container_ids_are_unique_across_server_restarts(
    monkeypatch, tmp_path
):
    from reliquary.environment.grader import server as srv

    captured: list[list[str]] = []

    class _FakeProc:
        pid = 4321
        stdin = None
        stdout = None

        def poll(self):
            return None

    monkeypatch.setattr(
        srv.subprocess,
        "Popen",
        lambda argv, **_kwargs: captured.append(list(argv)) or _FakeProc(),
    )

    for index in range(2):
        server = srv.GraderServer(
            socket_path=str(tmp_path / f"grader-{index}.sock"),
            pool_size=1,
            worker_argv=[
                "runsc", "run", "--bundle", "/b",
                srv.GRADER_CONTAINER_ID_PLACEHOLDER,
            ],
            metrics_port=0,
        )
        server._spawn_worker(0)

    assert len({argv[-1] for argv in captured}) == 2


def test_production_runsc_argv_disables_cgroups():
    """Production runsc argv must pass `--ignore-cgroups` as a GLOBAL flag
    (before `run`) so runsc never creates a per-sandbox cgroup — gVisor doesn't
    reap those on kill/recycle and they leak until `runsc run` fails ENOSPC,
    silently killing all code grading."""
    from reliquary.environment.grader import server as srv

    argv = srv.runsc_worker_argv("/opt/grader/bundle")

    assert argv[0] == "runsc"
    assert "--ignore-cgroups" in argv
    # Global flag: must precede the `run` subcommand, else this runsc build
    # rejects it ("flag provided but not defined").
    assert argv.index("--ignore-cgroups") < argv.index("run")
    assert argv[-1] == srv.GRADER_CONTAINER_ID_PLACEHOLDER
    assert "/opt/grader/bundle" in argv


def test_production_runsc_argv_can_select_bare_metal_kvm():
    from reliquary.environment.grader import server as srv

    argv = srv.runsc_worker_argv("/opt/grader/bundle", platform="kvm")

    assert "--platform=kvm" in argv
    assert argv.index("--platform=kvm") < argv.index("run")


def test_production_runsc_argv_rejects_unknown_platform():
    from reliquary.environment.grader import server as srv

    with pytest.raises(ValueError, match="runsc platform"):
        srv.runsc_worker_argv("/opt/grader/bundle", platform="not-a-platform")


def test_runsc_respawn_uses_fresh_id_before_cleanup(monkeypatch, tmp_path):
    from reliquary.environment.grader import server as srv

    deletes: list[list[str]] = []
    popens: list[list[str]] = []

    class _FakeProc:
        pid = 4321
        stdin = None
        stdout = None
        def poll(self):
            return None

    monkeypatch.setattr(srv.subprocess, "Popen", lambda argv, **kw: popens.append(list(argv)) or _FakeProc())
    monkeypatch.setattr(srv.subprocess, "run", lambda argv, **kw: deletes.append(list(argv)))
    monkeypatch.setattr(
        srv.threading,
        "Thread",
        lambda target, args=(), kwargs=None, daemon=None: type(
            "_T",
            (),
            {"start": lambda self: target(*args, **(kwargs or {}))},
        )(),
    )

    s = srv.GraderServer(
        socket_path=str(tmp_path / "g.sock"),
        pool_size=1,
        worker_argv=["runsc", "run", "--bundle", "/b", srv.GRADER_CONTAINER_ID_PLACEHOLDER],
        metrics_port=0,
    )
    old = s._spawn_worker(0)
    s._respawn(old, reason="death")

    container_ids = [argv[-1] for argv in popens]
    assert len(container_ids) == 2
    assert len(set(container_ids)) == 2
    assert deletes == [["runsc", "delete", "--force", old.container_id]]


def test_metrics_endpoint_exposes_eval_and_case_counters(grader_server):
    import urllib.request

    _request(grader_server.socket_path, code="def f(): return 1", cases=[
        _case({"kind": "function", "name": "f"}, [], 1),
    ])
    time.sleep(0.1)
    resp = urllib.request.urlopen(
        f"http://127.0.0.1:{grader_server.metrics_port}/metrics", timeout=2.0,
    )
    body = resp.read().decode()
    assert "grader_eval_total" in body
    assert "grader_case_total" in body


def test_stop_releases_metrics_listener_for_restart(tmp_path):
    from reliquary.environment.grader.server import GraderServer

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    for index in range(2):
        server = GraderServer(
            socket_path=str(tmp_path / f"grader-{index}.sock"),
            pool_size=0,
            metrics_port=port,
        )
        server._start_metrics_server()
        server.stop()


def test_idle_health_heartbeat_refreshes_process_telemetry():
    from reliquary.environment.grader.server import GraderServer
    from reliquary.infrastructure.process_health import (
        GRADER_HEALTH_STALE_SECONDS,
        collect_process_health,
    )

    with tempfile.TemporaryDirectory(prefix="ghb-", dir="/tmp") as tmp:
        root = Path(tmp)
        health_path = root / "grader-health.json"
        server = GraderServer(
            socket_path=str(root / "grader.sock"),
            pool_size=0,
            metrics_port=0,
            health_path=str(health_path),
            health_heartbeat_s=0.05,
        )
        server.start()
        try:
            initial = json.loads(health_path.read_text(encoding="utf-8"))
            refreshed = _wait_for_health_update(
                health_path,
                initial["updated_at"],
            )
            telemetry = collect_process_health(grader_health_path=health_path)

            assert GRADER_HEALTH_STALE_SECONDS == 120.0
            assert refreshed["workers_alive"] == 0
            assert refreshed["workers_busy"] == 0
            assert refreshed["shutdown_complete"] is False
            assert telemetry["grader"]["updated_at"] >= refreshed["updated_at"]
            assert telemetry["grader"]["age_seconds"] < 1.0
            assert telemetry["grader"]["stale"] is False
            assert "grader_health_stale" not in telemetry["warning_reasons"]
        finally:
            server.stop()


def test_shutdown_joins_health_heartbeat_and_publishes_completion():
    from reliquary.environment.grader.server import GraderServer

    with tempfile.TemporaryDirectory(prefix="ghb-", dir="/tmp") as tmp:
        root = Path(tmp)
        health_path = root / "grader-health.json"
        server = GraderServer(
            socket_path=str(root / "grader.sock"),
            pool_size=0,
            metrics_port=0,
            health_path=str(health_path),
            health_heartbeat_s=0.05,
        )
        server.start()
        heartbeat = server._health_heartbeat_thread
        assert heartbeat is not None and heartbeat.is_alive()

        server.stop()

        assert not heartbeat.is_alive()
        assert server._health_heartbeat_thread is None
        shutdown = json.loads(health_path.read_text(encoding="utf-8"))
        assert shutdown["shutdown_complete"] is True
        final_updated_at = shutdown["updated_at"]
        time.sleep(0.15)
        after_wait = json.loads(health_path.read_text(encoding="utf-8"))
        assert after_wait["updated_at"] == final_updated_at


def test_recycle_and_shutdown_publish_reaped_worker_counts():
    from reliquary.environment.grader.server import GraderServer

    short_tmp = tempfile.TemporaryDirectory(prefix="gr-", dir="/tmp")
    socket_path = os.path.join(short_tmp.name, "grader.sock")
    health_path = Path(short_tmp.name) / "grader-health.json"
    server = GraderServer(
        socket_path=socket_path,
        pool_size=1,
        worker_argv=[
            sys.executable,
            "-m",
            "reliquary.environment.grader.worker",
        ],
        eval_timeout_s=5.0,
        recycle_after_evals=1,
        metrics_port=0,
        health_path=str(health_path),
    )
    server.start()
    try:
        response = _request(
            socket_path,
            code="def f(): return 7",
            cases=[_case({"kind": "function", "name": "f"}, [], 7)],
        )
        assert response["passed"] == 1
        deadline = time.time() + 5.0
        while (
            server.health_snapshot()["worker_recycles_total"] < 1
            and time.time() < deadline
        ):
            time.sleep(0.05)
    finally:
        server.stop()

    snapshot = json.loads(health_path.read_text(encoding="utf-8"))
    assert snapshot["shutdown_complete"] is True
    assert snapshot["workers_alive"] == 0
    assert snapshot["workers_spawned_total"] >= 2
    assert snapshot["worker_recycles_total"] >= 1
    assert snapshot["worker_reaped_total"] == snapshot[
        "worker_terminations_total"
    ]
    assert snapshot["worker_reap_failures_total"] == 0
    short_tmp.cleanup()


def test_grader_sigterm_publishes_clean_shutdown():
    short_tmp = tempfile.TemporaryDirectory(prefix="gr-signal-", dir="/tmp")
    socket_path = os.path.join(short_tmp.name, "grader.sock")
    health_path = Path(short_tmp.name) / "grader-health.json"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "reliquary.environment.grader.server",
            "--socket",
            socket_path,
            "--pool-size",
            "0",
            "--metrics-port",
            "0",
            "--health-path",
            str(health_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 5.0
        while not os.path.exists(socket_path) and time.time() < deadline:
            time.sleep(0.05)
        assert os.path.exists(socket_path)

        process.terminate()
        assert process.wait(timeout=5.0) == 0
        snapshot = json.loads(health_path.read_text(encoding="utf-8"))
        assert snapshot["shutdown_complete"] is True
        assert snapshot["workers_alive"] == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5.0)
        short_tmp.cleanup()


def test_bundle_cpu_rlimit_covers_a_whole_worker_lifetime():
    """``RLIMIT_CPU`` is per-process CUMULATIVE, but a pool worker serves
    ``recycle_after_evals`` requests before it is recycled.

    Sizing it as if it were a per-request timeout kills healthy workers
    mid-request once their *accumulated* CPU crosses the limit. The miner whose
    case happened to be running is then blamed for a sandbox crash it did not
    cause. The per-request bound is the server's wall-clock reader deadline,
    not this rlimit.
    """
    from reliquary.environment.grader import server as srv

    config = json.loads(
        (Path(srv.__file__).parent / "bundle" / "config.json").read_text()
    )
    rlimits = {entry["type"]: entry for entry in config["process"]["rlimits"]}
    cpu = rlimits["RLIMIT_CPU"]
    required = srv.worker_lifetime_cpu_budget_seconds()

    assert cpu["soft"] >= required
    assert cpu["hard"] >= required


def test_worker_death_records_the_exit_signal(grader_server):
    """The exit status is the only evidence of *why* a worker died.

    Without it a SIGKILL from an exhausted rlimit is indistinguishable from a
    segfault in miner code, and the pool's failure mode cannot be diagnosed
    from telemetry at all.
    """
    worker = grader_server._workers[0]
    worker.proc.kill()
    worker.proc.wait(timeout=5.0)

    grader_server._respawn(worker, reason="death")

    body = grader_server._metrics.render()
    assert 'grader_worker_restarts_total{exit="SIGKILL",reason="death"}' in body


def test_worker_stderr_is_surfaced_when_it_dies(tmp_path, caplog):
    """A dying worker's stderr carries the interpreter-level cause (fault
    handler traceback, allocator abort). Discarding it leaves the death
    unexplainable."""
    import logging

    from reliquary.environment.grader.server import GraderServer

    server = GraderServer(
        socket_path=str(tmp_path / "g.sock"),
        pool_size=1,
        worker_argv=[
            sys.executable,
            "-c",
            "import sys, time; sys.stderr.write('worker-death-marker\\n');"
            " sys.stderr.flush(); time.sleep(30)",
        ],
        eval_timeout_s=1.0,
        metrics_port=0,
        health_path=str(tmp_path / "health.json"),
    )
    server.start()
    try:
        worker = server._workers[0]
        deadline = time.time() + 5.0
        while worker.proc.poll() is None and time.time() < deadline:
            time.sleep(0.05)
        worker.proc.kill()
        worker.proc.wait(timeout=5.0)

        with caplog.at_level(
            logging.WARNING, logger="reliquary.environment.grader.server"
        ):
            server._respawn(worker, reason="death")
    finally:
        server.stop()

    assert "worker-death-marker" in caplog.text


def test_worker_stderr_capture_survives_an_ungraceful_server_kill(tmp_path):
    """The validator is SIGKILLed in production (OOM). A per-worker stderr
    file that only gets cleaned up on the graceful path would accumulate a
    whole pool's worth of files on every such kill."""
    import glob

    before = set(glob.glob("/tmp/grader-worker-*"))
    script = (
        "import time\n"
        "from reliquary.environment.grader.server import GraderServer\n"
        "s = GraderServer(socket_path=%r, pool_size=3,\n"
        "                 worker_argv=[%r, '-m', 'reliquary.environment.grader.worker'],\n"
        "                 eval_timeout_s=1.0, metrics_port=0, health_path=%r)\n"
        "s.start()\n"
        "print('up', flush=True)\n"
        "time.sleep(60)\n"
    ) % (str(tmp_path / "g.sock"), sys.executable, str(tmp_path / "health.json"))

    proc = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True,
    )
    try:
        assert proc.stdout.readline().strip() == "up"
        time.sleep(0.5)
    finally:
        proc.kill()
        proc.wait(timeout=10.0)

    assert set(glob.glob("/tmp/grader-worker-*")) == before
