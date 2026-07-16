"""Tests for the trusted grader server."""

import json
import os
import socket
import sys
import tempfile
import threading
import time

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
