"""Tests for the CLI's grader auto-launch helpers."""

import os
import json
import socket
import tempfile
import threading
import time

import pytest


def test_grader_is_running_returns_false_for_missing_socket(tmp_path):
    from reliquary.cli.main import _grader_is_running
    assert _grader_is_running(str(tmp_path / "nope.sock")) is False


def test_grader_is_running_returns_true_when_listener_present(tmp_path):
    """Set up a real Unix socket listener — _grader_is_running should detect it."""
    from reliquary.cli.main import _grader_is_running
    tmp = tempfile.TemporaryDirectory(prefix="g-", dir="/tmp")
    sock_path = os.path.join(tmp.name, "g.sock")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_path)
    server.listen(1)

    def _accept_loop():
        try:
            conn, _ = server.accept()
            conn.close()
        except Exception:
            pass

    t = threading.Thread(target=_accept_loop, daemon=True)
    t.start()

    try:
        assert _grader_is_running(sock_path) is True
    finally:
        server.close()
        try:
            os.unlink(sock_path)
        except FileNotFoundError:
            pass
        tmp.cleanup()


def test_ensure_grader_refuses_unsandboxed_by_default(monkeypatch, tmp_path):
    from reliquary.cli import main

    monkeypatch.setattr(main, "_grader_is_running", lambda *a, **k: False)
    monkeypatch.setattr(main.shutil, "which", lambda name: None)
    monkeypatch.delenv("RELIQUARY_ALLOW_UNSANDBOXED_GRADER", raising=False)
    monkeypatch.setenv("GRADER_BUNDLE_PATH", str(tmp_path / "missing-bundle"))

    with pytest.raises(RuntimeError, match="requires the gVisor/runsc grader sandbox"):
        main._ensure_grader_running()


@pytest.mark.parametrize(
    ("mode", "expects_runsc"),
    [("shadow", True), ("remote", False)],
)
def test_ensure_grader_launches_safe_remote_rollout_mode(
    monkeypatch,
    tmp_path,
    mode,
    expects_runsc,
):
    from reliquary.cli import main

    calls = []
    reachability = iter([False, True])

    class _Process:
        pid = 42

        def poll(self):
            return None

    def _popen(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _Process()

    bundle_python = tmp_path / "rootfs" / "usr" / "local" / "bin" / "python3"
    bundle_python.parent.mkdir(parents=True)
    bundle_python.touch()
    monkeypatch.setattr(
        main,
        "_grader_is_running",
        lambda *args, **kwargs: next(reachability),
    )
    monkeypatch.setattr(main.shutil, "which", lambda name: "/usr/bin/runsc")
    monkeypatch.setattr(main.subprocess, "Popen", _popen)
    monkeypatch.setattr(main.atexit, "register", lambda callback: None)
    monkeypatch.setenv("GRADER_BUNDLE_PATH", os.fspath(tmp_path))
    monkeypatch.setenv(
        "RELIQUARY_GRADER_EXECUTOR_URL",
        "https://cpu-exec.internal:8443",
    )
    monkeypatch.setenv("RELIQUARY_GRADER_EXECUTOR_MODE", mode)
    monkeypatch.setattr(main, "_grader_proc", None)

    main._ensure_grader_running()

    assert len(calls) == 1
    cmd, kwargs = calls[0]
    assert ("--use-runsc" in cmd) is expects_runsc
    assert kwargs["env"]["RELIQUARY_GRADER_EXECUTOR_MODE"] == mode
    assert kwargs["env"]["RELIQUARY_GRADER_EXECUTOR_URL"].startswith("https://")


@pytest.mark.parametrize("exit_code", [0, 17, None])
def test_required_grader_startup_failure_raises_and_cleans_up(monkeypatch, exit_code):
    from reliquary.cli import main

    calls = []

    class Process:
        pid = 42
        code = exit_code

        def poll(self):
            return self.code

        def terminate(self):
            calls.append("terminate")
            self.code = 0

        def wait(self, timeout):
            calls.append("wait")

    clock = iter([0, 1, 16])
    monkeypatch.setattr(main._time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(main._time, "sleep", lambda _: None)
    monkeypatch.setattr(main, "_grader_is_running", lambda *a, **k: False)
    monkeypatch.setattr(main.subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(main, "_grader_proc", None)
    monkeypatch.setenv("RELIQUARY_GRADER_EXECUTOR_URL", "https://executor.example")
    monkeypatch.setenv("RELIQUARY_GRADER_EXECUTOR_MODE", "remote")

    with pytest.raises(RuntimeError, match="exited during startup|not ready within"):
        main._ensure_grader_running()
    assert main._grader_proc is None
    assert calls == (["terminate", "wait"] if exit_code is None else [])


def test_existing_incompatible_grader_is_refused_without_launch(monkeypatch):
    from reliquary.cli import main

    monkeypatch.setattr(main, "_grader_is_running", lambda *a, **k: not k)
    monkeypatch.setattr(main.subprocess, "Popen", lambda *a, **k: pytest.fail("unexpected launch"))
    monkeypatch.delenv("RELIQUARY_GRADER_EXECUTOR_URL", raising=False)
    with pytest.raises(RuntimeError, match="incompatible readiness"):
        main._ensure_grader_running(use_runsc=True)


@pytest.mark.parametrize("change,expected", [
    ({}, True), ({"updated_at": 0}, False), ({"workers_alive": 0}, False),
    ({"sandbox_backend": "python"}, False), ({"execution_backend": "remote"}, False),
    ({"shutdown_complete": True}, False), ({"server_pid": 9}, False),
])
def test_grader_readiness_checks_published_backend_and_workers(monkeypatch, tmp_path, change, expected):
    from reliquary.cli import main

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def settimeout(self, timeout):
            pass

        def connect(self, path):
            pass

    health = {"updated_at": time.time(), "execution_backend": "local",
              "sandbox_backend": "runsc", "workers_alive": 1,
              "server_pid": 42, "shutdown_complete": False, **change}
    path = tmp_path / "health.json"
    path.write_text(json.dumps(health))
    monkeypatch.setenv("GRADER_HEALTH_PATH", str(path))
    monkeypatch.delenv("RELIQUARY_GRADER_RUNTIME_ID", raising=False)
    monkeypatch.setattr(main._socket, "socket", lambda *a: Socket())
    assert main._grader_is_running("unused", expected_backend="local",
                                   expected_sandbox="runsc", expected_pid=42) is expected


def test_remote_grader_readiness_does_not_require_local_workers(monkeypatch, tmp_path):
    from reliquary.cli import main

    health = {"updated_at": time.time(), "execution_backend": "remote",
              "sandbox_backend": "remote", "workers_alive": 0,
              "shutdown_complete": False}
    path = tmp_path / "health.json"
    path.write_text(json.dumps(health))
    monkeypatch.setenv("GRADER_HEALTH_PATH", str(path))
    monkeypatch.delenv("RELIQUARY_GRADER_RUNTIME_ID", raising=False)
    from unittest.mock import MagicMock
    monkeypatch.setattr(main._socket, "socket", lambda *a: MagicMock())
    assert main._grader_is_running("unused", expected_backend="remote", expected_sandbox="remote")
