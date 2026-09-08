"""Tests for the CLI's grader auto-launch helpers."""

import os
import socket
import tempfile
import threading

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
        def poll(self):
            return 1

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
    main._grader_proc = None

    main._ensure_grader_running()

    assert len(calls) == 1
    cmd, kwargs = calls[0]
    assert ("--use-runsc" in cmd) is expects_runsc
    assert kwargs["env"]["RELIQUARY_GRADER_EXECUTOR_MODE"] == mode
    assert kwargs["env"]["RELIQUARY_GRADER_EXECUTOR_URL"].startswith("https://")
