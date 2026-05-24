"""Tests for the CLI's grader auto-launch helpers."""

import os
import socket
import tempfile
import threading


def test_grader_is_running_returns_false_for_missing_socket(tmp_path):
    from reliquary.cli.main import _grader_is_running
    assert _grader_is_running(str(tmp_path / "nope.sock")) is False


def test_grader_is_running_returns_true_when_listener_present(tmp_path):
    """Set up a real Unix socket listener — _grader_is_running should detect it."""
    from reliquary.cli.main import _grader_is_running
    sock_path = str(tmp_path / "fake-grader.sock")
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
