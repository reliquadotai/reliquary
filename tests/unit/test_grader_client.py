"""Tests for the grader IPC client.

Uses a real Unix socket server (asyncio) in-test to verify the wire
protocol round-trips. No grader server, no sandbox needed.
"""

import asyncio
import json
import os
import socket
import tempfile
import threading
import pytest


@pytest.fixture
def fake_grader_socket():
    """Spin up a tiny Unix socket server that returns canned responses.

    Yields (socket_path, set_response_fn). The response function lets
    each test queue what the server will reply for the next request.
    """
    tmp = tempfile.mkdtemp()
    sock_path = os.path.join(tmp, "fake-grader.sock")

    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_sock.bind(sock_path)
    server_sock.listen(8)

    state = {"response": None, "received": []}

    def run_server():
        while True:
            try:
                conn, _ = server_sock.accept()
            except OSError:
                return
            with conn:
                data = b""
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                    if b"\n" in data:
                        break
                if data:
                    line = data.split(b"\n", 1)[0]
                    state["received"].append(json.loads(line))
                if state["response"] is not None:
                    conn.sendall(json.dumps(state["response"]).encode() + b"\n")

    t = threading.Thread(target=run_server, daemon=True)
    t.start()

    def set_response(resp):
        state["response"] = resp

    yield sock_path, set_response, state

    server_sock.close()
    try:
        os.unlink(sock_path)
    except OSError:
        pass


def test_evaluate_round_trip(fake_grader_socket):
    """Client serializes a request and parses the server's response."""
    from reliquary.environment.grader_client import GraderClient

    sock_path, set_response, state = fake_grader_socket
    set_response({"req_id": "ignored", "passed": 3, "total": 5, "status": "ok"})

    client = GraderClient(socket_path=sock_path)
    result = client.evaluate(code="def f(): return 1", tests=["assert f() == 1"], timeout_s=5.0)

    assert result == 3 / 5
    assert state["received"][0]["code"] == "def f(): return 1"
    assert state["received"][0]["tests"] == ["assert f() == 1"]
    assert state["received"][0]["timeout_s"] == 5.0


def test_evaluate_returns_zero_when_status_not_ok(fake_grader_socket):
    from reliquary.environment.grader_client import GraderClient

    sock_path, set_response, _ = fake_grader_socket
    set_response({"req_id": "ignored", "passed": 0, "total": 3, "status": "timeout"})

    client = GraderClient(socket_path=sock_path)
    assert client.evaluate("", [], 5.0) == 0.0


def test_evaluate_returns_zero_when_grader_unreachable(tmp_path):
    """Missing socket → retry once → return 0.0, never raise."""
    from reliquary.environment.grader_client import GraderClient

    nonexistent = str(tmp_path / "nope.sock")
    client = GraderClient(socket_path=nonexistent)
    result = client.evaluate("def f(): pass", ["assert True"], 5.0)
    assert result == 0.0


def test_evaluate_returns_zero_on_malformed_response(fake_grader_socket):
    from reliquary.environment.grader_client import GraderClient

    sock_path, set_response, _ = fake_grader_socket
    set_response({"garbage": "no required fields"})

    client = GraderClient(socket_path=sock_path)
    assert client.evaluate("x = 1", ["assert x == 1"], 5.0) == 0.0


def test_evaluate_handles_zero_total_safely(fake_grader_socket):
    """If the dataset somehow sent 0 tests, return 0.0 not ZeroDivisionError."""
    from reliquary.environment.grader_client import GraderClient

    sock_path, set_response, _ = fake_grader_socket
    set_response({"req_id": "ignored", "passed": 0, "total": 0, "status": "ok"})

    client = GraderClient(socket_path=sock_path)
    assert client.evaluate("x = 1", [], 5.0) == 0.0
