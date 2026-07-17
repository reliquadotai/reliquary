"""Tests for the grader IPC client."""

import json
import os
import socket
import tempfile
import threading

import pytest


@pytest.fixture
def fake_grader_socket():
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
                    state["received"].append(json.loads(data.split(b"\n", 1)[0]))
                if state["response"] is not None:
                    conn.sendall(json.dumps(state["response"]).encode() + b"\n")

    threading.Thread(target=run_server, daemon=True).start()

    def set_response(resp):
        state["response"] = resp

    yield sock_path, set_response, state

    server_sock.close()
    try:
        os.unlink(sock_path)
    except OSError:
        pass


def _case():
    return {
        "entry": {"kind": "function", "name": "f"},
        "args": [1],
        "kwargs": {},
        "expected": 2,
        "compare": "exact",
    }


def test_evaluate_cases_round_trip(fake_grader_socket):
    from reliquary.environment.grader_client import GraderClient

    sock_path, set_response, state = fake_grader_socket
    set_response({"req_id": "ignored", "passed": 3, "total": 5, "status": "ok"})

    client = GraderClient(socket_path=sock_path)
    result = client.evaluate_cases("def f(x): return x+1", [_case()], timeout_s=5.0)

    assert result == 3 / 5
    assert state["received"][0]["code"] == "def f(x): return x+1"
    assert state["received"][0]["cases"] == [_case()]
    assert "tests" not in state["received"][0]
    assert state["received"][0]["timeout_s"] == 5.0


def test_evaluate_cases_returns_zero_when_status_not_ok(fake_grader_socket):
    from reliquary.environment.grader_client import GraderClient

    sock_path, set_response, _ = fake_grader_socket
    set_response({"req_id": "ignored", "passed": 0, "total": 3, "status": "timeout"})

    client = GraderClient(socket_path=sock_path)
    assert client.evaluate_cases("", [_case()], 5.0) == 0.0


def test_evaluate_cases_raises_when_grader_unreachable(tmp_path):
    from reliquary.environment.grader_client import (
        GraderClient,
        GraderInfrastructureError,
    )

    client = GraderClient(socket_path=str(tmp_path / "nope.sock"))
    with pytest.raises(GraderInfrastructureError, match="unreachable"):
        client.evaluate_cases("def f(): pass", [_case()], 5.0)


def test_evaluate_cases_raises_on_malformed_response(fake_grader_socket):
    from reliquary.environment.grader_client import (
        GraderClient,
        GraderInfrastructureError,
    )

    sock_path, set_response, _ = fake_grader_socket
    set_response({"garbage": "no required fields"})

    client = GraderClient(socket_path=sock_path)
    with pytest.raises(GraderInfrastructureError, match="malformed_response"):
        client.evaluate_cases("x = 1", [_case()], 5.0)


def test_evaluate_cases_raises_on_zero_total(fake_grader_socket):
    from reliquary.environment.grader_client import (
        GraderClient,
        GraderInfrastructureError,
    )

    sock_path, set_response, _ = fake_grader_socket
    set_response({"req_id": "ignored", "passed": 0, "total": 0, "status": "ok"})

    client = GraderClient(socket_path=sock_path)
    with pytest.raises(GraderInfrastructureError, match="invalid_score"):
        client.evaluate_cases("x = 1", [_case()], 5.0)


@pytest.mark.parametrize("status", ["crash", "grader_error", "unknown"])
def test_evaluate_cases_raises_on_infrastructure_status(
    fake_grader_socket, status,
):
    from reliquary.environment.grader_client import (
        GraderClient,
        GraderInfrastructureError,
    )

    sock_path, set_response, _ = fake_grader_socket
    set_response({
        "req_id": "ignored",
        "passed": 0,
        "total": 1,
        "status": status,
    })

    with pytest.raises(GraderInfrastructureError, match=status):
        GraderClient(socket_path=sock_path).evaluate_cases(
            "x = 1", [_case()], 5.0
        )


@pytest.mark.parametrize(
    "status",
    ["bad_output", "forbidden_import", "runtime_error", "tampered", "timeout"],
)
def test_evaluate_cases_scores_candidate_failures_zero(
    fake_grader_socket, status,
):
    from reliquary.environment.grader_client import GraderClient

    sock_path, set_response, _ = fake_grader_socket
    set_response({
        "req_id": "ignored",
        "passed": 0,
        "total": 1,
        "status": status,
    })

    assert GraderClient(socket_path=sock_path).evaluate_cases(
        "x = 1", [_case()], 5.0
    ) == 0.0


def test_evaluate_cases_returns_zero_for_empty_cases():
    from reliquary.environment.grader_client import GraderClient

    assert GraderClient(socket_path="/tmp/missing.sock").evaluate_cases("x=1", [], 5.0) == 0.0
