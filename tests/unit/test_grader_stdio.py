"""The grading service's stdio mode: a whole program run on one stdin.

The worker forks once per request and runs the package's own `guest.run` in
the child (the copy in tests/fixtures is pinned to the artifact's bytes); the
server never sees an expected output, so it returns what the program printed
and the trusted caller compares. CPU time and exit status come from the kernel,
never from what the child writes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

GUEST = (
    Path(__file__).resolve().parents[1] / "fixtures" / "competitive_code" / "guest.py"
).read_text()
SUM = "a, b = map(int, input().split())\nprint(a + b)\n"


def _worker(env_overrides=None):
    env = {**os.environ, "PYTHONHASHSEED": "0", **(env_overrides or {})}
    return subprocess.Popen(
        [sys.executable, "-m", "reliquary.environment.grader.worker"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1, env=env,
    )


def _ask(proc, **request) -> dict:
    import base64

    payload = {"req_id": "r", "mode": "stdio", "guest": GUEST, "output_cap": 1 << 16,
               "time_limit_s": 1.0, **request}
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()
    reply = json.loads(proc.stdout.readline())
    data = b""
    for _ in range(reply.get("stdout_chunks", 0)):
        proc.stdin.write("next\n")
        proc.stdin.flush()
        data += base64.b64decode(proc.stdout.readline())
    return {**reply, "stdout": data.decode()}


@pytest.fixture
def worker():
    proc = _worker()
    yield proc
    proc.kill()
    proc.wait()


def test_a_program_reads_stdin_and_its_stdout_comes_back(worker) -> None:
    reply = _ask(worker, code=SUM, stdin="2 3\n")
    assert (reply["req_id"], reply["status"], reply["stdout"]) == ("r", "ok", "5\n")
    assert reply["cpu_seconds"] >= 0.0


@pytest.mark.parametrize("code, status", [
    ("import os\nprint(1)\n", "forbidden_import"),
    ("print(1 // 0)\n", "runtime_error"),
    ("raise SystemExit(3)\n", "runtime_error"),
    ("while True:\n    pass\n", "timeout"),
    ("print('x' * 10_000_000)\n", "output_limit"),
])
def test_failures_are_named_by_the_guest_or_the_kernel(worker, code, status) -> None:
    reply = _ask(worker, code=code, stdin="")
    assert reply["status"] == status
    assert reply["stdout"] == ""


def test_the_worker_survives_and_serves_the_next_request(worker) -> None:
    assert _ask(worker, code="while True:\n    pass\n", stdin="")["status"] == "timeout"
    assert _ask(worker, code=SUM, stdin="1 1\n")["stdout"] == "2\n"


def test_each_run_starts_from_a_fresh_fork(worker) -> None:
    """Module state a submission changes dies with its child."""
    tamper = "import math\nmath.pi = 3\nprint(math.pi)\n"
    assert _ask(worker, code=tamper, stdin="")["stdout"] == "3\n"
    assert _ask(worker, code="import math\nprint(math.pi)\n", stdin="")["stdout"].startswith("3.14")


def test_output_is_deterministic_across_workers() -> None:
    """Set order of strings and the seeded `random` are the same on every
    replay: PYTHONHASHSEED=0 is inherited by the fork, the guest seeds random."""
    code = (
        "import random\n"
        "print(sorted({'b', 'a'}) == ['a', 'b'], list({'x', 'yy', 'zzz', 'w'}))\n"
        "print(random.random())\n"
    )
    outputs = []
    for _ in range(2):
        proc = _worker()
        try:
            outputs.append(_ask(proc, code=code, stdin="")["stdout"])
        finally:
            proc.kill()
            proc.wait()
    assert outputs[0] == outputs[1] and outputs[0]


def test_a_worker_without_a_pinned_hash_seed_refuses_to_run() -> None:
    proc = _worker({"PYTHONHASHSEED": "random"})
    try:
        assert _ask(proc, code=SUM, stdin="1 2\n")["status"] == "grader_error"
    finally:
        proc.kill()
        proc.wait()


def test_a_submission_cannot_forge_the_workers_reply(worker) -> None:
    """Escaping the guest's import gate reaches `os.write`; fd 1 of the child
    is /dev/null, so a forged line never reaches the server."""
    forge = (
        "w = [c for c in ().__class__.__base__.__subclasses__()"
        " if c.__name__ == '_wrap_close'][0].__init__.__globals__['write']\n"
        "w(1, b'{\"req_id\": \"r\", \"status\": \"ok\", \"stdout\": \"forged\"}\\n')\n"
        "print('honest')\n"
    )
    reply = _ask(worker, code=forge, stdin="")
    assert reply["stdout"] != "forged"


def test_a_child_cannot_forge_its_status_past_the_kernel(worker) -> None:
    """A program that burns its CPU limit and then writes an 'ok' result line
    to the result pipe is still a timeout: CPU comes from wait4."""
    burn = (
        "import time\n"
        "end = time.process_time() + 1.5\n"
        "while time.process_time() < end:\n"
        "    pass\n"
        "print('done')\n"
    )
    assert _ask(worker, code=burn, stdin="")["status"] == "timeout"


def test_an_idle_program_is_a_timeout(worker) -> None:
    assert _ask(worker, code="import time\ntime.sleep(30)\n", stdin="",
                time_limit_s=0.5)["status"] == "timeout"


def test_without_a_run_queue_an_idle_program_is_still_a_timeout(monkeypatch) -> None:
    """gVisor exposes no schedstat: a program that used almost no CPU over the
    whole wall window idled; one that used CPU may have been starved."""
    from reliquary.environment.grader import worker

    monkeypatch.setattr(worker, "_run_queue_wait_s", lambda pid: None)
    monkeypatch.setattr(worker.sys, "flags", type("F", (), {"hash_randomization": 0})())
    idle = worker.run_stdio(GUEST, "import time\ntime.sleep(30)\n", "", 1024, 0.3)
    assert idle["status"] == "timeout"
    monkeypatch.setattr(worker, "STDIO_IDLE_CPU_S", -1.0)
    starved = worker.run_stdio(GUEST, "import time\ntime.sleep(30)\n", "", 1024, 0.3)
    assert starved["status"] == "harness_overload"


@pytest.fixture
def server():
    from reliquary.environment.grader.server import GraderServer

    tmp = tempfile.TemporaryDirectory(prefix="g-", dir="/tmp")
    sock = os.path.join(tmp.name, "g.sock")
    server = GraderServer(
        socket_path=sock, pool_size=2,
        worker_argv=[sys.executable, "-m", "reliquary.environment.grader.worker"],
        eval_timeout_s=5.0, metrics_port=0,
        health_path=os.path.join(tmp.name, "health.json"),
    )
    server.start()
    deadline = time.time() + 5.0
    while not os.path.exists(sock) and time.time() < deadline:
        time.sleep(0.05)
    yield server
    server.stop()
    tmp.cleanup()


def _client(server):
    from reliquary.environment.grader_client import GraderClient

    return GraderClient(socket_path=server.socket_path)


def test_the_server_relays_a_stdio_run_without_any_expected_output(server) -> None:
    status, stdout = _client(server).run_stdio(
        code=SUM, guest=GUEST, stdin="40 2\n", output_cap=1 << 16, time_limit_s=1.0)
    assert (status, stdout) == ("ok", "42\n")


def test_candidate_failures_come_back_as_statuses(server) -> None:
    client = _client(server)
    status, stdout = client.run_stdio(code="while True:\n    pass\n", guest=GUEST,
                                      stdin="", output_cap=1024, time_limit_s=0.5)
    assert (status, stdout) == ("timeout", "")


def test_a_large_stdin_and_stdout_cross_the_socket(server) -> None:
    numbers = " ".join(str(i) for i in range(200_000))
    code = "import sys\nprint(sum(map(int, sys.stdin.read().split())))\nprint('y' * 300000)\n"
    status, stdout = _client(server).run_stdio(
        code=code, guest=GUEST, stdin=numbers, output_cap=1 << 20, time_limit_s=2.0)
    assert status == "ok"
    first, second = stdout.split("\n")[:2]
    assert int(first) == sum(range(200_000)) and second == "y" * 300000


@pytest.mark.parametrize("bad", [
    {"time_limit_s": 0},
    {"time_limit_s": 1e9},
    {"output_cap": 0},
    {"guest": ""},
    {"stdin": 3},
])
def test_a_malformed_stdio_request_is_a_grader_error(server, bad) -> None:
    from reliquary.environment.grader_client import GraderInfrastructureError

    request = dict(code=SUM, guest=GUEST, stdin="1 2\n", output_cap=1024, time_limit_s=1.0)
    request.update(bad)
    with pytest.raises(GraderInfrastructureError):
        _client(server).run_stdio(**request)


def test_harness_overload_is_infrastructure_never_a_verdict(server, monkeypatch) -> None:
    from reliquary.environment.grader_client import GraderInfrastructureError

    monkeypatch.setattr(server, "_evaluate_on_worker", lambda worker, req: {
        "req_id": req["req_id"], "status": "harness_overload", "stdout_chunks": 0,
        "cpu_seconds": 0.1})
    with pytest.raises(GraderInfrastructureError, match="harness_overload"):
        _client(server).run_stdio(code=SUM, guest=GUEST, stdin="1 2\n",
                                  output_cap=1024, time_limit_s=1.0)


def test_a_worker_that_stops_answering_is_never_the_programs_timeout(server, monkeypatch) -> None:
    """The server's own deadline answers "timeout" for a silent worker; in
    stdio mode the worker times the program itself, so that is infrastructure."""
    from reliquary.environment.grader_client import GraderInfrastructureError

    monkeypatch.setattr(server, "_evaluate_on_worker", lambda worker, req: {
        "req_id": req["req_id"], "output": None, "status": "timeout"})
    with pytest.raises(GraderInfrastructureError, match="worker_timeout"):
        _client(server).run_stdio(code=SUM, guest=GUEST, stdin="1 2\n",
                                  output_cap=1024, time_limit_s=1.0)


def test_a_long_stdout_comes_back_in_lines_runsc_can_carry(worker) -> None:
    """Under runsc a reply line past 64 KiB stalls; the stdout is sent in
    base64 chunks, each only when asked for."""
    import base64

    from reliquary.environment.grader.worker import STDIO_CHUNK_BYTES

    payload = {"req_id": "r", "mode": "stdio", "guest": GUEST, "output_cap": 1 << 20,
               "time_limit_s": 1.0, "code": "print('z' * 100000)\n", "stdin": ""}
    worker.stdin.write(json.dumps(payload) + "\n")
    worker.stdin.flush()
    header = json.loads(worker.stdout.readline())
    assert header["status"] == "ok" and "stdout" not in header
    assert header["stdout_chunks"] == -(-100001 // STDIO_CHUNK_BYTES)
    data = b""
    for _ in range(header["stdout_chunks"]):
        worker.stdin.write("next\n")
        worker.stdin.flush()
        line = worker.stdout.readline()
        assert len(line) < 64 * 1024
        data += base64.b64decode(line)
    assert data.decode() == "z" * 100000 + "\n"
