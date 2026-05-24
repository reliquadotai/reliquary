"""End-to-end grader test with real runsc + real bundle.

Skipped if runsc isn't installed or the bundle directory is missing.
Run manually with:

    pytest tests/integration/test_grader_e2e.py -v -s

or via the dedicated CI job.
"""

import os
import shutil
import socket
import json
import time
import threading
import pytest


BUNDLE_PATH = os.environ.get(
    "GRADER_BUNDLE_PATH",
    "/opt/reliquary/reliquary/environment/grader/bundle",
)


def _runsc_available() -> bool:
    return shutil.which("runsc") is not None and os.path.exists(
        os.path.join(BUNDLE_PATH, "rootfs", "usr", "local", "bin", "python3")
    )


pytestmark = pytest.mark.skipif(
    not _runsc_available(),
    reason="runsc or grader bundle not available on this host",
)


@pytest.fixture
def grader_server(tmp_path):
    from reliquary.environment.grader.server import GraderServer

    sock_path = str(tmp_path / "grader.sock")
    worker_argv = [
        "runsc", "--network=none", "run",
        "--bundle", BUNDLE_PATH, "grader-worker",
    ]
    server = GraderServer(
        socket_path=sock_path, pool_size=2,
        worker_argv=worker_argv, eval_timeout_s=5.0,
    )
    server.start()
    deadline = time.time() + 10.0
    while not os.path.exists(sock_path) and time.time() < deadline:
        time.sleep(0.1)
    yield server
    server.stop()


def _request(sock_path, code, tests, timeout_s=5.0):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(15.0)
        s.connect(sock_path)
        req = {"req_id": "e2e", "code": code, "tests": tests, "timeout_s": timeout_s}
        s.sendall(json.dumps(req).encode() + b"\n")
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(4096)
            if not chunk: break
            buf += chunk
    return json.loads(buf.split(b"\n", 1)[0])


def test_e2e_grades_correct_code(grader_server):
    resp = _request(
        grader_server.socket_path,
        code="def add(a,b): return a+b",
        tests=["assert add(1,2) == 3"],
    )
    assert resp["status"] == "ok"
    assert resp["passed"] == 1
    assert resp["total"] == 1


def test_e2e_blocks_network_access(grader_server):
    """Even if code tries socket(), runsc --network=none blocks it."""
    code = (
        "import socket\n"
        "try:\n"
        "    s = socket.socket()\n"
        "    s.connect(('8.8.8.8', 53))\n"
        "    HACKED = True\n"
        "except Exception:\n"
        "    HACKED = False\n"
    )
    tests = ["assert HACKED is False"]
    resp = _request(grader_server.socket_path, code=code, tests=tests)
    assert resp["passed"] == 1, "network should be blocked"


def test_e2e_blocks_filesystem_writes(grader_server):
    code = (
        "try:\n"
        "    open('/etc/hostname', 'w').write('pwned')\n"
        "    WROTE = True\n"
        "except Exception:\n"
        "    WROTE = False\n"
    )
    tests = ["assert WROTE is False"]
    resp = _request(grader_server.socket_path, code=code, tests=tests)
    assert resp["passed"] == 1


def test_e2e_kills_infinite_loop(grader_server):
    resp = _request(
        grader_server.socket_path,
        code="while True: pass",
        tests=["assert True"],
        timeout_s=1.0,
    )
    assert resp["status"] == "timeout"


def test_e2e_pool_recovers_after_worker_crash(grader_server):
    """Send hostile request, then a normal one — second should still work."""
    _request(grader_server.socket_path, code="while True: pass",
             tests=["assert True"], timeout_s=1.0)
    # After respawn, normal request must succeed.
    resp = _request(grader_server.socket_path, code="x = 1", tests=["assert x == 1"])
    assert resp["status"] == "ok"
    assert resp["passed"] == 1
