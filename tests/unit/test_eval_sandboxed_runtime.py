"""LiveCodeBench's verifier run against the grading service, not Docker.

The fixture is the taskset's own verify.py (research-environments @ f2e0d709,
Apache-2.0), pinned by its sha256. Workers run locally here; production wraps
them in runsc.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import pickle
import sys
import tempfile
import time
import zlib
from pathlib import Path

import pytest

from reliquary.eval import sandboxed_runtime as sr

VERIFY = (Path(__file__).resolve().parents[1] / "fixtures" / "livecodebench" / "verify.py").read_bytes()
SUM = "a, b = map(int, input().split())\nprint(a + b)\n"
STDIN_CASES = [{"input": "1 2\n", "output": "3\n", "testtype": "stdin"},
               {"input": "5 7\n", "output": "12\n", "testtype": "stdin"}]


def test_the_fixture_is_the_pinned_verifier():
    assert hashlib.sha256(VERIFY).hexdigest() == sr.LIVECODEBENCH_VERIFY_SHA256


@pytest.fixture(scope="module")
def socket_path():
    from reliquary.environment.grader.server import GraderServer

    tmp = tempfile.TemporaryDirectory(prefix="g-", dir="/tmp")
    path = os.path.join(tmp.name, "g.sock")
    server = GraderServer(
        socket_path=path, pool_size=4,
        worker_argv=[sys.executable, "-m", "reliquary.environment.grader.worker"],
        metrics_port=0, health_path=os.path.join(tmp.name, "health.json"))
    server.start()
    deadline = time.time() + 5.0
    while not os.path.exists(path) and time.time() < deadline:
        time.sleep(0.05)
    yield path
    server.stop()
    tmp.cleanup()


def _verify(socket_path, code, public, private="[]", fn_name=None, timeout=2, client=None):
    from reliquary.environment.grader_client import GraderClient

    runtime = sr.GraderRuntime(frozenset({sr.LIVECODEBENCH_VERIFY_SHA256}),
                               client or (lambda: GraderClient(socket_path)))

    async def run():
        await runtime.write("/w/solution.py", code.encode())
        await runtime.write("/w/tests.json", json.dumps({
            "public_test_cases": json.dumps(public), "private_test_cases": private,
            "fn_name": fn_name}).encode())
        result = await runtime.run_uv_script(VERIFY, args=["/w/solution.py", "/w/tests.json", str(timeout)])
        return json.loads(result.stdout.strip().splitlines()[-1])

    return asyncio.run(run()), runtime


def test_a_correct_stdin_program_passes(socket_path):
    report, _ = _verify(socket_path, SUM, STDIN_CASES)
    assert report["passed"] is True and report["num_test_cases"] == 2


@pytest.mark.parametrize("code", [
    "print(0)\n",                                   # wrong answer
    "raise SystemExit(3)\n",                        # exit status
    "import time\ntime.sleep(10)\n",                # wall clock
    "import sys\nsys.stdout.buffer.write(b'\\xff')\n",  # undecodable stdout
])
def test_a_failing_program_fails(socket_path, code):
    report, _ = _verify(socket_path, code, STDIN_CASES, timeout=1)
    assert report["passed"] is False


def test_the_whole_standard_library_and_real_stdin_are_there(socket_path):
    # The verifier's interpreter, not the competitive-code guest's gate.
    code = "import os, sys\nprint(sum(map(int, os.read(0, 100).split())))\n"
    report, _ = _verify(socket_path, code, STDIN_CASES)
    assert report["passed"] is True


def test_a_functional_test_calls_the_solution(socket_path):
    code = "class Solution:\n    def add(self, a, b):\n        return a + b\n"
    cases = [{"input": "1\n2", "output": "3", "testtype": "functional"}]
    private = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(
        [{"input": "[1]\n[2]", "output": "[1, 2]", "testtype": "functional"}])))).decode()
    report, _ = _verify(socket_path, code, cases, private=private, fn_name="add")
    assert report["passed"] is True and report["num_test_cases"] == 2


def test_hidden_tests_may_not_unpickle_a_global(socket_path):
    class Boom:
        def __reduce__(self):
            return (os.getcwd, ())

    private = base64.b64encode(zlib.compress(pickle.dumps(Boom()))).decode()
    with pytest.raises(pickle.UnpicklingError):
        _verify(socket_path, SUM, STDIN_CASES, private=private)


def test_a_test_input_is_read_as_a_literal_only(socket_path):
    cases = [{"input": "__import__('os').getcwd()", "output": "1", "testtype": "functional"}]
    with pytest.raises(ValueError, match="not a literal"):
        _verify(socket_path, "def f(x):\n    return 1\n", cases, fn_name="f")


def test_an_outage_raises_instead_of_scoring_zero():
    from reliquary.environment.grader_client import GraderInfrastructureError

    class Down:
        def run_stdio(self, **_):
            raise GraderInfrastructureError("unreachable")

    with pytest.raises(GraderInfrastructureError):
        _verify(None, SUM, STDIN_CASES, client=Down)


def test_tests_after_a_failure_are_not_run():
    calls = []

    class Wrong:
        def run_stdio(self, **kwargs):
            calls.append(kwargs)
            return "ok", "0\n"

    cases = [{"input": f"{i} 0\n", "output": f"{i}\n", "testtype": "stdin"} for i in range(1, 40)]
    report, runtime = _verify(None, SUM, cases, client=Wrong)
    assert report["passed"] is False
    assert len(calls) < len(cases) and runtime.last_calls == len(calls)


def test_an_unpinned_verifier_is_refused():
    runtime = sr.GraderRuntime(frozenset({sr.LIVECODEBENCH_VERIFY_SHA256}))
    with pytest.raises(RuntimeError, match="not pinned"):
        asyncio.run(runtime.run_uv_script(VERIFY + b"\n", args=[]))


def test_the_guest_unframes_the_program_and_its_stdin():
    namespace: dict = {}
    exec(sr.guest_source(2), namespace)
    program = "import sys\nprint(sys.stdin.read().upper(), end='')\n"
    assert namespace["run"]("", sr.frame(program, "a\nb"), 1 << 20) == {"status": "ok", "stdout": "A\nB"}
    assert namespace["run"]("", sr.frame("print('x' * 100)", ""), 10)["status"] == "output_limit"


def test_only_livecodebench_is_adapted():
    class Other:
        pass

    assert sr.runtime_for(Other()) is None
    assert set(sr.ADAPTED) == {"livecodebench.taskset"}


def test_a_livecodebench_task_scores_through_the_grading_service(socket_path, monkeypatch):
    lcb = pytest.importorskip("livecodebench.taskset")
    from reliquary.eval import verifiers_source as vs

    monkeypatch.setenv(sr.SOCKET_ENV, socket_path)
    task = lcb.LiveCodeBenchTask(lcb.LiveCodeBenchData(
        idx=0, name="t", prompt="add two numbers", system_prompt=lcb.SYSTEM_PROMPT,
        public_test_cases=json.dumps(STDIN_CASES), private_test_cases="[]", fn_name=None,
        question_id="t", difficulty="easy", platform="atcoder"), lcb.LiveCodeBenchTaskConfig())
    frozen = vs.freeze(task)
    assert vs.needs_runtime(task)
    assert vs.score_trace(task, frozen, f"```python\n{SUM}```") == 1.0
    assert vs.score_trace(task, frozen, "```python\nprint(0)\n```") == 0.0
    assert vs.score_trace(task, frozen, "no code") == 0.0
