"""The SciCode taskset (``tasksets/scicode_steps``): prompts and test files as
upstream builds them, the chain's bookkeeping, and its reward through the
grading service (a local worker; the real image's tests run on the eval host).
Skipped without `verifiers`."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

vf = pytest.importorskip("verifiers.v1")

from reliquary.eval import sandboxed_runtime as sr  # noqa: E402
from reliquary.eval import verifiers_source as vs  # noqa: E402

ROOT = Path(__file__).resolve().parents[2] / "tasksets" / "scicode_steps"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ts = importlib.import_module("scicode_steps.taskset")
chain = importlib.import_module("scicode_steps.chain")


def step(number, header, tests=("assert f(1) == target",), background="BG"):
    return {"step_number": number, "step_description_prompt": f"Describe {number}.",
            "step_background": f"{background} {number}", "function_header": header,
            "return_line": "    return x", "test_cases": list(tests)}


PROBLEMS = [
    {"problem_id": "1", "required_dependencies": "import numpy as np",
     "sub_steps": [step("1.1", "def f(a):\n    '''doc'''"), step("1.2", "def g(a):\n    '''doc'''"),
                   step("1.3", "def h(a):\n    '''doc'''")]},
    {"problem_id": "62", "required_dependencies": "import numpy as np\nfrom scipy.sparse import kron",
     "sub_steps": [step("62.1", "class EnlargedBlock:\n    def __init__(self, length):\n        '''x'''",
                        tests=()),
                   step("62.2", "def block_initial(model_d):\n    '''doc'''")]},
]


# Upstream's prompt assembly (eval/inspect_ai/scicode.py @ e3158ea0), for parity.
def upstream_prompt(problem, num_steps, previous_llm_code, with_background):
    output_lines, next_step, previous_code = [], [], []
    for i in range(num_steps - 1):
        output_lines.append(problem["sub_steps"][i]["step_description_prompt"] + '\n' +
                            problem["sub_steps"][i]["step_background"] if with_background
                            else problem["sub_steps"][i]["step_description_prompt"])
        output_lines.append(previous_llm_code[i])
        previous_code.append(previous_llm_code[i])
        output_lines.append("------")
    next_step.append(problem["sub_steps"][num_steps - 1]["step_description_prompt"] + '\n' +
                     problem["sub_steps"][num_steps - 1]["step_background"] if with_background
                     else problem["sub_steps"][num_steps - 1]["step_description_prompt"])
    s = problem["sub_steps"][num_steps - 1]
    next_step.append(f"{s['function_header']}\n\n{s['return_line']}")
    template = ts.TEMPLATES[with_background]
    return (template.format(problem_steps_str="\n\n".join(output_lines[:-1]),
                            next_step_str="\n\n".join(next_step),
                            dependencies=problem["required_dependencies"]),
            f'{problem["required_dependencies"]}\n{chr(10).join(previous_code)}\n')


def taskset(monkeypatch, **config):
    monkeypatch.setattr(ts, "read_problems", lambda split, revision=ts.DATASET_REVISION: PROBLEMS)
    return ts.SciCodeStepsTaskset(ts.SciCodeStepsConfig(**config))


def test_the_vendored_upstream_files_are_pinned():
    import hashlib

    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:12] for p in sorted(ts.DATA.iterdir())} == {
        "13.6.txt": "795a2b57c2d9", "62.1.txt": "bc9931d88a7d", "76.3.txt": "4758300d96ea",
        "background_comment_template.txt": "9b407bdd4493", "multistep_template.txt": "2c6946acf50f"}
    # No background is upstream's default: its "background comment" template.
    assert "starting with 'Background: '" in ts.TEMPLATES[False]
    assert hashlib.sha256(ts.VERIFY).hexdigest() == sr.SCICODE_VERIFY_SHA256


@pytest.mark.parametrize("with_background", [False, True])
def test_prompts_and_prefixes_match_upstream(monkeypatch, with_background):
    previous = {"1.1": "def f(a):\n    return a\n", "1.2": "\ndef g(a):\n    return f(a)\n"}
    tasks = list(taskset(monkeypatch, step=3, previous=previous, with_background=with_background))
    assert [t.data.step_number for t in tasks] == ["1.3"]
    prompt, prefix = upstream_prompt(PROBLEMS[0], 3, [previous["1.1"], previous["1.2"]], with_background)
    assert tasks[0].data.prompt == prompt and tasks[0].data.test_prefix == prefix
    assert ("BG 1.1" in prompt) is with_background
    frozen = vs.freeze(tasks[0])
    assert frozen.system is None and vs.refusal(tasks[0]) is None and vs.needs_runtime(tasks[0])


def test_each_position_asks_only_its_steps_and_needs_the_earlier_code(monkeypatch):
    assert [t.data.step_number for t in taskset(monkeypatch, step=1)] == ["1.1"]  # 62.1 is given
    with pytest.raises(ValueError, match="needs the code of 1.1"):
        list(taskset(monkeypatch, step=2))
    tasks = list(taskset(monkeypatch, step=2, previous={"1.1": "def f(a): return a"}))
    assert [t.data.step_number for t in tasks] == ["1.2", "62.2"]
    # The given step's code, as upstream takes it: the header's first def only.
    given = tasks[1].data.test_prefix
    assert "def __init__(self, length, basis_size, operator_dict)" in given
    assert "class EnlargedBlock" not in given and "class Block" not in given
    assert list(taskset(monkeypatch, step=4)) == []


@pytest.mark.parametrize("reply,code", [
    ("text\n```python\nimport numpy as np\nfrom a import b\ndef f(): pass\n```\n```python\nx=1\n```",
     "\n\ndef f(): pass\n"),
    ("```\nimport os\ny = 2\n```", "\ny = 2\n"),
    ("def f():\n    import os\n    return 1", "def f():\n\n    return 1"),
    ("", ""),
])
def test_code_is_extracted_as_upstream(reply, code):
    assert ts.extract_python_script(reply) == code


def test_the_test_file_is_upstreams():
    script = ts.test_script("DEPS\nPREV\n", "CODE", "1.2", ["assert a\nassert b", "assert c"], h5_path="/d.h5")
    assert script == ("DEPS\nPREV\n\nCODE\n\nfrom scicode.parse.parse import process_hdf5_to_tuple\n\n"
                      "targets = process_hdf5_to_tuple('1.2', 2, '/d.h5')\n"
                      "target = targets[0]\n\nassert a\nassert b\ntarget = targets[1]\n\nassert c\n")


class Recording:
    def __init__(self, passed=True):
        self.files, self.passed = {}, passed

    async def write(self, path, data):
        self.files[path] = data

    async def prepare_execution(self, routes):
        assert routes == []

    async def run_uv_script(self, script, args=None, env=None):
        assert script == ts.VERIFY
        self.spec = json.loads(self.files[args[0]])
        return type("R", (), {"exit_code": 0, "stdout": json.dumps({"passed": self.passed}), "stderr": ""})()


def score(task, reply, runtime):
    trace = vs._trace_for(task, vs.freeze(task), reply)

    async def run():
        await task.score(trace, runtime)
        return float(trace.reward)

    return asyncio.run(run())


def test_the_reward_runs_the_steps_test_file(monkeypatch):
    task = list(taskset(monkeypatch, step=2, previous={"1.1": "def f(a): return a"}))[0]
    runtime = Recording()
    assert score(task, "```python\nimport numpy as np\ndef g(a):\n    return f(a)\n```", runtime) == 1.0
    script = runtime.spec["script"]
    assert runtime.spec["timeout"] == 300
    assert script.startswith("import numpy as np\ndef f(a): return a\n\n\ndef g(a):")
    assert f"process_hdf5_to_tuple('1.2', 1, '{ts.H5_PATH}')" in script
    assert score(task, "", Recording(passed=False)) == 0.0


def test_the_chain_records_code_and_reports_both_rates():
    previous = chain.record({}, {"1.1": "```python\nimport os\ndef f(a): return a\n```", "62.2": ""})
    assert previous == {"1.1": "\ndef f(a): return a\n", "62.2": ""}
    with pytest.raises(ValueError, match="given as code"):
        chain.record({}, {"62.1": "x"})
    report = chain.summary("test", {"1.1": True, "1.2": True, "1.3": True, "62.2": False}, problems=PROBLEMS)
    assert report["problems_solved"] == 1 and report["steps_passed"] == 3
    assert report["main_problem_resolve_rate"] == 0.5 and report["subproblem_pass_rate"] == 0.75
    with pytest.raises(ValueError, match="outside"):
        chain.summary("test", {"62.1": True}, problems=PROBLEMS)


# --- through the grading service (a plain worker: no gVisor, no numpy) -------

@pytest.fixture(scope="module")
def socket_path():
    from reliquary.environment.grader.server import GraderServer

    tmp = tempfile.TemporaryDirectory(prefix="g-", dir="/tmp")
    path = os.path.join(tmp.name, "g.sock")
    server = GraderServer(socket_path=path, pool_size=2,
                          worker_argv=[sys.executable, "-m", "reliquary.environment.grader.worker"],
                          metrics_port=0, health_path=os.path.join(tmp.name, "health.json"),
                          stdio_max_seconds=600)
    server.start()
    deadline = time.time() + 5.0
    while not os.path.exists(path) and time.time() < deadline:
        time.sleep(0.05)
    yield path
    server.stop()
    tmp.cleanup()


def verdict(socket_path, script, timeout=5):
    from reliquary.environment.grader_client import GraderClient

    adapter = sr.ADAPTED["scicode_steps.taskset"]
    runtime = sr.GraderRuntime(adapter.verifiers, lambda: GraderClient(socket_path), adapter.guest_template)

    async def run():
        await runtime.write("/w/spec.json", json.dumps({"script": script, "timeout": timeout}).encode())
        result = await runtime.run_uv_script(ts.VERIFY, args=["/w/spec.json"])
        return json.loads(result.stdout.strip().splitlines()[-1])["passed"]

    return asyncio.run(run())


def test_a_step_passes_only_when_its_test_file_exits_zero(socket_path):
    assert verdict(socket_path, "def f(a):\n    return a\nassert f(2) == 2\nprint('x' * 10**6)\n") is True
    assert verdict(socket_path, "assert 1 == 2\n") is False
    assert verdict(socket_path, "g(1)\n") is False                 # the step's function is missing
    assert verdict(socket_path, "import time\ntime.sleep(5)\n", timeout=1) is False
    assert verdict(socket_path, "import os\nassert os.environ['OPENBLAS_NUM_THREADS'] == '1'\n") is True


def test_scicode_grades_only_through_its_own_service(monkeypatch):
    task = list(taskset(monkeypatch, step=1))[0]
    monkeypatch.delenv(sr.SCICODE_SOCKET_ENV, raising=False)
    monkeypatch.setenv(sr.SOCKET_ENV, "/nonexistent/default.sock")
    # An outage raises (verifiers wraps it), never a zero.
    with pytest.raises(Exception, match=f"{sr.SCICODE_SOCKET_ENV} is not set"):
        vs.score_trace(task, vs.freeze(task), "```python\ndef f(a): return a\n```")


def test_a_scicode_task_scores_through_the_grading_service(socket_path, monkeypatch):
    task = list(taskset(monkeypatch, step=1))[0]
    monkeypatch.setenv(sr.SCICODE_SOCKET_ENV, socket_path)
    # No `scicode` package here: the test file fails at its import, as a wrong answer would.
    assert vs.score_trace(task, vs.freeze(task), "```python\ndef f(a): return a\n```") == 0.0


def test_the_chain_knows_each_positions_steps_in_set_order(monkeypatch):
    previous = {"1.1": "def f(a): return a", "1.2": "", "62.2": ""}
    for position in (1, 2, 3):
        loaded = [t.data.step_number for t in taskset(monkeypatch, step=position, previous=previous)]
        assert chain.asked_at("test", position, problems=PROBLEMS) == loaded
    args = {"split": "test", "step": 2, "previous": {"1.1": "c"}}
    following = chain.next_args(args, {"1.2": "```python\ndef g(a): pass\n```", "62.2": ""}, problems=PROBLEMS)
    assert following == {"split": "test", "step": 3, "previous": {"1.1": "c", "1.2": "\ndef g(a): pass\n", "62.2": ""}}
    with pytest.raises(ValueError, match="asks 2 steps"):
        chain.next_args(args, {"1.2": ""}, problems=PROBLEMS)
