import json
import os
import subprocess
import sys

import pytest

from reliquary.validator.agentic_replay import TOOL_PROGRAM


def _run(tmp_path, tool, arguments, timeout=30):
    proc = subprocess.run([sys.executable, "-c", TOOL_PROGRAM],
                          input=json.dumps({"tool": tool, "arguments": arguments, "timeout": timeout}),
                          capture_output=True, text=True, cwd=tmp_path)
    return proc.stdout


def test_bash_returns_stdout_then_stderr(tmp_path):
    assert _run(tmp_path, "bash", json.dumps({"command": "echo out; echo err >&2"})) == "out\nerr\n"


def test_edit_replaces_exactly_once(tmp_path):
    (tmp_path / "f.py").write_text("a = 1\n")
    assert _run(tmp_path, "edit", json.dumps({"path": "f.py", "old_str": "1", "new_str": "2"})) == "Edited f.py"
    assert (tmp_path / "f.py").read_text() == "a = 2\n"


def test_edit_errors_match_verifiers(tmp_path):
    (tmp_path / "f.py").write_text("x x\n")
    assert _run(tmp_path, "edit", json.dumps({"path": "f.py", "old_str": "x", "new_str": "y"})) == \
        "error: old_str must appear exactly once in f.py (found 2)"


def test_invalid_json_arguments_reproduce_the_harness_message(tmp_path):
    out = _run(tmp_path, "bash", "{not json")
    assert out.startswith("error: invalid JSON in tool arguments (")


def test_non_object_arguments(tmp_path):
    assert _run(tmp_path, "bash", "[]") == \
        "error: tool arguments must be a JSON object, got list; resend as an object"


def test_unknown_tool(tmp_path):
    assert _run(tmp_path, "search", "{}") == "error: unknown tool 'search'"


def test_parity_with_pinned_verifiers(tmp_path):
    program = pytest.importorskip("verifiers.v1.harnesses.bash.program")
    (tmp_path / "g.py").write_text("k = 0\n")
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        want = program.run_edit("g.py", "0", "1")
    finally:
        os.chdir(cwd)
    (tmp_path / "g.py").write_text("k = 0\n")
    assert _run(tmp_path, "edit", json.dumps({"path": "g.py", "old_str": "0", "new_str": "1"})) == want


def test_bash_timeout_message_matches_the_pinned_harness(tmp_path):
    command = "sleep 5"
    out = _run(tmp_path, "bash", json.dumps({"command": command}), timeout=1)
    assert out == f"error: Command '['bash', '-c', '{command}']' timed out after 1 seconds"


def test_integral_timeout_is_sent_as_int():
    from reliquary.validator.agentic_replay import _timeout_arg
    assert _timeout_arg(3600.0) == 3600 and isinstance(_timeout_arg(3600.0), int)
    assert _timeout_arg(2.5) == 2.5


def test_the_program_reads_a_request_file_and_deletes_it(tmp_path):
    request = tmp_path / "req.json"
    request.write_text(json.dumps({"tool": "bash", "arguments": json.dumps({"command": "echo hi"}),
                                   "timeout": 30}))
    proc = subprocess.run([sys.executable, "-c", TOOL_PROGRAM, str(request)],
                          capture_output=True, text=True, cwd=tmp_path)
    assert proc.stdout == "hi\n"
    assert not request.exists()


# --- replay hardening, on a fake box (the real one needs Docker) -------------

import asyncio  # noqa: E402
import types  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

from reliquary.corpus.replay_compare import Action  # noqa: E402
from reliquary.validator import agentic_replay  # noqa: E402


class _FakeBox:
    def __init__(self, python="/opt/miniconda3/bin/python3", delay=0.0):
        self.python, self.delay = python, delay
        self.writes, self.runs, self.events = [], [], []

    async def prepare_setup(self):
        self.events.append("prepare_setup")

    async def prepare_execution(self, routes):
        self.events.append(("prepare_execution", routes))

    async def prepare_uv_script(self, script, env=None, *, activate=True):
        self.events.append(("uv_script", script, env))
        return ["/root/.cache/uv/python", "/tmp/vf-scripts/x.py"]

    async def write(self, path, data):
        self.writes.append(path)

    async def run(self, argv, env):
        self.runs.append(list(argv))
        if argv[:2] == ["sh", "-c"]:
            return types.SimpleNamespace(stdout=self.python + "\n", exit_code=0)
        await asyncio.sleep(self.delay)
        return types.SimpleNamespace(stdout=f"obs{len(self.runs)}", exit_code=0)


def _fake_verifiers(monkeypatch, box, updates=None, update_code=0):
    @asynccontextmanager
    async def provision_runtime(config, env, name=None):
        box.config, box.name = config, name
        yield box

    async def docker(*args):
        if updates is not None:
            updates.append((list(args), len(box.runs)))
        return update_code, "" if update_code == 0 else "no such container"

    monkeypatch.setattr(agentic_replay, "_docker", docker)

    v1 = types.ModuleType("verifiers.v1")
    v1.DockerConfig = lambda **kw: kw
    v1.Trace = lambda **kw: types.SimpleNamespace(info={"patch": "the diff"})
    v1.AgentInfo = v1.AgentConfig = v1.TraceTask = lambda **kw: kw
    runtimes = types.ModuleType("verifiers.v1.runtimes")
    runtimes.provision_runtime = provision_runtime
    monkeypatch.setitem(sys.modules, "verifiers", types.ModuleType("verifiers"))
    monkeypatch.setitem(sys.modules, "verifiers.v1", v1)
    monkeypatch.setitem(sys.modules, "verifiers.v1.runtimes", runtimes)
    bash = types.ModuleType("verifiers.v1.harnesses.bash.harness")
    bash.PROGRAM_SOURCE = "THE BASH HARNESS PROGRAM"
    for name in ("verifiers.v1.harnesses", "verifiers.v1.harnesses.bash"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "verifiers.v1.harnesses.bash.harness", bash)


def _task():
    async def noop(*_):
        pass
    return types.SimpleNamespace(
        data=types.SimpleNamespace(image="img", workdir="/testbed", network_allow=[]),
        key="k", hash="h", runtime_env=lambda: {}, setup=noop, finalize=noop)


def test_replay_resolves_python_once_then_passes_the_program_by_argv(monkeypatch):
    box = _FakeBox()
    _fake_verifiers(monkeypatch, box)
    actions = [Action("bash", '{"command": "ls"}', ""), Action("bash", '{"command": "pwd"}', "")]
    observations, diff = asyncio.run(agentic_replay.replay_swe(_task(), actions))
    assert diff == "the diff" and observations == ["obs2", "obs3"]
    assert box.runs[0][:2] == ["sh", "-c"]                     # resolved before any action
    assert sum(r[:2] == ["sh", "-c"] for r in box.runs) == 1     # and only once
    action_runs = box.runs[1:]
    assert len(action_runs) == 2
    for run, written in zip(action_runs, box.writes):
        assert run[0] == "/opt/miniconda3/bin/python3"           # absolute, never via PATH
        assert run[1:4] == ["-I", "-c", agentic_replay.TOOL_PROGRAM]  # isolated, program by argv
        assert run[4] == written                                 # its own fresh request file
    assert len(set(box.writes)) == 2
    assert not any(w.endswith(".py") for w in box.writes)       # no program file in the box


def test_replay_prepares_the_bash_harness_like_the_miner_before_the_network_cut(monkeypatch):
    # The miner's box runs verifiers' bash harness setup (pip install --user uv,
    # uv sync of its program) after task setup and before the cut: it leaves
    # /root/.local (on sys.path), /root/.cache/pip and uv in `pip list`, which
    # honest observations show. The replay box must carry the same footprint.
    box = _FakeBox()

    async def setup(trace, runtime):
        box.events.append("task_setup")
    task = _task()
    task.setup = setup
    _fake_verifiers(monkeypatch, box)
    asyncio.run(agentic_replay.replay_swe(task, [Action("bash", "{}", "")]))
    assert box.events == ["prepare_setup", "task_setup",
                          ("uv_script", "THE BASH HARNESS PROGRAM", {}),
                          ("prepare_execution", [])]


def test_replay_refuses_a_box_without_an_absolute_python(monkeypatch):
    _fake_verifiers(monkeypatch, _FakeBox(python=""))
    with pytest.raises(RuntimeError, match="no python"):
        asyncio.run(agentic_replay.replay_swe(_task(), [Action("bash", "{}", "")]))


def test_replay_past_its_deadline_raises_replay_timeout(monkeypatch):
    _fake_verifiers(monkeypatch, _FakeBox(delay=5.0))
    with pytest.raises(agentic_replay.ReplayTimeout, match="0 of 1 actions"):
        asyncio.run(agentic_replay.replay_swe(_task(), [Action("bash", "{}", "")], episode_deadline=0.2))


def test_the_box_is_bounded_before_any_action_runs(monkeypatch):
    box, updates = _FakeBox(), []
    _fake_verifiers(monkeypatch, box, updates)
    limits = agentic_replay.BoxLimits(cpu=1.5, memory_gb=2.0, pids=300)
    asyncio.run(agentic_replay.replay_swe(_task(), [Action("bash", "{}", "")], limits=limits))
    assert box.config["cpu"] == 1.5 and box.config["memory"] == 2.0
    assert box.name.startswith(agentic_replay.BOX_NAME_PREFIX)
    assert updates == [(["update", "--pids-limit", "300", "--memory", str(2 * 2 ** 30),
                         "--memory-swap", str(2 * 2 ** 30), box.name], 0)]   # before any run


def test_a_box_that_cannot_be_bounded_is_refused(monkeypatch):
    box = _FakeBox()
    _fake_verifiers(monkeypatch, box, update_code=1)
    with pytest.raises(RuntimeError, match="could not limit"):
        asyncio.run(agentic_replay.replay_swe(_task(), [Action("bash", "{}", "")]))
    assert box.runs == []


def test_default_episode_deadline_is_an_hour():
    import inspect
    sig = inspect.signature(agentic_replay.replay_swe)
    assert sig.parameters["episode_deadline"].default == 3600.0


def _shadowing_json(tmp_path):
    (tmp_path / "json.py").write_text("raise SystemExit('shadowed json imported')\n")
    request = tmp_path / "req.json"
    request.write_text('{"tool": "bash", "arguments": "{\\"command\\": \\"echo hi\\"}", "timeout": 30}')
    return request


def test_a_planted_module_in_the_working_directory_shadows_plain_python(tmp_path):
    # The attack the -I flag closes: without it, cwd comes first on sys.path.
    request = _shadowing_json(tmp_path)
    proc = subprocess.run([sys.executable, "-c", TOOL_PROGRAM, str(request)],
                          capture_output=True, text=True, cwd=tmp_path)
    assert "shadowed json imported" in proc.stderr


def test_isolated_mode_ignores_a_planted_module(tmp_path):
    request = _shadowing_json(tmp_path)
    proc = subprocess.run([sys.executable, "-I", "-c", TOOL_PROGRAM, str(request)],
                          capture_output=True, text=True, cwd=tmp_path)
    assert proc.stdout == "hi\n"
