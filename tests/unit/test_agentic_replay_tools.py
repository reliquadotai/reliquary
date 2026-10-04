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


def test_replay_past_its_trajectory_budget_raises_replay_timeout(monkeypatch):
    _fake_verifiers(monkeypatch, _FakeBox(delay=5.0))
    with pytest.raises(agentic_replay.ReplayTimeout, match="0 of 1 actions"):
        asyncio.run(agentic_replay.replay_swe(_task(), [Action("bash", "{}", "")],
                                              trajectory_budget=0.2))


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


def _timed_task(setup=900.0, agent=3600.0, finalize=900.0):
    task = _task()
    task.data.timeout = types.SimpleNamespace(setup=setup, agent=agent, finalize=finalize,
                                              scoring=1800.0)
    return task


def test_replay_deadlines_follow_the_tasks_own_timeouts():
    """Ruling P25: setup gets the task's setup timeout plus a margin (the
    executor's), the trajectory twice the miner's agent + finalize budget."""
    setup, budget = agentic_replay.replay_deadlines(_timed_task())
    assert setup == 900.0 + agentic_replay.SETUP_MARGIN_SECONDS
    assert budget == 9000.0                                      # SWE-smith: 2 x (3600 + 900)
    assert agentic_replay.replay_deadlines(_task()) == (           # unset phases: an hour each
        3600.0 + agentic_replay.SETUP_MARGIN_SECONDS, 2 * (3600.0 + 3600.0))
    # An honest miner's 3000 s of actions sits far inside the budget.
    assert budget > 3000.0 * 2


def test_actions_outlasting_the_setup_deadline_are_never_a_timeout(monkeypatch):
    # The two clocks are separate: 0.6 s of actions past a 0.2 s setup deadline
    # (the scaled image of an honest 3000 s episode) is within its own budget.
    _fake_verifiers(monkeypatch, _FakeBox(delay=0.3))
    observations, _ = asyncio.run(agentic_replay.replay_swe(
        _task(), [Action("bash", "{}", ""), Action("bash", "{}", "")],
        setup_deadline=0.2, trajectory_budget=2.0))
    assert len(observations) == 2


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


# F2 (ruling P23): what fails after the first recorded action started is the
# trajectory's doing, not the executor's; what fails before it is the executor's.

class _DyingBox(_FakeBox):
    def __init__(self, die_at_run, **kw):
        super().__init__(**kw)
        self._die_at = die_at_run

    async def run(self, argv, env):
        if len(self.runs) + 1 == self._die_at:
            self.runs.append(list(argv))
            raise RuntimeError("docker exec: container is not running")
        return await super().run(argv, env)


def test_a_box_that_dies_while_running_recorded_actions_is_box_lost(monkeypatch):
    _fake_verifiers(monkeypatch, _DyingBox(die_at_run=3))      # resolve, action 1, then dies
    actions = [Action("bash", '{"command": "kill -9 1"}', ""), Action("bash", "{}", "")]
    with pytest.raises(agentic_replay.BoxLost, match="after 1 of 2 actions"):
        asyncio.run(agentic_replay.replay_swe(_task(), actions))


def test_a_finalize_that_fails_after_the_actions_is_box_lost(monkeypatch):
    _fake_verifiers(monkeypatch, _FakeBox())
    task = _task()

    async def finalize(trace, box):
        raise RuntimeError("git diff: not a git repository")
    task.finalize = finalize
    with pytest.raises(agentic_replay.BoxLost, match="finalize"):
        asyncio.run(agentic_replay.replay_swe(task, [Action("bash", '{"command": "rm -rf .git"}', "")]))


def test_failures_before_any_recorded_action_stay_the_executors(monkeypatch):
    _fake_verifiers(monkeypatch, _FakeBox())
    task = _task()

    async def finalize(trace, box):
        raise RuntimeError("finalize broke on an untouched box")
    task.finalize = finalize
    with pytest.raises(RuntimeError) as caught:                 # no action ran: infrastructure
        asyncio.run(agentic_replay.replay_swe(task, []))
    assert not isinstance(caught.value, agentic_replay.BoxLost)

    _fake_verifiers(monkeypatch, _DyingBox(die_at_run=1))       # the python lookup itself
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(agentic_replay.replay_swe(_task(), [Action("bash", "{}", "")]))
    assert not isinstance(caught.value, agentic_replay.BoxLost)


def test_a_deadline_hit_by_the_actions_is_the_trajectorys_and_by_setup_the_executors(monkeypatch):
    _fake_verifiers(monkeypatch, _FakeBox(delay=5.0))
    with pytest.raises(agentic_replay.ReplayTimeout) as caught:
        asyncio.run(agentic_replay.replay_swe(_task(), [Action("bash", '{"command": "sleep 99999"}', "")],
                                              setup_deadline=10.0, trajectory_budget=0.2))
    assert caught.value.trajectory_caused is True

    _fake_verifiers(monkeypatch, _FakeBox())
    task = _task()

    async def slow_setup(trace, box):
        await asyncio.sleep(5.0)
    task.setup = slow_setup
    with pytest.raises(agentic_replay.ReplayTimeout) as caught:
        asyncio.run(agentic_replay.replay_swe(task, [Action("bash", "{}", "")],
                                              setup_deadline=0.2, trajectory_budget=60.0))
    assert caught.value.trajectory_caused is False             # slow setup: the executor's


# --- F4 (ruling P24): what a box hands back is bounded, its disk too ---------

def test_the_tool_program_truncates_its_output_to_the_requested_chars(tmp_path):
    proc = subprocess.run([sys.executable, "-c", TOOL_PROGRAM], cwd=tmp_path, capture_output=True,
                          text=True, input=json.dumps({
                              "tool": "bash", "arguments": json.dumps({"command": "yes | head -c 100000"}),
                              "timeout": 30, "max_chars": 1001}))
    assert len(proc.stdout) == 1001


def test_capped_exec_stops_reading_past_its_cap_and_ends_the_process():
    import time

    started = time.monotonic()
    code, out, err, truncated = asyncio.run(agentic_replay.capped_exec(
        [sys.executable, "-c", "import sys\nwhile True: sys.stdout.write('x' * 65536)"], max_bytes=100_000))
    assert truncated and len(out) == 100_000 and time.monotonic() - started < 30
    code, out, err, truncated = asyncio.run(agentic_replay.capped_exec(
        [sys.executable, "-c", "import sys; print('hi'); print('e', file=sys.stderr); sys.exit(3)"],
        max_bytes=100))
    assert (code, out, err, truncated) == (3, b"hi\n", b"e\n", False)


def test_an_observation_is_never_longer_than_a_lease_can_compare(monkeypatch):
    class Loud(_FakeBox):
        async def run(self, argv, env):
            self.runs.append(list(argv))
            return types.SimpleNamespace(stdout="y" * 5000, exit_code=0)

    monkeypatch.setattr(agentic_replay, "MAX_OBSERVATION_CHARS", 100)
    observation = asyncio.run(agentic_replay.run_action(Loud(), "/usr/bin/python3",
                                                       Action("bash", "{}", ""), 30))
    assert observation == "y" * 101                     # one past the bound: never equal to one within it


def test_the_bounded_exec_argv_is_verifiers_own(monkeypatch):
    """The bounded read runs exactly the `docker exec` verifiers' DockerRuntime.run runs
    (same env, workdir, container), so observations do not change."""
    docker_mod = pytest.importorskip("verifiers.v1.runtimes.docker")
    import verifiers.v1 as vf

    seen = []

    async def fake_docker(*args):
        seen.append(list(args))
        return types.SimpleNamespace(exit_code=0, stdout="", stderr="")

    monkeypatch.setattr(docker_mod, "docker", fake_docker)
    box = docker_mod.DockerRuntime(vf.DockerConfig(image="img", workdir="/testbed"), name="reliquary-gradebox-x")
    box._container = "reliquary-gradebox-x"
    box.env = {"A": "1"}
    for cut in (False, True):
        box._cut = cut
        if cut:
            box._proxy_env = lambda: {"HTTP_PROXY": "http://p"}
        seen.clear()
        asyncio.run(box.run(["python3", "-I", "-c", "x"], {"B": "2"}))
        assert agentic_replay.docker_exec_argv(box, ["python3", "-I", "-c", "x"], {"B": "2"}) == \
            ["docker", *seen[0]]


def test_the_box_disk_is_checked_before_any_action(monkeypatch):
    class Disk(_FakeBox):
        def __init__(self, kib):
            super().__init__()
            self.kib = kib

        async def run(self, argv, env):
            if argv[:2] == ["df", "-Pk"]:
                self.runs.append(list(argv))
                return types.SimpleNamespace(
                    stdout=f"Filesystem 1024-blocks Used Available Capacity Mounted on\n"
                           f"overlay {self.kib} 8 {self.kib - 8} 1% /\n", exit_code=0)
            return await super().run(argv, env)

    limits = agentic_replay.BoxLimits(disk_gb=2.0)
    bounded = Disk(2 * 2 ** 20)
    _fake_verifiers(monkeypatch, bounded)
    asyncio.run(agentic_replay.replay_swe(_task(), [Action("bash", "{}", "")], limits=limits))
    assert bounded.runs[0][:2] == ["df", "-Pk"]          # before the python lookup and any action
    unbounded = Disk(300 * 2 ** 20)                      # the host's 300 GB: no quota
    _fake_verifiers(monkeypatch, unbounded)
    with pytest.raises(RuntimeError, match="disk"):
        asyncio.run(agentic_replay.replay_swe(_task(), [Action("bash", "{}", "")], limits=limits))
    assert len(unbounded.runs) == 1                      # nothing ran after the check
    with pytest.raises(ValueError):
        agentic_replay.BoxLimits(disk_gb=0)


def test_finalize_gets_the_very_box_setup_got(monkeypatch):
    # reliquary-swe's finalize finds setup's base commit by id(runtime).
    box, seen = _FakeBox(), []
    task = _task()

    async def setup(trace, runtime):
        seen.append(runtime)

    async def finalize(trace, runtime):
        seen.append(runtime)
        await runtime.run(["sh", "-c", "git diff"], {})
    task.setup, task.finalize = setup, finalize
    _fake_verifiers(monkeypatch, box)
    asyncio.run(agentic_replay.replay_swe(task, [Action("bash", "{}", "")]))
    assert seen[0] is seen[1] is box
    assert "run" not in vars(box)                        # the bound is lifted after finalize
