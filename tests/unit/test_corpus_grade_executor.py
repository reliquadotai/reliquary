"""Grade leases and the executor's half, over fakes (Docker in tests/integration)."""

import asyncio
import json
import subprocess
import sys
import threading
import types

import httpx
import pytest
from pydantic import ValidationError

from reliquary.corpus.replay_compare import Action
from reliquary.validator import corpus_grade_executor
from reliquary.validator.agentic_replay import TOOL_PROGRAM, BoxLimits, ReplayTimeout
from reliquary.validator.corpus_grade_executor import GradeExecutor, run_grade_item
from reliquary.validator.corpus_grade_protocol import (
    GRADE_PROTOCOL,
    GradeAction,
    GradeItem,
    GradeLease,
    GradeResult,
)

SID = "a" * 64
ENV = {"package": "reliquary-swe", "version": "b" * 40}


def _item(mode="grade", **kw):
    item = {"submission_id": SID, "task_index": 3, "instance_id": "repo__x.1", "mode": mode,
            "final_diff": "diff --git a/x b/x\n",
            "actions": [{"tool": "bash", "arguments": '{"command": "ls"}', "observation": "a.py"},
                        {"tool": "bash", "arguments": '{"command": "pwd"}', "observation": None}]}
    item.update(kw)
    return GradeItem.model_validate(item)


def test_a_lease_holds_exactly_one_item():
    lease = {"protocol": GRADE_PROTOCOL, "lease_id": "c" * 32, "expires_at": 1.0, "env": ENV,
             "items": [_item().model_dump()]}
    GradeLease.model_validate(lease)
    with pytest.raises(ValidationError):
        GradeLease.model_validate({**lease, "items": [_item().model_dump()] * 2})
    with pytest.raises(ValidationError):
        GradeResult.model_validate({"results": []})


def _harness(tool, arguments):
    proc = subprocess.run([sys.executable, "-c", TOOL_PROGRAM], capture_output=True, text=True,
                          input=json.dumps({"tool": tool, "arguments": arguments, "timeout": 30}))
    return proc.stdout


@pytest.mark.parametrize("tool,arguments,observation", [
    (" bash", '{"command": "ls"}', "error: unknown tool ' bash'"),   # the harness's answer
    ("search", '{"query": "x"}', "error: unknown tool 'search'"),
    ("bash", '{"command ": "echo hi"}', ""),                          # .get: an empty command
    ("bash", '{"command": "echo hi", "timeout": 5}', "hi\n"),        # extra keys ignored
])
def test_any_call_is_leased_and_replays_as_the_harness_answers_it(tool, arguments, observation):
    GradeAction(tool=tool, arguments=arguments, observation=observation)
    assert _harness(tool, arguments) == observation


def test_only_sizes_bound_an_action():
    GradeAction(tool="", arguments="", observation=None)
    with pytest.raises(ValidationError):
        GradeAction(tool="x" * 1025, arguments="{}", observation=None)


async def _report(task, patch):
    return types.SimpleNamespace(applied=True, reward=1.0)


async def _replay(task, actions):
    assert [a.observation for a in actions] == ["a.py", None]
    assert all(isinstance(a, Action) for a in actions)
    return ["a.py\n", "/testbed\n"], "diff --git a/x b/x\n"


def test_grade_mode_reports_application_and_tests():
    result = asyncio.run(run_grade_item(_item(), task_for=lambda iid: iid, grade=_report))
    assert result == {"status": "ok", "diff_applied": True, "tests_passed": True}


def test_a_scoring_timeout_is_a_timeout():
    async def slow(task, patch):
        raise TimeoutError

    result = asyncio.run(run_grade_item(_item(), task_for=str, grade=slow))
    assert result["status"] == "timeout"


def test_the_task_is_built_off_the_event_loop():
    threads = []

    def task_for(instance_id):
        threads.append(threading.current_thread())
        return instance_id

    asyncio.run(run_grade_item(_item(), task_for=task_for, grade=_report))
    assert threads and threads[0] is not threading.main_thread()


def test_the_default_box_work_gets_the_executor_limits(monkeypatch):
    limits = BoxLimits(cpu=1.0, memory_gb=2.0, pids=128)
    seen = []

    async def grade(task, patch, *, limits):
        seen.append(("grade", limits))
        return types.SimpleNamespace(applied=False, reward=0.0)

    async def replay(task, actions, *, limits):
        seen.append(("replay", limits))
        return ["a.py", ""], "diff --git a/x b/x\n"

    monkeypatch.setattr(corpus_grade_executor, "grade_patch", grade)
    monkeypatch.setattr(corpus_grade_executor, "replay_swe", replay)
    asyncio.run(run_grade_item(_item(), task_for=str, limits=limits))
    asyncio.run(run_grade_item(_item("replay"), task_for=str, limits=limits))
    assert seen == [("grade", limits), ("replay", limits)]


@pytest.mark.parametrize("bad", [dict(cpu=0), dict(memory_gb=0), dict(pids=0), dict(pids=-1)])
def test_box_limits_are_never_unlimited(bad):
    with pytest.raises(ValueError):
        BoxLimits(**{"cpu": 2.0, "memory_gb": 6.0, "pids": 1024, **bad})


def test_a_partial_reward_is_not_a_pass():
    async def half(task, patch):
        return types.SimpleNamespace(applied=True, reward=0.5)

    result = asyncio.run(run_grade_item(_item(), task_for=str, grade=half))
    assert result == {"status": "ok", "diff_applied": True, "tests_passed": False}


def test_replay_mode_compares_stripped_observations_and_skips_unanswered_ones():
    result = asyncio.run(run_grade_item(_item("replay"), task_for=lambda iid: iid, replay=_replay))
    assert result == {"status": "ok", "replay_diff_equal": True, "observations_compared": 1,
                      "observations_mismatched": []}


def test_replay_mode_reports_mismatches_and_a_different_diff():
    async def other(task, actions):
        return ["b.py\n", "/testbed\n"], "diff --git a/y b/y\n"

    result = asyncio.run(run_grade_item(_item("replay"), task_for=str, replay=other))
    assert result == {"status": "ok", "replay_diff_equal": False, "observations_compared": 1,
                      "observations_mismatched": [0]}


def test_an_honest_max_turns_episode_whose_last_turn_writes_replays_to_its_diff():
    """F1: verifiers runs the calls of the turn that reached max_turns (the
    limit is checked before the next model call only), so the recorded diff
    holds what they wrote; the replay must run them too."""
    from reliquary.corpus.trajectory_parse import parse_trajectory
    from tests.unit.test_trajectory_parse import CALL, PROMPT, TERM, TEXT, R, build

    def harness(commands):
        return "".join(f"+{command}\n" for command in commands)   # each call writes one line

    turns = [([TEXT, CALL, TERM], ["a"]), ([TEXT, CALL, CALL, TERM], None)]
    tokens, spans = build(turns)
    served = [json.loads(arguments)["command"] for completion, _ in turns
              for arguments in [a for _, a in R.tool_calls(completion)]]
    recorded_diff = harness(served)                                  # the miner's box ran all three
    parsed = parse_trajectory(R, prompt_ids=PROMPT, tokens=tokens, spans=spans,
                              stop="max_turns", max_turns=2)

    async def box(task, actions):
        return ["a" if a.observation is not None else "" for a in actions], \
            harness(json.loads(a.arguments)["command"] for a in actions)

    item = _item("replay", final_diff=recorded_diff, actions=[
        {"tool": a.tool, "arguments": a.arguments, "observation": a.observation}
        for a in parsed.actions])
    result = asyncio.run(run_grade_item(item, task_for=str, replay=box))
    assert result == {"status": "ok", "replay_diff_equal": True, "observations_compared": 1,
                      "observations_mismatched": []}


def test_a_replay_past_its_deadline_is_a_timeout_and_a_crash_is_an_error():
    async def slow(task, actions):
        raise ReplayTimeout("replay exceeded 3600 s after 2 of 9 actions")

    async def broken(task, actions):
        raise RuntimeError("docker is gone")

    assert asyncio.run(run_grade_item(_item("replay"), task_for=str, replay=slow))["status"] == "timeout"
    assert asyncio.run(run_grade_item(_item("replay"), task_for=str, replay=broken))["status"] == "error"


def test_outcomes_the_trajectory_caused_are_reported_as_its_own():
    """Ruling P23: a box lost or a deadline spent once the recorded actions ran
    is the trajectory's outcome (box_lost / box_timeout), every executor gets
    it, and it is reported as a fact, not as this executor's error."""
    from reliquary.validator.agentic_replay import BoxLost
    from reliquary.validator.corpus_grade_executor import GradeTimeout

    async def lost(task, actions):
        raise BoxLost("the box failed in an action after 1 of 2 actions")

    async def spent(task, actions):
        raise ReplayTimeout("replay exceeded 3600 s after 1 of 2 actions", trajectory_caused=True)

    async def lost_grade(task, patch):
        raise BoxLost("the box failed after the patch was applied")

    async def spent_grade(task, patch):
        raise GradeTimeout("grading exceeded 1800 s", trajectory_caused=True)

    async def slow_setup(task, patch):
        raise GradeTimeout("grading exceeded 1800 s", trajectory_caused=False)

    run = lambda item, **kw: asyncio.run(run_grade_item(item, task_for=str, **kw))  # noqa: E731
    replayed = run(_item("replay"), replay=lost)
    assert replayed["status"] == "box_lost" and "1 of 2" in replayed["detail"]
    assert run(_item("replay"), replay=spent)["status"] == "box_timeout"
    assert run(_item(), grade=lost_grade)["status"] == "box_lost"
    assert run(_item(), grade=spent_grade)["status"] == "box_timeout"
    assert run(_item(), grade=slow_setup)["status"] == "timeout"
    for result in (replayed, run(_item(), grade=spent_grade)):
        GradeResult.model_validate({"results": [{**result, "submission_id": SID}]})


class _GradeBox:
    def __init__(self):
        self.runs = []

    async def prepare_setup(self):
        pass

    async def prepare_execution(self, routes):
        pass

    async def run(self, argv, env):
        self.runs.append(list(argv))
        return types.SimpleNamespace(exit_code=0, stdout="", stderr="")


def _fake_grading(monkeypatch, grade):
    from contextlib import asynccontextmanager

    box = _GradeBox()

    @asynccontextmanager
    async def bounded_box(task, limits):
        yield box

    monkeypatch.setattr(corpus_grade_executor, "bounded_box", bounded_box)
    package = types.ModuleType("reliquary_swe")
    package.grading = types.SimpleNamespace(grade=grade)
    monkeypatch.setitem(sys.modules, "reliquary_swe", package)
    return box


def _grade_task(scoring=None):
    return types.SimpleNamespace(data=types.SimpleNamespace(
        timeout=types.SimpleNamespace(scoring=scoring)))


def test_a_grade_box_failing_after_the_patch_is_applied_is_box_lost(monkeypatch):
    from reliquary.validator.agentic_replay import BoxLost

    async def grade(runtime, data, patch):
        await runtime.run(["git", "checkout", "-q", "--detach", "base"], {})
        await runtime.run(["git", "apply", "-v", "/tmp/agent.diff"], {})
        raise RuntimeError("docker exec: container is not running")   # conftest killed pid 1

    _fake_grading(monkeypatch, grade)
    with pytest.raises(BoxLost):
        asyncio.run(corpus_grade_executor.grade_patch(_grade_task(), "D"))


def test_a_grade_box_failing_before_the_patch_is_the_executors(monkeypatch):
    from reliquary.validator.agentic_replay import BoxLost

    async def grade(runtime, data, patch):
        await runtime.run(["git", "checkout", "-q", "--detach", "base"], {})
        raise RuntimeError("could not check out base")

    _fake_grading(monkeypatch, grade)
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(corpus_grade_executor.grade_patch(_grade_task(), "D"))
    assert not isinstance(caught.value, BoxLost)


@pytest.mark.parametrize("apply_first,caused", [(True, True), (False, False)])
def test_a_grade_deadline_is_the_trajectorys_once_the_patch_is_applied(monkeypatch, apply_first, caused):
    from reliquary.validator.corpus_grade_executor import GradeTimeout

    async def grade(runtime, data, patch):
        if apply_first:
            await runtime.run(["git", "apply", "-v", "/tmp/agent.diff"], {})
        await asyncio.sleep(5)                                         # a test that sleeps forever

    _fake_grading(monkeypatch, grade)
    with pytest.raises(GradeTimeout) as caught:
        asyncio.run(corpus_grade_executor.grade_patch(_grade_task(scoring=0.2), "D"))
    assert caught.value.trajectory_caused is caused


def test_every_result_fits_the_wire_model():
    async def broken(task, actions):
        raise RuntimeError("x" * 5000)

    for result in (asyncio.run(run_grade_item(_item(), task_for=str, grade=_report)),
                   asyncio.run(run_grade_item(_item("replay"), task_for=str, replay=_replay)),
                   asyncio.run(run_grade_item(_item("replay"), task_for=str, replay=broken))):
        GradeResult.model_validate({"results": [result]})


class _Http:
    """The control, answering one lease then nothing."""

    def __init__(self, env=ENV, expires_at=1e12):
        self.posts, self._lease_given, self._env = [], False, env
        self._expires_at = expires_at

    async def post(self, path, json, headers, timeout):
        self.posts.append((path, json))
        request = httpx.Request("POST", f"http://control{path}")
        if path.endswith("/heartbeat"):
            return httpx.Response(200, json={"executor_id": "g1", "model_id": ENV["package"],
                                             "model_revision": ENV["version"]}, request=request)
        if path.endswith("/claim"):
            if self._lease_given:
                return httpx.Response(204, request=request)
            self._lease_given = True
            return httpx.Response(200, request=request, json={
                "protocol": GRADE_PROTOCOL, "lease_id": "c" * 32,
                "expires_at": self._expires_at, "env": self._env, "items": [_item().model_dump()]})
        return httpx.Response(200, json={"outcome": "accepted"}, request=request)


def test_the_executor_claims_works_and_posts_one_result():
    http = _Http()

    async def item(grade_item):
        return {"status": "ok", "diff_applied": True, "tests_passed": False}

    async def go():
        executor = GradeExecutor(http=http, executor_id="g1", token="t" * 40, run_item=item,
                                 env_check=lambda package, version: None, sweep=lambda: 0)
        await executor.start()
        assert await executor.step() is True
        await asyncio.gather(*executor._running)
        assert await executor.step() is False

    asyncio.run(go())
    result = [(path, body) for path, body in http.posts if path.endswith("/result")]
    assert result == [(f"/corpus/internal/grade/{'c' * 32}/result",
                       {"results": [{"status": "ok", "diff_applied": True, "tests_passed": False,
                                     "submission_id": SID}]})]
    GradeResult.model_validate(result[0][1])
    claim = next(body for path, body in http.posts if path.endswith("/claim"))
    assert claim == {"executor_id": "g1", "env_package": ENV["package"], "env_version": ENV["version"]}


def test_an_expired_lease_is_skipped_not_worked():
    http = _Http(expires_at=1000.0)
    worked = []

    async def item(grade_item):
        worked.append(grade_item)
        return {"status": "ok"}

    async def go():
        executor = GradeExecutor(http=http, executor_id="g1", token="t" * 40, run_item=item,
                                 env_check=lambda p, v: None, sweep=lambda: 0,
                                 wall_clock=lambda: 1000.0)
        await executor.start()
        assert await executor.step() is True
        assert not executor._running

    asyncio.run(go())
    assert worked == [] and not [p for p, _ in http.posts if p.endswith("/result")]


def test_start_sweeps_orphaned_boxes_before_any_claim():
    http = _Http()
    swept = []

    async def go():
        executor = GradeExecutor(http=http, executor_id="g1", token="t" * 40,
                                 env_check=lambda p, v: None, sweep=lambda: swept.append(1) or 3)
        await executor.start()

    asyncio.run(go())
    assert swept == [1] and not [p for p, _ in http.posts if p.endswith("/claim")]


def test_the_executor_hands_its_limits_to_the_default_item_runner():
    limits = BoxLimits(cpu=1.0, memory_gb=3.0, pids=256)
    executor = GradeExecutor(http=_Http(), executor_id="g1", token="t" * 40, limits=limits)
    assert executor._run_item.keywords == {"limits": limits}


def test_the_executor_refuses_a_lease_for_another_env():
    http = _Http(env={"package": "reliquary-swe", "version": "e" * 40})

    async def go():
        executor = GradeExecutor(http=http, executor_id="g1", token="t" * 40,
                                 run_item=lambda item: None, env_check=lambda p, v: None,
                                 sweep=lambda: 0)
        await executor.start()
        await executor.step()

    with pytest.raises(RuntimeError, match="this executor grades"):
        asyncio.run(go())
    assert not [p for p, _ in http.posts if p.endswith("/result")]


def test_the_executor_holds_no_more_items_than_its_concurrency():
    http = _Http()

    async def go():
        release = asyncio.Event()

        async def item(grade_item):
            await release.wait()
            return {"status": "ok"}

        executor = GradeExecutor(http=http, executor_id="g1", token="t" * 40, run_item=item,
                                 concurrency=1, env_check=lambda p, v: None, sweep=lambda: 0)
        await executor.start()
        assert await executor.step() is True
        claims = sum(1 for p, _ in http.posts if p.endswith("/claim"))
        assert await executor.step() is False
        assert sum(1 for p, _ in http.posts if p.endswith("/claim")) == claims
        release.set()
        await asyncio.gather(*executor._running)

    asyncio.run(go())


def test_an_executor_with_another_env_installed_refuses_to_start():
    async def go():
        executor = GradeExecutor(http=_Http(), executor_id="g1", token="t" * 40,
                                 env_check=lambda package, version: "reliquary-swe is at another commit",
                                 sweep=lambda: 0)
        await executor.start()

    with pytest.raises(RuntimeError, match="another commit"):
        asyncio.run(go())


def test_an_executor_without_its_token_does_not_start():
    with pytest.raises(ValueError, match="RELIQUARY_EXECUTOR_TOKEN"):
        GradeExecutor(http=_Http(), executor_id="g1", token="")


def test_grade_is_a_registry_scope(monkeypatch):
    from reliquary.infrastructure import corpus_executor_store as executors
    from tests.unit.test_corpus_job_store import _FakeMultiObjectR2

    fake = _FakeMultiObjectR2()
    monkeypatch.setattr(executors, "get_s3_client", lambda **kw: fake)
    doc, created = asyncio.run(executors.register_executor(
        executor_id="g1", token_sha256="d" * 64, model_id="reliquary-swe", model_revision="b" * 40,
        expires_at=2e9, now=1000.0, scope="grade", provider_id="hetzner"))
    assert created and doc["scope"] == "grade"
    with pytest.raises(ValueError, match="env commit"):
        asyncio.run(executors.register_executor(
            executor_id="g2", token_sha256="d" * 64, model_id="reliquary-swe", model_revision="main",
            expires_at=2e9, now=1000.0, scope="grade", provider_id="hetzner"))


def test_the_admin_service_registers_a_grade_executor():
    from reliquary.admin.service import RegisterExecutor

    body = RegisterExecutor(executor_id="g1", token_sha256="d" * 64, model_id="reliquary-swe",
                            model_revision="b" * 40, expires_at=2e9, scope="grade")
    assert body.scope == "grade"


def test_the_register_command_binds_the_env_pin_and_prints_the_token_once(monkeypatch):
    import hashlib

    from typer.testing import CliRunner

    from reliquary.cli.main import app
    from reliquary.infrastructure import corpus_executor_store

    calls = []

    async def register(**kw):
        calls.append(kw)
        return {"executor_id": kw["executor_id"]}, True

    monkeypatch.setattr(corpus_executor_store, "register_executor", register)
    result = CliRunner().invoke(app, ["corpus", "register-grade-executor", "--executor-id", "g1",
                                      "--env-version", "b" * 40, "--provider-id", "hetzner"])
    assert result.exit_code == 0, result.output
    printed = json.loads(result.output.strip().splitlines()[-1])
    (call,) = calls
    assert call["scope"] == "grade" and call["model_id"] == "reliquary-swe"
    assert call["model_revision"] == "b" * 40
    assert call["token_sha256"] == hashlib.sha256(printed["token"].encode()).hexdigest()


def test_the_grade_executor_command_needs_its_token(monkeypatch):
    from typer.testing import CliRunner

    from reliquary.cli.main import app
    from reliquary.validator import corpus_grade_executor

    calls = []
    monkeypatch.setattr(corpus_grade_executor, "run_grade_executor", lambda **kw: calls.append(kw))
    monkeypatch.setattr(corpus_grade_executor, "docker_storage_refusal", lambda: None)
    monkeypatch.setattr(corpus_grade_executor, "docker_disk_refusal", lambda disk_gb, **kw: None)
    monkeypatch.delenv("RELIQUARY_EXECUTOR_TOKEN", raising=False)
    argv = ["corpus", "grade-executor", "--control-url", "https://control", "--executor-id", "g1"]
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 1 and "RELIQUARY_EXECUTOR_TOKEN" in result.output
    monkeypatch.setenv("RELIQUARY_EXECUTOR_TOKEN", "t" * 43)
    result = CliRunner().invoke(app, argv + ["--concurrency", "2", "--cpus", "1.5",
                                             "--memory-gb", "4", "--pids-limit", "512"])
    assert result.exit_code == 0, result.output
    assert calls == [{"control_url": "https://control", "executor_id": "g1", "concurrency": 2,
                      "limits": BoxLimits(cpu=1.5, memory_gb=4.0, pids=512, disk_gb=10.0)}]


def test_the_grade_executor_command_refuses_bad_box_limits_cleanly(monkeypatch):
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    calls = []
    monkeypatch.setattr(corpus_grade_executor, "run_grade_executor", lambda **kw: calls.append(kw))
    monkeypatch.setattr(corpus_grade_executor, "docker_storage_refusal", lambda: None)
    monkeypatch.setattr(corpus_grade_executor, "docker_disk_refusal", lambda disk_gb, **kw: None)
    monkeypatch.setenv("RELIQUARY_EXECUTOR_TOKEN", "t" * 43)
    argv = ["corpus", "grade-executor", "--control-url", "https://control", "--executor-id", "g1"]
    for bad in (["--cpus", "0"], ["--memory-gb", "-1"], ["--pids-limit", "0"], ["--cpus", "nan"],
                ["--disk-gb", "0"]):
        result = CliRunner().invoke(app, argv + bad)
        assert result.exit_code == 2, (bad, result.output)
        assert result.exception is None or isinstance(result.exception, SystemExit), bad
        assert "must be" in result.output, (bad, result.output)
    assert calls == []


def test_box_names_never_match_a_role_container():
    from reliquary.validator.agentic_replay import BOX_NAME_PREFIX

    assert BOX_NAME_PREFIX == "reliquary-gradebox-"
    for role in ("reliquary-grade-executor", "reliquary-grade", "reliquary-grader"):
        assert not role.startswith(BOX_NAME_PREFIX)


def _completed(returncode, stdout=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="boom")


def test_the_orphan_sweep_never_stops_the_executor(monkeypatch):
    from reliquary.validator import agentic_replay

    calls, logged = [], []
    monkeypatch.setattr(agentic_replay.logger, "error", lambda *a, **k: logged.append(a[0] % a[1:]))
    monkeypatch.setattr(agentic_replay.logger, "exception", lambda *a, **k: logged.append(a[0]))

    def listing_fails(argv, **kw):
        calls.append(argv)
        assert kw.get("check") is not True
        return _completed(1)

    monkeypatch.setattr(agentic_replay.subprocess, "run", listing_fails)
    assert agentic_replay.sweep_orphan_boxes() == 0
    assert len(calls) == 1 and "docker ps failed" in logged[-1]

    def removal_fails(argv, **kw):
        assert kw.get("check") is not True
        return _completed(0, "abc\ndef\n") if argv[1] == "ps" else _completed(1)

    monkeypatch.setattr(agentic_replay.subprocess, "run", removal_fails)
    assert agentic_replay.sweep_orphan_boxes() == 0 and "docker rm of 2 boxes" in logged[-1]

    def no_docker(argv, **kw):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(agentic_replay.subprocess, "run", no_docker)
    assert agentic_replay.sweep_orphan_boxes() == 0 and "sweep failed" in logged[-1]

    def removed(argv, **kw):
        return _completed(0, "abc\ndef\n") if argv[1] == "ps" else _completed(0)

    monkeypatch.setattr(agentic_replay.subprocess, "run", removed)
    assert agentic_replay.sweep_orphan_boxes() == 2


# --- ruling P20: replays need an xfs Docker root (directory order) ---------

_OVERLAY2_XFS = {"DockerRootDir": "/var/lib/docker", "Driver": "overlay2",
                 "DriverStatus": [["Backing Filesystem", "xfs"], ["Supports d_type", "true"]]}
_CONTAINERD = {"DockerRootDir": "/var/lib/docker", "Driver": "overlayfs",
               "DriverStatus": [["driver-type", "io.containerd.snapshotter.v1"]]}


def _fs(types):
    return lambda path: types[path]


def test_an_overlay2_root_on_xfs_is_accepted():
    refusal = corpus_grade_executor.docker_storage_refusal(
        info=_OVERLAY2_XFS, fs_type=_fs({"/var/lib/docker": "xfs"}))
    assert refusal is None


def test_a_docker_root_on_ext4_is_refused_naming_the_filesystem():
    refusal = corpus_grade_executor.docker_storage_refusal(
        info={**_OVERLAY2_XFS, "DriverStatus": [["Backing Filesystem", "extfs"]]},
        fs_type=_fs({"/var/lib/docker": "ext2/ext3"}))
    assert "ext2/ext3" in refusal and "/var/lib/docker" in refusal and "xfs" in refusal


def test_overlay2_backing_filesystem_must_be_xfs_too():
    refusal = corpus_grade_executor.docker_storage_refusal(
        info={**_OVERLAY2_XFS, "DriverStatus": [["Backing Filesystem", "extfs"]]},
        fs_type=_fs({"/var/lib/docker": "xfs"}))
    assert "extfs" in refusal


def test_the_containerd_snapshotter_root_is_checked_too():
    # sandbox-dev-01's shape: image layers live under /var/lib/containerd.
    refusal = corpus_grade_executor.docker_storage_refusal(
        info=_CONTAINERD, fs_type=_fs({"/var/lib/docker": "xfs", "/var/lib/containerd": "ext2/ext3"}))
    assert "ext2/ext3" in refusal and "/var/lib/containerd" in refusal
    assert corpus_grade_executor.docker_storage_refusal(
        info=_CONTAINERD, fs_type=_fs({"/var/lib/docker": "xfs", "/var/lib/containerd": "xfs"})) is None


def test_unreadable_docker_info_is_a_refusal():
    def broken():
        raise RuntimeError("docker info failed (1): Cannot connect")
    refusal = corpus_grade_executor.docker_storage_refusal(probe=broken, fs_type=_fs({}))
    assert "Cannot connect" in refusal


def test_the_grade_executor_command_refuses_a_non_xfs_docker_root(monkeypatch):
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    calls = []
    monkeypatch.setattr(corpus_grade_executor, "run_grade_executor", lambda **kw: calls.append(kw))
    monkeypatch.setattr(corpus_grade_executor, "docker_storage_refusal",
                        lambda: "Docker stores images on ext2/ext3 (/var/lib/docker), not xfs")
    monkeypatch.setenv("RELIQUARY_EXECUTOR_TOKEN", "t" * 43)
    argv = ["corpus", "grade-executor", "--control-url", "https://control", "--executor-id", "g1"]
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 1 and "ext2/ext3" in result.output and "--allow-non-xfs" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert calls == []
    result = CliRunner().invoke(app, argv + ["--allow-non-xfs"])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1


# --- ruling P24: each box's writable layer is bounded (xfs pquota) ----------

def _probe(kib):
    return lambda image: f"Filesystem 1024-blocks Used Available Capacity Mounted on\noverlay {kib} 8 1 1% /\n"


def test_a_daemon_default_box_size_within_the_limit_is_accepted():
    assert corpus_grade_executor.docker_disk_refusal(
        10.0, info=_OVERLAY2_XFS, run_probe=_probe(10 * 2 ** 20)) is None


def test_a_box_without_a_size_limit_is_refused():
    refusal = corpus_grade_executor.docker_disk_refusal(
        10.0, info=_OVERLAY2_XFS, run_probe=_probe(300 * 2 ** 20))
    assert "overlay2.size" in refusal and "pquota" in refusal


def test_the_containerd_image_store_cannot_bound_a_box():
    refusal = corpus_grade_executor.docker_disk_refusal(10.0, info=_CONTAINERD,
                                                        run_probe=_probe(10 * 2 ** 20))
    assert "overlay2" in refusal and "containerd-snapshotter" in refusal


def test_a_failing_disk_probe_is_a_refusal():
    def broken(image):
        raise RuntimeError("docker run failed: no such image")
    refusal = corpus_grade_executor.docker_disk_refusal(10.0, info=_OVERLAY2_XFS, run_probe=broken)
    assert "no such image" in refusal


def test_the_grade_executor_command_refuses_unbounded_box_disks(monkeypatch):
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    calls, asked = [], []
    monkeypatch.setattr(corpus_grade_executor, "run_grade_executor", lambda **kw: calls.append(kw))
    monkeypatch.setattr(corpus_grade_executor, "docker_storage_refusal", lambda: None)

    def disk(disk_gb, **kw):
        asked.append((disk_gb, kw))
        return "a box's / reads 300 GB: set overlay2.size"
    monkeypatch.setattr(corpus_grade_executor, "docker_disk_refusal", disk)
    monkeypatch.setenv("RELIQUARY_EXECUTOR_TOKEN", "t" * 43)
    argv = ["corpus", "grade-executor", "--control-url", "https://control", "--executor-id", "g1",
            "--disk-gb", "12"]
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 1 and "overlay2.size" in result.output and calls == []
    assert asked[0][0] == 12.0
    result = CliRunner().invoke(app, argv + ["--allow-non-xfs"])        # tests only: unchecked
    assert result.exit_code == 0, result.output
    assert calls[0]["limits"].disk_gb is None
