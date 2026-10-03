"""Grade leases and the executor's half, over fakes (Docker in tests/integration)."""

import asyncio
import json
import types

import httpx
import pytest
from pydantic import ValidationError

from reliquary.corpus.replay_compare import Action, harness_call_refusal
from reliquary.validator.agentic_replay import ReplayTimeout
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


@pytest.mark.parametrize("tool,arguments", [
    ("bash", '{"command": "ls"}'),
    ("edit", '{"path": "a.py", "old_str": "x", "new_str": "y"}'),
    ("edit", '{"new_str": "y", "path": "a.py", "old_str": "x"}'),
])
def test_the_harness_calls_are_accepted(tool, arguments):
    assert harness_call_refusal(tool, arguments) is None
    GradeAction(tool=tool, arguments=arguments, observation="x")


@pytest.mark.parametrize("tool,arguments", [
    (" bash", '{"command": "ls"}'),                      # not the harness's tool name
    ("bash ", '{"command": "ls"}'),
    ("Bash", '{"command": "ls"}'),
    ("search", '{"query": "x"}'),                         # search is off in this harness
    ("bash", '{"command ": "ls"}'),                      # a key the harness would not read
    ("bash", '{"cmd": "ls"}'),
    ("bash", '{"command": "ls", "timeout": 5}'),          # an extra key
    ("bash", "{}"),                                       # a missing key
    ("edit", '{"path": "a.py", "old_str": "x"}'),
    ("edit", '{"path": "a.py", "old_str": "x", "new_str": "y", "count": 2}'),
    ("bash", '["ls"]'),                                   # not an object
    ("bash", "not json"),
])
def test_a_call_outside_the_harness_is_refused_not_replayed_as_an_error(tool, arguments):
    assert harness_call_refusal(tool, arguments)
    with pytest.raises(ValidationError, match="harness"):
        GradeAction(tool=tool, arguments=arguments, observation="error: unknown tool")


async def _report(task, patch):
    return types.SimpleNamespace(applied=True, reward=1.0)


async def _replay(task, actions):
    assert [a.observation for a in actions] == ["a.py", None]
    assert all(isinstance(a, Action) for a in actions)
    return ["a.py\n", "/testbed\n"], "diff --git a/x b/x\n"


def test_grade_mode_reports_application_and_tests():
    result = asyncio.run(run_grade_item(_item(), task_for=lambda iid: iid, grade=_report))
    assert result == {"status": "ok", "diff_applied": True, "tests_passed": True}


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


def test_a_replay_past_its_deadline_is_a_timeout_and_a_crash_is_an_error():
    async def slow(task, actions):
        raise ReplayTimeout("replay exceeded 3600 s after 2 of 9 actions")

    async def broken(task, actions):
        raise RuntimeError("docker is gone")

    assert asyncio.run(run_grade_item(_item("replay"), task_for=str, replay=slow))["status"] == "timeout"
    assert asyncio.run(run_grade_item(_item("replay"), task_for=str, replay=broken))["status"] == "error"


def test_every_result_fits_the_wire_model():
    async def broken(task, actions):
        raise RuntimeError("x" * 5000)

    for result in (asyncio.run(run_grade_item(_item(), task_for=str, grade=_report)),
                   asyncio.run(run_grade_item(_item("replay"), task_for=str, replay=_replay)),
                   asyncio.run(run_grade_item(_item("replay"), task_for=str, replay=broken))):
        GradeResult.model_validate({"results": [result]})


class _Http:
    """The control, answering one lease then nothing."""

    def __init__(self, env=ENV):
        self.posts, self._lease_given, self._env = [], False, env

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
                "protocol": GRADE_PROTOCOL, "lease_id": "c" * 32, "expires_at": 1e12,
                "env": self._env, "items": [_item().model_dump()]})
        return httpx.Response(200, json={"outcome": "accepted"}, request=request)


def test_the_executor_claims_works_and_posts_one_result():
    http = _Http()

    async def item(grade_item):
        return {"status": "ok", "diff_applied": True, "tests_passed": False}

    async def go():
        executor = GradeExecutor(http=http, executor_id="g1", token="t" * 40, run_item=item,
                                 env_check=lambda package, version: None)
        await executor.start()
        assert await executor.step() is True
        await asyncio.gather(*executor._running)
        assert await executor.step() is False

    asyncio.run(go())
    result = [(path, body) for path, body in http.posts if path.endswith("/result")]
    assert result == [(f"/corpus/internal/grade/{'c' * 32}/result",
                       {"results": [{"status": "ok", "diff_applied": True, "tests_passed": False}]})]
    claim = next(body for path, body in http.posts if path.endswith("/claim"))
    assert claim == {"executor_id": "g1", "env_package": ENV["package"], "env_version": ENV["version"]}


def test_the_executor_refuses_a_lease_for_another_env():
    http = _Http(env={"package": "reliquary-swe", "version": "e" * 40})

    async def go():
        executor = GradeExecutor(http=http, executor_id="g1", token="t" * 40,
                                 run_item=lambda item: None, env_check=lambda p, v: None)
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
                                 concurrency=1, env_check=lambda p, v: None)
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
                                 env_check=lambda package, version: "reliquary-swe is at another commit")
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
        expires_at=2e9, now=1000.0, scope="grade"))
    assert created and doc["scope"] == "grade"
    with pytest.raises(ValueError, match="env commit"):
        asyncio.run(executors.register_executor(
            executor_id="g2", token_sha256="d" * 64, model_id="reliquary-swe", model_revision="main",
            expires_at=2e9, now=1000.0, scope="grade"))


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
                                      "--env-version", "b" * 40])
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
    monkeypatch.delenv("RELIQUARY_EXECUTOR_TOKEN", raising=False)
    argv = ["corpus", "grade-executor", "--control-url", "https://control", "--executor-id", "g1"]
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 1 and "RELIQUARY_EXECUTOR_TOKEN" in result.output
    monkeypatch.setenv("RELIQUARY_EXECUTOR_TOKEN", "t" * 43)
    result = CliRunner().invoke(app, argv + ["--concurrency", "2"])
    assert result.exit_code == 0, result.output
    assert calls == [{"control_url": "https://control", "executor_id": "g1", "concurrency": 2}]
