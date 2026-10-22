"""CPU contract fixtures do not qualify an actual CLI image or operator."""

import asyncio
import dataclasses
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("reliquary_sandbox_verifiers.cli_runner")

from reliquary.eval import protocol as ep, sets, signed_agentic
from reliquary.eval.platform_client import LeaseLost
from reliquary.eval.storage import subnet_key
from reliquary_sandbox.attest.canonical import canonical_json
from reliquary_sandbox_verifiers import cli, cli_runner
from tests.unit.test_eval_protocol import envelope, request_protocol
from tests.unit.test_evaluation_sessions import build as build_sessions, open_


@pytest.fixture
def approved(monkeypatch):
    names = (("mcp__signed_episode__bash", "bash"), ("mcp__signed_episode__edit", "edit"))
    tools = [{"type": "function", "function": {"name": name,
              "parameters": tool["inputSchema"], "description": tool["description"]}}
             for (name, _), tool in zip(names, cli.MCP_TOOLS, strict=True)]
    cap = cli.SignedCLICapability(protocol_id=ep.CLAUDE_CLI_HARNESS,
        runtime_image="fixture/cli@sha256:" + "f" * 64,
        model_tools_json=canonical_json(tools), model_tool_names=names,
        artifacts=(("@anthropic-ai/claude-code", "2.1.232", "sha512-fixture"),
                   ("@agentclientprotocol/claude-agent-acp", "0.67.0", "sha512-fixture")),
        qualification_sha256="b" * 64,
        runner_argv=("/opt/signed-cli/runtime/bin/python", "/opt/signed-cli/acp-runner.py"),
        runner_sha256="c" * 64, initial_template_sha256="d" * 64,
        date_recipe=cli.CLI_DATE_RECIPE)
    monkeypatch.setitem(cli._QUALIFIED, ep.CLAUDE_CLI_HARNESS, cap)
    monkeypatch.setattr("reliquary.eval.verifiers_source.installed_artifacts", lambda _: "9" * 64)
    monkeypatch.setattr(ep, "renderer_sha256", lambda: "2" * 64)
    return cap


def cli_environment(tmp_path, approved):
    env = build_sessions(tmp_path)
    cap_sha = cli_runner.capability_sha256(approved)
    source_sha = ep.SIGNED_SOURCE_POLICIES[("reliquary-swe==0.3.0", "eval")]
    policy_sha = ep.digest(ep.signed_evaluation_policy("reliquary-swe==0.3.0", "eval",
                                                    harness=ep.CLAUDE_CLI_HARNESS))
    private = {**env.card["signed_episode"], "package": "reliquary-swe==0.3.0",
        "env_package": "reliquary-swe==0.3.0+frozen", "source_policy_sha256": source_sha,
        "evaluation_policy_sha256": policy_sha,
        "cli_capability": cli_runner.capability_identity(approved), "cli_capability_sha256": cap_sha}
    reviewed = {"data": {"environment.toml": source_sha, "evaluation-cli-policy.json": policy_sha,
                          "signed-cli-capability.json": cap_sha}}
    card = ep.seal_set({**{key: value for key, value in env.card.items() if not key.startswith("protocol")},
        "source": "swe", "env": "swe", "split": "eval", "harness_id": ep.CLAUDE_CLI_HARNESS,
        "harness_max_turns": 48, "harness_max_episode_tokens": 65536,
        "signed_episode": private, "episode": {**env.card["episode"],
            "runtime_image_sha256": env.source.task.image.rsplit("sha256:", 1)[1],
            "tools_sha256": approved.model_tools_sha256}}, reviewed=reviewed)
    env.sessions.subnet._path(subnet_key(card["set_id"], "set.json")).write_bytes(ep.canonical(card))
    request = {**env.body["protocol"], "harness": ep.CLAUDE_CLI_HARNESS, "max_turns": 48,
        "max_episode_tokens": 65536, "sets": [{"set_id": card["set_id"], "env": "swe", "problems": 1, "samples": 1}],
        "protocol_bindings": {card["set_id"]: card["protocol_sha256"]}}
    env.body.update(protocol=request, protocol_json=ep.canonical(request).decode(),
        protocol_sha256=ep.digest(request), protocol_bindings=request["protocol_bindings"],
        selection={"env": "swe", "problems": 1, "samples": 1},
        set_protocol={key: card[key] for key in ("protocol", "protocol_json", "protocol_sha256")})
    env.card = card
    return env


def execution(env, grant):
    row = {**env.rows[0], "set_id": env.card["set_id"],
           "messages": [{"role": "user", "content": env.source.prompt(env.rows[0]["source_index"])}]}
    task = {"eval_id": env.body["eval_id"], "task_id": env.body["task_id"], "model": env.body["model"],
        "protocol": env.body["protocol"], "protocol_sha256": env.body["protocol_sha256"],
        "set_protocols": {env.card["set_id"]: env.body["set_protocol"]},
        "thinking": False, "max_new_tokens": 64, "sampling": env.body["protocol"]["sampling"]}
    return task, row


def test_registry_empty_and_other_harnesses_remain_unavailable(monkeypatch):
    monkeypatch.setattr(cli, "_QUALIFIED", {})
    with pytest.raises(ValueError, match="not approved"):
        ep.signed_cli_capability()
    for harness in (ep.CLAUDE_CLI_HARNESS, "signed-codex-cli-v1"):
        with pytest.raises(ValueError):
            ep.validate_request(envelope({**request_protocol(), "harness": harness,
                "interaction": "signed_episode", "max_turns": 48, "max_episode_tokens": 65536}))


def test_cli_source_binds_capability_and_exact_swe_only_policy(tmp_path, approved):
    env = cli_environment(tmp_path, approved)
    cap, identity, sha = ep.signed_cli_capability()
    assert cap is approved and ep.digest(identity) == sha
    assert identity["date_recipe"] == cli.CLI_DATE_RECIPE
    assert identity["date_template_sha256"] == cli.CLI_DATE_REMINDER_TEMPLATE_SHA256
    assert "context_date" not in identity and "context_date" not in ep.canonical(env.card).decode()
    assert env.card["protocol"]["harness"]["max_turns"] == 48
    assert env.card["protocol"]["source"]["episode"]["tools_sha256"] == approved.model_tools_sha256
    ep.validate_set(env.card, current_grader=True)
    ep.validate_execution({env.card["set_id"]: env.card}, env.body["protocol_bindings"], request=env.body["protocol"])
    for package, split in (("reliquary-terminal==0.3.1", "tmax"), ("reliquary-swe==0.3.0", "train")):
        with pytest.raises(ValueError):
            ep.signed_evaluation_policy(package, split, harness=ep.CLAUDE_CLI_HARNESS)
    for changes in ({"harness_max_turns": 49}, {"harness_max_episode_tokens": 65536.0},
                    {"signed_episode": {**env.card["signed_episode"], "cli_capability_sha256": "0" * 64}}):
        with pytest.raises(ValueError):
            ep.validate_set({**env.card, **changes})


def test_real_issuer_binds_cli_date_images_and_model_bounds(tmp_path, approved):
    env = cli_environment(tmp_path, approved)
    grant = open_(env)
    task, row = execution(env, grant)
    binding = grant["binding"]
    assert binding["context_date"] == cli.utc_context_date()
    assert binding["prompt_sha256"] == env.rows[0]["prompt_sha256"]
    assert binding["reference_sha256"] == env.rows[0]["reference_sha256"]
    assert binding["cli_runtime_image"] == approved.runtime_image != binding["image"]
    assert binding["max_turns"] == 48 and binding["max_episode_tokens"] == 65536
    assert signed_agentic._validate_cli_grant(task=task, row=row, grant=grant["grant"], binding=binding,
        source=env.card["protocol"]["source"]["episode"])[1] == cli_runner.capability_sha256(approved)


def test_builder_uses_actual_cli_tools_and_refuses_shared_task_cli_image(tmp_path, approved):
    from tests.unit.test_eval_signed_policy import Source
    from reliquary.sandbox.tasks import ResolvedTask

    class SweSource(Source):
        identity = "reliquary-swe==0.3.0+frozen"

    source_sha = ep.SIGNED_SOURCE_POLICIES[("reliquary-swe==0.3.0", "eval")]
    cap_sha = cli_runner.capability_sha256(approved)
    policy_sha = ep.digest(ep.signed_evaluation_policy("reliquary-swe==0.3.0", "eval",
                                                    harness=ep.CLAUDE_CLI_HARNESS))
    config = {"package": "reliquary-swe==0.3.0", "split": "eval", "options": {},
        "image_lock_sha256": "b" * 64, "renderer_sha256": "2" * 64,
        "tools_version": approved.tools_version, "tools": ["bash", "edit"],
        "max_turns": 48, "max_episode_tokens": 65536, "harness": ep.CLAUDE_CLI_HARNESS,
        "source_policy_sha256": source_sha, "evaluation_policy_sha256": policy_sha,
        "cli_capability_sha256": cap_sha}
    reviewed = {"data": {"environment.toml": source_sha, "evaluation-cli-policy.json": policy_sha,
                          "signed-cli-capability.json": cap_sha}}
    source = SweSource()
    def build(name):
        return sets.build_episode_set("swe", out=tmp_path / name, interaction="signed_episode",
            count=1, seed=0, episode_config=config, protocol=reviewed, open_episode=lambda *_: source)
    card = build("valid")
    assert card["episode"]["tools_sha256"] == approved.model_tools_sha256
    assert card["signed_episode"]["cli_capability"] == cli_runner.capability_identity(approved)
    async def same_image(index):
        return ResolvedTask(approved.runtime_image, limits={"wall_s": 300}, cpu_millis=1000)
    source.resolve = same_image
    with pytest.raises(ValueError, match="differ"):
        build("same-image")


def test_historical_signed_cli_receipt_retains_issuer_date(tmp_path, approved, monkeypatch):
    from reliquary_sandbox.attest import SessionClaims, chain_hash, sign_record
    from reliquary_sandbox.attest.records import parse_body
    from tests.unit.sandbox_fixtures import transcript

    env = cli_environment(tmp_path, approved)
    monkeypatch.setattr(cli, "utc_context_date", lambda: "2020-01-02")
    grant = open_(env)
    assert grant["binding"]["context_date"] == "2020-01-02"
    monkeypatch.setattr(cli, "utc_context_date", lambda: "2026-10-22")
    claims = SessionClaims.from_dict(grant["grant"]["token"]["claims"])
    receipt = transcript(env.validator, env.machine, claims, tools=("bash", "edit"), state=b"final")
    opened = dataclasses.replace(parse_body(receipt["records"][0]["body"]),
                                 env_package=env.card["signed_episode"]["env_package"])
    final = dataclasses.replace(parse_body(receipt["records"][-1]["body"]), prev=chain_hash(opened))
    receipt["records"] = [sign_record(env.machine, claims.machine_id, opened),
                          sign_record(env.machine, claims.machine_id, final)]
    assert asyncio.run(env.sessions.verify_result(grant["binding"], receipt, card=env.card,
                       row=env.rows[0], final_state=b"final")) == 1
    for key in ("prompt_sha256", "reference_sha256"):
        with pytest.raises(ValueError, match="identity_changed"):
            asyncio.run(env.sessions.verify_result({**grant["binding"], key: "0" * 64}, receipt,
                card=env.card, row=env.rows[0], final_state=b"final"))


@pytest.mark.parametrize("key,value", [
    ("context_date", "2020-01-01"), ("cli_capability_sha256", "0" * 64),
    ("cli_runtime_image", "other@sha256:" + "0" * 64), ("evaluation_policy_sha256", "0" * 64),
    ("max_turns", 48.0), ("max_episode_tokens", 65536.0), ("tools", ["bash"]),
    ("source_index", 99), ("image", "other@sha256:" + "0" * 64), ("worker_id", "other"),
    ("lease_fence", "0" * 64), ("episode", {}), ("harness_sha256", "0" * 64),
    ("prompt_sha256", "0" * 64), ("reference_sha256", "0" * 64),
])
def test_changed_grant_refuses_before_cli_dispatch(tmp_path, approved, key, value):
    env = cli_environment(tmp_path, approved)
    grant = open_(env)
    task, row = execution(env, grant)
    with pytest.raises(ValueError):
        signed_agentic._validate_cli_grant(task=task, row=row, grant=grant["grant"],
            binding={**grant["binding"], key: value}, source=env.card["protocol"]["source"]["episode"])


@pytest.mark.parametrize("messages", [
    [{"role": "user", "content": "changed task"}],
    [{"role": "user", "content": "Solve held-out task 0"}, {"role": "user", "content": "extra"}],
    [{"role": "system", "content": "Solve held-out task 0"}],
])
def test_actual_prompt_bytes_or_extra_messages_refuse_before_dispatch(tmp_path, approved, messages):
    env = cli_environment(tmp_path, approved)
    grant = open_(env)
    task, row = execution(env, grant)
    with pytest.raises(ValueError, match="prompt differs"):
        signed_agentic._validate_cli_grant(task=task, row={**row, "messages": messages},
            grant=grant["grant"], binding=grant["binding"], source=env.card["protocol"]["source"]["episode"])


@pytest.mark.asyncio
async def test_explicit_cli_dispatch_uses_only_injected_checkpoint_and_lease_failure(tmp_path, approved, monkeypatch):
    from reliquary.environment import agentic_swe
    env = await asyncio.to_thread(cli_environment, tmp_path, approved)
    grant = await asyncio.to_thread(open_, env)
    task, row = execution(env, grant)
    monkeypatch.setattr(agentic_swe, "_dist_commit", lambda _: agentic_swe.SUPPORTED_VERIFIERS)
    generator = SimpleNamespace(closed=False)
    generator.close = lambda: setattr(generator, "closed", True)
    clients = []
    def make_client(actual_generator, **kwargs):
        assert actual_generator is generator and kwargs["cli_bridge"] == "signed-bridge"
        clients.append(kwargs)
        return SimpleNamespace(token_ids=(11, 13), generated_spans=((0, 2),), turns=1,
            tool_calls=1, transcript_tokens=9, finish_reason="stop")
    monkeypatch.setattr(signed_agentic, "model_client", make_client)
    async def compose(**kwargs):
        assert kwargs["protocol_id"] == ep.CLAUDE_CLI_HARNESS
        assert kwargs["model"] == task["model"]["repo"] + "@" + task["model"]["revision"]
        assert kwargs["context_date"] == grant["binding"]["context_date"]
        assert kwargs["max_turns"] == 48 and kwargs["max_episode_tokens"] == 65536
        kwargs["make_model_client"]("signed-bridge", kwargs["deadline"] - 5)
        return SimpleNamespace(error=None, final={"body": {"status": "graded"}}, transcript={"signed": True}, state=b"state")
    monkeypatch.setattr(cli_runner, "run_cli_episode", compose)
    args = dict(generator=generator, task=task, row=row, sample_index=0, grant=grant,
                seed=7, deadline=time.monotonic() + 60)
    result = await signed_agentic.run_signed_episode(**args)
    assert len(clients) == 1 and result["episode"]["binding"]["context_date"] == cli.utc_context_date()
    assert result["episode"]["model_token_ids"] == [11, 13] and result["completion_tokens"] == 2
    async def lease_lost(**kwargs):
        raise cli_runner.CLILeaseLost("fixture lease lost")
    monkeypatch.setattr(cli_runner, "run_cli_episode", lease_lost)
    with pytest.raises(LeaseLost):
        await signed_agentic.run_signed_episode(**args)
    assert generator.closed


@pytest.mark.asyncio
async def test_legacy_episode_lease_watcher_cancels_tool_wait_and_closes_model():
    state = SimpleNamespace(lost=False, closed=False, cleaned=False)
    generator = SimpleNamespace(close=lambda: setattr(state, "closed", True))
    async def waiting():
        try:
            state.lost = True
            await asyncio.Future()
        finally:
            state.cleaned = True
    with pytest.raises(LeaseLost):
        await signed_agentic._lease_bound(waiting(), generator=generator, cancelled=lambda: state.lost)
    assert state.closed and state.cleaned
