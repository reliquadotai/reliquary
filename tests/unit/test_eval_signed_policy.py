"""Versioned signed episode bounds never widen historical source policies."""

import copy
import hashlib
import asyncio
import time

import pytest

from reliquary.eval import protocol as ep, sets
from reliquary.eval.grading import require_gradable
from reliquary.sandbox.tasks import ResolvedTask
from tests.unit.test_eval_protocol import envelope, request_protocol


@pytest.fixture(autouse=True)
def installed_identities(monkeypatch):
    monkeypatch.setattr("reliquary.eval.verifiers_source.installed_artifacts", lambda _: "9" * 64)
    monkeypatch.setattr(ep, "renderer_sha256", lambda: "2" * 64)


class Source:
    identity = "reliquary-terminal==0.3.1+frozen"
    options_sha256 = "a" * 64
    lock_sha256 = "b" * 64

    def __len__(self):
        return 1

    def manifest_sha256(self):
        return "d" * 64

    def prompt(self, index):
        return "Solve the frozen terminal task"

    async def resolve(self, index):
        return ResolvedTask("registry.example/task@sha256:" + "c" * 64,
                            limits={"wall_s": 300}, cpu_millis=1000)


def configuration(*, harness="signed-bash-v2"):
    package, split = "reliquary-terminal==0.3.1", "tmax"
    source_policy = ep.SIGNED_SOURCE_POLICIES[(package, split)]
    config = {"package": package, "split": split, "options": {},
        "image_lock_sha256": "b" * 64, "renderer_sha256": "2" * 64,
        "tools_version": "reliquary-tools/1", "tools": ["bash"],
        "max_turns": 40, "source_policy_sha256": source_policy}
    reviewed = {"data": {"environment.toml": source_policy}}
    if harness != "signed-bash-v1":
        policy = ep.signed_evaluation_policy(package, split, harness=harness)
        policy_sha = hashlib.sha256(ep.canonical(policy)).hexdigest()
        config.update(harness=harness, max_turns=48, max_episode_tokens=65536,
                      evaluation_policy_sha256=policy_sha)
        reviewed["data"]["evaluation-cli-policy.json"] = policy_sha
    return config, reviewed


def build(tmp_path, *, config=None, reviewed=None, name="set"):
    default_config, default_reviewed = configuration()
    return sets.build_episode_set("terminal", out=tmp_path / name,
        interaction="signed_episode", count=1, seed=0,
        episode_config=default_config if config is None else config,
        protocol=default_reviewed if reviewed is None else reviewed,
        open_episode=lambda *_: Source())


def test_terminal_48_policy_freezes_distinct_turn_context_and_source_identities(tmp_path):
    config, reviewed = configuration()
    assert ep.signed_evaluation_policy(config["package"], config["split"], harness=config["harness"]) == {
        "schema": "reliquary/evaluation-cli-policy/v1", "package": "reliquary-terminal==0.3.1",
        "split": "tmax", "source_policy_sha256": config["source_policy_sha256"],
        "harness": "signed-bash-v2", "max_turns": 48, "max_episode_tokens": 65536}
    card = build(tmp_path, config=config, reviewed=reviewed)
    harness = card["protocol"]["harness"]
    assert harness == {"id": "signed-bash-v2", "max_turns": 48,
                       "max_episode_tokens": 65536, "sha256": ep.harness_sha256("signed-bash-v2")}
    assert harness["sha256"] != ep.harness_sha256("signed-bash-v1")
    assert card["signed_episode"]["source_policy_sha256"] == reviewed["data"]["environment.toml"]
    assert card["signed_episode"]["evaluation_policy_sha256"] == reviewed["data"]["evaluation-cli-policy.json"]
    assert reviewed["data"]["environment.toml"] != reviewed["data"]["evaluation-cli-policy.json"]
    assert ep.validate_set(card, current_grader=True) == card["protocol_sha256"]
    require_gradable({card["set_id"]: (card, [])})
    request = {**request_protocol(), "harness": "signed-bash-v2", "interaction": "signed_episode",
               "max_turns": 48, "max_episode_tokens": 65536}
    ep.validate_request(envelope(request))
    ep.validate_execution({card["set_id"]: card}, {card["set_id"]: card["protocol_sha256"]}, request=request)
    for changes in ({"max_turns": 49}, {"max_episode_tokens": 65537}):
        with pytest.raises(ValueError):
            ep.validate_request(envelope({**request, **changes}))


def test_new_bounds_require_the_separate_exact_terminal_policy(tmp_path):
    config, reviewed = configuration()
    for index, changes in enumerate(({"max_turns": 49}, {"max_episode_tokens": 65537},
                                    {"max_episode_tokens": 65536.0},
                                    {"evaluation_policy_sha256": config["source_policy_sha256"]})):
        with pytest.raises(ValueError):
            build(tmp_path, name=f"invalid-{index}", config={**config, **changes}, reviewed=reviewed)
    for key in ("evaluation-cli-policy.json", "environment.toml"):
        changed = copy.deepcopy(reviewed)
        changed["data"].pop(key)
        with pytest.raises(ValueError):
            build(tmp_path, name="missing-" + key, config=config, reviewed=changed)
    with pytest.raises(ValueError):
        ep.signed_evaluation_policy("reliquary-swe==0.3.0", "eval", harness="signed-bash-v2")
    for identity in ("signed-codex-cli-v1", "signed-claude-code-cli-v1"):
        with pytest.raises(ValueError):
            ep.signed_evaluation_policy(config["package"], config["split"], harness=identity)
        request = {**request_protocol(), "harness": identity, "interaction": "signed_episode", "max_turns": 48}
        with pytest.raises(ValueError):
            ep.validate_request(envelope(request))


def test_historical_bash_and_telecom_remain_40_with_the_original_harness_shape(tmp_path):
    assert ep.SIGNED_SOURCE_POLICIES[("reliquary-swe==0.3.0", "eval")] == \
        "9456020aa9c05c193a3f3e1b9ec38db2646bcbf0d1e8cec5664b432c239a5640"
    assert {ep.SIGNED_SOURCE_POLICIES[("reliquary-terminal==0.3.1", split)]
            for split in ("tmax", "tmax_sft", "tmax_rl")} == {
                "8e5607fd25268c704becc4fb0bcdc48a983e35ef94530f90aef40c44c1194065"}
    config, reviewed = configuration(harness="signed-bash-v1")
    card = build(tmp_path, config=config, reviewed=reviewed)
    assert card["protocol"]["harness"] == {"id": "signed-bash-v1", "max_turns": 40,
                                           "sha256": ep.harness_sha256("signed-bash-v1")}
    assert "evaluation_policy_sha256" not in card["signed_episode"]
    for identity, interaction in (("signed-bash-v1", "signed_episode"), ("telecom-episode-v1", "episode")):
        request = {**request_protocol(), "harness": identity, "interaction": interaction,
                   "thinking": True, "max_turns": 41}
        with pytest.raises(ValueError):
            ep.validate_request(envelope(request))
    with pytest.raises(ValueError):
        build(tmp_path, name="legacy-48", config={**config, "max_turns": 48}, reviewed=reviewed)


def test_new_frozen_policy_cannot_be_changed_after_sealing(tmp_path):
    card = build(tmp_path)
    for changes in ({"harness_id": "signed-bash-v1"}, {"harness_max_turns": 40},
                    {"harness_max_episode_tokens": 32768},
                    {"harness_max_episode_tokens": 65536.0},
                    {"signed_episode": {**card["signed_episode"], "evaluation_policy_sha256": "0" * 64}}):
        with pytest.raises(ValueError):
            ep.validate_set({**card, **changes})
    changed = copy.deepcopy(card)
    changed["protocol"]["harness"]["max_episode_tokens"] = 65536.0
    changed["protocol_json"] = ep.canonical(changed["protocol"]).decode()
    changed["protocol_sha256"] = ep.digest(changed["protocol"])
    with pytest.raises(ValueError):
        ep.validate_execution({card["set_id"]: changed}, {card["set_id"]: changed["protocol_sha256"]})


def bounded_session(tmp_path):
    pytest.importorskip("reliquary_sandbox.attest")
    from reliquary.eval.storage import subnet_key
    from tests.unit.test_evaluation_sessions import build as build_session

    env = build_session(tmp_path)
    config, reviewed = configuration()
    card = ep.seal_set({**{key: value for key, value in env.card.items() if not key.startswith("protocol")},
        "harness_id": "signed-bash-v2", "harness_max_turns": 48, "harness_max_episode_tokens": 65536,
        "signed_episode": {**env.card["signed_episode"],
                           "evaluation_policy_sha256": config["evaluation_policy_sha256"]}}, reviewed=reviewed)
    env.sessions.subnet._path(subnet_key(card["set_id"], "set.json")).write_bytes(ep.canonical(card))
    request = {**env.body["protocol"], "harness": "signed-bash-v2", "max_turns": 48,
               "max_episode_tokens": 65536, "protocol_bindings": {card["set_id"]: card["protocol_sha256"]}}
    env.body.update(protocol=request, protocol_json=ep.canonical(request).decode(),
        protocol_sha256=ep.digest(request), protocol_bindings=request["protocol_bindings"],
        set_protocol={key: card[key] for key in ("protocol", "protocol_json", "protocol_sha256")})
    env.card = card
    return env


def test_actual_signed_grant_binds_new_policy_and_model_bounds(tmp_path):
    from tests.unit.test_evaluation_sessions import open_

    env = bounded_session(tmp_path)
    request = env.body["protocol"]
    grant = open_(env)
    binding = grant["binding"]
    assert binding["harness"] == "signed-bash-v2"
    assert binding["max_turns"] == 48 and binding["max_episode_tokens"] == 65536
    assert binding["evaluation_policy_sha256"] == env.card["signed_episode"]["evaluation_policy_sha256"]
    assert binding["prompt_sha256"] == env.rows[0]["prompt_sha256"]
    assert binding["reference_sha256"] == env.rows[0]["reference_sha256"]
    assert len(env.persisted.documents) == 1
    changed = {**request, "max_turns": 49}
    with pytest.raises(ValueError):
        open_(env, {"protocol": changed, "protocol_json": ep.canonical(changed).decode(),
                    "protocol_sha256": ep.digest(changed)})
    assert len(env.persisted.documents) == 1


def test_bounded_bash_actual_input_is_fenced_before_model_or_runner(tmp_path, monkeypatch):
    from reliquary.environment import agentic_swe
    from reliquary.eval import signed_agentic
    from tests.unit.test_evaluation_sessions import open_

    env = bounded_session(tmp_path)
    grant = open_(env)
    task = {"eval_id": env.body["eval_id"], "task_id": env.body["task_id"], "model": env.body["model"],
        "protocol": env.body["protocol"], "protocol_sha256": env.body["protocol_sha256"],
        "set_protocols": {env.card["set_id"]: env.body["set_protocol"]}}
    row = {**env.rows[0], "set_id": env.card["set_id"],
           "messages": [{"role": "user", "content": env.source.prompt(env.rows[0]["source_index"])}]}
    signed_agentic._validate_signed_prompt(row=row, binding=grant["binding"], token=grant["grant"]["token"])
    monkeypatch.setattr(agentic_swe, "_dist_commit", lambda _: agentic_swe.SUPPORTED_VERIFIERS)
    calls = []
    def model_forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("model construction preceded the input fence")
    monkeypatch.setattr(signed_agentic, "model_client", model_forbidden)
    changed = {**row, "messages": [{"role": "user", "content": "changed API task"}]}
    for actual_row, actual_grant in (
        (changed, grant),
        ({**row, "messages": row["messages"] * 2}, grant),
        (changed, {**grant, "binding": {**grant["binding"],
             "prompt_sha256": signed_agentic.prompt_sha256("changed API task")}}),
        (row, {**grant, "binding": {**grant["binding"], "reference_sha256": "0" * 64}}),
    ):
        with pytest.raises(ValueError, match="prompt"):
            asyncio.run(signed_agentic.run_signed_episode(generator=None, task=task, row=actual_row,
                sample_index=0, grant=actual_grant, seed=7, deadline=time.monotonic() + 60))
    assert calls == []


def test_bounded_receipt_hashes_are_authoritative_and_legacy_wire_is_unchanged(tmp_path):
    from tests.unit.test_evaluation_sessions import build as build_session, open_, signed_receipt

    legacy = build_session(tmp_path / "legacy")
    old_grant = open_(legacy)
    assert "prompt_sha256" not in old_grant["binding"] and "reference_sha256" not in old_grant["binding"]
    env = bounded_session(tmp_path / "bounded")
    grant = open_(env)
    receipt = signed_receipt(env, grant)
    for key in ("prompt_sha256", "reference_sha256"):
        with pytest.raises(ValueError, match="prompt_identity_changed"):
            asyncio.run(env.sessions.verify_result({**grant["binding"], key: "0" * 64}, receipt,
                card=env.card, row=env.rows[0], final_state=b"final\x00archive"))
    assert asyncio.run(env.sessions.verify_result(grant["binding"], receipt, card=env.card,
        row=env.rows[0], final_state=b"final\x00archive")) == 1
