"""`tasks create --model ...` without a template composes the contract from a
model, a named run policy and catalogued environments."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from reliquary.cli.main import app
from reliquary.environment.abi import canonical_sha256
from reliquary.protocol.composition import RUN_POLICIES, ModelSpec, compose_profile
from reliquary.protocol.environment_catalog import ENVIRONMENT_CATALOG
from reliquary.protocol.profiles import PROFILES, TOPLOC_DEPLOYED_DEFAULTS
from tests.unit.test_jobs_cli import _rl_entry, registry  # noqa: F401

V1 = PROFILES["qwen3-4b-base-dapo-reliquary-v1"]


def _args(*extra, **options):
    base = {
        "--task-id": "composed-run",
        "--model": "org/Model",
        "--model-revision": "abc123",
        "--model-architecture": "Qwen3ForCausalLM",
        "--prompt-encoding": "chat_template",
        "--envs": "reliquary_code_v1,reliquary_dapo_math_v1",
        "--run-policy": "suite-v9",
        "--cap": "0.3",
    }
    base.update(options)
    argv = ["tasks", "create"]
    for flag, value in base.items():
        if value is not None:
            argv += [flag, value]
    return argv + list(extra)


def _run(registry, *extra, **options):  # noqa: F811
    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    return CliRunner().invoke(app, _args(*extra, **options))


def test_the_composed_contract_is_what_compose_profile_builds(registry):  # noqa: F811
    result = _run(registry)
    assert result.exit_code == 0, result.output
    entry = registry["entries"]["composed-run"]
    expected = compose_profile(
        profile_id="composed-run",
        model=ModelSpec("org/Model", "abc123", "Qwen3ForCausalLM", "chat_template"),
        run=RUN_POLICIES["suite-v9"],
        environments=["reliquary_code_v1", "reliquary_dapo_math_v1"],
    ).to_generation_contract()
    assert entry.contract == expected
    assert entry.profile_id == "composed-run"
    assert entry.profile_sha256 == canonical_sha256(expected)
    assert entry.mechanism == "rl-discovered-price"


def test_composing_reliquary_v1_parts_gives_its_contract_back(registry):  # noqa: F811
    result = _run(
        registry,
        **{
            "--model": V1.model_id, "--model-revision": V1.model_revision,
            "--prompt-encoding": "raw", "--run-policy": "dapo-v6",
            "--envs": ",".join(V1.environments),
        },
    )
    assert result.exit_code == 0, result.output
    contract = dict(registry["entries"]["composed-run"].contract)
    assert contract.pop("model_architecture") == "Qwen3ForCausalLM"
    assert contract.pop("profile_id") == "composed-run"
    expected = V1.to_generation_contract()
    del expected["profile_id"]
    assert canonical_sha256(contract) == canonical_sha256(expected)


def test_set_overrides_a_tunable_field(registry):  # noqa: F811
    result = _run(
        registry,
        "--set", "reliquary_code_v1.max_new_tokens=16384",
        "--set", "reliquary_code_v1.thinking=false",
    )
    assert result.exit_code == 0, result.output
    code = registry["entries"]["composed-run"].contract["environments"]["reliquary_code_v1"]
    assert code["max_new_tokens"] == 16384
    assert code["thinking"] is False
    math = registry["entries"]["composed-run"].contract["environments"]["reliquary_dapo_math_v1"]
    assert math["max_new_tokens"] == ENVIRONMENT_CATALOG["reliquary_dapo_math_v1"].max_new_tokens


def test_set_overrides_an_episode_limit(registry):  # noqa: F811
    result = _run(
        registry, "--set", "reliquary_telecom_solo_v1.episode.max_turns=20",
        **{"--envs": "reliquary_telecom_solo_v1"},
    )
    assert result.exit_code == 0, result.output
    body = registry["entries"]["composed-run"].contract["environments"]["reliquary_telecom_solo_v1"]
    assert body["episode"]["max_turns"] == 20


@pytest.mark.parametrize(("setting", "message"), [
    ("reliquary_code_v1.answer_format=text", "cannot be overridden"),
    ("reliquary_code_v1.max_new_tokens=lots", "whole number"),
    ("reliquary_code_v1.thinking=yes", "true or false"),
    ("reliquary_code_v1", "ENV.FIELD=VALUE"),
    ("max_new_tokens=4096", "ENV.FIELD=VALUE"),
    ("openmathinstruct.max_new_tokens=4096", "not selected"),
])
def test_a_bad_set_is_refused_before_any_write(registry, setting, message):  # noqa: F811
    result = _run(registry, "--set", setting)
    assert result.exit_code != 0
    assert message in result.output
    assert "composed-run" not in registry["entries"]


def test_sampling_flags_override_the_run_policy(registry):  # noqa: F811
    result = _run(
        registry, "--rollouts", "8", "--temperature", "0.7", "--top-p", "0.95", "--top-k", "20",
    )
    assert result.exit_code == 0, result.output
    sampling = registry["entries"]["composed-run"].contract["sampling"]
    assert sampling == {
        "rollouts": 8, "temperature": 0.7, "top_p": 0.95, "top_k": 20,
        "do_sample": RUN_POLICIES["suite-v9"].sampling.do_sample,
    }


@pytest.mark.parametrize("mode", ["shadow", "enforce"])
def test_a_toploc_proof_is_carried_in_the_named_mode(registry, mode):  # noqa: F811
    result = _run(registry, "--proof", "toploc", "--proof-mode", mode)
    assert result.exit_code == 0, result.output
    proofs = registry["entries"]["composed-run"].contract["proofs"]
    expected = TOPLOC_DEPLOYED_DEFAULTS.to_contract()
    expected["mode"] = mode
    assert proofs == [expected]


def test_no_proof_flag_carries_no_proofs(registry):  # noqa: F811
    result = _run(registry)
    assert result.exit_code == 0, result.output
    assert "proofs" not in registry["entries"]["composed-run"].contract


@pytest.mark.parametrize("extra", [
    ["--proof", "toploc"],
    ["--proof-mode", "shadow"],
    ["--proof", "grail", "--proof-mode", "shadow"],
])
def test_an_incomplete_or_unknown_proof_is_refused(registry, extra):  # noqa: F811
    result = _run(registry, *extra)
    assert result.exit_code != 0
    assert "--proof" in result.output
    assert "composed-run" not in registry["entries"]


@pytest.mark.parametrize("flag", ["--prompt-encoding", "--envs", "--run-policy"])
def test_each_missing_compose_flag_is_named(registry, flag):  # noqa: F811
    result = _run(registry, **{flag: None})
    assert result.exit_code != 0
    assert flag in result.output
    assert "composed-run" not in registry["entries"]


def test_an_unknown_run_policy_is_refused_naming_the_known_ones(registry):  # noqa: F811
    result = _run(registry, **{"--run-policy": "fastest"})
    assert result.exit_code != 0
    assert "suite-v9" in result.output


def test_an_uncatalogued_environment_is_refused(registry):  # noqa: F811
    result = _run(registry, **{"--envs": "envscaler_tools_v1"})
    assert result.exit_code != 0
    assert "catalog" in result.output


@pytest.mark.parametrize("extra", [
    ["--run-policy", "suite-v9"],
    ["--set", "reliquary_code_v1.max_new_tokens=16384"],
    ["--prompt-encoding", "raw"],
    ["--rollouts", "8"],
    ["--proof", "toploc", "--proof-mode", "shadow"],
])
def test_from_profile_with_a_compose_flag_is_refused(registry, extra):  # noqa: F811
    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    argv = [
        "tasks", "create", "--task-id", "composed-run", "--cap", "0.3",
        "--model", "org/Model", "--model-revision", "abc123",
        "--model-architecture", "Qwen3ForCausalLM",
        "--from-profile", V1.profile_id, *extra,
    ]
    result = CliRunner().invoke(app, argv)
    assert result.exit_code != 0
    assert "--from-profile" in result.output
    assert extra[0] in result.output
    assert "composed-run" not in registry["entries"]


def test_a_compose_flag_without_model_is_refused(registry):  # noqa: F811
    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    result = CliRunner().invoke(app, [
        "tasks", "create", "--task-id", "legacy", "--cap", "0.3",
        "--profile-id", V1.profile_id, "--run-policy", "dapo-v6",
    ])
    assert result.exit_code != 0
    assert "--run-policy" in result.output
    assert "legacy" not in registry["entries"]
