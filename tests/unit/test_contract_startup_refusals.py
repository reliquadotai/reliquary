"""A contract the binary cannot execute must stop the process at startup, not
at the first window. The registry validates shape; this validates capability."""

import pytest

from reliquary.protocol.profiles import PROFILES
from reliquary.shared.task_registry import TaskEntry
from reliquary.validator.task_config import TaskConfigError, resolve_task_config

PARAMS = {
    "start": 1.0, "decay": 0.98, "rounds_per_step": 1000, "deadband": 0.80,
    "snap": 1.20, "floor": 0.05, "cap": 1.0, "median_rounds": 4800,
    "last_good_fills": 50,
}


def _profile():
    return PROFILES[sorted(PROFILES)[0]]


def _entry(profile, **overrides):
    from reliquary.environment.abi import canonical_sha256

    contract = profile.to_generation_contract()
    payload = {
        "task_id": "glm-run",
        "profile_id": profile.profile_id,
        "mechanism": "rl-discovered-price",
        "params": PARAMS,
        "status": "active",
        "retired_at": None,
        "contract": contract,
    }
    payload.update(overrides)
    # profile_sha256 pins whatever contract this entry actually carries, so a
    # `contract=` override stays self-consistent instead of silently tripping
    # the pre-existing digest check the environment/architecture tests do not
    # target.
    payload["profile_sha256"] = canonical_sha256(payload["contract"])
    return TaskEntry(**payload)


def test_a_carried_contract_passes_the_existing_checks_by_construction():
    # The process builds its profile FROM the contract, so profile_id and
    # profile_sha256 agree without any new logic.
    profile = _profile()
    entry = _entry(profile)
    config = resolve_task_config(
        {"glm-run": entry},
        "glm-run",
        profile_id=profile.profile_id,
        generation_contract=profile.to_generation_contract(),
    )
    assert config.task_id == "glm-run"


def test_a_contract_that_disagrees_with_the_registry_is_refused():
    # THE security case: the deployment mounts one contract, the registry
    # attests another. The existing profile_sha256 check is what catches it,
    # which is exactly why this design adds no new verification logic.
    profile = _profile()
    entry = _entry(profile)
    tampered = profile.to_generation_contract()
    tampered["model_id"] = "someone-elses/model"
    with pytest.raises(TaskConfigError):
        resolve_task_config(
            {"glm-run": entry},
            "glm-run",
            profile_id=profile.profile_id,
            generation_contract=tampered,
        )


def test_an_environment_the_binary_does_not_install_is_refused():
    profile = _profile()
    contract = profile.to_generation_contract()
    contract["environments"] = dict(contract["environments"])
    contract["environments"]["not-installed-anywhere"] = {
        "max_new_tokens": 128, "answer_format": None, "bft": None,
    }
    with pytest.raises(TaskConfigError) as caught:
        resolve_task_config(
            {"glm-run": _entry(profile, contract=contract)},
            "glm-run",
            profile_id=profile.profile_id,
            generation_contract=contract,
        )
    assert "not-installed-anywhere" in str(caught.value)


def test_an_unsupported_model_architecture_is_refused():
    profile = _profile()
    contract = profile.to_generation_contract()
    contract["model_architecture"] = "SomethingNobodyShips"
    with pytest.raises(TaskConfigError) as caught:
        resolve_task_config(
            {"glm-run": _entry(profile, contract=contract)},
            "glm-run",
            profile_id=profile.profile_id,
            generation_contract=contract,
        )
    assert "SomethingNobodyShips" in str(caught.value)


def test_a_contract_without_an_architecture_field_is_not_refused():
    # Historical contracts carry no architecture; refusing them would refuse
    # every task that exists today.
    profile = _profile()
    contract = profile.to_generation_contract()
    assert "model_architecture" not in contract
    resolve_task_config(
        {"glm-run": _entry(profile, contract=contract)},
        "glm-run",
        profile_id=profile.profile_id,
        generation_contract=contract,
    )
