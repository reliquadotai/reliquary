"""Rebuilding a profile from its generation contract must be exact: the
contract is what a task will carry, and a field lost here is a field the
fleet silently disagrees about."""

import pytest

from reliquary.protocol.profiles import (
    PROFILES,
    profile_from_contract,
)


def _any_profile_id():
    """Pick deterministically, so a failure is reproducible rather than
    depending on which profile happened to sort first that day."""
    return sorted(PROFILES)[0]


@pytest.mark.parametrize("profile_id", sorted(PROFILES))
def test_every_compiled_profile_survives_a_round_trip(profile_id):
    profile = PROFILES[profile_id]
    assert profile_from_contract(profile.to_generation_contract()) == profile


@pytest.mark.parametrize("profile_id", sorted(PROFILES))
def test_the_rebuilt_profile_renders_the_same_contract(profile_id):
    contract = PROFILES[profile_id].to_generation_contract()
    assert profile_from_contract(contract).to_generation_contract() == contract


def test_a_contract_that_is_not_an_object_is_refused():
    with pytest.raises(ValueError):
        profile_from_contract([])


@pytest.mark.parametrize(
    "missing",
    [
        "profile_id",
        "model_id",
        "model_revision",
        "protocol_version",
        "prompt_encoding",
        "collection_seconds",
        "upload_grace_seconds",
        "sampling",
        "environments",
    ],
)
def test_a_missing_top_level_field_is_refused(missing):
    contract = dict(PROFILES[_any_profile_id()].to_generation_contract())
    contract.pop(missing)
    with pytest.raises(ValueError):
        profile_from_contract(contract)


def test_an_environment_that_is_not_an_object_is_refused():
    contract = dict(PROFILES[_any_profile_id()].to_generation_contract())
    contract["environments"] = {"math": "nope"}
    with pytest.raises(ValueError):
        profile_from_contract(contract)


def test_an_environment_missing_its_budget_is_refused():
    contract = dict(PROFILES[_any_profile_id()].to_generation_contract())
    name, body = next(iter(contract["environments"].items()))
    broken = dict(body)
    broken.pop("max_new_tokens")
    contract["environments"] = {name: broken}
    with pytest.raises(ValueError):
        profile_from_contract(contract)
