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


def _profile_contract_with(field):
    """A contract and the environment in it that actually exercises an
    optional structure. Searching all profiles rather than the first is the
    point: a test that silently finds nothing is a test that asserts nothing.
    """
    for profile_id in sorted(PROFILES):
        contract = PROFILES[profile_id].to_generation_contract()
        for name, body in contract["environments"].items():
            if body.get(field) is not None:
                return contract, name
    pytest.fail(f"no compiled profile exercises {field!r}; this test cannot run")


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


def test_a_missing_nested_key_in_sampling_is_refused():
    contract = dict(PROFILES[_any_profile_id()].to_generation_contract())
    sampling = dict(contract["sampling"])
    sampling.pop("temperature")
    contract["sampling"] = sampling
    with pytest.raises(ValueError, match="temperature"):
        profile_from_contract(contract)


def test_a_null_required_top_level_string_field_is_refused():
    contract = dict(PROFILES[_any_profile_id()].to_generation_contract())
    contract["profile_id"] = None
    with pytest.raises(ValueError, match="null"):
        profile_from_contract(contract)


def test_a_null_required_top_level_int_field_is_refused():
    contract = dict(PROFILES[_any_profile_id()].to_generation_contract())
    contract["protocol_version"] = None
    with pytest.raises(ValueError, match="null"):
        profile_from_contract(contract)


def test_a_bool_as_int_field_is_refused():
    contract = dict(PROFILES[_any_profile_id()].to_generation_contract())
    contract["protocol_version"] = True
    with pytest.raises(ValueError, match="bool"):
        profile_from_contract(contract)


def test_a_list_as_int_field_is_refused():
    # int() raises a bare TypeError for a list; that must become a ValueError
    # naming the field, like every other rejection in this module.
    contract = dict(PROFILES[_any_profile_id()].to_generation_contract())
    contract["protocol_version"] = [1, 2]
    with pytest.raises(ValueError, match="protocol_version") as caught:
        profile_from_contract(contract)
    assert not isinstance(caught.value, TypeError)


def test_a_list_as_float_field_is_refused():
    # Same failure mode as the int case, for the other coercion that can
    # raise a bare TypeError: float().
    contract = dict(PROFILES[_any_profile_id()].to_generation_contract())
    sampling = dict(contract["sampling"])
    sampling["temperature"] = [1, 2]
    contract["sampling"] = sampling
    with pytest.raises(ValueError, match="temperature") as caught:
        profile_from_contract(contract)
    assert not isinstance(caught.value, TypeError)


def test_a_missing_nested_key_in_bft_is_refused():
    contract, name = _profile_contract_with("bft")
    env = dict(contract["environments"][name])
    bft = dict(env["bft"])
    bft.pop("thinking_budget")
    env["bft"] = bft
    contract["environments"] = {name: env}
    with pytest.raises(ValueError, match="thinking_budget"):
        profile_from_contract(contract)


def test_a_missing_nested_key_in_episode_is_refused():
    contract, name = _profile_contract_with("episode")
    env = dict(contract["environments"][name])
    episode = dict(env["episode"])
    episode.pop("max_turns")
    env["episode"] = episode
    contract["environments"] = {name: env}
    with pytest.raises(ValueError, match="max_turns"):
        profile_from_contract(contract)


def test_a_profile_with_legitimately_null_optional_fields_rebuilds():
    contract = dict(PROFILES[_any_profile_id()].to_generation_contract())
    # Ensure throughput_tiebreak, bft, answer_format, and prompt_template are
    # explicitly set to None (not missing). This is legitimate and must work.
    contract["throughput_tiebreak"] = None
    name = next(iter(contract["environments"].keys()))
    env = dict(contract["environments"][name])
    env["bft"] = None
    env["answer_format"] = None
    env["prompt_template"] = None
    contract["environments"] = {name: env}
    # Should not raise.
    rebuilt = profile_from_contract(contract)
    assert rebuilt is not None
    assert rebuilt.throughput_tiebreak is None


def test_a_missing_nested_key_in_prompt_template_is_refused():
    contract, name = _profile_contract_with("prompt_template")
    env = dict(contract["environments"][name])
    template = dict(env["prompt_template"])
    template.pop("id")
    env["prompt_template"] = template
    contract["environments"] = {name: env}
    with pytest.raises(ValueError, match="'id'"):
        profile_from_contract(contract)


def test_a_missing_nested_key_in_throughput_tiebreak_is_refused():
    # throughput_tiebreak is top-level, not per-environment
    contract = None
    for profile_id in sorted(PROFILES):
        c = PROFILES[profile_id].to_generation_contract()
        if c.get("throughput_tiebreak") is not None:
            contract = c
            break
    if contract is None:
        pytest.fail("no compiled profile exercises 'throughput_tiebreak'; this test cannot run")
    contract = dict(contract)
    tiebreak = dict(contract["throughput_tiebreak"])
    tiebreak.pop("token_cap")
    contract["throughput_tiebreak"] = tiebreak
    with pytest.raises(ValueError, match="token_cap"):
        profile_from_contract(contract)
