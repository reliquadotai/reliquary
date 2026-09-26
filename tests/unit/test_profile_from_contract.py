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


def test_a_carried_architecture_survives_the_round_trip():
    # The field the CLI seals into a contract. Dropped here, the digest the
    # process computes can never match the one the registry attests.
    contract = dict(PROFILES[_any_profile_id()].to_generation_contract())
    contract["model_architecture"] = "Qwen3ForCausalLM"
    rebuilt = profile_from_contract(contract)
    assert rebuilt.model_architecture == "Qwen3ForCausalLM"
    assert rebuilt.to_generation_contract() == contract


@pytest.mark.parametrize("profile_id", sorted(PROFILES))
def test_a_compiled_profile_emits_no_architecture_key(profile_id):
    # Emitted only when set, so the nine compiled contracts keep the exact
    # bytes -- and digests -- the fleet already attests.
    assert "model_architecture" not in PROFILES[profile_id].to_generation_contract()


def test_an_architecture_that_is_not_text_is_refused():
    contract = dict(PROFILES[_any_profile_id()].to_generation_contract())
    contract["model_architecture"] = 42
    with pytest.raises(ValueError, match="model_architecture"):
        profile_from_contract(contract)


# --- An unknown key is dropped in silence, then surfaces as an unattributable
# digest mismatch naming nothing. Refuse it where it can still be named. ---


def test_an_unknown_top_level_key_is_refused():
    contract = dict(PROFILES[_any_profile_id()].to_generation_contract())
    contract["surprise"] = 1
    with pytest.raises(ValueError, match="surprise"):
        profile_from_contract(contract)


def test_a_typo_in_sampling_is_refused():
    # "temperture" next to a valid "temperature" is the shape of the accident.
    contract = dict(PROFILES[_any_profile_id()].to_generation_contract())
    sampling = dict(contract["sampling"])
    sampling["temperture"] = 0.6
    contract["sampling"] = sampling
    with pytest.raises(ValueError, match="temperture"):
        profile_from_contract(contract)


def test_an_unknown_key_in_an_environment_is_refused():
    contract = dict(PROFILES[_any_profile_id()].to_generation_contract())
    name, body = next(iter(contract["environments"].items()))
    env = dict(body)
    env["surprise"] = 1
    contract["environments"] = {name: env}
    with pytest.raises(ValueError) as caught:
        profile_from_contract(contract)
    assert "surprise" in str(caught.value)
    assert name in str(caught.value)


def test_an_unknown_key_in_bft_is_refused():
    contract, name = _profile_contract_with("bft")
    env = dict(contract["environments"][name])
    env["bft"] = {**env["bft"], "surprise": 1}
    contract["environments"] = {name: env}
    with pytest.raises(ValueError, match="surprise"):
        profile_from_contract(contract)


def test_an_unknown_key_in_episode_is_refused():
    contract, name = _profile_contract_with("episode")
    env = dict(contract["environments"][name])
    env["episode"] = {**env["episode"], "surprise": 1}
    contract["environments"] = {name: env}
    with pytest.raises(ValueError, match="surprise"):
        profile_from_contract(contract)


def test_an_unknown_key_in_prompt_template_is_refused():
    contract, name = _profile_contract_with("prompt_template")
    env = dict(contract["environments"][name])
    env["prompt_template"] = {**env["prompt_template"], "surprise": 1}
    contract["environments"] = {name: env}
    with pytest.raises(ValueError, match="surprise"):
        profile_from_contract(contract)


def test_an_unknown_key_in_throughput_tiebreak_is_refused():
    for profile_id in sorted(PROFILES):
        contract = PROFILES[profile_id].to_generation_contract()
        if contract.get("throughput_tiebreak") is not None:
            break
    else:  # pragma: no cover - a profile always declares one
        pytest.fail("no compiled profile exercises 'throughput_tiebreak'")
    contract["throughput_tiebreak"] = {
        **contract["throughput_tiebreak"], "surprise": 1,
    }
    with pytest.raises(ValueError, match="surprise"):
        profile_from_contract(contract)


def test_a_float_where_a_whole_number_is_meant_is_refused():
    # protocol_version gates wire compatibility; truncating 3.7 to 3 answers a
    # question nobody asked.
    contract = dict(PROFILES[_any_profile_id()].to_generation_contract())
    contract["protocol_version"] = 3.7
    with pytest.raises(ValueError, match="protocol_version"):
        profile_from_contract(contract)


# --- The four values the environment coercions accepted unchanged. Each is a
# round-trip FIXED POINT, so the digest agrees and a malformed contract becomes
# a self-consistent, registry-attested task that boots. ---


@pytest.mark.parametrize("value", [True, 2.7, "abc", [1, 2]])
def test_an_unusable_batch_target_is_refused(value):
    contract = dict(PROFILES[_any_profile_id()].to_generation_contract())
    name, body = next(iter(contract["environments"].items()))
    contract["environments"] = {name: {**body, "batch_target": value}}
    with pytest.raises(ValueError) as caught:
        profile_from_contract(contract)
    assert "batch_target" in str(caught.value)
    assert name in str(caught.value)
    assert not isinstance(caught.value, TypeError)


def test_an_answer_format_that_is_not_text_is_refused():
    contract = dict(PROFILES[_any_profile_id()].to_generation_contract())
    name, body = next(iter(contract["environments"].items()))
    contract["environments"] = {name: {**body, "answer_format": {"a": 1}}}
    with pytest.raises(ValueError) as caught:
        profile_from_contract(contract)
    assert "answer_format" in str(caught.value)
    assert name in str(caught.value)


def test_an_environment_contract_id_that_is_not_text_is_refused():
    contract, name = _profile_contract_with("environment_contract_id")
    env = dict(contract["environments"][name])
    env["environment_contract_id"] = 42
    contract["environments"] = {name: env}
    with pytest.raises(ValueError) as caught:
        profile_from_contract(contract)
    assert "environment_contract_id" in str(caught.value)
    assert name in str(caught.value)


def test_an_environment_manifest_sha256_that_is_not_text_is_refused():
    contract, name = _profile_contract_with("environment_manifest_sha256")
    env = dict(contract["environments"][name])
    env["environment_manifest_sha256"] = 123
    contract["environments"] = {name: env}
    with pytest.raises(ValueError) as caught:
        profile_from_contract(contract)
    assert "environment_manifest_sha256" in str(caught.value)
    assert name in str(caught.value)
    assert not isinstance(caught.value, TypeError)


# --- `prompt_cooldown_windows` and `thinking` arrived with the Teutonic
# profile. The contract emitted them before this reader honoured them, so a
# task carrying that profile was attested and could not boot. ---

def test_prompt_cooldown_windows_survives_the_round_trip():
    contract, name = _profile_contract_with("prompt_cooldown_windows")
    expected = contract["environments"][name]["prompt_cooldown_windows"]
    rebuilt = profile_from_contract(contract)
    assert rebuilt.environments[name].prompt_cooldown_windows == expected


def test_thinking_survives_the_round_trip():
    # The only compiled profile that sets it sets it False, so an assertion
    # against None would pass on a reader that dropped the field entirely.
    contract, name = _profile_contract_with("thinking")
    expected = contract["environments"][name]["thinking"]
    assert isinstance(expected, bool)
    rebuilt = profile_from_contract(contract)
    assert rebuilt.environments[name].thinking is expected


def test_a_non_integer_prompt_cooldown_windows_is_refused():
    contract, name = _profile_contract_with("prompt_cooldown_windows")
    env = dict(contract["environments"][name])
    env["prompt_cooldown_windows"] = 2.5
    contract["environments"] = {name: env}
    with pytest.raises(ValueError) as caught:
        profile_from_contract(contract)
    assert "prompt_cooldown_windows" in str(caught.value)
    assert not isinstance(caught.value, TypeError)


def test_a_truthy_stand_in_for_thinking_is_refused():
    # `bool(1)` would be re-emitted as `true`, so the contract would no longer
    # hash to what it arrived as: a fixed point broken with nothing named.
    contract, name = _profile_contract_with("thinking")
    env = dict(contract["environments"][name])
    env["thinking"] = 1
    contract["environments"] = {name: env}
    with pytest.raises(ValueError) as caught:
        profile_from_contract(contract)
    assert "thinking" in str(caught.value)


def test_a_truthy_stand_in_for_force_answer_is_refused():
    # Same trap, one field over: `bool()` accepted anything here.
    contract, name = _profile_contract_with("bft")
    env = dict(contract["environments"][name])
    env["bft"] = {**env["bft"], "force_answer": 1}
    contract["environments"] = {name: env}
    with pytest.raises(ValueError) as caught:
        profile_from_contract(contract)
    assert "force_answer" in str(caught.value)
