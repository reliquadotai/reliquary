"""What a composed profile may change, and every way a declaration is refused."""

from dataclasses import replace

import pytest

from reliquary.protocol.composition import (
    RUN_POLICIES,
    ModelSpec,
    compose_profile,
    model_spec_of,
)
from reliquary.protocol.environment_catalog import ENVIRONMENT_CATALOG
from reliquary.protocol.profiles import (
    PROFILES,
    TOPLOC_DEPLOYED_DEFAULTS,
    BFTProfile,
    profile_from_contract,
)
from reliquary.protocol.release_contract import canonical_sha256

MODEL = ModelSpec(
    model_id="Qwen/Qwen3.8-27B",
    model_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    model_architecture="Qwen3_5ForConditionalGeneration",
    prompt_encoding="chat_template",
    proofs=(TOPLOC_DEPLOYED_DEFAULTS,),
)


def _compose(environments, **kwargs):
    kwargs.setdefault("model", MODEL)
    kwargs.setdefault("run", RUN_POLICIES["suite-v9"])
    return compose_profile(profile_id="composed", environments=environments, **kwargs)


def test_composed_contract_is_a_fixed_point():
    profile = _compose(["reliquary_code_v1", "reliquary_dapo_math_v1"])
    contract = profile.to_generation_contract()
    assert list(contract["environments"]) == ["reliquary_code_v1", "reliquary_dapo_math_v1"]
    rebuilt = profile_from_contract(contract).to_generation_contract()
    assert canonical_sha256(rebuilt) == canonical_sha256(contract)


def test_tunable_override_changes_only_that_field():
    profile = _compose(
        ["reliquary_code_v1", "reliquary_dapo_math_v1"],
        overrides={"reliquary_code_v1": {"max_new_tokens": 16384, "thinking": False}},
    )
    code = profile.environments["reliquary_code_v1"]
    assert code == replace(
        ENVIRONMENT_CATALOG["reliquary_code_v1"], max_new_tokens=16384, thinking=False
    )
    assert profile.environments["reliquary_dapo_math_v1"] == (
        ENVIRONMENT_CATALOG["reliquary_dapo_math_v1"]
    )


def test_episode_override_replaces_inside_the_episode():
    profile = _compose(
        ["reliquary_telecom_solo_v1"],
        overrides={"reliquary_telecom_solo_v1": {
            "episode.max_turns": 20, "episode.max_action_tokens": 2048,
        }},
    )
    episode = profile.environments["reliquary_telecom_solo_v1"].episode
    catalog = ENVIRONMENT_CATALOG["reliquary_telecom_solo_v1"].episode
    assert episode == replace(catalog, max_turns=20, max_action_tokens=2048)


def test_override_does_not_mutate_the_catalog():
    before = ENVIRONMENT_CATALOG["reliquary_code_v1"]
    _compose(["reliquary_code_v1"], overrides={"reliquary_code_v1": {"batch_target": 4}})
    assert ENVIRONMENT_CATALOG["reliquary_code_v1"] is before


@pytest.mark.parametrize("field", [
    "answer_format", "bft", "prompt_template", "environment_contract_id",
    "environment_manifest_sha256", "episode", "episode.renderer_id",
    "episode.max_observation_bytes", "episode.schema", "max_turns",
])
def test_non_tunable_field_is_refused(field):
    with pytest.raises(ValueError, match="cannot be overridden"):
        _compose(
            ["reliquary_telecom_solo_v1"],
            overrides={"reliquary_telecom_solo_v1": {field: None}},
        )


def test_episode_override_on_a_single_turn_environment_is_refused():
    with pytest.raises(ValueError, match="not an episode"):
        _compose(
            ["reliquary_code_v1"],
            overrides={"reliquary_code_v1": {"episode.max_turns": 4}},
        )


def test_override_for_an_environment_not_selected_is_refused():
    with pytest.raises(ValueError, match="not selected"):
        _compose(
            ["reliquary_code_v1"],
            overrides={"reliquary_dapo_math_v1": {"max_new_tokens": 4096}},
        )


@pytest.mark.parametrize("name", [
    "no_such_environment",
    # Installed, but declared by no profile: no reviewed body yet.
    "reliquary_stateful_tools_v2",
    "envscaler_tools_v1",
])
def test_unknown_or_uncatalogued_environment_is_refused(name):
    with pytest.raises(ValueError, match="catalog"):
        _compose([name])


def test_empty_or_duplicate_selection_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        _compose([])
    with pytest.raises(ValueError, match="duplicate"):
        _compose(["reliquary_code_v1", "reliquary_code_v1"])


def test_a_custom_catalog_entry_that_is_not_installed_is_refused():
    catalog = {"made_up_v1": ENVIRONMENT_CATALOG["reliquary_code_v1"]}
    with pytest.raises(ValueError, match="not installed"):
        _compose(["made_up_v1"], catalog=catalog)


@pytest.mark.parametrize(("name", "change", "what"), [
    ("reliquary_code_v1", {"environment_contract_id": "reliquary/boxed-answer/v1"}, "contract"),
    ("reliquary_code_v1", {"environment_manifest_sha256": "0" * 64}, "manifest"),
    ("reliquary_telecom_solo_v1", {"episode": "jsonl"}, "renderer"),
])
def test_spec_mismatch_is_refused(name, change, what):
    body = ENVIRONMENT_CATALOG[name]
    if change.get("episode") == "jsonl":
        change = {"episode": replace(body.episode, renderer_id="reliquary-jsonl-tools-v1")}
    catalog = {**ENVIRONMENT_CATALOG, name: replace(body, **change)}
    with pytest.raises(ValueError, match=what):
        _compose([name], catalog=catalog)


def test_int_temperature_is_refused_because_it_does_not_round_trip():
    run = RUN_POLICIES["suite-v9"]
    int_run = replace(run, sampling=replace(run.sampling, temperature=1))
    with pytest.raises(ValueError, match="round trip"):
        _compose(["reliquary_code_v1"], run=int_run)


@pytest.mark.parametrize("fields", [
    {"max_new_tokens": 16384.0},
    {"batch_target": True},
    {"thinking": 0},
])
def test_override_of_the_wrong_type_is_refused(fields):
    with pytest.raises(ValueError, match="round trip"):
        _compose(["reliquary_code_v1"], overrides={"reliquary_code_v1": fields})


def test_float_proof_threshold_given_as_int_is_refused():
    proof = replace(TOPLOC_DEPLOYED_DEFAULTS, mant_mean_threshold=40)
    with pytest.raises(ValueError, match="round trip"):
        _compose(["reliquary_code_v1"], model=replace(MODEL, proofs=(proof,)))


def test_unknown_prompt_encoding_is_refused():
    with pytest.raises(ValueError, match="prompt encoding"):
        _compose(["reliquary_code_v1"], model=replace(MODEL, prompt_encoding="chatml"))


def test_math_answer_format_constants_would_refuse_is_refused():
    catalog = {
        **ENVIRONMENT_CATALOG,
        "openmathinstruct": replace(ENVIRONMENT_CATALOG["openmathinstruct"], answer_format="text"),
    }
    with pytest.raises(ValueError, match="answer format"):
        _compose(["openmathinstruct"], catalog=catalog, run=RUN_POLICIES["dapo-v6"])


def test_math_cap_below_the_bft_budgets_is_refused():
    catalog = {
        **ENVIRONMENT_CATALOG,
        "openmathinstruct": replace(
            ENVIRONMENT_CATALOG["openmathinstruct"],
            bft=BFTProfile(thinking_budget=2048, answer_budget=512, force_answer=True),
        ),
    }
    # The body alone is fine; the override pushes the cap under the budgets.
    _compose(["openmathinstruct"], catalog=catalog, run=RUN_POLICIES["dapo-v6"])
    with pytest.raises(ValueError, match="BFT"):
        _compose(
            ["openmathinstruct"], catalog=catalog, run=RUN_POLICIES["dapo-v6"],
            overrides={"openmathinstruct": {"max_new_tokens": 2560}},
        )


def test_recomposing_a_compiled_profile_ignores_selection_order():
    profile = PROFILES["teutonic-9b-reliquary-suite-v9-dev1"]
    composed = compose_profile(
        profile_id=profile.profile_id,
        model=model_spec_of(profile),
        run=RUN_POLICIES["suite-v9"],
        environments=reversed(sorted(profile.environments)),
    )
    assert canonical_sha256(composed.to_generation_contract()) == canonical_sha256(
        profile.to_generation_contract()
    )


def test_invalid_sampling_is_refused_on_composition():
    run = RUN_POLICIES["suite-v9"]
    bad = replace(run, sampling=replace(run.sampling, top_p=0.0))
    with pytest.raises(ValueError, match="top_p"):
        _compose(["reliquary_code_v1"], run=bad)


def test_an_action_budget_above_the_episode_budget_is_refused():
    with pytest.raises(ValueError, match="max_action_tokens"):
        _compose(
            ["reliquary_telecom_solo_v1"],
            overrides={"reliquary_telecom_solo_v1": {"episode.max_action_tokens": 49153}},
        )
    # Equal is a single-turn episode's whole budget, and allowed.
    _compose(
        ["reliquary_telecom_solo_v1"],
        overrides={"reliquary_telecom_solo_v1": {"episode.max_action_tokens": 49152}},
    )
