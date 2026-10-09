"""The environment catalog is the compiled profiles' bodies, restated.

Editing a default changes every task declared afterwards, so each body is
pinned by digest and traced to the profile it was taken from.
"""

import pytest

from reliquary.environment.registry import ENVIRONMENT_SPECS
from reliquary.protocol.composition import compose_profile, model_spec_of, run_policy_of
from reliquary.protocol.environment_catalog import (
    CATALOG_PROVENANCE,
    ENVIRONMENT_CATALOG,
    TUNABLE_FIELDS,
)
from reliquary.protocol.profiles import PROFILES, ProtocolProfile
from reliquary.protocol.release_contract import canonical_sha256

V1 = "qwen3-4b-base-dapo-reliquary-v1"
TEUTONIC = "teutonic-9b-reliquary-suite-v9-dev1"

EXPECTED_PROVENANCE = {
    "openmathinstruct": V1,
    "opencodeinstruct": V1,
    "reliquary_logic_v2": V1,
    "reliquaryverifiable_v1": "qwen3-4b-reliquary-verifiable-v6-dev1",
    "reliquary_stateful_tools_v1": "qwen3-4b-reliquary-episode-v7-dev1",
    "reliquary_retrieval_tools_v1": "qwen3-4b-reliquary-episode-v7-dev1",
    "reliquary_workspace_tools_v1": "qwen3-4b-reliquary-episode-v7-dev1",
    "reliquarylogic_v1": "qwen3-4b-reliquary-logic-v8-dev1",
    "reliquary_dapo_math_v1": TEUTONIC,
    "reliquary_science_v1": TEUTONIC,
    "reliquary_hard_math_v1": TEUTONIC,
    "reliquary_instruction_following_v1": TEUTONIC,
    "reliquary_code_v1": TEUTONIC,
    "reliquary_competitive_code_v1": TEUTONIC,
    "reliquary_telecom_solo_v1": TEUTONIC,
}

# Canonical sha256 of each body as the contract writes it.
GOLDEN_BODY_SHA256 = {
    "openmathinstruct": "d575a401f626508e53c85ad3defd9501a18535f7bdab5068f587eebe07ce3a7f",
    "opencodeinstruct": "a16862cc515d58c645df6b1a593044bc4afd21ae70fcb2ac7a3eeac9384bd4bb",
    "reliquary_logic_v2": "43dc3b45a44c3c9b57495e6f8d24fcd967cf5b2737f3c20a3643be58b35fe664",
    "reliquaryverifiable_v1": "e5bf11ffdc68cadd269e48100b48df86b54b4317e7726a9f145ab6dd63a8b977",
    "reliquary_stateful_tools_v1": "3b4c195afeec0553641229475b2b9dd77ff69852f740b78db031df9bba51ce1f",
    "reliquary_retrieval_tools_v1": "5b968c03f906cc5b28d60d23560ac2ae52f56a34b52aed13a933085936e0dc1d",
    "reliquary_workspace_tools_v1": "c30e10b6c29c41799547887e5336f8a1f3ed6654edc74da43f9ef5896d55ec1d",
    "reliquarylogic_v1": "effc1b3a997f9acda707d9582cdfd9e3ebdd72b50d4415564dd8de8ecb0ec520",
    "reliquary_dapo_math_v1": "072af5acf59f88ff6308b857001844d1cdb0d5d0ca26d97a5e1dbd7b2af775a7",
    "reliquary_science_v1": "4a96faeb8130291002ec1679aa7790e9c2d7fbeeed3b1a07b6a68e8bcef528c8",
    "reliquary_hard_math_v1": "45997d290e97777a0cec48f13acdd1dbee52bff41e3049d12cc02314092535ba",
    "reliquary_instruction_following_v1": "6d6040473b809d87367198dec14eae5d72232b8c2ff3a773b81b7b51c1f0fbab",
    "reliquary_code_v1": "0724aa4fa09254d27547d757d8b77a7a0fbe6d3a78f8c9522be0ed23a27b4509",
    "reliquary_competitive_code_v1": "f6a580bf4b865bf59c7166dc526256ea3434bfb849a0f9bb84632e192485a56f",
    "reliquary_telecom_solo_v1": "5d7ec17b0250dc935a43fdf38a6873f34053caa460546b35331dde5870f9556a",
}

COMPOSABLE_PROFILES = [
    "qwen3-4b-base-dapo-reasoning-v5",
    "qwen3-4b-base-dapo-fill-closed-v6",
    V1,
    "qwen3-4b-reliquary-verifiable-v6-dev1",
    "qwen3-4b-reliquary-episode-v7-dev1",
    "qwen3-4b-reliquary-logic-v8-dev1",
    TEUTONIC,
]
# Their OMI/OCI bodies predate prompt templates; they stay reachable only
# through --from-profile.
NOT_REPRODUCIBLE = [
    "qwen35-2b-auction-v2",
    "qwen35-4b-auction-v3",
    "qwen3-4b-base-dapo-v4",
]


def _body_sha(name, body):
    # The body as the contract writes it: serialize through a one-environment profile.
    profile = PROFILES[V1]
    single = ProtocolProfile(
        profile_id="catalog-body", model_id=profile.model_id,
        model_revision=profile.model_revision,
        protocol_version=profile.protocol_version,
        collection_seconds=profile.collection_seconds,
        upload_grace_seconds=profile.upload_grace_seconds,
        prompt_encoding=profile.prompt_encoding, sampling=profile.sampling,
        environments={name: body},
    )
    return canonical_sha256(single.to_generation_contract()["environments"][name])


def test_catalog_and_provenance_cover_the_same_environments():
    assert dict(CATALOG_PROVENANCE) == EXPECTED_PROVENANCE
    assert set(ENVIRONMENT_CATALOG) == set(EXPECTED_PROVENANCE)


def test_environments_no_profile_declares_have_no_entry():
    assert "reliquary_stateful_tools_v2" not in ENVIRONMENT_CATALOG
    assert "envscaler_tools_v1" not in ENVIRONMENT_CATALOG


@pytest.mark.parametrize("name", sorted(EXPECTED_PROVENANCE))
def test_catalog_body_is_its_source_profile_body(name):
    source = PROFILES[CATALOG_PROVENANCE[name]]
    assert ENVIRONMENT_CATALOG[name] == source.environments[name]


@pytest.mark.parametrize("name", sorted(EXPECTED_PROVENANCE))
def test_catalog_body_digest_is_pinned(name):
    assert _body_sha(name, ENVIRONMENT_CATALOG[name]) == GOLDEN_BODY_SHA256[name]


@pytest.mark.parametrize("name", sorted(EXPECTED_PROVENANCE))
def test_catalog_body_agrees_with_the_installed_spec(name):
    spec = ENVIRONMENT_SPECS[name]
    body = ENVIRONMENT_CATALOG[name]
    if body.environment_contract_id is None:
        # Legacy runtimes: the spec pins no manifest either.
        assert spec.environment_manifest_sha256 is None
    else:
        assert body.environment_contract_id == spec.contract_version
        assert body.environment_manifest_sha256 == spec.environment_manifest_sha256
    assert (body.episode is not None) == (spec.interaction_mode == "episode")
    if body.episode is not None:
        assert body.episode.renderer_id == spec.renderer_id


def test_every_catalog_body_is_composable():
    # Corpus needs a template for a single-turn source; episodes render their own.
    for name, body in ENVIRONMENT_CATALOG.items():
        assert body.prompt_template is not None or body.episode is not None, name


@pytest.mark.parametrize("profile_id", COMPOSABLE_PROFILES)
def test_profile_recomposes_from_the_catalog(profile_id):
    profile = PROFILES[profile_id]
    composed = compose_profile(
        profile_id=profile.profile_id,
        model=model_spec_of(profile),
        run=run_policy_of(profile),
        environments=sorted(profile.environments),
    )
    assert composed == profile
    assert canonical_sha256(composed.to_generation_contract()) == canonical_sha256(
        profile.to_generation_contract()
    )


@pytest.mark.parametrize("profile_id", NOT_REPRODUCIBLE)
def test_legacy_profiles_are_not_reproducible_from_the_catalog(profile_id):
    profile = PROFILES[profile_id]
    assert any(
        ENVIRONMENT_CATALOG[name] != body for name, body in profile.environments.items()
    )


def test_every_profile_is_classified():
    assert set(COMPOSABLE_PROFILES) | set(NOT_REPRODUCIBLE) == set(PROFILES)


def test_tunable_fields():
    assert TUNABLE_FIELDS == frozenset({
        "max_new_tokens", "thinking", "batch_target", "prompt_cooldown_windows",
        "episode.max_turns", "episode.max_action_tokens", "episode.max_episode_tokens",
    })
