from __future__ import annotations

import dataclasses
import inspect

import pytest

import reliquary.protocol.release_contract as release_contract_module
from reliquary.protocol.profiles import to_generation_contract
from reliquary.protocol.release_contract import (
    CAP_FRESH_POST_SEAL_ORDERING,
    CAP_TICKETED_PACED_EPOCH,
    ContractComponent,
    EnvironmentContract,
    RELIQUARY_1_CAPABILITIES,
    ReleaseContract,
    ReleaseContractError,
    canonical_json_bytes,
    parse_release_contract,
    release_contract_to_dict,
)


def _component(name: str) -> ContractComponent:
    return ContractComponent.bind(
        f"{name}.contract/v1",
        {"domain": name, "schema": f"{name}/v1"},
    )


def _release() -> ReleaseContract:
    return ReleaseContract(
        release_id="reliquary-1/rc1",
        wire=_component("wire"),
        generation=ContractComponent.bind(
            "generation.qwen3-4b-reasoning/v5",
            to_generation_contract("qwen3-4b-base-dapo-reasoning-v5"),
        ),
        market=_component("market"),
        verification=_component("verification"),
        training=_component("training"),
        environments=(
            EnvironmentContract(
                "opencodeinstruct",
                _component("environment.opencodeinstruct"),
            ),
            EnvironmentContract(
                "openmathinstruct",
                _component("environment.openmathinstruct"),
            ),
        ),
        capabilities=RELIQUARY_1_CAPABILITIES,
    )


def test_release_contract_round_trip_and_hash_vector_are_stable() -> None:
    release = _release()

    assert parse_release_contract(release.to_bytes()) == release
    assert release.canonical_sha256 == (
        "da5def253914c9f8b926dc24aefdab84d538e5f009a1dca905800ccc6bb0d11e"
    )


@pytest.mark.parametrize(
    ("profile_id", "expected_sha256"),
    (
        (
            "qwen3-4b-base-dapo-v4",
            "e3098b00582d395fc7176ff835c988588e29a530343942c081781f1bab65de91",
        ),
        (
            "qwen3-4b-base-dapo-reasoning-v5",
            "19e98f5a3ddac1980efe66fd80db1ec0f8db87a5e60934efd5d0e8985435eadd",
        ),
    ),
)
def test_existing_generation_contract_hashes_are_unchanged(
    profile_id: str,
    expected_sha256: str,
) -> None:
    component = ContractComponent.bind(
        f"generation.compatibility/{profile_id}",
        to_generation_contract(profile_id),
    )

    assert component.canonical_sha256 == expected_sha256


def test_components_are_independently_bound() -> None:
    release = _release()
    changed_market = ContractComponent.bind(
        "market.contract/v1",
        {"domain": "market", "schema": "market/v1", "target": 17},
    )
    changed = dataclasses.replace(release, market=changed_market)

    assert changed.market.canonical_sha256 != release.market.canonical_sha256
    assert changed.generation == release.generation
    assert changed.wire == release.wire
    assert changed.canonical_sha256 != release.canonical_sha256

    assert release.market.matches({"domain": "market", "schema": "market/v1"})
    assert not release.market.matches(
        {"domain": "market", "schema": "market/v1", "target": 17}
    )


def test_component_ids_are_globally_unique() -> None:
    release = _release()

    with pytest.raises(ReleaseContractError, match="globally unique"):
        dataclasses.replace(release, market=release.wire)


def test_capabilities_replace_numeric_feature_dispatch() -> None:
    release = _release()

    assert release.supports(CAP_TICKETED_PACED_EPOCH)
    assert release.supports(CAP_FRESH_POST_SEAL_ORDERING)
    assert not release.supports("market.fill-closed-rate/v1")

    source = inspect.getsource(release_contract_module)
    assert "protocol_version" not in source
    assert ">=" not in source


def test_parser_rejects_noncanonical_bytes() -> None:
    raw = _release().to_bytes()

    with pytest.raises(ReleaseContractError, match="not canonical"):
        parse_release_contract(b"\n" + raw)


def test_parser_rejects_unknown_or_defaulted_fields() -> None:
    value = release_contract_to_dict(_release())
    value["protocol_version"] = 7

    with pytest.raises(ReleaseContractError, match="unknown"):
        parse_release_contract(canonical_json_bytes(value))

    del value["protocol_version"]
    del value["components"]["verification"]
    with pytest.raises(ReleaseContractError, match="missing"):
        parse_release_contract(canonical_json_bytes(value))


def test_parser_rejects_duplicate_json_keys() -> None:
    raw = b'{"schema":"one","schema":"two"}'

    with pytest.raises(ReleaseContractError, match="duplicate JSON key"):
        parse_release_contract(raw)


def test_parser_rejects_unsorted_capabilities_and_environments() -> None:
    value = release_contract_to_dict(_release())
    value["capabilities"]["values"].reverse()
    with pytest.raises(ReleaseContractError, match="capabilities must be sorted"):
        parse_release_contract(canonical_json_bytes(value))

    value = release_contract_to_dict(_release())
    value["components"]["environments"].reverse()
    with pytest.raises(ReleaseContractError, match="environments must be sorted"):
        parse_release_contract(canonical_json_bytes(value))


@pytest.mark.parametrize(
    "bad_hash",
    (
        "0" * 63,
        "A" * 64,
        "z" * 64,
    ),
)
def test_component_hash_is_exact_lowercase_sha256(bad_hash: str) -> None:
    with pytest.raises(ReleaseContractError, match="lowercase SHA-256"):
        ContractComponent("wire.contract/v1", bad_hash)
