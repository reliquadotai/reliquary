from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from types import MappingProxyType, SimpleNamespace

import pytest

from reliquary.environment.registry import (
    ENVIRONMENT_SPECS,
    EnvironmentSpec,
    _build_catalog,
    environment_catalog,
    environment_manifest,
    environment_manifest_sha256,
    get_environment_spec,
    resolve_environment_mix,
)


def test_catalog_is_immutable_and_contains_legacy_environments():
    assert isinstance(environment_catalog(), MappingProxyType)
    assert tuple(environment_catalog()) == (
        "openmathinstruct",
        "opencodeinstruct",
        "reliquaryverifiable_v1",
        "reliquarylogic_v1",
        "reliquary_stateful_tools_v1",
        "reliquary_stateful_tools_v2",
        "reliquary_retrieval_tools_v1",
        "reliquary_workspace_tools_v1",
    )
    with pytest.raises(TypeError):
        ENVIRONMENT_SPECS["new"] = ENVIRONMENT_SPECS["openmathinstruct"]
    with pytest.raises(FrozenInstanceError):
        ENVIRONMENT_SPECS["openmathinstruct"].name = "changed"


def test_duplicate_registration_fails_closed():
    spec = ENVIRONMENT_SPECS["openmathinstruct"]
    with pytest.raises(ValueError, match="duplicate environment"):
        _build_catalog((spec, spec))


def test_unknown_environment_error_preserves_legacy_prefix():
    with pytest.raises(ValueError, match=r"^Unknown environment: missing"):
        get_environment_spec("missing")


def test_environment_manifest_is_canonical_and_excludes_local_import_paths():
    manifest = environment_manifest()
    encoded = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert len(environment_manifest_sha256()) == 64
    assert "factory_path" not in encoded
    assert "scorer_path" not in encoded
    assert "admission_resource_class" not in encoded
    assert "openmathinstruct-runtime-v1" in encoded


def test_resolve_environment_mix_rejects_empty_duplicate_unknown_and_inactive():
    profiles = {
        "openmathinstruct": SimpleNamespace(batch_target=None),
        "opencodeinstruct": SimpleNamespace(batch_target=7),
    }
    with pytest.raises(ValueError, match="at least one"):
        resolve_environment_mix(
            [], profile_environments=profiles, default_batch_target=16
        )
    with pytest.raises(ValueError, match="duplicate"):
        resolve_environment_mix(
            ["openmathinstruct", "openmathinstruct"],
            profile_environments=profiles,
            default_batch_target=16,
        )
    with pytest.raises(ValueError, match="Unknown environment"):
        resolve_environment_mix(
            ["missing"],
            profile_environments=profiles,
            default_batch_target=16,
        )
    with pytest.raises(ValueError, match="not declared"):
        resolve_environment_mix(
            ["opencodeinstruct"],
            profile_environments={"openmathinstruct": profiles["openmathinstruct"]},
            default_batch_target=16,
        )


def test_resolve_environment_mix_uses_profile_target_or_legacy_default():
    profiles = {
        "openmathinstruct": SimpleNamespace(batch_target=None),
        "opencodeinstruct": SimpleNamespace(batch_target=7),
    }
    assert resolve_environment_mix(
        ["openmathinstruct", "opencodeinstruct"],
        profile_environments=profiles,
        default_batch_target=16,
    ) == [("openmathinstruct", 16), ("opencodeinstruct", 7)]


def test_resolve_environment_mix_rejects_contract_or_manifest_drift():
    spec = get_environment_spec("reliquary_stateful_tools_v1")
    base = {
        "batch_target": 16,
        "environment_contract_id": spec.contract_version,
        "environment_manifest_sha256": spec.environment_manifest_sha256,
    }
    with pytest.raises(ValueError, match="contract does not match"):
        resolve_environment_mix(
            [spec.name],
            profile_environments={
                spec.name: SimpleNamespace(
                    **{**base, "environment_contract_id": "wrong-v1"}
                )
            },
            default_batch_target=16,
        )
    with pytest.raises(ValueError, match="manifest does not match"):
        resolve_environment_mix(
            [spec.name],
            profile_environments={
                spec.name: SimpleNamespace(
                    **{**base, "environment_manifest_sha256": "0" * 64}
                )
            },
            default_batch_target=16,
        )


@pytest.mark.parametrize(
    "name",
    (
        "reliquary_stateful_tools_v1",
        "reliquary_retrieval_tools_v1",
        "reliquary_workspace_tools_v1",
    ),
)
def test_episode_specs_expose_the_binary_training_reward_lattice(name):
    spec = get_environment_spec(name)
    assert spec.reward_lattice_policy == "binary-v1"
    assert spec.attainable_rewards == (0.0, 1.0)


def test_spec_rejects_invalid_lattice_and_import_path():
    kwargs = dict(
        name="x",
        factory_path="module:factory",
        scorer_path="module:score",
        validator_authoritative_reward=True,
        admission_resource_class="cpu",
        termination_policy="eos_or_cap",
        final_answer_policy="json",
        reward_lattice_policy="binary-v1",
        attainable_rewards=(0.0, 1.0),
        contract_version="x-v1",
    )
    with pytest.raises(ValueError, match="unique and sorted"):
        EnvironmentSpec(**{**kwargs, "attainable_rewards": (1.0, 0.0)})
    with pytest.raises(ValueError, match="invalid import path"):
        EnvironmentSpec(**{**kwargs, "factory_path": "not-an-import-path"})


def test_legacy_factory_classes_are_preserved(monkeypatch):
    from reliquary.environment import load_environment
    from reliquary.environment.openmathinstruct import OpenMathInstructEnvironment
    from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment

    monkeypatch.setattr(OpenMathInstructEnvironment, "__init__", lambda self: None)
    monkeypatch.setattr(OpenCodeInstructEnvironment, "__init__", lambda self: None)
    assert isinstance(load_environment("openmathinstruct"), OpenMathInstructEnvironment)
    assert isinstance(load_environment("opencodeinstruct"), OpenCodeInstructEnvironment)


def test_reliquaryverifiable_factory_is_local_and_valid():
    from reliquary.environment import load_environment
    from reliquary.environment.reliquaryverifiable import (
        ReliquaryVerifiableEnvironment,
    )

    assert isinstance(
        load_environment("reliquaryverifiable_v1"),
        ReliquaryVerifiableEnvironment,
    )
