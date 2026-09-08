"""EnvScaler is installed, inert, and cannot be half-activated.

Registration and activation are separate layers here, and this environment
sits deliberately in the gap: the registry knows it so a measurement can
address it by name, while no profile names it, so no window can draw it.

The corpus is third-party and fetched rather than vendored, and the loader
reads it lazily. Without a guard the validator would boot happily, win
prompts, and only then discover the directory is missing — which is what
these tests exist to prevent.
"""

from __future__ import annotations

import pytest

from reliquary.environment.registry import (
    EnvironmentSpec,
    get_environment_spec,
)
from reliquary.protocol.profiles import PROFILES


ENVIRONMENT = "envscaler_tools_v1"
DATA_VARIABLE = "RELIQUARY_ENVSCALER_DATA"


def test_the_environment_is_registered() -> None:
    spec = get_environment_spec(ENVIRONMENT)

    assert spec.interaction_mode == "episode"
    assert spec.required_data_env_var == DATA_VARIABLE


def test_no_profile_draws_it() -> None:
    """Installed is not activated. A profile naming it is a deliberate act."""
    naming = [
        profile.profile_id
        for profile in PROFILES.values()
        if ENVIRONMENT in profile.environments
    ]

    assert naming == []


def test_the_worlds_run_third_party_source() -> None:
    """The corpus carries LLM-written Python that a rollout `exec`s."""
    assert get_environment_spec(ENVIRONMENT).admission_resource_class == "sandbox"


def test_the_reward_lattice_is_not_enumerated() -> None:
    """Reward is passed checks over total, and the denominator varies."""
    spec = get_environment_spec(ENVIRONMENT)

    assert spec.attainable_rewards == ()
    assert spec.reward_lattice_policy == "fractional-by-check-count-v1"


def test_creating_it_without_its_corpus_is_refused(monkeypatch) -> None:
    """Lazily, this would raise on the first task instead — mid-window."""
    monkeypatch.delenv(DATA_VARIABLE, raising=False)

    with pytest.raises(RuntimeError, match=DATA_VARIABLE):
        get_environment_spec(ENVIRONMENT).create()


def test_a_profile_naming_it_unset_refuses_to_boot(monkeypatch) -> None:
    from reliquary import constants

    monkeypatch.delenv(DATA_VARIABLE, raising=False)
    monkeypatch.setattr(constants, "ENVIRONMENT_MIX", [(ENVIRONMENT, 8)])

    with pytest.raises(ValueError, match=DATA_VARIABLE):
        constants._require_environment_corpora()


def test_the_guard_passes_once_the_corpus_is_configured(monkeypatch) -> None:
    from reliquary import constants

    monkeypatch.setenv(DATA_VARIABLE, "/anywhere")
    monkeypatch.setattr(constants, "ENVIRONMENT_MIX", [(ENVIRONMENT, 8)])

    constants._require_environment_corpora()


def test_the_guard_ignores_environments_that_vendor_their_corpus(
    monkeypatch,
) -> None:
    """Only an environment that declares the field is gated by it."""
    from reliquary import constants

    monkeypatch.delenv(DATA_VARIABLE, raising=False)
    monkeypatch.setattr(
        constants, "ENVIRONMENT_MIX", [("openmathinstruct", 8)]
    )

    constants._require_environment_corpora()


def test_an_unknown_name_is_left_to_the_environment_list(monkeypatch) -> None:
    """This guard is about corpora; unknown names fail their own way."""
    from reliquary import constants

    monkeypatch.setattr(constants, "ENVIRONMENT_MIX", [("nope", 8)])

    constants._require_environment_corpora()


def test_the_field_defaults_to_absent() -> None:
    """Existing environments must not become gated by adding the field."""
    assert EnvironmentSpec.__dataclass_fields__[
        "required_data_env_var"
    ].default is None
