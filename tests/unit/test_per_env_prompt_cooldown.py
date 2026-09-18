"""A curated corpus needs its own cooldown horizon.

The two cooldown maps were already per environment; only the horizon was
global, and it was sized for OpenMathInstruct's 14M prompts. At eight prompts
per window a 2,285-task environment is spent in three days and then serves
nothing, so an environment that knows its corpus is small declares it.
"""

from reliquary.constants import (
    BATCH_PROMPT_COOLDOWN_WINDOWS,
    prompt_cooldown_windows_for_environment,
)
from reliquary.protocol.profiles import (
    EnvironmentProfile,
    ProtocolProfile,
    SamplingProfile,
)

import pytest


def _profile(**environment_kwargs) -> ProtocolProfile:
    return ProtocolProfile(
        profile_id="test-cooldown",
        model_id="test/model",
        model_revision="0" * 40,
        protocol_version=9,
        collection_seconds=100,
        upload_grace_seconds=33,
        prompt_encoding="chat_template",
        sampling=SamplingProfile(
            rollouts=16, temperature=1.0, top_p=1.0, top_k=0, do_sample=True
        ),
        environments={
            "small": EnvironmentProfile(
                max_new_tokens=8192, bft=None, **environment_kwargs
            ),
        },
    )


def test_an_environment_without_a_declaration_keeps_the_global_horizon() -> None:
    """Every historical profile omits the field, so none of them may move."""
    assert prompt_cooldown_windows_for_environment("openmathinstruct") == (
        BATCH_PROMPT_COOLDOWN_WINDOWS
    )
    assert prompt_cooldown_windows_for_environment("not-an-environment") == (
        BATCH_PROMPT_COOLDOWN_WINDOWS
    )


def test_a_declared_horizon_is_the_one_used(monkeypatch) -> None:
    profile = _profile(prompt_cooldown_windows=286)
    monkeypatch.setattr(
        "reliquary.constants.ACTIVE_PROTOCOL_PROFILE", profile, raising=False
    )
    assert prompt_cooldown_windows_for_environment("small") == 286


def test_the_horizon_reaches_the_generation_contract() -> None:
    """Declared, not derived: a length read from an installed wheel is not
    something every validator can be made to agree on, so the value travels in
    the contract like `batch_target` does."""
    contract = _profile(prompt_cooldown_windows=286).to_generation_contract()
    assert contract["environments"]["small"]["prompt_cooldown_windows"] == 286


def test_an_undeclared_horizon_leaves_the_contract_untouched() -> None:
    contract = _profile().to_generation_contract()
    assert "prompt_cooldown_windows" not in contract["environments"]["small"]


def test_a_non_positive_horizon_is_refused() -> None:
    """Zero would mean a prompt returns in the very next window, which is the
    opposite of what a cooldown is for."""
    with pytest.raises(ValueError, match="prompt_cooldown_windows"):
        _profile(prompt_cooldown_windows=0)
