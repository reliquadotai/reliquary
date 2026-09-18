"""An external environment may choose how its answer is read, not what its
rewards are.

The boundary used to admit exactly one contract — `answer-json/v1` — which was
the shape of the only external environment that existed when it was written.
That is a review state, not a safety property: the episode adapter next to it
is exempt from all of it, which a real invariant would not be.

What is a safety property is the binary lattice. The reward a wheel returns is
checked against `attainable_rewards` at grading time, so with two values the
check is total; and an uncertain rollout is priced under every value the
lattice allows, so two values keep that enumeration cheap.
"""

import pytest

from reliquary.environment.registry import (
    EXTERNAL_SINGLE_TURN_CONTRACTS,
    EnvironmentSpec,
)


def _spec(**overrides) -> EnvironmentSpec:
    base = dict(
        name="external_probe_v1",
        factory_path="reliquary_logic:LogicEnvironment",
        scorer_path="reliquary.environment.agentic.external:score_external_answers",
        validator_authoritative_reward=True,
        admission_resource_class="cpu",
        termination_policy="eos_or_cap",
        final_answer_policy="json",
        reward_lattice_policy="binary-v1",
        attainable_rewards=(0.0, 1.0),
        contract_version="reliquary/answer-json/v1",
        environment_manifest_sha256="0" * 64,
        external_distribution="reliquary-logic",
        external_artifact_resource="reliquary_logic/artifact.json",
    )
    base.update(overrides)
    return EnvironmentSpec(**base)


@pytest.mark.parametrize("contract", sorted(EXTERNAL_SINGLE_TURN_CONTRACTS))
def test_every_qualified_contract_is_accepted(contract: str) -> None:
    assert _spec(contract_version=contract).contract_version == contract


@pytest.mark.parametrize(
    "policy", ["boxed", "fenced_python", "json"]
)
def test_how_the_answer_is_read_is_the_environment_s_business(policy) -> None:
    """`final_answer_policy` only ever switches the boxed-integrity check on.
    Dictating `json` forced a maths environment to declare something untrue
    about itself and gave up that check for nothing."""
    assert _spec(final_answer_policy=policy).final_answer_policy == policy


def test_an_unreviewed_contract_is_refused() -> None:
    with pytest.raises(ValueError, match="not one of"):
        _spec(contract_version="reliquary/something-new/v1")


def test_fractional_rewards_are_still_outside() -> None:
    """Not an oversight. A fractional lattice would have to come from the wheel
    and travel through the materials relay, which is a wider change than a
    name — so it stays refused until that exists."""
    with pytest.raises(ValueError, match="binary rewards"):
        _spec(
            contract_version="reliquary/boxed-answer/v1",
            reward_lattice_policy="fractional-by-case-count-v1",
            attainable_rewards=(),
        )


def test_a_reward_the_validator_does_not_own_is_refused() -> None:
    with pytest.raises(ValueError, match="binary rewards"):
        _spec(validator_authoritative_reward=False)


def test_a_wider_lattice_cannot_be_declared_binary() -> None:
    with pytest.raises(ValueError, match="binary rewards"):
        _spec(attainable_rewards=(0.0, 0.5, 1.0))
