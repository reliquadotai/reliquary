"""Every environment's generation ceiling must be reachable inside the window.

A miner generates its rollouts in ONE batch, so a group's wall time is set by
its LONGEST rollout. If the ceiling cannot be produced within the collection
window, the whole submission misses the deadline — the miner loses all of its
rollouts, not just the long one. A ceiling the window cannot serve is therefore
worse than a lower ceiling: a lower one truncates while there is still time to
submit, and the group is admitted with a truncated rollout instead of vanishing.
"""

import pytest

from reliquary.protocol.profiles import PROFILES

# Assumption, not a measurement: the live 2B fleet ran ~130 tok/s/seq and a 4B
# decodes roughly half as fast. Stated as a constant so it is reviewable — raise
# it only against observed 4B throughput, never to make a ceiling fit.
EXPECTED_FLEET_TOKENS_PER_SECOND = 65

V3 = "qwen35-4b-auction-v3"


def _ceiling(env_profile) -> int:
    """Tokens a rollout can actually generate: BFT binds before max_new_tokens
    where it applies, so reading max_new_tokens alone overstates Math."""
    if env_profile.bft is not None:
        return min(
            env_profile.max_new_tokens,
            env_profile.bft.thinking_budget + env_profile.bft.answer_budget,
        )
    return env_profile.max_new_tokens


@pytest.mark.parametrize("environment", sorted(PROFILES[V3].environments))
def test_v3_ceiling_is_reachable_within_the_collection_window(environment):
    profile = PROFILES[V3]
    ceiling = _ceiling(profile.environments[environment])
    required = ceiling / profile.collection_seconds

    assert required <= EXPECTED_FLEET_TOKENS_PER_SECOND, (
        f"{environment}: a {ceiling}-token ceiling in "
        f"{profile.collection_seconds}s demands {required:.0f} tok/s/seq, above "
        f"the expected {EXPECTED_FLEET_TOKENS_PER_SECOND}. A group that reaches "
        "it misses the deadline entirely and loses every rollout. Either lower "
        "the ceiling or lengthen the window — do not raise the assumed rate to "
        "make this pass."
    )


def test_v2_ceiling_debt_is_recorded_not_silently_accepted():
    """The deployed v2 profile does NOT satisfy the coupling: its Code ceiling
    is 32768 in a 100s window, demanding ~328 tok/s/seq. It never bit because
    the 2B seldom generated that long — the 4B is what makes a nominal ceiling
    real. Asserted as known debt so v2 is not mistaken for compliant, and so
    this test does not fail on a profile already in production.
    """
    profile = PROFILES["qwen35-2b-auction-v2"]
    code = _ceiling(profile.environments["opencodeinstruct"])
    required = code / profile.collection_seconds
    assert required > EXPECTED_FLEET_TOKENS_PER_SECOND    # still unreachable
    assert code == 32768 and profile.collection_seconds == 100


def test_lowering_the_ceiling_protects_rather_than_restricts():
    """The v3 Code ceiling must leave real margin, not sit exactly at the edge:
    the window has to absorb the ceiling plus scheduling and upload slack."""
    profile = PROFILES[V3]
    code = _ceiling(profile.environments["opencodeinstruct"])
    seconds_needed = code / EXPECTED_FLEET_TOKENS_PER_SECOND
    assert seconds_needed <= profile.collection_seconds * 0.9
