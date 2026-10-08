"""Decision H: an uncertain rollout no longer sinks a training group; the robust rule decides.

"Uncertain" = cut by the length cap (every environment), or properly terminated with no
``\\boxed`` where the contract makes a missing box uncertain (maths). Whatever a miner makes
uncertain, it can only lose admission compared with the true reward being known.
"""
from itertools import combinations, product
import random
from types import SimpleNamespace

import pytest

from reliquary.constants import M_ROLLOUTS
from reliquary.services import admission_policy as policy
from reliquary.services.admission_policy import (
    ExplorationEntitlement, default_missing_box, exploration_pay_entitlement, missing_box_is_uncertain,
    service_signal_admits, uncertain_rollout_indices,
)
from reliquary.services.scoring import classify_signal
from tests.unit.service_v2_fixtures import CODE, MATH, SCIENCE, contract_v2, contract_v2_dict

M = M_ROLLOUTS
BINARY = (0.0, 1.0)


def req(purpose="training"):
    return SimpleNamespace(service_binding={"purpose": purpose}, rollouts=[None] * M)


def contract(sigma_min_bps=2400):
    value = contract_v2_dict()
    value["scoring"]["sigma_min_bps"] = sigma_min_bps
    from reliquary.protocol.service_contract import ServiceContract
    return ServiceContract.from_dict(value)


def known_in_zone(rewards, sigma_min_bps):
    """The rule for a fully known vector, written independently of the code under test."""
    return classify_signal([round(r * 10000) / 10000 for r in rewards], expected=M,
                           sigma_min_bps=sigma_min_bps).in_zone


def completions(rewards, uncertain, lattice):
    for assignment in product(lattice, repeat=len(uncertain)):
        outcome = list(rewards)
        for index, value in zip(uncertain, assignment):
            outcome[index] = value
        yield outcome


def vector(ones):
    return [1.0] * ones + [0.0] * (M - ones)


# -- what is uncertain -------------------------------------------------------------------

def test_missing_box_is_uncertain_for_math_and_a_plain_zero_for_science_and_code():
    c = contract_v2(envs=(MATH, SCIENCE, CODE), shares={MATH: 4000, SCIENCE: 3000, CODE: 3000})
    assert missing_box_is_uncertain(MATH, c) is True
    assert missing_box_is_uncertain(SCIENCE, c) is False
    assert missing_box_is_uncertain(CODE, c) is False


def test_the_default_comes_from_the_environment_type():
    assert default_missing_box("reliquary_dapo_math_v1") == "uncertain"
    assert default_missing_box("reliquary_hard_math_v1") == "uncertain"
    assert default_missing_box("openmathinstruct") == "uncertain"
    assert default_missing_box(SCIENCE) == "graded"
    assert default_missing_box(CODE) == "graded"
    assert default_missing_box("an_env_added_later") == "graded"


def test_a_contract_cannot_make_a_box_matter_where_the_environment_has_none():
    c = contract_v2(envs=(MATH, CODE), missing_box="uncertain")
    assert c.environment(CODE)["missing_box"] == "uncertain"
    assert missing_box_is_uncertain(CODE, c) is False
    assert missing_box_is_uncertain("not_a_registered_env", c) is False


def test_length_capped_rollouts_are_uncertain_whatever_the_box_policy():
    # A science or code group has no unboxed index to offer; its capped rollouts still count.
    assert uncertain_rollout_indices(truncated_indices=(5, 2), unboxed_indices=(), size=M) == (5, 2)
    assert uncertain_rollout_indices(truncated_indices=(5,), unboxed_indices=(5, 1), size=M) == (5, 1)
    assert uncertain_rollout_indices(size=M) == ()
    with pytest.raises(ValueError):
        uncertain_rollout_indices(truncated_indices=(M,), size=M)
    with pytest.raises(ValueError):
        uncertain_rollout_indices(unboxed_indices=(True,), size=M)


# -- training lane -----------------------------------------------------------------------

@pytest.mark.parametrize("sigma_min_bps", [2400, 3400, 4500])
def test_math_group_with_missing_boxes_is_admitted_exactly_per_the_robust_rule(sigma_min_bps):
    """k successes, u terminated rollouts with no box (graded 0): admitted iff the group is in
    zone for every number of successes from k to k + u."""
    c, admitted, refused = contract(sigma_min_bps), 0, 0
    for ones in range(M + 1):
        for unboxed in range(M - ones + 1):
            rewards = vector(ones)
            uncertain = tuple(range(ones, ones + unboxed))
            expected = all(known_in_zone(vector(n), sigma_min_bps) for n in range(ones, ones + unboxed + 1))
            got = service_signal_admits(req(), c, rewards, uncertain_indices=uncertain, attainable_rewards=BINARY)
            assert got is expected, (ones, unboxed)
            admitted += got
            refused += not got
    assert admitted and refused


def test_the_boundary_at_the_default_threshold():
    c = contract(2400)
    top = max(n for n in range(M + 1) if known_in_zone(vector(n), 2400))  # most successes still in zone
    assert top == M - 1
    half = M // 2
    # half right, every failure but one unboxed: still in zone if they had all been right
    reach = tuple(range(half, half + (top - half)))
    assert service_signal_admits(req(), c, vector(half), uncertain_indices=reach, attainable_rewards=BINARY)
    # one more unboxed failure and the group could be uniform: refused
    over = tuple(range(half, M))
    assert not service_signal_admits(req(), c, vector(half), uncertain_indices=over, attainable_rewards=BINARY)


def test_one_uncertain_rollout_no_longer_rejects_a_training_group():
    assert service_signal_admits(req(), contract(), vector(M // 2), uncertain_indices=(0,), attainable_rewards=BINARY)
    assert service_signal_admits(req(), contract(), vector(M // 2), uncertain_indices=(M - 1,),
                                 attainable_rewards=BINARY)


def test_uncertainty_that_could_collapse_the_signal_is_still_refused():
    # the only success is uncertain: it may have been a failure, i.e. a uniform group
    assert not service_signal_admits(req(), contract(), vector(1), uncertain_indices=(0,), attainable_rewards=BINARY)
    # the only failure is uncertain: it may have been a success
    assert not service_signal_admits(req(), contract(), vector(M - 1), uncertain_indices=(M - 1,),
                                     attainable_rewards=BINARY)


def test_the_same_vector_in_a_science_env_has_no_uncertainty():
    """15/16 with the failure unboxed. Maths: the failure is uncertain, the group could be
    uniform, refused. Science: the missing box is a plain 0, the group is in zone."""
    c = contract_v2(envs=(MATH, SCIENCE), shares={MATH: 5000, SCIENCE: 5000})
    rewards, unboxed = vector(M - 1), (M - 1,)
    seen = {}
    for env in (MATH, SCIENCE):
        uncertain = uncertain_rollout_indices(
            unboxed_indices=unboxed if missing_box_is_uncertain(env, c) else (), size=M)
        seen[env] = (uncertain, service_signal_admits(req(), c, rewards, uncertain_indices=uncertain,
                                                      attainable_rewards=BINARY))
    assert seen == {MATH: ((M - 1,), False), SCIENCE: ((), True)}
    # ... and a length-capped rollout is uncertain in science too
    capped = uncertain_rollout_indices(truncated_indices=(M - 1,), size=M)
    assert not service_signal_admits(req(), c, rewards, uncertain_indices=capped, attainable_rewards=BINARY)


def test_an_unknown_lattice_proves_nothing():
    """Defaulting to {0, 1} would be unsound on a fractional environment: this group is in
    zone if the uncertain rollout scored 0 or 1, and out of zone if it scored 0.5."""
    high, low = M // 2, M - M // 2 - 1
    rewards = [0.74] * high + [0.26] * low + [0.0]
    c, uncertain = contract(2400), (M - 1,)
    assert all(known_in_zone(o, 2400) for o in completions(rewards, uncertain, BINARY))
    assert not known_in_zone(rewards[:-1] + [0.5], 2400)
    assert service_signal_admits(req(), c, rewards)                      # nothing uncertain: plain rule
    assert not service_signal_admits(req(), c, rewards, uncertain_indices=uncertain)   # no lattice given
    assert not service_signal_admits(req(), c, rewards, uncertain_indices=uncertain, attainable_rewards=())
    fine = tuple(n / 50 for n in range(51))
    assert not service_signal_admits(req(), c, rewards, uncertain_indices=uncertain, attainable_rewards=fine)
    assert not service_signal_admits(req(), c, rewards, uncertain_indices=uncertain, attainable_rewards=(0.0, 7.0))


def test_a_bad_index_refuses_instead_of_admitting():
    c = contract()
    for bad in ((M,), (-1,), (True,), ("0",)):
        assert not service_signal_admits(req(), c, vector(M // 2), uncertain_indices=bad, attainable_rewards=BINARY)


# -- what a miner can gain by making rollouts uncertain: nothing ---------------------------

@pytest.mark.parametrize("sigma_min_bps", [2400, 3400, 4500])
def test_binary_uncertainty_can_only_cost_admission_exhaustive(sigma_min_bps):
    """Every binary vector shape, every choice of uncertain rollouts: if the group is admitted
    with the uncertainty, it is admitted for every reward those rollouts could really have had."""
    c, admitted = contract(sigma_min_bps), 0
    for ones in range(M + 1):
        rewards = vector(ones)
        for from_ones in range(ones + 1):
            for from_zeros in range(M - ones + 1):
                uncertain = tuple(range(from_ones)) + tuple(range(ones, ones + from_zeros))
                if not service_signal_admits(req(), c, rewards, uncertain_indices=uncertain,
                                             attainable_rewards=BINARY):
                    continue
                admitted += 1
                # symmetric in the rollouts: one completion per number of uncertain successes
                for wins in range(len(uncertain) + 1):
                    outcome = list(rewards)
                    for rank, index in enumerate(uncertain):
                        outcome[index] = 1.0 if rank < wins else 0.0
                    assert service_signal_admits(req(), c, outcome), (rewards, uncertain, outcome)
                    assert known_in_zone(outcome, sigma_min_bps)
    assert admitted


@pytest.mark.parametrize("lattice", [(0.0, 0.5, 1.0), (0.0, 0.25, 0.5, 0.75, 1.0), tuple(n / 3 for n in range(4))])
def test_fractional_uncertainty_can_only_cost_admission(lattice):
    rng = random.Random(20261008)
    admitted = refused = 0
    for _ in range(400):
        sigma_min_bps = rng.choice([1500, 2400, 3400])
        c = contract(sigma_min_bps)
        rewards = [rng.choice(lattice) for _ in range(M)]
        uncertain = tuple(rng.sample(range(M), rng.randint(1, 3)))
        if not service_signal_admits(req(), c, rewards, uncertain_indices=uncertain, attainable_rewards=lattice):
            refused += 1
            continue
        admitted += 1
        for outcome in completions(rewards, uncertain, lattice):
            assert service_signal_admits(req(), c, outcome), (rewards, uncertain, outcome)
    assert admitted > 20 and refused > 20


def test_more_uncertainty_never_admits_more():
    c = contract(2400)
    for ones in range(M + 1):
        rewards = vector(ones)
        for size in range(1, 4):
            for uncertain in combinations(range(M), size):
                if service_signal_admits(req(), c, rewards, uncertain_indices=uncertain, attainable_rewards=BINARY):
                    assert service_signal_admits(req(), c, rewards)
                    for smaller in combinations(uncertain, size - 1):
                        assert service_signal_admits(req(), c, rewards, uncertain_indices=smaller,
                                                     attainable_rewards=BINARY)


def test_cutting_failures_cannot_manufacture_a_signal():
    """A uniform group stays refused whichever rollouts the miner lets run to the cap."""
    c = contract(2400)
    for rewards in (vector(0), vector(M), [0.5] * M):
        for size in range(M + 1):
            assert not service_signal_admits(req(), c, rewards, uncertain_indices=tuple(range(size)),
                                             attainable_rewards=(0.0, 0.5, 1.0))


def test_the_named_rule_is_the_one_consulted(monkeypatch):
    import reliquary.validator.admission as admission
    calls = []

    def spy(rewards, *, sigma_min, truncated_indices, attainable_rewards):
        calls.append((list(rewards), sigma_min, tuple(truncated_indices), tuple(attainable_rewards)))
        return False

    monkeypatch.setattr(admission, "robust_utility_admits", spy)
    assert not service_signal_admits(req(), contract(3400), vector(M // 2), uncertain_indices=(1, 1, 0),
                                     attainable_rewards=BINARY)
    assert calls == [(vector(M // 2), 0.34, (1, 0), BINARY)]
    assert service_signal_admits(req(), contract(3400), vector(M // 2))     # nothing uncertain: not consulted
    assert len(calls) == 1


# -- exploration lane --------------------------------------------------------------------

def test_exploration_observation_is_not_refused_for_an_uncertain_rollout():
    c = contract()
    assert service_signal_admits(req("exploration"), c, vector(0))
    assert service_signal_admits(req("exploration"), c, vector(0), uncertain_indices=(3,), attainable_rewards=BINARY)
    assert service_signal_admits(req("exploration"), c, vector(0), uncertain_indices=(3,))
    assert not service_signal_admits(req("exploration"), c, vector(M // 2), uncertain_indices=(3,),
                                     attainable_rewards=BINARY)


def test_exploration_all_terminated_including_missing_boxes_is_entitled_with_zeros_recorded():
    rewards = [0.0] * M
    got = exploration_pay_entitlement(rewards, truncated_indices=(), uncertain_indices=(1, 4))
    assert got == ExplorationEntitlement(True, None, tuple([0.0] * M))
    # the recorded reward of a terminated rollout with no box is 0 whatever was carried in
    carried = [1.0] * M
    got = exploration_pay_entitlement(carried, uncertain_indices=(1, 4))
    assert got.entitled and got.reason is None
    assert got.rewards == tuple(0.0 if i in (1, 4) else 1.0 for i in range(M))
    assert exploration_pay_entitlement(carried) == ExplorationEntitlement(True, None, tuple(carried))


def test_exploration_with_one_length_capped_rollout_is_unpaid_truncated():
    rewards = [0.0] * M
    got = exploration_pay_entitlement(rewards, truncated_indices=(M - 1,), uncertain_indices=(M - 1, 2))
    assert got == ExplorationEntitlement(False, "truncated", tuple(rewards))
    assert policy.EXPLORATION_TRUNCATED == "truncated"
    # the capped rollout keeps its graded reward; only the terminated one with no box is forced to 0
    graded = [1.0] * M
    got = exploration_pay_entitlement(graded, truncated_indices=(0,), uncertain_indices=(0, 2))
    assert (got.entitled, got.reason) == (False, "truncated")
    assert got.rewards == tuple(0.0 if i == 2 else 1.0 for i in range(M))
    # a capped rollout the caller forgot to list as uncertain still blocks pay
    assert not exploration_pay_entitlement(rewards, truncated_indices=(3,)).entitled


def test_exploration_entitlement_refuses_an_index_outside_the_group():
    with pytest.raises(ValueError):
        exploration_pay_entitlement([0.0] * M, truncated_indices=(M,))
    with pytest.raises(ValueError):
        exploration_pay_entitlement([0.0] * M, uncertain_indices=(-1,))
