import copy
import hashlib

import pytest

from reliquary.constants import M_ROLLOUTS
from reliquary.protocol.release_contract import canonical_json_bytes
from reliquary.protocol.seed_pool import (
    CAPABILITY, DRAW_DOMAIN, POOL_SCHEMA, ROLLOUT_SCHEMA, SELECTION_SCHEMA,
    PoolSelection, RolloutSeed, SeedPool, SeedPoolError, parse_rollout_binding,
    pool_from_service_policy, validate_rollout_selection,
)
from reliquary.protocol.service_contract import (
    SUPPORTED_V2_CAPABILITIES, ServiceContract, ServiceContractError,
)
from reliquary.protocol.service_schedule import initial_schedule
from tests.unit.service_v2_fixtures import CODE, MATH, contract_v2, contract_v2_dict

POOL_SEEDS = 2 * M_ROLLOUTS
FIRST = tuple(range(M_ROLLOUTS))
ODD = tuple(range(1, POOL_SEEDS, 2))


def _pool(**changes):
    context = {"environment": MATH, "prompt_idx": 7, "checkpoint_hash": "d" * 40, "pool_epoch": 3,
               "randomness": "ab" * 32, **changes}
    return SeedPool.from_contract(contract_v2(), **context)


def _commits(selection):
    return [{"rollout": {"seed_pool": selection.rollout_binding(i)}} for i in range(len(selection.seeds))]


# --- identities -----------------------------------------------------------------------------

def test_wire_identities_are_the_free_subset_ones():
    assert POOL_SCHEMA == "public-seed-pool/v3" and CAPABILITY == "public-seed-pool/v3"
    assert SELECTION_SCHEMA == "public-seed-selection/v2" and ROLLOUT_SCHEMA == "public-seed-rollout/v2"
    assert DRAW_DOMAIN == b"public-seed-draw/v3"
    assert CAPABILITY in SUPPORTED_V2_CAPABILITIES
    assert "public-group-pool/v1" not in SUPPORTED_V2_CAPABILITIES


def test_pool_is_two_m_public_seeds_immutable_and_round_trips():
    pool = _pool()
    assert pool.group_size == M_ROLLOUTS and pool.pool_seeds == POOL_SEEDS
    assert set(pool.to_dict()) == {"schema", "service_contract_sha256", "environment", "prompt_idx",
                                   "checkpoint_hash", "pool_epoch", "randomness", "group_size",
                                   "pool_seeds", "renewal_windows"}
    assert SeedPool.from_dict(pool.to_dict()) == pool
    assert not hasattr(pool, "pool_groups")
    with pytest.raises(AttributeError):
        pool.pool_epoch = 5


@pytest.mark.parametrize("pool_seeds", [M_ROLLOUTS, POOL_SEEDS - 1, POOL_SEEDS + 1, 3 * M_ROLLOUTS, True, None])
def test_pool_size_other_than_two_m_is_refused(pool_seeds):
    pool = _pool()
    with pytest.raises(SeedPoolError):
        SeedPool(pool.service_contract_sha256, MATH, 7, "d" * 40, 3, "ab" * 32, M_ROLLOUTS, pool_seeds, 1)
    with pytest.raises(SeedPoolError):
        SeedPool.from_dict({**pool.to_dict(), "pool_seeds": pool_seeds})
    with pytest.raises(ServiceContractError):
        contract_v2(pool_seeds=pool_seeds)


def test_contract_declares_the_pool_and_no_group_count():
    sampling = contract_v2().environment(MATH)["sampling"]
    assert sampling == {"kind": CAPABILITY, "group_size": M_ROLLOUTS, "pool_seeds": POOL_SEEDS,
                        "renewal_windows": 1}
    old = contract_v2_dict()
    for env in old["environments"].values():
        env["sampling"] = {"kind": "public-group-pool/v1", "group_size": M_ROLLOUTS, "pool_groups": 2,
                           "renewal_windows": 1}
    with pytest.raises(ServiceContractError):
        ServiceContract.from_dict(old)
    mixed = contract_v2_dict()
    mixed["environments"][MATH]["sampling"]["pool_groups"] = 2
    with pytest.raises(ServiceContractError):
        ServiceContract.from_dict(mixed)


# --- selection: one subset, one encoding ------------------------------------------------------

def test_any_m_distinct_seeds_form_a_selection():
    pool = _pool()
    for seeds in (FIRST, ODD, tuple(range(M_ROLLOUTS, POOL_SEEDS)), (0, *range(POOL_SEEDS - M_ROLLOUTS + 1, POOL_SEEDS))):
        selection = pool.selection(seeds)
        assert selection.seeds == seeds and type(selection.seeds) is tuple
        assert selection.to_dict() == {"schema": SELECTION_SCHEMA, "pool_sha256": pool.sha256, "seeds": list(seeds)}
        assert PoolSelection.from_dict(selection.to_dict()) == selection
        validate_rollout_selection(pool, selection, _commits(selection))


@pytest.mark.parametrize("seeds", [
    (0,) * M_ROLLOUTS,                                  # duplicates
    (0, 0, *range(2, M_ROLLOUTS)),                      # one duplicate
    tuple(reversed(range(M_ROLLOUTS))),                 # unsorted
    (1, 0, *range(2, M_ROLLOUTS)),                      # one inversion
    (*range(M_ROLLOUTS - 1), POOL_SEEDS),               # out of range (high)
    (-1, *range(1, M_ROLLOUTS)),                        # out of range (low)
    tuple(range(M_ROLLOUTS - 1)),                       # too few
    tuple(range(M_ROLLOUTS + 1)),                       # too many
    tuple(range(POOL_SEEDS)),                           # the whole pool
    (False, True, *range(2, M_ROLLOUTS)),               # bools
    (0.0, *range(1, M_ROLLOUTS)),                       # non-int
    ("0", *range(1, M_ROLLOUTS)),                       # non-int
    (),
    None,
    "0123456789abcdef",
])
def test_every_other_encoding_of_a_subset_is_refused(seeds):
    pool = _pool()
    with pytest.raises(SeedPoolError):
        pool.selection(seeds)
    with pytest.raises(SeedPoolError):
        selection = PoolSelection.from_dict({"schema": SELECTION_SCHEMA, "pool_sha256": pool.sha256,
                                             "seeds": list(seeds) if isinstance(seeds, tuple) else seeds})
        pool.validate_selection(selection, rollout_count=M_ROLLOUTS)


def test_selection_wire_form_is_exact_and_bounded():
    pool = _pool()
    good = pool.selection(FIRST).to_dict()
    for bad in ({**good, "extra": 1}, {**good, "schema": "public-group-selection/v1"},
                {"schema": good["schema"], "pool_sha256": good["pool_sha256"], "candidate_id": 0},
                {**good, "seeds": tuple(FIRST)}, {**good, "pool_sha256": "AB" * 32},
                {**good, "seeds": list(range(65))}, {**good, "seeds": list(range(100000))}):
        with pytest.raises(SeedPoolError):
            PoolSelection.from_dict(bad)
    with pytest.raises(SeedPoolError):
        pool.validate_selection(PoolSelection("ff" * 32, FIRST), rollout_count=M_ROLLOUTS)
    with pytest.raises(SeedPoolError):
        pool.validate_selection(pool.selection(FIRST), rollout_count=M_ROLLOUTS - 1)


# --- per-rollout binding -----------------------------------------------------------------------

def test_rollout_i_is_bound_to_the_i_th_chosen_seed():
    pool = _pool()
    selection = pool.selection(ODD)
    for index, seed in enumerate(ODD):
        binding = selection.rollout_binding(index)
        assert binding == {"schema": ROLLOUT_SCHEMA, "pool_sha256": pool.sha256, "seed_index": seed,
                           "rollout_index": index}
        assert parse_rollout_binding(binding) == RolloutSeed(pool.sha256, seed, index)
    with pytest.raises(SeedPoolError):
        selection.rollout_binding(M_ROLLOUTS)
    for bad in ({**selection.rollout_binding(0), "candidate_id": 0},
                {**selection.rollout_binding(0), "seed_index": True},
                {**selection.rollout_binding(0), "seed_index": 128},
                {**selection.rollout_binding(0), "rollout_index": 64},
                {**selection.rollout_binding(0), "schema": "public-group-rollout/v1"}, None, []):
        with pytest.raises(SeedPoolError):
            parse_rollout_binding(bad)


def test_rollouts_must_match_the_selection_and_their_position():
    pool = _pool()
    selection = pool.selection(ODD)
    commits = _commits(selection)
    validate_rollout_selection(pool, selection, commits)
    for invalid in (commits[:-1], commits[::-1], commits + commits[:1]):
        with pytest.raises(SeedPoolError):
            validate_rollout_selection(pool, selection, invalid)
    # two rollouts claiming the same seed
    twice = copy.deepcopy(commits)
    twice[1]["rollout"]["seed_pool"]["seed_index"] = twice[0]["rollout"]["seed_pool"]["seed_index"]
    with pytest.raises(SeedPoolError):
        validate_rollout_selection(pool, selection, twice)
    # a rollout drawn from a seed outside the signed selection
    foreign = copy.deepcopy(commits)
    foreign[0]["rollout"]["seed_pool"]["seed_index"] = 0
    with pytest.raises(SeedPoolError):
        validate_rollout_selection(pool, selection, foreign)
    # the right seeds, bound to other positions
    swapped = copy.deepcopy(commits)
    swapped[0]["rollout"]["seed_pool"], swapped[1]["rollout"]["seed_pool"] = (
        swapped[1]["rollout"]["seed_pool"], swapped[0]["rollout"]["seed_pool"])
    with pytest.raises(SeedPoolError):
        validate_rollout_selection(pool, selection, swapped)
    for field in ("seed_index", "rollout_index"):
        boolean = copy.deepcopy(_commits(pool.selection(FIRST)))
        boolean[1]["rollout"]["seed_pool"][field] = True     # True == 1 for dict equality
        with pytest.raises(SeedPoolError):
            validate_rollout_selection(pool, pool.selection(FIRST), boolean)
    episode = copy.deepcopy(commits)
    episode[0]["rollout"]["episode"] = {}
    with pytest.raises(SeedPoolError):
        validate_rollout_selection(pool, selection, episode)
    # a selection of another pool (other prompt) is not this pool's
    other = _pool(prompt_idx=8).selection(ODD)
    with pytest.raises(SeedPoolError):
        validate_rollout_selection(pool, other, _commits(other))


# --- draw ---------------------------------------------------------------------------------------

def test_draw_message_layout_is_pool_seed_position_only():
    pool = _pool()
    message = (b"public-seed-draw/v3" + bytes.fromhex(pool.sha256) + (5).to_bytes(4, "big")
               + (9).to_bytes(4, "big"))
    bits = int.from_bytes(hashlib.sha256(message).digest()[:8], "big") >> 11
    assert pool.uniform(5, 9) == bits / 2**53
    assert pool.sha256 == hashlib.sha256(canonical_json_bytes(pool.to_dict())).hexdigest()


def test_uniforms_are_distinct_bounded_and_refuse_bad_coordinates():
    pool = _pool()
    seed = M_ROLLOUTS                    # rank 0 of one subset, the last rank of another
    first = pool.selection(tuple(range(M_ROLLOUTS, POOL_SEEDS)))
    second = pool.selection((*range(M_ROLLOUTS - 1), seed))
    assert first.seeds.index(seed) == 0 and second.seeds.index(seed) == M_ROLLOUTS - 1
    assert pool.uniform(first.seeds[0], 3) == pool.uniform(second.seeds[-1], 3)
    assert _pool(pool_epoch=4).uniform(seed, 3) != pool.uniform(seed, 3)     # another pool, another value
    assert _pool(randomness="cd" * 32).uniform(seed, 3) != pool.uniform(seed, 3)
    assert len({pool.uniform(s, t) for s in range(POOL_SEEDS) for t in range(5)}) == POOL_SEEDS * 5
    assert all(0 <= pool.uniform(1, t) < 1 for t in range(256))
    for args in ((POOL_SEEDS, 0), (-1, 0), (0, -1), (True, 0), (0, True), (0.0, 0)):
        with pytest.raises(SeedPoolError):
            pool.uniform(*args)
    with pytest.raises(TypeError):
        pool.uniform(0, 0, 0)            # there is no rank coordinate any more


def test_a_seed_draws_the_same_in_every_subset_and_at_every_rank():
    """One seed is rollout 0 of a subset and the last rollout of another: same uniforms."""
    pool = _pool()
    seed = M_ROLLOUTS
    a = pool.selection(tuple(range(M_ROLLOUTS, POOL_SEEDS)))
    b = pool.selection((*range(M_ROLLOUTS - 1), seed))
    assert a.seeds.index(seed) == 0 and b.seeds.index(seed) == M_ROLLOUTS - 1
    assert set(a.seeds) & set(b.seeds) == {seed}
    draws = lambda selection, rank: [pool.uniform(selection.seeds[rank], t) for t in range(64)]
    assert draws(a, 0) == draws(b, M_ROLLOUTS - 1)
    # ...and choosing other companions cannot move it either
    assert draws(a, 0) == [pool.uniform(seed, t) for t in range(64)]


@pytest.mark.parametrize("change", [{"environment": CODE}, {"prompt_idx": 8}, {"checkpoint_hash": "e" * 40},
                                   {"pool_epoch": 4}, {"randomness": "cd" * 32}])
def test_draws_are_bound_to_authoritative_context(change):
    assert _pool().sha256 != _pool(**change).sha256
    assert _pool().uniform(1, 0) != _pool(**change).uniform(1, 0)


# --- group identity ------------------------------------------------------------------------------

def test_group_id_is_the_digest_of_the_selection():
    pool = _pool()
    # two miners choosing the same subset build equal selections: same group id, no hotkey in it
    miner_a, miner_b = pool.selection(ODD), PoolSelection.from_dict(pool.selection(ODD).to_dict())
    assert miner_a.sha256 == miner_b.sha256
    assert miner_a.sha256 == hashlib.sha256(canonical_json_bytes(miner_a.to_dict())).hexdigest()
    others = {pool.selection(FIRST).sha256, pool.selection(ODD).sha256,
              pool.selection((*FIRST[:-1], M_ROLLOUTS)).sha256, _pool(prompt_idx=8).selection(ODD).sha256,
              _pool(environment=CODE).selection(ODD).sha256, _pool(pool_epoch=4).selection(ODD).sha256}
    assert len(others) == 6


# --- announcement / legacy ------------------------------------------------------------------------

def test_pool_resolution_requires_server_announcement_and_capabilities():
    contract = contract_v2()
    capabilities = sorted(SUPPORTED_V2_CAPABILITIES)
    announcement = {"contract": contract.to_dict(), "schedule": initial_schedule(contract).to_dict(),
                    "checkpoint": {"checkpoint_n": 3, "repo": "models/test", "revision": "d" * 40, "sha256": "e" * 64},
                    "supported_capabilities": capabilities, "pool_epoch": 3, "pool_randomness": "ab" * 32}
    kw = {"environment": MATH, "prompt_idx": 7, "checkpoint_hash": "d" * 40}
    assert pool_from_service_policy(announcement, **kw) == _pool()
    assert pool_from_service_policy(announcement, **{**kw, "environment": CODE}).environment == CODE
    # an old miner's capability list does not cover the free-subset pool
    old = [c for c in capabilities if c != CAPABILITY] + ["public-group-pool/v1"]
    with pytest.raises(ValueError):
        pool_from_service_policy({**announcement, "supported_capabilities": old}, **kw)
    with pytest.raises(ValueError):
        pool_from_service_policy({**announcement, "supported_capabilities": ["legacy/v1"]}, **kw)
    with pytest.raises(SeedPoolError):
        pool_from_service_policy({**announcement, "pool_randomness": ""}, **kw)
    with pytest.raises(SeedPoolError):
        pool_from_service_policy({**announcement, "extra": 1}, **kw)


def test_legacy_paths_stay_inert():
    kw = {"environment": MATH, "prompt_idx": 7, "checkpoint_hash": "d" * 40}
    assert pool_from_service_policy(None, **kw) is None
    value = contract_v2_dict()
    for env in value["environments"].values():
        env["sampling"] = {"kind": "legacy/v1"}
    contract = ServiceContract.from_dict(value)
    announcement = {"contract": contract.to_dict(), "schedule": initial_schedule(contract).to_dict(),
                    "checkpoint": {"checkpoint_n": 3, "repo": "models/test", "revision": "d" * 40, "sha256": "e" * 64},
                    "supported_capabilities": sorted(SUPPORTED_V2_CAPABILITIES), "pool_epoch": 0,
                    "pool_randomness": ""}
    assert pool_from_service_policy(announcement, **kw) is None
    with pytest.raises(SeedPoolError):
        SeedPool.from_contract(contract, environment=MATH, prompt_idx=7, checkpoint_hash="d" * 40,
                               pool_epoch=0, randomness="ab" * 32)
