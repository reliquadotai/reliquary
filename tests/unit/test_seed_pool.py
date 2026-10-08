import copy
import pytest

from reliquary.constants import M_ROLLOUTS
from reliquary.protocol.seed_pool import (
    PoolSelection, SeedPool, SeedPoolError, pool_from_service_policy,
    validate_rollout_selection,
)
from reliquary.protocol.service_contract import SUPPORTED_V2_CAPABILITIES, ServiceContract
from reliquary.protocol.service_schedule import initial_schedule
from tests.unit.service_v2_fixtures import CODE, MATH, contract_v2


def _contract(**kw):
    return contract_v2(pool_groups=3, **kw)


def _pool(**changes):
    context = {"environment": MATH, "prompt_idx": 7, "checkpoint_hash": "d" * 40, "pool_epoch": 3,
               "randomness": "ab" * 32, **changes}
    return SeedPool.from_contract(_contract(), **context)


def test_public_identity_is_immutable_bounded_and_round_trips():
    pool = _pool()
    assert pool.exploration_rollouts == 3 * M_ROLLOUTS > M_ROLLOUTS
    assert SeedPool.from_dict(pool.to_dict()) == pool
    assert pool.uniform(1, 2, 3) == _pool().uniform(1, 2, 3)
    assert len({pool.uniform(c, r, t) for c in range(3) for r in range(M_ROLLOUTS)
                for t in range(5)}) == 3 * M_ROLLOUTS * 5
    assert all(0 <= pool.uniform(1, 0, t) < 1 for t in range(256))
    with pytest.raises(AttributeError):
        pool.pool_epoch = 5
    for args in ((3, 0, 0), (0, M_ROLLOUTS, 0), (0, 0, -1), (True, 0, 0)):
        with pytest.raises(SeedPoolError):
            pool.uniform(*args)


@pytest.mark.parametrize("change", [{"environment": CODE}, {"prompt_idx": 8}, {"checkpoint_hash": "e" * 40},
                                   {"pool_epoch": 4}, {"randomness": "cd" * 32}])
def test_draws_are_bound_to_authoritative_context(change):
    assert _pool().sha256 != _pool(**change).sha256
    assert _pool().uniform(1, 0, 0) != _pool(**change).uniform(1, 0, 0)


def test_one_complete_candidate_retains_original_rollout_indices():
    pool = _pool()
    selection = pool.selection(2)
    commits = [{"rollout": {"seed_pool": selection.rollout_binding(i)}} for i in range(M_ROLLOUTS)]
    validate_rollout_selection(pool, selection, commits)
    for invalid in (commits[:-1], commits[::-1]):
        with pytest.raises(SeedPoolError):
            validate_rollout_selection(pool, selection, invalid)
    mixed = copy.deepcopy(commits)
    mixed[0]["rollout"]["seed_pool"]["candidate_id"] = 1
    with pytest.raises(SeedPoolError):
        validate_rollout_selection(pool, selection, mixed)
    mixed = copy.deepcopy(commits)
    mixed[1]["rollout"]["seed_pool"]["rollout_index"] = True
    with pytest.raises(SeedPoolError):
        validate_rollout_selection(pool, selection, mixed)
    with pytest.raises(SeedPoolError):
        pool.validate_selection(PoolSelection("ff" * 32, 2), rollout_count=M_ROLLOUTS)


def test_pool_resolution_requires_server_announcement_and_capabilities():
    contract = _contract()
    capabilities = sorted(SUPPORTED_V2_CAPABILITIES)
    announcement = {"contract": contract.to_dict(), "schedule": initial_schedule(contract).to_dict(),
                    "checkpoint": {"checkpoint_n": 3, "repo": "models/test", "revision": "d" * 40, "sha256": "e" * 64},
                    "supported_capabilities": capabilities, "pool_epoch": 3, "pool_randomness": "ab" * 32}
    kw = {"environment": MATH, "prompt_idx": 7, "checkpoint_hash": "d" * 40}
    assert pool_from_service_policy(announcement, **kw) == _pool()
    assert pool_from_service_policy(None, **kw) is None
    assert pool_from_service_policy(announcement, **{**kw, "environment": CODE}).environment == CODE
    invalid = {**announcement, "supported_capabilities": ["legacy/v1"]}
    with pytest.raises(ValueError):
        pool_from_service_policy(invalid, **kw)
    invalid = {**announcement, "pool_randomness": ""}
    with pytest.raises(SeedPoolError):
        pool_from_service_policy(invalid, **kw)
    with pytest.raises(SeedPoolError):
        pool_from_service_policy({**announcement, "extra": 1}, **kw)
