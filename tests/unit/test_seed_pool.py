import copy
import json
from pathlib import Path

import pytest

from reliquary.constants import M_ROLLOUTS
from reliquary.protocol.seed_pool import (
    PoolSelection, SeedPool, SeedPoolError, pool_from_service_policy,
    validate_rollout_selection,
)
from reliquary.protocol.service_contract import ServiceContract


def _contract():
    value = json.loads((Path(__file__).parents[1] / "fixtures/service_contract_v1.json").read_text())
    value["policies"]["sampling"] = {"kind": "public-group-pool/v1", "group_size": M_ROLLOUTS,
                                     "pool_groups": 3, "renewal_windows": 2}
    return ServiceContract.from_dict(value)


def _pool(**changes):
    context = {"prompt_idx": 7, "checkpoint_hash": "d" * 40, "pool_epoch": 3,
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


@pytest.mark.parametrize("change", [{"prompt_idx": 8}, {"checkpoint_hash": "e" * 40},
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
    capabilities = ["environment-reward/v1", "public-group-pool/v1", "all/v1", "static/v1", "frozen/v1", "legacy/v1"]
    announcement = {"contract": contract.to_dict(), "supported_capabilities": capabilities,
                    "pool_epoch": 3, "pool_randomness": "ab" * 32}
    assert pool_from_service_policy(announcement, prompt_idx=7, checkpoint_hash="d" * 40) == _pool()
    assert pool_from_service_policy(None, prompt_idx=7, checkpoint_hash="d" * 40) is None
    invalid = {**announcement, "supported_capabilities": ["legacy/v1"]}
    with pytest.raises(ValueError):
        pool_from_service_policy(invalid, prompt_idx=7, checkpoint_hash="d" * 40)
    invalid = {**announcement, "pool_randomness": ""}
    with pytest.raises(SeedPoolError):
        pool_from_service_policy(invalid, prompt_idx=7, checkpoint_hash="d" * 40)
    draw_contract = contract.to_dict()
    draw_contract["policies"]["sampling"] = {"kind": "public-draw-pool/v1", "group_size": M_ROLLOUTS,
                                             "pool_draws": M_ROLLOUTS + 1, "renewal_windows": 2}
    draw = ServiceContract.from_dict(draw_contract)
    with pytest.raises(SeedPoolError, match="not implemented"):
        pool_from_service_policy({**announcement, "contract": draw.to_dict(),
                                  "supported_capabilities": [*capabilities, "public-draw-pool/v1"]},
                                 prompt_idx=7, checkpoint_hash="d" * 40)
