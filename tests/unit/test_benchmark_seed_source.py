"""The forced-seed qualification harness draws exactly what the validator recomputes.

No GPU, no model: the seed source, the subset policies and the group scoring are pure.
"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from reliquary.constants import (
    FORCED_SEED_CONSISTENCY_FLOOR, FORCED_SEED_ROLLOUT_FLOOR, M_ROLLOUTS,
)
from reliquary.environment.forced_sampling import u_at
from reliquary.protocol.seed_pool import SeedPool, pool_from_service_policy
from reliquary.protocol.service_contract import ServiceContract, ServiceContractError, SUPPORTED_V2_CAPABILITIES
from reliquary.protocol.service_schedule import initial_schedule
from tests.unit.service_v2_fixtures import MATH, contract_v2, contract_v2_dict

SCRIPT = Path(__file__).parents[2] / "scripts" / "benchmark_inference_contract.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_inference_contract_seed", SCRIPT)
B = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(B)

CKPT = "d" * 40
BEACON = "ab" * 32
POOL_SEEDS = 2 * M_ROLLOUTS


def _args(**kw):
    base = dict(seed_source="pool", randomness=BEACON, checkpoint_hash=CKPT, pool_env=MATH,
                pool_epoch=3, contract=contract_v2())
    base.update(kw)
    return SimpleNamespace(**base)


def _validator_pool(contract, *, prompt_idx, epoch=3, beacon=BEACON):
    """The pool exactly as the validator resolves it from the operator announcement."""
    announcement = {
        "contract": contract.to_dict(), "schedule": initial_schedule(contract).to_dict(),
        "checkpoint": {"checkpoint_n": 3, "repo": "models/test", "revision": CKPT, "sha256": "e" * 64},
        "supported_capabilities": sorted(SUPPORTED_V2_CAPABILITIES),
        "pool_epoch": epoch, "pool_randomness": beacon,
    }
    return pool_from_service_policy(announcement, environment=MATH, prompt_idx=prompt_idx, checkpoint_hash=CKPT)


def _row(seed, n_stoch=100, n_match=100):
    return {"seed_index": seed, "n_stochastic": n_stoch, "n_exact_match": n_match}


# --- seed source ---------------------------------------------------------------------------

def test_window_source_matches_the_protocol_u_at():
    args = SimpleNamespace(seed_source="window", randomness="42" * 32, checkpoint_hash="c" * 40)
    u = B.uniform_source(args, prompt_idx=3)
    assert u(1, 5) == u_at("42" * 32, 3, "c" * 40, 1, 5)


def test_pool_source_is_the_validators_seed_uniforms_path():
    contract = contract_v2()
    pool = _validator_pool(contract, prompt_idx=9)
    selection = pool.selection(list(range(1, POOL_SEEDS, 2)))
    u = B.uniform_source(_args(), prompt_idx=9)
    # validator: pool.uniform(selection.seeds[rollout_rank], position)
    for rank in range(M_ROLLOUTS):
        for position in (0, 1, 777):
            assert u(selection.seeds[rank], position) == pool.uniform(selection.seeds[rank], position)
    assert B.build_pool(_args(), prompt_idx=9) == pool
    assert B.build_pool(_args(), prompt_idx=9).sha256 == pool.sha256


def test_draw_depends_on_seed_not_on_rank_or_subset():
    u = B.uniform_source(_args(), prompt_idx=9)
    first, odd = list(range(M_ROLLOUTS)), list(range(1, POOL_SEEDS, 2))
    # seed 1 is rank 1 of the first subset and rank 0 of the odd one: same draw
    assert first[1] == odd[0] == 1
    assert [u(1, t) for t in range(50)] == [B.uniform_source(_args(), prompt_idx=9)(1, t) for t in range(50)]


def test_pool_source_is_deterministic_and_varies_with_the_inputs():
    base = [B.uniform_source(_args(), prompt_idx=9)(2, t) for t in range(20)]
    assert base == [B.uniform_source(_args(), prompt_idx=9)(2, t) for t in range(20)]
    for kw, prompt in ((dict(pool_epoch=4), 9), (dict(randomness="cd" * 32), 9),
                       (dict(checkpoint_hash="e" * 40), 9), ({}, 10)):
        assert [B.uniform_source(_args(**kw), prompt_idx=prompt)(2, t) for t in range(20)] != base


def test_pool_is_two_m_seeds_and_refuses_out_of_pool_seeds():
    pool = B.build_pool(_args(), prompt_idx=1)
    assert pool.pool_seeds == POOL_SEEDS == 2 * pool.group_size
    u = B.uniform_source(_args(), prompt_idx=1)
    u(POOL_SEEDS - 1, 0)
    with pytest.raises(ValueError):
        u(POOL_SEEDS, 0)


def test_r16_pools_renew_every_window_so_the_epoch_changes_the_pool():
    a = B.build_pool(_args(pool_epoch=3), prompt_idx=1)
    b = B.build_pool(_args(pool_epoch=4), prompt_idx=1)
    assert a.sha256 != b.sha256
    assert a.renewal_windows == 1
    value = contract_v2_dict()
    value["environments"][MATH]["sampling"]["renewal_windows"] = 2
    with pytest.raises(ServiceContractError):
        ServiceContract.from_dict(value)


def test_contract_can_be_given_as_a_file(tmp_path):
    path = tmp_path / "order.json"
    path.write_text(json.dumps(contract_v2().to_dict()), encoding="utf-8")
    assert B.build_pool(_args(contract=path), prompt_idx=2) == B.build_pool(_args(), prompt_idx=2)


# --- subset policies -----------------------------------------------------------------------

def test_first_policy_generates_m_seeds_all_policy_generates_the_whole_pool():
    pool = B.build_pool(_args(), prompt_idx=1)
    assert B.seeds_to_generate(pool, "first") == [list(range(M_ROLLOUTS))]
    chunks = B.seeds_to_generate(pool, "all")
    assert [len(c) for c in chunks] == [M_ROLLOUTS, M_ROLLOUTS]
    assert sorted(s for c in chunks for s in c) == list(range(POOL_SEEDS))
    with pytest.raises(ValueError):
        B.seeds_to_generate(pool, "random")


def test_picks_choose_m_ascending_distinct_seeds_the_pool_accepts():
    pool = B.build_pool(_args(), prompt_idx=1)
    agreement = {s: 0.5 + 0.01 * ((s * 7) % POOL_SEEDS) for s in range(POOL_SEEDS)}
    for pick in B.POOL_PICKS:
        chosen = B.choose_seeds(pool, pick, agreement)
        assert len(chosen) == M_ROLLOUTS and list(chosen) == sorted(set(chosen))
        pool.selection(chosen)  # a valid selection
    low = B.choose_seeds(pool, "lowest-agreement", agreement)
    high = B.choose_seeds(pool, "highest-agreement", agreement)
    assert max(agreement[s] for s in low) <= min(agreement[s] for s in high)
    assert B.choose_seeds(pool, "first", agreement) == tuple(range(M_ROLLOUTS))
    assert B.choose_seeds(pool, "last", agreement) == tuple(range(M_ROLLOUTS, POOL_SEEDS))
    with pytest.raises(ValueError):
        B.choose_seeds(pool, "best", agreement)
    with pytest.raises(ValueError):
        B.choose_seeds(pool, "first", {0: 1.0})


# --- scoring is the validator's ------------------------------------------------------------

def test_group_score_uses_the_validators_floors():
    ok = [_row(s, 100, 95) for s in range(M_ROLLOUTS)]
    score = B.score_group(ok)
    assert score["accepted"] and score["agreement"] == pytest.approx(0.95)
    assert score["group_floor"] == FORCED_SEED_CONSISTENCY_FLOOR
    assert score["rollout_floor"] == FORCED_SEED_ROLLOUT_FLOOR

    below_group = [_row(s, 100, 75) for s in range(M_ROLLOUTS)]  # 0.75 < 0.80
    assert B.score_group(below_group)["group_rejected"] and not B.score_group(below_group)["accepted"]

    one_swapped = [_row(s, 100, 98) for s in range(M_ROLLOUTS - 1)] + [_row(M_ROLLOUTS - 1, 100, 50)]
    s = B.score_group(one_swapped)
    assert not s["group_rejected"] and s["rollout_rejected"] and not s["accepted"] and s["rollouts_below_floor"] == 1

    thin = [_row(s, 1, 0) for s in range(M_ROLLOUTS)]  # too few stochastic positions: abstains
    assert B.score_group(thin)["accepted"] and B.score_group(thin)["group_abstained"]


def test_summary_rates():
    def group(rows):
        return {"score": B.score_group(rows), "rollouts": rows}
    good = group([_row(s, 100, 96) for s in range(M_ROLLOUTS)])
    bad = group([_row(s, 100, 60) for s in range(M_ROLLOUTS)])
    summary = B.forced_seed_summary([good, good, good, bad])
    assert summary["groups"] == 4
    assert summary["group_acceptance_rate"] == 0.75
    assert summary["rollout_floor_pass_rate"] == pytest.approx(48 / 64)
    assert summary["group_floor"] == FORCED_SEED_CONSISTENCY_FLOOR
