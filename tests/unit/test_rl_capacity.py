"""The RL session allocation: weighted water-filling over entities.
Every check runs in the production regime: production constants, small and large fleets, low and high demand."""
import random

import pytest

from reliquary.protocol.service_episode import GROUPS_IN_FLIGHT
from reliquary.sandbox.rl_capacity import (
    LOW_YIELD,
    LOW_YIELD_SHARE,
    MAIN,
    NEWCOMER,
    NEWCOMER_EXIT_GROUPS,
    NEWCOMER_MAX_GROUPS,
    NEWCOMER_SHARE,
    USEFUL_HORIZON_S,
    YIELD_MIN_DECIDED,
    YieldStats,
    allocate,
    entity_sizes,
    multiplier,
    potential,
    sessions_of,
    standing,
    water_fill,
    weight,
)

P = 32
FLEETS = (216, 2_160, 21_600)                      # small, Task 22a's, large


def test_nothing_is_keyed_per_operator_and_no_per_key_constant_remains():
    import reliquary.sandbox.rl_capacity as module

    for removed in ("QUOTA_KEYS", "QuotaKeys", "quota_key", "OPERATOR", "MAX_GROUPS", "YIELD_PRIOR_SESSIONS",
                    "GROWTH_MARGIN", "SESSION_SPAN_S", "main_live", "demand_live", "pools_for"):
        assert not hasattr(module, removed), removed


def test_standings_and_a_continuous_weight_through_the_floor():
    assert NEWCOMER_MAX_GROUPS == GROUPS_IN_FLIGHT
    assert standing(YieldStats(), pool_seeds=P) == NEWCOMER
    assert standing(YieldStats(useful=0, decided=NEWCOMER_EXIT_GROUPS * P), pool_seeds=P) == LOW_YIELD
    assert standing(YieldStats(useful=101, decided=200), pool_seeds=P) == MAIN
    assert standing(YieldStats(useful=100, decided=200), pool_seeds=P) == LOW_YIELD
    day = USEFUL_HORIZON_S
    just_above = weight(YieldStats(useful=101, decided=200, useful_s_24h=100 * day), pool_seeds=P)
    assert 0 < just_above < 2.0                        # 100 concurrent at 50.5 %: 1.7, no 2-group step
    assert weight(YieldStats(useful=50, decided=YIELD_MIN_DECIDED - 1, useful_s_24h=day), pool_seeds=P) == 0.0


@pytest.mark.parametrize("ratio, factor", [(1.0, 1.0), (0.8, 1.0), (0.65, 0.5), (0.5, 0.0), (0.2, 0.0)])
def test_the_multiplier_is_linear_between_50_and_80(ratio, factor):
    assert multiplier(ratio) == pytest.approx(factor)


def test_water_filling_gives_min_of_demand_and_weighted_share_and_respreads_the_rest():
    alloc = water_fill(100, {"a": (1.0, 10.0), "b": (1.0, 1_000.0), "c": (2.0, 1_000.0)})
    assert alloc == pytest.approx({"a": 10.0, "b": 30.0, "c": 60.0})
    assert water_fill(100, {"a": (1.0, 0.0), "b": (5.0, 7.0)}) == pytest.approx({"a": 0.0, "b": 7.0})


@pytest.mark.parametrize("capacity", FLEETS)
@pytest.mark.parametrize("load", ["low", "high"])
def test_splitting_a_main_key_never_increases_its_total(capacity, load):
    rng = random.Random(capacity + len(load))
    scale = 0.02 if load == "low" else 2.0             # total MAIN demand as a share of the fleet
    for _ in range(200):
        main = {f"k{i}": (rng.uniform(0.5, 300.0), rng.uniform(1.0, scale * capacity / 4)) for i in range(rng.randint(1, 8))}
        newcomers, low = rng.uniform(0, capacity), rng.uniform(0, capacity)
        whole = allocate(capacity, pool_seeds=P, main=main, newcomer_demand=newcomers, low_yield_demand=low)
        w, d = main["k0"]
        k = rng.randint(2, 40)                         # parts may fall far below one group
        parts = {**{key: v for key, v in main.items() if key != "k0"}, **{f"k0/{j}": (w / k, d / k) for j in range(k)}}
        split = allocate(capacity, pool_seeds=P, main=parts, newcomer_demand=newcomers, low_yield_demand=low)
        total = sum(split.main[f"k0/{j}"] for j in range(k))
        assert total == pytest.approx(whole.main["k0"], rel=1e-9, abs=1e-6)
        assert sum(sessions_of(split.main[f"k0/{j}"]) for j in range(k)) <= sessions_of(whole.main["k0"])
        assert split.newcomer == pytest.approx(whole.newcomer) and split.low_yield == pytest.approx(whole.low_yield)


@pytest.mark.parametrize("capacity", FLEETS)
def test_more_useful_volume_never_lowers_a_keys_allocation(capacity):
    rng = random.Random(capacity)
    for _ in range(100):
        others = {f"o{i}": (rng.uniform(1, 300), rng.uniform(1, capacity)) for i in range(rng.randint(0, 6))}
        demand = rng.uniform(1, capacity)
        previous = -1.0
        for w in sorted(rng.uniform(0.01, 500) for _ in range(20)):
            got = allocate(capacity, pool_seeds=P, main={**others, "me": (w, demand)}, newcomer_demand=capacity,
                           low_yield_demand=capacity).main["me"]
            assert got >= previous - 1e-9
            previous = got


@pytest.mark.parametrize("capacity", FLEETS)
@pytest.mark.parametrize("takes_spare", [True, False])
def test_more_fresh_keys_never_grow_the_newcomer_entity_beyond_its_rule(capacity, takes_spare):
    rng = random.Random(capacity)
    newcomer_n, _low = entity_sizes(capacity, pool_seeds=P)
    for _ in range(100):
        main = {f"k{i}": (rng.uniform(1, 300), rng.uniform(1, capacity)) for i in range(rng.randint(0, 6))}
        before = allocate(capacity, pool_seeds=P, main=main, newcomer_demand=2 * P, low_yield_demand=0,
                          takes_spare=takes_spare)
        flood = allocate(capacity, pool_seeds=P, main=main, newcomer_demand=100 * 2 * P, low_yield_demand=0,
                         takes_spare=takes_spare)
        assert all(flood.main[key] <= before.main[key] + 1e-9 for key in main)
        alone = allocate(capacity, pool_seeds=P, main={}, newcomer_demand=100 * 2 * P, low_yield_demand=0,
                         takes_spare=takes_spare)
        if takes_spare:      # its guarantee N, then at most a weighted share of what MAIN keys leave
            assert flood.newcomer <= alone.newcomer + 1e-9
            busy = {key: (w, 10.0 * capacity) for key, (w, _d) in main.items()}
            if busy:
                contended = allocate(capacity, pool_seeds=P, main=busy, newcomer_demand=100 * 2 * P,
                                     low_yield_demand=0, takes_spare=True)
                share = 32.0 / (32.0 + sum(w for w, _ in busy.values()))
                assert contended.newcomer <= newcomer_n + share * (capacity - newcomer_n) + 1e-6
        else:
            assert flood.newcomer <= newcomer_n


@pytest.mark.parametrize("capacity", FLEETS)
def test_the_low_yield_entity_is_a_hard_cap(capacity):
    _n, low_l = entity_sizes(capacity, pool_seeds=P)
    got = allocate(capacity, pool_seeds=P, main={}, newcomer_demand=0, low_yield_demand=10 * capacity)
    assert got.low_yield == low_l and low_l == max(P, int(LOW_YIELD_SHARE * capacity))
    assert entity_sizes(capacity, pool_seeds=P)[0] == max(P, int(NEWCOMER_SHARE * capacity))


def test_a_dormant_key_gets_nothing_and_a_returning_one_at_once():
    idle = allocate(2_160, pool_seeds=P, main={"big": (500.0, 0.0), "small": (5.0, 100.0)}, newcomer_demand=0,
                    low_yield_demand=0)
    assert idle.main["big"] == 0.0 and idle.main["small"] == 100.0
    back = allocate(2_160, pool_seeds=P, main={"big": (500.0, 3_000.0), "small": (5.0, 100.0)}, newcomer_demand=0,
                    low_yield_demand=0)
    assert back.main["big"] > 1_500                    # its weight, immediately: no history to rebuild


def test_the_potential_is_the_share_with_its_own_demand_unbounded():
    idle = potential(2_160, pool_seeds=P, main={}, hotkey="me", weight_h=5.0, newcomer_demand=0, low_yield_demand=0)
    assert idle == 2_160                                       # an idle fleet: all of it, whatever it asked so far
    busy = potential(2_160, pool_seeds=P, main={"big": (15.0, 10_000.0), "me": (5.0, 1.0)}, hotkey="me",
                     weight_h=5.0, newcomer_demand=0, low_yield_demand=0)
    assert busy == pytest.approx(540.0)                        # its weighted share, not its current demand of 1
