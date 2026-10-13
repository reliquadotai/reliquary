"""The reference miner draws environments by advertised RL pay."""

from __future__ import annotations

import collections
import random
from types import SimpleNamespace

import pytest

from reliquary.miner.engine import pick_env_and_prompt
from reliquary.miner.episode_mining import pick_episode_task
from reliquary.miner.rl_pay_client import PayWeights, env_choice_mode, weights_from_pay


def _pay(**rates):
    return {"environments": {
        name: {"now": {"alpha_per_mtoken_paid": rate},
               "market": {"paid_fraction_of_submitted": {"low": 0.25, "high": 1.0}}}
        for name, rate in rates.items()}}


def test_weights_are_pay_times_the_middle_of_the_paid_fraction_range():
    assert weights_from_pay(_pay(math=0.6, code=0.2)) == {
        "math": pytest.approx(0.375), "code": pytest.approx(0.125)}
    assert weights_from_pay({"status": "loading"}) is None
    assert weights_from_pay({"environments": {}}) is None


def test_mode_defaults_to_price_and_refuses_typos():
    assert env_choice_mode({}) == "price"
    assert env_choice_mode({"RELIQUARY_MINER_ENV_CHOICE": "MIX"}) == "mix"
    with pytest.raises(ValueError):
        env_choice_mode({"RELIQUARY_MINER_ENV_CHOICE": "cheapest"})


def test_pay_weights_cache_a_minute_and_fall_back_on_errors():
    calls, now = [], [0.0]

    def fetch(url):
        calls.append(url)
        if len(calls) == 2:
            raise RuntimeError("down")
        return _pay(math=0.6)

    pay = PayWeights("http://v:8080/", fetch=fetch, clock=lambda: now[0], background=False)
    assert pay.weights() == {"math": pytest.approx(0.375)}
    assert pay.weights() == {"math": pytest.approx(0.375)} and len(calls) == 1
    now[0] += 61
    assert pay.weights() is None                      # validator down: the mix
    assert calls[0] == "http://v:8080/rl/pay"


class _Env:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


ENVS = {"math": _Env(1000), "code": _Env(1000)}
MIX = [("math", 1), ("code", 1)]


def test_override_weights_bias_the_draw():
    rng = random.Random(0)
    drawn = collections.Counter(
        pick_env_and_prompt(ENVS, MIX, {}, rng=rng, weights_override={"math": 3.0, "code": 1.0})[0]
        for _ in range(4000))
    assert 0.72 < drawn["math"] / 4000 < 0.78


def test_no_override_keeps_todays_draw():
    a = [pick_env_and_prompt(ENVS, MIX, {}, rng=random.Random(5)) for _ in range(1)]
    b = [pick_env_and_prompt(ENVS, MIX, {}, rng=random.Random(5), weights_override=None) for _ in range(1)]
    assert a == b


def test_an_all_zero_override_falls_back_to_the_mix():
    rng = random.Random(1)
    drawn = collections.Counter(
        pick_env_and_prompt(ENVS, MIX, {}, rng=rng, weights_override={"other": 2.0})[0]
        for _ in range(2000))
    assert drawn["math"] > 800 and drawn["code"] > 800


def _view():
    return SimpleNamespace(accepting={"math": True, "code": True},
                           ranges={"math": (0, 100), "code": (0, 100)},
                           cooldowns={}, window=1)


def test_episode_order_follows_the_weights():
    rng = random.Random(2)
    first = collections.Counter(
        pick_episode_task(_view(), environments=["math", "code"], sizes={"math": 100, "code": 100},
                          taken=set(), rng=rng, weights={"math": 9.0, "code": 1.0})[0]
        for _ in range(2000))
    assert first["math"] / 2000 > 0.85


def test_episode_without_weights_is_todays_shuffle():
    kwargs = dict(environments=["math", "code"], sizes={"math": 100, "code": 100}, taken=set())
    assert (pick_episode_task(_view(), rng=random.Random(3), **kwargs)
            == pick_episode_task(_view(), rng=random.Random(3), weights=None, **kwargs))


def test_priced_envs_exhausted_falls_back_to_the_mix_instead_of_crashing():
    """C2: when the only env left has weight 0 under the override, the draw
    must use the mix, not raise ValueError out of the mining loop."""
    cooldowns = {"math": set(range(1000))}           # math fully in cooldown
    env, _ = pick_env_and_prompt(ENVS, MIX, cooldowns, rng=random.Random(0),
                                 weights_override={"math": 5.0})
    assert env == "code"


def test_unpriced_and_zero_low_envs_still_get_drawn():
    """I1: a new or starved env (no paid group yet) and an env whose low paid
    fraction is 0 keep a positive weight, so a rising price can attract miners."""
    body = _pay(math=0.6)
    body["environments"]["code"] = {"now": {"alpha_per_mtoken_paid": None},
                                    "market": {"paid_fraction_of_submitted": None}}
    body["environments"]["logic"] = {"now": {"alpha_per_mtoken_paid": 0.5},
                                     "market": {"paid_fraction_of_submitted": {"low": 0.0, "high": 0.8}}}
    weights = weights_from_pay(body)
    assert weights["code"] > 0 and weights["logic"] > 0


def test_pay_weights_never_block_the_caller_in_background_mode():
    """I5: the episode loop calls weights() on the event loop; a slow validator
    must not stall it. Background mode returns the last value immediately."""
    import threading
    gate = threading.Event()

    def slow_fetch(url):
        gate.wait(5)
        return _pay(math=0.6)

    pay = PayWeights("http://v", fetch=slow_fetch, background=True)
    assert pay.weights() is None                      # returns at once, fetch runs behind
    gate.set()
    pay.join(timeout=5)
    assert pay.weights() == {"math": pytest.approx(0.375)}
