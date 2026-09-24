"""The minimum-incentive floor is a property of each task: a hotkey's share is
measured within its task, and what the floor cuts stays within that task."""

import random
from dataclasses import replace

import pytest

from reliquary.constants import MIN_INCENTIVE_RAMP_START, MIN_INCENTIVE_SHARE
from reliquary.shared.task_registry import RegistryError, set_cap, validate_entry
from reliquary.validator.weight_only import WeightOnlyValidator as W
from tests.unit.test_task_set_cap import _entry  # an RL entry builder


def _archives(task_id, rewards, windows=range(100, 140)):
    return [
        {"task_id": task_id, "window_start": w, "window_status": "completed",
         "rewards_by_hotkey": dict(rewards)}
        for w in windows
    ]


def _merged(*streams):
    return sorted(
        [a for s in streams for a in s],
        key=lambda a: (a["window_start"], a["task_id"]),
    )


# -- the parameter ---------------------------------------------------------

def test_a_task_may_declare_its_own_floor():
    entry = _entry("default", 0.9)
    validate_entry(replace(entry, params={**entry.params, "min_incentive_share": 0.0}))
    validate_entry(replace(entry, params={
        **entry.params, "min_incentive_share": 0.05, "min_incentive_ramp_start": 0.02,
    }))


@pytest.mark.parametrize("params", [
    {"min_incentive_share": -0.01},
    {"min_incentive_share": 1.0},
    {"min_incentive_share": "0.02"},
    {"min_incentive_share": 0.01, "min_incentive_ramp_start": 0.02},
    {"min_incentive_ramp_start": 0.5},
])
def test_a_floor_that_cannot_mean_anything_is_refused(params):
    entry = _entry("default", 0.9)
    with pytest.raises(RegistryError):
        validate_entry(replace(entry, params={**entry.params, **params}))


def test_floors_default_to_the_protocol_floor():
    entry = _entry("default", 0.9)
    corpus = replace(entry, task_id="corpus-x",
                     params={**entry.params, "min_incentive_share": 0.0})
    floors = W._floors_by_task({"default": entry, "corpus-x": corpus})
    assert floors["default"] == (MIN_INCENTIVE_RAMP_START, MIN_INCENTIVE_SHARE)
    assert floors["corpus-x"] == (0.0, 0.0)


def test_a_share_alone_below_the_default_ramp_start_ramps_from_zero():
    entry = _entry("default", 0.9)
    low = replace(entry, params={**entry.params, "min_incentive_share": 0.005})
    validate_entry(low)
    assert W._floors_by_task({"default": low})["default"] == (0.0, 0.005)


# -- the replay ------------------------------------------------------------

def test_with_one_task_the_per_task_floor_is_the_old_global_floor():
    rng = random.Random(7)
    for _ in range(20):
        rewards = {f"h{i}": rng.random() for i in range(rng.randint(2, 40))}
        total = sum(rewards.values())
        rewards = {k: 0.9 * v / total for k, v in rewards.items()}
        archives = _archives("default", rewards)
        old = W._apply_min_incentive_share(
            W._replay_ema(archives, caps={"default": 0.9}),
            start=MIN_INCENTIVE_RAMP_START, threshold=MIN_INCENTIVE_SHARE,
        )
        new = W._replay_ema(
            archives, caps={"default": 0.9},
            floors={"default": (MIN_INCENTIVE_RAMP_START, MIN_INCENTIVE_SHARE)},
        )
        assert new == pytest.approx(old)


def test_small_corpus_miners_are_measured_within_the_corpus_task():
    rl = _archives("default", {"R1": 0.6, "R2": 0.3})
    corpus = _archives("corpus-x", {f"C{i}": 0.01 for i in range(10)})
    caps = {"default": 0.9, "corpus-x": 0.1}
    both = W._replay_ema(
        _merged(rl, corpus), caps=caps,
        floors={"default": (0.01, 0.02), "corpus-x": (0.01, 0.02)},
    )
    # Each corpus miner holds 10% of its task: well above a 2% floor there.
    assert all(both[f"C{i}"] > 0 for i in range(10))


def test_a_floor_never_moves_mass_between_tasks():
    rl = _archives("default", {"R1": 0.6, "R2": 0.3})
    # "tiny" holds 0.5% of the corpus task: under a 1% ramp start there.
    corpus = _archives("corpus-x", {"big": 0.0995, "tiny": 0.0005})
    caps = {"default": 0.9, "corpus-x": 0.1}
    floors = {"default": (0.01, 0.02), "corpus-x": (0.01, 0.02)}
    alone = W._replay_ema(rl, caps=caps, floors=floors)
    both = W._replay_ema(_merged(rl, corpus), caps=caps, floors=floors)
    assert both["R1"] == pytest.approx(alone["R1"])
    assert both["R2"] == pytest.approx(alone["R2"])
    # The corpus task keeps its whole mass: the cut hotkey's share goes to the
    # other corpus hotkey, not to RL.
    corpus_total = sum(v for k, v in both.items() if k in ("big", "tiny"))
    unfloored = W._replay_ema(_merged(rl, corpus), caps=caps)
    assert corpus_total == pytest.approx(unfloored["big"] + unfloored["tiny"])
    assert both.get("tiny", 0.0) == 0.0


def test_a_zero_floor_pays_every_corpus_miner_its_own_tokens():
    corpus = _archives("corpus-x", {"big": 0.098, "tiny": 0.002})
    floored = W._replay_ema(corpus, caps={"corpus-x": 0.1},
                            floors={"corpus-x": (0.0, 0.0)})
    assert floored == pytest.approx(W._replay_ema(corpus, caps={"corpus-x": 0.1}))
    assert floored["tiny"] > 0


# -- declaring it ----------------------------------------------------------

def test_set_cap_can_change_a_tasks_floor_and_nothing_else():
    entry = _entry("default", 1.0)
    updated = set_cap({"default": entry}, "default", 0.9, min_incentive_share=0.0)
    params = updated["default"].params
    assert params["cap"] == 0.9 and params["min_incentive_share"] == 0.0
    assert updated["default"].contract == entry.contract


def test_a_corpus_task_is_declared_with_no_floor_by_default():
    from reliquary.cli.main import build_corpus_task_entry

    entry = build_corpus_task_entry(
        task_id="corpus-x", job_id="job-x",
        from_profile="teutonic-9b-reliquary-suite-v9-dev1",
        model_id="org/M", model_revision="r", model_architecture="Qwen3_5ForCausalLM",
        prompt_source="reliquary_dapo_math_v1", cap=0.1, overrides={},
    )
    assert entry.params["min_incentive_share"] == 0.0
    validate_entry(entry)
