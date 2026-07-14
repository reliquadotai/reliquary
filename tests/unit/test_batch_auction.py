"""Difficulty-auction selection: the 8 HARDEST win, speed only breaks ties.

Replaces the drand-FCFS rule of ``select_batch_and_distribute``, under which the
8 submissions that merely ARRIVED first won and difficulty was never priced.
See docs/superpowers/specs/2026-07-14-difficulty-auction-design.md
"""

from dataclasses import dataclass

from reliquary.validator.batch_auction import select_batch_auction
from reliquary.validator.cooldown import CooldownMap


@dataclass
class FakeSubmission:
    hotkey: str
    prompt_idx: int
    drand_round: int
    value: float
    merkle_root: bytes = b"\x00" * 32
    selection_digest: bytes = b"\x00" * 32


def _sub(hotkey, prompt_idx, drand_round, value, digest=None):
    d = digest or hotkey.encode().ljust(32, b"\x00")
    return FakeSubmission(
        hotkey=hotkey,
        prompt_idx=prompt_idx,
        drand_round=drand_round,
        value=value,
        merkle_root=d,
        selection_digest=d,
    )


def _auction(subs, b=8, pool=1.0, window=100, **kw):
    return select_batch_auction(
        subs,
        b=b,
        cooldown_map=CooldownMap(cooldown_windows=50),
        current_window=window,
        pool=pool,
        **kw,
    )


def test_harder_group_beats_faster_group():
    """THE design. The slow miner with the hard prompt takes the slot from the
    fast miner with the easy one — the exact inversion of today's rule."""
    fast_easy = _sub("fast", prompt_idx=1, drand_round=1, value=0.11)   # k=6
    slow_hard = _sub("slow", prompt_idx=2, drand_round=9, value=0.32)   # k=2

    batch, rewards = _auction([fast_easy, slow_hard], b=1)

    assert [s.hotkey for s in batch] == ["slow"]
    assert rewards == {"slow": 1.0}


def test_equal_value_is_broken_by_speed():
    """Ties are the NORM, not an edge case: v(k) has only 7 distinct values.
    Within a difficulty class, the earlier drand round wins."""
    late = _sub("late", prompt_idx=1, drand_round=9, value=0.32)
    early = _sub("early", prompt_idx=2, drand_round=1, value=0.32)

    batch, _ = _auction([late, early], b=1)

    assert [s.hotkey for s in batch] == ["early"]


def test_equal_value_and_round_is_broken_canonically():
    """Two validators must converge on the same batch to agree on weights, so
    input order must not decide anything."""
    a = _sub("a", prompt_idx=1, drand_round=1, value=0.32, digest=b"\xaa" * 32)
    z = _sub("z", prompt_idx=2, drand_round=1, value=0.32, digest=b"\x01" * 32)

    picked_one = [s.hotkey for s in _auction([a, z], b=1)[0]]
    picked_other = [s.hotkey for s in _auction([z, a], b=1)[0]]

    assert picked_one == picked_other


def test_same_prompt_highest_value_takes_the_slot():
    """One slot per prompt. The loser on that prompt earns nothing — it did not
    win a slot, so there is no K-way split (unlike the rule it replaces)."""
    weak = _sub("weak", prompt_idx=7, drand_round=1, value=0.11)
    strong = _sub("strong", prompt_idx=7, drand_round=9, value=0.32)

    batch, rewards = _auction([weak, strong], b=8)

    assert [s.hotkey for s in batch] == ["strong"]
    assert rewards == {"strong": 0.125}


def test_zero_value_groups_are_never_selected():
    """k=0 and k=8 carry no gradient at all. This is what replaces SIGMA_MIN."""
    unanimous = _sub("unanimous", prompt_idx=1, drand_round=1, value=0.0)

    batch, rewards = _auction([unanimous], b=8)

    assert batch == [] and rewards == {}


def test_unfilled_slots_burn():
    """Two winners across eight slots pay 2/8 — the rest burns, no redistribution."""
    subs = [
        _sub("a", prompt_idx=1, drand_round=1, value=0.32),
        _sub("b", prompt_idx=2, drand_round=1, value=0.30),
    ]

    _, rewards = _auction(subs, b=8)

    assert rewards == {"a": 0.125, "b": 0.125}


def test_payout_is_flat_across_winners_regardless_of_value():
    """The score decides WHO gets in, not what a slot is worth. Paying
    proportionally to value would pull miners toward k=1 — the lucky-guess and
    broken-label region."""
    subs = [
        _sub("hardest", prompt_idx=1, drand_round=1, value=0.32),
        _sub("milder", prompt_idx=2, drand_round=1, value=0.18),
    ]

    _, rewards = _auction(subs, b=2)

    assert rewards["hardest"] == rewards["milder"] == 0.5


def test_coldkey_slot_cap_bounds_one_operator():
    """The cap today's per-PROMPT 8-distinct rule is missing: coldkey 5CQ6...
    took 13.1% of emission by flooding DISTINCT prompts across hotkeys."""
    whale = [
        _sub(f"hk{i}", prompt_idx=i, drand_round=1, value=0.32) for i in range(5)
    ]
    honest = _sub("honest", prompt_idx=99, drand_round=5, value=0.11)

    batch, rewards = _auction(
        whale + [honest],
        b=4,
        max_slots_per_coldkey=2,
        coldkey_of=lambda hk: "whale" if hk.startswith("hk") else hk,
    )

    assert sum(1 for s in batch if s.hotkey.startswith("hk")) == 2
    assert "honest" in rewards  # a lower-value honest group is promoted in


def test_prompts_in_cooldown_are_skipped():
    cd = CooldownMap(cooldown_windows=50)
    cd.record_batched(prompt_idx=1, window=100)

    batch, _ = select_batch_auction(
        [_sub("a", prompt_idx=1, drand_round=1, value=0.32)],
        b=8,
        cooldown_map=cd,
        current_window=101,
        pool=1.0,
    )

    assert batch == []
