"""The async prompt cooldown sized from the run (rl_cooldown.AdaptiveCooldown)."""
from __future__ import annotations

import random

import pytest

from reliquary.validator.rl_cooldown import AdaptiveCooldown, CooldownParams, cooldown_overrides
from reliquary.validator.rl_lease_quota import QuotaConfig, QuotaController
from reliquary.validator.rl_leases import IN_ZONE, TOO_HARD, LeaseBook, LeaseBookConfig

ENV = "math"
PARAMS = CooldownParams(min_first_scans=10, checkpoint_window=10, refresh_s=0.0)


def book_with(tmp_path, cooldown):
    book = LeaseBook(tmp_path / "leases.sqlite3",
                     LeaseBookConfig(expiry_s={ENV: 600.0}, prompt_range={ENV: (0, 1000)}, cooldown_of=cooldown),
                     quota=QuotaController(QuotaConfig(initial=4)), clock=lambda: 1.8e9, rng=random.Random(7))
    cooldown.attach(book)
    return book


def test_an_env_gets_the_initial_cooldown_until_it_is_measured(tmp_path):
    cooldown = AdaptiveCooldown(prompt_range={ENV: (0, 1000)}, params=PARAMS)
    book = book_with(tmp_path, cooldown)
    book.adopt_checkpoint(0)
    book.record_outcome(ENV, 1, 0, [1.0] * 8 + [0.0] * 8, IN_ZONE)
    assert cooldown(ENV) == PARAMS.initial and cooldown.state(ENV)["source"] == "initial"
    assert book.prompt_state()["prompts"][0]["until_k"] == PARAMS.initial


def test_the_cooldown_is_the_checkpoints_the_env_in_zone_prompts_last(tmp_path):
    cooldown = AdaptiveCooldown(prompt_range={ENV: (0, 1000)}, params=PARAMS)
    book = book_with(tmp_path, cooldown)
    prompt = 0
    for k in range(11):                                  # 10 checkpoints, 4 first scans each: half in zone
        book.adopt_checkpoint(k)
        for verdict in (IN_ZONE, IN_ZONE, TOO_HARD, TOO_HARD):
            book.record_outcome(ENV, prompt, k, [1.0] * 8 + [0.0] * 8 if verdict == IN_ZONE else [0.0] * 16, verdict)
            prompt += 1
    state = cooldown.state(ENV)
    # N=1000 prompts, p=0.5 in zone, Q=2 in-zone groups per checkpoint: 250 checkpoints of supply, x margin 0.5
    assert state["source"] == "measured" and state["p"] == pytest.approx(0.5) and state["q"] == pytest.approx(2.0)
    assert cooldown(ENV) == 125


def test_the_cooldown_is_bounded_and_an_operator_value_wins(tmp_path):
    tight = AdaptiveCooldown(prompt_range={ENV: (0, 10)}, params=PARAMS)
    book = book_with(tmp_path, tight)
    for k in range(11):
        book.adopt_checkpoint(k)
        for i in range(2):
            book.record_outcome(ENV, k * 2 + i, k, [1.0] * 8 + [0.0] * 8, IN_ZONE)
    assert tight(ENV) == PARAMS.floor                    # 10 prompts at 2 per checkpoint: 2.5, raised to the floor
    fixed = AdaptiveCooldown(prompt_range={ENV: (0, 10)}, overrides={ENV: 7}, params=PARAMS)
    assert fixed(ENV) == 7 and fixed.state(ENV)["source"] == "override"
    assert cooldown_overrides({"rl_cooldown_k": {ENV: 7}}) == {ENV: 7}
    for bad in ({ENV: -1}, {ENV: 1.5}, {ENV: True}, [7]):
        with pytest.raises(ValueError):
            cooldown_overrides({"rl_cooldown_k": bad})
    with pytest.raises(ValueError):
        CooldownParams(floor=200, initial=100)
