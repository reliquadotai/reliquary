"""Rows of an RL range that RL never draws (``rl_prompt_ranges.rl_excluded_prompts``): EnvScaler's RL worlds no
policy wins yet. The miner never picks one; the validator refuses a lease of one."""
from __future__ import annotations

import os
import random
from collections import Counter
from types import SimpleNamespace

import pytest

from reliquary.environment.agentic.envs.envscaler_tools_v2 import corpus as layout
from reliquary.environment.agentic.envs.envscaler_tools_v2.environment import EnvScalerToolsV2Environment
from reliquary.miner.rl_async import PromptPlanner
from reliquary.protocol.rl_prompt_ranges import RL_EXCLUDED_WORLDS, rl_excluded_prompts
from reliquary.validator.rl_cooldown import AdaptiveCooldown
from reliquary.validator.rl_lease_quota import QuotaConfig, QuotaController
from reliquary.validator.rl_leases import LeaseBook, LeaseBookConfig, LeaseRequest

ENV = "envscaler_tools_v2"
LENGTH = 7825                                   # the split RL_PROMPT_RANGES cut envscaler_tools_v2 against
WORLDS = sorted(RL_EXCLUDED_WORLDS[ENV][0])


class _Env:
    """``len`` and ``scenario(row).env_id``: what ``rl_excluded_prompts`` reads of an environment."""

    def __init__(self, world_of, length=LENGTH):
        self._world_of, self._length = world_of, length

    def __len__(self):
        return self._length

    def scenario(self, row):
        return SimpleNamespace(env_id=self._world_of(row))


@pytest.fixture(autouse=True)
def _real_ranges(real_rl_prompt_ranges):
    """The real RL ranges (the unit conftest offers every env its whole split otherwise)."""


def test_only_the_rl_rows_of_the_excluded_worlds_are_excluded():
    # Rows past the RL range [0, 1296) are SFT rows: never read, whatever world they name.
    env = _Env(lambda row: WORLDS[0] if row in (0, 7, 1295, 1296, 5000) else "env_141_rl")
    assert rl_excluded_prompts(ENV, env) == frozenset({0, 7, 1295})


def test_an_env_without_an_entry_excludes_nothing():
    assert rl_excluded_prompts("reliquary_hard_math_v1", _Env(lambda row: WORLDS[0], length=23_227)) == frozenset()


def test_the_exclusion_fails_closed():
    with pytest.raises(ValueError, match="every row"):
        rl_excluded_prompts(ENV, _Env(lambda row: WORLDS[0]))
    with pytest.raises(ValueError, match="cut against"):
        rl_excluded_prompts(ENV, _Env(lambda row: "env_141_rl", length=LENGTH + 1))


def test_the_validator_refuses_a_lease_of_an_excluded_row(tmp_path):
    book = LeaseBook(tmp_path / "leases.sqlite3",
                     LeaseBookConfig(expiry_s={ENV: 600.0}, prompt_range={ENV: (0, 1296)},
                                     prompt_excluded={ENV: frozenset({3})}),
                     quota=QuotaController(QuotaConfig(initial=4)), clock=lambda: 1.8e9, rng=random.Random(7))
    book.adopt_checkpoint(0)
    with pytest.raises(ValueError, match="excluded"):
        book.check_request(LeaseRequest(hotkey="A", environment=ENV, prompt_ids=(2, 3), checkpoint=0))
    book.check_request(LeaseRequest(hotkey="A", environment=ENV, prompt_ids=(2, 4), checkpoint=0))


def test_the_miner_never_picks_an_excluded_row():
    excluded = frozenset(range(0, 20, 2))
    planner = PromptPlanner(ENV, (0, 20), rng=random.Random(0), excluded=excluded)
    picks = {p for _ in range(50) for p in planner.pick(8)}
    assert picks and not picks & excluded
    assert not planner.leasable(4) and planner.leasable(5)


def test_the_cooldown_counts_only_the_rows_rl_draws():
    cooldown = AdaptiveCooldown(prompt_range={ENV: (0, 1296)}, excluded={ENV: 396})
    assert cooldown.state(ENV)["n"] == 900


@pytest.fixture(scope="module")
def real_corpus():
    if not os.environ.get(layout.DATA_ENV_VAR):
        pytest.skip(f"{layout.DATA_ENV_VAR} not set")
    return layout.default_corpus()


def test_the_real_corpus_excludes_the_eight_worlds_and_keeps_eighteen(real_corpus):
    env = EnvScalerToolsV2Environment(corpus=real_corpus)
    rows = rl_excluded_prompts(ENV, env)
    worlds = Counter(real_corpus.scenarios[row].env_id for row in rows)
    assert set(worlds) == set(WORLDS) and len(rows) == 396
    kept = {real_corpus.scenarios[row].env_id for row in range(1296) if row not in rows}
    assert len(kept) == 18 and not kept & set(WORLDS)
    assert set(WORLDS) <= set(layout.read_selection()["world_split"]["rl"])
