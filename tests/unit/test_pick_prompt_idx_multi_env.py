"""Tests for the multi-env prompt selection logic in the miner engine."""

import random
import pytest


class _FakeEnv:
    """Minimal Environment stub: name + len + get_problem stub."""
    def __init__(self, name: str, size: int):
        self.name = name
        self._size = size
    def __len__(self):
        return self._size
    def get_problem(self, idx):
        return {"prompt": f"{self.name}-{idx}", "ground_truth": "", "id": "x" * 16}


def test_pick_env_and_prompt_returns_env_and_idx():
    from reliquary.miner.engine import pick_env_and_prompt
    envs = {
        "openmathinstruct": _FakeEnv("openmathinstruct", 100),
        "opencodeinstruct": _FakeEnv("opencodeinstruct", 50),
    }
    mix = [("openmathinstruct", 8), ("opencodeinstruct", 8)]
    cooldown = {name: set() for name in envs}
    rng = random.Random(0)
    env_name, idx = pick_env_and_prompt(envs, mix, cooldown, rng=rng)
    assert env_name in {"openmathinstruct", "opencodeinstruct"}
    assert 0 <= idx < len(envs[env_name])


def test_pick_env_and_prompt_respects_weights():
    """With weights 1:9, the rare env should be chosen far less often."""
    from reliquary.miner.engine import pick_env_and_prompt
    envs = {"a": _FakeEnv("a", 10), "b": _FakeEnv("b", 10)}
    mix = [("a", 1), ("b", 9)]
    cooldown = {"a": set(), "b": set()}
    rng = random.Random(42)
    counts = {"a": 0, "b": 0}
    for _ in range(1000):
        env_name, _ = pick_env_and_prompt(envs, mix, cooldown, rng=rng)
        counts[env_name] += 1
    # 1:9 weights → roughly 100:900. Allow generous slack.
    assert 70 < counts["a"] < 150
    assert 850 < counts["b"] < 950


def test_pick_env_and_prompt_skips_env_in_full_cooldown():
    """If one env is fully in cooldown, sampling still works on the other."""
    from reliquary.miner.engine import pick_env_and_prompt
    envs = {"a": _FakeEnv("a", 5), "b": _FakeEnv("b", 5)}
    mix = [("a", 1), ("b", 1)]
    cooldown = {"a": set(range(5)), "b": set()}  # 'a' fully blocked
    rng = random.Random(0)
    for _ in range(50):
        env_name, idx = pick_env_and_prompt(envs, mix, cooldown, rng=rng)
        assert env_name == "b"  # never 'a'
