"""Tripwires on the plan 2C (signed episode) entry points ONLY, for the suites of a single-turn v2 order: they run
the phase 1 service stack on purpose, and must never reach an episode entry point.

``pytest -p tests.unit.episode_tripwires <suite>`` arms ``EPISODE_ARMED`` (raise and record) and
``EPISODE_CONDITIONAL`` (record when the episode branch is taken) of ``tests/unit/rl_tripwires.py`` for every test of
the suite, and fails a test afterwards if anything was recorded (``test_episode_inertness`` reruns the suites so).
"""
from __future__ import annotations

import pytest

from tests.unit import rl_tripwires as _wires


@pytest.fixture(autouse=True)
def episode_tripwires(monkeypatch):
    calls: list[str] = []
    _wires.arm_episode(monkeypatch, calls)
    yield calls
    assert not calls, f"signed-episode entry points were reached: {sorted(set(calls))}"
