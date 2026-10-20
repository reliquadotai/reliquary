"""The async submission grader's scoring isolation (``rl_submit.PooledScorer``) and the grader service start of the
async validator (``cli.main._envs_need_grader``)."""
from __future__ import annotations

import os
import time

import pytest

from reliquary.environment.grader_client import GraderInfrastructureError
from reliquary.validator.rl_submit import PooledScorer


def _score(environment, problem, texts, materials):
    return [1.0 if t == problem else 0.0 for t in texts]


def _slow(environment, problem, texts, materials):
    time.sleep(30)
    return [0.0] * len(texts)


def _dies(environment, problem, texts, materials):
    os._exit(3)


def test_a_group_is_scored_in_its_environments_pool():
    scorer = PooledScorer(["env"], workers=1, function=_score, wall_seconds=lambda env: 30.0)
    try:
        assert scorer("env", "a", ["a", "b"], None) == [1.0, 0.0]
        assert scorer("env", "b", ["a", "b"], None) == [0.0, 1.0]                # the pool is reused
        assert scorer.restarts == {} and scorer.timeouts == {}
    finally:
        scorer.close()


def test_a_group_that_overruns_its_wall_budget_is_the_graders_outage_and_the_pool_is_replaced():
    scorer = PooledScorer(["env"], workers=1, function=_slow, wall_seconds=lambda env: 1.0)
    try:
        started = time.monotonic()
        with pytest.raises(GraderInfrastructureError):
            scorer("env", "a", ["a"], None)
        assert time.monotonic() - started < 20                                    # never waits for the worker
        assert scorer.timeouts == {"env": 1} and scorer.restarts == {"env": 1}
        scorer.function = _score
        assert scorer("env", "a", ["a"], None) == [1.0]                           # a fresh pool serves the next
    finally:
        scorer.close()


def test_a_worker_that_dies_is_the_graders_outage():
    scorer = PooledScorer(["env"], workers=1, function=_dies, wall_seconds=lambda env: 30.0)
    try:
        with pytest.raises(GraderInfrastructureError):
            scorer("env", "a", ["a"], None)
        assert scorer.restarts == {"env": 1}
    finally:
        scorer.close()


def test_the_async_validator_starts_the_grader_service_only_for_code_envs():
    from reliquary.cli.main import _envs_need_grader

    assert _envs_need_grader(["openmathinstruct", "opencodeinstruct"])
    assert not _envs_need_grader(["openmathinstruct"])
    assert not _envs_need_grader(["swe_v1"])                  # a bridged episode env: scored by its sandbox
