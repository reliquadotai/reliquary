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


def _sleepy(environment, problem, texts, materials):
    time.sleep(float(problem))
    return [0.0] * len(texts)


def test_a_group_queued_behind_an_overrun_is_never_cancelled_into_a_reward_error():
    """Production 2026-10-22: one competitive-code group overran its 20 s, the pool was replaced, and the groups
    queued in it came back as CancelledError, which the grader turned into ``reward_error`` against their miners.
    Now a group waits for a slot before its clock starts, and whatever the pool raises is the grader's outage."""
    import threading

    scorer = PooledScorer(["env"], workers=1, function=_sleepy, wall_seconds=lambda env: 2.0)
    outcomes: dict[str, object] = {}

    def run(name, seconds):
        try:
            outcomes[name] = scorer("env", seconds, ["a"], None)
        except Exception as exc:                                   # noqa: BLE001  the test records the type
            outcomes[name] = exc

    try:
        slow = threading.Thread(target=run, args=("slow", 30))
        slow.start()
        time.sleep(0.5)
        queued = [threading.Thread(target=run, args=(f"q{i}", 0.2)) for i in range(3)]
        for thread in queued:
            thread.start()
        for thread in (slow, *queued):
            thread.join(timeout=60)
        assert isinstance(outcomes["slow"], GraderInfrastructureError)
        for i in range(3):                     # waited for the slot, then scored on a fresh pool within their wall
            assert outcomes[f"q{i}"] == [0.0], outcomes
        assert scorer.timeouts == {"env": 1}
    finally:
        scorer.close()


def test_a_cancelled_or_shut_down_pool_is_the_graders_outage(monkeypatch):
    from concurrent.futures import CancelledError, Future

    scorer = PooledScorer(["env"], workers=1, function=_score, wall_seconds=lambda env: 5.0)
    try:
        pool = scorer._pool("env")
        cancelled: Future = Future()
        cancelled.cancel()
        monkeypatch.setattr(pool, "submit", lambda *a, **k: cancelled)
        with pytest.raises(GraderInfrastructureError):
            scorer("env", "a", ["a"], None)
        with pytest.raises(CancelledError):
            cancelled.result()

        def shut(*args, **kwargs):
            raise RuntimeError("cannot schedule new futures after shutdown")

        monkeypatch.setattr(pool, "submit", shut)
        with pytest.raises(GraderInfrastructureError):
            scorer("env", "a", ["a"], None)
        assert scorer("env", "a", ["a"], None) == [1.0]                # replaced: a fresh pool serves the next
    finally:
        scorer.close()


def test_a_stdio_program_group_gets_wall_per_program_run_up_to_a_ceiling():
    from reliquary.constants import (
        CODE_ADMISSION_WALL_SECONDS, STDIO_ADMISSION_MAX_WALL_SECONDS, STDIO_ADMISSION_SECONDS_PER_RUN,
    )

    env = "reliquary_competitive_code_v1"
    scorer = PooledScorer([env], workers=1)
    tests = [{"stdin": "", "stdout": "", "time_limit_s": 1.0, "output_cap": 1}]
    assert scorer.budget(env, ["a"] * 16, tests * 10) == pytest.approx(
        CODE_ADMISSION_WALL_SECONDS + STDIO_ADMISSION_SECONDS_PER_RUN * 160)
    assert scorer.budget(env, ["a"] * 16, tests * 400) == STDIO_ADMISSION_MAX_WALL_SECONDS
    assert scorer.budget("openmathinstruct", ["a"] * 16, None) == 45.0          # other envs: unchanged


def test_the_async_validator_starts_the_grader_service_only_for_code_envs():
    from reliquary.cli.main import _envs_need_grader

    assert _envs_need_grader(["openmathinstruct", "opencodeinstruct"])
    assert not _envs_need_grader(["openmathinstruct"])
    assert not _envs_need_grader(["swe_v1"])                  # a bridged episode env: scored by its sandbox


RECYCLING = r"""
import bittensor  # noqa: F401  the parent imports it too, as the validator does
from reliquary.validator.rl_submit import PooledScorer

def score(environment, problem, texts, materials):
    import bittensor  # a grader that imports bittensor starts its logging listener in the worker
    return [0.0] * len(texts)

if __name__ == "__main__":
    scorer = PooledScorer(["env"], workers=1, max_tasks=1, function=score, wall_seconds=lambda env: 60.0)
    for _ in range(4):                                     # every call on a fresh worker: three retirements
        assert scorer("env", "p", ["a"], None) == [0.0]
    scorer.close()
    print("ok", flush=True)
"""


def test_retired_grading_workers_take_their_logging_listener_down_cleanly(tmp_path):
    import subprocess
    import sys

    script = tmp_path / "recycle.py"
    script.write_text(RECYCLING)
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=180,
                          cwd=os.getcwd(), env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert done.returncode == 0 and "ok" in done.stdout, done.stderr[-2000:]
    assert "EOFError" not in done.stderr and "Exception in thread" not in done.stderr, done.stderr[-2000:]
