"""A hand-written correct solution must score exactly 1.0.

Every earlier defect in this port was found by inspecting model behaviour,
which is a slow and indirect way to learn that the grader is wrong: a
greedy parser, an episode killed by the first raising tool, and a binary
conjunctive reward that forced sigma to zero all looked like model failure
first. An oracle closes that loop directly — if a solution known to be
correct does not score 1.0, the environment is broken regardless of what
any model does.

The negative controls matter as much: a grader that accepts everything
would also pass the positive case.
"""

from __future__ import annotations

import os

import pytest

from reliquary.environment.agentic.types import AssistantAction, EpisodeTrace


pytestmark = pytest.mark.skipif(
    not os.environ.get("RELIQUARY_ENVSCALER_DATA"),
    reason="needs the pinned EnvScaler corpus",
)

# env_151_rl-task_1: a clinical-trials world. Farida Youssef is enrolled in a
# suspended trial, her consent reads "approved" rather than "valid", and a
# withdrawn CT-C3 record blocks re-enrolment.
TASK_INDEX = 5
PARTICIPANT = "b55eb788-9cfd-4a51-b62a-3414e972abc2"


def _environment():
    from reliquary.environment.agentic.envs.envscaler_tools_v1.environment import (
        EnvScalerToolsEnvironment,
    )
    return EnvScalerToolsEnvironment()


def _trace(environment, task, reason="finished"):
    return EpisodeTrace(
        schema="reliquary/episode/v1", environment=environment.name,
        task_id=task.id, seed=0, events=(), actions=(), tokens=(),
        assistant_spans=(), observation_digests=(),
        termination_reason=reason,
    )


def _solution(state):
    """The four calls that satisfy every check, read off the initial state."""
    world = state.initial
    suspended = next(
        key for key, value in world["clinical_trials"].items()
        if value.get("name") == "NeuroNova IV"
    )
    stale = next(
        key for key, value in world["enrollments"].items()
        if value["participant_id"] == PARTICIPANT
        and value["trial_id"] == "CT-C3"
    )
    return [
        ("withdraw_participant_from_trial",
         {"participant_id": PARTICIPANT, "trial_id": suspended}),
        ("update_participant_consent_status",
         {"participant_id": PARTICIPANT, "new_consent_status": "valid"}),
        ("delete_enrollment", {"enrollment_id": stale}),
        ("enroll_participant_in_trial",
         {"participant_id": PARTICIPANT, "trial_id": "CT-C3",
          "enrollment_date": "2024-06-01"}),
    ]


def _run(environment, task, calls):
    state = environment.reset(task, seed=0).state
    for name, arguments in calls:
        result = environment.step(
            task, state,
            AssistantAction(kind="tool", tool=name, arguments=arguments),
        )
        assert not result.done, f"{name} ended the episode: {result.events}"
    return state


def test_a_correct_solution_scores_one():
    environment = _environment()
    task = environment.get_task(TASK_INDEX)
    reference = environment.reset(task, seed=0).state
    state = _run(environment, task, _solution(reference))

    report = environment.grade(task, state, _trace(environment, task))
    assert report.reward == pytest.approx(1.0)
    assert report.success is True


@pytest.mark.parametrize("dropped", range(4))
def test_dropping_any_step_costs_reward(dropped):
    """Every call in the solution is load-bearing."""
    environment = _environment()
    task = environment.get_task(TASK_INDEX)
    reference = environment.reset(task, seed=0).state
    calls = _solution(reference)
    partial = [call for index, call in enumerate(calls) if index != dropped]

    state = _run(environment, task, partial)
    report = environment.grade(task, state, _trace(environment, task))
    assert report.reward < 1.0
    assert report.success is False


def test_doing_nothing_scores_zero():
    """Under required-only grading there is no credit for the initial state."""
    if os.environ.get("RELIQUARY_ENVSCALER_REWARD") == "upstream":
        pytest.skip("upstream grading pays for checks already true at reset")
    environment = _environment()
    task = environment.get_task(TASK_INDEX)
    state = environment.reset(task, seed=0).state
    report = environment.grade(task, state, _trace(environment, task))
    assert report.reward == pytest.approx(0.0)


def test_no_task_pays_for_its_initial_state():
    """A no-op must score zero on every task, not just the one above.

    Upstream averages all checks, so an inert agent averages 0.166 across
    the corpus. That credit is not for anything the agent did, and it
    narrows the range the sigma gate has to work with.
    """
    environment = _environment()
    if os.environ.get("RELIQUARY_ENVSCALER_REWARD") == "upstream":
        pytest.skip("upstream grading pays for checks already true at reset")
    for index in range(12):
        task = environment.get_task(index)
        state = environment.reset(task, seed=0).state
        report = environment.grade(task, state, _trace(environment, task))
        assert report.reward == pytest.approx(0.0), task.id


def test_the_episode_survives_a_failed_call():
    """A wrong argument is an observation, not the end of the run."""
    environment = _environment()
    task = environment.get_task(TASK_INDEX)
    reference = environment.reset(task, seed=0).state
    state = environment.reset(task, seed=0).state

    bad = environment.step(
        task, state,
        AssistantAction(kind="tool", tool="delete_enrollment",
                        arguments={"enrollment_id": "does-not-exist"}),
    )
    assert bad.done is False

    for name, arguments in _solution(reference):
        environment.step(
            task, state,
            AssistantAction(kind="tool", tool=name, arguments=arguments),
        )
    report = environment.grade(task, state, _trace(environment, task))
    assert report.reward == pytest.approx(1.0)
