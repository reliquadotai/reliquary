from __future__ import annotations

import pytest

from reliquary.environment.agentic.renderer import CanonicalEpisodeRenderer
from reliquary.environment.agentic.replay import replay_tokenized_episode
from reliquary.environment.agentic.runner import EpisodeRunner, ScriptedPolicy
from reliquary.environment.agentic.types import AssistantAction, GeneratedAction
from reliquary.environment.registry import get_environment_spec


EPISODE_ENVS = (
    "reliquary_stateful_tools_v1",
    "reliquary_retrieval_tools_v1",
    "reliquary_workspace_tools_v1",
)


def _encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def _decode(tokens: list[int]) -> str:
    return bytes(tokens).decode("utf-8")


@pytest.mark.parametrize("name", EPISODE_ENVS)
def test_reference_episode_is_deterministic_and_replayable(name: str):
    spec = get_environment_spec(name)
    env = spec.create()
    task = env.get_task(3)
    runner = EpisodeRunner(renderer=CanonicalEpisodeRenderer(_encode))
    trace = runner.run(
        env,
        task,
        seed=41,
        policy=ScriptedPolicy(task.private["reference_actions"]),
    )

    assert trace.reward is not None
    assert trace.reward.reward == 1.0
    assert trace.reward.success is True
    assert len(trace.actions) == len(trace.assistant_spans)
    assert len(trace.actions) == len(trace.observation_digests)

    replay = replay_tokenized_episode(
        spec.create(),
        task_index=3,
        seed=41,
        tokens=list(trace.tokens),
        assistant_spans=trace.assistant_spans,
        decode=_decode,
        encode=_encode,
    )
    assert replay.tokens == trace.tokens
    assert replay.assistant_spans == trace.assistant_spans
    assert replay.trace_digest == trace.trace_digest
    assert replay.reward == trace.reward


@pytest.mark.parametrize("name", EPISODE_ENVS)
def test_task_generation_is_stable(name: str):
    env = get_environment_spec(name).create()
    first = env.get_task(11)
    second = env.get_task(11)
    assert first.to_public_wire() == second.to_public_wire()
    assert first.private == second.private


@pytest.mark.parametrize(
    ("name", "expected_families"),
    [
        (
            "reliquary_stateful_tools_v1",
            {"address_update", "refund", "support_note"},
        ),
        (
            "reliquary_retrieval_tools_v1",
            {"single_evidence", "multi_hop_alias", "revision_resolution"},
        ),
        (
            "reliquary_workspace_tools_v1",
            {
                "arithmetic_operator",
                "boundary_condition",
                "filter_predicate",
                "string_normalization",
            },
        ),
    ],
)
def test_initial_curriculum_covers_every_frozen_family(
    name: str,
    expected_families: set[str],
):
    env = get_environment_spec(name).create()
    observed = {
        str(env.get_task(index).metadata["family"])
        for index in range(12)
    }
    assert observed == expected_families


@pytest.mark.parametrize("name", EPISODE_ENVS)
def test_control_plane_problem_preserves_generated_task_family(name: str):
    environment = get_environment_spec(name).create()
    task = environment.get_task(1)
    assert environment.get_problem(1)["task_family"] == task.metadata["family"]


def test_replay_detects_a_noncanonical_observation_token():
    spec = get_environment_spec("reliquary_retrieval_tools_v1")
    env = spec.create()
    task = env.get_task(0)
    trace = EpisodeRunner(renderer=CanonicalEpisodeRenderer(_encode)).run(
        env,
        task,
        seed=0,
        policy=ScriptedPolicy(task.private["reference_actions"]),
    )
    tampered = list(trace.tokens)
    first_end = trace.assistant_spans[0][1]
    tampered[first_end + 1] ^= 1
    replay = replay_tokenized_episode(
        spec.create(),
        task_index=0,
        seed=0,
        tokens=tampered,
        assistant_spans=trace.assistant_spans,
        decode=_decode,
        encode=_encode,
    )
    assert tuple(tampered) != replay.tokens


def test_stateful_rejects_allowed_mutation_on_a_distractor():
    env = get_environment_spec("reliquary_stateful_tools_v1").create()
    task = env.get_task(0)  # address_update
    actions = list(task.private["reference_actions"])
    actions.insert(
        2,
        AssistantAction.tool_call(
            "update_shipping_address",
            order_id=task.private["other_order_id"],
            address="1 Collateral Damage Road",
        ),
    )
    trace = EpisodeRunner().run(
        env,
        task,
        seed=0,
        policy=ScriptedPolicy(actions),
    )
    assert trace.reward is not None
    assert trace.reward.reward == 0.0
    assert trace.reward.success is False
    assert not next(
        check for check in trace.reward.checks
        if check.name == "database_state_exact"
    ).passed


def test_retrieval_requires_exact_answer_and_one_exact_citation():
    env = get_environment_spec("reliquary_retrieval_tools_v1").create()
    task = env.get_task(0)
    target = task.private["expected_citations"][0]
    answer = task.private["answer"]
    prefix_answer = list(task.private["reference_actions"][:-1]) + [
        AssistantAction.tool_call(
            "finish",
            response=f"The code is {answer}",
            citations=[target],
        )
    ]
    duplicate_citation = list(task.private["reference_actions"][:-1]) + [
        AssistantAction.tool_call(
            "finish",
            response=answer,
            citations=[target, target],
        )
    ]
    for actions in (prefix_answer, duplicate_citation):
        trace = EpisodeRunner().run(
            env,
            task,
            seed=0,
            policy=ScriptedPolicy(actions),
        )
        assert trace.reward is not None
        assert trace.reward.reward == 0.0
        assert trace.reward.success is False


def test_workspace_partial_success_is_binary_failure():
    env = get_environment_spec("reliquary_workspace_tools_v1").create()
    task = env.get_task(0)
    actions = [
        action
        for action in task.private["reference_actions"]
        if action.tool != "run_tests"
    ]
    trace = EpisodeRunner().run(
        env,
        task,
        seed=0,
        policy=ScriptedPolicy(actions),
    )
    assert trace.reward is not None
    assert trace.reward.reward == 0.0
    assert trace.reward.success is False
    assert any(check.passed for check in trace.reward.checks)


def test_invalid_model_text_becomes_a_bounded_replayable_action():
    class InvalidThenFinal:
        def __init__(self):
            self.turn = 0

        def generate(self, **_):
            self.turn += 1
            if self.turn == 1:
                return GeneratedAction(text="not-json" * 1024)
            return GeneratedAction(text=AssistantAction.final("incorrect").to_json())

    env = get_environment_spec("reliquary_retrieval_tools_v1").create()
    task = env.get_task(0)
    trace = EpisodeRunner().run(
        env,
        task,
        seed=0,
        policy=InvalidThenFinal(),
    )
    assert trace.actions[0].tool == "__invalid_action__"
    assert len(trace.actions) == 1
    assert len(trace.actions[0].to_json().encode("utf-8")) < 1024
    assert trace.termination_reason == "invalid_action"
    assert trace.reward is not None and trace.reward.reward == 0.0


@pytest.mark.parametrize(
    "environment_name",
    [
        "reliquary_stateful_tools_v1",
        "reliquary_retrieval_tools_v1",
        "reliquary_workspace_tools_v1",
    ],
)
def test_invalid_action_is_an_immediate_binary_failure(environment_name):
    env = get_environment_spec(environment_name).create()
    task = env.get_task(0)
    trace = EpisodeRunner().run(
        env,
        task,
        seed=0,
        policy=ScriptedPolicy([AssistantAction.tool_call("unknown")]),
    )
    assert len(trace.actions) == 1
    assert trace.actions[0].tool == "unknown"
    assert trace.termination_reason == "invalid_action"
    assert trace.reward is not None
    assert trace.reward.reward == 0.0
    assert trace.reward.success is False
