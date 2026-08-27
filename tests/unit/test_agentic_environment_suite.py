from __future__ import annotations

import pytest

from reliquary.environment.agentic.renderer import CanonicalEpisodeRenderer
from reliquary.environment.agentic.replay import replay_tokenized_episode
from reliquary.environment.agentic.runner import EpisodeRunner, ScriptedPolicy
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

