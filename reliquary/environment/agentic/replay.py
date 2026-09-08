"""Validator-side deterministic replay helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from reliquary.environment.agentic.base import EpisodeEnvironment
from reliquary.environment.agentic.renderer import CanonicalEpisodeRenderer
from reliquary.environment.agentic.runner import EpisodeRunner, ScriptedPolicy
from reliquary.environment.agentic.types import (
    AssistantAction,
    EpisodeTrace,
    GeneratedAction,
)


def replay_episode(
    environment: EpisodeEnvironment,
    *,
    task_index: int,
    seed: int,
    actions: Iterable[AssistantAction],
) -> EpisodeTrace:
    """Re-run an action sequence without a tokenizer or model dependency."""

    task = environment.get_task(task_index)
    return EpisodeRunner().run(
        environment,
        task,
        seed=seed,
        policy=ScriptedPolicy(actions),
    )


def replay_wire_actions(
    environment: EpisodeEnvironment,
    *,
    task_index: int,
    seed: int,
    actions: Iterable[dict],
) -> EpisodeTrace:
    return replay_episode(
        environment,
        task_index=task_index,
        seed=seed,
        actions=[AssistantAction.from_wire(action) for action in actions],
    )


class _TokenizedReplayPolicy:
    def __init__(self, generated: Iterable[GeneratedAction]) -> None:
        self._generated = iter(generated)

    def generate(self, **_: object) -> GeneratedAction:
        try:
            return next(self._generated)
        except StopIteration:
            return GeneratedAction(text='{"final":"replay exhausted"}')


def replay_tokenized_episode(
    environment: EpisodeEnvironment,
    *,
    task_index: int,
    seed: int,
    tokens: list[int],
    assistant_spans: Iterable[tuple[int, int]],
    decode: Callable[[list[int]], str],
    encode: Callable[[str], list[int]],
    max_episode_tokens: int | None = None,
    max_observation_bytes: int = 64 * 1024,
) -> EpisodeTrace:
    """Replay while reconstructing the exact canonical flattened transcript."""

    generated = []
    for start, end in assistant_spans:
        piece = list(tokens[int(start):int(end)])
        generated.append(GeneratedAction(text=decode(piece), tokens=tuple(piece)))
    return EpisodeRunner(
        renderer=CanonicalEpisodeRenderer(encode),
        max_episode_tokens=max_episode_tokens,
        max_observation_bytes=max_observation_bytes,
    ).run(
        environment,
        environment.get_task(task_index),
        seed=seed,
        policy=_TokenizedReplayPolicy(generated),
    )
