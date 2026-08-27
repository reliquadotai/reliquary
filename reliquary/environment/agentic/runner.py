"""One understandable rollout loop shared by all episode environments."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Protocol

from reliquary.environment.agentic.base import EpisodeEnvironment
from reliquary.environment.agentic.renderer import CanonicalEpisodeRenderer
from reliquary.environment.agentic.types import (
    AssistantAction,
    EpisodeEvent,
    EpisodeTask,
    EpisodeTrace,
    GeneratedAction,
    RewardReport,
    canonical_json,
    sha256_json,
)


class EpisodePolicy(Protocol):
    def generate(
        self,
        *,
        task: EpisodeTask,
        events: tuple[EpisodeEvent, ...],
        tokens: tuple[int, ...],
        turn: int,
    ) -> GeneratedAction: ...


class ScriptedPolicy:
    """Dependency-free policy used for replay, qualification, and CPU tests."""

    def __init__(self, actions: Iterable[AssistantAction]) -> None:
        self._actions = iter(actions)

    def generate(self, **_: object) -> GeneratedAction:
        try:
            action = next(self._actions)
        except StopIteration:
            action = AssistantAction.final("script exhausted")
        return GeneratedAction(text=action.to_json())


class EpisodeRunner:
    def __init__(
        self,
        *,
        renderer: CanonicalEpisodeRenderer | None = None,
        max_turns: int | None = None,
        max_episode_tokens: int | None = None,
        max_observation_bytes: int = 64 * 1024,
    ) -> None:
        self.renderer = renderer
        self.max_turns = max_turns
        self.max_episode_tokens = max_episode_tokens
        self.max_observation_bytes = int(max_observation_bytes)

    def run(
        self,
        environment: EpisodeEnvironment,
        task: EpisodeTask,
        *,
        seed: int,
        policy: EpisodePolicy,
    ) -> EpisodeTrace:
        reset = environment.reset(task, seed)
        state = reset.state
        events: list[EpisodeEvent] = [
            EpisodeEvent(role="user", content=task.prompt),
            *reset.events,
        ]
        actions: list[AssistantAction] = []
        observation_digests: list[str] = []
        tokens: list[int] = []
        assistant_spans: list[tuple[int, int]] = []
        assistant_logprobs: list[float] = []
        if self.renderer is not None:
            tokens.extend(self.renderer.encode_initial(task))
        if (
            self.max_episode_tokens is not None
            and len(tokens) > self.max_episode_tokens
        ):
            raise ValueError("episode prompt exceeds total token budget")

        termination_reason = "turn_limit"
        reward: RewardReport | None = None
        limit = int(self.max_turns or environment.max_turns)
        try:
            for turn in range(limit):
                generated = policy.generate(
                    task=task,
                    events=tuple(events),
                    tokens=tuple(tokens),
                    turn=turn,
                )
                try:
                    action = AssistantAction.from_json(generated.text)
                except (TypeError, ValueError):
                    action = AssistantAction.tool_call(
                        "__invalid_action__", raw=generated.text
                    )
                actions.append(action)
                events.append(
                    EpisodeEvent(
                        role="assistant",
                        content=generated.text,
                        action_index=turn,
                    )
                )
                if self.renderer is not None:
                    generated_tokens = list(generated.tokens)
                    if not generated_tokens:
                        generated_tokens = self.renderer.encode_action(generated.text)
                    start = len(tokens)
                    tokens.extend(generated_tokens)
                    assistant_spans.append((start, len(tokens)))
                    if generated.token_logprobs:
                        assistant_logprobs.extend(generated.token_logprobs)
                    if (
                        self.max_episode_tokens is not None
                        and len(tokens) > self.max_episode_tokens
                    ):
                        raise ValueError("episode exceeds total token budget")

                result = environment.step(task, state, action)
                state = result.state
                observation_bytes = len(canonical_json(
                    [event.to_wire() for event in result.events]
                ).encode("utf-8"))
                if observation_bytes > self.max_observation_bytes:
                    raise ValueError("episode observation exceeds byte budget")
                events.extend(result.events)
                observation_digests.append(
                    sha256_json([event.to_wire() for event in result.events])
                )
                if result.done:
                    termination_reason = result.termination_reason or "environment_done"
                    if self.renderer is not None:
                        tokens.extend(self.renderer.encode_final_suffix(action))
                    if (
                        self.max_episode_tokens is not None
                        and len(tokens) > self.max_episode_tokens
                    ):
                        raise ValueError("episode exceeds total token budget")
                    break
                if self.renderer is not None:
                    tokens.extend(self.renderer.encode_observation(result.events))
                if (
                    self.max_episode_tokens is not None
                    and len(tokens) > self.max_episode_tokens
                ):
                    raise ValueError("episode exceeds total token budget")

            trace = EpisodeTrace(
                environment=environment.name,
                task_id=task.id,
                seed=seed,
                events=tuple(events),
                actions=tuple(actions),
                assistant_spans=tuple(assistant_spans),
                tokens=tuple(tokens),
                assistant_logprobs=tuple(assistant_logprobs),
                observation_digests=tuple(observation_digests),
                termination_reason=termination_reason,
            )
            reward = environment.grade(task, state, trace)
            return replace(trace, reward=reward)
        finally:
            environment.close(state)
