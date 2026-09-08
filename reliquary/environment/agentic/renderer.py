"""Canonical, append-only text rendering for Reliquary Episode v1."""

from __future__ import annotations

from collections.abc import Callable

from reliquary.environment.agentic.types import (
    AssistantAction,
    EpisodeEvent,
    EpisodeTask,
    canonical_json,
)


EPISODE_RENDERER_ID = "reliquary-jsonl-tools-v1"


class CanonicalEpisodeRenderer:
    """A tokenizer-neutral renderer whose previous tokens are never re-encoded."""

    id = EPISODE_RENDERER_ID

    def __init__(self, encode: Callable[[str], list[int]]) -> None:
        self._encode = encode

    @staticmethod
    def initial_text(task: EpisodeTask) -> str:
        tools = [tool.to_wire() for tool in task.tools]
        return (
            "<|reliquary_system|>\n"
            "You are operating a deterministic tool environment. Respond with "
            "exactly one JSON action per turn: "
            '{"tool":"name","arguments":{...}} or {"final":"answer"}.\n'
            f"TOOLS={canonical_json(tools)}\n"
            "<|reliquary_user|>\n"
            f"{task.prompt}\n"
            "<|reliquary_assistant|>\n"
        )

    @staticmethod
    def observation_text(events: tuple[EpisodeEvent, ...]) -> str:
        rendered = "\n<|reliquary_end|>\n"
        for event in events:
            rendered += (
                f"<|reliquary_{event.role}|>\n"
                f"{canonical_json(event.to_wire())}\n"
            )
        return rendered + "<|reliquary_assistant|>\n"

    @staticmethod
    def final_suffix(action: AssistantAction) -> str:
        del action
        return "\n<|reliquary_end|>\n"

    def encode_initial(self, task: EpisodeTask) -> list[int]:
        return list(self._encode(self.initial_text(task)))

    def encode_observation(self, events: tuple[EpisodeEvent, ...]) -> list[int]:
        return list(self._encode(self.observation_text(events)))

    def encode_action(self, text: str) -> list[int]:
        return list(self._encode(text))

    def encode_final_suffix(self, action: AssistantAction) -> list[int]:
        return list(self._encode(self.final_suffix(action)))
