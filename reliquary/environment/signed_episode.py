"""A signed-sandbox episode environment as a v2 batcher holds it.

The batcher needs an ``Environment``: a name, a length and ``get_problem`` (the task's prompt identity,
for the content cooldown and the admission's prompt materials). The reward never comes from here: it is
the signed final record's, verified at admission (``reliquary.validator.episode_admission``), and
``compute_reward`` refuses. The task source supplies the prompt exactly as the harness renders
it, and each task's pinned image and declared limits."""
from __future__ import annotations

import hashlib
from typing import Protocol

from reliquary.sandbox.tasks import ResolvedTask


class EpisodeTaskSource(Protocol):
    def __len__(self) -> int: ...

    def prompt(self, index: int) -> str: ...

    async def resolve(self, index: int) -> ResolvedTask: ...


class SignedEpisodeEnvironment:
    validator_authoritative_reward = True
    # The registry entry's mode, repeated for the batcher's fallback on a name the registry does not know (a
    # test env); admission reads the registry only and refuses an unregistered env's groups.
    interaction_mode = "signed_episode"

    def __init__(self, name: str, source: EpisodeTaskSource) -> None:
        self.name = name
        self.source = source

    def __len__(self) -> int:
        return len(self.source)

    def get_problem(self, index: int) -> dict:
        prompt = self.source.prompt(int(index))
        return {"prompt": prompt, "ground_truth": "",
                "id": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16], "task_index": int(index)}

    def compute_reward(self, problem: dict, completion: str) -> float:
        raise TypeError("a signed episode's reward is its signed final record's")


def signed_episode_scorer(problem, completion_texts, reward_materials=None):
    """The registry's scorer path for a signed-episode env: never called (the final record decides)."""
    raise TypeError("a signed episode is scored by its final record, never from text")


__all__ = ["EpisodeTaskSource", "SignedEpisodeEnvironment", "signed_episode_scorer"]
