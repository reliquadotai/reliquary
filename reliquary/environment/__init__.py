"""Reliquary environment module.

Provides the Environment protocol and a factory function to instantiate
concrete environments by name.
"""

from reliquary.environment.abi import (
    EnvironmentAdapter,
    EnvironmentManifest,
    EnvironmentRegistry,
    TaskEnvelope,
    TrajectoryEnvelope,
    TrajectoryEvent,
)
from reliquary.environment.base import Environment
from reliquary.environment.agentic.base import EpisodeEnvironment
from reliquary.environment.registry import get_environment_spec


def load_environment(name: str) -> Environment | EpisodeEnvironment:
    """Return a concrete Environment instance for the given *name*.

    Raises:
        ValueError: if *name* is not a recognised environment.
    """
    return get_environment_spec(name).create()


def load_environments(
    names: list[str],
) -> dict[str, Environment | EpisodeEnvironment]:
    """Return a dict {name: Environment} for each requested env.

    Raises ValueError if any name is not recognised. Single-env callers
    can keep using load_environment; multi-env callers (validator with
    ENVIRONMENT_MIX) use this.
    """
    return {name: load_environment(name) for name in names}


def load_episode_environment(name: str) -> EpisodeEnvironment:
    spec = get_environment_spec(name)
    if spec.interaction_mode != "episode":
        raise ValueError(f"environment {name!r} is not episode-based")
    environment = spec.create()
    if not isinstance(environment, EpisodeEnvironment):
        raise TypeError(f"environment {name!r} does not satisfy EpisodeEnvironment")
    return environment


__all__ = [
    "Environment",
    "EnvironmentAdapter",
    "EnvironmentManifest",
    "EnvironmentRegistry",
    "TaskEnvelope",
    "TrajectoryEnvelope",
    "TrajectoryEvent",
    "EpisodeEnvironment",
    "load_environment",
    "load_episode_environment",
    "load_environments",
]
