"""Reliquary environment module.

Provides the Environment protocol and a factory function to instantiate
concrete environments by name.
"""

from reliquary.environment.base import Environment
from reliquary.environment.openmathinstruct import OpenMathInstructEnvironment
from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment


def load_environment(name: str) -> Environment:
    """Return a concrete Environment instance for the given *name*.

    Raises:
        ValueError: if *name* is not a recognised environment.
    """
    if name == "openmathinstruct":
        return OpenMathInstructEnvironment()
    if name == "opencodeinstruct":
        return OpenCodeInstructEnvironment()
    raise ValueError(f"Unknown environment: {name}")


def load_environments(names: list[str]) -> dict[str, Environment]:
    """Return a dict {name: Environment} for each requested env.

    Raises ValueError if any name is not recognised. Single-env callers
    can keep using load_environment; multi-env callers (validator with
    ENVIRONMENT_MIX) use this.
    """
    return {name: load_environment(name) for name in names}


__all__ = [
    "Environment",
    "load_environment",
    "load_environments",
]
