"""reliquary-swe served through reliquary-sandbox's verifiers bridge: what a signed-sandbox
SWE job needs from the env (prompt, task id, image, limits, record-0 identity), read from
the same `VerifiersEnv` a gateway serves, so miner, validator and gateway agree on the
task. Replaces the removed `reliquary_swe.sandbox` adapter. `reliquary_sandbox_*` and
`reliquary_swe` are optional: imported inside functions only."""

from __future__ import annotations

import functools
import importlib.metadata
from typing import Any

PACKAGE = "reliquary-swe"
SERVED_SPLITS = {"train": {"split": "train"}, "r2e": {"split": "r2e"},
                 "polyglot": {"split": "polyglot"}}
GATEWAY_OPTIONS = {"tools": ["bash", "edit"], "splits": SERVED_SPLITS,
                   "defaults": {"per_call_timeout_s": 600}}
"""The options this process builds reliquary-swe with, which must be the ones the gateway
serves it with (`episode_env_options`): the image and limits a session token names are
computed with them. The gateway is the authority: its capacity report publishes the
digest of the options in effect, and no session is placed on a machine whose digest is
not `options_sha256()` (`SandboxFleet.pick`)."""


@functools.cache
def options_sha256() -> str:
    """The digest a gateway serving reliquary-swe with GATEWAY_OPTIONS publishes in its
    capacity report (`env_options_sha256`), computed by the sandbox's own code."""
    from reliquary_sandbox_verifiers.loading import options_sha256 as digest

    return digest(GATEWAY_OPTIONS)


@functools.cache
def _env(version: str) -> Any:
    from reliquary_sandbox_verifiers.loading import VerifiersEnv

    return VerifiersEnv(f"verifiers:{PACKAGE}=={version}", GATEWAY_OPTIONS)


def bridged_env() -> Any:
    return _env(importlib.metadata.version(PACKAGE))


def prompt_of(split: str, index: int) -> str:
    return bridged_env().task(split, int(index)).data.prompt


def row_of(split: str, index: int) -> tuple[None, Any]:
    return None, bridged_env().task(split, int(index)).data


def sandbox_task_of(split: str, index: int) -> Any:
    return bridged_env().sandbox_task(split, int(index))


def installed_env_package(package: str = PACKAGE) -> str:
    from reliquary_sandbox_service.episodes import bridged_env_package_of

    return bridged_env_package_of(package.replace("-", "_"))
