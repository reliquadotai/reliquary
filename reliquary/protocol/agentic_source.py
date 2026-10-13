"""The environment an agentic corpus job declares in its contract.

A multi-turn job's rows are SWE-smith tasks of the `reliquary-swe` package at
the commit its manifest's `episode.env.version` names, rendered by the
manifest's `episode.renderer` over the bash harness's system turn and tools,
and graded by grade executors. None of that is a catalog environment, so this
body is the contract's statement of it, as `external_eval` is for eval sets;
no RL task may select it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType

from reliquary.protocol.profiles import EnvironmentProfile, PromptTemplateProfile

AGENTIC_SWE_ENVIRONMENT = "reliquary_agentic_swe_v1"
AGENTIC_SWE_CONTRACT_ID = "reliquary/agentic-swe/v1"
AGENTIC_SWE_MANIFEST = {
    "contract": AGENTIC_SWE_CONTRACT_ID,
    "rows": "the SWE-smith tasks of reliquary-swe at the commit episode.env.version names",
    "rendering": "episode.renderer over the bash harness system turn, its bash and edit tools, "
                 "and the task prompt; one assistant span per model turn",
    "grading": "grade executors (scope grade) run reliquary-swe's grader and the replay, never the control",
}
AGENTIC_SWE_MANIFEST_SHA256 = hashlib.sha256(json.dumps(
    AGENTIC_SWE_MANIFEST, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
AGENTIC_SWE_BODY = EnvironmentProfile(
    max_new_tokens=8192,
    bft=None,
    answer_format="text",
    prompt_template=PromptTemplateProfile("reliquary-agentic-swe-prompt-v1", "$problem"),
    environment_contract_id=AGENTIC_SWE_CONTRACT_ID,
    environment_manifest_sha256=AGENTIC_SWE_MANIFEST_SHA256,
)
AGENTIC_BODIES: Mapping[str, EnvironmentProfile] = MappingProxyType(
    {AGENTIC_SWE_ENVIRONMENT: AGENTIC_SWE_BODY})

__all__ = ["AGENTIC_BODIES", "AGENTIC_SWE_BODY", "AGENTIC_SWE_CONTRACT_ID", "suite_episode_body",
           "AGENTIC_SWE_ENVIRONMENT", "AGENTIC_SWE_MANIFEST", "AGENTIC_SWE_MANIFEST_SHA256"]


def suite_episode_body(name: str, episode) -> EnvironmentProfile:
    """The contract body of a suite episode environment no profile declares
    (design 2026-10-13): what the registry spec pins (contract, manifest,
    renderer) and the job's own episode bounds. ``episode`` is the manifest's
    raw ``episode`` object; one that does not parse is refused here as it
    would be by ``parse_job``."""
    from reliquary.corpus.job import _parse_episode
    from reliquary.environment.registry import ENVIRONMENT_SPECS
    from reliquary.protocol.profiles import EpisodeProfile

    parsed = _parse_episode(episode)
    spec = ENVIRONMENT_SPECS.get(name)
    if spec is None or spec.interaction_mode != "episode":
        raise ValueError(f"{name!r} is not an episode environment of this binary's registry")
    return EnvironmentProfile(
        max_new_tokens=parsed.max_tokens_per_turn,
        bft=None,
        answer_format="episode_json_action_v1",
        environment_contract_id=spec.contract_version,
        environment_manifest_sha256=spec.environment_manifest_sha256,
        episode=EpisodeProfile(
            schema="reliquary/episode/v1",
            renderer_id=parsed.renderer,
            max_turns=parsed.max_turns,
            max_action_tokens=parsed.max_tokens_per_turn,
            max_episode_tokens=parsed.max_total_tokens,
            max_observation_bytes=65536,
        ),
    )
