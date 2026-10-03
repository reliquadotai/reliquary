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

__all__ = ["AGENTIC_BODIES", "AGENTIC_SWE_BODY", "AGENTIC_SWE_CONTRACT_ID",
           "AGENTIC_SWE_ENVIRONMENT", "AGENTIC_SWE_MANIFEST", "AGENTIC_SWE_MANIFEST_SHA256"]
