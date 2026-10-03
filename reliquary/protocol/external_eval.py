"""The environment a job on an external eval set declares in its contract.

A job's contract names one environment, composed from the catalog. A set built
from an external benchmark (a Verifiers taskset) has no catalog environment:
its rows come frozen from the set, rendered only by the model's own chat
template, and nothing that serves the job reads an environment's rows. The
contract still has to say so, honestly and in one place; this body is that
statement. It is not in `ENVIRONMENT_CATALOG`, whose bodies are pinned to the
compiled profiles they were taken from, and no RL task may select it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType

from reliquary.protocol.profiles import EnvironmentProfile, PromptTemplateProfile

EXTERNAL_EVAL_ENVIRONMENT = "reliquary_external_eval_v1"
EXTERNAL_EVAL_CONTRACT_ID = "reliquary/external-eval/v1"
# What the environment is, as the manifest its sha256 names: there is no
# package, so the statement itself is the thing pinned.
EXTERNAL_EVAL_MANIFEST = {
    "contract": EXTERNAL_EVAL_CONTRACT_ID,
    "rows": "the eval set's prompts.jsonl lines the job's eval-set prompt source names",
    "rendering": "the model's own chat template: an optional system turn, then one user turn",
    "grading": "the set's grading.jsonl on the admin host, never by the job's parties",
}
EXTERNAL_EVAL_MANIFEST_SHA256 = hashlib.sha256(json.dumps(
    EXTERNAL_EVAL_MANIFEST, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
EXTERNAL_EVAL_BODY = EnvironmentProfile(
    max_new_tokens=32768,
    bft=None,
    answer_format="text",
    prompt_template=PromptTemplateProfile("reliquary-external-prompt-v1", "$problem"),
    environment_contract_id=EXTERNAL_EVAL_CONTRACT_ID,
    environment_manifest_sha256=EXTERNAL_EVAL_MANIFEST_SHA256,
)
EXTERNAL_BODIES: Mapping[str, EnvironmentProfile] = MappingProxyType(
    {EXTERNAL_EVAL_ENVIRONMENT: EXTERNAL_EVAL_BODY})


def contract_environment_for(card: Mapping) -> str:
    """The environment an eval job on this set declares: its catalog source,
    or the external eval environment for a set built from a Verifiers taskset."""
    if card.get("source_kind") == "verifiers":
        return EXTERNAL_EVAL_ENVIRONMENT
    return card["source"]


__all__ = [
    "EXTERNAL_BODIES",
    "EXTERNAL_EVAL_BODY",
    "EXTERNAL_EVAL_CONTRACT_ID",
    "EXTERNAL_EVAL_ENVIRONMENT",
    "EXTERNAL_EVAL_MANIFEST",
    "EXTERNAL_EVAL_MANIFEST_SHA256",
    "contract_environment_for",
]
