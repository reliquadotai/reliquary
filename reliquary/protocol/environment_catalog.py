"""How a newly declared task uses each environment by default.

Each body is restated from the compiled profile named in
``CATALOG_PROVENANCE``; a test holds them equal and pins every digest. The
compiled profiles keep their own literals: this catalog is read only when a
task is declared, never by a running validator.

A model that needs another budget gets a per-task override, not a new default.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from reliquary.protocol.profiles import (
    EnvironmentProfile,
    EpisodeProfile,
    PromptTemplateProfile,
)

# The only fields a task may override. Everything else is what the environment
# is; `bft` is left out because only legacy thinking-model profiles use it.
TUNABLE_FIELDS = frozenset({
    "max_new_tokens",
    "thinking",
    "batch_target",
    "prompt_cooldown_windows",
    "episode.max_turns",
    "episode.max_action_tokens",
    "episode.max_episode_tokens",
})

_EXTERNAL_PROMPT = PromptTemplateProfile("reliquary-external-prompt-v1", "$problem")


def _jsonl_tools_episode(max_turns: int) -> EpisodeProfile:
    return EpisodeProfile(
        schema="reliquary/episode/v1",
        renderer_id="reliquary-jsonl-tools-v1",
        max_turns=max_turns,
        max_action_tokens=1024,
        max_episode_tokens=16384,
        max_observation_bytes=65536,
    )


ENVIRONMENT_CATALOG: Mapping[str, EnvironmentProfile] = MappingProxyType({
    "openmathinstruct": EnvironmentProfile(
        max_new_tokens=8192,
        bft=None,
        answer_format="boxed",
        prompt_template=PromptTemplateProfile(
            "openmathinstruct-step-by-step-v1",
            "Solve the following math problem step by step.\n\n"
            "$problem\n\n"
            "Put your final answer within \\boxed{}.",
        ),
    ),
    "opencodeinstruct": EnvironmentProfile(
        max_new_tokens=8192,
        bft=None,
        prompt_template=PromptTemplateProfile(
            "opencodeinstruct-step-by-step-v1",
            "Solve the following programming problem step by step.\n\n"
            "$problem$contract\n\n"
            "After your reasoning, provide the final implementation in the last "
            "fenced Python code block.",
        ),
    ),
    "reliquary_logic_v2": EnvironmentProfile(
        max_new_tokens=8192,
        bft=None,
        answer_format="last_json_object_v1",
        batch_target=16,
        prompt_template=_EXTERNAL_PROMPT,
        environment_contract_id="reliquary/answer-json/v1",
        environment_manifest_sha256=(
            "1e4e05cae799d8e71d8876b0f7526c5b09ca1d5a9ab05f364fb35539288c5019"
        ),
    ),
    "reliquaryverifiable_v1": EnvironmentProfile(
        max_new_tokens=1024,
        bft=None,
        answer_format="last_json_object_v1",
        prompt_template=PromptTemplateProfile("reliquary-records-v1", "$problem"),
        batch_target=16,
        environment_contract_id="reliquary-records-v1",
        environment_manifest_sha256=(
            "d0d5d838e40b383d1c95a62d1cdded8458f4a7b62df621c87c9435b62207929b"
        ),
    ),
    "reliquary_stateful_tools_v1": EnvironmentProfile(
        max_new_tokens=16384,
        bft=None,
        answer_format="episode_json_action_v1",
        batch_target=16,
        environment_contract_id="reliquary-stateful-tools-v1",
        environment_manifest_sha256=(
            "0f490881544ba065bf33b974032adbc3f844d2c3978bcd6ca8dbb7089baa8f18"
        ),
        episode=_jsonl_tools_episode(max_turns=8),
    ),
    "reliquary_retrieval_tools_v1": EnvironmentProfile(
        max_new_tokens=16384,
        bft=None,
        answer_format="episode_json_action_v1",
        batch_target=16,
        environment_contract_id="reliquary-retrieval-tools-v1",
        environment_manifest_sha256=(
            "1c53afdf6acc59dd7df0693b7486e47de94d79977404841d1368ffb2571c0c7d"
        ),
        episode=_jsonl_tools_episode(max_turns=6),
    ),
    "reliquary_workspace_tools_v1": EnvironmentProfile(
        max_new_tokens=16384,
        bft=None,
        answer_format="episode_json_action_v1",
        batch_target=16,
        environment_contract_id="reliquary-workspace-tools-v1",
        environment_manifest_sha256=(
            "7f0465cff80aefc489e0302d21222728115e0094df33858d6c613fa5423489e2"
        ),
        episode=_jsonl_tools_episode(max_turns=7),
    ),
    "reliquarylogic_v1": EnvironmentProfile(
        max_new_tokens=8192,
        bft=None,
        answer_format="last_json_object_v1",
        prompt_template=PromptTemplateProfile(
            "reliquary-logic-step-by-step-v1",
            "Solve the following problem step by step.\n\n"
            "$problem\n\n"
            "After your reasoning, give the final answer in the last fenced JSON "
            "code block.",
        ),
        batch_target=16,
        environment_contract_id="reliquary-logic-v1",
        environment_manifest_sha256=(
            "9cb29e487321b2e6c005f2a1a89ccffecf01b1c09bfd03337094e338ab912ca9"
        ),
    ),
    "reliquary_dapo_math_v1": EnvironmentProfile(
        max_new_tokens=32768,
        bft=None,
        answer_format="boxed",
        batch_target=8,
        prompt_template=_EXTERNAL_PROMPT,
        environment_contract_id="reliquary/boxed-answer/v1",
        environment_manifest_sha256=(
            "cca437d73e8183a6df4af4780e00e34cd26fed03238b18208898b1e2586d2035"
        ),
        prompt_cooldown_windows=1741,
    ),
    "reliquary_instruction_following_v1": EnvironmentProfile(
        max_new_tokens=8192,
        bft=None,
        answer_format="text",
        batch_target=16,
        prompt_template=_EXTERNAL_PROMPT,
        environment_contract_id="reliquary/checked-answer/v1",
        environment_manifest_sha256=(
            "84b8698446e68e057faea54ea4045cd198c21fb6bd0a7c8fc259a7659c4e483d"
        ),
        prompt_cooldown_windows=1839,
        thinking=False,
    ),
    "reliquary_code_v1": EnvironmentProfile(
        max_new_tokens=8192,
        bft=None,
        answer_format="fenced_python",
        batch_target=16,
        prompt_template=_EXTERNAL_PROMPT,
        environment_contract_id="reliquary/python-cases/v1",
        environment_manifest_sha256=(
            "71e4f23c614b7f321bbb9f6cf74f98b772137442bdf9960e5b6d9b6387f41216"
        ),
    ),
    "reliquary_telecom_solo_v1": EnvironmentProfile(
        max_new_tokens=49152,
        bft=None,
        answer_format="episode_json_action_v1",
        batch_target=4,
        environment_contract_id="reliquary/episode-json/v1",
        environment_manifest_sha256=(
            "74d0e7569247eabc3d4fb2d909773f2b1f1d8430a4ce4f4fd09c6ad6e6794b05"
        ),
        prompt_cooldown_windows=60,
        episode=EpisodeProfile(
            schema="reliquary/episode/v1",
            renderer_id="reliquary-chatml-tools-v1",
            max_turns=40,
            max_action_tokens=4096,
            max_episode_tokens=49152,
            max_observation_bytes=65536,
        ),
    ),
})

# Environment -> the compiled profile its body was taken from: the newest one
# declaring it, which for OMI/OCI is also the first with a prompt template.
CATALOG_PROVENANCE: Mapping[str, str] = MappingProxyType({
    "openmathinstruct": "qwen3-4b-base-dapo-reliquary-v1",
    "opencodeinstruct": "qwen3-4b-base-dapo-reliquary-v1",
    "reliquary_logic_v2": "qwen3-4b-base-dapo-reliquary-v1",
    "reliquaryverifiable_v1": "qwen3-4b-reliquary-verifiable-v6-dev1",
    "reliquary_stateful_tools_v1": "qwen3-4b-reliquary-episode-v7-dev1",
    "reliquary_retrieval_tools_v1": "qwen3-4b-reliquary-episode-v7-dev1",
    "reliquary_workspace_tools_v1": "qwen3-4b-reliquary-episode-v7-dev1",
    "reliquarylogic_v1": "qwen3-4b-reliquary-logic-v8-dev1",
    "reliquary_dapo_math_v1": "teutonic-9b-reliquary-suite-v9-dev1",
    "reliquary_instruction_following_v1": "teutonic-9b-reliquary-suite-v9-dev1",
    "reliquary_code_v1": "teutonic-9b-reliquary-suite-v9-dev1",
    "reliquary_telecom_solo_v1": "teutonic-9b-reliquary-suite-v9-dev1",
})


__all__ = ["CATALOG_PROVENANCE", "ENVIRONMENT_CATALOG", "TUNABLE_FIELDS"]
