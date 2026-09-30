"""Assemble a ``ProtocolProfile`` from a model, a run policy and environments.

Used only when a task is declared. The result is an ordinary profile, written
by the existing ``to_generation_contract``, so a composed contract has exactly
the shape every deployed reader already accepts. ``profiles`` never imports
this module, so resolving the active profile is unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from reliquary.protocol.environment_catalog import ENVIRONMENT_CATALOG
from reliquary.protocol.profiles import (
    PROFILES,
    EnvironmentProfile,
    ProofProfile,
    ProtocolProfile,
    SamplingProfile,
    ThroughputTiebreakProfile,
)

# What two corpus jobs must agree on to share a validator. `prompt_encoding`
# is on the spec but not here: corpus renders from its manifest instead.
MODEL_IDENTITY_FIELDS = ("model_id", "model_revision", "model_architecture", "proofs")

# The values `constants` accepts at import; refusing them here fails the
# declaration instead of every validator's startup.
_PROMPT_ENCODINGS = ("raw", "chat_template")
_MATH_ANSWER_FORMATS = ("boxed", "boxed_or_trailing_number")


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    model_revision: str
    model_architecture: str | None
    prompt_encoding: str
    proofs: tuple[ProofProfile, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "proofs", tuple(self.proofs))


@dataclass(frozen=True, slots=True)
class RunPolicy:
    protocol_version: int
    collection_seconds: int
    upload_grace_seconds: int
    sampling: SamplingProfile
    throughput_tiebreak: ThroughputTiebreakProfile | None


def model_spec_of(profile: ProtocolProfile) -> ModelSpec:
    return ModelSpec(
        model_id=profile.model_id,
        model_revision=profile.model_revision,
        model_architecture=profile.model_architecture,
        prompt_encoding=profile.prompt_encoding,
        proofs=profile.proofs,
    )


def run_policy_of(profile: ProtocolProfile) -> RunPolicy:
    return RunPolicy(
        protocol_version=profile.protocol_version,
        collection_seconds=profile.collection_seconds,
        upload_grace_seconds=profile.upload_grace_seconds,
        sampling=profile.sampling,
        throughput_tiebreak=profile.throughput_tiebreak,
    )


# Derived rather than restated, so a policy cannot drift from its profile.
# "corpus-v1" is the top level the live corpus-code-v1 task carries, which is
# teutonic-v9's field for field (pinned by the golden fixture).
RUN_POLICIES: Mapping[str, RunPolicy] = MappingProxyType({
    "dapo-v6": run_policy_of(PROFILES["qwen3-4b-base-dapo-reliquary-v1"]),
    "suite-v9": run_policy_of(PROFILES["teutonic-9b-reliquary-suite-v9-dev1"]),
    "episode-v7": run_policy_of(PROFILES["qwen3-4b-reliquary-episode-v7-dev1"]),
    "corpus-v1": run_policy_of(PROFILES["teutonic-9b-reliquary-suite-v9-dev1"]),
})


def check_profile_invariants(profile: ProtocolProfile) -> None:
    """The checks ``constants`` makes at import, without importing it."""
    if profile.prompt_encoding not in _PROMPT_ENCODINGS:
        raise ValueError(
            f"unknown prompt encoding {profile.prompt_encoding!r}; "
            f"expected one of {', '.join(_PROMPT_ENCODINGS)}"
        )
    math = profile.environments.get("openmathinstruct")
    if math is None:
        return
    if math.answer_format not in _MATH_ANSWER_FORMATS:
        raise ValueError(
            f"openmathinstruct answer format {math.answer_format!r} is not one "
            f"of {', '.join(_MATH_ANSWER_FORMATS)}"
        )
    bft = math.bft
    if bft is not None and math.max_new_tokens <= bft.thinking_budget + bft.answer_budget:
        raise ValueError(
            f"openmathinstruct cap {math.max_new_tokens} must exceed the BFT "
            f"budgets {bft.thinking_budget} + {bft.answer_budget}"
        )


def compose_profile(
    *,
    profile_id: str,
    model: ModelSpec,
    run: RunPolicy,
    environments: Iterable[str],
    catalog: Mapping[str, EnvironmentProfile] = ENVIRONMENT_CATALOG,
) -> ProtocolProfile:
    bodies = {name: catalog[name] for name in sorted(environments)}
    return ProtocolProfile(
        profile_id=profile_id,
        model_id=model.model_id,
        model_revision=model.model_revision,
        model_architecture=model.model_architecture,
        prompt_encoding=model.prompt_encoding,
        proofs=model.proofs,
        protocol_version=run.protocol_version,
        collection_seconds=run.collection_seconds,
        upload_grace_seconds=run.upload_grace_seconds,
        sampling=run.sampling,
        throughput_tiebreak=run.throughput_tiebreak,
        environments=bodies,
    )


__all__ = [
    "MODEL_IDENTITY_FIELDS",
    "RUN_POLICIES",
    "ModelSpec",
    "RunPolicy",
    "check_profile_invariants",
    "compose_profile",
    "model_spec_of",
    "run_policy_of",
]
