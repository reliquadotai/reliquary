"""Immutable, versioned protocol generation profiles.

Profiles live independently of ``reliquary.constants`` so a process can
advertise or validate an exact historical contract without inheriting the
currently deployed constants.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class SamplingProfile:
    rollouts: int
    temperature: float
    top_p: float
    top_k: int
    do_sample: bool


@dataclass(frozen=True, slots=True)
class BFTProfile:
    thinking_budget: int
    answer_budget: int
    force_answer: bool


@dataclass(frozen=True, slots=True)
class EnvironmentProfile:
    max_new_tokens: int
    bft: BFTProfile | None


@dataclass(frozen=True, slots=True)
class ThroughputTiebreakProfile:
    """Draw ordering among candidates of EQUAL difficulty.

    A tie-break never trades against training utility — difficulty ranks first,
    so this only orders submissions already judged equally useful. The question
    is which tie-break does least harm, and ordering by arrival round actively
    penalises long generation: a 16k-token rollout arrives later than a 500-token
    one and loses the slot at identical hardware. With binary rewards the
    difficulty score takes only nine values, so equal-difficulty ties are common,
    not marginal — the penalty applies broadly.

    Ordering by tokens-per-round is length-NEUTRAL: at equal hardware a long
    completion ranks the same as a short one. ``token_cap`` bounds the numerator
    so generating past the useful budget earns no rank, and because throughput is
    a rate rather than a total, padding adds tokens and time in step.

    INCOMPATIBLE WITH SPECULATIVE EARLY CLOSE. Sealing a window before its
    deadline requires that a leading candidate can no longer be overtaken, which
    held under arrival ordering — arriving later meant ranking later, full stop.
    Throughput ordering breaks that: a submission arriving later can still
    outrank an earlier one by serving faster, so leadership is not decided until
    the deadline. Proving mid-window would spend the bounded proof wall on
    candidates that later lose. (The final tiebreak also draws on seal randomness,
    which does not exist until the seal.) Dominance would only be provable for a
    candidate at both the value ceiling AND the maximum attainable throughput
    bucket, which is too rare to build a mechanism on. If early close is ever
    reconsidered, one of the two has to go.
    """

    token_cap: int
    bucket_tokens_per_round: int


@dataclass(frozen=True, slots=True)
class ProtocolProfile:
    profile_id: str
    model_id: str
    model_revision: str
    protocol_version: int
    collection_seconds: int
    upload_grace_seconds: int
    sampling: SamplingProfile
    environments: Mapping[str, EnvironmentProfile]
    throughput_tiebreak: ThroughputTiebreakProfile | None = None

    def __post_init__(self) -> None:
        # Copy before wrapping so caller-owned dictionaries cannot mutate a
        # profile after construction.
        object.__setattr__(
            self,
            "environments",
            MappingProxyType(dict(self.environments)),
        )

    def to_generation_contract(self) -> dict[str, Any]:
        """Return a detached contract containing only JSON-native values."""

        environments: dict[str, dict[str, Any]] = {}
        for name, environment in self.environments.items():
            bft = environment.bft
            environments[name] = {
                "max_new_tokens": environment.max_new_tokens,
                "bft": (
                    None
                    if bft is None
                    else {
                        "thinking_budget": bft.thinking_budget,
                        "answer_budget": bft.answer_budget,
                        "force_answer": bft.force_answer,
                    }
                ),
            }

        return {
            "profile_id": self.profile_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "protocol_version": self.protocol_version,
            "throughput_tiebreak": (
                None
                if self.throughput_tiebreak is None
                else {
                    "token_cap": self.throughput_tiebreak.token_cap,
                    "bucket_tokens_per_round": (
                        self.throughput_tiebreak.bucket_tokens_per_round
                    ),
                }
            ),
            "collection_seconds": self.collection_seconds,
            "upload_grace_seconds": self.upload_grace_seconds,
            "sampling": {
                "rollouts": self.sampling.rollouts,
                "temperature": self.sampling.temperature,
                "top_p": self.sampling.top_p,
                "top_k": self.sampling.top_k,
                "do_sample": self.sampling.do_sample,
            },
            "environments": environments,
        }


_SAMPLING = SamplingProfile(
    rollouts=8,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    do_sample=False,
)

_PROFILE_VALUES = (
    ProtocolProfile(
        profile_id="qwen35-2b-auction-v2",
        model_id="Qwen/Qwen3.5-2B",
        model_revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
        protocol_version=2,
        collection_seconds=100,
        upload_grace_seconds=33,
        sampling=_SAMPLING,
        environments={
            "openmathinstruct": EnvironmentProfile(
                max_new_tokens=32768,
                bft=BFTProfile(
                    thinking_budget=2048,
                    answer_budget=512,
                    force_answer=True,
                ),
            ),
            "opencodeinstruct": EnvironmentProfile(
                max_new_tokens=32768,
                bft=None,
            ),
        },
    ),
    ProtocolProfile(
        profile_id="qwen35-4b-auction-v3",
        model_id="Qwen/Qwen3.5-4B",
        model_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        protocol_version=3,
        collection_seconds=300,
        upload_grace_seconds=33,
        sampling=_SAMPLING,
        environments={
            "openmathinstruct": EnvironmentProfile(
                max_new_tokens=16384,
                bft=BFTProfile(
                    thinking_budget=15616,
                    answer_budget=512,
                    force_answer=True,
                ),
            ),
            "opencodeinstruct": EnvironmentProfile(
                max_new_tokens=32768,
                bft=None,
            ),
        },
        throughput_tiebreak=ThroughputTiebreakProfile(
            token_cap=15616,
            bucket_tokens_per_round=50,
        ),
    ),
)

PROFILES: Mapping[str, ProtocolProfile] = MappingProxyType(
    {profile.profile_id: profile for profile in _PROFILE_VALUES}
)
DEFAULT_PROFILE_ID = "qwen35-2b-auction-v2"
_PROFILE_ENV_VAR = "RELIQUARY_PROTOCOL_PROFILE"


def resolve_protocol_profile(profile_id: str | None = None) -> ProtocolProfile:
    """Resolve an explicit profile or the environment-selected default.

    Empty, misspelled, and otherwise unknown IDs are errors. Falling back after
    an explicit selection would silently put peers on different wire contracts.
    """

    selected_id = (
        os.environ.get(_PROFILE_ENV_VAR, DEFAULT_PROFILE_ID)
        if profile_id is None
        else profile_id
    )
    try:
        return PROFILES[selected_id]
    except KeyError as exc:
        available = ", ".join(PROFILES)
        raise ValueError(
            f"unknown protocol profile {selected_id!r}; "
            f"expected one of: {available}"
        ) from exc


ACTIVE_PROTOCOL_PROFILE = resolve_protocol_profile()


def to_generation_contract(
    profile: ProtocolProfile | str | None = None,
) -> dict[str, Any]:
    """Serialize a profile object, profile ID, or the active profile."""

    if profile is None:
        resolved = ACTIVE_PROTOCOL_PROFILE
    elif isinstance(profile, str):
        resolved = resolve_protocol_profile(profile)
    else:
        resolved = profile
    return resolved.to_generation_contract()


__all__ = [
    "ACTIVE_PROTOCOL_PROFILE",
    "BFTProfile",
    "DEFAULT_PROFILE_ID",
    "EnvironmentProfile",
    "PROFILES",
    "ProtocolProfile",
    "SamplingProfile",
    "resolve_protocol_profile",
    "to_generation_contract",
]
