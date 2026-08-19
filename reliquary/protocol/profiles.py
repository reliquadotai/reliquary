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
    answer_format: str | None = None


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
    prompt_encoding: str
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
                "answer_format": environment.answer_format,
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
            "prompt_encoding": self.prompt_encoding,
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

# DAPO/verl reference rollout sampling: full temperature, full support. Beyond
# matching the recipe, this is what makes warp() the identity softmax, so the
# PPO importance ratio is formed in the distribution the samples actually came
# from (the v3 values leave it distorted by r_raw^(1/T) and truncated to a
# 20-token nucleus). top_k is 0, not None: warp() guards on `top_k and
# top_k > 0`, but the miner's ForcedSeedLogitsProcessor coerces with int(top_k)
# and raises on None.
_SAMPLING_DAPO = SamplingProfile(
    # G=16, DAPO §4.1. With binary rewards the group mean and std are exact
    # functions of k, so the estimator's error lives entirely in k/G as a
    # binomial estimate of p — and the advantage √((1−p̂)/p̂) is non-linear in
    # it, so that error is a BIAS, which more prompts cannot average away.
    # Dynamic sampling makes this sharper: it admits k=1 and k=G−1 precisely
    # where small-G bias is worst.
    rollouts=16,
    temperature=1.0,
    top_p=1.0,
    top_k=0,
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
        prompt_encoding="chat_template",
        sampling=_SAMPLING,
        environments={
            "openmathinstruct": EnvironmentProfile(
                max_new_tokens=32768,
                answer_format="boxed_or_trailing_number",
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
        prompt_encoding="chat_template",
        sampling=_SAMPLING,
        environments={
            "openmathinstruct": EnvironmentProfile(
                max_new_tokens=16384,
                answer_format="boxed_or_trailing_number",
                bft=BFTProfile(
                    thinking_budget=15616,
                    answer_budget=512,
                    force_answer=True,
                ),
            ),
            "opencodeinstruct": EnvironmentProfile(
                max_new_tokens=16384,
                bft=None,
            ),
        },
        throughput_tiebreak=ThroughputTiebreakProfile(
            token_cap=15616,
            bucket_tokens_per_round=50,
        ),
    ),
    ProtocolProfile(
        profile_id="qwen3-4b-base-dapo-v4",
        # True base model: 0/32 spontaneous <think> under a raw prompt, against
        # 27/32 for Qwen3.5-4B-Base. Recent "-Base" releases are mid-trained on
        # reasoning traces, which spends the dispersion RL exists to convert.
        model_id="Qwen/Qwen3-4B-Base",
        model_revision="906bfd4b4dc7f14ee4320094d8b41684abff8539",
        protocol_version=4,
        # Length curriculum, start point. ck0 Qwen3-4B-Base terminates well
        # short of 16384 (measured on real OMI: median ~500, max ~1392 / 40
        # rollouts), so sizing the window for a 16384 worst case burns wall-clock
        # and seal-verify time from day one. Start the cap at 8192 (≈6× the
        # observed ck0 max, and above OVERLONG_PENALTY_CACHE_TOKENS=4096 so the
        # soft-overlong zone [cap-4096, cap] still sits ABOVE the natural length)
        # The cap is meant to ramp up with the policy's growing reasoning
        # length; watch the cap-hit rate as the thermostat before raising it.
        #
        # The window is now sized from measured arrivals rather than from the
        # cap. Over w29400-29440 (2175 submissions, R2 archives) submissions
        # land at median 25s / p95 67s / p99 93s / max 126s for math and
        # median 16s / p99 76s for code — nothing at all arrives between 126s
        # and the old 150s deadline, so that tail was pure dead air. 100s sits
        # just above the p99 and drops 0.7% of math submissions (0% of code),
        # spread thinly: per-hotkey medians all fall in 16-35s and the highest
        # -volume miners lose 0-1.5%, so no hardware class is excluded. The
        # measured window cycle is 385s median (collection 39%, seal proofs
        # 22%, train+archive 40%), so this returns ~15% more windows per hour.
        collection_seconds=100,
        upload_grace_seconds=33,
        prompt_encoding="raw",
        sampling=_SAMPLING_DAPO,
        environments={
            # No BFT anywhere: the base model emits no <think>, so there is no
            # block to force closed. Termination is trained by the soft overlong
            # punishment (Eq. 13) instead of forced by a budget.
            "openmathinstruct": EnvironmentProfile(
                max_new_tokens=8192,
                bft=None,
                answer_format="boxed",
            ),
            "opencodeinstruct": EnvironmentProfile(
                max_new_tokens=8192,
                bft=None,
            ),
        },
        throughput_tiebreak=ThroughputTiebreakProfile(
            # Without BFT the per-rollout generation budget is the env cap
            # itself, where v3 had to use its thinking budget. Tracks the cap.
            token_cap=8192,
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
