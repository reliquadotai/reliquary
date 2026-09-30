"""Assemble a ``ProtocolProfile`` from a model, a run policy and environments.

Used only when a task is declared. The result is an ordinary profile, written
by the existing ``to_generation_contract``, so a composed contract has exactly
the shape every deployed reader already accepts. ``profiles`` never imports
this module, so resolving the active profile is unchanged.
"""

from __future__ import annotations

import math

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from reliquary.protocol.environment_catalog import ENVIRONMENT_CATALOG, TUNABLE_FIELDS
from reliquary.protocol.profiles import (
    PROFILES,
    EnvironmentProfile,
    ProofProfile,
    ProtocolProfile,
    SamplingProfile,
    ThroughputTiebreakProfile,
    profile_from_contract,
)
from reliquary.protocol.release_contract import canonical_sha256

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


def _check_sampling(sampling: SamplingProfile) -> None:
    """Bounds every compiled profile sits inside; top_k 0 means no top-k cut."""
    if sampling.rollouts < 1:
        raise ValueError(f"sampling rollouts must be at least 1, got {sampling.rollouts}")
    if not math.isfinite(sampling.temperature) or sampling.temperature <= 0:
        raise ValueError(f"sampling temperature must be finite and positive, got {sampling.temperature}")
    if not math.isfinite(sampling.top_p) or not 0 < sampling.top_p <= 1:
        raise ValueError(f"sampling top_p must be in (0, 1], got {sampling.top_p}")
    if sampling.top_k < 0:
        raise ValueError(f"sampling top_k must be 0 (disabled) or positive, got {sampling.top_k}")


def check_profile_invariants(profile: ProtocolProfile) -> None:
    """The checks ``constants`` makes at import, without importing it."""
    if profile.prompt_encoding not in _PROMPT_ENCODINGS:
        raise ValueError(
            f"unknown prompt encoding {profile.prompt_encoding!r}; "
            f"expected one of {', '.join(_PROMPT_ENCODINGS)}"
        )
    _check_sampling(profile.sampling)
    for name, environment in sorted(profile.environments.items()):
        episode = environment.episode
        if episode is not None and episode.max_action_tokens > episode.max_episode_tokens:
            raise ValueError(
                f"environment {name!r} episode max_action_tokens {episode.max_action_tokens} "
                f"exceeds max_episode_tokens {episode.max_episode_tokens}"
            )
    omi = profile.environments.get("openmathinstruct")
    if omi is None:
        return
    if omi.answer_format not in _MATH_ANSWER_FORMATS:
        raise ValueError(
            f"openmathinstruct answer format {omi.answer_format!r} is not one "
            f"of {', '.join(_MATH_ANSWER_FORMATS)}"
        )
    bft = omi.bft
    if bft is not None and omi.max_new_tokens <= bft.thinking_budget + bft.answer_budget:
        raise ValueError(
            f"openmathinstruct cap {omi.max_new_tokens} must exceed the BFT "
            f"budgets {bft.thinking_budget} + {bft.answer_budget}"
        )


def _with_overrides(
    name: str, body: EnvironmentProfile, fields: Mapping[str, object]
) -> EnvironmentProfile:
    refused = sorted(set(fields) - TUNABLE_FIELDS)
    if refused:
        raise ValueError(
            f"environment {name!r}: {', '.join(refused)} cannot be overridden; "
            f"tunable fields are {', '.join(sorted(TUNABLE_FIELDS))}"
        )
    top = {k: v for k, v in fields.items() if not k.startswith("episode.")}
    episode = {
        k.removeprefix("episode."): v for k, v in fields.items() if k.startswith("episode.")
    }
    if episode:
        if body.episode is None:
            raise ValueError(f"environment {name!r} is not an episode environment")
        top["episode"] = replace(body.episode, **episode)
    return replace(body, **top)


def _check_against_spec(name: str, body: EnvironmentProfile) -> None:
    """The pure part of ``resolve_environment_mix``: the wheel need not be here."""
    # Imported here so the protocol layer stays importable without environment code.
    from reliquary.environment.registry import ENVIRONMENT_SPECS

    spec = ENVIRONMENT_SPECS.get(name)
    if spec is None:
        raise ValueError(f"environment {name!r} is not installed in this image")
    if (
        body.environment_contract_id is not None
        and body.environment_contract_id != spec.contract_version
    ):
        raise ValueError(
            f"environment {name!r} contract {body.environment_contract_id!r} "
            f"does not match installed {spec.contract_version!r}"
        )
    if (
        body.environment_manifest_sha256 is not None
        and body.environment_manifest_sha256 != spec.environment_manifest_sha256
    ):
        raise ValueError(f"environment {name!r} manifest does not match installed code")
    if (
        body.episode is not None
        and spec.renderer_id is not None
        and body.episode.renderer_id != spec.renderer_id
    ):
        raise ValueError(f"environment {name!r} renderer does not match installed code")


def _check_fixed_point(profile: ProtocolProfile) -> None:
    """Refuse a profile whose contract reads back as different bytes.

    ``1 == 1.0`` in Python, so dict equality would pass an int temperature;
    the canonical bytes, which the digests hash, do not.
    """
    contract = profile.to_generation_contract()
    try:
        rebuilt = profile_from_contract(contract).to_generation_contract()
    except ValueError as exc:
        raise ValueError(
            f"profile {profile.profile_id!r} does not survive its own round trip: {exc}"
        ) from exc
    if canonical_sha256(rebuilt) != canonical_sha256(contract):
        raise ValueError(
            f"profile {profile.profile_id!r} does not survive its own round trip: "
            "a value changes type when read back (an int where a float belongs?)"
        )


def compose_profile(
    *,
    profile_id: str,
    model: ModelSpec,
    run: RunPolicy,
    environments: Iterable[str],
    overrides: Mapping[str, Mapping[str, object]] | None = None,
    catalog: Mapping[str, EnvironmentProfile] = ENVIRONMENT_CATALOG,
) -> ProtocolProfile:
    """An ordinary profile from its parts, or a ``ValueError`` naming the part."""
    names = list(environments)
    if not names:
        raise ValueError("at least one environment must be selected")
    if len(set(names)) != len(names):
        raise ValueError("environment selection contains duplicate names")
    overrides = dict(overrides or {})
    unselected = sorted(set(overrides) - set(names))
    if unselected:
        raise ValueError(f"overrides name environments not selected: {', '.join(unselected)}")
    bodies = {}
    for name in sorted(names):
        if name not in catalog:
            raise ValueError(
                f"environment {name!r} has no catalog entry; add and review one first"
            )
        body = _with_overrides(name, catalog[name], overrides.get(name, {}))
        _check_against_spec(name, body)
        bodies[name] = body
    profile = ProtocolProfile(
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
    check_profile_invariants(profile)
    _check_fixed_point(profile)
    return profile


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
