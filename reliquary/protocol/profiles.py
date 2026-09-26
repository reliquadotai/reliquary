"""Immutable, versioned protocol generation profiles.

Profiles live independently of ``reliquary.constants`` so a process can
advertise or validate an exact historical contract without inheriting the
currently deployed constants.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from string import Template
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
class EpisodeProfile:
    schema: str
    renderer_id: str
    max_turns: int
    max_action_tokens: int
    max_episode_tokens: int
    max_observation_bytes: int

    def __post_init__(self) -> None:
        if self.schema != "reliquary/episode/v1":
            raise ValueError("unsupported episode schema")
        if not self.renderer_id:
            raise ValueError("episode renderer id must not be empty")
        if int(self.max_turns) <= 0:
            raise ValueError("episode max_turns must be positive")
        if int(self.max_action_tokens) <= 0:
            raise ValueError("episode max_action_tokens must be positive")
        if int(self.max_episode_tokens) <= 0:
            raise ValueError("episode max_episode_tokens must be positive")
        if int(self.max_observation_bytes) <= 0:
            raise ValueError("episode max_observation_bytes must be positive")


@dataclass(frozen=True, slots=True)
class PromptTemplateProfile:
    """Exact prompt text and rendering rule for a protocol environment.

    The template uses ``string.Template`` dollar placeholders so literal
    mathematical braces (for example ``\\boxed{}``) cannot be interpreted as
    formatting fields. Only ``$problem`` and ``$contract`` are legal, and the
    problem placeholder is mandatory.
    """

    template_id: str
    template: str

    def __post_init__(self) -> None:
        parsed = Template(self.template)
        if not self.template_id:
            raise ValueError("prompt template id must not be empty")
        # ``Template.is_valid/get_identifiers`` arrived in Python 3.11. The
        # project requires 3.11+, but this small fallback keeps lightweight
        # qualification hosts on 3.10 able to inspect contracts.
        if hasattr(parsed, "is_valid"):
            valid = parsed.is_valid()
            identifiers = set(parsed.get_identifiers())
        else:  # pragma: no cover - compatibility host only
            valid = True
            identifiers = set()
            for match in parsed.pattern.finditer(parsed.template):
                if match.group("invalid") is not None:
                    valid = False
                identifier = match.group("named") or match.group("braced")
                if identifier is not None:
                    identifiers.add(identifier)
        if not valid:
            raise ValueError(
                f"prompt template {self.template_id!r} is not valid"
            )
        unknown = identifiers - {"problem", "contract"}
        if unknown:
            raise ValueError(
                f"prompt template {self.template_id!r} has unknown "
                f"placeholders: {', '.join(sorted(unknown))}"
            )
        if "problem" not in identifiers:
            raise ValueError(
                f"prompt template {self.template_id!r} must contain $problem"
            )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.template.encode("utf-8")).hexdigest()

    def render(self, *, problem: str, contract: str = "") -> str:
        return Template(self.template).substitute(
            problem=problem,
            contract=contract,
        )

    def to_generation_contract(self) -> dict[str, str]:
        return {
            "id": self.template_id,
            "renderer": "dollar-substitution-v1",
            "template": self.template,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class EnvironmentProfile:
    max_new_tokens: int
    bft: BFTProfile | None
    answer_format: str | None = None
    prompt_template: PromptTemplateProfile | None = None
    # Optional per-environment selected-group target. Historical profiles omit
    # it and retain the protocol's legacy B_BATCH value byte-for-byte.
    batch_target: int | None = None
    # Consensus identity for generated/verifier-backed environments. Omitted
    # from historical profiles so v2-v5 generation contracts do not change.
    environment_contract_id: str | None = None
    environment_manifest_sha256: str | None = None
    # Present only for the Episode v1 fork. Historical profiles omit this
    # field and therefore retain their exact generation-contract bytes.
    episode: EpisodeProfile | None = None
    # How many windows a prompt of THIS environment sits out after it has been
    # trained on. `None` keeps BATCH_PROMPT_COOLDOWN_WINDOWS, so every
    # historical profile is unchanged.
    #
    # The global default was sized for OpenMathInstruct's 14M prompts, where
    # one million windows means "single use for the life of any real run". A
    # curated corpus breaks that: at eight prompts per window, 2,285 tasks are
    # exhausted in three days and the environment then serves nothing. The
    # value is declared rather than derived from the corpus at runtime — every
    # validator has to agree on which prompts are eligible, and a length read
    # from an installed wheel is not something consensus can rest on.
    #
    # The rule the numbers come from: one full pass through the corpus before
    # any prompt returns, i.e. `min(default, virtual_length // batch_target)`,
    # computed once here where it can be reviewed.
    prompt_cooldown_windows: int | None = None
    # Whether the chat template opens a reasoning block for this environment.
    # `None` keeps it open, which is what every historical profile did — the
    # value was hard-coded `True` in both places that render a prompt.
    #
    # It belongs to the environment rather than the run because the cost of
    # deliberating depends on what the grader reads. An environment graded on
    # the whole completion — no answer span to extract — checks the reasoning
    # against constraints written for the answer; measured on instruction
    # following, closing the block cut the band from 37.5% to 4.2%. One graded
    # on a world rather than a string pays nothing for it and plans better.
    thinking: bool | None = None

    def __post_init__(self) -> None:
        if int(self.max_new_tokens) <= 0:
            raise ValueError("environment max_new_tokens must be positive")
        if self.prompt_cooldown_windows is not None and (
            int(self.prompt_cooldown_windows) <= 0
        ):
            raise ValueError("environment prompt_cooldown_windows must be positive")
        if self.batch_target is not None and int(self.batch_target) <= 0:
            raise ValueError("environment batch_target must be positive")
        if bool(self.environment_contract_id) != bool(
            self.environment_manifest_sha256
        ):
            raise ValueError(
                "environment contract id and manifest sha256 must be set together"
            )
        digest = self.environment_manifest_sha256
        if digest is not None and (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("environment manifest sha256 must be lowercase hex")


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

    Arrival already appears in the elapsed-time denominator, so applying it
    again after this bucket would double-penalize later, longer answers. Exact
    bucket ties therefore go directly to post-seal randomness.

    An adaptive collection close is an operational pipeline policy, not proof
    that the economic leader is mathematically final: a later candidate could
    still have ranked higher. It must therefore never pre-prove or cache a
    mid-window leader. The validator instead expands productive capacity, keeps
    the profile's collection time as a hard ceiling, and may freeze only after a
    minimum collection period, a primary population, prior-GPU completion, and
    fully quiet/drained admission. Ranking and proof still begin only after that
    population is atomically frozen and post-seal randomness exists.
    """

    token_cap: int
    bucket_tokens_per_round: int


PROOF_SCHEME_GRAIL = "grail-v7"
PROOF_SCHEME_TOPLOC = "toploc-v1"
_PROOF_SCHEMES = (PROOF_SCHEME_GRAIL, PROOF_SCHEME_TOPLOC)
_PROOF_MODES = ("enforce", "shadow")
_TOPLOC_FIELDS = (
    "chunk_tokens", "topk", "exp_mismatch_threshold", "mant_mean_threshold",
    "mant_median_threshold", "min_allowed_failures", "ratio_allowed_failures",
)


@dataclass(frozen=True, slots=True)
class ProofProfile:
    """How a task's work is proven. ``shadow`` computes and records, never rejects."""

    scheme: str
    mode: str
    chunk_tokens: int | None = None
    topk: int | None = None
    exp_mismatch_threshold: int | None = None
    mant_mean_threshold: float | None = None
    mant_median_threshold: float | None = None
    min_allowed_failures: int | None = None
    ratio_allowed_failures: float | None = None

    def __post_init__(self) -> None:
        if self.scheme not in _PROOF_SCHEMES:
            raise ValueError(f"unknown proof scheme {self.scheme!r}")
        if self.mode not in _PROOF_MODES:
            raise ValueError(f"unknown proof mode {self.mode!r}")
        values = [getattr(self, name) for name in _TOPLOC_FIELDS]
        if self.scheme == PROOF_SCHEME_TOPLOC and any(v is None for v in values):
            raise ValueError("a toploc proof must state every threshold")
        if self.scheme == PROOF_SCHEME_GRAIL and any(v is not None for v in values):
            raise ValueError("a grail proof takes no toploc fields")
        if self.scheme == PROOF_SCHEME_TOPLOC:
            self._check_toploc_ranges()

    def _check_toploc_ranges(self) -> None:
        """A value outside these ranges boots a task every honest miner fails."""
        for name in ("chunk_tokens", "topk", "exp_mismatch_threshold", "min_allowed_failures"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be a whole number, got {value!r}")
        for name in ("mant_mean_threshold", "mant_median_threshold", "ratio_allowed_failures"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value!r}")
        if self.chunk_tokens < 1 or self.topk < 1:
            raise ValueError("chunk_tokens and topk must be at least 1")
        if min(self.exp_mismatch_threshold, self.mant_mean_threshold,
               self.mant_median_threshold, self.min_allowed_failures) < 0:
            raise ValueError("thresholds must not be negative")
        if not 0.0 <= self.ratio_allowed_failures <= 1.0:
            raise ValueError("ratio_allowed_failures must be within [0, 1]")

    def thresholds(self):
        from reliquary.protocol.toploc import ToplocThresholds

        if self.scheme != PROOF_SCHEME_TOPLOC:
            raise ValueError("only a toploc proof has thresholds")
        return ToplocThresholds(
            exp_mismatch=self.exp_mismatch_threshold,
            mant_mean=float(self.mant_mean_threshold),
            mant_median=float(self.mant_median_threshold),
            min_allowed_failures=self.min_allowed_failures,
            ratio_allowed_failures=float(self.ratio_allowed_failures),
        )

    def to_contract(self) -> dict[str, Any]:
        body: dict[str, Any] = {"scheme": self.scheme, "mode": self.mode}
        if self.scheme == PROOF_SCHEME_TOPLOC:
            body.update({name: getattr(self, name) for name in _TOPLOC_FIELDS})
        return body


# Prime Intellect's deployed default (toploc-validator @ 55c1a23, default.toml).
TOPLOC_DEPLOYED_DEFAULTS = ProofProfile(
    scheme=PROOF_SCHEME_TOPLOC,
    mode="enforce",
    chunk_tokens=32,
    topk=128,
    exp_mismatch_threshold=60,
    mant_mean_threshold=40.0,
    mant_median_threshold=40.0,
    min_allowed_failures=0,
    ratio_allowed_failures=0.0,
)


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
    # The architecture class the model's config declares, stated by whoever
    # sealed the contract. Compiled profiles omit it, so their contract bytes
    # are unchanged; a carried contract must name it, because the startup
    # refusal that checks it against this image's list has nothing else to read.
    model_architecture: str | None = None
    # How the task's work is proven. Empty means GRAIL enforced, as every
    # compiled profile has always meant, and keeps their contract bytes.
    proofs: tuple[ProofProfile, ...] = ()

    def __post_init__(self) -> None:
        # Copy before wrapping so caller-owned dictionaries cannot mutate a
        # profile after construction.
        object.__setattr__(
            self,
            "environments",
            MappingProxyType(dict(self.environments)),
        )
        object.__setattr__(self, "proofs", tuple(self.proofs))
        if sum(1 for p in self.proofs if p.mode == "enforce") > 1:
            raise ValueError("a task enforces at most one proof scheme")
        schemes = [p.scheme for p in self.proofs]
        if len(schemes) != len(set(schemes)):
            raise ValueError("a task names each proof scheme once")

    def to_generation_contract(self) -> dict[str, Any]:
        """Return a detached contract containing only JSON-native values."""

        environments: dict[str, dict[str, Any]] = {}
        for name, environment in self.environments.items():
            bft = environment.bft
            environment_contract: dict[str, Any] = {
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
            # Older generation contracts stay byte-for-byte unchanged. Prompt
            # text becomes an explicit signed field only on profiles that opt
            # into a versioned template (v5+).
            if environment.prompt_template is not None:
                environment_contract["prompt_template"] = (
                    environment.prompt_template.to_generation_contract()
                )
            if environment.batch_target is not None:
                environment_contract["batch_target"] = environment.batch_target
            if environment.prompt_cooldown_windows is not None:
                environment_contract["prompt_cooldown_windows"] = (
                    environment.prompt_cooldown_windows
                )
            if environment.thinking is not None:
                environment_contract["thinking"] = environment.thinking
            if environment.environment_contract_id is not None:
                environment_contract["environment_contract_id"] = (
                    environment.environment_contract_id
                )
                environment_contract["environment_manifest_sha256"] = (
                    environment.environment_manifest_sha256
                )
            if environment.episode is not None:
                episode = environment.episode
                environment_contract["episode"] = {
                    "schema": episode.schema,
                    "renderer_id": episode.renderer_id,
                    "max_turns": episode.max_turns,
                    "max_action_tokens": episode.max_action_tokens,
                    "max_episode_tokens": episode.max_episode_tokens,
                    "max_observation_bytes": episode.max_observation_bytes,
                }
            environments[name] = environment_contract

        contract: dict[str, Any] = {
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
        # Emitted only when set, like `episode` and `batch_target`: that is what
        # keeps the compiled profiles' contract bytes, and their digests,
        # identical to what the fleet already attests.
        if self.model_architecture is not None:
            contract["model_architecture"] = self.model_architecture
        if self.proofs:
            contract["proofs"] = [proof.to_contract() for proof in self.proofs]
        return contract


def _required(body: Mapping[str, Any], field: str, *, context: str = "generation contract") -> Any:
    """Refuse a contract missing a field or having a null value: a default here
    is a silent disagreement between two processes."""
    if field not in body:
        raise ValueError(f"{context} is missing {field!r}")
    value = body[field]
    if value is None:
        raise ValueError(f"{context} has a null {field!r}")
    return value


def _coerce_int(value: Any, field: str, *, context: str = "generation contract") -> int:
    """A whole number, or a ``ValueError`` naming the field.

    Nothing is converted: ``int(3.7)`` would answer a question nobody asked,
    and ``protocol_version`` gates wire compatibility. ``bool`` is named
    separately because it is an ``int`` subclass and would otherwise pass.
    """
    if isinstance(value, bool):
        raise ValueError(f"{context} {field!r} is a bool, not an int")
    if not isinstance(value, int):
        raise ValueError(f"{context} {field!r} is not an int: {value!r}")
    return value


def _coerce_float(value: Any, field: str, *, context: str = "generation contract") -> float:
    """A number, or a ``ValueError`` naming the field. A whole number is one."""
    if isinstance(value, bool):
        raise ValueError(f"{context} {field!r} is a bool, not a float")
    if not isinstance(value, (int, float)):
        raise ValueError(f"{context} {field!r} is not a float: {value!r}")
    return float(value)


# Every key each block of a generation contract may carry. `prompt_template`
# includes the two derived keys the writer emits but the reader recomputes.
_CONTRACT_FIELDS = (
    "profile_id", "model_id", "model_revision", "model_architecture",
    "protocol_version", "prompt_encoding", "throughput_tiebreak",
    "collection_seconds", "upload_grace_seconds", "sampling", "environments",
    "proofs",
)
_SAMPLING_FIELDS = ("rollouts", "temperature", "top_p", "top_k", "do_sample")
_ENVIRONMENT_FIELDS = (
    "max_new_tokens", "answer_format", "bft", "prompt_template",
    "batch_target", "environment_contract_id", "environment_manifest_sha256",
    "episode", "prompt_cooldown_windows", "thinking",
)
_BFT_FIELDS = ("thinking_budget", "answer_budget", "force_answer")
_PROMPT_TEMPLATE_FIELDS = ("id", "renderer", "template", "sha256")
_EPISODE_FIELDS = (
    "schema", "renderer_id", "max_turns", "max_action_tokens",
    "max_episode_tokens", "max_observation_bytes",
)
_TIEBREAK_FIELDS = ("token_cap", "bucket_tokens_per_round")
_PROOF_FIELDS = ("scheme", "mode", *_TOPLOC_FIELDS)


def _object(body: Any, known: tuple[str, ...], *, context: str) -> Mapping[str, Any]:
    """An object whose every key is one this reader honours.

    An ignored key is worse than a rejected one: the rebuild drops it, so the
    only symptom is a digest mismatch at startup that names nothing.
    """
    if not isinstance(body, Mapping):
        raise ValueError(f"{context} must be an object")
    unknown = sorted(set(body) - set(known))
    if unknown:
        raise ValueError(f"{context} has unknown fields: {unknown}")
    return body


def _coerce_str(value: Any, field: str, *, context: str = "generation contract") -> str | None:
    """An optional text field is text or absent, never a dict or a number.

    Waving one through is worse than a bad message: the wrong value survives
    the round trip unchanged, so the digest agrees and a malformed contract
    becomes a self-consistent, registry-attested task that boots.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{context} {field!r} must be text, got {value!r}")
    return value


def _coerce_bool(value: Any, field: str, *, context: str = "generation contract") -> bool:
    """A flag is a JSON boolean, never a truthy stand-in.

    ``bool(1)`` would be re-emitted as ``true``, so the contract would no
    longer hash to what it arrived as and startup would fail naming nothing.
    """
    if not isinstance(value, bool):
        raise ValueError(f"{context} {field!r} must be true or false, got {value!r}")
    return value


def _environment_from_contract(name: str, body: Any) -> EnvironmentProfile:
    if not isinstance(body, Mapping):
        raise ValueError(f"environment {name!r} is not an object")
    body = _object(body, _ENVIRONMENT_FIELDS, context=f"environment {name!r}")
    max_tokens = _required(body, "max_new_tokens", context=f"environment {name!r}")
    bft_body = body.get("bft")
    if bft_body is not None:
        bft_body = _object(bft_body, _BFT_FIELDS, context=f"environment {name!r} 'bft'")
    template = body.get("prompt_template")
    if template is not None:
        template = _object(
            template,
            _PROMPT_TEMPLATE_FIELDS,
            context=f"environment {name!r} 'prompt_template'",
        )
    episode_body = body.get("episode")
    if episode_body is not None:
        episode_body = _object(
            episode_body, _EPISODE_FIELDS, context=f"environment {name!r} 'episode'"
        )
    return EnvironmentProfile(
        max_new_tokens=_coerce_int(max_tokens, "max_new_tokens", context=f"environment {name!r}"),
        bft=(
            None
            if bft_body is None
            else BFTProfile(
                thinking_budget=_coerce_int(
                    _required(bft_body, "thinking_budget", context=f"environment {name!r} 'bft'"),
                    "thinking_budget",
                    context=f"environment {name!r} 'bft'",
                ),
                answer_budget=_coerce_int(
                    _required(bft_body, "answer_budget", context=f"environment {name!r} 'bft'"),
                    "answer_budget",
                    context=f"environment {name!r} 'bft'",
                ),
                force_answer=_coerce_bool(
                    _required(bft_body, "force_answer", context=f"environment {name!r} 'bft'"),
                    "force_answer",
                    context=f"environment {name!r} 'bft'",
                ),
            )
        ),
        answer_format=_coerce_str(
            body.get("answer_format"), "answer_format", context=f"environment {name!r}"
        ),
        prompt_template=(
            None
            if template is None
            # 'renderer' and 'sha256' are derived, so they are recomputed rather
            # than read; the round-trip test is what proves they still agree.
            else PromptTemplateProfile(
                template_id=str(_required(template, "id", context=f"environment {name!r} 'prompt_template'")),
                template=str(_required(template, "template", context=f"environment {name!r} 'prompt_template'")),
            )
        ),
        # All four optional fields go through a named coercion. A value the
        # bare `.get()` waved through was a round-trip fixed point, so its
        # digest agreed and the malformed contract booted as an attested task.
        batch_target=(
            None
            if body.get("batch_target") is None
            else _coerce_int(
                body["batch_target"], "batch_target", context=f"environment {name!r}"
            )
        ),
        environment_contract_id=_coerce_str(
            body.get("environment_contract_id"),
            "environment_contract_id",
            context=f"environment {name!r}",
        ),
        environment_manifest_sha256=_coerce_str(
            body.get("environment_manifest_sha256"),
            "environment_manifest_sha256",
            context=f"environment {name!r}",
        ),
        prompt_cooldown_windows=(
            None
            if body.get("prompt_cooldown_windows") is None
            else _coerce_int(
                body["prompt_cooldown_windows"],
                "prompt_cooldown_windows",
                context=f"environment {name!r}",
            )
        ),
        thinking=(
            None
            if body.get("thinking") is None
            else _coerce_bool(
                body["thinking"], "thinking", context=f"environment {name!r}"
            )
        ),
        episode=(
            None
            if episode_body is None
            else EpisodeProfile(
                schema=str(_required(episode_body, "schema", context=f"environment {name!r} 'episode'")),
                renderer_id=str(_required(episode_body, "renderer_id", context=f"environment {name!r} 'episode'")),
                max_turns=_coerce_int(
                    _required(episode_body, "max_turns", context=f"environment {name!r} 'episode'"),
                    "max_turns",
                    context=f"environment {name!r} 'episode'",
                ),
                max_action_tokens=_coerce_int(
                    _required(episode_body, "max_action_tokens", context=f"environment {name!r} 'episode'"),
                    "max_action_tokens",
                    context=f"environment {name!r} 'episode'",
                ),
                max_episode_tokens=_coerce_int(
                    _required(episode_body, "max_episode_tokens", context=f"environment {name!r} 'episode'"),
                    "max_episode_tokens",
                    context=f"environment {name!r} 'episode'",
                ),
                max_observation_bytes=_coerce_int(
                    _required(episode_body, "max_observation_bytes", context=f"environment {name!r} 'episode'"),
                    "max_observation_bytes",
                    context=f"environment {name!r} 'episode'",
                ),
            )
        ),
    )


def _proof_from_contract(index: int, body: Any) -> ProofProfile:
    context = f"generation contract 'proofs'[{index}]"
    body = _object(body, _PROOF_FIELDS, context=context)

    def integer(name):
        value = body.get(name)
        return None if value is None else _coerce_int(value, name, context=context)

    def real(name):
        value = body.get(name)
        return None if value is None else _coerce_float(value, name, context=context)

    return ProofProfile(
        scheme=str(_required(body, "scheme", context=context)),
        mode=str(_required(body, "mode", context=context)),
        chunk_tokens=integer("chunk_tokens"),
        topk=integer("topk"),
        exp_mismatch_threshold=integer("exp_mismatch_threshold"),
        mant_mean_threshold=real("mant_mean_threshold"),
        mant_median_threshold=real("mant_median_threshold"),
        min_allowed_failures=integer("min_allowed_failures"),
        ratio_allowed_failures=real("ratio_allowed_failures"),
    )


def profile_from_contract(contract: Mapping[str, Any]) -> ProtocolProfile:
    """Rebuild a profile from what ``to_generation_contract`` produced.

    The exact inverse, and it must stay exact: a task carries this contract, so
    a field lost in translation is a field the fleet disagrees about silently.
    """
    if not isinstance(contract, Mapping):
        raise ValueError("a generation contract must be an object")
    contract = _object(contract, _CONTRACT_FIELDS, context="generation contract")

    sampling_body = _required(contract, "sampling")
    if not isinstance(sampling_body, Mapping):
        raise ValueError("generation contract 'sampling' must be an object")
    sampling_body = _object(
        sampling_body, _SAMPLING_FIELDS, context="generation contract 'sampling'"
    )
    environments = _required(contract, "environments")
    if not isinstance(environments, Mapping):
        raise ValueError("generation contract 'environments' must be an object")
    tiebreak = contract.get("throughput_tiebreak")
    if tiebreak is not None:
        tiebreak = _object(
            tiebreak,
            _TIEBREAK_FIELDS,
            context="generation contract 'throughput_tiebreak'",
        )

    proofs_body = contract.get("proofs")
    if proofs_body is not None and not isinstance(proofs_body, list):
        raise ValueError("generation contract 'proofs' must be a list")

    return ProtocolProfile(
        profile_id=str(_required(contract, "profile_id")),
        model_id=str(_required(contract, "model_id")),
        model_revision=str(_required(contract, "model_revision")),
        protocol_version=_coerce_int(
            _required(contract, "protocol_version"),
            "protocol_version",
        ),
        collection_seconds=_coerce_int(
            _required(contract, "collection_seconds"),
            "collection_seconds",
        ),
        upload_grace_seconds=_coerce_int(
            _required(contract, "upload_grace_seconds"),
            "upload_grace_seconds",
        ),
        prompt_encoding=str(_required(contract, "prompt_encoding")),
        # Legitimately absent: every compiled profile predates the field.
        model_architecture=_coerce_str(
            contract.get("model_architecture"), "model_architecture"
        ),
        proofs=tuple(
            _proof_from_contract(i, body) for i, body in enumerate(proofs_body or ())
        ),
        sampling=SamplingProfile(
            rollouts=_coerce_int(
                _required(sampling_body, "rollouts", context="generation contract 'sampling'"),
                "rollouts",
                context="generation contract 'sampling'",
            ),
            temperature=_coerce_float(
                _required(sampling_body, "temperature", context="generation contract 'sampling'"),
                "temperature",
                context="generation contract 'sampling'",
            ),
            top_p=_coerce_float(
                _required(sampling_body, "top_p", context="generation contract 'sampling'"),
                "top_p",
                context="generation contract 'sampling'",
            ),
            top_k=_coerce_int(
                _required(sampling_body, "top_k", context="generation contract 'sampling'"),
                "top_k",
                context="generation contract 'sampling'",
            ),
            do_sample=bool(_required(sampling_body, "do_sample", context="generation contract 'sampling'")),
        ),
        environments={
            name: _environment_from_contract(name, body)
            for name, body in environments.items()
        },
        throughput_tiebreak=(
            None
            if tiebreak is None
            else ThroughputTiebreakProfile(
                token_cap=_coerce_int(
                    _required(tiebreak, "token_cap", context="generation contract 'throughput_tiebreak'"),
                    "token_cap",
                    context="generation contract 'throughput_tiebreak'",
                ),
                bucket_tokens_per_round=_coerce_int(
                    _required(tiebreak, "bucket_tokens_per_round", context="generation contract 'throughput_tiebreak'"),
                    "bucket_tokens_per_round",
                    context="generation contract 'throughput_tiebreak'",
                ),
            )
        ),
    )


def enforced_proof(profile: ProtocolProfile) -> ProofProfile:
    """The scheme that decides; GRAIL when the contract names none."""
    for proof in profile.proofs:
        if proof.mode == "enforce":
            return proof
    return ProofProfile(PROOF_SCHEME_GRAIL, "enforce")


def toploc_proof(profile: ProtocolProfile) -> ProofProfile | None:
    """The contract's toploc entry, whatever its mode."""
    for proof in profile.proofs:
        if proof.scheme == PROOF_SCHEME_TOPLOC:
            return proof
    return None


def proof_rejection(
    profile: ProtocolProfile, *, grail_passed: bool, toploc_passed: bool | None
) -> str | None:
    """Which enforced scheme refuses the rollout, if any; a shadow scheme never does.

    ``toploc_passed`` is None when no toploc verdict exists, which refuses under
    toploc enforcement: missing proofs are not a pass.
    """
    if enforced_proof(profile).scheme == PROOF_SCHEME_TOPLOC:
        return None if toploc_passed is True else "toploc_fail"
    return None if grail_passed else "grail_fail"


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


# The v4 base-model rollout accidentally omitted the semantic reasoning cue.
# Keep v4 immutable as the no-cue control and introduce the corrected prompts
# only through a new profile. The Math wording deliberately mirrors DAPO's
# released prompt prefix while retaining Reliquary's boxed reward channel.
_MATH_REASONING_PROMPT = PromptTemplateProfile(
    template_id="openmathinstruct-step-by-step-v1",
    template=(
        "Solve the following math problem step by step.\n\n"
        "$problem\n\n"
        "Put your final answer within \\boxed{}."
    ),
)

_CODE_REASONING_PROMPT = PromptTemplateProfile(
    template_id="opencodeinstruct-step-by-step-v1",
    template=(
        "Solve the following programming problem step by step.\n\n"
        "$problem$contract\n\n"
        "After your reasoning, provide the final implementation in the last "
        "fenced Python code block."
    ),
)

_RELIQUARY_RECORDS_PROMPT = PromptTemplateProfile(
    template_id="reliquary-records-v1",
    # The generated problem already contains the full input, ordered
    # operations, and exact answer channel. Keeping this wrapper as the
    # identity makes the rendered bytes explicit in the signed contract.
    template="$problem",
)

_RELIQUARY_LOGIC_PROMPT = PromptTemplateProfile(
    template_id="reliquary-logic-step-by-step-v1",
    # The generated problem carries the puzzle and the exact answer channel.
    # The wrapper adds the one thing a generator cannot state for itself:
    # that reasoning may come first. extract_json_answer already reads the
    # last fence, so reasoning costs the answer channel nothing.
    template=(
        "Solve the following problem step by step.\n\n"
        "$problem\n\n"
        "After your reasoning, give the final answer in the last fenced JSON "
        "code block."
    ),
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
    ProtocolProfile(
        profile_id="qwen3-4b-base-dapo-reasoning-v5",
        # Clean protocol fork from v4: the model, raw encoding, sampling,
        # budgets, and objective controls stay fixed. Only the canonical prompt
        # now asks the base model to reason step by step and, for Code, pins the
        # final implementation to the parser's last-fenced-block channel.
        model_id="Qwen/Qwen3-4B-Base",
        model_revision="906bfd4b4dc7f14ee4320094d8b41684abff8539",
        protocol_version=5,
        collection_seconds=100,
        upload_grace_seconds=33,
        prompt_encoding="raw",
        sampling=_SAMPLING_DAPO,
        environments={
            "openmathinstruct": EnvironmentProfile(
                max_new_tokens=8192,
                bft=None,
                answer_format="boxed",
                prompt_template=_MATH_REASONING_PROMPT,
            ),
            "opencodeinstruct": EnvironmentProfile(
                max_new_tokens=8192,
                bft=None,
                prompt_template=_CODE_REASONING_PROMPT,
            ),
        },
        throughput_tiebreak=ThroughputTiebreakProfile(
            token_cap=8192,
            bucket_tokens_per_round=50,
        ),
    ),
    ProtocolProfile(
        # Same model, same sampling, same budgets, same prompts as v5. v6
        # changes when a window ends and who gets admitted, never what a
        # miner generates -- so the generation contract is v5's, field for
        # field, and a diff of the two profiles shows only the window.
        profile_id="qwen3-4b-base-dapo-fill-closed-v6",
        model_id="Qwen/Qwen3-4B-Base",
        model_revision="906bfd4b4dc7f14ee4320094d8b41684abff8539",
        protocol_version=6,
        collection_seconds=100,
        upload_grace_seconds=33,
        prompt_encoding="raw",
        sampling=_SAMPLING_DAPO,
        environments={
            "openmathinstruct": EnvironmentProfile(
                max_new_tokens=8192,
                bft=None,
                answer_format="boxed",
                prompt_template=_MATH_REASONING_PROMPT,
            ),
            "opencodeinstruct": EnvironmentProfile(
                max_new_tokens=8192,
                bft=None,
                prompt_template=_CODE_REASONING_PROMPT,
            ),
        },
        # Fill-closed uses FIFO; rate and payload size never rank a group.
        throughput_tiebreak=None,
    ),
    ProtocolProfile(
        profile_id="qwen3-4b-base-dapo-reliquary-v1",
        model_id="Qwen/Qwen3-4B-Base",
        model_revision="906bfd4b4dc7f14ee4320094d8b41684abff8539",
        protocol_version=6,
        collection_seconds=100,
        upload_grace_seconds=33,
        prompt_encoding="raw",
        sampling=_SAMPLING_DAPO,
        environments={
            "openmathinstruct": EnvironmentProfile(
                max_new_tokens=8192, bft=None, answer_format="boxed",
                prompt_template=_MATH_REASONING_PROMPT,
            ),
            "opencodeinstruct": EnvironmentProfile(
                max_new_tokens=8192, bft=None,
                prompt_template=_CODE_REASONING_PROMPT,
            ),
            "reliquary_logic_v2": EnvironmentProfile(
                max_new_tokens=8192, bft=None,
                answer_format="last_json_object_v1", batch_target=16,
                prompt_template=PromptTemplateProfile(
                    "reliquary-external-prompt-v1", "$problem",
                ),
                environment_contract_id="reliquary/answer-json/v1",
                environment_manifest_sha256=(
                    "1e4e05cae799d8e71d8876b0f7526c5b09ca1d5a9ab05f364fb35539288c5019"
                ),
            ),
        },
        throughput_tiebreak=None,
    ),
    ProtocolProfile(
        profile_id="qwen3-4b-reliquary-verifiable-v6-dev1",
        # Isolated infrastructure/frontier profile. It deliberately reuses the
        # exact pinned v4/v5 base revision without joining their Math+Code
        # checkpoint lineage.
        model_id="Qwen/Qwen3-4B-Base",
        model_revision="906bfd4b4dc7f14ee4320094d8b41684abff8539",
        protocol_version=6,
        collection_seconds=100,
        upload_grace_seconds=33,
        prompt_encoding="raw",
        sampling=_SAMPLING_DAPO,
        environments={
            "reliquaryverifiable_v1": EnvironmentProfile(
                max_new_tokens=1024,
                bft=None,
                answer_format="last_json_object_v1",
                prompt_template=_RELIQUARY_RECORDS_PROMPT,
                batch_target=16,
                environment_contract_id="reliquary-records-v1",
                environment_manifest_sha256=(
                    "d0d5d838e40b383d1c95a62d1cdded8"
                    "458f4a7b62df621c87c9435b62207929b"
                ),
            ),
        },
        throughput_tiebreak=ThroughputTiebreakProfile(
            token_cap=1024,
            bucket_tokens_per_round=50,
        ),
    ),
    ProtocolProfile(
        profile_id="qwen3-4b-reliquary-episode-v7-dev1",
        # Opt-in development profile for the canonical multi-turn format. It
        # intentionally preserves the existing Qwen3-4B base revision so the
        # environment and assistant-mask fork can be evaluated independently.
        model_id="Qwen/Qwen3-4B-Base",
        model_revision="906bfd4b4dc7f14ee4320094d8b41684abff8539",
        protocol_version=7,
        collection_seconds=300,
        upload_grace_seconds=33,
        prompt_encoding="raw",
        sampling=_SAMPLING_DAPO,
        environments={
            "reliquary_stateful_tools_v1": EnvironmentProfile(
                max_new_tokens=16384,
                bft=None,
                answer_format="episode_json_action_v1",
                batch_target=16,
                environment_contract_id="reliquary-stateful-tools-v1",
                environment_manifest_sha256=(
                    "0f490881544ba065bf33b974032adbc3"
                    "f844d2c3978bcd6ca8dbb7089baa8f18"
                ),
                episode=EpisodeProfile(
                    schema="reliquary/episode/v1",
                    renderer_id="reliquary-jsonl-tools-v1",
                    max_turns=8,
                    max_action_tokens=1024,
                    max_episode_tokens=16384,
                    max_observation_bytes=65536,
                ),
            ),
            "reliquary_retrieval_tools_v1": EnvironmentProfile(
                max_new_tokens=16384,
                bft=None,
                answer_format="episode_json_action_v1",
                batch_target=16,
                environment_contract_id="reliquary-retrieval-tools-v1",
                environment_manifest_sha256=(
                    "1c53afdf6acc59dd7df0693b7486e47"
                    "de94d79977404841d1368ffb2571c0c7d"
                ),
                episode=EpisodeProfile(
                    schema="reliquary/episode/v1",
                    renderer_id="reliquary-jsonl-tools-v1",
                    max_turns=6,
                    max_action_tokens=1024,
                    max_episode_tokens=16384,
                    max_observation_bytes=65536,
                ),
            ),
            "reliquary_workspace_tools_v1": EnvironmentProfile(
                max_new_tokens=16384,
                bft=None,
                answer_format="episode_json_action_v1",
                batch_target=16,
                environment_contract_id="reliquary-workspace-tools-v1",
                environment_manifest_sha256=(
                    "7f0465cff80aefc489e0302d2122272"
                    "8115e0094df33858d6c613fa5423489e2"
                ),
                episode=EpisodeProfile(
                    schema="reliquary/episode/v1",
                    renderer_id="reliquary-jsonl-tools-v1",
                    max_turns=7,
                    max_action_tokens=1024,
                    max_episode_tokens=16384,
                    max_observation_bytes=65536,
                ),
            ),
        },
        throughput_tiebreak=ThroughputTiebreakProfile(
            token_cap=4096,
            bucket_tokens_per_round=50,
        ),
    ),
    ProtocolProfile(
        profile_id="qwen3-4b-reliquary-logic-v8-dev1",
        # Dormant development profile for the procedural logic suite. It
        # reuses the pinned v4/v5 base revision without joining any existing
        # checkpoint lineage, and declares the logic environment alone so a
        # canary never perturbs the live Math+Code lanes.
        model_id="Qwen/Qwen3-4B-Base",
        model_revision="906bfd4b4dc7f14ee4320094d8b41684abff8539",
        protocol_version=8,
        collection_seconds=100,
        upload_grace_seconds=33,
        prompt_encoding="raw",
        sampling=_SAMPLING_DAPO,
        environments={
            "reliquarylogic_v1": EnvironmentProfile(
                # The v4/v5 budget, unchanged: one base model, one budget to
                # think in. Nothing about a logic puzzle earns a tighter cap
                # than a math problem on the same weights, and the prompt now
                # asks for the reasoning first.
                max_new_tokens=8192,
                bft=None,
                answer_format="last_json_object_v1",
                prompt_template=_RELIQUARY_LOGIC_PROMPT,
                batch_target=16,
                environment_contract_id="reliquary-logic-v1",
                environment_manifest_sha256=(
                    "9cb29e487321b2e6c005f2a1a89ccff"
                    "ecf01b1c09bfd03337094e338ab912ca9"
                ),
            ),
        },
        throughput_tiebreak=ThroughputTiebreakProfile(
            token_cap=8192,
            bucket_tokens_per_round=50,
        ),
    ),

    ProtocolProfile(
        profile_id="teutonic-9b-reliquary-suite-v9-dev1",
        # Dormant development profile for the Teutonic-I run: four packaged
        # environments from reliquary-environments, each bound by the digest of
        # its artifact manifest. Nothing selects it until a task entry names it
        # and a validator is started with it; the live profile is untouched.
        #
        # Every per-environment number below was measured on this policy rather
        # than carried over from the 4B run — the budgets, the reasoning mode,
        # and the episode limits all moved when they were.
        model_id="ReliquaryForge/teutonic-i-graft-sft-cot-v2",
        model_revision="d5256c5ccc2c06d8f9bf3133b37ab2a5b95a224e",
        protocol_version=9,
        collection_seconds=100,
        upload_grace_seconds=33,
        # An instruct policy trained on the chat template, unlike the base
        # models every v4+ profile used. Raw completion would hand it a prompt
        # in a form it never saw.
        prompt_encoding="chat_template",
        sampling=_SAMPLING_DAPO,
        environments={
            "reliquary_dapo_math_v1": EnvironmentProfile(
                # 32,768, not the 8,192 the siblings need. Measured: band 70.8%
                # at 24,576 against 81.2% here, dead groups 26.0% against 17.7%.
                # Competition maths does not fit a budget chosen for answers.
                max_new_tokens=32768,
                bft=None,
                answer_format="boxed",
                # Half the siblings' prompt count, because a group here costs
                # what ten of theirs cost: 493k tokens against 48k, measured on
                # an H200 over a group of 16 at this budget. At 16 prompts this
                # environment alone is three quarters of the window's tokens,
                # for a band measured in protocol at 25% (3 groups of 12), so
                # the window would buy maths in code's and instruction
                # following's place.
                batch_target=8,
                prompt_template=PromptTemplateProfile(
                    "reliquary-external-prompt-v1", "$problem",
                ),
                environment_contract_id="reliquary/boxed-answer/v1",
                environment_manifest_sha256=(
                    "cca437d73e8183a6df4af4780e00e34cd26fed03238b18208898b1e2586d2035"
                ),
                # One pass through the 13,931-problem train split at 8 a window.
                prompt_cooldown_windows=1741,
            ),
            "reliquary_instruction_following_v1": EnvironmentProfile(
                max_new_tokens=8192,
                bft=None,
                answer_format="text",
                batch_target=16,
                prompt_template=PromptTemplateProfile(
                    "reliquary-external-prompt-v1", "$problem",
                ),
                environment_contract_id="reliquary/checked-answer/v1",
                environment_manifest_sha256=(
                    "84b8698446e68e057faea54ea4045cd198c21fb6bd0a7c8fc259a7659c4e483d"
                ),
                # One pass through the 29,435-prompt train split.
                prompt_cooldown_windows=1839,
                # Direct: every verifier reads the whole completion, so an open
                # reasoning block is graded against constraints written for the
                # answer. Measured on this policy, band 4.2% thinking against
                # 37.5% direct.
                thinking=False,
            ),
            "reliquary_code_v1": EnvironmentProfile(
                # The longest completion that finished on its own ran to about
                # 6,600 tokens; raising this measured worse, not better.
                max_new_tokens=8192,
                bft=None,
                answer_format="fenced_python",
                batch_target=16,
                prompt_template=PromptTemplateProfile(
                    "reliquary-external-prompt-v1", "$problem",
                ),
                environment_contract_id="reliquary/python-cases/v1",
                environment_manifest_sha256=(
                    "71e4f23c614b7f321bbb9f6cf74f98b772137442bdf9960e5b6d9b6387f41216"
                ),
                # No cooldown override: 2,481,806 prompts at 16 a window outlast
                # the global horizon, which is what that horizon was sized for.
            ),
            "reliquary_telecom_solo_v1": EnvironmentProfile(
                # For an episode this is the whole transcript, as `max_episode_tokens`.
                max_new_tokens=49152,
                bft=None,
                answer_format="episode_json_action_v1",
                # What this corpus supplies, not what it declares. Measured on
                # an H200 21-09 at 8 rollouts a task: `service_issue` solves
                # 54.7% of the time (47 tasks) and `mobile_data_issue` 9.8%
                # (254), while `mms_issue` — 1,984 of the 2,285 — solves 0.9%.
                # A group drawn there is sixteen zeroes: the sigma gate refuses
                # it and nothing is paid for it, so miners draw from the other
                # two and the usable corpus is ~240 tickets, not 1,827. Four
                # prompts a window is what that pool sustains. It goes back up
                # when the policy makes `mms_issue` solvable, which is a
                # property of the policy rather than of the environment.
                batch_target=4,
                environment_contract_id="reliquary/episode-json/v1",
                environment_manifest_sha256=(
                    "74d0e7569247eabc3d4fb2d909773f2b1f1d8430a4ce4f4fd09c6ad6e6794b05"
                ),
                # One pass through the tickets this policy can use, ~240 of
                # them, at four a window — not the 456 windows the corpus-wide
                # rule gives. Sized on the declared 1,827, the usable pool is
                # spent in sixty windows and the environment then serves
                # nothing for four hundred more, because every ticket still
                # eligible is one no group passes the gate on. This is the one
                # place that rule is deliberately relaxed, and it is a smaller
                # relaxation than it reads as: a ticket returns every sixty
                # windows instead of the environment going dark.
                prompt_cooldown_windows=60,
                episode=EpisodeProfile(
                    schema="reliquary/episode/v1",
                    # The policy's own template. Measured on H200 20-09 over 64
                    # episodes each: this dialect lands a valid call in 61 of 61,
                    # a median of 15 distinct tools and no invalid action, and
                    # solves 2 tickets outright; the JSONL dialect, whose framing
                    # this policy never saw, leaves 45 of 64 without a single
                    # valid call and none solved.
                    renderer_id="reliquary-chatml-tools-v1",
                    max_turns=40,
                    # Per turn. Measured over 11,308 turns: p99 2,001, and a
                    # higher cap buys three hundredths of a percent.
                    max_action_tokens=4096,
                    # Whole transcript: a 10,118-token opening (44 tool schemas
                    # and the policy), up to 24,324 generated, and tool results
                    # of about 63 tokens a turn — about 37,000 at worst.
                    max_episode_tokens=49152,
                    max_observation_bytes=65536,
                ),
            ),
        },
        throughput_tiebreak=None,
    ),
)

PROFILES: Mapping[str, ProtocolProfile] = MappingProxyType(
    {profile.profile_id: profile for profile in _PROFILE_VALUES}
)
DEFAULT_PROFILE_ID = "qwen35-2b-auction-v2"
_PROFILE_ENV_VAR = "RELIQUARY_PROTOCOL_PROFILE"
TASK_CONTRACT_ENV_VAR = "RELIQUARY_TASK_CONTRACT"


def resolve_protocol_profile(profile_id: str | None = None) -> ProtocolProfile:
    """Resolve an explicit profile, a task's carried contract, or the default.

    Empty, misspelled, and otherwise unknown IDs are errors. Falling back after
    an explicit selection would silently put peers on different wire contracts.

    An explicit id wins over the contract file: callers that pass one are
    naming a template, not asking what this process runs.
    """

    if profile_id is None:
        contract_path = os.environ.get(TASK_CONTRACT_ENV_VAR)
        # Absent (None) falls through to the compiled catalogue below; present
        # but empty is a broken deployment, not an unset one, and must fail
        # the same way a misspelled path does rather than run the default.
        if contract_path is not None:
            if not contract_path:
                raise ValueError(f"{TASK_CONTRACT_ENV_VAR} is set but empty")
            return _profile_from_contract_file(contract_path)

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


def _profile_from_contract_file(path: str) -> ProtocolProfile:
    """Read the contract this process was given, or refuse to start.

    Every failure here is fatal on purpose: a process that silently fell back
    to the compiled catalogue would generate under a contract nobody declared.
    """
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(
            f"cannot read the task contract at {path!r}: {exc}"
        ) from exc
    try:
        contract = json.loads(raw)
    except ValueError as exc:
        raise ValueError(
            f"the task contract at {path!r} is not JSON: {exc}"
        ) from exc
    try:
        return profile_from_contract(contract)
    except ValueError as exc:
        raise ValueError(
            f"the task contract at {path!r} is unusable: {exc}"
        ) from exc


ACTIVE_PROTOCOL_PROFILE = resolve_protocol_profile()


def render_active_prompt(
    environment: str,
    *,
    problem: str,
    contract: str = "",
) -> str | None:
    """Render the active profile's explicit prompt, if it declares one.

    ``None`` is an intentional legacy signal: v2-v4 continue through their
    original environment-local concatenation paths without changing a byte.
    """

    try:
        environment_profile = ACTIVE_PROTOCOL_PROFILE.environments[environment]
    except KeyError as exc:
        raise ValueError(
            f"active protocol profile has no environment {environment!r}"
        ) from exc
    prompt_template = environment_profile.prompt_template
    if prompt_template is None:
        return None
    return prompt_template.render(problem=problem, contract=contract)


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
    "EpisodeProfile",
    "PromptTemplateProfile",
    "PROFILES",
    "ProtocolProfile",
    "SamplingProfile",
    "TASK_CONTRACT_ENV_VAR",
    "profile_from_contract",
    "resolve_protocol_profile",
    "render_active_prompt",
    "to_generation_contract",
]
