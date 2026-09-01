"""Immutable, versioned protocol generation profiles.

Profiles live independently of ``reliquary.constants`` so a process can
advertise or validate an exact historical contract without inheriting the
currently deployed constants.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
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

    def __post_init__(self) -> None:
        if int(self.max_new_tokens) <= 0:
            raise ValueError("environment max_new_tokens must be positive")
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
    template_id="reliquary-logic-v1",
    # The generated problem already carries the puzzle and the exact answer
    # channel, so the wrapper is the identity — same rule as the records
    # template, and it keeps the rendered bytes explicit in the contract.
    template="$problem",
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
                    "746b3114299c07144cbfe24c339df938"
                    "9d5d606ffb2764ff0e9e5fd12cd86bb6"
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
                    "d8693064fd24b169cc6cf1f2d69997b"
                    "2b697a262ba6357d1cebfe74cd294a01c"
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
                    "92eba3051cb8a6c2bf60a27ccbd1eb1"
                    "13911e1daf1b5e589fa7c4b507fda347b"
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
                # Numbrix needs room to reason over a grid before emitting it;
                # boolean expressions finish far short of this cap.
                max_new_tokens=2048,
                bft=None,
                answer_format="last_json_object_v1",
                prompt_template=_RELIQUARY_LOGIC_PROMPT,
                batch_target=16,
                environment_contract_id="reliquary-logic-v1",
                environment_manifest_sha256=(
                    "9854f7e3f209676a1d5d4cdba6d85f6"
                    "1195bd107b03984ef123ebedef08d0db7"
                ),
            ),
        },
        throughput_tiebreak=ThroughputTiebreakProfile(
            token_cap=2048,
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
    "resolve_protocol_profile",
    "render_active_prompt",
    "to_generation_contract",
]
