"""What one corpus generation job declares, and the one rule for reading it.

A job is not an ``Environment``: that protocol couples prompt generation to a
reward because in RL the reward is the product. Here the grader only decides
corpus membership, never payment, so the two are independent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any

JOB_SCHEMA = "reliquary/corpus-job/v1"

# The wire's own bounds, duplicated here the way the reject reasons are, so this
# module stays free of pydantic like the rest of the pure layer; the schema test
# pins them equal. A job declared above either one sells slots that no valid
# request can fill, so it is refused at declaration rather than once per 422.
MAX_COMPLETION_TOKENS = 131072
MAX_COMPLETIONS_PER_SUBMISSION = 64

# The terminator counts toward the token budget, so a floor of 1 is a floor of
# nothing: a completion of just the eos id has empty text, clears every check,
# and is paid a slot for a corpus row with nothing in it. Two is the smallest
# floor that leaves one token behind.
MIN_NEW_TOKENS_FLOOR = 2

PROMPT_ORDER_MINER_WALK = "miner_walk"
PROMPT_ORDER_FREE = "free"
PROMPT_ORDERS = frozenset({PROMPT_ORDER_MINER_WALK, PROMPT_ORDER_FREE})

# `\Z`, not `$`: `$` also matches before a trailing newline, and a job id is
# interpolated into an object key by the job store, which has no strip of its own.
JOB_ID_RE = re.compile(r"\A[a-z0-9][a-z0-9-]{0,62}\Z")
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")

_SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_new_tokens", "max_new_tokens", "n")
_FILTER_FIELDS = ("grader_id", "threshold")
_JOB_FIELDS = (
    "schema",
    "job_id",
    "checkpoint_repo",
    "checkpoint_revision",
    "checkpoint_sha256",
    "prompt_source",
    "prompt_count",
    "renderer_id",
    "eos_token_id",
    "sampling",
    "slots_per_prompt",
    "filter",
    "prompt_order",
    "deadline_round",
)
# Optional, and written only when it differs from its default, so every
# manifest that predates it stores and hashes byte-identically. A binary that
# predates it refuses a manifest carrying it (unknown field): miners and
# validators of a job with ``prompt_start > 0`` need a build that knows it.
_OPTIONAL_JOB_FIELDS = ("prompt_start", "seed", "submit", "episode")
# The one value of `submit`: submissions go to the job's own scoped route
# (order jobs, on the order control); absent, to the legacy `/corpus/submit`.
SUBMIT_SCOPED = "scoped"

# An agentic (multi-turn) job: what one episode is and how it is bounded
# (spec §6). Gate M3 measured 60,000 tokens of prompt + trajectory as the most
# one 80 GB audit GPU prefills in one pass.
MAX_EPISODE_TOTAL_TOKENS = 60000
MAX_EPISODE_TURNS = 64
EPISODE_STOPS = ("agent_completed", "max_turns", "context_length")
EPISODE_ENV_PACKAGES = frozenset({"reliquary-swe"})
EPISODE_HARNESSES = frozenset({"bash"})
_EPISODE_FIELDS = ("env", "harness", "renderer", "verifiers", "max_turns",
                   "max_tokens_per_turn", "max_total_tokens", "replay_fraction_failed")
_EPISODE_ENV_FIELDS = ("package", "version", "split", "num_images")
_COMMIT_RE = re.compile(r"\A[0-9a-f]{40}\Z")
_RENDERER_RE = re.compile(r"\Arenderers:[a-z0-9.]+@\d+\.\d+\.\d+\Z")

# How an episode's environment steps are trusted. "replay" (the default, never
# written, so every manifest that predates it hashes byte-identically): the miner
# runs the box and grade executors replay it. "signed_sandbox": every step runs on
# our reliquary-sandbox machines and the validator verifies the signed transcript
# (docs/superpowers/plans/2026-10-06-catalyst-signed-episodes.md).
EXECUTION_REPLAY = "replay"
EXECUTION_SIGNED_SANDBOX = "signed_sandbox"
EXECUTIONS = frozenset({EXECUTION_REPLAY, EXECUTION_SIGNED_SANDBOX})
_EPISODE_OPTIONAL_FIELDS = ("execution", "sandbox")
_SANDBOX_FIELDS = ("env", "env_package", "tools", "sandbox_commit", "budgets")
SANDBOX_BUDGET_FIELDS = ("max_calls", "per_call_timeout_s", "cpu_s", "wall_s",
                         "memory_bytes", "pids", "disk_bytes")
# The gateway's default caps (reliquary-sandbox docs/deployment.md, EPISODE_MAX_*):
# a job above them could never open an episode.
_SANDBOX_BUDGET_BOUNDS = {"max_calls": 512, "per_call_timeout_s": 600, "cpu_s": 3600,
                          "wall_s": 14400, "memory_bytes": 8 * 1024**3, "pids": 1024,
                          "disk_bytes": 10 * 1024**3}
# verifiers' bash harness always offers bash; edit is optional. Record 0 signs the
# tools sorted and unique, so these are the only spellings it can carry.
_SANDBOX_TOOL_SETS = (["bash"], ["bash", "edit"])
# record 0's env_package: `<distribution>==<version>+g<first 16 hex of the code digest>`.
_ENV_PACKAGE_SUFFIX_RE = re.compile(r"\A[^=+\s]+\+g[0-9a-f]{16}\Z")


class JobError(ValueError):
    """The manifest does not describe a job this code can run."""


def _submit(raw) -> str | None:
    if "submit" not in raw:
        return None
    if raw["submit"] != SUBMIT_SCOPED:
        raise JobError(f"submit must be {SUBMIT_SCOPED!r} when present, got {raw['submit']!r}")
    return SUBMIT_SCOPED


@dataclass(frozen=True, slots=True)
class Sampling:
    """Imposed by the job, never chosen by the miner."""

    temperature: float
    top_p: float
    top_k: int
    min_new_tokens: int
    max_new_tokens: int
    n: int


@dataclass(frozen=True, slots=True)
class Filter:
    """The grader that decides corpus membership. It never decides payment."""

    grader_id: str
    threshold: float


@dataclass(frozen=True, slots=True)
class EpisodeEnv:
    """The environment package an episode runs in, pinned by commit."""

    package: str
    version: str
    split: str
    num_images: int

    def to_contract(self) -> dict[str, Any]:
        return {"package": self.package, "version": self.version, "split": self.split,
                "num_images": self.num_images}


@dataclass(frozen=True, slots=True)
class SandboxBudgets:
    """The session budgets a token carries (raised to the task's declared limits)."""

    max_calls: int
    per_call_timeout_s: int
    cpu_s: int
    wall_s: int
    memory_bytes: int
    pids: int
    disk_bytes: int

    def to_contract(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in SANDBOX_BUDGET_FIELDS}


@dataclass(frozen=True, slots=True)
class SandboxSpec:
    """What a signed-sandbox episode pins: the env name machines serve, the exact
    `env_package` record 0 must carry, the offered tools, the reliquary-sandbox
    commit and the session budgets."""

    env: str
    env_package: str
    tools: tuple[str, ...]
    sandbox_commit: str
    budgets: SandboxBudgets

    def to_contract(self) -> dict[str, Any]:
        return {"env": self.env, "env_package": self.env_package, "tools": list(self.tools),
                "sandbox_commit": self.sandbox_commit, "budgets": self.budgets.to_contract()}


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    """One agentic episode: the env, the harness and renderer that drive it,
    and its bounds. Only multi-turn jobs carry it."""

    env: EpisodeEnv
    harness: str
    renderer: str
    verifiers: str
    max_turns: int
    max_tokens_per_turn: int
    max_total_tokens: int
    replay_fraction_failed: float
    execution: str = EXECUTION_REPLAY
    sandbox: SandboxSpec | None = None

    def to_contract(self) -> dict[str, Any]:
        contract = {"env": self.env.to_contract(), "harness": self.harness,
                    "renderer": self.renderer, "verifiers": self.verifiers,
                    "max_turns": self.max_turns, "max_tokens_per_turn": self.max_tokens_per_turn,
                    "max_total_tokens": self.max_total_tokens,
                    "replay_fraction_failed": self.replay_fraction_failed}
        if self.execution != EXECUTION_REPLAY:
            contract["execution"] = self.execution
            contract["sandbox"] = self.sandbox.to_contract()
        return contract


@dataclass(frozen=True, slots=True)
class JobSpec:
    job_id: str
    checkpoint_repo: str
    checkpoint_revision: str
    checkpoint_sha256: str
    prompt_source: str
    prompt_count: int
    renderer_id: str
    eos_token_id: int
    sampling: Sampling
    slots_per_prompt: int
    filter: Filter | None
    prompt_order: str
    deadline_round: int | None
    # The job owns source rows [prompt_start, prompt_start + prompt_count), and
    # every prompt index it handles -- walk, submission, ledger, export -- is a
    # SOURCE index in that range.
    prompt_start: int = 0
    # Recorded for reproducibility (eval jobs); nothing verifies a miner used it.
    seed: int | None = None
    # "scoped" (order jobs only): miners submit on /corpus/jobs/{job_id}/submit.
    submit: str | None = None
    # Multi-turn agentic jobs only (spec §6); absent on every other job.
    episode: EpisodeSpec | None = None

    @property
    def prompt_end(self) -> int:
        """One past the last source row this job owns."""
        return self.prompt_start + self.prompt_count

    def owns(self, prompt_index: int) -> bool:
        return self.prompt_start <= prompt_index < self.prompt_end

    @property
    def total_slots(self) -> int:
        return self.prompt_count * self.slots_per_prompt

    @property
    def rejection_sampling(self) -> bool:
        """The flag: a job with a filter keeps only what the grader accepts."""
        return self.filter is not None

    def to_contract(self) -> dict[str, Any]:
        """A detached, JSON-native view, stable enough to hash and sign."""
        return {
            "schema": JOB_SCHEMA,
            "job_id": self.job_id,
            "checkpoint_repo": self.checkpoint_repo,
            "checkpoint_revision": self.checkpoint_revision,
            "checkpoint_sha256": self.checkpoint_sha256,
            "prompt_source": self.prompt_source,
            "prompt_count": self.prompt_count,
            "renderer_id": self.renderer_id,
            "eos_token_id": self.eos_token_id,
            "sampling": {
                "temperature": self.sampling.temperature,
                "top_p": self.sampling.top_p,
                "top_k": self.sampling.top_k,
                "min_new_tokens": self.sampling.min_new_tokens,
                "max_new_tokens": self.sampling.max_new_tokens,
                "n": self.sampling.n,
            },
            "slots_per_prompt": self.slots_per_prompt,
            "filter": (
                None
                if self.filter is None
                else {
                    "grader_id": self.filter.grader_id,
                    "threshold": self.filter.threshold,
                }
            ),
            "prompt_order": self.prompt_order,
            "deadline_round": self.deadline_round,
            **({"prompt_start": self.prompt_start} if self.prompt_start else {}),
            **({"seed": self.seed} if self.seed is not None else {}),
            **({"submit": self.submit} if self.submit is not None else {}),
            **({"episode": self.episode.to_contract()} if self.episode is not None else {}),
        }


def _text(raw: Mapping[str, Any], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value or value.strip() != value:
        raise JobError(f"{field} must be non-empty trimmed text, got {value!r}")
    return value


def _positive_int(raw: Mapping[str, Any], field: str) -> int:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise JobError(f"{field} must be a positive whole number, got {value!r}")
    return value


def _non_negative_int(raw: Mapping[str, Any], field: str) -> int:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise JobError(f"{field} must be a non-negative whole number, got {value!r}")
    return value


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JobError(f"{field} must be a number, got {value!r}")
    return float(value)


def _parse_sampling(raw: Any) -> Sampling:
    if not isinstance(raw, Mapping):
        raise JobError("sampling must be an object")
    unknown = set(raw) - set(_SAMPLING_FIELDS)
    if unknown:
        raise JobError(f"sampling has unknown fields: {sorted(unknown)}")
    missing = [f for f in _SAMPLING_FIELDS if f not in raw]
    if missing:
        raise JobError(f"sampling is missing: {', '.join(missing)}")
    temperature = _number(raw["temperature"], "sampling.temperature")
    if temperature <= 0:
        raise JobError(f"sampling.temperature must be positive, got {temperature}")
    top_p = _number(raw["top_p"], "sampling.top_p")
    if not 0.0 < top_p <= 1.0:
        raise JobError(f"sampling.top_p must be in (0, 1], got {top_p}")
    top_k = raw["top_k"]
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise JobError(f"sampling.top_k must be a non-negative whole number, got {top_k!r}")
    max_new_tokens = _positive_int(raw, "max_new_tokens")
    if max_new_tokens > MAX_COMPLETION_TOKENS:
        raise JobError(
            f"sampling.max_new_tokens {max_new_tokens} is above the wire's "
            f"{MAX_COMPLETION_TOKENS}, so a completion at this job's cap could "
            "never be submitted"
        )
    min_new_tokens = _positive_int(raw, "min_new_tokens")
    if min_new_tokens < MIN_NEW_TOKENS_FLOOR:
        raise JobError(
            f"sampling.min_new_tokens must be at least {MIN_NEW_TOKENS_FLOOR}, "
            f"got {min_new_tokens}: the terminator counts toward the budget, so "
            "a lower floor pays a slot for a completion with no text in it"
        )
    if min_new_tokens > max_new_tokens:
        raise JobError(
            f"sampling.min_new_tokens {min_new_tokens} exceeds max_new_tokens {max_new_tokens}"
        )
    n = _positive_int(raw, "n")
    if n > MAX_COMPLETIONS_PER_SUBMISSION:
        raise JobError(
            f"sampling.n {n} is above the wire's "
            f"{MAX_COMPLETIONS_PER_SUBMISSION} completions per submission, so "
            "every submission this job asks for would be refused unparsed"
        )
    return Sampling(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_new_tokens=min_new_tokens,
        max_new_tokens=max_new_tokens,
        n=n,
    )


def _parse_filter(raw: Any) -> Filter | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise JobError("filter must be an object or null")
    unknown = set(raw) - set(_FILTER_FIELDS)
    if unknown:
        raise JobError(f"filter has unknown fields: {sorted(unknown)}")
    missing = [f for f in _FILTER_FIELDS if f not in raw]
    if missing:
        raise JobError(f"filter is missing: {', '.join(missing)}")
    return Filter(
        grader_id=_text(raw, "grader_id"),
        threshold=_number(raw["threshold"], "filter.threshold"),
    )


def _fields(raw: Any, allowed: tuple[str, ...], label: str,
            optional: tuple[str, ...] = ()) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise JobError(f"{label} must be an object")
    unknown = set(raw) - set(allowed) - set(optional)
    if unknown:
        raise JobError(f"{label} has unknown fields: {sorted(unknown)}")
    missing = [f for f in allowed if f not in raw]
    if missing:
        raise JobError(f"{label} is missing: {', '.join(missing)}")
    return raw


def _bounded_int(raw: Mapping[str, Any], field: str, low: int, high: int, label: str) -> int:
    value = raw[field]
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise JobError(f"{label}.{field} must be a whole number in [{low}, {high}], got {value!r}")
    return value


def _parse_sandbox(raw: Any, package: str) -> SandboxSpec:
    raw = _fields(raw, _SANDBOX_FIELDS, "episode.sandbox")
    if raw["env"] != package:
        raise JobError(f"episode.sandbox.env must be the episode's package {package!r}")
    env_package = raw["env_package"]
    if (not isinstance(env_package, str) or not env_package.startswith(f"{package}==")
            or not _ENV_PACKAGE_SUFFIX_RE.match(env_package[len(package) + 2:])):
        raise JobError("episode.sandbox.env_package must read '<package>==<version>+g<16 hex>', "
                       "the env_package record 0 carries")
    tools = raw["tools"]
    if tools not in _SANDBOX_TOOL_SETS:
        raise JobError("episode.sandbox.tools must be ['bash'] or ['bash', 'edit']")
    commit = raw["sandbox_commit"]
    if not isinstance(commit, str) or not _COMMIT_RE.match(commit):
        raise JobError("episode.sandbox.sandbox_commit must be a 40-hex commit")
    budgets_raw = _fields(raw["budgets"], SANDBOX_BUDGET_FIELDS, "episode.sandbox.budgets")
    budgets = SandboxBudgets(**{
        name: _bounded_int(budgets_raw, name, 1, _SANDBOX_BUDGET_BOUNDS[name],
                           "episode.sandbox.budgets")
        for name in SANDBOX_BUDGET_FIELDS})
    return SandboxSpec(env=raw["env"], env_package=env_package, tools=tuple(tools),
                       sandbox_commit=commit, budgets=budgets)


def _parse_episode(raw: Any) -> EpisodeSpec:
    raw = _fields(raw, _EPISODE_FIELDS, "episode", optional=_EPISODE_OPTIONAL_FIELDS)
    env = _fields(raw["env"], _EPISODE_ENV_FIELDS, "episode.env")
    if not isinstance(env["package"], str) or env["package"] not in EPISODE_ENV_PACKAGES:
        raise JobError(f"episode.env.package must be one of {sorted(EPISODE_ENV_PACKAGES)}")
    if not isinstance(env["version"], str) or not _COMMIT_RE.match(env["version"]):
        raise JobError("episode.env.version must be the environment repository's 40-hex commit")
    if env["split"] != "train":
        raise JobError(f"episode.env.split must be 'train', got {env['split']!r}")
    if raw["harness"] not in EPISODE_HARNESSES:
        raise JobError(f"episode.harness must be one of {sorted(EPISODE_HARNESSES)}")
    if not isinstance(raw["renderer"], str) or not _RENDERER_RE.match(raw["renderer"]):
        raise JobError(f"episode.renderer must read renderers:<name>@<version>, got {raw['renderer']!r}")
    if not isinstance(raw["verifiers"], str) or not _COMMIT_RE.match(raw["verifiers"]):
        raise JobError("episode.verifiers must be a 40-hex commit")
    max_total = _bounded_int(raw, "max_total_tokens", 1, MAX_EPISODE_TOTAL_TOKENS, "episode")
    fraction = raw["replay_fraction_failed"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0.0 <= fraction <= 1.0:
        raise JobError(f"episode.replay_fraction_failed must be in [0, 1], got {fraction!r}")
    if "execution" in raw and raw["execution"] == EXECUTION_REPLAY:
        raise JobError("episode.execution is written only when it is not 'replay'")
    execution = raw.get("execution", EXECUTION_REPLAY)
    if not isinstance(execution, str) or execution not in EXECUTIONS:
        raise JobError(f"episode.execution must be one of {sorted(EXECUTIONS)}, got {execution!r}")
    sandbox = None
    if execution == EXECUTION_SIGNED_SANDBOX:
        if "sandbox" not in raw:
            raise JobError("a signed_sandbox episode needs episode.sandbox")
        sandbox = _parse_sandbox(raw["sandbox"], env["package"])
    elif "sandbox" in raw:
        raise JobError("episode.sandbox is only for execution 'signed_sandbox'")
    episode = EpisodeSpec(
        env=EpisodeEnv(package=env["package"], version=env["version"], split=env["split"],
                       num_images=_bounded_int(env, "num_images", 1, 1000, "episode.env")),
        harness=raw["harness"], renderer=raw["renderer"], verifiers=raw["verifiers"],
        max_turns=_bounded_int(raw, "max_turns", 1, MAX_EPISODE_TURNS, "episode"),
        max_tokens_per_turn=_bounded_int(raw, "max_tokens_per_turn", 1, max_total, "episode"),
        max_total_tokens=max_total,
        replay_fraction_failed=float(fraction),
        execution=execution, sandbox=sandbox,
    )
    return episode


def _check_episode_job(job: JobSpec) -> None:
    """The rules that bind the rest of a manifest once it carries `episode`."""
    episode = job.episode
    if job.sampling.n != 1:
        raise JobError("an episode job submits one trajectory per slot: sampling.n must be 1")
    if job.slots_per_prompt < 2:
        raise JobError("an episode job needs slots_per_prompt >= 2 (spec §6)")
    if job.filter is not None:
        raise JobError("an episode job is graded by its grade executors, not by a filter")
    if job.prompt_order != PROMPT_ORDER_FREE:
        raise JobError("an episode job runs episodes concurrently: prompt_order must be 'free'")
    if job.sampling.max_new_tokens != episode.max_tokens_per_turn:
        raise JobError("sampling.max_new_tokens must equal episode.max_tokens_per_turn")
    if job.renderer_id != episode.renderer:
        raise JobError("renderer_id must equal episode.renderer")
    if episode.execution == EXECUTION_SIGNED_SANDBOX:
        if episode.replay_fraction_failed != 0.0:
            raise JobError("a signed_sandbox episode is never replayed: "
                           "replay_fraction_failed must be 0")
        if episode.sandbox.budgets.max_calls < episode.max_turns:
            raise JobError("episode.sandbox.budgets.max_calls must be at least episode.max_turns")


def parse_job(raw: Mapping[str, Any]) -> JobSpec:
    """Read one manifest, or refuse it naming exactly what is wrong."""
    if not isinstance(raw, Mapping):
        raise JobError("a job manifest must be an object")
    unknown = set(raw) - set(_JOB_FIELDS) - set(_OPTIONAL_JOB_FIELDS)
    if unknown:
        raise JobError(f"unknown job fields: {sorted(unknown)}")
    missing = [f for f in _JOB_FIELDS if f not in raw]
    if missing:
        raise JobError(f"job is missing: {', '.join(missing)}")
    if raw["schema"] != JOB_SCHEMA:
        raise JobError(f"unsupported job schema {raw['schema']!r}")

    job_id = _text(raw, "job_id")
    if not JOB_ID_RE.match(job_id):
        raise JobError(f"unusable job id {job_id!r}")

    checkpoint_sha256 = _text(raw, "checkpoint_sha256")
    if not _SHA256_RE.match(checkpoint_sha256):
        raise JobError("checkpoint_sha256 must be 64 lowercase hex characters")

    prompt_order = _text(raw, "prompt_order")
    if prompt_order not in PROMPT_ORDERS:
        raise JobError(f"unknown prompt order {prompt_order!r}")

    deadline_round = raw["deadline_round"]
    if deadline_round is not None and (
        isinstance(deadline_round, bool)
        or not isinstance(deadline_round, int)
        or deadline_round < 0
    ):
        raise JobError(f"deadline_round must be a non-negative round or null, got {deadline_round!r}")

    job = JobSpec(
        job_id=job_id,
        checkpoint_repo=_text(raw, "checkpoint_repo"),
        checkpoint_revision=_text(raw, "checkpoint_revision"),
        checkpoint_sha256=checkpoint_sha256,
        prompt_source=_text(raw, "prompt_source"),
        prompt_count=_positive_int(raw, "prompt_count"),
        renderer_id=_text(raw, "renderer_id"),
        eos_token_id=_non_negative_int(raw, "eos_token_id"),
        sampling=_parse_sampling(raw["sampling"]),
        slots_per_prompt=_positive_int(raw, "slots_per_prompt"),
        filter=_parse_filter(raw["filter"]),
        prompt_order=prompt_order,
        deadline_round=deadline_round,
        prompt_start=(
            _non_negative_int(raw, "prompt_start") if "prompt_start" in raw else 0
        ),
        seed=_non_negative_int(raw, "seed") if "seed" in raw else None,
        submit=_submit(raw),
        episode=_parse_episode(raw["episode"]) if "episode" in raw else None,
    )
    if job.episode is not None:
        _check_episode_job(job)
    return job


def is_signed_sandbox(job: JobSpec) -> bool:
    return job.episode is not None and job.episode.execution == EXECUTION_SIGNED_SANDBOX


BRIDGED_SWESMITH_IMAGES = 20
"""SWE-smith's image count in the split a gateway serves through the verifiers bridge
(reliquary-swe's default)."""


def sandbox_split(episode: EpisodeSpec) -> str:
    """The split a session token names: reliquary-swe's `train` (SWE-smith at its default
    20 images, the split the bridge serves)."""
    if episode.env.num_images != BRIDGED_SWESMITH_IMAGES:
        raise ValueError(f"a signed reliquary-swe job serves SWE-smith at "
                         f"{BRIDGED_SWESMITH_IMAGES} images, not {episode.env.num_images}")
    return "train"
