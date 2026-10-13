"""One suite episode of a corpus job: a registered episode environment (first
``reliquary_telecom_solo_v1``) driven by the RL ``EpisodeRunner``, with every
assistant turn generated and proven by the miner's ``GenerateEngine``.

No verifiers, no Docker, no network: the environment is a pinned wheel and the
loop is the one the RL miner and the validator's replay already share. The
trajectory is the SWE corpus one (``corpus.trajectory``), with no diff.

Until ``corpus.job`` knows ``episode.harness == "suite"``, the suite manifest
is read here (``read_episode_job``); the rest of the job goes through
``parse_job`` unchanged.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from reliquary.corpus.job import (
    MAX_EPISODE_TOTAL_TOKENS,
    MAX_EPISODE_TURNS,
    PROMPT_ORDER_FREE,
    JobError,
    JobSpec,
    parse_job,
)
from reliquary.corpus.trajectory import BuiltTrajectory
from reliquary.miner.agentic_episode import EpisodeResult

logger = logging.getLogger(__name__)

SUITE_HARNESS = "suite"
SUITE_PACKAGE = "reliquary-suite"
# What a turn stops on: the ChatML renderer's turn terminators, in this order
# (the first one is the only one a turn other than the last may end on).
SUITE_STOP_TEXTS = ("<|im_end|>", "<|endoftext|>")
# The RL episode profile's observation bound (EpisodeRunner's default too).
MAX_OBSERVATION_BYTES = 64 * 1024
# A suite episode is CPU tool calls between turns: the bound is the turns.
DEADLINE_SECONDS = 3600.0

_REQUIRED_EPISODE = ("env", "harness", "renderer", "max_turns", "max_tokens_per_turn",
                     "max_total_tokens")
_REQUIRED_ENV = ("package", "name", "version", "split")


@dataclass(frozen=True, slots=True)
class SuiteEnv:
    package: str
    name: str
    version: str
    split: str

    def to_contract(self) -> dict[str, Any]:
        return {"package": self.package, "name": self.name, "version": self.version,
                "split": self.split}


@dataclass(frozen=True, slots=True)
class SuiteEpisodeSpec:
    """The episode of a suite job: a registered environment, its renderer and bounds."""

    env: SuiteEnv
    harness: str
    renderer: str
    max_turns: int
    max_tokens_per_turn: int
    max_total_tokens: int

    def to_contract(self) -> dict[str, Any]:
        return {"env": self.env.to_contract(), "harness": self.harness,
                "renderer": self.renderer, "max_turns": self.max_turns,
                "max_tokens_per_turn": self.max_tokens_per_turn,
                "max_total_tokens": self.max_total_tokens}


def is_suite_episode(episode) -> bool:
    return isinstance(episode, SuiteEpisodeSpec)


def _is_suite_manifest(raw: Any) -> bool:
    episode = raw.get("episode") if isinstance(raw, Mapping) else None
    return isinstance(episode, Mapping) and episode.get("harness") == SUITE_HARNESS


def _bounded(raw: Mapping[str, Any], field: str, high: int) -> int:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= high:
        raise JobError(f"episode.{field} must be a whole number in [1, {high}], got {value!r}")
    return value


def _text(raw: Mapping[str, Any], field: str, label: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value or value.strip() != value:
        raise JobError(f"{label}.{field} must be non-empty trimmed text, got {value!r}")
    return value


def parse_suite_episode(raw: Any) -> SuiteEpisodeSpec:
    """The fields this miner needs, read defensively: fields it does not use
    are ignored, so a manifest the validator side extends still mines."""
    if not isinstance(raw, Mapping):
        raise JobError("episode must be an object")
    missing = [f for f in _REQUIRED_EPISODE if f not in raw]
    if missing:
        raise JobError(f"episode is missing: {', '.join(missing)}")
    env = raw["env"]
    if not isinstance(env, Mapping):
        raise JobError("episode.env must be an object")
    missing = [f for f in _REQUIRED_ENV if f not in env]
    if missing:
        raise JobError(f"episode.env is missing: {', '.join(missing)}")
    if raw["harness"] != SUITE_HARNESS:
        raise JobError(f"episode.harness must be {SUITE_HARNESS!r}, got {raw['harness']!r}")
    if env["package"] != SUITE_PACKAGE:
        raise JobError(f"episode.env.package must be {SUITE_PACKAGE!r}, got {env['package']!r}")
    if env["split"] != "train":
        raise JobError(f"episode.env.split must be 'train', got {env['split']!r}")
    max_total = _bounded(raw, "max_total_tokens", MAX_EPISODE_TOTAL_TOKENS)
    return SuiteEpisodeSpec(
        env=SuiteEnv(package=env["package"], name=_text(env, "name", "episode.env"),
                     version=_text(env, "version", "episode.env"), split=env["split"]),
        harness=SUITE_HARNESS, renderer=_text(raw, "renderer", "episode"),
        max_turns=_bounded(raw, "max_turns", MAX_EPISODE_TURNS),
        max_tokens_per_turn=_bounded(raw, "max_tokens_per_turn", max_total),
        max_total_tokens=max_total)


def read_episode_job(raw: Mapping[str, Any]) -> JobSpec:
    """``parse_job``, plus suite episode manifests, which it does not know yet."""
    if not _is_suite_manifest(raw):
        return parse_job(raw)
    episode = parse_suite_episode(raw["episode"])
    job = replace(parse_job({k: v for k, v in raw.items() if k != "episode"}), episode=episode)
    if job.sampling.n != 1:
        raise JobError("an episode job submits one trajectory per slot: sampling.n must be 1")
    if job.prompt_order != PROMPT_ORDER_FREE:
        raise JobError("an episode job runs episodes concurrently: prompt_order must be 'free'")
    if job.renderer_id != episode.renderer:
        raise JobError("renderer_id must equal episode.renderer")
    return job


def episode_seed(job_id: str, prompt_index: int) -> int:
    """The environment seed of every episode on one prompt of one job: the
    first 8 bytes of sha256(f"{job_id}:{prompt_index}"), big-endian, the way
    the RL miner turns its seed material into an int (``int(hexdigest[:16], 16)``)."""
    return int.from_bytes(hashlib.sha256(f"{job_id}:{prompt_index}".encode()).digest()[:8], "big")


def suite_environment_spec(episode: SuiteEpisodeSpec):
    from reliquary.environment.registry import get_environment_spec

    return get_environment_spec(episode.env.name)


def suite_support_refusal(episode: SuiteEpisodeSpec) -> str | None:
    """Why this miner cannot run ``episode``, or None: the environment must be
    a registered episode environment written in the job's renderer, its
    manifest pinned by the job, and its wheel installed with that manifest."""
    from reliquary.environment.agentic.external import verify_external_artifact

    try:
        spec = suite_environment_spec(episode)
    except ValueError as exc:
        return f"episode.env.name: {exc}"
    if spec.interaction_mode != "episode":
        return f"{spec.name} is not an episode environment"
    if spec.renderer_id != episode.renderer:
        return (f"{spec.name} is rendered in {spec.renderer_id!r}, the job says "
                f"{episode.renderer!r}")
    if spec.environment_manifest_sha256 != episode.env.version:
        return (f"this reliquary pins {spec.name} at manifest {spec.environment_manifest_sha256}, "
                f"the job pins {episode.env.version}: update reliquary")
    if spec.external_distribution is not None:
        try:
            verify_external_artifact(spec)
        except (ValueError, TypeError) as exc:
            return (f"{spec.name} needs the {spec.external_distribution} wheel at manifest "
                    f"{episode.env.version}: {exc}")
    return None


def suite_stop_ids(encode: Callable[[str], list[int]]) -> list[int]:
    """The ids a turn stops on, in ``SUITE_STOP_TEXTS`` order."""
    ids = []
    for text in SUITE_STOP_TEXTS:
        encoded = list(encode(text))
        if len(encoded) != 1:
            raise ValueError(f"{text!r} is not one token in this tokenizer: {encoded}")
        ids.append(int(encoded[0]))
    return ids


def _stop_for(termination_reason: str) -> str:
    # The environment ended the episode (any reason it gives), or the runner's turn limit did.
    return "max_turns" if termination_reason == "turn_limit" else "agent_completed"


def replay_refusal(environment, *, prompt_index: int, seed: int, built: BuiltTrajectory,
                   renderer_id: str, encode, decode,
                   max_observation_bytes: int = MAX_OBSERVATION_BYTES) -> tuple[str, dict] | None:
    """The validator's replay (``replay_tokenized_episode``) of a built
    trajectory, or why it would not match: the replayed transcript must
    reproduce the prompt and every token up to the last turn, with the same
    assistant spans, and end the way the trajectory says it stopped."""
    from reliquary.environment.agentic.replay import replay_tokenized_episode

    prompt = list(built.prompt_ids)
    sequence = prompt + list(built.tokens)
    spans = [(len(prompt) + start, len(prompt) + end) for start, end in built.spans]
    try:
        trace = replay_tokenized_episode(
            environment, task_index=prompt_index, seed=seed, tokens=sequence,
            assistant_spans=spans, decode=decode, encode=encode,
            max_observation_bytes=max_observation_bytes, renderer_id=renderer_id)
    except (RecursionError, TypeError, ValueError) as exc:
        return "replay_failed", {"error": f"{type(exc).__name__}: {exc}"}
    replayed = [tuple(span) for span in trace.assistant_spans]
    if replayed[:len(spans)] != spans:
        return "replay_mismatch", {"why": "assistant spans differ"}
    if list(trace.tokens[:len(sequence)]) != sequence:
        return "replay_mismatch", {"why": "the replayed transcript differs"}
    ended = len(replayed) == len(spans) and trace.termination_reason != "turn_limit"
    if built.stop == "agent_completed" and not ended:
        return "replay_mismatch", {"why": "the environment does not end on the last turn"}
    if built.stop == "max_turns" and len(replayed) == len(spans) \
            and trace.termination_reason != "turn_limit":
        return "replay_mismatch", {"why": "the environment ended, the trajectory says max_turns"}
    return None


def trajectory_checks_refusal(built: BuiltTrajectory, *, episode: SuiteEpisodeSpec,
                              stop_ids: list[int], chunk_tokens: int) -> tuple[str, dict] | None:
    """The intake's span, budget, termination and proof-shape checks. A turn
    other than the last must end on ``<|im_end|>`` or at its cap: one that
    ended on ``<|endoftext|>`` mid-episode is refused here (bad_termination)."""
    from reliquary.corpus.checks import (
        check_short_turns,
        check_turn_budget,
        check_turn_proof_shape,
        check_turn_spans,
        check_turn_termination,
    )

    spans = [tuple(span) for span in built.spans]
    tokens = list(built.tokens)
    result = check_turn_spans(spans, len(tokens), episode.max_turns)
    if result.ok:
        for check in (
            lambda: check_short_turns(spans),
            lambda: check_turn_budget(spans, prompt_len=len(built.prompt_ids), length=len(tokens),
                                      max_tokens_per_turn=episode.max_tokens_per_turn,
                                      max_total_tokens=episode.max_total_tokens),
            lambda: check_turn_termination(tokens, spans, prompt_len=len(built.prompt_ids),
                                           terminator_id=stop_ids[0], stop_ids=set(stop_ids),
                                           max_tokens_per_turn=episode.max_tokens_per_turn,
                                           max_total_tokens=episode.max_total_tokens),
            lambda: check_turn_proof_shape(spans, [len(p) for p in built.proofs], chunk_tokens),
        ):
            result = check()
            if not result.ok:
                break
    if not result.ok:
        return result.reason or "bad_turns", dict(result.detail)
    return None


class _EnginePolicy:
    """One episode's turns, each generated by the engine on the event loop
    while the runner waits in its worker thread."""

    def __init__(self, *, engine, session_id: str, loop: asyncio.AbstractEventLoop,
                 decode: Callable[[list[int]], str]) -> None:
        self._engine = engine
        self._session = session_id
        self._loop = loop
        self._decode = decode

    def generate(self, *, tokens: tuple[int, ...], **_: object):
        from reliquary.environment.agentic.types import GeneratedAction

        answer = asyncio.run_coroutine_threadsafe(
            self._engine.generate(self._session, list(tokens), None), self._loop).result()
        ids = [int(t) for t in answer["choices"][0]["token_ids"]]
        # Raises past the action byte bound: the validator's replay would too.
        return GeneratedAction(text=self._decode(ids), tokens=tuple(ids))


class SuiteEpisodeRunner:
    """``SweEpisodeRunner``'s interface for a suite episode job."""

    def __init__(self, *, job, engine, encode: Callable[[str], list[int]],
                 decode: Callable[[list[int]], str], chunk_tokens: int | None = None,
                 environment=None) -> None:
        from reliquary.environment.agentic.renderers import renderer_for
        from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS

        self._job = job
        self._episode: SuiteEpisodeSpec = job.episode
        self._engine = engine
        self._encode = encode
        self._decode = decode
        self._chunk_tokens = int(chunk_tokens or TOPLOC_DEPLOYED_DEFAULTS.chunk_tokens)
        self._environment = environment if environment is not None \
            else suite_environment_spec(self._episode).create()
        self._renderer = renderer_for(self._episode.renderer, encode)
        self.stop_ids = suite_stop_ids(encode)
        self.max_turns = min(self._episode.max_turns, int(self._environment.max_turns))

    async def __aenter__(self) -> "SuiteEpisodeRunner":
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    def deadline(self, index: int) -> float:
        del index
        return DEADLINE_SECONDS

    def initial_ids(self, index: int) -> list[int]:
        return self._renderer.encode_initial(self._environment.get_task(index))

    def _run(self, index: int, session_id: str, loop: asyncio.AbstractEventLoop):
        from reliquary.environment.agentic.runner import EpisodeRunner

        return EpisodeRunner(
            renderer=self._renderer, max_turns=self.max_turns,
            max_episode_tokens=self._episode.max_total_tokens,
            max_observation_bytes=MAX_OBSERVATION_BYTES,
        ).run(self._environment, self._environment.get_task(index),
              seed=episode_seed(self._job.job_id, index),
              policy=_EnginePolicy(engine=self._engine, session_id=session_id, loop=loop,
                                   decode=self._decode))

    async def run(self, index: int, on_session=None) -> EpisodeResult:
        session_id = f"suite-{secrets.token_hex(12)}"
        if on_session is not None:
            on_session(session_id)
        try:
            trace = await asyncio.to_thread(self._run, index, session_id,
                                            asyncio.get_running_loop())
        except (RuntimeError, ValueError) as exc:
            # A budget overflow (the runner's own bound, or no room left for a
            # turn), an engine failure or an unprovable turn: nothing to sign.
            return EpisodeResult(session_id, "", None, False, None,
                                 error=f"{type(exc).__name__}: {exc}")
        if not trace.assistant_spans:
            return EpisodeResult(session_id, "", None, False, None, error="no assistant turn")
        reward = trace.reward.reward if trace.reward is not None else None
        return EpisodeResult(session_id, "", _stop_for(trace.termination_reason), True, reward)

    def precheck(self, built: BuiltTrajectory, prompt_index: int) -> tuple[str, dict] | None:
        """What the validator would refuse, before anything is signed: the
        intake's generic checks, then the exact replay."""
        refusal = trajectory_checks_refusal(built, episode=self._episode, stop_ids=self.stop_ids,
                                            chunk_tokens=self._chunk_tokens)
        if refusal is not None:
            return refusal
        if list(built.prompt_ids) != self.initial_ids(prompt_index):
            return "prompt_mismatch", {"why": "the first turn's prompt is not the task's"}
        return replay_refusal(self._environment, prompt_index=prompt_index,
                              seed=episode_seed(self._job.job_id, prompt_index), built=built,
                              renderer_id=self._episode.renderer, encode=self._encode,
                              decode=self._decode)


__all__ = ["SUITE_HARNESS", "SUITE_PACKAGE", "SuiteEnv", "SuiteEpisodeRunner",
           "SuiteEpisodeSpec", "episode_seed", "is_suite_episode", "parse_suite_episode",
           "read_episode_job", "replay_refusal", "suite_stop_ids", "suite_support_refusal",
           "trajectory_checks_refusal"]
