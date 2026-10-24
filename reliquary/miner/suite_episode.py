"""One suite episode of a corpus job (design 2026-10-13), on the miner: a
registry episode environment (first ``reliquary_telecom_solo_v2``) driven by
the RL ``EpisodeRunner``, every assistant turn generated and proven by the
miner's ``GenerateEngine``.

No verifiers, no Docker, no network: the environment is a pinned wheel and the
loop is the one RL mining and the validator's replay already share. The
trajectory is the SWE corpus one (``corpus.trajectory``), with no diff. What
the validator decides is shared, not copied: the seed, the turn ids, the
replay and its comparison all come from ``reliquary.corpus.suite_episode``.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import Callable

from reliquary.corpus.suite_episode import (
    SuiteSource,
    episode_seed,
    job_suite_source,
    replay_outcome,
    replay_stop,
    suite_spec,
    suite_span_refusal,
    suite_spec_refusal,
    turn_ids,
)
from reliquary.corpus.trajectory import BuiltTrajectory
from reliquary.miner.agentic_episode import EpisodeResult

logger = logging.getLogger(__name__)

# The RL episode profile's observation bound (EpisodeRunner's default too), as
# the validator's replay uses.
MAX_OBSERVATION_BYTES = 64 * 1024
# A suite episode is CPU tool calls between turns: the bound is the turns.
DEADLINE_SECONDS = 3600.0


def is_suite_episode(episode) -> bool:
    return bool(getattr(episode, "is_suite", False))


def suite_support(job) -> tuple[str | None, SuiteSource | None]:
    """``(refusal, None)`` when this miner cannot run the suite ``job``, else
    ``(None, source)``: the job's one source, for its runner. The validator's
    own refusals (registry, manifest pin, renderer, turn limit), then the
    installed wheel must be the artifact that manifest pins, and the job's
    source must build (for an eval job: its rows registered, at its split,
    every reference the instance it names)."""
    from reliquary.environment.agentic.external import verify_external_artifact

    episode = job.episode
    refusal = suite_spec_refusal(episode)
    if refusal:
        return refusal, None
    spec = suite_spec(episode)
    if spec.external_distribution is not None:
        try:
            verify_external_artifact(spec)
        except (ValueError, TypeError) as exc:
            return (f"{spec.name} needs the {spec.external_distribution} wheel at manifest "
                    f"{episode.env.version}: {exc}"), None
    try:
        return None, job_suite_source(job)
    except (ValueError, RuntimeError, TypeError) as exc:
        return str(exc), None


def suite_support_refusal(job) -> str | None:
    """Why this miner cannot run the suite ``job``, or None (``suite_support``)."""
    return suite_support(job)[0]


def suite_stop_ids(encode: Callable[[str], list[int]]) -> list[int]:
    """What a turn stops on: ``<|im_end|>`` first, then ``<|endoftext|>``."""
    terminator, stops = turn_ids(encode)
    return [terminator, *sorted(stops - {terminator})]


def trajectory_checks_refusal(built: BuiltTrajectory, *, episode, terminator_id: int,
                              stop_ids, chunk_tokens: int, encode: Callable[[str], list[int]],
                              decode: Callable[[list[int]], str]) -> tuple[str, dict] | None:
    """The suite intake's checks, in its order: spans, one turn per span
    (``suite_span_refusal``: no stop id before a span's end, no turn markup
    in it, so a turn that writes its own tool response and a second turn is
    refused), then short turns, budget, termination and proof shape. A turn
    other than the last must end on ``<|im_end|>`` or at its cap: one that
    ended on ``<|endoftext|>`` mid-episode is refused (bad_termination)."""
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
        inside = suite_span_refusal(tokens, spans, encode=encode, decode=decode)
        if inside is not None:
            return inside[0], dict(inside[1])
        for check in (
            lambda: check_short_turns(spans),
            lambda: check_turn_budget(spans, prompt_len=len(built.prompt_ids), length=len(tokens),
                                      max_tokens_per_turn=episode.max_tokens_per_turn,
                                      max_total_tokens=episode.max_total_tokens),
            lambda: check_turn_termination(tokens, spans, prompt_len=len(built.prompt_ids),
                                           terminator_id=terminator_id, stop_ids=set(stop_ids),
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
    """``SweEpisodeRunner``'s interface for a suite episode job. ``encode``
    and ``decode`` must code as the validator does
    (``reliquary.corpus.suite_episode.TokenizerCodec`` over the checkpoint's
    tokenizer)."""

    def __init__(self, *, job, engine, encode: Callable[[str], list[int]],
                 decode: Callable[[list[int]], str], chunk_tokens: int | None = None,
                 source=None) -> None:
        from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS

        self._job = job
        self._episode = job.episode
        self._engine = engine
        self._encode = encode
        self._decode = decode
        self._chunk_tokens = int(chunk_tokens or TOPLOC_DEPLOYED_DEFAULTS.chunk_tokens)
        self._source = source if source is not None else job_suite_source(job)
        self._environment = self._source.environment
        self._renderer = self._source.renderer(encode)
        self.terminator_id, stops = turn_ids(encode)
        self.stop_ids = [self.terminator_id, *sorted(stops - {self.terminator_id})]
        # `SuiteSource` refuses a job above the environment's; the min is RL's.
        self.max_turns = min(self._episode.max_turns, int(self._environment.max_turns))

    @property
    def environment(self):
        return self._environment

    async def __aenter__(self) -> "SuiteEpisodeRunner":
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    def deadline(self, index: int) -> float:
        del index
        return DEADLINE_SECONDS

    def initial_ids(self, index: int) -> list[int]:
        return self._renderer.encode_initial(self._source.task(index))

    def _run(self, index: int, session_id: str, loop: asyncio.AbstractEventLoop):
        from reliquary.environment.agentic.runner import EpisodeRunner

        # No max_episode_tokens: the runner would count the final suffix and
        # the observation after the last turn, which the trajectory never
        # holds, and drop episodes the validator accepts. The engine caps each
        # turn at what is left of max_total_tokens (and refuses a turn with no
        # room), and the precheck's budget check bounds what is submitted.
        return EpisodeRunner(
            renderer=self._renderer, max_turns=self.max_turns,
            max_observation_bytes=MAX_OBSERVATION_BYTES,
        ).run(self._environment, self._source.task(index),
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
            # No room left for a turn under max_total_tokens, an engine
            # failure or an unprovable turn: nothing to sign.
            return EpisodeResult(session_id, "", None, False, None,
                                 error=f"{type(exc).__name__}: {exc}")
        if not trace.assistant_spans:
            return EpisodeResult(session_id, "", None, False, None, error="no assistant turn")
        reward = trace.reward.reward if trace.reward is not None else None
        # The environment ended it: agent_completed; the turn limit: max_turns.
        return EpisodeResult(session_id, "", replay_stop(trace), True, reward)

    def precheck(self, built: BuiltTrajectory, prompt_index: int) -> tuple[str, dict] | None:
        """What the validator's suite intake would refuse, before anything is
        signed: its generic checks, the prompt, then its very replay and
        ``compare_replay``, which also refuses an observation carrying turn
        markup (an environment echoing a JSON-escaped argument)."""
        from reliquary.validator.suite_intake import replay_kwargs

        refusal = trajectory_checks_refusal(
            built, episode=self._episode, terminator_id=self.terminator_id,
            stop_ids=self.stop_ids, chunk_tokens=self._chunk_tokens, encode=self._encode,
            decode=self._decode)
        if refusal is not None:
            return refusal
        prompt = list(built.prompt_ids)
        if prompt != self.initial_ids(prompt_index) \
                or self._decode(prompt) != self._source.initial_text(prompt_index):
            return "prompt_mismatch", {"why": "the first turn's prompt is not the task's"}
        # This runner's environment is built at the job's split already.
        kwargs = replay_kwargs(self._job, self._source, prompt_index=prompt_index,
                               prompt_ids=prompt, tokens=list(built.tokens),
                               spans=list(built.spans), stop=built.stop, with_split=False)
        outcome = replay_outcome(self._environment, encode=self._encode, decode=self._decode,
                                 **kwargs)
        return outcome.refusal


__all__ = ["SuiteEpisodeRunner", "is_suite_episode", "suite_stop_ids", "suite_support",
           "suite_support_refusal", "trajectory_checks_refusal"]
