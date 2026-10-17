"""Suite episodes in a corpus job: one task of a registry episode environment,
run by the miner and replayed by the validator (design
``docs/design/2026-10-13-corpus-suite-episodes.md``).

The RL validator already verifies these episodes by replay
(``replay_tokenized_episode``): the miner's assistant spans are fed back to
the environment and every observation is re-rendered with the validator's
tokenizer, so the transcript either is the environment's or is refused. A
corpus job reuses that replay unchanged; what differs is only where a corpus
trajectory starts and stops. It holds the tokens after the initial prompt up
to the end of the last assistant span, and the replay's own tokens go on past
that (the final suffix, or the observation after a last call at the turn
limit), so the comparison stops where the trajectory does.

Nothing here imports a model or a tokenizer: callers pass ``encode`` and
``decode``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reliquary.corpus.job import SUITE_PACKAGE, SUITE_STOPS, EpisodeSpec
from reliquary.eval.rows import RowMap

# The renderers a suite job may declare: the ones whose turns end on a single
# terminator token, which the corpus termination check and the export's chat
# template transform both rely on. The JSONL dialect is RL-only.
SUITE_RENDERERS = frozenset({"reliquary-chatml-tools-v1"})
IM_END = "<|im_end|>"
END_OF_TEXT = "<|endoftext|>"
# The replay's turn-limit label (``EpisodeRunner``); every other termination
# is the environment's own, which a miner reports as ``agent_completed``.
TURN_LIMIT = "turn_limit"


def episode_seed(job_id: str, prompt_index: int) -> int:
    """The seed both sides run one prompt's episode with: the first 8 bytes,
    big-endian, of sha256(``f"{job_id}:{prompt_index}"``), a 64-bit seed as
    RL's. Never sent: the validator derives it, so a miner using another seed
    only gets refused where the environment's observations depend on it."""
    digest = hashlib.sha256(f"{job_id}:{int(prompt_index)}".encode()).hexdigest()
    return int(digest[:16], 16)


def _registry(specs: Mapping[str, Any] | None):
    if specs is not None:
        return specs
    from reliquary.environment.registry import environment_catalog

    return environment_catalog()


def suite_spec_refusal(episode: EpisodeSpec, *, specs: Mapping[str, Any] | None = None) -> str | None:
    """Why this binary cannot serve the suite ``episode``, or None: its
    environment must be in the registry, an episode environment, at the
    manifest the job pins, rendered in the dialect the job declares."""
    if not episode.is_suite or episode.env.package != SUITE_PACKAGE:
        return f"not a suite episode (harness {episode.harness!r})"
    name = episode.env.name
    spec = _registry(specs).get(name)
    if spec is None:
        return f"episode environment {name!r} is not in this binary's registry"
    if getattr(spec, "interaction_mode", None) != "episode":
        return f"environment {name!r} is {spec.interaction_mode!r}, not an episode environment"
    if spec.environment_manifest_sha256 != episode.env.version:
        return (f"environment {name!r} is at manifest {spec.environment_manifest_sha256}, "
                f"the job pins {episode.env.version}")
    if spec.renderer_id != episode.renderer:
        return (f"environment {name!r} renders with {spec.renderer_id!r}, the job declares "
                f"{episode.renderer!r}")
    if spec.renderer_id not in SUITE_RENDERERS:
        return (f"environment {name!r} renders with {spec.renderer_id!r}; a corpus job takes "
                f"only {sorted(SUITE_RENDERERS)}")
    return None


def suite_spec(episode: EpisodeSpec, *, specs: Mapping[str, Any] | None = None):
    """The registry spec of a suite episode, or a ``ValueError`` naming why not."""
    refusal = suite_spec_refusal(episode, specs=specs)
    if refusal:
        raise ValueError(refusal)
    return _registry(specs)[episode.env.name]


class TokenizerCodec:
    """``encode``/``decode`` as RL admission's replay workers run them: the
    fast tokenizer's backend, no special token added, none skipped, no
    clean-up. Every side of a suite replay (intake, grader, export) codes
    through this, so a decoded span reads the same everywhere. Thread-safe
    (one lock per codec): an HF tokenizer is not."""

    def __init__(self, tokenizer) -> None:
        import threading

        self._backend = getattr(tokenizer, "backend_tokenizer", None) or tokenizer
        self._lock = threading.Lock()

    def encode(self, text: str) -> list[int]:
        with self._lock:
            encoded = self._backend.encode(text, add_special_tokens=False)
        return [int(t) for t in getattr(encoded, "ids", encoded)]

    def decode(self, ids) -> str:
        with self._lock:
            return str(self._backend.decode([int(t) for t in ids], skip_special_tokens=False))


def single_id(encode: Callable[[str], list[int]], text: str) -> int:
    ids = list(encode(text))
    if len(ids) != 1:
        raise ValueError(f"{text!r} is not one token of this tokenizer: {ids}")
    return int(ids[0])


def turn_ids(encode: Callable[[str], list[int]]) -> tuple[int, frozenset[int]]:
    """``(terminator_id, stop_ids)`` of a ChatML suite turn. A turn before the
    last must end on ``<|im_end|>`` (or at the cap): the miner stops it there
    and the observation follows. The last turn may also end on
    ``<|endoftext|>``, which a model writes after a final answer; mid-episode
    the corpus refuses it (``bad_termination``) though the renderer would read
    the turn, because the exported transcript would not be the chat template's."""
    terminator = single_id(encode, IM_END)
    try:
        stops = frozenset({terminator, single_id(encode, END_OF_TEXT)})
    except ValueError:
        stops = frozenset({terminator})
    return terminator, stops


_MARKUP = ("<|im_start|>", "<tool_response>", "</tool_response>")


def turn_markup_ids(encode: Callable[[str], list[int]]) -> frozenset[int]:
    """The turn markup the renderer writes between spans, never inside one:
    ``<|im_start|>`` and the tool-response tags, where they are single tokens."""
    ids = set()
    for text in _MARKUP:
        try:
            ids.add(single_id(encode, text))
        except ValueError:
            pass
    return frozenset(ids)


def suite_span_refusal(tokens: Sequence[int], spans: Sequence[tuple[int, int]], *,
                       encode: Callable[[str], list[int]],
                       decode: Callable[[list[int]], str] | None = None) -> tuple[str, dict] | None:
    """``(reason, detail)`` when a span is not one assistant turn, else None.

    The SWE rule (``trajectory_parse.check_span_tokens``): a stop id only as
    the span's last token, no turn markup anywhere. Without it a span can hold
    ``<|im_end|>``, a made-up ``<|im_start|>user`` tool response and a second
    turn, and the replay, which reads the span as one action, accepts it.
    With ``decode``, the same markers spelled out as plain text are refused
    too. The miner's precheck calls this same function."""
    from reliquary.corpus.trajectory_parse import TrajectoryRefused, check_span_tokens

    _, stop_ids = turn_ids(encode)
    try:
        check_span_tokens(tokens, spans, stop_ids=stop_ids, markup_ids=turn_markup_ids(encode))
    except TrajectoryRefused as refused:
        return refused.reason, refused.detail
    if decode is not None:
        markers = (*_MARKUP, IM_END, END_OF_TEXT)
        for k, (start, end) in enumerate(spans):
            body = list(tokens[start:end])
            if body and body[-1] in stop_ids:
                body = body[:-1]
            text = decode(body)
            if any(marker in text for marker in markers):
                return "bad_turns", {"turn": k, "why": "turn markup spelled out inside the span"}
    return None


class SuiteSource:
    """A suite job's rows: the environment's tasks, through the job's
    ``RowMap``, and the renderer the job declares. ``initial_ids`` is what the
    miner prompts with and what the validator compares the trajectory's prompt
    to.

    Row ``i`` plays task ``rows.index(i)`` of the environment, which is built
    once, at ``rows.split``, by whoever builds the source (``suite_source``),
    never per call. Without ``rows`` the map is the identity over the whole
    environment on ``train``: a training job's ``prompt_index`` is the task's
    index, as it always was, and ``len`` is the environment's (the intake
    refuses a job owning rows past it)."""

    def __init__(self, environment, episode: EpisodeSpec, renderer_id: str,
                 rows: RowMap | None = None) -> None:
        # Refused, not capped: a job declaring more turns than the environment
        # runs would sell episodes that can never reach its limit.
        if episode.max_turns > int(environment.max_turns):
            raise ValueError(
                f"episode.max_turns {episode.max_turns} is above environment "
                f"{environment.name!r}'s own max_turns {environment.max_turns}; "
                f"declare at most {environment.max_turns}")
        if rows is None:
            rows = RowMap.identity("train", 0, len(environment))
        self.environment = environment
        self.episode = episode
        self.renderer_id = renderer_id
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def task(self, prompt_index: int):
        return self.environment.get_task(self.rows.index(int(prompt_index)))

    task_for = task

    def instance_id(self, prompt_index: int) -> str:
        return str(self.task(prompt_index).id)

    def renderer(self, encode: Callable[[str], list[int]]):
        from reliquary.environment.agentic.renderers import renderer_for

        return renderer_for(self.renderer_id, encode)

    def initial_text(self, prompt_index: int) -> str:
        return self.renderer(lambda text: []).initial_text(self.task(prompt_index))

    prompt = initial_text


def suite_source(episode: EpisodeSpec, *, rows: RowMap | None = None,
                 specs: Mapping[str, Any] | None = None) -> SuiteSource:
    """The source of a suite episode: its environment, built once at
    ``rows.split`` (``train`` without ``rows``: the identity over the whole
    environment, a training job's source). With ``rows``, every reference must
    rebuild into the instance it names, or a ``ValueError`` names the first
    row that does not: a set frozen against another build of the environment
    would otherwise play other tasks than the ones it was frozen on."""
    spec = suite_spec(episode, specs=specs)
    if rows is None:
        return SuiteSource(spec.create(), episode, spec.renderer_id)
    environment = spec.create(split=rows.split)
    rows.check_instances(lambda split, index: str(environment.get_task(index).id))
    return SuiteSource(environment, episode, spec.renderer_id, rows)


def job_suite_source(job, *, specs: Mapping[str, Any] | None = None) -> SuiteSource:
    """The one source of a suite job, for its intake, grader, export and miner.

    An eval job (its prompt source names a frozen set of row references) maps
    row ``i`` to the set's ``i``-th reference (``reliquary.eval.rows.job_row_map``),
    at the set's split. Any other job keeps the identity over the whole train
    environment, not just its ``[prompt_start, prompt_end)``: its
    ``prompt_index`` is a raw environment index, and its intake checks the job
    against the environment's own length."""
    from reliquary.eval.prompt_source import is_eval_source
    from reliquary.eval.rows import job_row_map

    rows = job_row_map(job) if is_eval_source(job.prompt_source) else None
    return suite_source(job.episode, rows=rows, specs=specs)


def replay_suite(environment, *, renderer_id: str, task_index: int, seed: int,
                 prompt_ids: Sequence[int], tokens: Sequence[int],
                 spans: Sequence[tuple[int, int]], encode, decode, max_turns: int,
                 max_observation_bytes: int = 64 * 1024):
    """The validator's replay of one trajectory (spans in trajectory
    coordinates); the trace's own coordinates start at the prompt.

    ``replay_tokenized_episode`` with the job's turn limit: that function has
    no ``max_turns`` and is in the builtin environments' pinned implementation
    digest, so it is not edited; this is its body with the limit passed on."""
    from reliquary.environment.agentic.renderers import renderer_for
    from reliquary.environment.agentic.replay import _TokenizedReplayPolicy
    from reliquary.environment.agentic.runner import EpisodeRunner
    from reliquary.environment.agentic.types import GeneratedAction

    full = list(prompt_ids) + list(tokens)
    offset = len(prompt_ids)
    generated = []
    for start, end in spans:
        piece = full[offset + int(start):offset + int(end)]
        generated.append(GeneratedAction(text=decode(piece), tokens=tuple(piece)))
    # The job's limit (`SuiteSource` refuses one above the environment's);
    # the min keeps a replay outside a job, as RL's, on the environment's.
    limit = min(int(max_turns), int(environment.max_turns))
    return EpisodeRunner(
        renderer=renderer_for(renderer_id, encode), max_turns=limit,
        max_observation_bytes=max_observation_bytes,
    ).run(environment, environment.get_task(int(task_index)), seed=int(seed),
          policy=_TokenizedReplayPolicy(generated))


def replay_stop(trace) -> str:
    return "max_turns" if trace.termination_reason == TURN_LIMIT else "agent_completed"


# What no observation the transcript holds may contain: the renderer writes an
# observation's text verbatim, so a stop or turn marker in it is encoded as the
# real token, and the trajectory then holds a turn nobody generated. An
# environment echoes what the model sent (an error quoting an argument), and a
# JSON argument can spell a marker with escapes no span check sees.
OBSERVATION_MARKERS = ("<|im_end|>", "<|im_start|>", "<|endoftext|>")


def observation_markup(trace, *, turns: int) -> dict | None:
    """``{"turn", "marker"}`` of the first observation rendered into the
    trajectory (those after turns ``0 .. turns - 2``) holding a marker of
    ``OBSERVATION_MARKERS``, else None."""
    turn = -1
    for event in trace.events:
        if event.role == "assistant":
            turn += 1
            continue
        if turn < 0 or turn >= turns - 1:
            continue                      # the task and reset events, or after the last turn
        for marker in OBSERVATION_MARKERS:
            if marker in event.content:
                return {"turn": turn, "marker": marker}
    return None


def compare_replay(trace, *, prompt_ids: Sequence[int], tokens: Sequence[int],
                   spans: Sequence[tuple[int, int]], stop: str) -> tuple[str, dict] | None:
    """``(reason, detail)`` when the trajectory is not the replay's, else None.

    As RL compares (admission): the tokens, the spans (and with them how many
    actions the environment took), and the termination. Only up to the end of
    the last span: what the replay renders after it is not the trajectory's."""
    offset = len(prompt_ids)
    full = list(prompt_ids) + list(tokens)
    replayed = list(trace.tokens)
    if replayed[:offset] != list(prompt_ids):
        return "bad_observation", {"why": "the replay's prompt is not the validator's"}
    expected = [(offset + int(s), offset + int(e)) for s, e in spans]
    got = [tuple(span) for span in trace.assistant_spans]
    if len(got) != len(expected):
        return "bad_turns", {"why": f"the environment took {len(got)} turns, the trajectory "
                                    f"has {len(expected)}"}
    if replayed[:len(full)] != full:
        at = next((i for i, (a, b) in enumerate(zip(replayed, full)) if a != b),
                  min(len(replayed), len(full)))
        turn = sum(1 for _, end in expected if end <= at)
        return "bad_observation", {"why": "a segment is not the environment's rendering",
                                   "turn": turn, "token": at - offset}
    if got != expected:
        return "bad_turns", {"why": "the spans are not the replay's"}
    marked = observation_markup(trace, turns=len(expected))
    if marked is not None:
        return "bad_observation", {"why": "an observation carries turn markup", **marked}
    if trace.reward is None:
        return "bad_observation", {"why": "the replay produced no reward"}
    if stop not in SUITE_STOPS:
        return "bad_stop", {"stop": stop, "allowed": list(SUITE_STOPS)}
    if replay_stop(trace) != stop:
        return "bad_stop", {"stop": stop, "replay": replay_stop(trace),
                            "termination": trace.termination_reason}
    return None


@dataclass(frozen=True)
class SuiteReplayOutcome:
    """What a replay concluded, picklable (it crosses the process pool)."""

    refusal: tuple[str, dict] | None
    task_id: str | None = None
    reward: float | None = None
    success: bool | None = None
    termination_reason: str | None = None
    trace_digest: str | None = None
    # The replay ran past its wall deadline on the worker: a refusal at
    # intake (the trajectory costs that much), a retry for the grader.
    deadline: bool = False


def replay_outcome(environment, *, renderer_id: str, task_index: int, seed: int,
                   prompt_ids, tokens, spans, stop: str, encode, decode,
                   max_turns: int) -> SuiteReplayOutcome:
    """Replay and compare; an environment or budget error is a refusal of the
    trajectory (the honest miner ran the same code and got through). A
    deadline (``TimeoutError``) is the caller's to handle."""
    try:
        trace = replay_suite(environment, renderer_id=renderer_id, task_index=task_index,
                             seed=seed, prompt_ids=prompt_ids, tokens=tokens, spans=spans,
                             encode=encode, decode=decode, max_turns=max_turns)
    except TimeoutError:
        raise
    except Exception as exc:  # noqa: BLE001 - whatever the environment raises refuses it
        return SuiteReplayOutcome(("bad_observation", {
            "why": f"replay failed: {type(exc).__name__}: {str(exc)[:200]}"}))
    refusal = compare_replay(trace, prompt_ids=prompt_ids, tokens=tokens, spans=spans, stop=stop)
    if refusal is not None:
        return SuiteReplayOutcome(refusal)
    return SuiteReplayOutcome(None, task_id=str(trace.task_id), reward=float(trace.reward.reward),
                              success=bool(trace.reward.success),
                              termination_reason=str(trace.termination_reason),
                              trace_digest=str(trace.trace_digest))


# -- export ---------------------------------------------------------------------


TRANSFORM_SINGLE_IM_END = "chatml-single-im-end-v1"
# Recorded only on the rows it applies to.
TRANSFORM_FINAL_ENDOFTEXT = "final-endoftext-as-im-end-v1"


def export_tokens(full: Sequence[int], spans: Sequence[tuple[int, int]], *, im_end_id: int,
                  newline_ids: Sequence[int], end_of_text_id: int | None = None,
                  ) -> tuple[list[int], list[int], list[str]]:
    """``(tokens, assistant_mask, transforms)`` of a trajectory, rewritten
    into the model's chat template (spans in ``full`` coordinates).

    ``chatml-single-im-end-v1``, always. A ChatML turn keeps the
    ``<|im_end|>`` the model stopped on (it is the token that ends the turn
    and must be learned), and the renderer's observation text opens with its
    own ``<|im_end|>\\n`` (the RL renderer is consensus and stays as it is).
    Where both are present the transcript holds ``<|im_end|><|im_end|>\\n``;
    the chat template writes one. So the observation's copy, never in the
    mask, is dropped; a turn cut at the cap keeps it (it is then the turn's
    only terminator). A last turn ending on ``<|im_end|>`` gets the ``\\n``
    the template writes after it, unmasked.

    ``final-endoftext-as-im-end-v1``, when it applies. A last turn that
    stopped on ``<|endoftext|>`` (accepted: the renderer reads the turn the
    same) has it replaced by ``<|im_end|>``, still in the mask, then the
    ``\\n``: the template ends an assistant turn on ``<|im_end|>``, and that is
    what a model trained on these rows should stop on. No other span token is
    ever touched."""
    full = [int(t) for t in full]
    transforms = [TRANSFORM_SINGLE_IM_END]
    if (spans and end_of_text_id is not None and end_of_text_id != im_end_id
            and full[spans[-1][1] - 1] == end_of_text_id):
        full[spans[-1][1] - 1] = im_end_id
        transforms.append(TRANSFORM_FINAL_ENDOFTEXT)
    dropped: set[int] = set()
    for start, end in spans[:-1]:
        if full[end - 1] == im_end_id and end < len(full) and full[end] == im_end_id:
            dropped.add(end)
    in_span = [0] * len(full)
    for start, end in spans:
        in_span[start:end] = [1] * (end - start)
    tokens = [t for i, t in enumerate(full) if i not in dropped]
    mask = [m for i, m in enumerate(in_span) if i not in dropped]
    if spans and full[spans[-1][1] - 1] == im_end_id:
        tokens += [int(t) for t in newline_ids]
        mask += [0] * len(newline_ids)
    return tokens, mask, transforms


def tools_of(task) -> list[dict]:
    """The task's tools as the chat template's ``tools=`` argument takes them
    (and as the ChatML renderer writes them into the system turn)."""
    return [{"type": "function", "function": tool.to_wire()} for tool in task.tools]


def _assistant_message(text: str, action) -> dict:
    from reliquary.environment.agentic.chatml import (
        CALL_OPEN, THINK_CLOSE, _without_turn_end,
    )

    text = _without_turn_end(text)
    if THINK_CLOSE in text:
        reasoning, after = text.rsplit(THINK_CLOSE, 1)
    else:
        reasoning, after = "", text
    message: dict[str, Any] = {"role": "assistant", "reasoning_content": reasoning.strip()}
    if action.kind == "tool" and not action.tool.startswith("__") and CALL_OPEN in after:
        message["content"] = after.split(CALL_OPEN, 1)[0].strip()
        message["tool_calls"] = [{"type": "function", "function": {
            "name": action.tool, "arguments": dict(action.arguments)}}]
    elif action.kind == "final":
        message["content"] = (action.content or "").strip()
    else:
        # A turn the renderer read as no action: what the model wrote, as text.
        message["content"] = after.strip()
    return message


def suite_messages(trace, *, task, decode, spans: Sequence[tuple[int, int]]) -> list[dict]:
    """The conversation of a replayed trajectory, for reading: system (the
    episode guidance; the tools go in ``tools``), the task, then each turn and
    the observation that answered it, up to the last turn. Tool results are
    ``tool`` messages; anything else the environment said is ``user``, as the
    renderer writes it. ``tokens`` stays the training form."""
    from reliquary.environment.agentic.chatml import EPISODE_GUIDANCE

    messages: list[dict] = [{"role": "system", "content": EPISODE_GUIDANCE},
                            {"role": "user", "content": task.prompt.strip()}]
    observations: list[list] = []
    seen_assistant = False
    for event in trace.events:
        if event.role == "assistant":
            seen_assistant = True
            observations.append([])
        elif seen_assistant:
            observations[-1].append(event)
    tokens = list(trace.tokens)
    for turn, ((start, end), action) in enumerate(zip(spans, trace.actions)):
        messages.append(_assistant_message(decode(tokens[start:end]), action))
        if turn == len(spans) - 1:
            break
        for event in observations[turn]:
            if any(marker in event.content for marker in OBSERVATION_MARKERS):
                # compare_replay refuses it first; never a row that holds one.
                raise ValueError(f"turn {turn}'s observation carries turn markup")
            if event.role == "tool":
                messages.append({"role": "tool", "name": event.name,
                                 "content": event.content.strip()})
            else:
                messages.append({"role": "user", "content": event.content.strip()})
    return messages


__all__ = ["SUITE_RENDERERS", "SuiteReplayOutcome", "SuiteSource", "compare_replay",
           "episode_seed", "export_tokens", "job_suite_source", "replay_outcome", "replay_stop",
           "replay_suite",
           "suite_messages", "suite_source", "suite_span_refusal", "suite_spec", "suite_spec_refusal",
           "tools_of", "turn_markup_ids",
           "turn_ids"]
