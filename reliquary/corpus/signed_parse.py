"""§5.C of the signed-episode spec: the model's calls are the signed records.

From the TOPLOC-proven tokens, each assistant span's calls are read with the one pinned
parser (`renderer.tool_calls`, verifiers' filter; it never raises anything but a
`bad_turns` refusal) and planned with `reliquary_sandbox.observation.plan_call` over
record 0's offered tools. Then:

* a `Send` consumes the next call record, which must have `turn` = the span's index,
  `k` = its `sent_positions` value, the same tool and canonical-equal arguments;
* a `Refuse` consumes no record; its observation is the refusal's fixed text;
* records left over, or a `Send` with no record left, are a mismatch;
* the final span of a `context_length` or `max_turns` stop has its sent calls recorded
  but no observation after it: they are counted, nothing is compared;
* an `episode_closed` stop (the bridge's stop after the gateway sent a final record on
  one of the last span's calls: budget, deadline or transcript cap) needs a final whose
  status is `expired` or `budget_exhausted`; any other final is `bad_stop`, so a miner
  cannot skip its last turn's calls and `/finish` a graded episode. The last span's
  records match a non-empty prefix of its sent calls (the closing call returned its
  record with the final; the calls after it were never sent) and no observation follows.
  What this parser checks is that no call record is left after that prefix; that the
  final record is the next record in the chain is §5.A's (`verify_transcript`);
* observations are never decoded or searched for: every other span's segment must be
  `renderer.next_prompt(prompt, completion, expected)`, token for token, where
  `expected` is `render_observation(record)` or the refusal text, in call order.
  Literal special-token text in an output is encoded by the renderer exactly as the
  miner's renderer encoded it (ruling 6), so it needs no rule of its own here.

Precondition: the transcript passed §5.A (`verify_transcript`: token, signatures,
session, contiguous chain, bindings). This parser compares record contents only: a
content-identical record taken from another session is caught there, not here.
The wire (`CorpusTrajectory.stop`) does not even carry `episode_closed` today; the rule
above holds here regardless, as defense in depth.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any, NamedTuple

from reliquary_sandbox.attest import (
    KIND_CALL, KIND_FINAL, KIND_OPEN, STATUS_BUDGET_EXHAUSTED, STATUS_EXPIRED, canonical_json,
    parse_body,
)
from reliquary_sandbox.attest.canonical import CanonicalError
from reliquary_sandbox.observation import Refuse, Send, plan_call, render_observation, sent_positions

from reliquary.corpus.checks import REASON_BAD_TURNS
from reliquary.corpus.replay_compare import Action
from reliquary.corpus.signed_reasons import REASON_SANDBOX_CALL_MISMATCH
from reliquary.corpus.trajectory_parse import (
    REASON_BAD_OBSERVATION, REASON_BAD_STOP, STOPS, ParsedTrajectory, ParsedTurn,
    TrajectoryRefused, TurnRenderer, check_span_structure,
)

STOP_EPISODE_CLOSED = "episode_closed"
"""The bridge's stop when the gateway closed the episode mid-turn
(`reliquary_sandbox_verifiers.task.SandboxEpisodeTask.episode_closed`). Signed episodes
only: the replay parser's `STOPS` do not hold it."""
SIGNED_STOPS = STOPS | {STOP_EPISODE_CLOSED}
CLOSED_ON_A_CALL = frozenset({STATUS_EXPIRED, STATUS_BUDGET_EXHAUSTED})
"""The final statuses a gateway sends on a call (deadline, budget, transcript cap)."""
_UNOBSERVED_LAST_TURN = ("context_length", "max_turns")


class SignedRecords(NamedTuple):
    tools: tuple[str, ...]
    calls: list
    final: Any


def signed_records(transcript: Mapping[str, Any]) -> SignedRecords:
    """Record 0's offered tools, the call bodies and the final body of a transcript.
    Parsed, not verified: callers verify first (`verify_transcript`) or read one an
    intake already accepted."""
    bodies = [parse_body(record["body"]) for record in transcript["records"]]
    if len(bodies) < 2 or bodies[0].kind != KIND_OPEN or bodies[-1].kind != KIND_FINAL:
        raise ValueError("a transcript runs from record 0 to a final record")
    return SignedRecords(tuple(bodies[0].tools),
                         [body for body in bodies[1:-1] if body.kind == KIND_CALL], bodies[-1])


def _same_arguments(signed: Any, sent: Mapping[str, Any]) -> bool:
    try:
        return canonical_json(signed) == canonical_json(sent)
    except CanonicalError:
        return False


def parse_signed_trajectory(renderer: TurnRenderer, *, prompt_ids: Sequence[int],
                            tokens: Sequence[int], spans: Sequence[tuple[int, int]], stop: str,
                            max_turns: int | None, calls: Sequence[Any],
                            offered: Collection[str], final: Any) -> ParsedTrajectory:
    """The trajectory's turns and actions, or `TrajectoryRefused`. `calls` are the
    verified transcript's call bodies in order; `offered` is record 0's `tools`;
    `final` is its final body (`signed_records(...).final`)."""
    if stop not in SIGNED_STOPS:
        raise TrajectoryRefused(REASON_BAD_STOP, {"stop": stop})
    if stop == STOP_EPISODE_CLOSED and final.status not in CLOSED_ON_A_CALL:
        raise TrajectoryRefused(REASON_BAD_STOP, {
            "stop": stop, "status": final.status,
            "why": "only a final sent on a call closes an episode mid-turn"})
    if not spans:
        raise TrajectoryRefused(REASON_BAD_TURNS, {"turns": 0})
    if stop == "max_turns" and len(spans) != max_turns:
        raise TrajectoryRefused(REASON_BAD_STOP, {"stop": stop, "turns": len(spans),
                                                  "max_turns": max_turns})
    check_span_structure(renderer, tokens, spans, stop)
    full = list(prompt_ids) + list(tokens)
    offset = len(prompt_ids)
    records = list(calls)
    offered = frozenset(offered)
    used = 0

    def take(turn: int, position: int, plan: Send):
        nonlocal used
        if used >= len(records):
            raise TrajectoryRefused(REASON_SANDBOX_CALL_MISMATCH, {
                "turn": turn, "k": position, "why": "a call the model sent has no signed record"})
        record = records[used]
        if ((record.turn, record.k, record.tool) != (turn, position, plan.tool)
                or not _same_arguments(record.arguments, plan.arguments)):
            raise TrajectoryRefused(REASON_SANDBOX_CALL_MISMATCH, {
                "turn": turn, "k": position, "record": record.i,
                "why": "the signed record is not the model's call"})
        used += 1
        return record

    turns: list[ParsedTurn] = []
    actions: list[Action] = []
    for k, (start, end) in enumerate(spans):
        completion = list(tokens[start:end])
        last = k == len(spans) - 1
        if last and stop == "agent_completed" and renderer.reasoning_unclosed(
                full[:offset + start], completion):
            raise TrajectoryRefused(REASON_BAD_STOP, {"turn": k, "why": "final reasoning is not closed"})
        if not renderer.span_is_canonical(full[:offset + start], completion):
            raise TrajectoryRefused(REASON_BAD_TURNS, {"turn": k, "why": "span does not re-render to itself"})
        pairs = list(renderer.tool_calls(completion))
        plans = [plan_call(name, arguments, offered) for name, arguments in pairs]
        positions = sent_positions(plans)
        sends = [(plan, position) for plan, position in zip(plans, positions)
                 if isinstance(plan, Send)]
        if last:
            if end != len(tokens):
                raise TrajectoryRefused(REASON_BAD_TURNS, {"trailing_tokens": len(tokens) - end})
            if stop == "agent_completed" and pairs:
                raise TrajectoryRefused(REASON_BAD_STOP, {"turn": k, "calls": len(pairs)})
            if stop in _UNOBSERVED_LAST_TURN:
                for plan, position in sends:
                    take(k, position, plan)
            elif stop == STOP_EPISODE_CLOSED:
                before = used
                for plan, position in sends:
                    if used == len(records):
                        break                       # the rest were never sent
                    take(k, position, plan)
                if used == before:
                    raise TrajectoryRefused(REASON_BAD_STOP, {
                        "turn": k, "why": "the episode closed on no recorded call of this turn"})
            actions += [Action(name, arguments, None) for name, arguments in pairs]
            turns.append(ParsedTurn(tuple(pairs), ()))
            continue
        if not pairs:
            raise TrajectoryRefused(REASON_BAD_OBSERVATION, {"turn": k, "why": "no tool call"})
        observations = [plan.observation if isinstance(plan, Refuse)
                        else render_observation(take(k, position, plan).to_dict())
                        for plan, position in zip(plans, positions)]
        expected = renderer.next_prompt(full[:offset + start], completion, observations)
        if expected is None or expected != full[:offset + spans[k + 1][0]]:
            raise TrajectoryRefused(REASON_BAD_OBSERVATION, {
                "turn": k, "why": "segment is not the rendering of the signed records"})
        turns.append(ParsedTurn(tuple(pairs), tuple(observations)))
        actions += [Action(name, arguments, observation)
                    for (name, arguments), observation in zip(pairs, observations)]
    if used != len(records):
        raise TrajectoryRefused(REASON_SANDBOX_CALL_MISMATCH, {
            "why": "signed records no call of the model explains", "left": len(records) - used})
    return ParsedTrajectory(tuple(turns), tuple(actions))


__all__ = ["CLOSED_ON_A_CALL", "SIGNED_STOPS", "STOP_EPISODE_CLOSED", "SignedRecords", "parse_signed_trajectory",
           "signed_records"]
