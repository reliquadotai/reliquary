"""Actions and observations come from the proven tokens, through the renderer."""

import json

import pytest

from reliquary.corpus.replay_compare import Action
from reliquary.corpus.trajectory_parse import TrajectoryRefused, parse_trajectory

TERM, EOT, TR, TRE, CALL, NL, GEN = 1, 2, 3, 4, 5, 10, 12
TEXT, CHAR = 200, 1000          # assistant text ids; observation chars are CHAR + ord(c)


class FakeRenderer:
    """A renderer over integer ids with the shape of the Qwen one: a turn closes
    on TERM, observations sit between TR and TRE, CALL is one tool call."""

    terminator_id = TERM
    stop_ids = frozenset({TERM, EOT})
    turn_markup_ids = frozenset({TR, TRE})
    canonical = True
    unclosed = False

    def span_is_canonical(self, prompt_ids, completion_ids):
        return self.canonical

    def reasoning_unclosed(self, prompt_ids, completion_ids):
        return self.unclosed

    def initial_ids(self, prompt):
        return [TEXT + (ord(c) % 50) for c in prompt] + [GEN]

    def tool_calls(self, completion_ids):
        calls = sum(1 for t in completion_ids if t == CALL)
        return [("bash", json.dumps({"command": f"c{i}"})) for i in range(calls)]

    def observations(self, segment_ids):
        out, inside = [], None
        for t in segment_ids:
            if t == TR:
                inside = []
            elif t == TRE and inside is not None:
                out.append("".join(chr(x - CHAR) for x in inside))
                inside = None
            elif inside is not None:
                inside.append(t)
        return out

    def next_prompt(self, prompt_ids, completion_ids, observations):
        ids = list(prompt_ids) + list(completion_ids)
        if not completion_ids or completion_ids[-1] not in self.stop_ids:
            ids.append(TERM)                        # the bridge closes a capped turn
        ids.append(NL)
        for text in observations:
            ids += [TR] + [CHAR + ord(c) for c in text] + [TRE]
        return ids + [GEN]

    def assistant_message(self, completion_ids):
        return {"role": "assistant", "content": "", "tool_calls": [
            {"id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": a}}
            for i, (n, a) in enumerate(self.tool_calls(completion_ids))]}


R = FakeRenderer()
PROMPT = R.initial_ids("fix it")


def build(turns):
    """turns: [(completion, observations or None)] -> (tokens, spans)."""
    full, spans = list(PROMPT), []
    for completion, observations in turns:
        start = len(full) - len(PROMPT)
        prefix = list(full)
        full = prefix + list(completion)
        spans.append((start, start + len(completion)))
        if observations is not None:
            full = R.next_prompt(prefix, completion, observations)
    return full[len(PROMPT):], spans


def parse(tokens, spans, stop="agent_completed"):
    return parse_trajectory(R, prompt_ids=PROMPT, tokens=tokens, spans=spans, stop=stop)


def test_an_honest_trajectory_yields_its_actions_in_order():
    tokens, spans = build([([TEXT, CALL, CALL, TERM], ["out a", "out b"]),
                           ([TEXT, CALL, TERM], ["x"]),
                           ([TEXT, TEXT, TERM], None)])
    parsed = parse(tokens, spans)
    assert parsed.actions == (Action("bash", '{"command": "c0"}', "out a"),
                              Action("bash", '{"command": "c1"}', "out b"),
                              Action("bash", '{"command": "c0"}', "x"))
    assert [t.observations for t in parsed.turns] == [("out a", "out b"), ("x",), ()]


def test_a_tool_call_without_an_answer_is_refused():
    tokens, spans = build([([TEXT, CALL, CALL, TERM], ["only one"]), ([TEXT, TERM], None)])
    with pytest.raises(TrajectoryRefused) as caught:
        parse(tokens, spans)
    assert caught.value.reason == "unanswered_tool_call"


def test_an_observation_without_a_call_is_refused():
    tokens, spans = build([([TEXT, CALL, TERM], ["a", "b"]), ([TEXT, TERM], None)])
    with pytest.raises(TrajectoryRefused, match="bad_observation"):
        parse(tokens, spans)


def test_a_turn_without_a_call_cannot_be_followed():
    tokens, spans = build([([TEXT, TERM], []), ([TEXT, TERM], None)])
    with pytest.raises(TrajectoryRefused, match="bad_observation"):
        parse(tokens, spans)


def test_a_segment_that_is_not_the_renderers_is_refused():
    tokens, spans = build([([TEXT, CALL, TERM], ["a"]), ([TEXT, TERM], None)])
    end = spans[0][1]
    forged = tokens[:end] + [NL] + tokens[end:]               # one extra scaffold token
    spans = [spans[0], (spans[1][0] + 1, spans[1][1] + 1)]
    with pytest.raises(TrajectoryRefused, match="bad_observation"):
        parse(forged, spans)


def test_trailing_tokens_after_the_final_turn_are_refused():
    tokens, spans = build([([TEXT, TERM], None)])
    with pytest.raises(TrajectoryRefused, match="bad_turns"):
        parse(tokens + [NL], spans)


def test_agent_completed_with_a_final_call_is_refused():
    tokens, spans = build([([TEXT, CALL, TERM], None)])
    with pytest.raises(TrajectoryRefused, match="bad_stop"):
        parse(tokens, spans)


def test_context_length_final_calls_are_replayed_without_observations():
    tokens, spans = build([([TEXT, CALL, TERM], ["a"]), ([TEXT, CALL, TERM], None)])
    parsed = parse(tokens, spans, stop="context_length")
    assert parsed.actions[-1] == Action("bash", '{"command": "c0"}', None)


def test_max_turns_final_calls_are_not_replayed():
    tokens, spans = build([([TEXT, CALL, TERM], ["a"]), ([TEXT, CALL, TERM], None)])
    parsed = parse_trajectory(R, prompt_ids=PROMPT, tokens=tokens, spans=spans,
                              stop="max_turns", max_turns=2)
    assert len(parsed.actions) == 1


def test_a_capped_turn_is_followed_by_a_synthesized_terminator():
    tokens, spans = build([([TEXT, CALL], ["a"]), ([TEXT, TERM], None)])  # no TERM: capped
    assert tokens[spans[0][1]] == TERM                       # the bridge's own close
    assert parse(tokens, spans).actions == (Action("bash", '{"command": "c0"}', "a"),)


def test_a_stop_token_inside_a_non_final_span_is_refused():
    tokens, spans = build([([TEXT, CALL, TERM, TEXT, CALL, TERM], ["a"]), ([TEXT, TERM], None)])
    with pytest.raises(TrajectoryRefused, match="bad_turns"):
        parse(tokens, spans)


def test_a_stop_token_inside_the_final_span_hides_nothing():
    tokens, spans = build([([TEXT, TERM, CALL, TERM], None)])
    with pytest.raises(TrajectoryRefused, match="bad_turns"):
        parse(tokens, spans)


@pytest.mark.parametrize("marker", [TR, TRE])
def test_turn_markup_inside_a_span_is_refused(marker):
    tokens, spans = build([([TEXT, CALL, marker, TERM], ["a"]), ([TEXT, TERM], None)])
    with pytest.raises(TrajectoryRefused, match="bad_turns"):
        parse(tokens, spans)


def test_a_truncated_final_turn_under_agent_completed_is_refused():
    tokens, spans = build([([TEXT, TEXT], None)])
    with pytest.raises(TrajectoryRefused, match="bad_turns"):
        parse(tokens, spans)


def test_an_unknown_stop_is_refused():
    tokens, spans = build([([TEXT, CALL, TERM], None)])
    with pytest.raises(TrajectoryRefused, match="bad_stop"):
        parse(tokens, spans, stop="bogus")


def test_max_turns_needs_the_jobs_turn_count():
    tokens, spans = build([([TEXT, CALL, TERM], ["a"]), ([TEXT, CALL, TERM], None)])
    with pytest.raises(TrajectoryRefused, match="bad_stop"):   # label without the job's limit
        parse(tokens, spans, stop="max_turns")
    with pytest.raises(TrajectoryRefused, match="bad_stop"):   # fewer turns than the limit
        parse_trajectory(R, prompt_ids=PROMPT, tokens=tokens, spans=spans, stop="max_turns", max_turns=5)
    ok = parse_trajectory(R, prompt_ids=PROMPT, tokens=tokens, spans=spans, stop="max_turns", max_turns=2)
    assert len(ok.actions) == 1


def test_spans_out_of_order_or_bounds_are_refused():
    tokens, spans = build([([TEXT, CALL, TERM], ["a"]), ([TEXT, TERM], None)])
    for bad in ([(spans[0][0], spans[0][0]), spans[1]], [spans[0], (spans[1][0], len(tokens) + 3)],
                [(spans[0][0], spans[1][1]), spans[1]]):
        with pytest.raises(TrajectoryRefused, match="bad_turns"):
            parse(tokens, bad)


def test_a_span_that_does_not_re_render_to_itself_is_refused(monkeypatch):
    tokens, spans = build([([TEXT, CALL, TERM], ["a"]), ([TEXT, TERM], None)])
    monkeypatch.setattr(FakeRenderer, "canonical", False)
    with pytest.raises(TrajectoryRefused, match="bad_turns"):
        parse(tokens, spans)


def test_an_unclosed_final_reasoning_is_refused_under_agent_completed(monkeypatch):
    tokens, spans = build([([TEXT, TEXT, TERM], None)])
    monkeypatch.setattr(FakeRenderer, "unclosed", True)
    with pytest.raises(TrajectoryRefused, match="bad_stop"):
        parse(tokens, spans)
