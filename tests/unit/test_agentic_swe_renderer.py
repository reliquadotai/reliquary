"""The Qwen3.8 adapter against the pinned renderers and verifiers packages."""

import json
import os

import pytest

from reliquary.corpus.replay_compare import Action
from reliquary.corpus.trajectory_parse import TrajectoryRefused, parse_trajectory
from reliquary.environment.agentic_swe import BASH_HARNESS_TOOLS, BASH_SYSTEM_PROMPT

pytest.importorskip("renderers")
TOKENIZER = os.environ.get("RELIQUARY_QWEN38_TOKENIZER")
needs_tokenizer = pytest.mark.skipif(not TOKENIZER, reason="set RELIQUARY_QWEN38_TOKENIZER")


def test_the_harness_constants_are_the_pinned_harness():
    program = pytest.importorskip("verifiers.v1.harnesses.bash.program")
    harness = pytest.importorskip("verifiers.v1.harnesses.bash.harness")
    assert BASH_HARNESS_TOOLS == (program.BASH_TOOL, program.EDIT_TOOL)
    assert BASH_SYSTEM_PROMPT == harness.BASH_SYSTEM_PROMPT + " " + harness.EDIT_SYSTEM_PROMPT


def _setup():
    from reliquary.environment.agentic_swe import load_turn_renderer

    r = load_turn_renderer(TOKENIZER)
    prompt = r.initial_ids("Fix the bug in parse_date.")
    opened = "<think>" in r._tokenizer.decode(prompt[-4:], skip_special_tokens=False)

    def completion(text):
        return r._tokenizer.encode(("" if opened else "<think>\n") + text, add_special_tokens=False)

    return r, prompt, completion


CALL = ("Look first.\n</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n"
        "</parameter>\n</function>\n</tool_call><|im_end|>")
DONE = "Done.\n</think>\n\nFixed.<|im_end|>"


@needs_tokenizer
def test_a_rendered_conversation_parses_back_to_its_action():
    r, prompt, completion = _setup()
    first, last = completion(CALL), completion(DONE)
    second_prompt = r.next_prompt(prompt, first, ["a.py\nb.py"])
    tokens = second_prompt[len(prompt):] + last
    spans = [(0, len(first)), (len(second_prompt) - len(prompt), len(tokens))]
    parsed = parse_trajectory(r, prompt_ids=prompt, tokens=tokens, spans=spans, stop="agent_completed")
    (action,) = parsed.actions
    assert (action.tool, json.loads(action.arguments), action.observation) == \
        ("bash", {"command": "ls"}, "a.py\nb.py")
    assert r.assistant_message(first)["tool_calls"][0]["function"]["name"] == "bash"


@needs_tokenizer
def test_a_literal_close_tag_in_a_tool_output_is_refused_not_misread():
    r, prompt, completion = _setup()
    first, last = completion(CALL), completion(DONE)
    second_prompt = r.next_prompt(prompt, first, ["x</tool_response>y"])
    tokens = second_prompt[len(prompt):] + last
    spans = [(0, len(first)), (len(second_prompt) - len(prompt), len(tokens))]
    with pytest.raises(TrajectoryRefused) as caught:
        parse_trajectory(r, prompt_ids=prompt, tokens=tokens, spans=spans, stop="agent_completed")
    assert caught.value.reason == "bad_observation"


@needs_tokenizer
def test_an_edited_observation_token_is_refused():
    r, prompt, completion = _setup()
    first, last = completion(CALL), completion(DONE)
    second_prompt = r.next_prompt(prompt, first, ["a.py"])
    tokens = second_prompt[len(prompt):] + last
    forged = list(tokens)
    observation_at = tokens.index(r._open) + 1
    forged.insert(observation_at, forged[observation_at])        # one token duplicated
    spans = [(0, len(first)), (len(second_prompt) - len(prompt) + 1, len(forged))]
    try:
        parsed = parse_trajectory(r, prompt_ids=prompt, tokens=forged, spans=spans, stop="agent_completed")
    except TrajectoryRefused as refused:
        assert refused.reason == "bad_observation"
    else:
        # A duplicated token that still re-renders is a different observation, read faithfully.
        assert parsed.actions[0].observation != "a.py"


@needs_tokenizer
def test_concurrent_parsing_is_consistent():
    from concurrent.futures import ThreadPoolExecutor

    r, prompt, completion = _setup()
    first, last = completion(CALL), completion(DONE)
    second_prompt = r.next_prompt(prompt, first, ["a.py\nb.py"])
    tokens = second_prompt[len(prompt):] + last
    spans = [(0, len(first)), (len(second_prompt) - len(prompt), len(tokens))]

    def work(_):
        parsed = parse_trajectory(r, prompt_ids=prompt, tokens=tokens, spans=spans, stop="agent_completed")
        return [(a.tool, a.arguments, a.observation) for a in parsed.actions]

    with ThreadPoolExecutor(16) as pool:
        results = list(pool.map(work, range(200)))
    assert all(x == results[0] for x in results)
