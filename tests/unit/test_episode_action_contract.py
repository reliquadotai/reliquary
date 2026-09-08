"""The action contract is consensus, and it now allows thinking first.

`AssistantAction.from_json` used to require the whole completion to be one
bare JSON object after `strip()`, which forbade any reasoning before the
call. Measured on EnvScaler: 78.2% of first turns carried a valid,
correctly named tool call while 27.1% satisfied that rule, and no Qwen3-14B
rollout in 768 was bare JSON on every turn. Reasoning is also the largest
single lever on those tasks — holding model and prompt fixed, removing it
took outright task completions from 5.1% to zero.

Widening it is only safe if it is a strict superset, so the first test here
is that everything which parsed before parses to the same action.
"""

from __future__ import annotations

import pytest

from reliquary.environment.agentic.types import (
    MAX_ACTION_CANDIDATES,
    AssistantAction,
)


_BARE = [
    '{"tool":"lookup","arguments":{}}',
    '{"tool":"lookup","arguments":{"id":"U1","deep":{"k":[1,2]}}}',
    '{"final":"done"}',
    '  {"final":"done"}  ',
]


@pytest.mark.parametrize("text", _BARE)
def test_bare_objects_are_unchanged(text):
    """The superset property: what parsed before parses the same way."""
    action = AssistantAction.from_json(text)
    assert AssistantAction.from_json(action.to_json()) == action


def test_reasoning_before_the_action_is_accepted():
    action = AssistantAction.from_json(
        "The account has to be read before it can be changed.\n"
        '{"tool":"get_user","arguments":{"id":"U1"}}'
    )
    assert action.kind == "tool"
    assert action.tool == "get_user"
    assert action.arguments == {"id": "U1"}


@pytest.mark.parametrize("wrapper", [
    "<think>{body}</think>\n{action}",
    "```json\n{action}\n```",
    "**{action}**",
    "{action}\n",
])
def test_common_wrappings_are_accepted(wrapper):
    text = wrapper.format(body="plan", action='{"tool":"a","arguments":{}}')
    assert AssistantAction.from_json(text).tool == "a"


def test_the_last_action_wins():
    """A model may weigh a call and then commit to another."""
    action = AssistantAction.from_json(
        '{"tool":"a","arguments":{}} on reflection: '
        '{"tool":"b","arguments":{}}'
    )
    assert action.tool == "b"


def test_an_echoed_prompt_schema_does_not_win():
    """The renderer's own example is literal, parseable JSON.

    A turn that repeats the instructions and then acts must be read as the
    action, not as the instruction — otherwise echoing the system prompt
    ends the episode.
    """
    action = AssistantAction.from_json(
        'Reply with {"final":"answer"} when finished.\n'
        '{"tool":"search","arguments":{"q":"x"}}'
    )
    assert action.kind == "tool"
    assert action.tool == "search"


def test_objects_that_are_not_actions_are_skipped():
    action = AssistantAction.from_json(
        'Observed {"status":"ok","rows":3} so now '
        '{"tool":"commit","arguments":{}}'
    )
    assert action.tool == "commit"


def test_braces_inside_strings_do_not_split_the_span():
    action = AssistantAction.from_json('{"tool":"a","arguments":{"k":"}{"}}')
    assert action.arguments == {"k": "}{"}


@pytest.mark.parametrize("text", [
    '{"foo":1}',
    "no json here at all",
    '{"tool":"a","arguments":{},"extra":1}',
    '{"tool":"a","arguments":{},"tool":"b"}',
    "{'tool':'a','arguments':{}}",
    '{"tool":123,"arguments":{}}',
    '{"tool":"a","arguments":[]}',
    '{"final":42}',
])
def test_still_rejected(text):
    with pytest.raises((ValueError, TypeError)):
        AssistantAction.from_json(text)


def test_unclosed_object_is_not_an_action():
    with pytest.raises(ValueError):
        AssistantAction.from_json('thinking… {"tool":"a","arguments":{}')


def test_candidate_scan_is_bounded():
    """A pathological turn must not cost more than it is worth."""
    noise = " ".join('{"tool":"n","arguments":{}}'
                     for _ in range(MAX_ACTION_CANDIDATES + 20))
    action = AssistantAction.from_json(noise + ' {"tool":"real","arguments":{}}')
    # The scan stops at the cap, so the trailing action is past it and the
    # last *examined* candidate wins. What matters is that it terminates and
    # still yields a valid action.
    assert action.kind == "tool"
    assert action.tool == "n"
