from __future__ import annotations

import pytest

from reliquary.environment.agentic.chatml import (
    CHATML_RENDERER_ID,
    EPISODE_GUIDANCE,
    ChatMLEpisodeRenderer as Renderer,
)
from reliquary.environment.agentic.types import AssistantAction, EpisodeEvent, EpisodeTask, ToolSpec
from reliquary.environment.registry import get_environment_spec


EPISODE_ENVS = (
    "reliquary_stateful_tools_v1",
    "reliquary_retrieval_tools_v1",
    "reliquary_workspace_tools_v1",
)

WEATHER = ToolSpec(
    name="get_weather",
    description="Current weather for a city",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)
TYPED = ToolSpec(
    name="book",
    description="Book seats",
    parameters={"type": "object", "properties": {
        "seats": {"type": "integer"}, "price": {"type": "number"}, "refundable": {"type": "boolean"},
        "note": {"type": ["string", "null"]}, "tags": {"type": "array"}, "meta": {"type": "object"},
        "code": {"type": "string"},
    }},
)
TASK = EpisodeTask(id="demo", prompt="  Weather in Lyon?  ", tools=(WEATHER, TYPED))


def _encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def test_initial_text_is_the_template_layout():
    text = Renderer.initial_text(TASK)
    assert text.startswith(
        "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
        '{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city", '
    )
    assert "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:" in text
    assert text.endswith(
        f"</IMPORTANT>\n\n{EPISODE_GUIDANCE}<|im_end|>\n"
        "<|im_start|>user\nWeather in Lyon?<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )
    assert Renderer.id == CHATML_RENDERER_ID


def test_tool_results_share_one_user_message_and_reopen_the_assistant():
    events = (
        EpisodeEvent(role="tool", content='{"ok": 1}', name="a"),
        EpisodeEvent(role="tool", content=' {"ok": 2} ', name="b"),
        EpisodeEvent(role="user", content="also Paris"),
    )
    assert Renderer.observation_text(events) == (
        "<|im_end|>\n"
        '<|im_start|>user\n<tool_response>\n{"ok": 1}\n</tool_response>\n<tool_response>\n{"ok": 2}\n</tool_response><|im_end|>\n'
        "<|im_start|>user\nalso Paris<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )


def test_action_text_follows_the_template_call_dialect():
    action = AssistantAction.tool_call("get_weather", city="Lyon")
    assert Renderer.action_text(action, reasoning=" check first ") == (
        "check first\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nLyon\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    assert Renderer.action_text(action, reasoning="x", preamble="Looking it up.").startswith(
        "x\n</think>\n\nLooking it up.\n\n<tool_call>\n"
    )
    assert Renderer.action_text(AssistantAction.final(" 14C, rain "), reasoning="") == "\n</think>\n\n14C, rain"


def test_arguments_round_trip_with_their_schema_types():
    action = AssistantAction.tool_call(
        "book", seats=3, price=12.5, refundable=True, note=None, tags=["a", "b"],
        meta={"k": [1, 2]}, code="007",
    )
    parsed = Renderer.parse_action(Renderer.action_text(action, reasoning="r"), TASK)
    assert parsed.to_wire() == action.to_wire()


def test_without_a_schema_every_argument_stays_text():
    text = Renderer.action_text(AssistantAction.tool_call("unknown", n=3, flag=False), reasoning="r")
    assert Renderer.parse_action(text, TASK).arguments == {"n": "3", "flag": "False"}


def test_reasoning_may_quote_call_markup_and_only_the_answer_counts():
    text = "maybe <tool_call>\n<function=get_weather>\n</function>\n</tool_call>? no.\n</think>\n\nIt rains."
    assert Renderer.parse_action(text, TASK).to_wire() == AssistantAction.final("It rains.").to_wire()


@pytest.mark.parametrize(
    "text, reason",
    [
        ("never closed <tool_call>", "never closed"),
        ("r\n</think>\n\n   ", "no answer"),
        ("r\n</think>\n\n" + "<tool_call>\n<function=get_weather>\n</function>\n</tool_call>" * 2, "more than one"),
        ("r\n</think>\n\n<tool_call>\n<function=get_weather>\n</function>\n", "unclosed"),
        ("r\n</think>\n\n<tool_call>\n<function=get_weather>\n</function>\n</tool_call> done", "after the function"),
        ("r\n</think>\n\n<tool_call>\nget_weather(city=Lyon)\n</tool_call>", "does not open a function"),
        ("r\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nA\n</parameter>\n"
         "<parameter=city>\nB\n</parameter>\n</function>\n</tool_call>", "given twice"),
        ("r\n</think>\n\n<function=get_weather>\n</function>", "outside a tool_call"),
    ],
)
def test_turns_that_commit_to_no_single_action_are_rejected(text: str, reason: str):
    with pytest.raises(ValueError, match=reason):
        Renderer.parse_action(text, TASK)


@pytest.mark.parametrize("name", EPISODE_ENVS)
def test_reference_episodes_survive_the_chatml_round_trip(name: str):
    env = get_environment_spec(name).create()
    task = env.get_task(3)
    renderer = Renderer(_encode)
    state = env.reset(task, seed=41).state
    pieces = [renderer.initial_text(task)]
    tokens = renderer.encode_initial(task)
    done = False
    for action in task.private["reference_actions"]:
        turn = renderer.action_text(action, reasoning="plan the next step")
        parsed = renderer.parse_action(turn, task)
        assert parsed.to_wire() == action.to_wire()
        pieces.append(turn)
        tokens += renderer.encode_action(turn)
        result = env.step(task, state, parsed)
        state = result.state
        if result.done:
            pieces.append(renderer.final_suffix(parsed))
            tokens += renderer.encode_final_suffix(parsed)
            done = True
            break
        pieces.append(renderer.observation_text(tuple(result.events)))
        tokens += renderer.encode_observation(tuple(result.events))
    assert done
    # Append-only: encoding the pieces one by one is encoding the whole transcript.
    assert tokens == _encode("".join(pieces))
    transcript = "".join(pieces)
    assert transcript.count("<|im_start|>assistant\n<think>\n") == transcript.count("</think>")
    assert "<|reliquary" not in transcript
