"""A ChatML episode rendered by a miner must replay byte for byte on a validator.

The validator re-renders every episode and compares the transcript to the
miner's tokens exactly. Both sides now choose their renderer from the
environment's declared id through one lookup; before that, both hard-coded the
JSONL renderer, and an environment declaring ChatML would have failed every
replay on `episode canonical transcript mismatch`.

These tests drive the two real code paths — the runner the miner uses, and the
`replay_tokenized_episode` the validator calls from admission — rather than
constructing the transcripts by hand.
"""

from __future__ import annotations

import pytest

from reliquary.environment.agentic.chatml import (
    CHATML_RENDERER_ID,
    ChatMLEpisodeRenderer,
)
from reliquary.environment.agentic.renderer import (
    EPISODE_RENDERER_ID,
    CanonicalEpisodeRenderer,
)
from reliquary.environment.agentic.renderers import RENDERER_IDS, renderer_for
from reliquary.environment.agentic.replay import replay_tokenized_episode
from reliquary.environment.agentic.runner import EpisodeRunner
from reliquary.environment.agentic.types import AssistantAction, GeneratedAction
from reliquary.environment.registry import get_environment_spec


def _encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def _decode(ids: list[int]) -> str:
    return bytes(ids).decode("utf-8")


class _ChatMLPolicy:
    """Writes each scripted action the way a ChatML-trained model would."""

    def __init__(self, actions: list[AssistantAction]) -> None:
        self._actions = iter(actions)

    def generate(self, **_: object) -> GeneratedAction:
        action = next(self._actions)
        return GeneratedAction(
            text=ChatMLEpisodeRenderer.action_text(
                action, reasoning="I should look this up first."
            )
        )


def _environment():
    return get_environment_spec("reliquary_stateful_tools_v1").create()


def _script(environment) -> list[AssistantAction]:
    task = environment.get_task(0)
    tool = task.tools[0]
    arguments = {
        name: "C-1001"
        for name in (tool.parameters.get("properties") or {})
    }
    return [
        AssistantAction(kind="tool", tool=tool.name, arguments=arguments),
        AssistantAction.final("The request has been handled."),
    ]


def _miner_trace(renderer_id: str, seed: int = 7):
    environment = _environment()
    task = environment.get_task(0)
    return EpisodeRunner(renderer=renderer_for(renderer_id, _encode)).run(
        environment, task, seed=seed, policy=_ChatMLPolicy(_script(environment))
    )


def test_a_chatml_episode_replays_byte_for_byte_on_the_validator_path() -> None:
    mined = _miner_trace(CHATML_RENDERER_ID)
    replayed = replay_tokenized_episode(
        _environment(),
        task_index=0,
        seed=mined.seed,
        tokens=list(mined.tokens),
        assistant_spans=mined.assistant_spans,
        decode=_decode,
        encode=_encode,
        renderer_id=CHATML_RENDERER_ID,
    )
    assert tuple(replayed.tokens) == tuple(mined.tokens)
    assert replayed.assistant_spans == mined.assistant_spans
    assert [a.to_wire() for a in replayed.actions] == [a.to_wire() for a in mined.actions]
    assert replayed.trace_digest == mined.trace_digest
    assert replayed.reward.state_digest == mined.reward.state_digest


def test_the_chatml_turns_are_read_as_actions_not_as_invalid_ones() -> None:
    """The failure this wiring prevents. Parsed as JSON, every ChatML turn is
    `__invalid_action__` and the episode ends on its first call."""
    mined = _miner_trace(CHATML_RENDERER_ID)
    tools = [a.tool for a in mined.actions if a.kind == "tool"]
    assert tools and "__invalid_action__" not in tools
    assert mined.actions[-1].kind == "final"


def test_replaying_with_the_wrong_renderer_is_caught() -> None:
    """If the two sides ever disagreed, the transcript comparison is what
    notices — which is why a single lookup matters more than either side."""
    mined = _miner_trace(CHATML_RENDERER_ID)
    replayed = replay_tokenized_episode(
        _environment(),
        task_index=0,
        seed=mined.seed,
        tokens=list(mined.tokens),
        assistant_spans=mined.assistant_spans,
        decode=_decode,
        encode=_encode,
        renderer_id=EPISODE_RENDERER_ID,
    )
    assert tuple(replayed.tokens) != tuple(mined.tokens)


def test_the_canonical_renderer_still_reads_json_exactly_as_before() -> None:
    """The existing environments pin these bytes; their behaviour must not
    move, only the source that implements it."""
    text = AssistantAction.final("done").to_json()
    assert CanonicalEpisodeRenderer.parse_action(text) == AssistantAction.from_json(text)


def test_every_declared_renderer_resolves_and_an_unknown_one_is_refused() -> None:
    assert RENDERER_IDS == {EPISODE_RENDERER_ID, CHATML_RENDERER_ID}
    assert isinstance(renderer_for(CHATML_RENDERER_ID, _encode), ChatMLEpisodeRenderer)
    with pytest.raises(ValueError, match="unknown episode renderer"):
        renderer_for("reliquary-invented-v1", _encode)


def test_a_malformed_chatml_call_becomes_an_invalid_action_not_a_crash() -> None:
    """A validator replays untrusted model text. A turn that closes its
    reasoning but writes a broken call must end the episode cleanly."""

    class _Broken:
        def generate(self, **_: object) -> GeneratedAction:
            return GeneratedAction(text="thinking\n</think>\n\n<tool_call>\n<function=x")

    environment = _environment()
    trace = EpisodeRunner(renderer=renderer_for(CHATML_RENDERER_ID, _encode)).run(
        environment, environment.get_task(0), seed=1, policy=_Broken()
    )
    assert trace.actions[0].tool == "__invalid_action__"


CALL_TURN = (
    "looking the customer up\n</think>\n\n<tool_call>\n"
    "<function=get_customer_by_phone>\n<parameter=phone_number>\n"
    "555-123-2002\n</parameter>\n</function>\n</tool_call>"
)


def test_a_turn_that_carries_its_own_terminator_still_commits() -> None:
    # What the policy actually writes: it stops on <|im_end|>, and the decoded
    # turn keeps it. Measured on Teutonic-I, every well-formed call arrived this
    # way and was refused as "text after the function call".
    action = ChatMLEpisodeRenderer.parse_action(CALL_TURN + "<|im_end|>")

    assert action.kind == "tool"
    assert action.tool == "get_customer_by_phone"
    assert dict(action.arguments) == {"phone_number": "555-123-2002"}


def test_both_terminators_are_consumed() -> None:
    for tail in ("<|endoftext|>", "<|im_end|>\n", "<|im_end|><|endoftext|>"):
        action = ChatMLEpisodeRenderer.parse_action(CALL_TURN + tail)
        assert action.tool == "get_customer_by_phone"


def test_a_final_answer_keeps_its_text_when_the_terminator_goes() -> None:
    action = ChatMLEpisodeRenderer.parse_action(
        "thinking\n</think>\n\nthe line is suspended<|im_end|>"
    )

    assert action.kind == "final"
    assert action.content == "the line is suspended"


def test_real_text_after_the_call_is_still_refused() -> None:
    import pytest

    with pytest.raises(ValueError):
        ChatMLEpisodeRenderer.parse_action(CALL_TURN + "\nand then I will check<|im_end|>")
