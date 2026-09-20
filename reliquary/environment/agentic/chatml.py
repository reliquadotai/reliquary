"""ChatML rendering of episodes, in the tool-calling dialect of the Qwen3.5 chat template.

The canonical renderer speaks a Reliquary-only frame, so a model trained on it speaks
nothing else: measured on a distilled Teutonic graft, a perfect three-turn episode in
that frame, and in its own chat template a call emitted before the weather it depended
on was known, then the same two calls repeated until the token cap. This renderer
produces, for one task and its tool results, the text that chat template produces, so
what the model learns here is what an application serving it will send.

Two deliberate departures from the template, both needed for token-exact replay:
past reasoning is never rewritten (the template drops it from turns that precede a
later user message), and exactly one function call is accepted per turn.

Not wired into the protocol yet: nothing here is listed in a pinned environment
manifest, so adding it changes no consensus digest.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from typing import Any

from reliquary.environment.agentic.types import (
    AssistantAction,
    EpisodeEvent,
    EpisodeTask,
    ToolSpec,
)


CHATML_RENDERER_ID = "reliquary-chatml-tools-v1"

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
# What a policy's turn ends on when it stops itself. The template writes the
# terminator as part of the *next* turn's framing, so a turn that carries its
# own is complete, not overlong: measured on Teutonic-I, every well-formed call
# arrived as `</tool_call><|im_end|>` and was refused as text after the call.
TURN_END = ("<|im_end|>", "<|endoftext|>")
THINK_OPEN = "<think>\n"
THINK_CLOSE = "</think>"
CALL_OPEN = "<tool_call>"
CALL_CLOSE = "</tool_call>"

# Verbatim from the Qwen3.5 chat template: the instructions the model was post-trained on.
_TOOLS_HEADER = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
_TOOLS_FOOTER = "\n</tools>"
_CALL_INSTRUCTIONS = (
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:"
    "\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
    "value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the "
    "second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n"
    "</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
    "format: an inner <function=...></function> block must be nested within "
    "<tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may "
    "provide optional reasoning for your function call in natural language BEFORE the "
    "function call, but NOT after\n- If there is no function call available, answer the "
    "question like normal with your current knowledge and do not tell the user about "
    "function calls\n</IMPORTANT>"
)
# The system message's own content. "Wait for its result" targets the failure above.
EPISODE_GUIDANCE = (
    "You are operating a deterministic tool environment. Work out what to do next, then "
    "either call exactly one function and wait for its result, or, when the task is "
    "complete, reply with the final answer and no function call."
)


def _tool_entry(tool: ToolSpec) -> str:
    # Same serialization as the chat template's `tojson` filter under transformers.
    return json.dumps({"type": "function", "function": tool.to_wire()}, ensure_ascii=False)


def _parameter_text(value: Any) -> str:
    # The template writes objects and lists as JSON and everything else through `string`.
    if isinstance(value, Mapping) or isinstance(value, (list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _without_turn_end(text: str) -> str:
    """The turn without the terminator the policy stopped on."""

    stripped = text.rstrip()
    changed = True
    while changed:
        changed = False
        for terminator in TURN_END:
            if stripped.endswith(terminator):
                stripped = stripped[: -len(terminator)].rstrip()
                changed = True
    return stripped


class ChatMLEpisodeRenderer:
    """Append-only ChatML episode text: no token already emitted is ever re-encoded."""

    id = CHATML_RENDERER_ID
    stop_text = IM_END

    def __init__(self, encode: Callable[[str], list[int]]) -> None:
        self._encode = encode

    @staticmethod
    def initial_text(task: EpisodeTask) -> str:
        if task.tools:
            tools = "".join("\n" + _tool_entry(tool) for tool in task.tools)
            system = _TOOLS_HEADER + tools + _TOOLS_FOOTER + _CALL_INSTRUCTIONS + "\n\n" + EPISODE_GUIDANCE
        else:
            system = EPISODE_GUIDANCE
        return (
            f"{IM_START}system\n{system}{IM_END}\n"
            f"{IM_START}user\n{task.prompt.strip()}{IM_END}\n"
            f"{IM_START}assistant\n{THINK_OPEN}"
        )

    @staticmethod
    def observation_text(events: tuple[EpisodeEvent, ...]) -> str:
        rendered = f"{IM_END}\n"
        responses: list[str] = []

        def flush() -> str:
            if not responses:
                return ""
            # Consecutive tool results share one user message, as in the template.
            block = "".join(f"\n<tool_response>\n{content}\n</tool_response>" for content in responses)
            responses.clear()
            return f"{IM_START}user{block}{IM_END}\n"

        for event in events:
            if event.role == "tool":
                responses.append(event.content.strip())
                continue
            rendered += flush()
            # The template allows a system message only first; mid-episode it is the user speaking.
            role = "assistant" if event.role == "assistant" else "user"
            rendered += f"{IM_START}{role}\n{event.content.strip()}{IM_END}\n"
        rendered += flush()
        return rendered + f"{IM_START}assistant\n{THINK_OPEN}"

    @staticmethod
    def final_suffix(action: AssistantAction) -> str:
        del action
        return f"{IM_END}\n"

    @staticmethod
    def action_text(action: AssistantAction, *, reasoning: str = "", preamble: str = "") -> str:
        """The turn a model writes after the prefilled `<think>`, as the template would render it."""

        text = reasoning.strip() + "\n" + THINK_CLOSE + "\n\n"
        if action.kind == "final":
            return text + (action.content or "").strip()
        preamble = preamble.strip()
        call = f"{CALL_OPEN}\n<function={action.tool}>\n"
        for name, value in action.arguments.items():
            call += f"<parameter={name}>\n{_parameter_text(value)}\n</parameter>\n"
        call += f"</function>\n{CALL_CLOSE}"
        return text + (f"{preamble}\n\n{call}" if preamble else call)

    @staticmethod
    def parse_action(text: str, task: EpisodeTask | None = None) -> AssistantAction:
        """The action a ChatML turn commits to; raises ValueError when it commits to none.

        Arguments are typed from the task's tool schema, because the dialect writes every
        value as text. A turn must close its reasoning, and may hold prose followed by at
        most one function call with nothing after it.
        """

        text = _without_turn_end(text)
        if THINK_CLOSE not in text:
            raise ValueError("turn never closed its reasoning")
        # The template also takes what follows the last closing tag.
        content = text.rsplit(THINK_CLOSE, 1)[1].strip()
        opened = content.count(CALL_OPEN)
        if opened == 0:
            if CALL_CLOSE in content or "<function=" in content:
                raise ValueError("function call markup outside a tool_call block")
            if not content:
                raise ValueError("turn holds no answer and no function call")
            return AssistantAction.final(content)
        if opened > 1:
            raise ValueError("more than one function call in one turn")
        start = content.index(CALL_OPEN)
        end = content.find(CALL_CLOSE, start)
        if end < 0:
            raise ValueError("unclosed tool_call block")
        if content[end + len(CALL_CLOSE):].strip():
            raise ValueError("text after the function call")
        name, raw_arguments = _parse_function(content[start + len(CALL_OPEN):end])
        schema = _parameter_schema(task, name)
        arguments = {key: _coerce(value, schema.get(key)) for key, value in raw_arguments.items()}
        return AssistantAction(kind="tool", tool=name, arguments=arguments)

    def encode_initial(self, task: EpisodeTask) -> list[int]:
        return list(self._encode(self.initial_text(task)))

    def encode_observation(self, events: tuple[EpisodeEvent, ...]) -> list[int]:
        return list(self._encode(self.observation_text(events)))

    def encode_action(self, text: str) -> list[int]:
        return list(self._encode(text))

    def encode_final_suffix(self, action: AssistantAction) -> list[int]:
        return list(self._encode(self.final_suffix(action)))


def _parse_function(block: str) -> tuple[str, dict[str, str]]:
    prefix = "\n<function="
    if not block.startswith(prefix):
        raise ValueError("tool_call block does not open a function")
    head_end = block.find(">\n", len(prefix))
    if head_end < 0:
        raise ValueError("malformed function header")
    name = block[len(prefix):head_end]
    if not name or "<" in name or "\n" in name:
        raise ValueError("malformed function name")
    cursor = head_end + 2
    arguments: dict[str, str] = {}
    while block.startswith("<parameter=", cursor):
        name_start = cursor + len("<parameter=")
        name_end = block.find(">\n", name_start)
        if name_end < 0:
            raise ValueError("malformed parameter header")
        key = block[name_start:name_end]
        value_end = block.find("\n</parameter>\n", name_end + 2)
        if not key or "<" in key or "\n" in key or value_end < 0:
            raise ValueError("malformed parameter")
        if key in arguments:
            raise ValueError(f"parameter {key!r} given twice")
        arguments[key] = block[name_end + 2:value_end]
        cursor = value_end + len("\n</parameter>\n")
    if block[cursor:] != "</function>\n":
        raise ValueError("unexpected text inside the function call")
    return name, arguments


def _parameter_schema(task: EpisodeTask | None, tool: str) -> Mapping[str, Any]:
    if task is None:
        return {}
    for spec in task.tools:
        if spec.name == tool:
            properties = spec.parameters.get("properties", {})
            return properties if isinstance(properties, Mapping) else {}
    return {}


def _coerce(value: str, schema: Any) -> Any:
    """Type one textual argument from its JSON schema; anything unexpected stays text."""

    declared = schema.get("type") if isinstance(schema, Mapping) else None
    types = declared if isinstance(declared, list) else [declared]
    if declared is None:
        # Untyped parameters are common in upstream tool schemas; the template wrote
        # objects and lists as JSON, so read those back and leave the rest as text.
        if value[:1] in ("{", "["):
            try:
                parsed = json.loads(value)
            except ValueError:
                return value
            if isinstance(parsed, (dict, list)):
                return parsed
        return value
    if "string" not in types and value in ("None", "null"):
        # An optional non-string argument left empty: the template writes Python's None.
        return None
    for kind in types:
        if kind == "null" and value in ("None", "null"):
            return None
        if kind == "boolean" and value in ("True", "true", "False", "false"):
            return value in ("True", "true")
        if kind == "integer":
            try:
                return int(value)
            except ValueError:
                continue
        if kind == "number":
            try:
                number = float(value)
            except ValueError:
                continue
            return int(number) if number.is_integer() and "." not in value and "e" not in value.lower() else number
        if kind in ("object", "array"):
            try:
                parsed = json.loads(value)
            except ValueError:
                continue
            if isinstance(parsed, dict if kind == "object" else list):
                return parsed
    return value
