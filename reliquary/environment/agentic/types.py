"""Small, wire-safe types shared by every Reliquary episode environment."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Literal, Mapping, Sequence


EPISODE_SCHEMA = "reliquary/episode/v1"
MAX_ACTION_BYTES = 16 * 1024
MAX_EVENT_BYTES = 64 * 1024
MAX_TOOL_ARGUMENTS = 64


def canonical_json(value: Any) -> str:
    """Return the consensus JSON representation used by traces and digests."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _load_json_object(text: str, *, max_bytes: int) -> dict[str, Any]:
    encoded = text.encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError(f"JSON payload exceeds {max_bytes} bytes")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(
        text,
        object_pairs_hook=reject_duplicate,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value: {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError("action must be a JSON object")
    return value


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.name or len(self.name) > 128:
            raise ValueError("tool name must contain 1..128 characters")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("tool parameters must be a mapping")

    def to_wire(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class EpisodeTask:
    id: str
    prompt: str
    tools: tuple[ToolSpec, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    private: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.id or len(self.id) > 128:
            raise ValueError("task id must contain 1..128 characters")
        if not self.prompt:
            raise ValueError("task prompt must not be empty")
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("task tool names must be unique")

    def to_public_wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "tools": [tool.to_wire() for tool in self.tools],
            "metadata": dict(self.metadata),
        }


ActionKind = Literal["tool", "final"]


@dataclass(frozen=True, slots=True)
class AssistantAction:
    kind: ActionKind
    tool: str | None = None
    arguments: Mapping[str, Any] = field(default_factory=dict)
    content: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "tool":
            if not self.tool:
                raise ValueError("tool action requires a tool name")
            if self.content is not None:
                raise ValueError("tool action cannot contain final content")
            if len(self.arguments) > MAX_TOOL_ARGUMENTS:
                raise ValueError("too many tool arguments")
        elif self.kind == "final":
            if self.tool is not None or self.arguments:
                raise ValueError("final action cannot contain a tool call")
            if self.content is None:
                raise ValueError("final action requires content")
        else:
            raise ValueError(f"unknown action kind: {self.kind}")
        if len(self.to_json().encode("utf-8")) > MAX_ACTION_BYTES:
            raise ValueError("action exceeds the byte limit")

    @classmethod
    def tool_call(cls, name: str, **arguments: Any) -> "AssistantAction":
        return cls(kind="tool", tool=name, arguments=arguments)

    @classmethod
    def final(cls, content: str) -> "AssistantAction":
        return cls(kind="final", content=content)

    @classmethod
    def from_json(cls, text: str) -> "AssistantAction":
        value = _load_json_object(text.strip(), max_bytes=MAX_ACTION_BYTES)
        if set(value) == {"tool", "arguments"}:
            if not isinstance(value["tool"], str):
                raise ValueError("tool must be a string")
            if not isinstance(value["arguments"], dict):
                raise ValueError("arguments must be an object")
            return cls(
                kind="tool",
                tool=value["tool"],
                arguments=value["arguments"],
            )
        if set(value) == {"final"} and isinstance(value["final"], str):
            return cls.final(value["final"])
        raise ValueError(
            'action must be {"tool":str,"arguments":object} or {"final":str}'
        )

    @classmethod
    def from_wire(cls, value: Mapping[str, Any]) -> "AssistantAction":
        return cls.from_json(canonical_json(dict(value)))

    def to_wire(self) -> dict[str, Any]:
        if self.kind == "tool":
            return {"tool": self.tool, "arguments": dict(self.arguments)}
        return {"final": self.content}

    def to_json(self) -> str:
        return canonical_json(self.to_wire())


EventRole = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class EpisodeEvent:
    role: EventRole
    content: str
    name: str | None = None
    action_index: int | None = None

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"unknown event role: {self.role}")
        if len(self.content.encode("utf-8")) > MAX_EVENT_BYTES:
            raise ValueError("event exceeds the byte limit")
        if self.role == "tool" and not self.name:
            raise ValueError("tool event requires a name")

    def to_wire(self) -> dict[str, Any]:
        value: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            value["name"] = self.name
        if self.action_index is not None:
            value["action_index"] = self.action_index
        return value


@dataclass(frozen=True, slots=True)
class GeneratedAction:
    """One policy-generated action and its exact sampled token information."""

    text: str
    tokens: tuple[int, ...] = ()
    token_logprobs: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if len(self.text.encode("utf-8")) > MAX_ACTION_BYTES:
            raise ValueError("generated action exceeds the byte limit")
        if self.token_logprobs and len(self.token_logprobs) != len(self.tokens):
            raise ValueError("token logprobs must match generated tokens")
        if any(not math.isfinite(float(value)) for value in self.token_logprobs):
            raise ValueError("token logprobs must be finite")


@dataclass(frozen=True, slots=True)
class ResetResult:
    state: Any
    events: tuple[EpisodeEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class StepResult:
    state: Any
    events: tuple[EpisodeEvent, ...]
    done: bool = False
    termination_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RewardCheck:
    name: str
    passed: bool
    weight: float = 1.0
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("reward check name must not be empty")
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("reward check weight must be finite and non-negative")

    def to_wire(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "weight": self.weight,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class RewardReport:
    reward: float
    success: bool
    checks: tuple[RewardCheck, ...]
    state_digest: str
    environment_error: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.reward) or not 0.0 <= self.reward <= 1.0:
            raise ValueError("episode reward must be finite and in [0, 1]")
        if len(self.state_digest) != 64:
            raise ValueError("state digest must be a SHA-256 hex digest")

    @classmethod
    def from_checks(
        cls,
        checks: Sequence[RewardCheck],
        *,
        state_digest: str,
        fatal: bool = False,
        binary: bool = False,
    ) -> "RewardReport":
        values = tuple(checks)
        success = bool(values) and not fatal and all(
            check.passed for check in values
        )
        denominator = sum(check.weight for check in values)
        reward = (
            float(success)
            if binary
            else 0.0
            if fatal or denominator <= 0
            else sum(check.weight for check in values if check.passed) / denominator
        )
        return cls(
            reward=float(reward),
            success=success,
            checks=values,
            state_digest=state_digest,
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "success": self.success,
            "checks": [check.to_wire() for check in self.checks],
            "state_digest": self.state_digest,
            "environment_error": self.environment_error,
        }


@dataclass(frozen=True, slots=True)
class EpisodeTrace:
    environment: str
    task_id: str
    seed: int
    events: tuple[EpisodeEvent, ...]
    actions: tuple[AssistantAction, ...]
    assistant_spans: tuple[tuple[int, int], ...] = ()
    tokens: tuple[int, ...] = ()
    assistant_logprobs: tuple[float, ...] = ()
    observation_digests: tuple[str, ...] = ()
    termination_reason: str = "unknown"
    reward: RewardReport | None = None
    schema: str = EPISODE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EPISODE_SCHEMA:
            raise ValueError(f"unsupported episode schema: {self.schema}")
        previous = 0
        assistant_tokens = 0
        for start, end in self.assistant_spans:
            if start < previous or end <= start or end > len(self.tokens):
                raise ValueError("assistant spans must be sorted token intervals")
            previous = end
            assistant_tokens += end - start
        if self.assistant_logprobs and len(self.assistant_logprobs) != assistant_tokens:
            raise ValueError("assistant logprobs must match assistant span tokens")

    @property
    def trace_digest(self) -> str:
        # Renderer output and span boundaries are verified separately. Keeping
        # this digest semantic makes tokenizer-backed generation and
        # dependency-free validator replay converge on the same identity.
        return sha256_json({
            "schema": self.schema,
            "environment": self.environment,
            "task_id": self.task_id,
            "seed": self.seed,
            "actions": [action.to_wire() for action in self.actions],
            "observation_digests": list(self.observation_digests),
            "termination_reason": self.termination_reason,
            "reward": None if self.reward is None else self.reward.to_wire(),
        })

    def to_wire(
        self,
        *,
        include_reward: bool = True,
        include_tokens: bool = False,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "environment": self.environment,
            "task_id": self.task_id,
            "seed": self.seed,
            "events": [event.to_wire() for event in self.events],
            "actions": [action.to_wire() for action in self.actions],
            "assistant_spans": [list(span) for span in self.assistant_spans],
            "observation_digests": list(self.observation_digests),
            "termination_reason": self.termination_reason,
        }
        if include_tokens:
            value["tokens"] = list(self.tokens)
            value["assistant_logprobs"] = list(self.assistant_logprobs)
        if include_reward:
            value["reward"] = None if self.reward is None else self.reward.to_wire()
        return value
