"""Token-exact evaluation of the artifact-verified telecom benchmark."""

from __future__ import annotations

import math
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from reliquary.environment.agentic.base import EpisodeEnvironment
from reliquary.environment.agentic.chatml import CHATML_RENDERER_ID
from reliquary.environment.agentic.runner import EpisodeRunner, ScriptedPolicy
from reliquary.environment.agentic.types import (
    EPISODE_SCHEMA, AssistantAction, EpisodeTask, EpisodeTrace, GeneratedAction, RewardReport,
)

TELECOM_ENVIRONMENT = "reliquary_telecom_solo_v2"


@dataclass(frozen=True)
class TelecomBenchmark:
    environment: EpisodeEnvironment
    manifest_sha256: str


@dataclass(frozen=True)
class EpisodeCompletion:
    """Display text may omit stop tokens; token_ids are the generation authority."""

    text: str
    token_ids: tuple[int, ...]
    finish_reason: str


@dataclass(frozen=True)
class EpisodeEvaluation:
    trace: EpisodeTrace
    finish_reasons: tuple[str, ...]
    generated_tokens: int
    tool_calls: int
    elapsed_seconds: float


def load_telecom_benchmark(expected_manifest_sha256: str) -> TelecomBenchmark:
    """Import only verified wheel bytes and explicitly select the held-out split."""
    from reliquary.environment.agentic.external import (
        ExternalEpisodeEnvironment, import_external_modules,
    )
    from reliquary.environment.registry import get_environment_spec

    spec = get_environment_spec(TELECOM_ENVIRONMENT)
    if spec.environment_manifest_sha256 != expected_manifest_sha256:
        raise ValueError("telecom environment manifest changed")
    module_name, _, factory_name = spec.factory_path.partition(":")
    _, (module,) = import_external_modules(spec, [module_name])
    backend = getattr(module, factory_name)(split="benchmark")
    environment = ExternalEpisodeEnvironment(backend, spec)
    if len(environment) != 134 or environment.max_turns != 40:
        raise ValueError("telecom benchmark shape changed")
    return TelecomBenchmark(environment, expected_manifest_sha256)


class _DeadlineEnvironment:
    """The pinned CPU environment has no external tools or network calls."""

    def __init__(self, environment, check):
        self._environment, self._check = environment, check
        self.name = environment.name
        self.max_turns = environment.max_turns
        self.validator_authoritative_reward = environment.validator_authoritative_reward
        self.state = None
        self.opened = False

    def reset(self, task, seed):
        self._check()
        result = self._environment.reset(task, seed)
        self.state, self.opened = result.state, True
        self._check()
        return result

    def step(self, task, state, action):
        self._check()
        result = self._environment.step(task, state, action)
        self.state = result.state
        self._check()
        return result

    def grade(self, task, state, trace):
        self._check()
        result = self._environment.grade(task, state, trace)
        self._check()
        return result

    def close(self, state):
        self.opened = False
        self._environment.close(state)


def run_telecom_episode(source: TelecomBenchmark, *, task: EpisodeTask, seed: int,
                        renderer, generate_turn, decode,
                        max_turns: int, max_new_tokens: int, max_episode_tokens: int,
                        episode_timeout_seconds: float,
                        clock: Callable[[], float] = time.monotonic,
                        cancelled: Callable[[], bool] = lambda: False) -> EpisodeEvaluation:
    """Run one validated benchmark task, then replay it without model generation.

    The caller validates the frozen set/request, runtime image and checkpoint,
    supplies the benchmark's exact task, and binds renderer/decode/generate_turn
    to the same verified local tokenizer/model. generate_turn must hard-stop its
    owned process at the supplied absolute monotonic deadline. Source identity
    and protocol validation remain in the existing publication/dispatch layer.
    """
    if (source.environment.name != TELECOM_ENVIRONMENT
            or source.environment.validator_authoritative_reward is not True
            or not isinstance(source.manifest_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", source.manifest_sha256)
            or not isinstance(task, EpisodeTask)
            or getattr(renderer, "id", None) != CHATML_RENDERER_ID):
        raise ValueError("unsupported telecom benchmark source or renderer")
    if (type(seed) is not int or not 0 <= seed < 2**31
            or type(max_turns) is not int
            or not 1 <= max_turns <= min(40, source.environment.max_turns)
            or type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096
            or type(max_episode_tokens) is not int or not 1 <= max_episode_tokens <= 65536
            or type(episode_timeout_seconds) not in (float, int)
            or not math.isfinite(episode_timeout_seconds)
            or not 0 < episode_timeout_seconds <= 3600):
        raise ValueError("invalid telecom episode budget or seed")
    started = clock()
    deadline = started + episode_timeout_seconds
    turns, finish_reasons, prefixes = [], [], []

    def check():
        if cancelled():
            raise RuntimeError("telecom evaluation cancelled")
        if clock() >= deadline:
            raise TimeoutError("telecom episode deadline exceeded")

    class Policy:
        def generate(self, *, tokens, **_):
            check()
            cap = min(max_new_tokens, max_episode_tokens - len(tokens))
            if cap <= 0:
                raise ValueError("telecom transcript token budget exhausted")
            # The owned local model process must hard-stop at this deadline.
            completion = generate_turn(tokens=tokens, max_new_tokens=cap,
                                       seed=seed, deadline=deadline)
            check()
            if (not isinstance(completion, EpisodeCompletion)
                    or completion.finish_reason not in {"stop", "length"}
                    or not completion.token_ids
                    or any(type(token) is not int or token < 0 for token in completion.token_ids)
                    or len(completion.token_ids) > cap):
                raise ValueError("telecom model turn is incomplete or token identity changed")
            # vLLM's display text can omit EOS. Parse the unchanged actual token sequence.
            action = GeneratedAction(text=decode(list(completion.token_ids)),
                                     tokens=tuple(completion.token_ids))
            prefixes.append(tokens)
            turns.append(action)
            finish_reasons.append(completion.finish_reason)
            return action

    def run(policy):
        environment = _DeadlineEnvironment(source.environment, check)
        try:
            return EpisodeRunner(renderer=renderer, max_turns=max_turns,
                                 max_episode_tokens=max_episode_tokens).run(
                environment, task, seed=seed, policy=policy)
        finally:
            # EpisodeRunner's prompt/render failures can precede its own cleanup block.
            if environment.opened:
                environment.close(environment.state)

    check()
    if len(renderer.encode_initial(task)) > max_episode_tokens:
        raise ValueError("telecom prompt exceeds transcript token budget")
    trace = run(Policy())
    if (trace.reward is None or trace.reward.environment_error is not None
            or trace.reward.reward not in {0.0, 1.0}
            or trace.reward.success != (trace.reward.reward == 1.0)):
        raise ValueError("telecom authoritative reward is unavailable")

    class ReplayPolicy:
        index = 0

        def generate(self, *, tokens, **_):
            check()
            index = self.index
            if index >= len(turns) or tokens != prefixes[index]:
                raise ValueError("telecom replay observations changed")
            self.index += 1
            return turns[index]

    replay = run(ReplayPolicy())
    check()
    if replay.to_wire(include_tokens=True) != trace.to_wire(include_tokens=True):
        raise ValueError("telecom deterministic replay changed")
    return EpisodeEvaluation(
        trace=trace,
        finish_reasons=tuple(finish_reasons),
        generated_tokens=sum(len(turn.tokens) for turn in turns),
        tool_calls=sum(action.kind == "tool" for action in trace.actions),
        elapsed_seconds=clock() - started,
    )


def grade_telecom_episode(source: TelecomBenchmark, *, task: EpisodeTask, seed: int,
                          episode_wire: dict, max_turns: int, max_new_tokens: int,
                          max_episode_tokens: int, episode_timeout_seconds: float = 300,
                          clock: Callable[[], float] = time.monotonic) -> RewardReport:
    """Recompute the frozen task's reward without trusting the executor's reward.

    Token counts are validated executor evidence; semantic replay does not prove
    model generation or independently authenticate tokenizer output.
    """
    from reliquary.environment.agentic.chatml import ChatMLEpisodeRenderer
    from reliquary.protocol.corpus_submission import MAX_TRANSCRIPT_BYTES
    import hashlib

    if (source.environment.name != TELECOM_ENVIRONMENT
            or source.environment.validator_authoritative_reward is not True
            or not isinstance(task, EpisodeTask)
            or type(seed) is not int or not 0 <= seed < 2**31
            or type(max_turns) is not int or not 1 <= max_turns <= min(40, source.environment.max_turns)
            or type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096
            or type(max_episode_tokens) is not int or not 1 <= max_episode_tokens <= 65536
            or type(episode_timeout_seconds) not in (int, float)
            or not math.isfinite(episode_timeout_seconds) or not 0 < episode_timeout_seconds <= 3600):
        raise ValueError("invalid telecom replay source or budget")
    if (not isinstance(episode_wire, dict) or episode_wire.get("schema") != EPISODE_SCHEMA
            or episode_wire.get("environment") != TELECOM_ENVIRONMENT
            or episode_wire.get("task_id") != task.id or type(episode_wire.get("seed")) is not int
            or episode_wire["seed"] != seed):
        raise ValueError("telecom replay task identity changed")
    if len(json.dumps(episode_wire, allow_nan=False).encode("utf-8")) > MAX_TRANSCRIPT_BYTES:
        raise ValueError("telecom replay transcript exceeds the byte budget")
    tokens, spans = episode_wire.get("tokens"), episode_wire.get("assistant_spans")
    actions, events = episode_wire.get("actions"), episode_wire.get("events")
    if (not isinstance(tokens, list) or not 1 <= len(tokens) <= max_episode_tokens
            or any(type(token) is not int or token < 0 for token in tokens)
            or not isinstance(actions, list) or not 1 <= len(actions) <= max_turns
            or not isinstance(spans, list) or len(spans) != len(actions)
            or not isinstance(events, list)
            or any(not isinstance(event, dict) for event in events)):
        raise ValueError("invalid telecom replay transcript bounds")
    previous = 0
    for span in spans:
        if (not isinstance(span, (list, tuple)) or len(span) != 2
                or any(type(value) is not int for value in span)
                or not previous <= span[0] < span[1] <= len(tokens)
                or span[1] - span[0] > max_new_tokens):
            raise ValueError("invalid telecom replay model turn span")
        previous = span[1]
    assistant = [event for event in events if event.get("role") == "assistant"]
    if len(assistant) != len(actions):
        raise ValueError("telecom replay model turns changed")
    parsed = []
    for index, (event, wire) in enumerate(zip(assistant, actions)):
        text = event.get("content")
        if (not isinstance(text, str) or len(text.encode("utf-8")) > 64 * 1024
                or type(event.get("action_index")) is not int or event["action_index"] != index):
            raise ValueError("invalid telecom replay model turn")
        try:
            action = ChatMLEpisodeRenderer.parse_action(text, task)
        except (RecursionError, TypeError, ValueError):
            # Keep the shared EpisodeRunner's explicit invalid-action accounting.
            raw = text.encode("utf-8", errors="backslashreplace")
            action = AssistantAction.tool_call("__invalid_action__",
                raw_prefix=raw[:256].decode("utf-8", errors="replace"),
                raw_sha256=hashlib.sha256(raw).hexdigest())
        if action.to_wire() != wire:
            raise ValueError("telecom replay action differs from the model turn")
        parsed.append(action)
    deadline = clock() + episode_timeout_seconds

    def check():
        if clock() >= deadline:
            raise TimeoutError("telecom replay deadline exceeded")

    environment = _DeadlineEnvironment(source.environment, check)
    try:
        replay = EpisodeRunner(max_turns=max_turns).run(
            environment, task, seed=seed, policy=ScriptedPolicy(parsed))
    finally:
        if environment.opened:
            environment.close(environment.state)
    check()
    actual = replay.to_wire()
    for key in ("actions", "observation_digests", "termination_reason", "reward"):
        if actual[key] != episode_wire.get(key):
            raise ValueError("telecom authoritative replay changed: " + key)
    if ([event for event in actual["events"] if event["role"] != "assistant"]
            != [event for event in events if event.get("role") != "assistant"]):
        raise ValueError("telecom authoritative replay observations changed")
    reward = replay.reward
    if (reward is None or reward.environment_error is not None or reward.reward not in {0.0, 1.0}
            or reward.success != (reward.reward == 1.0)):
        raise ValueError("telecom authoritative replay reward unavailable")
    return reward
