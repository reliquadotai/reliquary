"""Real shared episode loops with synthetic CPU tools and token-exact model turns."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from reliquary.eval.agentic import (
    EpisodeCompletion, TELECOM_ENVIRONMENT, TelecomBenchmark,
    run_telecom_episode, load_telecom_benchmark,
    grade_telecom_episode,
)
from reliquary.environment.agentic.chatml import ChatMLEpisodeRenderer
from reliquary.environment.agentic.types import (
    EpisodeEvent, EpisodeTask, ResetResult, RewardCheck, RewardReport, StepResult, ToolSpec,
)


def encode(text):
    result = []
    for index, part in enumerate(text.split("<|im_end|>")):
        if index:
            result.append(256)
        result.extend(ord(character) for character in part)
    return result


def decode(tokens):
    return "".join("<|im_end|>" if token == 256 else chr(token) for token in tokens)


CALL = ("</think>\n\n<tool_call>\n<function=connect>\n<parameter=enabled>\ntrue\n</parameter>\n"
        "</function>\n</tool_call><|im_end|>")
FINAL = "</think>\n\nConnection restored.<|im_end|>"


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value


class TelecomTools:
    name = TELECOM_ENVIRONMENT
    max_turns = 40
    validator_authoritative_reward = True

    def __init__(self, *, drift=None, clock=None):
        self.resets = 0
        self.closed = []
        self.drift = drift
        self.clock = clock

    def get_task(self, index):
        assert index == 0
        return EpisodeTask(
            id="benchmark-ticket", prompt="Restore network access.",
            tools=(ToolSpec("connect", "Set network connectivity.",
                            {"type": "object", "properties": {"enabled": {"type": "boolean"}},
                             "required": ["enabled"]}),),
            metadata={"split": "benchmark"},
        )

    def reset(self, task, seed):
        self.resets += 1
        return ResetResult({"connected": False, "finished": False})

    def step(self, task, state, action):
        if self.clock:
            self.clock.value += 40
        if action.kind == "final":
            state["finished"] = True
            return StepResult(state, (), done=True, termination_reason="finished")
        assert action.tool == "connect"
        state["connected"] = action.arguments["enabled"]
        content = "Connected" if self.drift != "observation" or self.resets == 1 else "Changed"
        return StepResult(state, (EpisodeEvent(role="tool", name="connect", content=content),))

    def grade(self, task, state, trace):
        success = state["connected"] and state["finished"]
        if self.drift == "reward" and self.resets > 1:
            success = False
        return RewardReport.from_checks(
            [RewardCheck("network_restored", success)],
            state_digest=("1" if success else "0") * 64, binary=True,
        )

    def close(self, state):
        self.closed.append(dict(state))


def run(environment=None, *, texts=(CALL, FINAL), reasons=("length", "stop"),
        clock=None, callback=None, cancelled=lambda: False, **limits):
    environment = environment or TelecomTools()
    source = TelecomBenchmark(environment, "a" * 64)
    task = environment.get_task(0)
    calls = []
    clock = clock or Clock()

    def generate_turn(**request):
        index = len(calls)
        calls.append(request)
        if callback:
            return callback(index, request)
        text = texts[index]
        return EpisodeCompletion(text, tuple(encode(text)), reasons[index])

    settings = {"max_turns": 40, "max_new_tokens": 512, "max_episode_tokens": 4096,
                "episode_timeout_seconds": 60, **limits}
    result = run_telecom_episode(source, task=task, seed=17,
                          renderer=ChatMLEpisodeRenderer(encode),
                          generate_turn=generate_turn, decode=decode, clock=clock,
                          cancelled=cancelled, **settings)
    return result, environment, calls


def test_actual_multiple_turns_observe_tools_and_replay_without_more_model_calls():
    result, environment, calls = run()
    assert len(calls) == 2
    assert environment.resets == len(environment.closed) == 2
    assert result.trace.reward.reward == 1
    assert result.trace.reward.success is True
    assert result.trace.termination_reason == "finished"
    assert [action.kind for action in result.trace.actions] == ["tool", "final"]
    assert "Connected" in decode(calls[1]["tokens"])
    assert CALL in decode(calls[1]["tokens"])
    assert result.finish_reasons == ("length", "stop")
    assert result.generated_tokens == len(encode(CALL)) + len(encode(FINAL))
    assert result.generated_tokens == sum(end - start for start, end in result.trace.assistant_spans)
    assert result.tool_calls == 1
    assert all(request["seed"] == 17 for request in calls)
    assert all(request["deadline"] == 160 for request in calls)


@pytest.mark.parametrize("completion", [
    EpisodeCompletion(CALL, tuple(encode(CALL)), "abort"),
    EpisodeCompletion(CALL, (), "stop"),
    EpisodeCompletion(CALL, (True,), "stop"),
    EpisodeCompletion(CALL, (-1,), "stop"),
])
def test_incomplete_or_mismatched_model_turn_cannot_mutate_the_world(completion):
    environment = TelecomTools()
    with pytest.raises(ValueError, match="model turn"):
        run(environment, callback=lambda *_: completion)
    assert environment.closed == [{"connected": False, "finished": False}]


def test_actual_stop_token_is_preserved_when_vllm_display_text_omits_it():
    def generated(index, _request):
        raw = (CALL, FINAL)[index]
        return EpisodeCompletion(raw.removesuffix("<|im_end|>"), tuple(encode(raw)), "stop")

    result, _, calls = run(callback=generated)
    assert result.trace.reward.reward == 1
    assert result.trace.events[1].content == CALL
    assert result.trace.tokens[result.trace.assistant_spans[0][1] - 1] == 256
    assert calls[1]["tokens"].count(256) >= 1
    assert result.generated_tokens == len(encode(CALL)) + len(encode(FINAL))


def test_actual_output_cannot_exceed_the_advertised_model_turn_cap():
    environment = TelecomTools()
    with pytest.raises(ValueError, match="model turn"):
        run(environment, max_new_tokens=len(encode(CALL)) - 1)
    assert environment.closed == [{"connected": False, "finished": False}]


def test_cumulative_budget_includes_actual_output_and_tool_observations():
    environment = TelecomTools()
    initial = len(ChatMLEpisodeRenderer(encode).encode_initial(environment.get_task(0)))
    with pytest.raises(ValueError, match="token budget"):
        run(environment, max_episode_tokens=initial + len(encode(CALL)))
    assert environment.closed == [{"connected": True, "finished": False}]


def test_turn_limit_preserves_native_failed_task_reward():
    result, environment, calls = run(texts=(CALL,), reasons=("stop",), max_turns=1)
    assert len(calls) == 1
    assert len(environment.closed) == 2
    assert result.trace.termination_reason == "turn_limit"
    assert result.trace.reward.reward == 0
    assert result.trace.reward.success is False


def test_slow_model_turn_raises_instead_of_reporting_a_completed_attempt():
    environment, clock = TelecomTools(), Clock()

    def delayed(*_):
        clock.value += 61
        return EpisodeCompletion(CALL, tuple(encode(CALL)), "stop")

    with pytest.raises(TimeoutError, match="episode deadline"):
        run(environment, clock=clock, callback=delayed)
    assert environment.closed == [{"connected": False, "finished": False}]


def test_whole_episode_deadline_covers_tool_execution_and_closes_state():
    clock = Clock()
    environment = TelecomTools(clock=clock)
    with pytest.raises(TimeoutError, match="episode deadline"):
        run(environment, clock=clock)
    assert len(environment.closed) == 1


def test_failure_after_a_real_tool_call_does_not_become_a_score():
    environment = TelecomTools()

    def failed(index, _request):
        if index:
            raise RuntimeError("local model unavailable")
        return EpisodeCompletion(CALL, tuple(encode(CALL)), "stop")

    with pytest.raises(RuntimeError, match="model unavailable"):
        run(environment, callback=failed)
    assert environment.closed == [{"connected": True, "finished": False}]
    assert environment.resets == 1


def test_cancellation_during_generation_closes_the_partial_environment():
    environment = TelecomTools()
    stopped = False

    def cancel(*_):
        nonlocal stopped
        stopped = True
        return EpisodeCompletion(CALL, tuple(encode(CALL)), "stop")

    with pytest.raises(RuntimeError, match="cancelled"):
        run(environment, callback=cancel, cancelled=lambda: stopped)
    assert len(environment.closed) == 1


@pytest.mark.parametrize("drift", ["observation", "reward"])
def test_replay_must_preserve_observations_and_authoritative_reward(drift):
    environment = TelecomTools(drift=drift)
    with pytest.raises(ValueError, match="replay"):
        run(environment)
    assert len(environment.closed) == 2


def test_loader_uses_the_verified_factory_with_an_explicit_benchmark_split(monkeypatch):
    from reliquary.environment import registry
    from reliquary.environment.agentic import external

    calls = []
    spec = SimpleNamespace(name=TELECOM_ENVIRONMENT, environment_manifest_sha256="a" * 64,
                           validator_authoritative_reward=True,
                           factory_path="reliquary_telecom_solo:TelecomSoloEnvironment")

    class Backend:
        name = TELECOM_ENVIRONMENT
        max_turns = 40
        validator_authoritative_reward = True

        def __len__(self):
            return 134

        def task(self, *args): pass
        def reset(self, *args): pass
        def step(self, *args): pass
        def grade(self, *args): pass
        def close(self, *args): pass

    def factory(**kwargs):
        calls.append(kwargs)
        return Backend()

    monkeypatch.setattr(registry, "get_environment_spec", lambda name: spec)

    def verified(actual, modules):
        assert actual is spec
        assert modules == ["reliquary_telecom_solo"]
        return {}, [SimpleNamespace(TelecomSoloEnvironment=factory)]

    monkeypatch.setattr(external, "import_external_modules", verified)
    source = load_telecom_benchmark("a" * 64)
    assert len(source.environment) == 134
    assert calls == [{"split": "benchmark"}]
    with pytest.raises(ValueError, match="manifest changed"):
        load_telecom_benchmark("b" * 64)
    assert len(calls) == 1


@pytest.mark.parametrize("limits", [
    {"max_turns": 0}, {"max_turns": 41}, {"max_new_tokens": 0},
    {"max_new_tokens": 4097}, {"max_episode_tokens": 65537},
    {"episode_timeout_seconds": float("nan")}, {"episode_timeout_seconds": True},
])
def test_unsupported_execution_limits_fail_before_any_model_or_tool_work(limits):
    environment = TelecomTools()
    with pytest.raises(ValueError, match="budget"):
        run(environment, callback=lambda *_: pytest.fail("model must not run"), **limits)
    assert environment.resets == 0


def test_another_environment_cannot_be_substituted_for_the_telecom_source():
    environment = TelecomTools()
    environment.name = "another-environment"
    with pytest.raises(ValueError, match="unsupported telecom"):
        run(environment, callback=lambda *_: pytest.fail("model must not run"))
    assert environment.resets == 0


def grade(result, *, environment=None, seed=17, **limits):
    environment = environment or TelecomTools()
    return grade_telecom_episode(
        TelecomBenchmark(environment, "a" * 64), task=environment.get_task(0), seed=seed,
        episode_wire=result, max_turns=limits.get("max_turns", 40),
        max_new_tokens=limits.get("max_new_tokens", 512),
        max_episode_tokens=limits.get("max_episode_tokens", 4096),
    )


def test_grader_recomputes_authoritative_reward_from_actual_actions():
    result, _, _ = run()
    environment = TelecomTools()
    reward = grade(result.trace.to_wire(include_tokens=True), environment=environment)
    assert reward.reward == 1 and reward.success is True
    assert environment.resets == len(environment.closed) == 1


@pytest.mark.parametrize("field,value", [
    ("seed", 18), ("task_id", "another-task"),
    ("environment", "another-environment"),
    ("termination_reason", "turn_limit"),
    ("reward", {"reward": 0, "success": False}),
    ("observation_digests", ["0" * 64]),
])
def test_grader_refuses_changed_identity_observations_or_reward(field, value):
    result, _, _ = run()
    wire = result.trace.to_wire(include_tokens=True)
    wire[field] = value
    with pytest.raises(ValueError, match="identity|replay changed"):
        grade(wire)


def test_grader_refuses_a_wire_action_unrelated_to_the_recorded_model_turn():
    result, _, _ = run()
    wire = result.trace.to_wire(include_tokens=True)
    wire["actions"][0]["arguments"]["enabled"] = False
    environment = TelecomTools()
    with pytest.raises(ValueError, match="action differs"):
        grade(wire, environment=environment)
    assert environment.resets == 0


def test_grader_preserves_a_failed_turn_limited_episode():
    result, _, _ = run(texts=(CALL,), reasons=("stop",), max_turns=1)
    reward = grade(result.trace.to_wire(include_tokens=True), max_turns=1)
    assert reward.reward == 0 and reward.success is False


def test_grader_refuses_a_model_turn_over_its_frozen_budget():
    result, _, _ = run()
    with pytest.raises(ValueError, match="model turn span"):
        grade(result.trace.to_wire(include_tokens=True), max_new_tokens=1)
