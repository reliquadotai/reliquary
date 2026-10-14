"""EnvScaler tool worlds, second version: ChatML, contained, binary, split.

What changed from ``envscaler_tools_v1``, which stays as it was:

* **Rendered in ChatML** (``reliquary-chatml-tools-v1``), the Qwen tool-call
  dialect the policy was post-trained on, as ``reliquary_telecom_solo_v1`` is.
  A turn ends on ``<|im_end|>``, so the RL miner and the corpus generator stop
  on the same token. A turn that holds a ``<tool_call>`` is a call; a turn that
  holds an answer and no call is the final answer and ends the episode, which
  is what the renderer's system guidance tells the model and what upstream
  does with a reply that calls nothing (``chat_with_user`` in
  non-conversational mode). A turn the renderer cannot read — reasoning never
  closed, malformed call markup, two calls — ends the episode as
  ``invalid_action``.

* **Contained.** Every execution of world or check code — reset, each step,
  grading — runs in a world process (``sandbox``/``worker``): rlimits, no
  network, no subprocess, no file access, a constructed environment. A world
  that loops, allocates without bound or reaches for the host fails as a tool
  that raised, with fixed bytes; it never takes the caller with it.

* **Deterministic by construction.** One fresh process per episode, with the
  hash seed, the time zone, the clock, uuid, random and secrets pinned or
  seeded inside it, and only plain JSON coming out. Nothing is cached across
  episodes, so the first rollout of a prompt and the eighth see the same
  identifiers, as does the validator replaying either.

* **Binary reward on the required checks**, pinned here rather than read from
  the host: ``envscaler_tools_v1`` switched modes on an environment variable,
  which made the reward a property of the machine. A check is *required* when
  it is false in the initial state (about 15.5% of checks are already true and
  measure nothing). The reward is 1 when every required check holds in the
  final state and the episode ended on a final answer, else 0. Binary beat the
  graded ratio in the earlier measurement and keeps the lattice ``(0, 1)``.

* **Split by world, laid out by range** (``corpus``): RL scenarios
  ``[0, RL_END)``, graded SFT scenarios ``[RL_END, SFT_GRADED_END)``, ungraded
  SFT scenarios ``[SFT_GRADED_END, LENGTH)``. The constants below are those
  boundaries; a test holds them equal to ``selection.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, ClassVar

from reliquary.environment.agentic.envs.envscaler_tools_v2.corpus import (
    Corpus,
    GRADED_RANGES,
    Scenario,
    default_corpus,
    world_source,
)
from reliquary.environment.agentic.envs.envscaler_tools_v2.sandbox import (
    WorldProcess,
    WorldProcessError,
)
from reliquary.environment.agentic.types import (
    AssistantAction,
    EpisodeEvent,
    EpisodeTask,
    EpisodeTrace,
    ResetResult,
    RewardCheck,
    RewardReport,
    StepResult,
    ToolSpec,
    canonical_json,
)


ENVIRONMENT_NAME = "envscaler_tools_v2"
GENERATOR_VERSION = "envscaler-v2"
CHECKER_VERSION = "envscaler-checker-v2"
REWARD_MODE = "binary-required-checks-finished-v1"
# An upper bound; the profile's episode.max_turns is what a run uses.
# Measured on Qwen3.8-27B over the graded SFT range (2026-10-13, 119
# episodes at 30): wins took 5-29 turns, 46 of 48 within 25; the 12 that
# reached turn 30 all lost (11 turn_limit at 11k trajectory tokens on
# average, one invalid_action at 26.7k). Past 25 a turn buys almost no win
# and the longest, costliest failures.
MAX_TURNS = 25

# The task prompt's last paragraph. The renderer's system message already asks
# for one call per turn; Qwen3.8 still put 2-11 calls in the first turn of 49
# episodes in 50, and the renderer reads such a turn as invalid_action, which
# ends the episode (5.6% won). Said again where the task is, with what it
# costs, 2 episodes in 119 did it (40% won).
TASK_INSTRUCTIONS = (
    "Call one tool per message, then wait for its result before the next call: "
    "a message with more than one tool call ends the session as a failure. When "
    "the task is complete, reply with a final message and no tool call."
)

# Index-space boundaries. Half-open, contiguous, in this order, and what a
# range guard reads (`reliquary.eval.sets` restates them; a test holds the
# two equal to each other and to `selection.json`):
#   rl            [0, 1296)     26 worlds, graded   (1300 upstream, 4 vendor)
#   sft_graded    [1296, 2546)  25 worlds, graded   (1250 upstream)
#   sft_ungraded  [2546, 7225)  140 worlds, no checks (4684 upstream, 5 vendor)
#   sft_corpus    [7225, 7825)  12 worlds, graded: sft_graded minus the 13
#                               worlds of corpus.SFT_CORPUS_EXCLUDED_WORLDS,
#                               the range a corpus job serves
# No scenario failed to start, none is won by a no-op agent (every one has a
# check false at reset), none has an initial state that differs between two
# processes.
RL_START = 0
RL_END = 1296
SFT_GRADED_START = RL_END
SFT_GRADED_END = 2546
SFT_UNGRADED_START = SFT_GRADED_END
SFT_UNGRADED_END = 7225
SFT_CORPUS_START = SFT_UNGRADED_END
LENGTH = 7825
RL_RANGE = (RL_START, RL_END)
SFT_GRADED_RANGE = (SFT_GRADED_START, SFT_GRADED_END)
SFT_UNGRADED_RANGE = (SFT_UNGRADED_START, SFT_UNGRADED_END)
SFT_CORPUS_RANGE = (SFT_CORPUS_START, LENGTH)

# The runner's name for a turn it could not read.
_UNREADABLE = "__invalid_action__"


@dataclass
class WorldState:
    scenario: Scenario
    process: WorldProcess | None = None
    # Set when the world never started, or its process died: the episode is
    # over and the reward is 0, whatever the trace says.
    failure: str | None = None
    invalid_actions: int = 0
    final_response: str | None = None
    calls: list[str] = field(default_factory=list)


def _tool_specs(world: dict[str, Any]) -> tuple[ToolSpec, ...]:
    """Upstream ships OpenAI function-calling shapes; map them straight over."""
    specs = []
    seen = set()
    for entry in world.get("tools", ()):
        function = entry.get("function") or {}
        name = function.get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        specs.append(ToolSpec(
            name=str(name),
            description=str(function.get("description", "")),
            parameters=dict(function.get("parameters") or {"type": "object"}),
        ))
    return tuple(specs)


def _event(name: str, value: Any) -> tuple[EpisodeEvent, ...]:
    return (EpisodeEvent(role="tool", name=name, content=canonical_json(value)),)


def _failure_digest(reason: str) -> str:
    return hashlib.sha256(
        canonical_json({"world_failure": reason}).encode("utf-8")
    ).hexdigest()


class EnvScalerToolsV2Environment:
    """Multi-turn tool use over the pinned, filtered EnvScaler scenarios."""

    name: ClassVar[str] = ENVIRONMENT_NAME
    validator_authoritative_reward: ClassVar[bool] = True
    max_turns: ClassVar[int] = MAX_TURNS

    def __init__(self, corpus: Corpus | None = None) -> None:
        # Injected by tests and by the filter script; otherwise the pinned
        # corpus under RELIQUARY_ENVSCALER_DATA, loaded once per process.
        self._injected = corpus

    @property
    def corpus(self) -> Corpus:
        return self._injected if self._injected is not None else default_corpus()

    def __len__(self) -> int:
        return len(self.corpus)

    def scenario(self, index: int) -> Scenario:
        scenarios = self.corpus.scenarios
        return scenarios[int(index) % len(scenarios)]

    def get_task(self, index: int) -> EpisodeTask:
        scenario = self.scenario(index)
        world = self.corpus.worlds[scenario.env_id]
        prompt = (
            f"{str(world['environment_introduction']).strip()}\n\n"
            "Rules:\n"
            + "\n".join(f"- {rule}" for rule in world.get("constraints_rules", ()))
            + f"\n\nTask:\n{scenario.task.strip()}"
            + f"\n\n{TASK_INSTRUCTIONS}"
        )
        return EpisodeTask(
            id=scenario.task_id,
            prompt=prompt,
            tools=_tool_specs(world),
            metadata={
                "family": scenario.env_id,
                "world": str(world.get("environment_summary", "")),
                "range": scenario.range,
                "graded": scenario.range in GRADED_RANGES,
                "checker_version": CHECKER_VERSION,
            },
            private={"index": int(index) % len(self.corpus.scenarios)},
        )

    def _scenario_of(self, task: EpisodeTask) -> Scenario:
        scenario = self.scenario(int(task.private["index"]))
        if scenario.task_id != task.id:
            raise ValueError("task does not belong to this corpus")
        return scenario

    def reset(self, task: EpisodeTask, seed: int) -> ResetResult:
        """A fresh world process for one episode.

        ``seed`` is accepted and ignored: everything a world could draw is
        seeded inside the process with a constant, so rollouts of one prompt
        differ only by what the policy does.
        """
        del seed
        scenario = self._scenario_of(task)
        world = self.corpus.worlds[scenario.env_id]
        state = WorldState(scenario=scenario)
        try:
            state.process = WorldProcess()
            reply = state.process.request({
                "op": "init",
                "source": world_source(scenario.env_id, str(world["env_class_code"])),
                "class_name": str(world["env_class_name"]),
                "init_config": dict(scenario.init_config),
            })
        except WorldProcessError as exc:
            state.failure = str(exc)
            return ResetResult(state=state)
        if not reply.get("ok"):
            state.failure = "world failed to start"
            state.process.close()
        return ResetResult(state=state)

    def step(
        self, task: EpisodeTask, state: WorldState, action: AssistantAction
    ) -> StepResult:
        if state.failure is not None:
            return StepResult(
                state,
                _event("error", {"success": False, "error": state.failure}),
                done=True,
                termination_reason="world_error",
            )
        if action.kind == "final":
            state.final_response = action.content or ""
            return StepResult(
                state,
                _event("final", {"accepted": True}),
                done=True,
                termination_reason="finished",
            )
        name = str(action.tool)
        if name == _UNREADABLE:
            state.invalid_actions += 1
            return StepResult(
                state,
                _event("error", {
                    "success": False,
                    "error": "unreadable turn: no final answer and no tool call",
                }),
                done=True,
                termination_reason="invalid_action",
            )
        if name not in {spec.name for spec in task.tools}:
            # Recoverable, as upstream: max_turns already bounds the cost.
            state.invalid_actions += 1
            return StepResult(
                state,
                _event("error", {
                    "success": False,
                    "error": f"unknown tool: {name[:80]}",
                }),
            )
        assert state.process is not None
        try:
            reply = state.process.request({
                "op": "call", "tool": name, "arguments": dict(action.arguments),
            })
        except WorldProcessError as exc:
            state.failure = str(exc)
            return StepResult(
                state,
                _event(name, {"success": False, "error": state.failure}),
                done=True,
                termination_reason="world_error",
            )
        if not reply.get("ok"):
            # Upstream's one fatal case: a tool that raises.
            state.invalid_actions += 1
            return StepResult(
                state,
                _event(name, {
                    "success": False, "error": str(reply.get("error", ""))[:400],
                }),
                done=True,
                termination_reason="tool_raised",
            )
        state.calls.append(name)
        return StepResult(state, _event(name, reply.get("result")))

    def grade(
        self, task: EpisodeTask, state: WorldState, trace: EpisodeTrace
    ) -> RewardReport:
        scenario = state.scenario
        finished = trace.termination_reason == "finished"
        tail = (
            RewardCheck("finished_explicitly", finished, 0.0),
            RewardCheck("no_invalid_tool_calls", state.invalid_actions == 0, 0.0),
        )
        if scenario.checks is None:
            # EnvScaler-SFT-Scenario ships no checks. Graded nowhere, so a job
            # over this range must not filter on reward.
            return RewardReport(
                reward=0.0,
                success=False,
                checks=(RewardCheck(
                    "ungraded", False, 0.0, "scenario ships no checks",
                ),) + tail,
                state_digest=self._digest(state),
                environment_error=state.failure,
            )
        if state.failure is not None or state.process is None:
            return self._failed(state, tail)
        try:
            reply = state.process.request({
                "op": "grade",
                "checks": [str(entry["check_func"]) for entry in scenario.checks],
            })
        except WorldProcessError as exc:
            state.failure = str(exc)
            return self._failed(state, tail)
        if not reply.get("ok"):
            state.failure = "grading failed"
            return self._failed(state, tail)
        checks: list[RewardCheck] = []
        required = passed = 0
        for position, (entry, before, after) in enumerate(zip(
            scenario.checks, reply["initial"], reply["final"], strict=True,
        )):
            # A check already true before the agent acted measures nothing; one
            # that cannot run on the initial state is not counted either.
            counted = before is False
            required += int(counted)
            passed += int(counted and after is True)
            checks.append(RewardCheck(
                name=f"check_{position}",
                passed=after is True,
                weight=1.0 if counted else 0.0,
                detail=str(entry.get("check_item", ""))[:200],
            ))
        success = required > 0 and passed == required and finished
        return RewardReport(
            reward=1.0 if success else 0.0,
            success=success,
            checks=tuple(checks) + tail,
            state_digest=str(reply["state_digest"]),
        )

    def _digest(self, state: WorldState) -> str:
        if state.failure is not None or state.process is None:
            return _failure_digest(state.failure or "no process")
        try:
            reply = state.process.request({"op": "grade", "checks": []})
        except WorldProcessError as exc:
            state.failure = str(exc)
            return _failure_digest(state.failure)
        return str(reply.get("state_digest") or _failure_digest("grading failed"))

    @staticmethod
    def _failed(state: WorldState, tail: tuple[RewardCheck, ...]) -> RewardReport:
        reason = state.failure or "no process"
        return RewardReport(
            reward=0.0,
            success=False,
            checks=(RewardCheck("world_ran", False, 0.0, reason),) + tail,
            state_digest=_failure_digest(reason),
            environment_error=reason,
        )

    def close(self, state: WorldState) -> None:
        if state.process is not None:
            state.process.close()


__all__ = [
    "ENVIRONMENT_NAME",
    "EnvScalerToolsV2Environment",
    "LENGTH",
    "MAX_TURNS",
    "REWARD_MODE",
    "RL_RANGE",
    "SFT_CORPUS_RANGE",
    "SFT_GRADED_RANGE",
    "SFT_UNGRADED_RANGE",
    "TASK_INSTRUCTIONS",
]
