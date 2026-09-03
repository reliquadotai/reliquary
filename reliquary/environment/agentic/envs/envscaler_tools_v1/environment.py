"""Episode environment over EnvScaler's programmatic tool worlds (MIT).

The upstream release carries each world as Python source and each scenario
as an initial state plus a task plus a checklist of terminal-state boolean
functions. This adapter presents that as a Reliquary episode: deterministic
from an index, replayable by the validator, graded on final state.

Three properties are load-bearing and each is pinned by a test:

* the execution namespace is frozen (``shims``), or replay diverges;
* the initial config is deep-copied per reset, or two rollouts of one prompt
  share mutable state;
* only checks that are *false* at the initial state count toward reward.
  Measured over 400 scenarios, 16.7% of checks are already true before the
  agent acts, so a raw pass rate pays a model that does nothing.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from typing import Any, ClassVar

from reliquary.environment.agentic.envs.envscaler_tools_v1.shims import (
    build_environment_class,
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


ENVIRONMENT_NAME = "envscaler_tools_v1"
GENERATOR_VERSION = "envscaler-v1"
CHECKER_VERSION = "envscaler-checker-v1"
MAX_TURNS = 12
# What runner.py calls a turn it could not read. Upstream reaches the same
# state by routing content-without-a-tool-call to chat_with_user.
_PROSE_TOOL = "__invalid_action__"
_DATA_ENV_VAR = "RELIQUARY_ENVSCALER_DATA"


@dataclass
class WorldState:
    instance: Any
    initial: dict[str, Any]
    invalid_actions: int = 0
    final_response: str | None = None
    calls: list[str] = field(default_factory=list)


def _data_root() -> Path:
    configured = os.environ.get(_DATA_ENV_VAR)
    if not configured:
        raise RuntimeError(
            f"{_DATA_ENV_VAR} must point at the pinned EnvScaler release"
        )
    return Path(configured)


@lru_cache(maxsize=1)
def _corpus() -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    root = _data_root()
    worlds = json.loads((root / "env_meta.json").read_text(encoding="utf-8"))
    if isinstance(worlds, dict):
        worlds = list(worlds.values())
    scenarios = json.loads((root / "rl_scen.json").read_text(encoding="utf-8"))
    by_id = {world["env_id"]: world for world in worlds}
    # Order is the corpus identity: scenarios are addressed by position.
    usable = tuple(
        scenario for scenario in scenarios if scenario["env_id"] in by_id
    )
    return by_id, usable


@lru_cache(maxsize=64)
def _world_class(env_id: str) -> type:
    worlds, _ = _corpus()
    world = worlds[env_id]
    return build_environment_class(
        world["env_class_code"], world["env_class_name"]
    )


def _tool_specs(world: dict[str, Any]) -> tuple[ToolSpec, ...]:
    """Upstream ships OpenAI function-calling shapes; map them straight over."""
    specs = []
    for entry in world.get("tools", ()):
        function = entry.get("function") or {}
        name = function.get("name")
        if not name:
            continue
        specs.append(ToolSpec(
            name=str(name),
            description=str(function.get("description", "")),
            parameters=dict(function.get("parameters") or {"type": "object"}),
        ))
    return tuple(specs)


def _state_of(instance: Any) -> dict[str, Any]:
    return deepcopy({
        key: value for key, value in vars(instance).items()
        if not (key.startswith("__") and key.endswith("__"))
    })


def _run_check(source: str, initial: dict, final: dict) -> bool | None:
    """Execute one terminal-state check. Returns None when it misbehaves."""
    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "initial_state": deepcopy(initial),
    }
    try:
        exec(source, namespace)  # noqa: S102 - dataset-carried, sandbox required
        checker = namespace.get("check_func")
        if checker is None:
            return None
        verdict = checker(deepcopy(final))
        return verdict if isinstance(verdict, bool) else None
    except Exception:
        return None


class EnvScalerToolsEnvironment:
    """Multi-turn tool use over a pinned EnvScaler scenario corpus."""

    name: ClassVar[str] = ENVIRONMENT_NAME
    validator_authoritative_reward: ClassVar[bool] = True
    max_turns: ClassVar[int] = MAX_TURNS

    def __len__(self) -> int:
        _worlds, scenarios = _corpus()
        return len(scenarios)

    def get_task(self, index: int) -> EpisodeTask:
        worlds, scenarios = _corpus()
        scenario = scenarios[int(index) % len(scenarios)]
        world = worlds[scenario["env_id"]]
        prompt = (
            f"{world['environment_introduction'].strip()}\n\n"
            "Rules:\n"
            + "\n".join(f"- {rule}" for rule in world.get("constraints_rules", ()))
            + f"\n\nTask:\n{scenario['task'].strip()}"
        )
        return EpisodeTask(
            id=str(scenario["task_id"]),
            prompt=prompt,
            tools=_tool_specs(world),
            metadata={
                "family": scenario["env_id"],
                "world": world["environment_summary"],
            },
            private={
                "env_id": scenario["env_id"],
                "init_config": scenario["init_config"],
                "checks": scenario["checklist_with_func"],
            },
        )

    def reset(self, task: EpisodeTask, seed: int) -> ResetResult:
        del seed
        world_class = _world_class(task.private["env_id"])
        # Deep-copied twice on purpose: the constructor may retain what it is
        # given, and two rollouts of one prompt must not share mutable state.
        config = deepcopy(dict(task.private["init_config"]))
        try:
            instance = world_class(deepcopy(config))
        except TypeError:
            instance = world_class()
        for key, value in config.items():
            setattr(instance, key, deepcopy(value))
        return ResetResult(
            state=WorldState(instance=instance, initial=_state_of(instance))
        )

    def _event(self, name: str, value: Any) -> tuple[EpisodeEvent, ...]:
        return (EpisodeEvent(
            role="tool", name=name, content=canonical_json(value)
        ),)

    def step(
        self, task: EpisodeTask, state: WorldState, action: AssistantAction
    ) -> StepResult:
        """Upstream's contract, followed deliberately.

        `EnvScalerBaseEnv.step` is forgiving in a specific shape, and the
        shape is the point: an unreadable turn or an unknown tool is an
        error *observation* the agent may recover from, while prose with no
        tool call is routed to `chat_with_user`, which in the non-conversational
        mode ends the episode and scores the state reached. Only a tool that
        raises is fatal. Departing from this measures our contract rather
        than theirs.
        """
        name = str(action.tool) if action.kind == "tool" else ""
        # runner.py turns a turn it cannot read into this call; upstream turns
        # the same turn into chat_with_user, which terminates. Same thing.
        if action.kind == "final" or name == _PROSE_TOOL:
            state.final_response = action.content or ""
            return StepResult(
                state,
                self._event("final", {"accepted": True}),
                done=True,
                termination_reason="finished",
            )
        known = {spec.name for spec in task.tools}
        if name not in known:
            # Recoverable, and uncapped: max_turns already bounds the cost.
            state.invalid_actions += 1
            return StepResult(
                state,
                self._event("error", {
                    "success": False,
                    "error": f"unknown tool: {name[:80]}",
                }),
            )
        try:
            result = getattr(state.instance, name)(**dict(action.arguments))
        except Exception as exc:
            state.invalid_actions += 1
            return StepResult(
                state,
                self._event(name, {
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}"[:400],
                }),
                done=True,
                termination_reason="tool_raised",
            )
        state.calls.append(name)
        return StepResult(state, self._event(name, result))

    def grade(
        self, task: EpisodeTask, state: WorldState, trace: EpisodeTrace
    ) -> RewardReport:
        """Upstream's reward: the share of checks true at the final state.

        `EnvScalerBaseEnv.calculate_reward` averages every check, including
        the ones already true at reset — a no-op agent scores 0.166 on
        average rather than zero, and breaking an invariant costs reward.
        Grading only the checks the agent must flip, and only all-or-nothing,
        makes the reward identically zero for every rollout of a 4B and
        leaves the sigma gate nothing to select on.
        """
        final = _state_of(state.instance)
        checks: list[RewardCheck] = []
        passed = 0
        for position, entry in enumerate(task.private["checks"]):
            after = _run_check(entry["check_func"], state.initial, final)
            passed += int(after is True)
            checks.append(RewardCheck(
                name=f"check_{position}",
                passed=after is True,
                weight=1.0,
                detail=str(entry.get("check_item", ""))[:200],
            ))
        total = len(task.private["checks"])
        reward = passed / total if total else 0.0
        checks.append(RewardCheck(
            "no_invalid_tool_calls", state.invalid_actions == 0, 0.0
        ))
        checks.append(RewardCheck(
            "finished_explicitly",
            trace.termination_reason == "finished",
            0.0,
        ))
        return RewardReport(
            reward=reward,
            success=total > 0 and passed == total,
            checks=tuple(checks),
            state_digest=hashlib.sha256(
                canonical_json(final).encode("utf-8")
            ).hexdigest(),
        )

    def close(self, state: WorldState) -> None:
        return None


__all__ = ["EnvScalerToolsEnvironment", "ENVIRONMENT_NAME", "MAX_TURNS"]
