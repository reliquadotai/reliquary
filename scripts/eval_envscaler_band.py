#!/usr/bin/env python3
"""Measure whether ``envscaler_tools_v1`` produces a training signal.

The environment follows upstream's contract: an unreadable turn or an
unknown tool is a recoverable error observation, prose with no tool call
terminates the episode and scores the state reached, and the reward is the
continuous share of checks true at the end. So the band is a threshold on
that reward's spread, not a count of successes.

Two questions, in the order that decides:

1. **Do 16 rollouts of one task disagree enough to select on?** That is
   ``sigma >= SIGMA_MIN``; below it the auction values the group at zero
   and the gate drops it, whatever the mean.
2. **What does the action contract cost?** Every turn is read twice from
   one generation — ``strict`` is production's ``AssistantAction.from_json``,
   which demands the whole completion be one bare JSON object, and
   ``lenient`` takes the first brace-balanced object anywhere in the turn.
   ``--contract`` picks which one drives the episode; the other is carried
   as a counterfactual.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from reliquary.environment.agentic.renderer import CanonicalEpisodeRenderer
from reliquary.environment.agentic.types import (
    AssistantAction,
    EpisodeTrace,
)

SIGMA_MIN = 0.24
ROLLOUTS = 16


def _json_objects(text: str):
    """Yield complete brace-balanced spans, outermost first, left to right.

    A greedy ``{.*}`` spans every object in the turn at once and parses as
    nothing — which is how a first pass read 96% invalid actions off a model
    that was emitting valid ones.
    """
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                yield text[start:index + 1]
                start = -1
            elif depth < 0:
                depth = 0


def parse_action(text: str):
    """First well-formed action in the turn.

    First, not last: the model often emits several candidate calls in one
    breath, and the environment executes one action per turn.
    """
    for block in _json_objects(text):
        try:
            value = json.loads(block)
        except Exception:
            continue
        if not isinstance(value, dict):
            continue
        if "final" in value:
            return ("final", None, str(value["final"]))
        if "tool" in value:
            name = value["tool"]
            if not isinstance(name, str) or not name.strip():
                continue
            args = value.get("arguments")
            return ("tool", name, args if isinstance(args, dict) else {})
    return None


def initial_text(task) -> str:
    return CanonicalEpisodeRenderer.initial_text(task)


def observation_text(events) -> str:
    return CanonicalEpisodeRenderer.observation_text(tuple(events))


def strict_action(text: str):
    """Production's reading: the whole turn must be one bare JSON object."""
    try:
        return AssistantAction.from_json(text)
    except (RecursionError, TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tasks", type=int, default=48)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-action-tokens", type=int, default=512)
    parser.add_argument("--gen-seed", type=int, default=7)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--contract", choices=("strict", "lenient"),
                        default="strict",
                        help="how a turn is read into an action; strict is "
                             "what runner.py does today")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from reliquary.environment.agentic.envs.envscaler_tools_v1.environment import (
        EnvScalerToolsEnvironment,
    )

    environment = EnvScalerToolsEnvironment()
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model, revision=args.revision, dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem, max_model_len=16384,
        seed=args.gen_seed,
    )
    sampling = SamplingParams(
        n=1, temperature=1.0, top_p=1.0, top_k=-1,
        max_tokens=args.max_action_tokens, seed=args.gen_seed,
        stop=["<|reliquary_end|>", "<|reliquary_user|>", "<|reliquary_tool|>",
              "<|reliquary_assistant|>", "<|reliquary_system|>"],
    )

    # One live rollout per (task, replica): the episode branches on what the
    # model emits, so turns cannot be batched across different histories.
    # Every rollout of every task advances one turn per generate() call.
    tasks = [environment.get_task(i) for i in range(args.tasks)]
    live = []
    for task in tasks:
        for replica in range(ROLLOUTS):
            state = environment.reset(task, seed=replica).state
            live.append({
                "task": task, "replica": replica, "state": state,
                "text": initial_text(task), "done": False, "reason": None,
                "tokens": 0, "invalid": False, "turns": 0,
                "strict_turns": 0, "strict_ok": 0, "strict_alive": True,
                "unreadable": 0,
            })

    started = time.time()
    for turn in range(args.max_turns):
        pending = [r for r in live if not r["done"]]
        if not pending:
            break
        prompts = [r["text"] for r in pending]
        outputs = llm.generate(prompts, sampling)
        for rollout, output in zip(pending, outputs):
            completion = output.outputs[0]
            rollout["tokens"] += len(completion.token_ids)
            rollout["turns"] += 1
            rollout["strict_turns"] += 1
            strict = strict_action(completion.text)
            rollout["strict_ok"] += int(strict is not None)
            if strict is None:
                rollout["strict_alive"] = False
            action = strict
            if action is None and args.contract == "lenient":
                parsed = parse_action(completion.text)
                if parsed is not None:
                    kind, name, payload = parsed
                    try:
                        action = (
                            AssistantAction(kind="final", content=payload)
                            if kind == "final"
                            else AssistantAction(kind="tool", tool=name,
                                                 arguments=payload)
                        )
                    except (TypeError, ValueError):
                        action = None
            if action is None:
                # runner.py does exactly this: the turn becomes a call to a
                # tool that does not exist, and the environment's error budget
                # decides whether the episode survives it.
                action = AssistantAction.tool_call("__invalid_action__")
                rollout["unreadable"] += 1
            rollout["text"] += completion.text
            result = environment.step(rollout["task"], rollout["state"], action)
            rollout["text"] += observation_text(result.events)
            if result.done:
                rollout["done"] = True
                rollout["reason"] = result.termination_reason
                rollout["invalid"] = result.termination_reason == "invalid_action"
        print(f"turn {turn + 1}: {len(pending)} live, "
              f"{sum(1 for r in live if r['done'])} finished", flush=True)
    for rollout in live:
        if not rollout["done"]:
            rollout["reason"] = "turn_limit"
    print(f"generation {time.time() - started:.0f}s", flush=True)

    rows = []
    for task in tasks:
        replicas = [r for r in live if r["task"].id == task.id]
        rewards = []
        for rollout in replicas:
            trace = EpisodeTrace(
                schema="reliquary/episode/v1", environment=environment.name,
                task_id=task.id, seed=rollout["replica"], events=(), actions=(),
                tokens=(), assistant_spans=(), observation_digests=(),
                termination_reason=rollout["reason"] or "turn_limit",
            )
            rewards.append(environment.grade(task, rollout["state"], trace))
        values = [r.reward for r in rewards]
        mean = sum(values) / len(values)
        sigma = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        rows.append({
            "task_id": task.id, "env_id": task.metadata["family"],
            "checks": len(task.private["checks"]),
            "rewards": [round(v, 4) for v in values],
            "reward_mean": mean, "reward_sigma": sigma,
            "reward_best": max(values),
            "solved": sum(1 for r in rewards if r.success),
            "invalid": sum(1 for r in replicas
                           if r["reason"] == "tool_raised"),
            "unreadable": sum(r["unreadable"] for r in replicas),
            "strict_turns": sum(r["strict_turns"] for r in replicas),
            "strict_ok": sum(r["strict_ok"] for r in replicas),
            "strict_alive": sum(1 for r in replicas if r["strict_alive"]),
            "finished": sum(1 for r in replicas if r["reason"] == "finished"),
            "turns_median": sorted(r["turns"] for r in replicas)[len(replicas) // 2],
            "tokens_median": sorted(r["tokens"] for r in replicas)[len(replicas) // 2],
        })

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump({"model": args.model, "sigma_min": SIGMA_MIN,
                   "contract": args.contract, "rows": rows},
                  handle, indent=1)

    n = len(rows)
    total = n * ROLLOUTS
    strict_turns = sum(r["strict_turns"] for r in rows)
    print(f"\nband = sigma >= {SIGMA_MIN} on the continuous reward\n")
    print(f"tasks                 {n}")
    print(f"mean reward           {sum(r['reward_mean'] for r in rows) / n:.3f}"
          "   (no-op baseline 0.166)")
    print(f"best-of-16 reward     {sum(r['reward_best'] for r in rows) / n:.3f}")
    print(f"tasks fully solved    {sum(r['solved'] for r in rows) / total:.1%}")
    print(f"GROUPS IN BAND        "
          f"{sum(r['reward_sigma'] >= SIGMA_MIN for r in rows) / n:.1%}")
    print(f"groups with spread    "
          f"{sum(r['reward_sigma'] > 0 for r in rows) / n:.1%}")
    print(f"median group sigma    "
          f"{sorted(r['reward_sigma'] for r in rows)[n // 2]:.3f}")
    print(f"episodes ended by a raising tool "
          f"{sum(r['invalid'] for r in rows) / total:.1%}")
    print(f"turns unreadable      "
          f"{sum(r['unreadable'] for r in rows) / max(strict_turns, 1):.1%}"
          f"   under --contract {args.contract}")
    print(f"turns that are bare JSON "
          f"{sum(r['strict_ok'] for r in rows) / max(strict_turns, 1):.1%}")
    print(f"explicit termination  {sum(r['finished'] for r in rows) / total:.1%}")
    print(f"turns  median         {sorted(r['turns_median'] for r in rows)[n // 2]}")
    print(f"tokens median         {sorted(r['tokens_median'] for r in rows)[n // 2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
