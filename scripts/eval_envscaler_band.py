#!/usr/bin/env python3
"""Measure whether ``envscaler_tools_v1`` is usable for RL at all.

Four questions, in the order that decides:

1. **Can the model emit a valid action?** On the branch's own episode suite
   35-43% of rollouts died on an invalid action, and stateful_tools_v1's
   apparent 62% success was 95% explained by whether the episode terminated
   cleanly. Format failure masks everything else.
2. **Is it in the sigma band** — the share of 16-rollout groups producing
   gradient. Scored twice from one generation: binary (every required check
   passes) and fractional over required checks only. That settles the reward
   contract without paying for two runs.
3. **How long is an episode** in tokens, for the window budget and w_env.
4. **Does success collapse with the number of required checks?** If it does,
   the checks are independent and binary reward is dead.

Only checks that are *false* at the initial state count: 16.7% of them are
already true before the agent acts.

Every turn is scored against **two** action contracts from one generation:

* ``strict`` — production's ``AssistantAction.from_json``, which demands the
  whole completion be one bare JSON object after ``strip()``;
* ``lenient`` — the first brace-balanced object anywhere in the turn.

The episode is advanced with the lenient reading so that later turns are
observable at all; strict is carried alongside as the counterfactual. The
gap between them is the cost of the contract rather than of the model, and
that is a decision the measurement should inform, not presuppose.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time

SIGMA_MIN = 0.24
ROLLOUTS = 16


def band_bounds(rollouts: int, sigma_min: float) -> tuple[int, int]:
    ok = [
        k for k in range(rollouts + 1)
        if ((k / rollouts) * (1 - k / rollouts)) ** 0.5 >= sigma_min
    ]
    return (min(ok), max(ok)) if ok else (1, rollouts - 1)


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
        EnvScalerToolsEnvironment, MAX_TOOL_ERRORS, _run_check,
    )
    from reliquary.environment.agentic.types import AssistantAction, EpisodeTrace
    from reliquary.environment.agentic.renderer import CanonicalEpisodeRenderer
    globals().update(
        AssistantAction=AssistantAction,
        CanonicalEpisodeRenderer=CanonicalEpisodeRenderer,
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
        state0 = environment.reset(task, seed=0).state
        required = [
            c["check_func"] for c in task.private["checks"]
            if _run_check(c["check_func"], state0.initial, state0.initial) is False
        ]
        binaries, fractions = [], []
        for rollout in replicas:
            trace = EpisodeTrace(
                schema="reliquary/episode/v1", environment=environment.name,
                task_id=task.id, seed=rollout["replica"], events=(), actions=(),
                tokens=(), assistant_spans=(), observation_digests=(),
                termination_reason=rollout["reason"] or "turn_limit",
            )
            report = environment.grade(task, rollout["state"], trace)
            binaries.append(report.reward)
            from reliquary.environment.agentic.envs.envscaler_tools_v1.environment import _state_of
            final = _state_of(rollout["state"].instance)
            hit = sum(
                1 for src in required
                if _run_check(src, state0.initial, final) is True
            )
            fractions.append(hit / len(required) if required else 0.0)
        mean_f = sum(fractions) / len(fractions)
        var_f = sum((v - mean_f) ** 2 for v in fractions) / len(fractions)
        rows.append({
            "frac_sigma": var_f ** 0.5,
            "fractions": [round(v, 4) for v in fractions],
            "task_id": task.id, "env_id": task.metadata["family"],
            "required": len(required),
            "k_binary": sum(1 for v in binaries if v >= 1.0),
            "frac_mean": sum(fractions) / len(fractions),
            "frac_best": max(fractions),
            "invalid": sum(1 for r in replicas if r["invalid"]),
            "unreadable": sum(r["unreadable"] for r in replicas),
            "strict_turns": sum(r["strict_turns"] for r in replicas),
            "strict_ok": sum(r["strict_ok"] for r in replicas),
            "strict_alive": sum(1 for r in replicas if r["strict_alive"]),
            "finished": sum(1 for r in replicas if r["reason"] == "finished"),
            "turns_median": sorted(r["turns"] for r in replicas)[len(replicas) // 2],
            "tokens_median": sorted(r["tokens"] for r in replicas)[len(replicas) // 2],
        })

    low, high = band_bounds(ROLLOUTS, SIGMA_MIN)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump({"model": args.model, "band": [low, high],
               "contract": args.contract, "rows": rows},
                  handle, indent=1)

    n = len(rows)
    total = n * ROLLOUTS
    print(f"\nband = {low}..{high} of {ROLLOUTS}\n")
    print(f"tasks                 {n}")
    print(f"pass@1 (binary)       {sum(r['k_binary'] for r in rows) / total:.3f}")
    print(f"band  (binary)        {sum(low <= r['k_binary'] <= high for r in rows) / n:.1%}")
    print(f"k=0   (binary)        {sum(r['k_binary'] == 0 for r in rows) / n:.1%}")
    print(f"fractional mean       {sum(r['frac_mean'] for r in rows) / n:.3f}")
    print(f"fractional best-of-16 {sum(r['frac_best'] for r in rows) / n:.3f}")
    # The band on fractional reward is the question binary cannot answer: a
    # group whose rollouts all score zero has sigma zero and is filtered,
    # however partial the credit.
    print(f"fractional sigma>={SIGMA_MIN}  "
          f"{sum(r['frac_sigma'] >= SIGMA_MIN for r in rows) / n:.1%}"
          "   (groups the gate would keep)")
    print(f"fractional sigma>0    {sum(r['frac_sigma'] > 0 for r in rows) / n:.1%}"
          "   (groups with any spread at all)")
    print(f"median group sigma    "
          f"{sorted(r['frac_sigma'] for r in rows)[n // 2]:.3f}")
    strict_turns = sum(r["strict_turns"] for r in rows)
    print(f"episodes killed by errors "
          f"{sum(r['invalid'] for r in rows) / total:.1%}"
          f"   (budget {MAX_TOOL_ERRORS})")
    print(f"turns unreadable      "
          f"{sum(r['unreadable'] for r in rows) / max(strict_turns, 1):.1%}"
          f"   under --contract {args.contract}")
    print(f"turns strict-parsed   {sum(r['strict_ok'] for r in rows) / max(strict_turns, 1):.1%}"
          f"   of {strict_turns} turns")
    print(f"rollouts strict-clean {sum(r['strict_alive'] for r in rows) / total:.1%}"
          "   (every turn bare JSON)")
    print(f"finished explicitly   {sum(r['finished'] for r in rows) / total:.1%}")
    print(f"turns  median         {sorted(r['turns_median'] for r in rows)[n // 2]}")
    print(f"tokens median         {sorted(r['tokens_median'] for r in rows)[n // 2]}")
    print(f"required checks med   {sorted(r['required'] for r in rows)[n // 2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
