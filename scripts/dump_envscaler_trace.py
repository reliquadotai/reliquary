#!/usr/bin/env python3
"""Dump complete EnvScaler episodes: every turn, verbatim, with its grading.

The band harness answers whether a corpus produces gradient. It cannot show
*what* the model does, and a reward of 0.424 tells you nothing about whether
the model reasoned its way to a goal or stumbled into a check. This writes
whole transcripts — raw completion, parsed action, observation, and the
per-check verdict at the end — so the behaviour can be read rather than
inferred.
"""

from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, "scripts")

from eval_envscaler_band import (  # noqa: E402
    initial_text,
    observation_text,
    parse_action,
    qwen_action,
    qwen_initial_text,
    qwen_observation_text,
    strict_action,
)
from reliquary.environment.agentic.types import (  # noqa: E402
    AssistantAction,
    EpisodeTrace,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tasks", default="15",
                        help="comma-separated corpus indices")
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-action-tokens", type=int, default=2048)
    parser.add_argument("--format", dest="fmt",
                        choices=("reliquary", "qwen"), default="reliquary")
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--prose", choices=("terminates", "retries"),
                        default="retries",
                        help="what an unreadable turn means; must match the "
                             "configuration whose numbers the traces "
                             "illustrate, or they show a different regime")
    parser.add_argument("--gen-seed", type=int, default=7)
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from reliquary.environment.agentic.envs.envscaler_tools_v1.environment import (
        EnvScalerToolsEnvironment, _PROSE_TOOL, _run_check, _state_of,
    )
    from vllm import LLM, SamplingParams

    environment = EnvScalerToolsEnvironment()
    qwen = args.fmt == "qwen"
    render_initial = (
        (lambda t: qwen_initial_text(t, args.no_think)) if qwen else initial_text
    )
    render_observation = (
        (lambda e: qwen_observation_text(e, args.no_think)) if qwen
        else observation_text
    )

    llm = LLM(
        model=args.model, revision=args.revision, dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem, max_model_len=32768,
        seed=args.gen_seed,
    )
    sampling = SamplingParams(
        n=1, temperature=1.0, top_p=1.0, top_k=-1,
        max_tokens=args.max_action_tokens, seed=args.gen_seed,
        stop=(["<|im_end|>", "</tool_call>"] if qwen else
              ["<|reliquary_end|>", "<|reliquary_user|>", "<|reliquary_tool|>",
               "<|reliquary_assistant|>", "<|reliquary_system|>"]),
    )

    indices = [int(v) for v in args.tasks.split(",")]
    live = []
    for index in indices:
        task = environment.get_task(index)
        for replica in range(args.rollouts):
            state = environment.reset(task, seed=replica).state
            live.append({
                "index": index, "task": task, "replica": replica,
                "state": state, "text": render_initial(task),
                "done": False, "reason": None, "turns": [],
            })

    for _ in range(args.max_turns):
        pending = [r for r in live if not r["done"]]
        if not pending:
            break
        outputs = llm.generate([r["text"] for r in pending], sampling)
        for rollout, output in zip(pending, outputs):
            completion = output.outputs[0]
            raw = completion.text
            if qwen:
                parsed = qwen_action(raw)
                action = (
                    AssistantAction(kind="tool", tool=parsed[1],
                                    arguments=parsed[2])
                    if parsed else None
                )
            else:
                action = strict_action(raw)
                how = "strict"
                if action is None:
                    parsed = parse_action(raw)
                    if parsed is not None:
                        kind, name, payload = parsed
                        try:
                            action = (
                                AssistantAction(kind="final", content=payload)
                                if kind == "final"
                                else AssistantAction(kind="tool", tool=name,
                                                     arguments=payload)
                            )
                            how = "lenient"
                        except (TypeError, ValueError):
                            action = None
            readable = action is not None
            if action is None:
                action = AssistantAction.tool_call(
                    _PROSE_TOOL if args.prose == "terminates"
                    else "__unreadable__"
                )

            errors_before = rollout["state"].invalid_actions
            rollout["text"] += raw
            result = environment.step(rollout["task"], rollout["state"], action)
            rollout["text"] += render_observation(result.events)
            failed = rollout["state"].invalid_actions > errors_before

            rollout["turns"].append({
                "raw": raw,
                "tokens": len(completion.token_ids),
                "finish_reason": completion.finish_reason,
                "readable": readable,
                "read_as": (None if not readable else
                            ("qwen_tool_call" if qwen else how)),
                "action": action.to_wire() if readable else None,
                "observation": result.events[0].content if result.events else None,
                "call_failed": failed,
                "done": result.done,
                "termination_reason": result.termination_reason,
            })
            if result.done:
                rollout["done"] = True
                rollout["reason"] = result.termination_reason

    records = []
    for rollout in live:
        task = rollout["task"]
        reason = rollout["reason"] or "turn_limit"
        trace = EpisodeTrace(
            schema="reliquary/episode/v1", environment=environment.name,
            task_id=task.id, seed=rollout["replica"], events=(), actions=(),
            tokens=(), assistant_spans=(), observation_digests=(),
            termination_reason=reason,
        )
        report = environment.grade(task, rollout["state"], trace)
        fresh = environment.reset(task, seed=0).state
        final = _state_of(rollout["state"].instance)
        checks = []
        for entry in task.private["checks"]:
            source = entry["check_func"]
            checks.append({
                "item": str(entry.get("check_item", ""))[:300],
                "true_at_reset": _run_check(source, fresh.initial, fresh.initial) is True,
                "true_at_end": _run_check(source, fresh.initial, final) is True,
            })
        records.append({
            "corpus_index": rollout["index"],
            "task_id": task.id,
            "world": task.metadata["family"],
            "replica": rollout["replica"],
            "prompt": task.prompt,
            "tools": [
                {"name": s.name, "description": s.description,
                 "parameters": dict(s.parameters)}
                for s in task.tools
            ],
            "termination_reason": reason,
            "reward": report.reward,
            "success": report.success,
            "invalid_actions": rollout["state"].invalid_actions,
            "checks": checks,
            "turns": rollout["turns"],
        })

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump({"model": args.model, "format": args.fmt,
                   "no_think": args.no_think, "prose": args.prose,
                   "records": records},
                  handle, indent=1, ensure_ascii=False)

    print(f"\n{len(records)} episodes -> {args.output}")
    for record in records:
        flipped = sum(1 for c in record["checks"]
                      if c["true_at_end"] and not c["true_at_reset"])
        todo = sum(1 for c in record["checks"] if not c["true_at_reset"])
        print(f"  {record['task_id']} r{record['replica']}  "
              f"reward {record['reward']:.3f}  "
              f"flipped {flipped}/{todo}  "
              f"turns {len(record['turns'])}  "
              f"readable {sum(1 for t in record['turns'] if t['readable'])}"
              f"/{len(record['turns'])}  {record['termination_reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
