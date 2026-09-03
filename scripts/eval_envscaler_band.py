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


def _iter_json_spans(text: str):
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
    for block in _iter_json_spans(text):
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


# --- Qwen's own tool-calling shape, for measuring what our frame costs ---
#
# Our renderer invents `<|reliquary_*|>` turn markers and asks for bare JSON.
# Qwen3 was post-trained on `<tool_call>` inside a ChatML frame, so those
# tokens are in the model's prior and ours are not. This is a measurement
# variant, not a protocol proposal: the manifest pins the real renderer.

_QWEN_SYSTEM = """<|im_start|>system
You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools}
</tools>

For each function call, return a json object with function name and \
arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call><|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""

_NO_THINK = "<think>\n\n</think>\n\n"


def qwen_initial_text(task, no_think: bool = False) -> str:
    tools = "\n".join(
        json.dumps({"type": "function", "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": dict(spec.parameters),
        }}, ensure_ascii=False)
        for spec in task.tools
    )
    text = _QWEN_SYSTEM.format(tools=tools, prompt=task.prompt)
    return text + _NO_THINK if no_think else text


def qwen_observation_text(events, no_think: bool = False) -> str:
    body = "".join(
        f"<tool_response>\n{event.content}\n</tool_response>\n"
        for event in events
    )
    text = (f"<|im_end|>\n<|im_start|>user\n{body}"
            f"<|im_end|>\n<|im_start|>assistant\n")
    return text + _NO_THINK if no_think else text


def qwen_action(text: str):
    """The call a turn settles on, in Qwen's `{"name","arguments"}` shape.

    Brace-balanced rather than regex-matched, and it keeps looking after a
    candidate fails to parse. A first version used
    ``<tool_call>\\s*(\\{.*?\\})\\s*(?:</tool_call>|$)`` and took only the
    first match: on a turn carrying two calls, the non-greedy span
    backtracked across both and parsed as neither. Measured on the dumped
    transcripts, 63% of the turns it called unreadable did contain a valid
    named object.
    """
    found = None
    for span in _iter_json_spans(text):
        try:
            value = json.loads(span)
        except Exception:
            continue
        if not isinstance(value, dict):
            continue
        name = value.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        args = value.get("arguments")
        found = ("tool", name, args if isinstance(args, dict) else {})
    return found


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
    parser.add_argument("--max-episode-tokens", type=int, default=16384,
                        help="the episode profile's budget; a rollout that "
                             "exceeds it ends, as runner.py would have it")
    parser.add_argument("--contract", choices=("strict", "lenient"),
                        default="strict",
                        help="how a turn is read into an action; strict is "
                             "what runner.py does today")
    parser.add_argument("--format", dest="fmt",
                        choices=("reliquary", "qwen"), default="reliquary",
                        help="prompt and action shape. 'qwen' is the "
                             "<tool_call> ChatML frame the model was actually "
                             "post-trained on, for measuring what our own "
                             "renderer costs")
    parser.add_argument("--no-think", action="store_true",
                        help="with --format qwen, pre-close the think block, "
                             "which is Qwen3's documented way to turn "
                             "reasoning off")
    parser.add_argument("--prose", choices=("terminates", "retries"),
                        default="terminates",
                        help="what an unreadable turn means. 'terminates' is "
                             "upstream: content with no tool call becomes "
                             "chat_with_user and ends the episode on the state "
                             "reached. 'retries' makes it a recoverable error, "
                             "which separates 'the base model rambles' from "
                             "'the environment cannot produce spread'")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from reliquary.environment.agentic.envs.envscaler_tools_v1.environment import (
        EnvScalerToolsEnvironment, _PROSE_TOOL,
    )
    globals()["_PROSE_TOOL"] = _PROSE_TOOL

    environment = EnvScalerToolsEnvironment()
    qwen = args.fmt == "qwen"
    if qwen:
        render_initial = lambda t: qwen_initial_text(t, args.no_think)
        render_observation = lambda e: qwen_observation_text(e, args.no_think)
    else:
        render_initial, render_observation = initial_text, observation_text
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model, revision=args.revision, dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        # Headroom over the episode budget so the budget check below is what
        # ends a long rollout, rather than vLLM raising mid-run.
        max_model_len=max(2 * args.max_episode_tokens, 16384),
        seed=args.gen_seed,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        n=1, temperature=1.0, top_p=1.0, top_k=-1,
        max_tokens=args.max_action_tokens, seed=args.gen_seed,
        stop=(["<|im_end|>", "</tool_call>"] if qwen else
              ["<|reliquary_end|>", "<|reliquary_user|>", "<|reliquary_tool|>",
               "<|reliquary_assistant|>", "<|reliquary_system|>"]),
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
                "text": render_initial(task), "done": False, "reason": None,
                "tokens": 0, "turns": 0,
                "strict_turns": 0, "strict_ok": 0, "strict_alive": True,
                "unreadable": 0, "tool_ok": 0, "tool_err": 0, "ended": 0,
            })

    started = time.time()
    for turn in range(args.max_turns):
        pending = []
        for rollout in (r for r in live if not r["done"]):
            if len(tokenizer.encode(rollout["text"])) > args.max_episode_tokens:
                rollout["done"] = True
                rollout["reason"] = "episode_token_limit"
                continue
            pending.append(rollout)
        if not pending:
            break
        prompts = [r["text"] for r in pending]
        outputs = llm.generate(prompts, sampling)
        for rollout, output in zip(pending, outputs):
            completion = output.outputs[0]
            rollout["tokens"] += len(completion.token_ids)
            rollout["turns"] += 1
            rollout["strict_turns"] += 1
            action = None
            if qwen:
                parsed = qwen_action(completion.text)
                rollout["strict_ok"] += int(parsed is not None)
                if parsed is None:
                    rollout["strict_alive"] = False
                else:
                    try:
                        action = AssistantAction(
                            kind="tool", tool=parsed[1], arguments=parsed[2])
                    except (TypeError, ValueError):
                        action = None
            else:
                strict = strict_action(completion.text)
                rollout["strict_ok"] += int(strict is not None)
                if strict is None:
                    rollout["strict_alive"] = False
                action = strict
            if action is None and not qwen and args.contract == "lenient":
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
                # Upstream routes content-without-a-tool-call to
                # chat_with_user, which terminates; the env recognises
                # runner.py's __invalid_action__ as the same thing. Under
                # --prose retries it is sent as a merely unknown tool, which
                # the env treats as recoverable.
                action = AssistantAction.tool_call(
                    "__invalid_action__" if args.prose == "terminates"
                    else "__unreadable__"
                )
                rollout["unreadable"] += 1
            rollout["text"] += completion.text
            # Every turn falls in exactly one of three buckets: unreadable,
            # readable but the call failed, readable and the world moved.
            # That is the line between a formatting problem and a competence
            # one, and it cannot be read off the invalid-turn rate alone.
            errors_before = rollout["state"].invalid_actions
            readable = (action.kind == "tool"
                        and action.tool not in (_PROSE_TOOL, "__unreadable__"))
            result = environment.step(rollout["task"], rollout["state"], action)
            if readable:
                if rollout["state"].invalid_actions > errors_before:
                    rollout["tool_err"] += 1
                else:
                    rollout["tool_ok"] += 1
            elif action.tool == _PROSE_TOOL:
                rollout["ended"] += 1
            rollout["text"] += render_observation(result.events)
            if result.done:
                rollout["done"] = True
                rollout["reason"] = result.termination_reason
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
            "tool_ok": sum(r["tool_ok"] for r in replicas),
            "tool_err": sum(r["tool_err"] for r in replicas),
            "strict_turns": sum(r["strict_turns"] for r in replicas),
            "strict_ok": sum(r["strict_ok"] for r in replicas),
            "strict_alive": sum(1 for r in replicas if r["strict_alive"]),
            "finished": sum(1 for r in replicas if r["reason"] == "finished"),
            "over_budget": sum(1 for r in replicas
                               if r["reason"] == "episode_token_limit"),
            "turns_median": sorted(r["turns"] for r in replicas)[len(replicas) // 2],
            "tokens_median": sorted(r["tokens"] for r in replicas)[len(replicas) // 2],
        })

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump({"model": args.model, "sigma_min": SIGMA_MIN,
                   "contract": args.contract, "prose": args.prose,
                   "format": args.fmt, "no_think": args.no_think,
                   "rows": rows},
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
          f"   --format {args.fmt} --contract {args.contract} "
          f"--prose {args.prose}")
    ok = sum(r["tool_ok"] for r in rows)
    err = sum(r["tool_err"] for r in rows)
    print(f"turns: readable+executed {ok / max(strict_turns, 1):.1%}"
          f" | readable+failed {err / max(strict_turns, 1):.1%}"
          f" | unreadable {sum(r['unreadable'] for r in rows) / max(strict_turns, 1):.1%}")
    print(f"of readable turns, executed "
          f"{ok / max(ok + err, 1):.1%}")
    print(f"turns with a readable action "
          f"{sum(r['strict_ok'] for r in rows) / max(strict_turns, 1):.1%}")
    print(f"explicit termination  {sum(r['finished'] for r in rows) / total:.1%}")
    print(f"over the 16k budget   {sum(r['over_budget'] for r in rows) / total:.1%}")
    print(f"turns  median         {sorted(r['turns_median'] for r in rows)[n // 2]}")
    print(f"tokens median         {sorted(r['tokens_median'] for r in rows)[n // 2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
