#!/usr/bin/env python3
"""Measure the sigma-gate band of ``reliquarylogic_v1`` per family.

The band — groups whose sigma clears ``SIGMA_MIN``, which at 16 rollouts
and 0.24 means 1 to 15 successes — is the only part of a corpus that
produces gradient. The bound is computed, not assumed.
Everything else is valued 0 by the auction and filtered by the gate.

Format and capability are reported separately on purpose: a family can score
zero because the model cannot solve it, or because a base model cannot emit
the answer envelope at all. Those call for opposite fixes.
"""

from __future__ import annotations

import argparse
import collections
import json
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--prompts-per-family", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--suite-seed", type=int, default=1337)
    parser.add_argument("--gen-seed", type=int, default=7)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dump-samples", type=int, default=0,
                        help="raw completions per family, for diagnosis")
    args = parser.parse_args()

    from reliquary.environment.logic_tasks import (
        VIRTUAL_LENGTH, active_families, generate_logic_task,
    )
    from reliquary.environment.reliquarylogic import ReliquaryLogicEnvironment
    from reliquary.environment.structured_output import extract_json_answer

    environment = ReliquaryLogicEnvironment()

    # Deterministic per-family sample drawn from the whole index space.
    import random

    rng = random.Random(args.suite_seed)
    wanted = args.prompts_per_family
    by_family: dict[str, list[int]] = collections.defaultdict(list)
    scanned = 0
    while scanned < 400_000 and any(
        len(v) < wanted for v in by_family.values()
    ) or not by_family:
        index = rng.randrange(VIRTUAL_LENGTH)
        family = generate_logic_task(index).family
        if len(by_family[family]) < wanted:
            by_family[family].append(index)
        scanned += 1
        if scanned > 2000 and all(
            len(v) >= wanted for v in by_family.values()
        ) and len(by_family) >= 6:
            break

    order = sorted(by_family)
    problems = [
        (family, index, environment.get_problem(index))
        for family in order
        for index in by_family[family]
    ]
    roster = active_families()
    print(f"{len(problems)} prompts over {len(order)} families", flush=True)
    print(f"roster: {','.join(roster)}", flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        revision=args.revision,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_new_tokens + 2048,
        seed=args.gen_seed,
    )
    # Protocol sampling: T=1.0, top_p=1.0, top_k disabled, raw prompt.
    sampling = SamplingParams(
        n=ROLLOUTS, temperature=1.0, top_p=1.0, top_k=-1,
        max_tokens=args.max_new_tokens, seed=args.gen_seed,
    )
    started = time.time()
    outputs = llm.generate(
        [problem["prompt"] for _family, _index, problem in problems], sampling
    )
    print(f"generation {time.time() - started:.0f}s", flush=True)

    low, high = band_bounds(ROLLOUTS, SIGMA_MIN)
    rows = []
    samples: dict[str, list[str]] = collections.defaultdict(list)
    for (family, index, problem), output in zip(problems, outputs):
        rewards, parsed, finished, lengths = [], 0, 0, []
        for completion in output.outputs:
            text = completion.text
            rewards.append(environment.compute_reward(problem, text))
            lengths.append(len(completion.token_ids))
            finished += int(completion.finish_reason == "stop")
            ok = False
            try:
                answer = extract_json_answer(text)
                ok = set(answer) == {"result"}
                parsed += int(ok)
            except Exception:
                pass
            # Keep unparsable completions: they are what needs diagnosing.
            if not ok and len(samples[family]) < args.dump_samples:
                samples[family].append(text[:400])
        successes = sum(1 for value in rewards if value >= 1.0)
        rows.append({
            "family": family,
            "index": index,
            "difficulty": problem["difficulty"],
            "k": successes,
            "in_band": low <= successes <= high,
            "parsed": parsed,
            "finished": finished,
            "median_tokens": sorted(lengths)[len(lengths) // 2],
        })

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump({
            "model": args.model,
            "revision": args.revision,
            # Toggling a family remaps the index space, so a measurement is
            # only comparable to another taken under the same roster.
            "roster": list(roster),
            "rollouts": ROLLOUTS,
            "sigma_min": SIGMA_MIN,
            "band": [low, high],
            "max_new_tokens": args.max_new_tokens,
            "rows": rows,
            "unparsable_samples": {k: v for k, v in samples.items()},
        }, handle, indent=1)

    print(f"\nband = {low}..{high} of {ROLLOUTS} successes\n")
    header = (
        f"{'family':<22}{'n':>4}{'pass@1':>8}{'BAND':>7}"
        f"{'k=0':>7}{'k=16':>7}{'json':>7}{'eos':>7}{'tok':>7}"
    )
    print(header)
    print("-" * len(header))
    for family in order:
        subset = [row for row in rows if row["family"] == family]
        n = len(subset)
        print(
            f"{family:<22}{n:>4}"
            f"{sum(r['k'] for r in subset) / (n * ROLLOUTS):>8.3f}"
            f"{sum(r['in_band'] for r in subset) / n:>7.1%}"
            f"{sum(r['k'] == 0 for r in subset) / n:>7.1%}"
            f"{sum(r['k'] == ROLLOUTS for r in subset) / n:>7.1%}"
            f"{sum(r['parsed'] for r in subset) / (n * ROLLOUTS):>7.1%}"
            f"{sum(r['finished'] for r in subset) / (n * ROLLOUTS):>7.1%}"
            f"{sorted(r['median_tokens'] for r in subset)[n // 2]:>7}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
