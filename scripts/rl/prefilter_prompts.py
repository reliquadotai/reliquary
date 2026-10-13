#!/usr/bin/env python3
"""Pre-filter an environment's RL prompts by how often the policy solves them.

For each prompt of the environment's RL rows this samples ``k`` completions
from the policy (vLLM on GPU, or transformers for a CPU dry run), grades them
the way admission does, and writes:

* ``passes.jsonl``: one line per prompt — rewards, how many were correct, how
  many ran to the token cap without terminating;
* ``keep.json``: the prompts whose correct count falls in the keep band
  (default 1 to 6 of 8), sorted, with the digest of the list;
* ``summary.json``: the share of prompts in each correct-count bucket, the
  truncation rate over all rollouts, and the kept share.

Prompts are rendered exactly as the miner renders them (``encode_prompt`` with
the environment's reasoning switch under the selected protocol profile), the
sampling parameters are the profile's, and the cap is the environment's. A
completion is graded on the text admission decodes: its tokens, the
terminator included, decoded with special tokens kept, then scored through the
environment's registered scorer with the reward materials admission would
build. A code environment therefore needs the grading service admission uses.

``passes.jsonl`` is appended as prompts finish, and a rerun skips the prompts
already in it, so a long GPU run can be resumed.

The protocol has no prompt allowlist: a window offers one contiguous slice of
each environment. ``keep.json`` is therefore an input for miners (who choose
which prompts to try) or for a future eligibility mechanism, not something the
validator reads today.

CPU dry run (tiny model, a few prompts)::

    RELIQUARY_PROTOCOL_PROFILE=teutonic-9b-reliquary-suite-v9-dev1 \\
    python scripts/rl/prefilter_prompts.py --backend hf --model sshleifer/tiny-gpt2 \\
        --env reliquary_dapo_math_v1 --count 4 --max-new-tokens 16 --output-dir /tmp/pf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

DEFAULT_PROFILE = "teutonic-9b-reliquary-suite-v9-dev1"
CORRECT_AT = 1.0 - 1e-9


def _json_line(row: dict) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--env", required=True, help="registered single-turn environment")
    parser.add_argument("--backend", choices=("vllm", "hf"), default="vllm")
    parser.add_argument("--k", type=int, default=8, help="completions per prompt")
    parser.add_argument("--keep-min", type=int, default=1)
    parser.add_argument("--keep-max", type=int, default=6)
    parser.add_argument("--start", type=int, default=None,
                        help="first row (default: the env's RL range)")
    parser.add_argument("--end", type=int, default=None,
                        help="row past the last (default: the env's RL range)")
    parser.add_argument("--count", type=int, default=None,
                        help="prompts to measure, drawn without replacement (default: every row)")
    parser.add_argument("--sample-seed", type=int, default=1337)
    parser.add_argument("--gen-seed", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="default: the environment's protocol cap")
    parser.add_argument("--prompts-per-call", type=int, default=64,
                        help="prompts handed to the engine at once")
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--gpu-mem", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if args.k < 1 or not 0 <= args.keep_min <= args.keep_max <= args.k:
        parser.error("need 0 <= --keep-min <= --keep-max <= --k")
    return args


def select_indices(start: int, end: int, count: int | None, seed: int) -> list[int]:
    if count is None or count >= end - start:
        return list(range(start, end))
    return sorted(random.Random(seed).sample(range(start, end), count))


def bucket_summary(rows: Sequence[dict], k: int, keep_min: int, keep_max: int) -> dict:
    total = len(rows)
    buckets = {str(n): 0 for n in range(k + 1)}
    rollouts = truncated = 0
    for row in rows:
        buckets[str(row["correct"])] += 1
        rollouts += len(row["rewards"])
        truncated += row["truncated"]
    kept = sum(1 for row in rows if keep_min <= row["correct"] <= keep_max)
    return {
        "prompts": total,
        "k": k,
        "correct_count_share": {n: (c / total if total else 0.0) for n, c in buckets.items()},
        "correct_count_prompts": buckets,
        "truncation_rate": truncated / rollouts if rollouts else 0.0,
        "keep_band": [keep_min, keep_max],
        "kept": kept,
        "kept_share": kept / total if total else 0.0,
    }


def keep_list(rows: Sequence[dict], keep_min: int, keep_max: int) -> tuple[list[int], str]:
    kept = sorted(row["prompt_idx"] for row in rows if keep_min <= row["correct"] <= keep_max)
    digest = hashlib.sha256(json.dumps(kept, separators=(",", ":")).encode()).hexdigest()
    return kept, digest


class Grader:
    """Admission's single-turn grading of one environment."""

    def __init__(self, env_name: str, tokenizer: Any) -> None:
        from reliquary.environment.registry import get_environment_spec

        self.spec = get_environment_spec(env_name)
        if self.spec.interaction_mode != "single_turn":
            raise SystemExit(f"{env_name} is not a single-turn environment")
        self.env = self.spec.create()
        self.tokenizer = tokenizer

    def problem(self, index: int) -> dict:
        return self.env.get_problem(index)

    def text(self, completion_ids: Sequence[int]) -> str:
        # What admission grades: the completion tokens, terminator included,
        # decoded with special tokens kept.
        return self.tokenizer.decode(list(completion_ids), skip_special_tokens=False)

    def rewards(self, problem: dict, completions: Sequence[Sequence[int]]) -> list[float]:
        materials = None
        method = self.spec.reward_materializer_method
        if method:
            materials = getattr(self.env, method)(problem)
        texts = [self.text(ids) for ids in completions]
        return [float(r) for r in self.spec.score_many(problem, texts, materials)]


class HFSampler:
    """transformers ``generate``: slow, for a CPU dry run."""

    def __init__(self, args, eos_ids: set[int]) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision)
        self.model.eval()
        self.eos_ids = eos_ids

    def sample(self, prompts: list[list[int]], *, k: int, max_new_tokens: int, sampling,
               seed: int) -> list[list[tuple[list[int], bool]]]:
        torch = self.torch
        out = []
        for offset, prompt in enumerate(prompts):
            torch.manual_seed(seed + offset)
            kwargs = dict(do_sample=True, temperature=sampling.temperature, top_p=sampling.top_p,
                          max_new_tokens=max_new_tokens, num_return_sequences=k,
                          eos_token_id=sorted(self.eos_ids),
                          pad_token_id=sorted(self.eos_ids)[0])
            if sampling.top_k > 0:
                kwargs["top_k"] = sampling.top_k
            with torch.no_grad():
                ids = torch.tensor([prompt])
                generated = self.model.generate(ids, attention_mask=torch.ones_like(ids), **kwargs)
            rows = []
            for sequence in generated.tolist():
                completion = sequence[len(prompt):]
                cut = next((i for i, t in enumerate(completion) if t in self.eos_ids), None)
                if cut is None:
                    rows.append((completion, True))
                else:
                    rows.append((completion[:cut + 1], False))
            out.append(rows)
        return out


class VLLMSampler:
    def __init__(self, args, eos_ids: set[int]) -> None:
        from vllm import LLM

        kwargs = dict(model=args.model, revision=args.revision,
                      tensor_parallel_size=args.tensor_parallel,
                      gpu_memory_utilization=args.gpu_mem, seed=args.gen_seed)
        if args.max_model_len:
            kwargs["max_model_len"] = args.max_model_len
        self.llm = LLM(**kwargs)
        self.eos_ids = eos_ids

    def sample(self, prompts: list[list[int]], *, k: int, max_new_tokens: int, sampling,
               seed: int) -> list[list[tuple[list[int], bool]]]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        params = [SamplingParams(n=k, temperature=sampling.temperature, top_p=sampling.top_p,
                                 top_k=sampling.top_k if sampling.top_k > 0 else -1,
                                 max_tokens=max_new_tokens, seed=seed + offset,
                                 stop_token_ids=sorted(self.eos_ids), skip_special_tokens=False)
                  for offset in range(len(prompts))]
        results = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params)
        out = []
        for result in results:
            rows = []
            for completion in result.outputs:
                ids = list(completion.token_ids)
                truncated = completion.finish_reason == "length"
                if not truncated and (not ids or ids[-1] not in self.eos_ids):
                    # Keep the terminator the miner commits and admission decodes.
                    stop = completion.stop_reason
                    ids.append(stop if isinstance(stop, int) else min(self.eos_ids))
                rows.append((ids, truncated))
            out.append(rows)
        return out


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("RELIQUARY_PROTOCOL_PROFILE", DEFAULT_PROFILE)

    from transformers import AutoTokenizer

    from reliquary.constants import (
        ACTIVE_PROTOCOL_PROFILE, max_new_tokens_for_environment, thinking_for_environment,
    )
    from reliquary.protocol.rl_prompt_ranges import rl_prompt_range
    from reliquary.protocol.tokens import encode_prompt
    from reliquary.shared.modeling import resolve_eos_token_ids

    if args.env not in ACTIVE_PROTOCOL_PROFILE.environments:
        raise SystemExit(f"{args.env} is not declared by {ACTIVE_PROTOCOL_PROFILE.profile_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    grader = Grader(args.env, tokenizer)
    rl_start, rl_end = rl_prompt_range(args.env, len(grader.env))
    start = rl_start if args.start is None else args.start
    end = rl_end if args.end is None else args.end
    if not rl_start <= start < end <= rl_end:
        raise SystemExit(f"[{start}, {end}) is outside {args.env}'s RL rows [{rl_start}, {rl_end})")
    sampling = ACTIVE_PROTOCOL_PROFILE.sampling
    max_new_tokens = args.max_new_tokens or max_new_tokens_for_environment(args.env)
    thinking = thinking_for_environment(args.env)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    passes_path = out_dir / "passes.jsonl"
    done: dict[int, dict] = {}
    if passes_path.exists():
        for line in passes_path.read_text().splitlines():
            row = json.loads(line)
            if row["env"] == args.env and row["k"] == args.k:
                done[row["prompt_idx"]] = row
    indices = [i for i in select_indices(start, end, args.count, args.sample_seed) if i not in done]

    model_for_eos = None
    if args.backend == "hf":
        sampler = HFSampler(args, set())
        model_for_eos = sampler.model
    eos_ids = resolve_eos_token_ids(model_for_eos, tokenizer)
    if not eos_ids:
        raise SystemExit("no EOS token id: a completion could never terminate")
    sampler = VLLMSampler(args, eos_ids) if args.backend == "vllm" else sampler
    sampler.eos_ids = eos_ids

    started = time.monotonic()
    with passes_path.open("a") as sink:
        for chunk_start in range(0, len(indices), args.prompts_per_call):
            chunk = indices[chunk_start:chunk_start + args.prompts_per_call]
            problems = [grader.problem(i) for i in chunk]
            prompts = [encode_prompt(tokenizer, p["prompt"], thinking=thinking) for p in problems]
            samples = sampler.sample(prompts, k=args.k, max_new_tokens=max_new_tokens,
                                     sampling=sampling, seed=args.gen_seed + chunk[0])
            for index, problem, rollouts in zip(chunk, problems, samples):
                rewards = grader.rewards(problem, [ids for ids, _ in rollouts])
                row = {
                    "env": args.env, "prompt_idx": index, "k": args.k,
                    "rewards": rewards,
                    "correct": sum(1 for r in rewards if r >= CORRECT_AT),
                    "truncated": sum(1 for _, cut in rollouts if cut),
                    "completion_tokens": [len(ids) for ids, _ in rollouts],
                }
                sink.write(_json_line(row))
                done[index] = row
            sink.flush()
            print(f"{len(done)} prompts, {time.monotonic() - started:.0f}s", file=sys.stderr)

    rows = [done[i] for i in sorted(done) if start <= i < end]
    kept, digest = keep_list(rows, args.keep_min, args.keep_max)
    identity = {
        "env": args.env, "profile": ACTIVE_PROTOCOL_PROFILE.profile_id,
        "model": args.model, "revision": args.revision, "range": [start, end],
        "k": args.k, "max_new_tokens": max_new_tokens, "backend": args.backend,
    }
    (out_dir / "keep.json").write_text(json.dumps(
        {**identity, "keep_band": [args.keep_min, args.keep_max],
         "prompt_indices": kept, "prompt_indices_sha256": digest}, sort_keys=True, indent=1) + "\n")
    summary = {**identity, **bucket_summary(rows, args.k, args.keep_min, args.keep_max)}
    (out_dir / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=1) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
