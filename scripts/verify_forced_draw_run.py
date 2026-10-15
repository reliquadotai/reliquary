#!/usr/bin/env python
"""Check a generated run with the validator's own forced-draw path.

Reads ``RUN/rollouts.pt`` (a list of dicts with ``prompt_index``, ``rollout_index``,
``prompt_ids``, ``tokens``, as offpolicy-lab ``measure.py gen`` / ``desync_probe_reliquary.py``
write them) and ``RUN/meta.json`` (``seed.randomness``, ``seed.checkpoint_hash``), runs the
proof forward of ``reliquary.validator.verifier`` (``forward_single_layer`` + ``_LazyLogitRows``,
fp32 head when the active profile says so) and its seed-consistency diagnostics under the
active profile's draw, then applies the production group/rollout floors.

    RELIQUARY_PROTOCOL_PROFILE=teutonic-9b-reliquary-suite-v10-gumbel-dev1 \\
      python scripts/verify_forced_draw_run.py --run RUN --model /root/opl/models/m9 [--fp32-head]

Writes ``RUN/verify_<draw>_<head>.json``. ``--fp32-head`` forces the fp32 head whatever the
profile (the contract field ``sampling.lm_head_fp32``).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fp32-head", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM

    from reliquary import constants as C
    from reliquary.environment.forced_sampling import u_at
    from reliquary.shared.forward import forward_single_layer
    from reliquary.validator import verifier

    if a.fp32_head:
        verifier.lm_head_fp32 = lambda: True  # type: ignore[assignment]
    fp32 = verifier.lm_head_fp32()
    settings = verifier.draw_settings()
    meta = json.load(open(os.path.join(a.run, "meta.json")))
    randomness, checkpoint = meta["seed"]["randomness"], meta["seed"]["checkpoint_hash"]
    rollouts = torch.load(os.path.join(a.run, "rollouts.pt"), weights_only=False)
    if a.limit:
        rollouts = rollouts[: a.limit]
    try:
        import flash_attn  # noqa: F401
        attn = "flash_attention_2"
    except ImportError:
        attn = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 attn_implementation=attn).to(a.device).eval()
    lm_head = model.lm_head
    vocab = verifier._lm_head_vocab_size(lm_head)

    per_rollout, misses = [], []
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    started = time.time()
    for ro in rollouts:
        ro = ro if isinstance(ro, dict) else vars(ro)
        prompt, tokens = list(ro["prompt_ids"]), list(ro["tokens"])
        if not tokens:
            continue
        ids = torch.tensor([prompt + tokens], device=a.device)
        with torch.no_grad():
            hidden, _ = forward_single_layer(model, ids, None, -1, materialize_logits=False)
        rows = verifier._LazyLogitRows(hidden[0], lm_head, vocab, fp32_head=fp32)
        P, n = len(prompt), len(tokens)
        us = [u_at(randomness, int(ro["prompt_index"]), checkpoint, int(ro["rollout_index"]), j)
              for j in range(n)]
        d = verifier._gpu_seed_consistency_diagnostics(
            rows, tokens, us, position_offsets=list(range(n)),
            logit_positions=[P - 1 + j for j in range(n)],
        )
        groups[int(ro["prompt_index"])].append((d.n_stochastic, d.n_exact_match))
        per_rollout.append(dict(prompt=int(ro["prompt_index"]), rollout=int(ro["rollout_index"]),
                                n=n, n_stochastic=d.n_stochastic, n_match=d.n_exact_match,
                                n_hard=d.n_hard_mismatch, max_miss=d.max_cdf_miss,
                                miss_gt_0_01=d.n_miss_gt_0_01, miss_gt_0_05=d.n_miss_gt_0_05,
                                miss_gt_0_10=d.n_miss_gt_0_10))
        misses.append(d.max_cdf_miss)

    from reliquary.validator.batcher import _forced_seed_rollout_reject, _forced_seed_verdict

    stoch = sum(r["n_stochastic"] for r in per_rollout)
    match = sum(r["n_match"] for r in per_rollout)
    rates = [r["n_match"] / r["n_stochastic"] for r in per_rollout
             if r["n_stochastic"] >= C.FORCED_SEED_ROLLOUT_MIN_STOCH]
    rejected = [p for p, rs in groups.items()
                if _forced_seed_verdict(sum(s for s, _ in rs), sum(m for _, m in rs), True)
                or _forced_seed_rollout_reject(rs, True)]
    summary = dict(
        draw=settings.draw, fp32_head=fp32, profile=C.PROTOCOL_PROFILE_ID,
        gpu=torch.cuda.get_device_name(0) if a.device.startswith("cuda") else a.device, torch=torch.__version__, seconds=time.time() - started,
        rollouts=len(per_rollout), groups=len(groups), stochastic=stoch,
        match=match / max(1, stoch), rollout_min=min(rates) if rates else None,
        rollouts_below_0_95=sum(r < 0.95 for r in rates), rollouts_below_0_80=sum(r < 0.80 for r in rates),
        groups_rejected=len(rejected), rejected_prompts=sorted(rejected),
        hard_mismatch_at_epsilon=sum(r["n_hard"] for r in per_rollout),
        boundary_epsilon=settings.boundary_epsilon,
        mismatch_miss_buckets={k: sum(r[k] for r in per_rollout)
                               for k in ("miss_gt_0_01", "miss_gt_0_05", "miss_gt_0_10")},
    )
    out = os.path.join(a.run, f"verify_{settings.draw}_{'fp32' if fp32 else 'bf16'}.json")
    json.dump(dict(summary=summary, per_rollout=per_rollout), open(out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
