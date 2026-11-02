"""Time the trainer's out-of-zone audit forward on one GPU: rollout by rollout (``rollout_diagnostics``) against the
packed group (``group_diagnostics``), on a synthetic group of the given rollout lengths, and count the rollouts whose
diagnostics agree.

    python scripts/bench_trainer_audit.py MODEL_DIR [--rollouts 16] [--prompt 600] [--lengths 2000,8000,30000]
        [--budget 32768]

The group's tokens are random (the audit's cost does not depend on them); the draw is the lease's, as in a real
audit. Run with the profile the run uses (RELIQUARY_PROTOCOL_PROFILE) so the head and the draw are the run's.
"""
from __future__ import annotations

import argparse
import random
import time


def main() -> None:
    import torch

    from reliquary.shared.modeling import load_text_generation_model
    from reliquary.trainer.audits import group_diagnostics, rollout_diagnostics
    from reliquary.validator.rl_buffer import SubmittedGroup

    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--rollouts", type=int, default=16)
    parser.add_argument("--prompt", type=int, default=600)
    parser.add_argument("--lengths", default="2000,4000,8000,16000")
    parser.add_argument("--budget", type=int, default=None)
    args = parser.parse_args()

    model = load_text_generation_model(args.model, torch_dtype=torch.bfloat16).to("cuda").eval()
    vocab = int(model.config.vocab_size)
    rng = random.Random(0)
    lengths = [int(x) for x in args.lengths.split(",")]
    completions = tuple(tuple(rng.randrange(vocab) for _ in range(lengths[i % len(lengths)]))
                        for i in range(args.rollouts))
    prompt = tuple(rng.randrange(vocab) for _ in range(args.prompt))
    group = SubmittedGroup("bench", "hotkey", "env", 1, 0, "ab" * 32, 0.0, (0.0,) * args.rollouts,
                           (False,) * args.rollouts, prompt, completions, (None,) * args.rollouts)

    def timed(fn):
        torch.cuda.synchronize()
        began = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        return value, time.perf_counter() - began

    one, t_one = timed(lambda: [rollout_diagnostics(model, group, i) for i in range(args.rollouts)])
    packed, t_packed = timed(lambda: group_diagnostics(model, group, token_budget=args.budget))
    agree = sum(a == b for a, b in zip(one, packed))
    tokens = sum(len(c) for c in completions) + args.prompt * args.rollouts
    print(f"rollout by rollout: {t_one:.2f} s; packed: {t_packed:.2f} s ({t_one / t_packed:.1f}x); "
          f"{tokens} tokens; diagnostics agree on {agree}/{args.rollouts} rollouts")


if __name__ == "__main__":
    main()
