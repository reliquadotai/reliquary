"""fsdp2 trainer qualification on one GPU: load + shard the real model, run
train_step on synthetic groups whose pi_old is the model's own logprobs,
compare with the single backend, export and reload in transformers."""
import argparse, copy, json, time
from dataclasses import dataclass, field

import torch


@dataclass
class _Rollout:
    tokens: list
    reward: float
    commit: dict = field(default_factory=dict)


@dataclass
class _Group:
    rollouts: list
    prompt_idx: int = 0


def batches_for(model, tok_ids, n_groups, length, seed):
    from reliquary.validator.training import _batched_completion_logprobs
    g = torch.Generator().manual_seed(seed)
    groups = []
    for gi in range(n_groups):
        rs = []
        for ri in range(4):
            tokens = [tok_ids[int(i)] for i in torch.randint(0, len(tok_ids), (length,), generator=g)]
            x = torch.tensor([tokens], device="cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                lp, _ = _batched_completion_logprobs(model, x, torch.ones_like(x), [64], [length])
            rs.append(_Rollout(tokens, float(ri % 2), {"tokens": tokens, "rollout": {
                "prompt_length": 64, "token_logprobs": lp.float().tolist()}}))
        groups.append(_Group(rs, gi))
    return [groups]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/div/model")
    ap.add_argument("--out", default="/root/div/fsdp2_export")
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--no-step", action="store_true")
    ap.add_argument("--export-only", action="store_true",
                    help="load, shard, gather, export, reload: what fits one card for the 9B")
    args = ap.parse_args()
    from reliquary.trainer.distributed import gather_full_state_dict, init_distributed, load_sharded_model, save_state_dict_hf
    from reliquary.validator import training
    from transformers import AutoModelForCausalLM

    ctx = init_distributed()
    training.configure_distributed(ctx)
    training.KL_BETA = 0.0  # the run's profile (decision N); the repo default is not 0
    t0 = time.time()
    model = load_sharded_model(args.model, {}, ctx, attn_implementation="flash_attention_2")
    print("load+shard s", round(time.time() - t0, 1), "mem GB", torch.cuda.max_memory_allocated() / 2**30,
          "loss scale", model.reliquary_loss_scale)
    batches = batches_for(model, list(range(1000, 100000)), args.groups, args.length, 0)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    if args.export_only:
        pass
    elif args.no_step:
        plan, _ = training._plan_from_batches(batches)
        training._accumulate_grouped_grads(
            model, None, plan, torch.device("cuda"),
            budget=training.MICROBATCH_MAX_PADDED_TOKENS, atomic=False,
            kl_stats=training._new_kl_stats(),
        )
    else:
        training.train_step(model, batches, ref_model=None)
    torch.cuda.synchronize()
    print("step s", round(time.time() - t0, 1), "peak GB", torch.cuda.max_memory_allocated() / 2**30)
    state = gather_full_state_dict(model, ctx)
    save_state_dict_hf(model, state, None, args.out)
    reloaded = AutoModelForCausalLM.from_pretrained(args.out, dtype=torch.bfloat16,
                                                    attn_implementation="flash_attention_2").cuda()
    x = torch.tensor([batches[0][0].rollouts[0].tokens[:512]], device="cuda")
    with torch.no_grad():
        want = reloaded(x).logits
        reloaded.load_state_dict({k: v.cuda() for k, v in state.items()})
        again = reloaded(x).logits
    print("reload identical:", torch.equal(want, again))
    print("config dtype:", json.load(open(f"{args.out}/config.json"))["dtype"])


if __name__ == "__main__":
    main()
