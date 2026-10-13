import copy

import pytest
import torch

from tests.unit.trainer_dist_helpers import run_ranks, tiny_qwen35


def test_backend_selection():
    from reliquary.trainer.distributed import trainer_backend

    assert trainer_backend({}) == "single"
    assert trainer_backend({"RELIQUARY_TRAINER_BACKEND": "FSDP2"}) == "fsdp2"
    with pytest.raises(ValueError):
        trainer_backend({"RELIQUARY_TRAINER_BACKEND": "deepspeed"})


def test_assign_bins_balances_and_is_deterministic():
    from reliquary.trainer.distributed import assign_bins

    costs = [100, 90, 50, 40, 10, 5]
    split = assign_bins(costs, 2)
    assert split == assign_bins(costs, 2)
    assert sorted(i for part in split for i in part) == list(range(6))
    loads = [sum(costs[i] for i in part) for part in split]
    assert max(loads) - min(loads) <= max(costs)
    assert assign_bins([7], 3) == [[0], [], []]


def _grad_equivalence():
    from reliquary.trainer.distributed import init_distributed, shard_model
    from reliquary.validator.training import _batched_completion_logprobs

    ctx = init_distributed(device_type="cpu")
    ref = tiny_qwen35()
    ref.gradient_checkpointing_enable()
    model = shard_model(copy.deepcopy(ref), ctx, param_dtype=torch.float32)
    g = torch.Generator().manual_seed(1)
    batches = [torch.randint(0, 300, (2, 600), generator=g) for _ in range(ctx.world_size)]

    def loss(m, x):
        lp, _ = _batched_completion_logprobs(m, x, torch.ones_like(x), [10, 10], [600, 600])
        return lp.sum()

    model.train(); ref.train()
    (loss(model, batches[ctx.rank]) * model.reliquary_loss_scale).backward()
    for x in batches:
        loss(ref, x).backward()
    worst = 0.0
    for (name, p), (_, q) in zip(model.named_parameters(), ref.named_parameters()):
        full = p.grad.full_tensor()
        worst = max(worst, float((full - q.grad).abs().max() / (q.grad.abs().max() + 1e-6)))
    return worst


def test_sharded_grads_equal_single_process_sum():
    results = run_ranks(_grad_equivalence, 2)
    assert max(results.values()) < 1e-4


def _gather_roundtrip():
    from reliquary.trainer.distributed import gather_full_state_dict, init_distributed, shard_model

    ctx = init_distributed(device_type="cpu")
    ref = tiny_qwen35()
    model = shard_model(copy.deepcopy(ref).to(torch.bfloat16), ctx)
    sd = gather_full_state_dict(model, ctx)
    if not ctx.is_main:
        return sd is None
    want = {k: v.to(torch.bfloat16) for k, v in ref.state_dict().items()}
    return set(sd) == set(want) and all(torch.equal(sd[k], want[k]) for k in want) and \
        all(v.dtype == torch.bfloat16 and v.device.type == "cpu" for v in sd.values())


def test_gather_full_state_dict_roundtrip():
    assert all(run_ranks(_gather_roundtrip, 2).values())


def _reduce_stats():
    from reliquary.trainer.distributed import all_reduce_stats, init_distributed

    ctx = init_distributed(device_type="cpu")
    stats = {"token_count": 10 * (ctx.rank + 1), "max": float(ctx.rank), "weighted_ppo": 0.5}
    all_reduce_stats(stats, ctx, max_keys=frozenset({"max"}))
    return stats


def test_all_reduce_stats_sums_and_maxes():
    out = run_ranks(_reduce_stats, 2)
    assert out[0] == out[1] == {"token_count": 30, "max": 1.0, "weighted_ppo": 1.0}


def test_fused_head_matches_the_chunk_loop():
    """head(hidden, targets) == the plain chunked path, values and gradients."""
    from reliquary.trainer.distributed import FusedLogprobHead
    from reliquary.validator.training import _selected_logprobs_from_hidden

    torch.manual_seed(0)
    plain = torch.nn.Linear(16, 50, bias=False)
    fused = copy.deepcopy(plain)
    fused.__class__ = FusedLogprobHead
    hidden = torch.randn(1300, 16, requires_grad=True)
    targets = torch.randint(0, 50, (1300,))
    a = _selected_logprobs_from_hidden(hidden, targets, plain, chunk=512)
    a.sum().backward()
    ga, gwa = hidden.grad.clone(), plain.weight.grad.clone()
    hidden.grad = None
    b = _selected_logprobs_from_hidden(hidden, targets, fused, chunk=512)
    b.sum().backward()
    assert torch.equal(a, b)
    assert torch.equal(ga, hidden.grad) and torch.equal(gwa, fused.weight.grad)
    assert torch.equal(fused(hidden), plain(hidden))


def test_shard_model_refuses_a_tied_head():
    from reliquary.trainer.distributed import DistContext, shard_model

    model = tiny_qwen35()
    model.lm_head.weight = model.model.embed_tokens.weight
    with pytest.raises(ValueError, match="untied"):
        shard_model(model, DistContext(0, 1, torch.device("cpu"), None))
