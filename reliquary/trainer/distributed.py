"""Multi-GPU support for the detached trainer: the FSDP2 backend.

Rank 0 runs the usual worker (journal, publication, control); the other ranks
follow the commands rank 0 broadcasts. Everything that knows about more than
one process lives here; train_step only asks for a rank split and reductions.
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.distributed as dist

BACKEND_ENV = "RELIQUARY_TRAINER_BACKEND"
_BACKENDS = ("single", "fsdp2")
# Rank 0 may wait hours for the next window while followers block on the
# command broadcast; only that group gets a timeout this long.
_CONTROL_TIMEOUT = datetime.timedelta(days=30)
COLLECTIVE_TIMEOUT_ENV = "RELIQUARY_TRAINER_COLLECTIVE_TIMEOUT_MINUTES"
# NCCL's own default. A collective waits only for the slowest rank to reach it,
# never for the work every rank does before it, so the first step's kernel
# compilation (4-6 min, run by all ranks at once) costs only the spread between
# ranks. Waits outside a step (windows, disk writes) are on the control group.
_DEFAULT_COLLECTIVE_TIMEOUT_MINUTES = 10


def trainer_backend(env: Mapping[str, str] = os.environ) -> str:
    value = str(env.get(BACKEND_ENV, "single")).strip().lower()
    if value not in _BACKENDS:
        raise ValueError(f"{BACKEND_ENV} must be one of {_BACKENDS}, got {value!r}")
    return value


def collective_timeout(env: Mapping[str, str] = os.environ) -> datetime.timedelta:
    raw = str(env.get(COLLECTIVE_TIMEOUT_ENV, "")).strip()
    minutes = int(raw, 10) if raw else _DEFAULT_COLLECTIVE_TIMEOUT_MINUTES
    if minutes < 1:
        raise ValueError(f"{COLLECTIVE_TIMEOUT_ENV} must be a positive number of minutes")
    return datetime.timedelta(minutes=minutes)


@dataclass(frozen=True)
class DistContext:
    rank: int
    world_size: int
    device: torch.device
    control_group: Any

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(*, device_type: str = "cuda") -> DistContext:
    """Join torchrun's process group (or a one-process group without torchrun)."""
    # Before the first CUDA allocation. The 9B on 4 x 80 GB sits at 70+ GB per rank; the default allocator
    # thrashes there: 2.5x slower steps than expandable segments (A100, 2026-10-16).
    for key, default in (("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", "29511"),
                         ("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0"),
                         ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")):
        os.environ.setdefault(key, default)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if device_type == "cuda":
        torch.cuda.set_device(local_rank)
        device, backend = torch.device("cuda", local_rank), "nccl"
    else:
        device, backend = torch.device("cpu"), "gloo"
    if not dist.is_initialized():
        # Bounded, so a rank that leaves a step early (or never enters it)
        # crashes torchrun instead of hanging the others. Measured 10-13:
        # without it, rank 0 sat over 15 min in an all-gather its followers
        # never joined, with no abort.
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size,
                                timeout=collective_timeout())
    control = dist.new_group(backend="gloo", timeout=_CONTROL_TIMEOUT)
    return DistContext(rank, world_size, device, control)


class FusedLogprobHead(torch.nn.Linear):
    """The LM head under FSDP2: ``head(hidden, targets)`` runs train_step's
    whole checkpointed chunk loop inside one module call.

    Called once per chunk instead, FSDP2 (torch 2.7) skips its hooks for the
    chunk recomputes issued during backward, so those chunks would read the
    sharded weight and their gradients would never be reduced. One call per
    micro-batch gathers the weight once for forward, once for backward, and
    reduces once. ``head(hidden)`` stays a plain projection. ``head(rows, audit=fn)`` hands ``fn`` the draw's
    projection over the gathered weight (``trainer.audits``: an out-of-zone audit's diagnostics, no gradient).
    """

    fused_selected_logprobs = True

    def forward(self, hidden, targets=None, draws=None, audit=None):
        if audit is not None:
            from reliquary.trainer.audits import head_projection

            return audit(head_projection(self.weight, self.bias))
        if targets is None:
            return super().forward(hidden)
        from reliquary.validator.training import _selected_logprobs_from_hidden

        # draws: the substitution loss's v* is picked here too, on the gathered weight.
        weight, bias = self.weight, self.bias
        return _selected_logprobs_from_hidden(
            hidden, targets, lambda rows: torch.nn.functional.linear(rows, weight, bias),
            draws=draws, weight=weight,
        )


def shard_model(model, ctx: DistContext, *, param_dtype=torch.bfloat16):
    """Shard every decoder layer, the LM head and the base, then the root.

    train_step calls the base model and the LM head directly (never the root
    forward), so each needs its own unit for its parameters to be gathered.
    The shards are cast to fp32 afterwards: they are the master weights, and
    the mixed-precision policy hands bf16 copies to the compute.
    """
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    base = getattr(model, model.base_model_prefix)
    if not hasattr(base, "layers"):
        raise ValueError(
            f"fsdp2 expects a text-only causal LM, got {type(model).__name__}"
        )
    embed = getattr(getattr(base, "embed_tokens", None), "weight", None)
    if embed is not None and model.lm_head.weight is embed:
        raise ValueError("fsdp2 needs an untied lm_head (tie_word_embeddings=false)")
    if type(model.lm_head) is not torch.nn.Linear:
        raise ValueError(f"fsdp2 expects an nn.Linear lm_head, got {type(model.lm_head).__name__}")
    model.lm_head.__class__ = FusedLogprobHead
    mesh = init_device_mesh(ctx.device.type, (ctx.world_size,))
    policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32)
    for layer in base.layers:
        fully_shard(layer, mesh=mesh, mp_policy=policy)
    fully_shard(model.lm_head, mesh=mesh, mp_policy=policy)
    fully_shard(base, mesh=mesh, mp_policy=policy, reshard_after_forward=False)
    fully_shard(model, mesh=mesh, mp_policy=policy)
    model.to(torch.float32)
    model.reliquary_loss_scale = _probe_loss_scale(model, ctx)
    return model


def _probe_loss_scale(model, ctx: DistContext) -> float:
    """The factor that turns FSDP2's gradient reduction into a sum.

    train_step's loss weights are already global, so per-rank gradients must
    be summed. What FSDP2 does differs by version and backend (torch 2.7
    averages on NCCL but sums on gloo), so it is measured, not assumed: a
    unit loss on the LM head gives every rank a gradient of exactly 1 per
    weight, which reduces to 1 under an average and to world_size under a sum.
    """
    head = model.lm_head
    probe = torch.ones(1, head.in_features, device=ctx.device)
    head(probe).float().sum().backward()
    reduced = float(head.weight.grad.to_local().float().mean())
    for parameter in model.parameters():
        parameter.grad = None
    scale = ctx.world_size / reduced
    if abs(scale - round(scale)) > 1e-3 or round(scale) not in (1, ctx.world_size):
        raise RuntimeError(f"unexpected FSDP2 gradient reduction (probe read {reduced})")
    return float(round(scale))


def gather_full_state_dict(model, ctx: DistContext) -> dict[str, torch.Tensor] | None:
    """Collective: every rank must call it. Rank 0 gets bf16 CPU tensors.

    Each shard is gathered and cast to bf16 on the GPU before the copy to host: 15 s instead of 50 s
    (fp32 offload, then a CPU cast) for the 9B on 4 A100s, same bytes."""
    state = {}
    for name, tensor in model.state_dict().items():
        full = (tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor).to(torch.bfloat16)
        if ctx.is_main:
            state[name] = full.cpu()
    return state if ctx.is_main else None


def assign_bins(costs: list[int], world_size: int) -> list[list[int]]:
    """Greedy longest-first split of micro-batch bins over ranks by cost.

    Deterministic (ties go to the lowest rank) so every rank computes the same
    split from the same plan without talking to the others.
    """
    loads = [0] * world_size
    parts: list[list[int]] = [[] for _ in range(world_size)]
    for index in sorted(range(len(costs)), key=lambda i: (-costs[i], i)):
        rank = min(range(world_size), key=lambda r: (loads[r], r))
        parts[rank].append(index)
        loads[rank] += costs[index]
    return parts


def all_reduce_stats(stats: dict, ctx: DistContext, *, max_keys: frozenset[str]) -> None:
    """Sum every counter across ranks, except ``max_keys`` which take the max."""
    keys = sorted(stats)
    sums = [k for k in keys if k not in max_keys]
    maxes = [k for k in keys if k in max_keys]
    for names, op in ((sums, dist.ReduceOp.SUM), (maxes, dist.ReduceOp.MAX)):
        if not names:
            continue
        values = torch.tensor([float(stats[k]) for k in names], dtype=torch.float64, device=ctx.device)
        dist.all_reduce(values, op=op)
        for name, value in zip(names, values.tolist()):
            stats[name] = int(value) if isinstance(stats[name], int) else value


def check_same_plan(signature: list[int], ctx: DistContext) -> None:
    """Fail fast when ranks built different micro-batch plans: their numbers of
    collectives would differ and the step would hang or mix gradients."""
    values = torch.tensor(signature, dtype=torch.int64, device=ctx.device)
    low, high = values.clone(), values.clone()
    dist.all_reduce(low, op=dist.ReduceOp.MIN)
    dist.all_reduce(high, op=dist.ReduceOp.MAX)
    if not torch.equal(low, high):
        raise RuntimeError(
            f"fsdp2 ranks disagree on the train plan: rank {ctx.rank} has {signature}, "
            f"min {low.tolist()} max {high.tolist()}"
        )


def all_reduce_tensor_sum(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


class CommandChannel:
    """Rank 0 broadcasts (command, payload) on the long-timeout gloo group."""

    def __init__(self, ctx: DistContext) -> None:
        self.ctx = ctx

    def send(self, command: str, payload: Any = None) -> None:
        if self.ctx.world_size == 1:
            return
        dist.broadcast_object_list([(command, payload)], src=0, group=self.ctx.control_group)

    def receive(self) -> tuple[str, Any]:
        message: list[Any] = [None]
        dist.broadcast_object_list(message, src=0, group=self.ctx.control_group)
        return message[0]


def run_follower(channel: CommandChannel, *, load, train, gather, audit=None) -> None:
    model = None
    while True:
        command, payload = channel.receive()
        if command == "LOAD":
            model = load(payload)
        elif command == "TRAIN":
            train(model, payload)
        elif command == "GATHER":
            gather(model)
        elif command == "AUDIT" and audit is not None:
            audit(model, payload)               # the same forward as rank 0's; only rank 0 reports
        elif command == "STOP":
            return
        else:
            raise ValueError(f"unknown trainer command {command!r}")


def broadcasting_train_step(channel: CommandChannel, train_step_fn):
    """Rank 0's train_step: the followers run the same call on the same batches."""

    def step(model, batches, **kwargs):
        channel.send("TRAIN", {"batches": batches, "kwargs": kwargs})
        return train_step_fn(model, batches, **kwargs)

    return step


def save_state_dict_hf(model, state_dict: dict, tokenizer, path) -> None:
    """Write an HF snapshot from a gathered bf16 state dict.

    The live shards are fp32; the published checkpoint is bf16 like today's,
    so config.json is written with dtype bfloat16 explicitly.
    """
    import copy
    from pathlib import Path

    from huggingface_hub import save_torch_state_dict

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_torch_state_dict(state_dict, path, max_shard_size="5GB")
    config = copy.deepcopy(model.config)
    config.dtype = torch.bfloat16
    config.save_pretrained(path)
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.save_pretrained(path)
    if tokenizer is not None:
        from reliquary.shared.modeling import save_tokenizer

        save_tokenizer(tokenizer, path)


def sharded_save_fn(model_ref, channel: CommandChannel, ctx: DistContext):
    """TrainerPublisher save_fn for rank 0: gather with every rank, then save."""

    def save(_model, tokenizer, path) -> None:
        model = model_ref()
        channel.send("GATHER")
        try:
            state = gather_full_state_dict(model, ctx)
        except Exception as exc:
            # The followers may still sit in a later collective of this
            # gather: a retried GATHER would pair mismatched parameters
            # (same-shaped layers gather "successfully"). Stop instead.
            from reliquary.trainer.train_runner import TrainerStateUnsafe

            raise TrainerStateUnsafe(
                "publish gather failed on rank 0; ranks are out of step, "
                "restart from the last published checkpoint"
            ) from exc
        save_state_dict_hf(model, state, tokenizer, path)

    return save


def load_sharded_model(model_path: str, load_kwargs: dict, ctx: DistContext, *,
                       attn_implementation: str):
    from reliquary.shared.modeling import load_text_generation_model

    model = load_text_generation_model(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation, **load_kwargs,
    ).to(ctx.device)
    model.gradient_checkpointing_enable()
    return shard_model(model, ctx)
