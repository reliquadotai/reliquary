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


def trainer_backend(env: Mapping[str, str] = os.environ) -> str:
    value = str(env.get(BACKEND_ENV, "single")).strip().lower()
    if value not in _BACKENDS:
        raise ValueError(f"{BACKEND_ENV} must be one of {_BACKENDS}, got {value!r}")
    return value


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
    for key, default in (("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", "29511"),
                         ("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")):
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
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    control = dist.new_group(backend="gloo", timeout=_CONTROL_TIMEOUT)
    return DistContext(rank, world_size, device, control)


def shard_model(model, ctx: DistContext, *, param_dtype=torch.bfloat16):
    """Shard every decoder layer, the LM head and the base, then the root.

    train_step calls the base model and the LM head directly (never the root
    forward), so each needs its own unit for its parameters to be gathered.
    The shards are cast to fp32 afterwards: they are the master weights, and
    the mixed-precision policy hands bf16 copies to the compute.
    """
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    mesh = init_device_mesh(ctx.device.type, (ctx.world_size,))
    policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32)
    base = getattr(model, model.base_model_prefix)
    for layer in base.layers:
        fully_shard(layer, mesh=mesh, mp_policy=policy)
    fully_shard(model.lm_head, mesh=mesh, mp_policy=policy, reshard_after_forward=False)
    fully_shard(base, mesh=mesh, mp_policy=policy, reshard_after_forward=False)
    fully_shard(model, mesh=mesh, mp_policy=policy)
    model.to(torch.float32)
    return model


def gather_full_state_dict(model, ctx: DistContext) -> dict[str, torch.Tensor] | None:
    """Collective: every rank must call it. Rank 0 gets bf16 CPU tensors."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    state = get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    if not ctx.is_main:
        return None
    return {name: tensor.to(torch.bfloat16) for name, tensor in state.items()}


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


def all_reduce_tensor_sum(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor
