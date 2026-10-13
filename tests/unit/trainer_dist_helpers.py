"""Helpers for multi-process FSDP2 tests on CPU (gloo)."""
from __future__ import annotations

import os
import pickle
import socket

import torch
import torch.multiprocessing as mp


def tiny_qwen35(seed: int = 0, vocab: int = 300):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(
        vocab_size=vocab, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=16,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        tie_word_embeddings=False,
    )
    return Qwen3_5ForCausalLM(cfg).float()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _entry(rank, world_size, port, fn, args, queue):
    os.environ.update({
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
        "RANK": str(rank), "WORLD_SIZE": str(world_size), "LOCAL_RANK": str(rank),
    })
    import faulthandler
    import sys

    # Ranks share the box's cores; oversubscribed threads make gloo tests crawl.
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // (2 * world_size)))
    faulthandler.dump_traceback_later(240, exit=False, file=sys.stderr)
    try:
        # Plain pickle copies tensor bytes; the queue's own pickler would
        # hand over shared memory that dies with this process.
        queue.put((rank, pickle.dumps(fn(*args))))
    except BaseException as exc:  # surface the failure to the parent
        queue.put((rank, exc))
        raise
    finally:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()


def run_ranks(fn, world_size: int = 2, *args, timeout: float = 300.0) -> dict:
    """Run ``fn(*args)`` on ``world_size`` gloo ranks; return {rank: result}."""
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_entry, args=(r, world_size, port, fn, args, queue))
        for r in range(world_size)
    ]
    for p in procs:
        p.start()
    results = {}
    for _ in procs:
        rank, value = queue.get(timeout=timeout)
        if isinstance(value, BaseException):
            for p in procs:
                p.kill()
            raise value
        results[rank] = pickle.loads(value)
    for p in procs:
        p.join(timeout=timeout)
    return results
