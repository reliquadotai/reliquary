"""Cheap factories/handlers for ProofWorkerPool tests.

Resolved by dotted path inside the spawned child, so they must live in an
importable module rather than a test-local closure.
"""

from __future__ import annotations

import os
import time
from typing import Any


def build_counter_context(*, device: str, start: int = 0) -> dict[str, Any]:
    return {"pid": os.getpid(), "calls": 0, "value": start,
            "device": device, "revision": None}


def echo_handler(context: dict[str, Any], payload: Any) -> dict[str, Any]:
    context["calls"] += 1
    return {
        "pid": context["pid"],
        "calls": context["calls"],
        "payload": payload,
        "revision": context["revision"],
    }


def reload_handler(
    context: dict[str, Any], snapshot_dir: str | None, revision: str,
    repo_id: str | None = None,
) -> None:
    context["revision"] = revision


def boom_handler(context: dict[str, Any], payload: Any) -> dict[str, Any]:
    raise ValueError(f"handler refused {payload!r}")


def crash_handler(context: dict[str, Any], payload: Any) -> dict[str, Any]:
    os._exit(9)


def slow_handler(context: dict[str, Any], payload: Any) -> dict[str, Any]:
    time.sleep(float(payload))
    return {"slept": payload}


def dispatch_handler(context: dict[str, Any], op: str, payload: Any) -> dict[str, Any]:
    """One handler with several behaviours, selected by the caller."""
    if op == "crash":
        os._exit(9)
    if op == "boom":
        raise ValueError(f"handler refused {payload!r}")
    if op == "sleep":
        time.sleep(float(payload))
    return echo_handler(context, payload)


def proof_result_handler(context: dict[str, Any], commit, randomness, seed_u_values):
    """Return a populated ProofResult, to prove it survives the pipe."""
    from reliquary.validator.verifier import ProofResult

    context["calls"] += 1
    return ProofResult(
        all_passed=True,
        passed=len(commit["tokens"]),
        checked=len(commit["tokens"]),
        sketch_diff_max=17,
        has_sparse_outputs=True,
        p_stop=0.125,
        challenge_lp_indices=[3, 5],
        challenge_lp_values=[-0.5, -1.5],
        completion_chosen_probs=[0.9] * 8,
        completion_argmax_probs=[0.95] * 8,
        completion_argmax_ids=[7] * 8,
        seed_n_stochastic=6,
        seed_n_match=5,
        terminal_pick_ok=True,
    )


def build_device_context(*, device: str, tag: str = "") -> dict[str, Any]:
    return {"pid": os.getpid(), "calls": 0, "device": device,
            "tag": tag, "revision": None}


def device_handler(context: dict[str, Any], payload: Any) -> dict[str, Any]:
    context["calls"] += 1
    return {"device": context["device"], "tag": context["tag"], "payload": payload}


def build_context_installing(*, device: str, revision: str | None = None) -> dict[str, Any]:
    """Factory that reports the revision it actually installed."""
    return {"pid": os.getpid(), "calls": 0, "device": device, "revision": revision}


def build_context_installing_dispatch(*, device: str, revision: str | None = None) -> dict[str, Any]:
    """Same as build_context_installing, for the crash/dispatch handler."""
    return {"pid": os.getpid(), "calls": 0, "device": device, "revision": revision}


def slow_reload_handler(
    context: dict[str, Any], seconds: str, revision: str,
    repo_id: str | None = None,
) -> None:
    time.sleep(float(seconds))
    context["revision"] = revision


def build_context_that_dies(*, device: str) -> dict[str, Any]:
    raise RuntimeError(f"cannot build a replica on {device}")
