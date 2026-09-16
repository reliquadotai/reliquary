"""One environment's burst must not reject another environment's miners.

Admission capacity is per environment: each batcher owns its own budget of
``MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW`` grading slots. The transport queue
that carries admitted work to the grading workers was not — it was one queue
per *resource class*, sized for the era when a class held exactly one
environment.

With several environments on a class, that shared queue saturates long before
the budgets that feed it, and the overflow is charged to whichever environment
happens to arrive next. An environment whose own budget is untouched then sees
``batch_filled`` — a reject it cannot act on and, since the price of an
environment is a function of what it turns away, one that corrupts its price
signal with its neighbour's load.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from reliquary.constants import MAX_PENDING_PROOF_QUEUE_DEPTH
from reliquary.protocol.submission import (
    BatchSubmissionResponse,
    RejectReason,
    WindowState,
)
from reliquary.validator.server import ValidatorServer

# Two environments the registry places on the CPU admission class.
_CPU_ENV_A = "openmathinstruct"
_CPU_ENV_B = "reliquarylogic_v1"
# ...and one on the sandbox class, which must stay separate from both.
_SANDBOX_ENV = "opencodeinstruct"


def _batcher(env_name: str) -> MagicMock:
    batcher = MagicMock()
    batcher.window_start = 500
    batcher.current_checkpoint_hash = "sha256:current"
    batcher.cooldown_prompts_snapshot = []
    batcher.env = MagicMock()
    batcher.env.name = env_name
    batcher.env.__len__.return_value = 1000
    batcher.is_sealed.return_value = False
    batcher.difficulty_auction_enabled = False
    batcher._seal_trigger_round = None
    batcher.prompt_range = None
    batcher.drand_round_check_enabled = False
    batcher.validate_drand_round.return_value = None
    batcher.prompt_submission_count.return_value = 0
    batcher.try_reserve_proof_admission.return_value = (True, None)
    batcher.accept_submission.return_value = BatchSubmissionResponse(
        accepted=True, reason=RejectReason.ACCEPTED,
    )
    return batcher


def _submission(env_name: str, *, hotkey: str = "hkA") -> dict:
    commit = {
        "tokens": list(range(36)),
        "commitments": [{"sketch": 0} for _ in range(36)],
        "proof_version": "v7",
        "model": {"name": "test", "layer_index": 6},
        "signature": "ab" * 32,
        "beacon": {"randomness": "cd" * 16},
        "rollout": {
            "prompt_length": 4, "completion_length": 32,
            "success": True, "total_reward": 1.0, "advantage": 0.0,
            "token_logprobs": [0.0] * 36,
        },
    }
    rollout = {
        "tokens": list(range(36)),
        "reward": 1.0,
        "commit": commit,
        "env_name": env_name,
    }
    return {
        "miner_hotkey": hotkey,
        "prompt_idx": 42,
        "window_start": 500,
        "merkle_root": "00" * 32,
        "rollouts": [deepcopy(rollout) for _ in range(8)],
        "checkpoint_hash": "sha256:current",
        "drand_round": 0,
        "protocol_version": 2,
    }


def test_cpu_environments_do_not_share_one_transport_queue():
    """Saturating one CPU environment's queue must leave the other's empty."""
    server = ValidatorServer()

    queue_a = server._submission_queue_for_environment(_CPU_ENV_A)
    queue_b = server._submission_queue_for_environment(_CPU_ENV_B)

    for _ in range(MAX_PENDING_PROOF_QUEUE_DEPTH):
        queue_a.put_nowait(object())

    assert queue_a.qsize() == MAX_PENDING_PROOF_QUEUE_DEPTH
    assert queue_b.qsize() == 0
    # Env B's own capacity is untouched, so its next item must fit.
    queue_b.put_nowait(object())


def test_sandbox_environment_keeps_its_own_queue():
    """The original CPU/sandbox split must survive the per-environment one."""
    server = ValidatorServer()

    cpu_queue = server._submission_queue_for_environment(_CPU_ENV_A)
    sandbox_queue = server._submission_queue_for_environment(_SANDBOX_ENV)

    for _ in range(MAX_PENDING_PROOF_QUEUE_DEPTH):
        cpu_queue.put_nowait(object())

    assert sandbox_queue.qsize() == 0
    sandbox_queue.put_nowait(object())


def test_every_environment_on_a_lane_gets_a_worker():
    """A queue nobody drains is worse than a shared one: it fills once and
    stays full for the rest of the window."""
    from reliquary.validator.server import lane_worker_allocation

    allocation = lane_worker_allocation(["a", "b", "c", "d", "e"], total=8)

    assert set(allocation) == {"a", "b", "c", "d", "e"}
    assert min(allocation.values()) >= 1


def test_lane_worker_total_does_not_grow_with_environment_count():
    """Workers are a validator-wide resource. Five environments on a lane must
    share the lane's workers, not multiply them."""
    from reliquary.validator.server import lane_worker_allocation

    one = lane_worker_allocation(["a"], total=8)
    five = lane_worker_allocation(["a", "b", "c", "d", "e"], total=8)

    assert sum(one.values()) == 8
    assert sum(five.values()) == 8


def test_lane_worker_allocation_is_even():
    """No environment may be starved relative to its neighbours."""
    from reliquary.validator.server import lane_worker_allocation

    allocation = lane_worker_allocation(["a", "b", "c"], total=8)

    assert sorted(allocation.values()) == [2, 3, 3]


def test_more_environments_than_workers_still_drains_each():
    """Below one worker per environment, the floor of 1 wins over the total —
    an undrained queue is a hard failure, an extra idle task is not."""
    from reliquary.validator.server import lane_worker_allocation

    allocation = lane_worker_allocation(["a", "b", "c", "d"], total=2)

    assert allocation == {"a": 1, "b": 1, "c": 1, "d": 1}


def test_full_neighbour_queue_does_not_reject_another_environment():
    """The miner-visible consequence: env B is admitted while env A is full."""
    server = ValidatorServer()
    server.set_current_state(WindowState.OPEN)
    server.set_active_batchers({
        _CPU_ENV_A: _batcher(_CPU_ENV_A),
        _CPU_ENV_B: _batcher(_CPU_ENV_B),
    })
    # Force the queue path; with no worker task /submit grades synchronously.
    server._worker_task = object()

    queue_a = server._submission_queue_for_environment(_CPU_ENV_A)
    for _ in range(MAX_PENDING_PROOF_QUEUE_DEPTH):
        queue_a.put_nowait(object())

    with TestClient(server.app) as client:
        response = client.post("/submit", json=_submission(_CPU_ENV_B))

    body = response.json()
    assert body["reason"] != RejectReason.BATCH_FILLED.value, body
    assert body["accepted"] is True, body
