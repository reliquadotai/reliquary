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

from reliquary.constants import (
    MATH_ADMISSION_WORKERS,
    MAX_PENDING_PROOF_QUEUE_DEPTH,
)
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
    server.set_admission_environments([_CPU_ENV_A, _CPU_ENV_B, _SANDBOX_ENV])

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


def test_admission_pools_do_not_multiply_with_environment_count():
    """Grading processes are the box's memory, not the lane's. Five CPU
    environments must share the CPU pool budget, not take it each."""
    from reliquary.constants import CODE_ADMISSION_WORKERS, MATH_ADMISSION_WORKERS
    from reliquary.validator.server import admission_pool_allocation

    sizes = admission_pool_allocation([
        "openmathinstruct",
        "reliquarylogic_v1",
        "reliquaryverifiable_v1",
        "reliquary_stateful_tools_v1",
        "reliquary_retrieval_tools_v1",
        "opencodeinstruct",
        "reliquary_workspace_tools_v1",
    ])

    cpu_total = sum(
        size for env, size in sizes.items()
        if env not in ("opencodeinstruct", "reliquary_workspace_tools_v1")
    )
    sandbox_total = sum(
        sizes[env]
        for env in ("opencodeinstruct", "reliquary_workspace_tools_v1")
    )
    assert cpu_total == MATH_ADMISSION_WORKERS
    assert sandbox_total == CODE_ADMISSION_WORKERS
    assert min(sizes.values()) >= 1


def test_single_environment_lane_keeps_its_whole_pool():
    """The shape production runs today must not change."""
    from reliquary.constants import MATH_ADMISSION_WORKERS
    from reliquary.validator.server import admission_pool_allocation

    sizes = admission_pool_allocation(["openmathinstruct"])

    assert sizes == {"openmathinstruct": MATH_ADMISSION_WORKERS}


def test_declared_environments_shrink_a_crowded_lane():
    """Four cpu environments share the cpu budget rather than take it each."""
    server = ValidatorServer()
    lane = [
        "openmathinstruct",
        "reliquarylogic_v1",
        "reliquaryverifiable_v1",
        "reliquary_stateful_tools_v1",
    ]

    server.set_admission_environments(lane)

    assert sum(server._admission_worker_count(env) for env in lane) == 8
    assert server._admission_worker_count("openmathinstruct") == 2


def test_drainers_and_pool_processes_agree_per_environment():
    """A pool larger than its drainers idles; drainers larger than their pool
    queue up behind it. Both read one allocation, and `ValidationService` can
    be given a strict subset of the profile's mix — so the profile's list is
    not that set."""
    server = ValidatorServer()
    running = ["openmathinstruct", "reliquarylogic_v1"]

    server.set_admission_environments(running)

    # `admission_allocation` is what `start` spawns from; `_admission_worker_count`
    # is what pool construction reads.
    cpu_lane = server.admission_allocation()["cpu"]
    for environment in running:
        assert cpu_lane[environment] == 4
        assert server._admission_worker_count(environment) == 4
    # An environment the validator does not run gets no drainers at all.
    assert "reliquaryverifiable_v1" not in cpu_lane


def test_empty_lane_fallback_keeps_its_own_queue():
    """A profile with no sandbox environment must not park sandbox drainers on
    the cpu queue. The synthetic key has no registry entry, so nothing may
    recover its class from its name."""
    server = ValidatorServer()
    # A profile whose mix declares no sandbox environment at all.
    server._default_queue_environment_by_class.pop("sandbox", None)

    synthetic = server._default_queue_environment("sandbox")

    assert synthetic not in ("openmathinstruct", "opencodeinstruct")
    assert (
        server._submission_queue_for_environment(synthetic)
        is not server._submission_queue_for_environment("openmathinstruct")
    )


def test_undeclared_environment_does_not_get_a_second_full_lane_budget():
    """Its lane's budget is already spoken for by the environments that own
    it, so an undeclared one takes the floor, not another whole share."""
    from reliquary.constants import MATH_ADMISSION_WORKERS

    server = ValidatorServer()
    server.set_admission_environments(["openmathinstruct"])

    assert (
        server._admission_worker_count("openmathinstruct")
        == MATH_ADMISSION_WORKERS
    )
    assert server._admission_worker_count("reliquarylogic_v1") == 1


def test_admission_environments_default_to_the_profile_mix():
    """An embedder that never declares its environments keeps today's shape."""
    from reliquary.constants import MATH_ADMISSION_WORKERS

    server = ValidatorServer()

    # Present in the allocation because the profile's mix supplied it — the
    # undeclared fallback would answer 1, not the lane budget.
    assert (
        server.admission_allocation()["cpu"]["openmathinstruct"]
        == MATH_ADMISSION_WORKERS
    )


def test_full_neighbour_queue_does_not_reject_another_environment():
    """The miner-visible consequence: env B is admitted while env A is full."""
    server = ValidatorServer()
    server.set_current_state(WindowState.OPEN)
    # As the service does: the environments it runs are the ones it declares.
    server.set_admission_environments([_CPU_ENV_A, _CPU_ENV_B])
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


def test_no_queue_exists_without_a_drainer():
    """A queue nobody drains is worse than a shared one: its items get no
    verdict at all and hold their proof-admission reservation until the window
    aborts. An environment outside the allocation has no drainers, so it must
    not get a queue of its own."""
    server = ValidatorServer()
    server.set_admission_environments([_CPU_ENV_A])

    drained = {
        environment
        for lane in server.admission_allocation().values()
        for environment in lane
    }
    assert _CPU_ENV_B not in drained

    assert (
        server._submission_queue_for_environment(_CPU_ENV_B)
        is server._submission_queue_for_environment(_CPU_ENV_A)
    )


def test_sandboxless_profile_keeps_its_sandbox_lane_off_the_cpu_queue(
    monkeypatch,
):
    """End state under a profile that declares no sandbox environment: the
    sandbox lane falls back to a synthetic key, and that key still resolves to
    a queue of its own. Two mechanisms defend this — the key is minted at
    construction and the allocation carries its class — so reverting either
    one alone leaves this green; `test_empty_lane_fallback_keeps_its_own_queue`
    is what pins the routing guard itself."""
    import reliquary.validator.server as server_module

    monkeypatch.setattr(
        server_module, "ENVIRONMENT_MIX", [(_CPU_ENV_A, 16)],
    )
    server = ValidatorServer()

    synthetic = server._default_queue_environment("sandbox")
    assert synthetic == "__sandbox_lane__"
    assert server._submission_queue_for_environment(synthetic) is not (
        server._submission_queue_for_environment(_CPU_ENV_A)
    )


def test_server_follows_the_runtime_environment_selection(monkeypatch):
    """The validator runs the mix named by RELIQUARY_ENVIRONMENTS, a strict
    subset of the profile's (its default is a single environment). The server
    reads that selection itself rather than being told, so the
    attestation-covered `service.py` does not have to carry it."""
    import reliquary.validator.server as server_module

    # A profile declaring three cpu environments...
    monkeypatch.setattr(
        server_module,
        "ENVIRONMENT_MIX",
        [(_CPU_ENV_A, 16), (_CPU_ENV_B, 16), ("reliquaryverifiable_v1", 16)],
    )
    # ...of which this validator actually runs two.
    monkeypatch.setenv("RELIQUARY_ENVIRONMENTS", f"{_CPU_ENV_A},{_CPU_ENV_B}")
    server = ValidatorServer()

    cpu_lane = server.admission_allocation()["cpu"]

    assert set(cpu_lane) == {_CPU_ENV_A, _CPU_ENV_B}
    # The budget goes to the two that run, not split three ways.
    assert cpu_lane[_CPU_ENV_A] == MATH_ADMISSION_WORKERS // 2


def test_explicit_declaration_overrides_the_environment_variable(monkeypatch):
    """An embedder that declares its set explicitly still wins."""
    monkeypatch.setenv("RELIQUARY_ENVIRONMENTS", _CPU_ENV_A)
    server = ValidatorServer()

    server.set_admission_environments([_CPU_ENV_A, _CPU_ENV_B])

    assert set(server.admission_allocation()["cpu"]) == {_CPU_ENV_A, _CPU_ENV_B}


def test_unset_environment_variable_falls_back_to_the_profile(monkeypatch):
    """No selection declared anywhere: the profile's mix is the answer."""
    monkeypatch.delenv("RELIQUARY_ENVIRONMENTS", raising=False)
    server = ValidatorServer()

    assert server.admission_allocation()["cpu"][_CPU_ENV_A] == (
        MATH_ADMISSION_WORKERS
    )
