"""Production bounds around deferred-proof auction admission and sealing."""

import asyncio
import threading

import pytest


def _request(
    *,
    prompt_idx=1,
    hotkey="miner",
    env_name="openmathinstruct",
    rewards=None,
):
    from tests.unit.test_grpo_window_batcher import _request as make_request

    request = make_request(
        prompt_idx=prompt_idx,
        hotkey=hotkey,
        rewards=rewards,
    )
    for rollout in request.rollouts:
        rollout.env_name = env_name
    return request


def _batcher(**kwargs):
    from tests.unit.test_grpo_window_batcher import _make_batcher

    return _make_batcher(**kwargs)


def test_payload_reservation_moves_pending_inflight_retained_then_releases():
    batcher = _batcher()
    request = _request()
    request._payload_bytes = 1234

    assert batcher.try_reserve_proof_admission(request) == (True, None)
    assert batcher.pending_payload_bytes == 1234
    assert batcher.reserved_payload_bytes == 1234
    assert batcher.payload_bytes_by_hotkey == {"miner": 1234}

    assert batcher.start_proof_admission(request) == (True, None)
    assert batcher.pending_payload_bytes == 0
    assert batcher.inflight_payload_bytes == 1234
    assert batcher.reserved_payload_bytes == 1234

    assert batcher.accept_submission(request).accepted is True
    batcher.finish_proof_admission(request)
    assert batcher.inflight_payload_bytes == 0
    assert batcher.retained_payload_bytes == 1234
    assert batcher.reserved_payload_bytes == 1234

    batcher.seal_batch()
    assert batcher.retained_payload_bytes == 0
    assert batcher.reserved_payload_bytes == 0
    assert batcher.payload_bytes_by_hotkey == {}


def test_payload_reservation_cancel_and_reject_refund_every_counter():
    batcher = _batcher()
    cancelled = _request(prompt_idx=1, hotkey="cancelled")
    cancelled._payload_bytes = 321
    assert batcher.try_reserve_proof_admission(cancelled) == (True, None)
    assert batcher.cancel_proof_admission(cancelled) is True

    rejected = _request(prompt_idx=2, hotkey="rejected")
    rejected._payload_bytes = 654
    assert batcher.try_reserve_proof_admission(rejected) == (True, None)
    assert batcher.start_proof_admission(rejected) == (True, None)
    batcher.finish_proof_admission(rejected)

    assert batcher.pending_payload_bytes == 0
    assert batcher.inflight_payload_bytes == 0
    assert batcher.retained_payload_bytes == 0
    assert batcher.payload_bytes_by_hotkey == {}


def test_seal_exception_still_releases_retained_payload(monkeypatch):
    batcher = _batcher()
    request = _request()
    request._payload_bytes = 777
    assert batcher.try_reserve_proof_admission(request) == (True, None)
    assert batcher.start_proof_admission(request) == (True, None)
    request._retain_payload = True
    batcher.finish_proof_admission(request)

    def fail(_pool):
        raise RuntimeError("seal failed")

    monkeypatch.setattr(batcher, "_seal_batch_inner", fail)
    with pytest.raises(RuntimeError, match="seal failed"):
        batcher.seal_batch()

    assert batcher.reserved_payload_bytes == 0
    assert batcher.payload_bytes_by_hotkey == {}


def test_payload_caps_apply_per_request_hotkey_and_environment(monkeypatch):
    import reliquary.validator.batcher as batcher_module

    monkeypatch.setattr(batcher_module, "MAX_SUBMISSION_PAYLOAD_BYTES", 100)
    monkeypatch.setattr(
        batcher_module, "MAX_PENDING_SUBMISSION_BYTES_PER_HOTKEY", 150
    )
    monkeypatch.setattr(
        batcher_module, "MAX_PENDING_SUBMISSION_BYTES_PER_ENV", 220
    )
    batcher = _batcher()

    oversized = _request(prompt_idx=1, hotkey="a")
    oversized._payload_bytes = 101
    assert batcher.try_reserve_proof_admission(oversized) == (
        False,
        "submission_payload_too_large",
    )

    first = _request(prompt_idx=2, hotkey="a")
    first._payload_bytes = 100
    assert batcher.try_reserve_proof_admission(first) == (True, None)

    same_hotkey = _request(prompt_idx=3, hotkey="a")
    same_hotkey._payload_bytes = 51
    assert batcher.try_reserve_proof_admission(same_hotkey) == (
        False,
        "pending_payload_bytes_hotkey_full",
    )

    other_hotkey = _request(prompt_idx=4, hotkey="b")
    other_hotkey._payload_bytes = 100
    assert batcher.try_reserve_proof_admission(other_hotkey) == (True, None)

    env_full = _request(prompt_idx=5, hotkey="c")
    env_full._payload_bytes = 21
    assert batcher.try_reserve_proof_admission(env_full) == (
        False,
        "pending_payload_bytes_env_full",
    )


def test_seal_snapshot_rejects_and_refunds_a_queued_reservation():
    batcher = _batcher()
    request = _request()
    request._payload_bytes = 4321
    assert batcher.try_reserve_proof_admission(request) == (True, None)

    batcher.begin_seal_snapshot()

    assert batcher.start_proof_admission(request) == (
        False,
        "auction_seal_snapshot_started",
    )
    assert batcher.reserved_payload_bytes == 0
    assert batcher.pending_proof_reservations == 0


@pytest.mark.parametrize(
    ("env_name", "env_factory"),
    [
        ("openmathinstruct", None),
        ("opencodeinstruct", "code"),
    ],
)
def test_operator_ownership_does_not_limit_ranked_winners_for_math_and_code(
    env_name,
    env_factory,
):
    from tests.unit.test_grpo_window_batcher import PrivateRewardFakeEnv

    hotkeys = ["a1", "a2", "a3", "b1", "b2"]
    mapping = {
        "a1": "operator-a",
        "a2": "operator-a",
        "a3": "operator-a",
        "b1": "operator-b",
        "b2": "operator-b",
    }
    kwargs = {"operator_by_hotkey": mapping}
    if env_factory == "code":
        kwargs["env"] = PrivateRewardFakeEnv()
    batcher = _batcher(**kwargs)

    for prompt_idx, hotkey in enumerate(hotkeys):
        assert batcher.accept_submission(
            _request(
                prompt_idx=prompt_idx,
                hotkey=hotkey,
                env_name=env_name,
            )
        ).accepted

    batcher.seal_batch()
    winners = batcher.valid_submissions()
    operator_a_winners = [
        winner for winner in winners if mapping[winner.hotkey] == "operator-a"
    ]

    assert {winner.hotkey for winner in winners} == set(hotkeys)
    assert len(operator_a_winners) == 3
    assert {winner.hotkey for winner in winners} & {"b1", "b2"} == {"b1", "b2"}
    assert all(row["status"] == "selected" for row in batcher.auction_candidates)


def test_missing_production_operator_mapping_fails_closed():
    batcher = _batcher(operator_by_hotkey={"mapped": "operator-a"})
    unmapped = batcher.accept_submission(
        _request(prompt_idx=1, hotkey="unmapped")
    )
    assert batcher.accept_submission(
        _request(prompt_idx=2, hotkey="mapped")
    ).accepted

    assert unmapped.accepted is False
    assert unmapped.reason.value == "registration_unavailable"
    batcher.seal_batch()

    assert [winner.hotkey for winner in batcher.valid_submissions()] == ["mapped"]
    assert batcher.auction_operator_unmapped_skips == 0
    assert {row["hotkey"] for row in batcher.auction_candidates} == {"mapped"}


def test_operator_tiebreak_does_not_change_when_hotkey_changes():
    def seal_with(operator_a_hotkey):
        mapping = {
            operator_a_hotkey: "operator-a",
            "operator-b-hotkey": "operator-b",
        }
        batcher = _batcher(operator_by_hotkey=mapping)
        batcher.seal_randomness = "future-drand-beacon"
        for hotkey in mapping:
            assert batcher.accept_submission(
                _request(prompt_idx=7, hotkey=hotkey)
            ).accepted
        batcher.seal_batch()
        winner = batcher.valid_submissions()[0]
        rows = {mapping[row["hotkey"]]: row for row in batcher.auction_candidates}
        return mapping[winner.hotkey], rows

    first_winner, first_rows = seal_with("operator-a-hotkey-1")
    second_winner, second_rows = seal_with("operator-a-hotkey-999")

    assert first_winner == second_winner
    assert first_rows["operator-a"]["operator_tiebreak"] == (
        second_rows["operator-a"]["operator_tiebreak"]
    )
    assert {
        row["rank_entropy_source"] for row in first_rows.values()
    } == {"seal_drand"}


def test_code_grader_outage_never_becomes_an_auction_negative():
    from reliquary.environment.grader_client import GraderInfrastructureError
    from reliquary.protocol.submission import RejectReason
    from tests.unit.test_grpo_window_batcher import PrivateRewardFakeEnv

    class FlakyCodeEnv(PrivateRewardFakeEnv):
        unavailable = True

        def compute_reward(self, problem, completion):
            if self.unavailable:
                raise GraderInfrastructureError("unreachable")
            return super().compute_reward(problem, completion)

    env = FlakyCodeEnv()
    batcher = _batcher(
        env=env,
        operator_by_hotkey={"miner": "operator-a"},
    )
    request = _request(
        prompt_idx=7,
        hotkey="miner",
        env_name="opencodeinstruct",
    )

    outage = batcher.accept_submission(request)

    assert outage.accepted is False
    assert outage.reason is RejectReason.WORKER_DROPPED
    assert batcher.pending_submissions() == []
    assert batcher.logical_group_reservation_count == 0
    assert batcher.grader_failures == {"unreachable": 1}

    env.unavailable = False
    retry = batcher.accept_submission(
        _request(
            prompt_idx=7,
            hotkey="miner",
            env_name="opencodeinstruct",
        )
    )
    assert retry.accepted is True
    assert len(batcher.pending_submissions()) == 1


def test_code_grader_crash_is_not_a_free_retry():
    from reliquary.environment.grader_client import GraderInfrastructureError
    from reliquary.protocol.submission import RejectReason
    from tests.unit.test_grpo_window_batcher import PrivateRewardFakeEnv

    class CrashingCodeEnv(PrivateRewardFakeEnv):
        def compute_reward(self, problem, completion):
            raise GraderInfrastructureError("crash")

    batcher = _batcher(
        env=CrashingCodeEnv(),
        operator_by_hotkey={
            "miner-a": "operator-a",
            "miner-a-sybil": "operator-a",
        },
    )

    crashed = batcher.accept_submission(
        _request(
            prompt_idx=7,
            hotkey="miner-a",
            env_name="opencodeinstruct",
        )
    )
    retry = batcher.accept_submission(
        _request(
            prompt_idx=7,
            hotkey="miner-a-sybil",
            env_name="opencodeinstruct",
        )
    )

    assert crashed.accepted is False
    assert crashed.reason is RejectReason.REWARD_MISMATCH
    assert retry.accepted is False
    assert retry.reason is RejectReason.HASH_DUPLICATE
    assert batcher.logical_group_reservation_count == 1
    assert batcher.grader_failures == {"crash": 1}


def test_failed_proof_does_not_consume_an_operator_slot(monkeypatch):
    batcher = _batcher(
        operator_by_hotkey={
            "failed": "operator-a",
            "winner-1": "operator-a",
            "winner-2": "operator-a",
        }
    )
    for prompt_idx, hotkey in enumerate(("failed", "winner-1", "winner-2")):
        rewards = (
            [1.0, 1.0] + [0.0] * 6
            if hotkey == "failed"
            else [1.0] * 4 + [0.0] * 4
        )
        assert batcher.accept_submission(
            _request(prompt_idx=prompt_idx, hotkey=hotkey, rewards=rewards)
        ).accepted

    original_verify = batcher._verify_expensive

    def verify(pending):
        if pending.hotkey == "failed":
            return None
        return original_verify(pending)

    monkeypatch.setattr(batcher, "_verify_expensive", verify)
    batcher.seal_batch()

    assert {winner.hotkey for winner in batcher.valid_submissions()} == {
        "winner-1",
        "winner-2",
    }
    rows = {row["hotkey"]: row for row in batcher.auction_candidates}
    assert rows["failed"]["status"] == "proof_failed"


def test_operator_failure_debt_blocks_multi_hotkey_proof_exhaustion(
    monkeypatch,
):
    from reliquary.constants import (
        B_BATCH,
        MAX_EXPENSIVE_PROOF_FAILURES_PER_OPERATOR_PER_WINDOW,
    )

    failure_cap = MAX_EXPENSIVE_PROOF_FAILURES_PER_OPERATOR_PER_WINDOW
    fake_hotkeys = [f"fake-{i}" for i in range(failure_cap + 3)]
    honest_hotkeys = [f"honest-{i}" for i in range(B_BATCH)]
    mapping = {
        **{hotkey: "operator-attacker" for hotkey in fake_hotkeys},
        **{
            hotkey: f"operator-honest-{i}"
            for i, hotkey in enumerate(honest_hotkeys)
        },
    }
    batcher = _batcher(operator_by_hotkey=mapping)
    for prompt_idx, hotkey in enumerate(fake_hotkeys):
        assert batcher.accept_submission(
            _request(
                prompt_idx=prompt_idx,
                hotkey=hotkey,
                rewards=[1.0, 1.0] + [0.0] * 6,
            )
        ).accepted
    for offset, hotkey in enumerate(honest_hotkeys, start=len(fake_hotkeys)):
        assert batcher.accept_submission(
            _request(
                prompt_idx=offset,
                hotkey=hotkey,
                rewards=[1.0] * 4 + [0.0] * 4,
            )
        ).accepted

    original_verify = batcher._verify_expensive
    attempted_hotkeys = []

    def verify(pending):
        attempted_hotkeys.append(pending.hotkey)
        if pending.hotkey.startswith("fake-"):
            return None
        return original_verify(pending)

    monkeypatch.setattr(batcher, "_verify_expensive", verify)
    batcher.seal_batch()

    assert sum(
        hotkey.startswith("fake-") for hotkey in attempted_hotkeys
    ) == failure_cap
    assert batcher.operator_proof_failure_debt("operator-attacker") == (
        failure_cap
    )
    assert {winner.hotkey for winner in batcher.valid_submissions()} == set(
        honest_hotkeys
    )
    fake_rows = {
        row["hotkey"]: row for row in batcher.auction_candidates
        if row["hotkey"].startswith("fake-")
    }
    assert sum(
        row["status"] == "operator_proof_debt"
        for row in fake_rows.values()
    ) == 3
    assert batcher.auction_operator_proof_debt_skips == 3


def test_proof_wall_budget_stops_ranked_work_and_archives_shortfall(monkeypatch):
    import reliquary.validator.batcher as batcher_module

    now = [0.0]
    mapping = {f"miner-{i}": f"operator-{i}" for i in range(5)}
    batcher = _batcher(
        time_fn=lambda: now[0],
        operator_by_hotkey=mapping,
    )
    for prompt_idx, hotkey in enumerate(mapping):
        assert batcher.accept_submission(
            _request(prompt_idx=prompt_idx, hotkey=hotkey)
        ).accepted

    original_verify = batcher._verify_expensive

    def verify(pending):
        result = original_verify(pending)
        now[0] += 0.6
        return result

    monkeypatch.setattr(batcher, "_verify_expensive", verify)
    monkeypatch.setattr(batcher_module, "MAX_PROOF_WALL_SECONDS", 1.0)

    batcher.seal_batch()

    assert batcher.proof_attempts == 2
    assert len(batcher.valid_submissions()) == 2
    assert batcher.proof_wall_exhausted is True
    assert batcher.difficulty_auction_shadow["proof_wall_exhausted"] is True
    assert any(
        row["status"] == "unobserved_wall_budget"
        for row in batcher.auction_candidates
    )


def test_transport_body_limits_reject_before_request_validation(monkeypatch):
    from fastapi.testclient import TestClient
    import reliquary.validator.server as server_module

    monkeypatch.setattr(server_module, "MAX_SUBMISSION_PAYLOAD_BYTES", 5)
    server = server_module.ValidatorServer()

    with TestClient(server.app) as client:
        response = client.post(
            "/submit",
            content=b"123456",
            headers={"content-type": "application/json"},
        )
        chunked = client.post(
            "/submit",
            content=(chunk for chunk in (b"123", b"456")),
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 413
    assert response.json() == {"detail": "submission_payload_too_large"}
    assert chunked.request.headers.get("content-length") is None
    assert chunked.status_code == 413
    assert chunked.json() == {"detail": "submission_payload_too_large"}


def test_code_rewards_use_all_sandbox_lanes_in_parallel():
    from reliquary.constants import M_ROLLOUTS
    from tests.unit.test_grpo_window_batcher import PrivateRewardFakeEnv

    class ConcurrentCodeEnv(PrivateRewardFakeEnv):
        def __init__(self):
            self.barrier = threading.Barrier(M_ROLLOUTS, timeout=2.0)
            self.thread_ids = set()
            self.thread_ids_lock = threading.Lock()

        def compute_reward(self, problem, completion):
            with self.thread_ids_lock:
                self.thread_ids.add(threading.get_ident())
            self.barrier.wait()
            return 1.0 if "CORRECT" in completion else 0.0

    env = ConcurrentCodeEnv()
    batcher = _batcher(env=env)
    request = _request(env_name="opencodeinstruct")

    response = batcher.accept_submission(request)

    assert response.accepted is True, response.reason
    assert len(env.thread_ids) == len(request.rollouts)


def test_math_and_code_submission_queues_are_independent_and_observable():
    from reliquary.validator.server import ValidatorServer

    server = ValidatorServer()
    math_queue = server._submission_queue_for_environment(
        "openmathinstruct"
    )
    code_queue = server._submission_queue_for_environment(
        "opencodeinstruct"
    )

    assert math_queue is server._submit_queue
    assert code_queue is server._code_submit_queue
    assert code_queue is not math_queue

    math_queue.put_nowait((object(), object()))
    code_queue.put_nowait((object(), object()))
    code_queue.put_nowait((object(), object()))

    assert server.submit_queue_depth == 3
    assert server.submit_queue_depth_by_environment == {
        "openmathinstruct": 1,
        "opencodeinstruct": 2,
    }
    assert server.proof_verification_inflight_by_environment == {
        "openmathinstruct": 0,
        "opencodeinstruct": 0,
    }


def test_health_reports_pending_auction_population_before_seal():
    from reliquary.validator.server import ValidatorServer

    batcher = _batcher()
    for prompt_idx in (1, 2):
        response = batcher.accept_submission(
            _request(prompt_idx=prompt_idx, hotkey=f"miner-{prompt_idx}")
        )
        assert response.accepted is True, response.reason

    server = ValidatorServer()
    server.set_active_batchers({"openmathinstruct": batcher})
    health = server._health_payload()

    assert health.valid_submissions_count == 2
    assert health.distinct_valid_prompt_count == 2
    assert (
        health.window_environments["openmathinstruct"][
            "valid_submissions_count"
        ]
        == 2
    )


@pytest.mark.asyncio
async def test_chunked_body_limit_rejects_before_downstream_parsing():
    from reliquary.validator.server import _SubmissionBodyLimitMiddleware

    downstream_called = False

    async def downstream(scope, receive, send):
        nonlocal downstream_called
        downstream_called = True
        while (await receive()).get("more_body", False):
            pass

    app = _SubmissionBodyLimitMiddleware(downstream, max_bytes=5)
    chunks = [
        {"type": "http.request", "body": b"123", "more_body": True},
        {"type": "http.request", "body": b"456", "more_body": False},
    ]
    sent = []

    async def receive():
        await asyncio.sleep(0)
        return chunks.pop(0)

    async def send(message):
        sent.append(message)

    await app(
        {
            "type": "http",
            "method": "POST",
            "path": "/submit",
            "headers": [],
        },
        receive,
        send,
    )

    assert downstream_called is True
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413


@pytest.mark.asyncio
async def test_worker_drains_predeadline_auction_item_after_collection_closes(
    monkeypatch,
):
    from reliquary.protocol.submission import WindowState
    from reliquary.validator.server import ValidatorServer

    cache_clears = []
    monkeypatch.setattr(
        "reliquary.validator.service._try_empty_cuda_cache",
        lambda: cache_clears.append(True),
    )

    server = ValidatorServer()
    batcher = _batcher()
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    request = _request()
    request._payload_bytes = 999
    assert batcher.try_reserve_proof_admission(request) == (True, None)

    # The HTTP deadline is closed, but this request was already admitted into
    # the queue. It belongs to the auction population until the service freezes
    # the snapshot.
    batcher._seal_flag.set()
    await server._submit_queue.put((request, batcher))
    worker = asyncio.create_task(server._submit_worker())
    try:
        for _ in range(100):
            if batcher.pending_count == 1:
                break
            await asyncio.sleep(0.01)
        assert batcher.pending_count == 1
        assert batcher.pending_proof_reservations == 0
        assert batcher.inflight_proof_reservations == 0
        assert batcher.retained_payload_bytes == 999
        assert cache_clears == []
    finally:
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

    batcher.seal_batch()
    assert batcher.reserved_payload_bytes == 0
