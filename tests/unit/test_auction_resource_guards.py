"""Production bounds around deferred-proof auction admission and sealing."""

import asyncio

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
def test_operator_cap_is_enforced_for_math_and_code(env_name, env_factory):
    from reliquary.constants import MAX_AUCTION_SLOTS_PER_OPERATOR
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

    assert len(operator_a_winners) == MAX_AUCTION_SLOTS_PER_OPERATOR
    assert {winner.hotkey for winner in winners} & {"b1", "b2"} == {"b1", "b2"}
    assert batcher.auction_operator_cap_skips == 1


def test_missing_production_operator_mapping_fails_closed():
    batcher = _batcher(operator_by_hotkey={"mapped": "operator-a"})
    assert batcher.accept_submission(
        _request(prompt_idx=1, hotkey="unmapped")
    ).accepted
    assert batcher.accept_submission(
        _request(prompt_idx=2, hotkey="mapped")
    ).accepted

    batcher.seal_batch()

    assert [winner.hotkey for winner in batcher.valid_submissions()] == ["mapped"]
    assert batcher.auction_operator_unmapped_skips == 1
    rows = {row["hotkey"]: row for row in batcher.auction_candidates}
    assert rows["unmapped"]["status"] == "operator_unmapped"


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
async def test_worker_drains_predeadline_auction_item_after_collection_closes():
    from reliquary.protocol.submission import WindowState
    from reliquary.validator.server import ValidatorServer

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
    finally:
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

    batcher.seal_batch()
    assert batcher.reserved_payload_bytes == 0
