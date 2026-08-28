from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import time

from fastapi.testclient import TestClient
import pytest

from reliquary.protocol.submission import RejectReason, WindowState
from reliquary.shared.checkpoint_epoch import (
    BeaconBinding,
    CheckpointBinding,
    ProtocolBinding,
    SignedEpochCommitmentSet,
    WindowSchedule,
    build_epoch_plan,
    canonical_manifest_bytes,
    commitment_set_sha256,
    manifest_sha256,
)
from reliquary.validator.observability import SubmitTelemetry
from reliquary.validator.server import ValidatorServer
from tests.unit.test_grpo_window_batcher import _make_batcher, _request
from tests.unit.test_validator_server import _batcher as _server_batcher
from tests.unit.test_validator_server import _precommit_for, _request as _server_request


def _plan(
    *,
    randomness: str = "4" * 64,
    window_count: int = 2,
    environments: dict[str, int] | None = None,
):
    return build_epoch_plan(
        protocol=ProtocolBinding(
            profile_id="checkpoint-epoch-test-only",
            protocol_version=99,
            generation_contract_sha256="1" * 64,
        ),
        checkpoint=CheckpointBinding(
            number=7,
            repo_id="example/checkpoint",
            revision="2" * 40,
            commit_observed_round=100,
        ),
        epoch_beacon=BeaconBinding(
            source="drand",
            chain="quicknet",
            chain_hash="3" * 64,
            round=101,
            randomness=randomness,
        ),
        beacon_delay_rounds=1,
        first_window=500,
        window_count=window_count,
        warmup_rounds=2,
        window_schedule=WindowSchedule(
            mode="concurrent_checkpoint_epoch",
            collection_seconds=60.0,
            timeout_seconds=7200,
        ),
        training_mode="sequential_steps",
        target_groups_per_environment_lane=16,
        candidate_limit_per_environment_lane=24,
        environment_universes=environments or {"math": 1_000},
        prompt_range_size=50,
    )


def _checkpoint(plan):
    return SimpleNamespace(
        checkpoint_n=plan.checkpoint.number,
        repo_id=plan.checkpoint.repo_id,
        revision=plan.checkpoint.revision,
        signature="ed25519:test",
    )


def _server(plan=None):
    server = ValidatorServer()
    server.set_active_batcher(_server_batcher(window_start=500))
    server.set_current_state(WindowState.OPEN)
    if plan is not None:
        server.set_current_checkpoint(_checkpoint(plan))
        server.set_checkpoint_epoch_plan(plan)
    return server


def test_disabled_surface_keeps_legacy_state_compatible():
    client = TestClient(_server().app)

    state = client.get("/state")

    assert state.status_code == 200
    assert "checkpoint_epoch_id" not in state.json()
    assert "checkpoint_epoch_manifest_sha256" not in state.json()
    assert "checkpoint_epoch_candidate_limit" not in state.json()
    assert "checkpoint_epoch_candidate_remaining" not in state.json()
    assert "checkpoint_epoch_collection_seconds" not in state.json()
    assert client.get("/checkpoint-epoch").status_code == 404


def test_active_surface_serves_exact_immutable_manifest():
    plan = _plan()
    server = _server(plan)
    client = TestClient(server.app)

    state = client.get("/state").json()
    endpoint = client.get("/checkpoint-epoch")

    assert state["checkpoint_epoch_id"] == plan.epoch_id
    assert state["checkpoint_epoch_manifest_sha256"] == manifest_sha256(plan)
    assert endpoint.content == canonical_manifest_bytes(plan)
    assert endpoint.headers["etag"] == f'"{manifest_sha256(plan)}"'
    assert "seal_randomness" not in endpoint.text
    assert state["checkpoint_epoch_target_groups"] == 16
    assert state["checkpoint_epoch_candidate_limit"] == 24
    assert state["checkpoint_epoch_candidate_remaining"] is None
    assert state["checkpoint_epoch_collection_seconds"] == 60.0
    assert state["checkpoint_epoch_reveal_seconds"] == 60.0


def test_checkpoint_transition_withdraws_old_plan():
    plan = _plan()
    server = _server(plan)
    server.set_current_checkpoint(SimpleNamespace(
        checkpoint_n=8,
        repo_id=plan.checkpoint.repo_id,
        revision="5" * 40,
        signature="ed25519:new",
    ))

    assert TestClient(server.app).get("/checkpoint-epoch").status_code == 404


def test_server_rejects_same_epoch_id_with_another_beacon_result():
    first = _plan(randomness="4" * 64)
    second = _plan(randomness="5" * 64)
    assert first.epoch_id == second.epoch_id
    server = _server(first)

    with pytest.raises(ValueError, match="equivocation"):
        server.set_checkpoint_epoch_plan(second)


def test_concurrent_state_routes_exact_environment_and_window():
    plan = _plan()
    first = _server_batcher(window_start=500)
    second = _server_batcher(window_start=501)
    environment = str(first.env.name)
    first.randomness = plan.windows[0].generation_randomness
    second.randomness = plan.windows[1].generation_randomness
    server = ValidatorServer()
    server.set_current_checkpoint(_checkpoint(plan))
    server.set_checkpoint_epoch_plan(plan)
    server.set_active_epoch_batchers({
        (environment, 500): first,
        (environment, 501): second,
    })
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)

    first_state = client.get(f"/state?env={environment}&window=500")
    second_state = client.get(f"/state?env={environment}&window=501")

    assert first_state.status_code == second_state.status_code == 200
    assert first_state.json()["window_n"] == 500
    assert second_state.json()["window_n"] == 501
    assert first_state.json()["randomness"] != second_state.json()["randomness"]
    assert client.get(
        f"/state?env={environment}&window=999"
    ).status_code == 404


def test_concurrent_lanes_share_one_admission_policy_window():
    plan = _plan(window_count=16)
    first = _server_batcher(window_start=plan.first_window)
    last = _server_batcher(
        window_start=plan.first_window + plan.window_count - 1
    )
    environment = str(first.env.name)
    server = ValidatorServer()
    server.set_checkpoint_epoch_plan(plan)
    server.set_active_epoch_batchers({
        (environment, first.window_start): first,
        (environment, last.window_start): last,
    })

    assert server._admission_policy_window(first.window_start) == plan.first_window
    assert server._admission_policy_window(last.window_start) == plan.first_window

    server.set_active_epoch_batchers({})
    assert server._admission_policy_window(last.window_start) == last.window_start


class _EpochLaneBatcher:
    def __init__(self, window: int, randomness: str) -> None:
        self.window_start = window
        self.randomness = ""
        self.checkpoint_epoch_generation_randomness = randomness
        self.env = SimpleNamespace(name="fake")
        self.model = None
        self.opened_at = None
        self.opened_wall = None

    def set_prompt_range(self) -> None:
        pass

    def mark_window_opened(self, *, monotonic_time, wall_time) -> None:
        self.opened_at = monotonic_time
        self.opened_wall = wall_time

    def bind_event_loop(self, _loop) -> None:
        pass

    def force_seal(self, _reason) -> None:
        pass


def test_epoch_commitment_cannot_upload_until_post_commit_selection():
    plan = _plan(environments={"fake": 1_000})
    batcher = _server_batcher(window_start=plan.first_window)
    batcher.experimental_epoch_ranking = True
    batcher.difficulty_auction_enabled = True
    batcher.collection_seconds = plan.window_schedule.collection_seconds
    batcher.current_checkpoint_hash = plan.checkpoint.revision
    prompt_idx = plan.windows[0].prompt_slices[0].start
    request = _server_request(
        prompt_idx=prompt_idx,
        window_start=plan.first_window,
        checkpoint_hash=plan.checkpoint.revision,
        valid_merkle=True,
    )
    batcher._operator_by_hotkey = {request.miner_hotkey: "operator-a"}
    payload = request.model_dump_json().encode("utf-8")
    precommit = _precommit_for(request, payload_bytes=len(payload))

    server = ValidatorServer()
    server._auction_admission_enabled = True
    server.set_current_checkpoint(_checkpoint(plan))
    server.set_checkpoint_epoch_plan(plan)
    server.set_active_epoch_batchers({
        ("fake", plan.first_window): batcher,
    })
    server.set_registered_hotkeys(
        {request.miner_hotkey},
        operator_by_hotkey={request.miner_hotkey: "operator-a"},
    )
    server.set_checkpoint_epoch_phase("commitment")
    server.set_current_state(WindowState.OPEN)

    with TestClient(server.app) as client:
        committed = client.post(
            "/submit/precommit",
            content=precommit.model_dump_json(),
            headers={"Content-Type": "application/json"},
        ).json()
        assert committed["accepted"] is True
        assert committed["upload_deadline_ts"] is None
        assert batcher.pending_upload_precommits == 0
        early = client.post(
            "/submit",
            content=payload,
            headers={
                "Content-Type": "application/json",
                "X-Reliquary-Precommit": committed["receipt_id"],
            },
        )

    assert early.json()["reason"] == RejectReason.REVEAL_NOT_SELECTED.value
    assert batcher.proof_grading_attempts == 0


@pytest.mark.parametrize(
    ("round_reject", "accepted"),
    (
        (RejectReason.STALE_ROUND, True),
        (RejectReason.FUTURE_ROUND, False),
    ),
)
def test_epoch_commitment_uses_receipt_deadline_not_stale_round_bucket(
    round_reject,
    accepted,
):
    from reliquary.validator.observability import DrandRoundObservation

    plan = _plan(environments={"fake": 1_000})
    batcher = _server_batcher(window_start=plan.first_window)
    batcher.experimental_epoch_ranking = True
    batcher.drand_round_check_enabled = True
    batcher.collection_seconds = plan.window_schedule.collection_seconds
    batcher.current_checkpoint_hash = plan.checkpoint.revision
    batcher.observe_drand_round = lambda *_args, **_kwargs: DrandRoundObservation(
        submitted_drand_round=100,
        arrival_drand_round=101,
        drand_delta=-1,
        drand_tolerance=0,
        drand_status=(
            "stale" if round_reject is RejectReason.STALE_ROUND else "future"
        ),
        reject_reason=round_reject,
    )
    prompt_idx = plan.windows[0].prompt_slices[0].start
    request = _server_request(
        prompt_idx=prompt_idx,
        window_start=plan.first_window,
        checkpoint_hash=plan.checkpoint.revision,
        valid_merkle=True,
    )
    batcher._operator_by_hotkey = {request.miner_hotkey: "operator-a"}
    payload = request.model_dump_json().encode("utf-8")
    precommit = _precommit_for(request, payload_bytes=len(payload))
    server = ValidatorServer()
    server._auction_admission_enabled = True
    server.set_current_checkpoint(_checkpoint(plan))
    server.set_checkpoint_epoch_plan(plan)
    server.set_active_epoch_batchers({("fake", plan.first_window): batcher})
    server.set_registered_hotkeys(
        {request.miner_hotkey},
        operator_by_hotkey={request.miner_hotkey: "operator-a"},
    )
    server.set_checkpoint_epoch_phase("commitment")
    server.set_current_state(WindowState.OPEN)

    with TestClient(server.app) as client:
        response = client.post(
            "/submit/precommit",
            content=precommit.model_dump_json(),
            headers={"Content-Type": "application/json"},
        ).json()

    assert response["accepted"] is accepted
    assert response["reason"] == (
        RejectReason.ACCEPTED.value if accepted else round_reject.value
    )


def test_selected_epoch_commitment_gets_bounded_reveal_right():
    plan = _plan(environments={"fake": 1_000})
    batcher = _server_batcher(window_start=plan.first_window)
    batcher.experimental_epoch_ranking = True
    batcher.difficulty_auction_enabled = True
    batcher.collection_seconds = plan.window_schedule.collection_seconds
    batcher.current_checkpoint_hash = plan.checkpoint.revision
    prompt_idx = plan.windows[0].prompt_slices[0].start
    request = _server_request(
        prompt_idx=prompt_idx,
        window_start=plan.first_window,
        checkpoint_hash=plan.checkpoint.revision,
        valid_merkle=True,
    )
    batcher._operator_by_hotkey = {request.miner_hotkey: "operator-a"}
    payload = request.model_dump_json().encode("utf-8")
    precommit = _precommit_for(request, payload_bytes=len(payload))
    server = ValidatorServer()
    server._auction_admission_enabled = True
    server.set_current_checkpoint(_checkpoint(plan))
    server.set_checkpoint_epoch_plan(plan)
    server.set_active_epoch_batchers({("fake", plan.first_window): batcher})
    server.set_registered_hotkeys(
        {request.miner_hotkey},
        operator_by_hotkey={request.miner_hotkey: "operator-a"},
    )
    server.set_checkpoint_epoch_phase("commitment")
    server.set_current_state(WindowState.OPEN)

    with TestClient(server.app) as client:
        committed = client.post(
            "/submit/precommit",
            content=precommit.model_dump_json(),
            headers={"Content-Type": "application/json"},
        ).json()
        commitment_close_round = plan.epoch_beacon.round + 9
        server.set_checkpoint_epoch_phase("selection")
        frozen = server.freeze_checkpoint_epoch_commitment_set(
            commitment_close_round=commitment_close_round,
            validator_hotkey="validator",
        )
        server.install_checkpoint_epoch_commitment_set(
            SignedEpochCommitmentSet(
                commitment_set=frozen,
                commitment_set_sha256=commitment_set_sha256(frozen),
                validator_signature="aa",
            )
        )
        with pytest.raises(ValueError, match="follow commitment close"):
            server.select_checkpoint_epoch_reveals(
                commitment_close_round=commitment_close_round,
                admission_beacon=BeaconBinding(
                    source="drand",
                    chain=plan.epoch_beacon.chain,
                    chain_hash=plan.epoch_beacon.chain_hash,
                    round=commitment_close_round,
                    randomness="e" * 64,
                ),
                reveal_deadline_ts=time.time() + 60.0,
            )
        counts = server.select_checkpoint_epoch_reveals(
            commitment_close_round=commitment_close_round,
            admission_beacon=BeaconBinding(
                source="drand",
                chain=plan.epoch_beacon.chain,
                chain_hash=plan.epoch_beacon.chain_hash,
                round=commitment_close_round + 1,
                randomness="e" * 64,
            ),
            reveal_deadline_ts=time.time() + 60.0,
        )
        status = client.get(
            f"/checkpoint-epoch/commitments/{committed['receipt_id']}"
        ).json()
        state = client.get(
            f"/state?env=fake&window={plan.first_window}"
        ).json()
        batcher.window_opened_wall_ts = (
            time.time() - plan.window_schedule.collection_seconds - 1.0
        )
        revealed = client.post(
            "/submit",
            content=payload,
            headers={
                "Content-Type": "application/json",
                "X-Reliquary-Precommit": committed["receipt_id"],
            },
        ).json()

    assert counts == {("fake", plan.first_window): 1}
    assert status["status"] == "selected"
    assert status["admission_beacon_round"] == commitment_close_round + 1
    assert state["checkpoint_epoch_phase"] == "reveal"
    assert state["checkpoint_epoch_admission_beacon_round"] == (
        commitment_close_round + 1
    )
    assert revealed["reason"] not in {
        RejectReason.REVEAL_NOT_SELECTED.value,
        RejectReason.PRECOMMIT_EXPIRED.value,
    }


def test_real_epoch_batcher_builder_honors_each_requested_logical_window(
    monkeypatch,
):
    import reliquary.validator.batcher as batcher_module
    from tests.unit.test_service_v2 import _build_late_drop_service

    monkeypatch.setattr(
        batcher_module,
        "DIFFICULTY_AUCTION_ENVIRONMENTS",
        frozenset({"fake"}),
    )
    plan = _plan(window_count=2, environments={"fake": 1_000})
    service = _build_late_drop_service()
    service._checkpoint_epoch_plan = plan
    service._candidate_window_n = plan.first_window

    first = service._build_window_batchers(plan.first_window)["fake"]
    second = service._build_window_batchers(plan.first_window + 1)["fake"]

    assert first.window_start == plan.first_window
    assert second.window_start == plan.first_window + 1
    assert first.randomness == ""
    assert second.randomness == ""
    assert first.collection_seconds == second.collection_seconds == 60.0
    assert first.max_productive_candidates == 24
    assert second.max_ranked_proof_attempts == 24


@pytest.mark.asyncio
async def test_epoch_runner_opens_sixteen_lanes_together_then_consumes_in_order(
    tmp_path,
):
    from reliquary.validator.checkpoint_epoch_runtime import EpochStore
    from tests.unit.test_service_v2 import _build_late_drop_service

    plan = _plan(window_count=16, environments={"fake": 1_000})
    service = _build_late_drop_service()
    service._checkpoint_epoch_plan = plan
    service.server.set_checkpoint_epoch_plan(plan)
    service._checkpoint_epoch_store = EpochStore(tmp_path)
    service._window_n = plan.first_window - 1
    chain_info = {
        "name": plan.epoch_beacon.chain,
        "hash": plan.epoch_beacon.chain_hash,
        "genesis_time": 0,
        "period": 30,
    }
    service._checkpoint_epoch_drand_snapshot = AsyncMock(
        side_effect=[(chain_info, 120), (chain_info, 149)]
    )
    built: list[_EpochLaneBatcher] = []

    def build(window_number: int):
        epoch_window = plan.windows[window_number - plan.first_window]
        batcher = _EpochLaneBatcher(
            window_number,
            epoch_window.generation_randomness,
        )
        built.append(batcher)
        return {"fake": batcher}

    service._build_window_batchers = build
    service._wait_for_checkpoint_epoch_phase_deadline = AsyncMock()
    service.server.drain_checkpoint_epoch_commitments = AsyncMock()
    admission_beacon = BeaconBinding(
        source="drand",
        chain=plan.epoch_beacon.chain,
        chain_hash=plan.epoch_beacon.chain_hash,
        round=150,
        randomness="a" * 64,
    )
    service._fetch_checkpoint_epoch_admission_beacon = AsyncMock(
        return_value=(149, admission_beacon)
    )
    service._fetch_checkpoint_epoch_seal_beacon = AsyncMock(
        return_value=(200, replace(plan.epoch_beacon, round=201))
    )
    service._freeze_auction_populations = AsyncMock(return_value={})
    service._train_and_publish = AsyncMock()

    await service._run_checkpoint_epoch()

    assert len(built) == 16
    assert len({batcher.opened_at for batcher in built}) == 1
    assert len({batcher.opened_wall for batcher in built}) == 1
    assert service._wait_for_checkpoint_epoch_phase_deadline.await_count == 2
    service.server.drain_checkpoint_epoch_commitments.assert_awaited_once_with()
    service._fetch_checkpoint_epoch_seal_beacon.assert_awaited_once_with(
        after_round=admission_beacon.round
    )
    assert [
        call.kwargs["window_n"]
        for call in service._train_and_publish.await_args_list
    ] == list(range(plan.first_window, plan.first_window + 16))
    assert [
        call.kwargs["epoch_finalize"]
        for call in service._train_and_publish.await_args_list
    ] == [False] * 15 + [True]
    assert service.server._active_batcher_values() == ()
    assert service._current_window_state is WindowState.READY
    assert service._checkpoint_epoch_store.terminal_status(plan) == "completed"
    pipeline = service.server._checkpoint_epoch_pipeline_state
    assert pipeline["finalization_policy"] == plan.finalization_policy
    assert pipeline["phase"] == "terminal"
    assert pipeline["terminal_status"] == "completed"
    assert pipeline["lanes_finalized"] == 16
    assert list(pipeline["lane_metrics"]) == [str(index) for index in range(16)]


@pytest.mark.asyncio
async def test_epoch_failure_closes_routes_and_tombstones_unconsumed_lanes(
    tmp_path,
):
    from reliquary.validator.checkpoint_epoch_runtime import EpochStore
    from reliquary.validator.service import CheckpointEpochExecutionError
    from tests.unit.test_service_v2 import _build_late_drop_service

    plan = _plan(window_count=16, environments={"fake": 1_000})
    service = _build_late_drop_service()
    service._checkpoint_epoch_plan = plan
    service.server.set_checkpoint_epoch_plan(plan)
    service._checkpoint_epoch_store = EpochStore(tmp_path)
    service._window_n = plan.first_window - 1
    chain_info = {
        "name": plan.epoch_beacon.chain,
        "hash": plan.epoch_beacon.chain_hash,
        "genesis_time": 0,
        "period": 30,
    }
    service._checkpoint_epoch_drand_snapshot = AsyncMock(
        side_effect=[(chain_info, 120), (chain_info, 149)]
    )

    def build(window_number: int):
        epoch_window = plan.windows[window_number - plan.first_window]
        return {
            "fake": _EpochLaneBatcher(
                window_number,
                epoch_window.generation_randomness,
            )
        }

    service._build_window_batchers = build
    service._wait_for_checkpoint_epoch_phase_deadline = AsyncMock()
    admission_beacon = BeaconBinding(
        source="drand",
        chain=plan.epoch_beacon.chain,
        chain_hash=plan.epoch_beacon.chain_hash,
        round=150,
        randomness="a" * 64,
    )
    service._fetch_checkpoint_epoch_admission_beacon = AsyncMock(
        return_value=(149, admission_beacon)
    )
    service._fetch_checkpoint_epoch_seal_beacon = AsyncMock(
        return_value=(200, replace(plan.epoch_beacon, round=201))
    )
    service._freeze_auction_populations = AsyncMock(return_value={})
    service._train_and_publish = AsyncMock(
        side_effect=[None, RuntimeError("stop")]
    )
    service._enqueue_aborted_window = Mock()

    with pytest.raises(CheckpointEpochExecutionError):
        await service._run_checkpoint_epoch()

    assert service._enqueue_aborted_window.call_count == 15
    assert service.server._active_batcher_values() == ()
    assert service._active_batchers == {}
    assert service._window_n == plan.first_window + plan.window_count - 1
    assert service._checkpoint_epoch_store.terminal_status(plan) == "aborted"
    assert service._current_window_state is WindowState.READY
    pipeline = service.server._checkpoint_epoch_pipeline_state
    assert pipeline["phase"] == "terminal"
    assert pipeline["terminal_status"] == "aborted"
    assert pipeline["failure_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_completed_epoch_cannot_reopen_without_successor_checkpoint(
    monkeypatch,
):
    import reliquary.validator.service as service_module
    from reliquary.validator.service import CheckpointEpochExecutionError
    from tests.unit.test_service_v2 import _build_late_drop_service

    plan = _plan(window_count=16, environments={"fake": 1_000})
    service = _build_late_drop_service()
    monkeypatch.setattr(
        service_module,
        "EXPERIMENTAL_CHECKPOINT_EPOCH_ENABLED",
        True,
    )
    monkeypatch.setattr(
        service_module,
        "CHECKPOINT_PUBLISH_INTERVAL_WINDOWS",
        16,
    )
    service.use_drand = True
    service._checkpoint_epoch_plan = plan
    service._window_n = plan.first_window + plan.window_count - 1
    service._checkpoint_store.current_manifest = Mock(
        return_value=_checkpoint(plan)
    )
    service._validate_checkpoint_epoch_runtime_config = Mock()

    with pytest.raises(
        CheckpointEpochExecutionError,
        match="without a successor checkpoint",
    ):
        await service._ensure_checkpoint_epoch_plan()


@pytest.mark.asyncio
async def test_restart_retires_activated_epoch_without_reopening_it(
    monkeypatch,
    tmp_path,
):
    import reliquary.validator.service as service_module
    from reliquary.validator.checkpoint_epoch_runtime import EpochStore
    from reliquary.validator.service import CheckpointEpochExecutionError
    from tests.unit.test_service_v2 import _build_late_drop_service

    plan = _plan(window_count=16, environments={"fake": 1_000})
    store = EpochStore(tmp_path)
    store.mark_activated(plan)
    store.load_current_plan = Mock(return_value=plan)
    service = _build_late_drop_service()
    monkeypatch.setattr(
        service_module,
        "EXPERIMENTAL_CHECKPOINT_EPOCH_ENABLED",
        True,
    )
    monkeypatch.setattr(
        service_module,
        "CHECKPOINT_PUBLISH_INTERVAL_WINDOWS",
        16,
    )
    service.use_drand = True
    service._window_n = plan.first_window - 1
    service._checkpoint_epoch_store = store
    service._checkpoint_store.current_manifest = Mock(
        return_value=_checkpoint(plan)
    )
    service._write_training_tombstone = Mock()

    with pytest.raises(
        CheckpointEpochExecutionError,
        match="requires a successor checkpoint",
    ):
        await service._ensure_checkpoint_epoch_plan()

    assert store.terminal_status(plan) == "aborted"
    assert service._write_training_tombstone.call_count == plan.window_count
    assert service._window_n == plan.first_window + plan.window_count - 1
    assert service._checkpoint_epoch_plan is None


def _accept_with_arrival(batcher, request, round_number: int) -> None:
    telemetry = SubmitTelemetry(
        window_n=batcher.window_start,
        prompt_idx=request.prompt_idx,
        hotkey=request.miner_hotkey,
        merkle_root="00" * 32,
        protocol_version=2,
        submitted_drand_round=round_number,
        t_arrival=float(round_number),
        prompt_hash_lead="",
        merkle_root_lead="",
        precommit_arrival_ts=float(round_number),
        arrival_drand_round=round_number,
    )
    assert batcher.accept_submission(request, telemetry=telemetry).accepted


def _epoch_rank_map(arrivals, seal_randomness="fresh-seal"):
    operators = {f"miner-{index}": f"operator-{index}" for index in range(12)}
    batcher = _make_batcher(
        operator_by_hotkey=operators,
        experimental_epoch_ranking=True,
    )
    rewards = [1.0, 1.0] + [0.0] * 6
    for index, arrival in enumerate(arrivals):
        _accept_with_arrival(
            batcher,
            _request(
                prompt_idx=index,
                hotkey=f"miner-{index}",
                rewards=rewards,
            ),
            arrival,
        )
    batcher.current_checkpoint_hash = "2" * 40
    batcher.collection_close_drand_round = 200
    batcher.seal_randomness = seal_randomness
    batcher.seal_beacon_round = 201
    batcher.seal_batch()
    return {
        row["prompt_idx"]: row["rank"]
        for row in batcher.auction_candidates
    }


def _production_rank_order(explicit_false: bool):
    operators = {f"miner-{index}": f"operator-{index}" for index in range(12)}
    kwargs = {"operator_by_hotkey": operators}
    if explicit_false:
        kwargs["experimental_epoch_ranking"] = False
    batcher = _make_batcher(**kwargs)
    rewards = [1.0, 1.0] + [0.0] * 6
    for index, arrival in enumerate(range(100, 112)):
        _accept_with_arrival(
            batcher,
            _request(
                prompt_idx=index,
                hotkey=f"miner-{index}",
                rewards=rewards,
            ),
            arrival,
        )
    batcher.current_checkpoint_hash = "2" * 40
    batcher.window_open_drand_round = 100
    batcher.seal_randomness = "fresh-seal"
    batcher.seal_batch()
    return [
        row["prompt_idx"]
        for row in sorted(
            batcher.auction_candidates,
            key=lambda item: item["rank"],
        )
    ]


def test_production_ranking_default_path_is_unchanged():
    assert _production_rank_order(False) == _production_rank_order(True)


def test_epoch_ranking_ignores_arrival_and_throughput(monkeypatch):
    import reliquary.validator.batcher as batcher_module

    monkeypatch.setattr(
        batcher_module,
        "throughput_rank",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("epoch ranking consulted throughput")
        ),
    )
    assert _epoch_rank_map(range(100, 112)) == _epoch_rank_map(
        reversed(range(100, 112))
    )


@pytest.mark.parametrize(
    ("randomness", "round_number"),
    (("", 201), ("fresh", None), ("fresh", 200)),
)
def test_epoch_ranking_requires_fresh_post_close_beacon(
    randomness,
    round_number,
):
    batcher = _make_batcher(
        operator_by_hotkey={"miner": "operator"},
        experimental_epoch_ranking=True,
    )
    _accept_with_arrival(
        batcher,
        _request(prompt_idx=1, hotkey="miner"),
        100,
    )
    batcher.collection_close_drand_round = 200
    batcher.seal_randomness = randomness
    batcher.seal_beacon_round = round_number

    with pytest.raises(RuntimeError, match="experimental epoch"):
        batcher.seal_batch()


def test_epoch_prompt_slice_is_enforced_even_before_production_cutover():
    batcher = _make_batcher(
        operator_by_hotkey={"miner": "operator"},
        experimental_epoch_ranking=True,
        experimental_prompt_range=(10, 20),
    )
    batcher.randomness = "1" * 64
    batcher.set_prompt_range()

    assert batcher.prompt_range == (10, 20)
    assert not batcher.accept_submission(
        _request(prompt_idx=9, hotkey="miner")
    ).accepted


def test_epoch_collection_and_candidate_limit_are_instance_bound():
    now = [0.0]
    operators = {f"miner-{index}": f"operator-{index}" for index in range(25)}
    batcher = _make_batcher(
        operator_by_hotkey=operators,
        experimental_epoch_ranking=True,
        collection_seconds=1_600.0,
        max_productive_candidates=24,
        max_ranked_proof_attempts=24,
        time_fn=lambda: now[0],
        wall_clock_fn=lambda: now[0],
    )
    batcher.mark_window_opened(monotonic_time=0.0, wall_time=0.0)

    now[0] = 100.0
    assert batcher.poll_deadline() is False
    for index in range(24):
        assert batcher.try_reserve_proof_admission(
            _request(prompt_idx=index, hotkey=f"miner-{index}")
        ) == (True, None)
    assert batcher.try_reserve_proof_admission(
        _request(prompt_idx=24, hotkey="miner-24")
    ) == (False, "proof_grading_attempts_full")

    now[0] = 1_601.0
    assert batcher.poll_deadline() is True


def test_epoch_collection_ignores_production_early_close(monkeypatch):
    import reliquary.validator.batcher as batcher_module
    from reliquary.constants import MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW

    monkeypatch.setattr(batcher_module, "AUCTION_EARLY_CLOSE_MODE", "enforce")
    now = [0.0]
    batcher = _make_batcher(
        operator_by_hotkey={"miner": "operator"},
        experimental_epoch_ranking=True,
        collection_seconds=1_600.0,
        time_fn=lambda: now[0],
        wall_clock_fn=lambda: now[0],
    )
    batcher.mark_window_opened(monotonic_time=0.0, wall_time=0.0)
    batcher._proof_grading_charged = MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW

    now[0] = 100.0
    assert batcher.poll_deadline() is False
    assert batcher.early_close_eligible_at is None
    assert batcher.early_close_sealed is False

    now[0] = 1_601.0
    assert batcher.poll_deadline() is True
    assert batcher.early_close_sealed is False
