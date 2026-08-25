from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from reliquary.protocol.submission import WindowState
from reliquary.shared.checkpoint_epoch import (
    BeaconBinding,
    CheckpointBinding,
    ProtocolBinding,
    WindowSchedule,
    build_epoch_plan,
    canonical_manifest_bytes,
    manifest_sha256,
)
from reliquary.validator.observability import SubmitTelemetry
from reliquary.validator.server import ValidatorServer
from tests.unit.test_grpo_window_batcher import _make_batcher, _request
from tests.unit.test_validator_server import _batcher as _server_batcher


def _plan(*, randomness: str = "4" * 64):
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
        window_count=2,
        warmup_rounds=2,
        window_schedule=WindowSchedule(
            mode="ordinary_window_state_machine",
            collection_seconds=60.0,
            timeout_seconds=7200,
        ),
        environment_universes={"math": 1_000},
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
