"""Final service verdicts describe the durable entitlement of every non-trained observation, including aborts.

Rewritten for service-contract/v2 (the v1 runtime this file imported is gone): the statuses are the
ones the exploration lane publishes (``tests/unit/test_service_exploration_lane.py`` produces them
through the real batcher and runtime; here they are fed to the verdict publisher directly).
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from reliquary.validator.service import ValidationService
from tests.unit.service_v2_fixtures import contract_v2


def service_and_batcher(status="exploration_pending", fraction=0.01, **row_fields):
    runtime = SimpleNamespace(contract=contract_v2())
    request = SimpleNamespace(merkle_root="e" * 64, service_binding={"purpose": "exploration"})
    pending = SimpleNamespace(hotkey="fixture-miner", prompt_idx=0, merkle_root=b"a", request=request,
                              reject_response=None, telemetry=None, rewards=[0.0, 0.0])
    row = {"status": status, "exploration_fraction": fraction, "proof_status": "passed", **row_fields}
    batcher = SimpleNamespace(window_start=1, difficulty_auction_enabled=True, service_runtime=runtime,
                              difficulty_auction_metadata_by_id={id(pending): row}, env=SimpleNamespace(name="math"),
                              pending_submissions=lambda: [pending], current_checkpoint_hash="d" * 40,
                              finalize_service_exploration=MagicMock())
    service = ValidationService.__new__(ValidationService)
    service._service_runtime = runtime
    service.server = SimpleNamespace(record_verdict=MagicMock(return_value={"record": True}),
                                     persist_final_verdicts=MagicMock(), complete_final_verdict_window=MagicMock())
    return service, batcher, row


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["exploration_pending", "exploration_audit_passed"])
async def test_a_paid_exploration_is_rewarded_without_being_selected_for_training(status):
    service, batcher, _ = service_and_batcher(status)
    await service._record_auction_final_verdicts(batcher, paid_groups=[])
    args = service.server.record_verdict.call_args.kwargs
    assert args["rewarded"] is True and args["selected_for_batch"] is False
    assert args["selection_reason"] == "exploration_reward_recorded"
    service.server.persist_final_verdicts.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [
    "exploration_forfeited", "exploration_unaudited", "already_scanned", "exploration_banned",
    "exploration_cap_reached", "exploration_unpaid", "exploration_audit_queued", "service_unproven_published",
])
async def test_every_unpaid_status_is_published_as_its_own_reason(status):
    service, batcher, _ = service_and_batcher(status, fraction=0.0)
    await service._record_auction_final_verdicts(batcher, paid_groups=[])
    args = service.server.record_verdict.call_args.kwargs
    assert args["rewarded"] is False and args["selected_for_batch"] is False
    assert args["selection_reason"] == status


@pytest.mark.asyncio
async def test_a_pending_row_with_no_fraction_is_not_paid():
    service, batcher, _ = service_and_batcher("exploration_pending", fraction=0.0)
    await service._record_auction_final_verdicts(batcher, paid_groups=[])
    args = service.server.record_verdict.call_args.kwargs
    assert args["rewarded"] is False and args["selection_reason"] == "exploration_pending"


@pytest.mark.asyncio
async def test_a_window_flagged_aborted_pays_nothing_even_for_a_pending_row():
    service, batcher, _ = service_and_batcher("exploration_pending", window_aborted=True)
    await service._record_auction_final_verdicts(batcher, paid_groups=[])
    args = service.server.record_verdict.call_args.kwargs
    assert args["rewarded"] is False and args["selection_reason"] == "exploration_pending"


def test_aborted_exploration_final_verdict_cannot_claim_its_burned_reward(monkeypatch, tmp_path):
    import reliquary.validator.service as module
    import reliquary.infrastructure.archive_queue as archive_module
    service, batcher, row = service_and_batcher("exploration_pending")
    monkeypatch.setattr(module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(archive_module, "get_archive_queue", lambda: object())
    service._active_batchers = {"math": batcher}
    service._archive_enqueued_windows = set()
    service._fill_closed_assemblers = {1: SimpleNamespace()}
    service._fill_closed_assembler = None
    service._close_and_commit_fill_closed_paid_side_effects = lambda *_: {"math": []}
    service._training_payload_queue_ref = lambda: SimpleNamespace(queue_dir=tmp_path)
    service._fill_closed_recovery_store = SimpleNamespace(quarantine_uncommitted=lambda *_: None,
        recover=lambda *_args, **_kw: {"window_start": 1, "window_status": "aborted", "rewards_by_hotkey": {}})
    service._fill_closed_rotation_store = SimpleNamespace(load=lambda: {})
    service._service_sealed_windows = set()
    service._service_recovery_attempts = {}
    service._service_recovery_context = {}
    service._cache_archived_hashes = lambda _: None
    service._enqueue_aborted_window(failure_stage="fixture", failure_type="interrupted")
    batcher.finalize_service_exploration.assert_called_once()      # what is open ends unaudited before the recovery
    args = service.server.record_verdict.call_args.kwargs
    assert args["rewarded"] is False and args["selected_for_batch"] is False
    assert args["selection_reason"] == "exploration_pending"
    assert row["exploration_fraction"] == 0.0 and row["window_aborted"] is True
    service.server.persist_final_verdicts.assert_called_once()
    service.server.complete_final_verdict_window.assert_called_once_with(1)
