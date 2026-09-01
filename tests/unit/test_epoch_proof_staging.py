from __future__ import annotations

import time

import pytest

from reliquary.validator.epoch_proof_staging import (
    EpochProofBinding,
    EpochProofStagingError,
    EpochProofStreamingUnsupported,
    TicketedEpochProofCoordinator,
    TicketedProofCandidate,
    require_streaming_runtime,
    streaming_runtime_blockers,
)
from reliquary.validator.proof_scheduler import (
    GlobalProofScheduler,
    ProofPlan,
    ProofSchedulerCapabilities,
)


def _binding(*, window_count: int = 2) -> EpochProofBinding:
    return EpochProofBinding(
        epoch_id="a" * 64,
        manifest_sha256="b" * 64,
        checkpoint_revision="immutable-checkpoint-revision",
        protocol_profile_id="experimental-ticketed-epoch",
        generation_contract_sha256="c" * 64,
        first_window=100,
        window_count=window_count,
        environments=("math", "code"),
        generation_randomness_by_offset=tuple(
            f"{offset + 1:064x}" for offset in range(window_count)
        ),
    )


def _candidate(
    rank: int,
    *,
    intent_id: str | None = None,
    window_number: int = 100,
    environment: str = "math",
    ticket_state: str = "primary",
    payload_digit: str | None = None,
) -> TicketedProofCandidate:
    return TicketedProofCandidate(
        intent_id=intent_id or f"intent-{rank}",
        operator_id=f"operator-{rank}",
        window_number=window_number,
        environment=environment,
        prompt_idx=rank,
        payload_sha256=(payload_digit or f"{rank + 4:x}") * 64,
        generation_randomness=f"{window_number - 99:064x}",
        selection_rank=rank,
        ticket_state=ticket_state,
    )


def test_current_scheduler_advertises_the_epoch_streaming_blockers():
    scheduler = GlobalProofScheduler(
        devices=("cpu",),
        environments=("math",),
        proof_callable=lambda _invocation: True,
        checkpoint_revision="revision",
    )
    try:
        assert scheduler.capabilities == ProofSchedulerCapabilities(
            max_live_plans_per_environment=1,
            durable_result_recovery=False,
            supports_predeclared_candidates=False,
            supports_rank_independent_extension=False,
        )
        blockers = streaming_runtime_blockers(
            scheduler.capabilities,
            concurrent_lanes=16,
        )
        assert blockers == (
            "insufficient_lane_isolation",
            "proof_results_are_not_durable",
            "ticket_slots_cannot_be_predeclared",
            "plan_extension_is_append_ordered",
        )
        with pytest.raises(EpochProofStreamingUnsupported, match="unavailable"):
            require_streaming_runtime(
                scheduler.capabilities,
                concurrent_lanes=16,
            )
    finally:
        scheduler.close()


def test_scheduler_rejects_a_second_live_plan_for_the_same_environment():
    scheduler = GlobalProofScheduler(
        devices=("cpu",),
        environments=("math",),
        proof_callable=lambda _invocation: True,
        checkpoint_revision="revision",
    )
    try:
        scheduler.submit(
            ProofPlan(
                plan_id="lane-0",
                environment="math",
                checkpoint_revision="revision",
                candidates=(),
                required_passes=1,
                deadline_at=time.monotonic() + 10.0,
                open_ended=True,
            )
        )
        with pytest.raises(ValueError, match="already has a plan"):
            scheduler.submit(
                ProofPlan(
                    plan_id="lane-1",
                    environment="math",
                    checkpoint_revision="revision",
                    candidates=(),
                    required_passes=1,
                    deadline_at=time.monotonic() + 10.0,
                    open_ended=True,
                )
            )
    finally:
        scheduler.close()


def test_only_activated_tickets_can_be_staged():
    with pytest.raises(ValueError, match="not activated"):
        _candidate(0, ticket_state="standby")
    with pytest.raises(ValueError, match="not activated"):
        _candidate(0, ticket_state="not_selected")


def test_stage_binds_randomness_and_rejects_intent_or_payload_rebinding():
    coordinator = TicketedEpochProofCoordinator(_binding())
    first = _candidate(0)
    assert coordinator.stage(first) is True
    assert coordinator.stage(first) is False

    with pytest.raises(EpochProofStagingError, match="rebound"):
        coordinator.stage(
            _candidate(0, intent_id=first.intent_id, payload_digit="e")
        )
    with pytest.raises(EpochProofStagingError, match="already staged"):
        coordinator.stage(
            _candidate(1, intent_id="other-intent", payload_digit="4")
        )
    changed_randomness = TicketedProofCandidate(
        **{
            **first.__dict__,
            "intent_id": "changed-randomness",
            "payload_sha256": "f" * 64,
            "generation_randomness": "9" * 64,
        }
    )
    with pytest.raises(EpochProofStagingError, match="randomness changed"):
        coordinator.stage(changed_randomness)


def test_frozen_dispatch_order_uses_ticket_rank_not_stage_order():
    coordinator = TicketedEpochProofCoordinator(_binding())
    for rank in (2, 0, 1):
        coordinator.stage(_candidate(rank))
    population = coordinator.freeze()
    assert len(population) == 64
    assert [
        candidate.selection_rank
        for candidate in coordinator.dispatch_order((100, "math"))
    ] == [0, 1, 2]

    with pytest.raises(EpochProofStagingError, match="ticket order"):
        coordinator.mark_dispatched("intent-2")
    assert coordinator.mark_dispatched("intent-0") is True
    assert coordinator.mark_dispatched("intent-0") is False
    assert coordinator.mark_dispatched("intent-1") is True
    assert coordinator.mark_dispatched("intent-2") is True


def test_lane_fails_closed_until_every_proof_is_terminal():
    coordinator = TicketedEpochProofCoordinator(_binding())
    coordinator.stage(_candidate(0))
    coordinator.stage(_candidate(1))
    coordinator.freeze()
    coordinator.mark_dispatched("intent-0")
    coordinator.mark_dispatched("intent-1")
    coordinator.record_terminal(
        "intent-0", passed=True, result_sha256="1" * 64
    )

    with pytest.raises(EpochProofStagingError, match="intent-1"):
        coordinator.assert_lane_terminal((100, "math"))
    coordinator.record_terminal(
        "intent-1", passed=False, result_sha256="2" * 64
    )
    assert coordinator.record_terminal(
        "intent-1", passed=False, result_sha256="2" * 64
    ) is False
    coordinator.assert_lane_terminal((100, "math"))
    with pytest.raises(EpochProofStagingError, match="cannot become terminal"):
        coordinator.record_terminal(
            "intent-1", passed=True, result_sha256="3" * 64
        )


def test_recovery_quarantines_inflight_work_and_never_reproofs_it():
    coordinator = TicketedEpochProofCoordinator(_binding())
    coordinator.stage(_candidate(0))
    coordinator.freeze()
    coordinator.mark_dispatched("intent-0")

    recovered = TicketedEpochProofCoordinator.from_snapshot_bytes(
        coordinator.snapshot_bytes()
    )
    assert recovered.quarantined_intent_ids() == ("intent-0",)
    with pytest.raises(EpochProofStagingError, match="quarantined"):
        recovered.mark_dispatched("intent-0")
    with pytest.raises(EpochProofStagingError, match="intent-0"):
        recovered.assert_lane_terminal((100, "math"))


def test_terminal_snapshot_is_canonical_and_byte_identical_on_restore():
    coordinator = TicketedEpochProofCoordinator(_binding())
    coordinator.stage(_candidate(0))
    coordinator.freeze()
    coordinator.mark_dispatched("intent-0")
    coordinator.record_terminal(
        "intent-0", passed=True, result_sha256="1" * 64
    )
    raw = coordinator.snapshot_bytes()
    recovered = TicketedEpochProofCoordinator.from_snapshot_bytes(raw)
    assert recovered.snapshot_bytes() == raw


def test_lane_finalization_claim_is_idempotent_and_cannot_change():
    coordinator = TicketedEpochProofCoordinator(_binding())
    for rank in (0, 1):
        coordinator.stage(_candidate(rank))
    coordinator.freeze()
    for rank in (0, 1):
        coordinator.mark_dispatched(f"intent-{rank}")
        coordinator.record_terminal(
            f"intent-{rank}",
            passed=True,
            result_sha256=f"{rank + 1:x}" * 64,
        )

    first = coordinator.claim_lane_finalization(
        (100, "math"), ("intent-0",)
    )
    replay = coordinator.claim_lane_finalization(
        (100, "math"), ("intent-0",)
    )
    assert first[1] is True
    assert replay[1] is False
    assert replay[0] == first[0]
    with pytest.raises(EpochProofStagingError, match="already claimed differently"):
        coordinator.claim_lane_finalization(
            (100, "math"), ("intent-1",)
        )


def test_fully_capable_scheduler_contract_opens_the_activation_guard():
    capabilities = ProofSchedulerCapabilities(
        max_live_plans_per_environment=16,
        durable_result_recovery=True,
        supports_predeclared_candidates=True,
        supports_rank_independent_extension=True,
    )
    assert streaming_runtime_blockers(
        capabilities, concurrent_lanes=16
    ) == ()
    require_streaming_runtime(capabilities, concurrent_lanes=16)
