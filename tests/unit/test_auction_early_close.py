"""Adaptive, GPU-aware auction collection between 60 and 100 seconds.

The old dominance rule sealed as soon as 64 short candidates filled productive
admission. Production archives showed that this systematically excluded later,
larger answers. The replacement keeps 100 seconds as a hard ceiling, expands
productive admission to 96, and permits an earlier close only after the primary
64-candidate population exists, the previous GPU half has finished, all upload
and grading work has drained, and arrivals have been quiet for a drand round.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading

import pytest

from tests.unit.test_grpo_window_batcher import _make_batcher


def _clock_batcher(mode: str, monkeypatch, *, drand_period: float = 3.0):
    import reliquary.validator.batcher as batcher_module

    monkeypatch.setattr(batcher_module, "AUCTION_EARLY_CLOSE_MODE", mode)
    now = [1000.0]
    wall = [10_000.0]
    batcher = _make_batcher(
        time_fn=lambda: now[0],
        wall_clock_fn=lambda: wall[0],
        drand_chain_info={"genesis_time": 0, "period": drand_period},
    )
    batcher.mark_window_opened()

    def advance(seconds: float) -> None:
        now[0] += seconds
        wall[0] += seconds

    return batcher, advance


def _primary_population(batcher, *, idle_seconds: float = 3.0) -> None:
    from reliquary.constants import (
        B_BATCH,
        PRIMARY_PROOF_GRADING_ATTEMPTS_PER_WINDOW,
    )

    # Use the batcher's public lock-free counters so this helper tests the seal
    # policy rather than running 64 complete reward-grading jobs per case.
    batcher.pending_count = PRIMARY_PROOF_GRADING_ATTEMPTS_PER_WINDOW
    batcher._proof_grading_charged = (
        PRIMARY_PROOF_GRADING_ATTEMPTS_PER_WINDOW
    )
    batcher.last_valid_submission_at = batcher._time_fn() - idle_seconds
    batcher.distinct_pending_prompt_count = lambda: B_BATCH


def test_never_closes_before_the_60_second_minimum(monkeypatch):
    from reliquary.constants import AUCTION_EARLY_CLOSE_MIN_SECONDS

    batcher, advance = _clock_batcher("enforce", monkeypatch)
    _primary_population(batcher, idle_seconds=20.0)

    advance(AUCTION_EARLY_CLOSE_MIN_SECONDS - 0.1)
    assert batcher.poll_deadline(pipeline_ready=True) is False
    assert batcher.early_close_blocker == "minimum_collection"

    advance(0.1)
    assert batcher.poll_deadline(pipeline_ready=True) is True
    assert batcher.early_close_sealed is True


def test_previous_gpu_half_must_finish_before_adaptive_close(monkeypatch):
    from reliquary.constants import AUCTION_EARLY_CLOSE_MIN_SECONDS

    batcher, advance = _clock_batcher("enforce", monkeypatch)
    advance(AUCTION_EARLY_CLOSE_MIN_SECONDS + 5.0)
    _primary_population(batcher, idle_seconds=5.0)

    assert batcher.poll_deadline(pipeline_ready=False) is False
    assert batcher.early_close_blocker == "previous_gpu_half"
    assert batcher.early_close_pipeline_ready is False

    assert batcher.poll_deadline(pipeline_ready=True) is True
    assert batcher.early_close_pipeline_ready is True
    assert batcher.early_close_pipeline_ready_at == pytest.approx(1065.0)


def test_primary_population_and_trainable_prompts_are_required(monkeypatch):
    from reliquary.constants import (
        AUCTION_EARLY_CLOSE_MIN_SECONDS,
        B_BATCH,
        PRIMARY_PROOF_GRADING_ATTEMPTS_PER_WINDOW,
    )

    batcher, advance = _clock_batcher("enforce", monkeypatch)
    advance(AUCTION_EARLY_CLOSE_MIN_SECONDS + 1.0)
    batcher.pending_count = PRIMARY_PROOF_GRADING_ATTEMPTS_PER_WINDOW - 1
    batcher.last_valid_submission_at = batcher._time_fn() - 10.0

    assert batcher.poll_deadline() is False
    assert batcher.early_close_blocker == "primary_candidate_target"

    batcher.pending_count += 1
    batcher.distinct_pending_prompt_count = lambda: B_BATCH - 1
    assert batcher.poll_deadline() is False
    assert batcher.early_close_blocker == "trainable_prompt_target"

    batcher.distinct_pending_prompt_count = lambda: B_BATCH
    assert batcher.poll_deadline() is True


def test_one_real_drand_round_of_quiet_is_required(monkeypatch):
    from reliquary.constants import AUCTION_EARLY_CLOSE_MIN_SECONDS

    batcher, advance = _clock_batcher(
        "enforce", monkeypatch, drand_period=30.0
    )
    advance(AUCTION_EARLY_CLOSE_MIN_SECONDS)
    _primary_population(batcher, idle_seconds=29.9)

    assert batcher.poll_deadline() is False
    assert batcher.early_close_blocker == "candidate_quiet_period"

    advance(0.1)
    assert batcher.poll_deadline() is True


@pytest.mark.parametrize(
    ("reservation_map", "blocker"),
    [
        ("_pending_proof_reservations", "pending_admission"),
        ("_inflight_proof_reservations", "inflight_admission"),
    ],
)
def test_admission_must_be_fully_drained(
    monkeypatch, reservation_map, blocker
):
    from reliquary.constants import AUCTION_EARLY_CLOSE_MIN_SECONDS

    batcher, advance = _clock_batcher("enforce", monkeypatch)
    advance(AUCTION_EARLY_CLOSE_MIN_SECONDS + 1.0)
    _primary_population(batcher, idle_seconds=5.0)
    reservations = getattr(batcher, reservation_map)
    reservations[1] = object()

    assert batcher.poll_deadline() is False
    assert batcher.early_close_blocker == blocker

    reservations.clear()
    assert batcher.poll_deadline() is True


def test_existing_upload_receipt_can_add_a_late_challenger(monkeypatch):
    """Adaptive close waits for receipts instead of stranding their bodies."""
    from reliquary.constants import AUCTION_EARLY_CLOSE_MIN_SECONDS

    batcher, advance = _clock_batcher("enforce", monkeypatch)
    advance(AUCTION_EARLY_CLOSE_MIN_SECONDS + 1.0)
    accepted, reason, _ = batcher.try_register_upload_precommit(
        "late-challenger",
        "miner",
        t_arrival_wall=batcher.window_opened_wall_ts + 61.0,
        payload_bytes=100,
    )
    assert accepted is True and reason is None
    _primary_population(batcher, idle_seconds=5.0)

    assert batcher.poll_deadline() is False
    assert batcher.early_close_blocker == "pending_uploads"
    assert batcher.early_close_eligible_at is None

    assert batcher.resolve_upload_precommit("late-challenger") is True
    assert batcher.poll_deadline() is True


def test_shadow_records_the_hypothetical_close_without_mutation(monkeypatch):
    from reliquary.constants import (
        AUCTION_EARLY_CLOSE_MIN_SECONDS,
        WINDOW_COLLECTION_SECONDS,
    )

    batcher, advance = _clock_batcher("shadow", monkeypatch)
    advance(AUCTION_EARLY_CLOSE_MIN_SECONDS + 2.0)
    _primary_population(batcher, idle_seconds=5.0)

    assert batcher.poll_deadline() is False
    assert batcher.is_sealed() is False
    assert batcher.early_close_eligible_at == pytest.approx(1062.0)
    assert batcher.early_close_sealed is False

    advance(WINDOW_COLLECTION_SECONDS)
    assert batcher.poll_deadline() is True
    assert batcher.early_close_sealed is False


def test_off_mode_only_uses_the_hard_ceiling(monkeypatch):
    from reliquary.constants import WINDOW_COLLECTION_SECONDS

    batcher, advance = _clock_batcher("off", monkeypatch)
    _primary_population(batcher, idle_seconds=20.0)
    advance(99.9)

    assert batcher.poll_deadline() is False
    assert batcher.early_close_blocker == "mode_off"
    advance(WINDOW_COLLECTION_SECONDS - 99.9)
    assert batcher.poll_deadline() is True
    assert batcher.early_close_sealed is False


def test_100_second_ceiling_ignores_every_adaptive_gate(monkeypatch):
    from reliquary.constants import WINDOW_COLLECTION_SECONDS

    batcher, advance = _clock_batcher("enforce", monkeypatch)
    advance(WINDOW_COLLECTION_SECONDS)

    assert batcher.poll_deadline(pipeline_ready=False) is True
    assert batcher.is_sealed() is True
    assert batcher.early_close_sealed is False
    assert batcher.early_close_eligible_at is None


def test_candidate_and_gpu_proof_limits_are_decoupled():
    from reliquary.constants import (
        B_BATCH,
        MAX_GRADING_STARTS_PER_WINDOW,
        MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW,
        MAX_RANKED_PROOF_ATTEMPTS_PER_WINDOW,
        PRIMARY_PROOF_GRADING_ATTEMPTS_PER_WINDOW,
        PROTOCOL_VERSION,
    )

    assert PRIMARY_PROOF_GRADING_ATTEMPTS_PER_WINDOW == 64
    assert MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW == 96
    assert MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW - 64 == 32
    assert MAX_RANKED_PROOF_ATTEMPTS_PER_WINDOW == (
        2 * B_BATCH if PROTOCOL_VERSION >= 3 else 64
    )
    assert MAX_GRADING_STARTS_PER_WINDOW == 256

    # The live v5 profile remains at the reviewed 32 ranked GPU attempts.
    code = (
        "from reliquary.constants import "
        "MAX_RANKED_PROOF_ATTEMPTS_PER_WINDOW as n; print(n)"
    )
    env = {
        **os.environ,
        "RELIQUARY_PROTOCOL_PROFILE": "qwen3-4b-base-dapo-reasoning-v5",
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.stdout.strip() == "32"


def test_productive_capacity_can_be_set_to_128_without_code_change():
    code = (
        "from reliquary.constants import "
        "MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW as n; print(n)"
    )
    env = {
        **os.environ,
        "RELIQUARY_MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW": "128",
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.stdout.strip() == "128"


def test_productive_capacity_rejects_values_outside_the_safe_range():
    code = "import reliquary.constants"
    for value in ("63", "129"):
        env = {
            **os.environ,
            "RELIQUARY_MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW": value,
        }
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode != 0
        assert "must be between 64 and 128" in result.stderr


def test_conservation_snapshot_exposes_cutover_evidence(monkeypatch):
    from reliquary.constants import AUCTION_EARLY_CLOSE_MIN_SECONDS

    batcher, advance = _clock_batcher("shadow", monkeypatch)
    advance(AUCTION_EARLY_CLOSE_MIN_SECONDS + 1.0)
    _primary_population(batcher, idle_seconds=5.0)
    batcher.poll_deadline(pipeline_ready=True)

    early = batcher.upload_precommit_conservation()["early_close"]
    assert early == {
        "mode": "shadow",
        "strategy": "adaptive_gpu_quiet",
        "minimum_collection_seconds": 60.0,
        "maximum_collection_seconds": 100.0,
        "quiet_seconds": 3.0,
        "primary_candidate_target": 64,
        "challenger_capacity": 32,
        "pipeline_ready": True,
        "pipeline_ready_offset_seconds": pytest.approx(61.0),
        "last_blocker": None,
        "eligible_offset_seconds": pytest.approx(61.0),
        "sealed_early": False,
        "sealed_offset_seconds": None,
        "refusing_precommits": False,
    }


def test_conservation_snapshot_does_not_deadlock(monkeypatch):
    batcher, _advance = _clock_batcher("enforce", monkeypatch)
    done = threading.Event()
    result: list[dict] = []

    def call() -> None:
        result.append(batcher.upload_precommit_conservation())
        done.set()

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    assert done.wait(5.0), "upload_precommit_conservation deadlocked"
    assert result[0]["early_close"]["refusing_precommits"] is False


def test_new_precommit_is_terminally_refused_after_adaptive_seal(monkeypatch):
    from reliquary.constants import AUCTION_EARLY_CLOSE_MIN_SECONDS

    batcher, advance = _clock_batcher("enforce", monkeypatch)
    advance(AUCTION_EARLY_CLOSE_MIN_SECONDS)
    _primary_population(batcher, idle_seconds=5.0)
    assert batcher.poll_deadline() is True

    accepted, reason, _ = batcher.try_register_upload_precommit(
        "too-late",
        "miner",
        t_arrival_wall=batcher.window_opened_wall_ts + 60.0,
        payload_bytes=100,
    )
    assert accepted is False
    assert reason == "collection_sealed"


def test_empty_mode_falls_back_to_shadow_and_invalid_mode_fails(monkeypatch):
    import importlib
    import reliquary.constants as constants

    try:
        monkeypatch.setenv("RELIQUARY_AUCTION_EARLY_CLOSE_MODE", "")
        reloaded = importlib.reload(constants)
        assert reloaded.AUCTION_EARLY_CLOSE_MODE == "shadow"

        monkeypatch.setenv(
            "RELIQUARY_AUCTION_EARLY_CLOSE_MODE", "sometimes"
        )
        with pytest.raises(ValueError, match="EARLY_CLOSE_MODE"):
            importlib.reload(constants)
    finally:
        monkeypatch.delenv(
            "RELIQUARY_AUCTION_EARLY_CLOSE_MODE", raising=False
        )
        importlib.reload(constants)
