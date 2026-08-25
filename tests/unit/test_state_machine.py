"""ValidationService state machine: OPEN → TRAINING → PUBLISHING → READY."""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from reliquary.constants import B_BATCH
from reliquary.protocol.submission import WindowState


@pytest.fixture(autouse=True)
def _isolated_validator_state(tmp_path, monkeypatch):
    """Unit services must never read or write the host's production state."""

    monkeypatch.setenv("RELIQUARY_STATE_DIR", str(tmp_path))


@dataclass
class _FakeEnv:
    def __len__(self): return 100
    def get_problem(self, i): return {"prompt": "p", "ground_truth": "", "id": f"p{i}"}
    def compute_reward(self, p, c): return 1.0

    @property
    def name(self): return "fake"


class _FakeWallet:
    class _Hk:
        ss58_address = "5FHk"
        @staticmethod
        def sign(d): return b"sig"
    hotkey = _Hk()


def _make_service():
    from reliquary.validator.service import ValidationService

    svc = ValidationService(
        wallet=_FakeWallet(),
        model=MagicMock(),
        tokenizer=MagicMock(),
        env=_FakeEnv(),
        netuid=99,
    )
    # These unit tests call _open_window directly and intentionally bypass the
    # startup restore that makes canonical cooldown state durable.
    svc._content_cooldown_health["complete"] = True
    return svc


def test_service_initial_state_is_ready():
    svc = _make_service()
    assert svc._current_window_state == WindowState.READY


@pytest.mark.asyncio
async def test_cross_environment_abort_discards_all_seal_side_effects():
    from reliquary.validator.proof_scheduler import SchedulerState

    svc = _make_service()
    math = MagicMock()
    code = MagicMock()
    for batcher in (math, code):
        batcher.beacon_invalid = False
        batcher.auction_admission_aborted = False
        batcher.proof_capacity_aborted = False
        batcher.proof_capacity_abort_reason = None
    math.seal_batch.return_value = ([MagicMock()], {"math-hotkey": 0.1})
    code.seal_batch.return_value = ([], {})
    code.proof_capacity_aborted = True
    code.proof_capacity_abort_reason = "attempt_limit"

    svc.env_mix = (
        ("openmathinstruct", 1.0),
        ("opencodeinstruct", 1.0),
    )
    expected_batchers = {
        "openmathinstruct": math,
        "opencodeinstruct": code,
    }
    svc._active_batchers = expected_batchers
    svc.proof_scheduler = SimpleNamespace(state=SchedulerState.RUNNING)
    svc._fetch_seal_randomness = AsyncMock(return_value="ab" * 32)
    svc._enqueue_aborted_window = MagicMock()

    await svc._train_and_publish()

    for batcher in (math, code):
        batcher.seal_batch.assert_called_once_with(
            pool=0.5,
            commit_side_effects=False,
        )
        batcher.discard_seal_side_effects.assert_called_once_with()
        batcher.commit_seal_side_effects.assert_not_called()
    svc._enqueue_aborted_window.assert_called_once_with(
        failure_stage="proof_capacity",
        failure_type="ProofCapacityAbort",
        batchers=expected_batchers,
        late_drops=None,
    )


@pytest.mark.asyncio
async def test_faulted_proof_scheduler_archives_then_requires_restart():
    from reliquary.validator.proof_scheduler import SchedulerState
    from reliquary.validator.service import FatalProofPlaneError

    svc = _make_service()
    batcher = MagicMock()
    batcher.beacon_invalid = False
    batcher.auction_admission_aborted = False
    batcher.proof_capacity_aborted = True
    batcher.proof_capacity_abort_reason = "active_proof_timeout"
    batcher.seal_batch.return_value = ([], {})
    expected_batchers = {"openmathinstruct": batcher}
    svc._active_batchers = expected_batchers
    svc.proof_scheduler = SimpleNamespace(state=SchedulerState.FAULTED)
    svc._fetch_seal_randomness = AsyncMock(return_value="ab" * 32)
    svc._enqueue_aborted_window = MagicMock()

    with pytest.raises(FatalProofPlaneError, match="restart required"):
        await svc._train_and_publish()

    batcher.discard_seal_side_effects.assert_called_once_with()
    svc._enqueue_aborted_window.assert_called_once_with(
        failure_stage="proof_capacity",
        failure_type="ProofCapacityAbort",
        batchers=expected_batchers,
        late_drops=None,
    )


@pytest.mark.asyncio
async def test_quiesced_replica_refresh_retries_at_next_boundary():
    from reliquary.validator.proof_scheduler import SchedulerState

    svc = _make_service()

    class _Scheduler:
        state = SchedulerState.QUIESCED
        revision = None

        def checkpoint_ready(self, revision):
            return (
                self.state is SchedulerState.RUNNING
                and self.revision == revision
            )

    scheduler = _Scheduler()
    svc.proof_scheduler = scheduler
    svc._verify_model_checkpoint_revision = "checkpoint-a"
    svc._checkpoint_store = MagicMock()
    svc._checkpoint_store.current_manifest.return_value = SimpleNamespace(
        revision="checkpoint-a"
    )
    calls = []

    def _refresh(revision):
        calls.append(revision)
        if len(calls) == 1:
            raise RuntimeError("transient replica load failure")
        scheduler.revision = revision
        scheduler.state = SchedulerState.RUNNING

    svc._synchronize_proof_models = MagicMock(side_effect=_refresh)

    with pytest.raises(RuntimeError, match="transient replica load failure"):
        await svc._ensure_proof_scheduler_ready()
    assert scheduler.state is SchedulerState.QUIESCED

    await svc._ensure_proof_scheduler_ready()

    assert calls == ["checkpoint-a", "checkpoint-a"]
    assert scheduler.checkpoint_ready("checkpoint-a")


@pytest.mark.asyncio
async def test_stale_verify_model_cannot_be_labeled_as_new_checkpoint():
    from reliquary.validator.proof_scheduler import SchedulerState

    svc = _make_service()

    class _Scheduler:
        state = SchedulerState.QUIESCED
        revision = None

        def checkpoint_ready(self, revision):
            return (
                self.state is SchedulerState.RUNNING
                and self.revision == revision
            )

    scheduler = _Scheduler()
    svc.proof_scheduler = scheduler
    svc._verify_model_checkpoint_revision = "checkpoint-old"
    svc._checkpoint_store = MagicMock()
    svc._checkpoint_store.current_manifest.return_value = SimpleNamespace(
        revision="checkpoint-new"
    )
    refresh_calls = []

    def _refresh(revision):
        refresh_calls.append(revision)
        if len(refresh_calls) == 1:
            svc._verify_model_checkpoint_revision = None
            raise RuntimeError("verify state copy failed")
        svc._verify_model_checkpoint_revision = revision

    def _synchronize(revision):
        assert svc._verify_model_checkpoint_revision == revision
        scheduler.revision = revision
        scheduler.state = SchedulerState.RUNNING

    svc._refresh_verify_model_from_train = MagicMock(side_effect=_refresh)
    svc._synchronize_proof_models = MagicMock(side_effect=_synchronize)

    with pytest.raises(RuntimeError, match="verify state copy failed"):
        await svc._ensure_proof_scheduler_ready()
    svc._synchronize_proof_models.assert_not_called()
    assert svc._verify_model_checkpoint_revision is None

    await svc._ensure_proof_scheduler_ready()

    assert refresh_calls == ["checkpoint-new", "checkpoint-new"]
    assert scheduler.checkpoint_ready("checkpoint-new")


def test_replica_synchronization_rejects_unlabeled_verify_weights():
    from reliquary.validator.proof_scheduler import SchedulerState

    svc = _make_service()
    scheduler = MagicMock()
    scheduler.state = SchedulerState.QUIESCED
    svc.proof_scheduler = scheduler
    svc._verify_model_checkpoint_revision = "checkpoint-old"

    with pytest.raises(RuntimeError, match="not certified"):
        svc._synchronize_proof_models("checkpoint-new")

    scheduler.mark_device_ready.assert_not_called()
    scheduler.resume.assert_not_called()


@pytest.mark.asyncio
async def test_faulted_scheduler_cannot_reach_window_open():
    from reliquary.validator.proof_scheduler import SchedulerState
    from reliquary.validator.service import FatalProofPlaneError

    svc = _make_service()
    svc.proof_scheduler = SimpleNamespace(state=SchedulerState.FAULTED)

    with pytest.raises(FatalProofPlaneError, match="process restart"):
        await svc._ensure_proof_scheduler_ready()


def test_proof_scheduler_health_flags_hung_and_recent_abort(monkeypatch):
    import reliquary.validator.service as service_module

    svc = _make_service()
    monkeypatch.setattr(service_module, "PROTOCOL_VERSION", 3)
    svc.proof_capacity_qualification = {"qualified": True}
    svc.proof_scheduler = SimpleNamespace(
        snapshot=lambda: {
            "state": "running",
            "active_checkpoint_revision": "checkpoint-a",
            "device_revisions": {"cuda:0": "checkpoint-a"},
            "active_by_device": {
                "cuda:0": {"age_seconds": 241.0},
            },
            "last_capacity_abort_age_seconds": 10.0,
        }
    )

    snapshot = svc._proof_scheduler_health_snapshot()

    assert snapshot["degraded_reasons"] == [
        "active_proof_over_wall",
        "recent_capacity_abort",
    ]


def test_proof_health_compares_memory_labels_to_published_manifest(monkeypatch):
    import reliquary.validator.service as service_module

    svc = _make_service()
    monkeypatch.setattr(service_module, "PROTOCOL_VERSION", 3)
    svc.proof_capacity_qualification = {"qualified": True}
    svc._checkpoint_store = MagicMock()
    svc._checkpoint_store.current_manifest.return_value = SimpleNamespace(
        revision="checkpoint-new"
    )
    svc._verify_model_checkpoint_revision = "checkpoint-old"
    svc.proof_scheduler = SimpleNamespace(
        snapshot=lambda: {
            "state": "running",
            "active_checkpoint_revision": "checkpoint-old",
            "device_revisions": {"cuda:0": "checkpoint-old"},
            "active_by_device": {"cuda:0": None},
            "last_capacity_abort_age_seconds": None,
        }
    )

    snapshot = svc._proof_scheduler_health_snapshot()

    assert "verify_checkpoint_mismatch" in snapshot["degraded_reasons"]
    assert "scheduler_checkpoint_mismatch" in snapshot["degraded_reasons"]


def test_proof_health_survives_checkpoint_manifest_failure(monkeypatch):
    import reliquary.validator.service as service_module

    svc = _make_service()
    monkeypatch.setattr(service_module, "PROTOCOL_VERSION", 3)
    svc.proof_capacity_qualification = {"qualified": True}
    svc._checkpoint_store = MagicMock()
    svc._checkpoint_store.current_manifest.side_effect = OSError(
        "checkpoint state unavailable"
    )
    svc._verify_model_checkpoint_revision = "checkpoint-old"
    svc.proof_scheduler = SimpleNamespace(
        snapshot=lambda: {
            "state": "running",
            "active_checkpoint_revision": "checkpoint-old",
            "device_revisions": {"cuda:0": "checkpoint-old"},
            "active_by_device": {"cuda:0": None},
            "last_capacity_abort_age_seconds": None,
        }
    )

    snapshot = svc._proof_scheduler_health_snapshot()

    assert snapshot["published_checkpoint_revision"] is None
    assert (
        "checkpoint_manifest_unavailable"
        in snapshot["degraded_reasons"]
    )


@pytest.mark.asyncio
async def test_scheduler_fault_after_seal_aborts_before_side_effect_commit():
    from reliquary.validator.proof_scheduler import SchedulerState
    from reliquary.validator.service import FatalProofPlaneError

    svc = _make_service()
    scheduler = SimpleNamespace(
        state=SchedulerState.RUNNING,
        snapshot=lambda: {"fault_reason": "proof_execution_error"},
    )
    batcher = MagicMock()
    batcher.beacon_invalid = False
    batcher.auction_admission_aborted = False
    batcher.proof_capacity_aborted = False
    batcher.proof_capacity_abort_reason = None

    def _seal(*_args, **_kwargs):
        scheduler.state = SchedulerState.FAULTED
        return ([MagicMock()], {"winner": 0.1})

    batcher.seal_batch.side_effect = _seal
    svc._active_batchers = {"openmathinstruct": batcher}
    svc.proof_scheduler = scheduler
    svc._fetch_seal_randomness = AsyncMock(return_value="ab" * 32)
    svc._enqueue_aborted_window = MagicMock()

    with pytest.raises(FatalProofPlaneError):
        await svc._train_and_publish()

    batcher.discard_seal_side_effects.assert_called_once_with()
    batcher.commit_seal_side_effects.assert_not_called()


@pytest.mark.asyncio
async def test_faulted_scheduler_shutdown_uses_short_timeout():
    from reliquary.validator.proof_scheduler import SchedulerState

    svc = _make_service()
    close = MagicMock(return_value=False)
    svc.proof_scheduler = SimpleNamespace(
        state=SchedulerState.FAULTED,
        close=close,
    )

    await svc._close_proof_scheduler()

    close.assert_called_once_with(5.0)


def test_open_window_sets_state_to_open():
    svc = _make_service()
    svc._open_window()
    svc._activate_window()
    assert svc._current_window_state == WindowState.OPEN
    assert svc._active_batcher is not None


def test_activation_reanchors_window_clock_after_preparation():
    svc = _make_service()
    svc._open_window()
    batcher = svc._active_batcher
    now = [123.0]
    wall = [1_234.0]
    batcher._time_fn = lambda: now[0]
    batcher._wall_clock = lambda: wall[0]
    batcher.window_opened_at = -1.0
    batcher.window_opened_wall_ts = -1.0

    svc._activate_window()

    assert batcher.window_opened_at == 123.0
    assert batcher.window_opened_wall_ts == 1_234.0


def test_open_window_reserves_candidate_without_committing_window_n():
    svc = _make_service()
    initial = svc._window_n
    svc._open_window()
    assert svc._window_n == initial
    assert svc._candidate_window_n == initial + 1
    assert svc._active_batcher.window_start == initial + 1


def test_failed_preopen_reuses_candidate_until_activation():
    svc = _make_service()
    initial = svc._window_n

    svc._open_window()
    svc._set_window_preparation_stage("prompt_manifest")
    svc._rollback_preopen_window(RuntimeError("source unavailable"))

    health = svc.server._health_payload()
    assert svc._window_n == initial
    assert svc._candidate_window_n == initial + 1
    assert health.status == "degraded"
    assert health.last_committed_window_n == initial
    assert health.candidate_window_n == initial + 1
    assert health.window_preparation_failures_total == 1
    assert health.window_preparation_failures_by_stage == {
        "prompt_manifest": 1
    }
    assert health.last_window_preparation_failure == {
        "candidate_window_n": initial + 1,
        "stage": "prompt_manifest",
        "error_type": "RuntimeError",
        "ts": health.last_window_preparation_failure["ts"],
    }

    svc._open_window()
    assert svc._active_batcher.window_start == initial + 1
    svc._activate_window()

    health = svc.server._health_payload()
    assert svc._window_n == initial + 1
    assert svc._candidate_window_n is None
    assert health.status == "ok"
    assert health.last_committed_window_n == initial + 1
    assert health.candidate_window_n is None
    assert health.last_window_preparation_failure is None


@pytest.mark.asyncio
async def test_prompt_preparation_failure_does_not_retry_randomness():
    from reliquary.environment.virtual_parquet import PromptSourceUnavailable

    svc = _make_service()
    svc._open_window()
    svc._derive_randomness = AsyncMock(return_value=("seed", None))
    svc._active_batcher.set_prompt_range = MagicMock(
        side_effect=PromptSourceUnavailable("manifest unavailable")
    )

    with pytest.raises(PromptSourceUnavailable, match="manifest unavailable"):
        await svc._set_window_randomness(subtensor=None)

    svc._derive_randomness.assert_awaited_once_with(
        None, svc._candidate_window_n
    )
    assert svc._window_preparation_stage == "prompt_manifest"


def test_set_state_transitions():
    svc = _make_service()
    for state in (WindowState.OPEN, WindowState.TRAINING,
                  WindowState.PUBLISHING, WindowState.READY):
        svc._set_state(state)
        assert svc._current_window_state == state


@pytest.mark.asyncio
async def test_train_and_publish_bumps_checkpoint_n(monkeypatch):
    # Patch B_BATCH to 0 so an empty sealed batch counts as "full" and the
    # train+publish path runs. Real behaviour with non-zero B_BATCH is
    # covered by the integration tests that exercise actual submissions.
    monkeypatch.setattr("reliquary.validator.service.B_BATCH", 0)

    svc = _make_service()
    initial_checkpoint = svc._checkpoint_n

    # Open a window so there's an active batcher + seal_event to drive
    svc._open_window()
    svc._activate_window()

    # Mock the checkpoint store to avoid HF calls
    svc._checkpoint_store = MagicMock()
    svc._checkpoint_store.current_manifest = MagicMock(return_value=None)
    from reliquary.validator.checkpoint import ManifestEntry
    fake_entry = ManifestEntry(
        checkpoint_n=initial_checkpoint + 1,
        repo_id="aivolutionedge/reliquary-sn",
        revision="rev_sha_x",
        signature="ed25519:x",
    )
    svc._checkpoint_store.publish = AsyncMock(return_value=fake_entry)

    # Mock storage.upload_window_dataset to avoid R2
    import reliquary.validator.service as svc_mod
    original_upload = svc_mod.storage.upload_window_dataset
    svc_mod.storage.upload_window_dataset = AsyncMock(return_value=True)

    try:
        await svc._train_and_publish()
    finally:
        svc_mod.storage.upload_window_dataset = original_upload

    assert svc._checkpoint_n == initial_checkpoint + 1
    assert svc._current_window_state == WindowState.READY
    assert svc._active_batcher is None
    svc._checkpoint_store.publish.assert_awaited_once()


@pytest.mark.asyncio
async def test_admission_drain_abort_never_ranks_rewards_or_trains(monkeypatch):
    svc = _make_service()
    svc._open_window()
    svc._activate_window()
    batcher = svc._active_batcher
    batcher.auction_admission_aborted = True
    batcher.auction_seal_drain = {
        "outcome": "aborted",
        "receipt_conservation": {"conserved": True, "pending": 0},
    }
    batcher.force_seal_reason = "auction_admission_drain_abort"
    batcher.seal_batch = MagicMock(
        side_effect=AssertionError("aborted population must not be ranked")
    )
    svc._archive_enqueued_windows = set()
    svc._window_iteration_stage = "seal_train_archive"
    checkpoint_n = svc._checkpoint_n
    tombstones = []

    class _StubQueue:
        def enqueue(self, window_start, archive):
            tombstones.append((window_start, archive))

    monkeypatch.setattr(
        "reliquary.infrastructure.archive_queue.get_archive_queue",
        lambda: _StubQueue(),
    )

    await svc._train_and_publish()

    batcher.seal_batch.assert_not_called()
    assert svc._checkpoint_n == checkpoint_n
    assert svc._current_window_state == WindowState.READY
    assert svc._active_batchers == {}
    assert len(tombstones) == 1
    archive = tombstones[0][1]
    assert archive["window_status"] == "aborted"
    assert archive["failure_stage"] == "admission_drain"
    assert archive["batch"] == []
    assert archive["rewards_by_hotkey"] == {}
    assert archive["training_accumulator"]["trained"] is False


def test_open_window_wires_checkpoint_hash_into_batcher():
    svc = _make_service()
    from reliquary.validator.checkpoint import ManifestEntry
    svc._checkpoint_store = MagicMock()
    svc._checkpoint_store.current_manifest = MagicMock(return_value=ManifestEntry(
        checkpoint_n=5,
        repo_id="aivolutionedge/reliquary-sn",
        revision="rev_sha_005",
        signature="ed25519:sig",
    ))
    svc._open_window()
    assert svc._active_batcher.current_checkpoint_hash == "rev_sha_005"


@pytest.mark.asyncio
async def test_activate_window_binds_batcher_loop_for_delayed_seal():
    """Regression: ``_activate_window`` must bind each batcher's event loop.

    ``accept_submission`` runs in a worker thread (``asyncio.to_thread``)
    with no running loop, so it cannot capture the loop itself — it reads
    the pre-bound ``batcher._loop`` to schedule the delayed drand-boundary
    seal via ``run_coroutine_threadsafe``. If ``_loop`` is left ``None``,
    the B-th distinct prompt seals the window synchronously, dropping every
    same-drand-round submission still in flight (BATCH_FILLED) and
    collapsing the boundary fair split — i.e. only ~B miners per round can
    ever earn emission.
    """
    import asyncio

    svc = _make_service()
    svc._open_window()
    svc._activate_window()

    running_loop = asyncio.get_running_loop()
    assert svc._active_batchers
    for batcher in svc._active_batchers.values():
        assert batcher._loop is running_loop


@pytest.mark.asyncio
async def test_wait_for_window_seal_force_seals_drained_proof_cap():
    """A full proof cap with no queued/in-flight work cannot fill later."""
    from reliquary.validator.service import MAX_GRADING_STARTS_PER_WINDOW

    svc = _make_service()
    svc._open_window()
    svc._activate_window()
    batcher = svc._active_batcher

    # Exhaustion is gated on the never-refunded grading-starts backstop:
    # non-productive rejects refund the productive admission budget, so only
    # the ceiling nothing gives back means "cannot fill later".
    batcher._proof_grading_attempts = MAX_GRADING_STARTS_PER_WINDOW
    assert batcher.valid_count == 0
    assert svc.server.submit_queue_depth == 0
    assert svc.server.proof_verification_inflight == 0

    reason = await svc._wait_for_window_seal()

    assert reason == "proof_admission_exhausted_drained"
    assert batcher.is_sealed()
    assert batcher.force_seal_reason == "proof_admission_exhausted_drained"


def test_proof_cap_breaker_waits_for_inflight_or_queued_work():
    from reliquary.validator.service import MAX_GRADING_STARTS_PER_WINDOW

    svc = _make_service()
    svc._open_window()
    svc._activate_window()
    batcher = svc._active_batcher
    batcher._proof_grading_attempts = MAX_GRADING_STARTS_PER_WINDOW

    svc.server._inflight_proofs = 1
    assert svc._proof_admission_exhausted_and_drained(batcher) is False

    svc.server._inflight_proofs = 0
    svc.server._submit_queue.put_nowait((object(), batcher, object()))
    assert svc._proof_admission_exhausted_and_drained(batcher) is False


def test_queue_drain_observes_batcher_reservation_dequeue_gap():
    """Queue empty + server inflight zero is not enough during dequeue."""
    svc = _make_service()
    svc._open_window()
    svc._activate_window()
    batcher = svc._active_batcher

    request = MagicMock()
    request.miner_hotkey = "miner"
    request._payload_bytes = 1
    request.drand_round = 0
    assert batcher.try_reserve_proof_admission(request) == (True, None)
    assert svc.server.submit_queue_depth == 0
    assert svc.server.proof_verification_inflight == 0

    assert svc._queue_and_proofs_drained() is False


@pytest.mark.asyncio
async def test_auction_freeze_marks_population_after_normal_drain():
    svc = _make_service()
    batcher = MagicMock()
    batcher.difficulty_auction_enabled = True
    batcher.pending_proof_reservations = 0
    batcher.inflight_proof_reservations = 0
    batcher.pending_upload_precommits = 0
    svc._active_batchers = {"openmathinstruct": batcher}

    timed_out = await svc._freeze_auction_populations([batcher])

    assert timed_out == {"openmathinstruct": False}
    batcher.begin_seal_snapshot.assert_called_once_with()


@pytest.mark.asyncio
async def test_auction_freeze_times_out_then_drops_pending_admission(monkeypatch):
    monkeypatch.setattr(
        "reliquary.validator.service.AUCTION_ADMISSION_DRAIN_DEADLINE_SECONDS",
        0.0,
    )
    svc = _make_service()
    batcher = MagicMock()
    batcher.difficulty_auction_enabled = True
    batcher.pending_proof_reservations = 1
    batcher.inflight_proof_reservations = 0
    batcher.pending_upload_precommits = 0
    svc._active_batchers = {"openmathinstruct": batcher}

    timed_out = await svc._freeze_auction_populations([batcher])

    assert timed_out == {"openmathinstruct": True}
    batcher.begin_seal_snapshot.assert_called_once_with()
    assert batcher.force_seal_reason == "auction_admission_drain_abort"
    assert batcher.auction_admission_aborted is True


@pytest.mark.asyncio
async def test_auction_drain_abort_expires_unrevealed_receipt(monkeypatch):
    from reliquary.protocol.submission import RejectReason
    from reliquary.validator.observability import DrandRoundObservation
    from reliquary.validator.server import _UploadPrecommitReceipt

    monkeypatch.setattr(
        "reliquary.validator.service.AUCTION_ADMISSION_DRAIN_DEADLINE_SECONDS",
        0.0,
    )
    svc = _make_service()
    svc._open_window()
    svc._activate_window()
    batcher = svc._active_batcher
    batcher.difficulty_auction_enabled = True
    # Production batchers receive this immutable ownership snapshot from the
    # metagraph; direct unit registration must model the same precondition.
    batcher._operator_by_hotkey["miner"] = "operator"
    receipt_id = "unrevealed-at-abort"
    accepted, reason, _deadline = batcher.try_register_upload_precommit(
        receipt_id,
        "miner",
        t_arrival_wall=batcher.window_opened_wall_ts,
        payload_bytes=512,
    )
    assert accepted is True
    assert reason is None
    receipt = _UploadPrecommitReceipt(
        receipt_id=receipt_id,
        precommit_signature="signed",
        miner_hotkey="miner",
        prompt_idx=1,
        window_start=batcher.window_start,
        merkle_root="00" * 32,
        checkpoint_hash=batcher.current_checkpoint_hash,
        environment="fake",
        payload_bytes=512,
        payload_sha256="11" * 32,
        drand_round=1,
        protocol_version=2,
        nonce="nonce",
        expires_at_wall=10**12,
        precommit_arrival_ts=1.0,
        drand_observation=DrandRoundObservation(
            submitted_drand_round=1,
            arrival_drand_round=1,
            drand_delta=0,
            drand_tolerance=0,
            drand_status="current",
            reject_reason=None,
        ),
        batcher=batcher,
    )
    svc.server._upload_precommit_receipts[receipt_id] = receipt
    svc.server._upload_precommit_by_signature["signed"] = receipt_id

    timed_out = await svc._freeze_auction_populations([batcher])

    assert timed_out == {"fake": True}
    assert receipt.terminal is True
    assert receipt.outcome is not None
    assert receipt.outcome.reason is RejectReason.PRECOMMIT_EXPIRED
    conservation = batcher.upload_precommit_conservation()
    assert conservation["accepted_receipts"] == 1
    assert conservation["revealed"] == 0
    assert conservation["expired"] == 1
    assert conservation["terminal_decisions"] == 0
    assert conservation["pending"] == 0
    assert conservation["capacity_reserved"] == 0
    assert conservation["conserved"] is True
    assert conservation["capacity_conserved"] is True
    assert batcher.auction_seal_drain["outcome"] == "aborted"


@pytest.mark.asyncio
async def test_auction_freeze_never_waits_for_inflight_after_snapshot(monkeypatch):
    monkeypatch.setattr(
        "reliquary.validator.service.AUCTION_ADMISSION_DRAIN_DEADLINE_SECONDS",
        0.0,
    )
    svc = _make_service()
    batcher = MagicMock()
    batcher.difficulty_auction_enabled = True
    batcher.env.name = "openmathinstruct"
    batcher.pending_proof_reservations = 0
    batcher.inflight_proof_reservations = 1
    batcher.pending_upload_precommits = 0
    svc._active_batchers = {"openmathinstruct": batcher}
    svc.server._inflight_proofs = 1
    svc.server._inflight_proofs_by_environment["openmathinstruct"] = 1

    timed_out = await svc._freeze_auction_populations([batcher])

    assert timed_out == {"openmathinstruct": True}
    batcher.begin_seal_snapshot.assert_called_once_with()


def test_proof_cap_breaker_uses_distinct_prompt_count():
    """Raw valid duplicates should not mask an unfillable trainable shortfall."""
    from reliquary.validator.service import MAX_GRADING_STARTS_PER_WINDOW

    svc = _make_service()
    svc._open_window()
    svc._activate_window()
    batcher = svc._active_batcher
    batcher.difficulty_auction_enabled = True
    # Proofs run at seal now, so the trainable fill level is the PENDING pool.
    batcher._pending = [
        SimpleNamespace(prompt_idx=i) for i in range(B_BATCH - 1)
    ] + [SimpleNamespace(prompt_idx=0)]
    batcher.pending_count = B_BATCH
    batcher._proof_grading_attempts = MAX_GRADING_STARTS_PER_WINDOW

    assert batcher.distinct_pending_prompt_count() == B_BATCH - 1
    assert svc._proof_admission_exhausted_and_drained(batcher) is True


@pytest.mark.asyncio
async def test_wait_for_window_seal_force_seals_duplicate_prompt_shortfall(monkeypatch):
    """A duplicate-filled raw batch must not wait for the long safety timeout."""
    monkeypatch.setattr(
        "reliquary.validator.service.MAX_SEAL_QUEUE_DRAIN_SECONDS", 0.0,
    )
    svc = _make_service()
    svc._open_window()
    svc._activate_window()
    batcher = svc._active_batcher
    batcher._valid = [
        SimpleNamespace(prompt_idx=i) for i in range(B_BATCH - 1)
    ] + [SimpleNamespace(prompt_idx=0)]
    batcher.valid_count = B_BATCH
    batcher._proof_admission_count = B_BATCH + 1

    reason = await svc._wait_for_window_seal()

    assert reason == "duplicate_prompt_distinct_shortfall_drained"
    assert batcher.is_sealed()
    assert batcher.force_seal_reason == reason


@pytest.mark.asyncio
async def test_wait_for_window_seal_force_seals_sparse_valid_idle(monkeypatch):
    """Sparse valid traffic should not wait for the long safety timeout."""
    monkeypatch.setattr(
        "reliquary.validator.service.SPARSE_VALID_IDLE_SEAL_SECONDS", 0.0,
    )
    monkeypatch.setattr(
        "reliquary.validator.service.SPARSE_VALID_IDLE_MIN_DISTINCT_PROMPTS", 4,
    )
    svc = _make_service()
    svc._open_window()
    svc._activate_window()
    batcher = svc._active_batcher
    batcher._valid = [SimpleNamespace(prompt_idx=i) for i in range(4)]
    batcher.valid_count = 4
    batcher.last_valid_submission_at = batcher._time_fn() - 1.0
    batcher.last_valid_submission_wall_ts = batcher._wall_clock() - 1.0

    reason = await svc._wait_for_window_seal()

    assert reason == "sparse_valid_idle_timeout"
    assert batcher.is_sealed()
    assert batcher.force_seal_reason == reason


@pytest.mark.asyncio
async def test_wait_for_window_seal_force_seals_sparse_valid_max_age(monkeypatch):
    """Very sparse windows eventually seal even below the idle distinct floor."""
    monkeypatch.setattr(
        "reliquary.validator.service.SPARSE_VALID_MAX_WINDOW_SECONDS", 0.0,
    )
    svc = _make_service()
    svc._open_window()
    svc._activate_window()
    batcher = svc._active_batcher
    batcher._valid = [SimpleNamespace(prompt_idx=123)]
    batcher.valid_count = 1
    batcher.last_valid_submission_at = batcher._time_fn()
    batcher.last_valid_submission_wall_ts = batcher._wall_clock()

    reason = await svc._wait_for_window_seal()

    assert reason == "sparse_valid_window_timeout"
    assert batcher.is_sealed()
    assert batcher.force_seal_reason == reason


@pytest.mark.asyncio
async def test_wait_for_window_seal_force_seals_zero_valid_max_age(monkeypatch):
    """A reset window with only rejected/stale miners must not freeze forever."""
    monkeypatch.setattr(
        "reliquary.validator.service.SPARSE_VALID_MAX_WINDOW_SECONDS", 0.0,
    )
    svc = _make_service()
    svc._open_window()
    svc._activate_window()
    batcher = svc._active_batcher
    assert batcher.valid_count == 0

    reason = await svc._wait_for_window_seal()

    assert reason == "zero_valid_window_timeout"
    assert batcher.is_sealed()
    assert batcher.force_seal_reason == reason


def test_open_window_empty_hash_pre_first_publish():
    svc = _make_service()
    # No checkpoint published yet → current_manifest returns None
    svc._checkpoint_store = MagicMock()
    svc._checkpoint_store.current_manifest = MagicMock(return_value=None)
    svc._open_window()
    assert svc._active_batcher.current_checkpoint_hash == ""


@pytest.mark.asyncio
async def test_publish_every_n_trained_windows(monkeypatch):
    """With _publish_every=3, publish is driven by successful trained windows.

    The first trained window publishes because no manifest exists. After that,
    the next publish happens after three more successful trained windows,
    regardless of the absolute window number.
    """
    # Patch B_BATCH so empty batches count as "full" (real-batch behaviour is
    # covered by the integration test that uses real submissions).
    monkeypatch.setattr("reliquary.validator.service.B_BATCH", 0)

    import reliquary.validator.service as svc_mod
    from reliquary.validator.checkpoint import ManifestEntry

    svc = _make_service()
    svc._publish_every = 3

    # Start with no manifest so first call always publishes.
    mock_store = MagicMock()
    mock_store.current_manifest = MagicMock(return_value=None)

    published_entries = []

    async def _fake_publish(checkpoint_n, model, profile_extra=None):
        entry = ManifestEntry(
            checkpoint_n=checkpoint_n,
            repo_id="aivolutionedge/reliquary-sn",
            revision=f"rev_{checkpoint_n:03d}",
            signature="ed25519:sig",
        )
        published_entries.append(entry)
        # After first publish, current_manifest returns the latest entry.
        mock_store.current_manifest.return_value = entry
        return entry

    mock_store.publish = AsyncMock(side_effect=_fake_publish)
    svc._checkpoint_store = mock_store

    original_upload = svc_mod.storage.upload_window_dataset
    svc_mod.storage.upload_window_dataset = AsyncMock(return_value=True)

    try:
        for _ in range(5):
            svc._open_window()
            svc._active_batcher.seal_event.set()
            await svc._train_and_publish()
    finally:
        svc_mod.storage.upload_window_dataset = original_upload

    # window_n increments: 1,2,3,4,5.
    # Publish fires when: window_n==1 (manifest is None), window_n==4
    # (three trained windows since the last publish). Windows 2,3,5 skip.
    # checkpoint_n advances only on publish.
    assert mock_store.publish.await_count == 2
    assert published_entries[0].checkpoint_n == 1  # first publish: next_n = 0+1 = 1
    assert published_entries[1].checkpoint_n == 2  # second publish: next_n = 1+1 = 2


@pytest.mark.asyncio
async def test_resume_from_path_installs_manifest():
    """resume_from="path:/tmp/ckpt_3" loads the directory AND installs a
    manifest so /state announces checkpoint_n=3 to miners immediately."""
    import tempfile, os
    from unittest.mock import MagicMock
    from reliquary.validator.service import ValidationService

    with tempfile.TemporaryDirectory() as td:
        ckpt_dir = os.path.join(td, "ckpt_3")
        os.makedirs(ckpt_dir)
        load_calls = []

        def fake_load(path):
            load_calls.append(path)
            return MagicMock(name="resumed_model")

        svc = ValidationService(
            wallet=_FakeWallet(),
            model=MagicMock(name="base_model"),
            tokenizer=MagicMock(),
            env=_FakeEnv(),
            netuid=99,
            resume_from=f"path:{ckpt_dir}",
            load_model_fn=fake_load,
        )
        await svc._apply_resume_from()

        assert svc.train_model is not None
        assert load_calls == [ckpt_dir]
        mf = svc._checkpoint_store.current_manifest()
        assert mf is not None
        assert mf.checkpoint_n == 3
        assert svc._checkpoint_n == 3
        assert svc._verify_model_checkpoint_revision == ckpt_dir


@pytest.mark.asyncio
async def test_resume_from_none_is_noop():
    """No resume_from → service boots with the base model, no manifest."""
    from reliquary.validator.service import ValidationService
    from unittest.mock import MagicMock
    svc = ValidationService(
        wallet=_FakeWallet(),
        model=MagicMock(),
        tokenizer=MagicMock(),
        env=_FakeEnv(),
        netuid=99,
    )
    await svc._apply_resume_from()
    assert svc._checkpoint_store.current_manifest() is None


@pytest.mark.asyncio
async def test_resume_from_load_failure_aborts():
    """If the resume source fails to load, abort — never fall back silently
    to the base model (would cause GRAIL mismatch on first submission)."""
    from unittest.mock import MagicMock
    from reliquary.validator.service import ValidationService
    import os, tempfile

    def failing_load(path):
        raise RuntimeError("corrupt checkpoint")

    with tempfile.TemporaryDirectory() as td:
        ckpt_dir = os.path.join(td, "ckpt_3")
        os.makedirs(ckpt_dir)
        svc = ValidationService(
            wallet=_FakeWallet(),
            model=MagicMock(),
            tokenizer=MagicMock(),
            env=_FakeEnv(),
            netuid=99,
            resume_from=f"path:{ckpt_dir}",
            load_model_fn=failing_load,
        )
        with pytest.raises(RuntimeError, match="corrupt checkpoint"):
            await svc._apply_resume_from()


@pytest.mark.asyncio
async def test_v3_resume_rejects_checkpoint_without_lineage_before_load(
    tmp_path,
    monkeypatch,
):
    from unittest.mock import MagicMock

    from reliquary.validator.checkpoint_profile import CheckpointProfileMismatch
    from reliquary.validator.service import ValidationService
    import reliquary.validator.service as service_module

    ckpt_dir = tmp_path / "ckpt_3"
    ckpt_dir.mkdir()
    load_calls = []

    def fake_load(path):
        load_calls.append(path)
        return MagicMock(name="resumed_model")

    monkeypatch.setattr(service_module, "PROTOCOL_VERSION", 3)
    svc = ValidationService(
        wallet=_FakeWallet(),
        model=MagicMock(name="base_model"),
        tokenizer=MagicMock(),
        env=_FakeEnv(),
        netuid=99,
        resume_from=f"path:{ckpt_dir}",
        load_model_fn=fake_load,
    )

    with pytest.raises(
        CheckpointProfileMismatch,
        match="no protocol-lineage metadata",
    ):
        await svc._apply_resume_from()
    assert load_calls == []
