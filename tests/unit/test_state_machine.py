"""ValidationService state machine: OPEN → TRAINING → PUBLISHING → READY."""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from reliquary.constants import B_BATCH
from reliquary.protocol.submission import WindowState


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
    return svc


def test_service_initial_state_is_ready():
    svc = _make_service()
    assert svc._current_window_state == WindowState.READY


def test_open_window_sets_state_to_open():
    svc = _make_service()
    svc._open_window()
    svc._activate_window()
    assert svc._current_window_state == WindowState.OPEN
    assert svc._active_batcher is not None


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
    from reliquary.validator.service import MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW

    svc = _make_service()
    svc._open_window()
    svc._activate_window()
    batcher = svc._active_batcher

    # Exhaustion is now gated on the never-refunded grading-attempts ceiling,
    # since out_of_zone refunds the GRAIL candidate budget.
    batcher._proof_grading_attempts = MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW
    assert batcher.valid_count == 0
    assert svc.server.submit_queue_depth == 0
    assert svc.server.proof_verification_inflight == 0

    reason = await svc._wait_for_window_seal()

    assert reason == "proof_admission_exhausted_drained"
    assert batcher.is_sealed()
    assert batcher.force_seal_reason == "proof_admission_exhausted_drained"


def test_proof_cap_breaker_waits_for_inflight_or_queued_work():
    from reliquary.validator.service import MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW

    svc = _make_service()
    svc._open_window()
    svc._activate_window()
    batcher = svc._active_batcher
    batcher._proof_grading_attempts = MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW

    svc.server._inflight_proofs = 1
    assert svc._proof_admission_exhausted_and_drained(batcher) is False

    svc.server._inflight_proofs = 0
    svc.server._submit_queue.put_nowait((object(), batcher, object()))
    assert svc._proof_admission_exhausted_and_drained(batcher) is False


def test_proof_cap_breaker_uses_distinct_prompt_count():
    """Raw valid duplicates should not mask an unfillable trainable shortfall."""
    from reliquary.validator.service import MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW

    svc = _make_service()
    svc._open_window()
    svc._activate_window()
    batcher = svc._active_batcher
    batcher._valid = [
        SimpleNamespace(prompt_idx=i) for i in range(B_BATCH - 1)
    ] + [SimpleNamespace(prompt_idx=0)]
    batcher.valid_count = B_BATCH
    batcher._proof_grading_attempts = MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW

    assert batcher.distinct_valid_prompt_count() == B_BATCH - 1
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

    async def _fake_publish(checkpoint_n, model):
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
