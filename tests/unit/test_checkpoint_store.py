"""CheckpointStore: save → upload to HuggingFace → sign → in-memory manifest."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from reliquary.validator.checkpoint import CheckpointStore, ManifestEntry


class FakeWallet:
    class _Hk:
        ss58_address = "5FHk"
        def sign(self, data):
            return b"signature_for_" + data[:16]
    hotkey = _Hk()


def _save_stub(model, tokenizer, path):
    """Pretend to save an HF-format snapshot — write two deterministic files."""
    (path / "model.safetensors").write_bytes(b"weights_for_" + str(id(model)).encode())
    (path / "config.json").write_text('{"model_type": "fake"}')
    if tokenizer is not None:
        (path / "tokenizer.json").write_text("{}")


def test_initial_manifest_is_none(tmp_path):
    store = CheckpointStore(
        validator_hotkey="5FHk",
        wallet=FakeWallet(),
        repo_id="aivolutionedge/reliquary-sn",
        staging_dir_path=str(tmp_path),
    )
    assert store.current_manifest() is None


@pytest.mark.asyncio
async def test_publish_writes_uploads_signs_and_serves(tmp_path):
    fake_upload = AsyncMock(return_value="abc123def456")
    store = CheckpointStore(
        validator_hotkey="5FHk",
        wallet=FakeWallet(),
        repo_id="aivolutionedge/reliquary-sn",
        staging_dir_path=str(tmp_path),
        upload_fn=fake_upload,
        save_fn=_save_stub,
    )
    model = MagicMock(name="mock_model")
    entry = await store.publish(checkpoint_n=1, model=model)
    assert isinstance(entry, ManifestEntry)
    assert entry.checkpoint_n == 1
    assert entry.repo_id == "aivolutionedge/reliquary-sn"
    assert entry.revision == "abc123def456"
    assert entry.signature.startswith("ed25519:")
    fake_upload.assert_awaited_once()
    assert store.current_manifest() is entry


@pytest.mark.asyncio
async def test_publish_increments_overrides_previous(tmp_path):
    fake_upload = AsyncMock(side_effect=[
        "rev_sha_001",
        "rev_sha_002",
    ])
    store = CheckpointStore(
        validator_hotkey="5FHk",
        wallet=FakeWallet(),
        repo_id="aivolutionedge/reliquary-sn",
        staging_dir_path=str(tmp_path),
        upload_fn=fake_upload,
        save_fn=_save_stub,
    )
    await store.publish(checkpoint_n=1, model=MagicMock())
    e2 = await store.publish(checkpoint_n=2, model=MagicMock())
    assert store.current_manifest() is e2
    assert store.current_manifest().checkpoint_n == 2
    assert store.current_manifest().revision == "rev_sha_002"


@pytest.mark.asyncio
async def test_upload_fn_receives_folder_and_kwargs(tmp_path):
    """upload_fn is called with (folder_path, repo_id, commit_message)."""
    captured = {}

    async def capturing_upload(folder_path, repo_id, commit_message):
        # Capture not just the path but also the *state* of the staging
        # directory at upload time — after publish() returns the dir is
        # cleaned up, so directory-content assertions have to be evaluated
        # here, while the directory still exists.
        from pathlib import Path as _P
        captured["folder_path"] = folder_path
        captured["repo_id"] = repo_id
        captured["commit_message"] = commit_message
        captured["is_dir_at_upload"] = _P(folder_path).is_dir()
        captured["has_safetensors"] = (_P(folder_path) / "model.safetensors").exists()
        captured["has_config"] = (_P(folder_path) / "config.json").exists()
        return "captured_revision_sha"

    store = CheckpointStore(
        validator_hotkey="5FHk",
        wallet=FakeWallet(),
        repo_id="aivolutionedge/reliquary-sn",
        staging_dir_path=str(tmp_path),
        upload_fn=capturing_upload,
        save_fn=_save_stub,
    )
    entry = await store.publish(5, model=object())
    assert captured["repo_id"] == "aivolutionedge/reliquary-sn"
    # folder_path was a real directory + had the snapshot files at upload time.
    assert captured["is_dir_at_upload"] is True
    assert captured["has_safetensors"] is True
    assert captured["has_config"] is True
    assert "5" in captured["commit_message"]
    assert entry.revision == "captured_revision_sha"


@pytest.mark.asyncio
async def test_save_receives_tokenizer(tmp_path):
    """Tokenizer passed through __init__ reaches the save function."""
    seen = {}

    def capturing_save(model, tokenizer, path):
        seen["tokenizer"] = tokenizer
        (path / "model.safetensors").write_bytes(b"x")
        (path / "config.json").write_text("{}")

    fake_tokenizer = object()
    fake_upload = AsyncMock(return_value="rev")
    store = CheckpointStore(
        validator_hotkey="5FHk",
        wallet=FakeWallet(),
        repo_id="org/repo",
        staging_dir_path=str(tmp_path),
        tokenizer=fake_tokenizer,
        upload_fn=fake_upload,
        save_fn=capturing_save,
    )
    await store.publish(1, model=object())
    assert seen["tokenizer"] is fake_tokenizer


@pytest.mark.asyncio
async def test_signature_includes_n_and_revision(tmp_path):
    """Signature payload = checkpoint_n || revision so tampering is detectable."""
    captured = {}

    class CapturingWallet:
        class _Hk:
            ss58_address = "5FHk"
            def sign(self, data):
                captured["signed"] = data
                return b"fake_sig"
        hotkey = _Hk()

    async def fake_upload(folder_path, repo_id, commit_message):
        return "revision_sha_42"

    store = CheckpointStore(
        validator_hotkey="5FHk",
        wallet=CapturingWallet(),
        repo_id="aivolutionedge/reliquary-sn",
        staging_dir_path=str(tmp_path),
        upload_fn=fake_upload,
        save_fn=_save_stub,
    )
    entry = await store.publish(42, model=object())
    assert b"42" in captured["signed"]
    assert b"revision_sha_42" in captured["signed"]
    assert entry.signature == "ed25519:" + b"fake_sig".hex()


@pytest.mark.asyncio
async def test_repo_id_stored_in_manifest(tmp_path):
    """ManifestEntry carries the repo_id so miners can do from_pretrained(repo_id, revision)."""
    async def fake_upload(folder_path, repo_id, commit_message):
        return "some_rev"

    store = CheckpointStore(
        validator_hotkey="5FHk",
        wallet=FakeWallet(),
        repo_id="myorg/my-model",
        staging_dir_path=str(tmp_path),
        upload_fn=fake_upload,
        save_fn=_save_stub,
    )
    entry = await store.publish(3, model=object())
    assert entry.repo_id == "myorg/my-model"
    assert entry.revision == "some_rev"


# ---- staging cleanup behaviour ---------------------------------------------
#
# The local ``ckpt_<N>`` directory is staging-only — once HuggingFace has
# acknowledged the upload (returning the revision SHA), the validator
# never reads the local copy again. Miners reload via
# ``snapshot_download(repo_id, revision)`` directly from HF. Keeping the
# local dir caused unbounded disk creep at ~7.6 GB / training step.
# These tests pin the post-fix invariant: the dir is gone when publish()
# returns, both on the happy path and on save/upload failure.


@pytest.mark.asyncio
async def test_publish_deletes_staging_dir_after_successful_upload(tmp_path):
    """After a successful publish(), ckpt_<N> must not exist on disk."""
    fake_upload = AsyncMock(return_value="rev_sha_001")
    store = CheckpointStore(
        validator_hotkey="5FHk",
        wallet=FakeWallet(),
        repo_id="org/repo",
        staging_dir_path=str(tmp_path),
        upload_fn=fake_upload,
        save_fn=_save_stub,
    )
    await store.publish(checkpoint_n=42, model=MagicMock())
    # The ckpt_42 directory must be gone — HF revision is the canonical
    # copy; keeping the local file would creep disk indefinitely.
    assert not (tmp_path / "ckpt_42").exists(), (
        "publish() leaked the local snapshot directory after upload"
    )


@pytest.mark.asyncio
async def test_publish_deletes_staging_dir_on_upload_failure(tmp_path):
    """If the HF upload raises, the staging dir is still cleaned up."""
    async def failing_upload(folder_path, repo_id, commit_message):
        raise RuntimeError("HF rate limit hit")

    store = CheckpointStore(
        validator_hotkey="5FHk",
        wallet=FakeWallet(),
        repo_id="org/repo",
        staging_dir_path=str(tmp_path),
        upload_fn=failing_upload,
        save_fn=_save_stub,
    )
    with pytest.raises(RuntimeError, match="HF rate limit hit"):
        await store.publish(checkpoint_n=7, model=MagicMock())
    # Even on upload failure, the half-written staging dir is cleaned up
    # so retried publishes don't accumulate orphaned checkpoints.
    assert not (tmp_path / "ckpt_7").exists(), (
        "publish() leaked the staging directory after upload failure"
    )


@pytest.mark.asyncio
async def test_publish_deletes_staging_dir_on_save_failure(tmp_path):
    """If the local save raises, the partial dir is still cleaned up."""
    def failing_save(model, tokenizer, path):
        # Simulate a save that partially writes then crashes.
        (path / "partial.bin").write_bytes(b"x")
        raise IOError("disk full")

    fake_upload = AsyncMock(return_value="never_called")
    store = CheckpointStore(
        validator_hotkey="5FHk",
        wallet=FakeWallet(),
        repo_id="org/repo",
        staging_dir_path=str(tmp_path),
        upload_fn=fake_upload,
        save_fn=failing_save,
    )
    with pytest.raises(IOError, match="disk full"):
        await store.publish(checkpoint_n=99, model=MagicMock())
    assert not (tmp_path / "ckpt_99").exists()
    fake_upload.assert_not_called()


@pytest.mark.asyncio
async def test_publish_repeated_calls_do_not_accumulate_directories(tmp_path):
    """N publishes leave 0 ckpt_* dirs behind (pre-fix: N dirs)."""
    fake_upload = AsyncMock(side_effect=[f"rev_{i}" for i in range(5)])
    store = CheckpointStore(
        validator_hotkey="5FHk",
        wallet=FakeWallet(),
        repo_id="org/repo",
        staging_dir_path=str(tmp_path),
        upload_fn=fake_upload,
        save_fn=_save_stub,
    )
    for i in range(1, 6):
        await store.publish(checkpoint_n=i, model=MagicMock())
    leftover = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert leftover == [], (
        f"publish() left orphan staging directories: {leftover}"
    )


@pytest.mark.asyncio
async def test_publish_does_not_block_event_loop_during_save(tmp_path):
    """The local snapshot save is offloaded to a thread, so other coroutines
    (the HTTP server serving /state, /submit) keep running while a multi-GB
    checkpoint is serialised. Regression: the save used to run synchronously
    on the event loop and froze every concurrent request for its duration."""
    import asyncio
    import time

    def slow_save(model, tokenizer, path):
        _save_stub(model, tokenizer, path)
        time.sleep(0.3)  # stand-in for multi-GB safetensors serialisation

    store = CheckpointStore(
        validator_hotkey="5FHk",
        wallet=FakeWallet(),
        repo_id="org/repo",
        staging_dir_path=str(tmp_path),
        upload_fn=AsyncMock(return_value="rev_sha_001"),
        save_fn=slow_save,
    )

    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    hb = asyncio.create_task(heartbeat())
    try:
        await store.publish(checkpoint_n=1, model=MagicMock())
    finally:
        hb.cancel()

    # The 0.3s save overlaps ~30 heartbeat ticks when the loop stays free.
    # If the save blocked the loop, ticks would be ~0. Slack for slow CI.
    assert ticks >= 5, f"event loop was starved during save (ticks={ticks})"
