"""Miner checkpoint activation binds number, repository, and immutable OID."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from reliquary.miner.engine import CheckpointIdentityError, maybe_pull_checkpoint


REPO = "aivolutionedge/reliquary-sn"
OTHER_REPO = "aivolutionedge/reliquary-sn-mirror"
REV_OLD = "4" * 40
REV_5 = "5" * 40
REV_7 = "7" * 40
REV_NEW = "a" * 40


@pytest.mark.asyncio
async def test_pull_when_remote_n_higher():
    state = MagicMock(
        checkpoint_n=5,
        checkpoint_repo_id=REPO,
        checkpoint_revision=REV_5,
    )
    download_fn = AsyncMock(return_value="/hf_cache/model_5")
    load_fn = MagicMock(return_value="loaded_model_5")

    result = await maybe_pull_checkpoint(
        state=state,
        local_n=4,
        local_repo_id=REPO,
        local_hash=REV_OLD,
        local_model="old_model",
        download_fn=download_fn,
        load_fn=load_fn,
    )

    assert result == (5, REPO, REV_5, "loaded_model_5")
    download_fn.assert_awaited_once_with(REPO, REV_5)
    load_fn.assert_called_once_with("/hf_cache/model_5")


@pytest.mark.asyncio
async def test_no_pull_when_local_identity_is_exact():
    state = MagicMock(
        checkpoint_n=5,
        checkpoint_repo_id=REPO,
        checkpoint_revision=REV_5,
    )
    download_fn = AsyncMock()

    result = await maybe_pull_checkpoint(
        state=state,
        local_n=5,
        local_repo_id=REPO,
        local_hash=REV_5,
        local_model="cached",
        download_fn=download_fn,
        load_fn=MagicMock(),
    )

    assert result == (5, REPO, REV_5, "cached")
    download_fn.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_pull_before_first_publish():
    state = MagicMock(
        checkpoint_n=0,
        checkpoint_repo_id=None,
        checkpoint_revision=None,
    )
    download_fn = AsyncMock()

    result = await maybe_pull_checkpoint(
        state=state,
        local_n=0,
        local_repo_id="",
        local_hash="",
        local_model="initial_model",
        download_fn=download_fn,
        load_fn=MagicMock(),
    )

    assert result == (0, "", "", "initial_model")
    download_fn.assert_not_awaited()


@pytest.mark.asyncio
async def test_partial_checkpoint_identity_is_rejected():
    state = MagicMock(
        checkpoint_n=3,
        checkpoint_repo_id=REPO,
        checkpoint_revision=None,
    )
    download_fn = AsyncMock()

    with pytest.raises(CheckpointIdentityError, match="invalid"):
        await maybe_pull_checkpoint(
            state=state,
            local_n=2,
            local_repo_id=REPO,
            local_hash=REV_OLD,
            local_model="local",
            download_fn=download_fn,
            load_fn=MagicMock(),
        )

    download_fn.assert_not_awaited()


@pytest.mark.asyncio
async def test_fresh_miner_joins_a_later_checkpoint():
    state = MagicMock(
        checkpoint_n=7,
        checkpoint_repo_id=REPO,
        checkpoint_revision=REV_7,
    )

    result = await maybe_pull_checkpoint(
        state=state,
        local_n=0,
        local_repo_id="",
        local_hash="",
        local_model=None,
        download_fn=AsyncMock(return_value="/hf_cache/model_7"),
        load_fn=MagicMock(return_value="model_7"),
    )

    assert result == (7, REPO, REV_7, "model_7")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("remote_repo", "remote_revision"),
    [(REPO, REV_NEW), (OTHER_REPO, REV_5)],
)
async def test_same_number_identity_rebinding_fails_closed(
    remote_repo,
    remote_revision,
):
    state = MagicMock(
        checkpoint_n=5,
        checkpoint_repo_id=remote_repo,
        checkpoint_revision=remote_revision,
    )
    download_fn = AsyncMock()

    with pytest.raises(CheckpointIdentityError, match="rebound"):
        await maybe_pull_checkpoint(
            state=state,
            local_n=5,
            local_repo_id=REPO,
            local_hash=REV_5,
            local_model="old_model",
            download_fn=download_fn,
            load_fn=MagicMock(),
        )

    download_fn.assert_not_awaited()


@pytest.mark.asyncio
async def test_same_number_can_initialize_an_unknown_local_identity():
    state = MagicMock(
        checkpoint_n=5,
        checkpoint_repo_id=REPO,
        checkpoint_revision=REV_5,
    )

    result = await maybe_pull_checkpoint(
        state=state,
        local_n=5,
        local_repo_id="",
        local_hash="",
        local_model="initial_model",
        download_fn=AsyncMock(return_value="/hf_cache/model_5"),
        load_fn=MagicMock(return_value="loaded_model_5"),
    )

    assert result == (5, REPO, REV_5, "loaded_model_5")


@pytest.mark.asyncio
async def test_mutable_revision_is_rejected_before_download():
    state = MagicMock(
        checkpoint_n=5,
        checkpoint_repo_id=REPO,
        checkpoint_revision="main",
    )
    download_fn = AsyncMock()

    with pytest.raises(CheckpointIdentityError, match="invalid"):
        await maybe_pull_checkpoint(
            state=state,
            local_n=4,
            local_repo_id=REPO,
            local_hash=REV_OLD,
            local_model="old_model",
            download_fn=download_fn,
            load_fn=MagicMock(),
        )

    download_fn.assert_not_awaited()


@pytest.mark.asyncio
async def test_loader_failure_does_not_advance_checkpoint_identity():
    state = MagicMock(
        checkpoint_n=5,
        checkpoint_repo_id=REPO,
        checkpoint_revision=REV_NEW,
    )

    with pytest.raises(RuntimeError, match="no activated model"):
        await maybe_pull_checkpoint(
            state=state,
            local_n=4,
            local_repo_id=REPO,
            local_hash=REV_OLD,
            local_model="old_model",
            download_fn=AsyncMock(return_value="/hf_cache/new"),
            load_fn=MagicMock(return_value=None),
        )
