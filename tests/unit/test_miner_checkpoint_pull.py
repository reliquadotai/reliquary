"""Miner detects new checkpoint_n via /state and downloads via HF."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from reliquary.miner.engine import maybe_pull_checkpoint


@pytest.mark.asyncio
async def test_pull_when_remote_n_higher():
    """Remote checkpoint_n > local → download triggered."""
    state = MagicMock()
    state.checkpoint_n = 5
    state.checkpoint_repo_id = "aivolutionedge/reliquary-sn"
    state.checkpoint_revision = "rev_sha_005"

    download_fn = AsyncMock(return_value="/hf_cache/model_5")
    load_fn = MagicMock(return_value="loaded_model_5")

    new_local_n, new_hash, new_model = await maybe_pull_checkpoint(
        state=state, local_n=4, local_hash="rev_old", local_model="old_model",
        download_fn=download_fn, load_fn=load_fn,
    )
    assert new_local_n == 5
    assert new_hash == "rev_sha_005"
    assert new_model == "loaded_model_5"
    download_fn.assert_awaited_once_with("aivolutionedge/reliquary-sn", "rev_sha_005")
    load_fn.assert_called_once_with("/hf_cache/model_5")


@pytest.mark.asyncio
async def test_no_pull_when_local_up_to_date():
    state = MagicMock()
    state.checkpoint_n = 5
    state.checkpoint_repo_id = "aivolutionedge/reliquary-sn"
    state.checkpoint_revision = "rev_sha_005"

    download_fn = AsyncMock()
    load_fn = MagicMock()

    new_local_n, new_hash, new_model = await maybe_pull_checkpoint(
        state=state, local_n=5, local_hash="rev_sha_005", local_model="cached",
        download_fn=download_fn, load_fn=load_fn,
    )
    assert new_local_n == 5
    assert new_hash == "rev_sha_005"
    assert new_model == "cached"
    download_fn.assert_not_called()


@pytest.mark.asyncio
async def test_no_pull_when_repo_missing():
    """Pre-first-publish: state has checkpoint_n=0 and repo_id=None — don't try to download."""
    state = MagicMock()
    state.checkpoint_n = 0
    state.checkpoint_repo_id = None
    state.checkpoint_revision = None

    download_fn = AsyncMock()
    load_fn = MagicMock()

    new_local_n, new_hash, new_model = await maybe_pull_checkpoint(
        state=state, local_n=0, local_hash="", local_model="initial_model",
        download_fn=download_fn, load_fn=load_fn,
    )
    assert new_local_n == 0
    assert new_model == "initial_model"
    download_fn.assert_not_called()


@pytest.mark.asyncio
async def test_no_pull_when_revision_missing():
    """repo_id present but revision=None — don't try to download."""
    state = MagicMock()
    state.checkpoint_n = 3
    state.checkpoint_repo_id = "aivolutionedge/reliquary-sn"
    state.checkpoint_revision = None

    download_fn = AsyncMock()
    load_fn = MagicMock()

    new_local_n, new_hash, new_model = await maybe_pull_checkpoint(
        state=state, local_n=2, local_hash="", local_model="local",
        download_fn=download_fn, load_fn=load_fn,
    )
    assert new_local_n == 2
    download_fn.assert_not_called()


@pytest.mark.asyncio
async def test_pull_from_zero_local_to_nonzero_remote():
    """Fresh miner (local_n=0) joins active subnet (remote_n=7)."""
    state = MagicMock()
    state.checkpoint_n = 7
    state.checkpoint_repo_id = "aivolutionedge/reliquary-sn"
    state.checkpoint_revision = "rev_sha_007"

    download_fn = AsyncMock(return_value="/hf_cache/model_7")
    load_fn = MagicMock(return_value="model_7")

    new_local_n, new_hash, new_model = await maybe_pull_checkpoint(
        state=state, local_n=0, local_hash="", local_model=None,
        download_fn=download_fn, load_fn=load_fn,
    )
    assert new_local_n == 7
    assert new_hash == "rev_sha_007"
    assert new_model == "model_7"


@pytest.mark.asyncio
async def test_pull_when_revision_changes_at_same_checkpoint_number():
    state = MagicMock()
    state.checkpoint_n = 5
    state.checkpoint_repo_id = "aivolutionedge/reliquary-sn"
    state.checkpoint_revision = "corrected_rev"
    download_fn = AsyncMock(return_value="/hf_cache/corrected")
    load_fn = MagicMock(return_value="corrected_model")

    result = await maybe_pull_checkpoint(
        state=state,
        local_n=5,
        local_hash="old_rev",
        local_model="old_model",
        download_fn=download_fn,
        load_fn=load_fn,
    )

    assert result == (5, "corrected_rev", "corrected_model")
    download_fn.assert_awaited_once()


@pytest.mark.asyncio
async def test_loader_failure_does_not_advance_checkpoint_identity():
    state = MagicMock()
    state.checkpoint_n = 5
    state.checkpoint_repo_id = "aivolutionedge/reliquary-sn"
    state.checkpoint_revision = "new_rev"

    with pytest.raises(RuntimeError, match="no activated model"):
        await maybe_pull_checkpoint(
            state=state,
            local_n=4,
            local_hash="old_rev",
            local_model="old_model",
            download_fn=AsyncMock(return_value="/hf_cache/new"),
            load_fn=MagicMock(return_value=None),
        )
