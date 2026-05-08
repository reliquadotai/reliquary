"""ValidationService._bootstrap_state_from_external: derives window_n + checkpoint_n + EMA from R2 + HF."""

from collections import defaultdict
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@dataclass
class _FakeEnv:
    @property
    def name(self): return "fake"
    def __len__(self): return 100
    def get_problem(self, i): return {"prompt": "p", "ground_truth": "", "id": f"p{i}"}
    def compute_reward(self, p, c): return 1.0


class _FakeWallet:
    class _Hk:
        ss58_address = "5FHk"
        @staticmethod
        def sign(d): return b"sig"
    hotkey = _Hk()


def _make_service():
    from reliquary.validator.service import ValidationService
    return ValidationService(
        wallet=_FakeWallet(), model=MagicMock(), tokenizer=MagicMock(),
        env=_FakeEnv(), netuid=99,
    )


@pytest.mark.asyncio
async def test_bootstrap_sets_window_n_from_r2():
    """window_n is set to max R2 window key."""
    svc = _make_service()

    with (
        patch(
            "reliquary.infrastructure.storage.list_all_window_keys",
            new=AsyncMock(return_value=[1, 5, 10, 42]),
        ),
        patch(
            "reliquary.infrastructure.storage.list_recent_datasets",
            new=AsyncMock(return_value=[]),
        ),
        patch("huggingface_hub.HfApi.list_repo_commits", return_value=[]),
    ):
        await svc._bootstrap_state_from_external()

    assert svc._window_n == 42


@pytest.mark.asyncio
async def test_bootstrap_window_n_zero_when_no_archives():
    """No archives → window_n stays 0."""
    svc = _make_service()

    with (
        patch(
            "reliquary.infrastructure.storage.list_all_window_keys",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "reliquary.infrastructure.storage.list_recent_datasets",
            new=AsyncMock(return_value=[]),
        ),
        patch("huggingface_hub.HfApi.list_repo_commits", return_value=[]),
    ):
        await svc._bootstrap_state_from_external()

    assert svc._window_n == 0


@pytest.mark.asyncio
async def test_bootstrap_checkpoint_n_from_hf_commits():
    """checkpoint_n is count of HF commits whose title starts with 'checkpoint '."""
    svc = _make_service()

    fake_commits = [
        MagicMock(title="checkpoint 1"),
        MagicMock(title="checkpoint 2"),
        MagicMock(title="checkpoint 3"),
        MagicMock(title="seed from Qwen/Qwen3-4B"),  # not a checkpoint commit
    ]

    with (
        patch(
            "reliquary.infrastructure.storage.list_all_window_keys",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "reliquary.infrastructure.storage.list_recent_datasets",
            new=AsyncMock(return_value=[]),
        ),
        patch("huggingface_hub.HfApi.list_repo_commits", return_value=fake_commits),
    ):
        await svc._bootstrap_state_from_external()

    assert svc._checkpoint_n == 3


@pytest.mark.asyncio
async def test_bootstrap_tolerates_r2_failure():
    """R2 failure during window key fetch → window_n stays 0, no exception raised."""
    svc = _make_service()

    with (
        patch(
            "reliquary.infrastructure.storage.list_all_window_keys",
            new=AsyncMock(side_effect=RuntimeError("R2 down")),
        ),
        patch(
            "reliquary.infrastructure.storage.list_recent_datasets",
            new=AsyncMock(return_value=[]),
        ),
        patch("huggingface_hub.HfApi.list_repo_commits", return_value=[]),
    ):
        await svc._bootstrap_state_from_external()  # must not raise

    assert svc._window_n == 0


@pytest.mark.asyncio
async def test_bootstrap_tolerates_hf_failure():
    """HF failure → checkpoint_n stays 0, no exception raised."""
    svc = _make_service()

    with (
        patch(
            "reliquary.infrastructure.storage.list_all_window_keys",
            new=AsyncMock(return_value=[1, 2]),
        ),
        patch(
            "reliquary.infrastructure.storage.list_recent_datasets",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "huggingface_hub.HfApi.list_repo_commits",
            side_effect=Exception("HF auth failed"),
        ),
    ):
        await svc._bootstrap_state_from_external()  # must not raise

    assert svc._checkpoint_n == 0
