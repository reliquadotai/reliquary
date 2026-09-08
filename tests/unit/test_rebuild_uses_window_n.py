"""Regression: _rebuild_cooldown_from_history uses self._window_n,
not a block-derived window. The archives on R2 are keyed by window_n
since v2.1."""

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


@pytest.mark.asyncio
async def test_rebuild_uses_state_window_n_not_block_derived():
    """Rebuild must call list_recent_datasets with _window_n + 1, not a
    block-derived value. Proves the bug fix holds."""
    from reliquary.validator.service import ValidationService

    svc = ValidationService(
        wallet=_FakeWallet(), model=MagicMock(), tokenizer=MagicMock(),
        env=_FakeEnv(), netuid=99,
    )
    svc._window_n = 42  # authoritative v2.1 counter

    captured = {}
    async def fake_list(current_window, n, *, strict=False):
        captured["current_window"] = current_window
        captured["strict"] = strict
        return [
            {"window_start": window_start, "batch": []}
            for window_start in range(1, 43)
        ]

    # No snapshot for the default run -> falls back to the bounded archive scan.
    with patch(
        "reliquary.infrastructure.storage.download_json",
        new=AsyncMock(return_value=None),
    ), patch(
        "reliquary.infrastructure.storage.list_recent_datasets",
        new=fake_list,
    ):
        await svc._rebuild_cooldown_from_history()

    # _rebuild passes current_window = _window_n + 1 to list_recent_datasets
    assert captured["current_window"] == 43
    assert captured["strict"] is True
