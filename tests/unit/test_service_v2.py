"""End-to-end: service creates GrpoWindowBatcher per window, seals at window
close, computes weights v2-flavoured."""

import asyncio
from dataclasses import dataclass
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reliquary.constants import B_BATCH
from reliquary.validator.batcher import GrpoWindowBatcher, ValidSubmission
from reliquary.validator.cooldown import CooldownMap


@dataclass
class _FakeEnv:
    def __len__(self): return 100
    def get_problem(self, i): return {"prompt": "p", "ground_truth": "", "id": f"p{i}"}
    def compute_reward(self, p, c): return 1.0


class _LateDropFakeEnv:
    name = "fake"
    def __len__(self): return 100
    def get_problem(self, i): return {"prompt": "p", "ground_truth": "a"}
    def compute_reward(self, p, c): return 0.0


class _LateDropFakeWallet:
    class _Hk:
        ss58_address = "5FHk"
        @staticmethod
        def sign(d): return b"sig"
    hotkey = _Hk()


def _build_late_drop_service():
    """Bare ValidationService for late-drop tests."""
    from unittest.mock import MagicMock
    from reliquary.validator.service import ValidationService
    fake_tok = MagicMock()
    fake_tok.eos_token_id = 99
    return ValidationService(
        wallet=_LateDropFakeWallet(), model=MagicMock(), tokenizer=fake_tok,
        env=_LateDropFakeEnv(), netuid=99,
    )


@pytest.mark.asyncio
async def test_registration_refresh_skips_chain_until_epoch_due():
    svc = _build_late_drop_service()
    svc.server.set_registered_hotkeys(
        {"hk-a"}, operator_by_hotkey={"hk-a": "operator-a"}
    )

    with patch(
        "reliquary.validator.service.chain.get_subtensor",
        new=AsyncMock(),
    ) as get_subtensor:
        refreshed = await svc._refresh_registered_hotkeys(
            reason="epoch_boundary"
        )

    assert refreshed is True
    get_subtensor.assert_not_awaited()


@pytest.mark.asyncio
async def test_registration_refresh_runs_once_snapshot_is_epoch_old():
    from reliquary.constants import REGISTERED_HOTKEY_CACHE_TTL_SECONDS

    svc = _build_late_drop_service()
    svc.server.set_registered_hotkeys(
        {"hk-old"},
        refreshed_at=time.time() - REGISTERED_HOTKEY_CACHE_TTL_SECONDS - 1,
        operator_by_hotkey={"hk-old": "operator-old"},
    )
    subtensor = object()
    neurons = [SimpleNamespace(hotkey="hk-new", coldkey="operator-new")]

    with (
        patch(
            "reliquary.validator.service.chain.get_subtensor",
            new=AsyncMock(return_value=subtensor),
        ) as get_subtensor,
        patch(
            "reliquary.validator.service.chain.get_neurons_lite",
            new=AsyncMock(return_value=neurons),
        ),
        patch(
            "reliquary.validator.service.chain.close_subtensor",
            new=AsyncMock(),
        ),
    ):
        refreshed = await svc._refresh_registered_hotkeys(
            reason="epoch_boundary"
        )

    assert refreshed is True
    assert svc.server._registered_hotkeys == frozenset({"hk-new"})
    get_subtensor.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_epoch_refresh_retries_at_next_boundary():
    from reliquary.constants import REGISTERED_HOTKEY_CACHE_TTL_SECONDS

    svc = _build_late_drop_service()
    old_refreshed_at = time.time() - REGISTERED_HOTKEY_CACHE_TTL_SECONDS - 1
    svc.server.set_registered_hotkeys(
        {"hk-a"},
        refreshed_at=old_refreshed_at,
        operator_by_hotkey={"hk-a": "operator-a"},
    )
    subtensor = object()

    with (
        patch(
            "reliquary.validator.service.chain.get_subtensor",
            new=AsyncMock(
                side_effect=[ConnectionError("chain down"), subtensor]
            ),
        ) as get_subtensor,
        patch(
            "reliquary.validator.service.chain.get_neurons_lite",
            new=AsyncMock(
                return_value=[
                    SimpleNamespace(hotkey="hk-b", coldkey="operator-b")
                ]
            ),
        ),
        patch(
            "reliquary.validator.service.chain.close_subtensor",
            new=AsyncMock(),
        ),
    ):
        first = await svc._refresh_registered_hotkeys(reason="epoch_boundary")
        second = await svc._refresh_registered_hotkeys(reason="epoch_boundary")

    assert first is False
    assert second is True
    assert get_subtensor.await_count == 2
    assert svc.server._registered_hotkeys == frozenset({"hk-b"})


@pytest.mark.asyncio
async def test_registration_refresh_uses_lite_neurons_and_updates_server():
    svc = _build_late_drop_service()
    subtensor = object()
    neurons = [
        SimpleNamespace(hotkey=" hk-a ", coldkey=" operator-a "),
        SimpleNamespace(hotkey="hk-b", coldkey="operator-b"),
        SimpleNamespace(hotkey="", coldkey="ignored"),
        SimpleNamespace(hotkey=None, coldkey="ignored"),
    ]

    with (
        patch(
            "reliquary.validator.service.chain.get_subtensor",
            new=AsyncMock(return_value=subtensor),
        ) as get_subtensor,
        patch(
            "reliquary.validator.service.chain.get_neurons_lite",
            new=AsyncMock(return_value=neurons),
        ) as get_neurons_lite,
        patch(
            "reliquary.validator.service.chain.close_subtensor",
            new=AsyncMock(),
        ) as close_subtensor,
    ):
        refreshed = await svc._refresh_registered_hotkeys(force=True)

    assert refreshed is True
    assert svc.server._registered_hotkeys == frozenset({"hk-a", "hk-b"})
    assert svc.server.operator_by_hotkey_snapshot() == {
        "hk-a": "operator-a",
        "hk-b": "operator-b",
    }
    health = svc.server._health_payload()
    assert health.registration_cache_refresh_attempts_total == 1
    assert health.registration_cache_refresh_successes_total == 1
    assert health.registration_cache_refresh_failures_total == 0
    assert health.registration_cache_last_refresh_reason == "unspecified"
    get_subtensor.assert_awaited_once()
    get_neurons_lite.assert_awaited_once_with(subtensor, 99)
    close_subtensor.assert_awaited_once_with(subtensor)


@pytest.mark.asyncio
async def test_registration_refresh_failure_preserves_last_known_good():
    svc = _build_late_drop_service()
    svc.server.set_registered_hotkeys(
        {"hk-a"},
        refreshed_at=123.0,
        operator_by_hotkey={"hk-a": "operator-a"},
    )

    with (
        patch(
            "reliquary.validator.service.chain.get_subtensor",
            new=AsyncMock(side_effect=ConnectionError("chain down")),
        ),
        patch(
            "reliquary.validator.service.chain.close_subtensor",
            new=AsyncMock(),
        ) as close_subtensor,
    ):
        refreshed = await svc._refresh_registered_hotkeys(force=True)

    assert refreshed is False
    assert svc.server._registered_hotkeys == frozenset({"hk-a"})
    assert svc.server.operator_by_hotkey_snapshot() == {
        "hk-a": "operator-a"
    }
    assert svc.server._registration_refreshed_at == 123.0
    health = svc.server._health_payload()
    assert health.registration_cache_refresh_attempts_total == 1
    assert health.registration_cache_refresh_successes_total == 0
    assert health.registration_cache_refresh_failures_total == 1
    assert health.registration_cache_last_refresh_failure_type == "ConnectionError"
    assert health.registration_cache_last_refresh_failure_reason == "unspecified"
    close_subtensor.assert_awaited_once_with(None)


@pytest.mark.asyncio
async def test_empty_lite_neuron_refresh_preserves_last_known_good():
    svc = _build_late_drop_service()
    svc.server.set_registered_hotkeys(
        {"hk-a"},
        refreshed_at=123.0,
        operator_by_hotkey={"hk-a": "operator-a"},
    )
    subtensor = object()

    with (
        patch(
            "reliquary.validator.service.chain.get_subtensor",
            new=AsyncMock(return_value=subtensor),
        ),
        patch(
            "reliquary.validator.service.chain.get_neurons_lite",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "reliquary.validator.service.chain.close_subtensor",
            new=AsyncMock(),
        ) as close_subtensor,
    ):
        refreshed = await svc._refresh_registered_hotkeys(force=True)

    assert refreshed is False
    assert svc.server._registered_hotkeys == frozenset({"hk-a"})
    assert svc.server.operator_by_hotkey_snapshot() == {
        "hk-a": "operator-a"
    }
    assert svc.server._registration_refreshed_at == 123.0
    close_subtensor.assert_awaited_once_with(subtensor)


@pytest.mark.asyncio
async def test_partial_operator_mapping_does_not_break_registration_refresh():
    svc = _build_late_drop_service()
    subtensor = object()
    neurons = [
        SimpleNamespace(hotkey="hk-a", coldkey="operator-a"),
        SimpleNamespace(hotkey="hk-b", coldkey=None),
    ]

    with (
        patch(
            "reliquary.validator.service.chain.get_subtensor",
            new=AsyncMock(return_value=subtensor),
        ),
        patch(
            "reliquary.validator.service.chain.get_neurons_lite",
            new=AsyncMock(return_value=neurons),
        ),
        patch(
            "reliquary.validator.service.chain.close_subtensor",
            new=AsyncMock(),
        ),
    ):
        refreshed = await svc._refresh_registered_hotkeys(force=True)

    assert refreshed is True
    assert svc.server._registered_hotkeys == frozenset({"hk-a", "hk-b"})
    assert svc.server.operator_by_hotkey_snapshot() == {
        "hk-a": "operator-a"
    }


@pytest.mark.asyncio
async def test_conflicting_lite_neuron_owners_are_left_unmapped():
    svc = _build_late_drop_service()
    subtensor = object()
    neurons = [
        SimpleNamespace(hotkey="hk-a", coldkey="operator-a"),
        SimpleNamespace(hotkey="hk-a", coldkey="operator-b"),
        SimpleNamespace(hotkey="hk-b", coldkey="operator-b"),
    ]

    with (
        patch(
            "reliquary.validator.service.chain.get_subtensor",
            new=AsyncMock(return_value=subtensor),
        ),
        patch(
            "reliquary.validator.service.chain.get_neurons_lite",
            new=AsyncMock(return_value=neurons),
        ),
        patch(
            "reliquary.validator.service.chain.close_subtensor",
            new=AsyncMock(),
        ),
    ):
        refreshed = await svc._refresh_registered_hotkeys(force=True)

    assert refreshed is True
    assert svc.server._registered_hotkeys == frozenset({"hk-a", "hk-b"})
    assert svc.server.operator_by_hotkey_snapshot() == {"hk-b": "operator-b"}


def test_service_creates_grpo_window_batcher():
    """The service's open_grpo_window() returns a GrpoWindowBatcher wired up
    with the shared CooldownMap."""
    from reliquary.validator.service import open_grpo_window

    shared_cooldown = CooldownMap(cooldown_windows=50)
    batcher = open_grpo_window(
        window_start=100,
        env=_FakeEnv(),
        model=None,
        cooldown_map=shared_cooldown,
        hash_set=None,
        tokenizer=MagicMock(),
        operator_by_hotkey={"hk-a": "operator-a"},
    )
    assert isinstance(batcher, GrpoWindowBatcher)
    assert batcher.window_start == 100
    assert batcher._cooldown is shared_cooldown
    assert batcher._operator_by_hotkey == {"hk-a": "operator-a"}


@pytest.mark.asyncio
async def test_rebuild_cooldown_from_history_populates_map():
    """ValidationService._rebuild_cooldown_from_history fetches archives
    from R2 and populates the cooldown map."""
    from reliquary.constants import BATCH_PROMPT_COOLDOWN_WINDOWS
    from reliquary.validator.service import ValidationService

    # Build a fake service — minimal state to call the rebuild method.
    class FakeWallet:
        class _Hk:
            ss58_address = "5FHk"
        hotkey = _Hk()

    svc = ValidationService(
        wallet=FakeWallet(),
        model=None,
        tokenizer=None,
        env=_FakeEnv(),
        netuid=99,
    )

    archives = [
        {"window_start": 100, "batch": [{"prompt_idx": 42}]},
        {"window_start": 101, "batch": [{"prompt_idx": 7}]},
    ]
    svc._window_n = 105  # authoritative counter

    with patch(
        "reliquary.infrastructure.storage.list_recent_datasets",
        new=AsyncMock(return_value=archives),
    ):
        await svc._rebuild_cooldown_from_history()

    # Should now know about prompts 42 and 7.
    assert len(svc._cooldown_map) == 2


def test_open_window_does_not_expose_batcher_before_activation():
    """_open_window builds the batcher in a non-active state; only
    _activate_window registers it with the HTTP server.

    Regression for the prod cascade where finney WebSocket 503s during
    _set_window_randomness left the batcher exposed with randomness=""
    and every miner submission crashed in indices_from_root('').
    """
    from reliquary.validator.service import ValidationService, WindowState

    class FakeWallet:
        class _Hk:
            ss58_address = "5FHk"
        hotkey = _Hk()

    svc = ValidationService(
        wallet=FakeWallet(),
        model=None,
        tokenizer=MagicMock(),
        env=_FakeEnv(),
        netuid=99,
    )
    svc.server = MagicMock()
    svc._checkpoint_store = MagicMock()
    svc._checkpoint_store.current_manifest.return_value = None

    svc._open_window()
    # Batcher exists internally but the server hasn't been told.
    assert svc._active_batcher is not None
    svc.server.set_active_batchers.assert_not_called()
    assert svc._current_window_state != WindowState.OPEN

    svc._activate_window()
    # Now the server is registered and the state has flipped to OPEN.
    svc.server.set_active_batchers.assert_called_once_with(svc._active_batchers)
    assert svc._current_window_state == WindowState.OPEN


def test_service_has_separate_verify_and_train_models():
    """ValidationService keeps verify_model and train_model as distinct
    PyTorch objects. The verify model is frozen (eval mode,
    requires_grad=False); the train model is trainable.
    """
    import torch.nn as nn
    from reliquary.validator.service import ValidationService

    train = nn.Linear(4, 4)
    svc = ValidationService(
        wallet=MagicMock(hotkey=MagicMock(ss58_address="x")),
        model=train,
        tokenizer=MagicMock(),
        env=_FakeEnv(),
        netuid=99,
    )
    assert svc.train_model is train
    assert svc.verify_model is not train
    assert all(not p.requires_grad for p in svc.verify_model.parameters())
    assert not svc.verify_model.training  # eval mode

    # In-place refresh works: mutate train, copy into verify, check.
    import torch
    with torch.no_grad():
        for p in svc.train_model.parameters():
            p.add_(1.0)
    svc.verify_model.load_state_dict(svc.train_model.state_dict())
    for p_t, p_v in zip(svc.train_model.parameters(), svc.verify_model.parameters()):
        assert torch.equal(p_t, p_v)


def test_service_constructs_hash_set_with_dedup_retention():
    """ValidationService owns a RolloutHashSet sized to HASH_DEDUP_RETENTION_WINDOWS,
    decoupled from the prompt cooldown horizon so the dedup outlives cooldown."""
    from unittest.mock import MagicMock
    from reliquary.constants import (
        BATCH_PROMPT_COOLDOWN_WINDOWS,
        HASH_DEDUP_RETENTION_WINDOWS,
    )
    from reliquary.validator.dedup import RolloutHashSet
    from reliquary.validator.service import ValidationService

    class _FakeEnv:
        name = "fake"
        def __len__(self): return 100
        def get_problem(self, i): return {"prompt": "p", "ground_truth": "a"}
        def compute_reward(self, p, c): return 0.0

    class _FakeWallet:
        class _Hk:
            ss58_address = "5FHk"
            @staticmethod
            def sign(d): return b"sig"
        hotkey = _Hk()

    fake_tok = MagicMock()
    fake_tok.eos_token_id = 99
    svc = ValidationService(
        wallet=_FakeWallet(), model=MagicMock(), tokenizer=fake_tok,
        env=_FakeEnv(), netuid=99,
    )
    assert isinstance(svc._hash_set, RolloutHashSet)
    assert svc._hash_set._retention_windows == HASH_DEDUP_RETENTION_WINDOWS
    # v2.3+: with ``BATCH_PROMPT_COOLDOWN_WINDOWS`` bumped to 1_000_000
    # (one-shot prompts on the 14M-prompt OpenMathInstruct env), cooldown
    # is now effectively unbounded for the lifetime of a training run.
    # The pre-v2.3 invariant "hash retention > cooldown" no longer holds
    # by design — a prompt that won a slot will never come back within
    # cooldown, so the hash horizon doesn't need to outlive cooldown.
    # What MUST still hold: hash retention is positive and finite, so the
    # in-memory hash set doesn't grow without bound (R2 archive replay
    # bounds it to HASH_DEDUP_RETENTION_WINDOWS recent windows).
    assert HASH_DEDUP_RETENTION_WINDOWS > 0
    assert HASH_DEDUP_RETENTION_WINDOWS < 1_000_000_000


@pytest.mark.asyncio
async def test_rebuild_hashes_from_history_populates_set():
    """_rebuild_hashes_from_history reads R2 archives and seeds the hash set."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from reliquary.validator.dedup import compute_rollout_hash
    from reliquary.validator.service import ValidationService

    class _FakeEnv:
        name = "fake"
        def __len__(self): return 100
        def get_problem(self, i): return {"prompt": "p", "ground_truth": "a"}
        def compute_reward(self, p, c): return 0.0

    class _FakeWallet:
        class _Hk:
            ss58_address = "5FHk"
            @staticmethod
            def sign(d): return b"sig"
        hotkey = _Hk()

    fake_tok = MagicMock()
    fake_tok.eos_token_id = 99
    svc = ValidationService(
        wallet=_FakeWallet(), model=MagicMock(), tokenizer=fake_tok,
        env=_FakeEnv(), netuid=99,
    )
    svc._window_n = 110

    # Two archive entries, one with explicit hash, one compat (tokens only).
    h_explicit = compute_rollout_hash([10, 20, 30]).hex()
    archives = [
        {
            "window_start": 100,
            "batch": [
                {
                    "prompt_idx": 7,
                    "rollouts": [
                        {"tokens": [10, 20, 30], "hash": h_explicit},
                        {"tokens": [40, 50, 60]},  # compat: no hash key
                    ],
                }
            ],
        }
    ]
    with patch(
        "reliquary.infrastructure.storage.list_recent_datasets",
        new=AsyncMock(return_value=archives),
    ):
        await svc._rebuild_hashes_from_history()

    assert bytes.fromhex(h_explicit) in svc._hash_set
    assert compute_rollout_hash([40, 50, 60]) in svc._hash_set


def test_record_late_drop_aggregates_per_hotkey_and_reason():
    """record_late_drop bumps counters keyed by (hotkey, reason), starting
    from empty, isolating different hotkeys."""
    svc = _build_late_drop_service()
    assert svc._late_drops == {}
    svc.record_late_drop("hkA", "window_not_active")
    svc.record_late_drop("hkA", "window_not_active")
    svc.record_late_drop("hkA", "worker_dropped")
    svc.record_late_drop("hkB", "worker_dropped")
    assert svc._late_drops == {
        "hkA": {"window_not_active": 2, "worker_dropped": 1},
        "hkB": {"worker_dropped": 1},
    }


def test_service_registers_late_drop_callback_on_server():
    """ValidationService wires record_late_drop into ValidatorServer at init."""
    svc = _build_late_drop_service()
    # Invoking via the server's stored callback must bump the service counter.
    svc.server._late_drop_callback("hkX", "window_not_active")
    assert svc._late_drops == {"hkX": {"window_not_active": 1}}
