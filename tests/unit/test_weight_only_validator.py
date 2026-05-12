"""WeightOnlyValidator main loop: read archives, replay EMA, submit on chain tempo."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class _FakeWallet:
    class _Hk:
        ss58_address = "5FReader"
    hotkey = _Hk()


def _archive(window_start, batch):
    return {
        "window_start": window_start,
        "batch": [{"hotkey": hk, "prompt_idx": 0} for hk in batch],
    }


@pytest.mark.asyncio
async def test_replay_ema_deterministic():
    from reliquary.validator.weight_only import WeightOnlyValidator
    archives = [
        _archive(1, ["alice", "bob"]),
        _archive(2, ["alice"]),
        _archive(3, ["bob", "carol"]),
    ]
    ema = WeightOnlyValidator._replay_ema(archives)
    assert "alice" in ema
    assert "bob" in ema
    assert "carol" in ema
    # alice and bob each contributed across more windows than carol
    assert ema["alice"] > ema["carol"]
    assert ema["bob"] > ema["carol"]


@pytest.mark.asyncio
async def test_replay_ema_is_order_invariant_given_sorted_input():
    from reliquary.validator.weight_only import WeightOnlyValidator
    archives = [
        _archive(2, ["alice"]),
        _archive(1, ["bob"]),
        _archive(3, ["alice", "bob"]),
    ]
    ema_a = WeightOnlyValidator._replay_ema(archives)
    ema_b = WeightOnlyValidator._replay_ema(list(reversed(archives)))
    # Both inputs sort to [1,2,3] internally → identical output
    assert ema_a == ema_b


@pytest.mark.asyncio
async def test_replay_ema_empty_archives():
    from reliquary.validator.weight_only import WeightOnlyValidator
    ema = WeightOnlyValidator._replay_ema([])
    assert ema == {}


@pytest.mark.asyncio
async def test_submit_weights_maps_hotkeys_to_uids():
    """_submit_weights calls chain.set_weights with correct uid mapping."""
    from reliquary.validator.weight_only import WeightOnlyValidator

    wov = WeightOnlyValidator(wallet=_FakeWallet(), netuid=81)

    fake_meta = MagicMock()
    fake_meta.hotkeys = ["alice", "bob", "carol"]
    fake_meta.uids = [10, 20, 30]

    import reliquary.validator.weight_only as wov_mod
    original_get_meta = wov_mod.chain.get_metagraph
    original_set_weights = wov_mod.chain.set_weights
    wov_mod.chain.get_metagraph = AsyncMock(return_value=fake_meta)
    captured = {}

    async def _fake_set_weights(subtensor, wallet, netuid, uids, weights):
        captured["uids"] = uids
        captured["weights"] = weights
        return True

    wov_mod.chain.set_weights = _fake_set_weights

    try:
        await wov._submit_weights(
            MagicMock(),
            {"alice": 0.5, "bob": 0.3},
            burn_weight=0.2,
        )
    finally:
        wov_mod.chain.get_metagraph = original_get_meta
        wov_mod.chain.set_weights = original_set_weights

    from reliquary.constants import UID_BURN
    assert 10 in captured["uids"]   # alice → uid 10
    assert 20 in captured["uids"]   # bob → uid 20
    assert UID_BURN in captured["uids"]


async def _run_one_iteration(wov):
    """Helper: run wov.run() until it completes one poll, then cancel."""
    import asyncio
    task = asyncio.create_task(wov.run())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def _patch_chain_and_storage(blocks_until: int, current_block: int = 1_000_000):
    """Patch the chain/storage entry points weight_only.run() touches.
    Returns (originals, captured_calls)."""
    from unittest.mock import AsyncMock, MagicMock
    import reliquary.validator.weight_only as wov_mod

    captured = {"submit_calls": 0}
    chain_mocks = {
        "get_subtensor": AsyncMock(return_value=MagicMock()),
        "close_subtensor": AsyncMock(),
        "blocks_until_next_epoch": AsyncMock(return_value=blocks_until),
        "get_current_block": AsyncMock(return_value=current_block),
    }
    storage_mocks = {
        "list_all_window_keys": AsyncMock(return_value=[1, 2, 3]),
        "list_recent_datasets": AsyncMock(return_value=[
            _archive(1, ["alice"]),
            _archive(2, ["alice"]),
            _archive(3, ["bob"]),
        ]),
    }
    originals = {
        "chain": {k: getattr(wov_mod.chain, k) for k in chain_mocks},
        "storage": {k: getattr(wov_mod.storage, k) for k in storage_mocks},
    }
    for k, v in chain_mocks.items():
        setattr(wov_mod.chain, k, v)
    for k, v in storage_mocks.items():
        setattr(wov_mod.storage, k, v)
    return originals, captured


def _restore(originals):
    import reliquary.validator.weight_only as wov_mod
    for k, v in originals["chain"].items():
        setattr(wov_mod.chain, k, v)
    for k, v in originals["storage"].items():
        setattr(wov_mod.storage, k, v)


async def _wire_submit_counter(wov, captured):
    async def _fake_submit(subtensor, miner_weights, burn_weight):
        captured["submit_calls"] += 1
        return True
    wov._submit_weights = _fake_submit


@pytest.mark.asyncio
async def test_bootstrap_submits_immediately_regardless_of_lead_window():
    """A freshly-booted validator submits on its first poll even if we're
    far from the epoch boundary."""
    from reliquary.validator.weight_only import WeightOnlyValidator
    wov = WeightOnlyValidator(wallet=_FakeWallet(), netuid=81)
    assert wov._last_submit_epoch is None
    # Far from boundary (200 blocks remain → outside any reasonable lead).
    originals, captured = _patch_chain_and_storage(blocks_until=200)
    await _wire_submit_counter(wov, captured)
    try:
        await _run_one_iteration(wov)
    finally:
        _restore(originals)
    assert captured["submit_calls"] == 1
    assert wov._last_submit_epoch == 1_000_200


@pytest.mark.asyncio
async def test_in_lead_window_submits():
    """Inside [tempo - EPOCH_SUBMIT_LEAD_BLOCKS, tempo - 1] → submit."""
    from reliquary.validator.weight_only import WeightOnlyValidator
    from reliquary.constants import EPOCH_SUBMIT_LEAD_BLOCKS
    wov = WeightOnlyValidator(wallet=_FakeWallet(), netuid=81)
    wov._last_submit_epoch = 999_000  # earlier epoch — not the current one
    originals, captured = _patch_chain_and_storage(
        blocks_until=EPOCH_SUBMIT_LEAD_BLOCKS,
    )
    await _wire_submit_counter(wov, captured)
    try:
        await _run_one_iteration(wov)
    finally:
        _restore(originals)
    assert captured["submit_calls"] == 1


@pytest.mark.asyncio
async def test_outside_lead_window_skips():
    """Far from boundary AND already submitted before → no submit."""
    from reliquary.validator.weight_only import WeightOnlyValidator
    from reliquary.constants import EPOCH_SUBMIT_LEAD_BLOCKS
    wov = WeightOnlyValidator(wallet=_FakeWallet(), netuid=81)
    wov._last_submit_epoch = 999_000  # earlier epoch
    originals, captured = _patch_chain_and_storage(
        blocks_until=EPOCH_SUBMIT_LEAD_BLOCKS + 50,  # well outside
    )
    await _wire_submit_counter(wov, captured)
    try:
        await _run_one_iteration(wov)
    finally:
        _restore(originals)
    assert captured["submit_calls"] == 0


@pytest.mark.asyncio
async def test_repeat_poll_in_same_epoch_skips():
    """Once submitted in epoch E, a second poll in E does not re-submit."""
    from reliquary.validator.weight_only import WeightOnlyValidator
    wov = WeightOnlyValidator(wallet=_FakeWallet(), netuid=81)
    wov._last_submit_epoch = 1_000_005  # = current_block + blocks_until
    originals, captured = _patch_chain_and_storage(
        blocks_until=5, current_block=1_000_000,
    )
    await _wire_submit_counter(wov, captured)
    try:
        await _run_one_iteration(wov)
    finally:
        _restore(originals)
    assert captured["submit_calls"] == 0


@pytest.mark.asyncio
async def test_next_epoch_submits_again():
    """After crossing the boundary, _last_submit_epoch != current_epoch_id
    and we're inside the lead window → submit fires again."""
    from reliquary.validator.weight_only import WeightOnlyValidator
    from reliquary.constants import EPOCH_SUBMIT_LEAD_BLOCKS
    wov = WeightOnlyValidator(wallet=_FakeWallet(), netuid=81)
    wov._last_submit_epoch = 1_000_000  # the prior epoch end
    originals, captured = _patch_chain_and_storage(
        blocks_until=EPOCH_SUBMIT_LEAD_BLOCKS,
        current_block=1_000_001,  # one block past the prior boundary
    )
    await _wire_submit_counter(wov, captured)
    try:
        await _run_one_iteration(wov)
    finally:
        _restore(originals)
    assert captured["submit_calls"] == 1
    assert wov._last_submit_epoch == 1_000_001 + EPOCH_SUBMIT_LEAD_BLOCKS


@pytest.mark.asyncio
async def test_blocks_until_timeout_recycles_subtensor():
    """On a chain-call TimeoutError, the run loop must close the wedged
    polling subtensor and pull a fresh one on the next iteration. The
    previous reconnect path leaked the old subtensor, whose background
    WebSocket task degraded the replacement and froze the loop for hours.
    """
    import asyncio as _asyncio
    from unittest.mock import AsyncMock, MagicMock
    from reliquary.validator.weight_only import WeightOnlyValidator
    import reliquary.validator.weight_only as wov_mod

    old_sub = MagicMock(name="old_sub")
    new_sub = MagicMock(name="new_sub")
    get_calls: list = []

    async def fake_get_subtensor():
        sub = old_sub if not get_calls else new_sub
        get_calls.append(sub)
        return sub

    close_calls: list = []

    async def fake_close(sub):
        close_calls.append(sub)

    bun_calls = {"n": 0}

    async def fake_blocks_until(sub, netuid):
        bun_calls["n"] += 1
        if bun_calls["n"] == 1:
            raise _asyncio.TimeoutError
        return 200

    originals = {
        "get_subtensor": wov_mod.chain.get_subtensor,
        "close_subtensor": wov_mod.chain.close_subtensor,
        "blocks_until_next_epoch": wov_mod.chain.blocks_until_next_epoch,
        "get_current_block": wov_mod.chain.get_current_block,
    }
    poll_orig = wov_mod.POLL_INTERVAL_SECONDS
    wov_mod.chain.get_subtensor = fake_get_subtensor
    wov_mod.chain.close_subtensor = fake_close
    wov_mod.chain.blocks_until_next_epoch = fake_blocks_until
    wov_mod.chain.get_current_block = AsyncMock(return_value=1_000_000)
    wov_mod.POLL_INTERVAL_SECONDS = 0  # don't wait between iterations in test

    wov = WeightOnlyValidator(wallet=_FakeWallet(), netuid=81)
    wov._submit_weights = AsyncMock(return_value=True)
    wov_mod.storage.list_all_window_keys = AsyncMock(return_value=[1])
    wov_mod.storage.list_recent_datasets = AsyncMock(return_value=[])

    try:
        task = _asyncio.create_task(wov.run())
        await _asyncio.sleep(0.1)  # let it iterate past the timeout + reconnect
        task.cancel()
        try:
            await task
        except _asyncio.CancelledError:
            pass
    finally:
        for k, v in originals.items():
            setattr(wov_mod.chain, k, v)
        wov_mod.POLL_INTERVAL_SECONDS = poll_orig

    # Initial connect + post-timeout reconnect = at least 2 get_subtensor calls.
    assert len(get_calls) >= 2
    # The wedged subtensor was closed before reconnect.
    assert old_sub in close_calls
