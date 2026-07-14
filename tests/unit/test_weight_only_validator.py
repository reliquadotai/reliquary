"""WeightOnlyValidator main loop: read archives, replay EMA, submit on chain tempo."""

from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reliquary.constants import B_BATCH


class _FakeWallet:
    class _Hk:
        ss58_address = "5FReader"
    hotkey = _Hk()


def _archive(window_start, batch, rewards_by_hotkey=None):
    """Build a minimal archive record.

    ``batch`` is the list of hotkeys present in the training batch (for
    documentation only; the EMA replay no longer consumes the ``batch``
    field). ``rewards_by_hotkey`` is the per-hotkey emission share — the
    field ``_replay_ema`` reads. When unspecified, it defaults to the
    pre-v2.3 behaviour ``count(hk) / B_BATCH`` so legacy fixtures stay
    semantically equivalent.
    """
    if rewards_by_hotkey is None:
        rewards = defaultdict(float)
        for hk in batch:
            rewards[hk] += 1.0 / B_BATCH
        rewards_by_hotkey = dict(rewards)
    return {
        "window_start": window_start,
        "batch": [{"hotkey": hk, "prompt_idx": 0} for hk in batch],
        "rewards_by_hotkey": rewards_by_hotkey,
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
        )
    finally:
        wov_mod.chain.get_metagraph = original_get_meta
        wov_mod.chain.set_weights = original_set_weights

    from reliquary.constants import UID_BURN
    assert 10 in captured["uids"]   # alice → uid 10
    assert 20 in captured["uids"]   # bob → uid 20
    assert UID_BURN in captured["uids"]
    assert sum(captured["weights"]) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_deregistered_ema_mass_is_conserved_as_burn():
    """Metagraph filtering must not redistribute vanished miner weight."""
    from reliquary.constants import UID_BURN
    from reliquary.validator.weight_only import WeightOnlyValidator

    wov = WeightOnlyValidator(wallet=_FakeWallet(), netuid=81)
    fake_meta = MagicMock(hotkeys=["alice"], uids=[10])
    captured = {}

    async def capture(_subtensor, _wallet, _netuid, uids, weights):
        captured.update(zip(uids, weights))
        return True

    with (
        patch(
            "reliquary.validator.weight_only.chain.get_metagraph",
            new=AsyncMock(return_value=fake_meta),
        ),
        patch(
            "reliquary.validator.weight_only.chain.set_weights",
            new=capture,
        ),
    ):
        submitted = await wov._submit_weights(
            MagicMock(), {"alice": 0.4, "deregistered": 0.3},
        )

    assert submitted is True
    assert captured[10] == pytest.approx(0.4)
    assert captured[UID_BURN] == pytest.approx(0.6)
    assert sum(captured.values()) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_uid_burn_earned_weight_is_aggregated_without_duplicate_uid():
    from reliquary.constants import UID_BURN
    from reliquary.validator.weight_only import WeightOnlyValidator

    wov = WeightOnlyValidator(wallet=_FakeWallet(), netuid=81)
    fake_meta = MagicMock(hotkeys=["owner", "alice"], uids=[UID_BURN, 10])
    captured = {}

    async def capture(_subtensor, _wallet, _netuid, uids, weights):
        captured["uids"] = uids
        captured["weights"] = weights
        return True

    with (
        patch(
            "reliquary.validator.weight_only.chain.get_metagraph",
            new=AsyncMock(return_value=fake_meta),
        ),
        patch(
            "reliquary.validator.weight_only.chain.set_weights",
            new=capture,
        ),
    ):
        await wov._submit_weights(
            MagicMock(), {"owner": 0.1, "alice": 0.4},
        )

    assert captured["uids"].count(UID_BURN) == 1
    by_uid = dict(zip(captured["uids"], captured["weights"]))
    assert by_uid[UID_BURN] == pytest.approx(0.6)
    assert sum(captured["weights"]) == pytest.approx(1.0)


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
    captured["chain_mocks"] = chain_mocks
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


async def _wire_submit_counter(wov, captured, *, result=True):
    async def _fake_submit(subtensor, miner_weights):
        captured["submit_calls"] += 1
        return result
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
async def test_failed_attempt_does_not_hammer_same_epoch():
    """A rejected set_weights call has already gone through bittensor's
    internal retries. Do not retry it every poll inside the same epoch.
    """
    from reliquary.validator.weight_only import WeightOnlyValidator
    wov = WeightOnlyValidator(wallet=_FakeWallet(), netuid=81)
    originals, captured = _patch_chain_and_storage(blocks_until=200)
    await _wire_submit_counter(wov, captured, result=False)
    try:
        await _run_one_iteration(wov)
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
async def test_epoch_position_is_pinned_to_current_block_snapshot():
    """The two values used for the epoch id must come from the same block.

    If the chain advances between independent reads, ``current_block +
    blocks_until`` can be off by one and trigger a duplicate set_weights call
    at the boundary.
    """
    from reliquary.validator.weight_only import WeightOnlyValidator

    wov = WeightOnlyValidator(wallet=_FakeWallet(), netuid=81)
    originals, captured = _patch_chain_and_storage(
        blocks_until=5, current_block=1_000_000,
    )
    await _wire_submit_counter(wov, captured)
    try:
        await _run_one_iteration(wov)
        chain_mocks = captured["chain_mocks"]
        calls = chain_mocks["blocks_until_next_epoch"].await_args_list
        assert calls
        assert all(call.kwargs == {"block": 1_000_000} for call in calls)
        assert all(
            call.args == (chain_mocks["get_subtensor"].return_value, 81)
            for call in calls
        )
    finally:
        _restore(originals)


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

    async def fake_blocks_until(sub, netuid, *, block=None):
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


# ---------------------------------------------------------------------------
# _replay_ema uses rewards_by_hotkey (post-alignment)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_replay_ema_reads_rewards_by_hotkey_field():
    """The on-chain EMA must derive from ``rewards_by_hotkey``, not from
    counting ``batch`` entries. Without this, the K-way same-prompt split
    and the boundary fair-split exist only in the archives and never
    reach the chain — sybils that lose the canonical-rep race silently
    forfeit their economically-earned share."""
    from reliquary.validator.weight_only import WeightOnlyValidator
    from reliquary.constants import EMA_ALPHA

    # Hand-craft rewards that DIFFER from what counting batch entries
    # would give: alice has 0 batch entries but a nonzero reward share
    # (her prompt won at the boundary; she didn't get the canonical pick).
    archive = {
        "window_start": 1,
        "batch": [{"hotkey": "carol", "prompt_idx": 0}],
        "rewards_by_hotkey": {"alice": 0.05, "bob": 0.05, "carol": 0.0625},
    }
    ema = WeightOnlyValidator._replay_ema([archive])
    # Each fraction × EMA_ALPHA on a fresh EMA:
    assert abs(ema["alice"] - EMA_ALPHA * 0.05) < 1e-9
    assert abs(ema["bob"] - EMA_ALPHA * 0.05) < 1e-9
    assert abs(ema["carol"] - EMA_ALPHA * 0.0625) < 1e-9


@pytest.mark.asyncio
async def test_replay_ema_k_way_split_applies_on_chain():
    """5 sybils on one winning prompt each earn ``slot_share / 5`` in
    ``rewards_by_hotkey``. The EMA must distribute the SAME total across
    all 5 hotkeys, not concentrate it on the canonical representative
    (the prior bug)."""
    from reliquary.validator.weight_only import WeightOnlyValidator
    from reliquary.constants import EMA_ALPHA, B_BATCH

    slot_share = 1.0 / B_BATCH
    archive = {
        "window_start": 1,
        # Batch only contains the canonical representative.
        "batch": [{"hotkey": "rep_canonical", "prompt_idx": 42}],
        # Economic distribution: all 5 sybils share the slot.
        "rewards_by_hotkey": {f"sybil{i}": slot_share / 5 for i in range(5)},
    }
    ema = WeightOnlyValidator._replay_ema([archive])
    # All 5 sybils get the same EMA contribution.
    expected = EMA_ALPHA * slot_share / 5
    for i in range(5):
        assert abs(ema[f"sybil{i}"] - expected) < 1e-9
    # The canonical rep — present in batch but NOT in rewards — gets nothing.
    assert "rep_canonical" not in ema
    # Total across the 5 sybils equals what 1 honest hotkey alone would
    # have earned on that prompt — sybil-neutral on-chain.
    assert abs(sum(ema.values()) - EMA_ALPHA * slot_share) < 1e-9


@pytest.mark.asyncio
async def test_replay_ema_boundary_fair_split_applies_on_chain():
    """Boundary-round miners that didn't make the training batch still
    earn on-chain EMA contribution, exactly as ``select_batch_and_distribute``
    computed for them. Pins the alignment between the seal-time logic
    and the weight-setter."""
    from reliquary.validator.weight_only import WeightOnlyValidator
    from reliquary.constants import EMA_ALPHA, B_BATCH

    slot_share = 1.0 / B_BATCH
    # Boundary scenario: 6 slots filled by round-1 prompts at full
    # slot_share each, plus a boundary round with 4 prompts sharing 2
    # remaining slots = per_prompt = 2 × slot_share / 4 = slot_share / 2.
    rewards = {f"r1_p{i}": slot_share for i in range(6)}
    for i in range(4):
        rewards[f"boundary_{i}"] = slot_share / 2

    # Only 2 of the 4 boundary prompts (canonical-hash winners) reach the
    # batch — the other 2 wouldn't appear at all under the old EMA.
    archive = {
        "window_start": 1,
        "batch": (
            [{"hotkey": f"r1_p{i}", "prompt_idx": i} for i in range(6)]
            + [{"hotkey": f"boundary_{i}", "prompt_idx": 100 + i} for i in range(2)]
        ),
        "rewards_by_hotkey": rewards,
    }
    ema = WeightOnlyValidator._replay_ema([archive])
    # Every miner — including the 2 non-trained boundary ones — earned EMA.
    for i in range(6):
        assert abs(ema[f"r1_p{i}"] - EMA_ALPHA * slot_share) < 1e-9
    for i in range(4):
        assert abs(ema[f"boundary_{i}"] - EMA_ALPHA * slot_share / 2) < 1e-9
    # Conservation: total EMA after one window = EMA_ALPHA × pool.
    assert abs(sum(ema.values()) - EMA_ALPHA * 1.0) < 1e-9


@pytest.mark.asyncio
async def test_replay_ema_empty_rewards_decays_existing_ema():
    """An archive with empty ``rewards_by_hotkey`` (no slots filled) must
    NOT zero out previously-earned EMA — it must apply the ``(1 - α)``
    decay multiplier so a single empty window doesn't wipe a miner's
    weight to zero."""
    from reliquary.validator.weight_only import WeightOnlyValidator
    from reliquary.constants import EMA_ALPHA

    archives = [
        # Window 1: alice earns half a slot.
        {"window_start": 1, "batch": [], "rewards_by_hotkey": {"alice": 0.5}},
        # Window 2: nobody earns anything.
        {"window_start": 2, "batch": [], "rewards_by_hotkey": {}},
    ]
    ema = WeightOnlyValidator._replay_ema(archives)
    # After window 1: alice = α × 0.5.
    # After window 2 (empty rewards): alice = (1-α) × (α × 0.5).
    expected = (1 - EMA_ALPHA) * EMA_ALPHA * 0.5
    assert abs(ema["alice"] - expected) < 1e-9


@pytest.mark.asyncio
async def test_replay_ema_conservation_bound():
    """Sum of EMA values after one window equals
    ``EMA_ALPHA × sum(rewards_by_hotkey)``. Pinning this invariant catches
    any future regression that would let the EMA either double-spend the
    pool or silently drop rewards."""
    from reliquary.validator.weight_only import WeightOnlyValidator
    from reliquary.constants import EMA_ALPHA

    rewards = {f"hk{i}": 0.1 for i in range(8)}  # sums to 0.8 (3 slots burned)
    archive = {
        "window_start": 1,
        "batch": [{"hotkey": f"hk{i}", "prompt_idx": i} for i in range(8)],
        "rewards_by_hotkey": rewards,
    }
    ema = WeightOnlyValidator._replay_ema([archive])
    expected_total = EMA_ALPHA * sum(rewards.values())
    assert abs(sum(ema.values()) - expected_total) < 1e-9
