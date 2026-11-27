"""The unpaid miner share, routed in part to a collector hotkey instead of the
burn. The hotkey and the tier schedule are code constants (``PAYOUT_HOTKEY``,
``PAYOUT_TIERS``), never the environment: a tier takes effect at the epoch its
activation block falls in, for every validator holding the release. Unset,
the vector is today's, float for float."""

from __future__ import annotations

import random
import struct
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# The public development key "Alice": an SS58 account with a valid checksum.
COLLECTOR = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
OWNER, OWNER_COLD = "owner", "owner-cold"

# The schedule a release ships. A tier pull request changes it here on purpose, with the constants.
EXPECTED_PAYOUT_HOTKEY = None
EXPECTED_PAYOUT_TIERS = ()


class _FakeWallet:
    class _Hk:
        ss58_address = "5FReader"
    hotkey = _Hk()


def _meta(*, collector_uid=77, collector_cold="collector-cold", owner_uid=42, owner_fields=True, block=None):
    hotkeys = ["alice", "5FReader"]
    uids = [10, 237]
    coldkeys = ["alice-cold", "reader-cold"]
    if owner_uid is not None:
        hotkeys.append(OWNER)
        uids.append(owner_uid)
        coldkeys.append(OWNER_COLD)
    if collector_uid is not None:
        hotkeys.append(COLLECTOR)
        uids.append(collector_uid)
        coldkeys.append(collector_cold)
    meta = SimpleNamespace(hotkeys=hotkeys, uids=uids, coldkeys=coldkeys)
    if owner_fields:
        meta.owner_hotkey, meta.owner_coldkey = OWNER, OWNER_COLD
    if block is not None:
        meta.block = block
    return meta


async def _vector(miner_weights, *, hotkey, tiers, epoch=1_000, meta=None, uid_burn=None):
    """The (uid -> weight) vector one weight-set submits for the epoch ending at ``epoch``."""
    import reliquary.constants as C
    from reliquary.validator.weight_only import WeightOnlyValidator

    captured: dict[int, float] = {}

    async def capture(_subtensor, _wallet, _netuid, uids, weights):
        captured.update(zip(uids, weights))
        return True

    with ExitStack() as stack:
        stack.enter_context(patch.object(C, "UID_BURN", uid_burn))
        stack.enter_context(patch.object(C, "PAYOUT_HOTKEY", hotkey))
        stack.enter_context(patch.object(C, "PAYOUT_TIERS", tiers))
        stack.enter_context(patch("reliquary.validator.weight_only.chain.get_metagraph",
                                  new=AsyncMock(return_value=meta or _meta())))
        stack.enter_context(patch("reliquary.validator.weight_only.chain.set_weights", new=capture))
        wov = WeightOnlyValidator(wallet=_FakeWallet(), netuid=81)
        wov._active_submit_epoch = epoch
        assert await wov._submit_weights(MagicMock(), dict(miner_weights)) is True
    return captured


@pytest.mark.parametrize(("fraction", "expected"), [
    (0.0, {10: 0.4, 42: 0.6}),
    (0.01, {10: 0.4, 77: 0.006, 42: 0.594}),
    (1.0, {10: 0.4, 77: 0.6}),
])
async def test_the_vector_at_fraction_0_one_percent_and_all(fraction, expected):
    vector = await _vector({"alice": 0.4, "deregistered": 0.1}, hotkey=COLLECTOR, tiers=((1_000, fraction),))
    assert set(vector) == set(expected)
    for uid, weight in expected.items():
        assert vector[uid] == pytest.approx(weight, abs=1e-15)
    assert sum(vector.values()) == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize(("epoch", "collector"), [(999, None), (1_000, 0.006), (1_999, 0.006), (2_000, 0.06),
                                                  (9_999, 0.06)])
async def test_a_tier_takes_effect_at_the_epoch_its_activation_block_falls_in(epoch, collector):
    tiers = ((1_000, 0.01), (2_000, 0.1))
    vector = await _vector({"alice": 0.4}, hotkey=COLLECTOR, tiers=tiers, epoch=epoch)
    assert vector.get(77) == (None if collector is None else pytest.approx(collector, abs=1e-15))
    assert sum(vector.values()) == pytest.approx(1.0, abs=1e-15)


async def test_without_an_epoch_id_the_metagraph_block_picks_the_tier():
    import reliquary.constants as C
    from reliquary.validator.weight_only import WeightOnlyValidator

    assert WeightOnlyValidator._metagraph_block(_meta(block=1_500)) == 1_500
    assert WeightOnlyValidator._metagraph_block(_meta()) is None
    before = await _vector({"alice": 0.4}, hotkey=COLLECTOR, tiers=((1_000, 0.5),), epoch=None,
                           meta=_meta(block=999))
    after = await _vector({"alice": 0.4}, hotkey=COLLECTOR, tiers=((1_000, 0.5),), epoch=None,
                          meta=_meta(block=1_000))
    assert 77 not in before and after[77] == pytest.approx(0.3)
    assert C.payout_fraction(((1_000, 0.5),), None) == 0.0


async def test_unset_is_todays_vector_float_for_float():
    miner = {"alice": 0.123456789, "deregistered": 0.3}
    unset = await _vector(miner, hotkey=None, tiers=())
    named_before_its_tier = await _vector(miner, hotkey=COLLECTOR, tiers=((5_000, 1.0),), epoch=4_999)
    named_at_zero = await _vector(miner, hotkey=COLLECTOR, tiers=((1_000, 0.0),))
    # Today's rule: the owner's UID takes 1 - the registered total.
    assert unset == named_before_its_tier == named_at_zero == {10: 0.123456789, 42: 1.0 - 0.123456789}


def _todays_vector(miner_weights, meta):
    """A frozen copy of the vector rule before collectors existed."""
    hotkey_to_uid = dict(zip(meta.hotkeys, meta.uids))
    weights_by_uid: dict[int, float] = {}
    registered_total = 0.0
    for hk, w in miner_weights.items():
        if w <= 0 or hk not in hotkey_to_uid:
            continue
        uid = int(hotkey_to_uid[hk])
        weights_by_uid[uid] = weights_by_uid.get(uid, 0.0) + w
        registered_total += w
    burn_weight = max(0.0, 1.0 - registered_total)
    if burn_weight > 0:
        burn_uid = int(hotkey_to_uid[meta.owner_hotkey])
        weights_by_uid[burn_uid] = weights_by_uid.get(burn_uid, 0.0) + burn_weight
    return weights_by_uid


async def test_with_no_tier_in_effect_the_vector_is_the_old_rules_bit_for_bit():
    rng = random.Random(81)
    for case in range(300):
        hotkeys = [f"hk{i}" for i in range(rng.randint(1, 12))]
        miner = {hk: rng.random() / len(hotkeys) for hk in hotkeys + ["gone"]}
        if case % 3 == 0:
            miner[OWNER] = rng.random() / 20         # the owner's UID also holds miner weight
        if case % 5 == 0:
            miner[COLLECTOR] = rng.random() / 20     # the collector hotkey also holds miner weight
        uids = rng.sample(range(256), len(hotkeys) + 2)
        meta = SimpleNamespace(hotkeys=hotkeys + [OWNER, COLLECTOR], uids=uids,
                               coldkeys=[f"c{i}" for i in range(len(hotkeys))] + [OWNER_COLD, "collector-cold"],
                               owner_hotkey=OWNER, owner_coldkey=OWNER_COLD)
        expected = _todays_vector(miner, meta)
        for hotkey, tiers, epoch in ((None, (), 1_000), (COLLECTOR, ((1_000, 0.0),), 1_000),
                                     (COLLECTOR, ((2_000, 1.0),), 1_999)):
            got = await _vector(miner, hotkey=hotkey, tiers=tiers, epoch=epoch, meta=meta)
            assert list(got) == list(expected)
            assert [struct.pack("<d", got[u]) for u in got] == [struct.pack("<d", expected[u]) for u in expected]


async def test_a_collector_without_a_uid_lets_its_part_burn(caplog):
    vector = await _vector({"alice": 0.4}, hotkey=COLLECTOR, tiers=((1_000, 0.5),), meta=_meta(collector_uid=None))
    assert vector == {10: 0.4, 42: pytest.approx(0.6)}
    assert "has no UID" in caplog.text


async def test_a_collector_held_by_the_owners_coldkey_lets_its_part_burn(caplog):
    vector = await _vector({"alice": 0.4}, hotkey=COLLECTOR, tiers=((1_000, 0.5),),
                           meta=_meta(collector_cold=OWNER_COLD))
    assert vector == {10: 0.4, 42: pytest.approx(0.6)}
    assert "subnet owner" in caplog.text


async def test_the_owner_hotkey_itself_is_never_the_collector():
    import reliquary.constants as C
    from reliquary.validator.weight_only import WeightOnlyValidator

    with patch.object(C, "PAYOUT_HOTKEY", OWNER), patch.object(C, "PAYOUT_TIERS", ((1_000, 1.0),)):
        assert WeightOnlyValidator._resolve_payout(_meta(), {OWNER: 42}, 1_000) == (None, 0.0)


async def test_a_snapshot_without_the_subnet_owner_fails_closed_as_without_a_collector(caplog):
    # At the whole share, a collector is not paid on a snapshot that cannot show it is not the owner's: the share
    # takes the burn path, which refuses to guess, exactly as without a collector.
    with pytest.raises(RuntimeError, match="subnet owner hotkey unavailable"):
        await _vector({"alice": 0.4}, hotkey=COLLECTOR, tiers=((1_000, 1.0),), meta=_meta(owner_fields=False))
    assert "does not name the subnet owner" in caplog.text


async def test_with_the_whole_share_routed_the_owner_needs_no_uid():
    vector = await _vector({"alice": 0.4}, hotkey=COLLECTOR, tiers=((1_000, 1.0),), meta=_meta(owner_uid=None))
    assert vector == {10: 0.4, 77: pytest.approx(0.6)}


async def test_an_explicit_burn_uid_still_takes_the_rest():
    vector = await _vector({"alice": 0.4}, hotkey=COLLECTOR, tiers=((1_000, 0.25),), uid_burn=0)
    assert vector == {10: 0.4, 77: pytest.approx(0.15), 0: pytest.approx(0.45)}


async def test_nothing_unpaid_routes_nothing():
    vector = await _vector({"alice": 1.0}, hotkey=COLLECTOR, tiers=((1_000, 1.0),))
    assert vector == {10: 1.0}


def test_the_hotkey_must_be_an_ss58_account_with_a_valid_checksum():
    from reliquary.constants import is_ss58_account

    assert is_ss58_account(COLLECTOR)
    assert is_ss58_account("5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty")
    for bad in (COLLECTOR[:-1] + ("R" if COLLECTOR[-1] != "R" else "S"), "5" * 46, COLLECTOR + "1",
                "not-an-address", "", None, b"5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "1" * 48):
        assert not is_ss58_account(bad), bad


def test_the_settings_refuse_what_is_not_a_hotkey_and_a_schedule():
    from reliquary.constants import payout_settings

    assert payout_settings(None, ()) == (None, ())
    assert payout_settings(COLLECTOR, ((10, 0.01), (20, 1))) == (COLLECTOR, ((10, 0.01), (20, 1.0)))
    assert payout_settings(None, ((10, 0.0),)) == (None, ((10, 0.0),))
    for hotkey, tiers in ((None, ((10, 0.5),)), (COLLECTOR, ((10, 1.5),)), (COLLECTOR, ((10, -0.1),)),
                          (COLLECTOR, ((10, float("nan")),)), (COLLECTOR, ((10, "0.1"),)), (COLLECTOR, ((10, True),)),
                          (COLLECTOR, ((True, 0.1),)), (COLLECTOR, ((-1, 0.1),)), (COLLECTOR, ((10.0, 0.1),)),
                          (COLLECTOR, ((20, 0.1), (10, 0.5))), (COLLECTOR, ((10, 0.1), (10, 0.5))),
                          (COLLECTOR, [(10, 0.1)]), (COLLECTOR, ((10,),)), ("5" * 46, ((10, 0.1),))):
        with pytest.raises(ValueError):
            payout_settings(hotkey, tiers)


def test_the_shipped_schedule_is_the_one_this_release_means():
    import reliquary.constants as C

    assert (C.PAYOUT_HOTKEY, C.PAYOUT_TIERS) == (EXPECTED_PAYOUT_HOTKEY, EXPECTED_PAYOUT_TIERS)


def test_no_environment_changes_the_routing():
    import os
    import subprocess
    import sys

    import reliquary.constants as C

    # A validator's environment cannot make it route differently from the others, whatever the schedule.
    env = {**os.environ, "RELIQUARY_PAYOUT_HOTKEY": COLLECTOR, "RELIQUARY_PAYOUT_FRACTION": "1",
           "RELIQUARY_PAYOUT_TIERS": "((0, 1.0),)"}
    out = subprocess.run([sys.executable, "-c", "import reliquary.constants as C; "
                          "print(repr((C.PAYOUT_HOTKEY, C.PAYOUT_TIERS)))"],
                         capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip().splitlines()[-1] == repr((C.PAYOUT_HOTKEY, C.PAYOUT_TIERS))
