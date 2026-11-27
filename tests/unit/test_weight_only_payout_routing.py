"""The unpaid miner share, routed in part to a collector hotkey instead of the
burn (RELIQUARY_PAYOUT_HOTKEY, RELIQUARY_PAYOUT_FRACTION). Unset, the vector
is today's, float for float."""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# The public development key "Alice": any well-formed SS58 address will do.
COLLECTOR = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
OWNER, OWNER_COLD = "owner", "owner-cold"


class _FakeWallet:
    class _Hk:
        ss58_address = "5FReader"
    hotkey = _Hk()


def _meta(*, collector_uid=77, collector_cold="collector-cold", owner_uid=42):
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
    return SimpleNamespace(hotkeys=hotkeys, uids=uids, coldkeys=coldkeys, owner_hotkey=OWNER,
                           owner_coldkey=OWNER_COLD)


async def _vector(miner_weights, *, hotkey, fraction, meta=None, uid_burn=None):
    """The (uid -> weight) vector one weight-set submits."""
    import reliquary.constants as C
    from reliquary.validator.weight_only import WeightOnlyValidator

    captured: dict[int, float] = {}

    async def capture(_subtensor, _wallet, _netuid, uids, weights):
        captured.update(zip(uids, weights))
        return True

    with ExitStack() as stack:
        stack.enter_context(patch.object(C, "UID_BURN", uid_burn))
        stack.enter_context(patch.object(C, "PAYOUT_HOTKEY", hotkey))
        stack.enter_context(patch.object(C, "PAYOUT_FRACTION", fraction))
        stack.enter_context(patch("reliquary.validator.weight_only.chain.get_metagraph",
                                  new=AsyncMock(return_value=meta or _meta())))
        stack.enter_context(patch("reliquary.validator.weight_only.chain.set_weights", new=capture))
        wov = WeightOnlyValidator(wallet=_FakeWallet(), netuid=81)
        assert await wov._submit_weights(MagicMock(), dict(miner_weights)) is True
    return captured


@pytest.mark.parametrize(("fraction", "expected"), [
    (0.0, {10: 0.4, 42: 0.6}),
    (0.01, {10: 0.4, 77: 0.006, 42: 0.594}),
    (1.0, {10: 0.4, 77: 0.6}),
])
async def test_the_vector_at_fraction_0_one_percent_and_all(fraction, expected):
    vector = await _vector({"alice": 0.4, "deregistered": 0.1}, hotkey=COLLECTOR, fraction=fraction)
    assert set(vector) == set(expected)
    for uid, weight in expected.items():
        assert vector[uid] == pytest.approx(weight, abs=1e-15)
    assert sum(vector.values()) == pytest.approx(1.0, abs=1e-15)


async def test_unset_is_todays_vector_float_for_float():
    miner = {"alice": 0.123456789, "deregistered": 0.3}
    unset = await _vector(miner, hotkey=None, fraction=0.0)
    named_at_zero = await _vector(miner, hotkey=COLLECTOR, fraction=0.0)
    # Today's rule: the owner's UID takes 1 - the registered total.
    assert unset == named_at_zero == {10: 0.123456789, 42: 1.0 - 0.123456789}


async def test_a_collector_without_a_uid_lets_its_part_burn(caplog):
    vector = await _vector({"alice": 0.4}, hotkey=COLLECTOR, fraction=0.5, meta=_meta(collector_uid=None))
    assert vector == {10: 0.4, 42: pytest.approx(0.6)}
    assert "has no UID" in caplog.text


async def test_a_collector_held_by_the_owners_coldkey_lets_its_part_burn(caplog):
    vector = await _vector({"alice": 0.4}, hotkey=COLLECTOR, fraction=0.5,
                           meta=_meta(collector_cold=OWNER_COLD))
    assert vector == {10: 0.4, 42: pytest.approx(0.6)}
    assert "subnet owner" in caplog.text


async def test_the_owner_hotkey_itself_is_never_the_collector():
    vector = await _vector({"alice": 0.4}, hotkey=OWNER, fraction=1.0)
    assert vector == {10: 0.4, 42: pytest.approx(0.6)}


async def test_with_the_whole_share_routed_the_owner_needs_no_uid():
    vector = await _vector({"alice": 0.4}, hotkey=COLLECTOR, fraction=1.0, meta=_meta(owner_uid=None))
    assert vector == {10: 0.4, 77: pytest.approx(0.6)}


async def test_an_explicit_burn_uid_still_takes_the_rest():
    vector = await _vector({"alice": 0.4}, hotkey=COLLECTOR, fraction=0.25, uid_burn=0)
    assert vector == {10: 0.4, 77: pytest.approx(0.15), 0: pytest.approx(0.45)}


async def test_nothing_unpaid_routes_nothing():
    vector = await _vector({"alice": 1.0}, hotkey=COLLECTOR, fraction=1.0)
    assert vector == {10: 1.0}


def test_the_settings_refuse_what_is_not_a_hotkey_and_a_fraction():
    from reliquary.constants import payout_settings

    assert payout_settings(None, 0.0) == (None, 0.0)
    assert payout_settings(COLLECTOR, 0.01) == (COLLECTOR, 0.01)
    assert payout_settings(COLLECTOR, 1) == (COLLECTOR, 1.0)
    assert payout_settings(COLLECTOR, 0.0) == (COLLECTOR, 0.0)
    for hotkey, fraction in ((None, 0.5), (COLLECTOR, 1.5), (COLLECTOR, -0.1), (COLLECTOR, float("nan")),
                             (COLLECTOR, "0.1"), (COLLECTOR, True), ("not-an-address", 0.1), ("5" * 10, 0.1),
                             (b"5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", 0.1)):
        with pytest.raises(ValueError):
            payout_settings(hotkey, fraction)


def test_the_defaults_route_nothing_and_no_environment_changes_them(monkeypatch):
    import subprocess
    import sys

    import reliquary.constants as C

    assert (C.PAYOUT_HOTKEY, C.PAYOUT_FRACTION) == (None, 0.0)
    # A validator's environment cannot make it route differently from the others.
    env = {"RELIQUARY_PAYOUT_HOTKEY": COLLECTOR, "RELIQUARY_PAYOUT_FRACTION": "1"}
    out = subprocess.run([sys.executable, "-c", "import reliquary.constants as C; "
                          "print(C.PAYOUT_HOTKEY, C.PAYOUT_FRACTION)"],
                         capture_output=True, text=True, env={**__import__("os").environ, **env}, check=True)
    assert out.stdout.split()[-2:] == ["None", "0.0"]
