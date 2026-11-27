"""The payer's chain adapter refuses what it cannot trust: a block number the
node has no hash for, a storage read that decodes as the empty default (what a
node's JSON-RPC error becomes), a block that cannot be fetched, a zero time;
and an extrinsic's outcome is read from its events, never assumed."""

from __future__ import annotations

import asyncio
import datetime as dt
from types import SimpleNamespace

import pytest

from reliquary.ledger import payout as po
from reliquary.ledger import payout_chain as pc

COLLECTOR, COLLECTOR_HOT = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"


class _Substrate:
    def __init__(self, *, hashes=None, account=None, owner=COLLECTOR, owner_hotkey=None, blocks=None, events=()):
        self.hashes = {} if hashes is None else hashes
        self.account = {"nonce": 7, "consumers": 0, "providers": 1, "sufficients": 0, "data": {}} \
            if account is None else account
        self.owner, self.owner_hotkey, self.blocks, self.events = owner, owner_hotkey, blocks or {}, events
        self.queries = []

    async def get_block_hash(self, number):
        return self.hashes.get(number, f"0x{number:064x}")

    async def get_chain_finalised_head(self):
        return "0xhead"

    async def get_block_number(self, block_hash):
        return 100

    async def query(self, module, storage, params, block_hash=None):
        self.queries.append((module, storage, block_hash))
        if storage == "Account":
            return SimpleNamespace(value=self.account)
        if storage == "SubnetOwner":
            return SimpleNamespace(value=self.owner)
        if storage == "SubnetOwnerHotkey":
            return SimpleNamespace(value=self.owner_hotkey or pc.ZERO_ACCOUNT)
        raise AssertionError(storage)

    async def get_block(self, block_number=None):
        return self.blocks.get(block_number)


class _Subtensor:
    def __init__(self, substrate, *, moment=1_795_000_000.0):
        self.substrate, self.moment, self.calls = substrate, moment, []

    async def get_stake(self, coldkey, hotkey, netuid, block_hash=None):
        self.calls.append(("get_stake", block_hash))
        return SimpleNamespace(rao=5)

    async def get_hotkey_owner(self, hotkey, block_hash=None):
        self.calls.append(("get_hotkey_owner", block_hash))
        return COLLECTOR

    async def get_timestamp(self, block_hash=None):
        return dt.datetime.fromtimestamp(self.moment, tz=dt.timezone.utc)


def _chain(substrate, **kwargs):
    return pc.SubtensorPayoutChain(_Subtensor(substrate, **kwargs), netuid=81, collector_coldkey=COLLECTOR,
                                   collector_hotkey=COLLECTOR_HOT)


def test_every_read_is_made_at_the_blocks_hash():
    substrate = _Substrate()
    chain = _chain(substrate)
    assert asyncio.run(chain.collector_nonce(42)) == 7
    assert asyncio.run(chain.collector_stake_rao(42)) == 5 and asyncio.run(chain.hotkey_owner(COLLECTOR_HOT, 42))
    assert substrate.queries == [("System", "Account", f"0x{42:064x}")]
    assert chain.subtensor.calls == [("get_stake", f"0x{42:064x}"), ("get_hotkey_owner", f"0x{42:064x}")]


def test_a_block_the_node_has_no_hash_for_is_refused_not_read_at_its_best_state():
    chain = _chain(_Substrate(hashes={42: None}))
    for read in (chain.collector_nonce(42), chain.collector_stake_rao(42), chain.hotkey_owner(COLLECTOR_HOT, 42),
                 chain.subnet_owner(42)):
        with pytest.raises(po.ChainReadError):
            asyncio.run(read)


@pytest.mark.parametrize("account", [
    {"nonce": 0, "consumers": 0, "providers": 0, "sufficients": 0, "data": {}},   # the empty default
    {"providers": 1},
    "not an account",
])
def test_an_account_read_that_decodes_as_the_empty_default_is_refused(account):
    with pytest.raises(po.ChainReadError):
        asyncio.run(_chain(_Substrate(account=account)).collector_nonce(1))


def test_the_subnet_owner_must_read_set_and_its_hotkey_may_be_unset():
    assert asyncio.run(_chain(_Substrate(owner=COLLECTOR)).subnet_owner(1)) == (COLLECTOR, None)
    assert asyncio.run(_chain(_Substrate(owner=COLLECTOR, owner_hotkey=COLLECTOR_HOT)).subnet_owner(1)) == (
        COLLECTOR, COLLECTOR_HOT)
    with pytest.raises(po.ChainReadError):
        asyncio.run(_chain(_Substrate(owner=pc.ZERO_ACCOUNT)).subnet_owner(1))


def test_a_zero_block_time_is_refused():
    with pytest.raises(po.ChainReadError):
        asyncio.run(_chain(_Substrate(), moment=0.0)._time(5))


def test_an_extrinsic_is_never_declared_absent_from_a_block_not_fetched():
    chain = _chain(_Substrate(blocks={10: {"extrinsics": [], "header": {"hash": "0x10"}}}))
    assert asyncio.run(chain.find_extrinsic("0xab", 10, 10)) is None
    with pytest.raises(po.ChainReadError):
        asyncio.run(chain.find_extrinsic("0xab", 10, 11))


@pytest.mark.parametrize("events, success", [
    ([{"event": {"module_id": "System", "event_id": "ExtrinsicSuccess", "attributes": {}}}], True),
    ([{"event": {"module_id": "System", "event_id": "ExtrinsicFailed", "attributes": {}}}], False),
    ([], None),                                       # what an events read that errored decodes to
])
def test_an_extrinsics_outcome_is_read_from_its_events(monkeypatch, events, success):
    import async_substrate_interface.async_substrate as asi

    class Receipt:
        def __init__(self, substrate, **kwargs):
            pass

        @property
        async def triggered_events(self):
            return events

        @property
        async def total_fee_amount(self):
            return 9

        @property
        async def error_message(self):
            return {"name": "AmountTooLow"}

    monkeypatch.setattr(asi, "AsyncExtrinsicReceipt", Receipt)
    block = {"extrinsics": [SimpleNamespace(extrinsic_hash=bytes.fromhex("ab"))], "header": {"hash": "0x10"}}
    found = asyncio.run(_chain(_Substrate(blocks={10: block})).find_extrinsic("0xAB", 10, 10))
    assert found.block_number == 10 and found.success is success
    assert found.fee_rao == (None if success is None else 9)
    assert (found.error is not None) is (success is False)


def test_find_nonce_names_what_the_collectors_key_signed():
    other = SimpleNamespace(value={"address": COLLECTOR_HOT, "nonce": 3}, extrinsic_hash=b"\x01")
    mine = SimpleNamespace(value={"address": COLLECTOR, "nonce": 3,
                                  "call": {"call_module": "SubtensorModule", "call_function": "add_stake"}},
                           extrinsic_hash=b"\x02")
    blocks = {10: {"extrinsics": [other], "header": {"hash": "0x10"}},
              11: {"extrinsics": [other, mine], "header": {"hash": "0x11"}}}
    found = asyncio.run(_chain(_Substrate(blocks=blocks)).find_nonce(3, 10, 11))
    assert found == {"block_number": 11, "block_hash": "0x11", "index": 1, "extrinsic_hash": "0x02",
                     "call": "SubtensorModule.add_stake"}
    assert asyncio.run(_chain(_Substrate(blocks=blocks)).find_nonce(4, 10, 11)) is None


def test_the_signer_composes_one_batch_all_of_transfer_stake_at_the_nonce_and_era_given():
    composed = []

    class Subtensor:
        def __init__(self):
            self.substrate = self

        async def compose_call(self, module, function, params):
            composed.append((module, function, params))
            return (module, function)

        async def create_signed_extrinsic(self, *, call, keypair, era, nonce):
            composed.append(("signed", call, era, nonce))
            return SimpleNamespace(extrinsic_hash=bytes.fromhex("cd"))

    wallet = SimpleNamespace(coldkeypub=SimpleNamespace(ss58_address=COLLECTOR), coldkey="key")
    signer = pc.WalletSigner(Subtensor(), wallet, netuid=81, collector_hotkey=COLLECTOR_HOT)
    transfers = [po.Transfer(coldkey="5A", amount_rao=10, obligation_ids=("x",)),
                 po.Transfer(coldkey="5B", amount_rao=20, obligation_ids=("y",))]
    signed = asyncio.run(signer.sign(transfers, nonce=4, era_block=1000, era_period=64))
    assert signed.extrinsic_hash == "0xcd" and (signed.nonce, signed.era_block, signed.era_period) == (4, 1000, 64)
    assert composed[:2] == [
        ("SubtensorModule", "transfer_stake", {"destination_coldkey": "5A", "hotkey": COLLECTOR_HOT,
                                               "origin_netuid": 81, "destination_netuid": 81, "alpha_amount": 10}),
        ("SubtensorModule", "transfer_stake", {"destination_coldkey": "5B", "hotkey": COLLECTOR_HOT,
                                               "origin_netuid": 81, "destination_netuid": 81, "alpha_amount": 20})]
    assert composed[2][:2] == ("Utility", "batch_all") and len(composed[2][2]["calls"]) == 2
    assert composed[3] == ("signed", ("Utility", "batch_all"), {"period": 64, "current": 1000}, 4)
