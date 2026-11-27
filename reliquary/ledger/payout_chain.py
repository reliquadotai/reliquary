"""The payer's chain, through the SDK: reads at given blocks, and the one write.

``SubtensorPayoutChain`` reads the collector's stake and nonce, a hotkey's
owner, block times and a block's extrinsics and events (an archive node, for
blocks days back), and sends a signed batch without waiting: the payer reads
what became of it from the chain (``payout.resolve_batch``).

``WalletSigner`` signs, with the collector's coldkey, at the nonce and the
mortal era it is given: one ``Utility.batch_all`` of
``SubtensorModule.transfer_stake`` calls (a payout), or one
``SubtensorModule.burn_alpha`` (a surplus burn). Only ``--execute`` builds one.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence

from reliquary.ledger.payout import Inclusion, SignedBatch, Transfer

READ_TIMEOUT_SECONDS = 120.0
BLOCK_SECONDS = 12.0


def _hex(value) -> str:
    if isinstance(value, (bytes, bytearray)):
        return "0x" + bytes(value).hex()
    text = str(value)
    return text.lower() if text.startswith("0x") else "0x" + text.lower()


def ss58(value, ss58_format: int = 42):
    """An account as SS58, from a decoded value: SS58 already, or its public key."""
    from scalecodec.utils.ss58 import ss58_encode

    if isinstance(value, (bytes, bytearray)) and len(value) == 32:
        return ss58_encode(bytes(value), ss58_format=ss58_format)
    if isinstance(value, str) and value.startswith("0x") and len(value) == 66:
        return ss58_encode(value, ss58_format=ss58_format)
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return ss58(value[0], ss58_format)
    if isinstance(value, (list, tuple)) and len(value) == 32 and all(isinstance(b, int) for b in value):
        return ss58_encode(bytes(value), ss58_format=ss58_format)
    return value


def normalize_event(event: Mapping) -> dict:
    """``{"module", "event", "attributes"}`` of one decoded event, accounts as SS58."""
    data = event.get("event") if isinstance(event.get("event"), Mapping) else event
    attributes = data.get("attributes")
    if isinstance(attributes, Mapping):
        attributes = {k: ss58(v) for k, v in attributes.items()}
    elif isinstance(attributes, (list, tuple)):
        attributes = [ss58(v) for v in attributes]
    return {"module": data.get("module_id"), "event": data.get("event_id"), "attributes": attributes}


class SubtensorPayoutChain:
    def __init__(self, subtensor, *, netuid: int, collector_coldkey: str, collector_hotkey: str) -> None:
        self.subtensor, self.netuid = subtensor, int(netuid)
        self.collector_coldkey, self.collector_hotkey = collector_coldkey, collector_hotkey
        self._times: dict[int, float] = {}

    async def _read(self, coro):
        return await asyncio.wait_for(coro, timeout=READ_TIMEOUT_SECONDS)

    async def finalized_block(self) -> int:
        substrate = self.subtensor.substrate
        return int(await self._read(substrate.get_block_number(await self._read(substrate.get_chain_finalised_head()))))

    async def collector_stake_rao(self, block: int) -> int:
        stake = await self._read(self.subtensor.get_stake(self.collector_coldkey, self.collector_hotkey, self.netuid,
                                                          block=int(block)))
        return int(stake.rao)

    async def collector_nonce(self, block: int) -> int:
        substrate = self.subtensor.substrate
        block_hash = await self._read(substrate.get_block_hash(int(block)))
        account = await self._read(substrate.query("System", "Account", [self.collector_coldkey],
                                                   block_hash=block_hash))
        value = getattr(account, "value", account)
        return int(value["nonce"])

    async def hotkey_owner(self, hotkey: str, block: int) -> str | None:
        return await self._read(self.subtensor.get_hotkey_owner(hotkey, block=int(block)))

    async def _time(self, block: int) -> float:
        if block not in self._times:
            stamp = await self._read(self.subtensor.get_timestamp(block=int(block)))
            self._times[block] = float(stamp.timestamp())
        return self._times[block]

    async def block_at(self, at: float) -> int:
        """The last block whose time is at or before ``at``."""
        head = await self.finalized_block()
        if await self._time(head) <= at:
            return head
        guess = head - int((await self._time(head) - float(at)) / BLOCK_SECONDS)
        low, high = max(1, guess - 300), min(head, guess + 300)
        while low > 1 and await self._time(low) > at:
            low = max(1, low - 3000)
        while high < head and await self._time(high) <= at:
            high = min(head, high + 3000)
        while low < high:
            middle = (low + high + 1) // 2
            if await self._time(middle) <= at:
                low = middle
            else:
                high = middle - 1
        return low

    async def find_extrinsic(self, extrinsic_hash: str, first_block: int, last_block: int) -> Inclusion | None:
        from async_substrate_interface.async_substrate import AsyncExtrinsicReceipt

        substrate = self.subtensor.substrate
        want = _hex(extrinsic_hash)
        for number in range(int(first_block), int(last_block) + 1):
            block = await self._read(substrate.get_block(block_number=number))
            if not block:
                continue
            for index, extrinsic in enumerate(block.get("extrinsics") or []):
                got = getattr(extrinsic, "extrinsic_hash", None)
                if got is None or _hex(got) != want:
                    continue
                block_hash = block["header"]["hash"]
                receipt = AsyncExtrinsicReceipt(substrate, extrinsic_hash=want, block_hash=block_hash,
                                                block_number=number, extrinsic_idx=index, finalized=True)
                success = bool(await receipt.is_success)
                events = tuple(normalize_event(e) for e in await receipt.triggered_events)
                fee = await receipt.total_fee_amount
                error = None if success else str(await receipt.error_message)
                return Inclusion(block_number=number, block_hash=str(block_hash), success=success, events=events,
                                 fee_rao=None if fee is None else int(fee), error=error)
        return None

    async def submit(self, signed: SignedBatch) -> None:
        await self._read(self.subtensor.substrate.submit_extrinsic(signed.payload, wait_for_inclusion=False,
                                                                   wait_for_finalization=False))


class WalletSigner:
    """Signs with the collector's coldkey (a bittensor wallet, unlocked by its
    owner when the key is first used)."""

    def __init__(self, subtensor, wallet, *, netuid: int, collector_hotkey: str) -> None:
        self.subtensor, self.wallet = subtensor, wallet
        self.netuid, self.collector_hotkey = int(netuid), collector_hotkey
        self.coldkey = str(wallet.coldkeypub.ss58_address)

    async def sign(self, transfers: Sequence[Transfer], *, nonce: int, era_block: int,
                   era_period: int) -> SignedBatch:
        calls = [await self.subtensor.compose_call("SubtensorModule", "transfer_stake", {
            "destination_coldkey": t.coldkey, "hotkey": self.collector_hotkey, "origin_netuid": self.netuid,
            "destination_netuid": self.netuid, "alpha_amount": int(t.amount_rao)}) for t in transfers]
        batch = await self.subtensor.compose_call("Utility", "batch_all", {"calls": calls})
        extrinsic = await self.subtensor.substrate.create_signed_extrinsic(
            call=batch, keypair=self.wallet.coldkey, era={"period": int(era_period), "current": int(era_block)},
            nonce=int(nonce))
        return SignedBatch(extrinsic_hash=_hex(extrinsic.extrinsic_hash), nonce=int(nonce), era_block=int(era_block),
                           era_period=int(era_period), payload=extrinsic)

    async def sign_burn(self, amount_rao: int, *, nonce: int, era_block: int, era_period: int) -> SignedBatch:
        """One ``SubtensorModule.burn_alpha`` of the collector's stake on its hotkey."""
        call = await self.subtensor.compose_call("SubtensorModule", "burn_alpha", {
            "hotkey": self.collector_hotkey, "amount": int(amount_rao), "netuid": self.netuid})
        extrinsic = await self.subtensor.substrate.create_signed_extrinsic(
            call=call, keypair=self.wallet.coldkey, era={"period": int(era_period), "current": int(era_block)},
            nonce=int(nonce))
        return SignedBatch(extrinsic_hash=_hex(extrinsic.extrinsic_hash), nonce=int(nonce), era_block=int(era_block),
                           era_period=int(era_period), payload=extrinsic)


__all__ = ["SubtensorPayoutChain", "WalletSigner", "normalize_event", "ss58"]
