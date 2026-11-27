"""The payer's chain, through the SDK: reads at given blocks, and the one write.

``SubtensorPayoutChain`` reads the collector's stake and nonce, a hotkey's
owner, the subnet's owner, block times and a block's extrinsics and events (an
archive node, for blocks days back), and sends a signed batch without waiting:
the payer reads what became of it from the chain (``payout.resolve_batch``).

Every read is made at an explicit block hash, and a value that cannot be right
raises ``ChainReadError`` instead of being returned: the SDK's storage reads
do not check a node's JSON-RPC errors, and decode one as the storage's empty
default (a nonce of 0, no events, a time of 0). A block number with no hash on
the node answering, a block that cannot be fetched, an account with no
provider and a zero time are all refused.

``WalletSigner`` signs one ``Utility.batch_all`` of
``SubtensorModule.transfer_stake`` calls with the collector's coldkey, at the
nonce and the mortal era it is given. Only ``reliquary ledger pay --execute``
builds one.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence

from reliquary.ledger.payout import ChainReadError, Inclusion, PoolRejected, SignedBatch, Transfer

READ_TIMEOUT_SECONDS = 120.0
BLOCK_SECONDS = 12.0
# A submission the node refused into its pool, by the error message author_submitExtrinsic answers with
# (Substrate's author RPC: invalid, unknown validity, priority too low for a transaction of the same nonce,
# immediately dropped). None of them is imported or gossiped. Every other error leaves the fate to the chain.
POOL_REFUSALS = ("Invalid Transaction", "Unknown Transaction Validity", "Priority is too low", "Immediately Dropped")
# The account every unset AccountId storage decodes to.
ZERO_ACCOUNT = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"


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


def outcome(events: Sequence[Mapping]) -> bool | None:
    """True with a ``System.ExtrinsicSuccess`` event, False with an
    ``ExtrinsicFailed`` one, None with neither (or both): unreadable."""
    names = {e.get("event") for e in events if e.get("module") == "System"}
    success, failed = "ExtrinsicSuccess" in names, "ExtrinsicFailed" in names
    if success == failed:
        return None
    return success


def _account(value) -> str | None:
    """A decoded AccountId as SS58; None when unset (the zero account)."""
    value = getattr(value, "value", value)
    value = ss58(value)
    if not isinstance(value, str) or not value or value == ZERO_ACCOUNT:
        return None
    return value


class SubtensorPayoutChain:
    def __init__(self, subtensor, *, netuid: int, collector_coldkey: str, collector_hotkey: str) -> None:
        self.subtensor, self.netuid = subtensor, int(netuid)
        self.collector_coldkey, self.collector_hotkey = collector_coldkey, collector_hotkey
        self._times: dict[int, float] = {}

    async def _read(self, coro):
        return await asyncio.wait_for(coro, timeout=READ_TIMEOUT_SECONDS)

    async def _hash(self, block: int) -> str:
        """The hash of block ``block``; refused when the node has none (it
        would read its own best state instead)."""
        value = await self._read(self.subtensor.substrate.get_block_hash(int(block)))
        if not value:
            raise ChainReadError(f"the node has no hash for block {block}")
        return str(value)

    async def finalized_block(self) -> int:
        substrate = self.subtensor.substrate
        head = await self._read(substrate.get_chain_finalised_head())
        if not head:
            raise ChainReadError("the node gave no finalized head")
        number = await self._read(substrate.get_block_number(head))
        if number is None or int(number) <= 0:
            raise ChainReadError(f"the finalized head {head} has no block number")
        return int(number)

    async def collector_stake_rao(self, block: int) -> int:
        stake = await self._read(self.subtensor.get_stake(self.collector_coldkey, self.collector_hotkey, self.netuid,
                                                          block_hash=await self._hash(block)))
        return int(stake.rao)

    async def collector_nonce(self, block: int) -> int:
        account = await self._read(self.subtensor.substrate.query("System", "Account", [self.collector_coldkey],
                                                                  block_hash=await self._hash(block)))
        value = getattr(account, "value", account)
        nonce = value.get("nonce") if isinstance(value, Mapping) else None
        providers = value.get("providers") if isinstance(value, Mapping) else None
        if isinstance(nonce, bool) or not isinstance(nonce, int) or nonce < 0 or not isinstance(providers, int) \
                or providers < 1:
            # The empty default (nonce 0, no provider) is what a node's error decodes to; a collector that pays
            # holds TAO for its fees, so its account exists.
            raise ChainReadError(f"the collector's account reads {value!r} at block {block}: not an existing account")
        return nonce

    async def hotkey_owner(self, hotkey: str, block: int) -> str | None:
        return await self._read(self.subtensor.get_hotkey_owner(hotkey, block_hash=await self._hash(block)))

    async def subnet_owner(self, block: int) -> tuple[str, str | None]:
        """(the subnet owner's coldkey, its owner hotkey or None when unset)."""
        substrate = self.subtensor.substrate
        block_hash = await self._hash(block)
        coldkey = _account(await self._read(substrate.query("SubtensorModule", "SubnetOwner", [self.netuid],
                                                            block_hash=block_hash)))
        hotkey = _account(await self._read(substrate.query("SubtensorModule", "SubnetOwnerHotkey", [self.netuid],
                                                           block_hash=block_hash)))
        if coldkey is None:
            raise ChainReadError(f"subnet {self.netuid}'s owner reads unset at block {block}")
        return coldkey, hotkey

    async def _time(self, block: int) -> float:
        if block not in self._times:
            stamp = await self._read(self.subtensor.get_timestamp(block_hash=await self._hash(block)))
            moment = float(stamp.timestamp())
            if moment <= 0:
                raise ChainReadError(f"block {block}'s time reads {moment}")
            self._times[block] = moment
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

    async def _block(self, number: int) -> Mapping:
        block = await self._read(self.subtensor.substrate.get_block(block_number=int(number)))
        if not block:
            raise ChainReadError(f"block {number} could not be fetched")
        return block

    async def find_extrinsic(self, extrinsic_hash: str, first_block: int, last_block: int) -> Inclusion | None:
        """Where the extrinsic of this hash landed in the blocks given, with its
        events; None if none of them holds it. A block that cannot be fetched
        raises: an extrinsic is never declared absent from a block not read."""
        from async_substrate_interface.async_substrate import AsyncExtrinsicReceipt

        substrate = self.subtensor.substrate
        want = _hex(extrinsic_hash)
        for number in range(int(first_block), int(last_block) + 1):
            block = await self._block(number)
            for index, extrinsic in enumerate(block.get("extrinsics") or []):
                got = getattr(extrinsic, "extrinsic_hash", None)
                if got is None or _hex(got) != want:
                    continue
                block_hash = block["header"]["hash"]
                receipt = AsyncExtrinsicReceipt(substrate, extrinsic_hash=want, block_hash=block_hash,
                                                block_number=number, extrinsic_idx=index, finalized=True)
                events = tuple(normalize_event(e) for e in await self._read(receipt.triggered_events))
                success = outcome(events)
                fee = await self._read(receipt.total_fee_amount) if success is not None else None
                error = str(await self._read(receipt.error_message)) if success is False else None
                return Inclusion(block_number=number, block_hash=str(block_hash), success=success, events=events,
                                 fee_rao=None if fee is None else int(fee), error=error)
        return None

    async def find_nonce(self, nonce: int, first_block: int, last_block: int) -> dict | None:
        """The extrinsic signed by the collector's coldkey at ``nonce`` in the
        blocks given (read-only: what took a batch's nonce), or None."""
        for number in range(int(first_block), int(last_block) + 1):
            block = await self._block(number)
            for index, extrinsic in enumerate(block.get("extrinsics") or []):
                value = getattr(extrinsic, "value", None)
                if not isinstance(value, Mapping) or value.get("nonce") != int(nonce):
                    continue
                if ss58(value.get("address")) != self.collector_coldkey:
                    continue
                call = value.get("call") if isinstance(value.get("call"), Mapping) else {}
                return {"block_number": number, "block_hash": str(block["header"]["hash"]), "index": index,
                        "extrinsic_hash": _hex(getattr(extrinsic, "extrinsic_hash", "") or ""),
                        "call": f"{call.get('call_module')}.{call.get('call_function')}"}
        return None

    async def alpha_price_rao(self, block: int) -> int:
        """The subnet's alpha price in TAO rao per alpha, as the chain prices a
        transfer's value against its minimum (``SwapRuntimeApi.current_alpha_price``)."""
        price = await self._read(self.subtensor.get_subnet_price(self.netuid, block_hash=await self._hash(block)))
        rao = int(getattr(price, "rao", 0) or 0)
        if rao <= 0:
            raise ChainReadError(f"subnet {self.netuid}'s alpha price reads {price!r} at block {block}")
        return rao

    async def max_extrinsic_weight(self) -> tuple[int, int | None] | None:
        """The runtime's most a Normal extrinsic may weigh (``System.BlockWeights``):
        (ref_time, proof_size), or None when the constant cannot be read."""
        constant = await self._read(self.subtensor.substrate.get_constant("System", "BlockWeights"))
        value = getattr(constant, "value", constant)
        try:
            most = value["per_class"]["normal"]["max_extrinsic"]
        except (KeyError, TypeError):
            return None
        if not isinstance(most, Mapping) or not most.get("ref_time"):
            return None
        return int(most["ref_time"]), None if most.get("proof_size") is None else int(most["proof_size"])

    async def submit(self, signed: SignedBatch) -> None:
        """Send without waiting. A refusal into the node's pool raises
        ``PoolRejected``: the batch was never sent."""
        from async_substrate_interface.errors import SubstrateRequestException

        try:
            await self._read(self.subtensor.substrate.submit_extrinsic(signed.payload, wait_for_inclusion=False,
                                                                       wait_for_finalization=False))
        except SubstrateRequestException as exc:
            message = str(exc.args[0] if exc.args else exc)
            if any(message.startswith(refusal) for refusal in POOL_REFUSALS):
                raise PoolRejected(message) from exc
            raise


class WalletSigner:
    """Signs with the collector's coldkey (a bittensor wallet, unlocked by its
    owner when the key is first used)."""

    def __init__(self, subtensor, wallet, *, netuid: int, collector_hotkey: str) -> None:
        self.subtensor, self.wallet = subtensor, wallet
        self.netuid, self.collector_hotkey = int(netuid), collector_hotkey
        self.coldkey = str(wallet.coldkeypub.ss58_address)

    async def _batch(self, transfers: Sequence[Transfer]):
        calls = [await self.subtensor.compose_call("SubtensorModule", "transfer_stake", {
            "destination_coldkey": t.coldkey, "hotkey": self.collector_hotkey, "origin_netuid": self.netuid,
            "destination_netuid": self.netuid, "alpha_amount": int(t.amount_rao)}) for t in transfers]
        return await self.subtensor.compose_call("Utility", "batch_all", {"calls": calls})

    async def declared_weight(self, transfers: Sequence[Transfer], *, nonce: int, era_block: int,
                              era_period: int) -> tuple[int, int | None]:
        """The batch's declared weight, (ref_time, proof_size), from the node's
        fee query of it unsigned: the public key only, nothing is signed."""
        info = await self.subtensor.substrate.get_payment_info(
            call=await self._batch(transfers), keypair=self.wallet.coldkeypub,
            era={"period": int(era_period), "current": int(era_block)}, nonce=int(nonce))
        weight = info.get("weight") if isinstance(info, Mapping) else None
        if isinstance(weight, Mapping) and weight.get("ref_time"):
            return int(weight["ref_time"]), None if weight.get("proof_size") is None else int(weight["proof_size"])
        if isinstance(weight, int) and weight > 0:
            return weight, None
        raise ChainReadError(f"the node's fee query gave no weight for the batch: {info!r}")

    async def sign(self, transfers: Sequence[Transfer], *, nonce: int, era_block: int,
                   era_period: int) -> SignedBatch:
        batch = await self._batch(transfers)
        extrinsic = await self.subtensor.substrate.create_signed_extrinsic(
            call=batch, keypair=self.wallet.coldkey, era={"period": int(era_period), "current": int(era_block)},
            nonce=int(nonce))
        return SignedBatch(extrinsic_hash=_hex(extrinsic.extrinsic_hash), nonce=int(nonce), era_block=int(era_block),
                           era_period=int(era_period), payload=extrinsic)


__all__ = ["POOL_REFUSALS", "SubtensorPayoutChain", "WalletSigner", "ZERO_ACCOUNT", "normalize_event", "outcome",
           "ss58"]
