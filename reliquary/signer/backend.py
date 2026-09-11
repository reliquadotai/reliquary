"""Wallet-owning backend for the narrow signer service."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ChainResult:
    accepted: bool
    message: str


async def weights_submission_wait_blocks(
    subtensor, netuid: int, hotkey: str, *, block: int | None = None,
) -> int:
    """Mirror the SDK's strict rate-limit gate before reserving a signer epoch."""
    if block is None:
        block = await asyncio.wait_for(
            subtensor.get_current_block(), timeout=30.0,
        )
    uid = await asyncio.wait_for(
        subtensor.get_uid_for_hotkey_on_subnet(hotkey, netuid, block=block),
        timeout=30.0,
    )
    if uid is None:
        raise RuntimeError("weight signer hotkey is not registered")
    elapsed, limit = await asyncio.wait_for(
        asyncio.gather(
            subtensor.blocks_since_last_update(netuid, uid, block=block),
            subtensor.weights_rate_limit(netuid, block=block),
        ),
        timeout=30.0,
    )
    if elapsed is None or limit is None:
        raise RuntimeError("weight rate-limit state is unavailable")
    return max(0, int(limit) + 1 - int(elapsed))


class SignerBackend(Protocol):
    hotkey_address: str

    def sign_checkpoint(self, payload: bytes) -> bytes: ...

    async def set_weights(
        self, *, netuid: int, uids: list[int], weights: list[float]
    ) -> ChainResult: ...

    async def serve_axon(self, *, netuid: int, ip: str, port: int) -> ChainResult: ...


class BittensorSignerBackend:
    """Loads one hotkey and constructs all chain calls inside signer-01."""

    def __init__(
        self,
        *,
        wallet_name: str,
        hotkey_name: str,
        wallet_path: str,
        network: str,
        expected_hotkey: str,
        chain_timeout_seconds: float = 240.0,
    ) -> None:
        import bittensor as bt

        self._bt = bt
        self.wallet = bt.Wallet(
            name=wallet_name,
            hotkey=hotkey_name,
            path=wallet_path,
        )
        self.hotkey_address = str(self.wallet.hotkey.ss58_address)
        if self.hotkey_address != expected_hotkey:
            raise RuntimeError(
                "loaded signer hotkey does not match RELIQUARY_SIGNER_EXPECTED_HOTKEY"
            )
        self.network = network
        self.chain_timeout_seconds = float(chain_timeout_seconds)

    def sign_checkpoint(self, payload: bytes) -> bytes:
        signature = bytes(self.wallet.hotkey.sign(payload))
        verifier = self._bt.Keypair(ss58_address=self.hotkey_address)
        if not verifier.verify(data=payload, signature=signature):
            raise RuntimeError(
                "signer produced a checkpoint signature it cannot verify"
            )
        return signature

    async def _subtensor(self):
        subtensor = self._bt.AsyncSubtensor(network=self.network)
        await asyncio.wait_for(subtensor.initialize(), timeout=120.0)
        return subtensor

    @staticmethod
    async def _close(subtensor) -> None:
        if subtensor is None:
            return
        try:
            await asyncio.wait_for(subtensor.close(), timeout=5.0)
        except Exception:
            pass

    async def set_weights(
        self, *, netuid: int, uids: list[int], weights: list[float]
    ) -> ChainResult:
        subtensor = await self._subtensor()
        try:
            wait_blocks = await weights_submission_wait_blocks(
                subtensor, netuid, self.hotkey_address,
            )
            if wait_blocks:
                return ChainResult(
                    accepted=False,
                    message=f"weights_rate_limited: retry_after_blocks={wait_blocks}",
                )
            response = await asyncio.wait_for(
                subtensor.set_weights(
                    wallet=self.wallet,
                    netuid=int(netuid),
                    uids=list(uids),
                    weights=list(weights),
                ),
                timeout=self.chain_timeout_seconds,
            )
            accepted = bool(getattr(response, "success", False))
            message = str(
                getattr(response, "message", None)
                or ("accepted" if accepted else "SDK rejected weights without a reason")
            )[:500]
            return ChainResult(accepted=accepted, message=message)
        finally:
            await self._close(subtensor)

    async def serve_axon(self, *, netuid: int, ip: str, port: int) -> ChainResult:
        subtensor = await self._subtensor()
        try:
            axon = self._bt.Axon(
                wallet=self.wallet,
                ip=ip,
                port=int(port),
                external_ip=ip,
                external_port=int(port),
            )
            response = await asyncio.wait_for(
                subtensor.serve_axon(
                    netuid=int(netuid),
                    axon=axon,
                    wait_for_inclusion=True,
                    wait_for_finalization=False,
                    raise_error=False,
                ),
                timeout=self.chain_timeout_seconds,
            )
            accepted_value = getattr(response, "success", None)
            if accepted_value is None:
                accepted_value = getattr(response, "is_success", False)
            message = str(getattr(response, "message", None) or "")[:500]
            return ChainResult(accepted=bool(accepted_value), message=message)
        finally:
            await self._close(subtensor)
