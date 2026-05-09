"""Bittensor chain interactions for Reliquary."""

import asyncio
import hashlib
import logging
import os
from typing import Any

from reliquary.constants import BLOCK_TIME_SECONDS

logger = logging.getLogger(__name__)

NETUID = int(os.getenv("NETUID", "81"))
NETWORK = os.getenv("BT_NETWORK", "finney")

# Timeouts for chain calls. Without these a wedged substrate WebSocket
# (TCP keepalive failed, NAT timeout, server-side drop) silently hangs
# the await forever — no exception, no log. We surface as TimeoutError so
# the caller can decide to reconnect.
CHAIN_READ_TIMEOUT = 30.0
# bittensor's set_weights retries up to 5x across consecutive blocks
# (~60s of retries), so allow a longer budget before declaring it hung.
CHAIN_WRITE_TIMEOUT = 180.0


async def get_subtensor():
    """Create async subtensor."""
    import bittensor as bt

    subtensor = bt.AsyncSubtensor(network=NETWORK)
    await asyncio.wait_for(subtensor.initialize(), timeout=120.0)
    return subtensor


async def get_metagraph(subtensor, netuid: int = NETUID):
    """Get subnet metagraph."""
    return await asyncio.wait_for(subtensor.metagraph(netuid), timeout=CHAIN_READ_TIMEOUT)


async def get_block_hash(subtensor, block_number: int) -> str:
    """Get block hash for a given block number."""
    return await asyncio.wait_for(
        subtensor.get_block_hash(block_number), timeout=CHAIN_READ_TIMEOUT,
    )


async def get_current_block(subtensor) -> int:
    """Get current block number."""
    return await asyncio.wait_for(subtensor.get_current_block(), timeout=CHAIN_READ_TIMEOUT)


async def set_weights(
    subtensor,
    wallet,
    netuid: int,
    uids: list[int],
    weights: list[float],
) -> bool:
    """Submit weights on-chain.

    Returns True only when the chain reports success. A rejected extrinsic
    (no validator permit, rate limit, bad weights, etc.) does NOT raise — it
    comes back as ``ExtrinsicResponse(success=False, message=...)`` — so we
    inspect the response explicitly instead of trusting "no exception ==
    success".

    bittensor's own ``set_weights`` already retries transient failures up to
    ``max_attempts=5`` times across consecutive blocks, so external retry
    wrapping here would be redundant. Wrapped in a write-timeout so a
    wedged WebSocket can't hang the validator loop indefinitely; the
    caller is expected to reconnect on TimeoutError.
    """
    try:
        response = await asyncio.wait_for(
            subtensor.set_weights(
                wallet=wallet,
                netuid=netuid,
                uids=uids,
                weights=weights,
            ),
            timeout=CHAIN_WRITE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error(
            "set_weights timed out after %.0fs — substrate WebSocket likely stalled",
            CHAIN_WRITE_TIMEOUT,
        )
        raise
    except Exception as e:
        logger.error("set_weights raised: %s", e, exc_info=True)
        return False

    success = bool(getattr(response, "success", False))
    if success:
        logger.info("set_weights OK — %d uids committed", len(uids))
        return True
    msg = getattr(response, "message", None) or "<no message>"
    err = getattr(response, "error", None)
    logger.error(
        "set_weights rejected by chain: %s (error=%r)", msg, err,
    )
    return False


def compute_drand_round_for_window(
    window_start_block: int, genesis_time: int, period: int
) -> int:
    """Deterministically compute which drand round to use for a window.

    Both miner and validator call this with the same inputs to agree
    on a single round — no "latest" fetch needed.

    Returns:
        The drand round number (>= 1).
    """
    window_timestamp = window_start_block * BLOCK_TIME_SECONDS
    if window_timestamp < genesis_time:
        return 1  # Clamp: can't go before round 1
    return 1 + (window_timestamp - genesis_time) // period


def compute_window_randomness(
    block_hash: str,
    drand_randomness: str | None = None,
    drand_round: int | None = None,
) -> str:
    """Combine block hash, drand randomness, and round into window randomness.

    Including the round number prevents a miner from choosing a round
    whose randomness is favorable.
    """
    clean_hash = block_hash.replace("0x", "")
    if drand_randomness:
        material = bytes.fromhex(clean_hash) + bytes.fromhex(drand_randomness)
        if drand_round is not None:
            material += drand_round.to_bytes(8, "big")
        combined = hashlib.sha256(material).hexdigest()
        return combined
    return clean_hash
