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


async def close_subtensor(subtensor) -> None:
    """Best-effort close of an ``AsyncSubtensor``.

    Long-lived subtensors on the public Finney endpoint accumulate WebSocket
    state and eventually wedge — every RPC call into them hangs past our
    timeouts. The recovery path is to throw the connection away and open a
    fresh one. This helper makes sure the old WebSocket + its background
    recv task get released, otherwise they linger in the event loop and
    have been observed to degrade subsequently-opened replacements too.

    Errors are swallowed: the common case is closing a connection that's
    already broken, and we don't want close failures to prevent the caller
    from opening the replacement.
    """
    if subtensor is None:
        return
    try:
        await asyncio.wait_for(subtensor.close(), timeout=5.0)
    except Exception as e:
        logger.warning("close_subtensor swallowed error: %s", e)


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


async def blocks_until_next_epoch(
    subtensor,
    netuid: int,
    *,
    block: int | None = None,
) -> int | None:
    """Blocks remaining in the current epoch for ``netuid``.

    All validators of the same netuid see the same boundary because the
    underlying SDK formula is purely a function of (netuid, current_block,
    tempo). Passing ``block`` pins the calculation to the same chain snapshot
    used to derive the epoch identifier, avoiding a boundary race between two
    independent head reads.
    """
    kwargs = {"block": block} if block is not None else {}
    return await asyncio.wait_for(
        subtensor.blocks_until_next_epoch(netuid, **kwargs),
        timeout=CHAIN_READ_TIMEOUT,
    )


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
    block_hash: str | None,
    drand_randomness: str | None = None,
    drand_round: int | None = None,
) -> str:
    """Combine drand randomness (+ optional block hash) into a window seed.

    Including the round number prevents a miner from choosing a round
    whose randomness is favorable.

    As of v2.3 ``block_hash`` may be None — the drand-only path. Drand
    quicknet provides threshold-BLS unpredictability that doesn't require
    a chain-anchored mix to be secure. Dropping the block_hash decouples
    window randomness from substrate availability, so a flaky WebSocket
    no longer stalls window OPEN.
    """
    if drand_randomness:
        material = b""
        if block_hash is not None:
            material += bytes.fromhex(block_hash.replace("0x", ""))
        material += bytes.fromhex(drand_randomness)
        if drand_round is not None:
            material += drand_round.to_bytes(8, "big")
        return hashlib.sha256(material).hexdigest()
    if block_hash is None:
        raise ValueError(
            "compute_window_randomness requires either block_hash or drand_randomness"
        )
    return block_hash.replace("0x", "")


def compute_current_drand_round(
    timestamp_seconds: float, genesis_time: int, period: int,
) -> int:
    """Drand round currently in progress at the given UNIX timestamp.

    Validator calls this at submit-receipt time with `time.time()` to
    decide whether the round attached by the miner equals the round
    publishing at receipt — the v2.3 zero-tolerance check.
    """
    if timestamp_seconds < genesis_time:
        return 1
    return 1 + int((timestamp_seconds - genesis_time) // period)


def seconds_until_next_drand_boundary(
    timestamp_seconds: float, genesis_time: int, period: int,
) -> float:
    """Seconds remaining until the next drand round publishes.

    Returns 0.0 if ``timestamp_seconds`` is exactly at a round boundary.
    Otherwise returns ``period - (timestamp_seconds - genesis_time) % period``.
    Used by the validator to align window OPEN to a drand round boundary,
    so the randomness round about to publish is not pre-fetchable by
    miners (which is what enables the v30 pre-spam exploit).
    """
    if timestamp_seconds < genesis_time:
        return float(genesis_time - timestamp_seconds)
    elapsed = timestamp_seconds - genesis_time
    rem = elapsed % period
    if rem == 0:
        return 0.0
    return period - rem
