"""Bittensor chain interactions for Reliquary."""

import asyncio
import logging
import os

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


async def miner_alpha_per_block(subtensor, netuid: int = NETUID) -> dict:
    """The subnet's alpha emitted to its miners per block: its alpha out,
    less the owner's cut, halved between miners and validators."""
    info = await asyncio.wait_for(subtensor.subnet(netuid), timeout=CHAIN_READ_TIMEOUT)
    raw_cut = await asyncio.wait_for(subtensor.query_subtensor("SubnetOwnerCut"),
                                     timeout=CHAIN_READ_TIMEOUT)
    cut = int(getattr(raw_cut, "value", raw_cut)) / 65535
    alpha_out = float(info.alpha_out_emission)
    return {"alpha_out_per_block": alpha_out, "owner_cut": cut,
            "miner_alpha_per_block": alpha_out * (1.0 - cut) / 2.0, "netuid": int(netuid)}


async def subnet_pool(subtensor, netuid: int = NETUID) -> dict:
    """The subnet pool's alpha price in TAO (TAO reserve over alpha reserve)
    and the miners' alpha per block (``miner_alpha_per_block``), in one read."""
    info = await asyncio.wait_for(subtensor.subnet(netuid), timeout=CHAIN_READ_TIMEOUT)
    raw_cut = await asyncio.wait_for(subtensor.query_subtensor("SubnetOwnerCut"),
                                     timeout=CHAIN_READ_TIMEOUT)
    cut = int(getattr(raw_cut, "value", raw_cut)) / 65535
    alpha_out = float(info.alpha_out_emission)
    tao_in, alpha_in = float(info.tao_in), float(info.alpha_in)
    return {"alpha_out_per_block": alpha_out, "owner_cut": cut,
            "miner_alpha_per_block": alpha_out * (1.0 - cut) / 2.0, "netuid": int(netuid),
            "tao_in": tao_in, "alpha_in": alpha_in,
            "alpha_price_tao": tao_in / alpha_in if alpha_in > 0 else None}


async def get_metagraph(subtensor, netuid: int = NETUID):
    """Get subnet metagraph."""
    return await asyncio.wait_for(subtensor.metagraph(netuid), timeout=CHAIN_READ_TIMEOUT)


async def get_neurons_lite(subtensor, netuid: int = NETUID):
    """Fetch the minimal per-neuron chain records for a subnet.

    Registration admission only needs each neuron's hotkey and coldkey.  The
    full metagraph additionally fetches mechanism and emission metadata, which
    can make an otherwise healthy registration refresh exceed its RPC budget.
    """
    return await asyncio.wait_for(
        subtensor.neurons_lite(netuid), timeout=CHAIN_READ_TIMEOUT,
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


