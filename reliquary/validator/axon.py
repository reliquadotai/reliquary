"""Publishing a validator's HTTP address on the metagraph (``serve_axon``), so miners find it with
``miner.submitter.discover_validator`` instead of ``--validator-url``. Used by the window validator and the async RL
validator alike."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def serve_axon_on_chain(subtensor, *, wallet, netuid: int, ip: str | None, port: int | None,
                              signer_client=None) -> bool:
    """Announce ``ip:port`` for the wallet's hotkey on ``netuid`` (through the remote signer when there is one).
    Never raises: a failure is logged and the miners need ``--validator-url``. True when published."""
    if not ip or not port:
        logger.warning("serve_axon skipped: no external_ip/external_port provided. Miners won't discover this "
                       "validator via metagraph; use --validator-url on the miner side.")
        return False
    if signer_client is not None:
        try:
            accepted = await signer_client.serve_axon(netuid=netuid, ip=ip, port=port)
        except Exception:
            logger.exception("remote signer serve_axon operation failed")
            return False
        if not accepted:
            logger.error("remote signer rejected serve_axon for %s:%d on netuid %d", ip, port, netuid)
            return False
        logger.info("remote signer published axon: %s:%d on netuid %d", ip, port, netuid)
        return True
    try:
        import bittensor as bt

        axon = bt.Axon(wallet=wallet, ip=ip, port=port, external_ip=ip, external_port=port)
        response = await subtensor.serve_axon(netuid=netuid, axon=axon, wait_for_inclusion=True,
                                              wait_for_finalization=False, raise_error=False)
    except Exception:
        logger.exception("serve_axon threw — miners will have to use --validator-url")
        return False
    if getattr(response, "is_success", None) is False:
        logger.error("serve_axon failed: %s:%d not published (response=%s). Likely: hotkey not registered on "
                     "netuid %d, or chain rejected.", ip, port, response, netuid)
        return False
    logger.info("serve_axon published: %s:%d announced on netuid %d", ip, port, netuid)
    return True
