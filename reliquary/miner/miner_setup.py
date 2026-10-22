"""``--share-setup``: send this process's hardware to the validator at start, then every few hours.

Best effort: a refusal or an unreachable validator is logged and never stops mining.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from reliquary.protocol.miner_setup import (
    MINER_SETUP_PATH, MinerSetupRequest, build_miner_setup_binding, collect_miner_setup, load_instance_id,
)

logger = logging.getLogger(__name__)

RESEND_EVERY_S = 6 * 3600


async def send_miner_setup(client, *, validator_url: str, validator_hotkey: str, wallet, report) -> bool:
    hotkey = wallet.hotkey.ss58_address
    at = int(time.time())
    binding = build_miner_setup_binding(hotkey=hotkey, at=at, report=report, validator_hotkey=validator_hotkey)
    request = MinerSetupRequest(miner_hotkey=hotkey, at=at, report=report,
                                signature=wallet.hotkey.sign(binding).hex())
    try:
        response = await client.post(validator_url.rstrip("/") + MINER_SETUP_PATH,
                                     json=request.model_dump(mode="json"), timeout=15.0)
    except Exception as exc:
        logger.warning("miner setup not sent (%s)", type(exc).__name__)
        return False
    if response.status_code != 200:
        logger.warning("miner setup refused: %s %s", response.status_code, response.text[:200])
        return False
    return True


async def share_setup_forever(client, *, validator_url: str, validator_hotkey: str, wallet, work_dir: str,
                              note: str = "", every_s: float = RESEND_EVERY_S) -> None:
    hotkey = wallet.hotkey.ss58_address
    instance_id = load_instance_id(Path(work_dir) / "state" / "miner-setup-instance-id")
    while True:
        report = await asyncio.to_thread(collect_miner_setup, hotkey=hotkey, instance_id=instance_id, note=note)
        if await send_miner_setup(client, validator_url=validator_url, validator_hotkey=validator_hotkey,
                                  wallet=wallet, report=report):
            logger.info("miner setup shared: %d GPU(s) %s", len(report.gpus),
                        sorted({gpu.name for gpu in report.gpus}))
        await asyncio.sleep(every_s)


__all__ = ["send_miner_setup", "share_setup_forever"]
