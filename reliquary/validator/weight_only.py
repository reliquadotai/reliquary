"""Weight-only validator — reads R2 archives, submits weights on-chain.

No model, no HTTP server, no HF writes. Meant to be run alongside a
trainer validator; any number of weight-only nodes can participate and
all submit consistent weights because they read the same R2 archives.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from reliquary.constants import (
    EMA_ALPHA,
    EPOCH_SUBMIT_LEAD_BLOCKS,
    POLL_INTERVAL_SECONDS,
    UID_BURN,
)
from reliquary.infrastructure import chain, storage

# EMA history depth — number of past windows replayed to compute miner
# scores. Independent of the on-chain tempo: 72 windows ≈ ~6 hours on a
# typical cadence, enough to smooth out per-window noise.
ROLLING_WINDOWS_HISTORY = 72

logger = logging.getLogger(__name__)


class WeightOnlyValidator:
    """Lightweight validator that only sets weights.

    Each subnet epoch (anchored on subtensor.blocks_until_next_epoch):
      1. Read last K archives from R2
      2. Replay EMA update
      3. Submit weights on-chain via chain.set_weights

    All validators of a netuid hit the same epoch boundary, so they submit
    inside a shared ~EPOCH_SUBMIT_LEAD_BLOCKS-block window and converge to
    identical weights from the deterministic EMA replay.

    A freshly-booted validator submits immediately on its first poll, then
    joins the synced cadence from the next epoch onward.

    No local state: every submit recomputes from scratch.
    """

    def __init__(self, wallet, netuid: int) -> None:
        self.wallet = wallet
        self.netuid = netuid
        self._last_submit_epoch: int | None = None

    async def run(self) -> None:
        """Poll the epoch boundary and submit weights once per epoch.

        Owns its own polling ``AsyncSubtensor``; closes and replaces it on
        any chain-call timeout so a wedged WebSocket can't permanently
        stall the loop. The trainer service runs on a separate subtensor,
        so neither side can poison the other's connection state.
        """
        logger.info(
            "Weight-only validator started (netuid=%d, hotkey=%s)",
            self.netuid, self.wallet.hotkey.ss58_address,
        )
        subtensor = None
        try:
            while True:
                try:
                    if subtensor is None:
                        subtensor = await chain.get_subtensor()
                    # Read the head once and pin the epoch calculation to that
                    # block. Separate head reads can straddle a boundary and
                    # manufacture a second epoch id for the same epoch.
                    current_block = await chain.get_current_block(subtensor)
                    blocks_until = await chain.blocks_until_next_epoch(
                        subtensor, self.netuid, block=current_block,
                    )
                    if blocks_until is None:
                        logger.warning("blocks_until_next_epoch returned None — retrying")
                        await asyncio.sleep(POLL_INTERVAL_SECONDS)
                        continue
                    # Stable per-epoch identifier: the absolute block number of the
                    # next epoch boundary stays constant for every poll inside the
                    # current epoch (current_block + blocks_until is invariant).
                    current_epoch_id = current_block + blocks_until

                    if self._last_submit_epoch == current_epoch_id:
                        await asyncio.sleep(POLL_INTERVAL_SECONDS)
                        continue

                    bootstrap = self._last_submit_epoch is None
                    in_lead_window = blocks_until <= EPOCH_SUBMIT_LEAD_BLOCKS
                    if not bootstrap and not in_lead_window:
                        await asyncio.sleep(POLL_INTERVAL_SECONDS)
                        continue

                    submitted = await self.submit_once()
                    self._last_submit_epoch = current_epoch_id
                    logger.info(
                        "Weight epoch attempt: epoch=%d snapshot_block=%d "
                        "blocks_until=%d success=%s",
                        current_epoch_id,
                        current_block,
                        blocks_until,
                        submitted,
                    )
                    if not submitted:
                        logger.warning(
                            "set_weights attempt for epoch %d failed; "
                            "waiting for the next epoch before retry",
                            current_epoch_id,
                        )
                except asyncio.CancelledError:
                    raise
                except asyncio.TimeoutError:
                    logger.warning(
                        "substrate call timed out — recycling polling subtensor",
                    )
                    await chain.close_subtensor(subtensor)
                    subtensor = None
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                except Exception:
                    logger.exception("weight-only loop iteration failed")
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
        finally:
            await chain.close_subtensor(subtensor)

    async def submit_once(self) -> bool:
        """Run one set_weights cycle: read R2 archives → replay EMA → submit
        on-chain. Returns True iff the chain accepted the extrinsic.

        Opens its own short-lived subtensor for the heavy metagraph +
        ``set_weights`` calls, then closes it. Keeps the polling subtensor
        clean: a hung extrinsic or stalled metagraph here cannot leak
        WebSocket state back into the main loop, and a poisoned polling
        connection cannot stall a submission. We open many of these per
        day (one per epoch), and ``initialize`` is cheap (~0.5s).
        """
        windows = await storage.list_all_window_keys()
        if not windows:
            logger.info("No archives yet; nothing to submit")
            return False

        archives = await storage.list_recent_datasets(
            current_window=max(windows) + 1,
            n=ROLLING_WINDOWS_HISTORY * 3,
        )
        ema = self._replay_ema(archives)
        miner_weights = dict(ema)

        subtensor = await chain.get_subtensor()
        try:
            submitted = await self._submit_weights(subtensor, miner_weights)
        finally:
            await chain.close_subtensor(subtensor)
        return submitted

    @staticmethod
    def _replay_ema(archives: list[dict]) -> dict[str, float]:
        """Replay the per-window emission distribution into an EMA.

        Reads ``rewards_by_hotkey`` from each archive — the single source of
        truth for what each miner earned that window, computed by
        ``select_batch_and_distribute`` at seal time. The dict's values are
        already in units of ``pool`` (≤ 1.0 per window), so they map
        directly onto the EMA fraction with no further normalization.

        Why this matters: the emission distribution has three properties
        the older "count entries in ``batch``" approach did not preserve
        on-chain:

          * K-way same-prompt split. K miners on one winning prompt each
            earn ``slot_share / K`` in ``rewards_by_hotkey``. The old code
            only counted the canonical training representative, which
            collapsed K sybils' total reward onto the rep — the K-way
            split existed in the archive but never reached the chain.
          * Boundary-round fair split. When the round that crosses
            ``B_BATCH`` has more prompts than remaining slots, every
            prompt in that round earns its share via
            ``rewards_by_hotkey`` (see ``select_batch_and_distribute``).
            The old code only credited the canonical-hash winners in the
            training batch.
          * Conservation. ``rewards_by_hotkey`` sums to at most ``pool``
            per window, with unfilled slots burning their share. The EMA
            fraction inherits the same bound naturally.
        """
        ema: dict[str, float] = {}
        alpha = EMA_ALPHA
        for record in sorted(archives, key=lambda r: int(r["window_start"])):
            rewards: dict[str, float] = record.get("rewards_by_hotkey", {})
            all_hotkeys = set(ema) | set(rewards)
            for hk in all_hotkeys:
                fraction = rewards.get(hk, 0.0)
                ema[hk] = alpha * fraction + (1 - alpha) * ema.get(hk, 0.0)
            ema = {hk: v for hk, v in ema.items() if v > 1e-6}
        return ema

    async def _submit_weights(
        self, subtensor, miner_weights: dict[str, float],
    ) -> bool:
        meta = await chain.get_metagraph(subtensor, self.netuid)
        hotkey_to_uid = dict(zip(meta.hotkeys, meta.uids))
        weights_by_uid: dict[int, float] = {}
        registered_total = 0.0
        registered_hotkey_count = 0
        omitted_total = 0.0
        for hk, w in miner_weights.items():
            if w <= 0:
                continue
            if hk not in hotkey_to_uid:
                omitted_total += w
                continue
            uid = int(hotkey_to_uid[hk])
            weights_by_uid[uid] = weights_by_uid.get(uid, 0.0) + w
            registered_total += w
            registered_hotkey_count += 1

        # The on-chain vector must conserve the full unit mass after filtering
        # against the current metagraph. Historical EMA belonging to a hotkey
        # that has since deregistered is no longer payable and therefore burns;
        # dropping it would make the vector sum below one and let chain-side
        # normalization redistribute that mass among remaining miners.
        burn_weight = max(0.0, 1.0 - registered_total)
        if burn_weight > 0:
            weights_by_uid[UID_BURN] = (
                weights_by_uid.get(UID_BURN, 0.0) + burn_weight
            )
        if not weights_by_uid:
            logger.info("No non-zero weights to submit; nothing to do.")
            return True

        uids = list(weights_by_uid)
        weight_vals = [weights_by_uid[uid] for uid in uids]
        submitted = await chain.set_weights(
            subtensor, self.wallet, self.netuid, uids, weight_vals,
        )
        if submitted:
            logger.info(
                "Submitted weights: %d registered miners "
                "(miner_total=%.4f, omitted=%.4f), burn=%.4f",
                registered_hotkey_count,
                registered_total,
                omitted_total,
                burn_weight,
            )
        return submitted
