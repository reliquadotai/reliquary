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
    TOPHEAVY_EMA_FLOOR,
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
                    blocks_until = await chain.blocks_until_next_epoch(
                        subtensor, self.netuid,
                    )
                    if blocks_until is None:
                        logger.warning("blocks_until_next_epoch returned None — retrying")
                        await asyncio.sleep(POLL_INTERVAL_SECONDS)
                        continue
                    current_block = await chain.get_current_block(subtensor)
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
        ema = self._apply_topheavy_cutoff(ema)
        miner_weights = dict(ema)
        total = sum(miner_weights.values())
        burn_weight = max(0.0, 1.0 - total)

        subtensor = await chain.get_subtensor()
        try:
            submitted = await self._submit_weights(
                subtensor, miner_weights, burn_weight,
            )
        finally:
            await chain.close_subtensor(subtensor)

        if submitted:
            logger.info(
                "Submitted weights: %d miners (total=%.4f), burn=%.4f",
                len(miner_weights), total, burn_weight,
            )
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

    @staticmethod
    def _apply_topheavy_cutoff(ema: dict[str, float]) -> dict[str, float]:
        """Zero miners below ``TOPHEAVY_EMA_FLOOR`` and redistribute their
        share pro-rata to the survivors.

        Keeps the total miner mass constant, so the burn weight is
        unchanged — the dropped miners' emission concentrates on the
        larger ones (top-heavy). Deterministic: every validator replays
        identical archives and applies the same floor, so on-chain weights
        still converge.

        Guard: if no miner clears the floor (e.g. a sparse early run),
        pass the EMA through unchanged rather than sending everything to
        burn.
        """
        floor = TOPHEAVY_EMA_FLOOR
        if floor <= 0.0:
            return ema
        survivors = {hk: v for hk, v in ema.items() if v >= floor}
        if not survivors:
            logger.warning(
                "Top-heavy cutoff: no miner ≥ floor %.4f; passing EMA through",
                floor,
            )
            return ema
        total = sum(ema.values())
        surv_total = sum(survivors.values())
        scale = total / surv_total
        dropped = len(ema) - len(survivors)
        if dropped:
            logger.info(
                "Top-heavy cutoff: dropped %d miners < %.4f → %d survivors (×%.3f)",
                dropped, floor, len(survivors), scale,
            )
        return {hk: v * scale for hk, v in survivors.items()}

    async def _submit_weights(
        self, subtensor, miner_weights: dict[str, float], burn_weight: float,
    ) -> bool:
        meta = await chain.get_metagraph(subtensor, self.netuid)
        hotkey_to_uid = dict(zip(meta.hotkeys, meta.uids))
        uids: list[int] = []
        weight_vals: list[float] = []
        for hk, w in miner_weights.items():
            if hk in hotkey_to_uid and w > 0:
                uids.append(int(hotkey_to_uid[hk]))
                weight_vals.append(w)
        if burn_weight > 0:
            uids.append(UID_BURN)
            weight_vals.append(burn_weight)
        if not uids:
            logger.info("No non-zero weights to submit; nothing to do.")
            return True
        return await chain.set_weights(
            subtensor, self.wallet, self.netuid, uids, weight_vals,
        )
