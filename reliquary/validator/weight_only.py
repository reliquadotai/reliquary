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

    A freshly-booted validator submits when chain rate limits permit, then
    joins the synced cadence. Remote signer attempts survive controller restarts.

    No local state: every submit recomputes from scratch.
    """

    def __init__(self, wallet, netuid: int, *, signer_client: Any | None = None) -> None:
        self.wallet = wallet
        self.netuid = netuid
        self.signer_client = signer_client
        self.validator_hotkey = str(wallet.hotkey.ss58_address)
        self._last_submit_epoch: int | None = None
        self._active_submit_epoch: int | None = None
        self._bootstrap_pending = True

    async def run(self) -> None:
        """Poll the epoch boundary and submit weights once per epoch.

        Owns its own polling ``AsyncSubtensor``; closes and replaces it on
        any chain-call timeout so a wedged WebSocket can't permanently
        stall the loop. The trainer service runs on a separate subtensor,
        so neither side can poison the other's connection state.
        """
        from reliquary.signer.backend import weights_submission_wait_blocks

        logger.info(
            "Weight-only validator started (netuid=%d, hotkey=%s)",
            self.netuid, self.validator_hotkey,
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

                    if self.signer_client is not None:
                        self._last_submit_epoch = await self.signer_client.last_weight_epoch()
                    if (
                        self._last_submit_epoch is not None
                        and self._last_submit_epoch >= current_epoch_id
                    ):
                        await asyncio.sleep(POLL_INTERVAL_SECONDS)
                        continue

                    # Restart catch-up still waits for an unattempted epoch and
                    # chain eligibility; a consumed epoch must not turn it into
                    # another full epoch of delay before the first refresh.
                    bootstrap = self._bootstrap_pending
                    in_lead_window = blocks_until <= EPOCH_SUBMIT_LEAD_BLOCKS
                    if not bootstrap and not in_lead_window:
                        await asyncio.sleep(POLL_INTERVAL_SECONDS)
                        continue

                    wait_blocks = await weights_submission_wait_blocks(
                        subtensor, self.netuid, self.validator_hotkey,
                        block=current_block,
                    )
                    if wait_blocks:
                        logger.info(
                            "Weight submission deferred: epoch=%d snapshot_block=%d "
                            "retry_after_blocks=%d",
                            current_epoch_id, current_block, wait_blocks,
                        )
                        await asyncio.sleep(POLL_INTERVAL_SECONDS)
                        continue

                    self._active_submit_epoch = current_epoch_id
                    submitted = await self.submit_once()
                    self._last_submit_epoch = current_epoch_id
                    self._bootstrap_pending = False
                    logger.info(
                        "Weight epoch attempt: epoch=%d snapshot_block=%d "
                        "blocks_until=%d success=%s bootstrap=%s",
                        current_epoch_id,
                        current_block,
                        blocks_until,
                        submitted,
                        bootstrap,
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

    async def submit_once(self, *, epoch_id: int | None = None) -> bool:
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
            previous_epoch = self._active_submit_epoch
            if epoch_id is not None:
                self._active_submit_epoch = int(epoch_id)
            try:
                submitted = await self._submit_weights(subtensor, miner_weights)
            finally:
                self._active_submit_epoch = previous_epoch
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

        The field is deliberately mechanism-agnostic. Historical archives may
        contain same-prompt/boundary splits; auction-v2 archives contain one
        uniform share per proven winner. Replaying the authoritative field
        preserves both eras without asking weight-only nodes to reimplement
        selection. Its sum remains at most one pool and unfilled shares burn.
        """
        ema: dict[str, float] = {}
        alpha = EMA_ALPHA
        for record in sorted(archives, key=lambda r: int(r["window_start"])):
            if record.get("window_status", "completed") == "aborted":
                continue
            rewards: dict[str, float] = record.get("rewards_by_hotkey", {})
            all_hotkeys = set(ema) | set(rewards)
            for hk in all_hotkeys:
                fraction = rewards.get(hk, 0.0)
                ema[hk] = alpha * fraction + (1 - alpha) * ema.get(hk, 0.0)
            ema = {hk: v for hk, v in ema.items() if v > 1e-6}
        return ema

    @staticmethod
    def _resolve_burn_uid(metagraph, hotkey_to_uid: dict) -> int:
        """Where the unpayable share goes.

        ``UID_BURN`` unset uses the owner hotkey carried by this exact
        metagraph snapshot. Missing owner state fails closed: paying a
        validator or a guessed UID is not burn.
        """
        # Lazy import so tests (and a redeploy-free env change) can rebind it.
        from reliquary.constants import UID_BURN as _uid_burn

        if _uid_burn is not None:
            return int(_uid_burn)
        owner_hotkey = getattr(metagraph, "owner_hotkey", None)
        if not isinstance(owner_hotkey, str) or not owner_hotkey:
            raise RuntimeError("subnet owner hotkey unavailable in metagraph")
        owner_uid = hotkey_to_uid.get(owner_hotkey)
        if owner_uid is None:
            raise RuntimeError(
                "subnet owner hotkey has no UID in the current metagraph"
            )
        return int(owner_uid)

    async def _submit_weights(
        self,
        subtensor,
        miner_weights: dict[str, float],
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
            burn_uid = self._resolve_burn_uid(meta, hotkey_to_uid)
            weights_by_uid[burn_uid] = (
                weights_by_uid.get(burn_uid, 0.0) + burn_weight
            )
        if not weights_by_uid:
            logger.info("No non-zero weights to submit; nothing to do.")
            return True

        uids = list(weights_by_uid)
        weight_vals = [weights_by_uid[uid] for uid in uids]
        if self.signer_client is None:
            submitted = await chain.set_weights(
                subtensor, self.wallet, self.netuid, uids, weight_vals,
            )
        else:
            if self._active_submit_epoch is None:
                raise RuntimeError("remote signer weight submission requires epoch_id")
            submitted = await self.signer_client.set_weights(
                epoch_id=self._active_submit_epoch,
                netuid=self.netuid,
                uids=uids,
                weights=weight_vals,
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
