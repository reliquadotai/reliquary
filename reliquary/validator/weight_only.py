"""Weight-only validator — reads R2 archives, submits weights on-chain.

No model, no HTTP server, no HF writes. Meant to be run alongside a
trainer validator; any number of weight-only nodes can participate and
all submit consistent weights because they read the same R2 archives.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any

from reliquary.constants import (
    EMA_ALPHA,
    EPOCH_SUBMIT_LEAD_BLOCKS,
    POLL_INTERVAL_SECONDS,
)
from reliquary.infrastructure import chain, storage
from reliquary.infrastructure.task_registry_store import read_registry

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
        from botocore.exceptions import ClientError

        by_task: dict[str, list[dict]] = {}
        try:
            for task_id in await storage.list_task_ids(strict=True):
                windows = await storage.list_all_window_keys(task_id=task_id, strict=True)
                if not windows:
                    continue
                by_task[task_id] = await storage.list_recent_datasets(
                    current_window=max(windows) + 1,
                    n=ROLLING_WINDOWS_HISTORY * 3,
                    task_id=task_id,
                )
        except ClientError:
            # A partial listing would submit a confident vector that pays
            # only the tasks we managed to see — worse than submitting
            # nothing. Abstain and let the next epoch retry the listing.
            logger.exception("Archive listing failed; abstaining from this epoch")
            return False
        if not by_task:
            logger.info("No archives yet; nothing to submit")
            return False

        try:
            declared, _ = await read_registry()
        except Exception:
            logger.exception("Task registry unreadable; abstaining from this epoch")
            return False
        undeclared = self._undeclared_tasks(by_task, declared)
        if undeclared:
            logger.error(
                "Tasks %s have archives but are not declared in the registry; "
                "abstaining rather than paying under unknown rules",
                undeclared,
            )
            return False

        archives = self._merge_archives(by_task)
        logger.info(
            "Replaying %d archives across %d task(s): %s",
            len(archives), len(by_task), ", ".join(sorted(by_task)),
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
    def _merge_archives(by_task: Mapping[str, list[dict]]) -> list[dict]:
        """One ordered stream out of every task's archives.

        Trusts the bucket key each archive was read under, not any
        ``task_id`` field in its body — the body can't forge which
        namespace an object physically lives in, and that's the only thing
        that should decide which task's decay clock and payout pool it
        joins.
        """
        merged = [
            {**archive, "task_id": task_id}
            for task_id, archives in by_task.items()
            for archive in archives
        ]
        return sorted(
            merged,
            key=lambda record: (int(record["window_start"]), str(record.get("task_id", ""))),
        )

    @staticmethod
    def _undeclared_tasks(by_task, declared) -> list[str]:
        """Archived tasks the registry does not know about."""
        return sorted(set(by_task) - set(declared))

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
        selection.

        Archives may come from several tasks (see ``_merge_archives``). Each
        task replays on its own decay clock, independent of every other
        task's windows — a task with no archives yet, or one that never pays
        a given hotkey, must not touch that hotkey's EMA. Each task's own
        ``rewards_by_hotkey`` already sums to at most that task's configured
        emission share (``window_pool``), so its own EMA sums to at most that
        share too; the combined total across tasks is therefore at most one
        pool when shares are configured correctly, and whatever is not paid
        burns. The clamp below is only a backstop against a misconfigured
        sum of shares exceeding one pool — it is logged, never silent,
        because it rescales every miner's emission.
        """
        by_task: dict[str, list[dict]] = {}
        for record in archives:
            by_task.setdefault(record.get("task_id", ""), []).append(record)

        combined: dict[str, float] = {}
        for records in by_task.values():
            ema: dict[str, float] = {}
            alpha = EMA_ALPHA
            for record in sorted(records, key=lambda r: int(r["window_start"])):
                if record.get("window_status", "completed") == "aborted":
                    continue
                rewards: dict[str, float] = record.get("rewards_by_hotkey", {})
                all_hotkeys = set(ema) | set(rewards)
                for hk in all_hotkeys:
                    fraction = rewards.get(hk, 0.0)
                    ema[hk] = alpha * fraction + (1 - alpha) * ema.get(hk, 0.0)
                ema = {hk: v for hk, v in ema.items() if v > 1e-6}
            for hk, v in ema.items():
                combined[hk] = combined.get(hk, 0.0) + v

        total = sum(combined.values())
        if total > 1.0:
            logger.warning(
                "Combined EMA across tasks %s totals %.4f (> 1.0 pool); "
                "rescaling every hotkey proportionally — check that "
                "configured task emission shares sum to at most 1.0",
                sorted(by_task), total,
            )
            combined = {hk: v / total for hk, v in combined.items()}
        return combined

    def _resolve_burn_uid(self, hotkey_to_uid: dict) -> int:
        """Where the unpayable share goes.

        ``UID_BURN`` unset means this validator's own uid, looked up from the
        metagraph by its hotkey, so the target follows re-registration and
        survives a subnet-ownership change. Falls back to 0 when this
        validator is absent from the metagraph: the burn MUST land somewhere,
        because a weight vector summing below one lets chain-side
        normalization redistribute that mass among the remaining miners.
        """
        # Lazy import so tests (and a redeploy-free env change) can rebind it.
        from reliquary.constants import UID_BURN as _uid_burn

        if _uid_burn is not None:
            return int(_uid_burn)
        own_uid = hotkey_to_uid.get(self.validator_hotkey)
        if own_uid is None:
            logger.warning(
                "burn uid: this validator's hotkey is absent from the "
                "metagraph; falling back to uid 0 to conserve weight mass"
            )
            return 0
        return int(own_uid)

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
            burn_uid = self._resolve_burn_uid(hotkey_to_uid)
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
