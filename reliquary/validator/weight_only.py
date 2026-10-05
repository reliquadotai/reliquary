"""Weight-only validator — reads R2 archives, submits weights on-chain.

No model, no HTTP server, no HF writes. Meant to be run alongside a
trainer validator; any number of weight-only nodes can participate and
all submit consistent weights because they read the same R2 archives.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Mapping
from typing import Any

from reliquary.constants import (
    EMA_ALPHA,
    EPOCH_SUBMIT_LEAD_BLOCKS,
    MIN_INCENTIVE_RAMP_START,
    MIN_INCENTIVE_SHARE,
    POLL_INTERVAL_SECONDS,
)
from reliquary.infrastructure import chain, storage
from reliquary.infrastructure.task_registry_store import read_registry
from reliquary.validator.task_config import legacy_registry_fallback

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
            windows_by_task: dict[str, list[int]] = {}
            for task_id in await storage.list_task_ids(strict=True):
                windows = await storage.list_all_window_keys(task_id=task_id, strict=True)
                if windows:
                    windows_by_task[task_id] = windows
            # ONE horizon, taken across every task, not each task's own last
            # window. Anchoring per task hands a task that stopped producing
            # its own final 216 windows forever, so its EMA is recomputed at
            # full strength and retirement never decays anything. With a
            # single task the global maximum IS its own maximum, so `default`
            # alone reads exactly the same archives as before.
            horizon = max(
                (max(w) for w in windows_by_task.values()), default=0
            ) + 1
            for task_id in windows_by_task:
                archives = await storage.list_recent_datasets(
                    current_window=horizon,
                    n=ROLLING_WINDOWS_HISTORY * 3,
                    task_id=task_id,
                    # The replay reads only these; the task comes from the R2 prefix.
                    fields=("window_start", "window_status", "rewards_by_hotkey"),
                )
                # A task whose last window fell out of the shared horizon
                # contributes nothing, and must not be counted as an
                # archived task either.
                if archives:
                    by_task[task_id] = archives
        except ClientError:
            # A partial listing would submit a confident vector that pays
            # only the tasks we managed to see — worse than submitting
            # nothing. Abstain and let the next epoch retry the listing.
            logger.exception("Archive listing failed; abstaining from this epoch")
            return False
        try:
            declared, _ = await read_registry()
        except Exception:
            logger.exception("Task registry unreadable; abstaining from this epoch")
            return False
        try:
            periods = await self._period_weights(declared)
        except Exception:
            # A period task's pay unreadable: a vector without it would hand its
            # share to nobody this epoch and look confident doing so.
            logger.exception("Period-settled pay unreadable; abstaining from this epoch")
            return False
        if not by_task and not periods:
            logger.info("No archives yet; nothing to submit")
            return False
        undeclared = self._undeclared_tasks(by_task, declared)
        if undeclared:
            if legacy_registry_fallback(declared, by_task):
                # No registry object yet. The legacy task predates it and the
                # startup path admits it for the same reason; abstaining here
                # would stop paying everyone instead of protecting anyone.
                logger.warning(
                    "No task registry in R2; paying the legacy task alone. "
                    "Declare it with `reliquary tasks create` to enable the "
                    "cross-task check."
                )
            else:
                logger.error(
                    "Tasks %s have archives but are not declared in the "
                    "registry; abstaining rather than paying under unknown "
                    "rules",
                    undeclared,
                )
                return False

        archives = self._merge_archives(by_task)
        logger.info(
            "Replaying %d archives across %d task(s): %s",
            len(archives), len(by_task), ", ".join(sorted(by_task)),
        )
        # The floor runs inside _replay_ema, per task: a hotkey's share is
        # measured against its own task, and what it cuts stays in that task.
        ema = self._replay_ema(
            archives,
            caps=self._caps_by_task(declared),
            floors=self._floors_by_task(declared),
            periods=periods,
        )
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
    async def _period_weights(declared: Mapping[str, Any], *, archives=None,
                              now: float | None = None,
                              genesis: float | None = None) -> dict[str, dict[str, float]]:
        """Each period-settled task's weights at the current drand period: its
        archives that entered within the replay depth, decayed once per period
        (design 2026-10-03). Window archives never hold these tasks' pay.

        Each archive is held to the task's cap: it is one period's pay, and
        ``_replay_ema`` bounds the task at ``CATCHUP_ENTRIES`` caps on that
        ground. An archive moved to an earlier entry (``replaces_entry_period``,
        scripts/requeue_period_archives.py) hides the one it replaces, should
        that one still be listed."""
        from reliquary.validator import corpus_periods as cp

        tasks = sorted(str(t) for t, e in (declared or {}).items() if cp.is_period_task(e))
        if not tasks:
            return {}
        if archives is None:
            from reliquary.infrastructure.corpus_period_store import R2PeriodArchives

            archives = R2PeriodArchives()
        if genesis is None:
            from reliquary.validator.corpus_period_settlement import _drand_genesis

            genesis = _drand_genesis()
        current = cp.period_of(time.time() if now is None else now, genesis)
        caps = WeightOnlyValidator._caps_by_task(declared)
        weights: dict[str, dict[str, float]] = {}
        for task_id in tasks:
            listed = await archives.list(task_id)
            keys = [(work, entry) for work, entry in listed
                    if 0 <= current - entry <= cp.REPLAY_DEPTH]
            # A key that may have been replaced: an earlier entry of its work
            # period exists. Those earlier archives are read to find out.
            entries_of: dict[int, list[int]] = {}
            for work, entry in listed:
                entries_of.setdefault(work, []).append(entry)
            read: dict[tuple[int, int], Mapping] = {}

            async def doc_of(work, entry):
                if (work, entry) not in read:
                    doc = await archives.read(task_id, work, entry)
                    if doc is None:
                        raise RuntimeError(f"period archive {task_id} {work}-{entry} "
                                           "listed but unreadable")
                    read[(work, entry)] = doc
                return read[(work, entry)]

            replaced: set[tuple[int, int]] = set()
            for work, entry in keys:
                for earlier in entries_of[work]:
                    if earlier < entry:
                        moved = (await doc_of(work, earlier)).get("replaces_entry_period")
                        if moved is not None and int(moved) == entry:
                            replaced.add((work, entry))
            cap = caps.get(task_id)
            docs = []
            for work, entry in keys:
                if (work, entry) in replaced:
                    continue
                rewards = {str(hk): float(v) for hk, v in
                           ((await doc_of(work, entry)).get("rewards_by_hotkey") or {}).items()}
                paid = sum(rewards.values())
                if cap is not None and paid > cap * (1 + 1e-9):
                    logger.warning("period archive %s %d-%d pays %.4f over its cap %.4f; "
                                   "scaled down", task_id, work, entry, paid, cap)
                    rewards = {hk: v * cap / paid for hk, v in rewards.items()}
                docs.append({"entry_period": entry, "rewards_by_hotkey": rewards})
            replayed = cp.replay(docs, current)
            if replayed:
                weights[task_id] = replayed
        return weights

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
    def _ramped_incentive(value: float, *, start: float, threshold: float) -> float:
        """How much of ``value`` a hotkey holding that share of the pool is paid.

        Linear ramp: nothing at or below ``start``, ``value`` in full at or
        above ``threshold``, and in between a fraction of ``value`` itself
        that grows linearly from 0 to 1 across the ramp -- replacing a hard
        cliff at ``threshold`` alone, which gives an infinite marginal return
        to crossing it and so pays miners to merge hotkeys.

        ``start == threshold`` collapses the ramp to zero width: every value
        is then either at/above it (paid in full) or below it (paid
        nothing), which is exactly today's cliff. Guarded explicitly so the
        division is never attempted for a zero- (or negative-) width ramp,
        even though that case cannot otherwise be reached with a validated
        ``start <= threshold``.
        """
        if value >= threshold:
            return value
        if start >= threshold or value < start:
            return 0.0
        return value * (value - start) / (threshold - start)

    @staticmethod
    def _apply_min_incentive_share(
        weights: Mapping[str, float], *, start: float, threshold: float
    ) -> dict[str, float]:
        """Ramp down hotkeys holding too small a share, and share their mass out.

        The floor reads each hotkey's share of the miner total, not its absolute
        weight, so a falling price never drops everyone under it; and the total is
        preserved, so only the price and the caps decide what burns.
        """
        total = math.fsum(weights.values())
        if total <= 0.0:
            return dict(weights)
        shares = {hotkey: value / total for hotkey, value in weights.items()}
        kept = {
            hotkey: WeightOnlyValidator._ramped_incentive(
                share, start=start, threshold=threshold
            )
            for hotkey, share in shares.items()
        }
        if all(kept[hotkey] == share for hotkey, share in shares.items()):
            return dict(weights)
        kept = {hotkey: share for hotkey, share in kept.items() if share > 0.0}
        kept_total = math.fsum(kept.values())
        if kept_total <= 0.0:
            # Filtering out every hotkey would switch payment off, not favour anyone.
            logger.warning(
                "Every hotkey holds less than %.4f of the miner total; "
                "paying the unfiltered weights",
                start,
            )
            return dict(weights)
        paid = {hotkey: total * share / kept_total for hotkey, share in kept.items()}
        # Rescaling can overshoot the total by a few ULPs; take them back from
        # the largest payment so the vector never sums past where it started.
        overshoot = math.fsum(paid.values()) - total
        if overshoot > 0.0:
            largest = max(paid, key=paid.__getitem__)
            paid[largest] -= overshoot
            while math.fsum(paid.values()) > total:
                paid[largest] = math.nextafter(paid[largest], 0.0)
        return paid

    @staticmethod
    def _floors_by_task(declared: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
        """Each declared task's (ramp start, share) floor; the protocol's by default.

        A task that names only a share below the protocol's ramp start ramps
        from zero, so a declared floor is never raised by the default one.
        """
        floors: dict[str, tuple[float, float]] = {}
        for task_id, entry in (declared or {}).items():
            params = getattr(entry, "params", None)
            if not isinstance(params, Mapping) or "min_incentive_share" not in params:
                floors[str(task_id)] = (MIN_INCENTIVE_RAMP_START, MIN_INCENTIVE_SHARE)
                continue
            share = float(params["min_incentive_share"])
            default_start = MIN_INCENTIVE_RAMP_START if MIN_INCENTIVE_RAMP_START <= share else 0.0
            start = float(params.get("min_incentive_ramp_start", default_start))
            floors[str(task_id)] = (start, share)
        return floors

    @staticmethod
    def _caps_by_task(declared: Mapping[str, Any]) -> dict[str, float]:
        """The most each declared task may pay, keyed by task id.

        ``read_registry`` validates every entry before returning it, so a real
        registry always yields a finite numeric ``cap`` here. An entry that
        does not carry one is left out rather than assumed: an unclamped task
        still meets the global backstop below, whereas guessing a cap would
        invent a number nobody declared.
        """
        caps: dict[str, float] = {}
        for task_id, entry in (declared or {}).items():
            params = getattr(entry, "params", None)
            if not isinstance(params, Mapping):
                continue
            try:
                caps[str(task_id)] = float(params["cap"])
            except (KeyError, TypeError, ValueError):
                continue
        return caps

    @staticmethod
    def _replay_ema(
        archives: list[dict],
        *,
        caps: Mapping[str, float] | None = None,
        floors: Mapping[str, tuple[float, float]] | None = None,
        periods: Mapping[str, Mapping[str, float]] | None = None,
    ) -> dict[str, float]:
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
        burns.

        ``caps`` is what the registry says each task may pay, and it is
        enforced HERE because this is where money is assigned. The producing
        validator's ``window_pool`` binds only itself: a box running a stale
        image, or one whose task was declared at a lower cap after it started,
        archives a full pool regardless. Clamping per task before combining
        keeps that task inside its own declared budget instead of letting the
        global backstop rescale every other task's miners to pay for it. A
        task with no declared cap (the legacy pre-registry fallback) is not
        clamped, so ``default`` alone is byte-for-byte unchanged.

        The global clamp at the end stays as a backstop against a
        misconfigured sum of shares exceeding one pool — it is logged, never
        silent, because it rescales every miner's emission.
        """
        by_task: dict[str, list[dict]] = {}
        for record in archives:
            by_task.setdefault(record.get("task_id", ""), []).append(record)

        combined: dict[str, float] = {}
        # Period-settled tasks arrive already replayed on their own clock, and
        # are added to whatever window archives the same task still has (a job
        # settled by window before its validator learnt periods): neither tail
        # is dropped. The cap and the floor apply to the sum, as to every task.
        periods = periods or {}
        for task_id in dict.fromkeys((*by_task, *periods)):
            ema = {}
            alpha = EMA_ALPHA
            for record in sorted(by_task.get(task_id, ()), key=lambda r: int(r["window_start"])):
                if record.get("window_status", "completed") == "aborted":
                    continue
                rewards: dict[str, float] = record.get("rewards_by_hotkey", {})
                all_hotkeys = set(ema) | set(rewards)
                for hk in all_hotkeys:
                    fraction = rewards.get(hk, 0.0)
                    ema[hk] = alpha * fraction + (1 - alpha) * ema.get(hk, 0.0)
                ema = {hk: v for hk, v in ema.items() if v > 1e-6}
            for hk, v in (periods.get(task_id) or {}).items():
                ema[hk] = ema.get(hk, 0.0) + float(v)
            cap = None if caps is None else caps.get(task_id)
            if cap is not None and task_id in periods:
                from reliquary.validator import corpus_periods as cp

                # Archives hold one cap each and up to CATCHUP_ENTRIES enter in
                # one period: a backlog paid back is that much at most, and
                # comes out of what would otherwise burn.
                cap = cap * cp.CATCHUP_ENTRIES
            ema = WeightOnlyValidator._clamp_to_cap(task_id, ema, cap)
            if floors is not None:
                start, threshold = floors.get(
                    task_id, (MIN_INCENTIVE_RAMP_START, MIN_INCENTIVE_SHARE)
                )
                if threshold > 0.0:
                    ema = WeightOnlyValidator._apply_min_incentive_share(
                        ema, start=start, threshold=threshold
                    )
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

    @staticmethod
    def _clamp_to_cap(
        task_id: str, ema: dict[str, float], cap: float | None
    ) -> dict[str, float]:
        """One task's EMA, scaled down to the most that task may pay."""
        if cap is None:
            return ema
        share = sum(ema.values())
        if share <= cap:
            return ema
        logger.warning(
            "Task %r replays to %.4f of the pool but is declared at cap "
            "%.4f; scaling its miners down to its own budget rather than "
            "letting it dilute the other tasks",
            task_id, share, cap,
        )
        if cap <= 0.0 or share <= 0.0:
            return {}
        scale = cap / share
        return {hk: v * scale for hk, v in ema.items()}

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
