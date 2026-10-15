"""`GET /rl/pay`: what each RL environment is paying, for miners choosing where
to put their GPUs. Read-only: it changes nothing in selection, payment or the
weights.

Everything comes from the last windows' archives (fed by the service as it
archives a window, and seeded from R2 at startup) and from the chain:

- A hotkey's weight is the EMA of its per-window share, and a window's share is
  paid out over the whole CYCLE (open to the next window's open), idle gap
  included. So the alpha a window pays is ``share x miner_alpha_per_block x
  cycle / 12 s``, and per paid token it is that over the window's completion
  tokens.
- Per GPU-hour is a different number in RL: an environment accepts groups only
  part of the cycle (its quota fills, then it refuses), and only some of the
  groups sent are paid. Both fractions are published, with the bounded
  estimate for a stated generation speed.

The whole answer is recomputed at most once per ``PAY_CACHE_SECONDS``; the
chain is read at most once per ``CHAIN_REFRESH_SECONDS`` and a failed read
keeps the last good one until ``CHAIN_STALE_SECONDS``.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import math
import statistics
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

PAY_CACHE_SECONDS = 60.0
CHAIN_REFRESH_SECONDS = 600.0
CHAIN_RETRY_SECONDS = 60.0
CHAIN_STALE_SECONDS = 3600.0
CHAIN_READ_BUDGET_SECONDS = 200.0
WINDOWS_KEPT = 72
SEED_WINDOWS = 24
HISTORY_WINDOWS = 12
ROUND_SECONDS = 3.0
BLOCK_SECONDS = 12.0
BLOCKS_PER_HOUR = 3600.0 / BLOCK_SECONDS

NOTE = (
    "An estimate, not a promise. A window pays each environment's pool split by its paid "
    "groups; a hotkey's weight is the EMA of its per-window share (half-life ~25 windows), "
    "so alpha is reckoned over the whole cycle from one window's open to the next. "
    "alpha_per_mtoken_paid counts every completion token of the paid groups. Per GPU-hour, "
    "an environment accepts groups only part of the cycle (accepting_fraction_of_cycle is a "
    "lower bound: the time to reach its target) and pays only part of what is sent "
    "(paid_fraction_of_submitted: low counts every refused precommit as a lost group, high counts none, since refusals are retried); estimate.alpha_per_gpu_hour, a low/high range, is an upper bound for a miner "
    "alone at the stated tokens_per_second, since miners share each environment's demand. "
    "A hotkey under 2 % of the miner total is ramped down, payment goes through Yuma "
    "consensus with the other validators, exploration groups are paid 15 %, and prices move "
    "every window."
)


def _median(values: Iterable[float]) -> float | None:
    values = [float(v) for v in values if v is not None]
    return statistics.median(values) if values else None


def window_stat(archive: Mapping[str, Any]) -> dict | None:
    """One archived window reduced to what the pay board needs, or None."""
    try:
        window = int(archive["window_start"])
        open_round = int(archive["window_open_round"])
        close_round = int(archive["window_close_round"])
    except (KeyError, TypeError, ValueError):
        return None
    if archive.get("window_status", "completed") == "aborted":
        return None
    environments = [str(e) for e in (archive.get("environments") or [])]
    picks = int(archive.get("picks_target") or 1)
    targets = archive.get("batch_targets") or {}
    pools = archive.get("window_pool_by_environment")
    ready = archive.get("collect_ready_round_by_environment") or {}
    shadow = ((archive.get("emission_price_shadow") or {}).get("by_environment") or {})
    paid = [g for g in archive.get("batch") or [] if g.get("selected_for_batch")]
    paid_total = math.fsum(float(v) for v in (archive.get("rewards_by_hotkey") or {}).values())
    out: dict[str, dict] = {}
    for environment in environments:
        groups = [g for g in paid if g.get("env_name") == environment]
        rejected = [r for r in archive.get("rejected") or [] if r.get("env_name") == environment]
        completion = prompt = 0
        for group in groups:
            for rollout in group.get("rollouts") or []:
                length = int(rollout.get("completion_length") or 0)
                completion += length
                tokens = rollout.get("tokens")
                if isinstance(tokens, list):
                    prompt += max(0, len(tokens) - length)
        slots = picks * int(targets.get(environment) or 0)
        if isinstance(pools, Mapping) and environment in pools:
            pool = float(pools[environment])
            share = pool * len(groups) / slots if slots else 0.0
        else:
            # Archives written before the per-environment pool: every paid group
            # was paid the same, so split the window's paid total by groups.
            pool = None
            share = paid_total * len(groups) / len(paid) if paid else 0.0
        reasons = collections.Counter(str(r.get("reject_reason")) for r in rejected)
        # Precommits refused for capacity (hotkey full, duplicate payload) carry
        # the hash of a group already generated: that work is lost too.
        conservation = ((archive.get("upload_precommit_conservation_by_environment") or {})
                        .get(environment) or {})
        refused_precommits = 0
        for reason, count in (conservation.get("capacity_rejections") or {}).items():
            if isinstance(count, int) and not isinstance(count, bool) and count > 0:
                reasons[str(reason)] += count
                refused_precommits += count
        ready_round = ready.get(environment)
        price = shadow.get(environment) or {}
        out[environment] = {
            "pool": pool,
            "slots": slots,
            "paid_groups": len(groups),
            "paid_share": share,
            "completion_tokens": completion,
            "prompt_tokens": prompt,
            # Uploaded groups (paid or rejected) and precommits refused for
            # capacity. A refusal is retried (the reference miner retries a
            # full batch up to 3 times), so refusals overcount lost groups:
            # the paid fraction is published as a range between the two.
            "submitted_groups": len(groups) + len(rejected),
            "refused_precommits": refused_precommits,
            "rejections": dict(reasons),
            "accept_seconds": (
                (int(ready_round) - open_round) * ROUND_SECONDS
                if isinstance(ready_round, int) and not isinstance(ready_round, bool) else None
            ),
            "price": price.get("price"),
            "regime": price.get("regime"),
        }
    return {"window": window, "open_round": open_round, "close_round": close_round,
            "next_open_round": None, "environments": out}


class RlPayBoard:
    """The last ``WINDOWS_KEPT`` windows and the answer built from them."""

    def __init__(self, *, chain_read: Callable[[], Awaitable[Mapping]] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self._stats: collections.deque[dict] = collections.deque(maxlen=WINDOWS_KEPT)
        self._chain_read = chain_read or read_subnet_pool
        self._clock = clock
        self._chain: dict | None = None
        self._chain_at: float | None = None
        self._chain_tried: float | None = None
        self._chain_failed = False
        self._cache: dict[Any, tuple[float, dict]] = {}
        self._chain_task: asyncio.Task | None = None

    def add(self, stat: dict | None) -> None:
        """Append a window; it gives its predecessor the cycle's end."""
        if stat is None:
            return
        if self._stats and self._stats[-1]["window"] == stat["window"] - 1:
            self._stats[-1]["next_open_round"] = stat["open_round"]
        self._stats.append(stat)

    def seed(self, stats: Iterable[dict | None]) -> None:
        """Merge archived windows with what is already held, oldest first."""
        merged = {s["window"]: s for s in stats if s}
        merged.update({s["window"]: s for s in self._stats})
        self._stats.clear()
        for window in sorted(merged)[-WINDOWS_KEPT:]:
            stat = merged[window]
            stat["next_open_round"] = None
            self.add(stat)
        self._cache.clear()

    async def refresh_chain(self) -> None:
        """One chain read; a failure keeps the last good value until it is stale."""
        self._chain_tried = self._clock()
        try:
            value = dict(await asyncio.wait_for(self._chain_read(), CHAIN_READ_BUDGET_SECONDS))
            if value.get("miner_alpha_per_block") is None:
                raise ValueError("chain read without miner_alpha_per_block")
            self._chain, self._chain_at, self._chain_failed = value, self._clock(), False
            self._cache.clear()
        except Exception as exc:  # noqa: BLE001 - a chain fault only nulls the alpha fields
            self._chain_failed = True
            logger.warning("rl pay: chain read failed: %r", exc)

    def _kick_chain(self) -> None:
        """Start a chain read BEHIND the request when one is due; never wait on it."""
        now = self._clock()
        wait = CHAIN_RETRY_SECONDS if self._chain_failed else CHAIN_REFRESH_SECONDS
        running = self._chain_task is not None and not self._chain_task.done()
        if running or (self._chain_tried is not None and now - self._chain_tried < wait):
            return
        self._chain_tried = now
        self._chain_task = asyncio.get_running_loop().create_task(self.refresh_chain())

    def _chain_value(self) -> dict | None:
        if self._chain is None or self._clock() - self._chain_at > CHAIN_STALE_SECONDS:
            return None
        return self._chain

    async def answer(self, tokens_per_second: float | None) -> dict:
        """The cached base answer, plus a per-request estimate when asked.

        Only the base is cached (one entry), so arbitrary speeds cannot grow
        memory; a failed rebuild serves the last good base.
        """
        if not self._stats:
            return {"status": "loading"}
        self._kick_chain()
        now = self._clock()
        cached = self._cache.get(None)
        if cached is None or now - cached[0] >= PAY_CACHE_SECONDS:
            try:
                cached = (now, self._build(self._chain_value()))
                self._cache[None] = cached
            except Exception:
                logger.exception("rl pay: rebuild failed; serving the last good answer")
                if cached is None:
                    raise
        base = cached[1]
        if tokens_per_second is None:
            return base
        return {**base, "tokens_per_second": tokens_per_second,
                "estimate": _estimate(base, float(tokens_per_second))}

    def _build(self, chain: dict | None) -> dict:
        stats = list(self._stats)
        cycled = [s for s in stats
                  if s["next_open_round"] is not None and s["next_open_round"] > s["open_round"]]
        mapb = None if chain is None else float(chain["miner_alpha_per_block"])

        def cycle_seconds(stat):
            return (stat["next_open_round"] - stat["open_round"]) * ROUND_SECONDS

        cycle_median = _median(cycle_seconds(s) for s in cycled)
        latest = stats[-1]
        environments: dict[str, dict] = {}
        names = list(latest["environments"])
        for environment in names:
            rows = [s for s in stats if environment in s["environments"]]
            cyc = [s for s in cycled if environment in s["environments"]]
            env_latest = latest["environments"][environment]

            def per_mtoken(stat):
                row = stat["environments"][environment]
                if mapb is None or row["completion_tokens"] <= 0:
                    return None
                e_cycle = mapb * cycle_seconds(stat) / BLOCK_SECONDS
                return row["paid_share"] * e_cycle / row["completion_tokens"] * 1e6

            per_group_tokens = [
                s["environments"][environment]["completion_tokens"] / s["environments"][environment]["paid_groups"]
                for s in rows if s["environments"][environment]["paid_groups"]
            ]
            per_group_prompt = [
                s["environments"][environment]["prompt_tokens"] / s["environments"][environment]["paid_groups"]
                for s in rows if s["environments"][environment]["paid_groups"]
            ]
            submitted = sum(s["environments"][environment]["submitted_groups"] for s in rows)
            refused = sum(s["environments"][environment].get("refused_precommits", 0) for s in rows)
            paid_groups = sum(s["environments"][environment]["paid_groups"] for s in rows)
            rejections: collections.Counter = collections.Counter()
            for s in rows:
                rejections.update(s["environments"][environment]["rejections"])
            market = {
                "alpha_per_mtoken_paid": _median(per_mtoken(s) for s in cyc),
                "paid_tokens_per_hour": _median(
                    s["environments"][environment]["completion_tokens"] / cycle_seconds(s) * 3600
                    for s in cyc),
                "alpha_per_hour": (
                    None if mapb is None else
                    _median(s["environments"][environment]["paid_share"] for s in cyc) * mapb * BLOCKS_PER_HOUR
                    if cyc else None),
                "accepting_fraction_of_cycle": _median(
                    min(1.0, s["environments"][environment]["accept_seconds"] / cycle_seconds(s))
                    for s in cyc if s["environments"][environment]["accept_seconds"] is not None),
                "paid_fraction_of_submitted": None if not submitted else {
                    # Every refused precommit a distinct lost group / none of them.
                    "low": paid_groups / (submitted + refused),
                    "high": paid_groups / submitted,
                },
                "rejections": dict(rejections),
                "fill_seconds_median": _median(
                    s["environments"][environment]["accept_seconds"] for s in rows),
            }
            pool_now = env_latest["pool"]
            if pool_now is None and env_latest["slots"] and env_latest["paid_groups"]:
                pool_now = env_latest["paid_share"] * env_latest["slots"] / env_latest["paid_groups"]
            tokens_per_group = _median(per_group_tokens)
            per_group_alpha = (
                None if mapb is None or cycle_median is None or pool_now is None or not env_latest["slots"]
                else pool_now / env_latest["slots"] * mapb * cycle_median / BLOCK_SECONDS
            )
            now_block = {
                "pool_share_per_window": pool_now,
                "alpha_per_paid_group": per_group_alpha,
                "alpha_per_mtoken_paid": (
                    None if per_group_alpha is None or not tokens_per_group
                    else per_group_alpha / tokens_per_group * 1e6),
                "completion_tokens_per_paid_group_median": tokens_per_group,
                "prompt_tokens_per_paid_group_median": _median(per_group_prompt),
            }
            environments[environment] = {
                "price": {"value": env_latest["price"], "regime": env_latest["regime"]},
                "now": now_block,
                "market": market,
                "history": [
                    {"window": s["window"], "alpha_per_mtoken_paid": per_mtoken(s),
                     "paid_groups": s["environments"][environment]["paid_groups"],
                     "cycle_seconds": cycle_seconds(s)}
                    for s in cyc[-HISTORY_WINDOWS:]
                ][::-1],
            }
        result = {
            "generated_at": self._clock(),
            "note": NOTE,
            "chain": None if chain is None else {
                "miner_alpha_per_block": chain.get("miner_alpha_per_block"),
                "alpha_price_tao": chain.get("alpha_price_tao"),
                "read_at": self._chain_at,
            },
            "cycle_seconds_median": cycle_median,
            "window_seconds_median": _median(
                (s["close_round"] - s["open_round"]) * ROUND_SECONDS for s in stats),
            "windows_counted": len(stats),
            "environments": environments,
        }
        return result


def _estimate(base: Mapping[str, Any], tokens_per_second: float) -> dict[str, dict]:
    """Per environment, the bounded alpha per GPU-hour at ``tokens_per_second``:
    paid tokens a miner alone could land (its speed x the accepting share of
    the cycle x the paid share, low/high) capped by the environment's demand."""
    out: dict[str, dict] = {}
    for name, entry in (base.get("environments") or {}).items():
        rate = entry["now"]["alpha_per_mtoken_paid"]
        market = entry["market"]
        accepting = market["accepting_fraction_of_cycle"]
        paid_fraction = market["paid_fraction_of_submitted"]
        demand = market["paid_tokens_per_hour"]
        if None in (rate, accepting, paid_fraction, demand):
            out[name] = {"alpha_per_gpu_hour": None}
            continue
        out[name] = {"alpha_per_gpu_hour": {
            bound: rate / 1e6 * min(tokens_per_second * 3600 * accepting * paid_fraction[bound], demand)
            for bound in ("low", "high")
        }}
    return out


async def read_subnet_pool() -> dict:
    """One chain read on a thread with its own event loop: importing
    bittensor and opening its websocket never hold the serving loop."""

    def run() -> dict:
        async def read() -> dict:
            from reliquary.infrastructure.chain import close_subtensor, get_subtensor, subnet_pool

            subtensor = await get_subtensor()
            try:
                return await subnet_pool(subtensor)
            finally:
                await close_subtensor(subtensor)

        return asyncio.run(read())

    return await asyncio.to_thread(run)
