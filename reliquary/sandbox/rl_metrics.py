"""Saturation metrics of the RL sandbox sessions and the operator's ALERTs. The sandbox fleet is fixed but grown by
hand, so the validator says when. Memory only, on the event loop.

* The grow-fleet ALERT counts only what more machines would fix: DISTINCT opens (one per engagement a minute, never
  the retries of one seed) refused for the fleet's capacity (``sandbox_saturated``, ``open_queue_full``: never a
  quota refusal, never the low-yield entity's hard cap), or the queue's p90 wait, and only while the fleet's free
  sessions are below ``SATURATION_ALERT_FREE_FRACTION`` of its sessions, for ``SATURATION_ALERT_FOR_S``.
* The check-slot ALERT: distinct groups refused ``episode_checker_busy`` above ``CHECKER_BUSY_ALERT_PER_MIN`` a
  minute for ``SATURATION_ALERT_FOR_S``; the check times' percentiles say how many slots the measured rate needs.
* The env ALERT: an episode env that no fresh machine serves any more (every machine lost mid-run); ALERTed again
  after a recovery followed by a new loss.
* The capacity-verdict ALERT, per hotkey: the sessions a capacity verdict made neutral (since this process
  started) reach ``CAPACITY_VERDICT_ALERT_SESSIONS`` and ``CAPACITY_VERDICT_ALERT_SHARE`` of its decided and
  neutral sessions: our capacity keeps failing that miner (grow the machines of its images), or it learnt to live
  on neutral verdicts."""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any

from reliquary.sandbox.rl_capacity import (
    CAPACITY_VERDICT_ALERT_SESSIONS,
    CAPACITY_VERDICT_ALERT_SHARE,
    CAPACITY_VERDICT_TOP,
    CHECKER_BUSY_ALERT_PER_MIN,
    SATURATION_ALERT_FOR_S,
    SATURATION_ALERT_FREE_FRACTION,
    SATURATION_ALERT_REFUSALS_PER_MIN,
    SATURATION_ALERT_WAIT_S,
    SATURATION_WINDOW_S,
)

logger = logging.getLogger(__name__)
FLEET_REASONS = frozenset({"sandbox_saturated", "open_queue_full"})
MAX_SAMPLES = 100_000          # each window's deques are bounded whatever the rate


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


class _Sustained:
    """A condition that must hold ``for_s`` before its ALERT; one ALERT until it clears."""

    def __init__(self) -> None:
        self.since: float | None = None
        self.alerted = False

    def step(self, above: bool, now: float) -> bool:
        if not above:
            self.since, self.alerted = None, False
            return False
        if self.since is None:
            self.since = now
        if self.alerted or now - self.since < SATURATION_ALERT_FOR_S:
            return False
        self.alerted = True
        return True


class SaturationMonitor:
    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._waits: deque[tuple[float, float]] = deque(maxlen=MAX_SAMPLES)          # (at, seconds)
        self._refusals: deque[tuple[float, str, str, str | None]] = deque(maxlen=MAX_SAMPLES)
        self._checks: deque[tuple[float, float]] = deque(maxlen=MAX_SAMPLES)         # (at, seconds)
        self._busy: deque[tuple[float, str]] = deque(maxlen=MAX_SAMPLES)             # (at, group)
        self._fleet = _Sustained()
        self._checker = _Sustained()
        self._lost: set[str] = set()                      # episode envs no machine serves now
        self._verdict_alerted: set[str] = set()           # hotkeys whose capacity-verdict ALERT is out

    # the grow-fleet condition's state (one ALERT until it clears)
    @property
    def _alerted(self) -> bool:
        return self._fleet.alerted

    def _trim(self, now: float) -> None:
        for samples in (self._waits, self._refusals, self._checks, self._busy):
            while samples and samples[0][0] < now - SATURATION_WINDOW_S:
                samples.popleft()

    # --- what is recorded ---
    def record_wait(self, seconds: float) -> None:
        self._waits.append((float(self._clock()), float(seconds)))

    def record_refusal(self, reason: str, env: str | None, engagement: str | None = None) -> None:
        self._refusals.append((float(self._clock()), str(reason), str(env or ""), engagement))

    def open_refused(self, reason: str, env: str | None, engagement: str | None) -> None:
        """The issuer's listener method (``FanOutEvents.open_refused``)."""
        self.record_refusal(reason, env, engagement)

    def opened(self, record: Any) -> None:            # a session listener hears every event; only refusals count
        pass

    def closed(self, record: Any, final: Any) -> None:
        pass

    def check_time(self, seconds: float) -> None:
        """The intake's listener: one transcript check took ``seconds``."""
        self._checks.append((float(self._clock()), float(seconds)))

    def checker_busy(self, hotkey: str, precommit_sha256: str | None) -> None:
        """The intake's listener: a group was refused for want of a check slot."""
        self._busy.append((float(self._clock()), f"{hotkey}:{precommit_sha256}"))

    # --- the operator's view ---
    def snapshot(self, *, queue: Any = None, quota: Any = None, fleet: Any = None, reservations: Any = None,
                 ledger: Any = None, issuer: Any = None) -> dict:
        now = float(self._clock())
        self._trim(now)
        waits = [seconds for _at, seconds in self._waits]
        refused: dict[str, dict[str, int]] = {}
        for _at, reason, env, _engagement in self._refusals:
            refused.setdefault(env, {}).setdefault(reason, 0)
            refused[env][reason] += 1
        checks = [seconds for _at, seconds in self._checks]
        snap = {"window_s": SATURATION_WINDOW_S,
                "queue": {"waiting": getattr(queue, "waiting", None), "waits": len(waits),
                          "wait_p50": _percentile(waits, 0.5), "wait_p90": _percentile(waits, 0.9),
                          "wait_p99": _percentile(waits, 0.99)},
                "refused": refused,
                "used": None if quota is None else quota.usage(int(now)),
                "fleet": None if fleet is None else self._fleet_view(fleet, now, reservations),
                "checks": {"count": len(checks), "p50_s": _percentile(checks, 0.5),
                           "p90_s": _percentile(checks, 0.9), "p99_s": _percentile(checks, 0.99),
                           "checker_busy": len({group for _at, group in self._busy}),
                           "checker_busy_last_minute": self.checker_busy_groups(now)},
                "envs_lost": sorted(self._lost)}
        if ledger is not None:
            snap["capacity_verdicts"] = self._verdicts_view(ledger)
        if issuer is not None:
            snap["attribution_failures"] = int(getattr(issuer, "attribution_failures", 0))
        return snap

    @staticmethod
    def _fleet_view(fleet: Any, now: float, reservations: Any) -> dict:
        reserved = reservations() if callable(reservations) else reservations
        if reserved is None:
            return fleet.capacity_by_env(now)
        return fleet.capacity_by_env(now, reservations=reserved)

    @staticmethod
    def _rows(ledger: Any) -> list[tuple[str, int, int, int, float]]:
        """(hotkey, verdicts, neutral sessions, decided sessions, neutral share) per hotkey with a verdict."""
        counts = ledger.capacity_counts()
        snapshot = ledger.snapshot()
        rows = []
        for hotkey, (verdicts, neutral) in counts.items():
            decided = int(getattr(snapshot.get(hotkey), "decided", 0) or 0)
            share = neutral / (neutral + decided) if neutral + decided else 0.0
            rows.append((hotkey, int(verdicts), int(neutral), decided, share))
        return rows

    def _verdicts_view(self, ledger: Any) -> list[dict]:
        rows = sorted(self._rows(ledger), key=lambda row: (-row[2], row[0]))[:CAPACITY_VERDICT_TOP]
        return [{"hotkey": hotkey, "verdicts": verdicts, "neutral_sessions": neutral, "decided": decided,
                 "neutral_share": round(share, 3)} for hotkey, verdicts, neutral, decided, share in rows]

    # --- the ALERTs ---
    def fleet_refusals(self, now: float) -> int:
        """Distinct opens refused for the fleet's capacity in the last minute (one per engagement)."""
        return len({(engagement or f"anon:{at}") for at, reason, _env, engagement in self._refusals
                    if at >= now - 60 and reason in FLEET_REASONS})

    def checker_busy_groups(self, now: float) -> int:
        """Distinct groups refused for want of a check slot in the last minute."""
        return len({group for at, group in self._busy if at >= now - 60})

    def check(self, now: float | None = None, *, free_fraction: float | None = None) -> str | None:
        """The grow-fleet ALERT, the first time the queue's p90 wait or the distinct fleet refusals of the last
        minute have stayed above their thresholds for SATURATION_ALERT_FOR_S while the fleet had less than
        SATURATION_ALERT_FREE_FRACTION of its sessions free; re-armed once the condition clears."""
        now = float(self._clock()) if now is None else float(now)
        self._trim(now)
        recent = [seconds for at, seconds in self._waits if at >= now - 60]
        per_minute = self.fleet_refusals(now)
        p90 = _percentile(recent, 0.9)
        full = free_fraction is not None and free_fraction < SATURATION_ALERT_FREE_FRACTION
        above = full and ((p90 is not None and p90 > SATURATION_ALERT_WAIT_S)
                          or per_minute > SATURATION_ALERT_REFUSALS_PER_MIN)
        if not self._fleet.step(above, now):
            return None
        text = (f"ALERT rl sandbox fleet saturated for {now - self._fleet.since:.0f} s: queue p90 wait "
                f"{p90 if p90 is not None else 0:.1f} s, {per_minute} distinct opens refused for capacity in the last "
                f"minute, {free_fraction:.0%} of the sessions free: grow the sandbox fleet (see the saturation "
                f"snapshot)")
        logger.error(text)
        return text

    def check_checker(self, now: float | None = None, *, running: int | None = None) -> str | None:
        """The check-slot ALERT: more than CHECKER_BUSY_ALERT_PER_MIN distinct groups a minute refused
        ``episode_checker_busy`` for SATURATION_ALERT_FOR_S. ``running``: the intake's check slots."""
        now = float(self._clock()) if now is None else float(now)
        self._trim(now)
        busy = self.checker_busy_groups(now)
        if not self._checker.step(busy > CHECKER_BUSY_ALERT_PER_MIN, now):
            return None
        times = [seconds for _at, seconds in self._checks]
        p90 = _percentile(times, 0.9)
        rate = (sum(1 for at, _s in self._checks if at >= now - 60) + busy) / 60.0       # groups a second asked
        needed = None if p90 is None else math.ceil(rate * p90) + 1
        text = (f"ALERT rl episode checks saturated for {now - self._checker.since:.0f} s: {busy} distinct groups "
                f"refused episode_checker_busy in the last minute with {running} check slots; check time p90 "
                f"{p90 if p90 is not None else 0:.2f} s at {rate:.2f} groups/s: raise MAX_CHECKS_RUNNING"
                + (f" to at least {needed}" if needed is not None else ""))
        logger.error(text)
        return text

    def check_envs(self, served: Mapping[str, bool], *, known_lost: set[str] | None = None) -> tuple[list[str], list[str]]:
        """``served``: {episode env: whether a fresh machine serves it now}. Returns (the ALERTs of envs that lost
        every machine since the last call, the envs served again). ``known_lost``: envs already ALERTed
        elsewhere (the boot's degraded start), taken as lost without a second ALERT."""
        if known_lost:
            self._lost |= set(known_lost) & {env for env, ok in served.items() if not ok}
        alerts, recovered = [], []
        for env, ok in sorted(served.items()):
            if ok and env in self._lost:
                self._lost.discard(env)
                recovered.append(env)
                logger.warning("episode environment %s is served by a machine again", env)
            elif not ok and env not in self._lost:
                self._lost.add(env)
                text = (f"ALERT episode environment {env} lost every sandbox machine: no fresh machine serves it at "
                        f"its pinned package and options; its opens are refused (environment_unserved, retryable) "
                        f"until one does")
                logger.error(text)
                alerts.append(text)
        return alerts, recovered

    def check_capacity_verdicts(self, ledger: Any) -> list[str]:
        """The per-hotkey capacity-verdict ALERTs (once per hotkey until its share falls back)."""
        alerts = []
        for hotkey, verdicts, neutral, decided, share in self._rows(ledger):
            above = neutral >= CAPACITY_VERDICT_ALERT_SESSIONS and share >= CAPACITY_VERDICT_ALERT_SHARE
            if not above:
                self._verdict_alerted.discard(hotkey)
                continue
            if hotkey in self._verdict_alerted:
                continue
            self._verdict_alerted.add(hotkey)
            text = (f"ALERT hotkey {hotkey}: {neutral} of its sessions were made neutral by {verdicts} capacity "
                    f"verdicts since start ({share:.0%} of its decided and neutral sessions): our capacity keeps "
                    f"refusing it (grow the machines of its tasks' images) or it lives on neutral verdicts")
            logger.error(text)
            alerts.append(text)
        return alerts


__all__ = ["FLEET_REASONS", "SaturationMonitor"]
