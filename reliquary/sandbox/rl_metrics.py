"""Saturation metrics of the RL sandbox sessions and the operator's ALERTs. The sandbox fleet is fixed but grown by
hand, so the validator says when. Memory only, on the event loop.

* The grow-fleet ALERT counts only what more machines would fix: DISTINCT opens (one per engagement a minute, never
  the retries of one seed) refused for the fleet's capacity (``sandbox_saturated``, ``open_queue_full``: never a
  quota refusal, never the low-yield entity's hard cap), or the queue's p90 wait, and only while free sessions are
  below ``SATURATION_ALERT_FREE_FRACTION``, for ``SATURATION_ALERT_FOR_S``. Judged PER ENV (its machines, its
  refusals and waits) and PER IMAGE SET (the machines reporting one image lock of the env, the refusals of its
  images): one saturated env or image set alerts while the rest of the fleet idles.
* The check-slot ALERT: distinct groups refused ``episode_checker_busy`` above ``CHECKER_BUSY_ALERT_PER_MIN`` a
  minute for ``SATURATION_ALERT_FOR_S``; the check times' percentiles say how many slots the measured rate needs.
* The env ALERT: an episode env that no fresh machine serves any more (every machine lost mid-run) for two rounds
  in a row; ALERTed again after a recovery followed by a new loss.
* The capacity-verdict ALERT, per hotkey: over the last ``USEFUL_HORIZON_S`` (both counts over the same window),
  the sessions a capacity verdict made neutral reach ``CAPACITY_VERDICT_ALERT_SESSIONS`` and
  ``CAPACITY_VERDICT_ALERT_SHARE`` of its decided and neutral sessions: our capacity keeps failing that miner (grow
  the machines of its images), or it learnt to live on neutral verdicts. Re-armed once its share falls back."""
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
    CAPACITY_VERDICT_SAMPLE_S,
    CAPACITY_VERDICT_TOP,
    CHECKER_BUSY_ALERT_PER_MIN,
    SATURATION_ALERT_FOR_S,
    SATURATION_ALERT_FREE_FRACTION,
    SATURATION_ALERT_REFUSALS_PER_MIN,
    SATURATION_ALERT_WAIT_S,
    SATURATION_WINDOW_S,
    USEFUL_HORIZON_S,
)

logger = logging.getLogger(__name__)
FLEET_REASONS = frozenset({"sandbox_saturated", "open_queue_full"})
MAX_SAMPLES = 100_000          # each window's deques are bounded whatever the rate
ENV_LOST_ROUNDS = 2            # an env is lost after this many rounds in a row with no machine (one late heartbeat
                               # of a one-machine env is not a loss)


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
        self._waits: deque[tuple[float, float, str]] = deque(maxlen=MAX_SAMPLES)     # (at, seconds, env)
        # (at, reason, env, engagement, image)
        self._refusals: deque[tuple[float, str, str, str | None, str | None]] = deque(maxlen=MAX_SAMPLES)
        self._checks: deque[tuple[float, float]] = deque(maxlen=MAX_SAMPLES)         # (at, seconds)
        self._busy: deque[tuple[float, str]] = deque(maxlen=MAX_SAMPLES)             # (at, group)
        self._fleet = _Sustained()
        self._checker = _Sustained()
        self._saturated: dict[tuple, _Sustained] = {}     # (env,) or (env, image lock) -> its condition
        self._lost: set[str] = set()                      # episode envs no machine serves now
        self._unserved_rounds: dict[str, int] = {}        # env -> consecutive rounds no machine served it
        self._verdict_alerted: set[str] = set()           # hotkeys whose capacity-verdict ALERT is out
        self._started = float(clock())
        # hotkey -> (at, neutral, decided) samples of the ledger's cumulative counts (the verdict window)
        self._verdict_samples: dict[str, deque[tuple[float, int, int]]] = {}

    # the grow-fleet condition's state (one ALERT until it clears)
    @property
    def _alerted(self) -> bool:
        return self._fleet.alerted

    def _trim(self, now: float) -> None:
        for samples in (self._waits, self._refusals, self._checks, self._busy):
            while samples and samples[0][0] < now - SATURATION_WINDOW_S:
                samples.popleft()

    # --- what is recorded ---
    def record_wait(self, seconds: float, cls: Any = None) -> None:
        """The queue's listener: a ticket of placement class ``cls`` ((env, cpu, memory) for RL) waited."""
        env = cls[0] if isinstance(cls, tuple) and cls else ""
        self._waits.append((float(self._clock()), float(seconds), str(env or "")))

    def record_refusal(self, reason: str, env: str | None, engagement: str | None = None,
                       image: str | None = None) -> None:
        self._refusals.append((float(self._clock()), str(reason), str(env or ""), engagement, image))

    def open_refused(self, reason: str, env: str | None, engagement: str | None, image: str | None = None) -> None:
        """The issuer's listener method (``FanOutEvents.open_refused``)."""
        self.record_refusal(reason, env, engagement, image)

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
        waits = [seconds for _at, seconds, _env in self._waits]
        refused: dict[str, dict[str, int]] = {}
        for _at, reason, env, _engagement, _image in self._refusals:
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
                "envs_lost": sorted(self._lost),
                "saturated": sorted("/".join(key) for key, cond in self._saturated.items() if cond.alerted)}
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

    def _rows(self, ledger: Any, now: float | None = None) -> list[tuple[str, int, int, int, float]]:
        """(hotkey, verdicts since start, neutral sessions, decided sessions, neutral share) per hotkey with a
        neutral session, the two session counts over the SAME window: the last ``USEFUL_HORIZON_S`` (the ledger's
        counts since start, less their sample from the window's start; zero before the first sample)."""
        now = float(self._clock()) if now is None else float(now)
        verdicts = ledger.capacity_counts()
        counts = ledger.session_counts()
        rows = []
        for hotkey, (neutral, decided) in counts.items():
            samples = self._verdict_samples.setdefault(hotkey, deque([(self._started, 0, 0)]))
            if now - samples[-1][0] >= CAPACITY_VERDICT_SAMPLE_S:
                samples.append((now, int(neutral), int(decided)))
            while len(samples) > 1 and samples[1][0] <= now - USEFUL_HORIZON_S:
                samples.popleft()
            base = samples[0]                     # the last sample at or before the window's start
            window_neutral, window_decided = int(neutral) - base[1], int(decided) - base[2]
            if window_neutral <= 0:
                continue
            share = window_neutral / (window_neutral + window_decided)
            rows.append((hotkey, int((verdicts.get(hotkey) or (0, 0))[0]), window_neutral, window_decided, share))
        return rows

    def _verdicts_view(self, ledger: Any) -> list[dict]:
        rows = sorted(self._rows(ledger), key=lambda row: (-row[2], row[0]))[:CAPACITY_VERDICT_TOP]
        return [{"hotkey": hotkey, "verdicts": verdicts, "neutral_sessions": neutral, "decided": decided,
                 "neutral_share": round(share, 3)} for hotkey, verdicts, neutral, decided, share in rows]

    # --- the ALERTs ---
    def fleet_refusals(self, now: float) -> int:
        """Distinct opens refused for the fleet's capacity in the last minute (one per engagement)."""
        return len({(engagement or f"anon:{at}") for at, reason, _env, engagement, _image in self._refusals
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
        recent = [seconds for at, seconds, _env in self._waits if at >= now - 60]
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

    def check_saturated(self, envs: Mapping[str, Mapping[str, Any]], *,
                        images_of: Any = None, now: float | None = None) -> list[str]:
        """The grow-fleet ALERTs per env and per image set. ``envs``: ``Fleet.capacity_by_env(reservations=)``
        ({env: {"slots", "free", "image_sets": {lock: {"slots", "free"}}}}); ``images_of(env, lock)``: the images
        of an image set (``Fleet.image_set_images``), to tell which refused opens were its. An env or image set is
        saturated while its free sessions are under ``SATURATION_ALERT_FREE_FRACTION`` and its distinct refused
        opens a minute exceed ``SATURATION_ALERT_REFUSALS_PER_MIN`` (or, for an env, its queue's p90 wait exceeds
        ``SATURATION_ALERT_WAIT_S``), for ``SATURATION_ALERT_FOR_S``."""
        now = float(self._clock()) if now is None else float(now)
        self._trim(now)
        refused: dict[str, dict[str, str | None]] = {}           # env -> {engagement: image}, the last minute
        for at, reason, env, engagement, image in self._refusals:
            if at >= now - 60 and reason in FLEET_REASONS:
                refused.setdefault(env, {})[engagement or f"anon:{at}"] = image
        waits: dict[str, list[float]] = {}
        for at, seconds, env in self._waits:
            if at >= now - 60:
                waits.setdefault(env, []).append(seconds)
        alerts, seen = [], set()

        def full(row: Mapping[str, Any]) -> bool:
            slots = int(row.get("slots") or 0)
            return slots > 0 and int(row.get("free") or 0) / slots < SATURATION_ALERT_FREE_FRACTION

        for env, row in sorted(envs.items()):
            mine = refused.get(env, {})
            p90 = _percentile(waits.get(env, []), 0.9)
            key = (env,)
            seen.add(key)
            above = full(row) and (len(mine) > SATURATION_ALERT_REFUSALS_PER_MIN
                                   or (p90 is not None and p90 > SATURATION_ALERT_WAIT_S))
            condition = self._saturated.setdefault(key, _Sustained())
            if condition.step(above, now):
                text = (f"ALERT rl sandbox env {env} saturated for {now - condition.since:.0f} s: {len(mine)} distinct "
                        f"opens refused for capacity in the last minute, queue p90 wait "
                        f"{p90 if p90 is not None else 0:.1f} s, {row.get('free')} of {row.get('slots')} sessions free: "
                        f"grow the machines serving {env}")
                logger.error(text)
                alerts.append(text)
            for lock, image_set in sorted((row.get("image_sets") or {}).items()):
                key = (env, lock)
                seen.add(key)
                above = False
                if full(image_set) and images_of is not None and mine:
                    images = images_of(env, lock)
                    count = sum(1 for image in mine.values() if image is not None and image in images)
                    above = count > SATURATION_ALERT_REFUSALS_PER_MIN
                condition = self._saturated.setdefault(key, _Sustained())
                if condition.step(above, now):
                    text = (f"ALERT rl sandbox image set {lock[:12]} of env {env} saturated for "
                            f"{now - condition.since:.0f} s: {image_set.get('free')} of {image_set.get('slots')} "
                            f"sessions free and its opens refused for capacity: grow the machines holding that image "
                            f"set (or pull its images on more machines)")
                    logger.error(text)
                    alerts.append(text)
        for key in [key for key in self._saturated if key not in seen]:
            del self._saturated[key]
        return alerts

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
            self._unserved_rounds[env] = 0 if ok else self._unserved_rounds.get(env, 0) + 1
            if ok and env in self._lost:
                self._lost.discard(env)
                recovered.append(env)
                logger.warning("episode environment %s is served by a machine again", env)
            elif not ok and env not in self._lost and self._unserved_rounds[env] >= ENV_LOST_ROUNDS:
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
        rows = self._rows(ledger)
        self._verdict_alerted &= {row[0] for row in rows}      # no neutral session in the window: re-armed
        for hotkey, verdicts, neutral, decided, share in rows:
            above = neutral >= CAPACITY_VERDICT_ALERT_SESSIONS and share >= CAPACITY_VERDICT_ALERT_SHARE
            if not above:
                self._verdict_alerted.discard(hotkey)
                continue
            if hotkey in self._verdict_alerted:
                continue
            self._verdict_alerted.add(hotkey)
            text = (f"ALERT hotkey {hotkey}: {neutral} of its sessions were made neutral by capacity verdicts in the "
                    f"last 24 h ({share:.0%} of its decided and neutral sessions; {verdicts} verdicts since start): "
                    f"our capacity keeps "
                    f"refusing it (grow the machines of its tasks' images) or it lives on neutral verdicts")
            logger.error(text)
            alerts.append(text)
        return alerts


__all__ = ["FLEET_REASONS", "SaturationMonitor"]
