"""The async prompt cooldown, sized from the run itself (decision E, ported to async).

A prompt scanned in zone or too hard cools down until checkpoint ``latest + cooldown`` (``LeaseBook.record_outcome``);
the cooldown counts checkpoints. ``AdaptiveCooldown`` is the book's ``cooldown_of``: per env

    cooldown = N x p / Q x margin, clamped to [floor, ceiling]

where ``N`` is the env's prompt range, ``p`` the in-zone share of its recent first scans and ``Q`` its in-zone groups per
checkpoint over the recent checkpoints. ``N x p / Q`` is how many checkpoints the env's in-zone prompts last at its
current consumption; a cooldown under it (``margin`` < 1) returns prompts before the env runs dry, so a too-hard prompt
is tried again once the policy has moved and an in-zone one is trained again in a later epoch. Until an env has
``min_first_scans`` first scans and in-zone groups in the window, it gets ``initial``. An operator override per env
(``params.rl_cooldown_k``) wins. Values are cached ``refresh_s`` seconds; ``snapshot`` feeds the controller metrics.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

logger = logging.getLogger(__name__)

COOLDOWN_PARAM = "rl_cooldown_k"


@dataclass(frozen=True)
class CooldownParams:
    margin: float = 0.5
    floor: int = 20
    ceiling: int = 1000
    initial: int = 100
    min_first_scans: int = 200
    first_scan_window: int = 2000            # recent first scans that measure p
    checkpoint_window: int = 50              # recent checkpoints that measure Q
    refresh_s: float = 60.0

    def __post_init__(self) -> None:
        if not 0.0 < self.margin <= 1.0:
            raise ValueError("margin must be in (0, 1]")
        if not 0 <= self.floor <= self.initial <= self.ceiling:
            raise ValueError("cooldown bounds must satisfy 0 <= floor <= initial <= ceiling")
        if min(self.min_first_scans, self.first_scan_window, self.checkpoint_window) < 1 or self.refresh_s < 0:
            raise ValueError("cooldown windows must be positive")


def cooldown_overrides(params: Mapping) -> dict[str, int]:
    """``params.rl_cooldown_k``: env -> cooldown in checkpoints (an operator's fixed value). ValueError when malformed."""
    raw = params.get(COOLDOWN_PARAM) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{COOLDOWN_PARAM} must map env names to checkpoints")
    out = {}
    for name, k in raw.items():
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError(f"{COOLDOWN_PARAM}[{name!r}] must be a non-negative integer of checkpoints")
        out[str(name)] = int(k)
    return out


class AdaptiveCooldown:
    """``LeaseBookConfig.cooldown_of``; ``attach(book)`` once the book exists (it reads the book's scan log)."""

    def __init__(self, *, prompt_range: Mapping[str, tuple[int, int]] | None = None,
                 overrides: Mapping[str, int] | None = None, params: CooldownParams | None = None,
                 excluded: Mapping[str, int] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.prompt_range = dict(prompt_range or {})
        self.excluded = dict(excluded or {})        # per env: rows of the range RL never draws (out of N)
        self.overrides = dict(overrides or {})
        self.params = params or CooldownParams()
        self.clock = clock
        self.book = None
        self._cache: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()

    def attach(self, book) -> "AdaptiveCooldown":
        self.book = book
        return self

    def __call__(self, environment: str) -> int:
        return int(self.state(environment)["cooldown"])

    def state(self, environment: str) -> dict:
        if environment in self.overrides:
            return {"cooldown": self.overrides[environment], "source": "override"}
        now = self.clock()
        with self._lock:
            cached = self._cache.get(environment)
            if cached is not None and now - cached[0] < self.params.refresh_s:
                return cached[1]
        value = self._measure(environment)
        with self._lock:
            self._cache[environment] = (now, value)
        return value

    def snapshot(self, environments) -> dict[str, float]:
        """Controller metrics: per env cooldown, p, Q and N."""
        out = {}
        for env in environments:
            s = self.state(env)
            for key in ("cooldown", "p", "q", "n"):
                if s.get(key) is not None:
                    out[f"rl/cooldown/{key}/{env}"] = float(s[key])
        return out

    def _measure(self, environment: str) -> dict:
        p = self.params
        span = self.prompt_range.get(environment)
        n = None if span is None else max(0, int(span[1]) - int(span[0]) - self.excluded.get(environment, 0))
        initial = {"cooldown": p.initial, "source": "initial", "n": n}
        if self.book is None or not n:
            return initial
        db, latest = self.book.db, self.book.latest_checkpoint
        firsts = db.execute("SELECT verdict FROM rl_prompt_scans WHERE environment=? AND first_scan=1 "
                            "ORDER BY seq DESC LIMIT ?", (environment, p.first_scan_window)).fetchall()
        if latest is None or len(firsts) < p.min_first_scans:
            return {**initial, "first_scans": len(firsts)}
        share = sum(1 for (verdict,) in firsts if verdict == "in_zone") / len(firsts)
        since = max(0, int(latest) - p.checkpoint_window)
        in_zone = db.execute("SELECT COUNT(*) FROM rl_prompt_scans WHERE environment=? AND verdict='in_zone' "
                             "AND k > ?", (environment, since)).fetchone()[0]
        q = in_zone / max(1, int(latest) - since)
        if share <= 0.0 or q <= 0.0:
            return {**initial, "p": share, "q": q}
        cooldown = round(n * share / q * p.margin)
        return {"cooldown": int(min(p.ceiling, max(p.floor, cooldown))), "source": "measured", "p": share, "q": q,
                "n": n}


__all__ = ["COOLDOWN_PARAM", "AdaptiveCooldown", "CooldownParams", "cooldown_overrides"]
