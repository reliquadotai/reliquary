"""Each env's share of a batch's groups at the target mix, and the per-env group limits drawn from it.

The targets are shares of trained tokens; the queue and the batch count groups, and an env's groups differ by ~80x
in tokens (instruction following ~6k, hard math ~530k on teutonic-rl-async-v1). At the target mix env e is
``target(e) / tokens_per_group(e)`` of the groups, normalised (instruction following: ~39 % of a batch for 1 % of
its tokens). The tokens per group are the mean of the env's last in-zone groups, re-read every ``refresh_s``.

* ``lane_max``: the most groups an env may hold in the queue (queued or taken), ``lane_batches`` batches of its
  share; past it its leases are refused (``LeaseBook``, refusal ``lane_full``) and its price weight descends. The
  other envs are untouched.
* ``batch_caps``: the most groups of an env one take hands the trainer, ``batch_factor`` x its share of the take
  (``TrainingBuffer.take``): a cheap env cannot fill the batch while the others are short
  (2026-10-24: 391 instruction-following groups of 512, a 35 M-token step instead of ~120 M).
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping

DEFAULT_GROUP_TOKENS = 100_000.0
RECENT_GROUPS = 200


def group_shares(db, targets: Mapping[str, float], *, recent: int = RECENT_GROUPS,
                 default_tokens: float = DEFAULT_GROUP_TOKENS) -> tuple[dict[str, float], dict[str, float]]:
    """(env -> share of the groups at the target mix, env -> mean completion tokens of its recent in-zone groups)."""
    tokens: dict[str, float] = {}
    has_groups = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='rl_groups'").fetchone()
    for env in sorted(targets):
        rows = [] if not has_groups else db.execute(
            "SELECT completion_tokens FROM rl_groups WHERE environment=? AND verdict='in_zone' "
            "AND completion_tokens>0 ORDER BY seq DESC LIMIT ?", (env, int(recent))).fetchall()
        tokens[env] = sum(r[0] for r in rows) / len(rows) if rows else float(default_tokens)
    weights = {env: float(share) / tokens[env] for env, share in targets.items() if float(share) > 0}
    total = math.fsum(weights.values())
    return ({env: w / total for env, w in weights.items()} if total > 0 else {}), tokens


class EnvGroupCaps:
    """Cached ``group_shares`` of ``book``'s file and the limits drawn from it (module doc). ``lane_batches`` or
    ``batch_factor`` None: that limit is off."""

    def __init__(self, book, targets: Mapping[str, float], *, batch_groups: int, lane_batches: float | None = None,
                 batch_factor: float | None = None, refresh_s: float = 300.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if batch_groups < 1 or (lane_batches is not None and lane_batches <= 0) \
                or (batch_factor is not None and batch_factor < 1):
            raise ValueError("batch_groups >= 1, lane_batches > 0, batch_factor >= 1")
        self.book, self.targets = book, dict(targets)
        self.batch_groups, self.lane_batches, self.batch_factor = int(batch_groups), lane_batches, batch_factor
        self.refresh_s, self.clock = float(refresh_s), clock
        self._cached: tuple[float, dict[str, float], dict[str, float]] | None = None

    def shares(self) -> dict[str, float]:
        return self._read()[0]

    def group_tokens(self) -> dict[str, float]:
        return self._read()[1]

    def _read(self) -> tuple[dict[str, float], dict[str, float]]:
        now = float(self.clock())
        if self._cached is None or now - self._cached[0] >= self.refresh_s:
            with self.book.lock:
                shares, tokens = group_shares(self.book.db, self.targets)
            self._cached = (now, shares, tokens)
        return self._cached[1], self._cached[2]

    def lane_max(self) -> dict[str, int]:
        if self.lane_batches is None:
            return {}
        return {env: max(1, math.ceil(self.lane_batches * self.batch_groups * share))
                for env, share in self.shares().items()}

    def batch_caps(self, limit: int) -> dict[str, int]:
        if self.batch_factor is None:
            return {}
        return {env: max(1, math.ceil(self.batch_factor * int(limit) * share)) for env, share in self.shares().items()}


__all__ = ["EnvGroupCaps", "group_shares"]
