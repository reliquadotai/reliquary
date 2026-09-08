"""Dependency-free accounting for the disabled fill qualification path.

Admission uses a monotonic per-environment budget. Completion requires the
configured pick ordinal for every environment, so a partially emitted
multi-environment event cannot close the shared window.
"""

from __future__ import annotations

import threading
from typing import Mapping


class FillState:
    def __init__(
        self, budgets: Mapping[str, int], picks_target: int
    ) -> None:
        if not budgets:
            raise ValueError("fill state requires at least one environment")
        if any(int(budget) <= 0 for budget in budgets.values()):
            raise ValueError("fill budgets must be positive")
        if int(picks_target) <= 0:
            raise ValueError("picks_target must be positive")
        self._budgets = {str(name): int(budget) for name, budget in budgets.items()}
        # Monotone: incremented only by ``reserve()``, never decremented by
        # ``release()``. This IS the budget -- a group that fails its proof
        # already spent real grading cost, so a failure frees proof
        # CAPACITY (``in_flight``) for the next candidate but does not
        # refund the budget that paid for the attempt.
        self._admitted = {name: 0 for name in self._budgets}
        self._proven = {name: 0 for name in self._budgets}
        self._in_flight = {name: 0 for name in self._budgets}
        self._picks_target = int(picks_target)
        # Per environment (R37), never a single window-wide counter: see
        # the module docstring for the half-event this exists to express.
        self._picks = {name: 0 for name in self._budgets}
        # Monotone cache generation for lock-free HTTP state-cache keys.  The
        # state endpoint takes ``lock`` only for the tiny snapshot copy; it does
        # not share the batcher's long-lived grading/proof lock.
        self._revision = 0
        # One window shares exactly one ``FillState`` across every
        # per-environment ``GrpoWindowBatcher`` instance (each batcher only
        # ever reserves/records on its own environment key, but
        # ``is_closed()`` is window-wide). The lock lives HERE, not on each
        # batcher, precisely because the instance is shared: two batchers
        # each taking their own lock around this same object would be no
        # lock at all. Methods below are deliberately lock-free -- callers
        # hold ``self.lock`` for the whole read-modify-write they need,
        # including reads paired with mutations of state that lives
        # elsewhere (a batcher's own ``_proven_groups``/pick counter),
        # which
        # is why this can't just be an internal per-method lock.
        self.lock = threading.Lock()

    def _known(self, environment: str) -> str:
        if environment not in self._budgets:
            raise ValueError(f"unknown environment {environment!r}")
        return environment

    def may_admit(self, environment: str) -> bool:
        name = self._known(environment)
        return self._admitted[name] < self._budgets[name]

    def reserve(self, environment: str) -> None:
        name = self._known(environment)
        self._in_flight[name] += 1
        self._admitted[name] += 1
        self._revision += 1

    def release(self, environment: str) -> None:
        name = self._known(environment)
        if self._in_flight[name] <= 0:
            raise ValueError(f"no reservation to release for {name!r}")
        self._in_flight[name] -= 1
        self._revision += 1
        # ``_admitted`` is untouched: see the budget comment above.

    def record_proven(self, environment: str) -> None:
        name = self._known(environment)
        if self._in_flight[name] > 0:
            self._in_flight[name] -= 1
        self._proven[name] += 1
        self._revision += 1

    def record_pick(self, environment: str) -> None:
        """Account this environment's share of one pick event (Component
        3 of the amendment).

        Two guards, both loud on purpose. Past ``picks_target`` is a
        caller bug: the window should already have sealed. And ordinals
        may differ by at most 1 -- exactly one event is ever in flight,
        so a wider spread means the caller drove one environment twice,
        or drove only one of them, which used to close a window at half
        its batches in silence. Validated BEFORE the counter moves, so a
        refused pick leaves the accounting exactly as it was.
        """
        name = self._known(environment)
        if self._picks[name] >= self._picks_target:
            raise ValueError(
                f"pick past picks_target for {name!r}"
            )
        prospective = dict(self._picks)
        prospective[name] += 1
        if max(prospective.values()) - min(prospective.values()) > 1:
            raise ValueError(
                "pick ordinals diverged past one in-flight event: "
                f"{prospective}"
            )
        self._picks[name] = prospective[name]
        self._revision += 1

    @property
    def revision(self) -> int:
        """Monotone mutation generation used only for state-cache expiry."""
        return self._revision

    def picks_taken(self, environment: str) -> int:
        """This environment's own pick ordinal -- what gates whether it
        may take another (a batcher's ``_claim_pick_chunk``), as opposed
        to whether the WINDOW is finished (``is_closed``)."""
        return self._picks[self._known(environment)]

    def is_closed(self) -> bool:
        return min(self._picks.values()) >= self._picks_target

    def snapshot(self) -> dict:
        return {
            "budgets": dict(self._budgets),
            "admitted": dict(self._admitted),
            "proven": dict(self._proven),
            "in_flight": dict(self._in_flight),
            "picks_by_environment": dict(self._picks),
            # Complete events only: the leader's half-taken event is not
            # one, so the service naming the next event to drive sees the
            # event still in flight rather than the one after it.
            "picks_emitted": min(self._picks.values()),
            "picks_target": self._picks_target,
            "closed": self.is_closed(),
        }
