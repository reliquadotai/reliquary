"""Per-environment fill accounting for the v6.1 window.

Pure and dependency-free: it counts, it decides, it touches no submission,
no model and no GPU. Admission and closing are governed by two, separate
rules --

    admit while   admitted < budget                (per environment)
    close when    picks_emitted >= picks_target     (window-wide)

which are deliberately decoupled (R33, R35, amendment v6.1). Admission no
longer gates on a per-environment "target" that also closes the window --
it gates on a budget, and over-collection against that budget is
deliberate: a pick only ever chooses among proven-and-unemitted groups
that already exist, so late, longer-generating groups have to be sitting
in the pool *before* a late pick can choose them. Sizing admission to the
window's close condition, as the old target did, would starve every pick
after the first of anything but what had already proven by the time
admission itself closed -- exactly the short bias this amendment removes.
Closing, in turn, is no longer a per-environment proven count at all: a
*pick* (Component 3 of the amendment) selects the best-by-rate proven
groups across the pool, and the window closes when the Nth pick has been
emitted, regardless of how much proven surplus is left in any
environment's pool. A proven group never picked by close is burned (R32):
logged, archived with a count, never redistributed and never paid.
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
        self._picks_emitted = 0
        # One window shares exactly one ``FillState`` across every
        # per-environment ``GrpoWindowBatcher`` instance (each batcher only
        # ever reserves/records on its own environment key, but
        # ``is_closed()`` is window-wide). The lock lives HERE, not on each
        # batcher, precisely because the instance is shared: two batchers
        # each taking their own lock around this same object would be no
        # lock at all. Methods below are deliberately lock-free -- callers
        # hold ``self.lock`` for the whole read-modify-write they need,
        # including reads paired with mutations of state that lives
        # elsewhere (a batcher's own ``_proven_groups``/watermark), which
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

    def release(self, environment: str) -> None:
        name = self._known(environment)
        if self._in_flight[name] <= 0:
            raise ValueError(f"no reservation to release for {name!r}")
        self._in_flight[name] -= 1
        # ``_admitted`` is untouched: see the budget comment above.

    def record_proven(self, environment: str) -> None:
        name = self._known(environment)
        if self._in_flight[name] > 0:
            self._in_flight[name] -= 1
        self._proven[name] += 1

    def record_pick(self) -> None:
        """Account one pick (Component 3 of the amendment). Window-wide,
        not per-environment: a pick selects across every environment's
        pool at once."""
        if self._picks_emitted >= self._picks_target:
            raise ValueError("pick past picks_target")
        self._picks_emitted += 1

    def is_closed(self) -> bool:
        return self._picks_emitted >= self._picks_target

    def snapshot(self) -> dict:
        return {
            "budgets": dict(self._budgets),
            "admitted": dict(self._admitted),
            "proven": dict(self._proven),
            "in_flight": dict(self._in_flight),
            "picks_emitted": self._picks_emitted,
            "picks_target": self._picks_target,
            "closed": self.is_closed(),
        }
