"""Per-environment fill accounting for the v6.1 window.

Pure and dependency-free: it counts, it decides, it touches no submission,
no model and no GPU. Admission and closing are governed by two, separate
rules --

    admit while   admitted < budget                (per environment)
    close when    every env's pick ordinal >= picks_target

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

R37: a pick EVENT is window-wide but it is taken one environment at a
time -- one event is one DAPO batch, built from every environment's own
k-th chunk -- so the ordinal counted here is PER ENVIRONMENT. A single
window-wide counter could not express the moment between the two halves
of an event: the first environment's Nth pick flipped the window closed
and locked its sibling out of the very event it was in the middle of,
tombstoning that half unpaid. ``picks_emitted`` is therefore the MIN over
environments (complete events only) and ``is_closed()`` asks every
environment, not the leader.
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
