"""Per-environment fill accounting for the v6 window.

Pure and dependency-free: it counts, it decides, it touches no submission,
no model and no GPU. The window's whole control loop is these two rules —

    admit while   proven + in_flight < target
    close when    proven >= target, for every environment

which are deliberately asymmetric. Gating admission on ``proven`` alone
would over-admit by the entire proof pipeline depth, because every
reservation still in flight would look like room. Closing on
``proven + in_flight`` would close on work that may still fail GRAIL, and a
group that fails its proof is not a group.
"""

from __future__ import annotations

from typing import Mapping


class FillState:
    def __init__(self, targets: Mapping[str, int]) -> None:
        if not targets:
            raise ValueError("fill state requires at least one environment")
        if any(int(target) <= 0 for target in targets.values()):
            raise ValueError("fill targets must be positive")
        self._targets = {str(name): int(target) for name, target in targets.items()}
        self._proven = {name: 0 for name in self._targets}
        self._in_flight = {name: 0 for name in self._targets}

    def _known(self, environment: str) -> str:
        if environment not in self._targets:
            raise ValueError(f"unknown environment {environment!r}")
        return environment

    def may_admit(self, environment: str) -> bool:
        name = self._known(environment)
        committed = self._proven[name] + self._in_flight[name]
        return committed < self._targets[name]

    def reserve(self, environment: str) -> None:
        self._in_flight[self._known(environment)] += 1

    def release(self, environment: str) -> None:
        name = self._known(environment)
        if self._in_flight[name] <= 0:
            raise ValueError(f"no reservation to release for {name!r}")
        self._in_flight[name] -= 1

    def record_proven(self, environment: str) -> None:
        name = self._known(environment)
        if self._in_flight[name] > 0:
            self._in_flight[name] -= 1
        self._proven[name] += 1

    def is_closed(self) -> bool:
        return all(
            self._proven[name] >= target
            for name, target in self._targets.items()
        )

    def snapshot(self) -> dict:
        return {
            "targets": dict(self._targets),
            "proven": dict(self._proven),
            "in_flight": dict(self._in_flight),
            "closed": self.is_closed(),
        }
