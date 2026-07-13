"""Bounded cross-window accumulation for balanced multi-environment training."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


class BalancedTrainingAccumulator:
    """Collect one checkpoint-consistent target batch per environment.

    Sparse environments may need several windows to reach their target. Fast
    environments stop contributing once their target is full, which preserves
    the configured environment weighting and bounds retained memory.
    """

    def __init__(self, targets: Mapping[str, int]) -> None:
        if not targets:
            raise ValueError("training accumulator requires at least one environment")
        if any(int(target) < 0 for target in targets.values()):
            raise ValueError("training accumulator targets must be non-negative")
        self.targets = {str(name): int(target) for name, target in targets.items()}
        self._groups: dict[str, list[Any]] = {name: [] for name in self.targets}
        self._source_windows: dict[str, list[int]] = {
            name: [] for name in self.targets
        }
        self.checkpoint_revision: str | None = None

    @property
    def ready(self) -> bool:
        return all(
            len(self._groups[name]) >= target
            for name, target in self.targets.items()
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "checkpoint_revision": self.checkpoint_revision,
            "targets": dict(self.targets),
            "counts": {
                name: len(self._groups[name]) for name in self.targets
            },
            "source_windows": {
                name: list(self._source_windows[name]) for name in self.targets
            },
            "ready": self.ready,
        }

    def reset(self) -> dict[str, Any]:
        previous = self.snapshot()
        for groups in self._groups.values():
            groups.clear()
        for windows in self._source_windows.values():
            windows.clear()
        self.checkpoint_revision = None
        return previous

    def add_window(
        self,
        batches: Mapping[str, Sequence[Any]],
        *,
        window_n: int,
        checkpoint_revision: str,
    ) -> dict[str, Any]:
        """Add as much of one clean window as each environment still needs."""
        unknown = set(batches) - set(self.targets)
        if unknown:
            raise ValueError(f"unknown training environments: {sorted(unknown)}")

        reset_snapshot = None
        if (
            self.checkpoint_revision is not None
            and self.checkpoint_revision != checkpoint_revision
        ):
            reset_snapshot = self.reset()
        self.checkpoint_revision = checkpoint_revision

        counts_before = {
            name: len(self._groups[name]) for name in self.targets
        }
        added: dict[str, int] = {}
        not_accumulated: dict[str, int] = {}
        for name, target in self.targets.items():
            incoming = list(batches.get(name, ()))
            capacity = max(0, target - len(self._groups[name]))
            accepted = incoming[:capacity]
            if accepted:
                self._groups[name].extend(accepted)
                if not self._source_windows[name] or (
                    self._source_windows[name][-1] != window_n
                ):
                    self._source_windows[name].append(window_n)
            added[name] = len(accepted)
            not_accumulated[name] = len(incoming) - len(accepted)

        return {
            "checkpoint_reset": reset_snapshot,
            "counts_before": counts_before,
            "added": added,
            "not_accumulated": not_accumulated,
            "snapshot": self.snapshot(),
        }

    def training_batches(self, env_order: Sequence[str]) -> list[list[Any]]:
        if not self.ready:
            raise RuntimeError("balanced training accumulator is not ready")
        if set(env_order) != set(self.targets) or len(env_order) != len(self.targets):
            raise ValueError("training environment order does not match accumulator")
        return [
            list(self._groups[name][: self.targets[name]])
            for name in env_order
        ]
