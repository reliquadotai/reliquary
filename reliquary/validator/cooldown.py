"""CooldownMap: tracks last rewarded-use window per prompt_idx.

A prompt that just earned emission is ineligible for the batch
for ``cooldown_windows`` following windows. This forces the curriculum
to rotate so the policy has time to shift between reuses of the same
prompt.
"""

from __future__ import annotations

import re


_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class CooldownMap:
    """Per-prompt "last rewarded at window N" store + eligibility predicate.

    The cooldown window is a half-open interval:
        ``[last_batched, last_batched + cooldown_windows)`` → ineligible.
    At ``current_window == last_batched + cooldown_windows`` the prompt
    becomes eligible again.
    """

    def __init__(self, cooldown_windows: int) -> None:
        if cooldown_windows < 0:
            raise ValueError("cooldown_windows must be non-negative")
        self._cooldown_windows = cooldown_windows
        self._last_batched: dict[int, int] = {}

    def record_batched(self, prompt_idx: int, window: int) -> None:
        """Mark *prompt_idx* as rewarded/used at *window*."""
        if prompt_idx < 0:
            raise ValueError("prompt_idx must be non-negative")
        if window < 0:
            raise ValueError("window must be non-negative")
        self._last_batched[prompt_idx] = window

    def is_in_cooldown(self, prompt_idx: int, current_window: int) -> bool:
        """True iff *prompt_idx* was rewarded within the cooldown horizon."""
        if self._cooldown_windows == 0:
            return False
        last = self._last_batched.get(prompt_idx)
        if last is None:
            return False
        return current_window - last < self._cooldown_windows

    def current_cooldown_set(self, current_window: int) -> set[int]:
        """All prompt_idx that are currently in cooldown."""
        if self._cooldown_windows == 0:
            return set()
        return {
            idx for idx, last in self._last_batched.items()
            if current_window - last < self._cooldown_windows
        }

    def __len__(self) -> int:
        return len(self._last_batched)

    # ---------- persistence ----------

    def save(self, path) -> None:
        """Serialise to JSON at *path*. Atomic via tmp-file + rename."""
        import json
        import os
        import tempfile

        path = str(path)
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=".cooldown.", dir=os.path.dirname(path) or "."
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(
                    {
                        "cooldown_windows": self._cooldown_windows,
                        "last_batched": self._last_batched,
                    },
                    f,
                )
            os.replace(tmp_path, path)
        except Exception:
            os.unlink(tmp_path)
            raise

    def load(self, path) -> None:
        """Load state from JSON at *path*. No-op if file doesn't exist."""
        import json
        import os

        path = str(path)
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        # JSON object keys are strings — coerce back to int.
        self._last_batched = {int(k): int(v) for k, v in data["last_batched"].items()}

    # ---------- snapshot state (run-keyed, persisted to R2) ----------

    def export_state(self) -> dict[int, int]:
        """Snapshot of ``prompt_idx -> last_batched_window`` for persistence."""
        return dict(self._last_batched)

    def import_state(self, last_batched: dict) -> None:
        """Replace state from an ``export_state`` snapshot (keys may be str)."""
        self._last_batched = {int(k): int(v) for k, v in last_batched.items()}

    # ---------- rebuild from archived window data ----------

    def apply_history(
        self,
        archived_windows: list[dict],
        current_window: int,
    ) -> None:
        """Merge rewarded prompts from *archived_windows*, keeping the most
        recent window per prompt. Does NOT clear existing state — use to top up
        a restored snapshot with the windows recorded since it was taken.

        Only windows within ``cooldown_windows`` of *current_window* matter —
        older entries have already expired and are skipped.
        """
        horizon = current_window - self._cooldown_windows
        for record in archived_windows:
            w = int(record["window_start"])
            if w <= horizon:
                continue
            rewarded_entries = list(record.get("batch", []))
            rewarded_entries.extend(
                entry for entry in record.get("runners_up", [])
                if entry.get("rewarded", False)
            )
            for entry in rewarded_entries:
                idx = int(entry["prompt_idx"])
                # Keep the most recent window for each prompt.
                if self._last_batched.get(idx, -1) < w:
                    self._last_batched[idx] = w

    def rebuild_from_history(
        self,
        archived_windows: list[dict],
        current_window: int,
    ) -> None:
        """Rebuild state from scratch from archived windows' rewarded prompts.

        *archived_windows* is a list of dicts, each with ``window_start``
        (int), ``batch`` (selected training prompts), and optionally
        ``runners_up`` entries carrying ``rewarded=True``. Typically fetched
        from the R2 dataset archive at validator startup.
        """
        self._last_batched.clear()
        self.apply_history(archived_windows, current_window)


class ContentCooldownMap:
    """Last-selected window keyed by a full canonical content digest."""

    def __init__(self, cooldown_windows: int) -> None:
        if cooldown_windows < 0:
            raise ValueError("cooldown_windows must be non-negative")
        self._cooldown_windows = cooldown_windows
        self._last_selected: dict[str, int] = {}

    @staticmethod
    def _digest(value: str) -> str:
        digest = str(value).strip().lower()
        if not _SHA256_HEX_RE.fullmatch(digest):
            raise ValueError("content digest must be 64 lowercase hex characters")
        return digest

    def record_selected(self, digest: str, window: int) -> None:
        if window < 0:
            raise ValueError("window must be non-negative")
        self._last_selected[self._digest(digest)] = int(window)

    def is_in_cooldown(self, digest: str, current_window: int) -> bool:
        if self._cooldown_windows == 0:
            return False
        last = self._last_selected.get(self._digest(digest))
        return last is not None and current_window - last < self._cooldown_windows

    def current_cooldown_set(self, current_window: int) -> set[str]:
        if self._cooldown_windows == 0:
            return set()
        return {
            digest
            for digest, last in self._last_selected.items()
            if current_window - last < self._cooldown_windows
        }

    def export_state(self) -> dict[str, int]:
        return dict(self._last_selected)

    def import_state(self, last_selected: dict) -> None:
        restored: dict[str, int] = {}
        for digest, window in last_selected.items():
            restored[self._digest(str(digest))] = int(window)
        self._last_selected = restored

    def __len__(self) -> int:
        return len(self._last_selected)
