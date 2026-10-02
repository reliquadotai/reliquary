"""Single-turn Verifiers tasksets as eval-set sources (filled in by the next task)."""

from __future__ import annotations

VERIFIERS_PREFIX = "verifiers:"


def is_verifiers_source(name) -> bool:
    return isinstance(name, str) and name.startswith(VERIFIERS_PREFIX)
