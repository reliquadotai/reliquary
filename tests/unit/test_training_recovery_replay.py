from __future__ import annotations

import pytest

from scripts.replay_training_recovery import reconstruct_balanced_batch


def _archive(
    window: int,
    *,
    before: dict[str, int],
    added: dict[str, int],
    counts: dict[str, int],
    math_groups: int,
    code_groups: int,
) -> dict:
    targets = {"openmathinstruct": 2, "opencodeinstruct": 1}
    batch = [
        {"env_name": "openmathinstruct", "prompt_idx": window * 10 + i}
        for i in range(math_groups)
    ] + [
        {"env_name": "opencodeinstruct", "prompt_idx": window * 10 + 8 + i}
        for i in range(code_groups)
    ]
    return {
        "window_start": window,
        "batch": batch,
        "training_accumulator": {
            "checkpoint_reset": None,
            "counts_before": before,
            "added": added,
            "snapshot": {
                "checkpoint_revision": "a" * 40,
                "targets": targets,
                "counts": counts,
                "ready": counts == targets,
            },
        },
    }


def test_reconstruct_balanced_batch_uses_accumulator_added_counts():
    first = _archive(
        10,
        before={"openmathinstruct": 0, "opencodeinstruct": 0},
        added={"openmathinstruct": 1, "opencodeinstruct": 1},
        counts={"openmathinstruct": 1, "opencodeinstruct": 1},
        math_groups=1,
        code_groups=1,
    )
    second = _archive(
        11,
        before={"openmathinstruct": 1, "opencodeinstruct": 1},
        added={"openmathinstruct": 1, "opencodeinstruct": 0},
        counts={"openmathinstruct": 2, "opencodeinstruct": 1},
        # The extra groups were sealed but not retained after capacity filled.
        math_groups=2,
        code_groups=1,
    )

    order, batches, metadata = reconstruct_balanced_batch([second, first])

    assert order == ["openmathinstruct", "opencodeinstruct"]
    assert [row["prompt_idx"] for row in batches[0]] == [100, 110]
    assert [row["prompt_idx"] for row in batches[1]] == [108]
    assert metadata["checkpoint_revision"] == "a" * 40
    assert metadata["source_windows"] == [10, 11]


def test_reconstruct_balanced_batch_rejects_missing_source_window():
    second = _archive(
        11,
        before={"openmathinstruct": 1, "opencodeinstruct": 1},
        added={"openmathinstruct": 1, "opencodeinstruct": 0},
        counts={"openmathinstruct": 2, "opencodeinstruct": 1},
        math_groups=1,
        code_groups=0,
    )

    with pytest.raises(ValueError, match="missing source archive"):
        reconstruct_balanced_batch([second])
