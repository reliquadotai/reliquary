from __future__ import annotations

import pytest

from scripts.replay_training_recovery import (
    parse_legacy_targets,
    reconstruct_balanced_batch,
    strip_synthetic_claim_metrics,
)


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


def test_reconstruct_legacy_full_window_requires_exact_explicit_contract():
    revision = "b" * 40
    archive = {
        "window_start": 20,
        "batch": [
            {
                "env_name": "math",
                "prompt_idx": 1,
                "claimed_checkpoint_hash": revision,
            },
            {
                "env_name": "math",
                "prompt_idx": 2,
                "claimed_checkpoint_hash": revision,
            },
            {
                "env_name": "code",
                "prompt_idx": 3,
                "claimed_checkpoint_hash": revision,
            },
        ],
    }

    order, batches, metadata = reconstruct_balanced_batch(
        [archive],
        legacy_checkpoint_revision=revision,
        legacy_targets={"math": 2, "code": 1},
    )

    assert order == ["math", "code"]
    assert [len(batch) for batch in batches] == [2, 1]
    assert metadata["checkpoint_revision"] == revision
    assert metadata["checkpoint_claims"] == [revision]
    assert metadata["checkpoint_claims_available"] is True
    assert metadata["legacy_full_window"] is True

    with pytest.raises(ValueError, match="immutable checkpoint"):
        reconstruct_balanced_batch(
            [archive], legacy_targets={"math": 2, "code": 1}
        )
    with pytest.raises(ValueError, match="counts do not match"):
        reconstruct_balanced_batch(
            [archive],
            legacy_checkpoint_revision=revision,
            legacy_targets={"math": 1, "code": 1},
        )


def test_reconstruct_legacy_full_window_rejects_claim_mismatch():
    explicit_revision = "b" * 40
    archive = {
        "window_start": 20,
        "batch": [
            {
                "env_name": "math",
                "prompt_idx": 1,
                "claimed_checkpoint_hash": "c" * 40,
            },
            {
                "env_name": "code",
                "prompt_idx": 2,
                "claimed_checkpoint_hash": "c" * 40,
            },
        ],
    }

    with pytest.raises(ValueError, match="claims do not match"):
        reconstruct_balanced_batch(
            [archive],
            legacy_checkpoint_revision=explicit_revision,
            legacy_targets={"math": 1, "code": 1},
        )


def test_reconstruct_legacy_full_window_allows_uniformly_missing_old_claims():
    revision = "b" * 40
    archive = {
        "window_start": 20,
        "batch": [
            {"env_name": "math", "prompt_idx": 1},
            {"env_name": "code", "prompt_idx": 2},
        ],
    }

    _order, _batches, metadata = reconstruct_balanced_batch(
        [archive],
        legacy_checkpoint_revision=revision,
        legacy_targets={"math": 1, "code": 1},
    )

    assert metadata["checkpoint_claims"] == []
    assert metadata["checkpoint_claims_available"] is False


def test_parse_legacy_targets_is_explicit_and_unique():
    assert parse_legacy_targets(["math=8", "code=8"]) == {
        "math": 8,
        "code": 8,
    }
    with pytest.raises(ValueError, match="unique"):
        parse_legacy_targets(["math=8", "math=8"])


def test_strip_synthetic_claim_metrics_keeps_policy_health_metrics():
    cleaned, ignored = strip_synthetic_claim_metrics({
        "train/grad_norm": 1.5,
        "train/pi_old_claim_abs_error_mean": 0.25,
        "train/pi_old_claim_token_count": 100.0,
        "train/ppo_clip_active_ratio": 0.01,
    })

    assert cleaned == {
        "train/grad_norm": 1.5,
        "train/ppo_clip_active_ratio": 0.01,
    }
    assert ignored == [
        "train/pi_old_claim_abs_error_mean",
        "train/pi_old_claim_token_count",
    ]
