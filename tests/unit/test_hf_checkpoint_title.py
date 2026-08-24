"""The startup bootstrap has to recognise the titles the trainer publishes.

`TrainerPublisher` writes "checkpoint N (cadence)"; the in-process path wrote
"checkpoint N". The bootstrap's title match required the number to END the
title, so from the detached-trainer cutover on 2026-08-22 it saw no published
checkpoint at all — it resolved "HF latest" as the last hand-titled one (572)
and kept trusting a stale operator pin while the fleet ran on 611. The
anti-regression guard it exists to provide never fired.
"""

from __future__ import annotations

import pytest

from reliquary.validator.service import checkpoint_n_from_commit_title


@pytest.mark.parametrize(
    "title, expected",
    [
        # what the detached trainer publishes (publisher.py:96)
        ("checkpoint 611 (cadence)", 611),
        ("checkpoint 612 (adaptive_policy_ratio_drift)", 612),
        # what the in-process path published (checkpoint.py:111)
        ("checkpoint 572", 572),
        ("Checkpoint 42", 42),
        ("checkpoint  7  ", 7),
        # not checkpoint publications
        ("initial commit", None),
        ("checkpoint", None),
        ("fix: checkpoint 5", None),
        ("checkpoints 5", None),
        ("checkpoint five", None),
        ("", None),
    ],
)
def test_reads_the_checkpoint_number_from_a_commit_title(title, expected):
    assert checkpoint_n_from_commit_title(title) == expected


def test_the_trainer_title_format_is_actually_recognised():
    """Guards the coupling directly: build the title the publisher builds."""
    title = f"checkpoint {611} ({'cadence'})"
    assert checkpoint_n_from_commit_title(title) == 611
