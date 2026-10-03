"""The child scorer survives a dying child, bounds a hung row, never blocks closing."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from reliquary.eval import verifiers_source as vs


def die_once(marker: str) -> str:
    path = Path(marker)
    if not path.exists():
        path.write_text("died")
        os._exit(1)
    return "survived"


def always_die() -> None:
    os._exit(1)


def hang() -> None:
    time.sleep(60)


def test_a_child_that_dies_is_replaced_once(tmp_path):
    scorer = vs.ChildScorer()
    try:
        assert scorer._call(die_once, str(tmp_path / "marker")) == "survived"
    finally:
        scorer.close()


def test_a_child_that_keeps_dying_is_an_error_not_a_score():
    from concurrent.futures.process import BrokenProcessPool

    scorer = vs.ChildScorer()
    try:
        with pytest.raises(BrokenProcessPool):
            scorer._call(always_die)
    finally:
        scorer.close()


def test_a_hung_row_is_cut_and_its_child_killed():
    scorer = vs.ChildScorer(timeout_seconds=3)
    started = time.monotonic()
    with pytest.raises(vs.ScoreTimeout, match="over 3 s"):
        scorer._call(hang)
    assert scorer._pool is None
    scorer.close()
    assert time.monotonic() - started < 30
