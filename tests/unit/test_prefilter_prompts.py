import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "rl" / "prefilter_prompts.py"
_SPEC = importlib.util.spec_from_file_location("prefilter_prompts", _PATH)
prefilter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prefilter)


def _row(index, correct, truncated=0, k=8):
    return {"prompt_idx": index, "correct": correct, "truncated": truncated,
            "rewards": [1.0] * correct + [0.0] * (k - correct)}


def test_buckets_truncation_and_keep_band():
    rows = [_row(10, 0, truncated=8), _row(11, 1), _row(12, 6, truncated=2), _row(13, 7), _row(14, 8)]
    summary = prefilter.bucket_summary(rows, 8, 1, 6)
    assert summary["correct_count_prompts"] == {"0": 1, "1": 1, "2": 0, "3": 0, "4": 0, "5": 0,
                                                "6": 1, "7": 1, "8": 1}
    assert summary["correct_count_share"]["0"] == pytest.approx(0.2)
    assert summary["truncation_rate"] == pytest.approx(10 / 40)
    assert (summary["kept"], summary["kept_share"]) == (2, pytest.approx(0.4))
    kept, digest = prefilter.keep_list(rows, 1, 6)
    assert kept == [11, 12] and len(digest) == 64
    assert prefilter.keep_list(list(reversed(rows)), 1, 6) == (kept, digest)


def test_sample_stays_inside_the_range_and_is_reproducible():
    picked = prefilter.select_indices(100, 200, 5, seed=3)
    assert picked == sorted(picked) == prefilter.select_indices(100, 200, 5, seed=3)
    assert len(set(picked)) == 5 and all(100 <= i < 200 for i in picked)
    assert prefilter.select_indices(100, 104, None, seed=3) == [100, 101, 102, 103]


def test_band_must_fit_k():
    with pytest.raises(SystemExit):
        prefilter.parse_args(["--model", "m", "--env", "e", "--output-dir", "o", "--keep-max", "9"])
