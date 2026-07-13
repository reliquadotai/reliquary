import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "report_forced_seed_cdf.py"
SPEC = importlib.util.spec_from_file_location("report_forced_seed_cdf", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(index: int, *, cdf_reject: bool = False) -> dict:
    return {
        "schema_version": 2,
        "ts_unix": float(index * 100),
        "miner_hotkey": f"hk-{index % 5}",
        "score": 0.96,
        "ratio_group_would_reject": False,
        "ratio_rollout_would_reject": False,
        "cdf_would_reject": cdf_reject,
        "max_cdf_miss": 0.1 if cdf_reject else 0.0,
    }


def test_report_requires_time_and_volume_before_canary():
    report = MODULE.summarize([_row(i) for i in range(100)])
    assert report["decision"] == "INSUFFICIENT_EVIDENCE"


def test_report_holds_on_ratio_clean_hard_mismatch():
    rows = [_row(i) for i in range(1000)]
    rows[-1]["ts_unix"] = 25 * 3600
    rows[10]["cdf_would_reject"] = True

    report = MODULE.summarize(rows)

    assert report["decision"] == "HOLD_AND_REVIEW_CDF_HARD_MISMATCHES"
    assert report["cdf_hard_mismatch_groups_among_ratio_clean"] == 1
