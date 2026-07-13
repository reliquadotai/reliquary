import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "report_forced_seed_cdf.py"
SPEC = importlib.util.spec_from_file_location("report_forced_seed_cdf", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(index: int, *, cdf_reject: bool = False) -> dict:
    return {
        "schema_version": 3,
        "ts_unix": float(index * 100),
        "miner_hotkey": f"hk-{index % 5}",
        "window_start": index // 10,
        "env_name": "openmathinstruct",
        "checkpoint_hash": "sha256:test",
        "score": 0.96,
        "ratio_group_would_reject": False,
        "ratio_rollout_would_reject": False,
        "cdf_would_reject": cdf_reject,
        "n_positions": 100,
        "n_hard_mismatch": 1 if cdf_reject else 0,
        "n_miss_gt_0_01": 1 if cdf_reject else 0,
        "n_miss_gt_0_05": 1 if cdf_reject else 0,
        "n_miss_gt_0_10": 0,
        "max_cdf_miss": 0.1 if cdf_reject else 0.0,
    }


def test_report_requires_time_and_volume_before_canary():
    rows = [{"schema_version": 1, "score": 0.01}]
    rows.extend(_row(i) for i in range(100))
    report = MODULE.summarize(rows)

    assert report["decision"] == "INSUFFICIENT_EVIDENCE"
    assert report["ratio_score"]["min"] == 0.96


def test_report_holds_on_ratio_clean_hard_mismatch():
    rows = [_row(i) for i in range(1000)]
    rows[-1]["ts_unix"] = 25 * 3600
    rows[10] = _row(10, cdf_reject=True)

    report = MODULE.summarize(rows)

    assert report["decision"] == "HOLD_AND_REVIEW_CDF_HARD_MISMATCHES"
    assert report["cdf_hard_mismatch_groups_among_ratio_clean"] == 1
    assert report["cdf_hard_mismatch_positions_among_ratio_clean"] == 1
    assert report["cdf_miss_severity_schema_v3"]["gt_0_05"] == 1
    assert report["by_environment_schema_v3"][0]["environment"] == (
        "openmathinstruct"
    )


def test_report_holds_immediately_when_small_sample_has_hard_mismatch():
    report = MODULE.summarize([_row(0, cdf_reject=True)])

    assert report["decision"] == "HOLD_AND_REVIEW_CDF_HARD_MISMATCHES"
