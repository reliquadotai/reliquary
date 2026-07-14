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
    assert report["by_forced_status_schema_v3"][0]["forced"] is False


def test_report_holds_immediately_when_small_sample_has_hard_mismatch():
    report = MODULE.summarize([_row(0, cdf_reject=True)])

    assert report["decision"] == "HOLD_AND_REVIEW_CDF_HARD_MISMATCHES"


def test_report_correlates_cdf_onset_with_repetition_by_termination_path():
    row = _row(0)
    row["schema_version"] = 4
    row["runtime_profile"] = {
        "profile_hash": "ab" * 32,
        "torch_version": "2.7.0+cu128",
        "transformers_version": "5.9.0",
        "fla_version": "0.5.0",
        "causal_conv1d_version": None,
        "qwen35_fast_path_all": False,
    }
    row["per_rollout"] = [
        {
            "termination_path": "forced_phase2_eos",
            "n_positions": 100,
            "n_hard_mismatch": 1,
            "first_hard_mismatch_offset": 10,
            "first_repeated_ngram_offset": 20,
            "repeated_ngram_fraction": 0.2,
            "tail_repeated_ngram_fraction": 0.4,
            "max_same_token_run": 9,
        },
        {
            "termination_path": "phase1_eos",
            "n_positions": 80,
            "n_hard_mismatch": 0,
            "first_hard_mismatch_offset": None,
            "first_repeated_ngram_offset": None,
            "repeated_ngram_fraction": 0.0,
            "tail_repeated_ngram_fraction": 0.0,
            "max_same_token_run": 1,
        },
    ]

    report = MODULE.summarize([row])

    assert report["records_schema_v4"] == 1
    assert report["rollouts_schema_v4"] == 2
    assert report["directionality_schema_v4"] == {
        "both_offsets_observed": 1,
        "cdf_mismatch_at_or_before_repetition": 1,
        "repetition_before_cdf_mismatch": 0,
    }
    assert {
        item["termination_path"]
        for item in report["by_termination_path_schema_v4"]
    } == {"forced_phase2_eos", "phase1_eos"}
    runtime = report["by_runtime_profile_schema_v4"][0]
    assert runtime["profile_hash"] == "ab" * 32
    assert runtime["n_hard_mismatch"] == 0
    assert runtime["fla_version"] == "0.5.0"


def test_select_rows_can_isolate_latest_checkpoint_and_recent_windows():
    rows = [
        {"schema_version": 3, "checkpoint_hash": "old", "window_start": 1},
        {"schema_version": 3, "checkpoint_hash": "new", "window_start": 2},
        {"schema_version": 3, "checkpoint_hash": "new", "window_start": 3},
        {"schema_version": 3, "checkpoint_hash": "new", "window_start": 3},
    ]

    selected = MODULE.select_rows(
        rows, latest_checkpoint=True, last_windows=1, last_records=1,
    )

    assert selected == [rows[-1]]
