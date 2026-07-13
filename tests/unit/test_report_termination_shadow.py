from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "report_termination_shadow.py"
SPEC = importlib.util.spec_from_file_location("report_termination_shadow", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(index: int, *, boundary: bool = False) -> dict:
    return {
        "event": "termination_shadow",
        "miner_hotkey": f"hk-{index % 5}",
        "window_start": index // 4,
        "env_name": "openmathinstruct",
        "checkpoint_hash": "sha256:test",
        "terminal_pick_ok": False,
        "terminal_pick_cdf_miss": 0.0005 if boundary else 0.1,
        "terminal_boundary_compatible": boundary,
        "natural_close_pick_ok": None,
        "natural_close_pick_cdf_miss": None,
        "natural_close_boundary_compatible": False,
        "termination_ok": not boundary,
        "cap_truncated": False,
        "would_exceed_truncation_budget": boundary,
    }


def test_report_holds_behavior_change_on_any_boundary_candidate():
    report = MODULE.summarize([_row(0, boundary=True)])

    assert report["decision"] == (
        "REVIEW_BOUNDARY_CANDIDATES_KEEP_GATE_UNCHANGED"
    )
    assert report["terminal_boundary_candidates"] == 1
    assert report["would_exceed_truncation_budget"] == 1


def test_report_requires_clean_volume_before_no_signal_conclusion():
    rows = [_row(index) for index in range(100)]
    report = MODULE.summarize(rows)

    assert report["decision"] == "NO_BOUNDARY_FALSE_POSITIVE_SIGNAL"
    assert report["distinct_hotkeys"] == 5
    assert report["distinct_windows"] == 25
