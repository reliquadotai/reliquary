from __future__ import annotations

import json

from scripts.report_legacy_merkle_shadow import summarize


def _line(stage: str, **fields) -> str:
    payload = {
        "event": "validator_submit_lifecycle",
        "stage": stage,
        "hotkey": "5HotkeyAlpha",
        "window_n": 10,
        "prompt_idx": 4,
        "merkle_root_lead": "abc123",
        "t_arrival": 100.0,
        "protocol_version": 1,
        **fields,
    }
    return "2026-07-14 | validator | INFO | event " + json.dumps(payload)


def test_report_correlates_mismatch_with_terminal_outcome():
    result = summarize(
        [
            _line(
                "legacy_merkle_checked",
                legacy_merkle_status="mismatch",
                submission_env_name="openmathinstruct",
            ),
            _line("candidate_rejected", reject_reason="seed_mismatch"),
        ],
        min_checks=1,
        min_hotkeys=1,
        min_windows=1,
        required_envs=["openmathinstruct"],
    )

    assert result["checks"] == 1
    assert result["mismatches"] == 1
    assert result["mismatch_outcomes"] == {"seed_mismatch": 1}
    assert result["ready_to_enforce"] is False
    assert "unresolved_mismatches" in result["enforcement_blockers"]


def test_report_requires_volume_diversity_and_environments():
    result = summarize(
        [
            _line(
                "legacy_merkle_checked",
                legacy_merkle_status="match",
                submission_env_name="openmathinstruct",
            ),
            _line("candidate_accepted", reject_reason="none"),
        ],
        min_checks=2,
        min_hotkeys=2,
        min_windows=2,
        required_envs=["openmathinstruct", "opencodeinstruct"],
    )

    assert result["matches"] == 1
    assert result["match_rate"] == 1.0
    assert result["ready_to_enforce"] is False
    assert result["missing_required_environments"] == ["opencodeinstruct"]
    assert result["enforcement_blockers"] == [
        "checks<2",
        "hotkeys<2",
        "windows<2",
        "missing_required_environments",
    ]


def test_report_marks_clean_calibration_ready():
    rows = []
    for index, env in enumerate(("openmathinstruct", "opencodeinstruct")):
        payload = json.loads(_line(
            "legacy_merkle_checked",
            legacy_merkle_status="match",
            submission_env_name=env,
        ).split("event ", 1)[1])
        payload["hotkey"] = f"5Hotkey{index}"
        payload["window_n"] = 10 + index
        payload["t_arrival"] = 100.0 + index
        rows.append(json.dumps(payload))

    result = summarize(
        rows,
        min_checks=2,
        min_hotkeys=2,
        min_windows=2,
        required_envs=["openmathinstruct", "opencodeinstruct"],
    )

    assert result["ready_to_enforce"] is True
    assert result["enforcement_blockers"] == []
