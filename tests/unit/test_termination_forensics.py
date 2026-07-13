from __future__ import annotations

import json

from reliquary.validator.auth_forensics import record_termination_shadow


def test_termination_shadow_records_boundary_candidate(tmp_path):
    path = tmp_path / "termination-shadow.jsonl"

    record_termination_shadow(
        hotkey="hk",
        window_start=500,
        env_name="openmathinstruct",
        checkpoint_hash="sha256:test",
        prompt_idx=42,
        rollout_idx=3,
        completion_length=700,
        p_stop=0.0001,
        terminal_pick_ok=False,
        terminal_pick_cdf_miss=0.0005,
        natural_close_pick_ok=None,
        natural_close_pick_cdf_miss=None,
        termination_ok=False,
        cap_truncated=False,
        would_exceed_truncation_budget=True,
        boundary_epsilon=0.002,
        path=path,
    )

    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["event"] == "termination_shadow"
    assert row["window_start"] == 500
    assert row["terminal_boundary_compatible"] is True
    assert row["natural_close_boundary_compatible"] is False
    assert row["would_exceed_truncation_budget"] is True
