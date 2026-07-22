from __future__ import annotations

import gzip
import json
import stat
from types import SimpleNamespace

from reliquary.validator.utility_telemetry import UtilityTelemetryWriter


def _submission(*, hotkey: str, prompt_idx: int, digest_byte: int):
    rollouts = []
    utility_rows = []
    for index in range(2):
        reward = float(index == 0)
        tokens = [10, 11, 20 + index, 99]
        rollouts.append(SimpleNamespace(
            reward=reward,
            commit={
                "tokens": tokens,
                "rollout": {
                    "prompt_length": 2,
                    "completion_length": 2,
                },
            },
        ))
        utility_rows.append({
            "rollout_idx": index,
            "reward": reward,
            "prompt_length": 2,
            "completion_length": 2,
            "natural_eos": True,
            "chosen_nll": {"mean": 0.2 + index, "p50": 0.2, "p90": 0.3},
            "full_policy_entropy": {
                "mean": 1.2 + index,
                "p50": 1.2,
                "p90": 1.3,
            },
            "hidden_start_f16_b64": "AAA=",
            "hidden_delta_f16_b64": "AQA=",
            "hidden_dim": 1,
            "hidden_end_completion_offset": 0,
            "representation_shift_l2": 3.0 + index,
            "token_degeneracy": {"unique_token_ratio": 1.0},
        })
    return SimpleNamespace(
        hotkey=hotkey,
        prompt_idx=prompt_idx,
        prompt_content_sha256=f"{prompt_idx:064x}",
        target_content_sha256=f"{prompt_idx + 1:064x}",
        selection_digest=bytes([digest_byte]) * 32,
        rollouts=rollouts,
        utility_rollouts=utility_rows,
    )


def _read(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def test_private_bundle_contains_winner_and_forensic_without_raw_identity(
    tmp_path,
):
    winner = _submission(hotkey="winner-hotkey", prompt_idx=7, digest_byte=1)
    forensic = _submission(
        hotkey="forensic-hotkey", prompt_idx=8, digest_byte=2
    )
    batcher = SimpleNamespace(
        _operator_by_hotkey={
            "winner-hotkey": "operator-one",
            "forensic-hotkey": "operator-two",
        },
        forensic_sample=[SimpleNamespace(
            submission=forensic,
            sample_role="counterfactual",
        )],
    )
    writer = UtilityTelemetryWriter(tmp_path)

    assert writer.write_window(
        window=42,
        checkpoint_revision="sha256:checkpoint",
        batchers={"openmathinstruct": batcher},
        selected_by_environment={"openmathinstruct": [winner]},
    )

    destination = tmp_path / "utility_telemetry" / "window-42.json.gz"
    payload = _read(destination)
    groups = payload["environments"]["openmathinstruct"]
    assert [group["role"] for group in groups] == ["winner", "forensic"]
    assert groups[0]["forensic_role"] is None
    assert groups[1]["forensic_role"] == "counterfactual"
    assert groups[0]["rollouts"][0]["tokens"] == [10, 11, 20, 99]
    assert groups[0]["group_summary"] == {
        "rollout_count": 2,
        "positive_reward_count": 1,
        "reward_mean": 0.5,
        "completion_length_mean": 2.0,
        "natural_eos_rate": 1.0,
        "representation_shift_l2_mean": 3.5,
        "positive_reward_shift_l2_mean": 3.0,
        "nonpositive_reward_shift_l2_mean": 4.0,
    }
    serialized = json.dumps(payload)
    assert "winner-hotkey" not in serialized
    assert "forensic-hotkey" not in serialized
    assert "operator-one" not in serialized
    assert "operator-two" not in serialized
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert stat.S_IMODE(
        (tmp_path / "utility_telemetry" / ".hmac-key").stat().st_mode
    ) == 0o600
    assert writer.snapshot()["last_window"] == 42


def test_operator_pseudonym_is_stable_across_windows(tmp_path):
    submission = _submission(hotkey="hk", prompt_idx=7, digest_byte=1)
    batcher = SimpleNamespace(
        _operator_by_hotkey={"hk": "operator"},
        forensic_sample=[],
    )
    writer = UtilityTelemetryWriter(tmp_path)

    for window in (10, 11):
        assert writer.write_window(
            window=window,
            checkpoint_revision="checkpoint",
            batchers={"env": batcher},
            selected_by_environment={"env": [submission]},
        )

    first = _read(
        tmp_path / "utility_telemetry" / "window-10.json.gz"
    )["environments"]["env"][0]
    second = _read(
        tmp_path / "utility_telemetry" / "window-11.json.gz"
    )["environments"]["env"][0]
    assert first["operator_pseudonym"] == second["operator_pseudonym"]
    assert first["candidate_id"] != second["candidate_id"]


def test_retention_prunes_old_windows(monkeypatch, tmp_path):
    monkeypatch.setenv("RELIQUARY_UTILITY_TELEMETRY_RETENTION_WINDOWS", "2")
    writer = UtilityTelemetryWriter(tmp_path)
    for window in (20, 21, 22):
        assert writer.write_window(
            window=window,
            checkpoint_revision="checkpoint",
            batchers={},
            selected_by_environment={},
        )

    assert sorted(path.name for path in writer.directory.glob("window-*.json.gz")) == [
        "window-21.json.gz",
        "window-22.json.gz",
    ]


def test_writer_fails_open_and_reports_health(monkeypatch, tmp_path):
    writer = UtilityTelemetryWriter(tmp_path)

    def _fail():
        raise OSError("disk unavailable")

    monkeypatch.setattr(writer, "_secret", _fail)
    assert not writer.write_window(
        window=5,
        checkpoint_revision="checkpoint",
        batchers={},
        selected_by_environment={},
    )
    assert writer.snapshot() == {
        "enabled": True,
        "schema_version": 1,
        "retention_windows": 2048,
        "writes_total": 0,
        "failures_total": 1,
        "last_window": None,
        "last_write_ts": None,
        "last_error_type": "OSError",
    }
