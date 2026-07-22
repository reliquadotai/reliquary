"""CooldownMap — in-memory lifecycle."""

import pytest

from reliquary.validator.cooldown import ContentCooldownMap, CooldownMap


def test_empty_map_never_in_cooldown():
    m = CooldownMap(cooldown_windows=50)
    assert m.is_in_cooldown(prompt_idx=42, current_window=100) is False


def test_just_batched_is_in_cooldown():
    m = CooldownMap(cooldown_windows=50)
    m.record_batched(prompt_idx=42, window=100)
    # Next window — still in cooldown
    assert m.is_in_cooldown(prompt_idx=42, current_window=101) is True


def test_cooldown_expires_after_N_windows():
    m = CooldownMap(cooldown_windows=50)
    m.record_batched(prompt_idx=42, window=100)
    # At window 150 — still in cooldown (100 + 50 inclusive boundary)
    assert m.is_in_cooldown(prompt_idx=42, current_window=149) is True
    # At window 150 — exactly the boundary → cooldown ends
    assert m.is_in_cooldown(prompt_idx=42, current_window=150) is False


def test_different_prompts_independent():
    m = CooldownMap(cooldown_windows=50)
    m.record_batched(prompt_idx=42, window=100)
    m.record_batched(prompt_idx=7, window=105)
    assert m.is_in_cooldown(42, 110) is True
    assert m.is_in_cooldown(7, 110) is True
    assert m.is_in_cooldown(99, 110) is False


def test_re_record_updates_last_seen():
    m = CooldownMap(cooldown_windows=50)
    m.record_batched(prompt_idx=42, window=100)
    # Same prompt re-enters at window 200 → cooldown resets from 200
    m.record_batched(prompt_idx=42, window=200)
    assert m.is_in_cooldown(42, 240) is True
    assert m.is_in_cooldown(42, 250) is False


def test_current_cooldown_set_at_window():
    m = CooldownMap(cooldown_windows=50)
    m.record_batched(prompt_idx=42, window=100)
    m.record_batched(prompt_idx=7, window=90)
    m.record_batched(prompt_idx=99, window=40)  # expired by window 130
    assert m.current_cooldown_set(current_window=130) == {42, 7}


def test_zero_cooldown_never_blocks():
    """With cooldown=0, no prompt is ever in cooldown."""
    m = CooldownMap(cooldown_windows=0)
    m.record_batched(prompt_idx=42, window=100)
    assert m.is_in_cooldown(42, 100) is False
    assert m.is_in_cooldown(42, 101) is False


def test_negative_prompt_idx_rejected():
    m = CooldownMap(cooldown_windows=50)
    with pytest.raises(ValueError):
        m.record_batched(prompt_idx=-1, window=100)


def test_content_cooldown_roundtrip_and_expiry():
    digest = "ab" * 32
    content = ContentCooldownMap(cooldown_windows=50)
    content.record_selected(digest, window=100)

    assert content.is_in_cooldown(digest.upper(), 149) is True
    assert content.is_in_cooldown(digest, 150) is False

    restored = ContentCooldownMap(cooldown_windows=50)
    restored.import_state(content.export_state())
    assert restored.export_state() == {digest: 100}


def test_content_cooldown_rejects_truncated_digest():
    content = ContentCooldownMap(cooldown_windows=50)
    with pytest.raises(ValueError, match="64 lowercase hex"):
        content.record_selected("ab", window=100)


import json
import tempfile
from pathlib import Path


def test_persist_and_load_roundtrip(tmp_path: Path):
    path = tmp_path / "cd.json"
    m1 = CooldownMap(cooldown_windows=50)
    m1.record_batched(prompt_idx=42, window=100)
    m1.record_batched(prompt_idx=7, window=105)
    m1.save(path)

    m2 = CooldownMap(cooldown_windows=50)
    m2.load(path)
    assert m2.is_in_cooldown(42, 110) is True
    assert m2.is_in_cooldown(7, 110) is True
    assert m2.is_in_cooldown(99, 110) is False


def test_load_missing_file_is_empty(tmp_path: Path):
    m = CooldownMap(cooldown_windows=50)
    m.load(tmp_path / "nonexistent.json")
    assert len(m) == 0


def test_load_malformed_file_raises(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    m = CooldownMap(cooldown_windows=50)
    with pytest.raises(json.JSONDecodeError):
        m.load(path)


def test_rebuild_from_history_takes_most_recent():
    """If the same prompt appeared in multiple archived windows, keep the latest."""
    archived = [
        {"window_start": 100, "batch": [{"prompt_idx": 42}, {"prompt_idx": 7}]},
        {"window_start": 105, "batch": [{"prompt_idx": 42}, {"prompt_idx": 99}]},
        {"window_start": 110, "batch": [{"prompt_idx": 7}]},
    ]
    m = CooldownMap(cooldown_windows=50)
    m.rebuild_from_history(archived, current_window=120)
    # Prompt 42 last seen at 105
    assert m.is_in_cooldown(42, 120) is True
    assert m.is_in_cooldown(42, 154) is True
    assert m.is_in_cooldown(42, 155) is False
    # Prompt 7 last seen at 110
    assert m.is_in_cooldown(7, 159) is True
    assert m.is_in_cooldown(7, 160) is False
    # Prompt 99 last seen at 105
    assert m.is_in_cooldown(99, 154) is True


def test_rebuild_from_history_includes_rewarded_runners_up():
    """Paid boundary runners are cooldowned after restart too."""
    archived = [
        {
            "window_start": 100,
            "batch": [{"prompt_idx": 42}],
            "runners_up": [
                {"prompt_idx": 7, "rewarded": True},
                {"prompt_idx": 99, "rewarded": False},
            ],
        },
    ]
    m = CooldownMap(cooldown_windows=50)
    m.rebuild_from_history(archived, current_window=120)
    assert m.is_in_cooldown(42, 120) is True
    assert m.is_in_cooldown(7, 120) is True
    assert m.is_in_cooldown(99, 120) is False


def test_rebuild_ignores_windows_older_than_cooldown():
    """Windows older than cooldown horizon are pointless to load."""
    archived = [
        {"window_start": 10, "batch": [{"prompt_idx": 42}]},   # way expired
        {"window_start": 105, "batch": [{"prompt_idx": 7}]},   # fresh
    ]
    m = CooldownMap(cooldown_windows=50)
    m.rebuild_from_history(archived, current_window=120)
    assert m.is_in_cooldown(42, 120) is False  # expired long ago
    assert m.is_in_cooldown(7, 120) is True


def test_export_import_state_roundtrip():
    """Snapshot persistence: export -> import restores cooldown exactly,
    including across the JSON str-key coercion the snapshot does."""
    m = CooldownMap(cooldown_windows=1000)
    m.record_batched(7, 100)
    m.record_batched(42, 250)
    state = m.export_state()

    restored = CooldownMap(cooldown_windows=1000)
    # mimic a JSON round-trip: keys come back as strings
    restored.import_state({str(k): v for k, v in state.items()})
    assert restored.is_in_cooldown(7, 300) is True
    assert restored.is_in_cooldown(42, 300) is True
    assert restored.export_state() == {7: 100, 42: 250}


def test_apply_history_merges_without_clearing():
    """apply_history tops up existing state (gap-replay) instead of clearing,
    keeping the most-recent window per prompt."""
    m = CooldownMap(cooldown_windows=1000)
    m.import_state({7: 100, 42: 100})  # restored snapshot
    gap = [
        {"window_start": 150, "batch": [{"prompt_idx": 7}]},   # newer for 7
        {"window_start": 160, "batch": [{"prompt_idx": 99}]},  # new prompt
    ]
    m.apply_history(gap, current_window=170)
    assert m.export_state() == {7: 150, 42: 100, 99: 160}


def test_apply_history_keeps_older_when_snapshot_is_newer():
    m = CooldownMap(cooldown_windows=1000)
    m.import_state({7: 200})
    m.apply_history([{"window_start": 150, "batch": [{"prompt_idx": 7}]}], current_window=210)
    assert m.export_state() == {7: 200}  # snapshot's newer window wins
