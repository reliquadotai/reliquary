"""Strictly-ordered journal consumption: payload, tombstone, or wait."""

import pytest

from reliquary.shared.training_payload import (
    TrainingPayloadProtocolMismatch,
    active_training_identity,
    encode_tombstone,
    encode_training_payload,
)
from reliquary.trainer.journal import WindowJournal

from tests.unit.test_training_payload_codec import _window_batches


def _payload_bytes(n):
    return encode_training_payload(
        _window_batches(), window_start=n, checkpoint_revision="rev",
        env_order=["openmathinstruct", "opencodeinstruct"],
        window_quarantine={"quarantined": False},
    )


def test_returns_payload_then_none():
    store = {"reliquary/training/window-101.npz": _payload_bytes(101)}
    j = WindowJournal(fetch_fn=store.get)
    kind, decoded = j.next_entry(100, stride=1)
    assert kind == "payload" and decoded.window_start == 101
    assert j.next_entry(101, stride=1) is None  # nothing yet -> wait


def test_tombstone_found_when_no_payload():
    store = {
        "reliquary/training/window-101.tombstone.json": encode_tombstone(
            window_start=101, failure_stage="s", failure_type="t",
        ),
    }
    j = WindowJournal(fetch_fn=store.get)
    kind, doc = j.next_entry(100, stride=1)
    assert kind == "tombstone" and doc["window_start"] == 101


def test_payload_wins_over_tombstone():
    store = {
        "reliquary/training/window-101.npz": _payload_bytes(101),
        "reliquary/training/window-101.tombstone.json": encode_tombstone(
            window_start=101, failure_stage="s", failure_type="t",
        ),
    }
    j = WindowJournal(fetch_fn=store.get)
    kind, _ = j.next_entry(100, stride=1)
    assert kind == "payload"


def test_stride_respected():
    store = {"reliquary/training/window-105.npz": _payload_bytes(105)}
    j = WindowJournal(fetch_fn=store.get)
    kind, decoded = j.next_entry(100, stride=5)
    assert kind == "payload" and decoded.window_start == 105


def test_v5_identity_mismatch_fails_closed():
    store = {"reliquary/training/window-101.npz": _payload_bytes(101)}
    expected = active_training_identity()
    expected["protocol_profile_id"] = "another-profile"
    journal = WindowJournal(
        fetch_fn=store.get,
        expected_identity=expected,
    )

    with pytest.raises(
        TrainingPayloadProtocolMismatch,
        match="protocol_profile_id",
    ):
        journal.next_entry(100, stride=1)


# ---- C3: journal key-space migration at resume (R25) ------------------


def test_a_raw_cursor_is_multiplied_once_when_v6_is_armed(monkeypatch):
    """R25: at cutover the trainer's cursor is a WINDOW number, but the v6
    journal is keyed window * EMISSIONS + batch. Resuming a raw cursor
    against the encoded space parks the trainer 15/16 of a window early
    forever, or raises on the window-start comparison."""
    import reliquary.trainer.journal as journal_module
    from reliquary.trainer.journal import migrate_journal_cursor

    monkeypatch.setattr(journal_module, "FILL_CLOSED_ENABLED", True)
    emissions = journal_module.FILL_CLOSED_EMISSIONS_PER_WINDOW

    cursor, key_space = migrate_journal_cursor(30_000, "raw")

    assert cursor == 30_000 * emissions
    assert key_space == "fill_closed"


def test_migration_is_idempotent(monkeypatch):
    """The marker is rewritten with the cursor, so a second resume from the
    migrated checkpoint must be a no-op -- otherwise every restart
    multiplies again."""
    import reliquary.trainer.journal as journal_module
    from reliquary.trainer.journal import migrate_journal_cursor

    monkeypatch.setattr(journal_module, "FILL_CLOSED_ENABLED", True)

    once = migrate_journal_cursor(30_000, "raw")
    twice = migrate_journal_cursor(*once)

    assert twice == once


def test_a_fill_closed_cursor_is_divided_when_the_gate_is_off(monkeypatch):
    """The other direction: rolling v6 back must not leave the cursor 16x
    past the end of the raw key space."""
    import reliquary.trainer.journal as journal_module
    from reliquary.trainer.journal import migrate_journal_cursor

    monkeypatch.setattr(journal_module, "FILL_CLOSED_ENABLED", False)
    emissions = journal_module.FILL_CLOSED_EMISSIONS_PER_WINDOW

    cursor, key_space = migrate_journal_cursor(30_000 * emissions, "fill_closed")

    assert cursor == 30_000
    assert key_space == "raw"


def test_an_absent_marker_reads_as_raw(monkeypatch):
    """Every checkpoint published before this field existed is raw-keyed."""
    import reliquary.trainer.journal as journal_module
    from reliquary.trainer.journal import migrate_journal_cursor

    monkeypatch.setattr(journal_module, "FILL_CLOSED_ENABLED", True)
    emissions = journal_module.FILL_CLOSED_EMISSIONS_PER_WINDOW

    assert migrate_journal_cursor(7, None) == (7 * emissions, "fill_closed")


def test_an_unknown_marker_refuses_to_guess(monkeypatch):
    import reliquary.trainer.journal as journal_module
    from reliquary.trainer.journal import migrate_journal_cursor

    monkeypatch.setattr(journal_module, "FILL_CLOSED_ENABLED", True)

    with pytest.raises(ValueError):
        migrate_journal_cursor(7, "something-else")


def test_the_matching_marker_leaves_the_cursor_alone(monkeypatch):
    import reliquary.trainer.journal as journal_module
    from reliquary.trainer.journal import migrate_journal_cursor

    monkeypatch.setattr(journal_module, "FILL_CLOSED_ENABLED", False)

    assert migrate_journal_cursor(30_000, "raw") == (30_000, "raw")


# ------------------------------------------------------- catch-up (R42)

def _store_with_entries(*window_numbers, tombstones=()):
    from reliquary.infrastructure.training_payload_queue import (
        payload_key, tombstone_key,
    )
    store = {}
    for n in window_numbers:
        store[payload_key(n)] = b"payload-bytes"
    for n in tombstones:
        store[tombstone_key(n)] = b"tombstone-bytes"
    return store


def test_backlog_depth_counts_consecutive_entries_and_stops_at_the_gap():
    journal = WindowJournal(_store_with_entries(101, 102, 103, 105).get)
    assert journal.backlog_depth(100, stride=1) == 3


def test_backlog_depth_counts_a_tombstone_as_an_entry():
    """A tombstoned key still occupies its slot: R18 keeps the journal
    gapless, so the first ABSENCE is the frontier, never a tombstone."""
    journal = WindowJournal(
        _store_with_entries(101, 103, tombstones=(102,)).get
    )
    assert journal.backlog_depth(100, stride=1) == 3


def test_backlog_depth_is_zero_at_the_frontier():
    journal = WindowJournal(_store_with_entries(101).get)
    assert journal.backlog_depth(101, stride=1) == 0


def test_backlog_depth_probes_existence_without_fetching_bodies():
    """The probe must not download megabytes of stale payloads it exists
    to avoid training: with an exists_fn injected, fetch is never hit."""
    from reliquary.infrastructure.training_payload_queue import payload_key
    fetched = []

    def fetch(key):
        fetched.append(key)
        return b"x"

    keys = {payload_key(n) for n in (101, 102)}
    journal = WindowJournal(fetch, exists_fn=lambda k: k in keys)
    assert journal.backlog_depth(100, stride=1) == 2
    assert fetched == []
