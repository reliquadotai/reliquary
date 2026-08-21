"""Strictly-ordered journal consumption: payload, tombstone, or wait."""

from reliquary.shared.training_payload import (
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
