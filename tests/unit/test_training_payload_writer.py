"""Flag-gated validator-side training payload / tombstone writers."""

from types import SimpleNamespace

import pytest

from reliquary import constants as C
from reliquary.shared.training_payload import (
    decode_tombstone,
    decode_training_payload,
)
from reliquary.validator.service import ValidationService

from tests.unit.test_training_payload_codec import _window_batches


class _RecordingQueue:
    def __init__(self):
        self.payloads = {}
        self.tombstones = {}

    def enqueue_payload(self, n, data):
        self.payloads[n] = data

    def enqueue_tombstone(self, n, data):
        self.tombstones[n] = data

    def enqueue_committed_payload(self, n, data):
        self.payloads[n] = data

    def enqueue_committed_tombstone(self, n, data):
        self.tombstones[n] = data


def _stub_service(queue):
    stub = SimpleNamespace(
        _training_payload_queue=queue,
        env_mix=[("openmathinstruct", 8), ("opencodeinstruct", 8)],
        _training_payload_queue_ref=(
            lambda self=None, q=queue: q
        ),
    )
    # The payload writer's failure path emits a tombstone via self.
    stub._write_training_tombstone = (
        lambda *a, **k: ValidationService._write_training_tombstone(
            stub, *a, **k,
        )
    )
    return stub


def test_writer_noop_when_flag_off(monkeypatch):
    monkeypatch.setattr(C, "WRITE_TRAINING_PAYLOADS", False)
    q = _RecordingQueue()
    ValidationService._write_training_payload(
        _stub_service(q), {}, 30100, "rev", {"quarantined": False},
    )
    ValidationService._write_training_tombstone(
        _stub_service(q), 30100, "stage", "Type",
    )
    assert q.payloads == {} and q.tombstones == {}


def test_writer_encodes_window(monkeypatch):
    monkeypatch.setattr(C, "WRITE_TRAINING_PAYLOADS", True)
    q = _RecordingQueue()
    ValidationService._write_training_payload(
        _stub_service(q), _window_batches(), 30100, "rev-abc",
        {"quarantined": False},
    )
    decoded = decode_training_payload(q.payloads[30100])
    assert decoded.checkpoint_revision == "rev-abc"
    assert decoded.env_order == ["openmathinstruct", "opencodeinstruct"]
    assert decoded.window_quarantine == {"quarantined": False}


def test_tombstone_writer(monkeypatch):
    monkeypatch.setattr(C, "WRITE_TRAINING_PAYLOADS", True)
    q = _RecordingQueue()
    ValidationService._write_training_tombstone(
        _stub_service(q), 30105, "proof_capacity", "ProofCapacityAbort",
    )
    doc = decode_tombstone(q.tombstones[30105])
    assert doc["failure_stage"] == "proof_capacity"


def test_writer_never_raises(monkeypatch):
    monkeypatch.setattr(C, "WRITE_TRAINING_PAYLOADS", True)

    class _Boom:
        def enqueue_payload(self, n, data):
            raise RuntimeError("disk full")

        def enqueue_tombstone(self, n, data):
            raise RuntimeError("disk full")

    stub = _stub_service(_Boom())
    ValidationService._write_training_payload(
        stub, _window_batches(), 30100, "rev", {},
    )
    ValidationService._write_training_tombstone(stub, 30100, "s", "t")


def test_fill_closed_writer_commits_through_the_immutable_queue_api(
    monkeypatch,
):
    monkeypatch.setattr(C, "WRITE_TRAINING_PAYLOADS", True)
    queue = _RecordingQueue()
    stub = _stub_service(queue)

    ValidationService._write_fill_closed_training_payload(
        stub, 481600, b"payload"
    )
    ValidationService._write_fill_closed_training_tombstone(
        stub, 481601, b"tombstone"
    )

    assert queue.payloads == {481600: b"payload"}
    assert queue.tombstones == {481601: b"tombstone"}


def test_fill_closed_writer_fails_closed_when_the_queue_does(monkeypatch):
    monkeypatch.setattr(C, "WRITE_TRAINING_PAYLOADS", True)

    class _Boom:
        def enqueue_committed_payload(self, n, data):
            raise OSError("disk full")

        def enqueue_committed_tombstone(self, n, data):
            raise OSError("disk full")

    stub = _stub_service(_Boom())
    with pytest.raises(OSError, match="disk full"):
        ValidationService._write_fill_closed_training_payload(
            stub, 481600, b"payload"
        )
    with pytest.raises(OSError, match="disk full"):
        ValidationService._write_fill_closed_training_tombstone(
            stub, 481601, b"tombstone"
        )


def test_fill_closed_writer_cannot_pay_with_journal_writing_disabled(
    monkeypatch,
):
    monkeypatch.setattr(C, "WRITE_TRAINING_PAYLOADS", False)
    stub = _stub_service(_RecordingQueue())

    with pytest.raises(RuntimeError, match="journal writing is disabled"):
        ValidationService._write_fill_closed_training_payload(
            stub, 481600, b"payload"
        )
