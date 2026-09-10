"""A partial fill window must not offset the next successor checkpoint."""

from types import SimpleNamespace

import pytest

from reliquary import constants
from reliquary.infrastructure.training_payload_queue import payload_key, tombstone_key
from reliquary.shared.training_payload import encode_tombstone, encode_training_payload
from reliquary.trainer import journal
from reliquary.trainer.worker import TrainerLockLost, TrainerWorker
from reliquary.validator.fill_closed_rotation import FillClosedRotationGate
from reliquary.validator.training import TrainingStepSkipped
from tests.unit.test_training_payload_codec import _window_batches


def _setup(monkeypatch, *, first_payloads=3, shadow=False, skip=False, quarantine=False):
    monkeypatch.setattr(journal, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(journal, "FILL_CLOSED_EMISSIONS_PER_WINDOW", 16)
    monkeypatch.setattr(constants, "FILL_CLOSED_EMISSIONS_PER_WINDOW", 16)
    store = {}
    for window, count in [(1, first_payloads), (2, 16)]:
        for index in range(16):
            key = window * 16 + index
            if index < count:
                store[payload_key(key)] = encode_training_payload(
                    _window_batches(), window_start=window,
                    checkpoint_revision="0" * 40,
                    env_order=["openmathinstruct", "opencodeinstruct"],
                    window_quarantine={"quarantined": quarantine},
                )
            else:
                store[tombstone_key(key)] = encode_tombstone(
                    window_start=window, failure_stage="underfill", failure_type="budget",
                )
    trained, published = [], []
    head = ["0" * 40]

    def train(decoded):
        if skip:
            raise TrainingStepSkipped("grad_norm", 123.0)
        trained.append(decoded.window_start)
        return True

    def publish(reason):
        head[0] = f"{len(published) + 1:040x}"
        published.append((worker.cursor, reason, head[0]))
        return head[0]

    worker = TrainerWorker(
        journal=journal.WindowJournal(store.get), train_fn=train,
        publish_fn=publish, head_revision_fn=lambda: head[0],
        cursor=15, stride=1, publish_every=16,
        last_published_revision=head[0], fill_closed=True, shadow=shadow,
        finish_fn=lambda: False,
    )
    return worker, trained, published, head


@pytest.mark.parametrize("skip,quarantine", [(False, False), (True, False), (False, True)])
def test_partial_then_full_window_publishes_covering_successor(monkeypatch, skip, quarantine):
    worker, trained, published, _ = _setup(monkeypatch, skip=skip, quarantine=quarantine)
    for _ in range(40):
        if worker.run_once() == "waited":
            break
    assert worker.cursor == 47
    assert [p[0] for p in published] == [31, 47]
    assert len(trained) == (0 if skip or quarantine else 19)
    assert worker.trained_since_publish == 0
    gate = FillClosedRotationGate(
        source_window=2, required_journal_key=47, parent_checkpoint_n=1,
        parent_revision=published[0][2], durable_payload_count=16,
        requires_successor=True,
    ).record_adoption(checkpoint_n=2, revision=published[-1][2], trained_cursor=47)
    assert gate.adoption_covers(SimpleNamespace(checkpoint_n=2, revision=published[-1][2]))


def test_adaptive_publish_mid_window_still_covers_boundary(monkeypatch):
    worker, _, published, _ = _setup(monkeypatch, first_payloads=16)
    assert worker.run_once() == "trained"

    def skip(_):
        raise TrainingStepSkipped("policy_ratio_drift", 0.0)

    worker._train_fn = skip
    assert worker.run_once() == "trained"
    assert worker.run_once() == "published"
    assert published[0][:2] == (17, "adaptive_policy_ratio_drift")
    for _ in range(14):
        assert worker.run_once() == "trained"
    assert worker.run_once() == "published"
    assert published[-1][:2] == (31, "fill_closed_boundary")


def test_boundary_refuses_to_publish_unflushed_accumulator(monkeypatch):
    worker, _, published, _ = _setup(monkeypatch)
    for _ in range(16):
        worker.run_once()

    def incomplete():
        raise RuntimeError("incomplete environment mix")

    worker._finish_fn = incomplete
    with pytest.raises(RuntimeError, match="incomplete environment mix"):
        worker.run_once()
    assert worker.cursor == 31 and not published


def test_empty_window_and_shadow_never_publish(monkeypatch):
    worker, _, published, _ = _setup(monkeypatch, first_payloads=0)
    for _ in range(16):
        assert worker.run_once() == "tombstone"
    assert worker.cursor == 31
    assert worker.run_once() == "trained"
    assert published == []
    worker, _, published, _ = _setup(monkeypatch, shadow=True)
    for _ in range(40):
        if worker.run_once() == "waited":
            break
    assert worker.cursor == 47 and published == []


def test_boundary_flush_and_publication_retry_keep_the_real_cursor(monkeypatch):
    worker, _, published, head = _setup(monkeypatch)
    for _ in range(16):
        worker.run_once()
    flushes = [True, False]
    worker._finish_fn = lambda: flushes.pop(0)
    head[0] = "foreign"
    with pytest.raises(TrainerLockLost):
        worker.run_once()
    assert worker.cursor == 31 and worker.trained_since_publish == 4
    assert not published
    head[0] = "0" * 40
    assert worker.run_once() == "published"
    assert published[0][:2] == (31, "fill_closed_boundary")
    assert worker.trained_since_publish == 0
