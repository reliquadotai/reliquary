from types import SimpleNamespace

import pytest

from reliquary.validator.fill_closed_rotation import (
    FillClosedRotationGate,
    FillClosedRotationStore,
)


def _gate() -> FillClosedRotationGate:
    return FillClosedRotationGate(
        source_window=42,
        required_journal_key=687,
        parent_checkpoint_n=7,
        parent_revision="parent",
        durable_payload_count=16,
        requires_successor=True,
    )


def test_rotation_gate_round_trips_byte_identically_across_restart(tmp_path):
    store1 = FillClosedRotationStore(tmp_path)
    gate = _gate()
    store1.save(gate)
    original = store1.path.read_bytes()

    store2 = FillClosedRotationStore(tmp_path)
    recovered = store2.load()

    assert recovered == gate
    store2.save(recovered)
    assert store2.path.read_bytes() == original


def test_covering_candidate_requires_the_matching_active_checkpoint(tmp_path):
    store = FillClosedRotationStore(tmp_path)
    candidate = _gate().record_adoption(
        checkpoint_n=8,
        revision="successor",
        trained_cursor=687,
    )
    store.save(candidate)
    recovered = FillClosedRotationStore(tmp_path).load()

    assert (
        recovered.adoption_covers(SimpleNamespace(checkpoint_n=7, revision="parent"))
        is False
    )
    assert (
        recovered.adoption_covers(SimpleNamespace(checkpoint_n=8, revision="other"))
        is False
    )
    assert (
        recovered.adoption_covers(SimpleNamespace(checkpoint_n=8, revision="successor"))
        is True
    )


def test_corrupt_rotation_state_fails_closed_on_restart(tmp_path):
    store = FillClosedRotationStore(tmp_path)
    store.path.write_text('{"schema_version":1}')

    with pytest.raises(ValueError, match="fields differ"):
        store.load()


def test_clear_is_idempotent(tmp_path):
    store = FillClosedRotationStore(tmp_path)
    store.save(_gate())
    store.clear()
    store.clear()
    assert store.load() is None
