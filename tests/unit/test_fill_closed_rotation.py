import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from reliquary.validator.fill_closed_rotation import (
    FillClosedRotationGate,
    FillClosedRotationStore,
)


PARENT_REVISION = "7" * 40
SUCCESSOR_REVISION = "8" * 40
OTHER_REVISION = "9" * 40


def _gate() -> FillClosedRotationGate:
    return FillClosedRotationGate(
        source_window=42,
        required_journal_key=687,
        parent_checkpoint_n=7,
        parent_revision=PARENT_REVISION,
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
        revision=SUCCESSOR_REVISION,
        trained_cursor=687,
    )
    store.save(candidate)
    recovered = FillClosedRotationStore(tmp_path).load()

    assert (
        recovered.adoption_covers(
            SimpleNamespace(checkpoint_n=7, revision=PARENT_REVISION)
        )
        is False
    )
    assert (
        recovered.adoption_covers(
            SimpleNamespace(checkpoint_n=8, revision=OTHER_REVISION)
        )
        is False
    )
    assert (
        recovered.adoption_covers(
            SimpleNamespace(checkpoint_n=8, revision=SUCCESSOR_REVISION)
        )
        is True
    )


def test_corrupt_rotation_state_fails_closed_on_restart(tmp_path):
    store = FillClosedRotationStore(tmp_path)
    store.path.write_text('{"schema_version":1}')

    with pytest.raises(ValueError, match="fields differ"):
        store.load()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("source_window", True),
        ("source_window", 42.0),
        ("source_window", "42"),
        ("required_journal_key", True),
        ("parent_checkpoint_n", "7"),
        ("durable_payload_count", 16.0),
    ],
)
def test_rotation_gate_rejects_coerced_durable_integers(field, value):
    payload = _gate().__dict__ | {field: value}

    with pytest.raises(ValueError):
        FillClosedRotationGate(**payload)


@pytest.mark.parametrize("revision", ["main", "7" * 39, "A" * 40])
def test_rotation_gate_requires_immutable_parent_revision(revision):
    payload = _gate().__dict__ | {"parent_revision": revision}

    with pytest.raises(ValueError, match="lowercase 40-character commit OID"):
        FillClosedRotationGate(**payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint_n", True),
        ("checkpoint_n", 8.0),
        ("checkpoint_n", "8"),
        ("trained_cursor", False),
        ("trained_cursor", 687.0),
        ("trained_cursor", "687"),
    ],
)
def test_rotation_adoption_does_not_coerce_durable_integers(field, value):
    kwargs = {
        "checkpoint_n": 8,
        "revision": SUCCESSOR_REVISION,
        "trained_cursor": 687,
        field: value,
    }

    with pytest.raises(ValueError, match="non-negative"):
        _gate().record_adoption(**kwargs)


@pytest.mark.parametrize("checkpoint_n", [True, 8.0, "8"])
def test_rotation_coverage_does_not_coerce_active_checkpoint(checkpoint_n):
    adopted = _gate().record_adoption(
        checkpoint_n=8,
        revision=SUCCESSOR_REVISION,
        trained_cursor=687,
    )

    assert (
        adopted.adoption_covers(
            SimpleNamespace(
                checkpoint_n=checkpoint_n,
                revision=SUCCESSOR_REVISION,
            )
        )
        is False
    )


@pytest.mark.parametrize(
    "raw",
    [
        b'{"schema_version":1,"schema_version":1}',
        b'{"schema_version":NaN}',
        b'{"schema_version":1e999}',
    ],
)
def test_rotation_store_rejects_ambiguous_json(tmp_path, raw):
    store = FillClosedRotationStore(tmp_path)
    store.path.write_bytes(raw)

    with pytest.raises(ValueError, match="invalid fill-closed rotation gate JSON"):
        store.load()


def test_rotation_store_rejects_noncanonical_bytes(tmp_path):
    store = FillClosedRotationStore(tmp_path)
    store.path.write_text(
        json.dumps(_gate().__dict__, sort_keys=True, indent=2)
    )

    with pytest.raises(ValueError, match="not canonical"):
        store.load()


def test_rotation_store_save_failure_preserves_previous_gate(tmp_path):
    store = FillClosedRotationStore(tmp_path)
    original = _gate()
    store.save(original)
    original_bytes = store.path.read_bytes()
    replacement = original.record_adoption(
        checkpoint_n=8,
        revision=SUCCESSOR_REVISION,
        trained_cursor=687,
    )

    with patch(
        "reliquary.validator.fill_closed_rotation.os.replace",
        side_effect=OSError("rename failed"),
    ):
        with pytest.raises(OSError, match="rename failed"):
            store.save(replacement)

    assert store.path.read_bytes() == original_bytes
    assert store.load() == original
    assert not store.path.with_suffix(".json.tmp").exists()


def test_clear_is_idempotent(tmp_path):
    store = FillClosedRotationStore(tmp_path)
    store.save(_gate())
    store.clear()
    store.clear()
    assert store.load() is None
