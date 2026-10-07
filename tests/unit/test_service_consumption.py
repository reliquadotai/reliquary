"""Measured Q comes from adopted durable trainer cursors, not proof passes."""
from types import SimpleNamespace

import pytest

from reliquary.services.runtime import ServiceRuntime
from tests.unit.test_service_runtime import example


def test_consumption_counts_distinct_groups_once_per_window_and_survives_restart(tmp_path):
    contract, qualification = example()
    path = tmp_path / "service.sqlite"
    runtime = ServiceRuntime(path, contract, qualification, now=0)
    a, b = SimpleNamespace(prompt_idx=0), SimpleNamespace(prompt_idx=1)
    runtime.record_training_journal(0, b"batch-a", is_tombstone=False, batches={"math": [a, a]}, stride=2)
    runtime.record_training_journal(1, b"batch-b", is_tombstone=False, batches={"math": [a, b]}, stride=2)
    assert runtime.measured_consumption() == 0
    assert runtime.record_consumption(1) == 2
    assert runtime.record_consumption(1) == 2
    runtime.record_training_journal(2, b"empty", is_tombstone=True, batches={}, stride=2)
    runtime.record_training_journal(3, b"batch-c", is_tombstone=False, batches={"math": [a]}, stride=2)
    assert runtime.record_consumption(2) == 2  # a partial next window preserves the preceding measurement
    assert runtime.record_consumption(3) == 1
    with pytest.raises(ValueError, match="backwards"):
        runtime.record_consumption(1)
    with pytest.raises(ValueError, match="conflicts"):
        runtime.record_training_journal(3, b"changed", is_tombstone=False, batches={"math": [a]}, stride=2)
    runtime.close()
    restored = ServiceRuntime(path, contract, qualification, now=1)
    assert restored.measured_consumption() == 1
    assert restored.record_consumption(5) == 0  # missing projection cannot invent throughput
    restored.close()
