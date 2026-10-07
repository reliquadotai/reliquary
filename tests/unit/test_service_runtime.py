"""Absolute budgets, context rollover and recovery on the real SQLite journal."""
import json
from pathlib import Path

import pytest

from reliquary.protocol.service_contract import ServiceContract
from reliquary.services.runtime import ServiceRuntime, SUPPORTED_SERVICE_CAPABILITIES


def example():
    value = json.loads((Path(__file__).parents[1] / "fixtures/service_contract_v1.json").read_text())
    value["service_kind"] = "adaptive_training"
    value["policies"]["checkpoint"] = {"kind": "trainer-driven/v1", "task_scoped": 1}
    value["policies"]["reward"] = {"kind": "exploration-discount/v1", "divisor": 4, "budget_bps": 2000, "refresh_windows": 10, "max_tokens_per_group": 100}
    value["limits"] = {"max_groups": 4, "max_tokens": 400, "deadline_seconds": 100}
    contract = ServiceContract.from_dict(value)
    qualification = {"schema": "service-runtime-qualification/v1", "qualified": True,
                     "contract_sha256": contract.sha256, "qualification_id": "test-qualified",
                     "supported_capabilities": sorted(SUPPORTED_SERVICE_CAPABILITIES), "row_ids": ["a", "b", "c"], "group_size": 2,
                     **{k: value[k] for k in ("dataset", "checkpoint", "environment", "generation_contract_sha256")}}
    return contract, qualification


def observation(contract, row="a", group="first", window=1):
    return {"schema": "prompt-observation/v1", "context_sha256": contract.context_sha256,
            "row_id": row, "group_id": group, "expected_samples": 2, "sample_ids": ["s0", "s1"],
            "rewards_bps": [0, 0], "tokens": [10, 10], "window": window,
            "verification": {"generation": "verified", "sampling": "verified", "grading": "graded"}, "source_sha256": "e" * 64}


def test_absolute_discount_freshness_archive_and_restart(tmp_path):
    contract, qualified = example()
    runtime = ServiceRuntime(tmp_path / "service.sqlite", contract, qualified, now=0)
    runtime.open_window(1, window_pool=0.5, slots=2)
    row = observation(contract)
    assert runtime.training_pool(0.5) == 0.4
    result = runtime.record_verified(row, hotkey="miner-a", purpose="exploration", window_pool=0.5, slots=2, now=1)
    assert result["amount"] == pytest.approx(0.0125)
    assert runtime.record_verified(row, hotkey="miner-a", purpose="exploration", window_pool=0.5, slots=2, now=1)["inserted"] is False
    row2 = observation(contract, group="second")
    assert runtime.record_verified(row2, hotkey="miner-b", purpose="exploration", window_pool=0.5, slots=2, now=1)["amount"] == 0
    original = {"window_start": 1, "payment_policy": "fixed-selected-group/v1", "rewards_by_hotkey": {"training": 0.4}}
    archive = runtime.reconcile_archive(original)
    assert archive["rewards_by_hotkey"] == {"training": 0.4, "miner-a": 0.0125}
    assert archive["payment_policy"] == "fixed-selected-group/v1"
    assert archive["service_payment_policy"] == "service-budgeted-exploration/v1"
    assert runtime.reconcile_archive(original) == archive
    runtime.close()
    restored = ServiceRuntime(tmp_path / "service.sqlite", contract, qualified, now=99)
    assert restored.reconcile_archive(original) == archive
    with pytest.raises(ValueError, match="settled"):
        restored.record_verified(observation(contract, row="b"), hotkey="late", purpose="exploration", window_pool=0.5, slots=2, now=99)
    restored.close()


def test_order_limits_and_clock_survive_checkpoint_adoption(tmp_path):
    contract, qualified = example()
    runtime = ServiceRuntime(tmp_path / "service.sqlite", contract, qualified, now=0)
    runtime.open_window(1, window_pool=1, slots=2)
    runtime.record_verified(observation(contract), hotkey="a", purpose="exploration", window_pool=1, slots=2, now=1)
    changed = runtime.adopt(repo=contract.to_dict()["checkpoint"]["repo"], revision="f" * 40, sha256="f" * 64)
    assert changed.context_sha256 != contract.context_sha256
    assert runtime.order_contract == contract
    runtime.open_window(2, window_pool=1, slots=2)
    for i in range(3):
        runtime.record_verified(observation(changed, group=f"new-{i}", window=2), hotkey="b", purpose="exploration", window_pool=1, slots=2, now=2)
    with pytest.raises(ValueError, match="budget"):
        runtime.record_verified(observation(changed, group="over", window=2), hotkey="b", purpose="exploration", window_pool=1, slots=2, now=2)
    assert runtime.active(now=100) is False
    assert runtime.active(now=1) is False
    runtime.close()


def test_unknown_unverified_and_aborted_groups_cannot_pay(tmp_path):
    contract, qualified = example()
    runtime = ServiceRuntime(tmp_path / "service.sqlite", contract, qualified, now=0)
    row = observation(contract)
    row["verification"]["sampling"] = "unverified"
    with pytest.raises(ValueError, match="complete verification"):
        runtime.record_verified(row, hotkey="bad", purpose="exploration", window_pool=1, slots=2, now=1)
    row = observation(contract)
    row["rewards_bps"][1] = None
    with pytest.raises(ValueError, match="signal"):
        runtime.record_verified(row, hotkey="bad", purpose="exploration", window_pool=1, slots=2, now=1)
    row = observation(contract)
    runtime.open_window(1, window_pool=1, slots=2)
    runtime.record_verified(row, hotkey="good", purpose="exploration", window_pool=1, slots=2, now=1)
    archive = runtime.reconcile_archive({"window_start": 1, "rewards_by_hotkey": {}}, aborted=True)
    assert archive["rewards_by_hotkey"] == {}
    with pytest.raises(ValueError, match="disposition"):
        runtime.reconcile_archive({"window_start": 1, "rewards_by_hotkey": {}})
    runtime.close()
