"""Frozen monetary envelopes and durable service archive replay."""

import math
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from reliquary.protocol.service_contract import ServiceContract
from reliquary.services.runtime import ServiceRuntime, validate_service_archive
from tests.unit.test_service_runtime import example, observation


def configured(*, groups=20, tokens=1000, divisor=4):
    contract, qualification = example()
    value = contract.to_dict()
    value["limits"].update(max_groups=groups, max_tokens=tokens)
    value["policies"]["reward"]["divisor"] = divisor
    contract = ServiceContract.from_dict(value)
    qualification["contract_sha256"] = contract.sha256
    return contract, qualification


def record(runtime, contract, *, row="a", group="first", window=1, hotkey="miner-a", now=1):
    return runtime.record_verified(observation(contract, row=row, group=group, window=window),
                                   hotkey=hotkey, purpose="exploration", window_pool=0.5, slots=2, now=now)


def test_verified_observation_requires_exact_window_envelope(tmp_path):
    contract, qualification = configured()
    runtime = ServiceRuntime(tmp_path / "runtime.sqlite", contract, qualification, now=0)
    with pytest.raises(ValueError, match="no frozen"):
        record(runtime, contract)
    runtime.open_window(1, window_pool=0.5, slots=2)
    for pool, slots in [(0.6, 2), (0.5, 3), (True, 2)]:
        with pytest.raises(ValueError, match="envelope|window pool"):
            runtime.record_verified(observation(contract), hotkey="miner-a", purpose="exploration",
                                    window_pool=pool, slots=slots, now=1)
    changed = runtime.adopt(repo=contract.to_dict()["checkpoint"]["repo"], revision="f" * 40, sha256="f" * 64)
    with pytest.raises(ValueError, match="frozen window"):
        record(runtime, changed)
    assert runtime.db.execute("SELECT COUNT(*) FROM service_observations").fetchone()[0] == 0
    assert runtime.db.execute("SELECT COUNT(*) FROM service_payments").fetchone()[0] == 0
    runtime.close()


@pytest.mark.parametrize("qualified_size", [None, True, 1, 65537, 8])
def test_verified_observation_requires_the_complete_qualified_group(tmp_path, qualified_size):
    contract, qualification = configured()
    qualification["group_size"] = qualified_size
    runtime = ServiceRuntime(tmp_path / "runtime.sqlite", contract, qualification, now=0)
    runtime.open_window(1, window_pool=0.5, slots=2)
    with pytest.raises(ValueError, match="group_size|qualified group size"):
        record(runtime, contract)
    assert runtime.db.execute("SELECT groups,tokens FROM service_orders").fetchone() == (0, 0)
    assert runtime.db.execute("SELECT COUNT(*) FROM service_observations").fetchone()[0] == 0
    assert runtime.db.execute("SELECT COUNT(*) FROM service_payments").fetchone()[0] == 0
    runtime.close()


def test_refresh_and_context_freshness_preserve_global_window_and_order_caps(tmp_path):
    contract, qualification = configured()
    runtime = ServiceRuntime(tmp_path / "runtime.sqlite", contract, qualification, now=0)
    runtime.open_window(1, window_pool=0.5, slots=2)
    assert record(runtime, contract)["amount"] == pytest.approx(0.0125)
    assert record(runtime, contract, group="duplicate-row", hotkey="miner-b")["amount"] == 0
    assert record(runtime, contract, row="b", group="second", hotkey="miner-b")["amount"] == pytest.approx(0.0125)
    assert record(runtime, contract, row="c", group="over-window", hotkey="miner-c")["amount"] == 0
    runtime.open_window(2, window_pool=0.5, slots=2)
    assert record(runtime, contract, group="same-refresh", window=2, hotkey="miner-c")["amount"] == 0
    runtime.open_window(10, window_pool=0.5, slots=2)
    assert record(runtime, contract, group="next-refresh", window=10)["amount"] == pytest.approx(0.0125)
    changed = runtime.adopt(repo=contract.to_dict()["checkpoint"]["repo"], revision="f" * 40, sha256="f" * 64)
    runtime.open_window(11, window_pool=0.5, slots=2)
    assert record(runtime, changed, group="adopted-context", window=11)["amount"] == pytest.approx(0.0125)
    assert runtime.db.execute("SELECT groups,tokens FROM service_orders").fetchone() == (7, 140)
    runtime.close()


def test_archive_reconciliation_is_idempotent_validated_and_freezes_reward_map(tmp_path):
    contract, qualification = configured()
    path = tmp_path / "runtime.sqlite"
    runtime = ServiceRuntime(path, contract, qualification, now=0)
    runtime.open_window(1, window_pool=0.5, slots=2)
    record(runtime, contract)
    for invalid in (0.401, float("nan"), -1, True):
        with pytest.raises(ValueError, match="reward"):
            runtime.reconcile_archive({"window_start": 1, "rewards_by_hotkey": {"training": invalid}})
        assert runtime.db.execute("SELECT COUNT(*) FROM service_settled").fetchone()[0] == 0
    original = {"window_start": 1, "window_status": "completed", "rewards_by_hotkey": {"training": 0.4}}
    result = runtime.reconcile_archive(original)
    assert runtime.reconcile_archive(result) == result
    assert runtime.reconcile_archive(original) == result
    changed = {**original, "rewards_by_hotkey": {"replacement": 0.4}}
    with pytest.raises(ValueError, match="reward map is already frozen"):
        runtime.reconcile_archive(changed)
    runtime.close()
    restored = ServiceRuntime(path, contract, qualification, now=200)
    assert restored.reconcile_archive(result) == result
    assert restored.db.execute("SELECT COUNT(*) FROM service_payments").fetchone()[0] == 1
    assert restored.active(now=0) is False
    assert restored.active(now=100) is False
    assert restored.active(now=0) is False
    restored.close()


def test_projection_retry_cannot_repeat_payment_or_budget_consumption(tmp_path):
    contract, qualification = configured()
    runtime = ServiceRuntime(tmp_path / "runtime.sqlite", contract, qualification, now=0)
    runtime.open_window(1, window_pool=0.5, slots=2)
    from reliquary.services.runtime_view import ServicePolicyView
    runtime.view = ServicePolicyView(contract, runtime.row_ids)
    runtime.view.observe = MagicMock(side_effect=[OSError("projection write interrupted"), None])
    with pytest.raises(OSError):
        record(runtime, contract)
    assert record(runtime, contract)["inserted"] is False
    assert runtime.view.observe.call_count == 2
    assert runtime.db.execute("SELECT groups,tokens FROM service_orders").fetchone() == (1, 20)
    result = runtime.reconcile_archive({"window_start": 1, "rewards_by_hotkey": {}})
    assert result["exploration_rewards_by_hotkey"] == {"miner-a": 0.0125}
    runtime.close()


@pytest.mark.parametrize("cap", [True, float("nan"), float("inf"), -1, 2])
def test_archive_reader_rejects_invalid_caps(tmp_path, cap):
    contract, qualification = configured()
    runtime = ServiceRuntime(tmp_path / "runtime.sqlite", contract, qualification, now=0)
    runtime.open_window(1, window_pool=0.5, slots=2)
    result = runtime.reconcile_archive({"window_start": 1, "rewards_by_hotkey": {}})
    with pytest.raises(ValueError, match="cap"):
        validate_service_archive(result, contract, cap=cap)
    runtime.close()


def test_service_replay_preserves_small_absolute_fractions_and_legacy_zero_floor_cutoff():
    from reliquary.constants import EMA_ALPHA
    from reliquary.validator.weight_only import WeightOnlyValidator

    archive = {"task_id": "service", "window_start": 1, "rewards_by_hotkey": {"small": 1e-8}}
    assert WeightOnlyValidator._replay_ema([archive]) == {}
    assert WeightOnlyValidator._replay_ema([archive], floors={"service": (0.0, 0.0)}) == {}
    legacy = {**archive, "task_id": "legacy", "rewards_by_hotkey": {"legacy-small": 1e-8}}
    assert WeightOnlyValidator._replay_ema([archive, legacy], floors={"service": (0.0, 0.0), "legacy": (0.0, 0.0)},
                                         service_tasks=frozenset({"service"})) == {
        "small": pytest.approx(EMA_ALPHA * 1e-8)
    }


def test_existing_task_run_cannot_change_order_policies_or_limits(tmp_path):
    contract, qualification = configured()
    path = tmp_path / "runtime.sqlite"
    runtime = ServiceRuntime(path, contract, qualification, now=0)
    runtime.open_window(1, window_pool=0.5, slots=2)
    record(runtime, contract)
    changed, changed_qualification = configured(groups=21)
    assert changed.context_sha256 == contract.context_sha256
    with pytest.raises(ValueError, match="another order.*new run"):
        ServiceRuntime(path, changed, changed_qualification, now=1)
    assert runtime.db.execute("SELECT id,groups,tokens FROM service_orders").fetchall() == [(contract.sha256, 1, 20)]
    runtime.close()
    restored = ServiceRuntime(path, contract, qualification, now=2)
    assert restored.order_contract == contract
    restored.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("training", [False, True])
async def test_passed_groups_queue_recovery_archive_retry_and_weight_replay(tmp_path, monkeypatch, training):
    """CPU proof fixtures drive real scheduler, SQLite, codec and disk queues."""
    import reliquary.infrastructure.training_payload_queue as queue_module
    import reliquary.validator.fill_closed_recovery as recovery_module
    import reliquary.validator.weight_only as weights_module
    from reliquary.constants import EMA_ALPHA
    from reliquary.infrastructure.archive_queue import ArchiveQueue
    from reliquary.shared.task_registry import MECHANISM_SERVICE_RL
    from reliquary.shared.training_payload import decode_training_payload, encode_training_payload
    from reliquary.validator.fill_closed_recovery import FillClosedRecoveryStore, accounting_rows
    from reliquary.validator.fill_closed_rotation import FillClosedRotationStore
    from reliquary.validator.proof_scheduler import GlobalProofScheduler, ProofDecisionStatus, ProofExecution, ProofPlan, RankedProof

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(queue_module, "FILL_CLOSED_EMISSIONS_PER_WINDOW", 2)
    monkeypatch.setattr(recovery_module, "B_BATCH", 2)
    monkeypatch.setattr(recovery_module, "FILL_CLOSED_EMISSIONS_PER_WINDOW", 2)
    monkeypatch.setattr(recovery_module, "FILL_CLOSED_PICKS_PER_WINDOW", 1)
    contract, qualification = configured(divisor=1000000)
    runtime_path = tmp_path / "runtime.sqlite"
    runtime = ServiceRuntime(runtime_path, contract, qualification, now=0)
    runtime.open_window(1, window_pool=0.5, slots=2)
    revision = contract.to_dict()["checkpoint"]["revision"]
    rollout = lambda reward: SimpleNamespace(reward=reward, env_name="math", commit={
        "tokens": [1, 2, 3], "rollout": {"prompt_length": 1, "completion_length": 2,
                                       "token_logprobs": [-0.1, -0.2]}})
    group = SimpleNamespace(hotkey="training", prompt_idx=1, sigma=0.5, eos_tokens=1,
                            claimed_checkpoint_hash=revision, merkle_root_bytes=b"a" * 32,
                            selection_digest=b"b" * 32, rollout_hashes=[b"c" * 32, b"d" * 32],
                            rollouts=[rollout(0), rollout(1)])
    candidates = [RankedProof("exploration", 0, "row-a", observation(contract), counts_toward_target=False)]
    if training:
        candidates.append(RankedProof("training", 1, "row-b", group))
    with GlobalProofScheduler(devices=("cpu-fixture",), environments=("math",),
                             checkpoint_revision=revision,
                             proof_callable=lambda call: ProofExecution(passed=True, value=call.candidate.payload)) as scheduler:
        result = scheduler.submit(ProofPlan("service-window", "math", revision, candidates, 1,
                                            time.monotonic() + 5, allow_shortfall=True)).result(1)
    assert all(decision.status is ProofDecisionStatus.PASSED for decision in result.decisions)
    assert result.winner_job_ids == (("training",) if training else ())
    explored = runtime.record_verified(result.decisions[0].value, hotkey="exploration", purpose="exploration",
                                       window_pool=0.5, slots=2, now=1)
    assert explored["amount"] == pytest.approx(0.00000005)
    store = FillClosedRecoveryStore(tmp_path / "state")
    store.begin(1, checkpoint_n=0, revision=revision, targets={"math": 2}, window_pool=runtime.training_pool(0.5))
    queue = queue_module.TrainingPayloadQueue(str(tmp_path / "payloads"))
    if training:
        batches = {"math": [result.decisions[1].value]}
        body = encode_training_payload(batches, window_start=1, checkpoint_revision=revision, env_order=["math"], window_quarantine={})
        decoded = decode_training_payload(body)
        assert decoded.batches()["math"][0].prompt_idx == 1
        queue.enqueue_committed_payload(2, body, accounting=accounting_rows(batches, batch_index=0))
    archives = ArchiveQueue(str(tmp_path / "archives"))
    broken = SimpleNamespace(enqueue=lambda *args: (_ for _ in ()).throw(OSError("archive enqueue interrupted")))
    rotation = FillClosedRotationStore(tmp_path / "state")
    with pytest.raises(OSError, match="enqueue interrupted"):
        store.recover(1, queue=queue, archives=broken, rotation=rotation, service_runtime=runtime)
    frozen = store.load(1)["archive"]
    runtime.close()
    restarted = ServiceRuntime(runtime_path, contract, qualification, now=2)
    recovered = FillClosedRecoveryStore(tmp_path / "state").recover(
        1, queue=queue_module.TrainingPayloadQueue(str(tmp_path / "payloads")), archives=archives,
        rotation=rotation, service_runtime=restarted)
    assert recovered == frozen
    assert archives.pending_archives(start_window=1, end_window=1)[1] == frozen
    assert len(list(queue._journal_commit_dir.glob("window-*.json"))) == 2
    assert not store.windows()
    validate_service_archive(recovered, contract, cap=0.5)
    assert recovered["window_status"] == ("recovered_partial" if training else "aborted")
    expected = {"training": 0.2, "exploration": explored["amount"]} if training else {}
    assert recovered["rewards_by_hotkey"] == expected
    assert restarted.reconcile_archive(recovered, aborted=not training) == recovered

    entry = SimpleNamespace(mechanism=MECHANISM_SERVICE_RL, service_contract=contract.to_dict(),
                            params={"cap": 0.5, "min_incentive_share": 0.0, "min_incentive_ramp_start": 0.0})
    monkeypatch.setattr(weights_module, "read_registry", AsyncMock(return_value=({"service": entry}, None)))
    monkeypatch.setattr(weights_module.storage, "list_task_ids", AsyncMock(return_value=["service"]))
    monkeypatch.setattr(weights_module.storage, "list_all_window_keys", AsyncMock(return_value=[1]))
    requested = []
    async def project(*args, fields, **kwargs):
        requested.append(fields)
        return [{field: recovered[field] for field in fields if field in recovered}]
    monkeypatch.setattr(weights_module.storage, "list_recent_datasets", project)
    monkeypatch.setattr(weights_module.chain, "get_subtensor", AsyncMock(return_value=object()))
    monkeypatch.setattr(weights_module.chain, "close_subtensor", AsyncMock())
    monkeypatch.setattr("reliquary.constants.UID_BURN", None)
    metagraph = SimpleNamespace(hotkeys=["owner", "training", "exploration"], uids=[0, 1, 2], owner_hotkey="owner")
    monkeypatch.setattr(weights_module.chain, "get_metagraph", AsyncMock(return_value=metagraph))
    validator = weights_module.WeightOnlyValidator(wallet=SimpleNamespace(hotkey=SimpleNamespace(ss58_address="reader")), netuid=81)
    captured = []
    async def submit(_subtensor, _wallet, _netuid, uids, fractions):
        captured.append(dict(zip(uids, fractions)))
        return True
    monkeypatch.setattr(weights_module.chain, "set_weights", submit)
    assert await validator.submit_once()
    assert "service_order_contract" in requested[0]
    paid = {metagraph.hotkeys[uid]: value for uid, value in captured[0].items() if uid != 0}
    assert paid == {key: pytest.approx(EMA_ALPHA * value) for key, value in expected.items()}
    assert math.fsum(paid.values()) <= 0.5
    assert captured[0][0] == pytest.approx(1 - math.fsum(paid.values()))
    assert math.fsum(captured[0].values()) == pytest.approx(1)
    restarted.close()
