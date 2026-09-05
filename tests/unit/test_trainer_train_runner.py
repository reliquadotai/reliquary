"""Trainer-side glue: accumulate decoded windows, then train_step."""

import pytest

from reliquary import constants as C
from reliquary.shared.training_payload import (
    CheckpointEpochTrainingBinding,
    decode_training_payload,
    encode_training_payload,
)
from reliquary.trainer.train_runner import TrainRunner

from tests.unit.test_training_payload_codec import _window_batches


def _decoded(n=30100):
    return decode_training_payload(encode_training_payload(
        _window_batches(), window_start=n, checkpoint_revision="rev",
        env_order=["openmathinstruct", "opencodeinstruct"],
        window_quarantine={"quarantined": False},
    ))


def _epoch_decoded(offset, *, mode, window_count=3):
    batches = _window_batches()
    for group in batches["opencodeinstruct"]:
        for rollout in group.rollouts:
            length = int(rollout.commit["rollout"]["completion_length"])
            rollout._validated_completion_logprobs = [-0.5] * length
    first_window = 30200
    binding = CheckpointEpochTrainingBinding(
        epoch_id="1" * 64,
        manifest_sha256="2" * 64,
        training_run_id=C.TRAINING_RUN_ID,
        training_mode=mode,
        first_window=first_window,
        lane_offset=offset,
        window_count=window_count,
        target_groups_per_environment_lane=1,
    )
    return decode_training_payload(encode_training_payload(
        batches,
        window_start=first_window + offset,
        checkpoint_revision="epoch-rev",
        env_order=["openmathinstruct", "opencodeinstruct"],
        window_quarantine={"quarantined": False},
        checkpoint_epoch=binding,
    ))


def test_accumulates_until_ready_then_trains(monkeypatch):
    monkeypatch.setattr(C, "KL_BETA", 0.0)
    calls = []

    def fake_train_step(model, batches, **kw):
        calls.append((batches, kw))
        return model

    runner = TrainRunner(
        model=object(),
        env_targets={"openmathinstruct": 2, "opencodeinstruct": 2},
        env_order=["openmathinstruct", "opencodeinstruct"],
        train_step_fn=fake_train_step,
        global_step_hint=42,
    )
    assert runner.step(_decoded(30100)) is False   # 1 group/env < target 2
    assert runner.step(_decoded(30101)) is True    # ready -> trained
    assert len(calls) == 1
    batches, kw = calls[0]
    assert kw["window_index"] == 30101
    assert kw["global_step_hint"] == 42
    assert kw["ref_model"] is None
    assert len(batches) == 2 and all(len(b) == 2 for b in batches)
    # Accumulator reset after consumption.
    assert runner.step(_decoded(30102)) is False


def test_sequential_epoch_runs_one_step_per_lane(monkeypatch):
    monkeypatch.setattr(C, "KL_BETA", 0.0)
    monkeypatch.setattr(C, "PROTOCOL_VERSION", 5)
    calls = []

    def fake_train_step(model, batches, **kwargs):
        calls.append((batches, kwargs))
        return model

    runner = TrainRunner(
        model=object(),
        env_targets={"openmathinstruct": 1, "opencodeinstruct": 1},
        env_order=["openmathinstruct", "opencodeinstruct"],
        train_step_fn=fake_train_step,
    )

    assert [
        runner.step(_epoch_decoded(offset, mode="sequential_steps"))
        for offset in range(3)
    ] == [True, True, True]
    assert len(calls) == 3
    assert [call[1]["window_index"] for call in calls] == [30200, 30201, 30202]
    assert all(len(batch) == 1 for call in calls for batch in call[0])
    assert runner.snapshot()["sequential_epoch_key"] is None


def test_aggregate_epoch_runs_one_step_after_final_lane(monkeypatch):
    monkeypatch.setattr(C, "KL_BETA", 0.0)
    monkeypatch.setattr(C, "PROTOCOL_VERSION", 5)
    calls = []

    def fake_train_step(model, batches, **kwargs):
        calls.append((batches, kwargs))
        return model

    runner = TrainRunner(
        model=object(),
        env_targets={"openmathinstruct": 1, "opencodeinstruct": 1},
        env_order=["openmathinstruct", "opencodeinstruct"],
        train_step_fn=fake_train_step,
    )

    assert [
        runner.step(_epoch_decoded(offset, mode="aggregate_one_step"))
        for offset in range(3)
    ] == [False, False, True]
    assert len(calls) == 1
    batches, kwargs = calls[0]
    assert kwargs["window_index"] == 30202
    assert all(len(batch) == 3 for batch in batches)
    assert runner.snapshot()["aggregate_epoch_key"] is None


def test_quarantined_accumulated_batch_resets_without_training(monkeypatch):
    monkeypatch.setattr(C, "KL_BETA", 0.0)
    calls = []

    def fake_train_step(model, batches, **kw):
        calls.append(batches)
        return model

    def quarantine_all(groups, reject_counts):
        class _V:
            quarantined = True
            reasons = ["test"]
        return _V()

    runner = TrainRunner(
        model=object(),
        env_targets={"openmathinstruct": 1, "opencodeinstruct": 1},
        env_order=["openmathinstruct", "opencodeinstruct"],
        train_step_fn=fake_train_step,
        assess_fn=quarantine_all,
    )
    assert runner.step(_decoded(30100)) is False
    assert calls == []
    # Reset happened: the next ready batch is assessed fresh.
    assert runner.step(_decoded(30101)) is False


def test_missing_pi_old_drops_whole_group(monkeypatch):
    monkeypatch.setattr(C, "KL_BETA", 0.0)
    monkeypatch.setattr(C, "T_PROTO", 1.0)
    monkeypatch.setattr(C, "PI_OLD_FROM_VERIFY_LOGPROBS", True)
    monkeypatch.setattr(C, "RECOMPUTE_PI_OLD_FROM_VERIFY", True)
    calls = []

    def fake_train_step(model, batches, **kw):
        calls.append(batches)
        return model

    runner = TrainRunner(
        model=object(),
        env_targets={"openmathinstruct": 1, "opencodeinstruct": 1},
        env_order=["openmathinstruct", "opencodeinstruct"],
        train_step_fn=fake_train_step,
    )
    # Fixture: the code group has one rollout WITHOUT pi_old -> whole
    # group dropped, so the accumulator never reaches the code target.
    assert runner.step(_decoded(30100)) is False
    assert runner.groups_dropped_missing_pi_old == 1
    assert calls == []


def test_kl_beta_guard(monkeypatch):
    monkeypatch.setattr(C, "KL_BETA", 0.01)
    with pytest.raises(RuntimeError, match="KL_BETA"):
        TrainRunner(
            model=object(),
            env_targets={"openmathinstruct": 1},
            env_order=["openmathinstruct"],
        )


def test_aggregate_episode_epoch_preserves_masks_and_rejects_target_drift(monkeypatch):
    from types import SimpleNamespace
    from reliquary.validator.training import _policy_token_positions
    from tests.unit.test_training_payload_codec import _group, _roll

    monkeypatch.setattr(C, "KL_BETA", 0.0)
    monkeypatch.setattr(C, "PROTOCOL_VERSION", 7)
    monkeypatch.setattr(C, "T_PROTO", 1.0)
    name = "reliquary_stateful_tools_v1"
    calls = []

    def train(model, batches, **kwargs):
        calls.append(batches)
        return model

    runner = TrainRunner(model=object(), env_targets={name: 1}, env_order=[name],
                         train_step_fn=train,
                         assess_fn=lambda *a, **kw: SimpleNamespace(quarantined=False))
    decoded = []
    for offset in range(2):
        rollout = _roll(1.0, 7, env=name, prompt_length=4)
        rollout.commit["rollout"]["episode"] = {"schema_version": "test"}
        rollout._validated_assistant_spans = ((4, 6), (9, 11))
        rollout._validated_completion_logprobs = [-0.5] * 4
        binding = CheckpointEpochTrainingBinding(
            epoch_id="1" * 64, manifest_sha256="2" * 64,
            training_run_id=C.TRAINING_RUN_ID, training_mode="aggregate_one_step",
            first_window=30200, lane_offset=offset, window_count=2,
            target_groups_per_environment_lane=1,
        )
        decoded.append(decode_training_payload(encode_training_payload(
            {name: [_group([rollout], prompt_idx=offset)]}, window_start=30200 + offset,
            checkpoint_revision="a" * 40, env_order=[name], env_targets={name: 1},
            window_quarantine={}, checkpoint_epoch=binding,
        )))
    assert runner.step(decoded[0]) is False
    before = runner.snapshot()
    decoded[1].env_targets[name] = 2
    with pytest.raises(ValueError, match="targets do not match"):
        runner.step(decoded[1])
    assert runner.snapshot() == before
    decoded[1].env_targets[name] = 1
    assert runner.step(decoded[1]) is True
    assert len(calls) == 1 and len(calls[0][0]) == 2
    for group in calls[0][0]:
        assert _policy_token_positions(group.rollouts[0]) == [4, 5, 9, 10]
        assert len(group.rollouts[0]._validated_completion_logprobs) == 4
