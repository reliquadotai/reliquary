"""Trainer-side glue: accumulate decoded windows, then train_step."""

import pytest

from reliquary import constants as C
from reliquary.shared.training_payload import (
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
