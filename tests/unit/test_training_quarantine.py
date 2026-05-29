from types import SimpleNamespace

from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP
from reliquary.validator.quarantine import assess_training_batch


def _rollout(reward: float, completion_length: int = 128):
    return SimpleNamespace(
        reward=reward,
        commit={
            "tokens": [1] * completion_length,
            "rollout": {
                "prompt_length": 0,
                "completion_length": completion_length,
            },
        },
    )


def _group(
    hotkey: str,
    rewards: str,
    completion_length: int = 128,
    reward_shape=None,
):
    return SimpleNamespace(
        hotkey=hotkey,
        reward_shape=reward_shape or {},
        rollouts=[
            _rollout(1.0 if bit == "1" else 0.0, completion_length)
            for bit in rewards
        ],
    )


def test_clean_mixed_batch_is_not_quarantined():
    batch = [
        _group("hk1", "11110000"),
        _group("hk2", "11001100"),
        _group("hk3", "10101010"),
        _group("hk4", "01010101"),
        _group("hk5", "00111100"),
        _group("hk6", "00001111"),
        _group("hk7", "10011001"),
        _group("hk8", "01100110"),
    ]

    decision = assess_training_batch(batch, reject_counts={})

    assert decision.quarantined is False
    assert decision.reasons == []


def test_single_cap_length_rollout_is_telemetry_not_quarantine():
    batch = [
        _group("hk1", "11110000", MAX_NEW_TOKENS_PROTOCOL_CAP),
    ] + [
        _group(f"hk{i}", "11001100")
        for i in range(2, 9)
    ]

    decision = assess_training_batch(batch, reject_counts={})

    assert decision.quarantined is False
    assert decision.metrics["cap_length_rollouts"] == 8
    assert decision.metrics["cap_length_groups"] == 1


def test_cap_length_density_quarantines_training():
    batch = [
        _group("hk1", "11110000", MAX_NEW_TOKENS_PROTOCOL_CAP),
        _group("hk2", "11001100", MAX_NEW_TOKENS_PROTOCOL_CAP),
    ] + [
        _group(f"hk{i}", "10101010")
        for i in range(3, 9)
    ]

    decision = assess_training_batch(batch, reject_counts={})

    assert decision.quarantined is True
    assert "cap_length_density" in decision.reasons


def test_dominant_reward_vector_is_telemetry_not_quarantine():
    batch = [
        _group(f"hk{i}", "11110000")
        for i in range(6)
    ] + [
        _group("hk6", "11001100"),
        _group("hk7", "10101010"),
    ]

    decision = assess_training_batch(batch, reject_counts={})

    assert decision.quarantined is False
    assert "reward_vector_dominance" not in decision.reasons
    assert decision.metrics["dominant_reward_vector"] == "11110000"
    assert decision.metrics["dominant_reward_vector_quarantine_threshold"] is True


def test_hotkey_dominance_alone_is_observability_not_quarantine():
    batch = [
        _group("solo", "11110000"),
        _group("solo", "11001100"),
        _group("solo", "10101010"),
        _group("solo", "01010101"),
        _group("solo", "00111100"),
        _group("solo", "00001111"),
        _group("solo", "10011001"),
        _group("other", "01100110"),
    ]

    decision = assess_training_batch(batch, reject_counts={})

    assert decision.quarantined is False
    assert "hotkey_batch_dominance" not in decision.reasons
    assert decision.metrics["max_hotkey_groups"] == 7
    assert decision.metrics["max_hotkey_share"] == 7 / 8


def test_high_risk_reject_spike_quarantines_training():
    batch = [_group(f"hk{i}", "11110000") for i in range(8)]

    decision = assess_training_batch(
        batch,
        reject_counts={"reward_distribution": 32},
    )

    assert decision.quarantined is True
    assert "high_risk_reject_spike" in decision.reasons


def test_reward_shape_density_quarantines_training():
    suspicious_shape = {
        "suspicious": True,
        "zero_length_mode": 120,
        "zero_length_mode_count": 4,
    }
    batch = [
        _group("hk1", "11110000", reward_shape=suspicious_shape),
        _group("hk2", "11110000", reward_shape=suspicious_shape),
    ] + [
        _group(f"hk{i}", "11001100")
        for i in range(3, 9)
    ]

    decision = assess_training_batch(batch, reject_counts={})

    assert decision.quarantined is True
    assert "reward_shape_density" in decision.reasons
    assert decision.metrics["reward_shape_groups"] == 2


def test_single_long_zero_tail_shape_quarantines_training():
    suspicious_shape = {
        "suspicious": True,
        "zero_length_mode": 4500,
        "zero_length_mode_count": 2,
    }
    batch = [
        _group("hk1", "11111100", reward_shape=suspicious_shape),
    ] + [
        _group(f"hk{i}", "11001100")
        for i in range(2, 9)
    ]

    decision = assess_training_batch(batch, reject_counts={})

    assert decision.quarantined is True
    assert "long_zero_tail_reward_shape" in decision.reasons
    assert decision.metrics["long_zero_tail_shape_groups"] == 1
