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


def _group(hotkey: str, rewards: str, completion_length: int = 128):
    return SimpleNamespace(
        hotkey=hotkey,
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


def test_cap_length_rollout_quarantines_training():
    batch = [
        _group("hk1", "11110000", MAX_NEW_TOKENS_PROTOCOL_CAP),
        _group("hk2", "11001100"),
    ]

    decision = assess_training_batch(batch, reject_counts={})

    assert decision.quarantined is True
    assert "cap_length_rollout" in decision.reasons


def test_dominant_reward_vector_quarantines_training():
    batch = [
        _group(f"hk{i}", "11110000")
        for i in range(6)
    ] + [
        _group("hk6", "11001100"),
        _group("hk7", "10101010"),
    ]

    decision = assess_training_batch(batch, reject_counts={})

    assert decision.quarantined is True
    assert "reward_vector_dominance" in decision.reasons


def test_high_risk_reject_spike_quarantines_training():
    batch = [_group(f"hk{i}", "11110000") for i in range(8)]

    decision = assess_training_batch(
        batch,
        reject_counts={"reward_distribution": 32},
    )

    assert decision.quarantined is True
    assert "high_risk_reject_spike" in decision.reasons
