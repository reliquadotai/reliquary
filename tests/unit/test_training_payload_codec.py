"""Round-trip fidelity of the R2 training payload.

The invariant: a decoded payload must drive train_step's metadata pass
(_plan_from_batches) and both pi_old accessors to the same values as the
live objects. A silently dropped field degrades the model, not the tests
— so equality is checked on the consumed accessors, not just raw fields.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from reliquary import constants as C
from reliquary.shared.training_payload import (
    PAYLOAD_SCHEMA_VERSION,
    TOMBSTONE_SCHEMA_VERSION,
    active_training_identity,
    decode_tombstone,
    decode_training_payload,
    encode_tombstone,
    encode_training_payload,
)
from reliquary.validator.training import (
    _completion_token_logprobs,
    _plan_from_batches,
    _validator_completion_logprobs,
)


def _roll(reward, length, *, forced=False, truncated=False,
          env="openmathinstruct", with_pi_old=True, prompt_length=7):
    tokens = list(range(prompt_length + length))
    meta = {
        "prompt_length": prompt_length,
        "completion_length": length,
        "token_logprobs": [-1.5] * length,
        "forced": forced,
        "truncated": truncated,
    }
    r = SimpleNamespace(
        reward=reward,
        env_name=env,
        commit={"tokens": tokens, "rollout": meta},
    )
    if with_pi_old:
        r._validated_completion_logprobs = [
            math.log(0.5) + 0.001 * i for i in range(length)
        ]
    return r


def _group(rollouts, prompt_idx=0):
    return SimpleNamespace(rollouts=rollouts, prompt_idx=prompt_idx)


def _window_batches():
    return {
        "openmathinstruct": [
            _group([_roll(1.0, 4), _roll(0.0, 6, forced=True)], prompt_idx=11),
        ],
        "opencodeinstruct": [
            _group([_roll(0.5, 5, env="opencodeinstruct"),
                    _roll(0.9, 3, env="opencodeinstruct", truncated=True,
                          with_pi_old=False)], prompt_idx=22),
        ],
    }


def _encode_decode(batches):
    blob = encode_training_payload(
        batches,
        window_start=30100,
        checkpoint_revision="rev-abc",
        env_order=["openmathinstruct", "opencodeinstruct"],
        window_quarantine={"quarantined": False, "reasons": []},
    )
    assert isinstance(blob, bytes)
    return decode_training_payload(blob)


def test_header_round_trip():
    decoded = _encode_decode(_window_batches())
    assert decoded.window_start == 30100
    assert decoded.checkpoint_revision == "rev-abc"
    assert decoded.env_order == ["openmathinstruct", "opencodeinstruct"]
    assert decoded.window_quarantine == {"quarantined": False, "reasons": []}


def test_consumed_accessors_round_trip():
    original = _window_batches()
    decoded = _encode_decode(original).batches()
    for env in original:
        assert len(decoded[env]) == len(original[env])
        for g0, g1 in zip(original[env], decoded[env]):
            assert g1.prompt_idx == g0.prompt_idx
            for r0, r1 in zip(g0.rollouts, g1.rollouts):
                assert r1.env_name == r0.env_name
                assert r1.reward == pytest.approx(r0.reward)
                assert list(r1.commit["tokens"]) == list(r0.commit["tokens"])
                assert _completion_token_logprobs(r1) == pytest.approx(
                    _completion_token_logprobs(r0)
                )
                meta0, meta1 = r0.commit["rollout"], r1.commit["rollout"]
                for key in ("prompt_length", "completion_length",
                            "forced", "truncated"):
                    assert meta1.get(key) == meta0.get(key)


def test_pi_old_fp32_exact_and_absent_when_missing(monkeypatch):
    monkeypatch.setattr(C, "T_PROTO", 1.0)
    monkeypatch.setattr(C, "PI_OLD_FROM_VERIFY_LOGPROBS", True)
    monkeypatch.setattr(C, "RECOMPUTE_PI_OLD_FROM_VERIFY", True)
    original = _window_batches()
    decoded = _encode_decode(original).batches()
    for env in original:
        for g0, g1 in zip(original[env], decoded[env]):
            for r0, r1 in zip(g0.rollouts, g1.rollouts):
                n = int(r0.commit["rollout"]["completion_length"])
                v0 = _validator_completion_logprobs(r0, n)
                v1 = _validator_completion_logprobs(r1, n)
                if v0 is None:
                    assert v1 is None
                else:
                    assert v1 == [float(np.float32(x)) for x in v0]


def test_plan_from_batches_equivalence():
    original = _window_batches()
    decoded = _encode_decode(original).batches()
    order = ["openmathinstruct", "opencodeinstruct"]
    plan0, skipped0 = _plan_from_batches([original[e] for e in order])
    plan1, skipped1 = _plan_from_batches([decoded[e] for e in order])
    assert skipped1 == skipped0
    assert len(plan1) == len(plan0)
    for (g0, adv0, scale0), (g1, adv1, scale1) in zip(plan0, plan1):
        assert g1.prompt_idx == g0.prompt_idx
        assert adv1 == pytest.approx(adv0)
        assert scale1 == pytest.approx(scale0)


def test_t_proto_gate_drops_pi_old(monkeypatch):
    monkeypatch.setattr(C, "T_PROTO", 0.6)
    decoded = _encode_decode(_window_batches()).batches()
    for env_groups in decoded.values():
        for g in env_groups:
            for r in g.rollouts:
                assert getattr(r, "_validated_completion_logprobs", None) is None


def test_force_span_and_termination_path_round_trip():
    from reliquary.validator.training import _completion_keep_list

    original = _window_batches()
    forced = original["openmathinstruct"][0].rollouts[1]
    forced._validated_force_span = (9, 12)
    forced._validated_termination_path = "bft_forced"
    decoded = _encode_decode(original).batches()
    d = decoded["openmathinstruct"][0].rollouts[1]
    assert d._validated_force_span == (9, 12)
    assert d._validated_termination_path == "bft_forced"
    # The consumer this exists for: BFT tokens masked from the loss.
    keep0 = _completion_keep_list(forced, 7, 6)
    keep1 = _completion_keep_list(d, 7, 6)
    assert keep1 == keep0 and keep1 is not None and False in keep1
    # Rollouts without a span stay span-free (no fabricated masking).
    other = decoded["openmathinstruct"][0].rollouts[0]
    assert getattr(other, "_validated_force_span", None) is None


def test_tombstone_round_trip():
    doc = decode_tombstone(encode_tombstone(
        window_start=30105, failure_stage="proof_capacity",
        failure_type="ProofCapacityAbort",
    ))
    assert doc["window_start"] == 30105
    assert doc["failure_stage"] == "proof_capacity"
    assert doc["failure_type"] == "ProofCapacityAbort"


def test_tombstone_rejects_duplicate_durable_identity():
    encoded = encode_tombstone(
        window_start=30105,
        failure_stage="proof_capacity",
        failure_type="ProofCapacityAbort",
    )
    ambiguous = encoded.replace(
        b'"window_start": 30105',
        b'"window_start": 1, "window_start": 30105',
        1,
    )

    with pytest.raises(ValueError, match="duplicate JSON key: window_start"):
        decode_tombstone(ambiguous)


@pytest.mark.parametrize(
    ("protocol_version", "payload_schema", "tombstone_schema"),
    [
        (4, 1, 1),
        (5, PAYLOAD_SCHEMA_VERSION, TOMBSTONE_SCHEMA_VERSION),
    ],
)
def test_artifact_schema_preserves_legacy_worker_compatibility(
    monkeypatch,
    protocol_version,
    payload_schema,
    tombstone_schema,
):
    monkeypatch.setattr(C, "PROTOCOL_VERSION", protocol_version)

    decoded = _encode_decode(_window_batches())
    tombstone = decode_tombstone(encode_tombstone(
        window_start=30105,
        failure_stage="proof_capacity",
        failure_type="ProofCapacityAbort",
    ))

    assert decoded.schema_version == payload_schema
    assert tombstone["schema_version"] == tombstone_schema
    if protocol_version >= 5:
        expected = active_training_identity()
        assert decoded.training_identity == expected
        for key, value in expected.items():
            assert tombstone[key] == value
    else:
        assert all(
            value is None for value in decoded.training_identity.values()
        )
        assert not set(active_training_identity()).intersection(tombstone)
