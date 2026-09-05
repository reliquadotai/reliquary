"""Public checkpoint state rejects identity-changing integer coercion."""

import pytest
from pydantic import ValidationError

from reliquary.protocol.submission import (
    GrpoBatchState,
    MinerState,
    WindowState,
)


def _grpo_state(**overrides):
    payload = {
        "state": WindowState.OPEN,
        "window_n": 1,
        "anchor_block": 2,
        "valid_submissions": 0,
    }
    payload.update(overrides)
    return payload


def _miner_state(**overrides):
    payload = {
        "state": WindowState.OPEN,
        "window_n": 1,
        "anchor_block": 2,
        "environments": {},
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    "model,payload",
    [
        (GrpoBatchState, _grpo_state),
        (MinerState, _miner_state),
    ],
)
@pytest.mark.parametrize("checkpoint_n", ["7", 7.0, True, "01"])
def test_public_checkpoint_number_rejects_coercible_values(
    model,
    payload,
    checkpoint_n,
):
    with pytest.raises(ValidationError, match="checkpoint_n"):
        model.model_validate(payload(checkpoint_n=checkpoint_n))


@pytest.mark.parametrize(
    "model,payload",
    [
        (GrpoBatchState, _grpo_state),
        (MinerState, _miner_state),
    ],
)
@pytest.mark.parametrize("checkpoint_n", [0, 7, None])
def test_public_checkpoint_number_accepts_canonical_values(
    model,
    payload,
    checkpoint_n,
):
    state = model.model_validate(payload(checkpoint_n=checkpoint_n))
    assert state.checkpoint_n == checkpoint_n


@pytest.mark.parametrize(
    "model,payload",
    [
        (GrpoBatchState, _grpo_state),
        (MinerState, _miner_state),
    ],
)
def test_public_checkpoint_number_remains_optional_for_legacy_state(
    model,
    payload,
):
    assert model.model_validate(payload()).checkpoint_n is None


@pytest.mark.parametrize(
    "model,payload",
    [
        (GrpoBatchState, _grpo_state),
        (MinerState, _miner_state),
    ],
)
def test_public_checkpoint_number_rejects_negative_integer(model, payload):
    with pytest.raises(ValidationError, match="checkpoint_n"):
        model.model_validate(payload(checkpoint_n=-1))
