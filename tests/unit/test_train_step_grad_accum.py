"""Tests for train_step's multi-batch (grad accumulation) mode."""

import inspect
import pytest


def test_train_step_accepts_list_of_batches_signature():
    """train_step's signature accepts a list of batches as the second arg."""
    from reliquary.validator.training import train_step
    sig = inspect.signature(train_step)
    params = list(sig.parameters.keys())
    assert params[0] == "model"
    assert params[1] == "batches"


def test_train_step_empty_batches_returns_model():
    """No data → no work, no crash."""
    from reliquary.validator.training import train_step

    class _Stub:
        pass

    model = _Stub()
    result = train_step(model, [], ref_model=None)
    assert result is model

    result = train_step(model, [[], []], ref_model=None)
    assert result is model
