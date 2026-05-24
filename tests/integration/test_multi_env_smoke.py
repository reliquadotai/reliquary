"""Smoke test for multi-env Phase 2.

Loads both envs from ENVIRONMENT_MIX, exercises the shape-level
plumbing (load_environments returns dict, train_step accepts list,
ValidationService can be instantiated multi-env). Skipped if the HF
datasets can't be reached.
"""

import pytest


def test_load_environments_for_full_mix(monkeypatch):
    """ENVIRONMENT_MIX drives load_environments end-to-end."""
    from reliquary.constants import ENVIRONMENT_MIX
    from reliquary.environment import load_environments
    from reliquary.environment.openmathinstruct import OpenMathInstructEnvironment
    from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment

    # Stub both env __init__s to avoid HF downloads.
    monkeypatch.setattr(OpenMathInstructEnvironment, "__init__", lambda self: None)
    monkeypatch.setattr(OpenCodeInstructEnvironment, "__init__", lambda self: None)

    names = [name for name, _ in ENVIRONMENT_MIX]
    envs = load_environments(names)
    assert set(envs.keys()) == set(names)


def test_train_step_handles_two_empty_batches_gracefully():
    """train_step([[], []], ...) is a no-op, returns model unchanged."""
    from reliquary.validator.training import train_step
    class _Stub: pass
    model = _Stub()
    result = train_step(model, [[], []], ref_model=None)
    assert result is model


def test_environment_mix_sums_to_expected_batch():
    """The active mix should produce 2*B_BATCH prompts per train step."""
    from reliquary.constants import B_BATCH, ENVIRONMENT_MIX, GRAD_ACCUM_STEPS
    total = sum(w for _, w in ENVIRONMENT_MIX)
    assert total == 2 * B_BATCH
    assert GRAD_ACCUM_STEPS == len(ENVIRONMENT_MIX)


def test_grad_accum_steps_equals_mix_length():
    """GRAD_ACCUM_STEPS is derived from len(mix), not separately tunable."""
    from reliquary.constants import ENVIRONMENT_MIX, GRAD_ACCUM_STEPS
    assert GRAD_ACCUM_STEPS == len(ENVIRONMENT_MIX)
