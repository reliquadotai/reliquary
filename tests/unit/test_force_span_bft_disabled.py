"""Forced-span claims must fail closed when the active profile has no BFT.

On v4 no honest rollout is ever ``forced``: the profile carries ``bft=None`` so
the miner never applies a force template. But ``forced`` and ``force_span``
remain wire fields a submission can set, and both validating predicates key off
budgets that are now 0 — so acceptance would rest on degenerate arithmetic
rather than on an explicit decision.

That matters because a valid span is not merely accepted, it *exempts* its
positions from the per-token authenticity and distribution checks
(``validate_force_span`` returns them as ``exempt``). A crafted forced claim is
therefore a way to carve tokens out of verification.

The fixtures below reuse the geometry of ``test_bft_carveout.py`` — atomic
``</think>`` = 777, canonical FORCE ids ``[777, 7, 8]``, prompt_length 2,
thinking budget 2 — which that file asserts is **accepted**. So in these tests
the BFT_ENABLED guard is the only thing that can reject it.
"""
from types import SimpleNamespace

import reliquary.constants as C
import reliquary.validator.admission as admission
from reliquary.validator.admission import _force_span_valid
from reliquary.validator.verifier import validate_force_span

_FORCE = [777, 7, 8]
_CLOSE = {777}
# prompt[0,1] + thinking[5,6] + force[777,7,8] + answer[55,99]
_TOKENS = [0, 1, 5, 6, 777, 7, 8, 55, 99]
_META = {"forced": True, "force_span": (4, 7), "prompt_length": 2}


def test_validate_force_span_rejects_forced_claim_when_bft_disabled(monkeypatch):
    monkeypatch.setattr(C, "BFT_ENABLED", False)

    ok, exempt = validate_force_span(
        _TOKENS, _META, _FORCE, 2, thinking_budget=2, think_close_ids=_CLOSE,
    )

    assert ok is False
    assert exempt == set()


def test_validate_force_span_leaves_non_forced_rollouts_alone(monkeypatch):
    monkeypatch.setattr(C, "BFT_ENABLED", False)

    ok, exempt = validate_force_span(
        [0, 1, 5, 6], {"forced": False}, _FORCE, 2,
        thinking_budget=2, think_close_ids=_CLOSE,
    )

    assert ok is True
    assert exempt == set()


def test_admission_rejects_forced_claim_when_bft_disabled(monkeypatch):
    monkeypatch.setattr(C, "BFT_ENABLED", False)
    # Module-top import in admission.py: patch the consuming module so the
    # fixture geometry stays valid and the guard is the only rejection cause.
    monkeypatch.setattr(admission, "BFT_THINKING_BUDGET", 2)
    ctx = SimpleNamespace(think_close_ids=_CLOSE, canonical_force_ids=_FORCE)

    assert _force_span_valid(_TOKENS, _META, ctx) is False


def test_admission_leaves_non_forced_rollouts_alone(monkeypatch):
    monkeypatch.setattr(C, "BFT_ENABLED", False)
    ctx = SimpleNamespace(think_close_ids=_CLOSE, canonical_force_ids=_FORCE)

    assert _force_span_valid([0, 1], {"forced": False}, ctx) is True
