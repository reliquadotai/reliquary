"""Pipelined window collection — orchestration units.

The loop itself is integration-scale; these tests pin the load-bearing
pieces: the flag default, the routing-ownership guard of the parameterized
GPU half, and the deferred verify-swap lifecycle (normal application at the
end of the capturing GPU half, and the abort safety net that certifies the
new revision before proving a post-publish window).
"""
import asyncio
from types import SimpleNamespace

import pytest

from reliquary.validator.service import ValidationService


def test_flag_defaults_off():
    from reliquary.constants import PIPELINED_WINDOWS
    assert PIPELINED_WINDOWS is False


class _Stub:
    """Minimal host for the real unbound coroutines."""

    # bind the real implementations so internal self.* calls resolve
    _apply_deferred_verify_swap = ValidationService._apply_deferred_verify_swap

    def __init__(self):
        self._pending_verify_swap = None
        self._pending_verify_swap_age = 0
        self._active_batchers = {"env": "SENTINEL"}
        self._window_n = 999
        self.proof_scheduler = None
        self.refreshed = []

    def _refresh_verify_model_from_train(self, revision):
        self.refreshed.append(revision)

    def _synchronize_proof_models(self, revision):
        raise AssertionError("no scheduler -> must not be called")


def _run(coro):
    return asyncio.run(coro)


def test_gpu_half_with_explicit_batchers_never_touches_routing():
    stub = _Stub()
    _run(ValidationService._train_and_publish(stub, batchers={}, window_n=42))
    # early return on empty batchers; the collecting window's routing state
    # must be untouched (owns_routing is False).
    assert stub._active_batchers == {"env": "SENTINEL"}


def test_deferred_swap_helper_applies_and_clears():
    stub = _Stub()
    stub._pending_verify_swap = "a" * 40
    stub._pending_verify_swap_age = 1
    _run(ValidationService._apply_deferred_verify_swap(stub, "a" * 40))
    assert stub.refreshed == ["a" * 40]
    assert stub._pending_verify_swap is None
    assert stub._pending_verify_swap_age == 0


def test_abort_safety_net_certifies_before_proving():
    """If the GPU half that captured the swap aborted, the NEXT one must
    apply it BEFORE its proofs (its window pinned the new revision)."""
    stub = _Stub()
    stub._pending_verify_swap = "b" * 40

    # first call captures (age 1), early-returns on empty batchers, applies
    # at its end because pending_swap_due was captured.
    _run(ValidationService._train_and_publish(stub, batchers={}, window_n=1))
    # NOTE: the empty-batchers early return exits BEFORE the end-of-half
    # application, mimicking an aborted half — pending must survive it.
    assert stub._pending_verify_swap == "b" * 40
    assert stub._pending_verify_swap_age == 1

    # second call: age reaches 2 -> safety net applies the swap up front.
    _run(ValidationService._train_and_publish(stub, batchers={}, window_n=2))
    assert stub.refreshed == ["b" * 40]
    assert stub._pending_verify_swap is None
