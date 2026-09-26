"""The batcher reads the contract: the enforced scheme refuses, shadow records."""

import dataclasses

import pytest
import torch

from reliquary.protocol.profiles import (
    ACTIVE_PROTOCOL_PROFILE,
    PROOF_SCHEME_GRAIL,
    TOPLOC_DEPLOYED_DEFAULTS,
    ProofProfile,
)
from reliquary.protocol.submission import RejectReason
from reliquary.validator import batcher as batcher_mod
from reliquary.validator.verifier import ProofResult
from tests.unit.test_grpo_window_batcher import _make_batcher, _prove_one, _request

SHADOW = dataclasses.replace(TOPLOC_DEPLOYED_DEFAULTS, mode="shadow")


def _contract(monkeypatch, *proofs):
    profile = dataclasses.replace(ACTIVE_PROTOCOL_PROFILE, proofs=proofs)
    monkeypatch.setattr(batcher_mod, "_active_profile", lambda: profile)


def _stub(*, grail, toploc, seen=None):
    def verify(commit, model, randomness):
        if seen is not None:
            seen.append(commit)
        return ProofResult(all_passed=grail, passed=int(grail), checked=1,
                           logits=torch.empty(0), toploc_checked=toploc is not None,
                           toploc_passed=bool(toploc), toploc_reason=None if toploc else "exp_mismatch")
    return verify


def test_toploc_enforced_decides_even_when_grail_fails(monkeypatch):
    _contract(monkeypatch, TOPLOC_DEPLOYED_DEFAULTS)
    b = _make_batcher(verify_commitment_proofs_fn=_stub(grail=False, toploc=True))
    assert _prove_one(b, _request()) is not None


def test_toploc_enforced_refuses_a_failed_proof(monkeypatch):
    _contract(monkeypatch, TOPLOC_DEPLOYED_DEFAULTS)
    b = _make_batcher(verify_commitment_proofs_fn=_stub(grail=True, toploc=False))
    assert _prove_one(b, _request()) is None
    assert b.reject_counts[RejectReason.TOPLOC_FAIL.value] == 1


def test_toploc_shadow_never_refuses(monkeypatch):
    _contract(monkeypatch, ProofProfile(PROOF_SCHEME_GRAIL, "enforce"), SHADOW)
    b = _make_batcher(verify_commitment_proofs_fn=_stub(grail=True, toploc=False))
    assert _prove_one(b, _request()) is not None


def test_the_verifier_gets_the_validators_spec_only_when_the_contract_names_toploc(monkeypatch):
    seen = []
    _contract(monkeypatch, SHADOW)
    b = _make_batcher(verify_commitment_proofs_fn=_stub(grail=True, toploc=True, seen=seen))
    _prove_one(b, _request())
    assert seen and all(c["toploc_spec"] == SHADOW.to_contract() for c in seen)

    seen.clear()
    _contract(monkeypatch)
    b = _make_batcher(verify_commitment_proofs_fn=_stub(grail=True, toploc=None, seen=seen))
    _prove_one(b, _request())
    assert seen and all("toploc_spec" not in c for c in seen)
