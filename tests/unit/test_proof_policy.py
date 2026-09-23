"""Which scheme decides a rollout is read from the task's contract alone."""

import dataclasses

import pytest

from reliquary.protocol.profiles import (
    PROFILES,
    PROOF_SCHEME_GRAIL,
    TOPLOC_DEPLOYED_DEFAULTS,
    ProofProfile,
    proof_rejection,
    toploc_proof,
)

BASE = PROFILES[sorted(PROFILES)[0]]
SHADOW = dataclasses.replace(TOPLOC_DEPLOYED_DEFAULTS, mode="shadow")


def _with(*proofs):
    return dataclasses.replace(BASE, proofs=proofs)


def test_a_compiled_profile_has_no_toploc():
    assert toploc_proof(BASE) is None


def test_the_toploc_entry_is_found_whatever_its_mode():
    assert toploc_proof(_with(ProofProfile(PROOF_SCHEME_GRAIL, "enforce"), SHADOW)) == SHADOW


@pytest.mark.parametrize(
    "grail,toploc,expected",
    [(True, None, None), (False, None, "grail_fail"), (True, False, None), (False, True, "grail_fail")],
)
def test_grail_decides_by_default_and_toploc_shadow_never_refuses(grail, toploc, expected):
    profile = _with(ProofProfile(PROOF_SCHEME_GRAIL, "enforce"), SHADOW)
    assert proof_rejection(profile, grail_passed=grail, toploc_passed=toploc) == expected
    assert proof_rejection(BASE, grail_passed=grail, toploc_passed=toploc) == expected


@pytest.mark.parametrize(
    "grail,toploc,expected",
    [(False, True, None), (True, False, "toploc_fail"), (True, None, "toploc_fail")],
)
def test_toploc_enforced_decides_alone_and_missing_proofs_refuse(grail, toploc, expected):
    profile = _with(TOPLOC_DEPLOYED_DEFAULTS)
    assert proof_rejection(profile, grail_passed=grail, toploc_passed=toploc) == expected


def test_two_entries_of_one_scheme_are_refused():
    with pytest.raises(ValueError, match="once"):
        _with(TOPLOC_DEPLOYED_DEFAULTS, SHADOW)
