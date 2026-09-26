"""A task's contract names how its work is proven. Absent means GRAIL as today,
so no compiled profile's contract bytes move."""

import dataclasses

import pytest

from reliquary.protocol.profiles import (
    PROFILES,
    PROOF_SCHEME_GRAIL,
    PROOF_SCHEME_TOPLOC,
    TOPLOC_DEPLOYED_DEFAULTS,
    ProofProfile,
    enforced_proof,
    profile_from_contract,
)


def _any():
    return PROFILES[sorted(PROFILES)[0]]


def test_the_deployed_defaults_are_prime_intellects():
    p = TOPLOC_DEPLOYED_DEFAULTS
    assert (p.scheme, p.mode, p.chunk_tokens, p.topk) == (PROOF_SCHEME_TOPLOC, "enforce", 32, 128)
    t = p.thresholds()
    assert (t.exp_mismatch, t.mant_mean, t.mant_median) == (60, 40.0, 40.0)
    assert (t.min_allowed_failures, t.ratio_allowed_failures) == (0, 0.0)


@pytest.mark.parametrize("profile_id", sorted(PROFILES))
def test_compiled_contracts_carry_no_proofs_key(profile_id):
    assert "proofs" not in PROFILES[profile_id].to_generation_contract()


def test_absent_means_grail_enforced():
    assert enforced_proof(_any()) == ProofProfile(PROOF_SCHEME_GRAIL, "enforce")


def test_a_contract_with_proofs_round_trips():
    grail_shadow_toploc = (
        ProofProfile(PROOF_SCHEME_GRAIL, "enforce"),
        dataclasses.replace(TOPLOC_DEPLOYED_DEFAULTS, mode="shadow"),
    )
    profile = dataclasses.replace(_any(), proofs=grail_shadow_toploc)
    contract = profile.to_generation_contract()
    assert [p["scheme"] for p in contract["proofs"]] == [PROOF_SCHEME_GRAIL, PROOF_SCHEME_TOPLOC]
    rebuilt = profile_from_contract(contract)
    assert rebuilt == profile
    assert enforced_proof(rebuilt).scheme == PROOF_SCHEME_GRAIL


def test_two_enforced_schemes_are_refused():
    with pytest.raises(ValueError):
        dataclasses.replace(
            _any(), proofs=(TOPLOC_DEPLOYED_DEFAULTS, ProofProfile(PROOF_SCHEME_GRAIL, "enforce"))
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scheme": "sha1", "mode": "enforce"},
        {"scheme": PROOF_SCHEME_GRAIL, "mode": "maybe"},
        {"scheme": PROOF_SCHEME_TOPLOC, "mode": "enforce"},           # thresholds missing
        {"scheme": PROOF_SCHEME_GRAIL, "mode": "enforce", "topk": 128},  # toploc field on grail
    ],
)
def test_an_incoherent_proof_is_refused(kwargs):
    with pytest.raises(ValueError):
        ProofProfile(**kwargs)


def test_an_unknown_proof_key_is_refused_on_read():
    contract = dataclasses.replace(_any(), proofs=(TOPLOC_DEPLOYED_DEFAULTS,)).to_generation_contract()
    contract["proofs"][0]["salt"] = 1
    with pytest.raises(ValueError, match="unknown fields"):
        profile_from_contract(contract)


def test_a_bool_threshold_is_refused_on_read():
    contract = dataclasses.replace(_any(), proofs=(TOPLOC_DEPLOYED_DEFAULTS,)).to_generation_contract()
    contract["proofs"][0]["exp_mismatch_threshold"] = True
    with pytest.raises(ValueError):
        profile_from_contract(contract)


@pytest.mark.parametrize(
    "field,value",
    [
        ("chunk_tokens", 0),
        ("chunk_tokens", -4),
        ("topk", 0),
        ("topk", -1),
        ("exp_mismatch_threshold", -1),
        ("mant_mean_threshold", -0.5),
        ("mant_median_threshold", float("nan")),
        ("mant_mean_threshold", float("inf")),
        ("min_allowed_failures", -3),
        ("ratio_allowed_failures", 5.0),
        ("ratio_allowed_failures", -0.1),
        ("topk", True),
    ],
)
def test_a_toploc_value_that_would_fail_every_honest_miner_is_refused(field, value):
    with pytest.raises(ValueError):
        dataclasses.replace(TOPLOC_DEPLOYED_DEFAULTS, **{field: value})
