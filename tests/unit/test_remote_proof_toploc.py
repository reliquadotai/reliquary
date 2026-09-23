"""The remote proof plane is another machine: the TOPLOC proofs and the
validator's toploc_spec must cross the wire, and nothing must change when the
contract has no toploc."""

from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS
from reliquary.validator.remote_proof_protocol import ProofInput, ProofValues
from reliquary.validator.verifier import ProofResult

SPEC = TOPLOC_DEPLOYED_DEFAULTS.to_contract()


def _input(**extra):
    return ProofInput(tokens=[1, 2, 3], commitments=[{"sketch": 0}] * 3, rollout={},
                      randomness="ab", seed_u_values=None, **extra)


def test_without_toploc_the_commit_keeps_its_old_shape():
    assert set(_input().commit()) == {"tokens", "commitments", "rollout"}


def test_toploc_crosses_the_wire():
    payload = _input(toploc_proofs=["/9kAAQ=="], toploc_spec=SPEC)
    reread = ProofInput.model_validate_json(payload.model_dump_json())
    commit = reread.commit()
    assert commit["toploc_proofs"] == ["/9kAAQ=="]
    assert commit["toploc_spec"] == SPEC


def test_the_toploc_verdict_survives_the_result_wire():
    result = ProofResult(all_passed=True, passed=1, checked=1, has_sparse_outputs=True,
                         toploc_checked=True,
                         toploc_passed=False, toploc_reason="exp_mismatch",
                         toploc_worst_exp=90, toploc_worst_mant_mean=12.5,
                         toploc_worst_mant_median=11.0)
    back = ProofValues.from_kernel(result).to_kernel()
    assert (back.toploc_checked, back.toploc_passed, back.toploc_reason) == (True, False, "exp_mismatch")
    assert (back.toploc_worst_exp, back.toploc_worst_mant_mean) == (90, 12.5)
