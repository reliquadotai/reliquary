"""TOPLOC checked on the validator's full-sequence hidden states, driven only
by the toploc_spec the validator itself put in the commit copy."""

import torch

from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF
from reliquary.protocol.toploc_proof import completion_proofs_b64
from reliquary.validator.toploc_check import toploc_verdict

KW = {"chunk_tokens": PROOF.chunk_tokens, "topk": PROOF.topk}
SPEC = PROOF.to_contract()


def _hidden(seed, rows=12 + 70, width=256):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, width, generator=g).to(torch.bfloat16)


def _commit(proofs=None, spec=SPEC):
    commit = {"tokens": list(range(82))}
    if spec is not None:
        commit["toploc_spec"] = spec
    if proofs is not None:
        commit["toploc_proofs"] = proofs
    return commit


def test_no_spec_means_no_toploc_check():
    assert toploc_verdict(_hidden(0), _commit(spec=None), 12) is None


def test_honest_proofs_pass():
    hidden = _hidden(0)
    verdict = toploc_verdict(hidden, _commit(completion_proofs_b64(hidden, 12, 82, **KW)), 12)
    assert (verdict.passed, verdict.reason, verdict.worst_exp) == (True, None, 0)


def test_missing_proofs_fail():
    verdict = toploc_verdict(_hidden(0), _commit(), 12)
    assert (verdict.passed, verdict.reason) == (False, "missing")


def test_another_models_proofs_fail():
    verdict = toploc_verdict(_hidden(0), _commit(completion_proofs_b64(_hidden(1), 12, 82, **KW)), 12)
    assert verdict.passed is False
    assert verdict.worst_exp > 60


def test_miscounted_proofs_fail_on_shape():
    hidden = _hidden(0)
    proofs = completion_proofs_b64(hidden, 12, 82, **KW)[:-1]
    assert toploc_verdict(hidden, _commit(proofs), 12).reason == "bad_proof_shape"


def test_a_shadow_configuration_error_is_recorded_not_raised():
    # topk wider than the model: the validator's own mistake. In shadow it must
    # not take the GRAIL proof down with it.
    import dataclasses

    spec = dataclasses.replace(PROOF, mode="shadow", topk=257).to_contract()
    hidden = _hidden(0)
    verdict = toploc_verdict(hidden, _commit(completion_proofs_b64(hidden, 12, 82, **KW), spec=spec), 12)
    assert verdict.passed is False
    assert verdict.reason == "error:ValueError"


def test_an_enforced_configuration_error_stays_loud():
    import dataclasses

    import pytest

    spec = dataclasses.replace(PROOF, topk=257).to_contract()
    hidden = _hidden(0)
    with pytest.raises(ValueError, match="configuration"):
        toploc_verdict(hidden, _commit(completion_proofs_b64(hidden, 12, 82, **KW), spec=spec), 12)
