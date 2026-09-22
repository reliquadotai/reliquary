"""Our port against bytes and results produced by the reference library
(toploc==0.1.6, C++ path). Regenerate with scripts/toploc_golden.py."""

import json
from pathlib import Path

import pytest
import torch

from reliquary.protocol.toploc_proof import build_chunk_proofs, verify_chunk_proofs

GOLDEN = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "toploc_golden.json").read_text()
)
KW = {"chunk_tokens": GOLDEN["chunk"], "topk": GOLDEN["topk"]}


def _tensor(bits):
    shape = (GOLDEN["rows"], GOLDEN["width"])
    return torch.tensor(bits, dtype=torch.int16).view(torch.bfloat16).reshape(shape)


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: f"seed{c['seed']}")
def test_proof_bytes_match_the_reference(case):
    ours = build_chunk_proofs(_tensor(case["hidden_bits"]), **KW)
    assert [p.hex() for p in ours] == case["proofs_hex"]


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: f"seed{c['seed']}")
@pytest.mark.parametrize("check", ["identical", "perturbed", "unrelated"])
def test_verification_matches_the_reference(case, check):
    expected = case["checks"][check]
    proofs = [bytes.fromhex(p) for p in case["proofs_hex"]]
    ours = verify_chunk_proofs(_tensor(expected["hidden_bits"]), proofs, **KW)
    flat = [v for r in ours for v in (r.exp_mismatches, r.mant_err_mean, r.mant_err_median)]
    assert flat == pytest.approx([v for row in expected["results"] for v in row])
