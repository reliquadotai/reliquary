"""Our port against bytes and results produced by the reference library
(toploc==0.1.6, C++ path). Regenerate with scripts/toploc_golden.py."""

import base64
import json
from pathlib import Path

import pytest
import torch

from reliquary.protocol.toploc_proof import build_chunk_proofs, verify_chunk_proofs

GOLDEN = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "toploc_golden.json").read_text()
)
KW = {"chunk_tokens": GOLDEN["chunk"], "topk": GOLDEN["topk"]}
CHECKS = [
    (case, check) for case in GOLDEN["cases"] for check in case["checks"]
]


def _tensor(case, encoded):
    raw = torch.frombuffer(bytearray(base64.b64decode(encoded)), dtype=torch.int16)
    return raw.view(torch.bfloat16).reshape(case["rows"], case["width"])


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["name"])
def test_proof_bytes_match_the_reference(case):
    ours = build_chunk_proofs(_tensor(case, case["hidden_b64"]), **KW)
    assert [p.hex() for p in ours] == case["proofs_hex"]


@pytest.mark.parametrize(
    "case,check", CHECKS, ids=[f"{c['name']}-{k}" for c, k in CHECKS]
)
def test_verification_matches_the_reference(case, check):
    expected = case["checks"][check]
    proofs = [bytes.fromhex(p) for p in case["proofs_hex"]]
    encoded = expected.get("hidden_b64", case["hidden_b64"])
    ours = verify_chunk_proofs(_tensor(case, encoded), proofs, **KW)
    flat = [v for r in ours for v in (r.exp_mismatches, r.mant_err_mean, r.mant_err_median)]
    assert flat == pytest.approx([v for row in expected["results"] for v in row])


def test_the_fixture_exercises_index_reduction():
    # A 2560-wide model has 81920 activations per chunk, beyond the field, so
    # the reference reduces indices and sometimes steps the modulus down.
    moduli = {int(p[:4], 16) for c in GOLDEN["cases"] for p in c["proofs_hex"]}
    assert any(m < 65497 for m in moduli)
    assert any(c["width"] * GOLDEN["chunk"] > 65497 for c in GOLDEN["cases"])
