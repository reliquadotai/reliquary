"""Batched, right-padded audit gives the same verdicts as one-by-one (measured
on Qwen3.8-27B: TOPLOC tolerances absorb the kernel differences)."""

import base64

import torch

from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF
from reliquary.protocol.toploc import sequence_verdict
from reliquary.protocol.toploc_proof import build_chunk_proofs, verify_chunk_proofs
from reliquary.validator.corpus_audit import (
    batch_completion_hidden_states,
    completion_hidden_states,
)
from tests.unit.test_corpus_audit import _tiny


def _seqs():
    return [(list(range(10, 18)) + list(range(100, 100 + n)), 8) for n in (70, 33, 100, 64)]


def test_batched_rows_have_the_one_by_one_shape_and_verdict():
    model = _tiny(0)
    seqs = _seqs()
    batched = batch_completion_hidden_states(model, seqs)
    for (tokens, n), rows in zip(seqs, batched):
        single = completion_hidden_states(model, tokens, n)
        assert rows.shape == single.shape
        proofs = build_chunk_proofs(single, chunk_tokens=PROOF.chunk_tokens, topk=PROOF.topk)
        ok, _ = sequence_verdict(
            verify_chunk_proofs(rows, proofs, chunk_tokens=PROOF.chunk_tokens, topk=PROOF.topk),
            PROOF.thresholds(),
        )
        assert ok


def test_the_final_hidden_state_is_the_only_one_computed(monkeypatch):
    model = _tiny(0)
    seen = {}
    original = model.forward

    def spy(*args, **kwargs):
        seen["output_hidden_states"] = kwargs.get("output_hidden_states")
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward", spy)
    completion_hidden_states(model, *_seqs()[0])
    assert not seen.get("output_hidden_states")
