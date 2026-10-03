"""Per-turn rows from one prefill of a whole trajectory: a span's rows must be
the ones a single-turn audit of that turn would take."""

import base64

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF
from reliquary.protocol.toploc_proof import build_chunk_proofs
from reliquary.validator.corpus_audit import (
    audit_completion,
    completion_hidden_states,
    span_hidden_states,
)


def _tiny(seed):
    torch.manual_seed(seed)
    config = AutoConfig.for_model(
        "qwen3", hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, vocab_size=512,
    )
    return AutoModelForCausalLM.from_config(config).to(torch.bfloat16).eval()


# prompt 0..8, turn A 8..40, observation 40..55, turn B 55..90
TOKENS = list(range(10, 100))
SPANS = [(8, 40), (55, 90)]


def test_each_span_equals_a_single_turn_audit_of_its_prefix():
    model = _tiny(0)
    rows = span_hidden_states(model, TOKENS, SPANS)
    for (start, end), got in zip(SPANS, rows):
        want = completion_hidden_states(model, TOKENS[:end], start)
        assert got.shape == want.shape == (end - start, 128)
        assert torch.equal(got, want)


def test_proofs_built_per_turn_pass_per_span():
    model = _tiny(0)
    for got in span_hidden_states(model, TOKENS, SPANS):
        proofs = [base64.b64encode(p).decode()
                  for p in build_chunk_proofs(got, chunk_tokens=PROOF.chunk_tokens, topk=PROOF.topk)]
        assert audit_completion(got, proofs, PROOF).passed


def test_span_touching_the_end_reads_no_padding():
    model = _tiny(0)
    (got,) = span_hidden_states(model, TOKENS, [(80, len(TOKENS))])
    assert got.shape == (10, 128)


@pytest.mark.parametrize("spans", [[(0, 5)], [(5, 4)], [(10, 20), (15, 30)], [(10, 200)]])
def test_malformed_spans_are_refused(spans):
    with pytest.raises(ValueError):
        span_hidden_states(_tiny(0), TOKENS, spans)
