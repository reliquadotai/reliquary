"""The audit's model-identity check on tiny random models, CPU only: the
proofs a model builds from its own prefill must pass, another model's fail."""

import base64

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF
from reliquary.protocol.toploc_proof import build_chunk_proofs
from reliquary.validator.corpus_audit import audit_completion, completion_hidden_states


def _tiny(seed):
    torch.manual_seed(seed)
    config = AutoConfig.for_model(
        "qwen3", hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, vocab_size=512,
    )
    return AutoModelForCausalLM.from_config(config).to(torch.bfloat16).eval()


TOKENS = list(range(10, 10 + 8 + 70))    # 8 prompt tokens, 70 completion tokens
PROMPT_LEN = 8


def _proofs(model):
    hidden = completion_hidden_states(model, TOKENS, PROMPT_LEN)
    raw = build_chunk_proofs(hidden, chunk_tokens=PROOF.chunk_tokens, topk=PROOF.topk)
    return [base64.b64encode(p).decode() for p in raw]


def test_rows_are_the_positions_that_produced_the_completion():
    hidden = completion_hidden_states(_tiny(0), TOKENS, PROMPT_LEN)
    assert tuple(hidden.shape) == (70, 128)


def test_a_model_passes_on_its_own_proofs():
    model = _tiny(0)
    hidden = completion_hidden_states(model, TOKENS, PROMPT_LEN)
    outcome = audit_completion(hidden, _proofs(model), PROOF)
    assert (outcome.passed, outcome.reason) == (True, None)
    assert len(outcome.results) == 3


def test_another_models_proofs_fail():
    hidden = completion_hidden_states(_tiny(0), TOKENS, PROMPT_LEN)
    outcome = audit_completion(hidden, _proofs(_tiny(1)), PROOF)
    assert outcome.passed is False
    assert outcome.reason == "exp_mismatch"


def test_a_missing_proof_fails_on_shape():
    model = _tiny(0)
    hidden = completion_hidden_states(model, TOKENS, PROMPT_LEN)
    outcome = audit_completion(hidden, _proofs(model)[:-1], PROOF)
    assert (outcome.passed, outcome.reason) == (False, "bad_proof_shape")


def test_undecodable_base64_fails_closed():
    model = _tiny(0)
    hidden = completion_hidden_states(model, TOKENS, PROMPT_LEN)
    proofs = _proofs(model)
    proofs[0] = "@@@"
    outcome = audit_completion(hidden, proofs, PROOF)
    assert (outcome.passed, outcome.reason) == (False, "proof_undecodable")


def test_a_grail_proof_profile_is_not_audited_here():
    from reliquary.protocol.profiles import ProofProfile

    with pytest.raises(ValueError):
        audit_completion(torch.zeros(1, 128), [], ProofProfile("grail-v7", "enforce"))


def test_a_prompt_length_outside_the_sequence_is_refused():
    with pytest.raises(ValueError):
        completion_hidden_states(_tiny(0), TOKENS, len(TOKENS))


def test_a_token_outside_the_vocabulary_is_refused_before_the_model_runs():
    # On CUDA an out-of-range embedding index kills the device context.
    with pytest.raises(ValueError, match="vocabulary"):
        completion_hidden_states(_tiny(0), [1, 2, 3, 512], 2)


def test_a_topk_wider_than_the_model_is_a_validator_error_not_the_miners():
    import dataclasses

    hidden = completion_hidden_states(_tiny(0), TOKENS, PROMPT_LEN)
    too_wide = dataclasses.replace(PROOF, topk=129)   # a 1-row chunk has 128 activations
    with pytest.raises(ValueError, match="configuration"):
        audit_completion(hidden[:33], ["AAAA"] * 2, too_wide)
