"""TOPLOC over every model span and the per-turn forced stop pick (plan 2C, Task 9). CPU, mocked forward."""
from unittest.mock import MagicMock, patch

import torch

from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO
from reliquary.environment import forced_sampling as fs
from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF
from reliquary.protocol.submission import SIGNED_EPISODE_SCHEMA
from reliquary.protocol.toploc_proof import completion_proofs_b64, span_proofs_b64
from reliquary.validator import verifier
from reliquary.validator.remote_proof_protocol import ProofValues
from tests.unit.episode_v2_fixtures import episode_contract, episode_pool

HIDDEN, VOCAB, STOP, OTHER = 128, 8, 3, 6
RANDOMNESS = "aa" * 32
PROMPT, SEQ = 2, 25
SPANS = [(2, 12), (15, 25)]            # 20 model tokens; positions 12..14 are a tool output
SEED = 5
POSITIONS = [t for start, end in SPANS for t in range(start, end)]
POOL = episode_pool(episode_contract())
U = [POOL.uniform(SEED, j) for j in range(len(POSITIONS))]


def _peaked(token):
    row = torch.full((VOCAB,), -10.0)
    row[token] = 10.0
    return row


def _episode(*, stop_rows=None):
    """Logits rows (row t-1 predicts token t), the honest tokens drawn from them with the seed's
    uniforms indexed by MODEL-token offset across turns, and the hidden rows."""
    torch.manual_seed(0)
    logits = torch.zeros(SEQ, VOCAB)
    for (_, end), token in zip(SPANS, stop_rows or (STOP, STOP)):
        logits[end - 2] = _peaked(token)                 # the row that draws the turn's last token
    tokens = [0, 1] + [OTHER] * (SEQ - PROMPT)
    for offset, t in enumerate(POSITIONS):
        probs = fs.warp(logits[t - 1], t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO)
        tokens[t] = fs.pick(probs, U[offset])
    for t in range(12, 15):
        tokens[t] = 7                                     # tool output: no draw, no position
    return logits, tokens, torch.randn(SEQ, HIDDEN)


def _commit(tokens, hidden, *, schema=SIGNED_EPISODE_SCHEMA, proofs=None):
    if proofs is None:
        proofs = span_proofs_b64(hidden, SPANS, chunk_tokens=PROOF.chunk_tokens, topk=PROOF.topk)
    return {"tokens": tokens, "commitments": [{"sketch": 0}] * len(tokens),
            "toploc_spec": PROOF.to_contract(), "toploc_proofs": proofs,
            "rollout": {"prompt_length": PROMPT, "completion_length": SEQ - PROMPT,
                        "episode": {"schema_version": schema, "assistant_spans": [list(s) for s in SPANS]}}}


def _verify(commit, logits, hidden):
    model = MagicMock()
    model.parameters.return_value = iter([torch.zeros(1)])
    with patch("reliquary.shared.forward.forward_single_layer", return_value=(hidden[None], logits[None])), \
            patch("reliquary.shared.hf_compat.resolve_hidden_size", return_value=HIDDEN), \
            patch.object(verifier, "resolve_eos_token_ids", lambda model, tokenizer: {STOP}):
        return verifier.verify_commitment_proofs(commit, model, RANDOMNESS, seed_u_values=U)


def test_an_honest_episode_passes_toploc_per_span_the_forced_seed_and_its_stops():
    logits, tokens, hidden = _episode()
    result = _verify(_commit(tokens, hidden), logits, hidden)
    assert result.toploc_checked and result.toploc_passed, result.toploc_reason
    assert result.seed_n_positions == len(POSITIONS)
    assert result.seed_n_stochastic > 0 and result.seed_n_match == result.seed_n_stochastic
    assert result.episode_stop_picks_ok is True and result.episode_stop_first_bad_turn is None
    assert len(result.completion_chosen_probs) == len(POSITIONS)


def test_a_forged_model_token_breaks_the_forced_seed():
    logits, tokens, hidden = _episode()
    probs = fs.warp(logits[4], t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO)
    tokens[5] = next(token for token in range(VOCAB) if token != tokens[5] and probs[token] > 0)
    result = _verify(_commit(tokens, hidden), logits, hidden)
    assert result.seed_n_match == result.seed_n_stochastic - 1


def test_a_stop_token_the_draw_did_not_pick_fails_its_turn():
    logits, tokens, hidden = _episode(stop_rows=(OTHER, STOP))   # turn 0's draw picks OTHER...
    tokens[SPANS[0][1] - 1] = STOP                               # ...and the miner wrote a stop
    result = _verify(_commit(tokens, hidden), logits, hidden)
    assert result.episode_stop_picks_ok is False and result.episode_stop_first_bad_turn == 0


def test_a_turn_cut_by_its_cap_has_no_stop_to_check():
    logits, tokens, hidden = _episode(stop_rows=(STOP, OTHER))   # the last turn ends on OTHER: its cap
    result = _verify(_commit(tokens, hidden), logits, hidden)
    assert result.episode_stop_picks_ok is True
    logits, tokens, hidden = _episode(stop_rows=(OTHER, OTHER))
    assert _verify(_commit(tokens, hidden), logits, hidden).episode_stop_picks_ok is None


def test_a_proof_of_other_activations_fails_toploc():
    logits, tokens, hidden = _episode()
    forged = span_proofs_b64(torch.randn(SEQ, HIDDEN), SPANS, chunk_tokens=PROOF.chunk_tokens, topk=PROOF.topk)
    result = _verify(_commit(tokens, hidden, proofs=forged), logits, hidden)
    assert result.toploc_checked and not result.toploc_passed


def test_a_legacy_episode_keeps_the_contiguous_toploc_and_no_stop_check():
    logits, tokens, hidden = _episode()
    proofs = completion_proofs_b64(hidden, PROMPT, SEQ, chunk_tokens=PROOF.chunk_tokens, topk=PROOF.topk)
    result = _verify(_commit(tokens, hidden, schema="reliquary/episode/v1", proofs=proofs), logits, hidden)
    assert result.toploc_checked and result.toploc_passed
    assert result.episode_stop_picks_ok is None


def test_the_spans_of_a_signed_episode_are_read_strictly():
    meta = {"episode": {"schema_version": SIGNED_EPISODE_SCHEMA, "assistant_spans": [[2, 12], [15, 25]]}}
    assert verifier.signed_episode_spans(meta, SEQ) == SPANS
    assert verifier.signed_episode_spans({"episode": {"schema_version": "reliquary/episode/v1"}}, SEQ) is None
    assert verifier.signed_episode_spans({}, SEQ) is None
    for bad in ([[0, 3]], [[2, 12], [10, 25]], [[2, 26]], [[2]]):
        assert verifier.signed_episode_spans({"episode": {"schema_version": SIGNED_EPISODE_SCHEMA,
                                                          "assistant_spans": bad}}, SEQ) == []


def test_the_remote_proof_wire_carries_the_stop_verdict():
    logits, tokens, hidden = _episode(stop_rows=(OTHER, STOP))
    tokens[SPANS[0][1] - 1] = STOP
    result = _verify(_commit(tokens, hidden), logits, hidden)
    back = ProofValues.from_kernel(result).to_kernel()
    assert (back.episode_stop_picks_ok, back.episode_stop_first_bad_turn) == (False, 0)


# --- Added by the implementer: edges the brief's tests did not pin. ---------------------------------

import base64

import pytest

from reliquary.protocol.toploc_proof import build_span_proofs
from reliquary.validator.toploc_check import toploc_span_verdict


def test_a_later_turn_s_forged_stop_names_that_turn():
    logits, tokens, hidden = _episode(stop_rows=(STOP, OTHER))   # turn 0 stops honestly, turn 1's draw is OTHER
    tokens[SPANS[1][1] - 1] = STOP
    result = _verify(_commit(tokens, hidden), logits, hidden)
    assert (result.episode_stop_picks_ok, result.episode_stop_first_bad_turn) == (False, 1)


def test_a_stop_the_u_stream_does_not_reach_fails_its_turn():
    logits, tokens, hidden = _episode()
    model = MagicMock()
    model.parameters.return_value = iter([torch.zeros(1)])
    with patch("reliquary.shared.forward.forward_single_layer", return_value=(hidden[None], logits[None])), \
            patch("reliquary.shared.hf_compat.resolve_hidden_size", return_value=HIDDEN), \
            patch.object(verifier, "resolve_eos_token_ids", lambda model, tokenizer: {STOP}):
        result = verifier.verify_commitment_proofs(_commit(tokens, hidden), model, RANDOMNESS,
                                                   seed_u_values=U[:len(POSITIONS) - 1])
    assert (result.episode_stop_picks_ok, result.episode_stop_first_bad_turn) == (False, 1)


def test_no_u_stream_means_no_stop_verdict():
    logits, tokens, hidden = _episode()
    model = MagicMock()
    model.parameters.return_value = iter([torch.zeros(1)])
    with patch("reliquary.shared.forward.forward_single_layer", return_value=(hidden[None], logits[None])), \
            patch("reliquary.shared.hf_compat.resolve_hidden_size", return_value=HIDDEN), \
            patch.object(verifier, "resolve_eos_token_ids", lambda model, tokenizer: {STOP}):
        result = verifier.verify_commitment_proofs(_commit(tokens, hidden), model, RANDOMNESS, seed_u_values=None)
    assert result.episode_stop_picks_ok is None and result.toploc_passed


def test_proofs_of_rows_shifted_by_one_fail_toploc():
    logits, tokens, hidden = _episode()
    shifted = span_proofs_b64(torch.cat([hidden[1:], hidden[:1]]), SPANS,
                              chunk_tokens=PROOF.chunk_tokens, topk=PROOF.topk)
    result = _verify(_commit(tokens, hidden, proofs=shifted), logits, hidden)
    assert result.toploc_checked and not result.toploc_passed


def test_span_proofs_are_the_rows_that_drew_each_span_concatenated():
    hidden = torch.randn(SEQ, HIDDEN)
    expected = [base64.b64encode(p).decode() for start, end in SPANS
                for p in build_span_proofs(hidden[start - 1:end - 1], chunk_tokens=4, topk=PROOF.topk,
                                           min_chunk_tokens=2)]
    assert len(expected) == 3 + 3
    assert span_proofs_b64(hidden, SPANS, chunk_tokens=4, topk=PROOF.topk, min_chunk_tokens=2) == expected
    for bad in ([(0, 3)], [(3, 3)], [(2, SEQ + 1)]):
        with pytest.raises(ValueError):
            span_proofs_b64(hidden, bad, chunk_tokens=8, topk=PROOF.topk)


def test_the_span_verdict_fails_closed():
    hidden = torch.randn(SEQ, HIDDEN)
    proofs = span_proofs_b64(hidden, SPANS, chunk_tokens=PROOF.chunk_tokens, topk=PROOF.topk)
    spec = PROOF.to_contract()
    assert toploc_span_verdict(hidden, {"toploc_proofs": proofs}, SPANS) is None
    assert toploc_span_verdict(hidden, {"toploc_spec": spec}, SPANS).reason == "missing"
    assert toploc_span_verdict(hidden, {"toploc_spec": spec, "toploc_proofs": proofs}, []).reason == "no_spans"
    short = toploc_span_verdict(hidden, {"toploc_spec": spec, "toploc_proofs": proofs[:-1]}, SPANS)
    assert not short.passed and short.reason == "bad_proof_shape"
    assert toploc_span_verdict(hidden, {"toploc_spec": spec, "toploc_proofs": proofs}, SPANS).passed


def test_a_span_too_short_to_judge_is_skipped_and_none_judged_fails():
    hidden = torch.randn(SEQ, HIDDEN)
    spec = PROOF.to_contract()
    mixed = [(2, 5), (15, 25)]                                 # 3 rows (< MIN_CHUNK_TOKENS) then 10
    forged_short = span_proofs_b64(torch.randn(SEQ, HIDDEN), [(2, 5)], chunk_tokens=PROOF.chunk_tokens,
                                   topk=PROOF.topk)
    honest_long = span_proofs_b64(hidden, [(15, 25)], chunk_tokens=PROOF.chunk_tokens, topk=PROOF.topk)
    verdict = toploc_span_verdict(hidden, {"toploc_spec": spec, "toploc_proofs": forged_short + honest_long}, mixed)
    assert verdict.passed
    only_short = span_proofs_b64(hidden, [(2, 5)], chunk_tokens=PROOF.chunk_tokens, topk=PROOF.topk)
    verdict = toploc_span_verdict(hidden, {"toploc_spec": spec, "toploc_proofs": only_short}, [(2, 5)])
    assert not verdict.passed and verdict.reason == "no_judged_spans"


def test_a_validator_error_raises_when_enforced_and_is_a_verdict_in_shadow():
    hidden = torch.randn(SEQ, 64)                             # narrower than topk: the validator's own error
    proofs = ["AA=="]
    with pytest.raises(ValueError):
        toploc_span_verdict(hidden, {"toploc_spec": PROOF.to_contract(), "toploc_proofs": proofs}, SPANS)
    shadow = {**PROOF.to_contract(), "mode": "shadow"}
    verdict = toploc_span_verdict(hidden, {"toploc_spec": shadow, "toploc_proofs": proofs}, SPANS)
    assert not verdict.passed and verdict.reason == "error:ValueError"


def test_spans_must_be_strict_integers_after_the_prompt():
    def meta(spans, prompt=PROMPT):
        return {"prompt_length": prompt,
                "episode": {"schema_version": SIGNED_EPISODE_SCHEMA, "assistant_spans": spans}}
    assert verifier.signed_episode_spans(meta([[1, 12]]), SEQ) == []           # starts inside the prompt
    assert verifier.signed_episode_spans(meta([[2, 12]]), SEQ) == [(2, 12)]
    assert verifier.signed_episode_spans(meta([[2.0, 12]]), SEQ) == []
    assert verifier.signed_episode_spans(meta([[True, 12]]), SEQ) == []
    assert verifier.signed_episode_spans(meta([["2", 12]]), SEQ) == []
    assert verifier.signed_episode_spans(meta("2,12"), SEQ) == []
    assert verifier.signed_episode_spans(meta(None), SEQ) == []
    assert verifier.signed_episode_spans(meta([]), SEQ) == []
    assert verifier.signed_episode_spans(meta([[2, 12]], prompt="x"), SEQ) == []


def test_a_malformed_signed_episode_fails_toploc():
    logits, tokens, hidden = _episode()
    commit = _commit(tokens, hidden)
    commit["rollout"]["episode"]["assistant_spans"] = [[2, 12], [10, 25]]
    result = _verify(commit, logits, hidden)
    assert result.toploc_checked and not result.toploc_passed and result.toploc_reason == "no_spans"
    assert result.episode_stop_picks_ok is None


def test_an_older_worker_s_answer_has_no_stop_verdict():
    logits, tokens, hidden = _episode()
    values = ProofValues.from_kernel(_verify(_commit(tokens, hidden), logits, hidden)).model_dump()
    del values["episode_stop_picks_ok"], values["episode_stop_first_bad_turn"]
    back = ProofValues.model_validate(values).to_kernel()
    assert back.episode_stop_picks_ok is None and back.episode_stop_first_bad_turn is None
