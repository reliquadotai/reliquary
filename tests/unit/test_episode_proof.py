"""TOPLOC over every model span and the per-turn forced stop pick. CPU, mocked forward."""
from unittest.mock import MagicMock, patch

import torch

from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO
from reliquary.environment import forced_sampling as fs
from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF
from reliquary.protocol.submission import SIGNED_EPISODE_SCHEMA
from reliquary.protocol.toploc_proof import completion_proofs_b64, span_proofs_b64
from reliquary.validator import verifier
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


def test_the_spans_of_a_signed_episode_are_read_strictly():
    meta = {"episode": {"schema_version": SIGNED_EPISODE_SCHEMA, "assistant_spans": [[2, 12], [15, 25]]}}
    assert verifier.signed_episode_spans(meta, SEQ) == SPANS
    assert verifier.signed_episode_spans({"episode": {"schema_version": "reliquary/episode/v1"}}, SEQ) is None
    assert verifier.signed_episode_spans({}, SEQ) is None
    for bad in ([[0, 3]], [[2, 12], [10, 25]], [[2, 26]], [[2]]):
        assert verifier.signed_episode_spans({"episode": {"schema_version": SIGNED_EPISODE_SCHEMA,
                                                          "assistant_spans": bad}}, SEQ) == []


# --- Added by the implementer: edges the brief's tests did not pin. ---------------------------------

import base64

import pytest

from reliquary.protocol.toploc_proof import build_span_proofs


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


# --- the failed turn's CDF distance (telemetry), no BFT carve-out for a signed episode. ---

