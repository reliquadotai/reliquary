"""Audit v2: one prefill of prompt + trajectory, proofs verified per assistant span."""

import asyncio
import base64

import torch

from reliquary.infrastructure.corpus_record_store import RECORD_SCHEMA_V2
from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF
from reliquary.protocol.toploc_proof import build_chunk_proofs, build_span_proofs
from reliquary.validator.corpus_audit import (
    completion_hidden_states,
    score_sequences,
    span_hidden_states,
    trajectory_outcome,
)
from reliquary.validator.corpus_auditor import CorpusAuditor
from tests.unit.test_corpus_audit import _tiny
from tests.unit.test_corpus_auditor import _Records, _Tokenizer

ID = "e" * 64
PROMPT = list(range(10, 18))                       # 8 prompt tokens
TOKENS = list(range(18, 108))                      # 90 trajectory tokens
SPANS = [(0, 40), (55, 90)]                        # 40 -> 32 + 8; 35 -> one merged chunk


def _b64(raw):
    return [base64.b64encode(p).decode() for p in raw]


def _turn_proofs(model, spans=SPANS):
    absolute = [(len(PROMPT) + s, len(PROMPT) + e) for s, e in spans]
    rows = span_hidden_states(model, PROMPT + TOKENS, absolute)
    return [_b64(build_span_proofs(r, chunk_tokens=PROOF.chunk_tokens, topk=PROOF.topk)) for r in rows]


def _record(model, spans=SPANS, proofs=None):
    proofs = proofs if proofs is not None else _turn_proofs(model, spans)
    return {"schema": RECORD_SCHEMA_V2, "hotkey": "5Hot", "rendered_prompt": "", "token_count": 75,
            "completions": [{"prompt_tokens": PROMPT, "tokens": TOKENS, "final_diff": "",
                             "stop": "agent_completed",
                             "turns": [{"start": s, "end": e, "proofs": p}
                                       for (s, e), p in zip(spans, proofs)]}]}


def _auditor(model, records):
    return CorpusAuditor(job_id="swe-agentic-v1", records=records, model=model,
                         tokenizer=_Tokenizer(), proof=PROOF)


def test_an_honest_trajectory_passes():
    model = _tiny(0)
    records = _Records({ID: _record(model)})
    verdict = asyncio.run(_auditor(model, records).audit(ID))
    assert verdict["passed"] is True, verdict
    assert records.verdicts[ID]["token_count"] == 75


def test_another_models_trajectory_fails():
    records = _Records({ID: _record(_tiny(1))})
    assert asyncio.run(_auditor(_tiny(0), records).audit(ID))["passed"] is False


def test_a_short_turn_is_reported_but_not_judged():
    model = _tiny(0)
    spans = [(0, 40), (55, 60), (70, 90)]           # the middle turn is 5 tokens
    proofs = _turn_proofs(model, spans)
    proofs[1] = _turn_proofs(_tiny(1), spans)[1]    # a wrong proof on the short turn
    tokens = PROMPT + TOKENS
    absolute = [(len(PROMPT) + s, len(PROMPT) + e) for s, e in spans]
    flat = [p for turn in proofs for p in turn]
    (scored,), _, _ = score_sequences(model, [(tokens, len(PROMPT), flat, absolute)],
                                      chunk_tokens=32, topk=128, batch_tokens=10_000)
    outcome = trajectory_outcome(*scored, [e - s for s, e in spans], PROOF)
    assert outcome.passed is True and len(outcome.results) == 4


def test_only_short_turns_fail_closed():
    model = _tiny(0)
    spans = [(0, 5), (85, 90)]
    flat = [p for turn in _turn_proofs(model, spans) for p in turn]
    absolute = [(len(PROMPT) + s, len(PROMPT) + e) for s, e in spans]
    (scored,), _, _ = score_sequences(model, [(PROMPT + TOKENS, len(PROMPT), flat, absolute)],
                                      chunk_tokens=32, topk=128, batch_tokens=10_000)
    outcome = trajectory_outcome(*scored, [5, 5], PROOF)
    assert (outcome.passed, outcome.reason) == (False, "no_judged_spans")


def test_a_wrong_proof_count_is_a_bad_shape():
    model = _tiny(0)
    flat = [p for turn in _turn_proofs(model) for p in turn][:-1]
    absolute = [(len(PROMPT) + s, len(PROMPT) + e) for s, e in SPANS]
    (scored,), _, _ = score_sequences(model, [(PROMPT + TOKENS, len(PROMPT), flat, absolute)],
                                      chunk_tokens=32, topk=128, batch_tokens=10_000)
    assert scored[0] == "bad_proof_shape"


def test_single_turn_and_trajectory_rows_score_together():
    model = _tiny(0)
    single = PROMPT + TOKENS[:70]
    hidden = completion_hidden_states(model, single, len(PROMPT))
    v1 = (single, len(PROMPT), _b64(build_chunk_proofs(hidden, chunk_tokens=32, topk=128)))
    absolute = [(len(PROMPT) + s, len(PROMPT) + e) for s, e in SPANS]
    v2 = (PROMPT + TOKENS, len(PROMPT), [p for t in _turn_proofs(model) for p in t], absolute)
    scores, _, _ = score_sequences(model, [v2, v1], chunk_tokens=32, topk=128, batch_tokens=10_000)
    assert [s[0] for s in scores] == ["ok", "ok"]
    assert trajectory_outcome(*scores[0], [40, 35], PROOF).passed


def test_item_spans_are_offset_by_the_prompt_length():
    model = _tiny(0)
    results, items = _auditor(model, _Records({})) ._prepare([_record(model)])
    (item,) = items
    assert item[5] == [(8, 48), (63, 98)] and item[3] == 8
    assert item[2] == PROMPT + TOKENS and results == [None]
