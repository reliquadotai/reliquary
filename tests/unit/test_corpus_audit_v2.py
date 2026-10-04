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


# ---- fix round 1: every scorer carries the spans, malformed records are ours ----

import copy
import json

import pytest

from reliquary.protocol.toploc import ChunkResult
from reliquary.validator.corpus_audit import SCORE_OK, rows_of_items
from reliquary.validator.corpus_gpu import decode_request, encode_request


def test_the_gpu_wire_carries_a_trajectorys_spans():
    rows = [(PROMPT + TOKENS, 8, ["a", "b"], [(8, 48), (63, 98)]), ([1, 2, 3], 1, ["c"])]
    back, chunk, topk = decode_request(encode_request(rows, chunk_tokens=32, topk=128))
    assert back == [(PROMPT + TOKENS, 8, ["a", "b"], [(8, 48), (63, 98)]), ([1, 2, 3], 1, ["c"])]
    assert (chunk, topk) == (32, 128)


def test_a_gpu_process_scorer_receives_the_spans_and_judges_a_trajectory():
    model = _tiny(0)
    seen = []

    async def scorer(rows):
        seen.extend(rows)
        rows = decode_request(encode_request(rows, chunk_tokens=32, topk=128))[0]
        scores, _, _ = score_sequences(model, rows, chunk_tokens=32, topk=128, batch_tokens=10_000)
        return scores, 0.0, 0.0

    records = _Records({ID: _record(model)})
    auditor = CorpusAuditor(job_id="swe-agentic-v1", records=records, model=None,
                            tokenizer=_Tokenizer(), proof=PROOF, scorer=scorer,
                            vocab_size=1000)
    verdict = asyncio.run(auditor.audit(ID))
    assert verdict["passed"] is True, verdict
    assert len(seen[0]) == 4 and seen[0][3] == [(8, 48), (63, 98)]


def test_rows_of_items_keeps_the_spans():
    items = [{"tokens": [1, 2], "prompt_len": 1, "proofs": ["p"]},
             {"tokens": [1, 2, 3], "prompt_len": 1, "proofs": ["p"], "spans": [(1, 3)]}]
    assert rows_of_items(items) == [([1, 2], 1, ["p"]), ([1, 2, 3], 1, ["p"], [(1, 3)])]


class _Remote:
    """A v2-capable remote: scores what it is sent with the real scorer."""

    def __init__(self, model):
        self.model, self.items = model, []

    def connected(self):
        return True

    def subscribe(self, listener):
        pass

    async def score(self, items):
        self.items.extend(items)
        scores, _, _ = score_sequences(self.model, rows_of_items(items),
                                       chunk_tokens=PROOF.chunk_tokens, topk=PROOF.topk,
                                       batch_tokens=4096)
        return [(status, chunks, "pod-1") for status, chunks in scores]


def test_a_trajectory_on_the_remote_branch_goes_with_its_spans_and_is_judged():
    model = _tiny(0)
    remote = _Remote(model)
    records = _Records({ID: _record(model)})
    auditor = CorpusAuditor(job_id="swe-agentic-v1", records=records, model=model,
                            tokenizer=_Tokenizer(), proof=PROOF, remote=remote)
    verdict = asyncio.run(auditor.audit(ID))
    assert remote.items[0]["spans"] == [(8, 48), (63, 98)]
    assert verdict["passed"] is True, verdict
    assert verdict["scored_by"] == ["pod-1"]


def test_the_eval_auditor_refuses_a_trajectory_too():
    from reliquary.validator.eval_control import eval_auditor

    auditor = eval_auditor(job_id="order-eval-1", records=None, tokenizer=_Tokenizer(),
                           proof=PROOF, vocab_size=1000, remote=_Remote(None))
    with pytest.raises(RuntimeError):
        asyncio.run(auditor._forward([_record(_tiny(0))], local=True))


def _broken(mutate):
    record = copy.deepcopy(_record(_tiny(0)))
    mutate(record)
    return record


@pytest.mark.parametrize("mutate", [
    lambda r: r["completions"][0].pop("turns"),
    lambda r: r["completions"][0].pop("prompt_tokens"),
    lambda r: r["completions"][0]["turns"][0].pop("proofs"),
    lambda r: r["completions"][0]["turns"][1].update(end=500),     # out of bounds
    lambda r: r["completions"][0]["turns"][1].update(start=30),    # overlaps the first
    lambda r: r["completions"].append(copy.deepcopy(r["completions"][0])),  # two completions
])
def test_a_malformed_v2_record_is_a_validator_error_and_spares_its_neighbours(mutate):
    model = _tiny(0)
    auditor = _auditor(model, _Records({}))
    good, bad = _record(model), _broken(mutate)
    outcomes = asyncio.run(auditor._audit_outcomes([bad, good]))
    assert isinstance(outcomes[0], str) and outcomes[0]
    assert outcomes[1]["passed"] is True


def test_the_first_failing_span_names_the_reason():
    ok, bad = ChunkResult(0, 0.0, 0.0), ChunkResult(10_000, 1e9, 1e9)
    outcome = trajectory_outcome("ok", [ok, bad, bad, ok], [40, 40, 35], PROOF)
    # spans: 40 -> 2 chunks (ok, bad); 40 -> 2 chunks (bad, ok); the third has none left.
    assert outcome.passed is False and outcome.reason
    assert outcome.reason == trajectory_outcome("ok", [ok, bad], [40], PROOF).reason
    first_good_then_bad = trajectory_outcome("ok", [ok, ok, bad, bad], [40, 40], PROOF)
    assert first_good_then_bad.passed is False


def test_a_bad_status_fails_the_trajectory_with_that_status():
    outcome = trajectory_outcome("proof_undecodable", (), [40, 35], PROOF)
    assert (outcome.passed, outcome.reason) == (False, "proof_undecodable")
