"""A draw job's sample carries no proof: the draw is its only check, computed by the same forward on every audit
path (in-process, the validator's GPU process, a remote executor) from its public draw context."""
from __future__ import annotations

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from reliquary.corpus import draw_audit, eval_draw
from reliquary.corpus.draw_audit import DrawSpec
from reliquary.environment import forced_sampling as fs
from reliquary.environment.forced_sampling import Warp
from reliquary.validator.corpus_audit import SCORE_OK, score_sequences

PROMPT = list(range(10, 18))


def tiny(seed=0):
    torch.manual_seed(seed)
    config = AutoConfig.for_model(
        "qwen3", hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, vocab_size=512,
    )
    return AutoModelForCausalLM.from_config(config).to(torch.bfloat16).eval()


def spec(rollout=0, warp=Warp(), margin=0.15):
    return DrawSpec("ab" * 32, 3, "rev1", rollout, eval_draw.EVAL_DRAW_DOMAIN, warp, margin)


def drawn(model, spec_, n=40, prompt=PROMPT):
    """What an honest miner generates: each token the draw's pick over the model's fp32-head logits."""
    tokens = list(prompt)
    u = spec_.uniforms(n)
    w = spec_.warp
    with torch.no_grad():
        for t in range(n):
            hidden = model.model(torch.tensor([tokens])).last_hidden_state[:, -1]
            logits = hidden.float() @ model.lm_head.weight.float().T
            tokens.append(int(fs.pick_rows(logits, [u[t]], draw=fs.DRAW_GUMBEL_MAX, t=w.t, top_k=w.top_k,
                                           top_p=w.top_p, min_p=w.min_p)[0]))
    return tokens


def _score(model, rows):
    scores, _, _ = score_sequences(model, rows, chunk_tokens=32, topk=8, batch_tokens=1 << 16)
    return scores


@pytest.mark.parametrize("warp", [Warp(), Warp(t=0.7, top_k=20, top_p=0.9)])
def test_an_honest_drawn_sample_follows_its_draw(warp):
    model = tiny()
    s = spec(warp=warp)
    tokens = drawn(model, s)
    ((status, chunks, diag),) = _score(model, [(tokens, len(PROMPT), None, None, s.to_wire())])
    assert (status, chunks) == (SCORE_OK, ())
    assert diag["positions"] == 40
    assert diag["exact"] >= 39
    assert eval_draw.judge([draw_audit.diagnostics_from_wire(diag)])[0]


def test_a_sample_from_another_draw_does_not():
    model = tiny()
    tokens = drawn(model, spec(rollout=1))
    ((_, _, diag),) = _score(model, [(tokens, len(PROMPT), None, None, spec(rollout=0).to_wire())])
    assert diag["exact"] < 30
    assert not eval_draw.judge([draw_audit.diagnostics_from_wire(diag)])[0]


def test_rows_without_a_draw_score_as_before():
    model = tiny()
    tokens = drawn(model, spec(), n=40)
    plain, proofless = _score(model, [(tokens, len(PROMPT), []), (tokens, len(PROMPT), None)])
    assert len(plain) == 2 and plain[0] != SCORE_OK      # no proof where one is due: a shape failure
    assert proofless == (SCORE_OK, ())


def test_a_trajectory_is_drawn_over_its_model_spans():
    """Positions count model tokens across turns; the tool output between spans is not drawn."""
    model = tiny()
    s = spec()
    first = drawn(model, s, n=10)
    tool = [200, 201, 202]
    u = s.uniforms(20)
    tokens = first + tool
    w = s.warp
    with torch.no_grad():
        for t in range(10, 20):
            hidden = model.model(torch.tensor([tokens])).last_hidden_state[:, -1]
            logits = hidden.float() @ model.lm_head.weight.float().T
            tokens.append(int(fs.pick_rows(logits, [u[t]], draw=fs.DRAW_GUMBEL_MAX)[0]))
    spans = [(len(PROMPT), len(PROMPT) + 10), (len(first) + 3, len(tokens))]
    ((_, _, diag),) = _score(model, [(tokens, len(PROMPT), None, spans, s.to_wire())])
    assert diag["positions"] == 20
    assert diag["exact"] >= 19
    del w


def test_a_draw_spec_round_trips_and_refuses_junk():
    s = spec(warp=Warp(t=0.6, top_k=20, top_p=0.95, min_p=0.05))
    assert DrawSpec.from_wire(s.to_wire()) == s
    for bad in ({**s.to_wire(), "extra": 1}, {**s.to_wire(), "edge_margin": 0.9},
                {**s.to_wire(), "prompt_idx": -1}, {**s.to_wire(), "warp": {"t": 0}}):
        with pytest.raises(ValueError):
            DrawSpec.from_wire(bad)


def test_diagnostics_from_the_wire_are_checked():
    good = {"positions": 10, "stochastic": 5, "exact": 9, "max_miss": 0.1, "last_miss": 0.0}
    assert draw_audit.diagnostics_from_wire(good).n_drawn_exact_match == 9
    for bad in ({**good, "exact": 11}, {**good, "max_miss": 2.0}, {**good, "positions": True},
                {**good, "extra": 1}):
        with pytest.raises(ValueError):
            draw_audit.diagnostics_from_wire(bad)


def test_the_spec_of_a_job_sample_is_its_draw_context_and_warp():
    from tests.unit.test_eval_draw import _job, _sampling

    job = _job(sampling=_sampling(temperature=0.6, top_k=20, top_p=0.95))
    s = draw_audit.spec_for(job, 4, 2, 1)
    context = eval_draw.draw_context(job, 4, 2, 1)
    assert (s.randomness, s.prompt_idx, s.rollout_index, s.domain) == (
        context["randomness"], 4, context["rollout_index"], eval_draw.EVAL_DRAW_DOMAIN)
    assert s.warp == Warp(t=0.6, top_k=20, top_p=0.95)
    assert s.edge_margin == job.judge.edge_margin
    assert s.uniforms(5) == eval_draw.uniforms(job, 4, 2, 1, 5)


# ------------------------------------------------------------------ the GPU process and the executors


def test_a_drawn_row_crosses_the_gpu_process_wire():
    from reliquary.validator.corpus_gpu import decode_request, encode_request, scores_from_wire, scores_to_wire

    s = spec(warp=Warp(t=0.6, top_k=20))
    rows = [([1, 2, 3, 4], 2, ["p"]), ([5, 6, 7, 8, 9], 1, None, None, s.to_wire()),
            ([1, 2, 3, 4, 5, 6], 1, None, [(1, 3), (4, 6)], s), ([7, 8, 9], 1, ["q"], [(1, 3)])]
    decoded, chunk_tokens, topk = decode_request(encode_request(rows, chunk_tokens=32, topk=8))
    assert (chunk_tokens, topk) == (32, 8)
    assert decoded[0] == rows[0] and decoded[3] == ([7, 8, 9], 1, ["q"], [(1, 3)])
    assert decoded[1] == ([5, 6, 7, 8, 9], 1, None, None, s.to_wire())
    assert decoded[2] == ([1, 2, 3, 4, 5, 6], 1, None, [(1, 3), (4, 6)], s.to_wire())
    diag = {"positions": 4, "stochastic": 2, "exact": 4, "max_miss": 0.0, "last_miss": 0.0}
    scores = [("ok", ()), ("ok", (), diag)]
    assert scores_from_wire(scores_to_wire(scores)) == scores


def test_an_executor_scores_a_v3_lease():
    from reliquary.validator.corpus_audit_executor import AuditExecutor
    from reliquary.validator.corpus_audit_protocol import AUDIT_PROTOCOL_V3, AuditLease, AuditResult

    model = tiny()
    s = spec(warp=Warp(t=0.7, top_k=20))
    honest, other = drawn(model, s), drawn(model, spec(rollout=3, warp=s.warp))
    lease = AuditLease.model_validate({
        "protocol": AUDIT_PROTOCOL_V3, "lease_id": "a" * 32, "model_id": "m", "model_revision": "r",
        "chunk_tokens": 32, "topk": 8, "expires_at": 1e12,
        "items": [{"tokens": t, "prompt_len": len(PROMPT), "proofs": [], "draw": s.to_wire()}
                  for t in (honest, other)]})
    executor = AuditExecutor(http=None, executor_id="pod", token="x", model_id="m", model_revision="r",
                             load_model=lambda m, r: None)
    executor._model = model
    scores = AuditResult.model_validate({"scores": executor._score(lease)}).scores
    assert [x.status for x in scores] == ["ok", "ok"]
    assert scores[0].draw["exact"] >= 39 and scores[1].draw["exact"] < 30


def test_a_v3_lease_refuses_proofs_and_mixed_items():
    from reliquary.validator.corpus_audit_protocol import AUDIT_PROTOCOL, AUDIT_PROTOCOL_V3, AuditLease

    s = spec().to_wire()
    base = {"lease_id": "a" * 32, "model_id": "m", "model_revision": "r", "chunk_tokens": 32, "topk": 8,
            "expires_at": 1e12}
    drawn_item = {"tokens": [1, 2, 3], "prompt_len": 1, "proofs": [], "draw": s}
    proven = {"tokens": [1, 2, 3], "prompt_len": 1, "proofs": ["p"]}
    AuditLease.model_validate({**base, "protocol": AUDIT_PROTOCOL_V3, "items": [drawn_item]})
    for protocol, items in ((AUDIT_PROTOCOL_V3, [drawn_item, proven]), (AUDIT_PROTOCOL, [drawn_item]),
                            (AUDIT_PROTOCOL_V3, [{**drawn_item, "proofs": ["p"]}])):
        with pytest.raises(ValueError):
            AuditLease.model_validate({**base, "protocol": protocol, "items": items})


def _diag(exact, positions=100):
    return {"positions": positions, "stochastic": positions // 2, "exact": exact, "max_miss": 0.0, "last_miss": 0.0}


def test_the_order_control_leases_drawn_items_to_v3_executors_and_judges_with_the_job():
    import asyncio

    from reliquary.corpus.job import DEFAULT_DRAW_JUDGE, DrawJudge
    from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF
    from reliquary.validator.corpus_audit_protocol import AUDIT_PROTOCOL, AUDIT_PROTOCOL_V3, AuditResult
    from tests.unit.test_eval_control import MODEL_A, _executor, _Harness

    items = [{"tokens": [1, 2, 3, 4], "prompt_len": 2, "proofs": None, "draw": spec().to_wire()}]
    v3 = (AUDIT_PROTOCOL, AUDIT_PROTOCOL_V3)

    def result(exact):
        return AuditResult.model_validate({"scores": [{"status": "ok", "chunks": [],
                                                       "draw": _diag(exact, 1000)}]})

    async def go(judge, exacts):
        h = _Harness()
        scoring = asyncio.ensure_future(h.dispatcher.view(*MODEL_A, PROOF, "j", judge).score(items))
        await asyncio.sleep(0)
        assert h.dispatcher.claim(_executor("old")) is None            # a v1 executor never gets it
        a = h.dispatcher.claim(_executor("a"), v3)
        b = h.dispatcher.claim(_executor("b"), v3)
        assert a["protocol"] == AUDIT_PROTOCOL_V3 and a["items"][0]["draw"] == items[0]["draw"]
        await h.dispatcher.result("a", a["lease_id"], result(exacts[0]))
        await h.dispatcher.result("b", b["lease_id"], result(exacts[1]))
        return (await scoring)[0], h

    (status, chunks, by, diag), h = asyncio.run(go(DEFAULT_DRAW_JUDGE, (960, 961)))
    assert (status, chunks, by, diag["exact"]) == ("ok", (), ("a", "b"), 960)
    assert h.quarantined == []
    # 960/1000 is under the default floor and over a 0.95 one: the pair agrees either way, the judge differs.
    from reliquary.corpus import eval_draw
    from reliquary.corpus.draw_audit import diagnostics_from_wire

    assert not eval_draw.judge([diagnostics_from_wire(diag)], DEFAULT_DRAW_JUDGE)[0]
    assert eval_draw.judge([diagnostics_from_wire(diag)], DrawJudge(0.95, 8, 4.0, 3.0))[0]

    async def disagree():
        h = _Harness()
        scoring = asyncio.ensure_future(h.dispatcher.view(*MODEL_A, PROOF, "j", DEFAULT_DRAW_JUDGE).score(items))
        await asyncio.sleep(0)
        a = h.dispatcher.claim(_executor("a"), v3)
        b = h.dispatcher.claim(_executor("b"), v3)
        await h.dispatcher.result("a", a["lease_id"], result(1000))
        await h.dispatcher.result("b", b["lease_id"], result(100))
        assert not scoring.done() and h.dispatcher.stats["disagreements"] == 1
        scoring.cancel()

    asyncio.run(disagree())


def test_a_v3_result_must_carry_the_draw():
    import asyncio

    from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF
    from reliquary.validator.corpus_audit_protocol import AUDIT_PROTOCOL, AUDIT_PROTOCOL_V3, AuditResult
    from reliquary.validator.corpus_audit_remote import LeaseRefused
    from tests.unit.test_eval_control import MODEL_A, _executor, _Harness

    items = [{"tokens": [1, 2, 3, 4], "prompt_len": 2, "proofs": None, "draw": spec().to_wire()}]

    async def go():
        h = _Harness()
        scoring = asyncio.ensure_future(h.dispatcher.view(*MODEL_A, PROOF, "j").score(items))
        await asyncio.sleep(0)
        a = h.dispatcher.claim(_executor("a"), (AUDIT_PROTOCOL, AUDIT_PROTOCOL_V3))
        with pytest.raises(LeaseRefused):
            await h.dispatcher.result("a", a["lease_id"], AuditResult.model_validate(
                {"scores": [{"status": "ok", "chunks": []}]}))
        scoring.cancel()

    asyncio.run(go())


# ------------------------------------------------------------------ the auditor's verdict


def _auditor(job):
    from reliquary.validator.corpus_auditor import CorpusAuditor

    auditor = CorpusAuditor.__new__(CorpusAuditor)
    auditor._draw_job = job
    return auditor


def test_a_draw_jobs_rows_carry_their_draw_and_no_proof():
    from tests.unit.test_eval_draw import _job

    job = _job()
    auditor = _auditor(job)
    records = [{"prompt_index": 4, "cursor": 2}]
    items = [(0, 1, [1, 2, 3, 4], 2, None, None)]
    ((tokens, n, proofs, spans, draw),) = auditor._rows(records, items)
    assert (tokens, n, proofs, spans) == ([1, 2, 3, 4], 2, None, None)
    assert draw == draw_audit.spec_for(job, 4, 2, 1).to_wire()
    (item,) = auditor._wire_items(records, items)
    assert item["draw"] == draw and item["proofs"] is None and "spans" not in item


def test_a_draw_jobs_verdict_is_its_judge_alone():
    from tests.unit.test_eval_draw import _job

    auditor = _auditor(_job())
    records = [{"completions": [{}, {}]}, {"completions": [{}]}, {"completions": [{}]}]
    items = [(0, 0, [], 1, None, None), (0, 1, [], 1, None, None), (1, 0, [], 1, None, None),
             (2, 0, [], 1, None, None)]
    scores = [("ok", (), _diag(100)), ("ok", (), _diag(99)), ("ok", (), _diag(50)), ("error", ())]
    results = auditor._verdicts(records, [None, None, None], items, scores)
    assert results[0]["passed"] and results[0]["forced_draw"]["match"] == 0.99
    assert not results[1]["passed"] and results[1]["reason"] == eval_draw.REASON_DRAW_MISMATCH
    assert not results[2]["passed"] and results[2]["reason"] == "error"
    with pytest.raises(RuntimeError):
        auditor._verdicts(records[:1], [None], items[:1], [("ok", ())])
