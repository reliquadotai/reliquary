"""An eval job's forced draw: one legal generation per (job, prompt, sample), checked by the audit."""
import random
from types import SimpleNamespace

import pytest
import torch

from reliquary.corpus import eval_draw
from reliquary.corpus.admission import admit
from reliquary.corpus.job import JOB_SCHEMA, JobError, parse_job
from reliquary.corpus.slots import ATTEMPTS_PER_SLOT, SlotLedger
from reliquary.corpus.walk import CursorLedger
from reliquary.environment.forced_sampling import DRAW_GUMBEL_MAX, pick_rows, u_range

SHA = "a" * 64
EOS = 151645
SOURCE = "eval-set:aime25-v1:30:" + "b" * 64


def _raw(**overrides):
    raw = {
        "schema": JOB_SCHEMA, "job_id": "eval-t-base-aime25",
        "checkpoint_repo": "ReliquaryForge/Teutonic", "checkpoint_revision": "rev1",
        "checkpoint_sha256": SHA, "prompt_source": SOURCE, "prompt_count": 30,
        "renderer_id": "chat-template-thinking-v1", "eos_token_id": EOS,
        "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_new_tokens": 2,
                     "max_new_tokens": 100, "n": 2, "draw": "gumbel_max"},
        "slots_per_prompt": 4, "filter": None, "prompt_order": "free", "deadline_round": None,
        "seed": 7,
    }
    raw.update(overrides)
    return raw


def _job(**overrides):
    return parse_job(_raw(**overrides))


def _sampling(**changes):
    return {**_raw()["sampling"], **changes}


def test_a_draw_job_parses_and_its_contract_carries_the_draw():
    job = _job()
    assert job.sampling.draw == DRAW_GUMBEL_MAX
    assert job.to_contract()["sampling"]["draw"] == DRAW_GUMBEL_MAX
    assert parse_job(job.to_contract()) == job


def test_a_free_job_contract_is_unchanged():
    job = _job(sampling=_sampling(draw=None))
    assert job.sampling.draw is None
    assert "draw" not in job.to_contract()["sampling"]


@pytest.mark.parametrize("change", [
    {"temperature": 0.6}, {"top_p": 0.95}, {"top_k": 20}, {"draw": "inverse_cdf"},
])
def test_a_draw_job_samples_at_t1_over_the_full_vocabulary(change):
    with pytest.raises(JobError):
        _job(sampling=_sampling(**change))


@pytest.mark.parametrize("overrides", [
    {"seed": None}, {"prompt_source": "openmathinstruct"}, {"prompt_order": "miner_walk"},
])
def test_a_draw_job_is_a_seeded_free_order_eval_job(overrides):
    raw = _raw(**overrides)
    if overrides.get("seed", 0) is None:
        del raw["seed"]
    with pytest.raises(JobError):
        parse_job(raw)


def test_the_stream_is_the_jobs_not_the_miners_and_each_sample_has_its_own():
    job = _job()
    a = eval_draw.uniforms(job, 3, 0, 0, 16)
    assert a == eval_draw.uniforms(job, 3, 0, 0, 16)
    assert a != eval_draw.uniforms(job, 3, 0, 1, 16)            # another completion
    assert a != eval_draw.uniforms(job, 3, 1, 0, 16)            # another draw slot
    assert a != eval_draw.uniforms(job, 4, 0, 0, 16)            # another prompt
    assert a != eval_draw.uniforms(_job(seed=8), 3, 0, 0, 16)   # another job
    # Domain-separated from the RL stream of the same key.
    context = eval_draw.draw_context(job, 3, 0, 0)
    assert a != u_range(context["randomness"], 3, context["checkpoint_hash"], context["rollout_index"], 16)
    assert eval_draw.rollout_index(job, 2, 1) == 2 * job.sampling.n + 1


def _admit(job, seen, *, cursor, prompt_index=5):
    return admit(job, hotkey="5Gx", cursor=cursor, prompt_index=prompt_index, checkpoint_sha256=SHA,
                 token_counts=[10, 12], last_token_ids=[EOS, EOS], digests=["d0", "d1"],
                 slots=SlotLedger(job.prompt_count, job.slots_per_prompt), cursors=CursorLedger(),
                 seen=seen)


def test_admission_takes_a_free_draw_slot_once():
    job = _job()
    assert _admit(job, frozenset(), cursor=0).accepted
    taken = frozenset({eval_draw.slot_key(5, 0)})
    refused = _admit(job, taken, cursor=0)
    assert (refused.accepted, refused.reason) == (False, eval_draw.REASON_DRAW_SLOT_TAKEN)
    assert _admit(job, taken, cursor=1).accepted
    assert _admit(job, taken, cursor=0, prompt_index=6).accepted


def test_admission_refuses_a_cursor_past_the_prompts_slots():
    job = _job()
    last = job.slots_per_prompt - 1
    assert eval_draw.slot_count(job) == last + 1
    assert _admit(job, frozenset(), cursor=last).accepted
    retired = _admit(job, frozenset(), cursor=last + 1)          # the former attempt range: taken
    assert (retired.accepted, retired.reason) == (False, eval_draw.REASON_DRAW_SLOT_TAKEN)
    refused = _admit(job, frozenset(), cursor=ATTEMPTS_PER_SLOT * job.slots_per_prompt)
    assert (refused.accepted, refused.reason) == (False, eval_draw.REASON_BAD_CURSOR)


def test_the_slot_key_is_a_digest_apart_from_completion_digests():
    key = eval_draw.slot_key(5, 0)
    assert len(key) == 64 and all(c in "0123456789abcdef" for c in key)
    assert key != eval_draw.slot_key(5, 1) != eval_draw.slot_key(6, 0)


def _sample(logits, u):
    """What the miner's processor picks: the draw's argmax over these logits."""
    return pick_rows(logits, u, draw=DRAW_GUMBEL_MAX).tolist()


def _logits(rows, vocab=257, peaked=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, vocab, generator=g) * 2
    if peaked:
        logits[:, 0] += peaked                         # near-deterministic positions
    return logits


def test_an_honest_sample_passes_and_an_edited_one_fails():
    job = _job()
    logits = _logits(200)
    u = eval_draw.uniforms(job, 1, 0, 0, 200)
    tokens = _sample(logits, u)
    ok, detail = eval_draw.judge([eval_draw.diagnostics(logits, tokens, u)])
    assert ok and detail["match"] == 1.0
    # Another sample's tokens (a re-roll) do not follow this sample's draw.
    other = _sample(logits, eval_draw.uniforms(job, 1, 1, 0, 200))
    ok, detail = eval_draw.judge([eval_draw.diagnostics(logits, other, u)])
    assert not ok and detail["match"] < 0.9


def test_edits_at_near_deterministic_positions_fail():
    job = _job()
    logits = _logits(300, peaked=30.0)
    u = eval_draw.uniforms(job, 1, 0, 0, 300)
    tokens = _sample(logits, u)
    edited = list(tokens)
    for p in (10, 50, 90):
        edited[p] = 1 if edited[p] != 1 else 2
    ok, _ = eval_draw.judge([eval_draw.diagnostics(logits, tokens, u)])
    assert ok
    ok, detail = eval_draw.judge([eval_draw.diagnostics(logits, edited, u)])
    assert not ok and detail["misses"] == 3 and detail["max_gap"] > eval_draw.EVAL_GAP_TOLERANCE


def test_one_bad_completion_fails_the_submission():
    job = _job()
    logits = _logits(100)
    u0, u1 = (eval_draw.uniforms(job, 2, 0, c, 100) for c in (0, 1))
    good = eval_draw.diagnostics(logits, _sample(logits, u0), u0)
    bad = eval_draw.diagnostics(logits, _sample(logits, u0), u1)
    assert eval_draw.judge([good])[0] and not eval_draw.judge([good, bad])[0]


def test_the_miner_processor_draws_the_eval_stream():
    from reliquary.miner.vllm_generation import _forced_state, _one_hot_picks, _uniform, forced_seed_extra_args

    job = _job()
    extra = forced_seed_extra_args(**eval_draw.draw_context(job, 4, 1, 1), base_offset=3)
    state = _forced_state(extra["forced_seed"])
    assert state["draw"] == DRAW_GUMBEL_MAX and state["domain"] == eval_draw.EVAL_DRAW_DOMAIN
    u = eval_draw.uniforms(job, 4, 1, 1, 8)
    assert [_uniform(state, 3 + t) for t in range(5)] == u[3:8]
    logits = _logits(5)
    forced = _one_hot_picks(logits, [0, 1, 2, 3, 4], u[3:8], kinds=[DRAW_GUMBEL_MAX] * 5)
    assert forced.argmax(-1).tolist() == _sample(logits, u[3:8])


def test_a_request_named_draw_is_picked_beside_the_profiles_rows(monkeypatch):
    from reliquary.miner import vllm_generation as vg

    logits = _logits(4)
    draws = [0.1, 0.5, 0.7, 0.9]
    monkeypatch.setattr(vg, "default_draw", lambda: "inverse_cdf")
    mixed = vg._row_picks(logits, draws, [None, DRAW_GUMBEL_MAX, None, DRAW_GUMBEL_MAX]).tolist()
    profile = vg._row_picks(logits, draws, None).tolist()
    gumbel = pick_rows(logits, draws, draw=DRAW_GUMBEL_MAX).tolist()
    assert mixed == [profile[0], gumbel[1], profile[2], gumbel[3]]


def test_the_auditor_judges_the_draw_on_the_audited_rows():
    """``CorpusAuditor._draw_diagnostics`` over an fp32 head: the miner's tokens pass, a re-roll fails."""
    from reliquary.validator.corpus_auditor import CorpusAuditor

    job = _job()
    torch.manual_seed(0)
    head = torch.nn.Linear(16, 257, bias=False)
    hidden = torch.randn(40, 16)
    logits = hidden.float() @ head.weight.float().T
    auditor = CorpusAuditor.__new__(CorpusAuditor)
    auditor._draw_job, auditor._model = job, SimpleNamespace(lm_head=head)
    prompt = [5, 6, 7]

    def item(tokens, c_idx=0):
        return (0, c_idx, prompt + tokens, len(prompt), [], None)

    record = {"prompt_index": 2, "cursor": 3}
    honest = _sample(logits, eval_draw.uniforms(job, 2, 3, 0, 40))
    reroll = _sample(logits, eval_draw.uniforms(job, 2, 4, 0, 40))
    good = auditor._draw_diagnostics(record, item(honest), hidden)
    bad = auditor._draw_diagnostics(record, item(reroll), hidden)
    results = [{"passed": True, "reason": None}, {"passed": True, "reason": None}]
    auditor._judge_draws(results, [(0, 0), (1, 0)], {0: good, 1: bad})
    assert results[0]["passed"] and results[0]["forced_draw"]["match"] == 1.0
    assert not results[1]["passed"] and results[1]["reason"] == eval_draw.REASON_DRAW_MISMATCH


def test_a_trajectory_is_drawn_over_its_model_tokens_across_turns():
    from reliquary.validator.corpus_auditor import CorpusAuditor

    job = _job()
    torch.manual_seed(1)
    head = torch.nn.Linear(8, 101, bias=False)
    auditor = CorpusAuditor.__new__(CorpusAuditor)
    auditor._draw_job, auditor._model = job, SimpleNamespace(lm_head=head)
    spans = [(3, 9), (12, 20)]                               # a tool observation at 9..12
    rows = [torch.randn(6, 8), torch.randn(8, 8)]
    logits = torch.cat(rows).float() @ head.weight.float().T
    u = eval_draw.uniforms(job, 1, 0, 0, 14)
    model_tokens = _sample(logits, u)
    tokens = [1, 2, 3] + model_tokens[:6] + [50, 51, 52] + model_tokens[6:]
    d = auditor._draw_diagnostics({"prompt_index": 1, "cursor": 0}, (0, 0, tokens, 1, [], spans), rows)
    assert d.n_positions == 14 and eval_draw.judge([d])[0]


def test_the_miner_picks_a_free_slot_on_an_open_prompt():
    import base64

    from reliquary.corpus.slots import OPEN_ENCODING
    from reliquary.miner.corpus_miner import pick_draw_slot

    job = _job()
    bitmap = bytearray(4)
    bitmap[0] = 0b01000000                                 # only prompt 1 open
    body = {"job_id": job.job_id, "prompt_start": 0, "prompt_count": 30, "open_count": 1, "as_of": 0,
            "encoding": OPEN_ENCODING, "open": base64.b64encode(bytes(bitmap)).decode(),
            "draw_slots": {"slots": 12, "taken": {"1": [0, 1]}}}
    rng = random.Random(0)
    assert pick_draw_slot(job, body, {}, set(), rng) == (1, 2)
    assert pick_draw_slot(job, body, {1: {2}}, set(), rng) == (1, 3)
    assert pick_draw_slot(job, body, {}, {1}, rng) is None
    assert pick_draw_slot(job, {**body, "open_count": 0}, {}, set(), rng) is None


def test_a_cut_last_turn_is_submitted_under_what_cut_it():
    from reliquary.miner.agentic_miner import drawn_stop

    job = SimpleNamespace(episode=SimpleNamespace(max_turns=3))
    stops = frozenset({9})
    turn = lambda *ids: SimpleNamespace(completion_ids=list(ids))  # noqa: E731
    assert drawn_stop("agent_completed", [turn(1, 9)], job, stops) == "agent_completed"
    assert drawn_stop("agent_completed", [turn(1, 2)], job, stops) == "context_length"
    assert drawn_stop("agent_completed", [turn(9), turn(9), turn(1, 2)], job, stops) == "max_turns"
    assert drawn_stop("max_turns", [turn(9)], job, stops) == "max_turns"


def _diag(**fields):
    from reliquary.environment.forced_sampling import SeedConsistencyDiagnostics

    return SeedConsistencyDiagnostics(**fields)


@pytest.mark.parametrize("positions, exact, ok", [
    (9618, 9585, True),       # the worst honest drift measured on H200 (0.69 % of stochastic positions)
    (5000, 4900, True),       # 2 % of every drawn position: the floor
    (5000, 4899, False),
    (36, 35, True),           # a short sample: a few honest near-ties are not a refusal
    (36, 28, True),           # EVAL_MATCH_GRACE
    (36, 27, False),
    (290, 284, True),         # live honest refusals at the old grace of 2 (2026-10-28)
    (24, 18, True),           # a quarter of a shorter sample, at most
    (24, 17, False),
])
def test_the_match_floor_is_two_percent_of_every_drawn_position(positions, exact, ok):
    d = _diag(n_positions=positions, n_stochastic=positions // 2, n_drawn_exact_match=exact)
    assert eval_draw.judge([d])[0] is ok


def test_a_low_temperature_sample_is_not_refused_for_having_few_stochastic_positions():
    """Honest misses are near-ties, which stay near-ties at any temperature, while the stochastic positions shrink:
    over stochastic positions only, 10 misses in 300 is 3.3 %; over the 1,000 drawn positions it is 1 %."""
    d = _diag(n_positions=1000, n_stochastic=300, n_exact_match=290, n_drawn_exact_match=990)
    assert eval_draw.judge([d])[0]


def test_a_miss_far_from_the_pick_is_an_edit_whatever_the_count():
    import math

    honest = 1 - math.exp(-0.6)         # an honest near-tie (honest gaps q99.9 0.64, max 2.9 on the 27B)
    edit = 1 - math.exp(-5.0)
    base = dict(n_positions=1000, n_stochastic=500, n_drawn_exact_match=999)
    assert eval_draw.judge([_diag(**base, max_cdf_miss=honest)])[0]
    ok, detail = eval_draw.judge([_diag(**base, max_cdf_miss=edit)])
    assert not ok and detail["max_gap"] > eval_draw.EVAL_GAP_TOLERANCE


def test_an_end_the_pick_did_not_make_is_refused():
    import math

    base = dict(n_positions=1000, n_stochastic=500, n_drawn_exact_match=999)
    assert eval_draw.judge([_diag(**base, last_miss=1 - math.exp(-0.5))])[0]
    ok, detail = eval_draw.judge([_diag(**base, last_miss=1 - math.exp(-5.0))])
    assert not ok and detail["last_gap"] > eval_draw.EVAL_TERMINAL_GAP


def test_nothing_drawn_is_nothing_verified():
    assert not eval_draw.judge([_diag()])[0]


def test_the_diagnostics_carry_the_last_positions_miss():
    from reliquary.environment.forced_sampling import DRAW_GUMBEL_MAX, seed_consistency_diagnostics

    logits = _logits(12, peaked=0.0)
    u = [((i * 7919) % 1000) / 1000.0 for i in range(12)]
    tokens = pick_rows(logits, u, draw=DRAW_GUMBEL_MAX).tolist()
    honest = seed_consistency_diagnostics(logits, tokens, u, t=1.0, top_k=0, top_p=1.0, stochastic_threshold=0.99,
                                          boundary_epsilon=0.05, draw=DRAW_GUMBEL_MAX)
    assert honest.last_miss == 0.0
    cut = tokens[:-1] + [(tokens[-1] + 1) % logits.shape[1]]
    edited = seed_consistency_diagnostics(logits, cut, u, t=1.0, top_k=0, top_p=1.0, stochastic_threshold=0.99,
                                          boundary_epsilon=0.05, draw=DRAW_GUMBEL_MAX)
    assert edited.last_miss > 0.0


class _Verdicts:
    """The auditor's record store: submissions, create-only verdicts."""

    def __init__(self, submissions):
        self.submissions, self.verdicts = submissions, {}

    async def write_verdict(self, job_id, submission_id, verdict):
        if submission_id in self.verdicts:
            return False
        self.verdicts[submission_id] = verdict
        return True

    async def read_verdict(self, job_id, submission_id):
        return self.verdicts.get(submission_id)

    async def read_submission(self, job_id, submission_id):
        return self.submissions.get(submission_id)

    async def list_verdict_ids(self, job_id):
        return list(self.verdicts)


def _draw_auditor(job, records, slot_store):
    from reliquary.validator.corpus_auditor import CorpusAuditor

    auditor = CorpusAuditor.__new__(CorpusAuditor)
    auditor._job_id, auditor._draw_job, auditor._records, auditor._slot_store = job.job_id, job, records, slot_store
    auditor._on_verdict = None
    auditor._mark_judged = lambda submission_id: None
    return auditor


def test_a_failed_draw_sample_reopens_its_prompts_slot(monkeypatch):
    """Else the prompt stays full without the sample and the job never completes."""
    import asyncio

    from reliquary.infrastructure import corpus_job_store as job_store
    from reliquary.infrastructure.corpus_job_store import BucketJobStore
    from reliquary.validator.corpus_service import rebuild_ledgers
    from tests.unit.test_corpus_job_store import _FakeMultiObjectR2

    monkeypatch.setattr(job_store, "get_s3_client", lambda **kw: _FakeMultiObjectR2())
    store, job = BucketJobStore(), _job()
    bad, good = "b" * 64, "a" * 64
    records = _Verdicts({bad: {"prompt_index": 2}, good: {"prompt_index": 3}})
    auditor = _draw_auditor(job, records, store)

    async def capacity(index):
        snapshot, _ = await store.read_ledgers(job.job_id)
        return rebuild_ledgers(job, snapshot).slots.capacity(index)

    async def go():
        await auditor._write(good, {"passed": True})
        await auditor._write(bad, {"passed": False, "reason": eval_draw.REASON_DRAW_MISMATCH})
        assert await capacity(2) == job.slots_per_prompt + 1
        assert await capacity(3) == job.slots_per_prompt
        # A restart records nothing twice.
        assert await auditor.reconcile_failures() == 1
        assert await capacity(2) == job.slots_per_prompt + 1

    asyncio.run(go())


def test_a_free_job_never_reopens_a_slot():
    import asyncio

    records = _Verdicts({"b" * 64: {"prompt_index": 2}})
    auditor = _draw_auditor(_job(), records, None)
    asyncio.run(auditor._write("b" * 64, {"passed": False}))
    assert asyncio.run(auditor.reconcile_failures()) == 0
