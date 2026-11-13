"""A forced draw on any job (generation included), prompts given as token ids, long-context jobs, and draw orders
from the admin: the pieces a batch generation API sits on."""
from __future__ import annotations

import hashlib
import json

import pytest

from reliquary.corpus.encoding import TOKEN_IDS_PREFIX, prompt_token_ids, token_ids_text
from reliquary.corpus.job import JobError, PROMPT_ORDER_FREE, parse_job


def _gen_raw(**overrides):
    from tests.unit.test_corpus_job import _raw

    raw = _raw(prompt_order=PROMPT_ORDER_FREE, seed=11)
    raw["sampling"] = {**raw["sampling"], "draw": "gumbel_max", "temperature": 0.7, "top_p": 0.95}
    raw.update(overrides)
    return raw


# ------------------------------------------------------------------ a draw on a generation job


def test_a_generation_job_takes_a_draw():
    job = parse_job(_gen_raw())
    assert job.sampling.draw == "gumbel_max" and job.seed == 11
    assert parse_job(job.to_contract()) == job


def test_a_draw_still_needs_its_seed_and_free_order():
    with pytest.raises(JobError, match="seed"):
        parse_job({k: v for k, v in _gen_raw().items() if k != "seed"})
    from tests.unit.test_corpus_job import PROMPT_ORDER_MINER_WALK

    with pytest.raises(JobError, match="miner_walk"):
        parse_job(_gen_raw(prompt_order=PROMPT_ORDER_MINER_WALK))


def test_a_draw_generation_jobs_failed_slot_reopens():
    import asyncio

    from reliquary.validator.corpus_service import record_prompt_failure

    class Store:
        snapshot, etag = {}, None

        async def read_ledgers(self, job_id):
            return self.snapshot, self.etag

        async def write_ledgers(self, job_id, snapshot, etag):
            self.snapshot, self.etag = snapshot, (etag or 0) + 1

    job = parse_job(_gen_raw())
    store = Store()
    assert asyncio.run(record_prompt_failure(store, job, 3, "s" * 64, slot=1, hotkey="h")) is True
    plain = parse_job({**_gen_raw(), "sampling": {k: v for k, v in _gen_raw()["sampling"].items() if k != "draw"}})
    with pytest.raises(ValueError, match="never reopen"):
        asyncio.run(record_prompt_failure(Store(), plain, 3, "s" * 64))


def test_a_draw_on_a_suite_episode_is_accepted():
    from reliquary.corpus.job import _check_draw_job
    from types import SimpleNamespace

    job = SimpleNamespace(seed=1, prompt_order=PROMPT_ORDER_FREE, episode=SimpleNamespace(harness="suite"))
    _check_draw_job(job)
    with pytest.raises(JobError, match="seed"):
        _check_draw_job(SimpleNamespace(seed=None, prompt_order=PROMPT_ORDER_FREE, episode=job.episode))


# ------------------------------------------------------------------ prompts as token ids


def test_token_ids_travel_as_canonical_text_and_come_back_exactly():
    text = token_ids_text([151644, 872, 198, 0])
    assert text.startswith(TOKEN_IDS_PREFIX)

    class NoTokenizer:
        def encode(self, *a, **k):
            raise AssertionError("token-id prompts never go through the tokenizer")

    assert prompt_token_ids(NoTokenizer(), text) == [151644, 872, 198, 0]
    for bad in ([], [-1], [1.5], [True]):
        with pytest.raises(ValueError):
            token_ids_text(bad)
    with pytest.raises(ValueError):
        prompt_token_ids(NoTokenizer(), TOKEN_IDS_PREFIX + "1,02")      # not canonical


def _set(rows):
    from reliquary.eval.prompt_source import EvalSource

    body = b"".join(json.dumps(r).encode() + b"\n" for r in rows)
    return EvalSource("batch-1", len(rows), hashlib.sha256(body).hexdigest()), body


def test_a_prompt_set_row_may_be_token_ids():
    from reliquary.eval.prompt_source import EvalSetEnvironment, register_eval_prompts
    from reliquary.validator.corpus_service import ChatTemplatePromptRenderer, EpisodeTask

    source, body = _set([{"problem_id": "r0", "prompt_token_ids": [5, 6, 7]},
                         {"problem_id": "r1", "messages": [{"role": "user", "content": "hi"}]}])
    rows = register_eval_prompts(source, body)
    env = EvalSetEnvironment(source, rows)
    problem = env.get_problem(0)
    assert problem["prompt"] == token_ids_text([5, 6, 7])

    class Template:
        def apply_chat_template(self, messages, **kw):
            return "<user>" + messages[-1]["content"]

    renderer = ChatTemplatePromptRenderer(Template(), thinking=False)
    assert renderer.initial_text(EpisodeTask(id="x", prompt=problem["prompt"], tools=())) == problem["prompt"]
    assert renderer.initial_text(EpisodeTask(id="y", prompt="hi", tools=())) == "<user>hi"


@pytest.mark.parametrize("row", [
    {"problem_id": "r", "prompt_token_ids": []},
    {"problem_id": "r", "prompt_token_ids": [1, -2]},
    {"problem_id": "r", "prompt_token_ids": [1], "messages": [{"role": "user", "content": "x"}]},
])
def test_a_bad_token_id_row_is_refused(row):
    from reliquary.eval.prompt_source import register_eval_prompts

    with pytest.raises(ValueError):
        register_eval_prompts(*_set([row]))


# ------------------------------------------------------------------ long context


def test_a_long_context_job_declares_its_longest_prompt():
    job = parse_job(_gen_raw(max_prompt_tokens=60000))
    assert job.max_prompt_tokens == 60000 and job.to_contract()["max_prompt_tokens"] == 60000
    assert "max_prompt_tokens" not in parse_job(_gen_raw()).to_contract()
    for bad in (0, 131073, "1"):
        with pytest.raises(JobError):
            parse_job(_gen_raw(max_prompt_tokens=bad))


def test_a_drawn_audit_item_holds_a_long_context_sample_a_proven_one_does_not():
    from reliquary.validator.corpus_audit_protocol import MAX_DRAWN_SEQUENCE_TOKENS, AuditItem
    from tests.unit.test_draw_only_audit import spec

    assert MAX_DRAWN_SEQUENCE_TOKENS >= 131072 + 60000
    AuditItem.model_validate({"tokens": [1] * 150000, "prompt_len": 100000, "proofs": [],
                              "draw": spec().to_wire()})
    with pytest.raises(ValueError):
        AuditItem.model_validate({"tokens": [1] * 150000, "prompt_len": 100000, "proofs": []})


# ------------------------------------------------------------------ admin orders


def test_an_order_names_its_draw_judge_and_min_p():
    from reliquary.admin.service import CreateJob, OrderSampling

    body = CreateJob.model_validate({
        "job_id": "order-gen-x", "model": "m", "env": "e", "prompt_count": 4, "samples_per_prompt": 2,
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.05}, "draw": "gumbel_max",
        "judge": {"floor_rate": 0.95}, "max_prompt_tokens": 40000})
    assert body.draw == "gumbel_max" and body.judge == {"floor_rate": 0.95}
    assert body.sampling.conditions()["min_p"] == 0.05
    # An order without min_p records the same conditions as before min_p existed.
    assert OrderSampling(temperature=1.0).conditions() == {"temperature": 1.0, "top_p": 1.0, "top_k": 0}
    with pytest.raises(ValueError):
        CreateJob.model_validate({"job_id": "j", "model": "m", "env": "e", "prompt_count": 1,
                                  "samples_per_prompt": 1, "draw": "inverse_cdf"})


# ------------------------------------------------------------------ the episode intake


def test_a_draw_episodes_turns_carry_no_proof_and_may_be_short():
    """The intake asks a draw job's turns for no proof, and no minimum length (the draw checks every model token);
    the same trajectory without proofs is refused on a job without a draw."""
    from types import SimpleNamespace

    pytest.importorskip("reliquary_harness")

    from reliquary.environment.agentic_swe import SweSource
    from reliquary.validator.agentic_intake import EpisodeIntake
    from tests.unit.test_corpus_job_episode import _manifest
    from tests.unit.test_corpus_route_episode import ROWS, TOKENIZER, TURNS
    from tests.unit.test_trajectory_parse import R, build

    raw = _manifest(prompt_count=3)
    drawn_raw = {**raw, "seed": 3, "prompt_order": PROMPT_ORDER_FREE,
                 "sampling": {**raw["sampling"], "draw": "gumbel_max"}}

    def refusal(job_raw):
        intake = EpisodeIntake(job=parse_job(job_raw), source=SweSource(ROWS), renderer=R, tokenizer=TOKENIZER,
                               vocab_size=None, chunk_tokens=32)
        tokens, spans = build(TURNS)
        trajectory = SimpleNamespace(tokens=tokens, turns=[SimpleNamespace(proofs=[]) for _ in spans])
        return intake._shape_refusal(trajectory, spans, list(range(10)))

    assert refusal(raw) is not None
    assert refusal(drawn_raw) is None
