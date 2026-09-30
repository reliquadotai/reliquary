"""Check before generating: the miner asks where its walk stands, skips full
prompts with a signed skip, and generates only for a prompt with a slot left.
Against a validator without those routes it mines exactly as before."""

import httpx
import pytest

from reliquary.corpus.walk import job_walk_index
from reliquary.miner.corpus_miner import (
    CorpusMinerHalted,
    HttpCorpusClient,
    build_skip,
    mine_steps,
)
from tests.unit.test_corpus_miner import _Client, _Generator, _job, _Tokenizer


class _SkippingClient(_Client):
    """A validator's walk bookkeeping: which prompts are full, where the
    hotkey stands, and the three answers next/skip/submit give from them."""

    def __init__(self, job, full, *, answers=None, skip_answers=None):
        super().__init__(answers or ["accepted"] * 100)
        self.job = job
        self.full = set(full)
        self.skips = []
        self.next_reads = 0
        self.skip_answers = list(skip_answers or [])

    def _index(self):
        return job_walk_index(self.job, "5Hot", self.position)

    def next_prompt(self, hotkey):
        self.next_reads += 1
        index = self._index()
        skip_to = self.position + 1
        while (job_walk_index(self.job, "5Hot", skip_to) in self.full
               and skip_to < self.position + 256):
            skip_to += 1
        return {"cursor": self.position, "prompt_index": index,
                "slots_remaining": 0 if index in self.full else 1, "skip_to": skip_to}

    def skip(self, body):
        self.skips.append(body)
        if self.skip_answers:
            return self.skip_answers.pop(0)
        assert body["cursor"] == self.position and body["prompt_index"] == self._index()
        for step in range(body["cursor"], body["to_cursor"]):
            assert job_walk_index(self.job, "5Hot", step) in self.full
        self.position = body["to_cursor"]
        return {"reason": "accepted", "skipped": True, "cursor": self.position}

    def submit(self, body):
        answer = super().submit(body)
        if answer["accepted"]:
            self.full.add(body["prompt_index"])
        return answer


def _mine(client, generator, **kwargs):
    kwargs.setdefault("sign_skip", lambda body: "skipsig")
    kwargs.setdefault("job", _job())
    return mine_steps(hotkey="5Hot", client=client, generator=generator, tokenizer=_Tokenizer(),
                      render=lambda i: f"q{i}", sign=lambda b: "sig", **kwargs)


def _first_open(job, full, start=0):
    cursor = start
    while job_walk_index(job, "5Hot", cursor) in full:
        cursor += 1
    return cursor


def test_full_prompts_are_skipped_without_generating():
    job = _job()
    full = {job_walk_index(job, "5Hot", c) for c in range(3)}
    client, generator = _SkippingClient(job, full), _Generator()
    open_cursor = _first_open(job, full)

    counts = _mine(client, generator, job=job, max_steps=1)

    assert len(generator.prompts) == 1
    # One skip crosses the whole run.
    assert [(b["cursor"], b["to_cursor"]) for b in client.skips] == [(0, open_cursor)]
    assert all(b["signature"] == "skipsig" for b in client.skips)
    assert [(b["cursor"], b["prompt_index"]) for b in client.submitted] == [
        (open_cursor, job_walk_index(job, "5Hot", open_cursor))]
    assert counts == {"skipped": 1, "accepted": 1}


def test_an_open_prompt_is_generated_for_with_no_skip():
    job = _job()
    client, generator = _SkippingClient(job, set()), _Generator()
    counts = _mine(client, generator, job=job, max_steps=3)
    assert client.skips == [] and len(generator.prompts) == 3
    assert counts == {"accepted": 3}


def test_the_miner_generates_only_for_open_prompts_late_in_a_job():
    """Two thirds of the source full: every generation lands on an open prompt."""
    job = _job(prompt_count=50)
    full = {i for i in range(50) if i % 3}
    client, generator = _SkippingClient(job, set(full)), _Generator()
    counts = _mine(client, generator, job=job, max_steps=5)
    assert len(generator.prompts) == 5 == len(client.submitted)
    assert all(b["prompt_index"] not in full for b in client.submitted)
    assert counts["skipped"] == len(client.skips) > 0
    # At most one skip per generation.
    assert len(client.skips) <= 5


def test_a_skip_refused_prompt_not_full_rereads_next_and_generates():
    job = _job()
    index = job_walk_index(job, "5Hot", 0)
    client = _SkippingClient(job, {index}, skip_answers=[
        {"reason": "prompt_not_full", "skipped": False, "slots_remaining": 1}])
    generator = _Generator()
    # Somebody freed nothing, but the validator says the prompt has a slot:
    # between our read and our skip the fake stops calling it full.
    original = client.skip

    def skip(body):
        client.full.discard(index)
        return original(body)

    client.skip = skip
    counts = _mine(client, generator, job=job, max_steps=1)
    assert len(generator.prompts) == 1
    assert client.submitted[0]["cursor"] == 0
    assert counts["skip_prompt_not_full"] == 1


def test_a_skip_that_reports_the_job_complete_stops_without_generating():
    job = _job()
    client = _SkippingClient(job, {job_walk_index(job, "5Hot", 0)},
                             skip_answers=[{"reason": "job_complete", "skipped": False}])
    generator = _Generator()
    counts = _mine(client, generator, job=job, max_steps=5)
    assert generator.prompts == [] and client.submitted == []
    assert counts == {"job_complete": 1}


@pytest.mark.parametrize("reason", ["hotkey_not_registered", "miner_banned"])
def test_a_skip_refusal_no_retry_can_change_halts(reason):
    job = _job()
    client = _SkippingClient(job, {job_walk_index(job, "5Hot", 0)},
                             skip_answers=[{"reason": reason, "skipped": False}])
    generator = _Generator()
    with pytest.raises(CorpusMinerHalted, match=reason):
        _mine(client, generator, job=job, max_steps=5)
    assert generator.prompts == []


def test_a_validator_that_cannot_verify_skips_is_mined_as_before():
    job = _job()
    full = {job_walk_index(job, "5Hot", 0)}
    client = _SkippingClient(job, full, answers=["prompt_full", "accepted", "accepted"],
                             skip_answers=[{"reason": "signature_unverifiable", "skipped": False}])
    generator = _Generator()
    original = client.submit

    def submit(body):
        answer = original(body)
        if answer["reason"] == "prompt_full":
            client.position += 1
        return answer

    client.submit = submit
    counts = _mine(client, generator, job=job, max_steps=3)
    # Asked once, then never again: every step generates, as today.
    assert len(client.skips) == 1
    assert len(generator.prompts) == 3
    assert counts["prompt_full"] == 1 and counts["accepted"] == 2


def test_repeated_stale_skips_give_up_and_generate():
    job = _job()
    client = _SkippingClient(job, {job_walk_index(job, "5Hot", 0)}, skip_answers=[
        {"reason": "bad_cursor", "skipped": False}] * 10)
    generator = _Generator()
    _mine(client, generator, job=job, max_steps=1)
    assert len(generator.prompts) == 1
    assert len(client.skips) <= 4


def test_a_walk_the_validator_disagrees_with_is_left_to_submit():
    job = _job()
    client = _SkippingClient(job, set())
    client.next_prompt = lambda hotkey: {"cursor": 0, "prompt_index": 10_000,
                                         "slots_remaining": 0, "skip_to": 1}
    generator = _Generator()
    _mine(client, generator, job=job, max_steps=1)
    assert client.skips == [] and len(generator.prompts) == 1


def test_without_a_skip_signer_the_miner_never_asks():
    job = _job()
    client = _SkippingClient(job, {job_walk_index(job, "5Hot", 0)},
                             answers=["accepted"])
    generator = _Generator()
    _mine(client, generator, job=job, max_steps=1, sign_skip=None)
    assert client.next_reads == 0 and client.skips == []
    assert len(generator.prompts) == 1


def test_a_free_job_never_skips():
    job = _job()
    job.prompt_order = "free"
    client = _SkippingClient(job, {job_walk_index(job, "5Hot", 0)})
    generator = _Generator()
    _mine(client, generator, job=job, max_steps=2)
    assert client.next_reads == 0 and len(generator.prompts) == 2


def test_an_old_client_without_the_calls_mines_as_before():
    client, generator = _Client(["accepted"] * 2), _Generator()
    counts = _mine(client, generator, max_steps=2)
    assert counts == {"accepted": 2} and len(generator.prompts) == 2


def test_the_skip_body_is_what_the_validator_verifies():
    body = build_skip(job=_job(), hotkey="5Hot", cursor=4, prompt_index=9, to_cursor=7,
                      sign=lambda b: f"signed:{sorted(b)}:{b['signature']!r}")
    assert body == {"job_id": "math-v1", "miner_hotkey": "5Hot", "cursor": 4, "prompt_index": 9,
                    "to_cursor": 7, "signature": "signed:['cursor', 'job_id', 'miner_hotkey', "
                    "'prompt_index', 'signature', 'to_cursor']:''"}


@pytest.mark.parametrize("skip_to", [None, 0, "x"])
def test_a_next_without_a_usable_skip_to_turns_skipping_off(skip_to):
    job = _job()
    client = _SkippingClient(job, {job_walk_index(job, "5Hot", 0)}, answers=["prompt_full"] * 3)
    answer = {"cursor": 0, "prompt_index": job_walk_index(job, "5Hot", 0), "slots_remaining": 0}
    if skip_to is not None:
        answer["skip_to"] = skip_to
    client.next_prompt = lambda hotkey: answer
    generator = _Generator()
    _mine(client, generator, job=job, max_steps=2)
    assert client.skips == [] and len(generator.prompts) == 2


# --------------------------------------------------------------------------
# over HTTP
# --------------------------------------------------------------------------


def _validator(*, has_skip, job_id=None):
    seen = []
    job = _job()

    def handle(request):
        seen.append((request.method, request.url.path))
        path = request.url.path
        prefix = "/corpus" if job_id is None else f"/corpus/jobs/{job_id}"
        if path == f"{prefix}/cursor/5Hot":
            return httpx.Response(200, json={"hotkey": "5Hot", "cursor": 0})
        if has_skip and path == f"{prefix}/next/5Hot":
            return httpx.Response(200, json={"cursor": 0, "prompt_index":
                                             job_walk_index(job, "5Hot", 0), "slots_remaining": 1,
                                             "skip_to": 1})
        if has_skip and path == f"{prefix}/skip":
            return httpx.Response(200, json={"reason": "accepted", "skipped": True, "cursor": 1})
        if path == "/corpus/submit":
            return httpx.Response(200, json={"accepted": True, "reason": "accepted"})
        return httpx.Response(404, json={"detail": "Not Found"})

    return httpx.Client(transport=httpx.MockTransport(handle), base_url="http://v"), seen


@pytest.mark.parametrize("job_id", [None, "math-v1"])
def test_the_client_uses_the_legacy_or_scoped_next_and_skip(job_id):
    http, seen = _validator(has_skip=True, job_id=job_id)
    client = HttpCorpusClient(http, job_id=job_id)
    prefix = "/corpus" if job_id is None else f"/corpus/jobs/{job_id}"
    assert client.next_prompt("5Hot")["slots_remaining"] == 1
    assert client.skip({"job_id": "math-v1"})["skipped"] is True
    assert seen == [("GET", f"{prefix}/next/5Hot"), ("POST", f"{prefix}/skip")]


@pytest.mark.parametrize("job_id", [None, "math-v1"])
def test_an_old_validator_answers_none_and_the_miner_falls_back(job_id):
    http, seen = _validator(has_skip=False, job_id=job_id)
    client = HttpCorpusClient(http, job_id=job_id)
    assert client.next_prompt("5Hot") is None
    assert client.skip({"job_id": "math-v1"}) is None

    seen.clear()
    generator = _Generator()
    counts = mine_steps(job=_job(), hotkey="5Hot", client=client, generator=generator,
                        tokenizer=_Tokenizer(), render=lambda i: f"q{i}", sign=lambda b: "sig",
                        sign_skip=lambda b: "skipsig", max_steps=3)
    assert counts == {"accepted": 3} and len(generator.prompts) == 3
    # One probe, then silence: the 404 is remembered for the run.
    assert [p for _, p in seen].count(
        "/corpus/next/5Hot" if job_id is None else f"/corpus/jobs/{job_id}/next/5Hot") == 1


def test_a_skip_route_that_fails_otherwise_is_not_mistaken_for_an_old_validator():
    from reliquary.miner.corpus_miner import CorpusPermanentFailure

    def handle(request):
        return httpx.Response(500, json={"detail": "corpus_ledger_corrupt"})

    client = HttpCorpusClient(httpx.Client(transport=httpx.MockTransport(handle), base_url="http://v"))
    with pytest.raises(CorpusPermanentFailure):
        client.next_prompt("5Hot")


def test_the_corpus_mine_command_signs_skips_with_the_skip_binding():
    import inspect

    from reliquary.cli import main

    source = inspect.getsource(main.corpus_mine)
    assert "sign_skip=lambda body: sign_corpus_skip(wallet, body)" in source
    assert "sign=lambda body: sign_corpus_submission(wallet, body)" in source
