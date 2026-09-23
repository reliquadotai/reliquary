"""One submission, end to end, against the real admission logic, the real job
store, and a fake bucket. The endpoint's whole job is to derive what admit()
must not be told."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reliquary.corpus.job import parse_job
from reliquary.environment.agentic.types import EpisodeTask
from reliquary.infrastructure import corpus_job_store as job_store
from reliquary.protocol.corpus_submission import CorpusSubmissionRequest

# The fake bucket is written once, in the job store's own tests, and proved
# there. The endpoint has to be exercised against the REAL store — its key
# layout and its compare-and-swap — so a second fake would only drift from it.
from tests.unit.test_corpus_job_store import _FakeMultiObjectR2

EOS = 151645
CHECKPOINT = "a" * 64
# A real installed environment the fidelity adapter genuinely serves: episode
# mode, procedural rows, no dataset and no network.
PROMPT_SOURCE = "reliquary_stateful_tools_v1"


def _manifest():
    return {
        "schema": "reliquary/corpus-job/v1",
        "job_id": "swe-v1",
        "checkpoint_repo": "org/Frozen",
        "checkpoint_revision": "abc123",
        "checkpoint_sha256": CHECKPOINT,
        "prompt_source": PROMPT_SOURCE,
        "prompt_count": 1000,
        "renderer_id": "reliquary-jsonl-tools-v1",
        "eos_token_id": EOS,
        "sampling": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_new_tokens": 16,
            "max_new_tokens": 4096,
            "n": 1,
        },
        "slots_per_prompt": 8,
        "filter": None,
        "prompt_order": "free",
        "deadline_round": 5_000_000,
    }


class _Tokenizer:
    """Decodes an id to its own digits, so a test can reason about the text."""

    def decode(self, ids, **kwargs):
        return "".join(str(i) for i in ids)


class _Renderer:
    """A stable string per task, standing in for a real episode renderer."""

    def initial_text(self, task):
        return f"<prompt {task.id}>"


class _Environment:
    """The prompt source, indexable without a dataset download."""

    name = PROMPT_SOURCE

    def __init__(self, rows: int = 1000):
        self._rows = rows

    def __len__(self):
        return self._rows

    def get_task(self, index):
        return EpisodeTask(id=f"row-{index}", prompt=f"question {index}", tools=())


class _Spec:
    """Stands in for an EnvironmentSpec: the adapter must not build the real
    one, which downloads a dataset."""

    interaction_mode = "episode"

    def __init__(self, environment):
        self._environment = environment

    def create(self):
        return self._environment


class _BlockingSpec:
    """A source whose build costs real time, the way a dataset-backed one does.

    It records whether it was built with an event loop running in its own
    thread, which is exactly the question: on the loop, or off it."""

    interaction_mode = "episode"

    def __init__(self, environment):
        self._environment = environment
        self.built_on_the_event_loop = None

    def create(self):
        import time

        try:
            asyncio.get_running_loop()
            self.built_on_the_event_loop = True
        except RuntimeError:
            self.built_on_the_event_loop = False
        time.sleep(0.05)
        return self._environment


class _UnbuildableSpec:
    """A source that is installed and episode-mode but cannot be built — a
    missing corpus directory, a broken wheel."""

    interaction_mode = "episode"

    def create(self):
        raise RuntimeError("corpus shard is missing")


class _CountingStore:
    """The real store bound to the fake bucket, counting ledger writes and
    able to lose a compare-and-swap race on demand."""

    def __init__(self, client_kwargs):
        self._kwargs = client_kwargs
        self.ledger_write_attempts = 0
        self.job_reads = 0
        self._conflicts_left = 0
        self._competing_snapshot = None

    async def read_job(self, job_id):
        self.job_reads += 1
        return await job_store.read_job(job_id, **self._kwargs)

    async def read_ledgers(self, job_id):
        return await job_store.read_ledgers(job_id, **self._kwargs)

    async def write_ledgers(self, job_id, snapshot, etag):
        self.ledger_write_attempts += 1
        if self._conflicts_left > 0:
            self._conflicts_left -= 1
            # A real conflict means somebody else's bytes landed, so the fake
            # lands them: retrying against the stale ETag must not succeed.
            competing = self._competing_snapshot
            if competing is None:
                competing, _ = await self.read_ledgers(job_id)
            await job_store.write_ledgers(job_id, competing, etag, **self._kwargs)
            raise job_store.CorpusStoreConflict("injected")
        return await job_store.write_ledgers(job_id, snapshot, etag, **self._kwargs)


class _SeededJob:
    def __init__(self, store, renderer, environment, raw):
        self.store = store
        self.renderer = renderer
        self.environment = environment
        self.raw = raw
        self.prompt_job_calls = 0

    @property
    def job(self):
        return parse_job(self.raw)

    def ledger_writes(self):
        return self.store.ledger_write_attempts

    def fail_next_ledger_write_with_conflict(self, *, competing_snapshot=None, times=1):
        self.store._conflicts_left = times
        self.store._competing_snapshot = competing_snapshot

    def seed_ledgers(self, snapshot):
        asyncio.run(
            job_store.write_ledgers("swe-v1", snapshot, None, **self.store._kwargs)
        )

    def environments(self, environment=None):
        return {PROMPT_SOURCE: _Spec(environment or self.environment)}

    def prompt_job_for(self, job):
        """The endpoint's seam, counting how often a job's source is resolved."""
        from reliquary.validator.corpus_service import prompt_job_for_spec

        self.prompt_job_calls += 1
        return prompt_job_for_spec(job, environments=self.environments())


@pytest.fixture
def _r2_client(monkeypatch):
    client = _FakeMultiObjectR2()
    monkeypatch.setattr(job_store, "get_s3_client", lambda **kw: client)
    return client


@pytest.fixture
def fake_r2(_r2_client):
    """The kwargs the store takes; the fake client is installed by monkeypatch."""
    return {}


@pytest.fixture
def seeded_job(fake_r2):
    raw = _manifest()
    asyncio.run(job_store.write_job(raw, None, **fake_r2))
    return _SeededJob(_CountingStore(fake_r2), _Renderer(), _Environment(), raw)


@pytest.fixture
def client(fake_r2, seeded_job):
    """A mounted router over a fake store, an identity tokenizer, and a
    signature verifier that accepts everything except the literal "bad"."""
    from reliquary.validator.corpus_service import build_corpus_router

    app = FastAPI()
    app.include_router(
        build_corpus_router(
            job_id="swe-v1",
            store=seeded_job.store,
            tokenizer=_Tokenizer(),
            renderer=seeded_job.renderer,
            verify_signature=lambda request: request.signature != "bad",
            prompt_job_for=seeded_job.prompt_job_for,
        )
    )
    return TestClient(app)


def _client_over(seeded_job, spec):
    """The same router, over a prompt source the test supplies."""
    from reliquary.validator.corpus_service import (
        build_corpus_router,
        prompt_job_for_spec,
    )

    app = FastAPI()
    app.include_router(
        build_corpus_router(
            job_id="swe-v1",
            store=seeded_job.store,
            tokenizer=_Tokenizer(),
            renderer=seeded_job.renderer,
            verify_signature=lambda request: True,
            prompt_job_for=lambda job: prompt_job_for_spec(
                job, environments={PROMPT_SOURCE: spec}
            ),
        )
    )
    return TestClient(app)


def _client_for_job(seeded_job, job_id):
    """The same router, serving a job id the test chooses. Production takes
    that id from the registry entry that pays for the job."""
    from reliquary.validator.corpus_service import build_corpus_router

    app = FastAPI()
    app.include_router(
        build_corpus_router(
            job_id=job_id,
            store=seeded_job.store,
            tokenizer=_Tokenizer(),
            renderer=seeded_job.renderer,
            verify_signature=lambda request: True,
            prompt_job_for=seeded_job.prompt_job_for,
        )
    )
    return TestClient(app)


def _faithful_prompt(prompt_index):
    """What the job's own renderer makes of that slot's source row."""
    return f"<prompt row-{prompt_index}>"


def _text_for(tokens):
    """What an honest miner sends: the decode of its tokens with the one
    trailing terminator dropped (ruling T3-a)."""
    body = tokens[:-1] if tokens and tokens[-1] == EOS else tokens
    return "".join(str(t) for t in body)


def _submit(
    client,
    *,
    tokens,
    text=None,
    prompt_index=0,
    cursor=0,
    signature="ok",
    rendered_prompt=None,
):
    """Build through the wire model so the schema is exercised, not bypassed."""
    request = CorpusSubmissionRequest(
        job_id="swe-v1",
        miner_hotkey="5Hot",
        cursor=cursor,
        prompt_index=prompt_index,
        checkpoint_sha256=CHECKPOINT,
        rendered_prompt=(
            _faithful_prompt(prompt_index) if rendered_prompt is None else rendered_prompt
        ),
        completions=[
            {
                "tokens": tokens,
                "text": _text_for(tokens) if text is None else text,
            }
        ],
        signature=signature,
    )
    return client.post("/corpus/submit", json=request.model_dump())


def test_a_well_formed_submission_is_accepted_and_consumes_a_slot(client):
    body = _submit(client, tokens=[7] * 16 + [EOS]).json()
    assert body["accepted"] is True
    assert body["slots_remaining"] == 7


def test_the_endpoint_derives_the_termination_from_the_tokens(client):
    """A completion that neither ends on the terminator nor reached the cap is
    a silent truncation. The wire carries no label to compare it against: the
    only source of one is `tokens[-1]`, here and in `check_termination`.

    Its counterpart -- a cap LABEL on an eos-ending completion, accepted --
    went with the field it depended on: the label is now unrepresentable on
    the wire rather than merely unread, so the case it discriminated cannot be
    built. `test_a_completion_may_not_declare_how_it_terminated` pins that."""
    body = _submit(client, tokens=[7] * 16 + [99]).json()
    assert body["accepted"] is False
    assert "termination" in body["reason"]


def test_a_completion_below_the_floor_is_refused(client, seeded_job):
    """`token_counts` is derived too, and nothing else in this file crosses
    either budget bound: pin it to `min_new_tokens` and every other test here
    stays green while a one-token completion gets paid."""
    before = seeded_job.ledger_writes()
    body = _submit(client, tokens=[7] * 8 + [EOS]).json()
    assert body["accepted"] is False
    assert body["reason"] == "token_budget_underrun"
    assert body["detail"]["tokens"] == 9
    assert seeded_job.ledger_writes() == before


def test_the_cheapest_possible_completion_is_refused(client):
    # One EOS token and empty text: what a pinned token count would buy.
    body = _submit(client, tokens=[EOS], text="").json()
    assert body["accepted"] is False
    assert body["reason"] == "token_budget_underrun"
    assert body["detail"]["tokens"] == 1


def test_the_empty_completion_is_refused_at_the_lowest_floor_a_job_may_declare(
    fake_r2, seeded_job
):
    """The boundary the CLI's own default used to sit on. Every fixture in
    this file declares a floor of 16, so nothing here approaches the floor a
    job may actually choose: at `min_new_tokens=1`, `tokens=[eos]` with
    `text=""` clears the budget, the termination check and the text check, and
    is paid a slot for an empty corpus row. The parser now refuses a floor of
    1, so this is the cheapest job that exists -- and it still refuses."""
    from reliquary.corpus.job import MIN_NEW_TOKENS_FLOOR
    from reliquary.validator.corpus_service import build_corpus_router

    raw = _manifest()
    raw["job_id"] = "floor-v1"
    raw["sampling"] = {**raw["sampling"], "min_new_tokens": MIN_NEW_TOKENS_FLOOR}
    asyncio.run(job_store.write_job(raw, None, **fake_r2))

    app = FastAPI()
    app.include_router(
        build_corpus_router(
            job_id="floor-v1",
            store=seeded_job.store,
            tokenizer=_Tokenizer(),
            renderer=seeded_job.renderer,
            verify_signature=lambda request: True,
            prompt_job_for=seeded_job.prompt_job_for,
        )
    )
    client = TestClient(app)

    def _post(tokens, text):
        request = CorpusSubmissionRequest(
            job_id="floor-v1",
            miner_hotkey="5Hot",
            cursor=0,
            prompt_index=0,
            checkpoint_sha256=CHECKPOINT,
            rendered_prompt=_faithful_prompt(0),
            completions=[{"tokens": tokens, "text": text}],
            signature="ok",
        )
        return client.post("/corpus/submit", json=request.model_dump()).json()

    before = seeded_job.ledger_writes()
    empty = _post([EOS], "")
    assert empty["accepted"] is False
    assert empty["reason"] == "token_budget_underrun"
    # Nothing moved: an accepted empty completion burns its slot permanently.
    assert seeded_job.ledger_writes() == before

    # One token of content at the same floor is accepted, so the refusal above
    # is the floor doing its job rather than this job refusing everything.
    assert _post([7, EOS], "7")["accepted"] is True


def test_a_completion_over_the_budget_is_refused(client, seeded_job):
    before = seeded_job.ledger_writes()
    body = _submit(client, tokens=[7] * 4097).json()
    assert body["accepted"] is False
    assert body["reason"] == "token_budget_exceeded"
    assert body["detail"]["tokens"] == 4097
    assert seeded_job.ledger_writes() == before


def test_a_completion_exactly_at_the_cap_without_eos_is_accepted(client):
    # The other side of the same bound, so the test above cannot be satisfied
    # by refusing everything long.
    body = _submit(client, tokens=[7] * 4096).json()
    assert body["accepted"] is True


def test_a_miner_walk_job_advances_the_cursor_it_persists(fake_r2, seeded_job):
    """Every other fixture in this file is `prompt_order="free"`, where the
    cursor is never read or written: replace `cursors.snapshot()` with `{}` in
    `ledger_snapshot` and 104 tests across six files stay green. In production
    that resets every miner on every request, so a `miner_walk` job serves one
    prompt per hotkey forever. This is the test that sees it."""
    from reliquary.corpus.walk import walk_index
    from reliquary.validator.corpus_service import build_corpus_router

    raw = _manifest()
    raw["job_id"] = "walk-v1"
    raw["prompt_order"] = "miner_walk"
    asyncio.run(job_store.write_job(raw, None, **fake_r2))

    app = FastAPI()
    app.include_router(
        build_corpus_router(
            job_id="walk-v1",
            store=seeded_job.store,
            tokenizer=_Tokenizer(),
            renderer=seeded_job.renderer,
            verify_signature=lambda request: True,
            prompt_job_for=seeded_job.prompt_job_for,
        )
    )
    client = TestClient(app)

    def _post(cursor, prompt_index, filler):
        tokens = [filler] * 16 + [EOS]
        request = CorpusSubmissionRequest(
            job_id="walk-v1",
            miner_hotkey="5Hot",
            cursor=cursor,
            prompt_index=prompt_index,
            checkpoint_sha256=CHECKPOINT,
            rendered_prompt=_faithful_prompt(prompt_index),
            completions=[
                {
                    "tokens": tokens,
                    "text": _text_for(tokens),
                }
            ],
            signature="ok",
        )
        return client.post("/corpus/submit", json=request.model_dump()).json()

    first_index = walk_index("walk-v1", "5Hot", 0, 1000)
    second_index = walk_index("walk-v1", "5Hot", 1, 1000)
    assert _post(0, first_index, 1)["accepted"] is True
    # The second submission only parses if the FIRST one's cursor survived the
    # round trip through the bucket: this is the read half.
    assert _post(1, second_index, 2)["accepted"] is True

    # And the write half, read back out of the store rather than off the
    # response, because the response would look identical either way.
    snapshot, _ = asyncio.run(job_store.read_ledgers("walk-v1", **fake_r2))
    assert snapshot["cursors"] == {"5Hot": 2}

    # A miner cannot pick its own step back: the cursor is where the ledger
    # says it is, not where the request says.
    replayed = _post(0, first_index, 3)
    assert replayed["accepted"] is False
    assert replayed["reason"] == "bad_cursor"
    assert replayed["detail"] == {"expected": 2, "got": 0}


def test_the_endpoint_derives_the_digest_from_the_tokens(client):
    """Two submissions with the same tokens must collide as duplicates
    whatever they declare, because the digest is computed here."""
    tokens = [7] * 16 + [EOS]
    assert _submit(client, tokens=tokens).json()["accepted"] is True
    second = _submit(client, tokens=tokens, prompt_index=0).json()
    assert second["accepted"] is False
    assert "duplicate" in second["reason"]


def test_the_same_tokens_under_another_prompt_are_not_a_duplicate(client):
    """The digest binds the tokens to the prompt they answer, so the endpoint
    must feed `completion_digest` the prompt index, not the tokens alone."""
    tokens = [7] * 16 + [EOS]
    assert _submit(client, tokens=tokens, prompt_index=0).json()["accepted"] is True
    assert _submit(client, tokens=tokens, prompt_index=1).json()["accepted"] is True


def _submission_for(job_id):
    return CorpusSubmissionRequest(
        job_id=job_id,
        miner_hotkey="5Hot",
        cursor=0,
        prompt_index=0,
        checkpoint_sha256=CHECKPOINT,
        rendered_prompt=_faithful_prompt(0),
        completions=[{"tokens": [1], "text": "1"}],
        signature="ok",
    )


def test_a_submission_for_another_job_never_reaches_this_validators_store(
    client, seeded_job
):
    """A validator is paid out of ONE task's share, for ONE job. Serving a
    submission for another job spends this task's slots and this task's
    bucket writes on work nobody here is paid for, so it is refused before
    the manifest is even read."""
    before_reads = seeded_job.store.job_reads
    before_writes = seeded_job.ledger_writes()

    body = client.post(
        "/corpus/submit", json=_submission_for("other-job").model_dump()
    ).json()

    assert body["accepted"] is False
    assert body["reason"] == "job_not_served"
    assert body["detail"]["serves"] == "swe-v1"
    assert seeded_job.store.job_reads == before_reads
    assert seeded_job.ledger_writes() == before_writes


def test_the_store_is_keyed_off_the_served_job_not_the_request(client, seeded_job):
    """The guard makes the two equal, so this cannot be observed through the
    wire: it is asserted on the calls themselves, because the property is that
    a request field never reaches a bucket key -- not that it happens to hold
    the right value today."""
    import inspect

    from reliquary.validator import corpus_service

    source = inspect.getsource(corpus_service.build_corpus_router)
    body = source[source.index("async def submit_corpus") :]
    for call in ("store.read_job(", "store.read_ledgers(", "store.write_ledgers("):
        argument = body[body.index(call) + len(call) :].lstrip()
        assert argument.startswith("job_id"), f"{call} is keyed off {argument[:40]!r}"


def test_the_job_this_validator_serves_is_still_read(client, seeded_job):
    """The other direction, so the refusal above cannot be satisfied by a
    router that refuses every job id it is given."""
    before_reads = seeded_job.store.job_reads

    body = client.post(
        "/corpus/submit", json=_submission_for("swe-v1").model_dump()
    ).json()

    assert body["reason"] != "job_not_served"
    assert seeded_job.store.job_reads == before_reads + 1


def test_a_validator_that_cannot_verify_says_so_instead_of_bad_signature(
    seeded_job, fake_r2
):
    """`bad_signature` tells a miner its signature was wrong. When this binary
    carries no corpus binding at all, the miner's signature was never the
    problem, and a miner debugging against the wrong reason burns real time."""
    from reliquary.validator.corpus_service import (
        build_corpus_router,
        refuse_unsigned_corpus_submissions,
    )

    app = FastAPI()
    app.include_router(
        build_corpus_router(
            job_id="swe-v1",
            store=seeded_job.store,
            tokenizer=_Tokenizer(),
            renderer=seeded_job.renderer,
            verify_signature=refuse_unsigned_corpus_submissions,
            prompt_job_for=seeded_job.prompt_job_for,
        )
    )
    unverifiable = TestClient(app)

    body = unverifiable.post(
        "/corpus/submit", json=_submission_for("swe-v1").model_dump()
    ).json()

    assert body["accepted"] is False
    assert body["reason"] == "signature_unverifiable"


def test_a_signature_that_does_not_verify_is_still_a_bad_signature(client):
    """The other direction: a verifier that CAN verify and refuses must not be
    reported as this validator's own failure."""
    body = _submit(client, tokens=[7] * 16 + [EOS], signature="bad").json()
    assert body["reason"] == "bad_signature"


def test_a_submission_for_an_unknown_job_is_refused_without_touching_the_ledgers(
    seeded_job
):
    """"Not mine" and "no such job" are different operational facts: this one
    is a task declaring a job whose manifest is absent from the bucket."""
    before = seeded_job.ledger_writes()
    ghost = _client_for_job(seeded_job, "ghost")

    body = ghost.post(
        "/corpus/submit", json=_submission_for("ghost").model_dump()
    ).json()

    assert body["accepted"] is False
    assert body["reason"] == "job_unknown"
    assert seeded_job.ledger_writes() == before


def test_a_job_id_that_could_never_name_a_job_is_refused_not_raised(seeded_job):
    """The id becomes a bucket key, so the store refuses it outright. The
    registry checks only that a corpus entry NAMES a job, not that the name
    is usable, so a hand-edited entry reaches the store: that must read as
    "no such job", not as a 500."""
    before = seeded_job.ledger_writes()
    hostile = _client_for_job(seeded_job, "../../etc/passwd")

    response = hostile.post(
        "/corpus/submit", json=_submission_for("../../etc/passwd").model_dump()
    )

    assert response.status_code == 200
    assert response.json()["reason"] == "job_unknown"
    assert seeded_job.ledger_writes() == before


def test_a_submission_whose_signature_does_not_verify_never_reaches_admit(
    client, seeded_job
):
    before = seeded_job.ledger_writes()
    body = _submit(client, tokens=[7] * 16 + [EOS], signature="bad").json()
    assert body["accepted"] is False
    assert body["reason"] == "bad_signature"
    assert seeded_job.ledger_writes() == before


def test_a_full_slot_is_reported_with_slots_remaining_zero(client):
    for i in range(8):
        body = _submit(client, tokens=[i] * 16 + [EOS]).json()
        assert body["accepted"] is True, body
    assert body["slots_remaining"] == 0
    refused = _submit(client, tokens=[42] * 16 + [EOS]).json()
    assert refused["accepted"] is False
    assert refused["reason"] == "prompt_full"
    assert refused["slots_remaining"] == 0


def test_text_that_does_not_match_its_tokens_is_refused(client, seeded_job):
    before = seeded_job.ledger_writes()
    body = _submit(client, tokens=[7] * 16 + [EOS], text="").json()
    assert body["accepted"] is False
    assert body["reason"] == "text_does_not_match_tokens"
    # The money leak: 17 tokens contributing nothing must not buy a slot.
    assert seeded_job.ledger_writes() == before


def test_an_honest_miner_that_omits_the_trailing_terminator_is_accepted(client):
    """Measured against the production tokenizer: ordinary generation never
    returns the terminator's text, so this is the honest default."""
    tokens = [7] * 16 + [EOS]
    body = _submit(client, tokens=tokens, text="7" * 16).json()
    assert body["accepted"] is True


def test_a_conflicting_ledger_write_is_retried_and_the_slot_consumed_once(
    client, seeded_job
):
    """Two validators admitting concurrently must not double-consume. The
    store's compare-and-swap is what makes this true; the endpoint has to
    honour it rather than writing blind."""
    seeded_job.fail_next_ledger_write_with_conflict()
    body = _submit(client, tokens=[7] * 16 + [EOS]).json()
    assert body["accepted"] is True
    assert body["slots_remaining"] == 7
    assert seeded_job.ledger_writes() == 2


def test_the_retry_admits_against_the_winner_s_ledgers_not_its_own(client, seeded_job):
    """The stronger half of the same rule: the writer that lost the race must
    re-admit against what the winner left, or it erases the winner's three
    consumed slots and sells them again."""
    seeded_job.fail_next_ledger_write_with_conflict(
        competing_snapshot={"slots": {"0": 3}, "cursors": {}, "seen": []}
    )
    body = _submit(client, tokens=[7] * 16 + [EOS]).json()
    assert body["accepted"] is True
    assert body["slots_remaining"] == 4


def test_a_ledger_that_never_settles_is_a_transient_failure_not_a_lost_slot(
    client, seeded_job
):
    seeded_job.fail_next_ledger_write_with_conflict(times=99)
    response = _submit(client, tokens=[7] * 16 + [EOS])
    assert response.status_code == 503
    # Nothing was consumed, so the same work resubmits cleanly.
    seeded_job.fail_next_ledger_write_with_conflict(times=0)
    body = _submit(client, tokens=[7] * 16 + [EOS]).json()
    assert body["accepted"] is True
    assert body["slots_remaining"] == 7


def test_a_corrupt_ledger_snapshot_is_named_rather_than_a_bare_lookup_error(
    client, seeded_job
):
    """The store moves ledger snapshots as raw dicts and has no job in scope
    to check them against, so this layer is the only one that can. It answers
    with a name: a bare "Internal Server Error" is indistinguishable from a
    bug, and the blast radius is every miner on the job."""
    from reliquary.validator.corpus_service import LedgerSnapshotError, rebuild_ledgers

    seeded_job.seed_ledgers({"slots": {"99999999": 1}, "cursors": {}, "seen": []})
    response = _submit(client, tokens=[7] * 16 + [EOS])
    assert response.status_code == 500
    assert response.json()["detail"] == "corpus_ledger_corrupt"

    with pytest.raises(LedgerSnapshotError) as caught:
        rebuild_ledgers(seeded_job.job, {"slots": {"99999999": 1}})
    assert "swe-v1" in str(caught.value)


def test_a_ledger_snapshot_with_a_field_this_binary_cannot_read_is_refused(
    client, seeded_job
):
    # An older reader that ignored it would delete it on its next write.
    seeded_job.seed_ledgers({"slots": {}, "cursors": {}, "seen": [], "reserved": {}})
    response = _submit(client, tokens=[7] * 16 + [EOS])
    assert response.status_code == 500
    assert response.json()["detail"] == "corpus_ledger_corrupt"


def test_a_ledger_snapshot_carrying_a_digest_that_is_not_one_is_refused(
    client, seeded_job
):
    seeded_job.seed_ledgers({"slots": {}, "cursors": {}, "seen": [17]})
    response = _submit(client, tokens=[7] * 16 + [EOS])
    assert response.status_code == 500
    assert response.json()["detail"] == "corpus_ledger_corrupt"


def test_a_prompt_the_job_never_assigned_is_refused(client, seeded_job):
    """Spec §7's prompt fidelity, now that the wire carries the prompt: a
    miner answering an easier question than its slot names is not paid."""
    before = seeded_job.ledger_writes()
    body = _submit(
        client,
        tokens=[7] * 16 + [EOS],
        rendered_prompt=_faithful_prompt(0) + "\nHint: the answer is 42.",
    ).json()
    assert body["accepted"] is False
    assert body["reason"] == "prompt_not_faithful"
    assert seeded_job.ledger_writes() == before


def test_a_prompt_rendered_for_another_slot_is_refused(client):
    body = _submit(
        client, tokens=[7] * 16 + [EOS], prompt_index=3,
        rendered_prompt=_faithful_prompt(4),
    ).json()
    assert body["accepted"] is False
    assert body["reason"] == "prompt_not_faithful"


def test_a_prompt_index_past_the_source_cannot_raise_out_of_the_adapter(client):
    """The fidelity check indexes the prompt source with a number the miner
    chose, so it has to be bounded before it reaches the environment."""
    response = _submit(client, tokens=[7] * 16 + [EOS], prompt_index=5000)
    assert response.status_code == 200
    assert response.json()["reason"] == "prompt_mismatch"


def test_the_prompt_source_is_resolved_once_per_job_not_once_per_request(
    client, seeded_job
):
    # Resolving a real source builds its environment; doing it per submission
    # would put a dataset load on the request path.
    for index in range(3):
        assert _submit(
            client, tokens=[index] * 16 + [EOS], prompt_index=index
        ).json()["accepted"] is True
    assert seeded_job.prompt_job_calls == 1


def test_a_refusal_that_moves_nothing_costs_no_ledger_write(client, seeded_job):
    """A miner spraying junk must not bill a bucket write per attempt."""
    before = seeded_job.ledger_writes()
    for _ in range(5):
        assert _submit(client, tokens=[7] * 16 + [99]).json()["accepted"] is False
    assert seeded_job.ledger_writes() == before


def test_a_manifest_that_no_longer_parses_is_named_not_called_unknown(
    client, seeded_job, _r2_client
):
    """`JobError` subclasses `ValueError`, so dropping its clause would
    disguise an operator's corrupt manifest as a miner's unknown job."""
    _r2_client.objects["reliquary/corpus/jobs/swe-v1.json"] = (
        b'{"schema": "nope"}',
        '"tampered"',
    )
    response = _submit(client, tokens=[7] * 16 + [EOS])
    assert response.status_code == 500
    assert response.json()["detail"] == "corpus_job_manifest_corrupt"


def test_the_ledger_object_carries_the_schema_it_was_written_under(
    client, seeded_job, _r2_client
):
    """Independently-operated validators share this object; without a marker,
    a later field could only be added by breaking every older reader."""
    import json

    from reliquary.validator.corpus_service import LEDGER_SCHEMA

    assert _submit(client, tokens=[7] * 16 + [EOS]).json()["accepted"] is True
    body, _ = _r2_client.objects["reliquary/corpus/jobs/swe-v1/ledgers.json"]
    assert json.loads(body)["schema"] == LEDGER_SCHEMA


def test_ledgers_written_under_another_schema_are_refused(client, seeded_job):
    seeded_job.seed_ledgers(
        {"schema": "reliquary/corpus-ledgers/v2", "slots": {}, "cursors": {}, "seen": []}
    )
    response = _submit(client, tokens=[7] * 16 + [EOS])
    assert response.status_code == 500
    assert response.json()["detail"] == "corpus_ledger_corrupt"


def test_the_prompt_source_is_never_built_on_the_event_loop(seeded_job):
    """Building a dataset-backed source reads from disk. Inside the handler
    that stalls every other request this validator is serving, which is the
    shape that froze `/state` once already."""
    spec = _BlockingSpec(seeded_job.environment)
    client = _client_over(seeded_job, spec)

    assert _submit(client, tokens=[7] * 16 + [EOS]).json()["accepted"] is True
    assert spec.built_on_the_event_loop is False


def test_a_prompt_source_that_cannot_be_built_is_named_not_anonymous(seeded_job):
    client = _client_over(seeded_job, _UnbuildableSpec())

    response = _submit(client, tokens=[7] * 16 + [EOS])
    assert response.status_code == 500
    assert response.json()["detail"] == "corpus_prompt_source_unusable"


def test_a_build_failure_carries_what_broke(seeded_job):
    from reliquary.validator.corpus_service import (
        CorpusPromptSourceError,
        prompt_job_for_spec,
    )

    with pytest.raises(CorpusPromptSourceError, match="corpus shard is missing"):
        prompt_job_for_spec(
            seeded_job.job, environments={PROMPT_SOURCE: _UnbuildableSpec()}
        )


def test_only_a_bounded_number_of_prompt_sources_is_held(seeded_job):
    """Each resolved source holds a built environment, so a cache keyed by job
    id grows with every job this process has ever served."""
    from reliquary.validator.corpus_service import (
        MAX_RESOLVED_PROMPT_SOURCES,
        PromptFidelity,
        prompt_job_for_spec,
    )

    assert MAX_RESOLVED_PROMPT_SOURCES < 1000
    built = []

    def resolve(job):
        built.append(job.job_id)
        return prompt_job_for_spec(job, environments=seeded_job.environments())

    fidelity = PromptFidelity(
        renderer=seeded_job.renderer, prompt_job_for=resolve, max_jobs=2
    )
    jobs = [parse_job({**seeded_job.raw, "job_id": f"job-{i}"}) for i in range(3)]

    async def visit(job):
        return await fidelity(_faithful_prompt(0), job=job, prompt_index=0)

    async def run():
        for job in jobs:
            assert (await visit(job)).ok
        # The third arrival evicted the first, so it costs one rebuild.
        assert (await visit(jobs[0])).ok
        # The most recent two are still held.
        assert (await visit(jobs[2])).ok

    asyncio.run(run())

    assert built == ["job-0", "job-1", "job-2", "job-0"]


def test_the_prompt_source_resolves_to_the_job_s_own_rows(seeded_job):
    """The adapter `check_prompt_fidelity` needs: a JobSpec names a prompt
    source, and the environment behind that name supplies the rows."""
    from reliquary.validator.corpus_service import prompt_job_for_spec

    prompts = prompt_job_for_spec(
        seeded_job.job, environments=seeded_job.environments()
    )

    assert prompts.task_for(7).id == "row-7"


def test_a_prompt_source_that_is_not_an_installed_environment_is_named(seeded_job):
    from reliquary.validator.corpus_service import (
        CorpusPromptSourceError,
        prompt_job_for_spec,
    )

    with pytest.raises(CorpusPromptSourceError) as caught:
        prompt_job_for_spec(seeded_job.job, environments={})
    assert PROMPT_SOURCE in str(caught.value)


def test_a_prompt_source_shorter_than_the_manifest_claims_is_refused(seeded_job):
    """Every slot past the last row would name a prompt nobody can render."""
    from reliquary.validator.corpus_service import (
        CorpusPromptSourceError,
        prompt_job_for_spec,
    )

    with pytest.raises(CorpusPromptSourceError):
        prompt_job_for_spec(
            seeded_job.job, environments=seeded_job.environments(_Environment(rows=10))
        )


def test_the_router_carries_the_prompt_fidelity_seam(seeded_job):
    """Bound on the router because the check needs the job's environment AND
    the job's renderer, and nothing else in the validator holds both."""
    from reliquary.validator.corpus_service import (
        build_corpus_router,
        prompt_job_for_spec,
    )

    environments = seeded_job.environments()
    router = build_corpus_router(
        job_id="swe-v1",
        store=seeded_job.store,
        tokenizer=_Tokenizer(),
        renderer=seeded_job.renderer,
        verify_signature=lambda request: True,
        prompt_job_for=lambda job: prompt_job_for_spec(job, environments=environments),
    )
    job = seeded_job.job

    async def run():
        # One loop for both, because the seam holds an asyncio.Lock and a
        # second `asyncio.run` would hand it a different loop.
        faithful = await router.prompt_fidelity(
            "<prompt row-3>", job=job, prompt_index=3
        )
        refused = await router.prompt_fidelity(
            "<prompt row-4>", job=job, prompt_index=3
        )
        return faithful, refused

    faithful, refused = asyncio.run(run())
    assert faithful.ok
    assert not refused.ok
    assert refused.reason == "prompt_not_faithful"
