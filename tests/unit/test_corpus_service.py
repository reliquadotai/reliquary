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


def _manifest():
    return {
        "schema": "reliquary/corpus-job/v1",
        "job_id": "swe-v1",
        "checkpoint_repo": "org/Frozen",
        "checkpoint_revision": "abc123",
        "checkpoint_sha256": CHECKPOINT,
        "prompt_source": "openmathinstruct",
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

    name = "openmathinstruct"

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


class _CountingStore:
    """The real store bound to the fake bucket, counting ledger writes and
    able to lose a compare-and-swap race on demand."""

    def __init__(self, client_kwargs):
        self._kwargs = client_kwargs
        self.ledger_write_attempts = 0
        self._conflicts_left = 0
        self._competing_snapshot = None

    async def read_job(self, job_id):
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
        return {"openmathinstruct": _Spec(environment or self.environment)}


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
            store=seeded_job.store,
            tokenizer=_Tokenizer(),
            renderer=seeded_job.renderer,
            verify_signature=lambda request: request.signature != "bad",
        )
    )
    return TestClient(app)


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
    termination="eos",
    prompt_index=0,
    cursor=0,
    signature="ok",
):
    """Build through the wire model so the schema is exercised, not bypassed."""
    request = CorpusSubmissionRequest(
        job_id="swe-v1",
        miner_hotkey="5Hot",
        cursor=cursor,
        prompt_index=prompt_index,
        checkpoint_sha256=CHECKPOINT,
        completions=[
            {
                "tokens": tokens,
                "text": _text_for(tokens) if text is None else text,
                "termination": termination,
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
    """A completion labelled "eos" whose last token is NOT the manifest's
    eos_token_id must be refused. If the endpoint forwarded the declared
    label, this passes and the check becomes decoration."""
    body = _submit(client, tokens=[7] * 16 + [99], termination="eos").json()
    assert body["accepted"] is False
    assert "termination" in body["reason"]


def test_a_cap_label_on_an_eos_ending_completion_is_also_derived_away(client):
    """The other direction, so the test above cannot be passed by hardcoding
    "cap": a completion that really did end on EOS is accepted however it is
    labelled, because the label never reaches the check."""
    body = _submit(client, tokens=[7] * 16 + [EOS], termination="cap").json()
    assert body["accepted"] is True


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


def test_a_submission_for_an_unknown_job_is_refused_without_touching_the_ledgers(
    client, seeded_job
):
    before = seeded_job.ledger_writes()
    request = CorpusSubmissionRequest(
        job_id="ghost",
        miner_hotkey="5Hot",
        cursor=0,
        prompt_index=0,
        checkpoint_sha256=CHECKPOINT,
        completions=[{"tokens": [1], "text": "1", "termination": "cap"}],
        signature="ok",
    )
    body = client.post("/corpus/submit", json=request.model_dump()).json()
    assert body["accepted"] is False
    assert body["reason"] == "job_unknown"
    assert seeded_job.ledger_writes() == before


def test_a_job_id_that_could_never_name_a_job_is_refused_not_raised(
    client, seeded_job
):
    """The id becomes a bucket key, so the store refuses it outright; that
    must read as "no such job", not as a 500 on a hostile request."""
    before = seeded_job.ledger_writes()
    request = CorpusSubmissionRequest(
        job_id="../../etc/passwd",
        miner_hotkey="5Hot",
        cursor=0,
        prompt_index=0,
        checkpoint_sha256=CHECKPOINT,
        completions=[{"tokens": [1], "text": "1", "termination": "cap"}],
        signature="ok",
    )
    response = client.post("/corpus/submit", json=request.model_dump())
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
    to check them against, so this layer is the only one that can."""
    from reliquary.validator.corpus_service import LedgerSnapshotError

    seeded_job.seed_ledgers({"slots": {"99999999": 1}, "cursors": {}, "seen": []})
    with pytest.raises(LedgerSnapshotError) as caught:
        _submit(client, tokens=[7] * 16 + [EOS])
    assert "swe-v1" in str(caught.value)


def test_a_ledger_snapshot_with_a_field_this_binary_cannot_read_is_refused(
    client, seeded_job
):
    # Slots we do not fully understand are slots we could sell twice.
    from reliquary.validator.corpus_service import LedgerSnapshotError

    seeded_job.seed_ledgers({"slots": {}, "cursors": {}, "seen": [], "reserved": {}})
    with pytest.raises(LedgerSnapshotError):
        _submit(client, tokens=[7] * 16 + [EOS])


def test_a_ledger_snapshot_carrying_a_digest_that_is_not_one_is_refused(
    client, seeded_job
):
    from reliquary.validator.corpus_service import LedgerSnapshotError

    seeded_job.seed_ledgers({"slots": {}, "cursors": {}, "seen": [17]})
    with pytest.raises(LedgerSnapshotError):
        _submit(client, tokens=[7] * 16 + [EOS])


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
    assert "openmathinstruct" in str(caught.value)


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
        store=seeded_job.store,
        tokenizer=_Tokenizer(),
        renderer=seeded_job.renderer,
        verify_signature=lambda request: True,
        prompt_job_for=lambda job: prompt_job_for_spec(job, environments=environments),
    )
    job = seeded_job.job

    assert router.prompt_fidelity("<prompt row-3>", job=job, prompt_index=3).ok
    refused = router.prompt_fidelity("<prompt row-4>", job=job, prompt_index=3)
    assert not refused.ok
    assert refused.reason == "prompt_not_faithful"
