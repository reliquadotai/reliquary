"""The submit route over ledger v2: the seen set is pending digests plus
sealed segments, and every verdict, slot and cursor must be exactly what the
v1 ledger (one plain set, rewritten whole) would have produced."""

from __future__ import annotations

import asyncio
import json
import random

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reliquary.corpus.admission import admit
from reliquary.corpus.checks import completion_digest
from reliquary.corpus.job import parse_job
from reliquary.corpus.slots import SlotLedger
from reliquary.corpus.walk import CursorLedger, walk_index
from reliquary.infrastructure import corpus_job_store as job_store
from reliquary.infrastructure.corpus_job_store import BucketJobStore
from reliquary.protocol.corpus_submission import CorpusSubmissionRequest
from tests.unit.test_corpus_ledger_v2 import seen_union
from tests.unit.test_corpus_route_records import _Records
from tests.unit.test_corpus_service import (  # noqa: F401 - fixtures
    CHECKPOINT,
    EOS,
    _faithful_prompt,
    _manifest,
    _r2_client,
    _Renderer,
    _text_for,
    _Tokenizer,
    fake_r2,
    seeded_job,
)

JOB = "seen-v1"


def _ledger_key(job_id=JOB):
    return f"reliquary/corpus/jobs/{job_id}/ledgers.json"


def _segment_keys(objects, job_id=JOB):
    prefix = f"reliquary/corpus/jobs/{job_id}/seen/"
    return sorted(k for k in objects if k.startswith(prefix))


def _write_job(job_id=JOB, **overrides):
    raw = _manifest()
    raw["job_id"] = job_id
    for key, value in overrides.items():
        if key == "n":
            raw["sampling"] = {**raw["sampling"], "n": value}
        else:
            raw[key] = value
    asyncio.run(job_store.write_job(raw, None))
    return parse_job(raw)


class _Store(BucketJobStore):
    """The real bound store, logging each bucket-facing call and able to fail
    or be raced at named points."""

    def __init__(self, *, yielding=False):
        super().__init__()
        self.log: list[str] = []
        self.faults: dict[str, list] = {}
        self.yielding = yielding
        self.conflicts = 0

    async def _hook(self, name):
        if self.yielding:
            await asyncio.sleep(0.001)
        queue = self.faults.get(name)
        if queue:
            fault = queue.pop(0)
            if isinstance(fault, BaseException):
                raise fault
            fault()

    async def read_ledgers(self, job_id):
        await self._hook("ledger_get")
        return await super().read_ledgers(job_id)

    async def write_ledgers(self, job_id, snapshot, etag):
        self.log.append("ledger_put")
        await self._hook("before_ledger_put")
        try:
            new = await super().write_ledgers(job_id, snapshot, etag)
        except job_store.CorpusStoreConflict:
            self.conflicts += 1
            raise
        await self._hook("after_ledger_put")
        return new

    async def write_seen_segment(self, job_id, digests):
        self.log.append("segment_put")
        await self._hook("segment_put")
        return await super().write_seen_segment(job_id, digests)

    async def read_seen_segment(self, job_id, segment_id):
        self.log.append("segment_get")
        await self._hook("segment_get")
        return await super().read_seen_segment(job_id, segment_id)


def _app(seeded_job, store, *, job_id=JOB, records=None, threshold=1024, segment_max=4096,
         attempts=4, seen_index=None):
    from reliquary.validator.corpus_service import build_corpus_router

    app = FastAPI()
    app.include_router(build_corpus_router(
        job_id=job_id, store=store, tokenizer=_Tokenizer(), renderer=_Renderer(),
        verify_signature=lambda request: True, prompt_job_for=seeded_job.prompt_job_for,
        records=records, seal_threshold=threshold, segment_max=segment_max,
        max_write_attempts=attempts, seen_index=seen_index,
    ))
    return app


def _tokens(fill):
    return [fill] * 16 + [EOS]


def _body(fill, *, prompt_index=0, hotkey="5Hot", cursor=0, job_id=JOB, copies=1):
    tokens = _tokens(fill)
    return CorpusSubmissionRequest(
        job_id=job_id, miner_hotkey=hotkey, cursor=cursor, prompt_index=prompt_index,
        checkpoint_sha256=CHECKPOINT, rendered_prompt=_faithful_prompt(prompt_index),
        completions=[{"tokens": tokens, "text": _text_for(tokens)}] * copies,
        signature="ok",
    ).model_dump()


def _digest(fill, prompt_index=0):
    return completion_digest(prompt_index, _tokens(fill))


def _post(client, body):
    response = client.post("/corpus/submit", json=body)
    return response.status_code, response.json()


def _ledger(r2, job_id=JOB):
    return json.loads(r2.objects[_ledger_key(job_id)][0])


def test_accept_grows_pending_and_writes_no_seen_list(seeded_job, _r2_client):
    _write_job()
    store = _Store()
    client = TestClient(_app(seeded_job, store))

    assert _post(client, _body(1))[1]["accepted"] is True
    assert _post(client, _body(2, prompt_index=1))[1]["accepted"] is True

    ledger = _ledger(_r2_client)
    assert "seen" not in ledger
    assert ledger["schema"] == "reliquary/corpus-ledgers/v2"
    assert ledger["seen_pending"] == sorted([_digest(1), _digest(2, 1)])
    assert ledger["seen_segments"] == []
    assert store.log == ["ledger_put", "ledger_put"]


def test_duplicate_in_pending_refused(seeded_job, _r2_client):
    _write_job()
    store = _Store()
    client = TestClient(_app(seeded_job, store))
    assert _post(client, _body(1))[1]["accepted"] is True

    for body in (_body(1), _body(1, hotkey="5Other")):
        status, verdict = _post(client, body)
        assert (status, verdict["accepted"], verdict["reason"]) == (200, False, "hash_duplicate")
    assert store.log == ["ledger_put"]


def test_duplicate_in_sealed_segment_refused(seeded_job, _r2_client):
    _write_job()
    store = _Store()
    client = TestClient(_app(seeded_job, store, threshold=2))
    assert _post(client, _body(1))[1]["accepted"] is True
    assert _post(client, _body(2))[1]["accepted"] is True

    ledger = _ledger(_r2_client)
    assert ledger["seen_pending"] == []
    assert [ref["count"] for ref in ledger["seen_segments"]] == [2]
    for fill in (1, 2):
        verdict = _post(client, _body(fill, hotkey="5Copy"))[1]
        assert verdict["reason"] == "hash_duplicate"


def test_duplicate_refused_after_restart(seeded_job, _r2_client):
    _write_job()
    client = TestClient(_app(seeded_job, _Store(), threshold=2))
    for fill in (1, 2, 3):
        assert _post(client, _body(fill))[1]["accepted"] is True

    # A new process: empty memory, empty index.
    store = _Store()
    restarted = TestClient(_app(seeded_job, store, threshold=2))
    for fill in (1, 2, 3):
        assert _post(restarted, _body(fill))[1]["reason"] == "hash_duplicate"
    assert store.log == ["segment_get"]
    assert _post(restarted, _body(4))[1]["accepted"] is True


def test_intra_submission_duplicate_refused(seeded_job, _r2_client):
    _write_job("pair-v1", n=2)
    store = _Store()
    client = TestClient(_app(seeded_job, store, job_id="pair-v1"))

    verdict = _post(client, _body(1, job_id="pair-v1", copies=2))[1]
    assert (verdict["accepted"], verdict["reason"]) == (False, "hash_duplicate")
    assert store.log == []


def test_seal_at_threshold_preserves_union(seeded_job, _r2_client):
    _write_job()
    client = TestClient(_app(seeded_job, _Store(), threshold=3, segment_max=2))
    accepted = set()
    for fill in range(1, 8):
        assert _post(client, _body(fill, prompt_index=fill % 4))[1]["accepted"] is True
        accepted.add(_digest(fill, fill % 4))
        assert seen_union(_r2_client.objects, JOB) == accepted
        ledger = _ledger(_r2_client)
        assert len(ledger["seen_pending"]) < 3
        assert all(ref["count"] <= 2 for ref in ledger["seen_segments"])
    assert _ledger(_r2_client)["slots"] == {"0": 1, "1": 2, "2": 2, "3": 2}


def test_refusal_writes_nothing(seeded_job, _r2_client):
    _write_job()
    store = _Store()
    client = TestClient(_app(seeded_job, store, threshold=2))
    assert _post(client, _body(1))[1]["accepted"] is True
    before = dict(_r2_client.objects)
    store.log.clear()

    # One more pending digest would seal: refusals must not.
    assert _post(client, _body(1, hotkey="5Copy"))[1]["reason"] == "hash_duplicate"
    bad = _body(9)
    bad["completions"][0]["tokens"][-1] = 99
    bad["completions"][0]["text"] = _text_for(bad["completions"][0]["tokens"])
    assert _post(client, bad)[1]["accepted"] is False
    assert store.log == []
    assert _r2_client.objects == before


def test_prompt_full_moves_cursor_not_seen(seeded_job, _r2_client):
    job = _write_job("walk-v1", prompt_order="miner_walk", slots_per_prompt=1)
    prompt = walk_index("walk-v1", "5Hot", 0, job.prompt_count)
    asyncio.run(job_store.write_ledgers("walk-v1", {
        "schema": "reliquary/corpus-ledgers/v2", "slots": {str(prompt): 1},
        "cursors": {}, "seen_pending": [], "seen_segments": [],
    }, None))
    client = TestClient(_app(seeded_job, _Store(), job_id="walk-v1", threshold=1))

    verdict = _post(client, _body(1, prompt_index=prompt, job_id="walk-v1"))[1]
    assert verdict["reason"] == "prompt_full"
    ledger = _ledger(_r2_client, "walk-v1")
    assert ledger["cursors"] == {"5Hot": 1}
    assert ledger["seen_pending"] == [] and ledger["seen_segments"] == []
    assert _segment_keys(_r2_client.objects, "walk-v1") == []


def _compete(r2, mutate, job_id=JOB):
    """Another validator's write landing first: new bytes, new ETag."""
    counter = [0]

    def land():
        counter[0] += 1
        ledger = _ledger(r2, job_id)
        mutate(ledger)
        r2.objects[_ledger_key(job_id)] = (job_store._encode(ledger), f'"rival{counter[0]}"')

    return land


def test_conflict_after_seal_orphan_unreferenced_retry_exact(seeded_job, _r2_client):
    _write_job()
    store = _Store()
    client = TestClient(_app(seeded_job, store, threshold=2, segment_max=2))
    assert _post(client, _body(1))[1]["accepted"] is True

    # A rival admits the same work at prompt 1 between our seal and our write.
    def rival_accepts(ledger):
        ledger["slots"]["1"] = 1
        ledger["seen_pending"] = sorted([*ledger["seen_pending"], _digest(2, 1)])

    store.faults["before_ledger_put"] = [_compete(_r2_client, rival_accepts)]
    verdict = _post(client, _body(2, prompt_index=1))[1]
    assert verdict["reason"] == "hash_duplicate"

    # The segment we sealed exists but nothing names it, so it counts for nothing.
    ledger = _ledger(_r2_client)
    assert ledger["seen_segments"] == []
    orphans = _segment_keys(_r2_client.objects)
    assert len(orphans) == 1
    assert seen_union(_r2_client.objects, JOB) == {_digest(1), _digest(2, 1)}

    # The next seal covers the rival's digest from the ledger, not the orphan.
    assert _post(client, _body(3))[1]["accepted"] is True
    ledger = _ledger(_r2_client)
    assert ledger["seen_pending"] == []
    assert orphans[0] not in {
        f"reliquary/corpus/jobs/{JOB}/seen/{ref['id']}.json" for ref in ledger["seen_segments"]
    }
    assert seen_union(_r2_client.objects, JOB) == {_digest(1), _digest(2, 1), _digest(3)}


def test_transport_error_after_seal_before_ledger_write_then_resend_accepted_once(
    seeded_job, _r2_client
):
    _write_job()
    store = _Store()
    client = TestClient(_app(seeded_job, store, threshold=2))
    assert _post(client, _body(1))[1]["accepted"] is True
    before = _ledger(_r2_client)

    store.faults["before_ledger_put"] = [OSError("reset")]
    status, body = _post(client, _body(2))
    assert (status, body) == (503, {"detail": "corpus_store_unavailable"})
    assert _ledger(_r2_client) == before
    assert len(_segment_keys(_r2_client.objects)) == 1

    assert _post(client, _body(2))[1]["accepted"] is True
    # The resend sealed the same content, so it reused the orphan by name.
    assert len(_segment_keys(_r2_client.objects)) == 1
    assert _ledger(_r2_client)["seen_segments"][0]["count"] == 2
    assert _post(client, _body(2))[1]["reason"] == "hash_duplicate"
    assert seen_union(_r2_client.objects, JOB) == {_digest(1), _digest(2)}
    assert _ledger(_r2_client)["slots"] == {"0": 2}


def test_ledger_write_landed_response_lost_resend_is_duplicate(seeded_job, _r2_client):
    _write_job()
    store = _Store()
    client = TestClient(_app(seeded_job, store, threshold=2))

    store.faults["after_ledger_put"] = [OSError("timeout")]
    assert _post(client, _body(1))[0] == 503
    # The accepted loss v1 had too: the slot is spent and the resend refused.
    assert _post(client, _body(1))[1]["reason"] == "hash_duplicate"
    assert _ledger(_r2_client)["slots"] == {"0": 1}


def test_record_write_fails_after_ledger_write_digest_still_seen(seeded_job, _r2_client):
    _write_job()
    records = _Records(fail=99)
    client = TestClient(_app(seeded_job, _Store(), records=records))

    assert _post(client, _body(1))[1]["accepted"] is True
    assert records.written == {}
    assert _post(client, _body(1))[1]["reason"] == "hash_duplicate"
    assert _digest(1) in seen_union(_r2_client.objects, JOB)


async def _gather_posts(pairs):
    clients = {}
    try:
        for app, _ in pairs:
            if id(app) not in clients:
                clients[id(app)] = httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://v"
                )
        responses = await asyncio.gather(*(
            clients[id(app)].post("/corpus/submit", json=body) for app, body in pairs
        ))
    finally:
        for http in clients.values():
            await http.aclose()
    return [r.json() for r in responses]


def test_concurrent_identical_submissions_exactly_one_accept(seeded_job, _r2_client):
    _write_job()
    app = _app(seeded_job, _Store(yielding=True), threshold=2)
    bodies = [_body(1)] * 4 + [_body(1, hotkey="5Copy")] * 2 + [_body(2)]

    results = asyncio.run(_gather_posts([(app, body) for body in bodies]))

    fill1 = [r["accepted"] for r in results[:6]]
    assert fill1.count(True) == 1
    assert {r["reason"] for r, ok in zip(results[:6], fill1) if not ok} == {"hash_duplicate"}
    assert results[6]["accepted"] is True
    assert seen_union(_r2_client.objects, JOB) == {_digest(1), _digest(2)}
    assert _ledger(_r2_client)["slots"] == {"0": 2}


def test_two_routers_same_bucket_exactly_one_accept(seeded_job, _r2_client):
    _write_job()
    stores = [_Store(yielding=True), _Store(yielding=True)]
    # Two writers genuinely race, so the retry budget is not what is tested.
    apps = [_app(seeded_job, store, threshold=2, attempts=50) for store in stores]
    pairs = [(apps[i % 2], _body(1, hotkey=f"5Hot{i}")) for i in range(6)]
    pairs += [(apps[i % 2], _body(10 + i, prompt_index=i)) for i in range(4)]

    results = asyncio.run(_gather_posts(pairs))

    assert [r["accepted"] for r in results[:6]].count(True) == 1
    assert all(r["accepted"] for r in results[6:])
    assert seen_union(_r2_client.objects, JOB) == {_digest(1)} | {
        _digest(10 + i, i) for i in range(4)
    }
    assert sum(_ledger(_r2_client)["slots"].values()) == 5


def test_two_routers_one_seals_other_learns_segment(seeded_job, _r2_client):
    _write_job()
    store_a, store_b = _Store(), _Store()
    a = TestClient(_app(seeded_job, store_a, threshold=2))
    b = TestClient(_app(seeded_job, store_b, threshold=2))

    assert _post(b, _body(1))[1]["accepted"] is True  # b now remembers the ledger
    assert _post(a, _body(2))[1]["accepted"] is True  # a seals [1, 2]
    assert _post(a, _body(3))[1]["accepted"] is True  # pending [3]
    assert len(_ledger(_r2_client)["seen_segments"]) == 1

    # b's memory is stale, so it admits 3 and seals [1, 3]; the compare-and-swap
    # sends it back to the bucket, where it loads a's segment and refuses.
    assert _post(b, _body(3, hotkey="5Copy"))[1]["reason"] == "hash_duplicate"
    assert "segment_get" in store_b.log
    assert _post(b, _body(4))[1]["accepted"] is True
    assert seen_union(_r2_client.objects, JOB) == {_digest(i) for i in (1, 2, 3, 4)}


def _sealed_bucket(seeded_job, r2):
    _write_job()
    client = TestClient(_app(seeded_job, _Store(), threshold=2))
    for fill in (1, 2):
        assert _post(client, _body(fill))[1]["accepted"] is True
    (key,) = _segment_keys(r2.objects)
    return key


def test_missing_referenced_segment_500(seeded_job, _r2_client):
    key = _sealed_bucket(seeded_job, _r2_client)
    del _r2_client.objects[key]
    before = _ledger(_r2_client)
    store = _Store()
    client = TestClient(_app(seeded_job, store, threshold=2))

    # Never read as empty: that would admit the duplicate below.
    assert _post(client, _body(1)) == (500, {"detail": "corpus_ledger_corrupt"})
    assert _post(client, _body(5)) == (500, {"detail": "corpus_ledger_corrupt"})
    assert "ledger_put" not in store.log
    assert _ledger(_r2_client) == before


def test_altered_referenced_segment_500(seeded_job, _r2_client):
    key = _sealed_bucket(seeded_job, _r2_client)
    body, etag = _r2_client.objects[key]
    _r2_client.objects[key] = (body.replace(b"]", b',"f"]'), etag)
    client = TestClient(_app(seeded_job, _Store(), threshold=2))
    assert _post(client, _body(5)) == (500, {"detail": "corpus_ledger_corrupt"})


def test_segment_get_transport_error_503(seeded_job, _r2_client):
    _sealed_bucket(seeded_job, _r2_client)
    store = _Store()
    store.faults["segment_get"] = [OSError("reset")]
    client = TestClient(_app(seeded_job, store, threshold=2))

    assert _post(client, _body(1)) == (503, {"detail": "corpus_store_unavailable"})
    assert _post(client, _body(1))[1]["reason"] == "hash_duplicate"


def test_cursor_route_does_not_load_segments(seeded_job, _r2_client):
    _sealed_bucket(seeded_job, _r2_client)
    store = _Store()
    client = TestClient(_app(seeded_job, store, threshold=2))

    response = client.get("/corpus/cursor/5Hot")
    assert response.status_code == 200
    assert response.json() == {"hotkey": "5Hot", "cursor": 0}
    assert "segment_get" not in store.log


# --------------------------------------------------------------------------
# The key test: v2 against a plain-set v1 reference.
# --------------------------------------------------------------------------

HOTKEYS = ("5HotA", "5HotB", "5HotC")


class _Reference:
    """v1's semantics: slots, cursors and one set, grown only on accept."""

    def __init__(self, job):
        self.job = job
        self.slots = SlotLedger(job.prompt_count, job.slots_per_prompt)
        self.cursors = CursorLedger()
        self.seen: set[str] = set()

    def decide(self, request):
        """The verdict, whether it would write, and a commit for the state."""
        slots = SlotLedger.from_snapshot(
            self.job.prompt_count, self.job.slots_per_prompt, self.slots.snapshot()
        )
        cursors = CursorLedger.from_snapshot(self.cursors.snapshot())
        digests = [completion_digest(request["prompt_index"], c["tokens"])
                   for c in request["completions"]]
        verdict = admit(
            self.job, hotkey=request["miner_hotkey"], cursor=request["cursor"],
            prompt_index=request["prompt_index"],
            checkpoint_sha256=request["checkpoint_sha256"],
            token_counts=[len(c["tokens"]) for c in request["completions"]],
            last_token_ids=[c["tokens"][-1] for c in request["completions"]],
            digests=digests, slots=slots, cursors=cursors, seen=frozenset(self.seen),
        )
        writes = verdict.accepted or (slots.snapshot(), cursors.snapshot()) != (
            self.slots.snapshot(), self.cursors.snapshot()
        )

        def commit():
            self.slots, self.cursors = slots, cursors
            if verdict.accepted:
                self.seen.update(digests)

        return verdict, writes, commit


def _rival_accepts(r2, reference, job, fill):
    """Another validator admits fresh work of its own, landing in the bucket
    and in the reference alike; returns the hook that lands it."""

    def land():
        free = [p for p in range(job.prompt_count) if not reference.slots.is_full(p)]
        prompt = free[fill % len(free)]
        digest = completion_digest(prompt, _tokens(fill))
        reference.slots.consume(prompt)
        reference.seen.add(digest)
        if job.prompt_order == "miner_walk":
            reference.cursors.advance("5Rival")

        def mutate(ledger):
            ledger["slots"][str(prompt)] = ledger["slots"].get(str(prompt), 0) + 1
            ledger["seen_pending"] = sorted([*ledger["seen_pending"], digest])
            if job.prompt_order == "miner_walk":
                ledger["cursors"]["5Rival"] = ledger["cursors"].get("5Rival", 0) + 1

        _compete(r2, mutate)()
        return digest

    return land


class _Unremembering(_Store):
    """Reads the bucket every time. The store's ledger memory assumes one
    writer: with two, a refusal judged against a stale memory writes nothing,
    so no compare-and-swap corrects it. That is v1 behaviour, unchanged here,
    so the two-router interleaving below takes it out of the comparison."""

    async def read_ledgers(self, job_id):
        self._ledgers.pop(job_id, None)
        return await super().read_ledgers(job_id)


def _assert_matches(r2, reference, step):
    if _ledger_key() not in r2.objects:
        assert reference.slots.filled == 0 and not reference.seen, step
        return
    ledger = _ledger(r2)
    assert ledger["slots"] == {str(k): v for k, v in reference.slots.snapshot().items()}, step
    assert ledger["cursors"] == reference.cursors.snapshot(), step
    assert seen_union(r2.objects, JOB) == reference.seen, step


@pytest.mark.parametrize("order", ["free", "miner_walk"])
@pytest.mark.parametrize("seed", range(4))
def test_v2_verdicts_match_v1_reference_model(seeded_job, _r2_client, order, seed):
    rng = random.Random(seed * 7919 + (order == "free"))
    # Large enough that 160 steps never complete the job.
    job = _write_job(prompt_order=order, prompt_count=40, slots_per_prompt=3)
    threshold, segment_max = 3, 2
    reference = _Reference(job)

    def router():
        store = _Unremembering()
        return store, TestClient(
            _app(seeded_job, store, threshold=threshold, segment_max=segment_max)
        )

    # Two routers on one bucket, each with its own memory and index.
    routers = [router(), router()]
    sent: list[dict] = []
    accepted_digests: list[str] = []
    next_fill = 1
    rivals = 0

    for step in range(160):
        if rng.random() < 0.05:
            routers[rng.randrange(2)] = router()  # a restart
        store, client = routers[rng.randrange(2)]
        hotkey = rng.choice(HOTKEYS)
        kind = rng.random()
        if kind < 0.55 or not sent:
            fill = next_fill = next_fill + 1
            if order == "miner_walk":
                cursor = reference.cursors.expected(hotkey)
                if rng.random() < 0.1:
                    cursor += 1
                prompt = walk_index(JOB, hotkey, cursor, job.prompt_count)
            else:
                cursor, prompt = 0, rng.randrange(job.prompt_count)
            body = _body(fill, prompt_index=prompt, hotkey=hotkey, cursor=cursor)
        elif kind < 0.8:
            body = rng.choice(sent)  # a resend, self or not
        else:
            # A cross-miner copy of earlier work, at the copier's own cursor.
            original = rng.choice(sent)
            body = {**original, "miner_hotkey": hotkey,
                    "cursor": reference.cursors.expected(hotkey)}
        sent.append(body)

        verdict, writes, commit = reference.decide(body)
        pending = len(_ledger(_r2_client)["seen_pending"]) if _ledger_key() in _r2_client.objects else 0
        seals = verdict.accepted and pending + 1 >= threshold
        fault = None
        roll = rng.random()
        if writes and roll < 0.08:
            fault = "crash_before_write"
            store.faults["before_ledger_put"] = [OSError("crash before the write")]
        elif writes and seals and roll < 0.14:
            fault = "crash_during_seal"
            store.faults["segment_put"] = [OSError("crash during the seal")]
        elif writes and 0.14 <= roll < 0.2:
            fault = "response_lost"
            store.faults["after_ledger_put"] = [OSError("response lost")]
        elif writes and 0.2 <= roll < 0.27:
            fault = "conflict"
            store.faults["before_ledger_put"] = [_compete(_r2_client, lambda ledger: None)]
        elif writes and 0.27 <= roll < 0.37 and _ledger_key() in _r2_client.objects:
            # A rival admits a different digest between our seal and our write:
            # our retry admits against what it left.
            fault = "rival"
            next_fill += 1
            rival = _rival_accepts(_r2_client, reference, job, 10_000 + next_fill)
            store.faults["before_ledger_put"] = [lambda: accepted_digests.append(rival())]
        landed = fault not in ("crash_before_write", "crash_during_seal")

        status, got = _post(client, body)
        store.faults.clear()
        if fault == "rival":
            rivals += 1
            # The rival landed first, so the verdict is the one against its ledger.
            verdict, writes, commit = reference.decide(body)
        if fault in ("crash_before_write", "crash_during_seal", "response_lost"):
            assert status == 503, (step, fault, got)
        else:
            assert status == 200, (step, got)
            assert (got["accepted"], got["reason"], got["slots_remaining"]) == (
                verdict.accepted, verdict.reason, verdict.slots_remaining
            ), (step, fault)
        if landed:
            commit()
            if verdict.accepted:
                accepted_digests.extend(
                    completion_digest(body["prompt_index"], c["tokens"])
                    for c in body["completions"]
                )
        _assert_matches(_r2_client, reference, (step, fault))

    assert not reference.slots.is_complete
    assert len(accepted_digests) == len(set(accepted_digests)) == len(reference.seen)
    assert rivals and _ledger(_r2_client)["seen_segments"], "the faults never fired"


class _OneFailsOthersLinger(_Store):
    """The first segment call fails at once; the others would finish later.
    Records which of those ever completed, and which were cancelled."""

    def __init__(self):
        super().__init__()
        self.completed: list[str] = []
        self.cancelled = 0
        self._calls = 0

    async def _slow(self, what):
        self._calls += 1
        if self._calls == 1:
            raise OSError("reset")
        try:
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        self.completed.append(what)

    async def write_seen_segment(self, job_id, digests):
        await self._slow("put")
        return await BucketJobStore.write_seen_segment(self, job_id, digests)

    async def read_seen_segment(self, job_id, segment_id):
        await self._slow("get")
        return await BucketJobStore.read_seen_segment(self, job_id, segment_id)


def test_a_failed_seal_cancels_its_sibling_writes(seeded_job, _r2_client):
    _write_job()
    store = _OneFailsOthersLinger()
    app = _app(seeded_job, store, threshold=6, segment_max=1)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://v"
        ) as http:
            for fill in range(1, 6):
                await http.post("/corpus/submit", json=_body(fill))
            failed = await http.post("/corpus/submit", json=_body(6))
            await asyncio.sleep(0.1)  # long enough for a surviving write to land
            return failed

    failed = asyncio.run(scenario())
    assert failed.status_code == 503
    assert store.completed == []
    assert store.cancelled == 5
    assert _segment_keys(_r2_client.objects) == []


def test_a_failed_segment_load_cancels_its_sibling_reads(seeded_job, _r2_client):
    from reliquary.validator.corpus_service import SeenIndex

    _write_job()
    client = TestClient(_app(seeded_job, _Store(), threshold=1, segment_max=1))
    for fill in range(1, 4):
        assert _post(client, _body(fill))[1]["accepted"] is True
    refs = _ledger(_r2_client)["seen_segments"]
    assert len(refs) == 3

    from reliquary.validator.corpus_service import SegmentRef

    store = _OneFailsOthersLinger()
    index = SeenIndex(store, JOB)

    async def scenario():
        with pytest.raises(OSError):
            await index.ensure([SegmentRef(r["id"], r["count"]) for r in refs])
        await asyncio.sleep(0.1)

    asyncio.run(scenario())
    assert store.completed == [] and store.cancelled == 2
    assert len(index) == 0
