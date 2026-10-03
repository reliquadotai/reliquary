"""A trajectory submission through the real route, with the real intake over a fake renderer."""

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reliquary.corpus.job import parse_job
from reliquary.environment.agentic_swe import SweSource
from reliquary.infrastructure import corpus_job_store as job_store
from reliquary.protocol.corpus_submission import CorpusSubmissionRequest
from reliquary.protocol.signatures import corpus_submission_id
from reliquary.validator.agentic_intake import EpisodeIntake
from tests.unit.test_corpus_job_episode import _manifest
from tests.unit.test_corpus_route_records import _Records
from tests.unit.test_corpus_service import (  # noqa: F401  (fixtures)
    _CountingStore, _r2_client, _Tokenizer, fake_r2, seeded_job,
)
from tests.unit.test_trajectory_parse import CALL, PROMPT, TERM, TEXT, R, build

PROOF = "A" * 200   # the wire bounds a trajectory's proof chars by its tokens (11 per token + 344)
TOKENIZER = _Tokenizer()


@pytest.fixture
def episode_store(fake_r2):
    asyncio.run(job_store.write_job(_manifest(prompt_count=3), None, **fake_r2))
    return _CountingStore(fake_r2)


def _intake():
    job = parse_job(_manifest(prompt_count=3))
    source = SweSource([("repo__a.1", "fix it"), ("repo__b.2", "x"), ("repo__c.3", "y")])
    return EpisodeIntake(job=job, source=source, renderer=R, tokenizer=TOKENIZER,
                         vocab_size=None, chunk_tokens=32)


def _client(store, records, accepted, intake=True):
    from reliquary.validator.corpus_service import build_corpus_router

    app = FastAPI()
    app.include_router(build_corpus_router(
        job_id="swe-agentic-v1", store=store, tokenizer=TOKENIZER, renderer=None,
        verify_signature=lambda r: True, records=records, on_accepted=accepted.append,
        proof_chunk_tokens=32, episode_intake=_intake() if intake else None))
    return TestClient(app)


TURNS = [([TEXT] * 8 + [CALL, TERM], ["a.py"]), ([TEXT] * 8 + [CALL, TERM], ["ok"]),
         ([TEXT] * 9 + [TERM], None)]


def _request(tokens=None, spans=None, rendered=None, **trajectory):
    if tokens is None:
        tokens, spans = build(TURNS)
    body = {"tokens": tokens, "turns": [{"start": s, "end": e, "proofs": [PROOF]} for s, e in spans],
            "final_diff": "diff --git a/x b/x\n", "stop": "agent_completed"}
    body.update(trajectory)
    return CorpusSubmissionRequest(
        job_id="swe-agentic-v1", miner_hotkey="5Hot", cursor=0, prompt_index=0,
        checkpoint_sha256="c" * 64,
        rendered_prompt=rendered if rendered is not None else TOKENIZER.decode(PROMPT),
        trajectory=body, signature="ok")


def _post(client, request):
    return client.post("/corpus/submit", json=request.model_dump()).json()


def test_an_honest_trajectory_is_accepted_and_recorded_as_v2(episode_store):
    records, accepted = _Records(), []
    request = _request()
    body = _post(_client(episode_store, records, accepted), request)
    assert body["accepted"] is True, body
    sid = corpus_submission_id(request)
    record = records.written[sid]
    assert record["schema"] == "reliquary/corpus-submission-record/v2"
    assert record["token_count"] == 30                             # assistant spans only
    assert record["completions"][0]["prompt_tokens"] == PROMPT
    assert record["completions"][0]["stop"] == "agent_completed" and accepted == [sid]


def test_a_forged_segment_is_refused(episode_store):
    tokens, spans = build(TURNS)
    end = spans[0][1]
    tokens = tokens[:end] + [10] + tokens[end:]
    spans = [spans[0]] + [(s + 1, e + 1) for s, e in spans[1:]]
    body = _post(_client(episode_store, _Records(), []), _request(tokens, spans))
    assert (body["accepted"], body["reason"]) == (False, "bad_observation")


def test_an_unfaithful_prompt_is_refused(episode_store):
    body = _post(_client(episode_store, _Records(), []), _request(rendered="something else"))
    assert body["reason"] == "prompt_not_faithful"


def test_a_wrong_proof_count_is_refused(episode_store):
    tokens, spans = build(TURNS)
    request = _request(tokens, spans)
    request.trajectory.turns[2].proofs.clear()       # the wire allows it; the shape check does not
    body = _post(_client(episode_store, _Records(), []), request)
    assert body["reason"] == "bad_proof_shape"


def test_the_same_trajectory_twice_is_a_duplicate(episode_store):
    client = _client(episode_store, _Records(), [])
    assert _post(client, _request())["accepted"] is True
    assert _post(client, _request(final_diff="other"))["reason"] == "hash_duplicate"


def test_an_episode_job_without_an_intake_is_a_server_error(episode_store):
    response = _client(episode_store, _Records(), [], intake=False).post(
        "/corpus/submit", json=_request().model_dump())
    assert response.status_code == 500


def test_a_trajectory_on_a_single_turn_job_is_malformed(seeded_job):
    from reliquary.validator.corpus_service import build_corpus_router

    app = FastAPI()
    app.include_router(build_corpus_router(
        job_id="swe-v1", store=seeded_job.store, tokenizer=TOKENIZER, renderer=seeded_job.renderer,
        verify_signature=lambda r: True, prompt_job_for=seeded_job.prompt_job_for))
    request = _request().model_copy(update={"job_id": "swe-v1"})
    body = TestClient(app).post("/corpus/submit", json=request.model_dump()).json()
    assert body["reason"] == "malformed_submission"


def test_a_single_turn_payload_on_an_episode_job_is_malformed(fake_r2):
    asyncio.run(job_store.write_job(_manifest(prompt_count=3), None, **fake_r2))
    store = _CountingStore(fake_r2)
    request = _request()
    single = CorpusSubmissionRequest(
        job_id="swe-agentic-v1", miner_hotkey="5Hot", cursor=0, prompt_index=0,
        checkpoint_sha256="c" * 64, rendered_prompt=request.rendered_prompt,
        completions=[{"tokens": [200] * 12, "text": "x" * 12, "proofs": [PROOF]}], signature="ok")
    records, accepted = _Records(), []
    body = _post(_client(store, records, accepted), single)
    assert (body["accepted"], body["reason"]) == (False, "malformed_submission")
    assert records.written == {} and accepted == []


def test_the_intake_passes_the_jobs_max_turns_to_the_parser(episode_store):
    # Three turns labelled max_turns on a job whose limit is not three.
    body = _post(_client(episode_store, _Records(), []), _request(stop="max_turns"))
    assert (body["accepted"], body["reason"]) == (False, "bad_stop")


def test_a_refusal_consumes_no_slot_and_no_cursor(episode_store):
    client = _client(episode_store, _Records(), [])
    assert _post(client, _request(rendered="nope"))["reason"] == "prompt_not_faithful"
    assert _post(client, _request())["accepted"] is True       # the same cursor still works


def test_the_record_keeps_its_trajectory_before_the_cursor_for_the_tail_read(episode_store):
    import json
    from reliquary.infrastructure.corpus_record_store import RECORD_SCHEMA_V2, submission_meta

    records = _Records()
    request = _request()
    assert _post(_client(episode_store, records, []), request)["accepted"] is True
    record = records.written[corpus_submission_id(request)]
    assert record["schema"] == RECORD_SCHEMA_V2
    stored = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    assert stored.index(b'"completions"') < stored.index(b'"cursor"')
    meta = submission_meta(stored[-400:])
    assert meta["hotkey"] == "5Hot" and meta["token_count"] == 30


def test_the_intake_decodes_under_a_lock():
    import threading

    class Probe(_Tokenizer):
        inside = 0
        overlapped = False

        def decode(self, ids, **kwargs):
            import time
            Probe.inside += 1
            Probe.overlapped |= Probe.inside > 1
            time.sleep(0.01)
            Probe.inside -= 1
            return super().decode(ids)

    job = parse_job(_manifest(prompt_count=3))
    source = SweSource([("a", "fix it"), ("b", "x"), ("c", "y")])
    intake = EpisodeIntake(job=job, source=source, renderer=R, tokenizer=Probe(),
                           vocab_size=None, chunk_tokens=32)
    request = _request()
    threads = [threading.Thread(target=intake.check, args=(request,)) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert Probe.overlapped is False


# --- the request-body cap (ruling P9) ---------------------------------------

from reliquary.validator.corpus_service import MIN_SUBMIT_BODY_BYTES as MAX_SUBMIT_BODY_BYTES  # noqa: E402


def _signature_spy():
    calls = []

    def verify(request):
        calls.append(request)
        return True
    return verify, calls


def test_an_over_cap_body_is_refused_before_it_is_parsed(episode_store):
    from reliquary.validator.corpus_service import build_corpus_router

    verify, calls = _signature_spy()
    app = FastAPI()
    app.include_router(build_corpus_router(
        job_id="swe-agentic-v1", store=episode_store, tokenizer=TOKENIZER, renderer=None,
        verify_signature=verify, proof_chunk_tokens=32, episode_intake=_intake()))
    client = TestClient(app)
    big = b"x" * (MAX_SUBMIT_BODY_BYTES + 1)         # not JSON: a parse would answer 422
    assert client.post("/corpus/submit", content=big).status_code == 413
    # Declared length absent (chunked): counted as it streams, refused at the cap.
    assert client.post("/corpus/submit", content=iter([big[:5_000_000], big[5_000_000:]])
                       ).status_code == 413
    assert calls == []


def test_the_scoped_and_legacy_routes_are_capped_too(episode_store):
    from reliquary.validator.corpus_service import (
        CorpusJobRoutes, build_corpus_jobs_router, build_corpus_router)

    router = build_corpus_router(
        job_id="swe-agentic-v1", store=episode_store, tokenizer=TOKENIZER, renderer=None,
        verify_signature=lambda r: True, proof_chunk_tokens=32, episode_intake=_intake())
    routes = CorpusJobRoutes()
    routes.add("swe-agentic-v1", router)
    app = FastAPI()
    app.include_router(build_corpus_jobs_router(routes, legacy=True))
    client = TestClient(app)
    big = b"x" * (MAX_SUBMIT_BODY_BYTES + 1)
    assert client.post("/corpus/submit", content=big).status_code == 413
    assert client.post("/corpus/jobs/swe-agentic-v1/submit", content=big).status_code == 413
    # Under the cap an honest trajectory still goes through, on both.
    assert client.post("/corpus/jobs/swe-agentic-v1/submit",
                       json=_request().model_dump()).json()["accepted"] is True


def test_an_honest_body_is_far_under_the_cap():
    assert len(_request().model_dump_json()) < MAX_SUBMIT_BODY_BYTES // 100


# --- ruling P10: the cap follows the job ------------------------------------

MIB = 1024 * 1024


def _single_turn_job(n=16, max_new_tokens=32768):
    raw = _manifest(with_episode=False, prompt_count=3)
    raw["sampling"] = {**raw["sampling"], "max_new_tokens": max_new_tokens, "n": n}
    return parse_job(raw)


def test_worst_case_body_by_job_kind():
    from reliquary.protocol.corpus_submission import MAX_RENDERED_PROMPT_CHARS
    from reliquary.validator.corpus_service import worst_case_body

    big = _single_turn_job()
    assert worst_case_body(big) == 16 * 32768 * 24 + MAX_RENDERED_PROMPT_CHARS + 64 * 1024
    assert worst_case_body(big) > 12_000_000
    assert worst_case_body(_single_turn_job(n=1, max_new_tokens=64)) == 8 * MIB     # never below
    assert worst_case_body(parse_job(_manifest(prompt_count=3))) == 8 * MIB          # episode


def _router(job, store, **kw):
    from reliquary.validator.corpus_service import build_corpus_router

    return build_corpus_router(
        job_id=str(job.job_id), store=store, tokenizer=TOKENIZER, renderer=None,
        verify_signature=lambda r: True, proof_chunk_tokens=32, job=job, **kw)


def test_a_single_turn_job_takes_a_twelve_megabyte_body(episode_store):
    job = _single_turn_job()
    app = FastAPI()
    app.include_router(_router(job, episode_store))
    client = TestClient(app)
    junk = b"x" * 12_000_000          # not JSON: past the cap check it parses and answers 422
    assert client.post("/corpus/submit", content=junk).status_code == 422
    over = b"x" * (_router(job, episode_store).body_cap + 1)
    assert client.post("/corpus/submit", content=over).status_code == 413


def test_an_episode_job_refuses_nine_mebibytes(episode_store):
    job = parse_job(_manifest(prompt_count=3))
    app = FastAPI()
    app.include_router(_router(job, episode_store, episode_intake=_intake()))
    response = TestClient(app).post("/corpus/submit", content=b"x" * (9 * MIB))
    assert response.status_code == 413


def test_scoped_uses_its_own_job_and_legacy_the_max_over_served_jobs(episode_store):
    from reliquary.validator.corpus_service import CorpusJobRoutes, build_corpus_jobs_router

    small = _router(parse_job(_manifest(prompt_count=3)), episode_store, episode_intake=_intake())
    routes = CorpusJobRoutes()
    routes.add("swe-agentic-v1", small)
    app = FastAPI()
    app.include_router(build_corpus_jobs_router(routes, legacy=True))
    client = TestClient(app)
    body = b"x" * (9 * MIB)
    assert client.post("/corpus/submit", content=body).status_code == 413
    # A big single-turn job joins while serving (hot add): read at request time.
    big_job = _single_turn_job()
    routes.add("big-v1", _router(big_job, episode_store))
    assert client.post("/corpus/submit", content=body).status_code == 422
    assert client.post("/corpus/jobs/big-v1/submit", content=body).status_code == 422
    assert client.post("/corpus/jobs/swe-agentic-v1/submit", content=body).status_code == 413


def test_every_check_reason_is_a_reject_reason():
    from reliquary.corpus import checks, trajectory_parse
    from reliquary.protocol.corpus_submission import CorpusRejectReason

    values = {r.value for r in CorpusRejectReason}
    found = [(m.__name__, name, v) for m in (checks, trajectory_parse) for name, v in vars(m).items()
             if name.startswith("REASON_") and isinstance(v, str)]
    assert found
    assert [f for f in found if f[2] not in values] == []


def test_an_unknown_intake_reason_is_a_server_error(episode_store):
    from reliquary.validator.agentic_intake import IntakeRefusal

    class Broken:
        def check(self, request):
            return IntakeRefusal("not_a_reason", {})

    from reliquary.validator.corpus_service import build_corpus_router

    app = FastAPI()
    app.include_router(build_corpus_router(
        job_id="swe-agentic-v1", store=episode_store, tokenizer=TOKENIZER, renderer=None,
        verify_signature=lambda r: True, proof_chunk_tokens=32, episode_intake=Broken()))
    assert TestClient(app).post("/corpus/submit", json=_request().model_dump()).status_code == 500
