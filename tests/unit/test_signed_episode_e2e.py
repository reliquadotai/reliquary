"""End to end without Docker: reliquary-sandbox's real gateway (`local_gateway` over
`FakeBoxes`: real routes, manager, signing and grading) on 127.0.0.1, this validator's
machine directory in (fake) R2, fleet, session route, signed intake, ledger, grader and
export, and a scripted miner. The miner signs with a real sr25519 hotkey, asks for a
token with the miner's own request helpers, opens through `EpisodeClient`, answers its
calls with `ToolRouter` (offered = record 0's tools), finishes and submits.

The first tests wire the components by hand; the last drives the real
`run_corpus_validator` startup (served by uvicorn on 127.0.0.1) with signed jobs, a
replay episode job beside them, and a restart in the middle of a session."""

from __future__ import annotations

import asyncio
import json
import socket
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

pytest.importorskip("reliquary_sandbox_service.episodes.testing")
bt = pytest.importorskip("bittensor")

from reliquary_sandbox.attest import Ed25519TokenVerifier  # noqa: E402
from reliquary_sandbox.episode_client import EpisodeClient  # noqa: E402
from reliquary_sandbox.tool_routing import ToolRouter  # noqa: E402
from reliquary_sandbox_service.episodes import registry, testing  # noqa: E402

from reliquary.corpus.checks import span_chunk_count  # noqa: E402
from reliquary.corpus.delivery import episode_rows  # noqa: E402
from reliquary.corpus.job import parse_job  # noqa: E402
from reliquary.corpus.signed_reasons import session_seen_key  # noqa: E402
from reliquary.environment.agentic_swe import SignedSweSource  # noqa: E402
from reliquary.infrastructure import corpus_job_store as job_store  # noqa: E402
from reliquary.infrastructure import corpus_record_store, sandbox_store  # noqa: E402
from reliquary.infrastructure.corpus_record_store import BucketRecordStore  # noqa: E402
from reliquary.miner.signed_episode import (  # noqa: E402
    signed_close_request, signed_open_request,
)
from reliquary.protocol.corpus_submission import CorpusSubmissionRequest  # noqa: E402
from reliquary.protocol.sandbox_session import sandbox_close_path, sandbox_open_path  # noqa: E402
from reliquary.protocol.signatures import (  # noqa: E402
    build_corpus_binding, verify_corpus_signature,
)
from reliquary.sandbox.fleet import Fleet, http_fetch_report  # noqa: E402
from reliquary.sandbox.routes import build_sandbox_sessions_router  # noqa: E402
from reliquary.sandbox.sessions import (  # noqa: E402
    CLOSED, CLOSED_GRADED, LIVE, SUBMITTED, CorpusEngagements, SandboxPolicy, SessionBook,
    SessionIssuer, SignedJobView,
)
from reliquary.sandbox.tasks import ResolvedTask  # noqa: E402
from reliquary.validator.corpus_service import build_corpus_router, rebuild_ledgers  # noqa: E402
from reliquary.validator.signed_grading import SignedEpisodeGrader  # noqa: E402
from reliquary.validator.signed_intake import SignedEpisodeIntake  # noqa: E402
from tests.unit.test_corpus_job_episode import _manifest  # noqa: E402
from tests.unit.test_corpus_job_signed_sandbox import signed_episode  # noqa: E402
from tests.unit.test_corpus_service import (  # noqa: E402, F401
    _CountingStore, _r2_client, _Tokenizer, fake_r2,
)
from tests.unit.test_corpus_validator import fixed_drand_chain  # noqa: E402, F401
from tests.unit.test_trajectory_parse import CHAR, TERM, TEXT, FakeRenderer  # noqa: E402

CMD_OPEN, CMD_CLOSE = 8, 9
MANIFEST = _manifest(prompt_count=3, episode=signed_episode())
JOB = parse_job(MANIFEST)
PROMPT_TEXT = "Write 42 to /work/answer.txt."
SOURCE = SignedSweSource("train:20", prompt_of=lambda split, index: PROMPT_TEXT,
                         row_of=lambda split, index: (None, SimpleNamespace(instance_id=f"fake-{index}")))
TOKENIZER = _Tokenizer()
PROOF = "A" * 200
CHUNK = 64
MINER = bt.Keypair.create_from_uri("//Alice")
HOTKEY = MINER.ss58_address
VALIDATOR = bt.Keypair.create_from_uri("//Bob")        # the audience of every session request
RIGHT, WRONG = "write /work/answer.txt 42", "echo hi"
BURN = "burn 4000000"                                  # 4000 s of CPU: past the job's cpu_s


class ScriptRenderer(FakeRenderer):
    """A bash call is CMD_OPEN, the command's characters, CMD_CLOSE."""

    def tool_calls(self, completion_ids):
        calls, text = [], None
        for token in completion_ids:
            if token == CMD_OPEN:
                text = []
            elif token == CMD_CLOSE and text is not None:
                calls.append(("bash", json.dumps({"command": "".join(chr(t - CHAR) for t in text)})))
                text = None
            elif text is not None:
                text.append(token)
        return calls


class ExportRenderer(ScriptRenderer):
    """The export's renderer: messages render to the ids it was shown (the fake has no
    chat template; the observations and calls are still parsed forward)."""

    rendered: list[int] = []

    def render_messages(self, messages):
        return list(self.rendered)

    def whitespace_free(self, ids):
        return tuple(ids)


R = ScriptRenderer()


def completion(command=None):
    body = [TEXT] * 8
    if command is not None:
        body += [CMD_OPEN] + [CHAR + ord(c) for c in command] + [CMD_CLOSE]
    return body + [TERM]


def sign(binding: bytes) -> str:
    return MINER.sign(binding).hex()


class Clock:
    """time.time() plus a skew the test moves (the directory's age, freshness)."""

    def __init__(self):
        self.skew = 0.0

    def __call__(self):
        return time.time() + self.skew


@pytest.fixture
def r2(fake_r2, _r2_client, monkeypatch):  # noqa: F811
    """One fake bucket under the job store, the record store and the sandbox store."""
    monkeypatch.setattr(sandbox_store, "get_s3_client", lambda **kw: _r2_client)
    monkeypatch.setattr(corpus_record_store, "get_s3_client", lambda **kw: _r2_client)
    asyncio.run(job_store.write_job(MANIFEST, None, **fake_r2))
    return SimpleNamespace(kwargs=fake_r2, client=_r2_client)


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    # The fake env lives in the reliquary-sandbox distribution; the machine reports the
    # job's env package for it so record 0 names what the job pins (packaging is not
    # what these tests are about).
    monkeypatch.setattr(registry, "_package_of", lambda entry: JOB.episode.sandbox.env_package)
    with testing.fake_gateway(tmp_path / "gw", envs={"reliquary-swe": testing.FAKE_ENV_ENTRY}) \
            as (served, boxes):
        served.boxes = boxes             # the FakeBoxes the gateway runs (a test may extend it)
        yield served


async def register(gateway) -> None:
    """The admin's `reliquary sandbox machines register`, into the fake bucket."""
    await sandbox_store.register_machine(
        machine_id=gateway.settings.machine_id, address=gateway.url, provider="local",
        capacity=4, key_id=gateway.machine.key_id, public_key_b64=gateway.machine.public_key_b64,
        valid_from=0, now=time.time())


def session_documents(r2) -> list[dict]:
    return [json.loads(body) for key, (body, _) in r2.client.objects.items()
            if key.startswith(sandbox_store.SESSION_PREFIX)]


def build_validator(gateway, r2, *, clock=None, policy=None, renderer=R, tokenizer=TOKENIZER,
                    chunk=CHUNK):
    """The validator's signed side, wired by hand over the shared fake bucket: a fresh
    process each call (a restart is a second call)."""
    clock = clock or Clock()
    policy = policy or SandboxPolicy()

    async def documents():
        return await sandbox_store.list_machines()

    fleet = Fleet(read_documents=documents, fetch_report=http_fetch_report, clock=clock)
    book = SessionBook(policy)
    tokens = Ed25519TokenVerifier({gateway.validator.key_id: gateway.validator.public_key_b64})
    views = {}
    issuer = SessionIssuer(book=book, store=sandbox_store.R2SessionStore(), fleet=fleet,
                           signer=gateway.validator, token_verifier=tokens,
                           engagements={"corpus": CorpusEngagements(views.get, book, clock)},
                           policy=policy, clock=clock)
    fleet.on_drained = issuer.void_machine
    intake = SignedEpisodeIntake(job=JOB, source=SOURCE, renderer=renderer, tokenizer=tokenizer,
                                 vocab_size=None, chunk_tokens=chunk,
                                 directory=fleet.directory_if_ready, token_verifier=tokens,
                                 sessions=issuer, seen=book.submitted_ids, clock=clock,
                                 retry_after_s=policy.retry_after_s)
    records = BucketRecordStore()
    accepted = []
    corpus = build_corpus_router(job_id=JOB.job_id, store=_CountingStore(r2.kwargs),
                                 tokenizer=tokenizer, renderer=None, records=records,
                                 verify_signature=verify_corpus_signature,
                                 on_accepted=accepted.append, proof_chunk_tokens=chunk,
                                 episode_intake=intake, job=JOB)

    async def resolve(index):
        return ResolvedTask(testing.FAKE_IMAGE, {})

    views[JOB.job_id] = SignedJobView(job=JOB, resolve_task=resolve,
                                      slots_remaining=corpus.slots_remaining)
    app = FastAPI()
    app.include_router(corpus)
    app.include_router(build_sandbox_sessions_router(
        issuer, policy=policy, validator_hotkey=VALIDATOR.ss58_address, clock=clock))
    return SimpleNamespace(app=app, fleet=fleet, book=book, issuer=issuer, records=records,
                           corpus=corpus, clock=clock, accepted=accepted, intake=intake)


async def started(gateway, r2, **kw):
    world = build_validator(gateway, r2, **kw)
    await world.issuer.restore()
    await world.fleet.refresh_directory()
    await world.fleet.poll_once()
    return world


def http(world) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=world.app), base_url="http://validator")


# -- the scripted miner -----------------------------------------------------------------

def open_body(request_id: str, *, now: float, prompt_index: int = 0, job_id: str = JOB.job_id,
              prefix: str = "/corpus") -> dict:
    return signed_open_request(hotkey=HOTKEY, job_id=job_id, prompt_index=prompt_index,
                               sign_binding=sign, now=now, request_id=request_id,
                               validator_hotkey=VALIDATOR.ss58_address,
                               path=sandbox_open_path(prefix))


def close_body(session_id: str, request_id: str, *, now: float, transcript, reason="final") -> dict:
    return signed_close_request(hotkey=HOTKEY, session_id=session_id, reason=reason,
                                transcript=transcript, sign_binding=sign, now=now,
                                request_id=request_id, validator_hotkey=VALIDATOR.ss58_address,
                                path=sandbox_close_path("/corpus", session_id))


class Played(SimpleNamespace):
    """One episode as a miner played it: what to submit."""


async def start_episode(grant: dict, command: str):
    """Open on the machine the validator named and route the first turn's calls; the
    episode is left open (the caller finishes it)."""
    prompt = R.initial_ids(SOURCE.prompt(0))
    first = completion(command)
    client = EpisodeClient(grant["gateway_url"])
    episode = await client.open(grant["token"])
    router = ToolRouter(episode, offered=tuple(episode.record0["body"]["tools"]))
    calls = [(f"call_{j}", name, arguments) for j, (name, arguments) in enumerate(R.tool_calls(first))]
    observations = [await router.answer(0, calls, call_id) for call_id, _, _ in calls]
    return Played(client=client, episode=episode, prompt=prompt, first=first, command=command,
                  full=R.next_prompt(prompt, first, observations), observations=observations)


async def reattach(played: Played) -> None:
    """The miner's HTTP client for an episode it keeps playing from another event loop
    (the episode's id and key are all the gateway needs)."""
    from reliquary_sandbox.episode_client import Episode

    played.client = EpisodeClient(played.client.base_url)
    played.episode = Episode(played.client, played.episode.id, played.episode.key,
                             played.episode.record0)


async def finish_episode(played: Played) -> Played:
    played.finished = await played.episode.finish()
    played.transcript = (await played.episode.transcript())["transcript"]
    await played.client.aclose()
    return played


async def play(grant: dict, command: str) -> Played:
    return await finish_episode(await start_episode(grant, command))


def submission(played: Played, *, last=None, transcript=None, full=None, chunk=CHUNK,
               final_diff=None) -> dict:
    """The miner's submission, signed by its hotkey (no stub verifier anywhere). `full`:
    the second turn's prompt the tokens show (the honest rendering by default)."""
    last = last or completion()
    full = played.full if full is None else full
    body_tokens = full[len(played.prompt):] + last
    spans = [(0, len(played.first)), (len(full) - len(played.prompt), len(body_tokens))]
    request = CorpusSubmissionRequest(
        job_id=JOB.job_id, miner_hotkey=HOTKEY, cursor=0, prompt_index=0,
        checkpoint_sha256=JOB.checkpoint_sha256, rendered_prompt=TOKENIZER.decode(played.prompt),
        trajectory={"tokens": body_tokens,
                    "turns": [{"start": s, "end": e, "proofs": [PROOF] * span_chunk_count(e - s, chunk)}
                              for s, e in spans],
                    "final_diff": (final_diff if final_diff is not None
                                   else (played.finished.state or b"").decode("utf-8")),
                    "stop": "agent_completed",
                    "transcript": transcript if transcript is not None else played.transcript},
        signature="unsigned")
    request = request.model_copy(update={"signature": sign(build_corpus_binding(request))})
    return request.model_dump()


async def grant_for(validator, world, request_id: str) -> dict:
    answer = await validator.post("/corpus/sandbox/sessions", json=open_body(request_id, now=world.clock()))
    grant = answer.json()
    assert answer.status_code == 200 and "token" in grant, grant
    return grant


def ledger(r2):
    snapshot, _ = asyncio.run(job_store.read_ledgers(JOB.job_id, **r2.kwargs))
    return rebuild_ledgers(JOB, snapshot)


# -- components wired by hand -----------------------------------------------------------

async def test_an_honest_episode_is_admitted_once_graded_and_exported_once(gateway, r2):
    await register(gateway)
    world = await started(gateway, r2)
    async with http(world) as validator:
        grant = await grant_for(validator, world, "a" * 32)
        assert world.book.reserved(JOB.job_id, 0, int(time.time())) == 1
        played = await play(grant, RIGHT)
        assert played.finished.record["body"]["status"] == "graded"
        assert played.finished.record["body"]["reward"] == 1.0
        answer = (await validator.post("/corpus/submit", json=submission(played))).json()
        assert answer["reason"] == "accepted", answer
        assert world.book.get(grant["session_id"]).state == SUBMITTED
        assert world.book.reserved(JOB.job_id, 0, int(time.time())) == 0
        assert await world.corpus.slots_remaining(0) == JOB.slots_per_prompt - 1
        # The same session again, with another last turn (another submission id).
        again = (await validator.post("/corpus/submit", json=submission(
            played, last=[TEXT] * 10 + [TERM]))).json()
        # Refused by the paid-session snapshot first (`Expected.seen_session_ids`); the
        # ledger's seen key is the backstop when the snapshot is late.
        assert again["reason"] == "sandbox_transcript_invalid", again
        assert again["detail"]["reasons"] == ["session_reused"]
        # The same open request resent after the episode: never a second token.
        resent = await validator.post("/corpus/sandbox/sessions",
                                      json=open_body("a" * 32, now=world.clock()))
        assert resent.status_code == 409 and resent.json()["reason"] == "request_reused"
    assert await world.corpus.slots_remaining(0) == JOB.slots_per_prompt - 1
    (submission_id,) = world.accepted
    assert session_seen_key(grant["session_id"]) in (await asyncio.to_thread(ledger, r2)).pending
    record = await world.records.read_submission(JOB.job_id, submission_id)
    assert record["completions"][0]["transcript"]["token"]["claims"]["session_id"] == grant["session_id"]

    # Graded from the transcript's final record; the audit's verdict stands in for TOPLOC.
    await world.records.write_verdict(JOB.job_id, submission_id, {"passed": True})
    grader = SignedEpisodeGrader(job=JOB, records=world.records, source=SOURCE)
    grade = await grader.grade_one(submission_id)
    assert grade["status"] == "ok" and grade["graded_success"] is True
    assert grade["graded_by"] == [f"sandbox:{gateway.settings.machine_id}"]
    assert grade["grade"]["session_id"] == grant["session_id"]

    renderer = ExportRenderer()
    renderer.rendered = played.full + completion()
    counts = {}
    rows = [row async for row in episode_rows(job=JOB, records=world.records, renderer=renderer,
                                              source=SOURCE, counts=counts, quarantined=())]
    assert counts["rows"] == 1 and counts["passing_submissions"] == 1
    (row,) = rows
    assert row["graded_success"] is True and row["replay_certified"] is True
    tool_messages = [m["content"] for m in json.loads(row["messages"]) if m["role"] == "tool"]
    assert tool_messages == played.observations
    # No session document ever holds the token or its signature.
    stored = session_documents(r2)
    assert [d["state"] for d in stored] == [SUBMITTED]
    assert all("token" not in d and "signature" not in d for d in stored)
    assert grant["token"]["signature"] not in json.dumps(stored)


async def test_an_unpaid_episode_closes_and_frees_its_slot(gateway, r2):
    await register(gateway)
    world = await started(gateway, r2)
    async with http(world) as validator:
        grant = await grant_for(validator, world, "b" * 32)
        played = await play(grant, BURN)
        assert played.finished.record["body"]["status"] == "budget_exhausted"
        closed = await validator.post(f"/corpus/sandbox/sessions/{grant['session_id']}/close",
                                      json=close_body(grant["session_id"], "c" * 32,
                                                      now=world.clock(), transcript=played.transcript))
        assert closed.status_code == 200, closed.json()
        assert closed.json()["state"] == CLOSED and closed.json()["status"] == "budget_exhausted"
        assert world.book.reserved(JOB.job_id, 0, int(time.time())) == 0
        assert await world.corpus.slots_remaining(0) == JOB.slots_per_prompt     # nothing paid
        # Its transcript is never paid afterwards.
        refused = (await validator.post("/corpus/submit", json=submission(played))).json()
        assert refused["reason"] == "sandbox_transcript_invalid", refused
        # The prompt is free again for this hotkey: a new open is granted.
        await grant_for(validator, world, "d" * 32)


async def test_an_unsubmitted_graded_close_holds_its_slot_until_it_is_submitted(gateway, r2):
    await register(gateway)
    world = await started(gateway, r2)
    async with http(world) as validator:
        grant = await grant_for(validator, world, "e" * 32)
        played = await play(grant, WRONG)
        assert played.finished.record["body"]["reward"] == 0.0
        closed = (await validator.post(f"/corpus/sandbox/sessions/{grant['session_id']}/close",
                                       json=close_body(grant["session_id"], "f" * 32,
                                                       now=world.clock(),
                                                       transcript=played.transcript))).json()
        assert closed["state"] == CLOSED_GRADED and closed["status"] == "graded"
        assert world.book.reserved(JOB.job_id, 0, int(time.time())) == 1
        answer = (await validator.post("/corpus/submit", json=submission(played))).json()
        assert answer["reason"] == "accepted", answer
        assert world.book.reserved(JOB.job_id, 0, int(time.time())) == 0
        assert world.book.get(grant["session_id"]).state == SUBMITTED


async def test_a_stale_directory_is_a_retryable_refusal_everywhere(gateway, r2):
    await register(gateway)
    world = await started(gateway, r2)
    async with http(world) as validator:
        grant = await grant_for(validator, world, "1" * 32)
        played = await play(grant, RIGHT)
        world.clock.skew = 200.0               # past the directory's 120 s, inside the token
        body = submission(played)
        refused = await validator.post("/corpus/submit", json=body)
        assert refused.status_code == 503 and refused.headers.get("Retry-After"), refused.json()
        assert refused.json()["detail"] == "sandbox_directory_unavailable"     # the 503 shape
        opened = await validator.post("/corpus/sandbox/sessions",
                                      json=open_body("2" * 32, now=world.clock(), prompt_index=1))
        assert opened.status_code == 503 and opened.headers.get("Retry-After")
        assert opened.json()["reason"] == "directory_unavailable"
        closed = await validator.post(f"/corpus/sandbox/sessions/{grant['session_id']}/close",
                                      json=close_body(grant["session_id"], "3" * 32,
                                                      now=world.clock(), transcript=played.transcript))
        assert closed.status_code == 503 and closed.json()["reason"] == "directory_unavailable"
        # Nothing was consumed by the refusals; once the directory is read again the
        # same submission is paid.
        assert world.book.get(grant["session_id"]).state == LIVE
        assert not world.book.is_claimed(grant["session_id"])
        await world.fleet.refresh_directory()
        answer = (await validator.post("/corpus/submit", json=body)).json()
        assert answer["reason"] == "accepted", answer


def _forge_output(transcript: dict, text: str) -> dict:
    forged = json.loads(json.dumps(transcript))
    for record in forged["records"]:
        if record["body"].get("output") is not None and record["body"].get("turn") == 0:
            record["body"]["output"] = text
    return forged


def _forge_reward(transcript: dict) -> dict:
    forged = json.loads(json.dumps(transcript))
    final = forged["records"][-1]["body"]
    final["reward"] = 1.0
    return forged


async def test_forged_transcripts_are_refused_and_consume_nothing(gateway, r2):
    await register(gateway)
    world = await started(gateway, r2)
    async with http(world) as validator:
        grant = await grant_for(validator, world, "4" * 32)
        played = await play(grant, "cat /work/answer.txt")
        assert played.finished.record["body"]["reward"] == 0.0
        # A failed episode's reward rewritten to 1.0: the machine's signature breaks.
        reward = (await validator.post("/corpus/submit", json=submission(
            played, transcript=_forge_reward(played.transcript)))).json()
        assert reward["reason"] == "sandbox_transcript_invalid", reward
        assert "bad_signature" in reward["detail"]["reasons"]
        # An observation rewritten in both the record and the tokens: the signature breaks.
        forged = _forge_output(played.transcript, "42\n")
        assert forged != played.transcript and _forge_reward(played.transcript) != played.transcript
        shown = R.next_prompt(played.prompt, played.first, ["42\n"])
        output = (await validator.post("/corpus/submit", json=submission(
            played, transcript=forged, full=shown))).json()
        assert output["reason"] == "sandbox_transcript_invalid", output
        assert "bad_signature" in output["detail"]["reasons"]
        # The honest transcript beside tokens showing another observation: §5.C.
        tokens = (await validator.post("/corpus/submit", json=submission(played, full=shown))).json()
        assert tokens["reason"] == "bad_observation", tokens
        # A final diff that is not the graded state: §5.D.
        diff = (await validator.post("/corpus/submit", json=submission(played, final_diff="x\n"))).json()
        assert diff["reason"] == "sandbox_state_mismatch", diff
        # None of them consumed a slot or the session: the honest one is still paid.
        assert await world.corpus.slots_remaining(0) == JOB.slots_per_prompt
        assert world.book.get(grant["session_id"]).state == LIVE
        honest = (await validator.post("/corpus/submit", json=submission(played))).json()
        assert honest["reason"] == "accepted", honest


async def test_a_restart_mid_session_restores_the_book(gateway, r2):
    await register(gateway)
    first = await started(gateway, r2)
    async with http(first) as validator:
        grant = await grant_for(validator, first, "5" * 32)
    played = await start_episode(grant, RIGHT)          # mid-episode: the validator restarts

    second = await started(gateway, r2)                 # a new process over the same bucket
    restored = second.book.get(grant["session_id"])
    assert restored is not None and restored.state == LIVE and restored.hotkey == HOTKEY
    assert second.book.reserved(JOB.job_id, 0, int(time.time())) == 1
    async with http(second) as validator:
        # The reservation and the hotkey's caps came back: no second session on the prompt.
        again = await validator.post("/corpus/sandbox/sessions",
                                     json=open_body("6" * 32, now=second.clock()))
        assert again.status_code == 429 and again.json()["reason"] == "prompt_live_cap"
        # Tokens are never persisted: the resent request gets no token back.
        resent = await validator.post("/corpus/sandbox/sessions",
                                      json=open_body("5" * 32, now=second.clock()))
        assert resent.status_code == 409 and resent.json()["reason"] == "request_reused"
        assert "token" not in resent.json()
        await finish_episode(played)
        answer = (await validator.post("/corpus/submit", json=submission(played))).json()
        assert answer["reason"] == "accepted", answer
    assert second.book.get(grant["session_id"]).state == SUBMITTED

    third = await started(gateway, r2)                  # and once more: paid once
    assert third.book.get(grant["session_id"]).state == SUBMITTED
    assert grant["session_id"] in third.book.submitted_ids()
    async with http(third) as validator:
        reused = (await validator.post("/corpus/submit", json=submission(
            played, last=[TEXT] * 10 + [TERM]))).json()
        assert reused["reason"] == "sandbox_transcript_invalid", reused
        assert reused["detail"]["reasons"] == ["session_reused"]
    assert await third.corpus.slots_remaining(0) == JOB.slots_per_prompt - 1


# -- the real validator startup -----------------------------------------------------------

JOB_B = parse_job(_manifest(prompt_count=3, job_id="swe-signed-b", episode=signed_episode()))
JOB_C = parse_job(_manifest(prompt_count=3, job_id="swe-replay-c"))       # replay, beside them
TASKS = {JOB.job_id: "corpus-signed-a", JOB_B.job_id: "corpus-signed-b",
         JOB_C.job_id: "corpus-replay-c"}


class _Stop(Exception):
    pass


def _entry(job_id, status="active"):
    return SimpleNamespace(task_id=TASKS[job_id], job_id=job_id, params={"cap": 0.1},
                           mechanism="corpus-generation", status=status,
                           retired_at=1 if status == "retired" else None, contract=None)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def validator_process(tmp_path, gateway, r2, fixed_drand_chain, monkeypatch):  # noqa: F811
    """`run_corpus_validator` for real (split front, so no GPU): signed jobs A and B and
    replay episode job C, the sandbox key from the environment, the machine directory
    in the fake bucket, uvicorn on 127.0.0.1. Stubbed: the GPU process, the model files,
    the replay job's task set and grade executors, the env packages' install pins, and
    the TOPLOC audit loop. `run(scenario)` starts it, runs `scenario(app, url)` while it
    serves, and stops it; each call is a fresh process over the same bucket."""
    import uvicorn

    import reliquary.shared.modeling as modeling
    from reliquary.environment import agentic_swe
    from reliquary.environment.agentic_swe import SweSource
    from reliquary.infrastructure import corpus_executor_store, storage, task_registry_store
    from reliquary.sandbox import tasks
    from reliquary.validator import (
        agentic_intake, corpus_auditor, corpus_gpu, corpus_grade_remote, corpus_grading,
        corpus_hot_jobs, corpus_service, corpus_validator,
    )
    from reliquary.validator.corpus_split import FrontSplit
    from tests.unit import corpus_split_fakes as fakes

    # Every R2 client is the fake bucket: nothing leaves the process.
    for module in (corpus_executor_store, task_registry_store, storage):
        monkeypatch.setattr(module, "get_s3_client", lambda **kw: r2.client)
    for job in (JOB_B, JOB_C):
        asyncio.run(job_store.write_job(job.to_contract(), None, **r2.kwargs))
    asyncio.run(register(gateway))
    key_file = tmp_path / "gw" / "gateway" / "validator.pem"
    assert key_file.is_file()
    monkeypatch.setenv("RELIQUARY_SANDBOX_VALIDATOR_KEY_FILE", str(key_file))
    monkeypatch.setenv("RELIQUARY_SANDBOX_VALIDATOR_KEY_ID", gateway.validator.key_id)

    async def idle(*args, **kwargs):
        await asyncio.sleep(3600)

    async def info(run_dir, **kw):
        return {"vocab_size": fakes.VOCAB}

    async def no_executors(**kw):
        return []

    monkeypatch.setattr(modeling, "load_tokenizer", lambda path: TOKENIZER)
    monkeypatch.setattr(corpus_gpu, "read_info", info)
    monkeypatch.setattr(corpus_service, "renderer_for_job", lambda job, encode, **kw: object())
    monkeypatch.setattr(corpus_validator, "_entry_profile", lambda entry: None)
    monkeypatch.setattr(corpus_hot_jobs, "hot_job_refusal", lambda *a, **kw: None)
    monkeypatch.setattr(corpus_auditor.CorpusAuditor, "run", idle)
    monkeypatch.setattr(corpus_grading.CorpusGrader, "run", idle)       # graded by hand below
    monkeypatch.setattr(corpus_validator, "settle_forever", idle)
    # The replay job C: its task set, renderer, lease and (empty) grade executor registry.
    monkeypatch.setattr(agentic_intake, "build_episode_intake", lambda job, **kw: SimpleNamespace(
        renderer=R, source=SweSource([("i0", "p"), ("i1", "q"), ("i2", "r")])))
    monkeypatch.setattr(agentic_intake, "build_grade_renderer", lambda job, **kw: SimpleNamespace())
    monkeypatch.setattr(corpus_grade_remote, "check_replay_lease", lambda job, **kw: None)
    monkeypatch.setattr(corpus_executor_store, "list_executors", no_executors)
    # The signed jobs: the real `build_signed_episode_intake` and `wire_signed_job`, over
    # this test's renderer and prompts instead of reliquary-swe's, without its install pins.
    monkeypatch.setattr(agentic_swe, "episode_support_refusal", lambda *a, **kw: None)
    monkeypatch.setattr(agentic_swe, "sandbox_support_refusal", lambda *a, **kw: None)
    monkeypatch.setattr(agentic_swe, "load_turn_renderer", lambda directory, **kw: R)
    monkeypatch.setattr(agentic_swe, "SignedSweSource", lambda split: SOURCE)
    monkeypatch.setattr(tasks.SweTaskResolver, "_resolve",
                        lambda self, index: ResolvedTask(testing.FAKE_IMAGE, {}))
    # Job B's wiring fails after its session view was registered.
    real_judge = corpus_validator.wire_job_judge

    def judge(w, **kw):
        if w.job.job_id == JOB_B.job_id:
            raise RuntimeError("job B's auditor cannot be wired")
        return real_judge(w, **kw)

    monkeypatch.setattr(corpus_validator, "wire_job_judge", judge)

    class _Server(uvicorn.Server):
        scenario = None

        async def serve(self, sockets=None):
            serving = asyncio.create_task(super().serve(sockets))
            try:
                deadline = time.monotonic() + 60
                while not self.started:
                    assert not serving.done() and time.monotonic() < deadline, "uvicorn did not start"
                    await asyncio.sleep(0.01)
                await _Server.scenario(self.config.app, f"http://127.0.0.1:{self.config.port}")
            finally:
                self.should_exit = True
                await asyncio.wait_for(serving, 60)
            raise _Stop()

    monkeypatch.setattr(uvicorn, "Server", _Server)

    def run(scenario, registry=None):
        """`registry`: the hot job set's entries (None: the jobs given are the jobs served)."""
        _Server.scenario = scenario

        async def read_entries():
            return dict(registry)

        split = FrontSplit(directory=str(tmp_path), fingerprint=JOB.checkpoint_sha256,
                           proof=fakes.PROOF, run_dir=str(tmp_path), links={})
        with pytest.raises(_Stop):
            asyncio.run(corpus_validator.run_corpus_validator(
                jobs=[(_entry(j), 0.1) for j in (JOB.job_id, JOB_B.job_id, JOB_C.job_id)],
                wallet=SimpleNamespace(hotkey=SimpleNamespace(ss58_address=VALIDATOR.ss58_address)),
                netuid=81, signer_client=None, http_host="127.0.0.1", http_port=_free_port(),
                set_weights=False, registration_gate=False, split=split,
                read_registry=read_entries if registry is not None else None,
                refresh_every_seconds=3600))

    return SimpleNamespace(run=run, chunk=fakes.PROOF.chunk_tokens)


def _sessions_client(url: str):
    """The miner's own session client, over HTTP to the served validator."""
    from reliquary.miner.signed_episode import HttpSandboxSessions

    client = httpx.Client(base_url=url, trust_env=False, timeout=30)
    return client, HttpSandboxSessions(client, validator_hotkey=VALIDATOR.ss58_address)


def _open(sessions, *, job_id=JOB.job_id, prompt_index=0, request_id):
    return sessions.open(open_body(request_id, now=time.time(), prompt_index=prompt_index,
                                   job_id=job_id, prefix=sessions.prefix))


async def _refused(sessions, **kw):
    from reliquary.protocol.sandbox_session import SessionRefused

    try:
        answer = await asyncio.to_thread(_open, sessions, **kw)
    except SessionRefused as refused:
        return refused
    raise AssertionError(f"granted: {sorted(answer)}")


def test_a_signed_job_served_by_the_real_validator_startup_across_a_restart(validator_process):
    from reliquary.corpus.trajectory import BuiltTrajectory
    from reliquary.miner.signed_episode import signed_trajectory_precheck, transcript_refusal
    from reliquary.validator.corpus_grading import CorpusGrader

    seen = {}
    held, regraded = [], []

    def hold(self, executor_id):
        held.append(self._job.job_id)

    async def regrade(self, executor_id):
        regraded.append(self._job.job_id)

    async def first_start(app, url):
        services, job_set = app.state.corpus_sandbox, app.state.corpus_jobs
        # Wired: A's view and its transcript grader; B's view forgotten when its
        # wiring failed; the replay job C has no view.
        assert set(services.jobs) == {JOB.job_id}
        assert set(job_set.served) == {JOB.job_id, JOB_C.job_id}
        assert isinstance(job_set.served[JOB.job_id].grader, SignedEpisodeGrader)
        assert job_set.served[JOB.job_id].grader._parse_executor is not None
        # A grade executor quarantine fans out to the replay grader only.
        dispatcher = app.state.corpus_grade_remote
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(CorpusGrader, "hold_executor", hold)
            patch.setattr(CorpusGrader, "regrade_executor", regrade)
            for holder in dispatcher._holders:
                holder("grade-executor-x")
            await asyncio.gather(*(listener("grade-executor-x")
                                   for listener in dispatcher._listeners))
        await services.fleet.poll_once()
        client, sessions = _sessions_client(url)
        try:
            refused = await _refused(sessions, job_id=JOB_B.job_id, request_id="b" * 32)
            assert (refused.reason, refused.status) == ("job_not_served", 404)
            grant = await asyncio.to_thread(_open, sessions, request_id="a" * 32)
        finally:
            client.close()
        seen["grant"] = grant
        seen["played"] = played = await start_episode(grant, RIGHT)
        await played.client.aclose()                            # then the validator stops

    validator_process.run(first_start)
    assert held == [JOB_C.job_id] and regraded == [JOB_C.job_id]
    grant, played = seen["grant"], seen["played"]

    async def second_start(app, url):
        services, job_set = app.state.corpus_sandbox, app.state.corpus_jobs
        # The book came back from the bucket before the routes served.
        restored = services.book.get(grant["session_id"])
        assert restored is not None and restored.state == LIVE
        assert services.book.reserved(JOB.job_id, 0, int(time.time())) == 1
        assert set(services.jobs) == {JOB.job_id}           # B forgotten again
        await services.fleet.poll_once()
        client, sessions = _sessions_client(url)
        try:
            refused = await _refused(sessions, request_id="c" * 32)
            assert (refused.reason, refused.status) == ("prompt_live_cap", 429)
            assert refused.retry_after is not None
            await reattach(played)
            await finish_episode(played)
            # The miner's own checks pass on what the validator will check.
            assert transcript_refusal(played.transcript, job=JOB, hotkey=HOTKEY, index=0,
                                      now=time.time()) is None
            last = completion()
            tokens = played.full[len(played.prompt):] + last
            spans = ((0, len(played.first)), (len(played.full) - len(played.prompt), len(tokens)))
            built = BuiltTrajectory(
                prompt_ids=tuple(played.prompt), tokens=tuple(tokens), spans=spans,
                proofs=tuple((PROOF,) * span_chunk_count(e - s, validator_process.chunk)
                             for s, e in spans),
                final_diff=played.finished.state.decode(), stop="agent_completed",
                transcript=played.transcript)
            assert signed_trajectory_precheck(R, max_turns=JOB.episode.max_turns)(built) is None
            request = CorpusSubmissionRequest(
                job_id=JOB.job_id, miner_hotkey=HOTKEY, cursor=0, prompt_index=0,
                checkpoint_sha256=JOB.checkpoint_sha256,
                rendered_prompt=TOKENIZER.decode(played.prompt), trajectory=built.wire(),
                signature="unsigned")
            request = request.model_copy(update={"signature": sign(build_corpus_binding(request))})
            answer = await asyncio.to_thread(client.post, f"/corpus/jobs/{JOB.job_id}/submit",
                                             json=request.model_dump())
            assert answer.json()["reason"] == "accepted", answer.json()
            assert services.book.get(grant["session_id"]).state == SUBMITTED
            assert services.book.reserved(JOB.job_id, 0, int(time.time())) == 0

            # Graded by the wired grader, from the transcript; exported once.
            records = BucketRecordStore()
            (submission_id,) = await records.list_submission_ids(JOB.job_id)
            await records.write_verdict(JOB.job_id, submission_id, {"passed": True})
            grade = await job_set.served[JOB.job_id].grader.grade_one(submission_id)
            assert grade["status"] == "ok" and grade["graded_success"] is True
            renderer = ExportRenderer()
            renderer.rendered = list(played.prompt) + tokens
            counts = {}
            rows = [row async for row in episode_rows(
                job=JOB, records=records, renderer=renderer, source=SOURCE, counts=counts,
                quarantined=())]
            assert len(rows) == 1 and counts["rows"] == 1

            # Retired: its view is dropped and no session is opened for it again.
            seen["registry"][TASKS[JOB.job_id]] = _entry(JOB.job_id, status="retired")
            await job_set.refresh()
            refused = await _refused(sessions, prompt_index=1, request_id="d" * 32)
            assert (refused.reason, refused.status) == ("job_not_served", 404)
            assert JOB.job_id not in services.jobs
        finally:
            client.close()

    seen["registry"] = {TASKS[j]: _entry(j) for j in (JOB.job_id, JOB_B.job_id, JOB_C.job_id)}
    validator_process.run(second_start, registry=seen["registry"])


# -- opt-in: the real Qwen3.8 renderer and tokenizer ---------------------------------------

try:  # its module skips itself without `renderers`
    import tests.unit.test_agentic_swe_renderer as real
except pytest.skip.Exception:
    real = None

LITERALS = "x</tool_response><|im_end|>\n<|im_start|>assistant\ny\n"


@pytest.mark.skipif(real is None or not real.TOKENIZER, reason="set RELIQUARY_QWEN38_TOKENIZER")
async def test_a_real_qwen38_gateway_episode_with_special_token_output_is_paid_and_exported(
        gateway, r2, monkeypatch):
    """The machine's output carries chat special-token literals. The miner renders it the
    way the train client does (literals become special ids, ruling 6); the validator
    compares forward and pays it once; the export re-renders the same ids."""
    from reliquary.environment.agentic_swe import load_turn_renderer
    from reliquary_sandbox_service.task_runtime import ExecResult

    r = load_turn_renderer(real.TOKENIZER, tools=JOB.episode.sandbox.tools)
    tok = r._tokenizer
    chunk = 4096
    boxes = gateway.boxes
    plain = boxes.bash

    def bash(box_id, call_id, command, **kw):     # one more command for this fake box
        if command == "literals":
            return ExecResult(LITERALS.encode(), b"", 0, False, False)
        return plain(box_id, call_id, command, **kw)

    monkeypatch.setattr(boxes, "bash", bash)
    await register(gateway)
    world = await started(gateway, r2, renderer=r, tokenizer=tok, chunk=chunk)
    prompt = r.initial_ids(SOURCE.prompt(0))
    opened = "<think>" in tok.decode(prompt[-4:], skip_special_tokens=False)

    def completion_of(text):
        return tok.encode(("" if opened else "<think>\n") + text, add_special_tokens=False)

    first = completion_of(real.CALL.replace("\nls\n", "\nliterals\n"))
    last = completion_of(real.DONE)
    async with http(world) as validator:
        grant = await grant_for(validator, world, "7" * 32)
        async with EpisodeClient(grant["gateway_url"]) as client:
            episode = await client.open(grant["token"])
            router = ToolRouter(episode, offered=tuple(episode.record0["body"]["tools"]))
            calls = [(f"call_{j}", name, arguments)
                     for j, (name, arguments) in enumerate(r.tool_calls(first))]
            assert [name for _, name, _ in calls] == ["bash"]
            observations = [await router.answer(0, calls, call_id) for call_id, _, _ in calls]
            assert LITERALS in observations[0]
            finished = await episode.finish()
            signed = (await episode.transcript())["transcript"]
        assert finished.record["body"]["status"] == "graded"
        full = r.next_prompt(prompt, first, observations)
        tokens = full[len(prompt):] + last
        spans = [(0, len(first)), (len(full) - len(prompt), len(tokens))]
        request = CorpusSubmissionRequest(
            job_id=JOB.job_id, miner_hotkey=HOTKEY, cursor=0, prompt_index=0,
            checkpoint_sha256=JOB.checkpoint_sha256,
            rendered_prompt=tok.decode(prompt, skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False),
            trajectory={"tokens": tokens, "stop": "agent_completed", "transcript": signed,
                        "final_diff": (finished.state or b"").decode("utf-8"),
                        "turns": [{"start": a, "end": b,
                                   "proofs": [PROOF] * span_chunk_count(b - a, chunk)}
                                  for a, b in spans]},
            signature="unsigned")
        request = request.model_copy(update={"signature": sign(build_corpus_binding(request))})
        answer = (await validator.post("/corpus/submit", json=request.model_dump())).json()
        assert answer["reason"] == "accepted", answer
    (submission_id,) = world.accepted
    await world.records.write_verdict(JOB.job_id, submission_id, {"passed": True})
    grade = await SignedEpisodeGrader(job=JOB, records=world.records, source=SOURCE).grade_one(
        submission_id)
    assert grade["status"] == "ok"
    counts = {}
    rows = [row async for row in episode_rows(job=JOB, records=world.records, renderer=r,
                                              source=SOURCE, counts=counts, quarantined=())]
    assert counts["rows"] == 1, counts
    tool_messages = [m["content"] for m in json.loads(rows[0]["messages"]) if m["role"] == "tool"]
    assert tool_messages == observations
