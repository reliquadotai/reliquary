"""A signed-sandbox submission through the real route and ledger: every Expected field,
the validator-clock deadline, record 0, §5.C, §5.D, the session claimed before the
write and paid once, recorded in the same compare-and-swap that consumes the slot."""

import asyncio
import dataclasses

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

attest = pytest.importorskip("reliquary_sandbox.attest")

from reliquary.corpus.admission import admit  # noqa: E402
from reliquary.corpus.checks import completion_digest  # noqa: E402
from reliquary.corpus.job import parse_job  # noqa: E402
from reliquary.corpus.signed_reasons import corpus_engagement, session_seen_key  # noqa: E402
from reliquary.environment.agentic_swe import SignedSweSource  # noqa: E402
from reliquary.infrastructure import corpus_job_store as job_store  # noqa: E402
from reliquary.infrastructure.sandbox_store import MemorySessionStore  # noqa: E402
from reliquary.protocol.corpus_submission import CorpusSubmissionRequest  # noqa: E402
from reliquary.sandbox.sessions import (  # noqa: E402
    ABORTED, CLOSED, CLOSED_GRADED, LIVE, SUBMITTED, VOIDED, SandboxPolicy, SessionBook,
    SessionIssuer, SessionRecord,
)
from reliquary.validator import signed_intake  # noqa: E402
from reliquary.validator.corpus_service import RECORD_SCHEMA_V2, rebuild_ledgers  # noqa: E402
from reliquary.validator.signed_intake import SignedEpisodeIntake, SignedIntakeFacts  # noqa: E402
from tests.unit.sandbox_fixtures import (  # noqa: E402
    ENV, MACHINE, NOW, claims, directory, signer, transcript,
)
from tests.unit.test_corpus_job_episode import _manifest  # noqa: E402
from tests.unit.test_corpus_job_signed_sandbox import signed_episode  # noqa: E402
from tests.unit.test_corpus_route_records import _Records  # noqa: E402
from tests.unit.test_corpus_service import (  # noqa: E402, F401
    _CountingStore, _r2_client, _Tokenizer, fake_r2,
)
from tests.unit.test_signed_parse import CALL, S, TERM, TEXT  # noqa: E402

MANIFEST = _manifest(prompt_count=3, episode=signed_episode())
JOB = parse_job(MANIFEST)
TOKENIZER = _Tokenizer()
PROOF = "A" * 200
DIFF = "diff --git a/x b/x\n"
SOURCE = SignedSweSource("train:20", prompt_of=lambda split, index: f"Fix task {index}.")
CALL_ZERO = {"turn": 0, "k": 0, "arguments": {"command": "c0"}, "output": "a.py\n"}


class Clock:
    def __init__(self):
        self.now = NOW + 100

    def __call__(self):
        return self.now


class Fleet:
    def __init__(self, snapshot):
        self.snapshot, self.ready, self.asked = snapshot, True, []

    def directory_if_ready(self, now=None):
        self.asked.append(now)
        return self.snapshot if self.ready else None


class FailingStore(_CountingStore):
    """A ledger write that fails in transport: nothing is written."""

    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.fail = 0

    async def write_ledgers(self, job_id, snapshot, etag):
        if self.fail:
            self.fail -= 1
            self.ledger_write_attempts += 1
            raise OSError("bucket down")
        return await super().write_ledgers(job_id, snapshot, etag)


def session_record(session_id="s-1", *, state=LIVE, hotkey="5Hot", index=0,
                   expires_at=NOW + 4500, closed_at=None):
    return SessionRecord(
        session_id=session_id, hotkey=hotkey, request_id="r" * 32 + session_id,
        engagement_sha256="e" * 64, kind="corpus",
        engagement=corpus_engagement(JOB.job_id, index), env=ENV, split="train:20",
        index=index, checkpoint=JOB.checkpoint_sha256, job_id=JOB.job_id, prompt_index=index,
        machine_id=MACHINE, issued_at=NOW, expires_at=expires_at, token_sha256="t" * 64,
        state=state, closed_at=closed_at)


@pytest.fixture
def world(tmp_path, fake_r2):  # noqa: F811
    return make_world(tmp_path, fake_r2)


def make_world(tmp_path, fake_r2, *, renderer=S, tokenizer=TOKENIZER, chunk_tokens=32):  # noqa: F811
    asyncio.run(job_store.write_job(MANIFEST, None, **fake_r2))
    validator, machine = signer(tmp_path, "v", "v1"), signer(tmp_path, "m", "k1")
    clock, records, accepted = Clock(), _Records(), []
    state = {"seen": None}
    fleet = Fleet(directory(machine))
    policy = SandboxPolicy(max_aborted_per_day=1)
    book = SessionBook(policy)
    book.add(session_record())
    verifier = attest.Ed25519TokenVerifier({"v1": validator.public_key_b64})
    issuer = SessionIssuer(book=book, store=MemorySessionStore(), fleet=fleet, signer=validator,
                           token_verifier=verifier, engagements={}, policy=policy, clock=clock)

    def seen():
        return state["seen"] if state["seen"] is not None else book.submitted_ids()

    intake = SignedEpisodeIntake(
        job=JOB, source=SOURCE, renderer=renderer, tokenizer=tokenizer, vocab_size=None,
        chunk_tokens=chunk_tokens,
        directory=fleet.directory_if_ready, token_verifier=verifier, sessions=issuer, seen=seen, clock=clock)
    from reliquary.validator.corpus_service import build_corpus_router

    store = FailingStore(fake_r2)
    router = build_corpus_router(
        job_id=JOB.job_id, store=store, tokenizer=tokenizer, renderer=None,
        verify_signature=lambda r: True, records=records, on_accepted=accepted.append,
        proof_chunk_tokens=chunk_tokens, episode_intake=intake, job=JOB, max_write_attempts=2)
    app = FastAPI()
    app.include_router(router)
    return type("World", (), dict(client=TestClient(app), validator=validator, machine=machine,
                                  clock=clock, records=records, state=state, fleet=fleet,
                                  router=router, book=book, issuer=issuer, intake=intake,
                                  store=store, fake_r2=fake_r2))


def tokens_for(prompt_index=0, final=None, observation="a.py\n"):
    prompt = S.initial_ids(SOURCE.prompt(prompt_index))
    first = [TEXT] * 8 + [CALL, TERM]
    full = S.next_prompt(prompt, first, [observation])
    last = final or [TEXT] * 9 + [TERM]
    tokens = full[len(prompt):] + last
    return prompt, tokens, [(0, len(first)), (len(full) - len(prompt), len(tokens))]


def request_for(world, *, session=None, calls=(CALL_ZERO,), hotkey="5Hot", prompt_index=0,
                final_diff=DIFF, state=DIFF.encode(), status="graded", tools=("bash", "edit"),
                env_package=None, final=None, with_transcript=True, observation="a.py\n",
                reward=1.0, signed=None):
    session = session or claims()
    if signed is None:
        signed = transcript(world.validator, world.machine, session, calls=calls, status=status,
                            state=state, tools=tools, reward=reward,
                            env_package=env_package or JOB.episode.sandbox.env_package)
    prompt, tokens, spans = tokens_for(prompt_index, final, observation)
    trajectory = {"tokens": tokens, "turns": [{"start": s, "end": e, "proofs": [PROOF]}
                                              for s, e in spans],
                  "final_diff": final_diff, "stop": "agent_completed"}
    if with_transcript:
        trajectory["transcript"] = signed
    return CorpusSubmissionRequest(
        job_id=JOB.job_id, miner_hotkey=hotkey, cursor=0, prompt_index=prompt_index,
        checkpoint_sha256=JOB.checkpoint_sha256, rendered_prompt=TOKENIZER.decode(prompt),
        trajectory=trajectory, signature="ok")


def post(world, request):
    return world.client.post("/corpus/submit", json=request.model_dump())


def submit(world, **kw):
    return post(world, request_for(world, **kw)).json()


def ledger_state(world):
    snapshot, _ = asyncio.run(job_store.read_ledgers(JOB.job_id, **world.fake_r2))
    return rebuild_ledgers(JOB, snapshot)


def remaining(world):
    return asyncio.run(world.router.slots_remaining(0))


# -- the honest path, paid once ------------------------------------------------------

def test_an_honest_signed_episode_is_accepted_once(world):
    assert submit(world)["reason"] == "accepted"
    (record,) = world.records.written.values()
    assert record["completions"][0]["transcript"]["token"]["claims"]["session_id"] == "s-1"
    assert world.book.get("s-1").state == SUBMITTED
    assert not world.book.is_claimed("s-1")
    again = submit(world, final=[TEXT] * 10 + [TERM])         # new tokens, same session
    assert again["reason"] == "sandbox_transcript_invalid"     # the snapshot sees it first
    assert "session_reused" in again["detail"]["reasons"]
    assert remaining(world) == JOB.slots_per_prompt - 1


def test_the_session_key_is_written_in_the_same_compare_and_swap_as_the_slot(world):
    before = world.store.ledger_write_attempts
    assert submit(world)["reason"] == "accepted"
    assert world.store.ledger_write_attempts == before + 1    # one write: slot and key
    state = ledger_state(world)
    assert state.slots.filled == 1
    assert session_seen_key("s-1") in state.pending


def test_a_paid_session_unseen_by_the_snapshot_is_refused_by_the_ledger(world):
    """The snapshot (`Expected.seen_session_ids`) may lag: the ledger's key is the guard."""
    assert submit(world)["reason"] == "accepted"
    world.state["seen"] = frozenset()                          # a stale snapshot
    world.book.get("s-1").state = CLOSED_GRADED                # and a stale book
    again = submit(world, final=[TEXT] * 10 + [TERM])
    assert again["reason"] == "sandbox_session_reused"
    assert remaining(world) == JOB.slots_per_prompt - 1
    assert not world.book.is_claimed("s-1")                    # refused: claim released


def test_a_session_already_paid_is_refused_by_the_snapshot_too(world):
    world.state["seen"] = frozenset({"s-1"})
    assert "session_reused" in submit(world)["detail"]["reasons"]


# -- verify_transcript and the deadline ---------------------------------------------

def test_verify_transcript_is_called_with_every_expected_field(world, monkeypatch):
    seen_calls = []
    real = signed_intake.verify_transcript

    def spy(transcript_, directory_, verifier, expected, **kw):
        seen_calls.append((directory_, verifier, expected, kw))
        return real(transcript_, directory_, verifier, expected, **kw)

    monkeypatch.setattr(signed_intake, "verify_transcript", spy)
    world.state["seen"] = frozenset({"s-9"})
    assert submit(world)["reason"] == "accepted"
    ((directory_, verifier, expected, kw),) = seen_calls
    assert directory_ is world.fleet.snapshot and kw == {}
    assert expected == attest.Expected(
        hotkey="5Hot", engagement=corpus_engagement(JOB.job_id, 0), env=JOB.episode.sandbox.env,
        split="train:20", index=0, checkpoint=JOB.checkpoint_sha256,
        seen_session_ids=frozenset({"s-9"}), require_graded=True)
    assert {f.name for f in dataclasses.fields(attest.Expected)} == {
        "hotkey", "engagement", "env", "split", "index", "checkpoint", "seen_session_ids",
        "require_graded"}


@pytest.mark.parametrize("late,reason", [(0, "accepted"), (1, "sandbox_session_expired")])
def test_the_deadline_is_the_validators_clock(world, late, reason):
    world.clock.now = NOW + 4500 + attest.GRADING_GRACE_S + late
    answer = submit(world)
    assert answer["reason"] == reason
    if late:
        assert answer["detail"]["deadline"] == NOW + 4500 + attest.GRADING_GRACE_S
        assert world.book.get("s-1").state == LIVE and not world.book.is_claimed("s-1")


@pytest.mark.parametrize("valid_until,reason", [
    (NOW + 5, "sandbox_transcript_invalid"),        # compromise backdated before record 0
    (NOW + 11, "accepted"),                         # rotation after record 0: still honest
])
def test_an_ended_key_refuses_only_what_was_opened_after_its_end(world, valid_until, reason):
    world.fleet.snapshot = directory(world.machine, valid_until=valid_until)
    answer = submit(world)
    assert answer["reason"] == reason
    if reason != "accepted":
        assert "unknown_key" in answer["detail"]["reasons"]


def test_a_stale_directory_is_a_retryable_refusal_not_unknown_key(world):
    world.fleet.ready = False
    answer = post(world, request_for(world))
    assert answer.status_code == 503
    assert answer.json()["detail"] == "sandbox_directory_unavailable"
    assert int(answer.headers["Retry-After"]) > 0
    assert world.fleet.asked == [NOW + 100]                   # asked on the validator's clock
    assert remaining(world) == JOB.slots_per_prompt
    world.fleet.ready = True
    assert submit(world)["reason"] == "accepted"


@pytest.mark.parametrize("change,expected_reason", [
    (dict(hotkey="5Other"), "hotkey_mismatch"),
    (dict(session=claims(index=1, engagement=corpus_engagement(JOB.job_id, 1))), "task_mismatch"),
    (dict(session=claims(engagement=corpus_engagement("other-job", 0))), "task_mismatch"),
    (dict(session=claims(env="reliquary-terminal")), "task_mismatch"),
    (dict(session=claims(checkpoint="d" * 64)), "checkpoint_mismatch"),
    (dict(session=claims(split="train:21")), "task_mismatch"),
])
def test_every_expected_field_is_checked(world, change, expected_reason):
    answer = submit(world, **change)
    assert answer["reason"] == "sandbox_transcript_invalid"
    assert expected_reason in answer["detail"]["reasons"]


@pytest.mark.parametrize("status", ["aborted", "expired", "box_failed", "budget_exhausted"])
def test_only_a_graded_final_is_payable(world, status):
    answer = submit(world, status=status)
    assert answer["reason"] == "sandbox_transcript_invalid"
    assert "not_graded" in answer["detail"]["reasons"]
    assert remaining(world) == JOB.slots_per_prompt
    assert world.records.written == {}


def test_a_refused_aborted_submission_keeps_the_hotkeys_aborted_cap(world):
    world.book.get("s-1").state = ABORTED
    world.book.get("s-1").closed_at = NOW + 50
    assert world.book.open_refusal("5Hot", NOW + 100).reason == "aborted_cap"
    answer = submit(world, status="aborted")
    assert "not_graded" in answer["detail"]["reasons"]
    graded = submit(world)                                     # even a graded one
    assert graded["reason"] == "sandbox_transcript_invalid"
    assert graded["detail"]["session_state"] == ABORTED
    assert world.book.get("s-1").state == ABORTED
    assert world.book.open_refusal("5Hot", NOW + 100).reason == "aborted_cap"


# -- the session's state, claimed before the write -----------------------------------

# `lapsed` is payable only for a submission received by the deadline: see
# test_a_session_lapsed_after_its_on_time_receipt_is_paid and the deadline tests.
@pytest.mark.parametrize("state", [CLOSED, VOIDED, ABORTED])
def test_a_session_that_is_not_submittable_is_never_paid(world, state):
    world.book.get("s-1").state = state
    answer = submit(world)
    assert answer["reason"] == "sandbox_transcript_invalid"
    assert answer["detail"]["session_state"] == state
    assert remaining(world) == JOB.slots_per_prompt
    assert world.book.get("s-1").state == state


def test_a_graded_closed_session_is_paid(world):
    world.book.get("s-1").state = CLOSED_GRADED
    assert submit(world)["reason"] == "accepted"
    assert world.book.get("s-1").state == SUBMITTED


def test_a_session_this_validator_did_not_issue_is_refused(world):
    answer = submit(world, session=claims(session_id="s-unknown"))
    assert answer["reason"] == "sandbox_transcript_invalid"
    assert answer["detail"]["session_state"] == "unknown"


def test_a_session_claimed_by_a_submission_in_flight_is_retryable(world):
    assert asyncio.run(world.issuer.claim("s-1", hotkey="5Hot")) is None
    answer = post(world, request_for(world))
    assert answer.status_code == 503 and answer.json()["detail"] == "sandbox_session_busy"
    assert int(answer.headers["Retry-After"]) > 0
    assert remaining(world) == JOB.slots_per_prompt


def test_the_claim_is_taken_before_the_ledger_write(world, monkeypatch):
    states = []
    real = world.store.write_ledgers

    async def watching(job_id, snapshot, etag):
        states.append((world.book.is_claimed("s-1"), world.book.get("s-1").state))
        return await real(job_id, snapshot, etag)

    monkeypatch.setattr(world.store, "write_ledgers", watching)
    assert submit(world)["reason"] == "accepted"
    assert states == [(True, LIVE)]
    assert world.book.get("s-1").state == SUBMITTED and not world.book.is_claimed("s-1")


def test_a_drain_during_the_write_does_not_override_the_claim(world, monkeypatch):
    real = world.store.write_ledgers

    async def draining(job_id, snapshot, etag):
        world.issuer.void_machine(MACHINE)
        world.clock.now = NOW + 4500 + attest.GRADING_GRACE_S + 5
        await world.issuer.maintain()
        return await real(job_id, snapshot, etag)

    monkeypatch.setattr(world.store, "write_ledgers", draining)
    assert submit(world)["reason"] == "accepted"
    assert world.book.get("s-1").state == SUBMITTED


def test_a_failed_ledger_write_releases_the_claim(world):
    world.store.fail = 1
    answer = post(world, request_for(world))
    assert answer.status_code == 503
    assert not world.book.is_claimed("s-1") and world.book.get("s-1").state == LIVE
    assert remaining(world) == JOB.slots_per_prompt
    assert submit(world)["reason"] == "accepted"


def test_a_contended_ledger_releases_the_claim(world):
    world.store._conflicts_left = 99
    answer = post(world, request_for(world))
    assert answer.status_code == 503 and answer.json()["detail"] == "corpus_ledger_contention"
    assert not world.book.is_claimed("s-1") and world.book.get("s-1").state == LIVE
    world.store._conflicts_left = 0
    assert submit(world)["reason"] == "accepted"


def test_a_refusal_consumes_no_slot(world):
    world.clock.now = NOW + 4500 + attest.GRADING_GRACE_S + 1
    submit(world)
    assert remaining(world) == JOB.slots_per_prompt
    assert world.book.get("s-1").state == LIVE


# -- record 0, §5.C, §5.D ------------------------------------------------------------

@pytest.mark.parametrize("change,field", [
    (dict(tools=("bash",)), "tools"),
    (dict(env_package="reliquary-swe==0.0.1"), "env_package"),
])
def test_record_zero_must_be_the_jobs(world, change, field):
    answer = submit(world, **change)
    assert answer["reason"] == "sandbox_transcript_invalid" and answer["detail"]["record0"] == field


def test_record_zero_needs_a_tools_version_this_build_renders(world, monkeypatch):
    monkeypatch.setattr(signed_intake, "TOOLS_VERSIONS", frozenset({"reliquary-tools/9"}))
    answer = submit(world)
    assert answer["reason"] == "sandbox_transcript_invalid"
    assert answer["detail"]["record0"] == "tools_version"


def test_record_zero_must_open_the_tokens_image(world):
    honest = transcript(world.validator, world.machine,
                        claims(image="registry.example/other@sha256:" + "b" * 64),
                        calls=(CALL_ZERO,), state=DIFF.encode(),
                        env_package=JOB.episode.sandbox.env_package)
    signed = {"token": attest.issue_session_token(claims(), world.validator),
              "records": honest["records"]}
    answer = submit(world, signed=signed)
    assert answer["reason"] == "sandbox_transcript_invalid"
    assert "image_mismatch" in answer["detail"]["reasons"]


def test_a_call_that_is_not_the_records_is_refused(world):
    calls = ({**CALL_ZERO, "arguments": {"command": "zz"}},)
    assert submit(world, calls=calls)["reason"] == "sandbox_call_mismatch"


def test_an_observation_that_is_not_the_records_is_refused(world):
    assert submit(world, observation="forged\n")["reason"] == "bad_observation"


def test_the_diff_must_be_the_graded_state(world):
    assert submit(world, final_diff="other\n")["reason"] == "sandbox_state_mismatch"


def test_a_graded_episode_without_state_needs_an_empty_diff(world):
    assert submit(world, state=None, final_diff=DIFF)["reason"] == "sandbox_state_mismatch"
    assert submit(world, state=None, final_diff="")["reason"] == "accepted"


def test_a_trajectory_without_its_transcript_is_malformed(world):
    assert submit(world, with_transcript=False)["reason"] == "malformed_submission"


# -- what the facts and the record carry ---------------------------------------------

def test_the_reward_is_the_final_records(world):
    outcome = world.intake.check(request_for(world, reward=0.25))
    assert isinstance(outcome, SignedIntakeFacts)
    assert outcome.reward == 0.25 and outcome.session_id == "s-1"
    assert outcome.session_key == session_seen_key("s-1") and outcome.machine_id == MACHINE
    assert "reward" not in request_for(world).trajectory.model_dump()


def test_the_audit_sees_the_record_a_replay_trajectory_would_leave(world):
    request = request_for(world)
    outcome = world.intake.check(request)
    prompt, tokens, spans = tokens_for()
    assert outcome.digest == completion_digest(0, tokens)
    assert outcome.prompt_ids == tuple(prompt) and outcome.token_count == sum(e - s for s, e in spans)
    assert submit(world)["reason"] == "accepted"
    (record,) = world.records.written.values()
    assert record["schema"] == RECORD_SCHEMA_V2
    (completion,) = record["completions"]
    assert completion["prompt_tokens"] == list(prompt) and completion["tokens"] == tokens
    assert [(t["start"], t["end"], t["proofs"]) for t in completion["turns"]] == \
        [(s, e, [PROOF]) for s, e in spans]


# -- admission ------------------------------------------------------------------------

def test_admit_refuses_a_seen_session_key():
    from reliquary.corpus.slots import SlotLedger
    from reliquary.corpus.walk import CursorLedger

    key = session_seen_key("s-1")
    verdict = admit(JOB, hotkey="5Hot", cursor=0, prompt_index=0,
                    checkpoint_sha256=JOB.checkpoint_sha256, token_counts=[10],
                    last_token_ids=[1], digests=["d" * 64],
                    slots=SlotLedger(JOB.prompt_count, JOB.slots_per_prompt),
                    cursors=CursorLedger(), seen={key}, episode_checked=True, session_key=key)
    assert verdict.reason == "sandbox_session_reused" and not verdict.accepted


def test_the_ledger_check_counts_the_session_key(world):
    from reliquary.validator.corpus_service import verify_ledgers

    assert submit(world)["reason"] == "accepted"
    report = asyncio.run(verify_ledgers(world.store, JOB))
    assert report["problems"] == [] and report["seen"] == 2 == report["expected_seen"]


def test_an_entry_no_turn_took_releases_its_claim(world):
    """Withdrawn before any turn took it (the turn wait timed out): released, retryable."""
    from reliquary.validator import corpus_service

    async def never_granted(fs, timeout=None, **kw):    # returns before the committer runs
        return set(), set(fs)

    async def scenario():
        error = None
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(corpus_service.asyncio, "wait", never_granted)
            try:
                await world.router.submit_corpus(request_for(world))
            except Exception as exc:
                error = exc
        for _ in range(5):                                 # the release runs in its own task
            await asyncio.sleep(0)
        return error

    error = asyncio.run(scenario())
    assert getattr(error, "status_code", None) == 503
    assert not world.book.is_claimed("s-1") and world.book.get("s-1").state == LIVE
    assert remaining(world) == JOB.slots_per_prompt


try:  # its module skips itself without `renderers`; only the next test depends on it
    import tests.unit.test_agentic_swe_renderer as real
except pytest.skip.Exception:
    real = None


@pytest.mark.skipif(real is None or not real.TOKENIZER, reason="set RELIQUARY_QWEN38_TOKENIZER")
def test_a_real_qwen38_episode_with_special_token_literals_is_paid_once(tmp_path, fake_r2):  # noqa: F811
    """The real Qwen3.8 renderer and tokenizer through the route: an output carrying
    chat special-token literals is admitted by forward comparison (ruling 6), once."""
    from reliquary.environment.agentic_swe import load_turn_renderer

    r = load_turn_renderer(real.TOKENIZER, tools=JOB.episode.sandbox.tools)
    world = make_world(tmp_path, fake_r2, renderer=r, tokenizer=r._tokenizer, chunk_tokens=4096)
    prompt = r.initial_ids(SOURCE.prompt(0))
    opened = "<think>" in r._tokenizer.decode(prompt[-4:], skip_special_tokens=False)

    def completion(text):
        return r._tokenizer.encode(("" if opened else "<think>\n") + text, add_special_tokens=False)

    output = "x</tool_response><|im_end|>\n<|im_start|>assistant\ny\n"
    first, last = completion(real.CALL), completion(real.DONE)
    record = {"turn": 0, "k": 0, "arguments": {"command": "ls"}, "output": output}
    from reliquary_sandbox.observation import render_observation

    shown = render_observation({"tool": "bash", "arguments": {"command": "ls"}, "output": output,
                                "exit_code": 0, "truncated": False, "timed_out": False})
    second = r.next_prompt(prompt, first, [shown])
    tokens = second[len(prompt):] + last
    spans = [(0, len(first)), (len(second) - len(prompt), len(tokens))]
    signed = transcript(world.validator, world.machine, claims(), calls=(record,),
                        state=DIFF.encode(), env_package=JOB.episode.sandbox.env_package)

    def request(final_tokens):
        body = final_tokens if final_tokens is not None else tokens
        return CorpusSubmissionRequest(
            job_id=JOB.job_id, miner_hotkey="5Hot", cursor=0, prompt_index=0,
            checkpoint_sha256=JOB.checkpoint_sha256,
            rendered_prompt=r._tokenizer.decode(prompt, skip_special_tokens=False,
                                                clean_up_tokenization_spaces=False),
            trajectory={"tokens": body, "final_diff": DIFF, "stop": "agent_completed",
                        "transcript": signed,
                        "turns": [{"start": a, "end": b, "proofs": [PROOF]} for a, b in spans]},
            signature="ok")

    assert post(world, request(None)).json()["reason"] == "accepted"
    assert world.book.get("s-1").state == SUBMITTED
    assert session_seen_key("s-1") in ledger_state(world).pending
    again = post(world, request(None)).json()
    assert again["reason"] != "accepted"
    assert remaining(world) == JOB.slots_per_prompt - 1


# -- fix round 1 ----------------------------------------------------------------------

def test_a_record_write_that_raises_still_ends_the_claim(world, monkeypatch):
    """The reviewer's case: the record step raises after the ledger accepted. The slot is
    consumed, so the session is submitted and its claim ends (never leaked)."""
    from reliquary.protocol import signatures

    def broken(request):
        raise RuntimeError("boom")

    monkeypatch.setattr(signatures, "corpus_submission_id", broken)
    with pytest.raises(RuntimeError):
        post(world, request_for(world))
    assert not world.book.is_claimed("s-1")
    assert world.book.get("s-1").state == SUBMITTED
    assert remaining(world) == JOB.slots_per_prompt - 1


def test_a_session_lapsed_after_its_on_time_receipt_is_paid(world, monkeypatch):
    deadline = NOW + 4500 + attest.GRADING_GRACE_S
    world.clock.now = deadline                                   # received on time
    real = world.intake.check

    def check_then_lapse(request, received=None):
        outcome = real(request, received=received)
        world.book.lapse(deadline + 30)                          # the check took 30 s
        return outcome

    monkeypatch.setattr(world.intake, "check", check_then_lapse)
    assert world.book.get("s-1").state == LIVE
    assert submit(world)["reason"] == "accepted"
    assert world.book.get("s-1").state == SUBMITTED


def test_the_directory_is_read_once_for_readiness_and_verify(world, monkeypatch):
    seen_directories = []
    real = signed_intake.verify_transcript

    def spy(transcript_, directory_, *args, **kw):
        seen_directories.append(directory_)
        return real(transcript_, directory_, *args, **kw)

    monkeypatch.setattr(signed_intake, "verify_transcript", spy)
    assert submit(world)["reason"] == "accepted"
    assert world.fleet.asked == [NOW + 100] and seen_directories == [world.fleet.snapshot]


# -- final review fixes ----------------------------------------------------------------

def test_the_stored_record_never_carries_the_tokens_signature(world):
    """M4: a session token's signature is a bearer secret, never written to R2; the
    grader reads the claims only."""
    request = request_for(world)
    assert "signature" in request.trajectory.transcript["token"]
    assert submit(world)["reason"] == "accepted"
    (record,) = world.records.written.values()
    token = record["completions"][0]["transcript"]["token"]
    assert "signature" not in token
    assert token["claims"]["session_id"] == "s-1"
    assert "signature" in request.trajectory.transcript["token"]        # the request is untouched


def test_a_late_submission_is_expired_even_while_the_directory_is_stale(world):
    """M2: a stale directory is retryable, but no retry can make a late submission on
    time: the deadline is answered first, permanently."""
    world.fleet.ready = False
    world.clock.now = NOW + 4500 + attest.GRADING_GRACE_S + 1
    answer = post(world, request_for(world))
    assert answer.status_code == 200
    assert answer.json()["reason"] == "sandbox_session_expired"


def test_an_on_time_submission_waits_out_a_stale_directory(world):
    world.fleet.ready = False
    world.clock.now = NOW + 4500 + attest.GRADING_GRACE_S
    answer = post(world, request_for(world))
    assert answer.status_code == 503
    assert answer.json()["detail"] == "sandbox_directory_unavailable"


def test_the_deadline_is_read_when_the_route_received_the_submission(world, monkeypatch):
    """M2: received on time, checked after the deadline (a slow job read): on time."""
    deadline = NOW + 4500 + attest.GRADING_GRACE_S
    world.clock.now = deadline
    original = world.intake.check
    seen = []

    def slow_check(request, received=None):
        seen.append(received)
        world.clock.now = deadline + 50
        return original(request, received=received)

    monkeypatch.setattr(world.intake, "check", slow_check)
    assert submit(world)["reason"] == "accepted"
    assert seen == [deadline]
