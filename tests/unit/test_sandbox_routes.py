"""The session routes: fresh, signed, registered requests only; refusals mapped to
statuses a miner can act on; the token goes to the miner and nowhere else."""

import json
import math
import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytest.importorskip("reliquary_sandbox.attest")

from reliquary.protocol.sandbox_session import (  # noqa: E402
    SandboxSessionCloseRequest, SandboxSessionOpenRequest, sandbox_close_path, sandbox_open_path,
)
from reliquary.protocol.signatures import (  # noqa: E402
    build_sandbox_close_binding, build_sandbox_open_binding, verify_sandbox_close_signature,
    verify_sandbox_open_signature,
)
from reliquary.sandbox import routes, sessions  # noqa: E402
from reliquary.sandbox.routes import (  # noqa: E402
    MAX_CLOSE_BODY_BYTES, MAX_OPEN_BODY_BYTES, REFUSAL_STATUS, build_sandbox_sessions_router,
)
from reliquary.sandbox.sessions import (  # noqa: E402
    ABORTED, Grant, Refusal, SandboxPolicy, SessionBook, SessionRecord,
)
from tests.unit.sandbox_fixtures import NOW  # noqa: E402

ALICE = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"          # //Alice, ss58 format 42
ALICE_FORMAT_0 = "15oF4uVJwmo4TdGW7VfQxNLavjCXviqxT9S1MgbjMNHr6Sp5"  # the same key, format 0
VALIDATOR = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"      # //Bob
OTHER_VALIDATOR = "5FLSigC9HGRKVhB9FiEo4Y3koPsNmBmLJbpXg2mp1hXcS59Y"  # //Charlie
OPEN, CLOSE = sandbox_open_path("/corpus"), sandbox_close_path("/corpus", "s-1")

TOKEN = {"v": 1, "claims": {"session_id": "s-1"}, "key_id": "v1", "signature": "SECRET-SIG"}
SECRET = "SECRET-SIG"


@pytest.fixture(autouse=True)
def _route_logs_reach_caplog():
    """Importing bittensor sets every logger that exists then to CRITICAL; without this,
    a log assertion here would pass on an empty capture."""
    saved = [(lg, lg.level) for lg in (routes.logger, sessions.logger)]
    for lg, _ in saved:
        lg.setLevel("DEBUG")
    yield
    for lg, level in saved:
        lg.setLevel(level)


class FakeIssuer:
    def __init__(self, outcome):
        self.outcome, self.calls = outcome, []

    async def open(self, **kw):
        self.calls.append(("open", kw))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    async def close(self, **kw):
        self.calls.append(("close", kw))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def accept(request, **audience):
    return True


def client(issuer, *, verify=accept, registration=None, now=NOW):
    app = FastAPI()
    app.include_router(build_sandbox_sessions_router(
        issuer, policy=SandboxPolicy(), validator_hotkey=VALIDATOR, verify_open=verify,
        verify_close=verify, registration=registration, clock=lambda: now))
    return TestClient(app, raise_server_exceptions=False)


def open_body(**overrides):
    body = {"miner_hotkey": ALICE, "request_id": "a" * 32, "at": NOW,
            "engagement": {"kind": "corpus", "job_id": "swe-agentic-v1", "prompt_index": 3},
            "signature": "00"}
    body.update(overrides)
    return body


def close_body(**overrides):
    body = {"miner_hotkey": ALICE, "request_id": "b" * 32, "at": NOW, "session_id": "s-1",
            "reason": "final", "transcript": {"token": {}, "records": []}, "signature": "00"}
    body.update(overrides)
    return body


def test_a_grant_answers_the_token(caplog):
    issuer = FakeIssuer(Grant("s-1", TOKEN, "http://10.0.0.5:8080", NOW + 4500))
    caplog.set_level("DEBUG")
    answer = client(issuer).post("/corpus/sandbox/sessions", json=open_body())
    assert answer.status_code == 200
    assert answer.json() == {"session_id": "s-1", "token": TOKEN,
                             "gateway_url": "http://10.0.0.5:8080", "expires_at": NOW + 4500}
    assert issuer.calls == [("open", {"hotkey": ALICE, "request_id": "a" * 32, "engagement": {
        "kind": "corpus", "job_id": "swe-agentic-v1", "prompt_index": 3}})]
    assert "s-1" in caplog.text                       # the capture is real
    assert SECRET not in caplog.text


def test_the_token_is_in_the_answer_exactly_once_and_never_cached():
    issuer = FakeIssuer(Grant("s-1", TOKEN, "http://10.0.0.5:8080", NOW + 4500))
    answer = client(issuer).post("/corpus/sandbox/sessions", json=open_body())
    assert answer.text.count(SECRET) == 1
    assert all(SECRET not in value for value in answer.headers.values())
    assert answer.headers["Cache-Control"] == "no-store"


def test_an_unsigned_request_never_reaches_the_issuer():
    issuer = FakeIssuer(None)
    answer = client(issuer, verify=lambda request, **audience: False).post("/corpus/sandbox/sessions",
                                                                json=open_body())
    assert answer.status_code == 403 and answer.json()["reason"] == "bad_signature"
    assert issuer.calls == []


def test_an_unsigned_close_never_reaches_the_issuer():
    issuer = FakeIssuer(None)
    answer = client(issuer, verify=lambda request, **audience: False).post(
        "/corpus/sandbox/sessions/s-1/close", json=close_body())
    assert answer.status_code == 403 and answer.json()["reason"] == "bad_signature"
    assert issuer.calls == []


def test_a_verifier_that_raises_is_a_bad_signature():
    def broken(request, **audience):
        raise RuntimeError("no bittensor")

    answer = client(FakeIssuer(None), verify=broken).post("/corpus/sandbox/sessions", json=open_body())
    assert answer.status_code == 403 and "no bittensor" not in answer.text


@pytest.mark.parametrize("at", [NOW - 121, NOW + 121])
def test_a_stale_request_is_refused(at):
    issuer = FakeIssuer(None)
    answer = client(issuer).post("/corpus/sandbox/sessions", json=open_body(at=at))
    assert answer.status_code == 400 and answer.json()["reason"] == "stale_request"
    assert issuer.calls == []


@pytest.mark.parametrize("at", [NOW - 120, NOW + 120])
def test_a_request_within_the_skew_is_fresh(at):
    issuer = FakeIssuer(Grant("s-1", TOKEN, "http://10.0.0.5:8080", NOW + 4500))
    assert client(issuer).post("/corpus/sandbox/sessions", json=open_body(at=at)).status_code == 200


def test_an_unregistered_hotkey_is_refused():
    issuer = FakeIssuer(None)

    async def registration(hotkey):
        return "not_registered"

    answer = client(issuer, registration=registration).post(
        "/corpus/sandbox/sessions", json=open_body())
    assert answer.status_code == 403 and answer.json()["reason"] == "hotkey_not_registered"
    assert issuer.calls == []


def test_a_registered_hotkey_reaches_the_issuer():
    issuer = FakeIssuer(Grant("s-1", TOKEN, "http://10.0.0.5:8080", NOW + 4500))
    seen = []

    async def registration(hotkey):
        seen.append(hotkey)
        return None

    assert client(issuer, registration=registration).post(
        "/corpus/sandbox/sessions", json=open_body()).status_code == 200
    assert seen == [ALICE] and len(issuer.calls) == 1


@pytest.mark.parametrize("why", ["unavailable", RuntimeError("chain down")])
def test_an_unknown_registration_is_retried_not_refused(why):
    async def registration(hotkey):
        if isinstance(why, Exception):
            raise why
        return why

    answer = client(FakeIssuer(None), registration=registration).post(
        "/corpus/sandbox/sessions", json=open_body())
    assert answer.status_code == 503 and answer.json()["reason"] == "registration_unavailable"
    assert answer.headers["Retry-After"] == "10" and "chain down" not in answer.text


@pytest.mark.parametrize("refusal,status", [
    (Refusal("prompt_unavailable", {"prompt_index": 3}), 409),
    (Refusal("job_not_served"), 404),
    (Refusal("live_cap"), 429), (Refusal("aborted_cap"), 429),
    (Refusal("prompt_live_cap"), 429), (Refusal("job_live_cap"), 429),
    (Refusal("request_conflict"), 409), (Refusal("miner_banned"), 403),
    (Refusal("engagement_kind_unsupported"), 409),
])
def test_refusals_map_to_statuses(refusal, status):
    answer = client(FakeIssuer(refusal)).post("/corpus/sandbox/sessions", json=open_body())
    assert answer.status_code == status and answer.json()["reason"] == refusal.reason


def test_a_full_fleet_answers_retry_after():
    answer = client(FakeIssuer(Refusal("sandbox_capacity", retry_after=10))).post(
        "/corpus/sandbox/sessions", json=open_body())
    assert answer.status_code == 503 and answer.headers["Retry-After"] == "10"


@pytest.mark.parametrize("path,body", [("/corpus/sandbox/sessions", open_body()),
                                       ("/corpus/sandbox/sessions/s-1/close", close_body())])
def test_a_stale_directory_answers_retry_after(path, body):
    refusal = Refusal("directory_unavailable", {"why": "the machine directory is stale"},
                      retry_after=10)
    answer = client(FakeIssuer(refusal)).post(path, json=body)
    assert answer.status_code == 503 and answer.headers["Retry-After"] == "10"
    assert answer.json()["reason"] == "directory_unavailable"


def test_every_refusal_the_sessions_emit_has_a_status():
    source = Path(sessions.__file__).read_text()
    emitted = set(re.findall(r'Refusal\(\s*"([a-z_]+)"', source))
    assert emitted and emitted <= set(REFUSAL_STATUS), emitted - set(REFUSAL_STATUS)
    assert all(code in (400, 403, 404, 408, 409, 413, 422, 429, 500, 503)
               for code in REFUSAL_STATUS.values())
    assert all(REFUSAL_STATUS[reason] == 503 for reason in (
        "sandbox_capacity", "directory_unavailable", "store_unavailable", "ledger_unavailable",
        "task_unavailable", "registration_unavailable", "close_busy", "job_not_ready"))


def test_an_unknown_engagement_kind_is_malformed():
    issuer = FakeIssuer(None)
    body = open_body(engagement={"kind": "batch"})
    answer = client(issuer).post("/corpus/sandbox/sessions", json=body)
    assert answer.status_code == 422 and answer.json()["reason"] == "malformed_request"
    assert issuer.calls == []


@pytest.mark.parametrize("raw", [b"{not json", b"[]", b'{"at": NaN}', "é".encode("latin-1")])
def test_a_body_that_is_not_a_json_object_is_malformed(raw):
    answer = client(FakeIssuer(None)).post("/corpus/sandbox/sessions", content=raw,
                                           headers={"content-type": "application/json"})
    assert answer.status_code == 422 and answer.json()["reason"] == "malformed_request"


def test_a_malformed_close_never_echoes_its_transcript(caplog):
    caplog.set_level("DEBUG")
    transcript = {"token": {"signature": SECRET}, "records": []}
    body = close_body(transcript=transcript, reason="nonsense")
    answer = client(FakeIssuer(None)).post("/corpus/sandbox/sessions/s-1/close", json=body)
    assert answer.status_code == 422 and answer.json()["reason"] == "malformed_request"
    assert SECRET not in answer.text and SECRET not in caplog.text
    assert answer.json()["detail"]["errors"][0]["loc"] == ["reason"]


def test_a_refused_close_never_echoes_its_transcript(caplog):
    caplog.set_level("DEBUG")
    transcript = {"token": {"signature": SECRET}, "records": []}
    for issuer, verify in [(FakeIssuer(Refusal("transcript_invalid", {"reasons": ["bad"]})), True),
                           (FakeIssuer(None), False)]:
        answer = client(issuer, verify=lambda request, v=verify, **audience: v).post(
            "/corpus/sandbox/sessions/s-1/close", json=close_body(transcript=transcript))
        assert answer.status_code in (403, 409)
        assert SECRET not in answer.text
    assert SECRET not in caplog.text


@pytest.mark.parametrize("path,cap", [("/corpus/sandbox/sessions", MAX_OPEN_BODY_BYTES),
                                      ("/corpus/sandbox/sessions/s-1/close", MAX_CLOSE_BODY_BYTES)])
def test_a_body_over_its_bound_is_refused_before_parsing(path, cap):
    issuer = FakeIssuer(None)
    answer = client(issuer).post(path, content=b"{" + b" " * cap + b"}",
                                 headers={"content-type": "application/json"})
    assert answer.status_code == 413 and answer.json()["reason"] == "body_too_large"
    assert issuer.calls == []


def test_a_chunked_body_over_its_bound_is_refused():
    issuer = FakeIssuer(None)

    def chunks():
        for _ in range(MAX_OPEN_BODY_BYTES // 1024 + 2):
            yield b" " * 1024

    answer = client(issuer).post("/corpus/sandbox/sessions", content=chunks(),
                                 headers={"content-type": "application/json"})
    assert answer.status_code == 413 and issuer.calls == []


def test_a_transcript_over_the_wire_cap_is_malformed():
    with pytest.raises(ValueError):
        SandboxSessionCloseRequest(**close_body(transcript={"x": "y" * (8 * 1024 * 1024)}))


def test_an_issuer_failure_answers_500_without_a_trace(caplog):
    caplog.set_level("DEBUG")
    issuer = FakeIssuer(RuntimeError(f"boom {SECRET}"))
    for path, body in [("/corpus/sandbox/sessions", open_body()),
                       ("/corpus/sandbox/sessions/s-1/close", close_body())]:
        answer = client(issuer).post(path, json=body)
        assert answer.status_code == 500 and answer.json() == {"reason": "internal_error",
                                                               "detail": {}}
    assert "Traceback" not in caplog.text and SECRET not in caplog.text
    assert "RuntimeError" in caplog.text


def test_a_close_reaches_the_issuer_with_its_transcript():
    issuer = FakeIssuer({"session_id": "s-1", "state": "closed", "status": "expired"})
    answer = client(issuer).post("/corpus/sandbox/sessions/s-1/close", json=close_body())
    assert answer.status_code == 200 and answer.json()["state"] == "closed"
    assert issuer.calls[0][1] == {"hotkey": ALICE, "session_id": "s-1", "reason": "final",
                                  "transcript": {"token": {}, "records": []}}


def test_a_close_for_another_session_is_unknown():
    answer = client(FakeIssuer(None)).post("/corpus/sandbox/sessions/s-2/close", json=close_body())
    assert answer.status_code == 404 and answer.json()["reason"] == "session_unknown"


def opened(body, validator=VALIDATOR, path=OPEN):
    return build_sandbox_open_binding(body, validator_hotkey=validator, path=path)


def closed(body, validator=VALIDATOR, path=CLOSE):
    return build_sandbox_close_binding(body, validator_hotkey=validator, path=path)


def test_bindings_ignore_absent_fields_and_bind_everything_else():
    model = SandboxSessionOpenRequest(**open_body())
    assert opened(model) == opened(open_body())
    other = open_body(engagement={"kind": "corpus", "job_id": "swe-agentic-v1", "prompt_index": 4})
    assert opened(other) != opened(open_body())
    for field, value in [("miner_hotkey", VALIDATOR), ("request_id", "c" * 32), ("at", NOW + 1)]:
        assert opened(open_body(**{field: value})) != opened(open_body())
        assert closed(close_body(**{field: value})) != closed(close_body())
    one = closed(close_body())
    two = closed(close_body(transcript={"token": {}, "records": [1]}))
    assert one != two != closed(close_body(transcript=None))
    assert one != closed(close_body(session_id="s-2"))
    assert one != closed(close_body(reason="open_failed"))
    assert SandboxSessionCloseRequest(**close_body()).session_id == "s-1"
    assert opened(open_body()) != closed(close_body())


def test_bindings_name_their_validator_and_path():
    assert opened(open_body()) != opened(open_body(), validator=OTHER_VALIDATOR)
    assert opened(open_body()) != opened(open_body(), path=sandbox_open_path("/rl"))
    assert closed(close_body()) != closed(close_body(), validator=OTHER_VALIDATOR)
    assert closed(close_body()) != closed(close_body(), path=sandbox_close_path("/corpus", "s-2"))
    assert OPEN == "/corpus/sandbox/sessions" and CLOSE == "/corpus/sandbox/sessions/s-1/close"


def test_a_real_hotkey_signature_verifies_and_binds_the_engagement():
    bt = pytest.importorskip("bittensor")
    keypair = bt.Keypair.create_from_uri("//Alice")
    audience = {"validator_hotkey": VALIDATOR, "path": OPEN}
    body = open_body()
    body["signature"] = keypair.sign(opened(body)).hex()
    assert verify_sandbox_open_signature(SandboxSessionOpenRequest(**body), **audience)
    prefixed = {**body, "signature": "0x" + body["signature"]}
    assert verify_sandbox_open_signature(SandboxSessionOpenRequest(**prefixed), **audience)
    tampered = {**body, "engagement": {**body["engagement"], "prompt_index": 4}}
    assert not verify_sandbox_open_signature(SandboxSessionOpenRequest(**tampered), **audience)
    close = close_body(signature=body["signature"])
    assert not verify_sandbox_close_signature(SandboxSessionCloseRequest(**close),
                                              validator_hotkey=VALIDATOR, path=CLOSE)
    bob = bt.Keypair.create_from_uri("//Bob")
    forged = {**body, "signature": bob.sign(opened(body)).hex()}
    assert not verify_sandbox_open_signature(SandboxSessionOpenRequest(**forged), **audience)
    assert not verify_sandbox_open_signature(SandboxSessionOpenRequest(**{**body, "signature": "zz"}),
                                             **audience)


def real_router(issuer, validator=VALIDATOR, registration=None):
    app = FastAPI()
    app.include_router(build_sandbox_sessions_router(issuer, policy=SandboxPolicy(),
                                                     validator_hotkey=validator,
                                                     registration=registration,
                                                     clock=lambda: NOW))
    return TestClient(app, raise_server_exceptions=False)


def test_the_default_router_checks_real_signatures():
    bt = pytest.importorskip("bittensor")
    keypair = bt.Keypair.create_from_uri("//Alice")
    issuer = FakeIssuer(Grant("s-1", TOKEN, "http://10.0.0.5:8080", NOW + 4500))
    http = real_router(issuer)
    body = open_body()
    assert http.post(OPEN, json=body).status_code == 403
    body["signature"] = keypair.sign(opened(body)).hex()
    assert http.post(OPEN, json=body).status_code == 200
    assert {route.path for route in build_sandbox_sessions_router(
        issuer, policy=SandboxPolicy(), validator_hotkey=VALIDATOR, prefix="/rl").routes} == {
        "/rl/sandbox/sessions", "/rl/sandbox/sessions/{session_id}/close"}


def test_an_open_signed_for_another_validator_or_path_is_refused():
    bt = pytest.importorskip("bittensor")
    keypair = bt.Keypair.create_from_uri("//Alice")
    issuer = FakeIssuer(Grant("s-1", TOKEN, "http://10.0.0.5:8080", NOW + 4500))
    for validator, path in [(OTHER_VALIDATOR, OPEN), (VALIDATOR, sandbox_open_path("/rl"))]:
        body = open_body()
        body["signature"] = keypair.sign(opened(body, validator=validator, path=path)).hex()
        answer = real_router(issuer).post(OPEN, json=body)
        assert answer.status_code == 403 and answer.json()["reason"] == "bad_signature"
    assert issuer.calls == []
    close = close_body()
    close["signature"] = keypair.sign(closed(close, path=sandbox_close_path("/corpus", "s-2"))).hex()
    assert real_router(issuer).post(CLOSE, json=close).status_code == 403
    close["signature"] = keypair.sign(closed(close)).hex()
    issuer.outcome = {"session_id": "s-1", "state": "closed", "status": "expired"}
    assert real_router(issuer).post(CLOSE, json=close).status_code == 200


def test_the_router_refuses_a_validator_hotkey_that_is_not_ss58():
    with pytest.raises(ValueError):
        build_sandbox_sessions_router(FakeIssuer(None), policy=SandboxPolicy(),
                                      validator_hotkey="5Hot")


def test_a_router_without_registration_warns(caplog):
    caplog.set_level("WARNING")
    build_sandbox_sessions_router(FakeIssuer(None), policy=SandboxPolicy(),
                                  validator_hotkey=VALIDATOR)
    assert "registration" in caplog.text


def test_the_hotkey_is_normalised_to_ss58_format_42():
    bt = pytest.importorskip("bittensor")
    keypair = bt.Keypair.create_from_uri("//Alice")
    issuer = FakeIssuer(Grant("s-1", TOKEN, "http://10.0.0.5:8080", NOW + 4500))
    seen = []

    async def registration(hotkey):
        seen.append(hotkey)
        return None

    body = open_body(miner_hotkey=ALICE_FORMAT_0)
    body["signature"] = keypair.sign(opened(body)).hex()
    assert real_router(issuer, registration=registration).post(OPEN, json=body).status_code == 200
    assert issuer.calls[0][1]["hotkey"] == ALICE and seen == [ALICE]


@pytest.mark.parametrize("hotkey", ["5Hot", "not-an-address", "1" * 48])
def test_a_hotkey_that_is_not_ss58_is_malformed(hotkey):
    issuer = FakeIssuer(None)
    answer = client(issuer).post(OPEN, json=open_body(miner_hotkey=hotkey))
    assert answer.status_code == 422 and answer.json()["reason"] == "malformed_request"
    assert answer.json()["detail"]["errors"][0]["loc"] == ["miner_hotkey"]
    assert issuer.calls == []


def test_a_deeply_nested_body_is_malformed():
    for raw in [b"[" * 15000, b'{"engagement": ' + b"[" * 15000 + b"]" * 15000 + b"}"]:
        answer = client(FakeIssuer(None)).post(OPEN, content=raw,
                                               headers={"content-type": "application/json"})
        assert answer.status_code in (413, 422)
    answer = client(FakeIssuer(None)).post(OPEN, content=b"[" * 15000,
                                           headers={"content-type": "application/json"})
    assert answer.status_code == 422 and answer.json()["reason"] == "malformed_request"
    deep = b'{"transcript": ' + b"[" * 15000 + b"]" * 15000 + b"}"
    answer = client(FakeIssuer(None)).post(CLOSE, content=deep,
                                           headers={"content-type": "application/json"})
    assert answer.status_code == 422 and answer.json()["reason"] == "malformed_request"


def test_a_stale_refusal_names_the_validator_clock():
    answer = client(FakeIssuer(None)).post(OPEN, json=open_body(at=NOW - 500))
    assert answer.json()["detail"] == {"max_skew_s": 120, "now": NOW}


@pytest.mark.parametrize("field,value", [("at", str(NOW)), ("at", float(NOW)), ("at", True)])
def test_integers_are_strict(field, value):
    with pytest.raises(ValueError):
        SandboxSessionOpenRequest.model_validate_json(json.dumps(open_body(**{field: value})))
    with pytest.raises(ValueError):
        SandboxSessionCloseRequest.model_validate_json(json.dumps(close_body(**{field: value})))


@pytest.mark.parametrize("value", ["3", 3.0, True])
def test_the_prompt_index_is_strict(value):
    body = open_body(engagement={"kind": "corpus", "job_id": "j", "prompt_index": value})
    with pytest.raises(ValueError):
        SandboxSessionOpenRequest.model_validate_json(json.dumps(body))
    assert client(FakeIssuer(None)).post(OPEN, json=body).status_code == 422


THROTTLED = sorted(reason for reason, status in REFUSAL_STATUS.items() if status in (429, 503))


@pytest.mark.parametrize("reason", THROTTLED)
def test_every_429_and_503_carries_retry_after(reason):
    answer = client(FakeIssuer(Refusal(reason))).post(OPEN, json=open_body())
    assert answer.status_code in (429, 503)
    assert answer.headers["Retry-After"] == str(SandboxPolicy().retry_after_s)


@pytest.mark.parametrize("given,header", [(2.2, "3"), (0, "1"), (-5, "1"), (3600, "3600")])
def test_retry_after_rounds_up_to_at_least_a_second(given, header):
    answer = client(FakeIssuer(Refusal("open_rate_cap", retry_after=given))).post(
        OPEN, json=open_body())
    assert answer.status_code == 429 and answer.headers["Retry-After"] == header


def test_no_other_status_carries_retry_after():
    for refusal in (Refusal("prompt_unavailable"), Refusal("job_not_served")):
        assert "Retry-After" not in client(FakeIssuer(refusal)).post(OPEN, json=open_body()).headers


def record(n, *, issued_at, state="submitted", closed_at=None):
    return SessionRecord(
        session_id=f"s-{n}", hotkey=ALICE, request_id=f"{n:032x}", engagement_sha256="e",
        kind="corpus", engagement="corpus:j:1", env="swe", split="train:1", index=1,
        checkpoint="c", job_id="j", prompt_index=n, machine_id="m", issued_at=issued_at,
        expires_at=issued_at + 100, token_sha256="t", state=state, closed_status=None,
        closed_at=closed_at)


def test_the_open_rate_cap_retries_when_the_oldest_counted_open_leaves_the_hour():
    policy = SandboxPolicy(max_opens_per_hour=3)
    book = SessionBook(policy)
    book.restore([record(n, issued_at=NOW - 3000 + 10 * n) for n in range(3)])
    refusal = book.open_refusal(ALICE, NOW)
    assert refusal.reason == "open_rate_cap" and refusal.retry_after == 600   # 3600 - 3000


def test_the_aborted_cap_retries_when_the_oldest_counted_abort_leaves_the_day():
    policy = SandboxPolicy(max_aborted_per_day=2)
    book = SessionBook(policy)
    book.restore([record(n, issued_at=NOW - 80000, state=ABORTED, closed_at=NOW - 86000 + n)
                  for n in range(2)])
    refusal = book.open_refusal(ALICE, NOW)
    assert refusal.reason == "aborted_cap" and refusal.retry_after == math.ceil(400)


def test_the_wire_models_refuse_unknown_fields():
    with pytest.raises(ValueError):
        SandboxSessionOpenRequest(**open_body(extra=1))
    with pytest.raises(ValueError):
        SandboxSessionOpenRequest(**open_body(request_id="A" * 32))
    assert json.loads(SandboxSessionOpenRequest(**open_body()).model_dump_json())["at"] == NOW


def test_closes_beyond_the_limit_wait_then_are_refused_busy():
    import asyncio

    import httpx

    release = asyncio.Event()

    class SlowIssuer(FakeIssuer):
        async def close(self, **kw):
            self.calls.append(("close", kw))
            await release.wait()
            return {"session_id": "s-1", "state": "closed", "status": "expired"}

    issuer = SlowIssuer(None)
    app = FastAPI()
    app.include_router(build_sandbox_sessions_router(
        issuer, policy=SandboxPolicy(), validator_hotkey=VALIDATOR, verify_open=accept,
        verify_close=accept, clock=lambda: NOW, max_concurrent_closes=1, close_wait_s=0.2))

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://v") as http:
            first = asyncio.ensure_future(http.post(CLOSE, json=close_body()))
            while not issuer.calls:
                await asyncio.sleep(0.01)
            busy = await http.post(CLOSE, json=close_body())
            release.set()
            done = await first
            after = await http.post(CLOSE, json=close_body())
        return busy, done, after

    busy, done, after = asyncio.run(run())
    assert busy.status_code == 503 and busy.json()["reason"] == "close_busy"
    assert int(busy.headers["Retry-After"]) >= 1
    assert done.status_code == 200 and after.status_code == 200
    assert len(issuer.calls) == 2                   # the refused close never reached it


def test_the_close_limit_must_be_positive():
    with pytest.raises(ValueError):
        build_sandbox_sessions_router(FakeIssuer(None), policy=SandboxPolicy(),
                                      validator_hotkey=VALIDATOR, max_concurrent_closes=0)


def test_a_stalled_close_body_times_out_without_holding_a_close_slot():
    import asyncio

    import httpx

    issuer = FakeIssuer({"session_id": "s-1", "state": "closed", "status": "expired"})
    app = FastAPI()
    app.include_router(build_sandbox_sessions_router(
        issuer, policy=SandboxPolicy(), validator_hotkey=VALIDATOR, verify_open=accept,
        verify_close=accept, clock=lambda: NOW, max_concurrent_closes=1, close_wait_s=0.2,
        close_body_timeout_s=0.5))
    gone = asyncio.Event()

    async def stalled():
        yield b'{"miner_hotkey": '
        await gone.wait()                               # the rest never arrives in time
        yield b'"x"}'

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://v") as http:
            slow = asyncio.ensure_future(http.post(CLOSE, content=stalled()))
            await asyncio.sleep(0.1)
            honest = await http.post(CLOSE, json=close_body())
            timed_out = await slow
            gone.set()
        return honest, timed_out

    honest, timed_out = asyncio.run(run())
    assert honest.status_code == 200                   # the stalled body held no slot
    assert timed_out.status_code == 408 and timed_out.json()["reason"] == "body_timeout"
    assert len(issuer.calls) == 1
