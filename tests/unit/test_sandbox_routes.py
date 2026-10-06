"""The session routes: fresh, signed, registered requests only; refusals mapped to
statuses a miner can act on; the token goes to the miner and nowhere else."""

import json
import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytest.importorskip("reliquary_sandbox.attest")

from reliquary.protocol.sandbox_session import (  # noqa: E402
    SandboxSessionCloseRequest, SandboxSessionOpenRequest,
)
from reliquary.protocol.signatures import (  # noqa: E402
    build_sandbox_close_binding, build_sandbox_open_binding, verify_sandbox_close_signature,
    verify_sandbox_open_signature,
)
from reliquary.sandbox import routes, sessions  # noqa: E402
from reliquary.sandbox.routes import (  # noqa: E402
    MAX_CLOSE_BODY_BYTES, MAX_OPEN_BODY_BYTES, REFUSAL_STATUS, build_sandbox_sessions_router,
)
from reliquary.sandbox.sessions import Grant, Refusal, SandboxPolicy  # noqa: E402
from tests.unit.sandbox_fixtures import NOW  # noqa: E402

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


def client(issuer, *, verify=lambda request: True, registration=None, now=NOW):
    app = FastAPI()
    app.include_router(build_sandbox_sessions_router(
        issuer, policy=SandboxPolicy(), verify_open=verify, verify_close=verify,
        registration=registration, clock=lambda: now))
    return TestClient(app, raise_server_exceptions=False)


def open_body(**overrides):
    body = {"miner_hotkey": "5Hot", "request_id": "a" * 32, "at": NOW,
            "engagement": {"kind": "corpus", "job_id": "swe-agentic-v1", "prompt_index": 3},
            "signature": "00"}
    body.update(overrides)
    return body


def close_body(**overrides):
    body = {"miner_hotkey": "5Hot", "request_id": "b" * 32, "at": NOW, "session_id": "s-1",
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
    assert issuer.calls == [("open", {"hotkey": "5Hot", "request_id": "a" * 32, "engagement": {
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
    answer = client(issuer, verify=lambda request: False).post("/corpus/sandbox/sessions",
                                                                json=open_body())
    assert answer.status_code == 403 and answer.json()["reason"] == "bad_signature"
    assert issuer.calls == []


def test_an_unsigned_close_never_reaches_the_issuer():
    issuer = FakeIssuer(None)
    answer = client(issuer, verify=lambda request: False).post(
        "/corpus/sandbox/sessions/s-1/close", json=close_body())
    assert answer.status_code == 403 and answer.json()["reason"] == "bad_signature"
    assert issuer.calls == []


def test_a_verifier_that_raises_is_a_bad_signature():
    def broken(request):
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
    assert seen == ["5Hot"] and len(issuer.calls) == 1


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
    assert all(code in (400, 403, 404, 409, 413, 422, 429, 500, 503)
               for code in REFUSAL_STATUS.values())
    assert all(REFUSAL_STATUS[reason] == 503 for reason in (
        "sandbox_capacity", "directory_unavailable", "store_unavailable", "ledger_unavailable",
        "task_unavailable", "registration_unavailable"))


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
        answer = client(issuer, verify=lambda request, v=verify: v).post(
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
    assert issuer.calls[0][1] == {"hotkey": "5Hot", "session_id": "s-1", "reason": "final",
                                  "transcript": {"token": {}, "records": []}}


def test_a_close_for_another_session_is_unknown():
    answer = client(FakeIssuer(None)).post("/corpus/sandbox/sessions/s-2/close", json=close_body())
    assert answer.status_code == 404 and answer.json()["reason"] == "session_unknown"


def test_bindings_ignore_absent_fields_and_bind_everything_else():
    model = SandboxSessionOpenRequest(**open_body())
    assert build_sandbox_open_binding(model) == build_sandbox_open_binding(open_body())
    other = open_body(engagement={"kind": "corpus", "job_id": "swe-agentic-v1", "prompt_index": 4})
    assert build_sandbox_open_binding(other) != build_sandbox_open_binding(open_body())
    for field, value in [("miner_hotkey", "5Other"), ("request_id", "c" * 32), ("at", NOW + 1)]:
        assert build_sandbox_open_binding(open_body(**{field: value})) != \
            build_sandbox_open_binding(open_body())
        assert build_sandbox_close_binding(close_body(**{field: value})) != \
            build_sandbox_close_binding(close_body())
    one = build_sandbox_close_binding(close_body())
    two = build_sandbox_close_binding(close_body(transcript={"token": {}, "records": [1]}))
    assert one != two != build_sandbox_close_binding(close_body(transcript=None))
    assert one != build_sandbox_close_binding(close_body(session_id="s-2"))
    assert one != build_sandbox_close_binding(close_body(reason="open_failed"))
    assert SandboxSessionCloseRequest(**close_body()).session_id == "s-1"
    assert build_sandbox_open_binding(open_body()) != build_sandbox_close_binding(close_body())


def test_a_real_hotkey_signature_verifies_and_binds_the_engagement():
    bt = pytest.importorskip("bittensor")
    keypair = bt.Keypair.create_from_uri("//Alice")
    body = open_body(miner_hotkey=keypair.ss58_address)
    body["signature"] = keypair.sign(build_sandbox_open_binding(body)).hex()
    assert verify_sandbox_open_signature(SandboxSessionOpenRequest(**body))
    tampered = {**body, "engagement": {**body["engagement"], "prompt_index": 4}}
    assert not verify_sandbox_open_signature(SandboxSessionOpenRequest(**tampered))
    close = close_body(miner_hotkey=keypair.ss58_address, signature=body["signature"])
    assert not verify_sandbox_close_signature(SandboxSessionCloseRequest(**close))
    bob = bt.Keypair.create_from_uri("//Bob")
    forged = {**body, "signature": bob.sign(build_sandbox_open_binding(body)).hex()}
    assert not verify_sandbox_open_signature(SandboxSessionOpenRequest(**forged))
    assert not verify_sandbox_open_signature(SandboxSessionOpenRequest(**{**body, "signature": "zz"}))


def test_the_default_router_checks_real_signatures():
    bt = pytest.importorskip("bittensor")
    keypair = bt.Keypair.create_from_uri("//Alice")
    issuer = FakeIssuer(Grant("s-1", TOKEN, "http://10.0.0.5:8080", NOW + 4500))
    app = FastAPI()
    app.include_router(build_sandbox_sessions_router(issuer, policy=SandboxPolicy(),
                                                     clock=lambda: NOW))
    http = TestClient(app, raise_server_exceptions=False)
    body = open_body(miner_hotkey=keypair.ss58_address)
    assert http.post("/corpus/sandbox/sessions", json=body).status_code == 403
    body["signature"] = keypair.sign(build_sandbox_open_binding(body)).hex()
    assert http.post("/corpus/sandbox/sessions", json=body).status_code == 200
    assert {route.path for route in build_sandbox_sessions_router(
        issuer, policy=SandboxPolicy(), prefix="/rl").routes} == {
        "/rl/sandbox/sessions", "/rl/sandbox/sessions/{session_id}/close"}


def test_the_wire_models_refuse_unknown_fields():
    with pytest.raises(ValueError):
        SandboxSessionOpenRequest(**open_body(extra=1))
    with pytest.raises(ValueError):
        SandboxSessionOpenRequest(**open_body(request_id="A" * 32))
    assert json.loads(SandboxSessionOpenRequest(**open_body()).model_dump_json())["at"] == NOW
