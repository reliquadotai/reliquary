"""Customer CPU-job contracts, request recovery and verified report publication."""

import copy
import hashlib
import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from reliquary.cli import platform as p

KEY = "rlq_" + "a" * 48
REQUEST_KEY = "00000000-0000-4000-8000-000000000001"
JOB_ID = "cpu:00000000-0000-4000-8000-000000000002"
UNIT_ID = "00000000-0000-4000-8000-000000000003"
ARTIFACT_ID = "00000000-0000-4000-8000-000000000004"
CONTENT = '{"text":"Bonjour é"}\n'.encode()
REF = {"dataset_id": REQUEST_KEY, "sha256": hashlib.sha256(CONTENT).hexdigest(),
       "format": "corpus", "size_bytes": len(CONTENT)}
SPEC = {"kind": "dataset_validation", "input": REF,
        "limits": {"deadline_seconds": 600, "max_attempts": 1}}
REPORT = {"schema_version": 1, "input_sha256": REF["sha256"], "format": "corpus",
          "rows": 1, "valid": True, "errors": []}
REPORT_BYTES = json.dumps(REPORT, ensure_ascii=False, separators=(",", ":")).encode()
ARTIFACT = {"id": ARTIFACT_ID, "job_id": JOB_ID, "work_unit_id": UNIT_ID, "attempt_id": None,
            "kind": "validation_report", "name": "validation-report.json", "content_type": "application/json",
            "size_bytes": len(REPORT_BYTES), "sha256": hashlib.sha256(REPORT_BYTES).hexdigest(), "created_at": 100}


def job(*, state="queued", artifacts=None, revision="1"):
    artifacts = artifacts or []
    return {"schema_version": 1, "id": JOB_ID, "kind": "dataset_validation",
            "source_ref": {"type": "dataset_validation", "id": JOB_ID[4:]}, "state": state,
            "native_phase": None, "revision": revision, "spec_sha256": "b" * 64, "spec": copy.deepcopy(SPEC),
            "current_attempt_id": None, "work_units": [{"id": UNIT_ID, "job_id": JOB_ID, "state": state,
                "attempt_ids": [], "artifact_ids": [a["id"] for a in artifacts]}],
            "attempts": [], "artifacts": copy.deepcopy(artifacts), "created_at": 100, "updated_at": 100,
            "finished_at": 100 if state in ("succeeded", "failed", "cancelled") else None}


def capabilities(*, can_submit=False):
    return {"schema_version": 1, "workloads": [{"kind": "dataset_validation", "execution": "cpu",
        "can_submit": can_submit, "attempts_supported": True, "reason": None if can_submit else "worker_unavailable",
        "input_formats": ["instruction", "preference", "corpus"], "controls": ["pause", "resume", "cancel"],
        "enrolled_workers": 1, "live_workers": 1 if can_submit else 0, "qualified_workers": 1 if can_submit else 0,
        "queued_jobs": 0, "running_jobs": 0, "oldest_queued_at": None}]}


def client(handler, url="https://test.invalid"):
    return p.PlatformClient(url, KEY, transport=httpx.MockTransport(handler))


def test_entire_customer_cpu_lifecycle_uses_scoped_contracts():
    requests = []

    def respond(request):
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer " + KEY
        path = request.url.path
        if path.endswith("/capabilities"):
            return httpx.Response(200, json=capabilities())
        if path == "/api/v1/job-inputs":
            assert json.loads(request.content) == {"schema_version": 1, "format": "corpus", "content": CONTENT.decode()}
            return httpx.Response(201, json={"schema_version": 1, "input": REF})
        if path == "/api/v1/jobs" and request.method == "POST":
            assert json.loads(request.content) == {"schema_version": 1, "spec": SPEC}
            return httpx.Response(201, json={"schema_version": 1, "job": job()})
        if path == "/api/v1/jobs":
            assert dict(request.url.params) == {"limit": "2", "cursor": "next_page"}
            return httpx.Response(200, json={"schema_version": 1, "jobs": [job()], "next_cursor": "page_2"})
        if path.endswith("/events"):
            return httpx.Response(200, json={"schema_version": 1, "events": [{"id": "event_1", "job_id": JOB_ID,
                "sequence": 1, "state": "queued", "native_phase": None, "code": None, "created_at": 100}], "next_cursor": None})
        if path.rsplit("/", 1)[-1] in ("pause", "resume", "cancel"):
            assert request.headers["X-Resource-Revision"] == "1"
            assert json.loads(request.content) == {"schema_version": 1, "revision": "1", "reason": "Reviewed control"}
            state = {"pause": "paused", "resume": "queued", "cancel": "cancelled"}[path.rsplit("/", 1)[-1]]
            return httpx.Response(200, json={"schema_version": 1, "job": job(state=state, revision="2")})
        if "/artifacts/" in path:
            return httpx.Response(200, json={"schema_version": 1, "artifact": ARTIFACT,
                                             "report": dict(reversed(list(REPORT.items())))})
        return httpx.Response(200, json={"schema_version": 1, "job": job(state="succeeded", artifacts=[ARTIFACT])})

    with client(respond) as api:
        assert api.capabilities()["workloads"][0]["can_submit"] is False
        assert api.upload(CONTENT, "corpus", REQUEST_KEY) == REF
        assert api.create(REF, REQUEST_KEY)["id"] == JOB_ID
        assert api.page(limit=2, cursor="next_page")["next_cursor"] == "page_2"
        assert api.page(job_id=JOB_ID)["events"][0]["sequence"] == 1
        for action in ("pause", "resume", "cancel"):
            assert api.control(JOB_ID, action, "1", "Reviewed control", REQUEST_KEY)["revision"] == "2"
        artifact, content = api.report(JOB_ID, ARTIFACT_ID)
        assert artifact == ARTIFACT and content == REPORT_BYTES
    assert api.http.is_closed
    assert all(r.headers["Idempotency-Key"] == REQUEST_KEY for r in requests if r.method == "POST")
    assert all(r.url.host == "test.invalid" for r in requests)


@pytest.mark.parametrize("url", ["http://example.com", "https://user:secret@example.com", "https://example.com/api",
                                  "https://example.com?key=secret", "https://example.com#secret", "https://example.com:bad",
                                  "https://@example.com", "https://example.com?", "https://example.com#"])
def test_invalid_origin_is_refused_without_exposing_it(url):
    with pytest.raises(p.PlatformError, match="invalid_origin_or_timeout") as caught:
        client(lambda r: pytest.fail("sent request"), url=url)
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize("url", ["http://127.0.0.1:8000", "http://[::1]:8000", "http://localhost:8000"])
def test_http_is_accepted_only_for_loopback_origins(url):
    with client(lambda request: httpx.Response(200, json=capabilities()), url=url) as api:
        assert api.capabilities()["schema_version"] == 1


@pytest.mark.parametrize("change", [{"qualified_workers": 1}, {"queued_jobs": 1}, {"enrolled_workers": True},
                                    {"execution": "subnet"}, {"input_formats": ["corpus", "corpus"]},
                                    {"controls": ["pause", "pause"]}])
def test_capabilities_reject_incoherent_capacity_contracts(change):
    value = capabilities()
    value["workloads"][0].update(change)
    with client(lambda r: httpx.Response(200, json=value)) as api:
        with pytest.raises(p.PlatformError, match="invalid_response"):
            api.capabilities()


@pytest.mark.parametrize("timeout", [0, -1, 121, float("nan"), float("inf")])
def test_invalid_request_timeout_is_refused(timeout):
    with pytest.raises(p.PlatformError, match="invalid_origin_or_timeout"):
        p.PlatformClient("https://test.invalid", KEY, timeout=timeout)


@pytest.mark.parametrize("key", ["", " secret", "secret\n", "x" * 4097])
def test_invalid_credential_is_not_displayed(key):
    with pytest.raises(p.PlatformError, match="credential_required") as caught:
        p.PlatformClient("https://test.invalid", key)
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize("failure", ["network", "redirect", "bad_json", "bad_schema", "oversize"])
def test_uncertain_mutations_are_never_retried_or_redirected(method, failure):
    calls = []

    def respond(request):
        calls.append(request)
        if failure == "network":
            raise httpx.ReadTimeout(KEY, request=request)
        if failure == "redirect":
            return httpx.Response(307, headers={"Location": "https://other.invalid/" + KEY})
        if failure == "bad_json":
            return httpx.Response(200, text=KEY)
        if failure == "bad_schema":
            return httpx.Response(200, json={"schema_version": True})
        return httpx.Response(200, content=b"x" * (4 * 1024 * 1024 + 1))

    with client(respond) as api:
        with pytest.raises(p.PlatformError) as caught:
            api.request(method, "/api/v1/jobs", key=REQUEST_KEY if method == "POST" else None)
    assert len(calls) == 1
    assert KEY not in str(caught.value)
    assert (caught.value.code == "outcome_unknown") == (method == "POST")
    if method == "POST":
        assert REQUEST_KEY in str(caught.value) and "same input and key" in str(caught.value)


@pytest.mark.parametrize("status", [408, 500, 503])
def test_server_mutation_failures_keep_original_request_identity(status):
    with client(lambda r: httpx.Response(status, json={"error": {"code": "internal_error", "message": KEY}})) as api:
        with pytest.raises(p.PlatformError, match="outcome_unknown"):
            api.create(REF, REQUEST_KEY)


@pytest.mark.parametrize("status", [401, 403, 409, 429])
def test_plain_client_refusals_are_definite_and_secret_free(status):
    with client(lambda r: httpx.Response(status, text=KEY)) as api:
        with pytest.raises(p.PlatformError) as caught:
            api.create(REF, REQUEST_KEY)
    assert caught.value.code == f"http_{status}" and KEY not in str(caught.value)


@pytest.mark.parametrize("content,format", [(b"", "corpus"), (b"x" * 32769, "corpus"), (b"\xff", "corpus"),
                                           (CONTENT, "unsupported")])
def test_import_limits_are_checked_before_sending(content, format):
    with client(lambda r: pytest.fail("sent request")) as api:
        with pytest.raises(ValueError):
            api.upload(content, format, REQUEST_KEY)


@pytest.mark.parametrize("key", [None, "", "not-a-uuid", "00000000-0000-1000-8000-000000000001"])
def test_mutations_require_caller_uuid_identity_before_sending(key):
    with client(lambda r: pytest.fail("sent request")) as api:
        with pytest.raises(ValueError):
            api.create(REF, key)


@pytest.mark.parametrize("change", [{"size_bytes": 1}, {"sha256": "0" * 64}, {"format": "instruction"}])
def test_upload_receipt_must_bind_actual_bytes(change):
    with client(lambda r: httpx.Response(201, json={"schema_version": 1, "input": {**REF, **change}})) as api:
        with pytest.raises(p.PlatformError, match="outcome_unknown"):
            api.upload(CONTENT, "corpus", REQUEST_KEY)


@pytest.mark.parametrize("revision", ["0", "-1", "1.2", "9007199254740992", "a" * 64])
def test_controls_require_reviewed_cpu_revision(revision):
    with client(lambda r: pytest.fail("sent request")) as api:
        with pytest.raises(ValueError):
            api.control(JOB_ID, "pause", revision, "Reviewed control", REQUEST_KEY)


def test_stale_revision_is_a_definite_refusal():
    with client(lambda r: httpx.Response(409, json={"error": {"code": "revision_conflict"}})) as api:
        with pytest.raises(p.PlatformError, match="current job revision") as caught:
            api.control(JOB_ID, "pause", "1", "Reviewed control", REQUEST_KEY)
    assert caught.value.code == "revision_conflict"


@pytest.mark.parametrize("change", [{"id": "subnet:" + JOB_ID[4:]}, {"state": "unknown"},
                                   {"revision": "0"}, {"finished_at": 100}, {"current_attempt_id": REQUEST_KEY},
                                   {"work_units": []}])
def test_invalid_job_receipt_never_becomes_a_success(change):
    with client(lambda r: httpx.Response(201, json={"schema_version": 1, "job": {**job(), **change}})) as api:
        with pytest.raises(p.PlatformError, match="outcome_unknown"):
            api.create(REF, REQUEST_KEY)


@pytest.mark.parametrize("change", [{"sha256": "0" * 64}, {"size_bytes": 1}, {"id": REQUEST_KEY},
                                    {"job_id": "cpu:" + REQUEST_KEY}, {"kind": "dataset"}])
def test_invalid_artifact_cannot_be_published(change):
    def respond(request):
        if "/artifacts/" in request.url.path:
            return httpx.Response(200, json={"schema_version": 1, "artifact": {**ARTIFACT, **change}, "report": REPORT})
        return httpx.Response(200, json={"schema_version": 1, "job": job(state="succeeded", artifacts=[ARTIFACT])})
    with client(respond) as api:
        with pytest.raises(p.PlatformError, match="invalid_response"):
            api.report(JOB_ID, ARTIFACT_ID)


@pytest.mark.parametrize("change", [{"rows": 0}, {"valid": False}, {"input_sha256": "0" * 64}, {"format": "instruction"},
                                    {"errors": [{"line": 2, "code": "invalid_json"}]}])
def test_report_contract_and_input_identity_are_verified(change):
    def respond(request):
        if "/artifacts/" in request.url.path:
            return httpx.Response(200, json={"schema_version": 1, "artifact": ARTIFACT, "report": {**REPORT, **change}})
        return httpx.Response(200, json={"schema_version": 1, "job": job(state="succeeded", artifacts=[ARTIFACT])})
    with client(respond) as api:
        with pytest.raises(p.PlatformError, match="invalid_response"):
            api.report(JOB_ID, ARTIFACT_ID)


def test_cli_persists_input_reference_and_verified_report_without_credentials(tmp_path, monkeypatch):
    input_file, reference, output = tmp_path / "input.jsonl", tmp_path / "input-ref.json", tmp_path / "report.json"
    input_file.write_bytes(CONTENT)

    def respond(request):
        if request.url.path == "/api/v1/job-inputs":
            return httpx.Response(201, json={"schema_version": 1, "input": REF})
        if "/artifacts/" in request.url.path:
            return httpx.Response(200, json={"schema_version": 1, "artifact": ARTIFACT, "report": REPORT})
        if request.method == "POST":
            return httpx.Response(201, json={"schema_version": 1, "job": job()})
        return httpx.Response(200, json={"schema_version": 1, "job": job(state="succeeded", artifacts=[ARTIFACT])})

    monkeypatch.setattr(p, "_client", lambda ctx: client(respond))
    runner = CliRunner()
    imported = runner.invoke(p.platform_app, ["import", str(input_file), "--format", "corpus", "--out", str(reference),
        "--idempotency-key", REQUEST_KEY, "--json"])
    assert imported.exit_code == 0, imported.output
    assert json.loads(reference.read_text()) == REF
    created = runner.invoke(p.platform_app, ["create", "--input", str(reference), "--idempotency-key", REQUEST_KEY, "--json"])
    assert created.exit_code == 0, created.output
    assert json.loads(created.stdout)["data"]["id"] == JOB_ID
    downloaded = runner.invoke(p.platform_app, ["download", JOB_ID, "--artifact", ARTIFACT_ID, "--out", str(output), "--json"])
    assert downloaded.exit_code == 0, downloaded.output
    assert output.read_bytes() == REPORT_BYTES
    assert json.loads(downloaded.stdout)["data"]["verified"] is True
    assert KEY not in imported.output + created.output + downloaded.output + reference.read_text() + output.read_text()
    again = runner.invoke(p.platform_app, ["download", JOB_ID, "--artifact", ARTIFACT_ID, "--out", str(output), "--json"])
    assert again.exit_code == 2 and output.read_bytes() == REPORT_BYTES


def test_file_publication_refuses_a_concurrent_destination(tmp_path):
    destination = tmp_path / "report.json"
    destination.write_bytes(b"keep")
    with pytest.raises(ValueError):
        p._publish(destination, REPORT_BYTES)
    assert destination.read_bytes() == b"keep"
    assert list(tmp_path.iterdir()) == [destination]


def test_cli_never_publishes_a_report_that_fails_digest_verification(tmp_path, monkeypatch):
    def respond(request):
        if "/artifacts/" in request.url.path:
            return httpx.Response(200, json={"schema_version": 1, "artifact": {**ARTIFACT, "sha256": "0" * 64}, "report": REPORT})
        return httpx.Response(200, json={"schema_version": 1, "job": job(state="succeeded", artifacts=[ARTIFACT])})
    monkeypatch.setattr(p, "_client", lambda ctx: client(respond))
    destination = tmp_path / "report.json"
    result = CliRunner().invoke(p.platform_app, ["download", JOB_ID, "--artifact", ARTIFACT_ID, "--out", str(destination), "--json"])
    assert result.exit_code == 1 and json.loads(result.stderr)["error"]["code"] == "invalid_response"
    assert not destination.exists() and list(tmp_path.iterdir()) == []


def test_environment_aliases_and_default_origin(monkeypatch):
    monkeypatch.delenv("RELIQUARY_API_KEY", raising=False)
    monkeypatch.delenv("RELIQUARY_API_URL", raising=False)
    monkeypatch.setenv("JOBS_API_KEY", KEY)
    monkeypatch.setenv("JOBS_API_ORIGIN", "https://test.invalid")
    seen = []
    monkeypatch.setattr(p, "PlatformClient", lambda url, key, timeout: seen.append((url, key, timeout)))
    p._client(type("Context", (), {"obj": {"url": "https://test.invalid", "timeout": 60}})())
    assert seen == [("https://test.invalid", KEY, 60)]


def test_versioned_json_failure_does_not_expose_credentials(monkeypatch):
    monkeypatch.setattr(p, "_client", lambda ctx: client(lambda request: (_ for _ in ()).throw(httpx.ConnectError(KEY))))
    result = CliRunner().invoke(p.platform_app, ["status", JOB_ID, "--json"])
    assert result.exit_code == 1 and KEY not in result.output
    error = json.loads(result.stderr)
    assert error["schema"] == "reliquary/cli/v1" and error["error"]["code"] == "upstream_unavailable"


@pytest.mark.parametrize("timeout,poll", [(0, 1), (-1, 1), (86401, 1), (float("inf"), 1),
                                        (float("nan"), 1), (1, 0), (1, -1), (1, float("nan"))])
def test_wait_requires_positive_finite_deadlines_before_requests(timeout, poll):
    with client(lambda r: pytest.fail("sent request")) as api:
        with pytest.raises(ValueError):
            api.wait(JOB_ID, timeout=timeout, poll_interval=poll)


def test_wait_caps_request_and_sleep_by_remaining_budget(monkeypatch):
    clock = [100.0]
    sleeps, deadlines = [], []
    def respond(request):
        assert request.method == "GET"
        deadlines.append(request.extensions["timeout"]["read"])
        clock[0] += 0.2
        return httpx.Response(200, json={"schema_version": 1, "job": job()})
    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds
    monkeypatch.setattr(p.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(p.time, "sleep", sleep)
    with client(respond) as api:
        with pytest.raises(p.PlatformError) as caught:
            api.wait(JOB_ID, timeout=1, poll_interval=2)
    assert caught.value.code == "wait_timeout" and caught.value.job["id"] == JOB_ID
    assert JOB_ID in str(caught.value) and "no cancellation" in str(caught.value)
    assert deadlines == [1] and sleeps == [pytest.approx(0.8)]


@pytest.mark.parametrize("state,exit_code", [("succeeded", 0), ("failed", 1), ("cancelled", 1), ("needs_attention", 1)])
def test_wait_cli_emits_state_and_reports_unsuccessful_outcomes(monkeypatch, state, exit_code):
    monkeypatch.setattr(p, "_client", lambda ctx: client(lambda r: httpx.Response(200, json={"schema_version": 1, "job": job(state=state)})))
    result = CliRunner().invoke(p.platform_app, ["wait", JOB_ID, "--timeout", "5", "--poll-seconds", "0.1", "--json"])
    assert result.exit_code == exit_code, result.output
    assert json.loads(result.stdout)["data"]["state"] == state
    if exit_code:
        assert json.loads(result.stderr)["error"]["code"] == "job_" + state


def test_wait_cli_timeout_preserves_last_state_and_job_identifier(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(p.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(p.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(p, "_client", lambda ctx: client(lambda r: httpx.Response(200, json={"schema_version": 1, "job": job()})))
    result = CliRunner().invoke(p.platform_app, ["wait", JOB_ID, "--timeout", "0.1", "--poll-seconds", "1", "--json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["data"]["id"] == JOB_ID
    assert json.loads(result.stderr)["error"]["code"] == "wait_timeout" and JOB_ID in result.stderr


def test_request_reaching_wait_deadline_keeps_resumable_job_identifier(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(p.time, "monotonic", lambda: clock[0])
    def respond(request):
        clock[0] += request.extensions["timeout"]["read"]
        raise httpx.ReadTimeout(KEY, request=request)
    with client(respond) as api:
        with pytest.raises(p.PlatformError) as caught:
            api.wait(JOB_ID, timeout=1)
    assert caught.value.code == "wait_timeout" and JOB_ID in str(caught.value) and KEY not in str(caught.value)


def test_streamed_chunks_stop_at_wait_budget_without_draining_response(monkeypatch):
    clock, chunks, closed = [100.0], [], []
    monkeypatch.setattr(p.time, "monotonic", lambda: clock[0])
    class SlowStream(httpx.SyncByteStream):
        def __iter__(self):
            for number in range(4):
                clock[0] += 0.6
                chunks.append(number)
                yield b"part"
        def close(self):
            closed.append(True)
    with client(lambda r: httpx.Response(200, stream=SlowStream())) as api:
        with pytest.raises(p.PlatformError) as caught:
            api.wait(JOB_ID, timeout=1)
    assert caught.value.code == "wait_timeout" and JOB_ID in str(caught.value)
    assert chunks == [0, 1] and closed == [True]


def test_failed_staging_flush_never_publishes_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(p.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("disk unavailable")))
    destination = tmp_path / "reference.json"
    with pytest.raises(OSError):
        p._publish(destination, b"reference")
    assert not destination.exists() and list(tmp_path.iterdir()) == []
