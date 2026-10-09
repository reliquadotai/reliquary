"""Scoped customer commands for the platform's CPU dataset-validation API."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from urllib.parse import quote, urlsplit

import httpx
import typer

from reliquary.cli.output import CLIGroup, JSON_OPTION, emit_result, fail
from reliquary.shared.strict_json import strict_json_loads

DEFAULT_URL = "https://api.reliqua.ai"
FORMATS = ("instruction", "preference", "corpus")
STATES = ("queued", "running", "paused", "cancelling", "succeeded", "failed", "cancelled", "needs_attention")
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z")
DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{1,100}\Z")
MAX_SAFE_INTEGER = 2**53 - 1


class PlatformError(RuntimeError):
    def __init__(self, code: str, *, key: str | None = None, status: int | None = None, job_id: str | None = None, job=None):
        self.code, self.status, self.job = code, status, job
        message = f"Platform request failed: {code}."
        if code == "outcome_unknown":
            message += f" Request key: {key}. Inspect status and reconcile with the same input and key; do not retry with a new key."
        elif code in ("revision_conflict", "stale_revision", "http_409"):
            message += " Read the current job revision before reviewing another control request."
        elif code == "wait_timeout":
            message += f" Job {job_id} continues remotely. Resume wait or inspect status; no cancellation was requested."
        elif code.startswith("job_"):
            message += f" Job {job_id} requires review; inspect status and events."
        super().__init__(message)


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError("Invalid platform contract.")


def _object(value, keys=None):
    _require(isinstance(value, dict) and (keys is None or set(value) == set(keys)))
    return value


def _integer(value, low=0, high=MAX_SAFE_INTEGER):
    _require(type(value) is int and low <= value <= high)
    return value


def _text(value, maximum=200):
    _require(isinstance(value, str) and 0 < len(value.encode("utf-16-le")) // 2 <= maximum
             and not re.search(r"[\x00-\x1f\x7f]", value))
    return value


def _choice(value, choices):
    _require(isinstance(value, str) and value in choices)
    return value


def _pattern(value, pattern):
    _require(isinstance(value, str) and pattern.fullmatch(value) is not None)
    return value


def _uuid(value):
    return _pattern(value, UUID_RE)


def _job_id(value):
    _require(isinstance(value, str) and value.startswith("cpu:"))
    _uuid(value[4:])
    return value


def _revision(value):
    _require(isinstance(value, str) and re.fullmatch(r"[1-9][0-9]{0,15}", value) is not None
             and int(value) <= MAX_SAFE_INTEGER)
    return value


def _list(value, maximum=100):
    _require(isinstance(value, list) and len(value) <= maximum)
    return value


def _time(value):
    return None if value is None else _integer(value)


def _input_ref(value):
    v = _object(value, ("dataset_id", "sha256", "format", "size_bytes"))
    return {"dataset_id": _uuid(v["dataset_id"]), "sha256": _pattern(v["sha256"], DIGEST_RE),
            "format": _choice(v["format"], FORMATS), "size_bytes": _integer(v["size_bytes"], 1, 32768)}


def _spec(value):
    v = _object(value, ("kind", "input", "limits"))
    _require(v["kind"] == "dataset_validation")
    limits = _object(v["limits"], ("deadline_seconds", "max_attempts"))
    return {"kind": v["kind"], "input": _input_ref(v["input"]), "limits": {
        "deadline_seconds": _integer(limits["deadline_seconds"], 60, 3600),
        "max_attempts": _integer(limits["max_attempts"], 1, 3)}}


def _artifact(value, job_id):
    v = _object(value)
    _require(v.get("job_id") == job_id and v.get("kind") == "validation_report"
             and v.get("name") == "validation-report.json" and v.get("content_type") == "application/json")
    return {"id": _pattern(v["id"], TOKEN_RE), "job_id": _job_id(v["job_id"]),
            "work_unit_id": None if v["work_unit_id"] is None else _uuid(v["work_unit_id"]),
            "attempt_id": None if v["attempt_id"] is None else _uuid(v["attempt_id"]),
            "kind": "validation_report", "name": v["name"], "content_type": v["content_type"],
            "size_bytes": _integer(v["size_bytes"], 1, 4 * 1024 * 1024),
            "sha256": _pattern(v["sha256"], DIGEST_RE), "created_at": _integer(v["created_at"])}


def _job(value, expected_id=None):
    v = _object(value)
    job_id = _job_id(v["id"])
    _require(type(v["schema_version"]) is int and v["schema_version"] == 1 and v["kind"] == "dataset_validation"
             and (expected_id is None or job_id == expected_id) and v["native_phase"] is None)
    source = _object(v["source_ref"])
    _require(source.get("type") == "dataset_validation" and source.get("id") == job_id[4:])
    spec = _spec(v["spec"])
    attempts = []
    for item in _list(v["attempts"], 3):
        a = _object(item)
        _require(a["job_id"] == job_id)
        parsed = {"id": _uuid(a["id"]), "job_id": job_id, "work_unit_id": _uuid(a["work_unit_id"]),
                  "attempt_number": _integer(a["attempt_number"], 1, spec["limits"]["max_attempts"]),
                  "worker_id": _uuid(a["worker_id"]), "fence": _integer(a["fence"], 1),
                  "state": _choice(a["state"], ("running", "succeeded", "failed", "expired", "cancelled")),
                  **{key: _integer(a[key]) for key in ("lease_expires_at", "heartbeat_at", "started_at")},
                  "finished_at": _time(a["finished_at"]),
                  "error_code": None if a["error_code"] is None else _pattern(a["error_code"], re.compile(r"[a-z][a-z0-9_]{0,79}\Z"))}
        _require(parsed["started_at"] <= parsed["heartbeat_at"] <= parsed["lease_expires_at"]
                 and (parsed["state"] == "running") == (parsed["finished_at"] is None)
                 and (parsed["finished_at"] is None or parsed["finished_at"] >= parsed["started_at"]))
        attempts.append(parsed)
    artifacts = [_artifact(a, job_id) for a in _list(v["artifacts"], 1000)]
    units = _list(v["work_units"], 1)
    _require(len(units) == 1)
    u = _object(units[0])
    unit = {"id": _uuid(u["id"]), "job_id": _job_id(u["job_id"]), "state": _choice(u["state"], STATES),
            "attempt_ids": [_uuid(a) for a in _list(u["attempt_ids"], 3)],
            "artifact_ids": [_pattern(a, TOKEN_RE) for a in _list(u["artifact_ids"], 1000)]}
    _require(unit["job_id"] == job_id and len({a["id"] for a in attempts}) == len(attempts)
             and len({a["id"] for a in artifacts}) == len(artifacts)
             and set(unit["attempt_ids"]) == {a["id"] for a in attempts}
             and set(unit["artifact_ids"]) == {a["id"] for a in artifacts}
             and len(set(unit["attempt_ids"])) == len(unit["attempt_ids"])
             and len(set(unit["artifact_ids"])) == len(unit["artifact_ids"])
             and all(a["work_unit_id"] == unit["id"] for a in attempts)
             and all(a["work_unit_id"] in (None, unit["id"]) and (
                 a["attempt_id"] is None or a["work_unit_id"] == unit["id"]
                 and a["attempt_id"] in unit["attempt_ids"]) for a in artifacts))
    current = None if v["current_attempt_id"] is None else _uuid(v["current_attempt_id"])
    _require(current is None or any(a["id"] == current and a["state"] == "running" for a in attempts))
    result = {"schema_version": 1, "id": job_id, "kind": "dataset_validation",
              "source_ref": {"type": "dataset_validation", "id": source["id"]},
              "state": _choice(v["state"], STATES), "native_phase": None, "revision": _revision(v["revision"]),
              "spec_sha256": _pattern(v["spec_sha256"], DIGEST_RE), "spec": spec,
              "current_attempt_id": current, "work_units": [unit], "attempts": attempts, "artifacts": artifacts,
              "created_at": _integer(v["created_at"]), "updated_at": _integer(v["updated_at"]),
              "finished_at": _time(v["finished_at"])}
    _require(result["updated_at"] >= result["created_at"] and (
        result["state"] in ("succeeded", "failed", "cancelled")) == (result["finished_at"] is not None)
        and (result["finished_at"] is None or result["created_at"] <= result["finished_at"] <= result["updated_at"]))
    return result


def _cursor(value):
    return None if value is None else _pattern(value, re.compile(r"[A-Za-z0-9_-]{1,512}\Z"))


def _job_response(value, expected_id=None):
    _object(value, ("schema_version", "job"))
    return _job(value["job"], expected_id)


def _report(value):
    v = _object(value, ("schema_version", "input_sha256", "format", "rows", "valid", "errors"))
    _require(type(v["schema_version"]) is int and v["schema_version"] == 1 and type(v["valid"]) is bool)
    rows = _integer(v["rows"], 0, 10000)
    errors = []
    for item in _list(v["errors"], 20):
        e = _object(item, ("line", "code"))
        errors.append({"line": _integer(e["line"], 0, rows), "code": _choice(e["code"], (
            "too_large", "invalid_utf8", "too_many_rows", "empty", "invalid_json", "invalid_fields"))})
    _require(v["valid"] == (not errors) and (not v["valid"] or rows > 0))
    return {"schema_version": 1, "input_sha256": _pattern(v["input_sha256"], DIGEST_RE),
            "format": _choice(v["format"], FORMATS), "rows": rows, "valid": v["valid"], "errors": errors}


class PlatformClient:
    def __init__(self, url: str, key: str, *, timeout: float = 60, transport=None):
        try:
            origin = urlsplit(url)
            _require(origin.scheme == "https" or origin.scheme == "http" and origin.hostname in ("127.0.0.1", "::1", "localhost"))
            _require(origin.hostname is not None and origin.username is None and origin.password is None
                     and origin.path in ("", "/") and "?" not in url and "#" not in url)
            origin.port
            _require(math.isfinite(timeout) and 0 < timeout <= 120)
        except (ValueError, TypeError):
            raise PlatformError("invalid_origin_or_timeout") from None
        if not key or key != key.strip() or len(key) > 4096 or re.search(r"[\x00-\x20\x7f]", key):
            raise PlatformError("credential_required")
        self._key = key
        self._timeout = timeout
        self.http = httpx.Client(base_url=url.rstrip("/"), timeout=timeout, follow_redirects=False,
                                 trust_env=False, transport=transport)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.http.close()

    def request(self, method, path, *, body=None, key=None, revision=None, params=None, timeout=None,
                deadline=None, parse=lambda value: value):
        if method == "POST":
            _uuid(key)
        headers = {"Authorization": f"Bearer {self._key}", "Accept": "application/json"}
        if key is not None:
            headers["Idempotency-Key"] = key
        if revision is not None:
            headers["X-Resource-Revision"] = revision
        try:
            options = {} if timeout is None else {"timeout": timeout}
            with self.http.stream(method, path, json=body, headers=headers, params=params, **options) as response:
                if deadline is not None and time.monotonic() >= deadline:
                    raise PlatformError("wait_timeout")
                status = response.status_code
                if 300 <= status < 400:
                    raise PlatformError("outcome_unknown" if method == "POST" else "redirect_refused", key=key, status=status)
                content = bytearray()
                for chunk in response.iter_bytes():
                    if deadline is not None and time.monotonic() >= deadline:
                        raise PlatformError("wait_timeout")
                    content.extend(chunk)
                    if len(content) > 4 * 1024 * 1024:
                        raise PlatformError("outcome_unknown" if method == "POST" else "response_too_large", key=key, status=status)
                if deadline is not None and time.monotonic() >= deadline:
                    raise PlatformError("wait_timeout")
                if method == "POST" and (status >= 500 or status == 408):
                    raise PlatformError("outcome_unknown", key=key, status=status)
                if not 200 <= status < 300:
                    code = f"http_{status}"
                    try:
                        value = strict_json_loads(content)
                        error = value.get("error") if isinstance(value, dict) else None
                        candidate = error.get("code") if isinstance(error, dict) else None
                        if (isinstance(candidate, str) and self._key not in candidate
                                and re.fullmatch(r"[a-z][a-z0-9_]{0,79}", candidate) is not None):
                            code = candidate
                    except (ValueError, UnicodeError):
                        pass
                    raise PlatformError(code, status=status)
                try:
                    value = strict_json_loads(content)
                    _object(value)
                    _require(self._key.encode() not in content)
                except (ValueError, UnicodeError):
                    raise PlatformError("outcome_unknown" if method == "POST" else "invalid_response", key=key, status=status) from None
                try:
                    _require(type(value.get("schema_version")) is int and value["schema_version"] == 1)
                    return parse(value)
                except (ValueError, KeyError, TypeError, UnicodeError):
                    raise PlatformError("outcome_unknown" if method == "POST" else "invalid_response", key=key, status=status) from None
        except httpx.RequestError:
            raise PlatformError("outcome_unknown" if method == "POST" else "upstream_unavailable", key=key) from None

    def capabilities(self):
        def parse(v):
            _object(v, ("schema_version", "workloads"))
            workloads = []
            for item in _list(v["workloads"], 2):
                w = _object(item)
                kind = _choice(w["kind"], ("dataset_validation", "corpus_generation"))
                _require(type(w["can_submit"]) is bool and type(w["attempts_supported"]) is bool)
                result = {"kind": kind, "execution": _choice(w["execution"], ("cpu", "subnet")),
                          "can_submit": w["can_submit"], "attempts_supported": w["attempts_supported"],
                          "reason": None if w["reason"] is None else _text(w["reason"], 500),
                          "input_formats": [_choice(f, FORMATS) for f in _list(w["input_formats"], 3)],
                          "controls": [_choice(c, ("pause", "resume", "cancel")) for c in _list(w["controls"], 3)],
                          **{k: _integer(w[k], 0, 8) for k in ("enrolled_workers", "live_workers", "qualified_workers", "queued_jobs", "running_jobs")},
                          "oldest_queued_at": _time(w["oldest_queued_at"])}
                _require(result["qualified_workers"] <= result["live_workers"] <= result["enrolled_workers"]
                         and result["queued_jobs"] + result["running_jobs"] <= 8
                         and (result["queued_jobs"] > 0) == (result["oldest_queued_at"] is not None)
                         and (kind == "dataset_validation") == (result["execution"] == "cpu")
                         and result["attempts_supported"] == (kind == "dataset_validation")
                         and len(set(result["input_formats"])) == len(result["input_formats"])
                         and len(set(result["controls"])) == len(result["controls"])
                         and (kind != "corpus_generation" or not result["can_submit"] and not result["input_formats"]
                              and result["queued_jobs"] == result["running_jobs"] == 0))
                workloads.append(result)
            _require(len({w["kind"] for w in workloads}) == len(workloads))
            return {"schema_version": 1, "workloads": workloads}
        return self.request("GET", "/api/v1/jobs/capabilities", parse=parse)

    def upload(self, content: bytes, format: str, key: str):
        _choice(format, FORMATS)
        _require(1 <= len(content) <= 32768)
        try:
            text = content.decode("utf-8")
        except UnicodeError:
            raise ValueError("Input must be UTF-8.") from None
        def parse(v):
            _object(v, ("schema_version", "input"))
            ref = _input_ref(v["input"])
            _require(ref["format"] == format and ref["size_bytes"] == len(content)
                     and ref["sha256"] == hashlib.sha256(content).hexdigest())
            return ref
        return self.request("POST", "/api/v1/job-inputs", key=key,
                            body={"schema_version": 1, "format": format, "content": text}, parse=parse)

    def create(self, ref, key, *, deadline=600, attempts=1):
        spec = _spec({"kind": "dataset_validation", "input": ref,
                      "limits": {"deadline_seconds": deadline, "max_attempts": attempts}})
        def parse(v):
            job = _job_response(v)
            _require(job["spec"] == spec)
            return job
        return self.request("POST", "/api/v1/jobs", key=key, body={"schema_version": 1, "spec": spec}, parse=parse)

    def status(self, job_id, *, timeout=None, deadline=None):
        _job_id(job_id)
        return self.request("GET", f"/api/v1/jobs/{quote(job_id, safe='')}", timeout=timeout, deadline=deadline,
                            parse=lambda v: _job_response(v, job_id))

    def wait(self, job_id, *, timeout=300, poll_interval=2):
        _job_id(job_id)
        _require(math.isfinite(timeout) and 0 < timeout <= 86400 and math.isfinite(poll_interval) and poll_interval > 0)
        deadline = time.monotonic() + timeout
        job = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PlatformError("wait_timeout", job_id=job_id, job=job)
            try:
                job = self.status(job_id, timeout=min(self._timeout, remaining), deadline=deadline)
            except PlatformError:
                if time.monotonic() >= deadline:
                    raise PlatformError("wait_timeout", job_id=job_id, job=job) from None
                raise
            if job["state"] in ("succeeded", "failed", "cancelled", "needs_attention"):
                return job
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PlatformError("wait_timeout", job_id=job_id, job=job)
            time.sleep(min(poll_interval, remaining))

    def page(self, *, job_id=None, cursor=None, limit=100):
        _integer(limit, 1, 100)
        params = {"limit": limit}
        if cursor is not None:
            params["cursor"] = _cursor(cursor)
        if job_id is not None:
            _job_id(job_id)
        path = "/api/v1/jobs" + (f"/{quote(job_id, safe='')}/events" if job_id is not None else "")
        def parse(v):
            field = "events" if job_id is not None else "jobs"
            _object(v, ("schema_version", field, "next_cursor"))
            items = []
            for item in _list(v[field]):
                if job_id is None:
                    items.append(_job(item))
                else:
                    e = _object(item)
                    _require(e["job_id"] == job_id and e["native_phase"] is None)
                    items.append({"id": _text(e["id"], 100), "job_id": job_id,
                                  "sequence": _integer(e["sequence"], 1), "state": _choice(e["state"], STATES),
                                  "native_phase": None, "code": None if e["code"] is None else _pattern(e["code"], re.compile(r"[a-z][a-z0-9_]{0,79}\Z")),
                                  "created_at": _integer(e["created_at"])})
            _require(len({i["id"] for i in items}) == len(items)
                     and (job_id is None or len({i["sequence"] for i in items}) == len(items)))
            return {"schema_version": 1, field: items, "next_cursor": _cursor(v["next_cursor"])}
        return self.request("GET", path, params=params, parse=parse)

    def control(self, job_id, action, revision, reason, key):
        _job_id(job_id)
        _choice(action, ("pause", "resume", "cancel"))
        _revision(revision)
        _text(reason, 500)
        reason = reason.strip()
        _require(len(reason) >= 3)
        return self.request("POST", f"/api/v1/jobs/{quote(job_id, safe='')}/{action}", key=key, revision=revision,
                            body={"schema_version": 1, "revision": revision, "reason": reason},
                            parse=lambda v: _job_response(v, job_id))

    def report(self, job_id, artifact_id):
        _job_id(job_id)
        _pattern(artifact_id, TOKEN_RE)
        job = self.status(job_id)
        def parse(v):
            _object(v, ("schema_version", "artifact", "report"))
            artifact = _artifact(v["artifact"], job_id)
            report = _report(v["report"])
            content = json.dumps(report, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            _require(artifact["id"] == artifact_id and report["input_sha256"] == job["spec"]["input"]["sha256"]
                     and report["format"] == job["spec"]["input"]["format"]
                     and artifact["size_bytes"] == len(content) and artifact["sha256"] == hashlib.sha256(content).hexdigest()
                     and any(a == artifact for a in job["artifacts"]))
            return artifact, content
        return self.request("GET", f"/api/v1/jobs/{quote(job_id, safe='')}/artifacts/{quote(artifact_id, safe='')}", parse=parse)


platform_app = typer.Typer(name="platform", cls=CLIGroup, help="Customer CPU dataset validation jobs; availability comes from capabilities.")


@platform_app.callback()
def platform_options(ctx: typer.Context, url: str = typer.Option(None, "--url", help="HTTPS API origin (HTTP loopback for local development)."),
                     timeout: float = typer.Option(60, "--timeout", help="Seconds per request, 0 < timeout <= 120.")):
    ctx.obj = {"url": url or os.getenv("RELIQUARY_API_URL") or os.getenv("JOBS_API_ORIGIN") or DEFAULT_URL,
               "timeout": timeout}


def _client(ctx):
    return PlatformClient(ctx.obj["url"], os.getenv("RELIQUARY_API_KEY") or os.getenv("JOBS_API_KEY") or "",
                          timeout=ctx.obj["timeout"])


def _run(ctx, action, as_json):
    try:
        with _client(ctx) as client:
            result = action(client)
        emit_result(result, as_json=as_json)
        return result
    except PlatformError as exc:
        if exc.job is not None:
            emit_result(exc.job, as_json=as_json)
        fail(str(exc), code=exc.code)
    except (ValueError, KeyError, TypeError, UnicodeError):
        fail("Invalid input or platform contract; check the file, identifiers and options.", code="invalid_input", exit_code=2)
    except OSError:
        fail("Local file operation failed; existing files are kept. Inspect status before repeating a write.", code="file_error")


def _new_file(path: Path):
    if path.exists() or path.is_symlink():
        raise ValueError("Output already exists.")


def _publish(path: Path, content: bytes):
    _new_file(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication refuses a destination created while the request ran.
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@platform_app.command("capabilities")
def capabilities(ctx: typer.Context, as_json: bool = JSON_OPTION):
    _run(ctx, lambda client: client.capabilities(), as_json)


@platform_app.command("import")
def import_input(ctx: typer.Context, file: Path = typer.Argument(...), format: str = typer.Option(..., "--format"),
                 out: Path = typer.Option(..., "--out", help="New file for the verified input reference."),
                 idempotency_key: str = typer.Option(..., "--idempotency-key", help="Caller UUID v4; preserve it for reconciliation."),
                 as_json: bool = JSON_OPTION):
    def action(client):
        _new_file(out)
        with file.open("rb") as handle:
            content = handle.read(32769)
        ref = client.upload(content, format, idempotency_key)
        _publish(out, json.dumps(ref, sort_keys=True).encode("utf-8"))
        return {"input": ref, "reference_file": str(out)}
    _run(ctx, action, as_json)


@platform_app.command("create")
def create_job(ctx: typer.Context, input: Path = typer.Option(..., "--input", help="Reference file written by platform import."),
               idempotency_key: str = typer.Option(..., "--idempotency-key"),
               deadline_seconds: int = typer.Option(600, "--deadline-seconds", min=60, max=3600),
               max_attempts: int = typer.Option(1, "--max-attempts", min=1, max=3), as_json: bool = JSON_OPTION):
    def action(client):
        with input.open("rb") as handle:
            ref = _input_ref(strict_json_loads(handle.read(32769)))
        return client.create(ref, idempotency_key, deadline=deadline_seconds, attempts=max_attempts)
    _run(ctx, action, as_json)


@platform_app.command("list")
def list_jobs(ctx: typer.Context, cursor: str = typer.Option(None, "--cursor"),
              limit: int = typer.Option(100, "--limit", min=1, max=100), as_json: bool = JSON_OPTION):
    _run(ctx, lambda client: client.page(cursor=cursor, limit=limit), as_json)


@platform_app.command("status")
def status_job(ctx: typer.Context, job_id: str = typer.Argument(...), as_json: bool = JSON_OPTION):
    _run(ctx, lambda client: client.status(job_id), as_json)


@platform_app.command("wait", help="Observe completion with a deadline; timeout or interruption leaves the remote job running.")
def wait_job(ctx: typer.Context, job_id: str = typer.Argument(...),
             wait_timeout: float = typer.Option(300, "--timeout", "--wait-timeout", help="Total wait seconds, 0 < timeout <= 86400."),
             poll_interval: float = typer.Option(2, "--poll-seconds", "--poll-interval", help="Positive seconds between status requests."),
             as_json: bool = JSON_OPTION):
    job = _run(ctx, lambda client: client.wait(job_id, timeout=wait_timeout, poll_interval=poll_interval), as_json)
    if job["state"] != "succeeded":
        exc = PlatformError("job_" + job["state"], job_id=job_id)
        fail(str(exc), code=exc.code)


@platform_app.command("events")
def job_events(ctx: typer.Context, job_id: str = typer.Argument(...), cursor: str = typer.Option(None, "--cursor"),
               limit: int = typer.Option(100, "--limit", min=1, max=100), as_json: bool = JSON_OPTION):
    _run(ctx, lambda client: client.page(job_id=job_id, cursor=cursor, limit=limit), as_json)


def _control_command(action):
    def command(ctx: typer.Context, job_id: str = typer.Argument(...),
                revision: str = typer.Option(..., "--revision", help="Current revision from status; never inferred."),
                reason: str = typer.Option(..., "--reason"), idempotency_key: str = typer.Option(..., "--idempotency-key"),
                as_json: bool = JSON_OPTION):
        _run(ctx, lambda client: client.control(job_id, action, revision, reason, idempotency_key), as_json)
    return command


for _action in ("pause", "resume", "cancel"):
    platform_app.command(_action, help=f"Request {_action} using the reviewed job revision; read status to observe effect.")(_control_command(_action))


@platform_app.command("download")
def download_report(ctx: typer.Context, job_id: str = typer.Argument(...),
                    artifact: str = typer.Option(..., "--artifact", help="Validation report artifact ID from status."),
                    out: Path = typer.Option(..., "--out", help="New output file; existing files are kept."),
                    as_json: bool = JSON_OPTION):
    def action(client):
        _new_file(out)
        metadata, content = client.report(job_id, artifact)
        _publish(out, content)
        return {"artifact": metadata, "out": str(out), "verified": True}
    _run(ctx, action, as_json)
