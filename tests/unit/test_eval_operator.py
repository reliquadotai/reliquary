"""The operator's evaluations: qualify then create, grade home, compare two checkpoints."""

from __future__ import annotations

import asyncio
import hashlib
import json
from types import SimpleNamespace

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from reliquary.eval import operator as op
from reliquary.eval import qualification as qual
from reliquary.infrastructure import corpus_job_store as job_store
from tests.unit.test_admin_eval_files import _complete_job
from tests.unit.test_admin_eval_jobs import (  # noqa: F401
    MODEL, REVISION, SAMPLING, SECRET, THRESHOLDS, admin,
)
from tests.unit.test_jobs_cli import registry  # noqa: F401


def client_for(admin):  # noqa: F811
    # The fixture's client keeps its event loop open, as a server does: a
    # grading runs beside the request that started it.
    return op.AdminClient("https://testserver", SECRET, http=admin.client)


def finish(qualification_id, status=qual.QUALIFIED):
    async def run():
        store = qual.QualificationStore()
        record, etag = await store.read(qualification_id)
        result = {"thresholds": THRESHOLDS, "architecture": "Qwen3ForCausalLM",
                  "checkpoint_sha256": "d" * 64, "eos_token_id": 151645,
                  "band": {"exp_mismatch": 50, "mant_mean": 27.6, "mant_median": 20.0,
                           "chunks": 90}, "clamped": [], "measurements": {}}
        if status != qual.QUALIFIED:
            result = {"refused_reason": "band_over_ceiling"}
        record.update(status=status, result=result)
        await store.write(record, etag)

    asyncio.run(run())


def card_of(admin):  # noqa: F811
    return json.loads((admin.root / "set" / "set.json").read_text())


def test_a_model_names_a_full_revision():
    assert op.split_model("org/Teutonic@" + "a" * 40) == ("org/Teutonic", "a" * 40)
    for bad in ("org/Teutonic", "org/Teutonic@main", "org/Teutonic@abc", "@" + "a" * 40):
        with pytest.raises(ValueError, match="repo@<40-hex commit>"):
            op.split_model(bad)


def test_ids_are_deterministic_and_bounded():
    conditions = {"model": "m", "revision": "a" * 40, "set_id": "s", "problems": 3}
    first = op.qualification_id_for("order-", conditions)
    assert first == op.qualification_id_for("order-", dict(reversed(conditions.items())))
    assert first.startswith("order-q-") and len(first) <= 63
    assert first != op.qualification_id_for("order-", {**conditions, "problems": 4})
    order = {"model": "org/m", "sampling": {"temperature": 0.6}, "max_new_tokens": 512,
             "thinking": True}
    short = op.default_job_id("order-", "a" * 40, "aime26", 30, 8, order)
    assert short.startswith("order-eval-aaaaaaaa-aime26-n30x8-") and len(short) == 39
    # Another sampling, budget or thinking mode is another job.
    for change in ({"sampling": {"temperature": 1.0}}, {"thinking": False},
                   {"max_new_tokens": 1024}, {"model": "org/other"}):
        assert op.default_job_id("order-", "a" * 40, "aime26", 30, 8,
                                 {**order, **change}) != short
    long = op.default_job_id("order-", "a" * 40, "verifiers-livecodebench-a1b2c3d4-r0-n1055",
                             1055, 1, order)
    assert len(long) <= 63 and long.startswith("order-eval-aaaaaaaa-verifiers")
    # A catalog set id holds underscores; a job id may not.
    catalog = op.default_job_id("order-", "a" * 40, "reliquary_dapo_math_v1-eval-r0-n100",
                                100, 4, order)
    assert op.checked_job_id(catalog) == catalog and "_" not in catalog
    with pytest.raises(ValueError, match=r"\[a-z0-9-\]"):
        op.checked_job_id("order-eval-a_b")


def test_create_qualifies_waits_then_creates_and_a_second_run_reuses_both(admin):  # noqa: F811
    client, logs, slept = client_for(admin), [], []
    card = card_of(admin)

    def sleep(seconds):
        slept.append(seconds)
        qid = [line.split()[1] for line in logs if line.endswith("requested for " + card["set_id"])][0]
        finish(qid)

    kwargs = dict(cards=[card], model=MODEL, revision=REVISION, samples=4, count=5,
                  max_new_tokens=512, thinking=False, sampling=SAMPLING, sleep=sleep,
                  log=logs.append, poll_seconds=7)
    created = op.create_evaluations(client, **kwargs)
    assert slept == [7]
    assert created[0]["job_id"].startswith(f"order-eval-{REVISION[:8]}-{card['set_id']}-n5x4-")
    job, _ = asyncio.run(job_store.read_job(created[0]["job_id"]))
    assert job.slots_per_prompt == 4 and job.checkpoint_revision == REVISION
    assert any("qualification" in line and "qualified" in line for line in logs)
    # The same command again finds its qualification done and the job declared.
    again = op.create_evaluations(client, **{**kwargs, "sleep": lambda s: pytest.fail("waited")})
    assert again[0]["job_id"] == created[0]["job_id"]
    assert again[0]["qualification_id"] == created[0]["qualification_id"]


def test_a_refused_model_gets_no_job(admin):  # noqa: F811
    client, logs = client_for(admin), []
    card = card_of(admin)

    def sleep(seconds):
        qid = [line.split()[1] for line in logs if line.endswith("requested for " + card["set_id"])][0]
        finish(qid, status=qual.REFUSED)

    with pytest.raises(RuntimeError, match="qualification refused: .*band_over_ceiling"):
        op.create_evaluations(client, cards=[card], model=MODEL, revision=REVISION, samples=2,
                              max_new_tokens=512, thinking=False, sampling=SAMPLING,
                              sleep=sleep, log=logs.append)
    assert not [k for k in admin.registry["entries"] if k.startswith("order-eval")]


def test_create_refuses_more_problems_than_the_set_holds(admin):  # noqa: F811
    with pytest.raises(ValueError, match="holds 8 problems, not 9"):
        op.create_evaluations(client_for(admin), cards=[card_of(admin)], model=MODEL,
                              revision=REVISION, samples=2, count=9, max_new_tokens=512,
                              thinking=False, sampling=SAMPLING)


def test_grade_brings_the_three_files_home(admin, tmp_path):  # noqa: F811
    _complete_job(admin)
    answer = op.grade_job(client_for(admin), "order-eval-7", out=tmp_path / "home",
                          sleep=lambda s: None)
    assert answer["complete"] is True
    report = json.loads((tmp_path / "home" / "report.json").read_text())
    assert report["eval_id"] == "order-eval-7"
    assert pq.read_table(tmp_path / "home" / "graded.parquet").num_rows == 20
    assert json.loads((tmp_path / "home" / "manifest.json").read_text())["complete"] is True


def write_grading(directory, *, correct, n_problems, samples=2, sampling=None, set_id="s1",
                  revision="a" * 40, ungraded=0, grader=None):
    """correct: {problem: [bool, ...]}; problems absent from it have no row."""
    directory.mkdir(parents=True)
    rows = [{"env": "math", "problem_id": p, "correct": c} for p, cs in correct.items() for c in cs]
    pq.write_table(pa.Table.from_pylist(rows), directory / "graded.parquet")
    rate = sum(sum(cs) for cs in correct.values()) / (n_problems * samples)
    (directory / "report.json").write_text(json.dumps({
        "eval_id": directory.name,
        "envs": {"math": {"pass@1": {"value": rate}, "n_problems": n_problems,
                          "ungraded_rows": ungraded, "missing_rows": 0}},
        "provenance": {"model": "org/m", "revision": revision,
                       "sampling": sampling or {"temperature": 0.6}, "max_new_tokens": 512,
                       "thinking": True,
                       "sets": [{"set_id": set_id, "env": "math", "problems": n_problems,
                                 "samples": samples,
                                 **({"source_kind": "verifiers",
                                     "taskset_at_grading": grader} if grader else {})}]}}))


def test_compare_two_checkpoints_on_the_same_problems(tmp_path):
    write_grading(tmp_path / "a", n_problems=4,
                  correct={"p0": [True, False], "p1": [False, False], "p2": [True, True]})
    write_grading(tmp_path / "b", n_problems=4, revision="b" * 40,
                  correct={"p0": [True, True], "p1": [True, False], "p2": [True, True]})
    result = op.compare_reports(tmp_path / "a", tmp_path / "b")
    math = result["envs"]["math"]
    assert math["a"] == pytest.approx(3 / 8) and math["b"] == pytest.approx(5 / 8)
    # p0 +0.5, p1 +0.5, p2 0, p3 (no row on either side) 0: over 4 problems.
    assert math["diff"] == pytest.approx(0.25)
    low, high = math["ci95"]
    assert low <= 0.25 <= high and low >= 0.0
    assert result["b"]["revision"] == "b" * 40


def test_compare_refuses_different_conditions(tmp_path):
    write_grading(tmp_path / "a", n_problems=2, correct={"p0": [True, True]})
    write_grading(tmp_path / "b", n_problems=2, correct={"p0": [True, True]},
                  sampling={"temperature": 1.0})
    with pytest.raises(ValueError, match=r"differ in \['sampling'\]"):
        op.compare_reports(tmp_path / "a", tmp_path / "b")
    write_grading(tmp_path / "c", n_problems=2, correct={"p0": [True, True]}, set_id="s2")
    with pytest.raises(ValueError, match=r"\['sets'\]"):
        op.compare_reports(tmp_path / "a", tmp_path / "c")


def test_a_failed_qualification_is_asked_again_only_with_another_attempt(admin):  # noqa: F811
    client, logs = client_for(admin), []
    card = card_of(admin)

    def qid_of():
        return [line.split()[1] for line in logs
                if line.endswith("requested for " + card["set_id"])][-1]

    kwargs = dict(cards=[card], model=MODEL, revision=REVISION, samples=2,
                  max_new_tokens=512, thinking=False, sampling=SAMPLING, log=logs.append)
    with pytest.raises(RuntimeError, match="rerun with --attempt 1"):
        op.create_evaluations(client, **kwargs,
                              sleep=lambda s: finish(qid_of(), status=qual.FAILED))
    failed = qid_of()
    created = op.create_evaluations(client, **kwargs, attempt=1,
                                    sleep=lambda s: finish(qid_of()))
    assert created[0]["qualification_id"] != failed


def test_a_bad_job_id_costs_no_qualification(admin):  # noqa: F811
    with pytest.raises(ValueError, match=r"\[a-z0-9-\]"):
        op.create_evaluations(client_for(admin), cards=[card_of(admin)], model=MODEL,
                              revision=REVISION, samples=2, max_new_tokens=512, thinking=False,
                              sampling=SAMPLING, job_id="order-eval-Bad_Id")
    card = card_of(admin)
    conditions = {"model": MODEL, "revision": REVISION, "set_id": card["set_id"],
                  "problems": card["count"], "sampling": SAMPLING, "max_new_tokens": 512,
                  "thinking": False}
    qid = op.qualification_id_for("order-", {**conditions, "completions": 32, "attempt": 0})
    record, _ = asyncio.run(qual.QualificationStore().read(qid))
    assert record is None


def test_compare_refuses_ungraded_rows_unless_told(tmp_path):
    write_grading(tmp_path / "a", n_problems=2, correct={"p0": [True, True]})
    write_grading(tmp_path / "b", n_problems=2, correct={"p0": [True, False]}, ungraded=1)
    with pytest.raises(ValueError, match="ungraded rows"):
        op.compare_reports(tmp_path / "a", tmp_path / "b")
    result = op.compare_reports(tmp_path / "a", tmp_path / "b", allow_ungraded=True)
    assert result["envs"]["math"]["ungraded_rows_b"] == 1


def test_compare_refuses_two_grader_versions(tmp_path):
    write_grading(tmp_path / "a", n_problems=1, correct={"p0": [True, True]},
                  grader={"package_version": "1.0", "verifiers_version": "0.3.1"})
    write_grading(tmp_path / "b", n_problems=1, correct={"p0": [True, True]},
                  grader={"package_version": "1.1", "verifiers_version": "0.3.1"})
    with pytest.raises(ValueError, match="grader versions"):
        op.compare_reports(tmp_path / "a", tmp_path / "b")


def test_explicit_seeds_name_distinct_jobs_and_reuse_qualification():
    requests = []

    class Client:
        def json(self, method, path, body=None):
            if method == "GET":
                return {"status": "qualified"}
            requests.append((path, body))
            return {}

    kwargs = dict(cards=[{"count": 3, "set_id": "s", "source": "math"}],
                  model=MODEL, revision=REVISION, samples=2, max_new_tokens=512,
                  thinking=False, sampling=SAMPLING, log=lambda line: None)
    unseeded, first, second, repeated = [op.create_evaluations(Client(), **kwargs, seed=seed)[0]
                                        for seed in (None, 1, 2, 1)]
    assert first["job_id"] != second["job_id"]
    assert first["job_id"] == repeated["job_id"]
    assert len({run["qualification_id"] for run in (unseeded, first, second)}) == 1
    conditions = {"model": MODEL, "revision": REVISION, "set_id": "s", "problems": 3,
                  "sampling": SAMPLING, "max_new_tokens": 512, "thinking": False}
    assert unseeded["job_id"] == op.default_job_id("order-", REVISION, "s", 3, 2, conditions)


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_admin_transport_errors_are_named_and_hide_transport_details(method):
    def fail(request):
        raise httpx.ReadTimeout(SECRET.decode(), request=request)

    with httpx.Client(base_url="https://test.invalid", transport=httpx.MockTransport(fail)) as http:
        with pytest.raises(op.AdminError, match="ReadTimeout") as caught:
            op.AdminClient("https://test.invalid", SECRET, http=http).json(method, "/admin/v1/jobs")
    assert caught.value.status is None
    assert SECRET.decode() not in str(caught.value)
    assert ("outcome is unknown" in str(caught.value)) == (method == "POST")


@pytest.mark.parametrize("status,content", [(502, b"<html>gateway error</html>"), (200, b"invalid"),
                                            (200, b"[]")])
def test_admin_invalid_json_is_a_typed_error(status, content):
    transport = httpx.MockTransport(lambda request: httpx.Response(status, content=content))
    with httpx.Client(base_url="https://test.invalid", transport=transport) as http:
        with pytest.raises(op.AdminError) as caught:
            op.AdminClient("https://test.invalid", SECRET, http=http).json("POST", "/admin/v1/jobs")
    assert "outcome is unknown" in str(caught.value)
    assert content.decode() not in str(caught.value)


def test_admin_error_redacts_secret_in_json_detail():
    transport = httpx.MockTransport(lambda request: httpx.Response(422, json={"detail": SECRET.decode()}))
    with httpx.Client(base_url="https://test.invalid", transport=transport) as http:
        with pytest.raises(op.AdminError) as caught:
            op.AdminClient("https://test.invalid", SECRET, http=http).json("GET", "/admin/v1/jobs")
    assert SECRET.decode() not in str(caught.value)
    assert caught.value.detail == "[redacted]"


def test_admin_context_closes_only_owned_http_clients(monkeypatch):
    closed = []
    http = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(op.httpx, "Client", lambda **kwargs: http)
    with op.AdminClient("https://test.invalid", SECRET):
        pass
    assert closed == [True]
    with op.AdminClient("https://test.invalid", SECRET, http=http):
        pass
    assert closed == [True]


def _download_files():
    files = {"report.json": json.dumps({"eval_id": "order-eval-7"}).encode(),
             "graded.parquet": b"graded rows"}
    files["manifest.json"] = json.dumps({
        "schema": "reliquary/eval-report/v2", "eval_id": "order-eval-7",
        "files": [{"name": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                  for name, content in files.items()]}).encode()
    return files


def _grade_http(files, *, failed_file=None, state="done"):
    def respond(request):
        if request.url.path.endswith("/status"):
            return httpx.Response(200, json={"manifest": {
                "prompt_source": "eval-set:s:1:" + "a" * 64, "slots_per_prompt": 1,
                "sampling": {"n": 1}}})
        if request.url.path.endswith("/grade"):
            return httpx.Response(200 if state == "done" else 202,
                                  json={"state": state, "eval_id": "order-eval-7"})
        name = request.url.path.rsplit("/", 1)[-1]
        if name == failed_file:
            return httpx.Response(503, json={"detail": "unavailable"})
        return httpx.Response(200, content=files[name])

    return httpx.Client(base_url="https://test.invalid", transport=httpx.MockTransport(respond))


@pytest.mark.parametrize("failed_file", ["manifest.json", "graded.parquet"])
def test_failed_bundle_download_leaves_no_partial_result(tmp_path, failed_file):
    destination = tmp_path / "result"
    destination.mkdir()
    with _grade_http(_download_files(), failed_file=failed_file) as http:
        client = op.AdminClient("https://test.invalid", SECRET, http=http)
        with pytest.raises(op.AdminError, match="download failed"):
            op.grade_job(client, "order-eval-7", out=destination)
    assert list(destination.iterdir()) == []
    assert list(tmp_path.iterdir()) == [destination]


def test_corrupt_download_is_not_installed(tmp_path):
    files = _download_files()
    files["graded.parquet"] = b"changed rows"
    with _grade_http(files) as http:
        with pytest.raises(ValueError, match="does not match its manifest"):
            op.grade_job(op.AdminClient("https://test.invalid", SECRET, http=http),
                         "order-eval-7", out=tmp_path / "result")
    assert list(tmp_path.iterdir()) == []


def test_existing_results_are_preserved_before_any_request(tmp_path):
    previous = tmp_path / "report.json"
    previous.write_text("previous result")
    with pytest.raises(ValueError, match="new or empty directory"):
        op.grade_job(None, "order-eval-7", out=tmp_path)
    assert previous.read_text() == "previous result"


def test_completed_bundle_replaces_empty_destination(tmp_path):
    destination = tmp_path / "result"
    destination.mkdir()
    files = _download_files()
    with _grade_http(files) as http:
        op.grade_job(op.AdminClient("https://test.invalid", SECRET, http=http),
                     "order-eval-7", out=destination)
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == files
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("options", [{"poll_seconds": 0}, {"poll_seconds": float("nan")},
                                     {"timeout_seconds": -1}, {"timeout_seconds": float("inf")}])
def test_invalid_wait_options_cost_no_request(options):
    with pytest.raises(ValueError, match="positive finite"):
        op.grade_job(None, "order-eval-7", **options)
    with pytest.raises(ValueError, match="positive finite"):
        op.create_evaluations(None, cards=[], model=MODEL, revision=REVISION, samples=2,
                              max_new_tokens=512, thinking=False, sampling=SAMPLING, **options)


@pytest.mark.parametrize("job_id", [None, 7, "", "order-eval-Bad", "order eval", "a" * 64])
def test_invalid_grade_job_id_is_refused_before_any_request(job_id):
    requests = []
    with httpx.Client(transport=httpx.MockTransport(
            lambda request: requests.append(request) or httpx.Response(500))) as http:
        client = op.AdminClient("https://test.invalid", SECRET, http=http)
        with pytest.raises(ValueError, match=r"\[a-z0-9-\]"):
            op.grade_job(client, job_id)
    assert requests == []


@pytest.mark.parametrize("eval_id", [7, "", "order eval", "_result", "a" * 129])
def test_invalid_evaluation_id_is_refused_before_any_request(eval_id):
    requests = []
    with httpx.Client(transport=httpx.MockTransport(
            lambda request: requests.append(request) or httpx.Response(500))) as http:
        client = op.AdminClient("https://test.invalid", SECRET, http=http)
        with pytest.raises(ValueError, match="delivery id"):
            op.grade_job(client, "order-eval-7", eval_id=eval_id)
    assert requests == []


@pytest.mark.parametrize("eval_id", ["order-eval-Result_2", "order-" + "a" * 122])
def test_evaluation_id_uses_the_server_delivery_name_rule(eval_id):
    requests = []

    def respond(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"manifest": {
                "prompt_source": "eval-set:s:1:" + "a" * 64,
                "slots_per_prompt": 1, "sampling": {"n": 1}}})
        return httpx.Response(200, json={"state": "done", "eval_id": eval_id})

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        result = op.grade_job(op.AdminClient("https://test.invalid", SECRET, http=http),
                              "order-eval-7", eval_id=eval_id)
    assert result["eval_id"] == eval_id
    assert [(r.method, r.url.path) for r in requests] == [
        ("GET", "/admin/v1/jobs/order-eval-7/status"),
        ("POST", f"/admin/v1/evaluations/{eval_id}/grade"),
    ]


def test_qualification_poll_does_not_sleep_past_its_deadline():
    now, sleeps = [0.0], []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    client = SimpleNamespace(json=lambda *args: {"status": "pending"})
    with pytest.raises(TimeoutError, match="rerun the same command"):
        op._wait(client, ["order-q-7"], poll_seconds=30, timeout_seconds=1,
                 sleep=sleep, log=lambda line: None, clock=lambda: now[0])
    assert sleeps == [1]


def test_no_wait_create_only_requests_qualifications():
    requests = []
    client = SimpleNamespace(json=lambda *args: requests.append(args))
    answer = op.create_evaluations(client, cards=[{"count": 3, "set_id": "s", "source": "math"}],
                                  model=MODEL, revision=REVISION, samples=2, max_new_tokens=512,
                                  thinking=False, sampling=SAMPLING, wait=False, log=lambda line: None)
    assert len(requests) == 1 and requests[0][1] == "/admin/v1/qualifications"
    assert answer[0]["state"] == "qualification_requested"


def test_no_wait_grade_returns_running_without_output(tmp_path):
    with _grade_http({}, state="running") as http:
        answer = op.grade_job(op.AdminClient("https://test.invalid", SECRET, http=http),
                              "order-eval-7", wait=False, out=tmp_path / "result",
                              sleep=lambda seconds: pytest.fail("waited"))
    assert answer == {"state": "running", "eval_id": "order-eval-7"}
    assert list(tmp_path.iterdir()) == []


def test_duplicate_sets_are_refused_before_requesting_qualification():
    card = {"count": 3, "set_id": "s", "source": "math"}
    with pytest.raises(ValueError, match="each --set must be distinct"):
        op.create_evaluations(None, cards=[card, card], model=MODEL, revision=REVISION,
                              samples=2, max_new_tokens=512, thinking=False, sampling=SAMPLING)


def test_unknown_qualification_state_is_a_typed_error():
    client = SimpleNamespace(json=lambda *args: {"status": "unknown"})
    with pytest.raises(op.AdminError, match="unknown qualification state"):
        op._wait(client, ["order-q-7"], poll_seconds=30, timeout_seconds=1,
                 sleep=lambda seconds: pytest.fail("waited"), log=lambda line: None, clock=lambda: 0)


def test_invalid_job_status_is_a_typed_error():
    client = SimpleNamespace(json=lambda *args: {"manifest": None})
    with pytest.raises(op.AdminError, match="invalid evaluation job status"):
        op.grade_job(client, "order-eval-7")


def test_failed_staged_write_leaves_destination_unchanged(tmp_path, monkeypatch):
    from pathlib import Path

    original = Path.write_bytes

    def write(path, content):
        if path.name == "manifest.json":
            raise OSError("disk write failed")
        return original(path, content)

    monkeypatch.setattr(Path, "write_bytes", write)
    with _grade_http(_download_files()) as http:
        with pytest.raises(OSError, match="disk write failed"):
            op.grade_job(op.AdminClient("https://test.invalid", SECRET, http=http),
                         "order-eval-7", out=tmp_path / "result")
    assert list(tmp_path.iterdir()) == []


def test_manifest_cannot_name_unexpected_result_files(tmp_path):
    files = _download_files()
    manifest = json.loads(files["manifest.json"])
    manifest["files"][0]["name"] = ["report.json"]
    files["manifest.json"] = json.dumps(manifest).encode()
    with _grade_http(files) as http:
        with pytest.raises(ValueError, match="must describe both result files"):
            op.grade_job(op.AdminClient("https://test.invalid", SECRET, http=http),
                         "order-eval-7", out=tmp_path / "result")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("url", [
    "http://admin.example", "https://user:password@admin.example",
    "https://admin.example?option=1", "https://admin.example#section",
    "https://admin.example?", "https://admin.example#",
    "https://admin.example/service", "ftp://admin.example", "https://admin.example:0",
])
def test_administrator_origin_validation_applies_to_injected_transport(url):
    with httpx.Client(transport=httpx.MockTransport(lambda r: pytest.fail("unexpected request"))) as http:
        with pytest.raises(ValueError, match="HTTPS origin or HTTP loopback"):
            op.AdminClient(url, SECRET, http=http)


@pytest.mark.parametrize("url", ["https://admin.example", "http://127.0.0.1:8790",
                                  "http://localhost:8790", "http://[::1]:8790"])
def test_administrator_requests_use_the_validated_origin(url):
    requests = []
    with httpx.Client(base_url="https://different.example", transport=httpx.MockTransport(
            lambda request: requests.append(request.url) or httpx.Response(200, json={})
    )) as http:
        op.AdminClient(url, SECRET, http=http).json("GET", "/admin/v1/jobs")
    assert str(requests[0]) == f"{url}/admin/v1/jobs"


def test_administrator_redirect_is_not_followed_with_an_injected_client():
    requests = []

    def redirect(request):
        requests.append(request.url)
        return httpx.Response(307, headers={"location": "https://different.example"},
                              json={"detail": "redirect refused"})

    with httpx.Client(transport=httpx.MockTransport(redirect), follow_redirects=True) as http:
        with pytest.raises(op.AdminError) as raised:
            op.AdminClient("https://admin.example", SECRET, http=http).json("POST", "/admin/v1/jobs", {})
    assert raised.value.status == 307
    assert len(requests) == 1


def test_owned_administrator_client_disables_environment_proxies_and_redirects(monkeypatch):
    options = []
    monkeypatch.setattr(op.httpx, "Client", lambda **kwargs: options.append(kwargs) or SimpleNamespace())
    op.AdminClient("https://admin.example", SECRET)
    assert options[0]["trust_env"] is False
    assert options[0]["follow_redirects"] is False


def test_failed_download_sync_preserves_empty_destination(monkeypatch, tmp_path):
    import os

    destination = tmp_path / "result"
    destination.mkdir()
    monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("sync unavailable")))
    with _grade_http(_download_files()) as http:
        with pytest.raises(OSError, match="sync unavailable"):
            op.grade_job(op.AdminClient("https://test.invalid", SECRET, http=http),
                         "order-eval-7", out=destination)
    assert list(destination.iterdir()) == []
    assert list(tmp_path.iterdir()) == [destination]


def test_download_files_and_directory_are_synced_before_publication(monkeypatch, tmp_path):
    import os
    from pathlib import Path

    calls = []
    real_replace, real_sync = Path.replace, os.fsync

    def sync(fd):
        calls.append("sync")
        real_sync(fd)

    def replace(source, destination):
        assert calls == ["sync"] * 4
        calls.append("publish")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(Path, "replace", replace)
    with _grade_http(_download_files()) as http:
        op.grade_job(op.AdminClient("https://test.invalid", SECRET, http=http),
                     "order-eval-7", out=tmp_path / "result")
    assert calls == ["sync"] * 4 + ["publish", "sync"]
