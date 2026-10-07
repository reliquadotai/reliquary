"""A signed job's grades come from its transcripts; a validator wires the sandbox
services only from its own key and never mixes signed jobs into the replay pins."""

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

attest = pytest.importorskip("reliquary_sandbox.attest")

from reliquary.corpus.job import parse_job  # noqa: E402
from reliquary.environment.agentic_swe import SignedSweSource  # noqa: E402
from reliquary.infrastructure.corpus_record_store import RECORD_SCHEMA_V2  # noqa: E402
from reliquary.infrastructure.sandbox_store import MemorySessionStore  # noqa: E402
from reliquary.validator.sandbox_wiring import (  # noqa: E402
    SandboxValidatorConfig, build_sandbox_services, wire_signed_grader, wire_signed_job,
)
from reliquary.validator.signed_grading import SignedEpisodeGrader  # noqa: E402
from tests.unit.sandbox_fixtures import NOW, claims, signer, transcript  # noqa: E402
from tests.unit.test_corpus_job_episode import _manifest  # noqa: E402
from tests.unit.test_corpus_job_signed_sandbox import signed_episode  # noqa: E402

JOB = parse_job(_manifest(prompt_count=3, episode=signed_episode()))
SOURCE = SignedSweSource("train:20", prompt_of=lambda s, i: "p",
                         row_of=lambda s, i: (None, SimpleNamespace(instance_id=f"repo__{i}")))
VALIDATOR = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"      # //Bob, ss58 format 42


class Records:
    def __init__(self, submissions, verdicts=None):
        self.submissions, self.verdicts, self.grades = submissions, dict(verdicts or {}), {}

    async def read_submission(self, job_id, sid):
        return self.submissions.get(sid)

    async def read_verdict(self, job_id, sid):
        return self.verdicts.get(sid)

    async def write_grade(self, job_id, sid, document):
        self.grades[sid] = document
        return True

    async def list_grade_ids(self, job_id):
        return list(self.grades)

    async def list_submission_ids(self, job_id):
        return list(self.submissions)


def record(signed, received_at=NOW + 50.0):
    return {"schema": RECORD_SCHEMA_V2, "hotkey": "5Hot", "prompt_index": 0,
            "received_at": received_at, "completions": [{"transcript": signed}]}


def grade(records, sid="sub-1"):
    grader = SignedEpisodeGrader(job=JOB, records=records, source=SOURCE, clock=lambda: NOW + 60)
    return grader, asyncio.run(grader.grade_one(sid))


@pytest.mark.parametrize("reward,success", [(1.0, True), (0.0, False)])
def test_the_grade_is_the_final_records(tmp_path, reward, success):
    signed = transcript(signer(tmp_path, "v", "v1"), signer(tmp_path, "m", "k1"), claims(),
                        reward=reward)
    records = Records({"sub-1": record(signed)})
    grader, document = grade(records)
    assert document["status"] == "ok" and document["graded_success"] is success
    assert document["grade"]["reward"] == reward and document["grade"]["session_id"] == "s-1"
    assert document["graded_by"] == ["sandbox:machine-1"] and document["replay_certified"] is True
    assert document["instance_id"] == "repo__0"
    assert asyncio.run(grader.ready(["sub-1"])) == {"sub-1"}


def test_a_failed_audit_is_graded_audit_failed(tmp_path):
    signed = transcript(signer(tmp_path, "v", "v1"), signer(tmp_path, "m", "k1"), claims())
    records = Records({"sub-1": record(signed)}, verdicts={"sub-1": {"passed": False}})
    assert grade(records)[1]["status"] == "audit_failed"


def test_a_record_without_a_transcript_is_unparseable():
    assert grade(Records({"sub-1": record(None)}))[1]["status"] == "unparseable"


def test_an_ungraded_signed_submission_holds_its_period():
    grader = SignedEpisodeGrader(job=JOB, records=Records({"sub-1": record(None, NOW + 7.0)}),
                                 source=SOURCE, clock=lambda: NOW + 60)
    grader._enqueue = lambda sid: None                 # list it, do not grade it yet
    asyncio.run(grader.rescan_once())
    assert grader.oldest_unready_received_at() == NOW + 7.0


def test_the_config_needs_both_key_settings(tmp_path):
    assert SandboxValidatorConfig.from_env({}) is None
    with pytest.raises(ValueError):
        SandboxValidatorConfig.from_env({"RELIQUARY_SANDBOX_VALIDATOR_KEY_ID": "v1"})
    config = SandboxValidatorConfig.from_env({
        "RELIQUARY_SANDBOX_VALIDATOR_KEY_FILE": str(tmp_path / "v.pem"),
        "RELIQUARY_SANDBOX_VALIDATOR_KEY_ID": "v1",
        "RELIQUARY_SANDBOX_VALIDATOR_RETIRED_KEYS": '{"v0": "AAAA"}',
        "RELIQUARY_SANDBOX_MAX_LIVE_PER_HOTKEY": "3",
        "RELIQUARY_SANDBOX_CLAIM_TTL_S": "600"})
    assert config.key_id == "v1" and config.retired_keys == {"v0": "AAAA"}
    assert config.policy.max_live_per_hotkey == 3 and config.policy.claim_ttl_s == 600


def test_the_config_reads_the_directory_and_close_settings(tmp_path):
    keys = {"RELIQUARY_SANDBOX_VALIDATOR_KEY_FILE": str(tmp_path / "v.pem"),
            "RELIQUARY_SANDBOX_VALIDATOR_KEY_ID": "v1"}
    default = SandboxValidatorConfig.from_env(keys)
    assert default.directory_refresh_s == 30.0 and default.directory_max_age_s == 120.0
    assert default.directory_read_timeout_s == 15.0 and default.close_concurrency == 4
    config = SandboxValidatorConfig.from_env({
        **keys, "RELIQUARY_SANDBOX_DIRECTORY_REFRESH_S": "10",
        "RELIQUARY_SANDBOX_DIRECTORY_MAX_AGE_S": "45",
        "RELIQUARY_SANDBOX_DIRECTORY_READ_TIMEOUT_S": "5",
        "RELIQUARY_SANDBOX_CLOSE_CONCURRENCY": "2"})
    assert (config.directory_refresh_s, config.directory_max_age_s,
            config.directory_read_timeout_s, config.close_concurrency) == (10.0, 45.0, 5.0, 2)
    for name in ("RELIQUARY_SANDBOX_DIRECTORY_REFRESH_S", "RELIQUARY_SANDBOX_CLOSE_CONCURRENCY"):
        with pytest.raises(ValueError):
            SandboxValidatorConfig.from_env({**keys, name: "0"})


def test_the_config_never_shows_its_key_file_contents(tmp_path, caplog):
    from reliquary.validator import sandbox_wiring

    sandbox_wiring.logger.setLevel(logging.DEBUG)      # importing bittensor mutes loggers
    caplog.set_level(logging.DEBUG)
    validator, services = _services(tmp_path)
    asyncio.run(services.start())
    secret = Path(tmp_path / "v.pem").read_text().split("-----")[2].strip()
    assert "sandbox sessions restored" in caplog.text            # the capture is real
    assert secret and secret not in caplog.text and secret not in repr(services)
    assert validator.public_key_b64 == services.signer.public_key_b64


def _services(tmp_path, **config_kw):
    validator = signer(tmp_path, "v", "v1")
    config = SandboxValidatorConfig(Path(tmp_path / "v.pem"), "v1", {}, None, **config_kw)

    async def read_documents():
        return []

    async def fetch(address):
        return None

    return validator, build_sandbox_services(
        config, validator_hotkey=VALIDATOR, session_store=MemorySessionStore(),
        read_documents=read_documents, fetch_report=fetch)


def test_services_sign_with_the_configured_key_and_mount_the_route(tmp_path):
    validator, services = _services(tmp_path)
    assert services.signer.public_key_b64 == validator.public_key_b64
    paths = {route.path for route in services.router.routes}
    assert paths == {"/corpus/sandbox/sessions", "/corpus/sandbox/sessions/{session_id}/close"}
    assert services.fleet.on_drained == services.issuer.void_machine
    asyncio.run(services.start())
    background = services.background()
    assert len(background) == 2
    for coroutine in background:
        coroutine.close()


def test_the_fleet_takes_the_directory_settings(tmp_path):
    _, services = _services(tmp_path, directory_refresh_s=10.0, directory_max_age_s=45.0,
                            directory_read_timeout_s=5.0)
    fleet = services.fleet
    assert (fleet._refresh_s, fleet._max_age_s, fleet.directory_read_timeout_s) == (10.0, 45.0, 5.0)


def test_a_failed_restore_aborts_the_start(tmp_path):
    _, services = _services(tmp_path)

    class Broken:
        async def list_recent(self, now):
            raise OSError("R2 down")

    services.issuer._store = Broken()
    with pytest.raises(OSError):
        asyncio.run(services.start())


def test_the_background_loops_stop_on_cancel(tmp_path):
    _, services = _services(tmp_path)

    async def run():
        await services.start()
        tasks = [asyncio.ensure_future(c) for c in services.background()]
        await asyncio.sleep(0.05)
        for task in tasks:
            task.cancel()
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)
        return results

    assert all(isinstance(r, asyncio.CancelledError) for r in asyncio.run(run()))


def test_a_signed_job_is_wired_with_its_intake_view_and_grader(tmp_path):
    _, services = _services(tmp_path)
    built = {}

    def intake_factory(job, **kw):
        built.update(kw)
        return SimpleNamespace(source=SOURCE)

    class Router:
        async def slots_remaining(self, index):
            return 2

    routes = SimpleNamespace(routers={JOB.job_id: Router()})
    w = SimpleNamespace(job=JOB, is_banned=None)
    wire_signed_job(w, services=services, routes=lambda: routes, checkpoint_dir="/ck",
                    tokenizer=None, vocab_size=None, chunk_tokens=32, intake_factory=intake_factory,
                    resolver_factory=lambda split: SimpleNamespace(resolve=None))
    assert built["sessions"] is services.issuer
    assert built["seen"] == services.book.submitted_ids
    assert built["directory"] == services.fleet.directory_if_ready
    assert built["token_verifier"] is services.token_verifier
    assert built["retry_after_s"] == services.issuer.policy.retry_after_s
    view = services.jobs[JOB.job_id]
    assert view.job is JOB and asyncio.run(view.slots_remaining(0)) == 2
    wire_signed_grader(w, judge_records=Records({}))
    assert isinstance(w.grader, SignedEpisodeGrader)


def test_a_replay_validator_never_imports_the_sandbox():
    import subprocess
    import sys

    code = ("import sys, reliquary.validator.corpus_validator, reliquary.validator.corpus_service, "
            "reliquary.corpus.admission, reliquary.corpus.delivery, reliquary.miner.agentic_miner; "
            "print(any(m.startswith('reliquary_sandbox') for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=Path(__file__).resolve().parents[2])
    assert out.stdout.strip().splitlines()[-1] == "False"


def test_signed_jobs_never_join_the_replay_grade_pins():
    from reliquary.validator.corpus_validator import replay_episode_pins

    replay = parse_job(_manifest())
    single_turn = parse_job(_manifest(
        with_episode=False, prompt_source="openmathinstruct", renderer_id="x-v1",
        sampling={"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_new_tokens": 2,
                  "max_new_tokens": 4096, "n": 1}))
    wiring = [SimpleNamespace(job=JOB), SimpleNamespace(job=replay),
              SimpleNamespace(job=single_turn)]
    assert replay_episode_pins(wiring) == {("reliquary-swe", "a" * 40)}


# -- fix round 1 ---------------------------------------------------------------


def test_a_signed_grade_is_certified_for_the_export(tmp_path):
    from reliquary.corpus.delivery import certified, held_by_quarantine

    signed = transcript(signer(tmp_path, "v", "v1"), signer(tmp_path, "m", "k1"), claims())
    _, document = grade(Records({"sub-1": record(signed)}))
    assert document["replay"] == {"signed": True, "status": "ok", "certified": True}
    assert certified(document)
    assert held_by_quarantine(document, {"executor-1", "machine-1"}) == []
    failed = grade(Records({"sub-1": record(signed)}, verdicts={"sub-1": {"passed": False}}))[1]
    assert not certified(failed)


def test_a_signed_grade_round_trips_through_the_export(tmp_path):
    from reliquary.corpus.delivery import episode_rows
    from tests.unit.test_corpus_export_signed import build

    trajectory, renderer = build(tmp_path)
    stored = {"schema": RECORD_SCHEMA_V2, "hotkey": "5Hot", "prompt_index": 0,
              "received_at": NOW + 50.0, "completions": [trajectory]}

    class Exported(Records):
        async def list_verdict_ids(self, job_id):
            return ["sub-1"]

        async def list_voided_ids(self, job_id):
            return []

        async def read_regrade(self, job_id, sid):
            return None

        async def read_grade(self, job_id, sid):
            return self.grades.get(sid)

    records = Exported({"sub-1": stored}, verdicts={"sub-1": {"passed": True}})
    source = SignedSweSource("train:20", prompt_of=lambda s, i: "Fix task 0.",
                             row_of=lambda s, i: (None, SimpleNamespace(instance_id=f"repo__{i}")))
    grader = SignedEpisodeGrader(job=JOB, records=records, source=source, clock=lambda: NOW + 60)
    assert asyncio.run(grader.grade_one("sub-1"))["status"] == "ok"

    async def export():
        counts = {}
        rows = [r async for r in episode_rows(job=JOB, records=records, renderer=renderer,
                                               source=source, counts=counts, quarantined=())]
        return rows, counts

    rows, counts = asyncio.run(export())
    assert counts["rows"] == 1 and counts["uncertified"] == 0 and counts["held"] == 0
    assert rows[0]["graded_success"] is True and rows[0]["replay_certified"] is True


def test_the_grader_parses_transcripts_on_its_parse_executor(tmp_path, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from reliquary.validator import signed_grading

    seen = []
    real = signed_grading.signed_records

    def spy(transcript_):
        seen.append(threading.current_thread().name)
        return real(transcript_)

    monkeypatch.setattr(signed_grading, "signed_records", spy)
    pool = ThreadPoolExecutor(1, thread_name_prefix="corpus-grade-parse")
    signed = transcript(signer(tmp_path, "v", "v1"), signer(tmp_path, "m", "k1"), claims())
    grader = SignedEpisodeGrader(job=JOB, records=Records({"sub-1": record(signed)}),
                                 source=SOURCE, clock=lambda: NOW + 60, parse_executor=pool)
    assert asyncio.run(grader.grade_one("sub-1"))["status"] == "ok"
    pool.shutdown()
    assert seen and seen[0].startswith("corpus-grade-parse")


@pytest.mark.parametrize("name,value", [
    ("RELIQUARY_SANDBOX_DIRECTORY_REFRESH_S", "nan"),
    ("RELIQUARY_SANDBOX_DIRECTORY_MAX_AGE_S", "inf"),
    ("RELIQUARY_SANDBOX_CLOSE_BODY_TIMEOUT_S", "-1"),
    ("RELIQUARY_SANDBOX_VALIDATOR_KEY_ID", "  "),
])
def test_the_config_refuses_non_finite_settings_and_a_blank_key_id(tmp_path, name, value):
    keys = {"RELIQUARY_SANDBOX_VALIDATOR_KEY_FILE": str(tmp_path / "v.pem"),
            "RELIQUARY_SANDBOX_VALIDATOR_KEY_ID": "v1"}
    with pytest.raises(ValueError):
        SandboxValidatorConfig.from_env({**keys, name: value})


def test_the_close_body_timeout_is_a_setting(tmp_path):
    keys = {"RELIQUARY_SANDBOX_VALIDATOR_KEY_FILE": str(tmp_path / "v.pem"),
            "RELIQUARY_SANDBOX_VALIDATOR_KEY_ID": "v1"}
    assert SandboxValidatorConfig.from_env(keys).close_body_timeout_s == 30.0
    assert SandboxValidatorConfig.from_env(
        {**keys, "RELIQUARY_SANDBOX_CLOSE_BODY_TIMEOUT_S": "5"}).close_body_timeout_s == 5.0


class FlakyStore(MemorySessionStore):
    def __init__(self, failures, hang=False):
        super().__init__()
        self.failures, self.hang, self.calls = failures, hang, 0

    async def list_recent(self, now):
        self.calls += 1
        if self.calls <= self.failures:
            if self.hang:
                await asyncio.sleep(3600)
            raise OSError("R2 down")
        return await super().list_recent(now)


def _flaky_services(tmp_path, store):
    _, services = _services(tmp_path)
    services.issuer._store = store
    services.restore_backoff_s = 0.0
    services.restore_timeout_s = 0.05
    return services


def test_restore_is_retried_before_the_start_aborts(tmp_path):
    store = FlakyStore(failures=2)
    asyncio.run(_flaky_services(tmp_path, store).start())
    assert store.calls == 3


def test_a_hung_restore_times_out_and_is_retried(tmp_path):
    store = FlakyStore(failures=1, hang=True)
    asyncio.run(_flaky_services(tmp_path, store).start())
    assert store.calls == 2


def test_a_restore_that_keeps_failing_aborts_the_start(tmp_path):
    store = FlakyStore(failures=99)
    services = _flaky_services(tmp_path, store)
    with pytest.raises(OSError):
        asyncio.run(services.start())
    assert store.calls == services.restore_attempts


def test_a_failed_directory_read_at_start_does_not_abort(tmp_path):
    validator = signer(tmp_path, "v", "v1")
    config = SandboxValidatorConfig(Path(tmp_path / "v.pem"), "v1", {}, None)
    reads = []

    async def broken():
        reads.append(1)
        raise OSError("R2 down")

    async def fetch(address):
        return None

    services = build_sandbox_services(config, validator_hotkey=VALIDATOR,
                                      session_store=MemorySessionStore(),
                                      read_documents=broken, fetch_report=fetch)
    asyncio.run(services.start())                      # logged, retried by the fleet
    assert reads == [1] and services.fleet._next_refresh_at is not None
    assert services.signer.public_key_b64 == validator.public_key_b64


def test_stop_awaits_the_void_persists_within_its_bound(tmp_path):
    _, services = _services(tmp_path)

    async def run():
        written = []

        async def slow_update(document):
            await asyncio.sleep(0.05)
            written.append(document["session_id"])

        async def stuck_update(document):
            await asyncio.sleep(3600)

        services.issuer._store.update = slow_update
        task = asyncio.get_running_loop().create_task(slow_update({"session_id": "s-1"}))
        services.issuer._tasks.add(task)
        await services.stop(timeout=5)
        assert written == ["s-1"]
        stuck = asyncio.get_running_loop().create_task(stuck_update({}))
        services.issuer._tasks.add(stuck)
        started = asyncio.get_running_loop().time()
        await services.stop(timeout=0.1)
        assert asyncio.get_running_loop().time() - started < 2 and stuck.cancelled()

    asyncio.run(run())


def _wired(tmp_path, routes):
    _, services = _services(tmp_path)
    w = SimpleNamespace(job=JOB)

    async def resolve(index):
        raise AssertionError("not reached")

    wire_signed_job(w, services=services, routes=lambda: routes, checkpoint_dir="/ck",
                    tokenizer=None, vocab_size=None, chunk_tokens=32,
                    intake_factory=lambda job, **kw: SimpleNamespace(source=SOURCE),
                    resolver_factory=lambda split: SimpleNamespace(resolve=resolve))
    return services


def _terms(services, index=0):
    engagement = {"kind": "corpus", "job_id": JOB.job_id, "prompt_index": index}
    return asyncio.run(services.issuer._engagements["corpus"].terms("5Hot", engagement))


def test_a_job_not_adopted_yet_is_retryable(tmp_path):
    from reliquary.sandbox.routes import REFUSAL_STATUS

    services = _wired(tmp_path, SimpleNamespace(routers={}, retired=set()))
    refusal = _terms(services)
    assert refusal.reason == "job_not_ready" and refusal.retry_after
    assert REFUSAL_STATUS["job_not_ready"] == 503
    assert _terms(_wired(tmp_path, None)).reason == "job_not_ready"


def test_a_retired_job_loses_its_session_view(tmp_path):
    class Router:
        async def slots_remaining(self, index):
            return 0

    routes = SimpleNamespace(routers={JOB.job_id: Router()}, retired={JOB.job_id})
    services = _wired(tmp_path, routes)
    assert _terms(services).reason == "job_not_served"
    assert JOB.job_id not in services.jobs


def test_a_failed_job_can_be_forgotten(tmp_path):
    services = _wired(tmp_path, None)
    services.forget(JOB.job_id)
    assert JOB.job_id not in services.jobs and _terms(services).reason == "job_not_served"


def test_the_pre_auth_close_limit_is_a_setting_passed_to_the_route(tmp_path, monkeypatch):
    from reliquary.sandbox import routes

    keys = {"RELIQUARY_SANDBOX_VALIDATOR_KEY_FILE": str(tmp_path / "v.pem"),
            "RELIQUARY_SANDBOX_VALIDATOR_KEY_ID": "v1"}
    assert SandboxValidatorConfig.from_env(keys).close_preauth_concurrency == 8
    assert SandboxValidatorConfig.from_env(
        {**keys, "RELIQUARY_SANDBOX_CLOSE_PREAUTH_CONCURRENCY": "3"}).close_preauth_concurrency == 3
    with pytest.raises(ValueError):
        SandboxValidatorConfig.from_env({**keys, "RELIQUARY_SANDBOX_CLOSE_PREAUTH_CONCURRENCY": "0"})
    built = {}
    real = routes.build_sandbox_sessions_router

    def spy(*args, **kw):
        built.update(kw)
        return real(*args, **kw)

    monkeypatch.setattr(routes, "build_sandbox_sessions_router", spy)
    _services(tmp_path, close_preauth_concurrency=3)
    assert built["max_preauth_closes"] == 3


def test_the_runbook_lists_every_sandbox_setting_with_its_default():
    """M6: every RELIQUARY_SANDBOX_* setting the code reads is in the runbook's table."""
    import re

    root = Path(__file__).resolve().parents[2]
    names = set()
    for path in (root / "reliquary").rglob("*.py"):
        names |= set(re.findall(r"RELIQUARY_SANDBOX_[A-Z_0-9]+", path.read_text()))
    runbook = (root / "docs/runbooks/agentic-corpus-swe.md").read_text()
    rows = {m.group(1): m.group(2) for m in re.finditer(
        r"^\| `(RELIQUARY_SANDBOX_[A-Z_0-9]+)` \| ([^|]+) \|", runbook, re.M)}
    assert names and names <= set(rows), sorted(names - set(rows))
    assert all(default.strip() for default in rows.values())


def test_a_signed_jobs_view_is_published_only_once_its_wiring_completed(tmp_path):
    """The session issuer never sees a job whose wiring is still in progress (or later
    fails): the validator wires with publish=False and publishes after the job's
    grader and auditor are wired."""
    _, services = _services(tmp_path)
    w = SimpleNamespace(job=JOB, is_banned=None)
    wire_signed_job(w, services=services, routes=lambda: None, checkpoint_dir="/ck",
                    tokenizer=None, vocab_size=None, chunk_tokens=32,
                    intake_factory=lambda job, **kw: SimpleNamespace(source=SOURCE),
                    resolver_factory=lambda split: SimpleNamespace(resolve=None), publish=False)
    assert services.jobs == {} and w.sandbox_view.job is JOB
    services.publish(w)
    assert services.jobs[JOB.job_id] is w.sandbox_view


def test_the_validator_publishes_a_signed_view_only_after_audit_and_settle():
    """Source pin: both the startup loop and the hot add wire with publish=False and
    publish after `audit_and_settle` succeeded."""
    import inspect

    from reliquary.validator import corpus_validator

    source = inspect.getsource(corpus_validator)
    assert "publish=False" in source
    assert source.count("sandbox_services.publish(w)") == 2
