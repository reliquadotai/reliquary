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
