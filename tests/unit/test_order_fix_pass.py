"""Dataset orders on any model, review fix pass: refusals before any executor
is rented, a control that trusts no registry entry, miners that route without
configuration, and the sampled audit of a generation order on the GPU-less
control, through executor pairs, up to its settlement."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from reliquary.eval import qualification as qual
from tests.unit.test_admin_eval_jobs import admin  # noqa: F401
from tests.unit.test_eval_control_process import world  # noqa: F401
from tests.unit.test_jobs_cli import registry  # noqa: F401
from tests.unit.test_order_any_model import (
    GEN_ENV,
    GEN_THRESHOLDS,
    _gen_job,
    _gen_qualification,
    _qualify_gen,
    stub_catalog_env,
)


# -- I1. architecture and loadability at qualification creation -------------------


@pytest.mark.parametrize("kind", ["eval", "generation"])
@pytest.mark.parametrize("facts,status,detail", [
    ({"architecture": "MambaForCausalLM", "eos_token_id": 2}, 422, "architecture_unsupported"),
    (ValueError("no config.json"), 422, "model_files_unreadable"),
    (ConnectionError("hub down"), 503, "model_files_unavailable"),
])
def test_a_qualification_is_refused_at_creation_before_any_executor(
        admin, monkeypatch, kind, facts, status, detail):  # noqa: F811
    from tests.unit.test_admin_eval_jobs import _qualification

    stub_catalog_env(monkeypatch)
    admin.facts["customer/Bad"] = facts
    body = (_gen_qualification(model="customer/Bad") if kind == "generation"
            else _qualification(model="customer/Bad"))
    response = admin("POST", "/admin/v1/qualifications", body)
    assert response.status_code == status, response.text
    assert detail in response.text
    record, _ = asyncio.run(qual.QualificationStore().read(body["qualification_id"]))
    assert record is None  # nothing queued: no executor is ever rented for it


def test_model_facts_turns_missing_files_into_a_refusal(tmp_path, monkeypatch):
    from reliquary.validator import eval_control

    monkeypatch.setattr(eval_control, "_model_files", lambda repo, revision: str(tmp_path))
    with pytest.raises(ValueError, match="config.json"):
        eval_control.model_facts("adapter/only", "r" * 40)
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"],
                                                      "vocab_size": 10}))
    # A config and no tokenizer: not loadable either, never retried forever.
    with pytest.raises(ValueError, match="tokenizer"):
        eval_control.model_facts("adapter/only", "r" * 40)
    with pytest.raises(ValueError, match="tokenizer"):
        eval_control.load_cpu_tokenizer("adapter/only", "r" * 40)


def test_a_pending_qualification_expires(monkeypatch):
    from reliquary.infrastructure import corpus_job_store as job_store
    from tests.unit.test_corpus_job_store import _FakeMultiObjectR2
    from tests.unit.test_eval_qualification import _executor, _queue, _request

    fake = _FakeMultiObjectR2()
    monkeypatch.setattr(job_store, "get_s3_client", lambda **kw: fake)
    store = qual.QualificationStore()
    now = [0.0]
    queue = _queue(store, now)
    _request(store)  # requested at 0
    asyncio.run(queue.refresh())
    assert qual.qualification_expiry_seconds() == 6 * 3600
    now[0] = 6 * 3600 - 1
    asyncio.run(queue.refresh())
    assert asyncio.run(store.read("order-q1"))[0]["status"] == qual.PENDING
    now[0] = 6 * 3600
    # Expired with no executor at all: nothing is billed forever.
    asyncio.run(queue.refresh())
    record, _ = asyncio.run(store.read("order-q1"))
    assert record["status"] == qual.FAILED
    assert record["result"] == {"failed_reason": "qualification_expired"}
    assert asyncio.run(queue.claim(_executor("e1"))) is None
    monkeypatch.setenv("RELIQUARY_QUALIFY_EXPIRY_SECONDS", "60")
    assert qual.qualification_expiry_seconds() == 60.0


# -- I2. the whole range at qualification creation ---------------------------------


def test_a_generation_range_past_the_source_is_refused_at_qualification(admin, monkeypatch):  # noqa: F811
    stub_catalog_env(monkeypatch, rows=1000)
    past = admin("POST", "/admin/v1/qualifications", _gen_qualification(prompt_start=900))
    assert past.status_code == 422 and "1000" in past.text
    fits = admin("POST", "/admin/v1/qualifications", _gen_qualification(
        prompt_start=500, problems=500))
    assert fits.status_code == 201, fits.text


# -- I3. the control trusts no registry entry; only the admin declares orders ------


def test_the_order_env_set_is_pinned():
    from reliquary.eval.qualification import ORDER_ENVIRONMENTS, order_environment_refusal

    assert ORDER_ENVIRONMENTS == frozenset({
        "reliquary_logic_v2", "reliquary_dapo_math_v1",
        "reliquary_instruction_following_v1", "reliquary_code_v1"})
    # A packaged single-turn source outside the list is not enabled by itself.
    assert order_environment_refusal("reliquary_telecom_solo_v1") is not None


def _declared_gen(admin, monkeypatch):  # noqa: F811
    from reliquary.infrastructure import corpus_job_store as job_store

    stub_catalog_env(monkeypatch)
    _qualify_gen(admin)
    created = admin("POST", "/admin/v1/jobs", _gen_job())
    assert created.status_code == 201, created.text
    job, _ = asyncio.run(job_store.read_job("order-gen-11"))
    entry = admin.registry["entries"]["order-gen-11"]
    declared = asyncio.run(qual.EvalJobStore().read("order-gen-11"))
    record, _ = asyncio.run(qual.QualificationStore().read("order-gq1"))
    return entry, job, declared, record


def test_a_generation_order_is_recorded_with_its_qualification(admin, monkeypatch):  # noqa: F811
    _, _, declared, _ = _declared_gen(admin, monkeypatch)
    assert declared == {
        "schema": qual.ORDER_JOB_SCHEMA, "job_id": "order-gen-11", "kind": "generation",
        "qualification_id": "order-gq1", "env": GEN_ENV, "prompt_start": 100,
        "problems": 500, "samples": 2,
        "sampling": {"temperature": 0.7, "top_p": 0.95, "top_k": 0},
        "max_new_tokens": 1024, "thinking": True}


def test_the_control_wires_only_what_its_qualification_backs(admin, monkeypatch):  # noqa: F811
    from reliquary.validator.eval_control import qualification_refusal

    entry, job, declared, record = _declared_gen(admin, monkeypatch)
    assert qualification_refusal(entry, job, declared, record) is None
    assert "no order record" in qualification_refusal(entry, job, None, record)
    assert "qualification" in qualification_refusal(entry, job, declared, None)
    for change, word in [
            ({"status": "pending"}, "qualified"),
            ({"kind": None}, "kind"),
            ({"revision": "0" * 40}, "model"),
            ({"prompt_start": 0}, "prompt_start"),
            ({"environment_manifest_sha256": "0" * 64}, "package"),
            ({"result": {**record["result"], "architecture": "MambaForCausalLM"}}, "architecture"),
            ({"result": {**record["result"], "checkpoint_sha256": "0" * 64}}, "checkpoint"),
            ({"result": {**record["result"], "eos_token_id": 1}}, "eos"),
            ({"result": {**record["result"], "thresholds": {
                **GEN_THRESHOLDS, "exp_mismatch_threshold": 99}}}, "thresholds")]:
        assert word in qualification_refusal(entry, job, declared, {**record, **change}), change
    # The manifest repeats the order's conditions.
    assert "sampling" in qualification_refusal(
        entry, replace(job, sampling=replace(job.sampling, temperature=1.0)), declared, record)
    # M1: the package this control loads is the one the order was qualified on.
    contract = json.loads(json.dumps(entry.contract))
    contract["environments"][GEN_ENV]["environment_manifest_sha256"] = "0" * 64
    assert "package" in qualification_refusal(replace(entry, contract=contract), job, declared,
                                              record)


def test_the_control_refuses_an_unsupported_architecture_whatever_wrote_the_entry(monkeypatch):
    from reliquary.protocol import profiles
    from reliquary.validator.eval_control import order_job_refusal
    from tests.unit.test_order_any_model import _gen_entry_and_job

    stub_catalog_env(monkeypatch)
    entry, job = _gen_entry_and_job()
    monkeypatch.setattr(profiles, "profile_from_contract", lambda c: SimpleNamespace(
        model_id=c["model_id"], model_revision=c["model_revision"], proofs=c["proofs"],
        model_architecture=c.get("model_architecture")))
    monkeypatch.setattr(profiles, "toploc_proof", lambda p: SimpleNamespace(mode="enforce"))
    entry.contract["model_architecture"] = "MambaForCausalLM"
    assert "architecture" in order_job_refusal(entry, job)
    entry.contract["model_architecture"] = "Qwen3ForCausalLM"
    assert order_job_refusal(entry, job) is None


@pytest.mark.parametrize("job_id", ["order-gen-5", "order-eval-5"])
def test_jobs_create_refuses_order_ids(job_id, monkeypatch):
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    result = CliRunner().invoke(app, [
        "jobs", "create", "--job-id", job_id, "--model", "m", "--model-revision", "r",
        "--model-architecture", "Qwen3ForCausalLM", "--checkpoint-sha256", "a" * 64,
        "--env", GEN_ENV, "--prompt-count", "4", "--renderer-id", "chat-template-v1",
        "--eos-token-id", "2", "--slots-per-prompt", "1", "--cap", "0.01",
        "--fleet-knows-corpus-generation"])
    assert result.exit_code != 0
    assert "admin" in result.output


@pytest.mark.parametrize("task_id", ["order-gen-x", "order-eval-x"])
def test_an_order_task_id_needs_an_order_job_id(admin, monkeypatch, task_id):  # noqa: F811
    stub_catalog_env(monkeypatch)
    response = admin("POST", "/admin/v1/jobs", {
        "job_id": "order-foo", "task_id": task_id, "model": "customer/Gen-8B",
        "env": GEN_ENV, "prompt_count": 5, "samples_per_prompt": 1, "cap": 0.01})
    assert response.status_code == 422 and "order" in response.text


def test_a_refused_order_entry_is_named_in_the_control_status(monkeypatch):
    from reliquary.protocol import profiles
    from reliquary.validator.eval_control import PairedAuditDispatcher, build_order_control
    from tests.unit.test_order_any_model import _gen_entry_and_job

    stub_catalog_env(monkeypatch)
    monkeypatch.setattr(profiles, "profile_from_contract", lambda c: SimpleNamespace(
        model_id=c["model_id"], model_revision=c["model_revision"], proofs=c["proofs"],
        model_architecture=c.get("model_architecture")))
    monkeypatch.setattr(profiles, "toploc_proof", lambda p: SimpleNamespace(mode="enforce"))
    entry, job = _gen_entry_and_job()
    entry = SimpleNamespace(task_id="order-gen-1", job_id="order-gen-1", status="active",
                            mechanism="corpus-generation",
                            params={"cap": 0.02, "settlement": "period-ema-v1"},
                            contract={**entry.contract, "model_architecture": "Mamba"})

    async def read_entries():
        return {"order-gen-1": entry}

    class _Store:
        async def read_job(self, job_id):
            return job, None

    async def go():
        dispatcher = PairedAuditDispatcher(directory=SimpleNamespace(revoke_locally=None))
        app, job_set = build_order_control(
            store=_Store(), records=object(), dispatcher=dispatcher,
            directory=SimpleNamespace(), verify_signature=lambda r: True,
            read_entries=read_entries, qualification_store=object(), order_jobs=object())
        await job_set.refresh()
        assert job_set.served == {}
        status = await app.state.control_status()
        reasons = status["jobs"]["order-gen-1"]["reasons"]
        assert status["jobs"]["order-gen-1"]["needs_attention"] is True
        assert any("refused" in r and "architecture" in r for r in reasons)

    asyncio.run(go())


# -- I4. miners route without configuration; one source for the nginx regex --------


def _miner_posts(job, job_id):
    """Where a miner set up as `reliquary corpus mine` does posts a submission."""
    from reliquary.miner.corpus_miner import HttpCorpusClient, submits_scoped

    paths = []

    class _Http:
        def post(self, path, json=None):
            paths.append(path)
            return SimpleNamespace(status_code=200, json=lambda: {"ok": True},
                                   raise_for_status=lambda: None)

    client = HttpCorpusClient(_Http(), job_id=job_id)
    client.scoped_submit = submits_scoped(job)
    client.submit({})
    return paths


def test_a_miner_on_an_operator_corpus_job_keeps_the_legacy_submit(admin, monkeypatch):  # noqa: F811
    """Deployed corpus controls answer 404 on the scoped submit: a prod job
    (no marker in its manifest) is submitted on /corpus/submit, --job-id or not."""
    from tests.unit.test_corpus_export import _job_spec

    prod = _job_spec(job_id="code-qwen38-27b-v1", prompt_source="reliquary_code_v1")
    assert _miner_posts(prod, "code-qwen38-27b-v1") == ["/corpus/submit"]
    assert _miner_posts(prod, None) == ["/corpus/submit"]
    # An order job declared by the admin carries the marker: its own route.
    entry, job, _, _ = _declared_gen(admin, monkeypatch)
    assert job.submit == "scoped"
    assert _miner_posts(job, job.job_id) == [f"/corpus/jobs/{job.job_id}/submit"]


def test_a_manifest_without_the_marker_stores_and_parses_as_before():
    from reliquary.corpus.job import JobError, parse_job
    from tests.unit.test_corpus_export import _job_spec

    plain = _job_spec().to_contract()
    assert "submit" not in plain and parse_job(plain).submit is None
    assert parse_job({**plain, "submit": "scoped"}).submit == "scoped"
    with pytest.raises(JobError):
        parse_job({**plain, "submit": "legacy"})


def test_the_order_control_refuses_a_generation_job_without_the_marker(monkeypatch):
    from reliquary.protocol import profiles
    from reliquary.validator.eval_control import order_job_refusal
    from tests.unit.test_order_any_model import _gen_entry_and_job

    stub_catalog_env(monkeypatch)
    monkeypatch.setattr(profiles, "profile_from_contract", lambda c: SimpleNamespace(
        model_id=c["model_id"], model_revision=c["model_revision"], proofs=c["proofs"],
        model_architecture=c.get("model_architecture")))
    monkeypatch.setattr(profiles, "toploc_proof", lambda p: SimpleNamespace(mode="enforce"))
    entry, job = _gen_entry_and_job()
    assert order_job_refusal(entry, job) is None
    assert "scoped" in order_job_refusal(entry, replace(job, submit=None))


@pytest.mark.parametrize("prefix", ["order-", "sn81-", "a.b-"])
def test_the_order_routes_follow_the_configured_prefix(prefix):
    import re

    from reliquary.eval.prompt_source import order_jobs_route_regex, order_routes_nginx

    pattern = re.compile(order_jobs_route_regex(prefix))
    assert pattern.match(f"/corpus/jobs/{prefix}gen-1/submit")
    assert pattern.match(f"/corpus/jobs/{prefix}eval-1/job")
    assert not pattern.match("/corpus/jobs/code-qwen38-27b-v1/submit")
    assert not pattern.match(f"/corpus/jobs/{prefix}other-1/job")
    snippet = order_routes_nginx(prefix, port=8791)
    assert order_jobs_route_regex(prefix) in snippet
    assert "^/corpus/internal/eval-audit/" in snippet and "127.0.0.1:8791" in snippet


def test_the_nginx_snippet_is_printed_from_the_prefix_and_documented(monkeypatch):
    from pathlib import Path

    from typer.testing import CliRunner

    from reliquary.cli.main import app

    monkeypatch.setenv("RELIQUARY_ADMIN_TASK_PREFIX", "sn81-")
    result = CliRunner().invoke(app, ["corpus", "order-nginx", "--port", "8791"])
    assert result.exit_code == 0 and "sn81-" in result.output
    text = Path("docs/design/2026-10-01-evaluation-on-subnet-design.md").read_text()
    assert "reliquary corpus order-nginx" in text


# -- I5. the sampled audit of a generation order, through executor pairs ------------


class _PairRemote:
    """The pair dispatcher's answer: every audited batch scored by two
    agreeing executors; with ``fail_all`` every record scores far over the
    thresholds."""

    fail_all = False

    def __init__(self):
        self.scored: list[int] = []

    def connected(self):
        return True

    def subscribe(self, listener):
        pass

    async def score(self, items):
        from reliquary.protocol.toploc import ChunkResult

        self.scored.append(len(items))
        out = []
        for item in items:
            exp = 999 if self.fail_all else 0
            out.append(("ok", tuple(ChunkResult(exp, 0.0, 0.0) for _ in item["proofs"]),
                        ("e1", "e2")))
        return out


def _gen_auditor(records, states, clock, remote, **params):
    from reliquary.corpus.audit_policy import AuditParams
    from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF
    from reliquary.validator.eval_control import eval_auditor
    from tests.unit.test_corpus_auditor import _Tokenizer
    from tests.unit.test_corpus_judge import HOLD, Q, _Beacon, _round_at

    return eval_auditor(
        job_id="order-gen-1", records=records, tokenizer=_Tokenizer(), proof=PROOF,
        vocab_size=1000, remote=remote,
        params=AuditParams(q=Q, hold_seconds=HOLD, ban_after_failures=params.get("ban", 10)),
        miner_states=states, beacon=_Beacon(), round_at=_round_at, clock=clock,
        accept_slack_seconds=0.0)


def test_a_generation_order_samples_its_audits_on_the_gpu_less_control():
    from tests.unit.test_corpus_auditor import _Records
    from tests.unit.test_corpus_judge import (
        HK, HOLD, Q, SAMPLED, T0, _Clock, _ids, _rec, _round_at, _States,
    )

    drawn, undrawn = _ids(True, 2), _ids(False, 3)
    records = _Records({sid: _rec(0) for sid in drawn + undrawn})
    states = _States({HK: SAMPLED})
    clock = _Clock(T0 + 10)
    remote = _PairRemote()
    auditor = _gen_auditor(records, states, clock, remote)
    assert auditor.reopen_slots is False
    asyncio.run(auditor.judge_many(drawn + undrawn))
    # Drawn: audited at once, by the pair. Undrawn: wait out the hold.
    assert set(records.verdicts) == set(drawn)
    for sid in drawn:
        v = records.verdicts[sid]
        assert (v["passed"], v["audited"], v["scored_by"]) == (True, True, ["e1", "e2"])
        assert v["draw"] == {"round": _round_at(T0) + 1, "q": Q, "drawn": True}
    scored = sum(remote.scored)
    clock.now = T0 + HOLD
    asyncio.run(auditor.judge_many(undrawn))
    for sid in undrawn:
        v = records.verdicts[sid]
        assert (v["passed"], v["audited"]) == (True, False)
    assert sum(remote.scored) == scored  # passed unaudited: no executor asked


def test_a_confirmed_failure_makes_the_hotkey_suspect_and_audits_its_held_records():
    from tests.unit.test_corpus_auditor import _Records
    from tests.unit.test_corpus_judge import HK, SAMPLED, T0, _Clock, _ids, _rec, _States

    hit, held = _ids(True, 1)[0], _ids(False, 3)
    records = _Records({sid: _rec(1) for sid in [hit, *held]})
    states = _States({HK: SAMPLED})
    remote = _PairRemote()
    remote.fail_all = True
    auditor = _gen_auditor(records, states, _Clock(T0 + 10), remote)
    asyncio.run(auditor.judge_many(held))
    assert records.verdicts == {}
    asyncio.run(auditor.judge_many([hit]))
    assert set(records.verdicts) == {hit, *held}
    assert all(v["passed"] is False and v["audited"] is True
               and v["scored_by"] == ["e1", "e2"] for v in records.verdicts.values())
    state = states.states[HK]
    assert len(state.confirmed_failures) == 4 and state.suspect_until == T0 + 10 + 86400


def test_a_ban_voids_a_generation_orders_pending_records():
    from reliquary.corpus.audit_policy import MinerState
    from tests.unit.test_corpus_auditor import _Records
    from tests.unit.test_corpus_judge import HK, T0, _Clock, _ids, _rec, _States

    held = _ids(False, 2)
    records = _Records({sid: _rec(0) for sid in held})
    states = _States({HK: MinerState(audited_passed=150, banned_until=T0 + 500)})
    remote = _PairRemote()
    auditor = _gen_auditor(records, states, _Clock(T0 + 10), remote)
    asyncio.run(auditor.judge_many(held))
    assert all(records.verdicts[sid]["reason"] == "banned" for sid in held)
    assert remote.scored == []


# sha256 of the canonical (archive, settlement state) of the run below, on the
# period settler that pays every corpus task since 2026-10-08.
GEN_SETTLEMENT_GOLDEN = "7b6222ab6128d107f306f9e988b1f4d4d715269e814e7c785e5b12244b1c8566"


def test_a_generation_order_settles_through_the_order_archives():
    from reliquary.validator.eval_control import OrderArchives
    from tests.unit.test_corpus_settlement import WORK, _Archives, _Records, _settler, _v

    # Audited passes, an unaudited pass after its hold, and a failure.
    verdicts = {"1" * 64: {**_v("A", 10), "audited": True},
                "2" * 64: {**_v("B", 30), "audited": False},
                "3" * 64: {**_v("C", 900, ok=False), "audited": True}}
    records = _Records(verdicts)
    guard = OrderArchives(served=lambda: {"order-gen-1"})

    class _Guarded(_Archives):
        async def write(self, task_id, work, entry, document):
            guard.refuse_unserved(task_id)
            await super().write(task_id, work, entry, document)

    archives = _Guarded()
    settler = _settler(records, archives, task_id="order-gen-1", job_id="order-gen-1", cap=0.02)
    assert asyncio.run(settler.settle_once()) == WORK
    archive = archives.written[WORK]
    assert archive["rewards_by_hotkey"] == pytest.approx({"A": 0.005, "B": 0.015})
    digest = hashlib.sha256(json.dumps({"archive": archive, "state": records.state},
                                       sort_keys=True, separators=(",", ":")).encode()
                            ).hexdigest()
    assert digest == GEN_SETTLEMENT_GOLDEN


# -- I5. what the order-control image needs ------------------------------------------


def test_the_order_control_runtime_check_names_every_missing_piece(monkeypatch):
    from reliquary.validator import eval_control

    seen = []
    monkeypatch.setattr(eval_control, "_import", lambda name: seen.append(name))
    monkeypatch.setattr(eval_control, "_verify_environment", lambda env: None)
    report = eval_control.order_control_runtime_check()
    assert set(seen) >= {"bittensor_drand", "huggingface_hub", "transformers"}
    assert set(report["environments"]) == set(qual.ORDER_ENVIRONMENTS)
    assert report["ok"] is True

    def missing(env):
        raise ValueError(f"external environment distribution for {env} is not installed")

    monkeypatch.setattr(eval_control, "_verify_environment", missing)
    report = eval_control.order_control_runtime_check()
    assert report["ok"] is False and "not installed" in report["environments"][GEN_ENV]


def test_the_order_control_image_installs_and_checks_its_runtime():
    from pathlib import Path

    dockerfile = Path("docker/Dockerfile.order-control").read_text()
    assert "order-control-check" in dockerfile
    for distribution in ("reliquary-logic", "reliquary-dapo-math",
                         "reliquary-instruction-following", "reliquary-code"):
        assert distribution in dockerfile
    assert "order-control" in Path("docs/runbooks/order-control.md").read_text()


# -- M3. reopening is off unless asked -----------------------------------------------


def test_slot_reopening_is_off_by_default():
    from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS
    from reliquary.validator.eval_control import eval_auditor

    auditor = eval_auditor(job_id="order-eval-1", records=None, tokenizer=None,
                           proof=TOPLOC_DEPLOYED_DEFAULTS, vocab_size=10,
                           remote=SimpleNamespace(subscribe=lambda listener: None))
    assert auditor.reopen_slots is False


# -- platform I2: a drained order's status survives a restart; a retired order
# not yet drained is wired again at boot and finishes. ---------------------------


def test_an_order_job_finishes_across_a_restart(world, monkeypatch):  # noqa: F811
    from reliquary.infrastructure.corpus_job_store import BucketJobStore
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from reliquary.validator import corpus_auditor, corpus_period_settlement
    from reliquary.validator.eval_control import (
        EvalExecutorDirectory,
        PairedAuditDispatcher,
        build_order_control,
    )
    from tests.unit.test_eval_control_process import _Tokenizer

    import httpx

    entries, _ = world
    pending = {"order-eval-a": [], "order-eval-b": []}

    async def idle(self):
        await asyncio.sleep(3600)

    async def pending_ids(self):
        return list(pending[self._job_id])

    monkeypatch.setattr(corpus_auditor.CorpusAuditor, "run", idle)
    monkeypatch.setattr(corpus_auditor.CorpusAuditor, "pending_ids", pending_ids)
    monkeypatch.setattr(corpus_period_settlement.CorpusPeriodSettler, "settle_once", lambda self: idle(self))

    async def read_entries():
        return dict(entries["entries"])

    def control():
        directory = EvalExecutorDirectory(list_documents=lambda: asyncio.sleep(0, []))
        return build_order_control(
            store=BucketJobStore(), records=BucketRecordStore(),
            dispatcher=PairedAuditDispatcher(directory=directory), directory=directory,
            verify_signature=lambda request: True,
            tokenizer_for=lambda repo, revision: (_Tokenizer(), 100), read_entries=read_entries)

    def stop(job_set):
        for tasks in job_set._tasks.values():
            for task in tasks:
                task.cancel()

    async def status(app, job_id):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://order") as client:
            return await client.get(f"/corpus/jobs/{job_id}/status")

    async def go():
        records = BucketRecordStore()
        app, job_set = control()
        await job_set.refresh()
        assert sorted(job_set.served) == ["order-eval-a", "order-eval-b"]
        # b is still auditing when both are retired: only a drains.
        pending["order-eval-b"] = ["s" * 64]
        for job_id in ("order-eval-a", "order-eval-b"):
            entries["entries"][job_id] = replace(entries["entries"][job_id], status="retired",
                                                 retired_at=5)
        await job_set.refresh()
        await job_set.refresh()
        assert sorted(job_set.served) == ["order-eval-b"]
        final = await records.read_final_status("order-eval-a")
        assert final["state"] == "drained" and final["job_id"] == "order-eval-a"
        assert {"prompts_total", "prompts_complete", "prompts_exhausted", "complete"} <= set(final)
        assert await records.read_final_status("order-eval-b") is None
        stop(job_set)

        # A restart: a's final status is served from the bucket, never re-wired;
        # b, retired and not drained, is wired again, admission closed.
        app, job_set = control()
        await job_set.refresh()
        assert sorted(job_set.served) == ["order-eval-b"]
        assert job_set.is_retired("order-eval-b")
        assert (await status(app, "order-eval-a")).json() == final
        pending["order-eval-b"] = []
        await job_set.refresh()
        await job_set.refresh()
        assert job_set.served == {}
        stored = await records.read_final_status("order-eval-b")
        assert stored["state"] == "drained"
        assert (await status(app, "order-eval-b")).json() == stored
        stop(job_set)

    asyncio.run(go())


def test_the_cap_guard_refusal_is_a_409_with_a_fixed_detail(admin, monkeypatch):  # noqa: F811
    stub_catalog_env(monkeypatch)
    _qualify_gen(admin)
    # The admin pool is 0.3 here: one order alone may not take more.
    response = admin("POST", "/admin/v1/jobs", _gen_job(cap=0.31))
    assert response.status_code == 409
    assert response.json() == {
        "detail": "corpus caps would total 0.3100, above the admin pool of 0.3000"}
    # Retryable: the same order with a cap that fits is declared.
    assert admin("POST", "/admin/v1/jobs", _gen_job(cap=0.02)).status_code == 201
