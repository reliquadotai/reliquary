"""The async RL service wiring (``validator/rl_async_service.py``, ``trainer/rl_link.py``, ``trainer/async_cli.py``):
the mode switch, the registry declaration, the trainer link, and one CPU end-to-end run over real HTTP: the
reference miner leases, generates (fake) and submits against the mounted routes, the async trainer takes the
buffered groups through the link, steps (fake step), reports and publishes a delta to a fake bucket, the ledger
accrues and ``/rl/pay`` shows it, the miner syncs the new checkpoint, and the period settles."""
from __future__ import annotations

import asyncio
import dataclasses
import json
import random
import socket
import threading
import time

import pytest
import torch

bt = pytest.importorskip("bittensor")
httpx = pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from reliquary.validator.rl_async_service import (  # noqa: E402
    ASYNC, AuditModel, PLANNED_TOKENS_PARAM, WINDOW, RlAsyncSettings, build_rl_async, build_rl_trainer_app,
    build_rl_trainer_router, mount_rl_async, rl_async_loops, rl_async_settings, rl_mode, trainer_bind,
)
from tests.unit.test_rl_submit import CAP, ENV, MathEnv, Tokenizer, group  # noqa: E402

MINER = bt.Keypair.create_from_uri("//Alice")
VALIDATOR = bt.Keypair.create_from_uri("//Bob")
TOKEN = "t" * 32


@pytest.fixture(autouse=True)
def small_cap(monkeypatch):
    monkeypatch.setattr("reliquary.validator.admission.max_new_tokens_for_environment", lambda env: CAP)
    monkeypatch.setattr("reliquary.validator.rl_submit.max_new_tokens_for_environment", lambda env: CAP)


def test_the_mode_switch_defaults_to_windows():
    assert rl_mode({}) == WINDOW and rl_mode({"RELIQUARY_RL_MODE": " Async "}) == ASYNC
    with pytest.raises(ValueError):
        rl_mode({"RELIQUARY_RL_MODE": "both"})


def test_an_async_task_is_declared_paid_by_period_with_its_mix():
    from reliquary.cli.main import build_task_entry, with_rl_async_params
    from reliquary.constants import PROTOCOL_PROFILE_ID
    from reliquary.protocol.profiles import resolve_protocol_profile
    from reliquary.shared.task_registry import parse_registry, render_registry, validate_entry
    from reliquary.validator import corpus_periods as cp

    envs = sorted(resolve_protocol_profile(PROTOCOL_PROFILE_ID).environments)
    split = {env: (1.0 if i == 0 else 0.0) for i, env in enumerate(envs)}
    entry = build_task_entry(task_id="rl-async", profile_id=PROTOCOL_PROFILE_ID, cap=0.4, overrides={},
                             env_split=split)
    with pytest.raises(ValueError, match="settled by period"):
        rl_async_settings(entry, 0.4)                                   # a window task is not an async one
    declared = with_rl_async_params(entry, planned_tokens_per_period=2e8, lease_tokens_per_s=10.0)
    validate_entry(declared)
    assert cp.is_period_task(declared) and declared.params[PLANNED_TOKENS_PARAM] == 2e8
    assert parse_registry(render_registry({"rl-async": declared}))["rl-async"] == declared
    settings = rl_async_settings(declared, 0.4, run="r1", max_tokens=lambda env: 4000)
    assert settings.targets == {envs[0]: 1.0} and settings.run == "r1" and settings.cap == 0.4
    assert settings.lease_expiry_s == {envs[0]: 4000 / 10.0 * 2.0}
    with pytest.raises(ValueError):
        with_rl_async_params(entry, planned_tokens_per_period=None)


def test_an_async_task_declares_a_start_pool_under_its_cap():
    from dataclasses import replace

    from reliquary.cli.main import build_task_entry, with_rl_async_params
    from reliquary.constants import PROTOCOL_PROFILE_ID
    from reliquary.protocol.profiles import resolve_protocol_profile
    from reliquary.shared.task_registry import RegistryError, set_cap, validate_entry
    from reliquary.validator.rl_async_service import START_POOL_PARAM

    envs = sorted(resolve_protocol_profile(PROTOCOL_PROFILE_ID).environments)
    split = {env: (1.0 if i == 0 else 0.0) for i, env in enumerate(envs)}
    entry = build_task_entry(task_id="rl-async", profile_id=PROTOCOL_PROFILE_ID, cap=0.4, overrides={},
                             env_split=split)
    declared = with_rl_async_params(entry, planned_tokens_per_period=2e8, start_pool=0.1)
    validate_entry(declared)
    assert declared.params[START_POOL_PARAM] == 0.1
    settings = rl_async_settings(declared, 0.4, max_tokens=lambda env: 4000)
    assert settings.start_pool == 0.1 and settings.cap == 0.4
    legacy = rl_async_settings(with_rl_async_params(entry, planned_tokens_per_period=2e8), 0.4,
                               max_tokens=lambda env: 4000)
    assert legacy.start_pool is None                                    # the controller prices on the cap
    for bad in (0.0, -0.1, 0.5):
        with pytest.raises(ValueError, match="--start-pool"):
            with_rl_async_params(entry, planned_tokens_per_period=2e8, start_pool=bad)
    for bad in (0.0, 0.5, "x"):
        with pytest.raises(RegistryError):
            validate_entry(replace(declared, params={**declared.params, START_POOL_PARAM: bad}))
    with pytest.raises(RegistryError, match="async RL"):                # a window task has no start pool
        validate_entry(replace(entry, params={**entry.params, START_POOL_PARAM: 0.1}))
    lowered = set_cap({"rl-async": declared}, "rl-async", 0.05)["rl-async"]
    assert lowered.params[START_POOL_PARAM] == 0.05                     # a lowered cap bounds the start pool
    assert set_cap({"rl-async": declared}, "rl-async", 0.3)["rl-async"].params[START_POOL_PARAM] == 0.1


def test_the_cli_takes_a_start_pool_only_with_rl_async(monkeypatch):
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    created = []

    async def create(entry):
        created.append(entry)

    from reliquary.constants import PROTOCOL_PROFILE_ID
    from reliquary.protocol.profiles import resolve_protocol_profile
    from reliquary.validator.rl_async_service import START_POOL_PARAM

    envs = sorted(resolve_protocol_profile(PROTOCOL_PROFILE_ID).environments)
    split = ",".join(f"{env}={1.0 if i == 0 else 0.0}" for i, env in enumerate(envs))
    monkeypatch.setattr("reliquary.infrastructure.task_registry_store.create_task", create)
    base = ["tasks", "create", "--task-id", "rl-async", "--profile-id", PROTOCOL_PROFILE_ID, "--cap", "0.4",
            "--env-split", split]
    result = CliRunner().invoke(app, [*base, "--start-pool", "0.1"])
    assert result.exit_code != 0 and "need --rl-async" in result.output
    result = CliRunner().invoke(app, [*base, "--rl-async", "--planned-tokens-per-period", "2e8", "--start-pool", "0.5"])
    assert result.exit_code != 0 and "--start-pool" in result.output
    assert created == []
    result = CliRunner().invoke(app, [*base, "--rl-async", "--planned-tokens-per-period", "2e8", "--start-pool", "0.1"])
    assert result.exit_code == 0, result.output
    assert [e.params[START_POOL_PARAM] for e in created] == [0.1]


def settings_for(**kw) -> RlAsyncSettings:
    return RlAsyncSettings(**{"task_id": "rl-async", "cap": 0.5, "run": "smoke", "targets": {ENV: 1.0},
                              "planned_tokens_per_period": 1e6, "lease_expiry_s": {ENV: 600.0}, **kw})


def components(tmp_path, board=None):
    from reliquary.validator.rl_submit import GroupGrader

    grader = GroupGrader({ENV: MathEnv()}, Tokenizer(), eos_token_ids=(1,))
    return build_rl_async(settings_for(), path=tmp_path / "rl-async.sqlite3", grader=grader, board=board,
                          prompt_range={ENV: (0, 50)}, collector_interval_s=0.0)


def test_the_run_benchmark_feed_is_served_from_the_private_bucket(tmp_path):
    from reliquary.validator.server import ValidatorServer

    rl = components(tmp_path)
    server = ValidatorServer()
    mount_rl_async(server, rl, validator_hotkey=VALIDATOR.ss58_address)
    public = TestClient(server.app)
    assert public.get("/rl/benchmarks", params={"run_id": "abc-r1-trainer"}).status_code == 404   # none yet
    feeds = {"abc-r1-trainer": b'{"schema":"reliquary.run-benchmarks.v1"}'}
    rl.benchmarks = feeds.get
    served = public.get("/rl/benchmarks", params={"run_id": "abc-r1-trainer"})
    assert served.status_code == 200 and served.content == feeds["abc-r1-trainer"]
    assert served.headers["content-type"] == "application/json"
    assert public.get("/rl/benchmarks", params={"run_id": "other"}).status_code == 404
    for bad in ("", "../x", "a/b", "a" * 129):
        assert public.get("/rl/benchmarks", params={"run_id": bad}).status_code == 422


def test_the_trainer_link_needs_its_token_and_round_trips(tmp_path):
    from reliquary.trainer.rl_link import TrainerLink, TrainerLinkError
    from reliquary.validator.rl_leases import LeaseRequest

    rl = components(tmp_path)
    app = FastAPI()
    app.include_router(build_rl_trainer_router(rl, token=TOKEN))
    client = TestClient(app)
    assert client.get("/rl/trainer/config").status_code == 401
    assert client.get("/rl/trainer/config", headers={"Authorization": "Bearer " + "x" * 32}).status_code == 401
    link = TrainerLink("http://testserver", TOKEN, client=client, patience_s=0.0)
    assert link.config() == {"task_id": "rl-async", "run": "smoke", "targets": {ENV: 1.0}, "latest": None}
    link.adopt_checkpoint(0)
    assert link.latest_checkpoint == 0
    lease = rl.book.grant([LeaseRequest("A", ENV, (3,), 0)])[0].leases[0]
    assert rl.submissions.submit(lease.lease_id, "A", group(8))["verdict"] == "in_zone"
    taken = link.take(4)
    assert [g.lease_id for g in taken] == [lease.lease_id] and link.depth() == {}
    assert link.requeue() == 1 and link.depth() == {ENV: 1}
    assert link.take(4)[0].completions == taken[0].completions
    link.report_training(lease.lease_id, rollout_match=[1.0] * 16, accepted=True, matched_tokens=[3] * 16)
    with pytest.raises(TrainerLinkError) as missing:
        link.report_training(lease.lease_id, rollout_match=[1.0] * 16, accepted=True)
    assert missing.value.status == 404                                  # a first attempt: not a resend
    with pytest.raises(TrainerLinkError):
        link.adopt_checkpoint(-1)
    rl.close()


def test_the_trainer_takes_the_audits_and_reports_their_verdicts_through_its_link(tmp_path):
    from reliquary.trainer.rl_link import TrainerLink, TrainerLinkError
    from reliquary.validator.rl_leases import LeaseRequest
    from reliquary.validator.rl_submit import AuditResult

    rl = components(tmp_path)
    app = FastAPI()
    app.include_router(build_rl_trainer_router(rl, token=TOKEN))
    client = TestClient(app)
    assert client.post("/rl/trainer/audits", json={"limit": 1}).status_code == 401
    link = TrainerLink("http://testserver", TOKEN, client=client, patience_s=0.0)
    link.adopt_checkpoint(0)
    lease = rl.book.grant([LeaseRequest("A", ENV, (4,), 0)])[0].leases[0]
    assert rl.submissions.submit(lease.lease_id, "A", group(0))["verdict"] == "too_hard"   # probation: drawn
    [job] = link.take_audits(8)
    assert job.lease_id == lease.lease_id and link.take_audits(8)[0].lease_id == lease.lease_id   # until reported
    for bad in (0, 33):
        with pytest.raises(TrainerLinkError) as refused:
            link.take_audits(bad)
        assert refused.value.status == 422
    link.report_audit(lease.lease_id, AuditResult(True, None, (0.99,) * 16))
    assert rl.submissions.group(lease.lease_id)["audit"] == "passed" and link.take_audits(8) == []
    with pytest.raises(TrainerLinkError) as missing:
        link.report_audit(lease.lease_id, AuditResult(True, None, (0.99,) * 16))
    assert missing.value.status == 404
    rl.close()


def test_the_trainer_link_has_its_own_listener_never_the_public_port(tmp_path):
    from reliquary.trainer.rl_link import TrainerLink
    from reliquary.validator.server import ValidatorServer

    assert trainer_bind({}) == ("127.0.0.1", 8889)
    with pytest.raises(ValueError, match="not loopback"):                 # plain HTTP, a token that mints pay
        trainer_bind({"RELIQUARY_RL_TRAINER_BIND": "10.0.0.5:7000"})
    private = {"RELIQUARY_RL_TRAINER_BIND": "10.0.0.5:7000", "RELIQUARY_RL_TRAINER_INSECURE_BIND": "1"}
    assert trainer_bind(private) == ("10.0.0.5", 7000)                    # the operator says: a private network
    assert trainer_bind({"RELIQUARY_RL_TRAINER_BIND": "[::1]:7000"}) == ("::1", 7000)
    assert trainer_bind({"RELIQUARY_RL_TRAINER_BIND": "localhost:7000"}) == ("localhost", 7000)
    for bad in ("7000", "host:", "host:99999", ":80"):
        with pytest.raises(ValueError):
            trainer_bind({"RELIQUARY_RL_TRAINER_BIND": bad})

    class Board:
        async def answer(self):
            return {"index": {"latest": {"version": 3}}, "signature": "s"}

    rl = components(tmp_path, board=Board())
    server = ValidatorServer()
    mount_rl_async(server, rl, validator_hotkey=VALIDATOR.ss58_address)
    public = TestClient(server.app)
    auth = {"Authorization": f"Bearer {TOKEN}"}
    assert public.get("/rl/trainer/config", headers=auth).status_code == 404     # not on the public port
    assert public.get("/rl/checkpoint").json()["signature"] == "s"
    trainer = TestClient(build_rl_trainer_app(rl, token=TOKEN))
    assert trainer.get("/rl/trainer/checkpoint").status_code == 401
    assert trainer.get("/rl/checkpoint").status_code == 404                      # only the link lives here
    link = TrainerLink("http://testserver", TOKEN, client=trainer, patience_s=0.0)
    assert link.checkpoint()["index"]["latest"]["version"] == 3 and link.config()["run"] == "smoke"
    rl.close()


def test_a_lost_report_answer_is_resent_and_counts_as_done(tmp_path):
    from reliquary.trainer.rl_link import TrainerLink

    class LosesFirstAnswer:
        def __init__(self, inner):
            self.inner, self.calls = inner, 0

        def request(self, *args, **kwargs):
            self.calls += 1
            answer = self.inner.request(*args, **kwargs)
            if self.calls == 1:
                raise httpx.ReadTimeout("lost")
            return answer

    rl = components(tmp_path)
    app = FastAPI()
    app.include_router(build_rl_trainer_router(rl, token=TOKEN))
    from reliquary.validator.rl_leases import LeaseRequest

    rl.book.adopt_checkpoint(0)
    lease = rl.book.grant([LeaseRequest("A", ENV, (3,), 0)])[0].leases[0]
    rl.submissions.submit(lease.lease_id, "A", group(8))
    rl.buffer.take(1)
    flaky = LosesFirstAnswer(TestClient(app))
    link = TrainerLink("http://testserver", TOKEN, client=flaky, sleep=lambda s: None)
    link.report_training(lease.lease_id, rollout_match=[1.0] * 16, accepted=True)
    assert flaky.calls == 2 and [e["kind"] for e in rl.submissions.events()] == ["trained"]
    rl.close()


def test_the_trainer_resumes_from_the_latest_published_checkpoint(tmp_path):
    from reliquary.trainer.async_cli import resume_weights

    class Link:
        def __init__(self, latest):
            self.latest_checkpoint, self.adopted = latest, []

        def adopt_checkpoint(self, k):
            self.adopted.append(k)

    class Sync:
        model_dir = tmp_path / "files"

        def __init__(self, version):
            self.version = version

        def sync(self, response, state, version):
            assert state is None and version is None                      # always rebuilt from an anchor
            return {"w": torch.zeros(1)}, self.version, None

    fresh = Link(None)
    assert resume_weights(fresh, validator_hotkey="v", work_dir=tmp_path, sync=Sync(0), fetch=dict) is None
    behind = Link(3)
    state, version, files = resume_weights(behind, validator_hotkey="v", work_dir=tmp_path, sync=Sync(4),
                                           fetch=dict)
    assert version == 4 and behind.adopted == [4] and files == Sync.model_dir      # published, never adopted
    with pytest.raises(RuntimeError):
        resume_weights(Link(5), validator_hotkey="v", work_dir=tmp_path, sync=Sync(4), fetch=dict)


# --- end to end ---------------------------------------------------------------------------------------------------
class MemoryStore:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def put_bytes(self, key, data):
        self.objects[key] = bytes(data)

    def put_file(self, key, path):
        self.objects[key] = open(path, "rb").read()

    def get_bytes(self, key):
        return self.objects.get(key)

    def delete(self, key):
        self.objects.pop(key, None)


class Served:
    """The app on a real socket (uvicorn in a thread)."""

    def __init__(self, app):
        import uvicorn

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        self.server = uvicorn.Server(uvicorn.Config(app, log_level="warning", ws="none", lifespan="off"))
        self.thread = threading.Thread(target=self.server.run, kwargs={"sockets": [sock]}, daemon=True)
        self.thread.start()
        for _ in range(500):
            if self.server.started:
                return
            time.sleep(0.01)
        raise RuntimeError("server did not start")

    def stop(self):
        self.server.should_exit = True
        self.thread.join(10)


class Archives:
    def __init__(self):
        self.written = []

    async def read(self, task_id, work_period, entry_period):
        return next((doc for task, work, entry, doc in self.written
                     if (task, work, entry) == (task_id, work_period, entry_period)), None)

    async def write(self, task_id, work_period, entry_period, doc):
        self.written.append((task_id, work_period, entry_period, doc))


def in_zone_or_hard(leases):
    """A generator stand-in: an even prompt gets an in-zone group (8/16 right), an odd one 0/16 (too hard)."""
    return [[list(r.tokens) for r in group(8 if lease.prompt_idx % 2 == 0 else 0)] for lease in leases]


async def test_end_to_end_lease_submit_train_pay_and_sync(tmp_path):
    from reliquary.miner.ckpt_sync import CheckpointSync
    from reliquary.miner.rl_async import (
        AsyncMiner, AsyncMinerConfig, FakeGroupGenerator, HttpRlTransport, PromptPlanner,
    )
    from reliquary.shared.ckpt_index import index_key
    from reliquary.trainer.async_cli import run_forever
    from reliquary.trainer.async_trainer import AsyncTrainer, AsyncTrainerConfig
    from reliquary.trainer.ckpt_publisher import DeltaPublisher
    from reliquary.trainer.rl_link import TrainerLink
    from reliquary.validator.rl_checkpoint import RlCheckpointBoard
    from reliquary.validator.server import ValidatorServer

    started = time.monotonic()
    store = MemoryStore()
    board = RlCheckpointBoard(load_index=lambda: store.get_bytes(index_key("smoke")),
                              sign_fn=lambda payload: VALIDATOR.sign(payload), validator_hotkey=VALIDATOR.ss58_address,
                              presign=lambda key: f"mem://{key}", refresh_seconds=0)
    rl = components(tmp_path, board=board)
    server = ValidatorServer()
    mount_rl_async(server, rl, validator_hotkey=VALIDATOR.ss58_address)
    served, trainer_served = Served(server.app), Served(build_rl_trainer_app(rl, token=TOKEN))
    try:
        # --- the trainer: a tiny model, a fresh book -> checkpoint 0 published and adopted through the link
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "tiny"}))
        (model_dir / "tokenizer.json").write_text("{}")
        model = torch.nn.Linear(8, 8).to(torch.bfloat16)

        def fake_step(model, batches, *, gate, **_):
            for g in (g for b in batches for g in b):
                n = len(g.rollouts[0].commit["tokens"]) - g.rollouts[0].commit["rollout"]["prompt_length"]
                gate.admit(g.rollouts, [(30, 30, n, n)] * len(g.rollouts))
            gate.judge([])
            with torch.no_grad():
                model.weight[0, :3] += 1.0
            return model

        link = TrainerLink(trainer_served.url, TOKEN, patience_s=5.0)
        publisher = DeltaPublisher(run="smoke", store=store, staging_dir=tmp_path / "staging", model_dir=model_dir)
        trainer = AsyncTrainer(model, config=AsyncTrainerConfig(batch_groups=4),
                               buffer=link, submissions=link, book=link, publisher=publisher,
                               train_step_fn=fake_step)
        assert rl.book.latest_checkpoint == 0 and trainer.version == 0

        # --- the miner over HTTP: syncs checkpoint 0 from /rl/checkpoint, leases, generates, submits
        def download(url, dest):
            dest.write_bytes(store.objects[url.removeprefix("mem://")])

        sync = CheckpointSync(validator_hotkey=VALIDATOR.ss58_address, work_dir=tmp_path / "miner-dl",
                              download=download, model_dir=tmp_path / "miner-model")
        generator = FakeGroupGenerator()
        generator.generate = in_zone_or_hard
        async with httpx.AsyncClient(timeout=30.0) as client:
            transport = HttpRlTransport(client, base_url=served.url, hotkey=MINER.ss58_address,
                                        validator_hotkey=VALIDATOR.ss58_address,
                                        sign_binding=lambda binding: MINER.sign(binding).hex())
            miner = AsyncMiner(AsyncMinerConfig(environments=(ENV,), batch_groups=4, submit_backoff_s=(0.0,)),
                               transport=transport, generator=generator,
                               planners={ENV: PromptPlanner(ENV, (0, 50), rng=random.Random(0))},
                               checkpoint_sync=sync, rng=random.Random(0))
            assert await miner.follow_checkpoint_once() == 0
            assert (tmp_path / "miner-model" / "config.json").is_file()        # the anchor's model files
            assert await miner.swap_if_ready() and miner.loaded_version == 0
            await miner.refresh_quota()
            await miner.refresh_state()
            assert await miner.lease_more() == 4                              # the starting quota
            assert await miner.generate_once() == 4
            await miner.drain()
            assert generator.loads == [0] and miner.stats["submitted"] == 4
            verdicts = miner.stats["verdicts"]
            assert sum(verdicts.values()) == 4 and verdicts.get("in_zone", 0) >= 1 and verdicts.get("too_hard", 0) >= 1
            in_zone = verdicts["in_zone"]
            assert rl.buffer.depth() == {ENV: in_zone}

            # --- the background loops: expiry, audits (no verify model: inconclusive), ledger, prices
            archives = Archives()
            loops = [asyncio.create_task(c) for c in rl_async_loops(
                rl, archives=archives, every={"expire": 0.01, "audit": 0.01, "ledger": 0.01, "price": 0.01})]
            for _ in range(200):
                await asyncio.sleep(0.01)
                if not rl.submissions.audit_jobs(10):
                    break
            assert rl.submissions.audit_jobs(10) == []                          # every out-of-zone group concluded

            # --- one trainer step through the link (the batch is what passed): report, publish 1, adopt, speed
            trainer.config = dataclasses.replace(trainer.config, batch_groups=in_zone)
            await asyncio.to_thread(run_forever, trainer, steps=1)
            await asyncio.to_thread(trainer.close)
            assert trainer.last_step["result"] == "trained" and trainer.version == 1
            assert rl.book.latest_checkpoint == 1 and rl.buffer.depth() == {}
            kinds = [e["kind"] for e in rl.submissions.events()]
            assert kinds.count("trained") == in_zone
            assert kinds.count("first_scan_forfeited") == verdicts["too_hard"]  # probation audits, inconclusive

            # --- pay: the ledger accrues the trained groups, /rl/pay shows them
            await asyncio.to_thread(rl.ledger.ingest)
            pay = (await client.get(f"{served.url}/rl/pay", params={"hotkey": MINER.ss58_address})).json()
            assert pay["hotkeys"][MINER.ss58_address]["accrued"] > 0
            assert pay["environments"][ENV]["target_share"] == 1.0

            # --- the miner syncs checkpoint 1 (a delta) and swaps it in before its next batch
            assert await miner.follow_checkpoint_once() == 1
            assert torch.equal(miner._state["weight"].view(torch.int16),
                               trainer.model.state_dict()["weight"].view(torch.int16))
            await miner.lease_more()
            await miner.generate_once()
            await miner.drain()
            assert generator.loads == [0, 1]

            # --- the validator's verify model follows the checkpoints; audits wait while it lags
            audit_sync = CheckpointSync(validator_hotkey=VALIDATOR.ss58_address, work_dir=tmp_path / "audit-dl",
                                        download=download)
            audit = AuditModel(torch.nn.Linear(8, 8).to(torch.bfloat16), audit_sync)
            assert audit.audit(1, rl.submissions, 8) is None
            assert await audit.follow_once(board) == 1 and await audit.follow_once(board) is None
            assert torch.equal(audit.model.weight, trainer.model.weight)
            for task in loops:
                task.cancel()
            await asyncio.gather(*loops, return_exceptions=True)

        # --- settlement: the period closes, its archive pays the miner under the cap
        settled = rl.ledger.settle(now=time.time() + 3 * 4320)
        assert settled and settled[0]["rewards_by_hotkey"][MINER.ss58_address] > 0
        assert await rl.ledger.publish(archives) == len(settled) and archives.written[0][0] == "rl-async"
    finally:
        served.stop()
        trainer_served.stop()
        rl.close()
    assert time.monotonic() - started < 60


# --- discovery ------------------------------------------------------------------------------------------------------
async def test_the_async_validator_serves_its_axon_like_the_window_one(monkeypatch):
    from types import SimpleNamespace

    from reliquary.validator.axon import serve_axon_on_chain

    served = []

    class Subtensor:
        async def serve_axon(self, **kw):
            served.append(kw)
            return SimpleNamespace(is_success=True)

    monkeypatch.setattr(bt, "Axon", lambda **kw: SimpleNamespace(**kw))
    wallet = SimpleNamespace(hotkey=VALIDATOR)
    assert await serve_axon_on_chain(Subtensor(), wallet=wallet, netuid=81, ip="1.2.3.4", port=8888)
    assert served[0]["netuid"] == 81 and (served[0]["axon"].ip, served[0]["axon"].port) == ("1.2.3.4", 8888)
    assert not await serve_axon_on_chain(Subtensor(), wallet=wallet, netuid=81, ip=None, port=8888)   # no address

    class Signer:
        async def serve_axon(self, **kw):
            served.append(("signer", kw))
            return True

    assert await serve_axon_on_chain(None, wallet=wallet, netuid=81, ip="1.2.3.4", port=9000, signer_client=Signer())
    assert served[-1] == ("signer", {"netuid": 81, "ip": "1.2.3.4", "port": 9000})


def test_mine_async_finds_the_validator_on_the_metagraph(monkeypatch):
    from types import SimpleNamespace

    from typer.testing import CliRunner

    import reliquary.cli.main as cli
    from reliquary.miner.submitter import discover_validator

    meta = SimpleNamespace(validator_permit=[False, True], hotkeys=["5Miner", "5Validator"],
                           axons=[SimpleNamespace(ip="9.9.9.9", port=1), SimpleNamespace(ip="1.2.3.4", port=8888)])
    assert discover_validator(meta) == ("http://1.2.3.4:8888", "5Validator")

    async def discovered():
        return discover_validator(meta)

    runs = []

    async def run_async_miner(config, **kw):
        runs.append(kw)

    monkeypatch.setattr(cli, "_discover_rl_validator", discovered)
    monkeypatch.setattr("reliquary.miner.rl_async.run_async_miner", run_async_miner)
    monkeypatch.setattr(bt, "Wallet", lambda **kw: SimpleNamespace(**kw))
    runner = CliRunner()
    result = runner.invoke(cli.app, ["mine-async", "--environments", ENV])
    assert result.exit_code == 0, result.output
    assert (runs[-1]["validator_url"], runs[-1]["validator_hotkey"]) == ("http://1.2.3.4:8888", "5Validator")
    result = runner.invoke(cli.app, ["mine-async", "--environments", ENV, "--validator-url", "http://x:1/",
                                     "--validator-hotkey", "5Override"])
    assert result.exit_code == 0 and (runs[-1]["validator_url"], runs[-1]["validator_hotkey"]) == ("http://x:1",
                                                                                                    "5Override")
    assert runner.invoke(cli.app, ["mine-async", "--environments", ENV, "--validator-url", "http://x:1"]).exit_code == 2


def test_the_controller_wandb_row_reads_prices_buffer_leases_and_miners(tmp_path):
    import time

    from reliquary.validator.rl_leases import LeaseRequest
    from reliquary.validator.rl_metrics import controller_metrics

    rl = components(tmp_path)
    rl.book.adopt_checkpoint(0)
    leases = rl.book.grant([LeaseRequest("A", ENV, (3, 4), 0)])[0].leases
    assert rl.submissions.submit(leases[0].lease_id, "A", group(8))["verdict"] == "in_zone"
    row = controller_metrics(rl, time.time())
    assert row[f"price/{ENV}/multiplier"] > 0 and row[f"buffer/{ENV}/queued"] == 1
    assert row["leases/open"] == 1 and row["leases/submitted"] == 1 and row[f"leases_issued/{ENV}"] == 2
    assert row[f"groups/{ENV}/in_zone"] == 1 and row["miners/active"] == 1 and row["miners/banned"] == 0
    assert row["leases/sandbox_failure_released"] == 0
    rl.book.sandbox_failure = lambda lease: ("box_failed", "rollout 2")     # our sandbox failed its episode
    rl.book.forfeit(leases[1].lease_id, "A")
    row = controller_metrics(rl, time.time())
    assert row["leases/sandbox_failure_released"] == 1 and row[f"leases_released/{ENV}/box_failed"] == 1
    assert row["leases/released"] == 1 and "leases/open" not in row
    rl.close()



def test_card_class_turns_declared_names_into_short_keys():
    from reliquary.validator.rl_metrics import card_class

    assert card_class("NVIDIA GeForce RTX 4090") == "rtx-4090"
    assert card_class("NVIDIA H100 80GB HBM3") == "h100-80gb-hbm3"
    assert card_class("  ") == "unknown"


def test_the_controller_row_breaks_the_trainer_verdicts_down_per_declared_card(tmp_path):
    import json
    import time

    from reliquary.validator.miner_setup import MinerSetupStore
    from reliquary.validator.rl_leases import LeaseRequest
    from reliquary.validator.rl_metrics import CardFeed, controller_metrics

    now = time.time()
    setups = MinerSetupStore(tmp_path / "setups.sqlite3")
    h100 = {"name": "NVIDIA H100 80GB HBM3", "memory_mib": 81559}
    setups.put("A", "a" * 16, int(now), now, {"gpus": [{**h100, "id_hash": "1" * 16}, {**h100, "id_hash": "2" * 16}]})
    setups.put("B", "b" * 16, int(now), now, {"gpus": [{"name": "NVIDIA GeForce RTX 4090", "id_hash": "3" * 16}]})
    rl = components(tmp_path)
    rl.book.adopt_checkpoint(0)
    feed = CardFeed(setups, tmp_path / "cards.json")
    first_start = feed.since
    leases = {}
    for hotkey, idx in (("A", 3), ("B", 4), ("C", 5)):
        [lease] = rl.book.grant([LeaseRequest(hotkey, ENV, (idx,), 0)])[0].leases
        assert rl.submissions.submit(lease.lease_id, hotkey, group(8))["verdict"] == "in_zone"
        leases[hotkey] = lease.lease_id
    rl.submissions.report_training(leases["A"], rollout_match=[0.99] * 16, accepted=True,
                                   matched_tokens=[1] * 16)
    rl.submissions.report_training(leases["B"], rollout_match=[0.5] * 15 + [0.25], accepted=False)
    rl.submissions.report_training(leases["C"], rollout_match=[0.0] * 16, accepted=False)   # never judged
    at = time.time() + 1
    row = controller_metrics(rl, at, cards=feed)
    h, r, u = "cards/h100-80gb-hbm3/", "cards/rtx-4090/", "cards/undeclared/"
    # The platform's shared/rl-card-fleet.ts: a same-row vector for every model, window start in Unix seconds.
    assert row["_timestamp"] == at and row["cards/window_start"] == first_start
    models = {k.split("/")[1] for k in row if k.startswith("cards/") and k.count("/") == 2}
    assert models == {"h100-80gb-hbm3", "rtx-4090", "mixed", "undeclared", "other"}
    for m in models:
        for key in ("gpus", "miners", "groups", "groups_rejected", "match_weight", "rejection_weight",
                    "trained_tokens", "trained_token_share"):
            assert isinstance(row[f"cards/{m}/{key}"], (int, float)), (m, key)
    assert sum(row[f"cards/{m}/trained_token_share"] for m in models) == pytest.approx(1.0)
    assert row[h + "gpus"] == 2 and row[h + "miners"] == 1 and row[r + "gpus"] == 1 and row[u + "miners"] == 1
    assert row[h + "groups"] == 1 and row[h + "groups_trained"] == 1 and row[h + "rejected_ratio"] == 0
    assert row[h + "rollout_match_mean"] == pytest.approx(0.99) and row[h + "match_weight"] == 16
    assert row[h + "trained_tokens"] == 16 and row[h + "trained_token_share"] == 1.0
    assert row[r + "rejected_ratio"] == 1.0 and row[r + "rejection_weight"] == 1
    assert row[r + "rollout_match_min"] == 0.25 and row[r + "rollout_match_mean"] == pytest.approx(0.484375)
    # The skipped group says nothing about the card: zero weight, no ratio.
    assert row[u + "groups"] == 0 and row[u + "match_weight"] == 0 and u + "rejected_ratio" not in row
    assert json.loads((tmp_path / "cards.json").read_text()) == ["h100-80gb-hbm3", "rtx-4090"]
    # Next row: disjoint window starting at the previous end, nothing new judged, same models, shares kept.
    later = controller_metrics(rl, at + 61, cards=feed)
    assert later["cards/window_start"] == at and later[h + "groups"] == 0 and h + "rollout_match_mean" not in later
    assert later[h + "trained_token_share"] == 1.0 and later[h + "gpus"] == 2
    # A stall never yields a window the dashboard drops (> 300 s).
    assert controller_metrics(rl, at + 1000, cards=feed)["cards/window_start"] == at + 1000 - 240
    # A restarted controller keeps publishing the same models (the resumed run's summary keeps them).
    assert CardFeed(setups, tmp_path / "cards.json").named == ["h100-80gb-hbm3", "rtx-4090"]
    assert not any(k.startswith("cards/") for k in controller_metrics(rl, time.time()))
    rl.close()


def test_a_tagged_group_counts_for_its_miner_process_cards(tmp_path):
    import time

    from reliquary.validator.miner_setup import MinerSetupStore
    from reliquary.validator.rl_leases import LeaseRequest
    from reliquary.validator.rl_metrics import CardFeed, controller_metrics

    now = time.time()
    setups = MinerSetupStore(tmp_path / "setups.sqlite3")
    setups.put("M", "c" * 16, int(now), now, {"gpus": [{"name": "NVIDIA H100 80GB HBM3", "id_hash": "1" * 16}]})
    setups.put("M", "d" * 16, int(now), now, {"gpus": [{"name": "NVIDIA GeForce RTX 4090", "id_hash": "2" * 16}]})
    rl = components(tmp_path)
    rl.book.adopt_checkpoint(0)
    feed = CardFeed(setups)
    for idx, instance in ((3, "c" * 16), (4, "d" * 16), (5, None), (6, "e" * 16)):   # e: never declared
        [lease] = rl.book.grant([LeaseRequest("M", ENV, (idx,), 0)])[0].leases
        assert rl.submissions.submit(lease.lease_id, "M", group(8))["verdict"] == "in_zone"
        if instance is not None:
            rl.submissions.tag_instance(lease.lease_id, "M", instance)
        rl.submissions.report_training(lease.lease_id, rollout_match=[0.99] * 16, accepted=True,
                                       matched_tokens=[1] * 16)
    row = controller_metrics(rl, time.time() + 1, cards=feed)
    # The fleet is still per hotkey (mixed); each tagged group goes to its process's cards, the rest to the hotkey's.
    assert row["cards/mixed/miners"] == 1 and row["cards/mixed/gpus"] == 2
    assert row["cards/h100-80gb-hbm3/groups"] == 1 and row["cards/rtx-4090/groups"] == 1
    assert row["cards/mixed/groups"] == 2
    rl.close()


def test_past_nine_named_cards_the_rest_is_other(tmp_path):
    import time

    from reliquary.validator.miner_setup import MinerSetupStore
    from reliquary.validator.rl_leases import LeaseRequest
    from reliquary.validator.rl_metrics import CardFeed, controller_metrics

    now = time.time()
    setups = MinerSetupStore(tmp_path / "setups.sqlite3")
    rl = components(tmp_path)
    rl.book.adopt_checkpoint(0)
    for i in range(11):
        setups.put(f"H{i}", "x" * 16, int(now), now, {"gpus": [{"name": f"Card {i}", "id_hash": f"{i:016d}"}]})
        rl.book.grant([LeaseRequest(f"H{i}", ENV, (i,), 0)])
    row = controller_metrics(rl, time.time() + 1, cards=CardFeed(setups))
    models = {k.split("/")[1] for k in row if k.startswith("cards/") and k.endswith("/gpus")}
    assert len(models) == 12 and row["cards/other/gpus"] == 2
    rl.close()


async def test_with_audits_on_the_trainer_the_validator_loop_never_concludes_them_itself(tmp_path):
    from reliquary.validator.rl_leases import LeaseRequest

    rl = components(tmp_path)
    rl.book.adopt_checkpoint(0)
    lease = rl.book.grant([LeaseRequest("A", ENV, (5,), 0)])[0].leases[0]
    rl.submissions.submit(lease.lease_id, "A", group(0))
    loops = [asyncio.create_task(c) for c in rl_async_loops(
        rl, archives=Archives(), trainer_audits=True, every={"expire": 9, "audit": 0.01, "ledger": 9, "price": 9})]
    await asyncio.sleep(0.2)
    for task in loops:
        task.cancel()
    await asyncio.gather(*loops, return_exceptions=True)
    assert [g.lease_id for g in rl.submissions.audit_jobs(10)] == [lease.lease_id]   # left to the trainer
    rl.close()


def test_card_windows_do_not_replay_groups_after_restart(tmp_path, monkeypatch):
    from reliquary.validator.miner_setup import MinerSetupStore
    from reliquary.validator.rl_leases import LeaseRequest
    from reliquary.validator.rl_metrics import CardFeed, controller_metrics

    setups = MinerSetupStore(tmp_path / "setups.sqlite3")
    rl = components(tmp_path)
    rl.book.adopt_checkpoint(0)
    feed = CardFeed(setups)
    [lease] = rl.book.grant([LeaseRequest("A", ENV, (3,), 0)])[0].leases
    assert rl.submissions.submit(lease.lease_id, "A", group(8))["verdict"] == "in_zone"
    rl.submissions.report_training(lease.lease_id, rollout_match=[1.0] * 16, accepted=True)
    first_end = time.time() + 1
    first = controller_metrics(rl, first_end, cards=feed)
    assert first["cards/undeclared/groups"] == 1
    monkeypatch.setattr(time, "time", lambda: first_end + 1)
    restarted = CardFeed(setups)
    after = controller_metrics(rl, first_end + 2, cards=restarted)
    assert first_end <= after["cards/window_start"] <= after["_timestamp"]
    assert after["cards/undeclared/groups"] == 0
    # A backward wall-clock correction produces no negative interval or replay.
    backward = controller_metrics(rl, first_end, cards=restarted)
    assert backward["cards/window_start"] == backward["_timestamp"] == first_end
    forward = controller_metrics(rl, first_end + 3, cards=restarted)
    assert forward["cards/window_start"] == after["_timestamp"]
    rl.close()


@pytest.mark.parametrize("judged, expected_groups", [(True, 1), (False, 0), (None, 0)])
def test_zero_match_quality_preserves_judged_skip_and_legacy_distinction(tmp_path, judged, expected_groups):
    from reliquary.trainer.rl_link import TrainerLink
    from reliquary.validator.miner_setup import MinerSetupStore
    from reliquary.validator.rl_leases import LeaseRequest
    from reliquary.validator.rl_metrics import CardFeed, controller_metrics

    rl = components(tmp_path)
    rl.book.adopt_checkpoint(0)
    feed = CardFeed(MinerSetupStore(tmp_path / "setups.sqlite3"))
    [lease] = rl.book.grant([LeaseRequest("A", ENV, (3,), 0)])[0].leases
    assert rl.submissions.submit(lease.lease_id, "A", group(8))["verdict"] == "in_zone"
    app = FastAPI()
    app.include_router(build_rl_trainer_router(rl, token=TOKEN))
    link = TrainerLink("http://testserver", TOKEN, client=TestClient(app), patience_s=0.0)
    link.report_training(lease.lease_id, rollout_match=[0.0] * 16, accepted=False, judged=judged)
    raw = rl.book.db.execute("SELECT detail FROM rl_events WHERE kind='untrained'").fetchone()[0]
    detail = json.loads(raw)
    assert detail.get("judged") is judged
    assert ("judged" in detail) is (judged is not None)
    assert detail["failure_class"] is None
    assert rl.book.db.execute("SELECT COUNT(*) FROM rl_verdicts WHERE source='training'").fetchone()[0] == 0
    row = controller_metrics(rl, time.time() + 1, cards=feed)
    assert row["cards/undeclared/groups"] == expected_groups
    assert row["cards/undeclared/groups_rejected"] == expected_groups
    assert row["cards/undeclared/match_weight"] == 16 * expected_groups
    assert row["cards/undeclared/trained_tokens"] == 0
    if judged:
        assert row["cards/undeclared/rejected_ratio"] == 1.0
        assert row["cards/undeclared/rollout_match_mean"] == 0.0
    rl.close()


@pytest.mark.parametrize("judged", [0, 1, "true"])
def test_invalid_judged_flag_does_not_consume_a_group(tmp_path, judged):
    from reliquary.validator.rl_leases import LeaseRequest

    rl = components(tmp_path)
    rl.book.adopt_checkpoint(0)
    [lease] = rl.book.grant([LeaseRequest("A", ENV, (3,), 0)])[0].leases
    assert rl.submissions.submit(lease.lease_id, "A", group(8))["verdict"] == "in_zone"
    app = FastAPI()
    app.include_router(build_rl_trainer_router(rl, token=TOKEN))
    response = TestClient(app).post("/rl/trainer/report", headers={"Authorization": "Bearer " + TOKEN}, json={
        "lease_id": lease.lease_id, "rollout_match": [0.0] * 16, "accepted": False, "judged": judged})
    assert response.status_code == 422
    assert rl.buffer.get(lease.lease_id) is not None
    rl.close()


def test_controller_reopen_does_not_reissue_groups_held_by_a_live_trainer(tmp_path):
    from reliquary.trainer.async_trainer import AsyncTrainer, AsyncTrainerConfig
    from reliquary.trainer.rl_link import TrainerLink
    from reliquary.validator.rl_leases import LeaseRequest

    live = {"rl": components(tmp_path)}
    live["rl"].book.adopt_checkpoint(0)

    def client_for(rl):
        return TestClient(build_rl_trainer_app(rl, token=TOKEN))

    def submit(prompt):
        rl = live["rl"]
        held = rl.book.grant([LeaseRequest("A", ENV, (prompt,), 0)])[0].leases[0]
        rl.submissions.submit(held.lease_id, "A", group(8))
        return held.lease_id

    live["client"] = client_for(live["rl"])
    leases = [submit(3)]
    reopened = []

    class Transport:
        def request(self, *args, **kwargs):
            return live["client"].request(*args, **kwargs)

        def close(self):
            live["client"].close()

    def wait_for_more(_):
        assert not reopened, "the trainer should finish when one new group arrives"
        held = list(live["rl"].book.db.execute("SELECT lease_id,state,attempts FROM rl_buffer"))
        paid = live["rl"].book.quota.state("A")["counters"].get("paid", 0)
        live["client"].close()
        live["rl"].close()
        live["rl"] = components(tmp_path)                    # same durable book; trainer remains alive
        assert list(live["rl"].book.db.execute("SELECT lease_id,state,attempts FROM rl_buffer")) == held
        assert live["rl"].book.quota.state("A")["counters"].get("paid", 0) == paid
        live["client"] = client_for(live["rl"])
        leases.append(submit(4))
        reopened.append(True)

    link = TrainerLink("http://testserver", TOKEN, client=Transport())
    trainer = AsyncTrainer(None, config=AsyncTrainerConfig(2), buffer=link, submissions=link, book=link,
                           sleep=wait_for_more)
    try:
        batch = trainer.assemble()
        assert [g.lease_id for g in batch] == leases
        assert dict(live["rl"].book.db.execute("SELECT lease_id,attempts FROM rl_buffer")) == dict.fromkeys(leases, 1)
        trainer._report_untrained(batch)
        assert live["rl"].buffer.depth() == {}
        assert live["rl"].book.db.execute("SELECT COUNT(*) FROM rl_buffer").fetchone()[0] == 0
        assert [(e["lease_id"], e["kind"]) for e in live["rl"].submissions.events()] == [
            (lease_id, "untrained") for lease_id in leases]
    finally:
        trainer.close()
        link.close()
        live["rl"].close()


def test_the_operator_sets_the_global_cap_and_the_lanes_by_env_var():
    from reliquary.cli.main import build_task_entry, with_rl_async_params
    from reliquary.constants import PROTOCOL_PROFILE_ID
    from reliquary.protocol.profiles import resolve_protocol_profile
    from reliquary.validator.rl_async_service import DEFAULT_MAX_PENDING_GROUPS, lane_reserves

    envs = sorted(resolve_protocol_profile(PROTOCOL_PROFILE_ID).environments)[:2]
    entry = build_task_entry(task_id="rl-async", profile_id=PROTOCOL_PROFILE_ID, cap=0.4, overrides={},
                             env_split={envs[0]: 0.75, envs[1]: 0.25})
    declared = with_rl_async_params(entry, planned_tokens_per_period=2e8)
    plain = rl_async_settings(declared, 0.4, max_tokens=lambda env: 4000, environ={})
    assert plain.max_pending_groups == DEFAULT_MAX_PENDING_GROUPS and plain.lane_reserves() == {}
    laned = rl_async_settings(declared, 0.4, max_tokens=lambda env: 4000,
                              environ={"RELIQUARY_RL_MAX_PENDING_GROUPS": "1536", "RELIQUARY_RL_LANE_RESERVE": "0.5"})
    assert laned.max_pending_groups == 1536 and laned.lane_reserves() == {envs[0]: 576, envs[1]: 192}
    assert sum(lane_reserves({"a": 1, "b": 1, "c": 1}, 10, 0.99).values()) <= 10
    for bad in ({"RELIQUARY_RL_LANE_RESERVE": "1"}, {"RELIQUARY_RL_MAX_PENDING_GROUPS": "0"}):
        with pytest.raises(ValueError):
            rl_async_settings(declared, 0.4, max_tokens=lambda env: 4000, environ=bad)
