"""A training order runs end to end without an operator: the admin creates it, the supervisor starts its controller,
a miner picks the task through the front, leases and submits, a trainer pod with its own executor token trains the
batch and publishes a checkpoint through presigned URLs, the ledger settles and publishes the bill, the admin
retires the order, the bill turns final and the supervisor stops the controller.

In process on CPU: the controller is the real app (lease routes, trainer link, ledger loops) over a fake grader, the
front reaches it over ASGI by the port the supervisor mapped, and the bucket is a dict behind presigned URLs. The
clock is shared and moved by hand so the drand periods close in seconds."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path

import pytest

bt = pytest.importorskip("bittensor")
httpx = pytest.importorskip("httpx")

from tests.unit.test_admin_eval_jobs import admin  # noqa: E402,F401
from tests.unit.test_admin_token_ceilings import rates  # noqa: E402,F401
from tests.unit.test_admin_training import PROFILE  # noqa: E402
from tests.unit.test_jobs_cli import registry  # noqa: E402,F401
from tests.unit.test_rl_presigned_checkpoints import _Http  # noqa: E402
from tests.unit.test_rl_submit import MathEnv, Tokenizer, group  # noqa: E402

ENV = "reliquary_dapo_math_v1"
MINER = bt.Keypair.create_from_uri("//Alice")
VALIDATOR = bt.Keypair.create_from_uri("//Bob")
POD_TOKEN = "pod-" + "c" * 40
LOOP_EVERY = {"ledger": 0.02, "registry": 0.02, "expire": 0.02, "audit": 3600.0, "price": 3600.0}


class Clock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class Bucket(_Http):
    """The public bucket: presigned PUT/GET URLs name the object; the controller's own writes land here too."""

    def presign_put(self, key):
        return f"https://bucket.invalid/put/{key}"

    def presign_get(self, key):
        return f"https://bucket.invalid/get/{key}"

    def delete(self, key):
        self.objects.pop(key, None)

    async def public_put(self, key, body, content_type, cache_control):
        self.objects[key] = bytes(body)


def boxed_four(environment, problem, texts, materials):
    """The fake grader: a rollout is right when it boxes the answer."""
    return [1.0 if "\\boxed{4}" in text else 0.0 for text in texts]


class Controller:
    """One task's controller as the supervisor's process handle: its app, and its background loops on this loop."""

    def __init__(self, entry, *, path, bucket, clock, registry_entries) -> None:
        from fastapi import FastAPI

        from reliquary.validator.rl_async_service import (
            build_rl_async, build_rl_trainer_router, executor_trainer_auth, mount_rl_async, rl_async_loops,
            rl_async_settings,
        )
        from reliquary.validator.rl_spend import SpendPublisher
        from reliquary.validator.rl_submit import GroupGrader
        from reliquary.validator.server import ValidatorServer

        task_id = entry.task_id
        settings = rl_async_settings(entry, entry.params["cap"], environ={})
        grader = GroupGrader({ENV: MathEnv()}, Tokenizer(), eos_token_ids=(1,), scorer=boxed_four)
        self.rl = rl = build_rl_async(settings, path=path, grader=grader, prompt_range={ENV: (0, 50)},
                                      collector_interval_s=0.0, clock=clock)
        rl.presign_put, rl.presign_get, rl.delete_object = bucket.presign_put, bucket.presign_get, bucket.delete
        rl.public_put = bucket.public_put
        server = ValidatorServer()
        mount_rl_async(server, rl, validator_hotkey=VALIDATOR.ss58_address, clock=clock)
        server.app.include_router(build_rl_trainer_router(rl, auth=executor_trainer_auth(task_id)))
        self.app: FastAPI = server.app

        async def status():
            found = registry_entries().get(task_id)
            return None if found is None else found.status

        loops = rl_async_loops(rl, archives=None, clock=clock, every=LOOP_EVERY,
                               spend=SpendPublisher(rl, rl.public_put, clock=clock), registry_status=status)
        self.tasks = [asyncio.ensure_future(loop) for loop in loops]
        self.code: int | None = None

    def poll(self):
        return self.code

    def terminate(self) -> None:
        for task in self.tasks:
            task.cancel()
        self.code = -15

    kill = terminate

    async def closed(self) -> None:
        await asyncio.gather(*self.tasks, return_exceptions=True)
        if self.tasks:
            self.tasks = []
            self.rl.close()


class Terminator(httpx.BaseTransport):
    """The host's TLS terminator, for a pod's synchronous client: relays each request to the front on the test's
    loop with ``X-Forwarded-Proto: https``."""

    def __init__(self, front, loop) -> None:
        self.front, self.loop = front, loop

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        body = request.read()

        async def relay() -> httpx.Response:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.front)) as client:
                headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
                r = await client.request(request.method, str(request.url).replace("https://", "http://", 1),
                                         headers={**headers, "X-Forwarded-Proto": "https"}, content=body)
            out = {k: v for k, v in r.headers.items() if k.lower() not in ("content-encoding", "content-length")}
            return httpx.Response(r.status_code, headers=out, content=r.content)

        return asyncio.run_coroutine_threadsafe(relay(), self.loop).result(timeout=30)


async def until(condition, *, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        await asyncio.sleep(0.01)


def _order():
    return {"training_id": "e2e", "profile_id": PROFILE,
            "environments": [{"name": ENV, "share_bps": 10000}],
            "limits": {"max_steps": 50, "deadline_seconds": 86400}, "max_price_usd_per_mtoken": 2.0,
            "batch_groups": 2}


@pytest.fixture
def on(monkeypatch, request, rates):  # noqa: F811
    from reliquary.admin.training import TRAINING_ORDERS_ENV

    monkeypatch.setenv(TRAINING_ORDERS_ENV, "on")
    return request.getfixturevalue("admin")


@pytest.mark.slow
async def test_a_training_order_runs_end_to_end_without_an_operator(on, monkeypatch, tmp_path):
    from reliquary.admin import service
    from reliquary.miner.rl_async import HeldLease, HttpRlTransport, encode_submission
    from reliquary.miner.rl_task_choice import TaskChooser, fetch_tasks, task_prefix
    from reliquary.shared.ckpt_index import index_key
    from reliquary.trainer.ckpt_publisher import DeltaPublisher
    from reliquary.trainer.presigned_store import PresignedObjectStore
    from reliquary.trainer.async_cli import EXECUTOR_ID_ENV, async_trainer_settings, trainer_link
    from reliquary.trainer.rl_link import TrainerLinkError
    from reliquary.validator import corpus_periods as cp
    from reliquary.validator import rl_spend
    from reliquary.validator.rl_front import behind_terminator, build_front_app
    from reliquary.validator.rl_async_service import DB_NAME
    from reliquary.validator.rl_supervisor import Supervisor, admin_final, subprocess_spawn
    from tests.unit.test_ckpt_delta import chain

    clock = Clock(time.time())               # the admin dates the order's deadline on the wall clock
    bucket = Bucket()
    monkeypatch.setattr(rl_spend, "read_spend", lambda run: (json.loads(bucket.objects[rl_spend.spend_key(run)])
                                                             if rl_spend.spend_key(run) in bucket.objects else None))
    retired_at: list[float] = []
    monkeypatch.setattr(service, "_round_time", lambda r: retired_at[0])

    # --- the platform orders a training run, and registers the pod it rented for it
    created = on("POST", "/admin/v1/training", _order())
    assert created.status_code == 201, created.text
    task = created.json()["task_id"]
    pod = {"executor_id": "pod-e2e", "token_sha256": hashlib.sha256(POD_TOKEN.encode()).hexdigest(),
           "model_id": "m", "model_revision": "r", "expires_at": 4e12, "scope": "trainer", "task_id": task}
    assert on("POST", "/admin/v1/executors", pod).status_code == 201

    # --- the supervisor starts the task's controller and maps its port
    entries = lambda: on.registry["entries"]  # noqa: E731
    controllers: dict[int, Controller] = {}

    children: dict[str, dict] = {}
    monkeypatch.setattr("subprocess.Popen", lambda cmd, **kw: children.update({kw["env"]["RELIQUARY_TASK_ID"]: {
        "cmd": cmd, "env": kw["env"]}}))
    child = subprocess_spawn(["reliquary", "validate"], tmp_path / "controllers",
                             {"RELIQUARY_RL_TRAINER_TOKEN": "s" * 32, "RELIQUARY_STATE_DIR": str(tmp_path / "host")})

    def spawn(task_id, port, entry):
        child(task_id, port, entry)                      # the production child: its command line and environment
        env = children[task_id]["env"]
        controllers[port] = Controller(entry, path=Path(env["RELIQUARY_STATE_DIR"]) / DB_NAME, bucket=bucket,
                                       clock=clock, registry_entries=entries)
        return controllers[port]

    map_path = tmp_path / "front" / "map.json"
    sup = Supervisor(entries, spawn, ports=range(9100, 9102), map_path=map_path, clock=clock, final_of=admin_final)
    try:
        sup.tick()
        assert list(controllers) == [9100]
        controller = controllers[9100]
        started = children[task]
        assert started["env"]["RELIQUARY_STATE_DIR"] == str(tmp_path / "controllers" / task)
        assert "RELIQUARY_RL_TRAINER_TOKEN" not in started["env"] and "--no-set-weights" in started["cmd"]
        assert started["env"]["RELIQUARY_PROTOCOL_PROFILE"] == PROFILE
        mapped = json.loads(map_path.read_text())
        assert mapped["tasks"][task]["port"] == 9100 and mapped["default"] == task

        front = behind_terminator(build_front_app(map_path, client_factory=lambda port: httpx.AsyncClient(
            transport=httpx.ASGITransport(app=controllers[port].app))))          # as `reliquary rl front` serves it
        loop = asyncio.get_running_loop()
        pod_http = httpx.Client(transport=Terminator(front, loop))
        pod_env = {"RELIQUARY_RL_TRAINER_URL": "https://front", "RELIQUARY_RL_TRAINER_TOKEN": POD_TOKEN,
                   "RELIQUARY_RL_VALIDATOR_HOTKEY": VALIDATOR.ss58_address, "RELIQUARY_TASK_ID": task,
                   EXECUTOR_ID_ENV: pod["executor_id"]}
        link = trainer_link(async_trainer_settings(pod_env), client=pod_http)     # as the pod's entry builds it

        # --- the pod starts: its task, and the base checkpoint published through presigned URLs
        assert (await asyncio.to_thread(link.config))["task_id"] == task
        states = chain(1)
        publisher = await asyncio.to_thread(DeltaPublisher, run=task, store=PresignedObjectStore(link, http=bucket),
                                            staging_dir=tmp_path / "staging", anchor_interval=2, prune_grace_s=0.0)
        await asyncio.to_thread(publisher.publish, 0, states[0])
        await asyncio.to_thread(link.adopt_checkpoint, 0)

        # --- a miner picks the task that pays, offers, leases and submits through the front
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=front), base_url="http://front") as client:
            chooser = TaskChooser(lambda: fetch_tasks(client, "http://front"), served_envs={ENV}, asks={ENV: 0.25})
            chosen = await chooser.choose()
            assert chosen == task and chooser.listed[task]["model"]["id"] == "Qwen/Qwen3.5-4B"
            miner = HttpRlTransport(client, base_url="http://front", hotkey=MINER.ss58_address,
                                    validator_hotkey=VALIDATOR.ss58_address, prefix=task_prefix(chosen), clock=clock,
                                    sign_binding=lambda binding: MINER.sign(binding).hex())
            assert (await miner.ask(ENV, 250_000, 4))["accepted"] is True
            leased = await miner.lease(ENV, 0, [1, 2])
            held = [HeldLease.from_wire(item) for item in leased["leases"]]
            assert len(held) == 2, leased
            for lease in held:
                graded = await miner.submit(encode_submission(lease, [list(r.tokens) for r in group(8)], 0))
                assert graded["verdict"] == "in_zone", graded

        # --- the pod trains the batch, reports it and publishes the next checkpoint
        batch = await asyncio.to_thread(link.take, 2)
        assert sorted(g.lease_id for g in batch) == sorted(h.lease_id for h in held)
        for g in batch:
            await asyncio.to_thread(link.report_training, g.lease_id, rollout_match=[1.0] * len(g.completions),
                                    accepted=True)
        await asyncio.to_thread(publisher.publish, 1, states[1])
        await asyncio.to_thread(link.adopt_checkpoint, 1)
        assert json.loads(bucket.objects[index_key(task)])["latest"]["version"] == 1
        assert controller.rl.book.latest_checkpoint == 1
        for key in (rl_spend.spend_key(task), f"rl/{task}/anchors/00000001/again.safetensors"):
            with pytest.raises(TrainerLinkError):                    # the pod writes only new checkpoints
                await asyncio.to_thread(link.upload_urls, [key])

        # --- its period closes and settles: the bill shows the trained tokens and the pay, not yet final
        def spend():
            answer = on("GET", f"/admin/v1/training/{task}/spend")
            return answer.json() if answer.status_code == 200 else None

        period = cp.period_of(clock())
        margin = controller.rl.ledger.config.outage_grace_s
        clock.t = cp.period_end(period) + margin + 1
        await until(lambda: (s := spend()) is not None and s["periods"] and s["periods"][0]["settled"])
        bill = spend()
        [first, *_] = bill["periods"]
        assert first["period"] == period and first["paid_alpha"] > 0
        assert first["buckets"][ENV][0][1] > 0                       # trained tokens, at the lease's price
        assert bill["usd_per_alpha"] == pytest.approx(4.0) and bill["latest_checkpoint"] == 1
        assert bill["final"] is False

        # --- the platform retires the order: the controller stops leasing, settles on, then the bill is final
        retired_at.append(clock())
        assert on("POST", f"/admin/v1/training/{task}/retire", {}).json()["status"] == "retired"
        assert on("GET", f"/admin/v1/executors/{pod['executor_id']}").json()["status"] == "revoked"
        await until(lambda: controller.rl.book.auctions.stopped() == "retired")
        clock.t += rl_spend.SPEND_EVERY_S + 1                         # its next bill names the stop
        await until(lambda: (s := spend()) is not None and s["stopped"] == "retired")
        assert spend()["final"] is False and spend()["stopped_at"] < cp.period_end(cp.period_of(retired_at[0]))
        sup.tick()
        assert controller.poll() is None                              # not final yet: it keeps settling
        clock.t = cp.period_end(cp.period_of(clock())) + margin + 1
        await until(lambda: (s := spend()) is not None and s["final"] is True)
        assert all(p["settled"] for p in spend()["periods"])
        assert spend()["settled_through"] >= cp.period_of(spend()["stopped_at"])

        sup.tick()
        assert controller.poll() is not None and json.loads(map_path.read_text())["tasks"] == {}
        link.close()
    finally:
        for running in controllers.values():
            running.terminate()
            await running.closed()
