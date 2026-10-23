"""Episodes on the async RL service (``validator/rl_async_episodes.py``, ``rl_async_service`` wiring, ``mine-async``'s
episode generator): the policies from the registry, the lease lifetimes, the ``/rl/submit`` body cap and its gzip
encoding, and one CPU end-to-end run through the async validator app: the reference miner (``AsyncMiner`` over
``HttpRlTransport``) leases an episode env's prompt, opens its 16 sessions on the mounted ``/rl/sandbox/sessions``
route (the issuer, the ``RlLeaseEngagements`` book and the fleet over a fake directory and capacity report), plays
them (transcripts signed by a fake machine for the issued tokens), closes them, submits the group gzipped, and the
validator grades it with its ``LeaseEpisodeChecker`` into the training buffer."""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

bt = pytest.importorskip("bittensor")
httpx = pytest.importorskip("httpx")

from reliquary.protocol.rl_submit import (  # noqa: E402
    MAX_SUBMIT_BODY_BYTES, encode_tokens, gzip_body, inflate_body,
)
from reliquary.validator.rl_async_episodes import (  # noqa: E402
    EPISODE_ENVS_PARAM, EPISODE_SESSION_SLACK_S, LEASE_EXPIRY_PARAM, MEMORY_FLOOR_ENV, async_episode_policy,
    episode_blocks, episode_lease_expiry_s, episode_policies, memory_floors, validate_episode_block,
)
from reliquary.validator.rl_async_service import (  # noqa: E402
    PLANNED_TOKENS_PARAM, RlAsyncSettings, build_rl_async, mount_rl_async, rl_async_params, rl_async_settings,
    wire_episode_sandbox,
)
from tests.unit.episode_v2_fixtures import BUDGETS, ENV_PACKAGE, EPISODE, GIB, SANDBOX_ENV, episode_block  # noqa: E402
from tests.unit.test_rl_submit import CAP, ENV, MathEnv, Tokenizer  # noqa: E402

MINER = bt.Keypair.create_from_uri("//Alice")
VALIDATOR = bt.Keypair.create_from_uri("//Bob")


@pytest.fixture(autouse=True)
def small_cap(monkeypatch):
    monkeypatch.setattr("reliquary.validator.admission.max_new_tokens_for_environment", lambda env: CAP)
    monkeypatch.setattr("reliquary.validator.rl_submit.max_new_tokens_for_environment", lambda env: CAP)


def entry(params, split=None):
    return SimpleNamespace(task_id="rl-ep", params={"settlement": "period-ema-v1", PLANNED_TOKENS_PARAM: 1e6,
                                                     **params},
                           env_split=split or {ENV: 0.5, EPISODE: 0.5})


# --- policies and lease lifetimes from the registry ----------------------------------------------------------------
def test_episode_blocks_are_checked_with_the_orders_rules():
    block = validate_episode_block(EPISODE, {"episode": episode_block()})
    policy = async_episode_policy(EPISODE, block)
    assert policy.sandbox_env == SANDBOX_ENV and policy.env_package == ENV_PACKAGE
    assert policy.pool_seeds == 32 and policy.budgets_dict() == BUDGETS        # 2 x the lease's 16 rollouts
    for bad in ({"episode": episode_block(tools=["rm"])}, {"episode": episode_block(max_turns=0)},
                {"episode": episode_block(), "extra": 1}, {"catalogue": {}}, {"episode": episode_block(),
                                                                              "version": "x"}):
        with pytest.raises(ValueError):
            validate_episode_block(EPISODE, bad)
    with pytest.raises(ValueError, match="env_split"):
        episode_blocks({EPISODE_ENVS_PARAM: {EPISODE: {"episode": episode_block()}}}, envs=(ENV,))


def test_an_episode_env_declares_its_rows_and_bad_rows_are_refused(tmp_path, monkeypatch):
    from reliquary.cli.main import _read_episode_envs, with_rl_async_params
    from reliquary.protocol.rl_prompt_ranges import PromptRows
    from reliquary.validator.rl_async_episodes import episode_prompt_rows, load_episode_environment

    rows = {"start": 100, "end": 900, "exclude": [[200, 300], 450]}
    block = validate_episode_block(EPISODE, {"episode": episode_block(), "rows": rows})
    assert block["rows"] == rows
    assert episode_prompt_rows(EPISODE, block, 1000) == PromptRows(100, 900, ((200, 300), (450, 451)))
    assert episode_prompt_rows(EPISODE, {"episode": episode_block()}, 1000) == PromptRows(0, 1000)
    with pytest.raises(ValueError, match="past the installed taskset"):
        episode_prompt_rows(EPISODE, block, 899)
    # a 1000-row registered env: rows past it fail the load (validator and miner alike)
    from tests.unit.episode_v2_fixtures import make_test_episode_env

    monkeypatch.setattr("reliquary.environment.registry.find_environment_spec",
                        lambda name: SimpleNamespace(interaction_mode="signed_episode"))
    monkeypatch.setattr("reliquary.environment.load_environment", lambda name: make_test_episode_env())
    assert len(load_episode_environment(EPISODE, block)) == 1000
    with pytest.raises(ValueError, match="past the installed taskset"):
        load_episode_environment(EPISODE, {**block, "rows": {"start": 0, "end": 1001}})
    # refused when the task is declared
    path = tmp_path / "eps.json"
    base = SimpleNamespace(mechanism="rl-discovered-price", env_split={ENV: 0.5, EPISODE: 0.5}, params={})
    monkeypatch.setattr("dataclasses.replace", lambda obj, **kw: SimpleNamespace(**{**vars(obj), **kw}))
    for bad in ({"start": 0}, {"start": 0, "end": 10, "exclude": [[2, 5], 4]}, {"start": 0, "end": 1, "exclude": [0]}):
        path.write_text(json.dumps({EPISODE: {"episode": episode_block(), "rows": bad}}))
        with pytest.raises(ValueError, match="rows"):
            with_rl_async_params(base, planned_tokens_per_period=1e6, episode_envs=_read_episode_envs(path))
    path.write_text(json.dumps({EPISODE: {"episode": episode_block(), "rows": rows}}))
    declared = with_rl_async_params(base, planned_tokens_per_period=1e6, episode_envs=_read_episode_envs(path))
    assert declared.params[EPISODE_ENVS_PARAM][EPISODE]["rows"] == rows


def test_an_rl_episode_is_bounded_by_the_rl_cap_not_the_corpus_one():
    from reliquary.protocol.service_contract import MAX_EPISODE_TOKENS, MAX_RL_EPISODE_TOKENS

    assert MAX_EPISODE_TOKENS == 60000 and MAX_RL_EPISODE_TOKENS == 65536
    block = validate_episode_block(EPISODE, {"episode": episode_block(max_tokens_per_turn=8192,
                                                                      max_episode_tokens=65536)})
    assert async_episode_policy(EPISODE, block).max_episode_tokens == 65536
    for bad in (episode_block(max_episode_tokens=65537), episode_block(max_tokens_per_turn=65537)):
        with pytest.raises(ValueError, match="max_"):
            validate_episode_block(EPISODE, {"episode": bad})


def test_the_settings_carry_the_episode_policies_and_a_lease_lifetime_with_sandbox_time():
    params = rl_async_params(planned_tokens_per_period=1e6, lease_tokens_per_s=20.0, lease_margin=2.0,
                             episode_envs={EPISODE: {"episode": episode_block()}})
    settings = rl_async_settings(entry(params), 0.3, run="r", max_tokens=lambda env: 4000)
    assert set(settings.episodes) == {EPISODE} and settings.episode_policies()[EPISODE].max_episode_tokens == 4096
    decode = 4096 / 20.0 * 2.0
    assert settings.lease_expiry_s[EPISODE] == decode + BUDGETS["wall_s"] + EPISODE_SESSION_SLACK_S
    assert settings.lease_expiry_s[ENV] == 4000 / 20.0 * 2.0                  # single-turn: unchanged rule
    assert episode_lease_expiry_s(settings.episode_policies()[EPISODE], reference_tokens_per_s=20.0, margin=2.0,
                                  slack_s=0.0) == decode + BUDGETS["wall_s"]
    # the registry overrides any env's lifetime; an override outside the split is refused
    params = {**params, LEASE_EXPIRY_PARAM: {EPISODE: 5400, ENV: 90.0}}
    settings = rl_async_settings(entry(params), 0.3, max_tokens=lambda env: 4000)
    assert settings.lease_expiry_s == {EPISODE: 5400.0, ENV: 90.0}
    with pytest.raises(ValueError, match="outside the env_split"):
        rl_async_settings(entry({**params, LEASE_EXPIRY_PARAM: {"other": 5.0}}), 0.3, max_tokens=lambda env: 4000)
    with pytest.raises(ValueError):
        rl_async_params(planned_tokens_per_period=1e6, lease_expiry_s={EPISODE: -1})


def test_the_memory_floor_is_read_strictly(monkeypatch):
    monkeypatch.delenv(MEMORY_FLOOR_ENV, raising=False)
    assert memory_floors() == {} and memory_floors("") == {}
    assert memory_floors(json.dumps({EPISODE: 4 * GIB})) == {EPISODE: 4 * GIB}
    for bad in ("{", "[]", '{"e": "4294967296"}', '{"e": 4.0}', '{"e": true}', '{"e": 0}', '{"e": -1}',
                json.dumps({"e": 8 * GIB + 1})):
        with pytest.raises(ValueError, match=MEMORY_FLOOR_ENV):
            memory_floors(bad)
    monkeypatch.setenv(MEMORY_FLOOR_ENV, "{nope")
    with pytest.raises(ValueError, match=MEMORY_FLOOR_ENV):     # refused at startup
        rl_async_settings(entry(rl_async_params(planned_tokens_per_period=1e6,
                                                episode_envs={EPISODE: {"episode": episode_block()}})), 0.3,
                          max_tokens=lambda env: 4000)


def test_the_memory_floor_only_raises_the_issued_memory_budget(monkeypatch):
    monkeypatch.delenv(MEMORY_FLOOR_ENV, raising=False)
    small = {**BUDGETS, "memory_bytes": 2 * GIB}
    blocks = {EPISODE: validate_episode_block(EPISODE, {"episode": episode_block(budgets=small)})}
    plain = async_episode_policy(EPISODE, blocks[EPISODE])
    assert episode_policies(blocks) == {EPISODE: plain}                  # no floor: the block's policy, unchanged
    raised = episode_policies(blocks, {EPISODE: 4 * GIB})[EPISODE]
    assert raised.budgets_dict() == {**small, "memory_bytes": 4 * GIB}
    assert raised == plain.__class__(**{**{f: getattr(plain, f) for f in plain.__slots__},
                                        "budgets": raised.budgets})     # nothing else moves
    assert episode_policies(blocks, {EPISODE: GIB})[EPISODE] == plain   # never lowered
    monkeypatch.setenv(MEMORY_FLOOR_ENV, json.dumps({EPISODE: 4 * GIB}))
    assert episode_policies(blocks)[EPISODE] == raised
    assert blocks[EPISODE]["episode"]["budgets"] == small                # the block (/rl/episodes) is untouched
    with pytest.raises(ValueError, match="not episode envs"):
        episode_policies(blocks, {"other": 4 * GIB})


def test_the_cli_declares_episode_envs_and_lease_lifetimes(tmp_path, monkeypatch):
    from reliquary.cli.main import _parse_lease_expiry, _read_episode_envs, with_rl_async_params

    path = tmp_path / "eps.json"
    path.write_text(json.dumps({EPISODE: {"episode": episode_block()}}))
    base = SimpleNamespace(mechanism="rl-discovered-price", env_split={ENV: 0.5, EPISODE: 0.5}, params={})
    monkeypatch.setattr("dataclasses.replace", lambda obj, **kw: SimpleNamespace(**{**vars(obj), **kw}))
    declared = with_rl_async_params(base, planned_tokens_per_period=1e6, episode_envs=_read_episode_envs(path),
                                    lease_expiry_s=_parse_lease_expiry(f"{EPISODE}=4000"))
    assert declared.params[EPISODE_ENVS_PARAM][EPISODE]["episode"]["env_package"] == ENV_PACKAGE
    assert declared.params[LEASE_EXPIRY_PARAM] == {EPISODE: 4000.0}
    with pytest.raises(ValueError, match="outside --env-split"):
        with_rl_async_params(base, planned_tokens_per_period=1e6, lease_expiry_s={"other": 1.0})
    with pytest.raises(ValueError):
        _parse_lease_expiry("nonsense")


# --- the /rl/submit body ------------------------------------------------------------------------------------------
def test_gzip_bodies_inflate_within_the_cap():
    raw = json.dumps({"x": "a" * 100_000}).encode()
    assert inflate_body(gzip_body(raw), cap=len(raw)) == raw
    with pytest.raises(OverflowError):
        inflate_body(gzip_body(raw), cap=len(raw) - 1)                          # no decompression bomb
    with pytest.raises(ValueError):
        inflate_body(b"not gzip", cap=10**6)
    with pytest.raises(ValueError):
        inflate_body(gzip_body(raw) + gzip_body(raw), cap=10**7)                # one stream only


# --- the async validator app with its sandbox side ------------------------------------------------------------------
def _settings(budgets=None, **kw) -> RlAsyncSettings:
    block = validate_episode_block(EPISODE, {"episode": episode_block(**({"budgets": budgets} if budgets else {}))})
    values = dict(task_id="rl-ep", cap=0.5, run="ep", targets={ENV: 0.5, EPISODE: 0.5}, planned_tokens_per_period=1e6,
                  lease_expiry_s={ENV: 600.0, EPISODE: 5000.0}, episodes={EPISODE: block})
    values.update(kw)
    return RlAsyncSettings(**values)


class ImageSource:
    """``episode_v2_fixtures.FixedSource`` with its image given (no import of the sandbox service's testing kit)."""

    def __init__(self, image):
        from tests.unit.episode_v2_fixtures import FixedSource

        self.inner, self.image = FixedSource(image=image), image

    def __len__(self):
        return len(self.inner)

    def prompt(self, index):
        return self.inner.prompt(index)

    async def resolve(self, index):
        from reliquary.sandbox.tasks import ResolvedTask

        return ResolvedTask(self.image, {})


class World:
    """The async validator (lease book, submissions, the sandbox side over a fake machine) and its ASGI app."""

    def __init__(self, tmp_path, monkeypatch, *, machine_ids=None, budgets=None, floor=None, capacity=32,
                 resources=None):
        attest = pytest.importorskip("reliquary_sandbox.attest")
        from reliquary.environment.signed_episode import SignedEpisodeEnvironment
        from reliquary.infrastructure.sandbox_store import MemorySessionStore
        from reliquary.validator.rl_submit import GroupGrader
        from reliquary.validator.sandbox_wiring import SandboxValidatorConfig
        from reliquary.validator.server import ValidatorServer
        from tests.unit.episode_v2_fixtures import episode_signers, register_episode_env
        from tests.unit.sandbox_fixtures import IMAGE, capacity_report, machine_document
        from tests.unit.test_trajectory_parse import FakeRenderer

        register_episode_env(monkeypatch)
        if floor is None:
            monkeypatch.delenv(MEMORY_FLOOR_ENV, raising=False)
        else:
            monkeypatch.setenv(MEMORY_FLOOR_ENV, json.dumps({EPISODE: floor}))
        self.memory = max(floor or 0, (budgets or BUDGETS)["memory_bytes"])     # what a session is issued
        # Without an installed reliquary-sandbox distribution (a source checkout on the path) no commit is known.
        monkeypatch.setattr("reliquary.sandbox.sandbox_commit_refusal", lambda pinned: None)
        self.attest = attest
        self.validator, self.machine = episode_signers(tmp_path / "keys")
        grader = GroupGrader({ENV: MathEnv()}, Tokenizer(), eos_token_ids=(1,))
        self.rl = build_rl_async(_settings(budgets), path=tmp_path / "rl-async.sqlite3", grader=grader,
                                 prompt_range={ENV: (0, 50), EPISODE: (0, 1000)}, collector_interval_s=0.0)
        self.rl.book.adopt_checkpoint(0)
        machine = self.machine
        self.fetched = []

        async def read_documents():
            return [machine_document(machine, capacity=capacity)]

        async def fetch_report(address):
            self.fetched.append(address)
            return capacity_report(machine, at=int(time.time()), capacity=capacity, free=capacity, images=(IMAGE,),
                                   env_packages={SANDBOX_ENV: ENV_PACKAGE}, resources=resources)

        config = SandboxValidatorConfig(key_file=Path("unused"), key_id=self.validator.key_id, retired_keys={},
                                        machine_ids=machine_ids)
        self.store = MemorySessionStore()
        wire_episode_sandbox(self.rl, config, validator_hotkey=VALIDATOR.ss58_address,
                             environments={EPISODE: SignedEpisodeEnvironment(EPISODE, ImageSource(image=IMAGE))},
                             renderer_for=lambda policy: FakeRenderer(), signer=self.validator,
                             session_store=self.store, read_documents=read_documents, fetch_report=fetch_report)
        self.server = ValidatorServer()
        mount_rl_async(self.server, self.rl, validator_hotkey=VALIDATOR.ss58_address)

    async def start(self):
        await self.rl.sandbox.start()
        await self.rl.sandbox.fleet.refresh_directory()
        await self.rl.sandbox.fleet.poll_once()

    def client(self):
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.server.app), base_url="http://validator")


@pytest.mark.parametrize("scope,capacity", [(None, 32), ({"machine-1"}, 32), (set(), 0), ({"unknown"}, 0)])
async def test_machine_scope_reaches_async_lease_polling_and_capacity(tmp_path, monkeypatch, scope, capacity):
    world = World(tmp_path, monkeypatch, machine_ids=scope)
    await world.start()
    fleet = world.rl.sandbox.fleet
    assert fleet.session_capacity() == capacity
    assert len(world.fetched) == (1 if capacity else 0)
    assert fleet.directory_if_ready().public_key("machine-1", world.machine.key_id, int(time.time())) is not None


@pytest.mark.parametrize("floor,hostable", [(None, 320), (4 * GIB, 160)])
async def test_the_fleet_counts_sessions_at_the_floored_memory(tmp_path, monkeypatch, floor, hostable):
    from reliquary.sandbox.rl_capacity import SessionTiers

    resources = {"cpu_millis": 320_000, "memory_bytes": 640 * GIB, "used_cpu_millis": 0, "used_memory_bytes": 0,
                 "free_cpu_millis": 320_000, "free_memory_bytes": 640 * GIB, "max_box_cpu_millis": 1_000,
                 "boxes_per_episode": 1, "cpu_overcommit_permille": 1_000, "present_window_s": 120, "sessions": []}
    world = World(tmp_path, monkeypatch, budgets={**BUDGETS, "memory_bytes": 2 * GIB}, floor=floor, capacity=320,
                  resources=resources)
    await world.start()
    try:
        tiers = SessionTiers(world.rl.settings.episode_policies())
        assert tiers.reservation(EPISODE)["memory_bytes"] == world.memory
        assert world.rl.sandbox.fleet.session_capacity(reservations=tiers.reservations()) == hostable
        assert await world.rl.sandbox.lease_refusal(EPISODE, 0) is None        # placement at the floored budget
    finally:
        await world.rl.sandbox.stop()
        world.rl.close()


def sign(binding: bytes) -> str:
    return MINER.sign(binding).hex()


async def open_session(client, lease_id, index):
    from reliquary.miner.rl_episode_client import signed_lease_open_request

    body = signed_lease_open_request(hotkey=MINER.ss58_address, lease_id=lease_id, rollout_index=index,
                                     sign_binding=sign, now=time.time(), request_id=os.urandom(16).hex(),
                                     validator_hotkey=VALIDATOR.ss58_address, path="/rl/sandbox/sessions")
    return await client.post("/rl/sandbox/sessions", json=body)


class RouteEpisodes:
    """The miner's episode generator stand-in: for each leased rollout a real session open on the validator's route,
    the episode the fake machine signs for the issued token (two turns, one bash call; reward by rollout parity),
    the real close; ``EpisodeRollout`` out, as ``EpisodeLeaseGenerator`` returns."""

    def __init__(self, world, client):
        self.world, self.client, self.opened, self.loads = world, client, [], []

    def generate(self, leases):
        raise RuntimeError("agenerate")

    def load_weights(self, state, version):
        self.loads.append(version)

    async def agenerate(self, leases):
        return [await self.play(lease) for lease in leases]

    async def play(self, lease):
        from reliquary_sandbox.attest import SessionClaims
        from reliquary_sandbox.observation import render_observation

        from reliquary.corpus.signed_parse import signed_records
        from reliquary.miner.rl_async import EpisodeRollout
        from reliquary.miner.signed_episode import signed_close_request
        from tests.unit.episode_v2_fixtures import FIRST_TURN_TEXT, PROMPT_TEXT
        from tests.unit.sandbox_fixtures import transcript
        from tests.unit.test_trajectory_parse import CALL, TERM, TEXT, FakeRenderer

        renderer, rollouts = FakeRenderer(), []
        for index in range(16):
            answer = await open_session(self.client, lease.lease_id, index)
            assert answer.status_code == 200, answer.text
            grant = answer.json()
            self.opened.append(grant["session_id"])
            claims = SessionClaims.from_dict(grant["token"]["claims"])
            assert claims.engagement == f"rl-lease:{lease.lease_id}:{index}" and claims.checkpoint == "k0"
            assert claims.budgets.memory_bytes == self.world.memory               # the floor, signed
            signed = transcript(self.world.validator, self.world.machine, claims, status="graded",
                                reward=float(index % 2 == 0), env_package=ENV_PACKAGE, at=claims.issued_at,
                                calls=[{"turn": 0, "k": 0, "arguments": {"command": "c0"}, "output": "ok"}])
            assert signed["token"] == grant["token"]                         # the machine's records name this token
            prompt = renderer.initial_ids(PROMPT_TEXT)
            first = [TEXT] * FIRST_TURN_TEXT + [CALL, TERM]
            observations = [render_observation(body.to_dict()) for body in signed_records(signed).calls[:1]]
            full = renderer.next_prompt(prompt, first, observations)
            tokens = full + [TEXT] * 9 + [TERM]
            p = len(prompt)
            close = signed_close_request(hotkey=MINER.ss58_address, session_id=grant["session_id"], reason="final",
                                         transcript=signed, sign_binding=sign, now=time.time(),
                                         request_id=os.urandom(16).hex(), validator_hotkey=VALIDATOR.ss58_address,
                                         path=f"/rl/sandbox/sessions/{grant['session_id']}/close")
            closed = await self.client.post(f"/rl/sandbox/sessions/{grant['session_id']}/close", json=close)
            assert closed.status_code == 200, closed.text
            rollouts.append(EpisodeRollout(tuple(tokens[p:]), ((0, len(first)), (len(full) - p, len(tokens) - p)),
                                           "agent_completed", signed))
        return rollouts


@pytest.mark.parametrize("floor", [None, 6 * GIB])
async def test_an_episode_lease_end_to_end_through_the_async_validator_app(tmp_path, monkeypatch, floor):
    from reliquary.miner.rl_async import AsyncMiner, AsyncMinerConfig, HttpRlTransport, PromptPlanner

    world = World(tmp_path, monkeypatch, floor=floor)
    await world.start()
    sent = []
    try:
        async with world.client() as client:
            async def record(request):
                if request.url.path.endswith("/rl/submit"):
                    sent.append((request.headers.get("content-encoding"), len(request.content)))

            client.event_hooks["request"] = [record]
            transport = HttpRlTransport(client, base_url="http://validator", hotkey=MINER.ss58_address,
                                        validator_hotkey=VALIDATOR.ss58_address, sign_binding=sign)
            # the miner reads the episode envs the validator grades with
            episodes = await transport.episodes()
            assert set(episodes) == {EPISODE} and episodes[EPISODE]["episode"] == episode_block()
            assert episodes[EPISODE]["lease_expiry_s"] == 5000.0
            generator = RouteEpisodes(world, client)
            miner = AsyncMiner(AsyncMinerConfig(environments=(EPISODE,), lease_batch=1, batch_groups=1,
                                                submit_backoff_s=(0.0,)),
                               transport=transport, generator=generator,
                               planners={EPISODE: PromptPlanner(EPISODE, (0, 1000))}, loaded_version=0)
            await miner.refresh_quota()
            assert miner.planners[EPISODE].lease_expiry_s == 5000.0             # sandbox time included
            assert await miner.lease_more() == 1
            (lease,) = miner.queue
            assert await miner.generate_once() == 1
            await miner.drain()
            assert miner.stats["verdicts"] == {"in_zone": 1}, miner.stats
            assert sent and sent[-1][0] == "gzip"                               # the 16 transcripts, gzipped
            (buffered,) = world.rl.buffer.take(4)
            assert buffered.spans is not None and len(buffered.spans) == 16
            group = world.rl.submissions.group(lease.lease_id)
            model = sum(e - s for s, e in buffered.spans[0])
            assert group["completion_tokens"] == 16 * model                     # model tokens only

            # one session per (lease, rollout); the lease closed at its submission
            again = await open_session(client, lease.lease_id, 3)
            assert again.status_code != 200 and again.json()["reason"] in ("lease_closed", "engagement_taken")
            # the sessions are the lease's, stored under the async prefix, closed final
            records = {r.session_id: r for r in world.rl.sandbox.book.records()}
            assert set(generator.opened) <= set(records)
            assert {records[s].engagement for s in generator.opened} == {
                f"rl-lease:{lease.lease_id}:{i}" for i in range(16)}
    finally:
        await world.rl.sandbox.stop()
        world.rl.close()


async def test_a_second_open_of_a_rollout_and_a_foreign_hotkey_are_refused(tmp_path, monkeypatch):
    from reliquary.validator.rl_leases import LeaseRequest

    world = World(tmp_path, monkeypatch)
    await world.start()
    try:
        lease = world.rl.book.grant([LeaseRequest(MINER.ss58_address, EPISODE, (7,), 0)])[0].leases[0]
        async with world.client() as client:
            first = await open_session(client, lease.lease_id, 0)
            assert first.status_code == 200, first.text
            second = await open_session(client, lease.lease_id, 0)
            assert second.json()["reason"] == "engagement_taken"
            out = await open_session(client, lease.lease_id, 16)
            assert out.json()["reason"] == "seed_out_of_pool"
        other = world.rl.book.grant([LeaseRequest(VALIDATOR.ss58_address, EPISODE, (8,), 0)])[0].leases[0]
        async with world.client() as client:
            foreign = await open_session(client, other.lease_id, 0)
            assert foreign.json()["reason"] == "lease_not_yours"
    finally:
        await world.rl.sandbox.stop()
        world.rl.close()


async def test_the_submit_route_caps_bodies_per_kind_and_reads_gzip(tmp_path, monkeypatch):
    """A single-turn group stays within 8 MiB (even gzipped, even on a validator serving episodes); a malformed gzip
    body is refused; an unknown encoding is refused."""
    from reliquary.protocol.signatures import build_rl_submit_binding
    from reliquary.protocol.rl_submit import HOTKEY_HEADER, rollouts_digest
    from reliquary.validator.rl_leases import LeaseRequest

    world = World(tmp_path, monkeypatch)
    await world.start()
    try:
        lease = world.rl.book.grant([LeaseRequest(MINER.ss58_address, ENV, (3,), 0)])[0].leases[0]
        chunk = encode_tokens(list(range(10, 10 + 1)))
        pad = "A" * (MAX_SUBMIT_BODY_BYTES // 16 + 1024 - len(chunk))
        rollouts = [{"index": i, "tokens": chunk + pad} for i in range(16)]    # not decodable, but signed
        at = int(time.time())
        digest = rollouts_digest([(r["index"], r["tokens"], None) for r in rollouts])
        body = {"miner_hotkey": MINER.ss58_address, "at": at, "lease_id": lease.lease_id, "rollouts": rollouts,
                "signature": sign(build_rl_submit_binding(hotkey=MINER.ss58_address, at=at, lease_id=lease.lease_id,
                                                          rollouts_digest=digest,
                                                          validator_hotkey=VALIDATOR.ss58_address,
                                                          path="/rl/submit"))}
        raw = json.dumps(body).encode()
        assert len(raw) > MAX_SUBMIT_BODY_BYTES
        async with world.client() as client:
            mine = {HOTKEY_HEADER: MINER.ss58_address}
            answer = await client.post("/rl/submit", content=gzip_body(raw), headers={
                **mine, "Content-Encoding": "gzip", "Content-Type": "application/json"})
            assert answer.status_code == 413 and answer.json()["reason"] == "body_too_large"
            bad = await client.post("/rl/submit", content=b"\x1f\x8b junk", headers={**mine, "Content-Encoding": "gzip"})
            assert bad.status_code == 422
            odd = await client.post("/rl/submit", content=b"{}", headers={**mine, "Content-Encoding": "br"})
            assert odd.status_code == 422 and "encoding" in odd.json()["detail"]["why"]
    finally:
        await world.rl.sandbox.stop()
        world.rl.close()


def test_a_validator_without_episode_envs_keeps_the_8_mib_cap(tmp_path):
    from fastapi.testclient import TestClient

    from reliquary.validator.rl_submit import GroupGrader
    from reliquary.validator.server import ValidatorServer

    rl = build_rl_async(_settings(targets={ENV: 1.0}, episodes={}, lease_expiry_s={ENV: 600.0}),
                        path=tmp_path / "a.sqlite3", grader=GroupGrader({ENV: MathEnv()}, Tokenizer(),
                                                                        eos_token_ids=(1,)),
                        prompt_range={ENV: (0, 50)})
    server = ValidatorServer()
    mount_rl_async(server, rl, validator_hotkey=VALIDATOR.ss58_address)
    client = TestClient(server.app)
    from reliquary.protocol.rl_submit import HOTKEY_HEADER

    big = b"{" + b" " * (MAX_SUBMIT_BODY_BYTES + 10) + b"}"
    assert client.post("/rl/submit", content=big, headers={HOTKEY_HEADER: MINER.ss58_address}).status_code == 413
    assert client.get("/rl/episodes").status_code == 404
    rl.close()


# --- mine-async: the episode generator's wiring -------------------------------------------------------------------
class FakeLLM:
    def __init__(self):
        self.pushed, self.resets, self.fail = [], 0, False

    def collective_rpc(self, fn, args=()):
        if self.fail:
            raise RuntimeError("push refused")
        self.pushed.append([Path(p).name for p in args[0]])

    def reset_prefix_cache(self, reset_running_requests=False):
        self.resets += 1
        self.reset_running = reset_running_requests


class FakeCore:
    def __init__(self, model_dir, **kwargs):
        self.model_dir, self.kwargs = model_dir, kwargs
        self._llm = FakeLLM()

    def has_unfinished(self):
        return False

    def step(self):
        return []


def fake_deps(made, released, served):
    from reliquary.miner.episode_mining import EpisodeStackDeps
    from tests.unit.test_trajectory_parse import FakeRenderer

    def make_core(model_dir, **kwargs):
        made.append(FakeCore(model_dir, **kwargs))
        return made[-1]

    async def serve(app, port):
        served.append((app, port))
        done = asyncio.get_running_loop().create_future()
        done.set_result(None)
        return SimpleNamespace(should_exit=False), done

    return EpisodeStackDeps(load_renderer=lambda directory, tools: FakeRenderer(), make_core=make_core,
                            release_core=released.append, serve=serve,
                            sync_http=lambda url: httpx.Client(base_url=url), package_refusal=lambda policy: None,
                            sampling=lambda: SimpleNamespace(temperature=1.0, top_p=1.0, top_k=0))


def base_model(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    (base / "tokenizer.json").write_text("{}")
    return base


def episode_source(name, block):
    from tests.unit.episode_v2_fixtures import FixedSource

    return SimpleNamespace(source=FixedSource(rows=40))


async def test_the_episode_generator_pushes_each_checkpoint_into_its_engine(tmp_path):
    import torch

    from reliquary.miner.rl_async_episodes import EpisodeEngineHost, build_episode_generator
    from reliquary.miner.rl_episode_client import LeaseEpisodeRunner
    from reliquary.miner.rl_async import HeldLease

    made, released, served = [], [], []
    generator = build_episode_generator(
        {EPISODE: {"episode": episode_block()}}, validator_url="http://v", validator_hotkey=VALIDATOR.ss58_address,
        hotkey=MINER.ss58_address, sign_binding=sign, base_model_dir=str(base_model(tmp_path)),
        work_dir=str(tmp_path / "work"), deps=fake_deps(made, released, served), load_environment=episode_source)
    host = generator.host
    assert isinstance(host, EpisodeEngineHost) and host.engine is None and len(generator.sources[EPISODE]) == 40
    assert generator.policies[EPISODE].pool_seeds == 32
    with pytest.raises(RuntimeError, match="no weights"):
        await host.ensure_serving()
    state = {"w": torch.zeros(2, 2)}
    await asyncio.to_thread(generator.load_weights, state, 3)
    first = host.engine
    assert made[-1].model_dir.endswith("episode-weights-00000003") and (Path(made[-1].model_dir) / "config.json").is_file()
    assert made[-1].kwargs["max_total_tokens"] == 4096 and made[-1].kwargs["stop_token_ids"]
    await host.ensure_serving()
    assert served[0][1] == 8013 and host.take_session.__self__ is first          # the app reaches the engine
    # a new checkpoint: pushed into the same core by the same engine, turns in flight resumed, no model load
    await asyncio.to_thread(generator.load_weights, state, 4)
    assert released == [] and len(made) == 1 and host.restarts == 1 and host.core is made[0]
    assert host.engine is first and made[0]._llm.pushed and made[0]._llm.resets == 1 and made[0]._llm.reset_running
    # a push that fails falls back to a restart on an HF directory of the new weights
    made[0]._llm.fail = True
    await asyncio.to_thread(generator.load_weights, state, 5)
    assert released == [made[0]] and host.restarts == 2 and made[1].model_dir.endswith("episode-weights-00000005")
    assert not Path(made[0].model_dir).exists() and Path(made[1].model_dir).is_dir()
    lease = HeldLease("L1", EPISODE, 5, 4, "ab" * 32, time.time() + 600)
    runner = generator._factory(generator.policies[EPISODE], lease, "prompt")
    assert isinstance(runner, LeaseEpisodeRunner) and runner._lease is lease
    await host.close()
    assert released == [made[0], made[1]] and len(served) == 1


def test_one_engine_serves_only_envs_with_the_same_caps(tmp_path):
    from reliquary.miner.rl_async_episodes import build_episode_generator

    other = "reliquary_test_episode_v2"
    blocks = {EPISODE: {"episode": episode_block()}, other: {"episode": episode_block(max_tokens_per_turn=256)}}
    with pytest.raises(ValueError, match="miner process of its own"):
        build_episode_generator(blocks, validator_url="http://v", validator_hotkey=VALIDATOR.ss58_address,
                                hotkey=MINER.ss58_address, sign_binding=sign, base_model_dir=str(tmp_path),
                                work_dir=str(tmp_path), deps=fake_deps([], [], []), load_environment=episode_source)


async def test_mine_async_routes_episode_envs_to_the_episode_engine_and_the_rest_to_vllm(tmp_path):
    from reliquary.miner.rl_async import AsyncMinerConfig, HeldLease, miner_generator
    from reliquary.miner.rl_async_episodes import MixedGenerator

    class Single:
        def __init__(self, **kw):
            self.kw, self.loads = kw, []

        def generate(self, leases):
            return [[[1, 2]] * 16 for _ in leases]

        def load_weights(self, state, version):
            self.loads.append(version)

    class Episodes:
        policies = {EPISODE: object()}

        def __init__(self, blocks, **kw):
            self.blocks, self.kw, self.loads = blocks, kw, []

        async def agenerate(self, leases):
            return [[("episode", lease.lease_id)] * 16 for lease in leases]

        def load_weights(self, state, version):
            self.loads.append(version)

    wallet = SimpleNamespace(hotkey=MINER)
    common = dict(base_model_dir=str(tmp_path), work_dir=str(tmp_path), wallet=wallet, build_single=Single,
                  build_episodes=Episodes, validator_url="http://v", validator_hotkey="5V")
    only_single = miner_generator(AsyncMinerConfig(environments=(ENV,)), blocks={}, **common)
    assert isinstance(only_single, Single)
    blocks = {EPISODE: {"episode": episode_block()}}
    only_episodes = miner_generator(AsyncMinerConfig(environments=(EPISODE,)), blocks=blocks,
                                    episode_options={"generate_port": 9001}, **common)
    assert isinstance(only_episodes, MixedGenerator) and only_episodes.single is None
    assert only_episodes.episodes.kw["generate_port"] == 9001 and only_episodes.episodes.kw["hotkey"] == MINER.ss58_address
    mixed = miner_generator(AsyncMinerConfig(environments=(ENV, EPISODE)), blocks=blocks, **common)
    leases = [HeldLease(f"L{i}", env, i, 0, "ab" * 32, 1e12) for i, env in enumerate((EPISODE, ENV, EPISODE))]
    groups = await mixed.agenerate(leases)
    assert groups[0][0] == ("episode", "L0") and groups[1][0] == [1, 2] and groups[2][0] == ("episode", "L2")
    mixed.load_weights({}, 7)
    assert mixed.single.loads == [7] and mixed.episodes.loads == [7]


def test_the_cli_entry_splits_onto_the_episode_envs():
    from reliquary.cli.main import build_task_entry
    from reliquary.constants import PROTOCOL_PROFILE_ID
    from reliquary.protocol.profiles import resolve_protocol_profile

    envs = resolve_protocol_profile(PROTOCOL_PROFILE_ID).environments
    split = {**{env: 0.0 for env in envs}, EPISODE: 1.0}
    entry = build_task_entry(task_id="rl-ep", profile_id=PROTOCOL_PROFILE_ID, cap=1.0, overrides={},
                             env_split=split, episode_envs=(EPISODE,))
    assert entry.env_split[EPISODE] == 1.0
    with pytest.raises(ValueError, match="does not declare"):
        build_task_entry(task_id="rl-ep", profile_id=PROTOCOL_PROFILE_ID, cap=1.0, overrides={}, env_split=split)
