"""The RL validator's sandbox side (plan 2C, Task 12) and the deferred items it closes."""
import asyncio
import dataclasses
import json
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

attest = pytest.importorskip("reliquary_sandbox.attest")

from reliquary.infrastructure import sandbox_store  # noqa: E402
from reliquary.infrastructure.sandbox_store import MemorySessionStore  # noqa: E402
from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS  # noqa: E402
from reliquary.protocol.submission import RejectReason  # noqa: E402
from reliquary.sandbox import rl_engagements  # noqa: E402
from reliquary.sandbox.routes import REFUSAL_STATUS  # noqa: E402
from reliquary.sandbox.sessions import (  # noqa: E402
    HANDED_BACK, SUBMITTED, SandboxPolicy, SessionBook,
)
from reliquary.validator.cooldown import CooldownMap  # noqa: E402
from reliquary.validator.rl_sandbox_wiring import (  # noqa: E402
    RL_PREFIX, RlEpisodeServices, build_rl_episode_services, episode_stop_set_refusal, rl_sandbox_policy,
    stop_ids_from_metadata,
)
from reliquary.validator.sandbox_wiring import SandboxValidatorConfig  # noqa: E402
from reliquary.validator.service import ValidationService  # noqa: E402
from tests.unit.episode_v2_fixtures import (  # noqa: E402
    EPISODE, WINDOW_BEACON, episode_precommit, episode_runtime, episode_signers,
    make_test_episode_env,
)
from tests.unit.test_trajectory_parse import EOT, TERM, FakeRenderer  # noqa: E402

PATHS = {f"{RL_PREFIX}/sandbox/sessions", f"{RL_PREFIX}/sandbox/sessions/{{session_id}}/close",
         f"{RL_PREFIX}/episodes/precommit"}
VALIDATOR_SS58 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"     # //Bob
STOPS = {TERM, EOT}


def _paths(app) -> set[str]:
    return set(app.openapi()["paths"])


def _services(tmp_path, *, store=None, **kwargs):
    validator, _machine = episode_signers(tmp_path / "keys")
    rt = kwargs.pop("runtime", None) or episode_runtime(tmp_path / "rt")
    config = SandboxValidatorConfig(key_file=Path("unused"), key_id=validator.key_id, retired_keys={})

    async def no_machines():
        return []

    services = build_rl_episode_services(
        config, validator_hotkey=VALIDATOR_SS58, runtime=rt, environments={EPISODE: make_test_episode_env()},
        renderer_for=lambda policy: FakeRenderer(), current_window=lambda: 1, chunk_tokens=32, signer=validator,
        session_store=store if store is not None else MemorySessionStore(), read_documents=no_machines, **kwargs)
    return services, rt


def test_the_services_serve_the_rl_routes_and_one_checker_per_episode_env(tmp_path):
    services, rt = _services(tmp_path)
    app = FastAPI()
    for router in services.routers:
        app.include_router(router)
    assert PATHS <= _paths(app)
    assert services.intake.environments == (EPISODE,) == services.environments
    assert set(services.issuer._engagements) == {"rl_precommit"}
    engagements = services.issuer._engagements["rl_precommit"]
    # item 11: the precommit age is read from the runtime's own row
    assert engagements._recorded_at == rt.episode_precommit_recorded_at


def test_the_rl_policy_holds_every_group_in_flight_and_scales_its_caps(tmp_path):
    """Item 7: max_live_per_hotkey >= (2M + 1) per group in flight; the hourly-open and daily-abort caps
    are scaled with it."""
    services, rt = _services(tmp_path)
    pool = rt.contract.episode_policy(EPISODE).pool_seeds
    policy = services.book.policy
    assert policy.max_live_per_hotkey >= (pool + 1) * 2
    default = SandboxPolicy()
    scale = -(-policy.max_live_per_hotkey // default.max_live_per_hotkey)
    assert policy.max_aborted_per_day == default.max_aborted_per_day * scale > default.max_aborted_per_day
    assert policy.max_opens_per_hour == default.max_opens_per_hour * scale
    roomy = SandboxPolicy(max_live_per_hotkey=1000, max_aborted_per_day=5000)
    assert rl_sandbox_policy(roomy, pool).max_live_per_hotkey == 1000
    assert rl_sandbox_policy(roomy, pool).max_aborted_per_day == 5000


async def test_start_restores_the_rl_sessions_and_fails_closed(tmp_path):
    """Item 10: the RL store is read back before serving; a store that cannot be read stops the start."""
    store = MemorySessionStore()
    services, _ = _services(tmp_path, store=store)
    record = _record("s-restored", sha="a" * 64)
    store.documents[record.session_id] = record.to_document()
    await services.start()
    assert services.book.get("s-restored") is not None
    await services.stop()
    broken = MemorySessionStore()
    broken.fail = True

    async def boom(now):
        raise OSError("bucket down")

    broken.list_recent = boom
    failing, _ = _services(tmp_path / "b", store=broken)
    failing.restore_attempts, failing.restore_backoff_s = 2, 0.0
    with pytest.raises(OSError):
        await failing.start()


def test_the_rl_session_store_is_kept_under_its_own_prefix(tmp_path, monkeypatch):
    seen = {}

    class Store:
        def __init__(self, *, prefix, **kw):
            seen["prefix"] = prefix

    monkeypatch.setattr(sandbox_store, "R2SessionStore", Store)
    validator, _ = episode_signers(tmp_path / "keys")
    config = SandboxValidatorConfig(key_file=Path("unused"), key_id=validator.key_id, retired_keys={})
    build_rl_episode_services(
        config, validator_hotkey=VALIDATOR_SS58, runtime=episode_runtime(tmp_path / "rt"),
        environments={EPISODE: make_test_episode_env()}, renderer_for=lambda policy: FakeRenderer(),
        current_window=lambda: 1, chunk_tokens=32, signer=validator, read_documents=lambda: [])
    assert seen["prefix"] == sandbox_store.RL_SESSION_PREFIX != sandbox_store.SESSION_PREFIX


def test_every_rl_engagement_refusal_has_an_explicit_status():
    """Items 10-11: the RL book's refusals are mapped, never the UNMAPPED fallback."""
    source = Path(rl_engagements.__file__).read_text()
    emitted = set(re.findall(r'Refusal\(\s*"([a-z_]+)"', source))
    assert {"precommit_unknown", "precommit_stale", "seed_out_of_pool", "environment_not_served",
            "session_too_long"} <= emitted
    assert emitted <= set(REFUSAL_STATUS), emitted - set(REFUSAL_STATUS)
    assert {r: REFUSAL_STATUS[r] for r in ("precommit_unknown", "precommit_stale", "seed_out_of_pool",
                                           "session_too_long", "environment_not_served")} == {
        "precommit_unknown": 409, "precommit_stale": 409, "seed_out_of_pool": 409, "session_too_long": 409,
        "environment_not_served": 503}


# --- the validation service ---
def _service(tmp_path, monkeypatch, *, runtime, key=True, stops=STOPS):
    from tests.unit.test_corpus_job_store import _FakeMultiObjectR2

    fake = _FakeMultiObjectR2()
    monkeypatch.setattr(sandbox_store, "get_s3_client", lambda **kw: fake)
    if key:
        validator_dir = tmp_path / "validator-key"
        validator_dir.mkdir(parents=True, exist_ok=True)
        key_file = validator_dir / "v.pem"
        key_file.write_bytes(attest.generate_private_key_pem())
        key_file.chmod(0o600)
        monkeypatch.setenv("RELIQUARY_SANDBOX_VALIDATOR_KEY_FILE", str(key_file))
        monkeypatch.setenv("RELIQUARY_SANDBOX_VALIDATOR_KEY_ID", "v1")
    else:
        monkeypatch.delenv("RELIQUARY_SANDBOX_VALIDATOR_KEY_FILE", raising=False)
        monkeypatch.delenv("RELIQUARY_SANDBOX_VALIDATOR_KEY_ID", raising=False)

    async def registered(hotkey):
        return None

    service = ValidationService.__new__(ValidationService)
    service._service_runtime = runtime
    if runtime is not None:
        runtime.task_in_cooldown = lambda *_: False
    service.envs = {EPISODE: make_test_episode_env()}
    service.wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address=VALIDATOR_SS58))
    service.server = SimpleNamespace(app=FastAPI(), _episode_intake=None, _active_batcher_values=lambda: (),
                                     _registration_reject_reason=registered)
    service.tokenizer = SimpleNamespace(eos_token_id=sorted(stops))
    service.verify_model = None
    service._proof_worker_pool = None
    service._episode_renderer_for = lambda policy: FakeRenderer()
    return service


async def test_a_legacy_task_builds_nothing(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch, runtime=None)
    await service._start_episode_services()
    assert getattr(service, "_episode_services", None) is None and service.server._episode_intake is None


async def test_an_episode_order_mounts_its_routes_and_its_intake(tmp_path, monkeypatch):
    monkeypatch.setattr("reliquary.protocol.profiles.toploc_proof", lambda profile: TOPLOC_DEPLOYED_DEFAULTS)
    service = _service(tmp_path, monkeypatch, runtime=episode_runtime(tmp_path / "rt"))
    await service._start_episode_services()
    try:
        assert PATHS <= _paths(service.server.app)
        assert service.server._episode_intake is service._episode_services.intake
        assert len(service._episode_tasks) == 3            # fleet, session maintenance, precommit pruning
        intake = service._episode_services.intake
        # item 13 wiring: the intake takes groups only for a window this process noted
        assert intake._window_open(5) is False
        service._note_episode_window(5, resumed=False)
        assert intake._window_open(5) is True
        # item 2 wiring: the retention the service computes
        assert service._episode_services.precommit_retention_s == service._episode_precommit_retention_s()
    finally:
        await service._stop_episode_services()
    assert service._episode_services is None and service._episode_tasks == []


async def test_an_episode_order_refuses_to_start_without_its_key_or_toploc(tmp_path, monkeypatch):
    rt = episode_runtime(tmp_path / "rt")
    monkeypatch.setattr("reliquary.protocol.profiles.toploc_proof", lambda profile: None)
    with pytest.raises(ValueError, match="TOPLOC"):
        await _service(tmp_path, monkeypatch, runtime=rt)._start_episode_services()
    monkeypatch.setattr("reliquary.protocol.profiles.toploc_proof", lambda profile: TOPLOC_DEPLOYED_DEFAULTS)
    with pytest.raises(ValueError, match="RELIQUARY_SANDBOX_VALIDATOR_KEY_FILE"):
        await _service(tmp_path / "nokey", monkeypatch, runtime=rt, key=False)._start_episode_services()


async def test_an_episode_order_refuses_to_start_without_the_cooldown_hook(tmp_path, monkeypatch):
    """Item 1: the hook is asserted at boot (no silent fail-open)."""
    monkeypatch.setattr("reliquary.protocol.profiles.toploc_proof", lambda profile: TOPLOC_DEPLOYED_DEFAULTS)
    service = _service(tmp_path, monkeypatch, runtime=episode_runtime(tmp_path / "rt"))
    service._service_runtime.task_in_cooldown = None
    with pytest.raises(ValueError, match="cooldown"):
        await service._start_episode_services()
    assert getattr(service, "_episode_services", None) is None


async def test_an_episode_order_refuses_to_start_when_a_renderer_stop_is_not_proven(tmp_path, monkeypatch):
    """Item 14: a renderer stop outside the proof's stop set refuses the start."""
    monkeypatch.setattr("reliquary.protocol.profiles.toploc_proof", lambda profile: TOPLOC_DEPLOYED_DEFAULTS)
    service = _service(tmp_path, monkeypatch, runtime=episode_runtime(tmp_path / "rt"), stops={TERM})
    with pytest.raises(ValueError, match="outside the proof's stop set"):
        await service._start_episode_services()


def test_the_stop_sets_must_agree():
    """Item 14: renderer stops within the proof's, and the remote worker's set equal to the batcher's."""
    assert episode_stop_set_refusal({EPISODE: STOPS}, STOPS) is None
    assert episode_stop_set_refusal({EPISODE: STOPS}, STOPS | {7}, STOPS | {7}) is None
    assert "outside" in episode_stop_set_refusal({EPISODE: STOPS}, {TERM})
    assert "empty" in episode_stop_set_refusal({EPISODE: STOPS}, set())
    assert "differs" in episode_stop_set_refusal({EPISODE: STOPS}, STOPS, {TERM})
    # a remote worker's metadata: generation config, config and its nested text config, plus the tokenizer
    assert stop_ids_from_metadata({"text_config": {"eos_token_id": 9}}, {"eos_token_id": [TERM]},
                                  SimpleNamespace(eos_token_id=EOT)) == {TERM, EOT, 9}


def test_the_service_reads_the_remote_workers_stop_set(tmp_path):
    service = ValidationService.__new__(ValidationService)
    service.tokenizer = SimpleNamespace(eos_token_id=EOT)
    proxy = SimpleNamespace(config=SimpleNamespace(eos_token_id=TERM), generation_config=None)
    service._proof_models = {"gpu0": proxy}
    service._proof_worker_pool = SimpleNamespace(health=SimpleNamespace(
        config={"eos_token_id": TERM, "text_config": {"eos_token_id": 9}}, generation_config={}))
    service.verify_model = None
    proof, remote = service._episode_stop_sets()
    assert proof == {TERM, EOT} and remote == {TERM, EOT, 9}
    assert episode_stop_set_refusal({EPISODE: STOPS}, proof, remote) is not None


def test_the_current_window_is_the_announced_service_window_this_process_opened(tmp_path):
    """Items 4 and 13: memory only (the runtime is never read), and a resumed window is not open."""
    rt = episode_runtime(tmp_path)
    service = ValidationService.__new__(ValidationService)

    class NoRuntime:
        def __getattr__(self, name):
            raise AssertionError(f"the current window read the runtime ({name})")

    service._service_runtime = NoRuntime()
    announced = SimpleNamespace(service_policy=rt.announcement(window=1, randomness=WINDOW_BEACON), window_start=1)
    service.server = SimpleNamespace(_active_batcher_values=lambda: (announced,))
    assert service._current_service_window() is None                     # not opened by this process
    service._note_episode_window(1, resumed=False)
    assert service._current_service_window() == 1
    service.server = SimpleNamespace(_active_batcher_values=lambda: (SimpleNamespace(service_policy=None, window_start=9),))
    assert service._current_service_window() is None


def test_a_window_frozen_before_this_process_opened_it_takes_no_episode(tmp_path):
    """Item 13: a restart never resumes a window for episode intake; an earlier open of this process does."""
    rt = episode_runtime(tmp_path)
    service = ValidationService.__new__(ValidationService)
    assert ValidationService._service_window_frozen(rt, 1) is True
    assert ValidationService._service_window_frozen(rt, 2) is False
    service._note_episode_window(1, resumed=True)
    assert service._episode_window_started_at(1) is None
    service._note_episode_window(2, resumed=False)
    first = service._episode_window_started_at(2)
    service._note_episode_window(2, resumed=True)                          # this process froze it earlier
    assert service._episode_window_started_at(2) == first
    for window in range(3, 9):
        service._note_episode_window(window, resumed=False)
    assert sorted(service._episode_windows_opened) == [5, 6, 7, 8]        # bounded memory


def test_the_service_opens_a_window_for_episodes_only_when_it_froze_it(tmp_path):
    """Item 13 through the real ``_open_service_window``."""
    rt = episode_runtime(tmp_path)                                       # window 1 frozen "before the restart"
    service = ValidationService.__new__(ValidationService)
    service._service_runtime = rt
    service._episode_services = object()
    service._refresh_service_active = lambda: None
    plan = dict(window=1, pools={name: 0.25 for name in rt.contract.environments},
                env_mix=[(EPISODE, 1.0)])
    from reliquary.services.runtime import protocol_slot_geometry

    plan["picks_target"], plan["batch_slots"] = protocol_slot_geometry()
    service._candidate_service_window = plan
    service._open_service_window(1, WINDOW_BEACON)
    assert service._episode_window_started_at(1) is None
    service._candidate_service_window = dict(plan, window=2)
    service._open_service_window(2, WINDOW_BEACON)
    assert service._episode_window_started_at(2) is not None


def test_an_episode_group_of_a_resumed_window_is_refused_before_any_check(tmp_path):
    """Item 13 at the intake."""
    from tests.unit.test_episode_intake import admit, world

    w = world(tmp_path, intake_kwargs={"window_open": lambda window: False})
    prepared, claim = admit(w)
    assert claim is None and prepared.reject_reason is RejectReason.WINDOW_NOT_ACTIVE
    assert prepared.reject_stage == "episode_window_resumed" and w.sessions.claimed == []
    w = world(tmp_path / "open", intake_kwargs={"window_open": lambda window: window == 1})
    _, claim = admit(w)
    assert claim is not None


def test_the_service_starts_the_rl_side_before_serving():
    import inspect

    source = inspect.getsource(ValidationService.run)
    assert source.index('startup_step("episode_services"') < source.index("await self.server.start()")
    assert "await self._stop_serving()" in source


def _stopping_service(calls, *, fail=None):
    service = ValidationService.__new__(ValidationService)

    def step(name):
        async def run():
            calls.append(name)
            if name == fail:
                raise RuntimeError(name)
        return run

    service._stop_observation_publication = step("observations")
    service._close_proof_scheduler = step("proof_scheduler")
    service._stop_episode_services = step("episode_services")
    service.server = SimpleNamespace(stop=step("server"))
    service._service_runtime = SimpleNamespace(close=lambda: calls.append("runtime"))
    return service


async def test_the_episode_services_stop_after_the_proof_plane_and_the_server():
    """I1: late hand-backs (from the proof plane's last verdicts) still find the services running."""
    calls = []
    await _stopping_service(calls)._stop_serving()
    assert calls == ["observations", "proof_scheduler", "server", "episode_services", "runtime"]
    calls = []
    with pytest.raises(RuntimeError):
        await _stopping_service(calls, fail="server")._stop_serving()
    assert calls == ["observations", "proof_scheduler", "server", "episode_services", "runtime"]


def test_the_validators_prompt_cooldown_hook_reads_its_live_maps():
    """Item 1 (replaces the source-grep test): the hook the runtime gets, against real CooldownMaps."""
    service = ValidationService.__new__(ValidationService)
    cooling = CooldownMap(cooldown_windows=3)
    cooling.record_batched(7, window=10)
    service._cooldown_per_env = {EPISODE: cooling}
    assert service._episode_task_in_cooldown(EPISODE, 7, 11) is True
    assert service._episode_task_in_cooldown(EPISODE, 8, 11) is False
    assert service._episode_task_in_cooldown("other_env", 7, 11) is False
    fresh = CooldownMap(cooldown_windows=3)                              # a new window replaced the map
    service._cooldown_per_env = {EPISODE: fresh}
    assert service._episode_task_in_cooldown(EPISODE, 7, 11) is False


# --- item 6: a stalled window opens no session ---
def test_a_session_is_refused_when_its_window_is_older_than_the_longest_window(tmp_path):
    from reliquary.constants import FILL_CLOSED_MAX_SECONDS
    from tests.unit.sandbox_fixtures import NOW
    from tests.unit.test_rl_engagements import build, engagement

    env = build(tmp_path)
    started = {1: float(NOW)}
    terms = env.issuer._engagements["rl_precommit"]
    terms._window_started_at = started.get
    env.wall["now"] = float(NOW) + FILL_CLOSED_MAX_SECONDS + rl_engagements.WINDOW_AGE_MARGIN_S - 1
    assert asyncio.run(terms.terms("5Hot", engagement(env.precommit, 0))).exclusive is True
    env.wall["now"] += 2
    refused = asyncio.run(terms.terms("5Hot", engagement(env.precommit, 0)))
    assert refused.reason == "precommit_stale" and "old" in refused.detail["why"]
    started.clear()                                                       # a window this process did not open
    env.wall["now"] = float(NOW)
    assert asyncio.run(terms.terms("5Hot", engagement(env.precommit, 0))).reason == "precommit_stale"


# --- item 5 (+ fix round 1, I2/I3/I3b): the precommit route is bounded before its signature check ---
ALICE = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
CHARLIE = "5FLSigC9HGRKVhB9FiEo4Y3koPsNmBmLJbpXg2mp1hXcS59Y"
DAVE = "5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy"


def _recorded(value):
    return True, "a" * 64


def _precommit_router(rt, *, verify, record=_recorded, **kwargs):
    from reliquary.sandbox.rl_routes import build_episode_precommit_router

    return build_episode_precommit_router(
        record=record, current_window=lambda: 1, validator_hotkey=VALIDATOR_SS58,
        policy=SandboxPolicy(), verify=verify, **kwargs)


def _precommit_client(rt, *, verify, **kwargs):
    app = FastAPI()
    app.include_router(_precommit_router(rt, verify=verify, **kwargs))
    return TestClient(app)


def _precommit_body(rt, hotkey=VALIDATOR_SS58, signature="00"):
    precommit = episode_precommit(rt.contract, hotkey=hotkey)
    return {"miner_hotkey": hotkey, "at": int(time.time()), "precommit": precommit.to_dict(),
            "signature": signature}


def _signed_only(calls):
    def verify(hotkey, precommit, *, signature, **kwargs):
        calls.append(hotkey)
        return signature == "good"
    return verify


class _Body:
    """A request whose body arrives when ``gate`` is set (at once without one)."""

    headers = {}

    def __init__(self, raw, gate=None):
        self.raw, self.gate = raw, gate

    async def stream(self):
        if self.gate is not None:
            await self.gate.wait()
        yield self.raw


def _endpoint(router):
    return next(route.endpoint for route in router.routes)


def test_the_precommit_route_rate_limits_each_hotkey_once_it_signed(tmp_path):
    """I2: a signed request over the hotkey's minute is refused before its signature is verified."""
    rt = episode_runtime(tmp_path)
    calls = []
    client = _precommit_client(rt, verify=_signed_only(calls), max_per_minute=2)
    answers = [client.post(f"{RL_PREFIX}/episodes/precommit", json=_precommit_body(rt, signature="good"))
               for _ in range(3)]
    assert [a.status_code for a in answers] == [200, 200, 429]
    assert answers[2].json()["reason"] == "precommit_rate" and int(answers[2].headers["Retry-After"]) >= 1
    assert len(calls) == 2                                                # the third never reached the verify


def test_unsigned_requests_claiming_a_hotkey_do_not_spend_its_rate(tmp_path):
    """I2: 30 unsigned requests naming H, then H's signed one goes through."""
    rt = episode_runtime(tmp_path)
    client = _precommit_client(rt, verify=_signed_only([]))
    junk = [client.post(f"{RL_PREFIX}/episodes/precommit", json=_precommit_body(rt)) for _ in range(30)]
    assert {a.status_code for a in junk} == {403}
    signed = client.post(f"{RL_PREFIX}/episodes/precommit", json=_precommit_body(rt, signature="good"))
    assert signed.status_code == 200, signed.json()


def test_an_unregistered_hotkey_is_refused_before_its_signature_and_never_tracked(tmp_path):
    """I2 + I3b: the in-memory registration check runs first; an unregistered hotkey costs no verify and
    takes no place among the tracked hotkeys (capped: the oldest is forgotten)."""
    from reliquary.validator.corpus_registration import NOT_REGISTERED

    rt = episode_runtime(tmp_path)
    calls = []

    async def registration(hotkey):
        return NOT_REGISTERED if hotkey == DAVE else None

    client = _precommit_client(rt, verify=_signed_only(calls), registration=registration, max_per_minute=1,
                               max_tracked_hotkeys=1)
    post = lambda hotkey: client.post(f"{RL_PREFIX}/episodes/precommit",  # noqa: E731
                                      json=_precommit_body(rt, hotkey=hotkey, signature="good"))
    assert post(ALICE).status_code == 200
    for _ in range(5):
        answer = post(DAVE)
        assert answer.status_code == 403 and answer.json()["reason"] == "hotkey_not_registered"
    assert calls == [ALICE]                                               # Dave was never verified
    assert post(ALICE).status_code == 429                                 # ... nor tracked: Alice still is
    assert post(CHARLIE).status_code == 200                               # one more tracked hotkey ...
    assert post(ALICE).status_code == 200                                 # ... and the oldest is forgotten


async def test_the_precommit_route_bounds_the_requests_it_verifies_at_once(tmp_path):
    from reliquary.validator.corpus_registration import NOT_REGISTERED

    rt = episode_runtime(tmp_path)
    held = asyncio.Event()

    async def registration(hotkey):                                       # inside the bound: holds its slot
        await held.wait()
        return NOT_REGISTERED

    endpoint = _endpoint(_precommit_router(rt, verify=_signed_only([]), registration=registration,
                                           max_preauth=1))
    raw = json.dumps(_precommit_body(rt)).encode()
    first = asyncio.ensure_future(endpoint(_Body(raw)))
    await asyncio.sleep(0.05)
    busy = await asyncio.wait_for(endpoint(_Body(raw)), 5)
    assert busy.status_code == 503 and json.loads(busy.body)["reason"] == "precommit_busy"
    held.set()
    assert (await asyncio.wait_for(first, 5)).status_code == 403
    assert (await asyncio.wait_for(endpoint(_Body(raw)), 5)).status_code == 403     # the slot is free again


async def test_slow_bodies_do_not_hold_the_precommit_routes_verification_slots(tmp_path):
    """I3: the bound covers decode + verify, not the body read: 8 slow bodies, then an honest one."""
    rt = episode_runtime(tmp_path)
    endpoint = _endpoint(_precommit_router(rt, verify=_signed_only([])))
    arrive = asyncio.Event()
    slow = [asyncio.ensure_future(endpoint(_Body(json.dumps(_precommit_body(rt)).encode(), arrive)))
            for _ in range(8)]
    await asyncio.sleep(0.05)
    honest = await asyncio.wait_for(
        endpoint(_Body(json.dumps(_precommit_body(rt, signature="good")).encode())), 5)
    assert honest.status_code == 200, json.loads(honest.body)
    arrive.set()
    assert {(await asyncio.wait_for(task, 5)).status_code for task in slow} <= {403, 503}


async def test_the_precommit_route_bounds_the_time_its_body_has(tmp_path):
    rt = episode_runtime(tmp_path)
    endpoint = _endpoint(_precommit_router(rt, verify=lambda *a, **k: True, body_timeout_s=0.2))

    class SlowBody:
        headers = {}

        async def stream(self):
            await asyncio.sleep(5)
            yield b"{}"

    started = time.monotonic()
    answer = await endpoint(SlowBody())
    assert answer.status_code == 408 and json.loads(answer.body)["reason"] == "body_timeout"
    assert time.monotonic() - started < 2


# --- M2: the RL open route is bounded like the precommit route ---
async def test_the_rl_open_route_bounds_the_time_its_body_has(tmp_path):
    services, _ = _services(tmp_path, open_body_timeout_s=0.2)
    endpoint = next(route.endpoint for route in services.routers[0].routes
                    if route.path == f"{RL_PREFIX}/sandbox/sessions")

    class SlowBody:
        headers = {}

        async def stream(self):
            await asyncio.sleep(5)
            yield b"{}"

    started = time.monotonic()
    answer = await endpoint(SlowBody())
    assert answer.status_code == 408 and json.loads(answer.body)["reason"] == "body_timeout"
    assert time.monotonic() - started < 2


async def test_the_open_route_bounds_the_opens_it_verifies_at_once_but_not_their_bodies():
    from reliquary.sandbox.routes import build_sandbox_sessions_router
    from reliquary.validator.corpus_registration import NOT_REGISTERED

    held = asyncio.Event()

    async def registration(hotkey):
        await held.wait()
        return NOT_REGISTERED

    router = build_sandbox_sessions_router(
        SimpleNamespace(), policy=SandboxPolicy(), validator_hotkey=VALIDATOR_SS58, prefix=RL_PREFIX,
        verify_open=lambda *a, **k: True, registration=registration, max_preauth_opens=1,
        open_body_timeout_s=5.0)
    endpoint = next(route.endpoint for route in router.routes if route.path == f"{RL_PREFIX}/sandbox/sessions")
    raw = json.dumps({"miner_hotkey": ALICE, "at": int(time.time()), "request_id": "a" * 32,
                      "engagement": {"kind": "rl_precommit", "precommit": {"seed_index": 0}},
                      "signature": "00"}).encode()
    arrive = asyncio.Event()
    slow = asyncio.ensure_future(endpoint(_Body(raw, arrive)))           # a slow body holds no slot
    await asyncio.sleep(0.05)
    first = asyncio.ensure_future(endpoint(_Body(raw)))                    # held in its slot
    await asyncio.sleep(0.05)
    busy = await asyncio.wait_for(endpoint(_Body(raw)), 5)
    assert busy.status_code == 503 and json.loads(busy.body)["reason"] == "open_busy", json.loads(busy.body)
    held.set()
    assert (await asyncio.wait_for(first, 5)).status_code == 403
    arrive.set()
    assert (await asyncio.wait_for(slow, 5)).status_code == 403


# --- item 2: precommit rows retention ---
def _settle(rt, window):
    with rt._txn():
        rt.db.execute("INSERT INTO service_settled VALUES(?,?,?,?)", (rt.contract.sha256, window, 0, "{}"))


def test_precommit_rows_of_settled_windows_are_pruned_unless_a_session_needs_them(tmp_path):
    rt = episode_runtime(tmp_path)
    rt.task_in_cooldown = None
    old = episode_precommit(rt.contract, hotkey="5Aaa")
    held = episode_precommit(rt.contract, hotkey="5Bbb")
    for precommit in (old, held):
        rt.record_episode_precommit(precommit, now=time.time() - 10_000)
    recent = episode_precommit(rt.contract, hotkey="5Ccc")
    rt.record_episode_precommit(recent)
    before = time.time() - 1_000
    assert rt.prune_episode_precommits(before=before) == 0              # window 1 not settled: kept
    _settle(rt, 1)
    assert rt.prune_episode_precommits(before=before, keep={held.sha256}) == 1
    assert rt.episode_precommit(old.sha256) is None
    assert rt.episode_precommit(held.sha256) == held and rt.episode_precommit(recent.sha256) == recent


async def test_the_services_prune_what_no_held_session_names(tmp_path):
    services, rt = _services(tmp_path, precommit_retention_s=1_000)
    rt.task_in_cooldown = None
    gone = episode_precommit(rt.contract, hotkey="5Ddd")
    held = episode_precommit(rt.contract, hotkey="5Bbb")
    for precommit in (gone, held):
        rt.record_episode_precommit(precommit, now=time.time() - 10_000)
    _settle(rt, 1)
    services.book.add(_record("s-held", sha=held.sha256, hotkey="5Bbb", state="live",
                              expires_at=int(time.time()) + 600))
    assert await services.prune_precommits() == 1
    assert rt.episode_precommit(gone.sha256) is None and rt.episode_precommit(held.sha256) == held


# --- the hand-back hook (items 16 and the Task 10 carry) ---
def _record(session_id, *, sha, hotkey="5Hot", state=SUBMITTED, seed=0, expires_at=None, issued_at=None):
    from reliquary.protocol.service_episode import rl_engagement
    from reliquary.sandbox.sessions import SessionRecord

    now = int(time.time())
    return SessionRecord(
        session_id=session_id, hotkey=hotkey, request_id=f"r-{session_id}", engagement_sha256="e" * 64,
        kind="rl_precommit", engagement=rl_engagement(1, sha, seed), env="reliquary-swe", split="train:rl",
        index=3, checkpoint="d" * 40, job_id=None, prompt_index=None, machine_id="m",
        issued_at=issued_at or now - 60, expires_at=expires_at or now + 600, token_sha256="t" * 64, state=state,
        closed_status="graded" if state == SUBMITTED else None, closed_at=now - 30 if state == SUBMITTED else None)


class Outcomes:
    def __init__(self):
        self.calls = []

    def group_settled(self, **kwargs):
        self.calls.append(("settled", kwargs))

    def group_handed_back(self, **kwargs):
        self.calls.append(("handed_back", kwargs))


async def test_a_handed_back_group_is_scheduled_without_blocking_and_marked_in_book_and_store(tmp_path):
    outcomes = Outcomes()
    store = MemorySessionStore()
    services, rt = _services(tmp_path, store=store, outcomes=outcomes)
    rt.task_in_cooldown = None
    precommit = episode_precommit(rt.contract, hotkey="5Hot")
    rt.record_episode_precommit(precommit)
    other = "b" * 64
    records = [_record(f"s-{n}", sha=precommit.sha256, seed=n) for n in range(3)]
    records.append(_record("s-other", sha=other))                         # another precommit's: untouched
    for record in records:
        await store.create(record.to_document())
        services.book.add(record)
    await services.start()
    hook = services.batcher_hook(EPISODE, 1)
    pending = SimpleNamespace(hotkey="5Hot", prompt_idx=precommit.task_index)
    await services.issuer._lock.acquire()                                 # the issuer is busy
    try:
        started = time.monotonic()
        caller = threading.Thread(target=hook, args=(pending, "validator_lost"))   # a proof-plane thread
        caller.start()
        caller.join(2)
        hook(pending, "validator_lost")                                   # and the loop itself, inside a seal
        assert not caller.is_alive() and time.monotonic() - started < 1
        assert all(services.book.get(r.session_id).closed_status == "graded" for r in records)
    finally:
        services.issuer._lock.release()
    for _ in range(200):
        await asyncio.sleep(0.01)
        if len(outcomes.calls) == 1 and not services._tasks:
            break
    await asyncio.sleep(0.05)
    assert not services._tasks
    marked = [r.session_id for r in records if services.book.get(r.session_id).closed_status == HANDED_BACK]
    assert marked == ["s-0", "s-1", "s-2"]
    assert all(services.book.get(i).state == SUBMITTED for i in marked)   # still taken: no same-window retry
    assert all(store.documents[i]["closed_status"] == HANDED_BACK and store.documents[i]["state"] == SUBMITTED
               for i in marked)
    assert store.documents["s-other"]["closed_status"] == "graded"
    assert [c for c in outcomes.calls if c[0] == "handed_back"] == [("handed_back", {
        "hotkey": "5Hot", "precommit_sha256": precommit.sha256, "session_ids": ("s-0", "s-1", "s-2"),
        "stage": "validator_lost"})]                                      # once, the second found none left
    taken = await services.issuer.claim_all(["s-0"], hotkey="5Hot", precommit_sha256=precommit.sha256)
    assert taken is not None and taken[1].reason == "precommit_submitted"       # still taken
    await services.stop()


def test_a_handed_back_session_still_counts_against_the_miners_open_rate():
    """M1: a hand-back gives no opens back (the sessions were opened; only the yield/quota excludes them)."""
    book = SessionBook(SandboxPolicy(max_opens_per_hour=2))
    now = int(time.time())
    for n in range(2):
        book.add(_record(f"s-{n}", sha="c" * 64, seed=n, issued_at=now - 60))
    assert book.open_refusal("5Hot", now).reason == "open_rate_cap"
    for n in range(2):
        book.get(f"s-{n}").closed_status = HANDED_BACK
    assert book.open_refusal("5Hot", now).reason == "open_rate_cap"


async def test_the_store_takes_a_hand_back_and_nothing_else_on_a_submitted_record():
    store = MemorySessionStore()
    record = _record("s-1", sha="c" * 64)
    await store.create(record.to_document())
    handed = dataclasses.replace(record, closed_status=HANDED_BACK, closed_at=record.closed_at + 5)
    await store.update(handed.to_document())
    assert store.documents["s-1"]["closed_status"] == HANDED_BACK
    with pytest.raises(sandbox_store.SessionStoreConflict):
        await store.update(dataclasses.replace(record, closed_status="whatever").to_document())
    with pytest.raises(sandbox_store.SessionStoreConflict):                 # nothing else may change with it
        await store.update(dataclasses.replace(handed, machine_id="other").to_document())


async def test_the_service_hands_its_episode_batchers_to_the_services(monkeypatch, tmp_path):
    """M3: the real ``_build_window_batchers`` installs the hand-back hook on episode-env batchers only."""
    from tests.unit.service_v2_fixtures import CODE, MATH
    from tests.unit.test_service_window_build import _open
    from tests.unit.test_service_window_build import _service as window_service

    svc = window_service(monkeypatch, tmp_path)
    hooks = []

    def batcher_hook(environment, window):
        hooks.append((environment, window))
        return ("hook", environment, window)

    svc._episode_services = SimpleNamespace(environments=(MATH,), batcher_hook=batcher_hook)
    batchers = await _open(svc, activate=False)
    assert batchers[MATH].episode_proof_inconclusive == ("hook", MATH, 1)
    assert batchers[CODE].episode_proof_inconclusive is None
    assert hooks == [(MATH, 1)]


async def test_the_rl_registration_reads_the_servers_gate():
    from reliquary.validator.corpus_registration import NOT_REGISTERED
    from reliquary.validator.rl_sandbox_wiring import rl_registration

    reasons = {"a": None, "b": RejectReason.HOTKEY_NOT_REGISTERED, "c": RejectReason.REGISTRATION_UNAVAILABLE}

    async def gate(hotkey):
        return reasons[hotkey]

    registration = rl_registration(SimpleNamespace(_registration_reject_reason=gate))
    assert [await registration(h) for h in "abc"] == [None, NOT_REGISTERED, "unavailable"]


def test_the_mounted_precommit_route_asks_the_servers_registration_gate(tmp_path, monkeypatch):
    monkeypatch.setattr("reliquary.protocol.profiles.toploc_proof", lambda profile: TOPLOC_DEPLOYED_DEFAULTS)
    rt = episode_runtime(tmp_path / "rt")
    service = _service(tmp_path, monkeypatch, runtime=rt)
    asked = []

    async def gate(hotkey):
        asked.append(hotkey)
        return RejectReason.HOTKEY_NOT_REGISTERED

    service.server._registration_reject_reason = gate

    async def start():
        await service._start_episode_services()
        for task in service._episode_tasks:
            task.cancel()

    asyncio.run(start())
    answer = TestClient(service.server.app).post(f"{RL_PREFIX}/episodes/precommit", json=_precommit_body(rt))
    assert answer.status_code == 403 and answer.json()["reason"] == "hotkey_not_registered"
    assert asked == [VALIDATOR_SS58]


def test_the_precommit_retention_covers_the_longest_episode_cooldown():
    from reliquary.constants import FILL_CLOSED_MAX_SECONDS
    from reliquary.validator.rl_sandbox_wiring import MIN_PRECOMMIT_RETENTION_S

    service = ValidationService.__new__(ValidationService)
    horizons = {"a": 3, "b": 10_000}
    service._service_runtime = SimpleNamespace(
        contract=SimpleNamespace(episode_environments=("a", "b")),
        schedule=SimpleNamespace(cooldown_windows=lambda name: horizons[name]))
    assert service._episode_precommit_retention_s() == 10_000 * FILL_CLOSED_MAX_SECONDS
    horizons["b"] = 1
    assert service._episode_precommit_retention_s() == MIN_PRECOMMIT_RETENTION_S


def test_a_v2_order_without_episode_envs_never_loads_the_sandbox_package(tmp_path):
    """M3: a validator with no signed-episode env never imports reliquary_sandbox."""
    import subprocess
    import sys

    script = (
        "import asyncio, sys, pathlib\n"
        "import pytest\n"
        "from tests.unit.test_service_window_build import _open, _service\n"
        "assert 'reliquary_sandbox' not in sys.modules\n"
        "mp = pytest.MonkeyPatch()\n"
        "svc = _service(mp, pathlib.Path(sys.argv[1]))\n"
        "assert not svc._service_runtime.contract.episode_environments\n"
        "async def main():\n"
        "    await svc._start_episode_services()\n"
        "    await _open(svc)\n"
        "asyncio.run(main())\n"
        "assert getattr(svc, '_episode_services', None) is None\n"
        "print('LOADED' if 'reliquary_sandbox' in sys.modules else 'CLEAN')\n")
    root = Path(__file__).resolve().parents[2]
    done = subprocess.run([sys.executable, "-c", script, str(tmp_path)], cwd=root, capture_output=True,
                          text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == "CLEAN"


def test_the_nested_eos_of_a_remote_config_joins_the_boot_check_but_not_the_legacy_stop_set():
    """I5 (ruling): the legacy proxies stay as before (flat config, nested eos unseen by the batcher's
    resolver); only the episode boot check reads the nested ``text_config`` eos."""
    from reliquary.shared.modeling import resolve_eos_token_ids
    from reliquary.validator.remote_proof import RemoteProofPool

    pool = RemoteProofPool.__new__(RemoteProofPool)
    pool.pipeline_depth = 1
    pool.health = SimpleNamespace(slots=[SimpleNamespace(device_id="s0")],
                                  config={"eos_token_id": TERM, "text_config": {"eos_token_id": 9}},
                                  generation_config={"eos_token_id": [TERM]})
    proxies = pool.proxies()
    tok = SimpleNamespace(eos_token_id=EOT)
    assert resolve_eos_token_ids(proxies["s0"], tok) == {TERM, EOT}               # legacy: as at 703bee9a
    assert pool.health.config["text_config"] == {"eos_token_id": 9}                # the health is untouched

    service = ValidationService.__new__(ValidationService)
    service.tokenizer = tok
    service._proof_models = proxies
    service._proof_worker_pool = pool
    service.verify_model = None
    proof, remote = service._episode_stop_sets()
    assert proof == {TERM, EOT, 9} and remote == {TERM, EOT, 9}                    # the boot check sees it


# --- item 17: nothing published carries a transcript ---
async def test_the_window_archive_and_journal_carry_no_signed_episode_transcript(monkeypatch, tmp_path):
    from reliquary.protocol.submission import SIGNED_EPISODE_SCHEMA
    from tests.unit import test_legacy_archive_golden as golden

    secret = "SECRET-TRANSCRIPT-" + "x" * 64
    original = golden._valid_submission

    def signed_submission(**kwargs):
        group = original(**kwargs)
        for rollout in group.rollouts:
            rollout.commit["rollout"]["episode"] = {
                "schema_version": SIGNED_EPISODE_SCHEMA, "precommit_sha256": "a" * 64, "seed_index": 0,
                "assistant_spans": [[2, len(rollout.commit["tokens"])]], "stop": "agent_completed",
                "transcript": {"token": secret}}
            object.__setattr__(rollout, "_validated_assistant_spans", [[2, len(rollout.commit["tokens"])]])
        return dataclasses.replace(group, completion_texts=[""] * len(group.rollouts))

    monkeypatch.setattr(golden, "_valid_submission", signed_submission)
    # The training payload's own strip is Task 11's (test_episode_training); this legacy-profile harness
    # cannot encode an episode payload, so the encoder is stubbed and only told what it would encode.
    encoded = []
    monkeypatch.setattr("reliquary.validator.fill_closed_batch_assembler.encode_training_payload",
                        lambda batches, **kw: encoded.append(batches) or b"payload")
    archive = await golden._legacy_archive(monkeypatch, tmp_path)
    assert len(archive["batch"]) == 1                                     # the signed group was archived
    assert secret not in json.dumps(archive)
    written = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert written and not [p for p in written if secret.encode() in p.read_bytes()]


def test_services_are_a_plain_dataclass_with_their_background_jobs(tmp_path):
    services, _ = _services(tmp_path)
    assert isinstance(services, RlEpisodeServices)
    jobs = services.background()
    try:
        assert len(jobs) == 3
    finally:
        for job in jobs:
            job.close()
