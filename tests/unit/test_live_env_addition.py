"""The validator adds an env to the running run: built when the request is first seen, checked and staged at
the window boundary, served from the next window, rebuilt at boot."""
import asyncio
import errno
import hashlib
import json
import os
import types
from types import SimpleNamespace

import pytest

attest = pytest.importorskip("reliquary_sandbox.attest")

from reliquary.environment import bridged_episode, registry  # noqa: E402
from reliquary.protocol.service_schedule import next_schedule  # noqa: E402
from reliquary.services.schedule import (  # noqa: E402
    ScheduleNotYet, ScheduleRefused, ScheduleRequestStore, build_request,
)
from reliquary.validator import catalogue_activation, service  # noqa: E402
from reliquary.validator.catalogue_activation import ADDED_PACKAGES_FILE  # noqa: E402
from reliquary.validator.proof_scheduler import GlobalProofScheduler  # noqa: E402
from reliquary.validator.service import ValidationService  # noqa: E402
from tests.unit.episode_v2_fixtures import (  # noqa: E402
    ADDED, ADDED_IDENTITY, ADDED_PACKAGE, EPISODE, added_block, episode_runtime,
)
from tests.unit.service_v2_fixtures import MATH  # noqa: E402
from tests.unit.test_catalogue_activation import BOOT, CURRENT, IMAGE_A, IMAGE_B, Source, attestation  # noqa: E402

BLOCK = added_block()
SHARES = {MATH: 4000, EPISODE: 3000, ADDED: 3000}
METHODS = ("_added_digest", "_catalogue_check", "_prestage_requested_envs", "_active_bridged_sources", "_stage_added_envs",
           "_install_env", "_install_added_envs", "_commit_staged_envs", "_all_env_mix", "_activation_validity_s",
           "_request_settled", "_service_installed_version", "_service_environment_problem",
           "_service_activation_version", "_require_service_environments", "_open_service_window")


@pytest.fixture(autouse=True)
def fresh_catalogue(monkeypatch):
    monkeypatch.setattr(registry, "_CATALOGUE_SPECS", {})
    monkeypatch.setattr(bridged_episode, "_LOADED_IDENTITIES", {})
    # the distributions installed now: the boot's plus the added one, installed alone
    monkeypatch.setattr(catalogue_activation, "installed_distributions", lambda *a: dict(CURRENT))


def built(config=None, **source):
    from reliquary.environment.signed_episode import SignedEpisodeEnvironment

    return SignedEpisodeEnvironment(ADDED, Source(**source))


def allow(folder):
    (folder / ADDED_PACKAGES_FILE).write_text(ADDED_PACKAGE + "\n")
    os.chmod(folder / ADDED_PACKAGES_FILE, 0o600)


def serving_fleet(*, complete=True, serving=("m",)):
    return SimpleNamespace(serving=lambda *a, **k: list(serving), reported_packages=lambda e: [],
                           view_complete=lambda *a, **k: complete,
                           images_served=lambda *a, **k: frozenset({IMAGE_A, IMAGE_B}))


def serving_services(**fleet):
    return SimpleNamespace(fleet=serving_fleet(**fleet), environment_refusal=lambda policy: None,
                           book=SimpleNamespace(policy=SimpleNamespace(open_window_s=120)))


def validator(tmp_path, *, services=None, boot=BOOT):
    rt = episode_runtime(tmp_path)
    me = SimpleNamespace(_service_runtime=rt, _episode_services=services, _staged_envs={}, _prebuilt={}, envs={},
                         env_mix=[], env_targets={}, _service_folder=tmp_path, _activation_memo={},
                         _prestage_refused={}, _prestaging=set(), _prestage_guard=service.threading.Lock(),
                         _boot_distributions=boot, _service_schedule_store=ScheduleRequestStore(tmp_path))
    for name in METHODS:
        setattr(me, name, types.MethodType(getattr(ValidationService, name), me))
    return me, rt


def add_request(rt, **kwargs):
    return build_request(order_sha256=rt.contract.sha256, revision=rt.schedule.revision + 1,
                         add={ADDED: {**BLOCK, "attestation": "conformance.json"}}, **kwargs)


def test_an_env_cannot_be_added_to_a_run_without_episode_services(tmp_path):
    me, _ = validator(tmp_path)
    with pytest.raises(ScheduleRefused, match="launched without a signed-episode environment"):
        me._catalogue_check(ADDED, BLOCK, attestation())


def test_a_passing_activation_stages_the_built_env(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())
    me, _ = validator(tmp_path, services=serving_services())
    allow(tmp_path)
    me._catalogue_check(ADDED, BLOCK, attestation())
    assert me._staged_envs[ADDED].source.identity == ADDED_IDENTITY
    assert me._prebuilt[ADDED][1] is me._staged_envs[ADDED]          # a later boundary reuses the build


def test_a_package_the_owner_did_not_allow_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())
    me, _ = validator(tmp_path, services=serving_services())
    with pytest.raises(ScheduleRefused, match="allowlist"):
        me._catalogue_check(ADDED, BLOCK, attestation())


def test_the_env_is_built_when_the_request_is_first_seen_not_at_the_boundary(tmp_path, monkeypatch):
    builds = []
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: builds.append(config) or built())
    me, rt = validator(tmp_path, services=serving_services())
    allow(tmp_path)
    me._service_schedule_store.write(add_request(rt))
    assert me._prestage_requested_envs() == [ADDED] and len(builds) == 1
    assert me._prestage_requested_envs() == [] and len(builds) == 1          # the same block: not rebuilt
    me._catalogue_check(ADDED, BLOCK, attestation())                         # the boundary reuses it
    assert len(builds) == 1 and ADDED in me._staged_envs


def test_a_package_the_owner_did_not_allow_is_never_built(tmp_path, monkeypatch):
    builds = []
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: builds.append(config) or built())
    me, rt = validator(tmp_path, services=serving_services())               # no allowlist file
    me._service_schedule_store.write(add_request(rt))
    assert me._prestage_requested_envs() == [] and builds == []             # its code never ran here


def test_the_prestage_checks_the_installed_distributions_before_building(tmp_path, monkeypatch):
    builds = []
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: builds.append(config) or built())
    moved = {**CURRENT, "verifiers": frozenset({("0.2", "9" * 64, frozenset({"verifiers"}))})}
    monkeypatch.setattr(catalogue_activation, "installed_distributions", lambda *a: moved)
    me, rt = validator(tmp_path, services=serving_services())
    allow(tmp_path)
    me._service_schedule_store.write(add_request(rt))
    assert me._prestage_requested_envs() == [] and builds == []


def test_a_block_the_prestage_refused_is_tried_again_only_later(tmp_path, monkeypatch):
    builds = []
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: builds.append(config) or built())
    me, rt = validator(tmp_path, services=serving_services())
    me._service_schedule_store.write(add_request(rt))
    reads = []
    real = catalogue_activation.allowed_packages
    monkeypatch.setattr(catalogue_activation, "allowed_packages", lambda folder: reads.append(1) or real(folder))
    assert me._prestage_requested_envs(now=1000.0) == []                     # not allowed yet
    allow(tmp_path)
    assert me._prestage_requested_envs(now=1000.0 + service.PRESTAGE_RETRY_SECONDS - 1) == [] and len(reads) == 1
    assert me._prestage_requested_envs(now=1000.0 + service.PRESTAGE_RETRY_SECONDS + 1) == [ADDED]
    assert len(builds) == 1


def test_a_transient_build_error_in_the_prestage_is_tried_again_at_once(tmp_path, monkeypatch):
    calls = []

    def flaky(config):
        calls.append(config)
        if len(calls) == 1:
            raise OSError(errno.EIO, "dataset download failed")
        return built()

    monkeypatch.setattr(bridged_episode, "build_environment", flaky)
    me, rt = validator(tmp_path, services=serving_services())
    allow(tmp_path)
    me._service_schedule_store.write(add_request(rt))
    assert me._prestage_requested_envs(now=1.0) == []
    assert me._prestage_requested_envs(now=2.0) == [ADDED]


def test_the_prestage_leaves_a_settled_request_alone(tmp_path, monkeypatch):
    builds = []
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: builds.append(config) or built())
    me, rt = validator(tmp_path, services=serving_services())
    allow(tmp_path)
    request = add_request(rt)
    store = me._service_schedule_store
    store.write(request)
    store.report(request["request_id"], status="refused", detail="no", revision=0)
    assert me._prestage_requested_envs() == [] and builds == []


def test_with_no_add_request_the_prestage_reads_nothing_else(tmp_path, monkeypatch):
    monkeypatch.setattr(catalogue_activation, "installed_distributions", lambda *a: pytest.fail("read"))
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: pytest.fail("built"))
    me, rt = validator(tmp_path, services=serving_services())
    assert me._prestage_requested_envs() == []
    me._service_schedule_store.write(build_request(order_sha256=rt.contract.sha256,
                                                   revision=rt.schedule.revision + 1, cooldowns={MATH: 10}))
    assert me._prestage_requested_envs() == []


def test_the_boundary_waits_while_the_prestage_builds_the_same_env(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: pytest.fail("built twice"))
    me, _ = validator(tmp_path, services=serving_services())
    allow(tmp_path)
    me._prestaging.add(ADDED)
    with pytest.raises(ScheduleNotYet, match="being built"):
        me._catalogue_check(ADDED, BLOCK, attestation())
    assert me._staged_envs == {}


def test_a_renderer_outside_the_proofs_stop_set_refuses_the_addition(tmp_path):
    services = SimpleNamespace(fleet=None, environment_refusal=lambda policy: "renderer stop ids [9] are outside")
    me, _ = validator(tmp_path, services=services)
    with pytest.raises(ScheduleRefused, match="outside"):
        me._catalogue_check(ADDED, BLOCK, attestation())


def test_a_renderer_that_cannot_be_loaded_now_waits(tmp_path):
    def loading(policy):
        raise TimeoutError("tokenizer download timed out")

    services = serving_services()
    services.environment_refusal = loading
    me, _ = validator(tmp_path, services=services)
    with pytest.raises(ScheduleNotYet, match="renderer"):
        me._catalogue_check(ADDED, BLOCK, attestation())


@pytest.mark.parametrize("error", [OSError(errno.EIO, "I/O error"), TimeoutError("read timed out"),
                                   ConnectionError("network down")])
def test_a_transient_build_error_at_the_boundary_waits(tmp_path, monkeypatch, error):
    def failing(config):
        raise error

    monkeypatch.setattr(bridged_episode, "build_environment", failing)
    me, _ = validator(tmp_path, services=serving_services())
    allow(tmp_path)
    with pytest.raises(ScheduleNotYet):
        me._catalogue_check(ADDED, BLOCK, attestation())
    assert me._staged_envs == {}


def test_a_transient_error_in_a_later_check_waits(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())

    def unreadable(*a, **k):
        raise OSError(errno.EIO, "report unreadable")

    monkeypatch.setattr(catalogue_activation, "machines_problem", unreadable)
    me, _ = validator(tmp_path, services=serving_services())
    allow(tmp_path)
    with pytest.raises(ScheduleNotYet, match="cannot run now"):
        me._catalogue_check(ADDED, BLOCK, attestation())


def test_a_stale_fleet_view_waits_and_a_complete_one_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())
    me, _ = validator(tmp_path, services=serving_services(complete=False, serving=()))
    allow(tmp_path)
    with pytest.raises(ScheduleNotYet, match="not yet"):
        me._catalogue_check(ADDED, BLOCK, attestation())
    me._episode_services = serving_services(complete=True, serving=())
    with pytest.raises(ScheduleRefused, match="machine"):
        me._catalogue_check(ADDED, BLOCK, attestation())


def test_without_the_boot_snapshot_nothing_can_be_added(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: pytest.fail("built"))
    me, _ = validator(tmp_path, services=serving_services(), boot=None)
    allow(tmp_path)
    with pytest.raises(ScheduleRefused, match="restart"):
        me._catalogue_check(ADDED, BLOCK, attestation())


def test_the_activation_gets_the_order_the_boot_snapshot_and_the_token_validity(tmp_path, monkeypatch):
    seen = {}

    def spy(name, added, attestation, **kwargs):
        seen.update(kwargs)
        return None, built()

    monkeypatch.setattr(catalogue_activation, "activation_problem", spy)
    me, rt = validator(tmp_path, services=serving_services())
    me._catalogue_check(ADDED, BLOCK, attestation())
    from reliquary.sandbox.sessions import DEFAULT_PREPARE_TIMEOUT_S, token_validity_s

    assert seen["order"] is rt.order_contract and seen["boot"] is BOOT
    assert seen["validity_s"] == token_validity_s(120, BLOCK["environment"]["episode"]["budgets"]["wall_s"],
                                                  DEFAULT_PREPARE_TIMEOUT_S)
    assert seen["memo"] is me._activation_memo[ADDED]


async def test_staged_envs_are_installed_between_windows_and_served(tmp_path):
    added = []
    services = SimpleNamespace(add_environment=lambda name, env, policy: added.append((name, policy.split)))
    me, rt = validator(tmp_path, services=services)
    rt.apply_schedule(next_schedule(rt.order_contract, rt.schedule, add={ADDED: BLOCK}, active=[MATH, EPISODE, ADDED],
                                    shares=SHARES), request_id="add-1", window=1)
    me._staged_envs = {ADDED: built(), "not_in_the_schedule": built()}
    assert me._service_environment_problem(ADDED) is None     # staged: its version, rows and group count
    assert (ADDED, me._all_env_mix()[-1][1]) in me._all_env_mix()
    assert await me._commit_staged_envs() == [ADDED]
    assert ADDED in me.envs and (ADDED, me.env_targets[ADDED]) in me.env_mix and me._staged_envs == {}
    assert added == [(ADDED, "tmax")]
    assert registry.find_environment_spec(ADDED).environment_manifest_sha256 == BLOCK["environment"]["version"]


async def test_an_env_the_services_refuse_is_never_installed_and_stays_staged(tmp_path):
    def refuse(name, env, policy):
        raise ValueError("episode env name 'x' names two envs")

    me, rt = validator(tmp_path, services=SimpleNamespace(add_environment=refuse))
    rt.apply_schedule(next_schedule(rt.order_contract, rt.schedule, add={ADDED: BLOCK}, active=[MATH, EPISODE, ADDED],
                                    shares=SHARES), request_id="add-1", window=1)
    me._staged_envs = {ADDED: built()}
    with pytest.raises(ValueError, match="two envs"):
        await me._commit_staged_envs()
    assert ADDED not in me.envs and ADDED in me._staged_envs and registry.find_environment_spec(ADDED) is None
    # the next boundary: the env is still checked as staged, and served once the services take it
    me._stage_added_envs(rt.schedule)
    assert me._service_environment_problem(ADDED) is None
    me._episode_services = SimpleNamespace(add_environment=lambda name, env, policy: None)
    assert await me._commit_staged_envs() == [ADDED]


async def test_an_installed_env_gets_its_cooldown_maps_and_its_proof_plans(tmp_path):
    from reliquary.validator.batcher import audit_scheduler_environment
    from reliquary.validator.cooldown import ContentCooldownMap, CooldownMap

    me, rt = validator(tmp_path)
    rt.apply_schedule(next_schedule(rt.order_contract, rt.schedule, add={ADDED: BLOCK}), request_id="add-1", window=1)
    me._cooldown_per_env = {MATH: CooldownMap(cooldown_windows=3)}
    me._content_cooldown_per_env = {MATH: ContentCooldownMap(cooldown_windows=3)}
    me._content_cooldown_health = {"counts_by_environment": {MATH: 0}}
    scheduler = GlobalProofScheduler(devices=("cpu",), environments=(MATH,), proof_callable=lambda *a: None)
    try:
        me.proof_scheduler = scheduler
        me._staged_envs = {ADDED: built()}
        assert await me._commit_staged_envs() == [ADDED]
        assert me._cooldown_per_env[ADDED].cooldown_windows == BLOCK["environment"]["cooldown_windows"]
        assert me._content_cooldown_per_env[ADDED].cooldown_windows == BLOCK["environment"]["cooldown_windows"]
        assert me._content_cooldown_health["counts_by_environment"][ADDED] == 0
        assert {ADDED, audit_scheduler_environment(ADDED)} <= scheduler._environment_set
        assert scheduler._active_plan_by_environment[ADDED] is None
    finally:
        scheduler.close()


def test_the_proof_scheduler_takes_new_envs_once():
    scheduler = GlobalProofScheduler(devices=("cpu",), environments=(MATH,), proof_callable=lambda *a: None)
    try:
        assert scheduler.add_environments((ADDED, MATH)) == (ADDED,)
        assert scheduler.add_environments((ADDED,)) == ()
        assert scheduler._environments == (MATH, ADDED)
    finally:
        scheduler.close()


def test_a_restart_rebuilds_the_added_envs_before_checking_the_schedule(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())
    me, rt = validator(tmp_path)
    schedule = next_schedule(rt.order_contract, rt.schedule, add={ADDED: BLOCK}, active=[MATH, EPISODE, ADDED],
                             shares=SHARES)
    me._install_added_envs(schedule)
    assert ADDED in me.envs and registry.find_environment_spec(ADDED) is not None


def test_a_restart_installs_an_added_env_left_inactive(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())
    me, rt = validator(tmp_path)
    me._install_added_envs(next_schedule(rt.order_contract, rt.schedule, add={ADDED: BLOCK}))
    assert ADDED in me.envs and me._staged_envs == {}


def test_with_nothing_added_the_boot_builds_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: pytest.fail("built"))
    me, rt = validator(tmp_path)
    me._install_added_envs(rt.schedule)
    assert me.envs == {} and me.env_mix == []


def test_an_added_env_this_validator_cannot_build_stops_the_window(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built(identity="x==1+g0.vfb2"))
    me, rt = validator(tmp_path)
    schedule = next_schedule(rt.order_contract, rt.schedule, add={ADDED: BLOCK})
    with pytest.raises(ValueError, match="installed"):
        me._stage_added_envs(schedule)


# ---------------------------------------------------------------- through the runtime (the boundary's path)

ATTESTATION = json.dumps(attestation(), sort_keys=True).encode()
ATTESTED = added_block(catalogue={"conformance_sha256": hashlib.sha256(ATTESTATION).hexdigest()})


def attested_folder(tmp_path):
    os.chmod(tmp_path, 0o700)
    (tmp_path / "conformance.json").write_bytes(ATTESTATION)
    os.chmod(tmp_path / "conformance.json", 0o600)
    allow(tmp_path)


def write_add(me, rt, **kwargs):
    request = build_request(order_sha256=rt.contract.sha256, revision=rt.schedule.revision + 1,
                            add={ADDED: {**ATTESTED, "attestation": "conformance.json"}}, **kwargs)
    me._service_schedule_store.write(request)
    return request


def boundary(me, rt, window, **kwargs):
    """``_service_window_plan``'s schedule part, as the validator runs it."""
    schedule = rt.apply_pending_schedule_request(me._service_schedule_store, window=window,
                                                 installed_version=me._service_activation_version,
                                                 catalogue_check=me._catalogue_check, **kwargs)
    me._stage_added_envs(schedule)
    if ADDED in schedule.active_environments():          # R7 as ``_require_service_environments`` runs it
        assert me._service_environment_problem(ADDED) is None
    return schedule


async def test_an_inactive_add_then_set_active_through_the_real_activation_version(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())
    me, rt = validator(tmp_path, services=serving_services())
    me._episode_services.add_environment = lambda name, env, policy: None
    attested_folder(tmp_path)
    write_add(me, rt)
    schedule = boundary(me, rt, 2)
    assert ADDED in schedule.added_environments() and ADDED not in schedule.active_environments()
    assert await me._commit_staged_envs() == [ADDED] and ADDED in me.envs
    me._service_schedule_store.write(build_request(order_sha256=rt.contract.sha256, revision=rt.schedule.revision + 1,
                                                   active=[MATH, EPISODE, ADDED], shares=SHARES))
    schedule = boundary(me, rt, 3)
    assert ADDED in schedule.active_environments(), me._service_schedule_store.status()
    assert (ADDED, me.env_targets[ADDED]) in me._all_env_mix()


async def test_set_active_after_a_restart_through_the_real_activation_version(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())
    me, rt = validator(tmp_path, services=serving_services())
    rt.apply_schedule(next_schedule(rt.order_contract, rt.schedule, add={ADDED: ATTESTED}), request_id="add-1",
                      window=1)
    # a restart: nothing installed in this process yet, then the boot installs what the schedule added
    me._service_schedule_store.write(build_request(order_sha256=rt.contract.sha256, revision=rt.schedule.revision + 1,
                                                   active=[MATH, EPISODE, ADDED], shares=SHARES))
    before = me._service_activation_version
    with pytest.raises(ScheduleRefused, match="not loaded"):
        before(ADDED)
    me._install_added_envs(rt.schedule)
    schedule = boundary(me, rt, 2)
    assert ADDED in schedule.active_environments(), me._service_schedule_store.status()


async def test_an_add_whose_checks_cannot_be_settled_waits_and_then_lands(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())
    services = serving_services(complete=False, serving=())
    services.add_environment = lambda name, env, policy: None
    me, rt = validator(tmp_path, services=services)
    attested_folder(tmp_path)
    request = write_add(me, rt, active=[MATH, EPISODE, ADDED], shares=SHARES)
    schedule = boundary(me, rt, 2)
    assert ADDED not in schedule.added_environments() and me._staged_envs == {}
    status = me._service_schedule_store.status()
    assert status["request_id"] == request["request_id"] and status["status"] == "waiting"
    assert rt.db.execute("SELECT * FROM service_schedule_requests").fetchall() == []
    services.fleet = serving_fleet()
    schedule = boundary(me, rt, 3)
    assert ADDED in schedule.active_environments() and me._service_schedule_store.status()["status"] == "applied"
    assert await me._commit_staged_envs() == [ADDED]


async def test_a_transient_build_error_through_the_runtime_waits_not_refused(tmp_path, monkeypatch):
    def failing(config):
        raise OSError(errno.ENETUNREACH, "network unreachable")

    monkeypatch.setattr(bridged_episode, "build_environment", failing)
    me, rt = validator(tmp_path, services=serving_services())
    attested_folder(tmp_path)
    write_add(me, rt)
    boundary(me, rt, 2)
    assert me._service_schedule_store.status()["status"] == "waiting"
    assert rt.db.execute("SELECT * FROM service_schedule_requests").fetchall() == []


def test_a_wait_rewrites_the_status_only_when_its_reason_changes(tmp_path, monkeypatch):
    me, rt = validator(tmp_path, services=serving_services())
    attested_folder(tmp_path)
    write_add(me, rt)
    store = me._service_schedule_store
    reasons = iter(["stale", "stale", "stale", "other"])

    def not_yet(name, added, attestation):
        raise ScheduleNotYet(next(reasons))

    writes = []
    real = store.report
    monkeypatch.setattr(store, "report", lambda *a, **k: writes.append(k.get("detail")) or real(*a, **k))
    for window in (2, 3, 4):
        rt.apply_pending_schedule_request(store, window=window, catalogue_check=not_yet)
    assert len(writes) == 1 and "stale" in writes[0]
    checks = rt.db.execute("SELECT checks FROM service_schedule_waits").fetchone()[0]
    assert checks == 3                                                    # still counted in the database
    rt.apply_pending_schedule_request(store, window=5, catalogue_check=not_yet)
    assert len(writes) == 2 and "other" in writes[1]


async def test_with_the_real_services_the_renderer_is_built_off_the_loop(tmp_path, monkeypatch):
    from tests.unit.episode_v2_fixtures import make_test_episode_env
    from tests.unit.test_rl_episode_wiring import _services
    from tests.unit.test_trajectory_parse import EOT, TERM, FakeRenderer

    monkeypatch.setattr(catalogue_activation, "machines_problem", lambda *a, **k: None)
    rt = episode_runtime(tmp_path / "rt")
    services, _ = _services(tmp_path, runtime=rt, proof_stop_ids={TERM, EOT})
    builds = []
    services.renderer_for = lambda policy: builds.append(policy) or FakeRenderer()
    env = make_test_episode_env()
    env.name = ADDED
    env.source.identity = ADDED_IDENTITY
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: env)
    monkeypatch.setattr(catalogue_activation, "install_problem", lambda *a: None)
    monkeypatch.setattr(catalogue_activation, "module_problem", lambda *a: None)
    me, _ = validator(tmp_path, services=services)
    me._service_runtime = rt
    attested_folder(tmp_path)
    write_add(me, rt)
    me._service_schedule_store = ScheduleRequestStore(tmp_path)
    await asyncio.to_thread(boundary, me, rt, 2)
    assert len(builds) == 1 and ADDED in me._staged_envs
    services.renderer_for = lambda policy: pytest.fail("a renderer built on the event loop")
    monkeypatch.setattr(type(env.source), "manifest_sha256", lambda self: BLOCK["environment"]["version"],
                        raising=False)
    assert await me._commit_staged_envs() == [ADDED]
    assert ADDED in services.views and ADDED in me.envs
    if services.ledger is not None:
        services.ledger.close()


# ---------------------------------------------------------------- window open: only its envs count

def test_each_window_open_tells_the_services_its_envs(tmp_path):
    told = []
    me = SimpleNamespace(
        _service_runtime=SimpleNamespace(open_window=lambda *a, **k: None,
                                         announcement=lambda **k: {"pool_randomness": "r"}),
        _candidate_service_window={"window": 2, "pools": {MATH: 0.4, EPISODE: 0.3}, "env_mix": [(MATH, 8)],
                                   "picks_target": 1, "batch_slots": 8},
        _episode_services=SimpleNamespace(set_active_envs=told.append),
        _service_window_frozen=lambda runtime, window: False, _note_episode_window=lambda *a, **k: None,
        _refresh_service_active=lambda: None)
    me._open_service_window = types.MethodType(ValidationService._open_service_window, me)
    me._open_service_window(2, "r")
    assert told == [sorted([EPISODE, MATH])]


# ---------------------------------------------------------------- the validator's own boundary

def planning(me):
    from tests.unit.episode_v2_fixtures import REVISION

    manifest = SimpleNamespace(revision=REVISION, repo_id="models/test", checkpoint_n=0)
    me._checkpoint_store = SimpleNamespace(current_manifest=lambda: manifest)
    me._service_installed_receipt = None
    me._emission_cap = 1.0
    me._window_pool_for = lambda env_order, declared=None: declared
    me.env_mix = [(MATH, 8), (EPISODE, 8)]
    checked = []
    me._require_service_environments = lambda schedule: checked.append(
        me._service_environment_problem(ADDED) if ADDED in schedule.active_environments() else None)
    for name in ("_service_window_plan", "_service_window_pool"):
        setattr(me, name, types.MethodType(getattr(ValidationService, name), me))
    return checked


@pytest.mark.parametrize("active", [True, False])
def test_the_window_plan_checks_stages_and_plans_an_added_env(tmp_path, monkeypatch, active):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())
    me, rt = validator(tmp_path, services=serving_services())
    attested_folder(tmp_path)
    write_add(me, rt, **({"active": [MATH, EPISODE, ADDED], "shares": SHARES} if active else {}))
    checked = planning(me)
    plan = me._service_window_plan(2)
    assert me._service_schedule_store.status()["status"] == "applied" and ADDED in me._staged_envs
    assert (ADDED in dict(plan["env_mix"])) is active and (ADDED in plan["pools"]) is active
    assert checked == [None]


async def test_the_boundary_commits_the_staged_env_before_the_window_is_built(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())
    added = []
    services = serving_services()
    services.add_environment = lambda name, env, policy: added.append(name)
    me, rt = validator(tmp_path, services=services)
    attested_folder(tmp_path)
    write_add(me, rt, active=[MATH, EPISODE, ADDED], shares=SHARES)
    planning(me)
    me._candidate_window_n = 2
    me._set_window_preparation_stage = lambda stage: None
    me._service_window_boundary = lambda window: {"recovered": [], "attempted": False, "error": None,
                                                  "plan": me._service_window_plan(window)}
    me._prepare_service_window = types.MethodType(ValidationService._prepare_service_window, me)
    assert await me._prepare_service_window() is True
    assert added == [ADDED] and ADDED in me.envs and me._staged_envs == {}
    assert ADDED in dict(me._candidate_service_window["env_mix"])


def test_the_boot_snapshots_the_installs_first_and_installs_added_envs_before_reading_the_contract():
    """``ValidationService.__init__`` is too large to run here: its order is pinned on its source."""
    import inspect

    source = inspect.getsource(ValidationService.__init__)
    snapshot = source.index("self._boot_distributions = installed_distributions()")
    install = source.index("self._install_added_envs(self._service_runtime.schedule)")
    assert snapshot < source.index("self._load_launch_environments(")       # before any env is built
    assert install < source.index("self._require_service_environments(self._service_runtime.schedule)")
    for reader in ("scheduler_environments = tuple(self.envs)", "self._cooldown_per_env: dict",
                   "self._service_runtime.contract.environments", "ValidatorServer("):
        assert install < source.index(reader), reader


def test_the_window_plan_stages_an_env_the_schedule_added_without_this_process(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())
    me, rt = validator(tmp_path, services=serving_services())
    rt.apply_schedule(next_schedule(rt.order_contract, rt.schedule, add={ADDED: BLOCK}, active=[MATH, EPISODE, ADDED],
                                    shares=SHARES), request_id="add-1", window=1)
    checked = planning(me)
    plan = me._service_window_plan(2)
    assert ADDED in me._staged_envs and ADDED in dict(plan["env_mix"]) and checked == [None]


def test_unreadable_installs_in_the_prestage_are_tried_again_at_once(tmp_path, monkeypatch):
    builds = []
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: builds.append(config) or built())
    reads = []

    def flaky(*a):
        reads.append(1)
        if len(reads) == 1:
            raise OSError(errno.EIO, "metadata unreadable")
        return dict(CURRENT)

    monkeypatch.setattr(catalogue_activation, "installed_distributions", flaky)
    me, rt = validator(tmp_path, services=serving_services())
    allow(tmp_path)
    me._service_schedule_store.write(add_request(rt))
    assert me._prestage_requested_envs(now=1.0) == [] and builds == []
    assert me._prestage_requested_envs(now=2.0) == [ADDED]


def test_an_env_whose_name_another_spec_holds_is_not_installed(tmp_path):
    from reliquary.environment.bridged_episode import catalogue_spec

    me, _ = validator(tmp_path)
    registry.register_catalogue_spec(catalogue_spec(ADDED, "0" * 64))
    with pytest.raises(ValueError, match="another environment spec"):
        me._install_env(ADDED, built())
    assert ADDED not in me.envs and me.env_mix == []
