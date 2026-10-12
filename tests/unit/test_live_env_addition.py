"""The validator adds an env to the running run: built in the background when the request is first seen, checked
in full and staged at the window boundary, served from the next window, rebuilt at boot. One added env, broken,
refused or slow, never stops a window from opening nor the validator from booting."""
import asyncio
import errno
import hashlib
import inspect
import json
import logging
import os
import threading
import time
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
from reliquary.validator.catalogue_activation import ADDED_PACKAGES_FILE, transient_error  # noqa: E402
from reliquary.validator.proof_scheduler import GlobalProofScheduler  # noqa: E402
from reliquary.validator.service import ValidationService  # noqa: E402
from tests.unit.episode_v2_fixtures import (  # noqa: E402
    ADDED, ADDED_IDENTITY, ADDED_PACKAGE, EPISODE, added_block, episode_runtime,
)
from tests.unit.service_v2_fixtures import MATH  # noqa: E402
from tests.unit.test_catalogue_activation import BOOT, CURRENT, IMAGE_A, IMAGE_B, Source, attestation  # noqa: E402

BLOCK = added_block()
SHARES = {MATH: 4000, EPISODE: 3000, ADDED: 3000}
METHODS = ("_added_digest", "_added_policy", "_catalogue_check", "_prestage_requested_envs", "_active_bridged_sources",
           "_stage_added_envs", "_install_env", "_uninstall_env", "_install_added_envs", "_commit_staged_envs",
           "_all_env_mix", "_activation_validity_s", "_request_settled", "_added_spec_problem", "_added_env_problem",
           "_prebuilt_for", "_build_failure", "_start_build", "_build_job", "_join_builds", "_added_unavailable",
           "_added_served", "_added_failed_attempt", "_added_now", "_supersede_pending_request", "_retry_added_builds", "_evict_unwanted_builds", "_deactivate_added_envs",
           "_replan_without", "_service_installed_version", "_service_environment_problem",
           "_service_activation_version", "_require_service_environments", "_open_service_window")
WAIT = 10.0


@pytest.fixture(autouse=True)
def fresh_catalogue(monkeypatch):
    monkeypatch.setattr(registry, "_CATALOGUE_SPECS", {})
    monkeypatch.setattr(bridged_episode, "_LOADED_IDENTITIES", {})
    # the distributions installed now: the boot's plus the added one, installed alone
    monkeypatch.setattr(catalogue_activation, "installed_distributions", lambda *a: dict(CURRENT))


def built(config=None, **source):
    from reliquary.environment.signed_episode import SignedEpisodeEnvironment

    return SignedEpisodeEnvironment(ADDED, Source(**source))


def counting(builds, **source):
    def build(config):
        builds.append(threading.current_thread().name)
        return built(**source)
    return build


def allow(folder):
    (folder / ADDED_PACKAGES_FILE).write_text(ADDED_PACKAGE + "\n")
    os.chmod(folder / ADDED_PACKAGES_FILE, 0o600)


def serving_fleet(*, complete=True, serving=("m",)):
    return SimpleNamespace(serving=lambda *a, **k: list(serving), reported_packages=lambda e: [],
                           view_complete=lambda *a, **k: complete,
                           images_served=lambda *a, **k: frozenset({IMAGE_A, IMAGE_B}))


def serving_services(**fleet):
    return SimpleNamespace(fleet=serving_fleet(**fleet), environment_refusal=lambda policy: None,
                           add_environment=lambda name, env, policy: None,
                           book=SimpleNamespace(policy=SimpleNamespace(open_window_s=120)))


def validator(tmp_path, *, services=None, boot=BOOT):
    rt = episode_runtime(tmp_path)
    me = SimpleNamespace(_service_runtime=rt, _episode_services=services, _staged_envs={}, _prebuilt={}, envs={},
                         env_mix=[], env_targets={}, _service_folder=tmp_path, _activation_memo={},
                         _prestage_refused={}, _builds={}, _build_failures={}, _added_retry={}, _added_alerts={},
                         _prestage_guard=threading.RLock(), _windows_since_cooldown_snapshot=0, _window_n=1,
                         _boot_distributions=boot, _service_schedule_store=ScheduleRequestStore(tmp_path))
    for name in METHODS:
        setattr(me, name, types.MethodType(getattr(ValidationService, name), me))
    return me, rt


def add_request(rt, **kwargs):
    return build_request(order_sha256=rt.contract.sha256, revision=rt.schedule.revision + 1,
                         add={ADDED: {**BLOCK, "attestation": "conformance.json"}}, **kwargs)


def prestaged(me):
    started = me._prestage_requested_envs()
    assert me._join_builds(WAIT)
    return started


def add_to_schedule(rt, *, active, block=BLOCK):
    extra = {"active": [MATH, EPISODE, ADDED], "shares": SHARES} if active else {}
    rt.apply_schedule(next_schedule(rt.order_contract, rt.schedule, add={ADDED: block}, **extra),
                      request_id="add-1", window=1)
    return rt.schedule


# ---------------------------------------------------------------- the activation check

def test_an_env_cannot_be_added_to_a_run_without_episode_services(tmp_path):
    me, _ = validator(tmp_path)
    with pytest.raises(ScheduleRefused, match="launched without a signed-episode environment"):
        me._catalogue_check(ADDED, BLOCK, attestation())


def test_a_passing_activation_stages_the_env_built_ahead(tmp_path, monkeypatch):
    builds = []
    monkeypatch.setattr(bridged_episode, "build_environment", counting(builds))
    me, rt = validator(tmp_path, services=serving_services())
    allow(tmp_path)
    me._service_schedule_store.write(add_request(rt))
    assert prestaged(me) == [ADDED]
    me._catalogue_check(ADDED, BLOCK, attestation())
    assert me._staged_envs[ADDED].source.identity == ADDED_IDENTITY
    assert len(builds) == 1 and builds[0].startswith("added-env-build")     # in the background builder


def test_a_package_the_owner_did_not_allow_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: pytest.fail("built"))
    me, _ = validator(tmp_path, services=serving_services())
    with pytest.raises(ScheduleRefused, match="allowlist"):
        me._catalogue_check(ADDED, BLOCK, attestation())
    assert me._builds == {}


def test_the_env_is_built_when_the_request_is_first_seen_not_at_the_boundary(tmp_path, monkeypatch):
    builds = []
    monkeypatch.setattr(bridged_episode, "build_environment", counting(builds))
    me, rt = validator(tmp_path, services=serving_services())
    allow(tmp_path)
    me._service_schedule_store.write(add_request(rt))
    assert prestaged(me) == [ADDED] and len(builds) == 1
    assert prestaged(me) == [] and len(builds) == 1          # the same block: not rebuilt
    me._catalogue_check(ADDED, BLOCK, attestation())          # the boundary reuses it
    assert len(builds) == 1 and ADDED in me._staged_envs


def test_a_package_the_owner_did_not_allow_is_never_built(tmp_path, monkeypatch):
    builds = []
    monkeypatch.setattr(bridged_episode, "build_environment", counting(builds))
    me, rt = validator(tmp_path, services=serving_services())               # no allowlist file
    me._service_schedule_store.write(add_request(rt))
    assert prestaged(me) == [] and builds == []             # its code never ran here


def test_the_prestage_checks_the_installed_distributions_before_building(tmp_path, monkeypatch):
    builds = []
    monkeypatch.setattr(bridged_episode, "build_environment", counting(builds))
    moved = {**CURRENT, "verifiers": frozenset({("0.2", "9" * 64, frozenset({"verifiers"}))})}
    monkeypatch.setattr(catalogue_activation, "installed_distributions", lambda *a: moved)
    me, rt = validator(tmp_path, services=serving_services())
    allow(tmp_path)
    me._service_schedule_store.write(add_request(rt))
    assert prestaged(me) == [] and builds == []


def test_a_block_the_prestage_refused_is_tried_again_only_later(tmp_path, monkeypatch):
    builds = []
    monkeypatch.setattr(bridged_episode, "build_environment", counting(builds))
    me, rt = validator(tmp_path, services=serving_services())
    me._service_schedule_store.write(add_request(rt))
    reads = []
    real = catalogue_activation.allowed_packages
    monkeypatch.setattr(catalogue_activation, "allowed_packages", lambda folder: reads.append(1) or real(folder))
    assert me._prestage_requested_envs(now=1000.0) == []                     # not allowed yet
    allow(tmp_path)
    assert me._prestage_requested_envs(now=1000.0 + service.PRESTAGE_RETRY_SECONDS - 1) == [] and len(reads) == 1
    assert me._prestage_requested_envs(now=1000.0 + service.PRESTAGE_RETRY_SECONDS + 1) == [ADDED]
    assert me._join_builds(WAIT) and len(builds) == 1


def test_a_transient_build_error_is_built_again_at_once(tmp_path, monkeypatch):
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
    assert prestaged(me) == [ADDED] and me._prebuilt == {}
    assert prestaged(me) == [ADDED] and ADDED in me._prebuilt


def test_unreadable_installs_in_the_prestage_are_tried_again_at_once(tmp_path, monkeypatch):
    builds = []
    monkeypatch.setattr(bridged_episode, "build_environment", counting(builds))
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
    assert me._join_builds(WAIT)


def test_the_prestage_leaves_a_settled_request_alone_and_drops_its_build(tmp_path, monkeypatch):
    builds = []
    monkeypatch.setattr(bridged_episode, "build_environment", counting(builds))
    me, rt = validator(tmp_path, services=serving_services())
    allow(tmp_path)
    request = add_request(rt)
    store = me._service_schedule_store
    store.write(request)
    assert prestaged(me) == [ADDED] and ADDED in me._prebuilt
    store.report(request["request_id"], status="refused", detail="no", revision=0)
    assert prestaged(me) == [] and len(builds) == 1
    assert me._prebuilt == {}                                   # a refused request's build is not kept


def test_a_withdrawn_request_drops_its_build(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())
    me, rt = validator(tmp_path, services=serving_services())
    allow(tmp_path)
    me._service_schedule_store.write(add_request(rt))
    prestaged(me)
    assert ADDED in me._prebuilt
    me._service_schedule_store.request_path.unlink()
    prestaged(me)
    assert me._prebuilt == {}


def test_with_no_add_request_the_prestage_reads_nothing_else(tmp_path, monkeypatch):
    monkeypatch.setattr(catalogue_activation, "installed_distributions", lambda *a: pytest.fail("read"))
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: pytest.fail("built"))
    me, rt = validator(tmp_path, services=serving_services())
    assert prestaged(me) == []
    me._service_schedule_store.write(build_request(order_sha256=rt.contract.sha256,
                                                   revision=rt.schedule.revision + 1, cooldowns={MATH: 10}))
    assert prestaged(me) == []


# ---------------------------------------------------------------- the boundary never builds

def gated(builds, release):
    def build(config):
        builds.append(threading.current_thread().name)
        assert release.wait(WAIT)
        return built()
    return build


def test_a_request_just_before_a_boundary_never_delays_the_window(tmp_path, monkeypatch):
    builds, release = [], threading.Event()
    monkeypatch.setattr(bridged_episode, "build_environment", gated(builds, release))
    me, _ = validator(tmp_path, services=serving_services())
    allow(tmp_path)
    try:
        started = time.monotonic()
        with pytest.raises(ScheduleNotYet, match="being built"):
            me._catalogue_check(ADDED, BLOCK, attestation())      # no build yet: it starts one and waits
        assert time.monotonic() - started < 5.0
        with pytest.raises(ScheduleNotYet, match="being built"):
            me._catalogue_check(ADDED, BLOCK, attestation())      # still building: no second build
        assert me._prestage_requested_envs() == []                 # the prestage shares the claim
    finally:
        release.set()
    assert me._join_builds(WAIT) and len(builds) == 1 and builds[0].startswith("added-env-build")
    me._catalogue_check(ADDED, BLOCK, attestation())
    assert ADDED in me._staged_envs


def test_a_build_running_too_long_is_reported_stalled_and_waits(tmp_path, monkeypatch):
    builds, release = [], threading.Event()
    monkeypatch.setattr(bridged_episode, "build_environment", gated(builds, release))
    monkeypatch.setattr(service, "ADDED_BUILD_TIMEOUT_S", -1.0)
    me, _ = validator(tmp_path, services=serving_services())
    allow(tmp_path)
    try:
        with pytest.raises(ScheduleNotYet):
            me._catalogue_check(ADDED, BLOCK, attestation())
        with pytest.raises(ScheduleNotYet, match="stalled"):
            me._catalogue_check(ADDED, BLOCK, attestation())
    finally:
        release.set()
    assert me._join_builds(WAIT) and len(builds) == 1


def test_the_background_build_warms_the_renderer_so_the_boundary_never_loads_one(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())
    loads = []
    services = serving_services()
    services.environment_refusal = lambda policy: loads.append(threading.current_thread().name)
    me, _ = validator(tmp_path, services=services)
    allow(tmp_path)
    with pytest.raises(ScheduleNotYet):
        me._catalogue_check(ADDED, BLOCK, attestation())
    assert me._join_builds(WAIT) and len(loads) == 1 and loads[0].startswith("added-env-build")


# ---------------------------------------------------------------- refusals and waits

def test_a_renderer_outside_the_proofs_stop_set_refuses_the_addition(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())
    services = serving_services()
    services.environment_refusal = lambda policy: "renderer stop ids [9] are outside"
    me, _ = validator(tmp_path, services=services)
    allow(tmp_path)
    with pytest.raises(ScheduleNotYet):
        me._catalogue_check(ADDED, BLOCK, attestation())
    assert me._join_builds(WAIT)
    with pytest.raises(ScheduleRefused, match="outside"):
        me._catalogue_check(ADDED, BLOCK, attestation())


def test_a_renderer_that_cannot_be_loaded_now_waits(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())

    def loading(policy):
        raise TimeoutError("tokenizer download timed out")

    services = serving_services()
    services.environment_refusal = loading
    me, _ = validator(tmp_path, services=services)
    allow(tmp_path)
    for _ in range(2):
        with pytest.raises(ScheduleNotYet):
            me._catalogue_check(ADDED, BLOCK, attestation())
        assert me._join_builds(WAIT)


class HttpError(OSError):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.response = SimpleNamespace(status_code=status)


def named(name, base=Exception):
    return type(name, (base,), {})


TRANSIENT = [TimeoutError("read timed out"), ConnectionError("reset"), ConnectionResetError("peer"),
             OSError(errno.EIO, "I/O"), OSError(errno.EMFILE, "files"), OSError(errno.EAGAIN, "again"),
             HttpError(503), HttpError(429), named("ClientPayloadError", named("ClientError"))("payload"),
             named("ServerDisconnectedError", named("ClientError"))("gone"),
             named("LocalEntryNotFoundError", FileNotFoundError)("offline"),
             RuntimeError("wrapped")]
TRANSIENT[-1].__cause__ = TimeoutError("inner")
PERMANENT = [FileNotFoundError("missing"), PermissionError("denied"), HttpError(404), HttpError(401), HttpError(403),
             named("RepositoryNotFoundError", OSError)("no repo"), named("GatedRepoError", OSError)("gated"),
             named("DatasetNotFoundError", FileNotFoundError)("no dataset"), ValueError("bad split"),
             OSError("no errno"), KeyError("column"), FileNotFoundError("missing after a retry")]
PERMANENT[-1].__context__ = TimeoutError("the first attempt timed out")      # the last error decides


@pytest.mark.parametrize("error", TRANSIENT, ids=lambda e: type(e).__name__)
def test_transient_errors_wait(error):
    assert transient_error(error) is True


@pytest.mark.parametrize("error", PERMANENT, ids=lambda e: type(e).__name__)
def test_permanent_errors_are_verdicts(error):
    assert transient_error(error) is False


@pytest.mark.parametrize("error,waits", [(OSError(errno.EIO, "I/O error"), True), (TimeoutError("timed out"), True),
                                         (ConnectionError("network down"), True), (HttpError(503), True),
                                         (FileNotFoundError("no package"), False), (HttpError(404), False),
                                         (PermissionError("denied"), False)],
                         ids=lambda v: type(v).__name__ if isinstance(v, BaseException) else str(v))
def test_a_build_error_at_the_boundary_waits_or_refuses_by_its_kind(tmp_path, monkeypatch, error, waits):
    def failing(config):
        raise error

    monkeypatch.setattr(bridged_episode, "build_environment", failing)
    me, _ = validator(tmp_path, services=serving_services())
    allow(tmp_path)
    with pytest.raises(ScheduleNotYet):
        me._catalogue_check(ADDED, BLOCK, attestation())        # building
    assert me._join_builds(WAIT)
    with pytest.raises(ScheduleNotYet if waits else ScheduleRefused):
        me._catalogue_check(ADDED, BLOCK, attestation())
    assert me._staged_envs == {}


def built_ahead(me, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())
    allow(me._service_folder)
    with pytest.raises(ScheduleNotYet):
        me._catalogue_check(ADDED, BLOCK, attestation())
    assert me._join_builds(WAIT)


@pytest.mark.parametrize("error,waits", [(OSError(errno.EIO, "report unreadable"), True),
                                         (FileNotFoundError("gone"), False)])
def test_an_error_in_a_later_check_waits_or_refuses_by_its_kind(tmp_path, monkeypatch, error, waits):
    me, _ = validator(tmp_path, services=serving_services())
    built_ahead(me, monkeypatch)

    def failing(*a, **k):
        raise error

    monkeypatch.setattr(catalogue_activation, "machines_problem", failing)
    with pytest.raises(ScheduleNotYet if waits else ScheduleRefused):
        me._catalogue_check(ADDED, BLOCK, attestation())


def test_a_stale_fleet_view_waits_and_a_complete_one_refuses(tmp_path, monkeypatch):
    me, _ = validator(tmp_path, services=serving_services(complete=False, serving=()))
    built_ahead(me, monkeypatch)
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
    me, rt = validator(tmp_path, services=serving_services())
    built_ahead(me, monkeypatch)

    def spy(name, added, attestation, **kwargs):
        seen.update(kwargs)
        return None, built()

    monkeypatch.setattr(catalogue_activation, "activation_problem", spy)
    me._catalogue_check(ADDED, BLOCK, attestation())
    from reliquary.sandbox.sessions import DEFAULT_PREPARE_TIMEOUT_S, token_validity_s

    assert seen["order"] is rt.order_contract and seen["boot"] is BOOT
    assert seen["validity_s"] == token_validity_s(120, BLOCK["environment"]["episode"]["budgets"]["wall_s"],
                                                  DEFAULT_PREPARE_TIMEOUT_S)
    assert seen["memo"] is me._activation_memo[ADDED]


@pytest.mark.parametrize("source", ["services", "spec"])
def test_what_the_commit_would_refuse_is_refused_before_the_schedule_commits(tmp_path, monkeypatch, source):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())
    me, rt = validator(tmp_path, services=serving_services())
    attested_folder(tmp_path)
    write_add(me, rt, active=[MATH, EPISODE, ADDED], shares=SHARES)
    assert prestaged(me) == [ADDED] and ADDED in me._prebuilt       # the background dry run passed then
    if source == "services":
        me._episode_services.add_check = lambda name, env, policy: "its task source cannot be served"
        match = "cannot be served"
    else:
        from reliquary.environment.bridged_episode import catalogue_spec

        registry.register_catalogue_spec(catalogue_spec(ADDED, "0" * 64))
        match = "another environment spec"
    rt.apply_pending_schedule_request(me._service_schedule_store, window=2,
                                      installed_version=me._service_activation_version,
                                      catalogue_check=me._catalogue_check)
    status = me._service_schedule_store.status()
    assert status["status"] == "refused" and match in status["detail"]
    assert ADDED not in rt.schedule.added_environments() and me._staged_envs == {}


# ---------------------------------------------------------------- commit: never raised for, an active one switched off

async def test_staged_envs_are_installed_between_windows_and_served(tmp_path):
    added = []
    services = SimpleNamespace(add_environment=lambda name, env, policy: added.append((name, policy.split)))
    me, rt = validator(tmp_path, services=services)
    add_to_schedule(rt, active=True)
    me._staged_envs = {ADDED: built(), "not_in_the_schedule": built()}
    assert me._service_environment_problem(ADDED) is None     # staged: its version, rows and group count
    assert (ADDED, me._all_env_mix()[-1][1]) in me._all_env_mix()
    assert await me._commit_staged_envs() == ([ADDED], {})
    assert ADDED in me.envs and (ADDED, me.env_targets[ADDED]) in me.env_mix and me._staged_envs == {}
    assert added == [(ADDED, "tmax")]
    assert registry.find_environment_spec(ADDED).environment_manifest_sha256 == BLOCK["environment"]["version"]
    # its cooldown goes into the snapshot taken at the next seal
    assert me._windows_since_cooldown_snapshot == service.COOLDOWN_SNAPSHOT_INTERVAL_WINDOWS


def refusing(name, env, policy):
    raise ValueError("episode env name 'x' names two envs")


async def test_an_inactive_env_the_services_refuse_after_the_commit_never_stops_the_window(tmp_path, caplog):
    me, rt = validator(tmp_path, services=SimpleNamespace(add_environment=refusing))
    add_to_schedule(rt, active=False)
    me._staged_envs = {ADDED: built()}
    with caplog.at_level(logging.ERROR):
        assert await me._commit_staged_envs(active={MATH, EPISODE}) == ([], {})
    assert ADDED not in me.envs and me._staged_envs == {} and registry.find_environment_spec(ADDED) is None
    assert "ALERT" in caplog.text and ADDED in me._added_retry
    with pytest.raises(ScheduleRefused, match="not loaded"):
        me._service_activation_version(ADDED)                 # it cannot be switched on while unloaded


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


def preparing(me):
    me._candidate_window_n = 2
    me._set_window_preparation_stage = lambda stage: None
    me._service_window_boundary = lambda window: {"recovered": [], "attempted": False, "error": None,
                                                  "plan": me._service_window_plan(window)}
    me._prepare_service_window = types.MethodType(ValidationService._prepare_service_window, me)


async def test_an_active_env_the_services_refuse_after_the_commit_is_switched_off_and_the_window_opens(
        tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())
    me, rt = validator(tmp_path, services=serving_services())
    me._episode_services.add_environment = refusing
    attested_folder(tmp_path)
    write_add(me, rt, active=[MATH, EPISODE, ADDED], shares=SHARES)
    prestaged(me)
    planning(me)
    preparing(me)
    with caplog.at_level(logging.ERROR):
        assert await me._prepare_service_window() is True
    plan = me._candidate_service_window
    assert ADDED not in dict(plan["env_mix"]) and set(plan["pools"]) == {MATH, EPISODE}
    schedule = rt.schedule
    assert ADDED in schedule.added_environments() and ADDED not in schedule.active_environments()
    assert schedule.share_bps(MATH) + schedule.share_bps(EPISODE) == 10000
    assert (schedule.share_bps(MATH), schedule.share_bps(EPISODE)) == (5714, 4286)
    assert rt.db.execute("SELECT request_id FROM service_schedules WHERE revision=?",
                         (schedule.revision,)).fetchone()[0] == f"auto-deactivate-r{schedule.revision}"
    assert "ALERT" in caplog.text and ADDED not in me.envs


def test_shares_are_renormalised_in_their_proportions():
    assert service._renormalised_shares({MATH: 4000, EPISODE: 3000}) == {MATH: 5714, EPISODE: 4286}
    assert sum(service._renormalised_shares({"a": 1, "b": 1, "c": 1}).values()) == 10000
    assert service._renormalised_shares({"a": 0, "b": 0}) == {"a": 5000, "b": 5000}


def test_no_env_left_active_is_the_only_halt(tmp_path):
    me, rt = validator(tmp_path)
    current = rt.schedule
    rt.apply_schedule(next_schedule(rt.contract, current, add={ADDED: BLOCK}, active=[ADDED], shares={ADDED: 10000}),
                      request_id="only", window=1)
    with pytest.raises(ValueError, match="no other env is active"):
        me._deactivate_added_envs({ADDED: "broken"}, 2)


async def test_an_installed_env_gets_its_cooldown_maps_and_its_proof_plans(tmp_path):
    from reliquary.validator.batcher import audit_scheduler_environment
    from reliquary.validator.cooldown import ContentCooldownMap, CooldownMap

    me, rt = validator(tmp_path)
    add_to_schedule(rt, active=False)
    me._cooldown_per_env = {MATH: CooldownMap(cooldown_windows=3)}
    me._content_cooldown_per_env = {MATH: ContentCooldownMap(cooldown_windows=3)}
    me._content_cooldown_health = {"counts_by_environment": {MATH: 0}}
    scheduler = GlobalProofScheduler(devices=("cpu",), environments=(MATH,), proof_callable=lambda *a: None)
    try:
        me.proof_scheduler = scheduler
        me._staged_envs = {ADDED: built()}
        assert await me._commit_staged_envs() == ([ADDED], {})
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


def test_an_env_whose_name_another_spec_holds_is_not_installed(tmp_path):
    from reliquary.environment.bridged_episode import catalogue_spec

    me, _ = validator(tmp_path)
    registry.register_catalogue_spec(catalogue_spec(ADDED, "0" * 64))
    with pytest.raises(ValueError, match="another environment spec"):
        me._install_env(ADDED, built())
    assert ADDED not in me.envs and me.env_mix == []


# ---------------------------------------------------------------- boot

def test_a_restart_rebuilds_the_added_envs_before_checking_the_schedule(tmp_path, monkeypatch):
    builds = []
    monkeypatch.setattr(bridged_episode, "build_environment", counting(builds))
    me, rt = validator(tmp_path)
    schedule = add_to_schedule(rt, active=True)
    assert me._install_added_envs(schedule).revision == schedule.revision
    assert ADDED in me.envs and registry.find_environment_spec(ADDED) is not None
    assert builds[0].startswith("added-env-build")


def test_a_restart_installs_an_added_env_left_inactive(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())
    me, rt = validator(tmp_path)
    me._install_added_envs(add_to_schedule(rt, active=False))
    assert ADDED in me.envs and me._staged_envs == {}


def test_with_nothing_added_the_boot_builds_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: pytest.fail("built"))
    me, rt = validator(tmp_path)
    assert me._install_added_envs(rt.schedule).revision == rt.schedule.revision
    assert me.envs == {} and me.env_mix == [] and me._builds == {}


@pytest.mark.parametrize("failure", ["raises", "identity"])
def test_a_broken_inactive_added_env_never_stops_the_boot(tmp_path, monkeypatch, caplog, failure):
    def broken(config):
        if failure == "raises":
            raise FileNotFoundError("the package was uninstalled")
        return built(identity="x==1+g0.vfb2")

    monkeypatch.setattr(bridged_episode, "build_environment", broken)
    me, rt = validator(tmp_path)
    schedule = add_to_schedule(rt, active=False)
    with caplog.at_level(logging.ERROR):
        assert me._install_added_envs(schedule).revision == schedule.revision     # no switch-off needed
    assert ADDED not in me.envs and "ALERT" in caplog.text
    with pytest.raises(ScheduleRefused, match="not loaded"):
        me._service_activation_version(ADDED)                    # `set --active` is refused while unloaded
    assert ADDED in me._added_retry


def test_a_broken_active_added_env_is_switched_off_at_boot(tmp_path, monkeypatch, caplog):
    def broken(config):
        raise ConnectionError("the hub is down")

    monkeypatch.setattr(bridged_episode, "build_environment", broken)
    me, rt = validator(tmp_path)
    schedule = add_to_schedule(rt, active=True)
    me.envs = {MATH: object(), EPISODE: object()}
    with caplog.at_level(logging.ERROR):
        now = me._install_added_envs(schedule)
    assert now.revision == schedule.revision + 1 and ADDED not in now.active_environments()
    assert set(now.active_environments()) == {MATH, EPISODE} and "switched off" in caplog.text


def test_the_boot_waits_a_bounded_time_for_a_stalled_build(tmp_path, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(bridged_episode, "build_environment", gated([], release))
    monkeypatch.setattr(service, "BOOT_ADDED_BUILD_TIMEOUT_S", 0.2)
    me, rt = validator(tmp_path)
    try:
        started = time.monotonic()
        now = me._install_added_envs(add_to_schedule(rt, active=True))
        assert time.monotonic() - started < 5.0
        assert ADDED not in now.active_environments() and ADDED not in me.envs
    finally:
        release.set()
    assert me._join_builds(WAIT) and ADDED in me._prebuilt          # served later, once switched on again


def test_an_unloaded_added_env_is_retried_with_a_backoff(tmp_path, monkeypatch):
    builds = []

    def broken(config):
        builds.append(1)
        raise FileNotFoundError("gone")

    monkeypatch.setattr(bridged_episode, "build_environment", broken)
    me, rt = validator(tmp_path)
    clock = SimpleNamespace(now=100.0)
    me._added_clock = lambda: clock.now
    schedule = add_to_schedule(rt, active=False)
    me._added_unavailable(ADDED, "not built yet")                   # no attempt yet: a retry is due now
    me._added_unavailable(ADDED, "not built yet")                   # never moved by a second call
    assert me._added_retry[ADDED] == (0, 100.0)
    assert me._retry_added_builds(schedule) == [ADDED]
    assert me._retry_added_builds(schedule) == []                    # spent: one build per scheduled retry
    assert me._join_builds(WAIT) and builds == [1]
    assert me._added_retry[ADDED] == (1, 100.0 + service.ADDED_RETRY_MIN_S)        # the failed build counts
    me._added_unavailable(ADDED, "gone")                             # a boundary: the retry is not moved
    assert me._added_retry[ADDED] == (1, 100.0 + service.ADDED_RETRY_MIN_S)
    clock.now = 100.0 + service.ADDED_RETRY_MIN_S
    assert me._retry_added_builds(schedule) == [ADDED]
    assert me._join_builds(WAIT) and builds == [1, 1]
    assert me._added_retry[ADDED] == (2, clock.now + 2 * service.ADDED_RETRY_MIN_S)  # doubled per failure
    for _ in range(20):
        me._added_failed_attempt(ADDED)
    assert me._added_retry[ADDED][1] == clock.now + service.ADDED_RETRY_MAX_S       # capped


def test_after_an_outage_an_unloaded_env_is_built_and_loaded_within_one_backoff(tmp_path, monkeypatch):
    """Network down 3 h, prestage tick every 30 s, boundary every 300 s, simulated clock over 8 h: the env is
    loaded within one backoff of the network coming back, and at most one build runs per scheduled retry."""
    outage, end = 3 * 3600.0, 8 * 3600.0
    clock = SimpleNamespace(now=0.0)
    starts = []

    def build(config):
        starts.append(clock.now)
        if clock.now < outage:
            raise ConnectionError("network down")
        return built()

    monkeypatch.setattr(bridged_episode, "build_environment", build)
    me, rt = validator(tmp_path)
    me._added_clock = lambda: clock.now
    schedule = add_to_schedule(rt, active=False)
    me._added_unavailable(ADDED, "the boot could not build it")
    loaded_at, deadlines = None, set()
    while clock.now <= end and loaded_at is None:
        due = me._added_retry.get(ADDED, (0, None))[1]
        if me._retry_added_builds(schedule):                    # the prestage tick
            deadlines.add(due)
        assert me._join_builds(WAIT)
        if int(clock.now) % 300 == 0:                           # a boundary
            me._stage_added_envs(schedule, window=int(clock.now // 300) + 2)
            assert me._join_builds(WAIT)
            for name, environment in list(me._staged_envs.items()):
                me._install_env(name, environment)
                me._staged_envs.pop(name)
                loaded_at = clock.now
        clock.now += 30.0
    assert loaded_at is not None and loaded_at <= outage + service.ADDED_RETRY_MAX_S + 300
    assert len(starts) == len(deadlines)                         # one build per scheduled retry
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert all(gap >= service.ADDED_RETRY_MIN_S for gap in gaps)  # never rebuilt every 30 s
    assert len(starts) <= 12 and ADDED in me.envs


def test_at_a_boundary_an_unloaded_inactive_env_is_left_and_an_active_one_switched_off(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: pytest.fail("built at the boundary"))
    me, rt = validator(tmp_path)
    me._start_build = lambda name, added, now=None: "started"           # the background builder, not here
    schedule = add_to_schedule(rt, active=False)
    assert me._stage_added_envs(schedule, window=2) is not None and me._staged_envs == {}
    assert rt.schedule.revision == schedule.revision
    rt.apply_schedule(next_schedule(rt.contract, rt.schedule, active=[MATH, EPISODE, ADDED], shares=SHARES),
                      request_id="on", window=2)
    now = me._stage_added_envs(rt.schedule, window=3)
    assert ADDED not in now.active_environments()


def test_the_boundary_runs_the_services_check_on_an_env_it_stages(tmp_path, monkeypatch):
    me, rt = validator(tmp_path, services=serving_services())
    add_to_schedule(rt, active=False)
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())
    me._retry_added_builds()
    assert me._join_builds(WAIT) and ADDED in me._prebuilt
    me._episode_services.add_check = lambda name, env, policy: "names two envs"
    me._stage_added_envs(rt.schedule, window=2)
    assert me._staged_envs == {} and ADDED not in me._prebuilt
    attempts, due = me._added_retry[ADDED]
    assert attempts == 1 and due < float("inf")                   # a failed attempt: retried after a backoff


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
    schedule = me._stage_added_envs(schedule, window=window)
    if ADDED in schedule.active_environments():          # R7 as ``_require_service_environments`` runs it
        assert me._service_environment_problem(ADDED) is None
    return schedule


def built_then_boundary(me, rt, window):
    boundary(me, rt, window)                              # the build starts; the request waits
    assert me._service_schedule_store.status()["status"] == "waiting"
    assert me._join_builds(WAIT)
    return boundary(me, rt, window + 1)


async def test_an_inactive_add_then_set_active_through_the_real_activation_version(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())
    me, rt = validator(tmp_path, services=serving_services())
    attested_folder(tmp_path)
    write_add(me, rt)
    schedule = built_then_boundary(me, rt, 2)
    assert ADDED in schedule.added_environments() and ADDED not in schedule.active_environments()
    assert await me._commit_staged_envs() == ([ADDED], {}) and ADDED in me.envs
    me._service_schedule_store.write(build_request(order_sha256=rt.contract.sha256, revision=rt.schedule.revision + 1,
                                                   active=[MATH, EPISODE, ADDED], shares=SHARES))
    schedule = boundary(me, rt, 4)
    assert ADDED in schedule.active_environments(), me._service_schedule_store.status()
    assert (ADDED, me.env_targets[ADDED]) in me._all_env_mix()


async def test_set_active_after_a_restart_through_the_real_activation_version(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())
    me, rt = validator(tmp_path, services=serving_services())
    add_to_schedule(rt, active=False, block=ATTESTED)
    me._service_schedule_store.write(build_request(order_sha256=rt.contract.sha256, revision=rt.schedule.revision + 1,
                                                   active=[MATH, EPISODE, ADDED], shares=SHARES))
    with pytest.raises(ScheduleRefused, match="not loaded"):
        me._service_activation_version(ADDED)
    me._install_added_envs(rt.schedule)
    schedule = boundary(me, rt, 2)
    assert ADDED in schedule.active_environments(), me._service_schedule_store.status()


async def test_an_add_whose_checks_cannot_be_settled_waits_and_then_lands(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())
    services = serving_services(complete=False, serving=())
    me, rt = validator(tmp_path, services=services)
    attested_folder(tmp_path)
    request = write_add(me, rt, active=[MATH, EPISODE, ADDED], shares=SHARES)
    schedule = built_then_boundary(me, rt, 2)
    assert ADDED not in schedule.added_environments() and me._staged_envs == {}
    status = me._service_schedule_store.status()
    assert status["request_id"] == request["request_id"] and status["status"] == "waiting"
    assert rt.db.execute("SELECT * FROM service_schedule_requests").fetchall() == []
    services.fleet = serving_fleet()
    schedule = boundary(me, rt, 4)
    assert ADDED in schedule.active_environments() and me._service_schedule_store.status()["status"] == "applied"
    assert await me._commit_staged_envs() == ([ADDED], {})


async def test_a_transient_build_error_through_the_runtime_waits_not_refused(tmp_path, monkeypatch):
    def failing(config):
        raise OSError(errno.ENETUNREACH, "network unreachable")

    monkeypatch.setattr(bridged_episode, "build_environment", failing)
    me, rt = validator(tmp_path, services=serving_services())
    attested_folder(tmp_path)
    write_add(me, rt)
    built_then_boundary(me, rt, 2)
    assert me._service_schedule_store.status()["status"] == "waiting"
    assert rt.db.execute("SELECT * FROM service_schedule_requests").fetchall() == []
    assert me._join_builds(WAIT)


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
    services.renderer_for = lambda policy: builds.append(threading.current_thread().name) or FakeRenderer()
    env = make_test_episode_env()
    env.name = ADDED
    env.source.identity = ADDED_IDENTITY
    monkeypatch.setattr(type(env.source), "manifest_sha256", lambda self: BLOCK["environment"]["version"],
                        raising=False)
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: env)
    monkeypatch.setattr(catalogue_activation, "install_problem", lambda *a: None)
    monkeypatch.setattr(catalogue_activation, "module_problem", lambda *a: None)
    me, _ = validator(tmp_path, services=services)
    me._service_runtime = rt
    attested_folder(tmp_path)
    write_add(me, rt)
    await asyncio.to_thread(built_then_boundary, me, rt, 2)
    assert len(builds) == 1 and builds[0].startswith("added-env-build") and ADDED in me._staged_envs
    services.renderer_for = lambda policy: pytest.fail("a renderer built on the event loop")
    assert await me._commit_staged_envs() == ([ADDED], {})
    assert ADDED in services.views and ADDED in me.envs


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

@pytest.mark.parametrize("active", [True, False])
def test_the_window_plan_checks_stages_and_plans_an_added_env(tmp_path, monkeypatch, active):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())
    me, rt = validator(tmp_path, services=serving_services())
    attested_folder(tmp_path)
    write_add(me, rt, **({"active": [MATH, EPISODE, ADDED], "shares": SHARES} if active else {}))
    checked = planning(me)
    me._service_window_plan(2)                               # the build starts, the request waits
    assert me._join_builds(WAIT)
    plan = me._service_window_plan(2)
    assert me._service_schedule_store.status()["status"] == "applied" and ADDED in me._staged_envs
    assert (ADDED in dict(plan["env_mix"])) is active and (ADDED in plan["pools"]) is active
    assert checked[-1] is None


async def test_the_boundary_commits_the_staged_env_before_the_window_is_built(tmp_path, monkeypatch):
    added = []
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())
    me, rt = validator(tmp_path, services=serving_services())
    me._episode_services.add_environment = lambda name, env, policy: added.append(name)
    attested_folder(tmp_path)
    write_add(me, rt, active=[MATH, EPISODE, ADDED], shares=SHARES)
    prestaged(me)
    planning(me)
    preparing(me)
    assert await me._prepare_service_window() is True
    assert added == [ADDED] and ADDED in me.envs and me._staged_envs == {}
    assert ADDED in dict(me._candidate_service_window["env_mix"])


def test_the_window_plan_stages_an_env_the_schedule_added_without_this_process(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())
    me, rt = validator(tmp_path, services=serving_services())
    add_to_schedule(rt, active=True)
    me._retry_added_builds()
    assert me._join_builds(WAIT)
    checked = planning(me)
    plan = me._service_window_plan(2)
    assert ADDED in me._staged_envs and ADDED in dict(plan["env_mix"]) and checked == [None]


# ---------------------------------------------------------------- a restart after an add (cooldown snapshots)

def cooldown_snapshot(window, envs):
    from reliquary.constants import TRAINING_RUN_ID
    from reliquary.shared.checkpoint_namespace import active_checkpoint_namespace

    return {"schema_version": 2, **active_checkpoint_namespace().identity, "run_id": TRAINING_RUN_ID,
            "snapshot_window": window, "complete": True, "envs": {name: {} for name in envs}}


def test_a_snapshot_from_before_an_add_is_valid_without_the_added_env():
    snapshot = cooldown_snapshot(5, [MATH])
    validate = ValidationService._validate_cooldown_snapshot
    assert validate(snapshot, {MATH, ADDED}, 7, frozenset({ADDED})) == 5
    with pytest.raises(ValueError, match="incomplete"):
        validate(snapshot, {MATH, ADDED}, 7)                       # an env of the order is never optional
    with pytest.raises(ValueError, match="incomplete"):
        validate(snapshot, {MATH, EPISODE, ADDED}, 7, frozenset({ADDED}))


def test_a_content_snapshot_from_before_an_add_is_valid_without_the_added_env():
    from reliquary.constants import TRAINING_RUN_ID
    from reliquary.shared.checkpoint_namespace import active_checkpoint_namespace

    snapshot = {"schema_version": 1, **active_checkpoint_namespace().identity, "run_id": TRAINING_RUN_ID,
                "complete": True, "snapshot_window": 5, "envs": {MATH: {}}}
    validate = ValidationService._validate_content_snapshot
    assert validate(snapshot, {MATH, ADDED}, 7, frozenset({ADDED})) == 5
    with pytest.raises(ValueError, match="incomplete"):
        validate(snapshot, {MATH, ADDED}, 7)
    # one holding an added env this process does not serve now
    holding = {**snapshot, "envs": {MATH: {}, ADDED: {}}}
    assert validate(holding, {MATH}, 7, frozenset({ADDED})) == 5


async def test_a_restart_with_the_snapshot_from_before_the_add_boots_and_replays(tmp_path, monkeypatch):
    from reliquary.validator.cooldown import CooldownMap

    me, rt = validator(tmp_path)
    add_to_schedule(rt, active=True)
    me._cooldown_per_env = {MATH: CooldownMap(cooldown_windows=3), ADDED: CooldownMap(cooldown_windows=3)}
    me._window_n = 7
    replayed = []

    async def replay(current, gap):
        replayed.append((current, gap))

    async def no_remote(*a, **k):
        return None

    me._replay_cooldown_gap = replay
    me._snapshot_optional_envs = types.MethodType(ValidationService._snapshot_optional_envs, me)
    me._validate_cooldown_snapshot = ValidationService._validate_cooldown_snapshot
    monkeypatch.setattr(service, "_read_gzip_json", lambda path: cooldown_snapshot(5, [MATH]))
    monkeypatch.setattr(service.storage, "download_json", no_remote)
    await ValidationService._rebuild_cooldown_from_history(me)
    assert replayed == [(7, 2)] and len(me._cooldown_per_env[ADDED]) == 0


# ---------------------------------------------------------------- pinned on the source

def test_the_boot_snapshots_the_installs_first_and_installs_added_envs_before_reading_the_contract():
    """``ValidationService.__init__`` is too large to run here: its order is pinned on its source."""
    source = inspect.getsource(ValidationService.__init__)
    snapshot = source.index("self._boot_distributions = installed_distributions()")
    install = source.index("self._require_service_environments(self._install_added_envs(self._service_runtime"
                           ".schedule))")
    assert snapshot < source.index("self._load_launch_environments(")       # before any env is built
    launch = source.index("self._launch_envs = dict(self.envs)")
    assert launch < install and "environments=self._launch_envs" in source   # circuits: the launch envs only
    for reader in ("scheduler_environments = tuple(self.envs)", "self._cooldown_per_env: dict",
                   "self._service_runtime.contract.environments", "ValidatorServer("):
        assert install < source.index(reader), reader


def test_the_episode_services_start_the_prestage_and_take_added_envs_after_the_launch_ones():
    source = inspect.getsource(ValidationService._start_episode_services)
    assert "self._prestage_forever()" in source
    assert "name not in added" in source and "await self._serve_added_envs_at_start(services)" in source
    assert "if name in added:\n                continue" in source     # an unloaded added env never stops it


async def test_an_added_env_the_services_refuse_at_start_is_unloaded_and_switched_off(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())
    me, rt = validator(tmp_path)
    me._install_added_envs(add_to_schedule(rt, active=True))
    assert ADDED in me.envs
    services = serving_services()
    services.add_environment = refusing
    me._episode_services = services
    with caplog.at_level(logging.ERROR):
        await ValidationService._serve_added_envs_at_start(me, services)
    assert ADDED not in me.envs and ADDED not in dict(me.env_mix) and "ALERT" in caplog.text
    assert ADDED not in rt.schedule.active_environments()


def test_the_content_top_up_skips_an_added_env_not_served_now(tmp_path):
    from reliquary.validator.cooldown import ContentCooldownMap, CooldownMap

    me, _ = validator(tmp_path)
    prompts = CooldownMap(cooldown_windows=3)
    prompts.record_batched(4, 6)
    me._cooldown_per_env = {ADDED: prompts}
    me._content_cooldown_per_env = {ADDED: ContentCooldownMap(cooldown_windows=3)}
    assert ValidationService._top_up_content_cooldown_from_prompt_state(me, 5) == 0


def test_an_added_env_that_fails_its_install_at_boot_is_switched_off(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())
    me, rt = validator(tmp_path)

    def failing(name, environment):
        raise ValueError("the registry refused it")

    me._install_env = failing
    now = me._install_added_envs(add_to_schedule(rt, active=True))
    assert ADDED not in now.active_environments() and ADDED not in me.envs


def test_an_inactive_added_env_still_building_is_left_alone_at_the_boundary(tmp_path, monkeypatch):
    builds, release = [], threading.Event()
    monkeypatch.setattr(bridged_episode, "build_environment", gated(builds, release))
    me, rt = validator(tmp_path)
    schedule = add_to_schedule(rt, active=False)
    try:
        me._retry_added_builds(schedule)
        me._stage_added_envs(schedule, window=2)
        assert me._added_retry[ADDED][0] == 0 and me._added_alerts == {}   # no failure counted
    finally:
        release.set()
    assert me._join_builds(WAIT)
    me._stage_added_envs(schedule, window=3)
    assert ADDED in me._staged_envs and len(builds) == 1


def test_a_request_whose_revision_an_auto_deactivation_took_is_refused_saying_so(tmp_path, caplog):
    me, rt = validator(tmp_path)
    add_to_schedule(rt, active=True)
    store = me._service_schedule_store
    pending = build_request(order_sha256=rt.contract.sha256, revision=rt.schedule.revision + 1, cooldowns={MATH: 10})
    store.write(pending)
    with caplog.at_level(logging.ERROR):
        me._deactivate_added_envs({ADDED: "broken"}, 2)
    status = store.status()
    assert status["request_id"] == pending["request_id"] and status["status"] == "refused"
    assert "superseded by auto-deactivate" in status["detail"]
    assert f"resubmit against revision {rt.schedule.revision + 1}" in status["detail"]
    assert "superseded by auto-deactivate" in caplog.text
    rt.apply_pending_schedule_request(store, window=3)                    # recorded: never read again
    assert store.status()["detail"] == status["detail"]


def test_a_request_for_a_later_revision_is_left_alone_by_an_auto_deactivation(tmp_path):
    me, rt = validator(tmp_path)
    add_to_schedule(rt, active=True)
    store = me._service_schedule_store
    pending = build_request(order_sha256=rt.contract.sha256, revision=rt.schedule.revision + 2, cooldowns={MATH: 10})
    store.write(pending)
    me._deactivate_added_envs({ADDED: "broken"}, 2)
    assert store.status() is None or store.status().get("request_id") != pending["request_id"]


async def test_an_added_env_unloaded_at_boot_keeps_its_cooldown_from_the_snapshot(tmp_path, monkeypatch):
    from reliquary.validator.cooldown import ContentCooldownMap, CooldownMap

    me, rt = validator(tmp_path)
    add_to_schedule(rt, active=False)
    me.envs = {MATH: object()}                                      # ADDED could not be loaded at boot
    me._cooldown_per_env = {MATH: CooldownMap(cooldown_windows=3), ADDED: CooldownMap(cooldown_windows=10)}
    me._content_cooldown_per_env = {MATH: ContentCooldownMap(cooldown_windows=3),
                                    ADDED: ContentCooldownMap(cooldown_windows=10)}
    me._window_n = 5
    snapshot = cooldown_snapshot(5, [MATH, ADDED])
    snapshot["envs"][ADDED] = {"4": 3}

    async def no_remote(*a, **k):
        return None

    me._snapshot_optional_envs = types.MethodType(ValidationService._snapshot_optional_envs, me)
    me._validate_cooldown_snapshot = ValidationService._validate_cooldown_snapshot
    monkeypatch.setattr(service, "_read_gzip_json", lambda path: snapshot)
    monkeypatch.setattr(service.storage, "download_json", no_remote)
    await ValidationService._rebuild_cooldown_from_history(me)
    assert me._cooldown_per_env[ADDED].export_state() == {4: 3}
    kept = me._cooldown_per_env[ADDED]
    me._install_env(ADDED, built())                                   # loaded later: the map is kept as is
    assert me._cooldown_per_env[ADDED] is kept


def test_the_boot_keeps_cooldown_maps_for_added_envs_it_could_not_load():
    source = inspect.getsource(ValidationService.__init__)
    assert "unloaded = [name for name in schedule.added_environments() if name not in self.envs]" in source
    assert "for name in [*self.envs, *unloaded]:" in source


def test_a_build_the_dry_run_refuses_is_a_failed_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: built())
    services = serving_services()
    services.environment_refusal = lambda policy: "renderer stop ids [9] are outside"
    me, rt = validator(tmp_path, services=services)
    schedule = add_to_schedule(rt, active=False)
    assert me._retry_added_builds(schedule) == [ADDED] and me._join_builds(WAIT)
    attempts, due = me._added_retry[ADDED]
    assert attempts == 1 and due < float("inf") and me._retry_added_builds(schedule) == []
