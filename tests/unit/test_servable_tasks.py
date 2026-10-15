"""Precommits are limited to the tasks some machine can serve: a task is servable where the open path itself could
place it; a fleet view that is not ready is answered retryably, a task no serving machine can take is refused for
the window, and nothing is recorded either way."""
import asyncio
import dataclasses
from types import SimpleNamespace

import pytest

pytest.importorskip("reliquary_sandbox.attest")

from reliquary.sandbox.rl_routes import PRECOMMIT_STATUS  # noqa: E402
from reliquary.sandbox.sessions import token_validity_s  # noqa: E402
from reliquary.sandbox.tasks import ResolvedTask  # noqa: E402
from tests.unit.episode_v2_fixtures import ENV_PACKAGE, EPISODE, SANDBOX_ENV, TASK  # noqa: E402
from reliquary.sandbox.fleet import HEARTBEAT_MAX_AGE_S, Fleet  # noqa: E402
from tests.unit.sandbox_fixtures import (  # noqa: E402
    ADDRESS, GIB, IMAGE, NOW, capacity_report, machine_document, signer,
)
from tests.unit.test_fleet_resources import RESOURCES  # noqa: E402
from tests.unit.test_rl_episode_wiring import _services  # noqa: E402
from tests.unit.test_sandbox_fleet import PICK, Clock, make  # noqa: E402

OTHER = "registry.example/other@sha256:" + "b" * 64


def serving_fleet(tmp_path, **report):
    fleet, machine, served, clock = make(tmp_path)
    report.setdefault("resources", RESOURCES)
    served[ADDRESS] = capacity_report(machine, at=NOW, env_packages={SANDBOX_ENV: ENV_PACKAGE}, **report)
    asyncio.run(fleet.poll_once())
    fleet.resources_required = True
    fleet._test_machine, fleet._test_served = machine, served
    return fleet


def task(**change):
    value = {**PICK, "env": SANDBOX_ENV, "env_package": ENV_PACKAGE, "image": IMAGE, "now": NOW,
             "reserve": {"cpu_millis": 500, "memory_bytes": GIB}}
    value.update(change)
    return value


def test_a_task_is_servable_exactly_where_the_open_path_could_place_it(tmp_path):
    fleet = serving_fleet(tmp_path)
    cases = {
        "servable": ({}, 1),
        "image absent": ({"image": OTHER}, 0),
        "box too small": ({"reserve": {"cpu_millis": 4_000, "memory_bytes": GIB}}, 0),
        "memory too big": ({"reserve": {"cpu_millis": None, "memory_bytes": 32 * GIB}}, 0),
        "another identity": ({"env_package": "x==1+g0.vfb2"}, 0),
        "token too long for the machine": ({"validity_s": 10**9}, 0),
    }
    for name, (change, machines) in cases.items():
        assert fleet.servable(**task(**change)) == machines, name
        assert fleet.could_place(**task(**change)) is (machines > 0), name          # one filter, both paths
    assert IMAGE in fleet.images_served(SANDBOX_ENV, ENV_PACKAGE, now=NOW)
    assert OTHER not in fleet.images_served(SANDBOX_ENV, ENV_PACKAGE, now=NOW)


def test_a_task_is_servable_only_where_its_machines_can_host_a_group_at_once(tmp_path):
    fleet = serving_fleet(tmp_path)
    hosts = fleet.sessions_hostable(fleet.servable_machines(**task()), task()["reserve"])
    assert hosts == 4                                        # its 4 slots; 2 000 mCPU / 500 and 16 GiB / 1 GiB allow more
    assert fleet.servable(**task(), group_size=hosts) == 1
    assert fleet.servable(**task(), group_size=hosts + 1) == 0
    big = task(reserve={"cpu_millis": 1_000, "memory_bytes": GIB})
    assert fleet.servable(**big, group_size=2) == 1         # 2 000 mCPU / 1 000: two at once
    assert fleet.servable(**big, group_size=3) == 0
    assert fleet.could_place(**task())                       # the open path itself is unchanged


def test_a_report_of_thousands_of_images_is_indexed_once_per_report(tmp_path):
    many = [f"registry.example/task-{i}@sha256:" + f"{i:064x}" for i in range(5_000)]
    fleet = serving_fleet(tmp_path, images=many)
    report = fleet._reports["machine-1"]
    indexed = fleet._images_of("machine-1", report)
    assert len(indexed) == 5_000 and many[4_321] in indexed
    for image in (many[0], many[-1]):
        assert fleet.servable(**task(image=image)) == 1
    assert fleet._images_of("machine-1", report) is indexed          # never rebuilt per request
    served = fleet.images_served(SANDBOX_ENV, ENV_PACKAGE, now=NOW)
    assert fleet.images_served(SANDBOX_ENV, ENV_PACKAGE, now=NOW) is served
    # The next report drops an image: the next lookup sees it gone.
    fleet._test_served[ADDRESS] = capacity_report(fleet._test_machine, at=NOW, images=many[1:],
                                                  env_packages={SANDBOX_ENV: ENV_PACKAGE}, resources=RESOURCES)
    asyncio.run(fleet.poll_once())
    assert fleet.servable(**task(image=many[0])) == 0
    assert fleet.servable(**task(image=many[1])) == 1
    assert many[0] not in fleet.images_served(SANDBOX_ENV, ENV_PACKAGE, now=NOW)


def admission(services, image=IMAGE, cpu_millis=None, prepare_timeout_s=None, lock=None):
    view = services.views[EPISODE]

    async def resolve(index):
        return ResolvedTask(image, {}, None, prepare_timeout_s, cpu_millis=cpu_millis, env_lock_sha256=lock)

    services.views[EPISODE] = type(view)(policy=view.policy, resolve_task=resolve)
    return SimpleNamespace(environment=EPISODE, task_index=TASK, window=1)


async def test_a_precommit_for_an_unservable_task_is_refused_without_sanction(tmp_path):
    services, _ = _services(tmp_path)
    services.fleet.serving = lambda *a, **k: ["m"]                 # fresh machines serve the env
    services.fleet.servable = lambda **k: 0                        # ... but none can take this task
    services.fleet.could_place = lambda **k: False                 # ... and the open path would say so
    value = admission(services, image=OTHER)
    reason, detail = await services.precommit_admission(value)
    assert reason == "task_unservable" and detail["image"] == OTHER and detail["task_index"] == TASK
    services.fleet.servable = lambda **k: 1
    assert await services.precommit_admission(value) is None


def test_an_image_on_no_machine_of_a_served_env_is_refused_at_the_precommit(tmp_path):
    """A real fleet: the env is served, the task's image is on none of its machines."""
    services, _ = _services(tmp_path)
    policy = services.views[EPISODE].policy
    assert (policy.sandbox_env, policy.env_package) == (SANDBOX_ENV, ENV_PACKAGE) and policy.pool_seeds // 2 == 8
    roomy = {**RESOURCES, "cpu_millis": 4_000, "free_cpu_millis": 4_000, "memory_bytes": 32 * GIB,
             "free_memory_bytes": 32 * GIB}
    (tmp_path / "fleet").mkdir()
    fleet = serving_fleet(tmp_path / "fleet", capacity=8, free=8, resources=roomy)
    services.fleet, services.clock = fleet, fleet._clock
    reason, detail = asyncio.run(services.precommit_admission(admission(services, image=OTHER, cpu_millis=500)))
    assert reason == "task_unservable" and detail["machines"] == 0
    assert asyncio.run(services.precommit_admission(admission(services, image=IMAGE, cpu_millis=500))) is None
    # 8 slots, but 4 000 mCPU hold only 4 sessions of 1 000: its group of 8 cannot run at once on this fleet
    reason, _ = asyncio.run(services.precommit_admission(admission(services, image=IMAGE, cpu_millis=1_000)))
    assert reason == "task_unservable"


SECOND = "http://10.0.0.6:8080"


def two_machine_services(tmp_path):
    """The test env (M = 8) over two machines of 4 sessions each: its group fits only on both together."""
    services, _ = _services(tmp_path)
    keys = tmp_path / "two"
    keys.mkdir()
    first, second = signer(keys, "a", "ka"), signer(keys, "b", "kb")
    documents = [machine_document(first), machine_document(second, machine_id="machine-2", address=SECOND)]
    served, clock = {}, Clock()

    async def read_documents():
        return documents

    async def fetch(address):
        return served.get(address)

    fleet = Fleet(read_documents=read_documents, fetch_report=fetch, clock=clock)
    asyncio.run(fleet.refresh_directory())
    fleet.resources_required = True

    def report(signed, at, **change):
        kw = dict(at=at, env_packages={SANDBOX_ENV: ENV_PACKAGE}, resources=RESOURCES)
        if signed is second:
            kw.update(machine_id="machine-2", address=SECOND)
        kw.update(change)
        return capacity_report(signed, **kw)

    services.fleet, services.clock = fleet, clock
    return services, fleet, clock, served, first, second, report


def test_a_group_needing_a_machine_that_missed_its_heartbeat_is_answered_retryably(tmp_path):
    services, fleet, clock, served, first, second, report = two_machine_services(tmp_path)
    value = admission(services, cpu_millis=500)
    served[ADDRESS], served[SECOND] = report(first, NOW), report(second, NOW)
    asyncio.run(fleet.poll_once())
    assert asyncio.run(services.precommit_admission(value)) is None                # 4 + 4 sessions hold M = 8
    # The second machine misses its heartbeats: only the first is fresh, and alone it holds 4 of the 8.
    clock.now = NOW + HEARTBEAT_MAX_AGE_S + 10
    served[ADDRESS], served[SECOND] = report(first, clock.now), None
    asyncio.run(fleet.poll_once())
    assert "machine-2" not in fleet.drained
    assert fleet.could_place(**task(now=clock.now, validity_s=4500)) is True       # the open path places it
    reason, _ = asyncio.run(services.precommit_admission(value))
    assert reason == "fleet_unavailable"                                            # never final on a gap


def test_a_group_no_complete_fresh_view_can_host_is_refused_for_the_window(tmp_path):
    services, fleet, clock, served, first, second, report = two_machine_services(tmp_path)
    served[ADDRESS], served[SECOND] = report(first, NOW), report(second, NOW, images=(OTHER,))
    asyncio.run(fleet.poll_once())
    reason, detail = asyncio.run(services.precommit_admission(admission(services, cpu_millis=500)))
    assert reason == "task_unservable" and detail["machines"] == 0


async def test_a_slow_task_source_is_answered_retryably_never_admitted_unchecked(tmp_path):
    services, _ = _services(tmp_path)
    services.book.policy = dataclasses.replace(services.book.policy, io_timeout_s=0.01)
    view = services.views[EPISODE]

    async def slow(index):
        await asyncio.sleep(3600)

    services.views[EPISODE] = type(view)(policy=view.policy, resolve_task=slow)
    services.fleet.serving = lambda *a, **k: pytest.fail("no fleet check without the task")
    reason, detail = await services.precommit_admission(SimpleNamespace(environment=EPISODE, task_index=TASK,
                                                                        window=1))
    assert reason == "fleet_unavailable" and detail["why"] == "task_resolve_timeout"


async def test_a_task_the_open_path_cannot_judge_now_is_answered_retryably(tmp_path):
    services, _ = _services(tmp_path)
    services.fleet.serving = lambda *a, **k: ["m"]
    services.fleet.servable = lambda **k: 0
    services.fleet.could_place = lambda **k: None                  # a machine holding it is stale or drained
    reason, _ = await services.precommit_admission(admission(services))
    assert reason == "fleet_unavailable"


async def test_a_fleet_view_that_is_not_ready_is_answered_retryably(tmp_path):
    services, _ = _services(tmp_path)
    services.fleet.serving = lambda *a, **k: []                    # a stale directory, a missed heartbeat
    services.fleet.servable = lambda **k: pytest.fail("no task check without a fleet view")
    reason, _ = await services.precommit_admission(admission(services))
    assert reason == "fleet_unavailable"


async def test_an_admission_that_fails_is_answered_retryably(tmp_path):
    services, _ = _services(tmp_path)

    def broken(*a, **k):
        raise RuntimeError("boom")

    services.fleet.serving = broken
    reason, _ = await services.precommit_admission(admission(services))
    assert reason == "fleet_unavailable"


async def test_the_admission_passes_the_sessions_own_budgets_and_validity(tmp_path):
    services, _ = _services(tmp_path)
    seen = {}
    services.fleet.serving = lambda *a, **k: ["m"]
    services.fleet.servable = lambda **k: seen.update(k) or 1
    assert await services.precommit_admission(admission(services, cpu_millis=500)) is None
    policy = services.views[EPISODE].policy
    assert seen["budgets"] == policy.budgets_dict()                # no task limit raises them here
    assert seen["validity_s"] == token_validity_s(services.book.policy.open_window_s, int(seen["budgets"]["wall_s"]))
    assert seen["reserve"] == {"cpu_millis": 500, "memory_bytes": int(seen["budgets"]["memory_bytes"])}
    assert seen["group_size"] == policy.pool_seeds // 2
    # a long preparation lengthens the token, as the open path's own validity does
    assert await services.precommit_admission(admission(services, prepare_timeout_s=1_800)) is None
    wall = int(seen["budgets"]["wall_s"])
    assert seen["validity_s"] == token_validity_s(services.book.policy.open_window_s, wall, 1_800)
    assert seen["validity_s"] > services.book.policy.open_window_s + wall


async def test_a_precommit_for_a_degraded_env_is_refused_retryably(tmp_path):
    services, _ = _services(tmp_path)
    services.unserved_envs = {EPISODE}
    reason, _ = await services.precommit_admission(SimpleNamespace(environment=EPISODE, task_index=TASK, window=1))
    assert reason == "environment_unserved"


async def test_a_task_the_source_cannot_resolve_is_left_to_the_runtimes_checks(tmp_path):
    services, _ = _services(tmp_path)
    view = services.views[EPISODE]

    async def broken(index):
        raise IndexError(index)

    services.views[EPISODE] = type(view)(policy=view.policy, resolve_task=broken)
    assert await services.precommit_admission(SimpleNamespace(environment=EPISODE, task_index=10**6, window=1)) is None


def test_the_engagements_and_the_admission_read_the_same_views(tmp_path):
    services, _ = _services(tmp_path)
    engagements = services.issuer._engagements["rl_precommit"]
    assert engagements._environments(EPISODE) is services.views[EPISODE]


def test_the_refusal_statuses():
    # task_unservable is final for this task in this window (the miner moves on); the others are retried
    assert PRECOMMIT_STATUS["task_unservable"] == 409
    assert PRECOMMIT_STATUS["environment_unserved"] == PRECOMMIT_STATUS["fleet_unavailable"] == 503


async def test_the_precommit_admission_counts_machines_under_the_tasks_image_lock(tmp_path):
    services, _ = _services(tmp_path)
    seen = []
    services.fleet.serving = lambda *a, **k: ["m"]
    services.fleet.servable = lambda **k: seen.append(k) or 1
    assert await services.precommit_admission(admission(services, lock="9" * 64)) is None
    assert seen[-1]["env_lock_sha256"] == "9" * 64
    assert await services.precommit_admission(admission(services)) is None
    assert "env_lock_sha256" not in seen[-1]                       # an unlocked task keeps its exact keywords
