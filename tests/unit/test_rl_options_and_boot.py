"""Bridged RL envs are placed only where the gateway serves them with the validator's options, and an episode
env no machine serves is started degraded, never stopping the validator."""
import asyncio
import logging

import pytest

pytest.importorskip("reliquary_sandbox.attest")

from reliquary.sandbox.routes import REFUSAL_STATUS  # noqa: E402
from reliquary.sandbox.tasks import ResolvedTask  # noqa: E402
from tests.unit.episode_v2_fixtures import ENV_PACKAGE, EPISODE, SANDBOX_ENV  # noqa: E402
from tests.unit.sandbox_fixtures import ADDRESS, BUDGETS, ENV, IMAGE, MACHINE, NOW, capacity_report  # noqa: E402
from tests.unit.sandbox_fixtures import ENV_PACKAGE as FLEET_PACKAGE  # noqa: E402
from tests.unit.test_fleet_resources import RESERVE, RESOURCES, fleet_with  # noqa: E402
from tests.unit.test_rl_engagements import build, open_  # noqa: E402
from tests.unit.test_rl_episode_wiring import _services  # noqa: E402
from tests.unit.test_sandbox_fleet import make  # noqa: E402

DIGEST = "1" * 64
LOCK = "2" * 64


def test_the_rl_open_places_with_the_tasks_options_digest(tmp_path):
    env = build(tmp_path)
    view = env.issuer._engagements["rl_precommit"]._environments(EPISODE)

    async def resolve(index):
        return ResolvedTask(IMAGE, {}, DIGEST)

    env.issuer._engagements["rl_precommit"]._environments = lambda name: type(view)(policy=view.policy,
                                                                                    resolve_task=resolve)
    open_(env, 0)
    assert env.issuer._fleet.picked[-1]["env_options_sha256"] == DIGEST


def test_an_open_of_a_degraded_env_is_refused_before_placement(tmp_path):
    env = build(tmp_path)
    engagements = env.issuer._engagements["rl_precommit"]
    asked = []

    def serving_refusal(environment):
        from reliquary.sandbox.sessions import Refusal

        asked.append(environment)
        return Refusal("environment_unserved", {"environment": environment}, retry_after=5)

    engagements.serving_refusal = serving_refusal
    refusal = open_(env, 0)
    assert refusal.reason == "environment_unserved" and refusal.retry_after == 5
    assert asked == [env.precommit.environment] and env.issuer._fleet.picked == []
    engagements.serving_refusal = lambda environment: None
    assert not hasattr(open_(env, 1), "reason")                      # served again: placed


def test_serving_requires_the_exact_identity_options_resources_and_lock(tmp_path):
    fleet, machine, served, _ = make(tmp_path)
    served[ADDRESS] = capacity_report(
        machine, at=NOW, env_packages={SANDBOX_ENV: ENV_PACKAGE}, env_options_sha256={SANDBOX_ENV: DIGEST},
        env_image_locks={SANDBOX_ENV: {"sha256": LOCK, "images": 3, "present": 2}}, resources=RESOURCES)
    asyncio.run(fleet.poll_once())
    assert fleet.serving(SANDBOX_ENV, ENV_PACKAGE, now=NOW) == [MACHINE]
    assert fleet.serving(SANDBOX_ENV, ENV_PACKAGE, env_options_sha256=DIGEST, now=NOW) == [MACHINE]
    assert fleet.serving(SANDBOX_ENV, ENV_PACKAGE.removesuffix(".vfb2"), now=NOW) == []
    assert fleet.serving(SANDBOX_ENV, ENV_PACKAGE, env_options_sha256="9" * 64, now=NOW) == []
    assert fleet.serving(SANDBOX_ENV, ENV_PACKAGE, lock_sha256=LOCK, now=NOW) == []          # 2 of 3 images
    assert fleet.serving(SANDBOX_ENV, ENV_PACKAGE, lock_sha256=LOCK, min_present=2, now=NOW) == [MACHINE]
    assert fleet.serving(SANDBOX_ENV, ENV_PACKAGE, lock_sha256="9" * 64, min_present=0, now=NOW) == []
    assert fleet.serving(SANDBOX_ENV, ENV_PACKAGE, now=NOW + 31) == []                       # stale report
    fleet.drained.add(MACHINE)
    assert fleet.serving(SANDBOX_ENV, ENV_PACKAGE, now=NOW) == []
    assert fleet.reported_packages(SANDBOX_ENV) == [ENV_PACKAGE]
    assert fleet.reported_packages("nobody-serves-this") == []


def test_once_resources_are_required_a_machine_without_them_serves_nothing(tmp_path):
    fleet, machine, served, _ = make(tmp_path)
    served[ADDRESS] = capacity_report(machine, at=NOW, env_packages={SANDBOX_ENV: ENV_PACKAGE})
    asyncio.run(fleet.poll_once())
    assert fleet.serving(SANDBOX_ENV, ENV_PACKAGE, now=NOW) == [MACHINE]
    fleet.resources_required = True
    assert fleet.serving(SANDBOX_ENV, ENV_PACKAGE, now=NOW) == []


def test_required_resources_no_machine_reports_cannot_be_judged_not_never(tmp_path, caplog):
    """A fleet that requires resources while a machine serving the env reports none (a configuration miss) is
    our state: the open is retryable, never a final ``task_unplaceable`` for the seed; the machine is named in
    an ALERT. A machine whose valid resources say "never" (a box smaller than the task) still answers False."""
    fleet, machine, served, _ = make(tmp_path)
    served[ADDRESS] = capacity_report(machine, at=NOW)
    asyncio.run(fleet.poll_once())
    fleet.resources_required = True
    place = dict(image=IMAGE, env=ENV, env_package=FLEET_PACKAGE, budgets=BUDGETS, validity_s=60, now=NOW,
                 reserve=RESERVE)
    with caplog.at_level(logging.ERROR):
        assert fleet.could_place(**place) is None
        assert fleet.pick(**place) is None
    assert any("ALERT" in r.message and MACHINE in r.message for r in caplog.records)
    (tmp_path / "small").mkdir()
    small, *_ = fleet_with(tmp_path / "small", max_box_cpu_millis=500)
    small.resources_required = True
    assert small.could_place(**place) is False                        # its resources say never: final


async def test_an_env_no_machine_serves_starts_degraded_and_is_refused_retryably(tmp_path, caplog):
    services, _ = _services(tmp_path, boot_wait_s=0.0)
    assert services.served == {EPISODE: (SANDBOX_ENV, ENV_PACKAGE, None)}
    with caplog.at_level(logging.ERROR):
        await services.start()                                       # never raises
    assert services.unserved_envs == {EPISODE}
    alert = [r.message for r in caplog.records if "ALERT" in r.message]
    assert alert and ENV_PACKAGE in alert[0] and "nothing" in alert[0]
    refusal = services.serving_refusal(EPISODE)
    assert refusal.reason == "environment_unserved" and refusal.retry_after
    assert services.issuer._engagements["rl_precommit"].serving_refusal == services.serving_refusal
    await services.stop()


async def test_a_degraded_env_is_served_as_soon_as_a_machine_reports_it(tmp_path):
    services, _ = _services(tmp_path, boot_wait_s=0.0)
    await services.start()
    assert services.unserved_envs == {EPISODE}
    services.fleet.serving = lambda *args, **kwargs: ["m"]           # a machine now reports it
    watch = asyncio.ensure_future(services.watch_served_forever(0.01))
    for _ in range(100):
        if not services.unserved_envs:
            break
        await asyncio.sleep(0.01)
    watch.cancel()
    assert services.unserved_envs == set() and services.serving_refusal(EPISODE) is None
    await services.stop()


async def test_the_boot_wait_polls_until_served_and_the_watch_runs_in_the_background(tmp_path):
    services, _ = _services(tmp_path, boot_wait_s=30.0)
    polls = []
    real_poll = services.fleet.poll_once

    async def poll_once():
        polls.append(1)
        await real_poll()

    services.fleet.poll_once = poll_once
    services.fleet.serving = lambda *args, **kwargs: ["m"] if len(polls) >= 2 else []

    async def no_sleep(_seconds):
        return None

    import reliquary.validator.rl_sandbox_wiring as wiring

    real_sleep = wiring.asyncio.sleep
    wiring.asyncio.sleep = no_sleep
    try:
        await services.start()
    finally:
        wiring.asyncio.sleep = real_sleep
    assert len(polls) == 2 and services.unserved_envs == set()
    jobs = services.background()
    names = [job.cr_code.co_name for job in jobs]
    for job in jobs:
        job.close()
    assert "watch_served_forever" in names
    await services.stop()


async def test_without_a_boot_wait_the_start_is_unchanged(tmp_path):
    services, _ = _services(tmp_path)
    await services.start()
    assert services.unserved_envs == set()
    jobs = services.background()
    names = [job.cr_code.co_name for job in jobs]
    for job in jobs:
        job.close()
    assert "watch_served_forever" not in names
    await services.stop()


def test_the_unserved_refusal_is_retried():
    from reliquary.miner.rl_episode_client import RL_THROTTLES

    assert REFUSAL_STATUS["environment_unserved"] == 503
    assert "environment_unserved" in RL_THROTTLES          # the miner reopens; never "not serving"
