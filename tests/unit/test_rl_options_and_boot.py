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
from tests.unit.test_sandbox_fleet import make  # noqa: E402

DIGEST = "1" * 64
LOCK = "2" * 64


@pytest.fixture(autouse=True)
def _sandbox_alerts_reach_caplog(caplog):
    # Bittensor changes existing logger levels during import; keep alert assertions independent of test order.
    for name in ("reliquary.sandbox.fleet", "reliquary.sandbox.sessions", "reliquary.validator.rl_sandbox_wiring"):
        caplog.set_level(logging.ERROR, logger=name)


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


def test_the_unserved_refusal_is_retried():
    from reliquary.miner.rl_episode_client import RL_THROTTLES

    assert REFUSAL_STATUS["environment_unserved"] == 503
    assert "environment_unserved" in RL_THROTTLES          # the miner reopens; never "not serving"


@pytest.mark.parametrize("window", ["missing", 121, 600, 0, -5, "120", 120.0, True, None])
def test_a_resources_block_without_a_present_window_of_at_most_120_s_is_invalid(tmp_path, caplog, window):
    """A gateway accepting a token after 120 s would open a session that neither the hold nor the quota counts
    any more: its resources are invalid. Never a final refusal: the open cannot be judged (retryable), the env is
    unserved (degraded) and the machine is named in an ALERT."""
    resources = {k: v for k, v in RESOURCES.items() if k != "present_window_s"}
    if window != "missing":
        resources["present_window_s"] = window
    fleet, machine, served, _ = make(tmp_path)
    served[ADDRESS] = capacity_report(machine, at=NOW, resources=resources)
    asyncio.run(fleet.poll_once())
    fleet.resources_required = True
    place = dict(image=IMAGE, env=ENV, env_package=FLEET_PACKAGE, budgets=BUDGETS, validity_s=60, now=NOW,
                 reserve=RESERVE)
    with caplog.at_level(logging.ERROR):
        assert fleet.pick(**place) is None
        assert fleet.could_place(**place) is None                      # never final for the seed
    assert fleet.serving(ENV, FLEET_PACKAGE, now=NOW) == []
    assert any("ALERT" in r.message and MACHINE in r.message and "present_window" in r.message
               for r in caplog.records)


@pytest.mark.parametrize("window", [120, 45, 10])
def test_a_present_window_of_at_most_120_s_is_valid(tmp_path, window):
    from reliquary.sandbox.fleet import MAX_PRESENT_WINDOW_S
    from reliquary.sandbox.rl_capacity import PRESENT_WINDOW_S

    assert MAX_PRESENT_WINDOW_S == PRESENT_WINDOW_S
    fleet, machine, served, _ = make(tmp_path)
    served[ADDRESS] = capacity_report(machine, at=NOW, resources={**RESOURCES, "present_window_s": window})
    asyncio.run(fleet.poll_once())
    fleet.resources_required = True
    place = dict(image=IMAGE, env=ENV, env_package=FLEET_PACKAGE, budgets=BUDGETS, validity_s=60, now=NOW,
                 reserve=RESERVE)
    assert fleet.pick(**place) is not None and fleet.could_place(**place) is True
    assert fleet.serving(ENV, FLEET_PACKAGE, now=NOW) == [MACHINE]


# --- our state is never charged to the miner: an unserved env is retryable and untallied ------------------------

def _set_task_digest(env, digest):
    engagements = env.issuer._engagements["rl_precommit"]
    view = engagements._environments(EPISODE)

    async def resolve(index):
        return ResolvedTask(IMAGE, {}, digest)

    engagements._environments = lambda name: type(view)(policy=view.policy, resolve_task=resolve)


