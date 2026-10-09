"""The runtime's episode precommits (plan 2C, Task 3)."""
import time

import pytest

from reliquary.protocol.service_episode import EpisodePrecommit
from reliquary.services.runtime import EpisodePrecommitRefused, ServicePolicyLimit, ServiceRuntime
from tests.unit.episode_v2_fixtures import EPISODE, TASK, episode_precommit, episode_runtime
from tests.unit.service_v2_fixtures import MATH, qualification_v2

HOTKEY = "5Hot"


def _refused(rt, precommit):
    with pytest.raises(EpisodePrecommitRefused) as refused:
        rt.record_episode_precommit(precommit)
    return refused.value


def test_a_precommit_is_recorded_once_and_read_back(tmp_path):
    rt = episode_runtime(tmp_path)
    precommit = episode_precommit(rt.contract, hotkey=HOTKEY)
    assert rt.record_episode_precommit(precommit) == (True, precommit.sha256)
    assert rt.record_episode_precommit(precommit) == (False, precommit.sha256)
    assert rt.episode_precommit(precommit.sha256) == precommit
    assert rt.episode_precommit("0" * 64) is None
    assert rt.episode_precommit("not a digest") is None


def test_each_hotkey_and_task_has_its_own_precommit(tmp_path):
    rt = episode_runtime(tmp_path)
    precommits = [episode_precommit(rt.contract, hotkey=HOTKEY), episode_precommit(rt.contract, hotkey="5Another"),
                  episode_precommit(rt.contract, hotkey=HOTKEY, task=TASK + 1)]
    assert [rt.record_episode_precommit(p)[0] for p in precommits] == [True, True, True]
    assert len({p.sha256 for p in precommits}) == 3


def test_one_live_precommit_per_hotkey_env_task_and_window(tmp_path):
    rt = episode_runtime(tmp_path)
    with rt.lock, rt.db:                  # a row of the same key under another body (a future field)
        rt.db.execute("INSERT INTO service_episode_precommits VALUES(?,?,?,?,?,?,?,?)",
                      (rt.contract.sha256, "f" * 64, 1, EPISODE, TASK, HOTKEY, "{}", time.time()))
    refused = _refused(rt, episode_precommit(rt.contract, hotkey=HOTKEY))
    assert (refused.reason, refused.existing) == ("precommit_exists", "f" * 64)


@pytest.mark.parametrize("change, reason", [
    (dict(order="e" * 64), "order_mismatch"),
    (dict(environment=MATH), "environment_not_episode"),
    (dict(environment="unknown_env"), "environment_not_episode"),
    (dict(task_index=1000), "task_out_of_range"),
    (dict(checkpoint="e" * 40), "checkpoint_mismatch"),
    (dict(pool_sha256="e" * 64), "pool_mismatch"),
    (dict(window=2), "window_not_open"),
])
def test_a_precommit_the_frozen_window_does_not_allow_is_refused_and_not_written(tmp_path, change, reason):
    rt = episode_runtime(tmp_path)
    precommit = EpisodePrecommit.from_dict({**episode_precommit(rt.contract, hotkey=HOTKEY).to_dict(), **change})
    assert _refused(rt, precommit).reason == reason
    assert rt.db.execute("SELECT COUNT(*) FROM service_episode_precommits").fetchone()[0] == 0


def test_a_settled_window_takes_no_precommit(tmp_path):
    rt = episode_runtime(tmp_path)
    with rt.lock, rt.db:
        rt.db.execute("INSERT INTO service_settled VALUES(?,?,?,?)", (rt.contract.sha256, 1, 0, "{}"))
    assert _refused(rt, episode_precommit(rt.contract, hotkey=HOTKEY)).reason == "window_closed"


def test_a_precommit_survives_a_restart(tmp_path):
    rt = episode_runtime(tmp_path)
    precommit = episode_precommit(rt.contract, hotkey=HOTKEY)
    rt.record_episode_precommit(precommit)
    rt.close()
    again = ServiceRuntime(tmp_path / "runtime.sqlite3", rt.contract, qualification_v2(rt.contract),
                           drand_round_at=lambda instant: 1_000)
    assert again.episode_precommit(precommit.sha256) == precommit
    assert again.record_episode_precommit(precommit) == (False, precommit.sha256)


def test_a_window_with_a_precommit_is_never_discarded(tmp_path):
    rt = episode_runtime(tmp_path)
    rt.record_episode_precommit(episode_precommit(rt.contract, hotkey=HOTKEY))
    with pytest.raises(ServicePolicyLimit, match="precommit"):
        rt.discard_unactivated_window(1)


def test_a_task_in_cooldown_in_the_window_is_refused_and_not_written(tmp_path):
    rt = episode_runtime(tmp_path)
    asked = []

    def cooling(environment, task_index, window):
        asked.append((environment, task_index, window))
        return task_index == TASK

    rt.task_in_cooldown = cooling
    assert _refused(rt, episode_precommit(rt.contract, hotkey=HOTKEY)).reason == "task_in_cooldown"
    assert asked == [(EPISODE, TASK, 1)]
    assert rt.db.execute("SELECT COUNT(*) FROM service_episode_precommits").fetchone()[0] == 0
    other = episode_precommit(rt.contract, hotkey=HOTKEY, task=TASK + 1)      # a task out of cooldown
    assert rt.record_episode_precommit(other) == (True, other.sha256)


def test_a_precommit_recorded_before_the_cooldown_is_still_answered(tmp_path):
    rt = episode_runtime(tmp_path)
    precommit = episode_precommit(rt.contract, hotkey=HOTKEY)
    rt.record_episode_precommit(precommit)
    rt.task_in_cooldown = lambda *_: True
    assert rt.record_episode_precommit(precommit) == (False, precommit.sha256)   # idempotent retry


def test_the_validator_installs_its_prompt_cooldown_on_the_runtime():
    import inspect

    from reliquary.validator import service
    source = inspect.getsource(service)
    assert "task_in_cooldown" in source and "_cooldown_per_env[environment].is_in_cooldown" in source
