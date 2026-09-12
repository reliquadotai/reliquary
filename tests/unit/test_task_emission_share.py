"""A new task pays nobody until someone decides otherwise."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


def _share(env: dict) -> subprocess.CompletedProcess:
    clean = {k: v for k, v in os.environ.items() if not k.startswith("RELIQUARY_")}
    clean.update(env)
    return subprocess.run(
        [sys.executable, "-c", "import reliquary.constants as c; print(c.TASK_EMISSION_SHARE)"],
        capture_output=True, text=True, env=clean,
    )


def test_the_legacy_task_keeps_the_whole_pool():
    completed = _share({})

    assert completed.returncode == 0, completed.stderr
    assert float(completed.stdout.strip()) == 1.0


def test_a_new_task_takes_nothing_by_default():
    completed = _share({"RELIQUARY_TASK_ID": "logic-probe"})

    assert completed.returncode == 0, completed.stderr
    assert float(completed.stdout.strip()) == 0.0


def test_a_share_can_be_declared():
    completed = _share({"RELIQUARY_TASK_ID": "logic-probe", "RELIQUARY_TASK_EMISSION_SHARE": "0.25"})

    assert float(completed.stdout.strip()) == 0.25


@pytest.mark.parametrize("bad", ["-0.1", "1.5", "nan", "abc"])
def test_an_impossible_share_refuses_to_import(bad):
    completed = _share({"RELIQUARY_TASK_ID": "logic-probe", "RELIQUARY_TASK_EMISSION_SHARE": bad})

    assert completed.returncode != 0
    assert "RELIQUARY_TASK_EMISSION_SHARE" in completed.stderr


def test_a_zero_share_pays_nobody():
    from reliquary.validator.token_rewards import AcceptedGroup, split_environment_pool

    groups = [AcceptedGroup(hotkey="hk", operator_id="hk", eos_tokens=10)]

    assert split_environment_pool(groups, pool=0.0) == {"hk": 0.0}


def test_window_environment_pool_reads_the_journalled_pool():
    from reliquary.validator.fill_closed_recovery import window_environment_pool

    record = {"window_pool": 0.5, "environments": ["math", "code"], "picks_target": 4}

    assert window_environment_pool(record) == 0.5 / 2 / 4


def test_window_environment_pool_defaults_a_pre_upgrade_journal_to_the_whole_pool():
    from reliquary.validator.fill_closed_recovery import window_environment_pool

    record = {"environments": ["math", "code"], "picks_target": 4}

    assert window_environment_pool(record) == 1.0 / 2 / 4


def test_window_environment_pool_pays_nobody_for_a_journalled_zero_share():
    from reliquary.validator.fill_closed_recovery import window_environment_pool

    record = {"window_pool": 0.0, "environments": ["math", "code"], "picks_target": 4}

    assert window_environment_pool(record) == 0.0


def test_begin_round_trips_the_declared_pool(tmp_path):
    from reliquary.validator.fill_closed_recovery import FillClosedRecoveryStore

    store = FillClosedRecoveryStore(tmp_path)
    store.begin(1, checkpoint_n=1, revision="a" * 40, targets={"math": 1}, window_pool=0.5)

    assert store.load(1)["window_pool"] == 0.5


@pytest.mark.parametrize("bad", [1.5, -0.1, True])
def test_begin_refuses_an_out_of_range_pool(tmp_path, bad):
    from reliquary.validator.fill_closed_recovery import FillClosedRecoveryStore

    store = FillClosedRecoveryStore(tmp_path)

    with pytest.raises(ValueError):
        store.begin(1, checkpoint_n=1, revision="a" * 40, targets={"math": 1}, window_pool=bad)
