"""A rented trainer pod gets no RELIQUARY_PROTOCOL_PROFILE: it takes the profile (or the carried contract) its
controller runs from ``/rl/trainer/config`` and starts again under it, before any model work; an operator's
explicit profile must match the controller's."""
import json

import pytest

from reliquary.trainer.async_cli import adopt_task_profile


class Restarted(Exception):
    pass


def restart_into(seen):
    def reexec(env):
        seen.append(env)
        raise Restarted

    return reexec


def test_a_pod_without_a_profile_restarts_under_the_controllers(tmp_path):
    seen = []
    task = {"profile": {"profile_id": "qwen35-4b-reliquary-suite-v10-gumbel-dev1", "contract": None}}
    with pytest.raises(Restarted):
        adopt_task_profile(task, {"PATH": "/bin"}, active="qwen35-2b-auction-v2", state_dir=tmp_path,
                           reexec=restart_into(seen))
    assert seen == [{"PATH": "/bin", "RELIQUARY_PROTOCOL_PROFILE": "qwen35-4b-reliquary-suite-v10-gumbel-dev1",
                     "RELIQUARY_RL_TRAINER_PROFILE_ADOPTED": "1"}]


def test_a_carried_contract_is_written_and_named(tmp_path):
    seen = []
    contract = {"profile_id": "order-77", "model_id": "Qwen/Qwen3.5-4B"}
    task = {"profile": {"profile_id": "order-77", "contract": contract}}
    with pytest.raises(Restarted):
        adopt_task_profile(task, {}, active="qwen35-2b-auction-v2", state_dir=tmp_path, reexec=restart_into(seen))
    path = seen[0]["RELIQUARY_TASK_CONTRACT"]
    assert json.loads(open(path).read()) == contract and "RELIQUARY_PROTOCOL_PROFILE" not in seen[0]


def test_nothing_happens_once_the_profiles_agree(tmp_path):
    task = {"profile": {"profile_id": "p", "contract": None}}
    adopt_task_profile(task, {"RELIQUARY_PROTOCOL_PROFILE": "p"}, active="p", state_dir=tmp_path,
                       reexec=restart_into([]))
    adopt_task_profile(task, {}, active="p", state_dir=tmp_path, reexec=restart_into([]))   # the default is it


def test_an_explicit_profile_that_differs_is_refused(tmp_path):
    task = {"profile": {"profile_id": "p", "contract": None}}
    with pytest.raises(SystemExit, match="'q'.*'p'"):
        adopt_task_profile(task, {"RELIQUARY_PROTOCOL_PROFILE": "q"}, active="q", state_dir=tmp_path,
                           reexec=restart_into([]))


def test_a_restart_that_did_not_take_is_refused_rather_than_looping(tmp_path):
    task = {"profile": {"profile_id": "p", "contract": None}}
    with pytest.raises(SystemExit):
        adopt_task_profile(task, {"RELIQUARY_RL_TRAINER_PROFILE_ADOPTED": "1"}, active="q", state_dir=tmp_path,
                           reexec=restart_into([]))


def test_an_older_controller_without_a_profile_keeps_the_local_one(tmp_path, caplog):
    import logging

    caplog.set_level(logging.WARNING, logger="reliquary.trainer.async_cli")
    adopt_task_profile({"task_id": 1}, {}, active="q", state_dir=tmp_path, reexec=restart_into([]))
    assert any("profile" in r.getMessage() for r in caplog.records)


def test_ranks_cannot_restart(tmp_path):
    task = {"profile": {"profile_id": "p", "contract": None}}
    with pytest.raises(SystemExit, match="RELIQUARY_PROTOCOL_PROFILE"):
        adopt_task_profile(task, {}, active="q", state_dir=tmp_path, reexec=None)
