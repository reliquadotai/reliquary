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


def _contract(profile_id="order-77"):
    from reliquary.protocol.profiles import PROFILES

    contract = PROFILES["qwen35-4b-reliquary-suite-v10-gumbel-dev1"].to_generation_contract()
    return {**contract, "profile_id": profile_id}


def test_a_carried_contract_is_written_and_named(tmp_path):
    seen = []
    contract = _contract()
    task = {"profile": {"profile_id": "order-77", "contract": contract}}
    with pytest.raises(Restarted):
        adopt_task_profile(task, {}, active="qwen35-2b-auction-v2", state_dir=tmp_path, reexec=restart_into(seen))
    path = seen[0]["RELIQUARY_TASK_CONTRACT"]
    assert json.loads(open(path).read()) == contract and "RELIQUARY_PROTOCOL_PROFILE" not in seen[0]


def test_nothing_happens_once_the_profiles_agree(tmp_path):
    task = {"profile": {"profile_id": "qwen35-2b-auction-v2", "contract": None}}
    adopt_task_profile(task, {"RELIQUARY_PROTOCOL_PROFILE": "qwen35-2b-auction-v2"}, active="qwen35-2b-auction-v2",
                       state_dir=tmp_path, reexec=restart_into([]))
    adopt_task_profile(task, {}, active="qwen35-2b-auction-v2", state_dir=tmp_path,
                       reexec=restart_into([]))                                        # the default is it


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
    task = {"profile": {"profile_id": "qwen35-4b-reliquary-suite-v10-gumbel-dev1", "contract": None}}
    with pytest.raises(SystemExit, match="RELIQUARY_PROTOCOL_PROFILE"):
        adopt_task_profile(task, {}, active="q", state_dir=tmp_path, reexec=None)


@pytest.mark.parametrize("wanted", [
    {"profile_id": "no-such-profile", "contract": None},                    # not in this trainer's catalogue
    {"profile_id": "order-77", "contract": {"profile_id": "order-77"}},     # a contract that does not parse
    {"profile_id": "order-77", "contract": _contract("order-78")},          # a contract under another id
])
def test_what_the_controller_sends_is_checked_before_any_restart(tmp_path, wanted):
    seen = []
    with pytest.raises(SystemExit, match="controller"):
        adopt_task_profile({"profile": wanted}, {}, active="qwen35-2b-auction-v2", state_dir=tmp_path,
                           reexec=restart_into(seen))
    assert seen == [] and not list(tmp_path.iterdir())


def test_the_restart_keeps_the_interpreter_flags_and_the_module_form(monkeypatch):
    """``python -u -m reliquary.cli.main train-async`` restarts as itself (``sys.orig_argv``), output flushed."""
    import os
    import sys

    from reliquary.trainer import async_cli

    calls, flushed = [], []
    monkeypatch.setattr(sys, "orig_argv", ["/usr/bin/python3", "-u", "-m", "reliquary.cli.main", "train-async"])
    monkeypatch.setattr(sys, "argv", ["/x/reliquary/cli/main.py", "train-async"])
    monkeypatch.setattr(sys.stdout, "flush", lambda: flushed.append("out"))
    monkeypatch.setattr(sys.stderr, "flush", lambda: flushed.append("err"))
    monkeypatch.setattr(os, "execve", lambda path, args, env: calls.append((path, args, env)))
    async_cli._restart({"A": "1"})
    assert calls == [(sys.executable, [sys.executable, "-u", "-m", "reliquary.cli.main", "train-async"], {"A": "1"})]
    assert flushed == ["out", "err"]
