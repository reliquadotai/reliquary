"""The async trainer's audit and replay knobs, read from the environment (``async_trainer_settings``)."""
from pathlib import Path

import pytest

from reliquary.trainer import async_cli
from reliquary.trainer.miles_gateway import miles_settings

ENV = {"RELIQUARY_RL_TRAINER_URL": "http://c", "RELIQUARY_RL_TRAINER_TOKEN": "t",
       "RELIQUARY_RL_VALIDATOR_HOTKEY": "5Hot"}
MILES = {"RELIQUARY_MILES_LAUNCHER": "/l.py", "RELIQUARY_MILES_MODEL_TYPE": "qwen3.5-4B",
         "RELIQUARY_MILES_OUTPUT_DIR": "/o"}


def test_defaults_bound_the_audits_by_a_share_of_training_and_dump_nothing():
    settings = async_cli.async_trainer_settings(ENV)
    assert settings["audit_budget_s"] is None                  # no fixed budget: the share rule decides
    assert (settings["audit_min_s"], settings["audit_share"]) == (60.0, 0.25)
    assert settings["audit_rollouts_per_call"] == 64
    assert settings["dump_dir"] is None


def test_the_old_fixed_budget_still_overrides_and_the_rest_is_read():
    settings = async_cli.async_trainer_settings({
        **ENV, "RELIQUARY_RL_TRAINER_AUDIT_BUDGET_S": "420", "RELIQUARY_RL_TRAINER_AUDIT_MIN_S": "30",
        "RELIQUARY_RL_TRAINER_AUDIT_SHARE": "0.5", "RELIQUARY_RL_TRAINER_AUDIT_ROLLOUTS_PER_CALL": "128",
        "RELIQUARY_RL_TRAINER_DUMP_DIR": "/data/dumps"})
    assert settings["audit_budget_s"] == 420.0
    assert (settings["audit_min_s"], settings["audit_share"]) == (30.0, 0.5)
    assert settings["audit_rollouts_per_call"] == 128 and settings["dump_dir"] == Path("/data/dumps")


@pytest.mark.parametrize("name,value", [("RELIQUARY_RL_TRAINER_AUDIT_SHARE", "-1"),
                                        ("RELIQUARY_RL_TRAINER_AUDIT_MIN_S", "-5"),
                                        ("RELIQUARY_RL_TRAINER_AUDIT_BUDGET_S", "-1"),
                                        ("RELIQUARY_RL_TRAINER_AUDIT_ROLLOUTS_PER_CALL", "0")])
def test_negative_audit_knobs_are_refused(name, value):
    with pytest.raises(ValueError, match=name):
        async_cli.async_trainer_settings({**ENV, name: value})


def test_the_miles_chunk_is_a_setting():
    assert miles_settings(MILES)["chunk"] == 64
    assert miles_settings({**MILES, "RELIQUARY_MILES_CHUNK": "1024"})["chunk"] == 1024
    with pytest.raises(ValueError):
        miles_settings({**MILES, "RELIQUARY_MILES_CHUNK": "0"})
