"""What a window pays comes from the registry, and `default` is unchanged."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from reliquary.environment.abi import canonical_sha256
from reliquary.shared.task_registry import MECHANISM_RL_DISCOVERED_PRICE, TaskEntry
from reliquary.validator.task_config import resolve_task_config

PARAMS = {
    "start": 1.0, "decay": 0.99, "rounds_per_step": 1000,
    "deadband": 0.80, "snap": 1.20, "floor": 0.05, "cap": 1.0,
    "median_rounds": 4800,
}


def test_the_emission_share_env_var_is_gone():
    clean = {k: v for k, v in os.environ.items() if not k.startswith("RELIQUARY_")}
    clean["RELIQUARY_TASK_EMISSION_SHARE"] = "0.25"
    completed = subprocess.run(
        [sys.executable, "-c",
         "import reliquary.constants as c; print(hasattr(c, 'TASK_EMISSION_SHARE'))"],
        capture_output=True, text=True, env=clean,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"


def test_the_legacy_task_still_pays_the_whole_pool():
    from reliquary.constants import PROTOCOL_GENERATION_CONTRACT, PROTOCOL_PROFILE_ID

    entry = TaskEntry(
        task_id="default",
        profile_id=PROTOCOL_PROFILE_ID,
        profile_sha256=canonical_sha256(PROTOCOL_GENERATION_CONTRACT),
        mechanism=MECHANISM_RL_DISCOVERED_PRICE,
        params=dict(PARAMS),
        status="active",
        retired_at=None,
    )

    config = resolve_task_config(
        {"default": entry}, "default",
        profile_id=PROTOCOL_PROFILE_ID,
        generation_contract=PROTOCOL_GENERATION_CONTRACT,
    )

    assert config.emission_cap == 1.0


def test_no_module_still_reads_the_removed_constant():
    """The env-var path is gone, not merely unused."""
    import pathlib
    import subprocess

    root = pathlib.Path(__file__).resolve().parents[2] / "reliquary"
    hits = subprocess.run(
        ["grep", "-rn", "TASK_EMISSION_SHARE", str(root)],
        capture_output=True, text=True,
    ).stdout.strip()

    assert hits == "", f"TASK_EMISSION_SHARE still referenced:\n{hits}"
