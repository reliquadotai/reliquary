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


def test_no_registry_at_all_starts_the_legacy_task_at_the_full_pool():
    """The fallback that keeps `default` running the day this ships: a
    wholly absent registry (today's reality for every validator) is not a
    refusal, only a present-but-wrong one is."""
    from reliquary.validator.emission_price import PRODUCTION_PRICE_PARAMS
    from reliquary.validator.task_config import legacy_task_config

    config = legacy_task_config()

    assert config.task_id == "default"
    assert config.entry is None
    assert config.emission_cap == 1.0
    assert config.price_params == PRODUCTION_PRICE_PARAMS


# --- Crash-recovery payment coverage, restored from the deleted
# test_task_emission_share.py and adapted to the registry: `window_pool` is
# now a value threaded from the registry through ValidationService and
# FillClosedRecoveryStore.begin(), rather than a process-global constant, so
# these exercise the pool argument directly instead of the env var. ---

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


def test_a_zero_share_pays_nobody():
    from reliquary.validator.token_rewards import AcceptedGroup, split_environment_pool

    groups = [AcceptedGroup(hotkey="hk", operator_id="hk", eos_tokens=10)]

    assert split_environment_pool(groups, pool=0.0) == {"hk": 0.0}


def test_recovered_archive_carries_the_pool_the_window_actually_opened_with(
    tmp_path, monkeypatch,
):
    """The one test that would fail if fill_closed_recovery.py's archive
    construction hardcoded 1.0 (or misspelt the key) instead of reading back
    what begin() journalled: everything else in this file exercises the
    payout math or the store in isolation, not the archived field itself."""
    import reliquary.infrastructure.training_payload_queue as queue_module
    import reliquary.validator.fill_closed_recovery as recovery_module
    from reliquary.infrastructure.archive_queue import ArchiveQueue
    from reliquary.infrastructure.training_payload_queue import TrainingPayloadQueue
    from reliquary.validator.fill_closed_recovery import FillClosedRecoveryStore
    from reliquary.validator.fill_closed_rotation import FillClosedRotationStore

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(queue_module, "FILL_CLOSED_EMISSIONS_PER_WINDOW", 16)
    monkeypatch.setattr(recovery_module, "FILL_CLOSED_EMISSIONS_PER_WINDOW", 16)
    monkeypatch.setattr(recovery_module, "FILL_CLOSED_PICKS_PER_WINDOW", 16)

    store = FillClosedRecoveryStore(tmp_path)
    store.begin(42, checkpoint_n=7, revision="a" * 40,
                targets={"math": 16, "code": 16}, window_pool=0.4)
    queue = TrainingPayloadQueue(str(tmp_path / "payloads"))
    archives = ArchiveQueue(str(tmp_path / "archives"))
    rotation = FillClosedRotationStore(tmp_path)

    rows = [{"env_name": env, "batch_index": 0, "hotkey": "alice",
             "prompt_idx": 1, "eos_tokens": 16, "claimed_checkpoint_hash": "a" * 40}
            for env in ("math", "code")]
    queue.enqueue_committed_tombstone(672, b"training-quarantine", accounting=rows)

    store.recover(42, queue=queue, archives=archives, rotation=rotation)
    archive = archives.pending_archives(start_window=42, end_window=42)[42]

    assert archive["task_emission_share"] == 0.4
