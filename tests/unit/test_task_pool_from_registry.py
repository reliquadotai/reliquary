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
    "median_rounds": 4800, "last_good_fills": 50,
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


def test_the_production_service_is_handed_the_per_environment_caps():
    """The window pool is per-environment only if the call site says so.

    Task 4 gave ValidationService an ``env_caps`` parameter and the assembler
    a per-environment pool; neither does anything unless the production
    construction actually passes it. This asserts the wire, not the plumbing.
    """
    import ast
    import pathlib

    source = pathlib.Path("reliquary/cli/main.py").read_text()
    tree = ast.parse(source)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ValidationService"
    ]

    assert calls, "no ValidationService construction found in cli/main.py"
    for call in calls:
        passed = {kw.arg for kw in call.keywords}
        assert "env_caps" in passed, (
            "the production ValidationService call must pass env_caps, or the "
            "per-environment window pool is dead code"
        )
        assert "emission_cap" in passed
        assert "price_params" in passed


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

def _recovery_scaffold(tmp_path, monkeypatch, picks=16):
    """A 16-slot, two-environment window ready to be crash-recovered."""
    import reliquary.infrastructure.training_payload_queue as queue_module
    import reliquary.validator.fill_closed_recovery as recovery_module
    from reliquary.infrastructure.archive_queue import ArchiveQueue
    from reliquary.infrastructure.training_payload_queue import TrainingPayloadQueue
    from reliquary.validator.fill_closed_recovery import FillClosedRecoveryStore
    from reliquary.validator.fill_closed_rotation import FillClosedRotationStore

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(queue_module, "FILL_CLOSED_EMISSIONS_PER_WINDOW", 16)
    monkeypatch.setattr(recovery_module, "B_BATCH", 16)
    monkeypatch.setattr(recovery_module, "FILL_CLOSED_EMISSIONS_PER_WINDOW", 16)
    monkeypatch.setattr(recovery_module, "FILL_CLOSED_PICKS_PER_WINDOW", picks)
    return (
        FillClosedRecoveryStore(tmp_path),
        TrainingPayloadQueue(str(tmp_path / "payloads")),
        ArchiveQueue(str(tmp_path / "archives")),
        FillClosedRotationStore(tmp_path),
    )


def _recover_one_paid_batch(store, queue, archives, rotation):
    """One paid batch for `alice` in both environments, then recovery."""
    rows = [{"env_name": env, "batch_index": 0, "hotkey": "alice",
             "prompt_idx": 1, "eos_tokens": 16, "claimed_checkpoint_hash": "a" * 40}
            for env in ("math", "code")]
    queue.enqueue_committed_tombstone(42 * 16, b"training-quarantine", accounting=rows)
    store.recover(42, queue=queue, archives=archives, rotation=rotation)
    return archives.pending_archives(start_window=42, end_window=42)[42]


def test_recovery_pays_from_the_journalled_pool(tmp_path, monkeypatch):
    """Recovery divides the pool the window OPENED with -- not a default --
    by environment and by pick, then by the fixed payment slots."""
    store, queue, archives, rotation = _recovery_scaffold(tmp_path, monkeypatch)
    store.begin(42, checkpoint_n=7, revision="a" * 40,
                targets={"math": 16, "code": 16}, window_pool=0.5)

    archive = _recover_one_paid_batch(store, queue, archives, rotation)

    assert archive["window_pool"] == 0.5
    # 0.5 / 2 environments / 16 picks, then one of 16 fixed slots, in each of
    # the two environments alice was paid in.
    assert archive["rewards_by_hotkey"] == {"alice": 2 * (0.5 / 2 / 16) / 16}


def test_recovery_defaults_a_pre_upgrade_journal_to_the_whole_pool(
    tmp_path, monkeypatch,
):
    """A schema_version 1 journal predates `window_pool` entirely; recovery
    must read it back as the whole pool rather than paying nothing."""
    from reliquary.shared.training_payload import active_training_identity
    from reliquary.validator.control import write_json

    store, queue, archives, rotation = _recovery_scaffold(tmp_path, monkeypatch)
    write_json(store._path(42), {
        "schema_version": 1, "window_start": 42,
        "identity": active_training_identity(), "parent_checkpoint_n": 7,
        "parent_revision": "a" * 40, "environments": ["math", "code"],
        "batch_targets": {"math": 16, "code": 16}, "archive": None,
        "picks_target": 16,
    })

    archive = _recover_one_paid_batch(store, queue, archives, rotation)

    assert archive["window_pool"] == 1.0
    # Pre-upgrade journals also predate the fixed-slot policy, so the whole
    # environment pool goes to the only paid hotkey: 1.0 / 2 / 16, twice.
    assert archive["rewards_by_hotkey"] == {"alice": 2 * (1.0 / 2 / 16)}


def test_activating_a_window_journals_the_pool_it_opened_with(tmp_path, monkeypatch):
    """The leg no other test covers: ValidationService must hand its own
    emission cap to recovery.begin(). Dropping that keyword is silent -- the
    journal simply falls back to 1.0 and a recovered window pays the whole
    pool instead of this task's share."""
    import types

    import reliquary.validator.service as service_mod
    from reliquary.validator.service import ValidationService

    store, _queue, _archives, _rotation = _recovery_scaffold(tmp_path, monkeypatch)
    monkeypatch.setattr(service_mod, "FILL_CLOSED_ENABLED", True)
    batcher = types.SimpleNamespace(
        window_start=42, mark_window_opened=lambda: None,
        bind_event_loop=lambda loop: None, current_checkpoint_hash="a" * 40,
    )
    service = types.SimpleNamespace(
        _active_batchers={"math": batcher}, _candidate_window_n=42,
        _set_window_preparation_stage=lambda stage: None,
        _candidate_fill_closed_assembler=types.SimpleNamespace(window_start=42),
        _fill_closed_assembler=None, _fill_closed_assemblers={},
        _fill_closed_recovery_store=store,
        _checkpoint_store=types.SimpleNamespace(
            current_manifest=lambda: types.SimpleNamespace(
                revision="a" * 40, checkpoint_n=7)),
        env_mix=[("math", 16), ("code", 16)], _emission_cap=0.37,
        _window_n=None, _candidate_activation_nonce=None,
        _window_preparation_stage=None,
        server=types.SimpleNamespace(
            set_active_batchers=lambda batchers: None,
            clear_window_preparation_failure=lambda: None),
        _publish_window_preparation_state=lambda: None,
        _set_state=lambda state: None,
    )

    ValidationService._activate_window(service)

    assert store.load(42)["window_pool"] == 0.37


def test_recovery_pays_nobody_for_a_journalled_zero_share(tmp_path, monkeypatch):
    store, queue, archives, rotation = _recovery_scaffold(tmp_path, monkeypatch)
    store.begin(42, checkpoint_n=7, revision="a" * 40,
                targets={"math": 16, "code": 16}, window_pool=0.0)

    archive = _recover_one_paid_batch(store, queue, archives, rotation)

    assert archive["window_pool"] == 0.0
    assert sum(archive["rewards_by_hotkey"].values()) == 0.0


def test_begin_round_trips_the_declared_pool(tmp_path, monkeypatch):
    import reliquary.validator.fill_closed_recovery as recovery_module
    from reliquary.validator.fill_closed_recovery import FillClosedRecoveryStore

    monkeypatch.setattr(recovery_module, "B_BATCH", 1)
    store = FillClosedRecoveryStore(tmp_path)
    store.begin(1, checkpoint_n=1, revision="a" * 40, targets={"math": 1}, window_pool=0.5)

    assert store.load(1)["window_pool"] == 0.5


@pytest.mark.parametrize("bad", [1.5, -0.1, True])
def test_begin_refuses_an_out_of_range_pool(tmp_path, monkeypatch, bad):
    import reliquary.validator.fill_closed_recovery as recovery_module
    from reliquary.validator.fill_closed_recovery import FillClosedRecoveryStore

    monkeypatch.setattr(recovery_module, "B_BATCH", 1)
    store = FillClosedRecoveryStore(tmp_path)

    with pytest.raises(ValueError, match="pool"):
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
    monkeypatch.setattr(recovery_module, "B_BATCH", 16)
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
