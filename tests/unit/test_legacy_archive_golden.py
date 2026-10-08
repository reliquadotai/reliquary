# tests/unit/test_legacy_archive_golden.py
"""A legacy (non-service) fill-closed window archives exactly what it archived before the
service-contract/v2 wiring: same keys, same values, no ``service_`` field.

The golden file was produced by this very test body, with ``RELIQUARY_WRITE_GOLDEN=<path>``, on
commit adce8966: the merge base of this branch with ``main``, where ``validator/service.py`` has no
service code path at all (an export of that tree, ``git archive``, with this file copied in). The
same bytes come out of 820c3d2b, the commit phase 1 of the next RL run started from.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import reliquary.validator.service as service_module
from reliquary.validator.fill_closed_recovery import FillClosedRecoveryStore
from tests.unit.test_archive_window_content import _valid_submission
from tests.unit.test_service_v2 import _build_late_drop_service

GOLDEN = Path(__file__).parent / "data" / "legacy_fill_closed_archive_adce8966.json"
ROOT = "d" * 40
VOLATILE = {"window_opened_wall_ts_by_environment"}                     # the wall clock at activation


def _stable(value, key=None):
    """The archive with its only run-dependent values named instead of copied.

    Those are the wall clock of the activation and the ``repr`` of the fake model's attributes
    (it carries an object id). Everything else is compared as it is.
    """
    if key in VOLATILE:
        return {name: type(item).__name__ for name, item in sorted(value.items())}
    if isinstance(value, dict):
        return {name: _stable(item, name) for name, item in value.items()}
    if isinstance(value, list):
        return [_stable(item) for item in value]
    if isinstance(value, str) and value.startswith("<MagicMock "):
        return "<MagicMock>"
    return value


async def _legacy_archive(monkeypatch, tmp_path) -> dict:
    """One legacy fill-closed window through the real open, activation, seal and archive."""
    import reliquary.infrastructure.training_payload_queue as queue_module
    from reliquary.infrastructure.archive_queue import ArchiveQueue
    from reliquary.infrastructure.training_payload_queue import TrainingPayloadQueue
    from reliquary.validator.fill_closed_rotation import FillClosedRotationStore

    monkeypatch.setattr(service_module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr("reliquary.constants.EMISSION_PRICE_ARMED", False)
    monkeypatch.setattr("reliquary.constants.WRITE_TRAINING_PAYLOADS", True)
    monkeypatch.setattr("reliquary.constants.DETACHED_TRAINER", True)
    monkeypatch.setenv("RELIQUARY_STATE_DIR", str(tmp_path / "state"))
    svc = _build_late_drop_service()
    svc._derive_randomness = AsyncMock(return_value=("drand-material", None))
    svc._checkpoint_store = SimpleNamespace(current_manifest=lambda: SimpleNamespace(
        repo_id="models/test", revision=ROOT, checkpoint_n=0))
    svc._fill_closed_recovery_store = FillClosedRecoveryStore(tmp_path / "state")
    svc._fill_closed_rotation_store = FillClosedRotationStore(tmp_path / "state")
    svc._training_payload_queue = TrainingPayloadQueue(str(tmp_path / "payloads"))
    archives = ArchiveQueue(str(tmp_path / "archives"))
    monkeypatch.setattr("reliquary.infrastructure.archive_queue.get_archive_queue", lambda: archives)
    svc._open_window()
    await svc._set_window_randomness(subtensor=None)
    svc._activate_window()
    window = svc._window_n
    env_name = next(iter(svc._active_batchers))
    group = dataclasses.replace(
        _valid_submission(prompt_idx=3, hotkey="alice", eos_first=True, eos_tokens=5), claimed_checkpoint_hash=ROOT)
    svc._fill_closed_assembler.accept(env_name, [group], window, ROOT)
    batchers = dict(svc._active_batchers)
    for batcher in batchers.values():
        batcher.force_seal("unit")
    svc._close_and_commit_fill_closed_paid_side_effects(batchers, svc._fill_closed_assembler)
    await svc._archive_window(batchers, {name: ([], {}) for name in batchers})
    return archives.pending_archives(start_window=window, end_window=window)[window]


@pytest.mark.asyncio
async def test_a_legacy_archive_is_what_it_was_before_the_service_wiring(monkeypatch, tmp_path):
    archive = await _legacy_archive(monkeypatch, tmp_path)
    assert not [key for key in archive if key.startswith("service_")]
    assert archive["rewards_by_hotkey"] and len(archive["batch"]) == 1     # a paid window, not an empty shell
    stable = json.dumps(_stable(archive), sort_keys=True, indent=1) + "\n"
    target = os.environ.get("RELIQUARY_WRITE_GOLDEN")
    if target:
        Path(target).write_text(stable)
    golden = GOLDEN.read_text()
    assert set(json.loads(stable)) == set(json.loads(golden))               # the key set, named on failure
    assert stable == golden                                                 # and every byte of it
