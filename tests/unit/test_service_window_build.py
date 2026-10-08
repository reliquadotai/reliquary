# tests/unit/test_service_window_build.py
"""A service-contract/v2 task in the validator: schedule-driven windows over several envs.

Every test drives the REAL ``ValidationService`` methods and the REAL ``ServiceRuntime``
(one SQLite file per test); only the model, the chain and the env datasets are fakes.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import reliquary.validator.service as service_module
from reliquary.constants import B_BATCH, FILL_CLOSED_ADMISSION_BUDGET_PER_ENV
from reliquary.protocol.service_schedule import next_schedule
from reliquary.services.runtime import FROZEN_ARCHIVE_FIELDS, ServiceRuntime, protocol_slot_geometry
from reliquary.services.schedule import ScheduleRequestStore
from reliquary.services.settlement import SettlementError
from reliquary.validator.cooldown import ContentCooldownMap, CooldownMap
from reliquary.validator.fill_closed_recovery import FillClosedRecoveryStore
from tests.unit.service_v2_fixtures import CODE, MATH, SCIENCE, contract_v2, contract_v2_dict, qualification_v2
from tests.unit.test_archive_window_content import _valid_submission
from tests.unit.test_grpo_window_batcher import FakeEnv
from tests.unit.test_service_v2 import _build_late_drop_service

PICKS, SLOTS = protocol_slot_geometry()
ROOT = "d" * 40
BEACON = "ab" * 32
CAP = 0.5


class _Env(FakeEnv):
    def __init__(self, name, rows=1000):
        self.name = name
        self._rows = rows

    def __len__(self):
        return self._rows


def drand_round(instant: float) -> int:
    return int(instant) // 3


def _runtime(folder, contract):
    return ServiceRuntime(folder / "runtime.sqlite3", contract, qualification_v2(contract), now=0,
                          drand_round_at=drand_round)


def _service(monkeypatch, tmp_path, contract=None, *, loaded=None, cap=CAP, revision=ROOT, checkpoint_n=0):
    """A real ValidationService wired the way ``__init__`` wires a service task."""
    monkeypatch.setattr(service_module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr("reliquary.constants.EMISSION_PRICE_ARMED", False)
    contract = contract or contract_v2()
    folder = tmp_path / "service"
    folder.mkdir(mode=0o700, exist_ok=True)
    folder.chmod(0o700)   # the request folder must not be group/other-writable whatever the umask
    svc = _build_late_drop_service()
    names = list(loaded if loaded is not None else
                 [name for name in (MATH, CODE, SCIENCE) if name in contract.environments])
    svc.envs = {name: _Env(name) for name in names}
    svc.env_mix = [(name, B_BATCH) for name in names]
    svc.env_targets = dict(svc.env_mix)
    svc.env = svc.envs[names[0]]
    svc._cooldown_per_env = {name: CooldownMap(cooldown_windows=50) for name in names}
    svc._content_cooldown_per_env = {name: ContentCooldownMap(cooldown_windows=50) for name in names}
    svc._cooldown_map = svc._cooldown_per_env[names[0]]
    svc._emission_cap = cap
    svc._env_caps = {}
    svc._service_runtime = _runtime(folder, contract)
    svc._service_schedule_store = ScheduleRequestStore(folder)
    svc._service_installed_version = lambda name: contract.environments[name]["version"]
    svc._checkpoint_store = SimpleNamespace(current_manifest=lambda: SimpleNamespace(
        repo_id="models/test", revision=revision, checkpoint_n=checkpoint_n))
    svc._fill_closed_recovery_store = FillClosedRecoveryStore(tmp_path / "state")
    svc._derive_randomness = AsyncMock(return_value=("drand-material", None))
    return svc


async def _open(svc, *, activate=True):
    """One pass of the main loop's open sequence, as ``run`` performs it."""
    await svc._prepare_service_window()
    svc._open_window()
    await svc._set_window_randomness(subtensor=None)
    if activate:
        svc._activate_window()
    return svc._active_batchers


def _archived(svc, window):
    """What the end of ``_archive_window`` leaves behind for a window whose archive is enqueued."""
    svc._fill_closed_recovery_store._path(window).unlink()
    svc._fill_closed_assemblers.pop(window, None)
    svc._fill_closed_assembler = None
    svc._active_batchers = {}


def _journal(svc, monkeypatch, tmp_path):
    """Real trainer journal, archive queue and rotation store under ``tmp_path``."""
    import reliquary.infrastructure.training_payload_queue as queue_module
    from reliquary.infrastructure.archive_queue import ArchiveQueue
    from reliquary.infrastructure.training_payload_queue import TrainingPayloadQueue
    from reliquary.validator.fill_closed_rotation import FillClosedRotationStore
    from reliquary.validator.training_accumulator import BalancedTrainingAccumulator

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr("reliquary.constants.WRITE_TRAINING_PAYLOADS", True)
    monkeypatch.setattr("reliquary.constants.DETACHED_TRAINER", True)
    monkeypatch.setenv("RELIQUARY_STATE_DIR", str(tmp_path / "state"))
    svc._training_payload_queue = TrainingPayloadQueue(str(tmp_path / "payloads"))
    archives = ArchiveQueue(str(tmp_path / "archives"))
    monkeypatch.setattr("reliquary.infrastructure.archive_queue.get_archive_queue", lambda: archives)
    monkeypatch.setattr(archives, "run_forever", AsyncMock())
    svc._fill_closed_rotation_store = FillClosedRotationStore(tmp_path / "state")
    svc._training_accumulator = BalancedTrainingAccumulator(dict(svc.env_mix))
    svc._fetch_seal_randomness = AsyncMock(return_value="")
    svc._detached_checkpoint_tick = AsyncMock()
    return archives


def _paid(hotkey, prompt):
    """A proven group as the assembler pays it, generated on the window's checkpoint."""
    return dataclasses.replace(_valid_submission(prompt_idx=prompt, hotkey=hotkey, eos_first=True, eos_tokens=5),
                               claimed_checkpoint_hash=ROOT)


def _pay_one_group_per_env(svc, window=1):
    svc._fill_closed_assembler.accept(MATH, [_paid("alice", 3)], window, ROOT)
    svc._fill_closed_assembler.accept(CODE, [_paid("bob", 4)], window, ROOT)


def _seal(svc):
    for batcher in svc._active_batchers.values():
        batcher.force_seal("unit")


def _money(archive):
    return json.dumps({key: archive.get(key) for key in FROZEN_ARCHIVE_FIELDS}, sort_keys=True)


async def _run(svc, monkeypatch, *, windows, mid_window=None):
    """The REAL ``ValidationService.run`` loop until ``windows`` windows are behind it.

    Stubbed: chain, R2 history, drand waits, the proof plane and the seal wait (it seals at once).
    ``mid_window(n)`` runs while window ``n`` is open (admission exposed), just before it seals.
    """
    monkeypatch.setattr(service_module, "POLL_INTERVAL_SECONDS", 0)
    for name in ("_refresh_registered_hotkeys", "_apply_resume_from", "_bootstrap_state_from_external",
                 "_ensure_proof_scheduler_ready", "_rebuild_cooldown_from_history", "_restore_content_cooldown",
                 "_rebuild_hashes_from_history", "_restore_price_walk", "_serve_axon_on_chain",
                 "_wait_for_next_drand_boundary", "_close_proof_scheduler", "_snapshot_committed_cooldowns"):
        monkeypatch.setattr(svc, name, AsyncMock())
    monkeypatch.setattr(svc.server, "start", AsyncMock())
    monkeypatch.setattr(svc.server, "stop", AsyncMock())
    monkeypatch.setattr(svc.server, "prepare_admission_pools", AsyncMock())
    monkeypatch.setattr(svc, "_log_startup_config_banner", MagicMock())
    monkeypatch.setattr(svc, "_arm_fill_closed_rotation_gate", MagicMock())
    monkeypatch.setattr(svc, "_wait_for_fill_closed_rotation", AsyncMock(return_value="not_armed"))
    if svc._service_runtime is not None:
        monkeypatch.setattr(svc._service_runtime, "close", MagicMock())   # the tests read it afterwards
    iterations, stalled = [], []

    async def top_of_loop():
        iterations.append(svc._window_n)
        if svc._window_n >= windows:
            raise asyncio.CancelledError
        if len(iterations) > 3 * windows + 3:                        # never spin: stop, then fail below
            stalled.append(True)
            raise asyncio.CancelledError
        return False

    async def seal_wait(**kwargs):
        if mid_window is not None:
            mid_window(svc._window_n)
        _seal(svc)
        return "sealed"

    monkeypatch.setattr(svc, "_pause_for_control_drain", top_of_loop)
    monkeypatch.setattr(svc, "_wait_for_window_seal", seal_wait)
    with pytest.raises(asyncio.CancelledError):
        await svc.run(None)
    assert not stalled, f"the loop does not advance: {iterations}"
    return iterations


def _request(svc, **changes):
    runtime = svc._service_runtime
    return svc._service_schedule_store.submit(
        order_sha256=runtime.contract.sha256, revision=runtime.schedule.revision + 1, **changes)


def _row(hotkey, env, prompt):
    return {"hotkey": hotkey, "env_name": env, "prompt_idx": prompt}


def _archive(window, rows=(), status="completed"):
    return {"window_start": window, "window_status": status, "batch": list(rows), "rewards_by_hotkey": {}}


# ---------------------------------------------------------------- cooldown maps (R6)

@pytest.mark.parametrize("kind", [CooldownMap, ContentCooldownMap])
def test_a_horizon_change_is_a_fresh_map_with_the_same_history(kind):
    running = kind(cooldown_windows=10)
    key = 5 if kind is CooldownMap else "ab" * 32
    (running.record_batched if kind is CooldownMap else running.record_selected)(key, 100)
    changed = running.with_cooldown_windows(3)
    assert changed is not running and type(changed) is kind
    assert changed.cooldown_windows == 3 and running.cooldown_windows == 10
    assert changed.is_in_cooldown(key, 102) and not changed.is_in_cooldown(key, 103)
    assert running.is_in_cooldown(key, 109) and not running.is_in_cooldown(key, 110)   # untouched
    (changed.record_batched if kind is CooldownMap else changed.record_selected)(key, 200)
    assert running.export_state() == {key: 100}                                        # no shared state
    with pytest.raises(ValueError):
        running.with_cooldown_windows(-1)


# ---------------------------------------------------------------- window build

@pytest.mark.asyncio
async def test_two_active_envs_get_two_batchers_a_constant_group_count_and_cap_times_share(monkeypatch, tmp_path):
    svc = _service(monkeypatch, tmp_path, contract_v2(shares={MATH: 6000, CODE: 4000}))
    batchers = await _open(svc)
    assert list(batchers) == [MATH, CODE] and svc._window_env_mix == [(MATH, B_BATCH), (CODE, B_BATCH)]
    expected = {MATH: CAP * 6000 / 10000, CODE: CAP * 4000 / 10000}
    assembler = svc._fill_closed_assembler
    for name, batcher in batchers.items():
        # R9: the share moves the emission, never the group count per pick.
        assert batcher.batch_target == B_BATCH == SLOTS
        assert batcher.service_runtime is svc._service_runtime and batcher.service_environment == name
        assert batcher.env.name == name and batcher.prompt_range == (0, 1000)
        assert assembler.pool_for(name) == expected[name]                 # exactly, not approximately
    assert svc._candidate_service_pools == expected
    assert assembler.picks_target == PICKS
    fill = batchers[MATH].fill_state
    assert fill is batchers[CODE].fill_state
    assert fill.snapshot()["budgets"] == {MATH: FILL_CLOSED_ADMISSION_BUDGET_PER_ENV,
                                         CODE: FILL_CLOSED_ADMISSION_BUDGET_PER_ENV}
    envelope = svc._service_runtime.envelope(1)
    assert envelope["pools"] == expected and (envelope["picks_target"], envelope["batch_slots"]) == (PICKS, SLOTS)
    assert envelope["checkpoint"]["revision"] == ROOT
    record = svc._fill_closed_recovery_store.load(1)
    assert record["environments"] == [MATH, CODE] and record["batch_targets"] == {MATH: B_BATCH, CODE: B_BATCH}
    for batcher in batchers.values():
        policy = batcher.service_policy
        assert policy["pool_epoch"] == 1 and policy["schedule"] == envelope["schedule"]
        assert policy["pool_randomness"] == batcher.randomness
        assert policy["checkpoint"]["revision"] == batcher.current_checkpoint_hash == ROOT


@pytest.mark.asyncio
async def test_an_armed_price_scales_a_service_pool_below_cap_times_share(monkeypatch, tmp_path):
    from reliquary.validator.emission_price import PriceState
    svc = _service(monkeypatch, tmp_path, contract_v2(shares={MATH: 6000, CODE: 4000}))
    monkeypatch.setattr("reliquary.constants.EMISSION_PRICE_ARMED", True)
    svc._price_shadow_state = PriceState(price=0.5, last_good=0.5)
    await _open(svc)
    assert svc._candidate_service_pools == pytest.approx({MATH: CAP * 0.6 * 0.5, CODE: CAP * 0.4 * 0.5})
    assert svc._service_runtime.envelope(1)["pools"] == svc._candidate_service_pools


@pytest.mark.asyncio
async def test_a_deactivation_lands_at_the_next_window_and_the_running_window_still_settles(monkeypatch, tmp_path):
    svc = _service(monkeypatch, tmp_path)
    runtime = svc._service_runtime
    first = await _open(svc)
    running = dict(first)
    _request(svc, active=[MATH], shares={MATH: 10000})
    # Mid-window nothing reads the request: the schedule moves only at the boundary.
    assert runtime.schedule.revision == 0 and svc._service_schedule_store.status() is None
    assert set(svc._active_batchers) == {MATH, CODE} and svc._active_batchers[CODE] is running[CODE]
    # The running window settles its deactivated env from its frozen envelope.
    settled = await svc._settle_service_archive(runtime, _archive(1, [_row("alice", CODE, 3), _row("bob", MATH, 4)]))
    assert settled["rewards_by_hotkey"] == {"alice": pytest.approx(CAP * 0.5 / (PICKS * SLOTS)),
                                            "bob": pytest.approx(CAP * 0.5 / (PICKS * SLOTS))}
    assert set(settled["service_pools_by_environment"]) == {MATH, CODE}
    _archived(svc, 1)
    second = await _open(svc)
    assert runtime.schedule.revision == 1 and svc._service_schedule_store.status()["status"] == "applied"
    assert list(second) == [MATH] and svc._window_env_mix == [(MATH, B_BATCH)]
    assert second[MATH].batch_target == B_BATCH                      # still the constant group count
    assert svc._candidate_service_pools == {MATH: CAP * 10000 / 10000}
    assert second[MATH].fill_state.snapshot()["budgets"] == {MATH: FILL_CLOSED_ADMISSION_BUDGET_PER_ENV}
    assert runtime.envelope(2)["pools"] == {MATH: CAP} and set(runtime.envelope(1)["pools"]) == {MATH, CODE}
    assert svc._fill_closed_recovery_store.load(2)["environments"] == [MATH]


@pytest.mark.asyncio
async def test_an_env_activated_mid_window_gets_no_batcher_until_the_next_window(monkeypatch, tmp_path):
    contract = contract_v2(envs=(MATH, CODE, SCIENCE), shares={MATH: 5000, CODE: 5000, SCIENCE: 0})
    svc = _service(monkeypatch, tmp_path, contract)
    runtime = svc._service_runtime
    first = await _open(svc)
    assert list(first) == [MATH, CODE]                               # declared, loaded, inactive: no batcher
    running = dict(first)
    _request(svc, active=[MATH, CODE, SCIENCE], shares={MATH: 4000, CODE: 4000, SCIENCE: 2000})
    assert svc._active_batchers == running and SCIENCE not in runtime.envelope(1)["pools"]
    # Even a second boundary pass on the SAME window (a retried open) keeps it frozen.
    plan = await asyncio.to_thread(svc._service_window_plan, 1)
    assert runtime.schedule.revision == 1                            # the request is applied ...
    assert [name for name, _ in plan["env_mix"]] == [MATH, CODE]     # ... and window 1 does not see it
    assert plan["pools"] == runtime.envelope(1)["pools"]
    _archived(svc, 1)
    second = await _open(svc)
    assert list(second) == [MATH, CODE, SCIENCE]
    assert svc._candidate_service_pools == {MATH: CAP * 4000 / 10000, CODE: CAP * 4000 / 10000,
                                            SCIENCE: CAP * 2000 / 10000}
    assert all(batcher.batch_target == B_BATCH for batcher in second.values())


@pytest.mark.asyncio
async def test_a_cooldown_change_builds_a_fresh_map_and_never_touches_the_running_one(monkeypatch, tmp_path):
    svc = _service(monkeypatch, tmp_path)
    first = await _open(svc)
    running_map, running_content = first[MATH]._cooldown, first[MATH]._content_cooldown
    untouched_code = first[CODE]._cooldown
    assert running_map is svc._cooldown_per_env[MATH] and running_map.cooldown_windows == 50
    running_map.record_batched(7, 1)
    _request(svc, cooldowns={MATH: 3})
    _archived(svc, 1)
    second = await _open(svc)
    fresh = second[MATH]._cooldown
    assert fresh is not running_map and fresh is svc._cooldown_per_env[MATH] is svc._cooldown_map
    assert second[MATH]._content_cooldown is not running_content
    assert fresh.cooldown_windows == 3 and second[MATH]._content_cooldown.cooldown_windows == 3
    assert running_map.cooldown_windows == 50 and running_content.cooldown_windows == 50   # R6
    assert fresh.export_state() == {7: 1}                           # history kept
    assert second[MATH].cooldown_prompts_membership == frozenset({7})                      # window 2 < 1 + 3
    assert second[CODE]._cooldown is untouched_code                  # unchanged env keeps its map
    fresh.record_batched(8, 2)
    assert running_map.export_state() == {7: 1}


@pytest.mark.asyncio
async def test_window_open_order_is_checkpoint_then_freeze_then_announce_then_journal_then_admission(
        monkeypatch, tmp_path):
    svc = _service(monkeypatch, tmp_path)
    runtime, calls = svc._service_runtime, []
    for name in ("apply_pending_schedule_request", "ensure_checkpoint", "adopt", "open_window", "announcement"):
        def spy(*args, _real=getattr(runtime, name), _name=name, **kwargs):
            calls.append(_name)
            return _real(*args, **kwargs)
        monkeypatch.setattr(runtime, name, spy)
    recovery = svc._fill_closed_recovery_store
    real_begin = recovery.begin
    monkeypatch.setattr(recovery, "begin", lambda *a, **k: (calls.append("recovery.begin"), real_begin(*a, **k))[1])
    real_expose = svc.server.set_active_batchers
    monkeypatch.setattr(svc.server, "set_active_batchers",
                        lambda batchers: (calls.append("admission"), real_expose(batchers))[1])
    real_open = service_module.open_grpo_window
    monkeypatch.setattr(service_module, "open_grpo_window",
                        lambda **kwargs: (calls.append("batcher"), real_open(**kwargs))[1])
    await _open(svc)
    assert calls == ["apply_pending_schedule_request", "ensure_checkpoint", "adopt", "batcher", "batcher",
                     "open_window", "announcement", "announcement", "recovery.begin", "admission"]


@pytest.mark.asyncio
async def test_a_window_the_runtime_did_not_open_is_never_journalled_nor_exposed(monkeypatch, tmp_path):
    svc = _service(monkeypatch, tmp_path)
    await svc._prepare_service_window()
    svc._open_window()
    for batcher in svc._active_batchers.values():                    # randomness without the runtime open
        batcher.randomness = BEACON
    exposed = []
    monkeypatch.setattr(svc.server, "set_active_batchers", exposed.append)
    with pytest.raises(RuntimeError, match="not opened in the runtime"):
        svc._activate_window()
    assert svc._fill_closed_recovery_store.windows() == [] and exposed == []
    # A window built without its boundary pass is refused too.
    svc._candidate_service_window = None
    with pytest.raises(RuntimeError, match="not prepared at the boundary"):
        svc._build_window_batchers(1)


@pytest.mark.asyncio
async def test_the_window_opens_on_the_validator_clock(monkeypatch, tmp_path):
    svc = _service(monkeypatch, tmp_path)
    monkeypatch.setattr(service_module.time, "time", lambda: 1234.5)
    await _open(svc, activate=False)
    runtime = svc._service_runtime
    assert runtime.db.execute("SELECT opened_at FROM service_windows WHERE window=1").fetchone()[0] == 1234.5


# ---------------------------------------------------------------- restart / retry

@pytest.mark.asyncio
async def test_a_restart_before_activation_rebuilds_the_same_envelope_and_pools(monkeypatch, tmp_path):
    contract = contract_v2(shares={MATH: 6000, CODE: 4000})
    svc = _service(monkeypatch, tmp_path, contract)
    await _open(svc, activate=False)                                 # frozen and announced, then the crash
    envelope = svc._service_runtime.envelope(1)
    announced = svc._active_batchers[MATH].service_policy
    _request(svc, shares={MATH: 2000, CODE: 8000}, cooldowns={MATH: 9})
    svc._service_runtime.close()

    again = _service(monkeypatch, tmp_path, contract)                # restart: same file, same request folder
    again._derive_randomness = AsyncMock(return_value=("other-drand-material", None))
    # Whatever the price walk says after the restart, the frozen window keeps the pools it was priced with.
    from reliquary.validator.emission_price import PriceState
    monkeypatch.setattr("reliquary.constants.EMISSION_PRICE_ARMED", True)
    again._price_shadow_state = PriceState(price=0.5, last_good=0.5)
    assert again._window_n == svc._window_n                          # the same window number is rebuilt
    batchers = await _open(again)
    runtime = again._service_runtime
    assert runtime.schedule.revision == 1                            # the request was taken at the boundary ...
    assert runtime.envelope(1) == envelope                           # ... and window 1 stays what was frozen
    assert again._candidate_service_pools == envelope["pools"] == {MATH: CAP * 0.6, CODE: CAP * 0.4}
    assert again._fill_closed_assembler.pool_for(CODE) == CAP * 4000 / 10000
    assert batchers[MATH]._cooldown.cooldown_windows == 50           # the frozen schedule's cooldown
    assert batchers[MATH].service_policy == announced                # first beacon is permanent
    assert batchers[MATH].randomness != announced["pool_randomness"]
    _archived(again, 1)
    await _open(again)
    assert again._candidate_service_pools == pytest.approx({MATH: CAP * 0.2 * 0.5, CODE: CAP * 0.8 * 0.5})
    assert again._active_batchers[MATH]._cooldown.cooldown_windows == 9


@pytest.mark.asyncio
async def test_a_frozen_window_is_never_rebuilt_on_another_checkpoint(monkeypatch, tmp_path):
    contract = contract_v2()
    svc = _service(monkeypatch, tmp_path, contract)
    await _open(svc, activate=False)
    svc._service_runtime.adopt(checkpoint_n=1, repo="models/test", revision="f" * 40, sha256="e" * 64)
    svc._service_runtime.close()
    again = _service(monkeypatch, tmp_path, contract, revision="f" * 40, checkpoint_n=1)
    with pytest.raises(ValueError, match="frozen on checkpoint"):
        await again._prepare_service_window()
    assert again._candidate_service_window is None


@pytest.mark.asyncio
async def test_an_installed_checkpoint_outside_the_lineage_opens_nothing(monkeypatch, tmp_path, caplog):
    svc = _service(monkeypatch, tmp_path, revision="9" * 40, checkpoint_n=4)
    built = MagicMock(wraps=svc._build_window_batchers)
    svc._build_window_batchers = built
    with caplog.at_level(logging.ERROR), pytest.raises(ValueError, match="lineage"):
        await svc._prepare_service_window()
    assert "service window 1: boundary preparation failed" in caplog.text
    with pytest.raises(RuntimeError, match="not prepared at the boundary"):
        svc._open_window()
    with pytest.raises(Exception):
        svc._service_runtime.envelope(1)


# ---------------------------------------------------------------- R7: installed envs

def test_a_persisted_schedule_with_an_env_not_loaded_or_at_another_version_is_refused(monkeypatch, tmp_path):
    contract = contract_v2(envs=(MATH, CODE, SCIENCE), shares={MATH: 5000, CODE: 5000, SCIENCE: 0})
    svc = _service(monkeypatch, tmp_path, contract, loaded=[MATH, CODE])
    runtime = svc._service_runtime
    svc._require_service_environments(runtime.schedule)             # R7: SCIENCE is declared, inactive, absent
    pinned = svc._service_installed_version
    svc._service_installed_version = lambda name: "0" * 64 if name == CODE else pinned(name)
    with pytest.raises(ValueError, match=f"{CODE} is installed at another version"):
        svc._require_service_environments(runtime.schedule)
    svc._service_installed_version = pinned
    svc.envs[CODE] = _Env(CODE, rows=999)
    with pytest.raises(ValueError, match=f"{CODE} has another dataset size"):
        svc._require_service_environments(runtime.schedule)
    svc.envs[CODE] = _Env("other")
    with pytest.raises(ValueError, match=f"{CODE} is loaded under another name"):
        svc._require_service_environments(runtime.schedule)
    svc.envs[CODE] = _Env(CODE)
    svc.env_targets[CODE] = B_BATCH * 2
    with pytest.raises(ValueError, match="group count per pick"):
        svc._require_service_environments(runtime.schedule)
    svc.env_targets[CODE] = B_BATCH
    # The schedule PERSISTED by an earlier process activates SCIENCE: the contract's launch shares do not.
    changed = next_schedule(contract, runtime.schedule, active=[MATH, CODE, SCIENCE],
                            shares={MATH: 4000, CODE: 4000, SCIENCE: 2000})
    assert runtime.apply_schedule(changed, request_id="earlier-process", window=0)
    with pytest.raises(ValueError, match=f"{SCIENCE} is not loaded by this validator"):
        svc._require_service_environments(runtime.schedule)
    with pytest.raises(ValueError, match=f"{SCIENCE} is not loaded by this validator"):
        svc._service_window_plan(1)                                  # and no window is derived from it


@pytest.mark.asyncio
async def test_a_request_cannot_activate_an_env_this_process_did_not_load(monkeypatch, tmp_path, caplog):
    contract = contract_v2(envs=(MATH, CODE, SCIENCE), shares={MATH: 5000, CODE: 5000, SCIENCE: 0})
    svc = _service(monkeypatch, tmp_path, contract, loaded=[MATH, CODE])
    _request(svc, active=[MATH, CODE, SCIENCE], shares={MATH: 4000, CODE: 4000, SCIENCE: 2000})
    with caplog.at_level(logging.ERROR):
        batchers = await _open(svc)
    status = svc._service_schedule_store.status()
    assert status["status"] == "refused" and "not installed" in status["detail"]
    assert svc._service_runtime.schedule.revision == 0 and list(batchers) == [MATH, CODE]
    assert f"service environment {SCIENCE} cannot be activated: it is not loaded" in caplog.text


# ---------------------------------------------------------------- failures at open

@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["open_window", "announcement"])
async def test_a_runtime_failure_at_open_leaves_nothing_open_and_the_window_is_retried(
        monkeypatch, tmp_path, caplog, failing):
    svc = _service(monkeypatch, tmp_path)
    runtime = svc._service_runtime
    real = getattr(runtime, failing)
    monkeypatch.setattr(runtime, failing, MagicMock(side_effect=OSError("disk I/O error")))
    exposed = []
    real_expose = svc.server.set_active_batchers
    monkeypatch.setattr(svc.server, "set_active_batchers",
                        lambda b: (exposed.append(dict(b)) if b else None, real_expose(b))[1])
    with caplog.at_level(logging.ERROR), pytest.raises(OSError) as error:
        await _open(svc)
    assert "service window 1: the runtime could not open it (OSError: disk I/O error)" in caplog.text
    # Nothing is half open: no randomness, no policy, no recovery record, nothing exposed.
    assert all(b.randomness == "" and b.service_policy is None for b in svc._active_batchers.values())
    assert svc._fill_closed_recovery_store.windows() == [] and exposed == []
    assert svc._window_n == 0 and svc._candidate_window_n == 1
    # What the main loop's handler does with any pre-open failure, then the next iteration.
    svc._rollback_preopen_window(error.value)
    assert svc._active_batchers == {} and svc._candidate_window_n == 1
    monkeypatch.setattr(runtime, failing, real)
    batchers = await _open(svc)
    assert svc._window_n == 1 and list(batchers) == [MATH, CODE]
    assert svc._fill_closed_recovery_store.windows() == [1] and len(exposed) == 1
    assert all(b.service_policy["pool_epoch"] == 1 for b in batchers.values())


# ---------------------------------------------------------------- legacy tasks

@pytest.mark.asyncio
async def test_a_legacy_task_runs_none_of_the_service_wiring(monkeypatch, tmp_path):
    monkeypatch.setattr(service_module, "FILL_CLOSED_ENABLED", True)
    svc = _build_late_drop_service()
    assert svc._service_runtime is None and svc._service_schedule_store is None
    spies = {}
    for name in ("_prepare_service_window", "_service_window_plan", "_open_service_window",
                 "_settle_service_archive", "_refresh_service_cooldown_advice", "_service_window_pool",
                 "_require_service_environments", "_service_environment_problem"):
        spies[name] = MagicMock(side_effect=AssertionError(f"{name} ran for a legacy task"))
        monkeypatch.setattr(svc, name, spies[name])
    svc._fill_closed_recovery_store = FillClosedRecoveryStore(tmp_path / "state")
    svc._checkpoint_store = SimpleNamespace(current_manifest=lambda: SimpleNamespace(
        repo_id="models/test", revision=ROOT, checkpoint_n=0))
    svc._derive_randomness = AsyncMock(return_value=("drand-material", None))
    maps = dict(svc._cooldown_per_env)
    svc._open_window()
    await svc._set_window_randomness(subtensor=None)
    svc._activate_window()
    batcher = next(iter(svc._active_batchers.values()))
    assert batcher.service_runtime is None and batcher.service_policy is None
    assert not hasattr(batcher, "service_environment")
    assert svc._window_env_mix == svc.env_mix and svc._candidate_service_window is None
    assert svc._cooldown_per_env == maps and batcher._cooldown is next(iter(maps.values()))
    record = svc._fill_closed_recovery_store.load(svc._window_n)
    assert record["environments"] == [name for name, _ in svc.env_mix]
    assert all(not spy.called for spy in spies.values())


# ---------------------------------------------------------------- the real main loop

def _spy_runtime(svc, monkeypatch):
    """Record every runtime call of the window lifecycle with the thread it ran on."""
    runtime, calls = svc._service_runtime, []
    for name in ("apply_pending_schedule_request", "ensure_checkpoint", "open_window", "announcement",
                 "reconcile_archive", "refresh_cooldown_advice"):
        def spy(*args, _real=getattr(runtime, name), _name=name, **kwargs):
            window = kwargs.get("window", args[0] if args else None)
            if isinstance(window, dict):
                window = window["window_start"]
            calls.append((_name, window, threading.get_ident()))
            return _real(*args, **kwargs)
        monkeypatch.setattr(runtime, name, spy)
    return calls


@pytest.mark.asyncio
async def test_the_loop_applies_requests_before_open_refreshes_advice_after_settle_all_off_the_event_loop(
        monkeypatch, tmp_path):
    svc = _service(monkeypatch, tmp_path)
    archives = _journal(svc, monkeypatch, tmp_path)
    calls = _spy_runtime(svc, monkeypatch)
    runtime = svc._service_runtime

    seen = []

    def mid_window(window):
        if window == 1:                                              # the operator acts while window 1 runs
            _request(svc, active=[MATH], shares={MATH: 10000})
        seen.append((window, runtime.schedule.revision, sorted(svc._active_batchers), len(calls)))
    iterations = await _run(svc, monkeypatch, windows=2, mid_window=mid_window)
    assert iterations == [0, 1, 2]
    # Mid-window the request is only a file: no runtime call between the announcement and the seal.
    assert seen == [(1, 0, sorted([MATH, CODE]), 5), (2, 1, [MATH], 11)]
    loop_thread = threading.get_ident()
    assert [(name, window) for name, window, _ in calls] == [
        ("apply_pending_schedule_request", 1), ("ensure_checkpoint", None), ("open_window", 1),
        ("announcement", 1), ("announcement", 1), ("reconcile_archive", 1), ("refresh_cooldown_advice", 1),
        ("apply_pending_schedule_request", 2), ("ensure_checkpoint", None), ("open_window", 2),
        ("announcement", 2), ("reconcile_archive", 2), ("refresh_cooldown_advice", 2)]
    assert all(thread != loop_thread for _, _, thread in calls)      # SQLite never runs on the event loop
    pending = archives.pending_archives(start_window=1, end_window=2)
    assert set(pending[1]["service_pools_by_environment"]) == {MATH, CODE}
    assert pending[2]["service_pools_by_environment"] == {MATH: CAP} and pending[2]["environments"] == [MATH]
    assert set(runtime.cooldown_advice()) == {MATH, CODE}
    assert svc._fill_closed_recovery_store.windows() == [] and svc._active_batchers == {}
    runtime.close.assert_called_once()


@pytest.mark.asyncio
async def test_a_runtime_failure_at_the_boundary_or_at_open_does_not_stop_the_loop(monkeypatch, tmp_path, caplog):
    svc = _service(monkeypatch, tmp_path)
    archives = _journal(svc, monkeypatch, tmp_path)
    runtime = svc._service_runtime
    failures = {"ensure_checkpoint": [OSError("disk I/O error")], "open_window": [OSError("database is locked")]}
    for name, queue in failures.items():
        def flaky(*args, _real=getattr(runtime, name), _queue=queue, **kwargs):
            if _queue:
                raise _queue.pop()
            return _real(*args, **kwargs)
        monkeypatch.setattr(runtime, name, flaky)
    with caplog.at_level(logging.ERROR):
        iterations = await _run(svc, monkeypatch, windows=2)
    assert iterations == [0, 0, 0, 1, 2]                             # two failed opens of window 1, then on
    assert "service window 1: boundary preparation failed (OSError: disk I/O error)" in caplog.text
    assert "service window 1: the runtime could not open it (OSError: database is locked)" in caplog.text
    pending = archives.pending_archives(start_window=1, end_window=2)
    assert sorted(pending) == [1, 2] and all(a["window_status"] == "completed" for a in pending.values())
    assert svc._fill_closed_recovery_store.windows() == []


@pytest.mark.asyncio
async def test_a_settlement_failure_at_seal_archives_the_window_from_its_receipts_and_the_loop_goes_on(
        monkeypatch, tmp_path, caplog):
    svc = _service(monkeypatch, tmp_path)
    archives = _journal(svc, monkeypatch, tmp_path)
    runtime = svc._service_runtime
    real, seen = runtime.reconcile_archive, []

    def flaky(archive, **kwargs):
        seen.append((archive["window_start"], archive["window_status"], kwargs))
        if len(seen) == 1:
            raise SettlementError("unit: the live archive cannot be settled")
        return real(archive, **kwargs)
    monkeypatch.setattr(runtime, "reconcile_archive", flaky)
    with caplog.at_level(logging.ERROR):
        iterations = await _run(svc, monkeypatch, windows=1)
    assert iterations == [0, 1]
    assert "service window 1: settlement failed at seal (SettlementError" in caplog.text
    # Second settlement: the same window, rebuilt from the journal receipts (no paid group here -> aborted).
    assert seen == [(1, "completed", {}), (1, "aborted", {"aborted": True})]
    archive = archives.pending_archives(start_window=1, end_window=1)[1]
    assert archive["window_status"] == "aborted" and archive["rewards_by_hotkey"] == {}
    assert archive["service_exploration_by_environment"] == {}
    assert svc._fill_closed_recovery_store.windows() == [] and svc._active_batchers == {}


@pytest.mark.asyncio
async def test_a_failed_seal_settlement_keeps_every_committed_training_payment(monkeypatch, tmp_path, caplog):
    svc = _service(monkeypatch, tmp_path)
    archives = _journal(svc, monkeypatch, tmp_path)
    runtime = svc._service_runtime
    await _open(svc)
    _pay_one_group_per_env(svc)
    real, seen = runtime.reconcile_archive, []

    def flaky(archive, **kwargs):
        seen.append((archive["window_status"], kwargs))
        if len(seen) == 1:
            raise SettlementError("unit: the live archive cannot be settled")
        return real(archive, **kwargs)
    monkeypatch.setattr(runtime, "reconcile_archive", flaky)
    _seal(svc)
    with caplog.at_level(logging.ERROR):
        await svc._train_and_publish()                               # the seal path of the main loop
    assert seen == [("completed", {}), ("recovered_partial", {"aborted": False})]
    archive = archives.pending_archives(start_window=1, end_window=1)[1]
    unit = CAP * 0.5 / (PICKS * SLOTS)
    assert archive["window_status"] == "recovered_partial"
    assert archive["rewards_by_hotkey"] == {"alice": unit, "bob": unit}
    assert [(row["hotkey"], row["env_name"], row["prompt_idx"]) for row in archive["batch"]] == [
        ("alice", MATH, 3), ("bob", CODE, 4)]
    assert archive["service_training_by_environment"] == {MATH: {"alice": unit}, CODE: {"bob": unit}}
    assert svc._fill_closed_recovery_store.windows() == [] and svc._active_batchers == {}
    assert caplog.text.count("service window 1: settlement failed at seal") == 1


@pytest.mark.asyncio
async def test_a_window_the_runtime_cannot_settle_is_left_unarchived_and_settled_at_the_next_start(
        monkeypatch, tmp_path, caplog):
    svc = _service(monkeypatch, tmp_path)
    archives = _journal(svc, monkeypatch, tmp_path)
    runtime = svc._service_runtime
    real = runtime.reconcile_archive
    broken = {"on": True}

    def reconcile(archive, **kwargs):
        if broken["on"] and archive["window_start"] == 1:
            raise SettlementError("unit: window 1 cannot be settled")
        return real(archive, **kwargs)
    monkeypatch.setattr(runtime, "reconcile_archive", reconcile)

    def mid_window(window):                                          # groups are paid during window 1
        if window == 1:
            _pay_one_group_per_env(svc)
    with caplog.at_level(logging.ERROR):
        iterations = await _run(svc, monkeypatch, windows=2, mid_window=mid_window)
    assert iterations == [0, 1, 2]                                   # the loop went on to window 2
    pending = archives.pending_archives(start_window=1, end_window=2)
    assert sorted(pending) == [2]                                    # nothing was paid for window 1 ...
    assert svc._fill_closed_recovery_store.windows() == [1]          # ... and its record is kept
    assert svc._fill_closed_recovery_store.load(1)["archive"] is None
    assert "service window 1: recovery could not settle it (SettlementError" in caplog.text
    assert "Failed to enqueue aborted-window tombstone" in caplog.text
    # Next start (``_initialize_fill_closed_rotation_store``): the same record, the committed payments.
    broken["on"] = False
    recovered = svc._fill_closed_recovery_store.recover(
        1, queue=svc._training_payload_queue, archives=archives, rotation=svc._fill_closed_rotation_store,
        service_runtime=runtime)
    unit = CAP * 0.5 / (PICKS * SLOTS)
    assert recovered["window_status"] == "recovered_partial"
    assert recovered["rewards_by_hotkey"] == {"alice": unit, "bob": unit}
    assert sorted(archives.pending_archives(start_window=1, end_window=2)) == [1, 2]
    assert svc._fill_closed_recovery_store.windows() == []


# ---------------------------------------------------------------- recovery never settles an archived window

class _Crash(BaseException):
    """The process dies here."""


@pytest.mark.asyncio
async def test_a_crash_between_settlement_and_archive_enqueue_recovers_the_same_money(monkeypatch, tmp_path, caplog):
    contract = contract_v2()
    svc = _service(monkeypatch, tmp_path, contract)
    archives = _journal(svc, monkeypatch, tmp_path)
    runtime = svc._service_runtime
    batchers = await _open(svc)
    _pay_one_group_per_env(svc)
    settled = []
    real = runtime.reconcile_archive
    monkeypatch.setattr(runtime, "reconcile_archive",
                        lambda archive, **kw: (settled.append(real(archive, **kw)), settled[-1])[1])
    recovery = svc._fill_closed_recovery_store
    monkeypatch.setattr(recovery, "finish", MagicMock(side_effect=_Crash()))   # dies before the record is written
    with pytest.raises(_Crash):
        await svc._archive_window(dict(batchers), {name: ([], {}) for name in batchers})
    assert len(settled) == 1 and recovery.load(1)["archive"] is None       # settled, nothing enqueued
    assert archives.pending_archives(start_window=1, end_window=1) == {}
    runtime.close()

    restarted = _runtime(tmp_path / "service", contract)                   # the next process
    again = FillClosedRecoveryStore(tmp_path / "state")
    with caplog.at_level(logging.ERROR, logger="reliquary.services.runtime"):
        archive = again.recover(1, queue=svc._training_payload_queue, archives=archives,
                                rotation=svc._fill_closed_rotation_store, service_runtime=restarted)
    # Same paid rows, same disposition: the runtime sees no flip and no other batch.
    assert "settled" not in caplog.text
    assert archive["window_status"] == "recovered_partial"
    assert _money(archive) == _money(settled[0])
    assert archives.pending_archives(start_window=1, end_window=1)[1] == archive
    assert again.windows() == []
    restarted.close()


@pytest.mark.asyncio
async def test_a_committed_archive_is_enqueued_again_as_it_is_and_never_settled_again(monkeypatch, tmp_path):
    contract = contract_v2()
    svc = _service(monkeypatch, tmp_path, contract)
    archives = _journal(svc, monkeypatch, tmp_path)
    runtime = svc._service_runtime
    batchers = await _open(svc)
    _pay_one_group_per_env(svc)
    real_enqueue = archives.enqueue
    monkeypatch.setattr(archives, "enqueue", MagicMock(side_effect=_Crash()))   # dies after the record is written
    with pytest.raises(_Crash):
        await svc._archive_window(dict(batchers), {name: ([], {}) for name in batchers})
    committed = svc._fill_closed_recovery_store.load(1)["archive"]
    assert committed is not None and committed["window_status"] == "completed"
    # The archived batch is the assembler's paid set: the service money equals what the assembler paid.
    assert committed["service_training_recomputed_delta"] == 0.0
    assert committed["rewards_by_hotkey"] == {"alice": CAP * 0.5 / (PICKS * SLOTS), "bob": CAP * 0.5 / (PICKS * SLOTS)}
    monkeypatch.setattr(archives, "enqueue", real_enqueue)

    never = MagicMock()
    never.reconcile_archive.side_effect = AssertionError("a committed archive was settled again")
    again = FillClosedRecoveryStore(tmp_path / "state")
    archive = again.recover(1, queue=svc._training_payload_queue, archives=archives,
                            rotation=svc._fill_closed_rotation_store, service_runtime=never)
    assert archive == committed and archives.pending_archives(start_window=1, end_window=1)[1] == committed
    never.reconcile_archive.assert_not_called()
    assert again.windows() == []
    with pytest.raises(FileNotFoundError):                           # once enqueued the window cannot be recovered
        again.recover(1, queue=svc._training_payload_queue, archives=archives,
                      rotation=svc._fill_closed_rotation_store, service_runtime=never)
    runtime.close()


@pytest.mark.asyncio
async def test_the_advice_refresh_never_raises_and_says_which_window(monkeypatch, tmp_path, caplog):
    svc = _service(monkeypatch, tmp_path)
    await svc._refresh_service_cooldown_advice(3)
    advice = svc._service_runtime.cooldown_advice()
    assert set(advice) == {MATH, CODE} and all(entry["current_windows"] == 50 for entry in advice.values())
    monkeypatch.setattr(svc._service_runtime, "refresh_cooldown_advice", MagicMock(side_effect=OSError("disk")))
    with caplog.at_level(logging.ERROR):
        await svc._refresh_service_cooldown_advice(4)
    assert "service window 4: cooldown advice refresh failed (OSError: disk)" in caplog.text


@pytest.mark.asyncio
async def test_a_legacy_task_goes_through_the_real_loop_without_any_service_call(monkeypatch, tmp_path):
    monkeypatch.setattr(service_module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr("reliquary.constants.EMISSION_PRICE_ARMED", False)
    svc = _build_late_drop_service()
    svc._derive_randomness = AsyncMock(return_value=("drand-material", None))
    svc._checkpoint_store = SimpleNamespace(current_manifest=lambda: SimpleNamespace(
        repo_id="models/test", revision=ROOT, checkpoint_n=0))
    archives = _journal(svc, monkeypatch, tmp_path)
    spies = {}
    for name in ("_prepare_service_window", "_service_window_plan", "_open_service_window",
                 "_settle_service_archive", "_refresh_service_cooldown_advice", "_service_window_pool",
                 "_require_service_environments"):
        spies[name] = MagicMock(side_effect=AssertionError(f"{name} ran for a legacy task"))
        monkeypatch.setattr(svc, name, spies[name])
    iterations = await _run(svc, monkeypatch, windows=2)
    assert iterations == [0, 1, 2]
    pending = archives.pending_archives(start_window=1, end_window=2)
    assert sorted(pending) == [1, 2]
    assert not any(key.startswith("service_") for archive in pending.values() for key in archive)
    assert all(not spy.called for spy in spies.values())


# ---------------------------------------------------------------- boot (the real constructor)

def _boot(monkeypatch, tmp_path, contract, *, loaded, installed=None, **overrides):
    from reliquary.validator.service import ValidationService
    from tests.unit.test_service_v2 import _LateDropFakeWallet

    for name, value in {"DETACHED_TRAINER": True, "FILL_CLOSED_ENABLED": True, "PIPELINED_WINDOWS": False,
                        "ENFORCE_ENVELOPE_SIGNATURE": True, **overrides}.items():
        monkeypatch.setattr(f"reliquary.constants.{name}", value)
    monkeypatch.setenv("RELIQUARY_TASK_SCOPED_CHECKPOINTS", "1")
    monkeypatch.setenv("RELIQUARY_TASK_ID", "task")
    monkeypatch.setenv("RELIQUARY_TRAINING_RUN_ID", "run")
    monkeypatch.setenv("RELIQUARY_STATE_DIR", str(tmp_path / "state"))
    qualification = tmp_path / "qualification.json"
    qualification.write_text(json.dumps(qualification_v2(contract)))
    monkeypatch.setenv("RELIQUARY_SERVICE_QUALIFICATION", str(qualification))
    monkeypatch.setattr(service_module, "load_environments", lambda names: {name: _Env(name) for name in names})
    monkeypatch.setattr("reliquary.services.schedule.default_installed_version",
                        installed or (lambda name: contract.environments[name]["version"]))
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 99
    return ValidationService(wallet=_LateDropFakeWallet(), model=MagicMock(), tokenizer=tokenizer, netuid=99,
                             env_mix=[(name, B_BATCH) for name in loaded], service_contract=contract,
                             emission_cap=CAP)


def _bootable_contract(**kwargs):
    from reliquary.protocol.service_contract import ServiceContract
    from reliquary.shared.training_payload import active_training_identity

    value = contract_v2_dict(**kwargs)
    value["generation_contract_sha256"] = active_training_identity()["generation_contract_sha256"]
    return ServiceContract.from_dict(value)


def _persisted_folder(tmp_path):
    folder = tmp_path / "state" / "service-policies" / "task" / "run"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def test_boot_wires_the_runtime_the_request_folder_and_the_persisted_cooldowns(monkeypatch, tmp_path):
    contract = _bootable_contract(envs=(MATH, CODE, SCIENCE), shares={MATH: 5000, CODE: 5000, SCIENCE: 0})
    folder = _persisted_folder(tmp_path)
    earlier = _runtime(folder, contract)                             # an earlier process changed a cooldown
    assert earlier.apply_schedule(next_schedule(contract, earlier.schedule, cooldowns={MATH: 9}),
                                  request_id="earlier", window=0)
    earlier.close()
    # R7: SCIENCE is declared, inactive and NOT loaded; boot accepts it.
    svc = _boot(monkeypatch, tmp_path, contract, loaded=[MATH, CODE])
    assert svc._service_runtime.schedule.revision == 1
    assert svc._service_schedule_store.folder == folder
    assert svc._service_runtime.db.execute("PRAGMA database_list").fetchone()[2] == str(folder / "runtime.sqlite3")
    assert svc._cooldown_per_env[MATH].cooldown_windows == 9 == svc._content_cooldown_per_env[MATH].cooldown_windows
    assert svc._cooldown_per_env[CODE].cooldown_windows == 50
    assert svc._cooldown_map is svc._cooldown_per_env[MATH]
    svc._service_runtime.close()


@pytest.mark.parametrize("problem", ["not_loaded", "version", "rows"])
def test_boot_refuses_a_persisted_schedule_whose_active_env_this_validator_cannot_run(
        monkeypatch, tmp_path, problem):
    contract = _bootable_contract(envs=(MATH, CODE, SCIENCE), shares={MATH: 5000, CODE: 5000, SCIENCE: 0})
    folder = _persisted_folder(tmp_path)
    installed = None
    if problem == "not_loaded":
        # The contract's launch shares leave SCIENCE off; the schedule a previous process persisted has it on.
        earlier = _runtime(folder, contract)
        assert earlier.apply_schedule(
            next_schedule(contract, earlier.schedule, active=[MATH, CODE, SCIENCE],
                          shares={MATH: 4000, CODE: 4000, SCIENCE: 2000}), request_id="earlier", window=0)
        earlier.close()
        match = f"{SCIENCE} is not loaded by this validator"
    elif problem == "version":
        installed = lambda name: "0" * 64 if name == CODE else contract.environments[name]["version"]  # noqa: E731
        match = f"{CODE} is installed at another version"
    else:
        monkeypatch.setattr(_Env, "__len__", lambda self: 999 if self.name == CODE else 1000)
        match = f"{CODE} has another dataset size"
    with pytest.raises(ValueError, match=match):
        _boot(monkeypatch, tmp_path, contract, loaded=[MATH, CODE], installed=installed)


def test_boot_refuses_a_service_task_outside_its_execution_mode(monkeypatch, tmp_path):
    contract = _bootable_contract()
    with pytest.raises(ValueError, match="signed, serial fill-closed detached"):
        _boot(monkeypatch, tmp_path, contract, loaded=[MATH, CODE], PIPELINED_WINDOWS=True)
    from reliquary.validator.service import ValidationService
    real_init = ValidationService.__init__
    monkeypatch.setattr(ValidationService, "__init__",
                        lambda self, *a, **k: real_init(self, *a, **{**k, "use_drand": False}))
    with pytest.raises(ValueError, match="drand"):
        _boot(monkeypatch, tmp_path, contract, loaded=[MATH, CODE])
