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
import os
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import reliquary.validator.service as service_module
from reliquary.constants import B_BATCH, FILL_CLOSED_ADMISSION_BUDGET_PER_ENV, M_ROLLOUTS
from reliquary.protocol.release_contract import canonical_sha256
from reliquary.protocol.service_schedule import next_schedule
from reliquary.services.runtime import (
    FROZEN_ARCHIVE_FIELDS, ServicePolicyLimit, ServiceRuntime, protocol_slot_geometry,
)
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
NEXT = "f" * 40
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


def _runtime(folder, contract, started=0):
    return ServiceRuntime(folder / "runtime.sqlite3", contract, qualification_v2(contract), now=started,
                          drand_round_at=drand_round)


def _service(monkeypatch, tmp_path, contract=None, *, loaded=None, cap=CAP, revision=ROOT, checkpoint_n=0,
             started=0):
    """A real ValidationService wired the way ``__init__`` wires a service task.

    ``started`` is the order's start on the validator clock: a test that records observations
    on the wall clock starts its order now (an order started at 0 is long past its deadline).
    """
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
    svc._service_runtime = _runtime(folder, contract, started)
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


def _explore(svc, *, window=1, hotkey="explorer", prompt=9, env=MATH):
    """One exploration group of a never-scanned prompt, recorded, drawn and audited: owed at the seal."""
    runtime = svc._service_runtime
    opened = runtime.db.execute("SELECT opened_at FROM service_windows WHERE window=?", (window,)).fetchone()[0]
    pool = runtime.seed_pool(environment=env, prompt_idx=prompt, window=window)
    selection = pool.selection(list(range(M_ROLLOUTS)))
    result = runtime.record_exploration(
        environment=env, prompt_idx=prompt, hotkey=hotkey, window=window, rewards=[0.0] * M_ROLLOUTS,
        group_id=selection.sha256, candidate={"pool_sha256": selection.pool_sha256, "seeds": list(selection.seeds)},
        token_count=100, now=opened + 1)
    assert result["entitled"], result
    runtime.resolve_draws(window, beacon_for_round=lambda round_id: "cd" * 32, now=opened + 1000)
    assert runtime.record_audit(result["observation_id"], passed=True, now=opened + 1000).kind == "passed"
    return result


def _exploration_price(pool):
    return 0.15 * pool / (PICKS * SLOTS)


def _seal(svc):
    for batcher in svc._active_batchers.values():
        batcher.force_seal("unit")


def _money(archive):
    return json.dumps({key: archive.get(key) for key in FROZEN_ARCHIVE_FIELDS}, sort_keys=True)


async def _run(svc, monkeypatch, *, windows, mid_window=None, rotation_wait=None):
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
    monkeypatch.setattr(svc, "_wait_for_fill_closed_rotation",
                        rotation_wait or AsyncMock(return_value="not_armed"))
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
    # A boundary pass on the ACTIVATED window is refused: it is never rebuilt, nor its envelope dropped.
    envelope = runtime.envelope(1)
    with pytest.raises(RuntimeError, match="window 1 was activated"):
        await asyncio.to_thread(svc._service_window_plan, 1)
    assert runtime.envelope(1) == envelope and runtime.schedule.revision == 0
    assert svc._active_batchers[MATH].service_policy["pool_randomness"] == svc._active_batchers[MATH].randomness
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
async def test_a_window_frozen_before_a_restart_and_never_activated_is_frozen_again_on_what_is_true_now(
        monkeypatch, tmp_path, caplog):
    """I3 + m4: the unactivated envelope is discarded; current schedule, current price, fresh beacon."""
    contract = contract_v2(shares={MATH: 6000, CODE: 4000})
    svc = _service(monkeypatch, tmp_path, contract)
    await _open(svc, activate=False)                                 # frozen and announced, then the crash
    envelope = svc._service_runtime.envelope(1)
    announced = svc._active_batchers[MATH].service_policy
    assert envelope["pools"] == {MATH: CAP * 0.6, CODE: CAP * 0.4}
    _request(svc, shares={MATH: 2000, CODE: 8000}, cooldowns={MATH: 9})
    svc._service_runtime.close()

    again = _service(monkeypatch, tmp_path, contract)                # restart: same file, same request folder
    again._derive_randomness = AsyncMock(return_value=("other-drand-material", None))
    from reliquary.validator.emission_price import PriceState
    monkeypatch.setattr("reliquary.constants.EMISSION_PRICE_ARMED", True)
    again._price_shadow_state = PriceState(price=0.5, last_good=0.5)
    assert again._window_n == svc._window_n                          # the same window number is rebuilt
    with caplog.at_level(logging.WARNING):
        batchers = await _open(again)
    assert "service window 1 was frozen on checkpoint dddddddddddd (schedule revision 0) and never activated" \
        in caplog.text
    runtime = again._service_runtime
    fresh = runtime.envelope(1)
    # No miner was ever admitted on the first envelope: the window takes the request and today's price.
    assert runtime.schedule.revision == 1 == fresh["schedule"]["revision"]
    assert again._candidate_service_pools == fresh["pools"] == pytest.approx({MATH: CAP * 0.2 * 0.5,
                                                                             CODE: CAP * 0.8 * 0.5})
    assert again._fill_closed_assembler.pool_for(CODE) == fresh["pools"][CODE]
    assert batchers[MATH]._cooldown.cooldown_windows == 9
    for batcher in batchers.values():
        policy = batcher.service_policy
        # m4: what miners derive their seed pool from IS what the batcher verifies with.
        assert policy["pool_randomness"] == batcher.randomness != announced["pool_randomness"]
        assert policy["schedule"] == fresh["schedule"] and policy["pool_epoch"] == 1
    assert runtime.db.execute("SELECT randomness FROM service_pools WHERE window=1").fetchall() == [
        (batchers[MATH].randomness,)]
    # Now the window IS activated: from here its envelope and its beacon are permanent.
    with pytest.raises(RuntimeError, match="window 1 was activated"):
        again._service_window_plan(1)
    assert runtime.envelope(1) == fresh


@pytest.mark.asyncio
async def test_a_checkpoint_installed_between_two_attempts_of_one_window_does_not_block_it(
        monkeypatch, tmp_path, caplog):
    """I3 (liveness): the first attempt froze window 1 on the root; the retry runs on its child."""
    contract = contract_v2()
    svc = _service(monkeypatch, tmp_path, contract)
    runtime = svc._service_runtime
    first_open = []
    real_time = service_module.time.time
    monkeypatch.setattr(service_module.time, "time", lambda: 1000.0)
    await svc._prepare_service_window()
    svc._open_window()
    await svc._set_window_randomness(subtensor=None)                 # frozen + announced on ROOT ...
    first_open.append((runtime.envelope(1), svc._active_batchers[MATH].randomness))
    svc._rollback_preopen_window(RuntimeError("unit: the open failed after the freeze"))   # ... never activated
    assert svc._fill_closed_recovery_store.windows() == [] and svc._candidate_window_n == 1
    # The rotation gate installs the trainer's next checkpoint before the retry.
    runtime.adopt(checkpoint_n=1, repo="models/test", revision=NEXT, sha256="e" * 64)
    svc._checkpoint_store = SimpleNamespace(current_manifest=lambda: SimpleNamespace(
        repo_id="models/test", revision=NEXT, checkpoint_n=1))
    svc._derive_randomness = AsyncMock(return_value=("later-drand-material", None))
    monkeypatch.setattr(service_module.time, "time", lambda: 2000.0)
    with caplog.at_level(logging.WARNING):
        batchers = await _open(svc)
    monkeypatch.setattr(service_module.time, "time", real_time)
    assert "that envelope and its beacon are discarded" in caplog.text
    envelope = runtime.envelope(1)
    assert first_open[0][0]["checkpoint"]["revision"] == ROOT and envelope["checkpoint"]["revision"] == NEXT
    assert runtime.db.execute("SELECT opened_at FROM service_windows WHERE window=1").fetchone()[0] == 2000.0
    assert svc._window_n == 1 and svc._fill_closed_recovery_store.load(1)["parent_revision"] == NEXT
    for batcher in batchers.values():
        assert batcher.current_checkpoint_hash == NEXT == batcher.service_policy["checkpoint"]["revision"]
        assert batcher.randomness == batcher.service_policy["pool_randomness"] != first_open[0][1]   # m4
    # The seed pool miners derive is the one of the fresh beacon and the installed checkpoint.
    pool = runtime.seed_pool(environment=MATH, prompt_idx=3, window=1)
    from reliquary.protocol.seed_pool import pool_from_service_policy
    assert pool.sha256 == pool_from_service_policy(batchers[MATH].service_policy, environment=MATH, prompt_idx=3,
                                                   checkpoint_hash=NEXT).sha256


@pytest.mark.asyncio
async def test_a_window_with_an_observation_is_never_frozen_again(monkeypatch, tmp_path, caplog):
    """The runtime's own refusal reaches the loop as a boundary failure: nothing is opened."""
    svc = _service(monkeypatch, tmp_path, started=service_module.time.time())
    runtime = svc._service_runtime
    await _open(svc, activate=False)
    _explore(svc)                                                    # cannot happen unactivated; if it did ...
    envelope = runtime.envelope(1)
    svc._rollback_preopen_window(RuntimeError("unit"))
    with caplog.at_level(logging.ERROR), pytest.raises(ServicePolicyLimit, match="has observations"):
        await svc._prepare_service_window()
    assert "service window 1: boundary preparation failed (ServicePolicyLimit" in caplog.text
    assert runtime.envelope(1) == envelope and svc._candidate_service_window is None


@pytest.mark.asyncio
async def test_a_beacon_that_is_not_the_windows_randomness_opens_nothing(monkeypatch, tmp_path):
    """m4 as a rule: an announcement carrying another beacon than the batchers' randomness is refused."""
    svc = _service(monkeypatch, tmp_path)
    runtime = svc._service_runtime
    real = runtime.announcement
    monkeypatch.setattr(runtime, "announcement",
                        lambda **kwargs: {**real(**kwargs), "pool_randomness": "77" * 32})
    with pytest.raises(RuntimeError, match="announced with another beacon"):
        await _open(svc)
    assert all(b.randomness == "" and b.service_policy is None for b in svc._active_batchers.values())
    assert svc._fill_closed_recovery_store.windows() == [] and not svc._candidate_service_window["opened"]


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
    assert status["status"] == "refused"
    assert status["detail"] == f"cannot activate {SCIENCE}: the environment is not loaded by this validator"
    assert svc._service_runtime.schedule.revision == 0 and list(batchers) == [MATH, CODE]
    assert f"service environment {SCIENCE} cannot be activated: it is not loaded" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("problem, said", [
    ("rows", "has another dataset size than the order"),
    ("name", "is loaded under another name"),
    ("group_count", "has a group count per pick that differs from the protocol's"),
    ("version", "is installed at another version than the one the order pins"),
])
async def test_a_refused_activation_says_what_is_wrong_with_the_env(monkeypatch, tmp_path, problem, said):
    """m3: the operator reads the real reason, not "not installed"."""
    contract = contract_v2(envs=(MATH, CODE, SCIENCE), shares={MATH: 5000, CODE: 5000, SCIENCE: 0})
    svc = _service(monkeypatch, tmp_path, contract)
    if problem == "rows":
        svc.envs[SCIENCE] = _Env(SCIENCE, rows=999)
    elif problem == "name":
        svc.envs[SCIENCE] = _Env("other")
    elif problem == "group_count":
        svc.env_targets[SCIENCE] = B_BATCH * 2
    else:
        pinned = svc._service_installed_version
        svc._service_installed_version = lambda name: "0" * 64 if name == SCIENCE else pinned(name)
    _request(svc, active=[MATH, CODE, SCIENCE], shares={MATH: 4000, CODE: 4000, SCIENCE: 2000})
    batchers = await _open(svc)
    status = svc._service_schedule_store.status()
    assert status["status"] == "refused" and status["detail"] == f"cannot activate {SCIENCE}: the environment {said}"
    assert "not installed" not in status["detail"]
    assert svc._service_runtime.schedule.revision == 0 and list(batchers) == [MATH, CODE]
    # An env this process can run is still switched on, with its version.
    assert svc._service_activation_version(MATH) == contract.environments[MATH]["version"]


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
    assert all(b.service_policy["pool_randomness"] == b.randomness for b in batchers.values())   # m4


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
async def test_a_checkpoint_swap_between_two_windows_in_the_real_loop(monkeypatch, tmp_path):
    """The REAL ``_swap_staged_checkpoint`` at the rotation gate between window 1 and window 2."""
    from reliquary.trainer.publisher import PUBLICATION_RECEIPT

    svc = _service(monkeypatch, tmp_path)
    archives = _journal(svc, monkeypatch, tmp_path)
    runtime = svc._service_runtime
    current = {"entry": SimpleNamespace(repo_id="models/test", revision=ROOT, checkpoint_n=0)}

    def install(number, revision):
        current["entry"] = SimpleNamespace(repo_id="models/test", revision=revision, checkpoint_n=number)
        return current["entry"]
    svc._checkpoint_store = SimpleNamespace(repo_id="models/test", current_manifest=lambda: current["entry"],
                                            install_external=install)
    manifest = {"checkpoint_n": 1, "repo_id": "models/test", "revision": NEXT, "trained_window_cursor": 0}
    files = {"model.safetensors": {"size": 9, "sha256": "2" * 64, "blob_id": "2" * 40}}
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / PUBLICATION_RECEIPT).write_text(json.dumps({
        "publication_id": "unit", "parent_revision": ROOT, "files": files,
        "manifest": {key: value for key, value in manifest.items() if key != "revision"}}))
    svc._checkpoint_intake = SimpleNamespace(take_staged=lambda: (manifest, stage), mark_installed=MagicMock(),
                                             snapshot=lambda: {}, staged_ready=False)
    refreshed = MagicMock()                                              # the verify plane's weight load
    monkeypatch.setattr(svc, "_refresh_verify_model_from_dir", refreshed)
    monkeypatch.setattr(svc.server, "set_current_checkpoint", MagicMock())
    calls = _spy_runtime(svc, monkeypatch)
    real_adopt = runtime.adopt
    monkeypatch.setattr(runtime, "adopt", lambda **kwargs: (
        calls.append(("adopt", kwargs["revision"], threading.get_ident())), real_adopt(**kwargs))[1])

    async def rotation_gate():
        if svc._window_n == 1 and current["entry"].revision == ROOT:
            await svc._swap_staged_checkpoint(1)                       # where the loop installs a checkpoint
            return "checkpoint_adopted"
        return "not_armed"
    seen = {}

    def mid_window(window):
        revision = ROOT if window == 1 else NEXT
        group = dataclasses.replace(_paid("alice", 3 + window), claimed_checkpoint_hash=revision)
        svc._fill_closed_assembler.accept(MATH, [group], window, revision)
        seen[window] = (svc._fill_closed_recovery_store.load(window)["parent_revision"],
                        {b.current_checkpoint_hash for b in svc._active_batchers.values()},
                        {b.service_policy["checkpoint"]["revision"] for b in svc._active_batchers.values()},
                        all(b.randomness == b.service_policy["pool_randomness"]
                            for b in svc._active_batchers.values()))
    iterations = await _run(svc, monkeypatch, windows=2, mid_window=mid_window,
                            rotation_wait=AsyncMock(side_effect=rotation_gate))
    assert iterations == [0, 1, 2]
    assert seen == {1: (ROOT, {ROOT}, {ROOT}, True), 2: (NEXT, {NEXT}, {NEXT}, True)}
    # Window 1 settles on the root before the swap; window 2 is frozen on the child.
    names = [(name, window) for name, window, _ in calls]
    assert names.index(("reconcile_archive", 1)) < names.index(("adopt", NEXT)) \
        < names.index(("apply_pending_schedule_request", 2)) < names.index(("open_window", 2))
    assert runtime.envelope(1)["checkpoint"]["revision"] == ROOT
    assert runtime.envelope(2)["checkpoint"] == {"checkpoint_n": 1, "repo": "models/test", "revision": NEXT,
                                                  "sha256": canonical_sha256(files)}
    assert runtime.db.execute("SELECT revision FROM service_checkpoints ORDER BY seq").fetchall() == [
        (ROOT,), (NEXT,)]
    pending = archives.pending_archives(start_window=1, end_window=2)
    unit = CAP * 0.5 / (PICKS * SLOTS)
    assert all(pending[window]["window_status"] == "completed" and
               pending[window]["rewards_by_hotkey"] == {"alice": unit} for window in (1, 2))
    assert [row["prompt_idx"] for window in (1, 2) for row in pending[window]["batch"]] == [4, 5]
    # One run across the swap: the observations and scans of window 1 are still there.
    assert svc._checkpoint_n == 1 and svc._fill_closed_recovery_store.windows() == []
    refreshed.assert_called_once_with(stage, NEXT)


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
async def test_a_settlement_failure_at_seal_settles_the_sealed_window_again_and_the_loop_goes_on(
        monkeypatch, tmp_path, caplog):
    """I2: the window SEALED; with no paid training group it is still settled not aborted."""
    svc = _service(monkeypatch, tmp_path, started=service_module.time.time())
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
        iterations = await _run(svc, monkeypatch, windows=1,
                                mid_window=lambda window: _explore(svc) if window == 1 else None)
    assert iterations == [0, 1]
    assert "service window 1: settlement failed at seal (SettlementError" in caplog.text
    # Second settlement: the same window, rebuilt from the journal receipts, told that it sealed.
    assert seen == [(1, "completed", {}), (1, "completed", {"aborted": False})]
    archive = archives.pending_archives(start_window=1, end_window=1)[1]
    assert archive["window_status"] == "completed" and archive["batch"] == []
    assert archive["service_exploration_by_environment"] == {MATH: {"explorer": 1}}
    assert archive["rewards_by_hotkey"] == {"explorer": pytest.approx(_exploration_price(CAP * 0.5))}
    assert runtime.window_disposition(1) == "settled" and runtime.log.is_scanned(MATH, 9)
    assert svc._fill_closed_recovery_store.windows() == [] and svc._active_batchers == {}
    assert svc._service_sealed_windows == set()


@pytest.mark.asyncio
async def test_the_seal_writes_the_window_then_settles_then_commits_the_archive(monkeypatch, tmp_path):
    """I2: nothing that can fail stands between the settlement and ``finish``."""
    svc = _service(monkeypatch, tmp_path)
    _journal(svc, monkeypatch, tmp_path)
    runtime, recovery, order = svc._service_runtime, svc._fill_closed_recovery_store, []
    for owner, name in ((svc._utility_telemetry, "write_window"), (runtime, "reconcile_archive"),
                        (recovery, "finish")):
        def spy(*args, _real=getattr(owner, name), _name=name, **kwargs):
            order.append(_name)
            return _real(*args, **kwargs)
        monkeypatch.setattr(owner, name, spy)
    batchers = await _open(svc)
    _seal(svc)
    await svc._archive_window(dict(batchers), {name: ([], {}) for name in batchers})
    assert order == ["write_window", "reconcile_archive", "finish"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["write_window", "finish"])
async def test_an_exploration_only_window_keeps_its_pay_when_the_archive_step_fails(
        monkeypatch, tmp_path, caplog, failing):
    """I2: a failure before the settlement (never settled) or right after it (settled, not committed)."""
    svc = _service(monkeypatch, tmp_path, started=service_module.time.time())
    archives = _journal(svc, monkeypatch, tmp_path)
    runtime, recovery = svc._service_runtime, svc._fill_closed_recovery_store
    settled, attempts = [], []
    real = runtime.reconcile_archive
    monkeypatch.setattr(runtime, "reconcile_archive",
                        lambda archive, **kw: (settled.append((kw, real(archive, **kw))), settled[-1][1])[1])
    owner = svc._utility_telemetry if failing == "write_window" else recovery

    def flaky(*args, _real=getattr(owner, failing), **kwargs):
        attempts.append(failing)
        if len(attempts) == 1:
            raise OSError("unit: disk full")
        return _real(*args, **kwargs)
    monkeypatch.setattr(owner, failing, flaky)
    paid = {}
    with caplog.at_level(logging.ERROR):
        iterations = await _run(svc, monkeypatch, windows=1,
                                mid_window=lambda window: paid.update(_explore(svc)) if window == 1 else None)
    assert iterations == [0, 1] and "window archive failed" in caplog.text
    # The seal settled it (or never got to), then the recovery settled it: not aborted, both times.
    assert [kw for kw, _ in settled] == ([{"aborted": False}] if failing == "write_window"
                                         else [{}, {"aborted": False}])
    assert "is now settled as" not in caplog.text and "settled again with another batch" not in caplog.text
    archive = archives.pending_archives(start_window=1, end_window=1)[1]
    assert archive["window_status"] == "completed" and archive["batch"] == []
    assert archive["service_exploration_by_environment"] == {MATH: {"explorer": 1}}
    assert archive["rewards_by_hotkey"] == {"explorer": pytest.approx(_exploration_price(CAP * 0.5))}
    assert all(_money(result) == _money(archive) for _, result in settled)
    assert runtime.log.is_scanned(MATH, 9)                              # the paid first scan is kept
    statuses = [event["status"] for _, event in runtime.events(limit=1000)
                if event["id"] == paid["observation_id"] and event["type"] == "settle"]
    assert statuses[-1] == "exploration_paid" and "exploration_unpaid" not in statuses
    assert recovery.windows() == [] and svc._service_sealed_windows == set()


@pytest.mark.asyncio
async def test_a_window_that_fails_before_its_seal_is_aborted_and_pays_no_exploration(monkeypatch, tmp_path):
    """The other side of I2: only a SEALED window keeps exploration with an empty batch."""
    svc = _service(monkeypatch, tmp_path, started=service_module.time.time())
    archives = _journal(svc, monkeypatch, tmp_path)
    runtime = svc._service_runtime
    await _open(svc)
    _explore(svc)
    svc._enqueue_aborted_window(failure_stage="active", failure_type="UnitFailure")
    archive = archives.pending_archives(start_window=1, end_window=1)[1]
    assert archive["window_status"] == "aborted" and archive["rewards_by_hotkey"] == {}
    assert archive["service_exploration_by_environment"] == {}
    assert runtime.window_disposition(1) == "aborted" and not runtime.log.is_scanned(MATH, 9)
    assert svc._service_sealed_windows == set()


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


def _broken_settlement(svc, monkeypatch, *, window=1, failures=None):
    """``reconcile_archive`` of ``window`` fails ``failures`` times (None: until switched off)."""
    runtime = svc._service_runtime
    real, state = runtime.reconcile_archive, {"on": True, "calls": 0}

    def reconcile(archive, **kwargs):
        if archive["window_start"] == window:
            state["calls"] += 1
            if state["on"] and (failures is None or state["calls"] <= failures):
                raise SettlementError(f"unit: window {window} cannot be settled")
        return real(archive, **kwargs)
    monkeypatch.setattr(runtime, "reconcile_archive", reconcile)
    return state


@pytest.mark.asyncio
async def test_a_window_the_runtime_cannot_settle_is_retried_once_per_boundary_and_never_blocks_the_next(
        monkeypatch, tmp_path, caplog):
    """I4: persistent failure. The record is kept, nothing is paid or aborted, the loop goes on."""
    svc = _service(monkeypatch, tmp_path)
    archives = _journal(svc, monkeypatch, tmp_path)
    runtime = svc._service_runtime
    broken = _broken_settlement(svc, monkeypatch)
    boundary_threads = []
    real_recover = svc._recover_leftover_service_windows
    monkeypatch.setattr(svc, "_recover_leftover_service_windows", lambda target: (
        boundary_threads.append(threading.get_ident()), real_recover(target))[1])

    def mid_window(window):                                          # groups are paid during window 1
        if window == 1:
            _pay_one_group_per_env(svc)
    with caplog.at_level(logging.ERROR):
        iterations = await _run(svc, monkeypatch, windows=3, mid_window=mid_window)
    # Each boundary after window 1: one recovery attempt, back to the rotation wait, then the window opens.
    assert iterations == [0, 1, 1, 2, 2, 3]
    pending = archives.pending_archives(start_window=1, end_window=3)
    assert sorted(pending) == [2, 3]                                 # nothing was paid for window 1 ...
    assert svc._fill_closed_recovery_store.windows() == [1]          # ... and its record is kept
    assert svc._fill_closed_recovery_store.load(1)["archive"] is None
    assert runtime.window_disposition(1) is None                     # never aborted on its own
    # Seal, its recovery, the loop handler's, then ONE attempt at the boundary of window 2 and ONE at 3's.
    assert broken["calls"] == 5
    assert svc._service_recovery_attempts == {1: 3}
    # The only failed iteration is window 1's own seal: a retry that fails never fails the boundary.
    assert caplog.text.count("Window iteration failed") == 1
    assert threading.get_ident() not in boundary_threads             # in the boundary's worker thread
    assert "service window 1: recovery could not settle it (SettlementError" in caplog.text
    assert "Failed to enqueue aborted-window tombstone" in caplog.text
    assert "service window 1 was left unarchived by an earlier failure; settling it again at the boundary " \
           "of window 2" in caplog.text
    assert "service window 1: still not archived at the boundary of window 3 (SettlementError" in caplog.text
    # The rotation barrier that recovery wrote for window 1 is the one the loop waits on again.
    assert svc._fill_closed_rotation_gate is not None and svc._fill_closed_rotation_gate.source_window == 1
    # Next start (``_initialize_fill_closed_rotation_store``): the same record, the committed payments.
    broken["on"] = False
    recovered = svc._fill_closed_recovery_store.recover(
        1, queue=svc._training_payload_queue, archives=archives, rotation=svc._fill_closed_rotation_store,
        service_runtime=runtime)
    unit = CAP * 0.5 / (PICKS * SLOTS)
    assert recovered["window_status"] == "recovered_partial"
    assert recovered["rewards_by_hotkey"] == {"alice": unit, "bob": unit}
    assert sorted(archives.pending_archives(start_window=1, end_window=3)) == [1, 2, 3]
    assert svc._fill_closed_recovery_store.windows() == []


@pytest.mark.asyncio
async def test_a_transient_settlement_failure_heals_at_the_next_boundary_without_a_restart(
        monkeypatch, tmp_path, caplog):
    """I4: seal, recovery and the loop handler all fail on window 1; the boundary of window 2 archives it."""
    svc = _service(monkeypatch, tmp_path)
    archives = _journal(svc, monkeypatch, tmp_path)
    runtime = svc._service_runtime
    broken = _broken_settlement(svc, monkeypatch, failures=3)
    waits = []
    rotation_wait = AsyncMock(side_effect=lambda: (
        waits.append((svc._window_n, getattr(svc._fill_closed_rotation_gate, "source_window", None))),
        "not_armed")[1])

    def mid_window(window):
        if window == 1:
            _pay_one_group_per_env(svc)
            assert svc._fill_closed_recovery_store.windows() == [1]
        else:
            # Window 2 runs: window 1 was archived at its boundary, by this process.
            assert sorted(archives.pending_archives(start_window=1, end_window=2)) == [1]
            assert svc._fill_closed_recovery_store.windows() == [2]
    with caplog.at_level(logging.WARNING):
        iterations = await _run(svc, monkeypatch, windows=2, mid_window=mid_window, rotation_wait=rotation_wait)
    assert iterations == [0, 1, 1, 2] and broken["calls"] == 4
    assert caplog.text.count("Window iteration failed") == 1         # window 1's seal, nothing else
    assert "service window 2: the recovery of an unarchived window armed its rotation barrier; back to the " \
           "rotation wait before this window opens" in caplog.text
    pending = archives.pending_archives(start_window=1, end_window=2)
    unit = CAP * 0.5 / (PICKS * SLOTS)
    assert pending[1]["window_status"] == "recovered_partial"
    assert pending[1]["rewards_by_hotkey"] == {"alice": unit, "bob": unit}
    assert pending[2]["window_status"] == "completed"
    assert runtime.window_disposition(1) == "settled"
    assert svc._fill_closed_recovery_store.windows() == []
    assert svc._service_recovery_attempts == {} and 1 not in svc._fill_closed_assemblers
    assert svc._cooldown_durable_window >= 1
    # The existing rule is kept: the barrier recovery wrote for window 1 is waited on before window 2 opens.
    assert waits == [(0, None), (1, 1), (1, 1)]
    assert "service window 1 was left unarchived by an earlier failure" in caplog.text
    assert "still not archived" not in caplog.text


@pytest.mark.asyncio
async def test_a_sealed_window_healed_at_the_boundary_keeps_its_exploration(monkeypatch, tmp_path):
    """I2 + I4: the boundary retry knows the window sealed, as the first recovery did."""
    svc = _service(monkeypatch, tmp_path, started=service_module.time.time())
    archives = _journal(svc, monkeypatch, tmp_path)
    _broken_settlement(svc, monkeypatch, failures=3)
    await _run(svc, monkeypatch, windows=2, mid_window=lambda window: _explore(svc) if window == 1 else None)
    archive = archives.pending_archives(start_window=1, end_window=2)[1]
    assert archive["window_status"] == "completed" and archive["batch"] == []
    assert archive["rewards_by_hotkey"] == {"explorer": pytest.approx(_exploration_price(CAP * 0.5))}
    assert svc._service_sealed_windows == set()


@pytest.mark.asyncio
async def test_the_boundary_never_stops_on_the_records_of_unarchived_windows(monkeypatch, tmp_path, caplog):
    svc = _service(monkeypatch, tmp_path)
    _journal(svc, monkeypatch, tmp_path)
    assert svc._recover_leftover_service_windows(1) == ([], False)   # nothing left: nothing attempted
    monkeypatch.setattr(svc._fill_closed_recovery_store, "windows", MagicMock(side_effect=ValueError("corrupt")))
    with caplog.at_level(logging.ERROR):
        assert await svc._prepare_service_window() is True
    assert "the records of unarchived windows cannot be read (ValueError: corrupt)" in caplog.text
    assert svc._candidate_service_window["window"] == 1


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
async def test_a_crash_after_the_seal_settlement_of_an_exploration_only_window_keeps_its_disposition(
        monkeypatch, tmp_path, caplog):
    """I2 across a restart: nobody tells the next process the window sealed; the runtime remembers."""
    contract = contract_v2()
    started = service_module.time.time()
    svc = _service(monkeypatch, tmp_path, contract, started=started)
    archives = _journal(svc, monkeypatch, tmp_path)
    runtime = svc._service_runtime
    batchers = await _open(svc)
    _explore(svc)
    settled = []
    real = runtime.reconcile_archive
    monkeypatch.setattr(runtime, "reconcile_archive",
                        lambda archive, **kw: (settled.append(real(archive, **kw)), settled[-1])[1])
    monkeypatch.setattr(svc._fill_closed_recovery_store, "finish", MagicMock(side_effect=_Crash()))
    _seal(svc)
    with pytest.raises(_Crash):
        await svc._archive_window(dict(batchers), {name: ([], {}) for name in batchers})
    assert len(settled) == 1 and settled[0]["rewards_by_hotkey"] == {
        "explorer": pytest.approx(_exploration_price(CAP * 0.5))}
    runtime.close()

    restarted = _runtime(tmp_path / "service", contract, started)
    again = FillClosedRecoveryStore(tmp_path / "state")
    with caplog.at_level(logging.ERROR, logger="reliquary.services.runtime"):
        archive = again.recover(1, queue=svc._training_payload_queue, archives=archives,
                                rotation=svc._fill_closed_rotation_store, service_runtime=restarted)
    assert "settled" not in caplog.text                                  # no flip, no other batch
    assert archive["window_status"] == "completed" and _money(archive) == _money(settled[0])
    assert restarted.log.is_scanned(MATH, 9) and restarted.window_disposition(1) == "settled"
    assert archives.pending_archives(start_window=1, end_window=1)[1] == archive
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


def test_boot_closes_the_runtime_when_the_constructor_fails_later(monkeypatch, tmp_path):
    """m2: whatever fails after the runtime was opened, its SQLite handle is closed."""
    import sqlite3

    contract = _bootable_contract()
    opened, closed = [], []
    real_init, real_close = ServiceRuntime.__init__, ServiceRuntime.close
    monkeypatch.setattr(ServiceRuntime, "__init__",
                        lambda self, *a, **k: (real_init(self, *a, **k), opened.append(self))[0])
    monkeypatch.setattr(ServiceRuntime, "close", lambda self: (closed.append(self), real_close(self))[1])
    monkeypatch.setattr(service_module, "ValidatorServer", MagicMock(side_effect=RuntimeError("unit: no port")))
    with pytest.raises(RuntimeError, match="unit: no port"):
        _boot(monkeypatch, tmp_path, contract, loaded=[MATH, CODE])
    assert len(opened) == 1 and closed == opened
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].db.execute("SELECT 1")
    # The same for a failure right after the runtime (the request store) and for a boot refusal (R7).
    monkeypatch.setattr("reliquary.services.schedule.ScheduleRequestStore", MagicMock(side_effect=OSError("unit")))
    with pytest.raises(OSError, match="unit"):
        _boot(monkeypatch, tmp_path, contract, loaded=[MATH, CODE])
    assert len(opened) == 2 and closed == opened
    monkeypatch.undo()
    monkeypatch.setattr(ServiceRuntime, "close", lambda self: (closed.append(self), real_close(self))[1])
    with pytest.raises(ValueError, match="another version"):
        _boot(monkeypatch, tmp_path, contract, loaded=[MATH, CODE], installed=lambda name: "0" * 64)
    assert len(closed) == 3


@pytest.mark.parametrize("umask", [0o000, 0o002, 0o022, 0o277])
def test_the_request_folder_the_validator_creates_is_0700_whatever_the_umask(tmp_path, umask):
    """I5: the request store refuses a folder group or others can write."""
    target = tmp_path / "state" / "service-policies" / "task" / "run"
    target.parent.mkdir(parents=True)
    previous = os.umask(umask)
    try:
        assert service_module._service_request_folder(target) == target
    finally:
        os.umask(previous)
    assert target.stat().st_mode & 0o7777 == 0o700
    store = ScheduleRequestStore(target)
    assert store.take() is None                                      # safe: read without a refusal


def test_boot_creates_the_request_folder_0700_and_takes_requests_from_it(monkeypatch, tmp_path):
    contract = _bootable_contract()
    previous = os.umask(0o002)                                       # the umask of this machine's operator
    try:
        svc = _boot(monkeypatch, tmp_path, contract, loaded=[MATH, CODE])
    finally:
        os.umask(previous)
    folder = tmp_path / "state" / "service-policies" / "task" / "run"
    assert folder.stat().st_mode & 0o7777 == 0o700 and svc._service_schedule_store.folder == folder
    _request(svc, cooldowns={MATH: 9})
    schedule = svc._service_runtime.apply_pending_schedule_request(svc._service_schedule_store, window=1)
    assert schedule.revision == 1 and svc._service_schedule_store.status()["status"] == "applied"
    svc._service_runtime.close()


@pytest.mark.parametrize("mode, refused", [(0o775, True), (0o757, True), (0o755, False), (0o700, False)])
def test_boot_never_chmods_an_existing_request_folder_and_refuses_an_unsafe_one_once(
        monkeypatch, tmp_path, caplog, mode, refused):
    """I5: an existing folder is the operator's. Unsafe: logged once per boot, requests refused."""
    contract = _bootable_contract()
    folder = _persisted_folder(tmp_path)
    folder.chmod(mode)
    with caplog.at_level(logging.ERROR, logger="reliquary.validator.service"):
        svc = _boot(monkeypatch, tmp_path, contract, loaded=[MATH, CODE])
        assert folder.stat().st_mode & 0o7777 == mode                # not ours to change
        _request(svc, cooldowns={MATH: 9})
        for window in (1, 2, 3):                                     # three boundaries, one boot
            schedule = svc._service_runtime.apply_pending_schedule_request(svc._service_schedule_store, window=window)
    assert caplog.text.count("service request folder refused") == (1 if refused else 0)
    if refused:
        assert f"is writable by group or others (mode {mode:04o})" in caplog.text
        assert schedule.revision == 0                                # the request is never read from there
    else:
        assert schedule.revision == 1 and svc._service_schedule_store.status()["status"] == "applied"
    assert folder.stat().st_mode & 0o7777 == mode
    svc._service_runtime.close()


def test_a_legacy_task_still_boots_without_drand(monkeypatch):
    """Only a service task needs drand window randomness."""
    from reliquary.validator.service import ValidationService
    from tests.unit.test_service_v2 import _LateDropFakeEnv, _LateDropFakeWallet

    tokenizer = MagicMock()
    tokenizer.eos_token_id = 99
    svc = ValidationService(wallet=_LateDropFakeWallet(), model=MagicMock(), tokenizer=tokenizer,
                            env=_LateDropFakeEnv(), netuid=99, use_drand=False)
    assert svc.use_drand is False and svc._service_runtime is None and svc._service_schedule_store is None
    assert svc._service_sealed_windows == set() and svc._service_recovery_attempts == {}
