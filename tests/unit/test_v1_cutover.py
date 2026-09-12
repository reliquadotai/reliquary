"""Cutover invariants: no next window, final publication, paid crash recovery."""

import asyncio
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from reliquary import constants as C
from reliquary.infrastructure.archive_queue import ArchiveQueue
from reliquary.infrastructure.training_payload_queue import TrainingPayloadQueue
from reliquary.trainer.train_runner import TrainerStateUnsafe, TrainRunner
from reliquary.validator.control import ControlStore
from reliquary.validator.fill_closed_recovery import FillClosedRecoveryStore
from reliquary.validator.fill_closed_rotation import FillClosedRotationStore
from tests.unit.test_trainer_train_runner import _decoded
from tests.unit.test_trainer_worker import _Decoded, _Env, _worker


def test_local_control_requires_fresh_matching_acknowledgement(tmp_path):
    store = ControlStore(tmp_path, start_closed=True)
    assert store.request()["mode"] == "drain"
    request = store.set_mode("drain")
    store.report(request, phase="drained")
    assert store.status()["drained"]
    store.set_mode("run")
    assert not store.status()["fresh"]
    assert ControlStore(tmp_path, start_closed=True).request()["mode"] == "run"
    store.path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError):
        store.request()


def test_trainer_drain_publishes_below_cadence_and_stops_at_target(tmp_path):
    store = ControlStore(tmp_path)
    store.set_mode("drain", target_cursor=102)
    env = _Env({101: ("payload", _Decoded(101)),
                102: ("tombstone", {}), 103: ("payload", _Decoded(103))})
    worker = _worker(env, publish_every=16, drain_request_fn=store.request, finish_fn=lambda: False)
    assert worker.run_once() == "trained"
    assert worker.run_once() == "tombstone"
    env.head = None
    with pytest.raises(RuntimeError, match="HEAD unavailable"):
        worker.run_once()
    assert not env.published and worker.cursor == 102
    env.head = "rev-0"
    assert worker.run_once() == "published"
    assert env.published == ["cutover_drain"]
    assert worker.run_once() == worker.run_once() == "drained"
    assert worker.cursor == 102 and len(env.trained) == 1
    store.set_mode("run")
    assert worker.run_once() == "trained" and worker.cursor == 103


def test_retry_of_pending_publication_preserves_live_training_state():
    env = _Env({101: ("payload", _Decoded(101))})
    pending = [False]
    worker = _worker(env, publish_every=1, publication_pending_fn=lambda: pending[0])

    def publish(reason):
        if not pending[0]:
            pending[0] = True
            env.head = "exact-committed-revision"
            raise OSError("R2 upload interrupted after HF committed")
        # Production publisher verifies this exact transaction and its parent.
        assert env.head == "exact-committed-revision"
        pending[0] = False
        return env.head

    worker._publish_fn = publish
    assert worker.run_once() == "trained"
    with pytest.raises(OSError):
        worker.run_once()
    assert worker.cursor == 101 and worker.trained_since_publish == 1
    assert worker.run_once() == "published"
    assert worker.last_published_revision == env.head
    assert len(env.trained) == 1


def test_tombstone_only_drain_preserves_restored_lr_position(monkeypatch):
    from reliquary.trainer.cli import _publication_lr_step
    from reliquary.validator import training

    monkeypatch.setattr(training, "current_lr_schedule_step", lambda: None)
    assert _publication_lr_step(123) == 123
    monkeypatch.setattr(training, "current_lr_schedule_step", lambda: 124)
    assert _publication_lr_step(123) == 124


def test_trainer_flushes_balanced_partial_and_halts_on_optimizer_failure(monkeypatch):
    monkeypatch.setattr(C, "KL_BETA", 0.0)
    monkeypatch.setattr(C, "RECOMPUTE_PI_OLD_FROM_VERIFY", False)
    calls = []
    runner = TrainRunner(object(), env_targets={"openmathinstruct": 16, "opencodeinstruct": 16},
                         env_order=["openmathinstruct", "opencodeinstruct"],
                         train_step_fn=lambda model, batches, **kw: calls.append(batches) or model)
    assert not runner.step(_decoded())
    assert runner.finish() and len(calls) == 1
    assert not runner.finish()
    assert not any(runner.snapshot()["accumulator"]["counts"].values())

    def failed(model, batches, **kw):
        raise RuntimeError("partial optimizer mutation")

    runner._train_step = failed
    runner.step(_decoded())
    with pytest.raises(TrainerStateUnsafe):
        runner.finish()


def _recovery_setup(tmp_path, monkeypatch, picks=16):
    import reliquary.infrastructure.training_payload_queue as queue_module
    import reliquary.validator.fill_closed_recovery as recovery_module

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(queue_module, "FILL_CLOSED_EMISSIONS_PER_WINDOW", 16)
    monkeypatch.setattr(recovery_module, "FILL_CLOSED_EMISSIONS_PER_WINDOW", 16)
    monkeypatch.setattr(recovery_module, "FILL_CLOSED_PICKS_PER_WINDOW", picks)
    store = FillClosedRecoveryStore(tmp_path)
    store.begin(42, checkpoint_n=7, revision="a" * 40, targets={"math": 16, "code": 16})
    queue = TrainingPayloadQueue(str(tmp_path / "payloads"))
    archives = ArchiveQueue(str(tmp_path / "archives"))
    rotation = FillClosedRotationStore(tmp_path)
    return store, queue, archives, rotation


def test_crash_after_payload_upload_keeps_paid_groups_and_pads_only_unwritten_slots(tmp_path, monkeypatch):
    store, queue, archives, rotation = _recovery_setup(tmp_path, monkeypatch)
    rows = [{"env_name": env, "batch_index": 0, "hotkey": hotkey,
             "prompt_idx": 12, "eos_tokens": tokens,
             "claimed_checkpoint_hash": "a" * 40}
            for env, hotkey, tokens in [("math", "alice", 30), ("math", "bob", 10), ("code", "bob", 4)]]
    # The body was uploaded/deleted before the assembler could credit RAM.
    payload = queue.enqueue_committed_payload(42 * 16, b"committed-training-body", accounting=rows)
    payload.unlink()
    # The next body never acquired a receipt and must never reach the trainer.
    orphan = queue._journal_commit_dir / f"window-{42 * 16 + 1}.payload.body"
    orphan.write_bytes(b"uncommitted")
    store.quarantine_uncommitted(queue.queue_dir)
    restarted = TrainingPayloadQueue(str(queue.queue_dir))
    store.recover(42, queue=restarted, archives=archives, rotation=rotation)
    archive = archives.pending_archives(start_window=42, end_window=42)[42]
    assert archive["window_status"] == "recovered_partial"
    assert archive["batch"] == rows
    assert archive["rewards_by_hotkey"] == {"alice": 0.75 / 32, "bob": 1.25 / 32}
    assert len(list(queue._journal_commit_dir.glob("window-*.json"))) == 16
    assert not store.windows()
    assert rotation.load().required_journal_key == 42 * 16 + 15
    assert not rotation.load().requires_successor
    first = json.loads((queue._journal_commit_dir / f"window-{42 * 16}.json").read_text())
    assert first["kind"] == "payload"


def test_crash_during_archive_enqueue_replays_exact_archive(tmp_path, monkeypatch):
    store, queue, archives, rotation = _recovery_setup(tmp_path, monkeypatch)
    archive = {"window_start": 42, "batch": [], "window_status": "completed"}
    broken = SimpleNamespace(enqueue=lambda *args: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        store.finish(42, archive, broken)
    FillClosedRecoveryStore(tmp_path).recover(42, queue=queue, archives=archives, rotation=rotation)
    assert archives.pending_archives(start_window=42, end_window=42)[42] == archive
    assert not store.windows()


def test_quarantined_training_still_preserves_miner_payment_after_crash(tmp_path, monkeypatch):
    store, queue, archives, rotation = _recovery_setup(tmp_path, monkeypatch)
    rows = [{"env_name": env, "batch_index": 0, "hotkey": "alice",
             "prompt_idx": 1, "eos_tokens": 16, "claimed_checkpoint_hash": "a" * 40}
            for env in ("math", "code")]
    queue.enqueue_committed_tombstone(672, b"training-quarantine", accounting=rows)
    store.recover(42, queue=queue, archives=archives, rotation=rotation)
    archive = archives.pending_archives(start_window=42, end_window=42)[42]
    assert archive["rewards_by_hotkey"] == {"alice": 1 / 16}
    assert archive["durable_payload_count"] == 0
    assert not rotation.load().requires_successor


def test_validator_drain_finishes_pipeline_before_acknowledging(tmp_path, monkeypatch):
    from reliquary.validator.service import ValidationService

    monkeypatch.setattr(C, "WRITE_TRAINING_PAYLOADS", True)
    control = ControlStore(tmp_path)
    control.set_mode("drain")
    uploads = {"depth": 1}
    calls = []

    async def finish(**kwargs):
        calls.append(kwargs)

    async def no_sleep(_):
        pass

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    service = SimpleNamespace(
        _control_store=control, _window_n=42,
        server=SimpleNamespace(set_active_batchers=lambda batchers: calls.append(batchers)),
        _set_state=lambda state: None,
        _gpu_backlog=({"math": object()}, 42, None, {}, {}),
        _archive_queue=SimpleNamespace(snapshot=lambda: uploads),
        _training_payload_queue_ref=lambda: SimpleNamespace(snapshot=lambda: {"depth": 0}),
        _train_and_publish=finish,
    )
    assert asyncio.run(ValidationService._pause_for_control_drain(service))
    assert service._gpu_backlog is None and calls[1]["window_n"] == 42
    assert not control.status()["drained"]
    uploads["depth"] = 0
    assert asyncio.run(ValidationService._pause_for_control_drain(service))
    assert control.status()["drained"]


@pytest.mark.parametrize("crash_stage", ["body", "receipt", "visible"])
def test_process_death_at_journal_commit_boundaries(tmp_path, crash_stage):
    # Actual os._exit: no Python finally handlers or in-memory assembler state
    # survive. Each recovery runs in another fresh interpreter.
    environment = {key: value for key, value in os.environ.items() if not key.startswith("RELIQUARY_")}
    environment.update(RELIQUARY_PROTOCOL_PROFILE="qwen3-4b-base-dapo-fill-closed-v6",
                       RELIQUARY_EXPERIMENTAL_FILL_CLOSED_ENABLED="1")
    common = '''
import os, sys
from pathlib import Path
from reliquary.infrastructure.archive_queue import ArchiveQueue
from reliquary.infrastructure.training_payload_queue import TrainingPayloadQueue
from reliquary.validator.fill_closed_recovery import FillClosedRecoveryStore
from reliquary.validator.fill_closed_rotation import FillClosedRotationStore
root=Path(sys.argv[1]); store=FillClosedRecoveryStore(root)
'''
    crash = common + '''
store.begin(42, checkpoint_n=7, revision="a"*40, targets={"math":16,"code":16})
queue=TrainingPayloadQueue(str(root/"payloads"))
stage=sys.argv[2]
write=queue._enqueue_durable
def die_after_write(name, data):
    result=write(name,data)
    if (stage=="body" and name.endswith(".body")) or (stage=="receipt" and name.endswith(".json")):
        os._exit(91)
    return result
queue._enqueue_durable=die_after_write
rows=[{"env_name":e,"batch_index":0,"hotkey":"alice","prompt_idx":1,"eos_tokens":4,"claimed_checkpoint_hash":"a"*40} for e in ("math","code")]
queue.enqueue_committed_payload(672,b"training-body",accounting=rows)
os._exit(91)
'''
    result = subprocess.run([sys.executable, "-c", crash, str(tmp_path), crash_stage], env=environment,
                            capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 91, result.stderr
    recover = common + '''
store.quarantine_uncommitted(root/"payloads")
queue=TrainingPayloadQueue(str(root/"payloads"))
archives=ArchiveQueue(str(root/"archives"))
store.recover(42,queue=queue,archives=archives,rotation=FillClosedRotationStore(root))
archive=archives.pending_archives(start_window=42,end_window=42)[42]
expected={} if sys.argv[2]=="body" else {"alice":1/16}
assert archive["rewards_by_hotkey"]==expected, archive
assert len(list(queue._journal_commit_dir.glob("window-*.json")))==16
assert not store.windows()
'''
    result = subprocess.run([sys.executable, "-c", recover, str(tmp_path), crash_stage], env=environment,
                            capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr


def test_live_accounting_commit_and_abort_release_stale_assembler(tmp_path, monkeypatch):
    from reliquary.validator.fill_closed_recovery import accounting_rows
    from reliquary.validator.service import ValidationService

    store, queue, archives, rotation = _recovery_setup(tmp_path, monkeypatch)
    group = SimpleNamespace(hotkey="alice", prompt_idx=12, sigma=1.0,
        eos_tokens=30, claimed_checkpoint_hash="a" * 40,
        merkle_root_bytes=b"m" * 32, selection_digest=b"s" * 32,
        rollout_hashes=[b"h" * 32],
        rollouts=[SimpleNamespace(commit={"tokens": [1, 2]}, reward=1.0)])
    rows = accounting_rows({"math": [group]}, batch_index=0)
    queue.enqueue_committed_payload(42 * 16, b"training-body", accounting=rows)
    receipt_path = queue._journal_commit_dir / f"window-{42 * 16}.json"
    original = receipt_path.read_bytes()
    assert json.loads(original)["accounting"][0]["rollouts"][0]["hash"] == (b"h" * 32).hex()
    monkeypatch.setattr("reliquary.validator.service.FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr("reliquary.infrastructure.archive_queue.get_archive_queue", lambda: archives)
    stale = SimpleNamespace(window_start=42)
    svc = SimpleNamespace(_active_batchers={"math": SimpleNamespace(window_start=42)},
        _archive_enqueued_windows=set(), _fill_closed_recovery_store=store,
        _training_payload_queue_ref=lambda: queue, _fill_closed_rotation_store=rotation,
        _fill_closed_assemblers={42: stale}, _fill_closed_assembler=stale)
    ValidationService._enqueue_aborted_window(svc, failure_stage="active", failure_type="RuntimeError")
    assert svc._fill_closed_assembler is None and not svc._fill_closed_assemblers
    assert receipt_path.read_bytes() == original
    assert archives.pending_archives(start_window=42, end_window=42)[42]["batch"] == rows


@pytest.mark.parametrize('legacy', [False, True])
def test_recovery_preserves_original_payment_denominator_and_journal_stride(tmp_path, monkeypatch, legacy):
    import reliquary.validator.fill_closed_recovery as recovery
    from reliquary.validator.control import write_json
    monkeypatch.setattr(recovery, 'FILL_CLOSED_PICKS_PER_WINDOW', 10)
    store, queue, archives, rotation = _recovery_setup(tmp_path, monkeypatch, picks=10)
    if legacy:
        record = store.load(42)
        record.pop('picks_target')
        write_json(store._path(42), record)
    rows = [{'env_name': env, 'batch_index': 0, 'hotkey': 'alice', 'prompt_idx': 1,
             'eos_tokens': 16, 'claimed_checkpoint_hash': 'a' * 40} for env in ('math', 'code')]
    queue.enqueue_committed_payload(42 * 16, b'committed-training-body', accounting=rows)
    store.recover(42, queue=queue, archives=archives, rotation=rotation)
    record = archives.pending_archives(start_window=42, end_window=42)[42]
    assert record['rewards_by_hotkey'] == {'alice': 1 / (16 if legacy else 10)}
    assert len(list(queue._journal_commit_dir.glob('window-*.json'))) == 16
    assert rotation.load().required_journal_key == 42 * 16 + 15


def test_recovery_bridge_accepts_v6_record_and_keeps_v1_writer(tmp_path, monkeypatch):
    import reliquary.validator.fill_closed_recovery as recovery
    from reliquary.validator.control import write_json

    monkeypatch.setattr(recovery, "B_BATCH", 16)
    store, queue, archives, rotation = _recovery_setup(tmp_path, monkeypatch, picks=10)
    active = store.load(42)
    assert active["schema_version"] == 1
    assert not {"journal_slots", "payment_policy", "selection_policy", "window_pool"} & active.keys()

    active.update(
        schema_version=2,
        journal_slots=16,
        payment_policy="fixed-selected-group/v1",
        selection_policy="fifo-ingress/v1",
        window_pool=0.8,
    )
    for field, invalid, error in (
        ("payment_policy", "wrong", "payment policy"),
        ("selection_policy", "wrong", "selection policy"),
        ("journal_slots", 15, "journal stride"),
        ("window_pool", -0.1, "window pool"),
    ):
        malformed = {**active, field: invalid}
        write_json(store._path(42), malformed)
        with pytest.raises(ValueError, match=error):
            store.load(42)

    write_json(store._path(42), active)
    rows = [
        {"env_name": env, "batch_index": 0, "hotkey": hotkey,
         "prompt_idx": index, "eos_tokens": tokens,
         "claimed_checkpoint_hash": "a" * 40}
        for index, (env, hotkey, tokens) in enumerate((
            ("math", "alice", 30), ("math", "bob", 0), ("code", "bob", 4)
        ))
    ]
    queue.enqueue_committed_payload(42 * 16, b"body", accounting=rows)
    store.recover(42, queue=queue, archives=archives, rotation=rotation)

    archive = archives.pending_archives(start_window=42, end_window=42)[42]
    share = 0.8 / 2 / 10 / 16
    assert archive["rewards_by_hotkey"] == pytest.approx({"alice": share, "bob": 2 * share})
    assert archive["batch"] == [
        {**row, "selected_for_batch": True, "rewarded": True,
         "payment_source": "fill_closed_fixed_group"}
        for row in rows
    ]
    assert {key: archive[key] for key in (
        "journal_slots", "payment_policy", "selection_policy", "window_pool"
    )} == {
        "journal_slots": 16,
        "payment_policy": "fixed-selected-group/v1",
        "selection_policy": "fifo-ingress/v1",
        "window_pool": 0.8,
    }


def test_control_heartbeat_never_acknowledges_a_new_request(tmp_path):
    from reliquary.validator.control import ControlStore
    store = ControlStore(tmp_path)
    run = store.set_mode('run')
    store.report(run, phase='running', window=10)
    store.heartbeat(window=11)
    assert store.status()['fresh'] and store.status()['status']['window'] == 11
    store.set_mode('drain')
    store.heartbeat(window=11)
    assert not store.status()['fresh'] and not store.status()['drained']
