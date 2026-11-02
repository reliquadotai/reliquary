"""Archive assembly retains its bytes without starving the serving loop."""

import asyncio
import contextvars
import gzip
import functools
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.unit.test_archive_window_content import (
    _FakeEnv,
    _FakeWallet,
    _valid_submission,
)


def setup_archive(monkeypatch, tmp_path, *, env=None, fill=False):
    import reliquary.validator.service as module
    from reliquary.infrastructure.archive_queue import ArchiveQueue

    monkeypatch.setattr(module, "FILL_CLOSED_ENABLED", fill)
    tok = MagicMock()
    tok.eos_token_id = 99
    svc = module.ValidationService(
        wallet=_FakeWallet(),
        model=MagicMock(),
        tokenizer=tok,
        env=env or _FakeEnv(),
        netuid=99,
    )
    svc._utility_telemetry.write_window = MagicMock()
    svc.kl_reference_state = {
        **svc.kl_reference_state,
        "dtype": "float32",
        "device": "cpu",
    }
    svc._cache_archived_hashes = MagicMock()
    svc.server.complete_final_verdict_window = MagicMock()
    svc._record_auction_final_verdicts = AsyncMock()
    sub = _valid_submission(prompt_idx=7)
    events = []
    batcher = SimpleNamespace(
        window_start=500,
        randomness="abcd",
        window_opened_at=0.0,
        reject_counts={},
        rejected_submissions=[],
        valid_submissions=lambda: [sub],
        prepare_fill_closed_paid_side_effects=lambda rows: events.append("prepare"),
        commit_seal_side_effects=lambda: events.append("commit"),
    )
    assembler = SimpleNamespace(
        window_start=500,
        payment_policy="fixed-selected-group-v1",
        picks_target=16,
        window_pool=1.0,
        close=lambda **kw: events.append("close"),
        paid_groups=lambda: {"fake": [(0, sub)]},
        reward_map=lambda: {"hk": 1.0},
        pool_for=lambda name: 1.0,
    )
    if fill:
        svc._fill_closed_assemblers[500] = assembler
        svc._fill_closed_assembler = assembler
    queue = ArchiveQueue(queue_dir=str(tmp_path / "queue"))
    monkeypatch.setattr(
        "reliquary.infrastructure.archive_queue.get_archive_queue", lambda: queue
    )
    return svc, batcher, sub, queue, events


class BlockingStep:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.responsive = False

    def __call__(self):
        self.started.set()
        self.responsive = self.release.wait(1.0)

    async def pulse(self):
        while not self.started.is_set():
            await asyncio.sleep(0)
        self.release.set()


def read_archive(queue):
    files = queue._pending()
    assert len(files) == 1
    return json.loads(gzip.decompress(files[0].read_bytes()))


@pytest.mark.asyncio
async def test_legacy_dataset_read_keeps_serving_loop_responsive(monkeypatch, tmp_path):
    block = BlockingStep()

    class Slow(_FakeEnv):
        def get_problem(self, index):
            block()
            return super().get_problem(index)

    svc, batcher, sub, queue, _ = setup_archive(monkeypatch, tmp_path, env=Slow())
    pulse = asyncio.create_task(block.pulse())
    await svc._archive_window(batcher, [sub])
    await pulse
    assert block.responsive
    assert read_archive(queue)["batch"][0]["prompt"] == "question 7"


@pytest.mark.asyncio
@pytest.mark.parametrize("fill", [False, True])
@pytest.mark.parametrize("phase", ["compress", "fsync"])
async def test_durable_enqueue_keeps_serving_loop_responsive(
    monkeypatch, tmp_path, fill, phase
):
    import reliquary.infrastructure.archive_queue as module

    svc, batcher, sub, queue, events = setup_archive(monkeypatch, tmp_path, fill=fill)
    block = BlockingStep()
    original = module.gzip.compress if phase == "compress" else module.os.fsync

    def slow(*args, **kwargs):
        if not block.started.is_set():
            block()
        return original(*args, **kwargs)

    monkeypatch.setattr(
        module.gzip if phase == "compress" else module.os,
        "compress" if phase == "compress" else "fsync",
        slow,
    )
    if fill:

        def finish(window, archive, output):
            events.append("finish")
            output.enqueue(window, archive)

        svc._fill_closed_recovery_store = SimpleNamespace(finish=finish)
    pulse = asyncio.create_task(block.pulse())
    await svc._archive_window(batcher, [sub])
    await pulse
    assert block.responsive
    assert 500 in svc._archive_enqueued_windows
    assert read_archive(queue)["batch"][0]["prompt"] == "question 7"
    if fill:
        assert events == ["close", "prepare", "commit", "finish"]


async def wait_started(block):
    async def wait():
        while not block.started.is_set():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("fill", [False, True])
async def test_cancellation_waits_for_enqueue_and_marks_only_committed_archive(
    monkeypatch, tmp_path, fill
):
    svc, batcher, sub, queue, events = setup_archive(monkeypatch, tmp_path, fill=fill)
    block = BlockingStep()
    original = queue.enqueue
    loop_thread = threading.get_ident()

    class LoopOwnedSet(set):
        def add(self, value):
            assert threading.get_ident() == loop_thread
            super().add(value)

    svc._archive_enqueued_windows = LoopOwnedSet()

    def enqueue(*args):
        block()
        original(*args)

    queue.enqueue = enqueue
    if fill:
        svc._fill_closed_recovery_store = SimpleNamespace(
            finish=lambda window, archive, output: output.enqueue(window, archive)
        )
    task = asyncio.create_task(svc._archive_window(batcher, [sub]))
    await wait_started(block)
    task.cancel("first stop")
    await asyncio.sleep(0)
    task.cancel("second stop")
    await asyncio.sleep(0.01)
    assert not task.done()
    assert 500 not in svc._archive_enqueued_windows
    assert queue._pending() == []
    if fill:
        assert 500 in svc._fill_closed_assemblers
    block.release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 3)
    assert 500 in svc._archive_enqueued_windows
    assert read_archive(queue)["window_start"] == 500
    if fill:
        assert 500 not in svc._fill_closed_assemblers


@pytest.mark.asyncio
@pytest.mark.parametrize("fill", [False, True])
async def test_enqueue_failure_preserves_error_and_recovery_owner(
    monkeypatch, tmp_path, fill
):
    svc, batcher, sub, queue, events = setup_archive(monkeypatch, tmp_path, fill=fill)
    error = OSError("synthetic durable write unavailable")

    def fail(*args):
        raise error

    queue.enqueue = fail
    if fill:
        svc._fill_closed_recovery_store = SimpleNamespace(
            finish=lambda w, a, q: q.enqueue(w, a)
        )
    with pytest.raises(OSError) as captured:
        await svc._archive_window(batcher, [sub])
    assert captured.value is error
    assert 500 not in svc._archive_enqueued_windows
    assert queue._pending() == []
    if fill:
        assert 500 in svc._fill_closed_assemblers


@pytest.mark.asyncio
async def test_concurrent_flusher_never_observes_partial_archive(monkeypatch, tmp_path):
    import reliquary.infrastructure.archive_queue as module

    svc, batcher, sub, queue, events = setup_archive(monkeypatch, tmp_path)
    queue.enqueue(499, {"window_start": 499, "task_id": "synthetic"})
    old_path = queue._pending()[0]
    uploaded = []

    def put(bucket, key, body, *credentials):
        uploaded.append(json.loads(gzip.decompress(body))["window_start"])

    monkeypatch.setattr("reliquary.infrastructure.storage._sync_boto3_put", put)
    block = BlockingStep()
    original = module.os.fsync

    def slow(fd):
        if not block.started.is_set():
            block()
        return original(fd)

    monkeypatch.setattr(module.os, "fsync", slow)
    task = asyncio.create_task(svc._archive_window(batcher, [sub]))
    await wait_started(block)
    assert queue._pending() == [old_path]
    assert await queue._try_upload(old_path)
    assert uploaded == [499] and queue._pending() == []
    block.release.set()
    await task
    assert read_archive(queue)["window_start"] == 500
    assert not list(queue.queue_dir.glob("*.tmp"))


@pytest.mark.asyncio
async def test_seal_reserve_close_commit_order_and_cancellation(monkeypatch, tmp_path):
    from reliquary.validator.service import _complete_archive_commit

    svc, batcher, sub, queue, events = setup_archive(monkeypatch, tmp_path, fill=True)
    svc._active_batchers = {"fake": batcher}
    block = BlockingStep()

    def arm(*, reserve_full_range=False):
        events.append("reserve" if reserve_full_range else "final-gate")

    def close(**kwargs):
        events.append("close")
        block()

    svc._arm_fill_closed_rotation_gate = arm
    svc._fill_closed_assembler.close = close
    task = asyncio.create_task(
        _complete_archive_commit(
            functools.partial(svc._commit_fill_closed_seal, "completed")
        )
    )
    await wait_started(block)
    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done() and events == ["reserve", "close"]
    block.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert events == ["reserve", "close", "prepare", "commit", "final-gate"]
    assert svc._active_batchers == {"fake": batcher}


@pytest.mark.asyncio
async def test_cancelled_commit_failure_does_not_hide_durable_error(
    monkeypatch, tmp_path
):
    from reliquary.validator.service import _complete_archive_commit

    block = BlockingStep()
    error = OSError("synthetic journal unavailable")

    def operation():
        block()
        raise error

    task = asyncio.create_task(_complete_archive_commit(operation))
    await wait_started(block)
    task.cancel()
    block.release.set()
    with pytest.raises(OSError) as captured:
        await task
    assert captured.value is error


def install_recovery(svc, monkeypatch, tmp_path):
    import reliquary.validator.service as module
    from reliquary.validator.fill_closed_recovery import FillClosedRecoveryStore

    assembler = svc._fill_closed_assembler
    assembler.payment_policy = module.FIXED_GROUP_PAYMENT_POLICY
    from reliquary.constants import FILL_CLOSED_PICKS_PER_WINDOW

    assembler.picks_target = FILL_CLOSED_PICKS_PER_WINDOW
    store = FillClosedRecoveryStore(str(tmp_path / "recovery"))
    store.begin(
        500,
        checkpoint_n=7,
        revision="a" * 40,
        targets={"fake": module.B_BATCH},
        window_pool=1.0,
    )
    svc._fill_closed_recovery_store = store
    return store


@pytest.mark.asyncio
async def test_real_recovery_finish_is_durable_before_cancellation(
    monkeypatch, tmp_path
):
    import reliquary.validator.fill_closed_recovery as module

    svc, batcher, sub, queue, events = setup_archive(monkeypatch, tmp_path, fill=True)
    store = install_recovery(svc, monkeypatch, tmp_path)
    block = BlockingStep()
    original = module.os.fsync

    def slow(fd):
        if not block.started.is_set():
            block()
        return original(fd)

    monkeypatch.setattr(module.os, "fsync", slow)
    task = asyncio.create_task(svc._archive_window(batcher, [sub]))
    await wait_started(block)
    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done() and store._path(500).exists()
    assert 500 not in svc._archive_enqueued_windows
    block.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not store._path(500).exists()
    assert read_archive(queue)["window_start"] == 500
    assert 500 in svc._archive_enqueued_windows
    assert 500 not in svc._fill_closed_assemblers


@pytest.mark.asyncio
async def test_real_recovery_keeps_exact_archive_for_failed_enqueue_retry(
    monkeypatch, tmp_path
):
    svc, batcher, sub, queue, events = setup_archive(monkeypatch, tmp_path, fill=True)
    store = install_recovery(svc, monkeypatch, tmp_path)
    original = queue.enqueue
    error = OSError("synthetic queue unavailable")

    def fail(*args):
        raise error

    queue.enqueue = fail
    with pytest.raises(OSError) as captured:
        await svc._archive_window(batcher, [sub])
    assert captured.value is error
    saved = store.load(500)["archive"]
    assert saved["window_start"] == 500
    assert 500 not in svc._archive_enqueued_windows
    assert 500 in svc._fill_closed_assemblers
    queue.enqueue = original
    await svc._enqueue_window_archive(500, saved, svc._fill_closed_assembler)
    assert read_archive(queue) == saved
    assert not store._path(500).exists()
    assert 500 in svc._archive_enqueued_windows
    assert 500 not in svc._fill_closed_assemblers


@pytest.mark.asyncio
async def test_archive_commit_preserves_task_context():
    from reliquary.validator.service import _complete_archive_commit

    task = contextvars.ContextVar("synthetic_archive_task")
    token = task.set("task-a")
    try:
        assert await _complete_archive_commit(task.get) == "task-a"
    finally:
        task.reset(token)


@pytest.mark.asyncio
async def test_offloaded_enqueue_preserves_compact_archive_bytes(tmp_path):
    from reliquary.infrastructure.archive_queue import ArchiveQueue
    from reliquary.validator.service import _complete_archive_commit

    queue = ArchiveQueue(str(tmp_path))
    payload = {
        "window_start": 500,
        "batch": [{"prompt": "∑ x", "ground_truth": None, "reward": 0.5}],
        "window_status": "completed",
    }
    path = await _complete_archive_commit(functools.partial(queue.enqueue, 500, payload))

    assert gzip.decompress(path.read_bytes()) == (
        b'{"window_start":500,"batch":[{"prompt":"\\u2211 x",'
        b'"ground_truth":null,"reward":0.5}],"window_status":"completed"}'
    )
