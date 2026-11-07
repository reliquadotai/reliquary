"""An audit executor downloads to the pod's volume and refuses a disk that cannot hold the checkpoint."""

import asyncio
from types import SimpleNamespace

import pytest

from reliquary.validator import corpus_audit_executor as ex


def _info(*sizes):
    return SimpleNamespace(siblings=[SimpleNamespace(size=s) for s in sizes])


def test_a_checkpoint_that_does_not_fit_is_refused():
    gb = 10**9
    with pytest.raises(ex.DiskTooSmall, match="needs 59 GB"):
        ex._check_disk("org/m", "a" * 40, "/x", info=_info(40 * gb, 15 * gb), free=24 * gb)
    ex._check_disk("org/m", "a" * 40, "/x", info=_info(40 * gb, 15 * gb), free=300 * gb)


def test_the_volume_holds_the_cache_unless_the_operator_chose_one(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setattr(__import__("os").path, "ismount", lambda p: True)
    monkeypatch.setattr(__import__("os"), "access", lambda p, mode: True)
    assert ex._cache_dir("/workspace") == "/workspace/hf/hub"
    monkeypatch.setenv("HF_HOME", "/data/hf")
    assert ex._cache_dir("/workspace") is None
    monkeypatch.delenv("HF_HOME")
    monkeypatch.setattr(__import__("os").path, "ismount", lambda p: False)
    assert ex._cache_dir("/workspace") is None


async def test_a_pod_without_room_goes_silent_after_its_first_heartbeat():
    posts = []

    class Http:
        async def post(self, path, json=None, **kw):
            posts.append(path)
            return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                                   json=lambda: {"model_id": "m", "model_revision": "r"})

    def full_disk(model_id, revision):
        raise ex.DiskTooSmall("m@r needs 59 GB at /root, 2 GB free")

    executor = ex.AuditExecutor(http=Http(), executor_id="pod-1", token="x", model_id="m",
                                model_revision="r", load_model=full_disk, heartbeat_seconds=0.01)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(executor.run(), timeout=0.3)
    assert [p for p in posts if p.endswith("/heartbeat")] == [posts[0]] and len(posts) == 1


async def test_a_slow_load_keeps_heartbeating():
    import threading
    import time

    posts = []

    class Http:
        async def post(self, path, json=None, **kw):
            posts.append((path, json))
            return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                                   json=lambda: {"model_id": "m", "model_revision": "r"})

    released = threading.Event()

    def slow(model_id, revision):
        released.wait(2)
        return "model"

    executor = ex.AuditExecutor(http=Http(), executor_id="pod-1", token="x", model_id="m",
                                model_revision="r", load_model=slow, heartbeat_seconds=0.02)
    start = asyncio.ensure_future(executor.start())
    await asyncio.sleep(0.15)
    released.set()
    await start
    beats = [body for path, body in posts if path.endswith("/heartbeat")]
    assert len(beats) >= 3 and executor._model == "model"
    assert beats[-1]["detail"]["loaded"] is False


def test_a_download_is_too_slow_once_its_rate_cannot_end_within_the_budget():
    gb = 10**9
    # 54 GB at 30 MB/s needs 30 min: past a 25 min budget. Before the probe ends, never judged.
    assert ex.too_slow(0, (0, 0), 180, int(5.4 * gb), 54 * gb, probe=120, budget=1500)
    assert not ex.too_slow(0, (0, 0), 60, 0, 54 * gb, probe=120, budget=1500)
    # 54 GB at 100 MB/s: 9 min, in time. Bytes already there when the watch began do not count as speed.
    assert not ex.too_slow(0, (0, 0), 180, 18 * gb, 54 * gb, probe=120, budget=1500)
    assert ex.too_slow(0, (0, 40 * gb), 180, 40 * gb, 54 * gb, probe=120, budget=1500)
    assert not ex.too_slow(0, (0, 40 * gb), 180, 54 * gb, 54 * gb, probe=120, budget=1500)


def test_the_progress_counts_allocated_bytes_of_the_repo_blobs(tmp_path, monkeypatch):
    blobs = tmp_path / "models--org--m" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "a").write_bytes(b"x" * 8192)
    with open(blobs / "b.incomplete", "wb") as f:
        f.truncate(10**6)  # sized, not yet written
    monkeypatch.setitem(ex._CHECKPOINT_BYTES, ("org/m", "r"), 10**6)
    done, total = ex.download_progress("org/m", "r", cache_dir=str(tmp_path))
    assert total == 10**6 and 8192 <= done < 10**5


async def test_a_pod_downloading_too_slowly_goes_silent():
    import threading

    posts = []

    class Http:
        async def post(self, path, json=None, **kw):
            posts.append(path)
            return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                                   json=lambda: {"model_id": "m", "model_revision": "r"})

    released = threading.Event()
    now = [0.0]

    def clock():
        now[0] += 60.0
        return now[0]

    def stuck(model_id, revision):
        released.wait(5)
        return "model"

    executor = ex.AuditExecutor(http=Http(), executor_id="pod-1", token="x", model_id="m",
                                model_revision="r", load_model=stuck, heartbeat_seconds=0.01,
                                progress=lambda m, r: (10**9, 54 * 10**9), clock=clock)
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(executor.start(), timeout=0.3)
        beats = len(posts)
        await asyncio.sleep(0.1)
        assert len(posts) == beats and executor._model is None
    finally:
        released.set()
