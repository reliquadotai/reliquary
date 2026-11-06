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
