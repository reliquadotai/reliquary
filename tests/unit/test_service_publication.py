"""Signed static publication of the run observation log (decision D, validator side)."""
import asyncio
import gzip
import hashlib
import json
from types import SimpleNamespace

import pytest
from bittensor_wallet import Keypair

from reliquary.infrastructure import storage
from reliquary.services import publication
from reliquary.services.publication import (
    INDEX_CACHE, SEGMENT_CACHE, ObservationPublisher, encode_segment, index_key, public_events, segment_key,
    verify_index,
)
from tests.unit.test_service_runtime_v2 import build, explore, runtime

SECRET_HOTKEY = "5MinerHotkeyThatMustNeverBePublished"


def keypair():
    return Keypair.create_from_seed("0x" + "03" * 32)


def publisher(rt, puts, *, clock=lambda: 1000.7, fail=None):
    key = keypair()

    async def put(key_, body, content_type, cache_control):
        if fail is not None:
            fail(key_)
        puts.append((key_, body, content_type, cache_control))
    return ObservationPublisher(rt, run_id="r1", task_id="next-rl", wallet=SimpleNamespace(hotkey=key),
                                put=put, clock=clock), key


def lines(body):
    return [json.loads(x) for x in gzip.decompress(body).decode().splitlines()]


def test_keys_follow_the_decided_layout():
    assert segment_key("r1", 7) == "observations/run-r1/seg-000007.jsonl.gz"
    assert index_key("r1") == "observations/run-r1/index.json"


def test_segment_bytes_are_deterministic():
    events = [{"type": "observation", "id": "a"}, {"type": "settle", "id": "a"}]
    assert encode_segment(events) == encode_segment(events)
    assert gzip.decompress(encode_segment(events)).decode().splitlines()[0] == '{"id":"a","type":"observation"}'
    assert encode_segment(events)[4:8] == b"\0\0\0\0"  # gzip mtime field


def test_flush_writes_an_immutable_segment_then_a_signed_short_cache_index(tmp_path):
    rt = runtime(tmp_path)
    explore(rt, hotkey=SECRET_HOTKEY)
    puts = []
    pub, key = publisher(rt, puts)
    asyncio.run(pub.flush())
    (seg_key, seg_body, seg_type, seg_cache), (idx_key, idx_body, idx_type, idx_cache) = puts
    assert seg_key == "observations/run-r1/seg-000001.jsonl.gz" and seg_cache == SEGMENT_CACHE
    assert "immutable" in seg_cache and "max-age=31536000" in seg_cache and seg_type == "application/gzip"
    assert idx_key == "observations/run-r1/index.json" and idx_cache == INDEX_CACHE == "public, max-age=15"
    index = verify_index(idx_body, key.ss58_address)
    entry = index["segments"][0]
    assert entry["key"] == seg_key and entry["first_seq"] == 1 and entry["size"] == len(seg_body)
    assert entry["sha256"] == hashlib.sha256(seg_body).hexdigest()
    assert index["run_id"] == "r1" and index["order_sha256"] == rt.contract.sha256 and index["schema"]
    assert asyncio.run(pub.flush()) is None and len(puts) == 2


def test_tampered_index_is_refused(tmp_path):
    rt = runtime(tmp_path)
    explore(rt)
    puts = []
    pub, key = publisher(rt, puts)
    asyncio.run(pub.flush())
    document = json.loads(puts[1][1])
    document["segments"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="signature"):
        verify_index(json.dumps(document).encode(), key.ss58_address)
    other = Keypair.create_from_seed("0x" + "04" * 32)
    with pytest.raises(ValueError, match="another validator"):
        verify_index(puts[1][1], other.ss58_address)
    with pytest.raises(ValueError):
        verify_index(b"not json", key.ss58_address)


def test_restart_rebuilds_the_same_segment_bytes(tmp_path):
    rt = runtime(tmp_path)
    explore(rt)
    plan = rt.plan_segment(max_events=1000, now=1000.7)
    first = encode_segment(public_events([e for _, e in rt.events(after=plan["first_seq"] - 1, limit=10)],
                                         flush_at=plan["flush_at"]))
    explore(rt, prompt=9, hotkey="later")
    again = rt.plan_segment(max_events=1000, now=5000.0)  # a later clock must not change the plan
    assert again == plan
    rt.close()
    rt2 = build(tmp_path / "runtime.sqlite3")  # a real restart on the same journal
    assert rt2.plan_segment(max_events=1000, now=9999.0) == plan
    puts = []
    pub, _ = publisher(rt2, puts, clock=lambda: 9999.0)
    asyncio.run(pub.flush())
    assert puts[0][1] == first


def test_crash_between_segment_upload_and_commit_reuploads_identical_bytes(tmp_path):
    rt = runtime(tmp_path)
    explore(rt)
    puts = []
    pub, _ = publisher(rt, puts)
    real = rt.commit_segment
    rt.commit_segment = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("crash"))
    with pytest.raises(RuntimeError):
        asyncio.run(pub.flush())
    assert len(puts) == 1 and rt.published_segments() == []
    rt.commit_segment = real
    rt.close()
    rt2 = build(tmp_path / "runtime.sqlite3")
    pub2, key = publisher(rt2, puts, clock=lambda: 4000.0)
    asyncio.run(pub2.flush())
    assert puts[0][:2] == puts[1][:2]                        # same key, same bytes
    assert verify_index(puts[2][1], key.ss58_address)["segments"][0]["sha256"] == hashlib.sha256(puts[0][1]).hexdigest()


def test_crash_between_segment_and_index_writes_the_index_after_restart(tmp_path):
    rt = runtime(tmp_path)
    explore(rt)
    puts = []
    pub, key = publisher(rt, puts, fail=lambda k: (_ for _ in ()).throw(OSError("down")) if k.endswith("index.json") else None)
    with pytest.raises(OSError):
        asyncio.run(pub.flush())
    assert [p[0] for p in puts] == [segment_key("r1", 1)] and len(rt.published_segments()) == 1
    # same process, store back up: the next flush has no new events but still writes the index
    pub.put = publisher(rt, puts)[0].put
    index = asyncio.run(pub.flush())
    assert index is not None and puts[-1][0] == index_key("r1")
    # a restarted process writes it too (its first flush), without touching the segment again
    rt.close()
    rt2 = build(tmp_path / "runtime.sqlite3")
    pub2, _ = publisher(rt2, puts)
    n = len(puts)
    asyncio.run(pub2.flush())
    assert [p[0] for p in puts[n:]] == [index_key("r1")]
    verify_index(puts[-1][1], key.ss58_address)


def test_a_later_segment_extends_the_index(tmp_path):
    rt = runtime(tmp_path)
    explore(rt)
    puts = []
    pub, key = publisher(rt, puts)
    asyncio.run(pub.flush())
    explore(rt, prompt=9, hotkey="later")
    asyncio.run(pub.flush())
    index = verify_index(puts[-1][1], key.ss58_address)
    assert [s["number"] for s in index["segments"]] == [1, 2]
    assert index["segments"][1]["first_seq"] == index["segments"][0]["last_seq"] + 1
    assert [p[0] for p in puts] == [segment_key("r1", 1), index_key("r1"), segment_key("r1", 2), index_key("r1")]


def test_nothing_is_published_without_events(tmp_path):
    rt = runtime(tmp_path)
    puts = []
    pub, _ = publisher(rt, puts)
    assert asyncio.run(pub.flush()) is None and puts == []


def test_reason_mapping_hides_submitter_state():
    events = [{"type": "observation", "id": "a", "reason": r, "ts": 1.0} for r in
              ("probation_limit", "banned", "already_scanned", "cap", "trained")]
    assert [e["reason"] for e in public_events(events, flush_at=5.0)] == \
        ["refused", "refused", "already_scanned", "cap", "trained"]
    assert [e["reason"] for e in events][:2] == ["probation_limit", "banned"]  # the input is untouched


def test_timestamps_are_rounded_to_the_flush_time(tmp_path):
    rt = runtime(tmp_path)
    r = explore(rt, now=120.0)
    explore(rt, prompt=9, now=133.3)
    with rt._txn():
        rt.log.settle(r["observation_id"], status="exploration_forfeited", proof="x", at=140.9)
    puts = []
    pub, _ = publisher(rt, puts, clock=lambda: 1000.7)
    asyncio.run(pub.flush())
    got = lines(puts[0][1])
    assert len(got) == 3 and {e["ts"] for e in got} == {1000.0}


def test_no_hotkey_salt_or_meta_in_any_published_byte(tmp_path):
    rt = runtime(tmp_path)
    explore(rt, hotkey=SECRET_HOTKEY)
    puts = []
    pub, key = publisher(rt, puts)
    asyncio.run(pub.flush())
    blob = b"".join(gzip.decompress(b) if k.endswith(".gz") else b for k, b, _, _ in puts)
    salt = rt.log.run_salt
    for needle in (SECRET_HOTKEY.encode(), salt, salt.hex().encode(), b"run_salt", b"run_meta", b"token_count"):
        assert needle not in blob
    assert key.ss58_address.encode() in blob  # the validator's own address is the signer, public
    assert set().union(*(set(e) for k, b, _, _ in puts if k.endswith(".gz") for e in lines(b))) <= publication.PUBLIC_KEYS


def test_transient_statuses_are_never_published(tmp_path):
    events = [{"type": "observation", "id": "a", "status": s} for s in
              ("exploration_recording", "service_unproven_recording", "exploration_pending")]
    assert [e["status"] for e in public_events(events, flush_at=1.0)] == ["exploration_pending"]
    rt = runtime(tmp_path)
    r = explore(rt)
    with rt._txn():
        rt.log.settle(r["observation_id"], status="exploration_recording", proof="x", at=1.0)
    puts = []
    pub, _ = publisher(rt, puts)
    asyncio.run(pub.flush())
    assert all("recording" not in e["status"] for e in lines(puts[0][1]))


def test_a_failed_upload_is_retried_with_the_same_bytes_and_never_blocks_the_loop(tmp_path):
    rt = runtime(tmp_path)
    explore(rt)
    puts, attempts = [], []

    def fail(key):
        attempts.append(key)
        if len(attempts) == 1:
            raise OSError("r2 down")
    pub, key = publisher(rt, puts, fail=fail)

    async def scenario():
        stop = asyncio.Event()
        ticks = 0
        task = asyncio.create_task(pub.run(stop, interval=0.01))
        while len(puts) < 2:       # (bounded by wait_for below) the loop keeps running while publication fails then recovers
            ticks += 1
            await asyncio.sleep(0.005)
        stop.set()
        await asyncio.wait_for(task, 2)
        return ticks
    assert asyncio.run(asyncio.wait_for(scenario(), 10)) > 0
    assert attempts[0] == attempts[1] == segment_key("r1", 1)   # the retry is the same segment
    assert [p[0] for p in puts] == [segment_key("r1", 1), index_key("r1")]
    assert len(rt.published_segments()) == 1


def test_flush_does_not_block_the_event_loop(tmp_path):
    rt = runtime(tmp_path)
    explore(rt)
    pub, _ = publisher(rt, [])
    seen = []
    real = rt.plan_segment

    def slow(**kw):
        import time
        time.sleep(0.2)
        return real(**kw)
    rt.plan_segment = slow

    async def scenario():
        async def beat():
            for _ in range(10):
                seen.append(1)
                await asyncio.sleep(0.01)
        await asyncio.gather(pub.flush(), beat())
    asyncio.run(scenario())
    assert len(seen) == 10


def test_publication_off_never_uploads(monkeypatch):
    from reliquary.validator.service import ValidationService
    monkeypatch.delenv("RELIQUARY_OBSERVATIONS_BUCKET", raising=False)
    monkeypatch.setattr(publication, "r2_put", lambda *_: pytest.fail("publication is off"))
    for runtime_, signer in ((object(), None), (None, None)):
        me = SimpleNamespace(_service_runtime=runtime_, _signer_client=signer)
        ValidationService._start_observation_publication(me)
        assert not hasattr(me, "_observation_task")
    # a bucket but no service runtime (legacy task / corpus job): still nothing
    monkeypatch.setenv("RELIQUARY_OBSERVATIONS_BUCKET", "fake-public-bucket")
    me = SimpleNamespace(_service_runtime=None, _signer_client=None)
    ValidationService._start_observation_publication(me)
    assert not hasattr(me, "_observation_task")
    # a remote signer cannot sign the index: not started either
    me = SimpleNamespace(_service_runtime=object(), _signer_client=object())
    ValidationService._start_observation_publication(me)
    assert not hasattr(me, "_observation_task")


# --- storage.upload_bytes (fake client, no network) ---

class _Client:
    def __init__(self, store, *, existing=None):
        self.store, self.calls = store, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def put_object(self, **kw):
        from botocore.exceptions import ClientError
        self.calls.append(kw)
        if kw.get("IfNoneMatch") == "*" and kw["Key"] in self.store:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}, "ResponseMetadata": {"HTTPStatusCode": 412}}, "PutObject")
        self.store[kw["Key"]] = kw["Body"]

    async def get_object(self, **kw):
        body = self.store[kw["Key"]]
        return {"Body": SimpleNamespace(read=_async(body), close=lambda: None)}


def _async(value):
    async def read():
        return value
    return read


def test_upload_bytes_sets_headers_and_never_replaces_with_different_bytes(monkeypatch):
    store = {}
    client = _Client(store)
    monkeypatch.setattr(storage, "get_s3_client", lambda **kw: client)
    run = lambda body, **kw: asyncio.run(storage.upload_bytes("k", body, content_type="application/gzip",
                                                              cache_control=SEGMENT_CACHE, bucket_name="fake", **kw))
    assert run(b"one", if_absent=True) is True
    assert client.calls[0]["CacheControl"] == SEGMENT_CACHE and client.calls[0]["ContentType"] == "application/gzip"
    assert client.calls[0]["IfNoneMatch"] == "*" and client.calls[0]["Bucket"] == "fake"
    assert run(b"one", if_absent=True) is True                      # identical re-upload: idempotent
    with pytest.raises(ValueError, match="different bytes"):
        run(b"two", if_absent=True)
    assert store["k"] == b"one"
    assert run(b"two") is True and store["k"] == b"two"             # the index (no if_absent) is replaced
    assert "IfNoneMatch" not in client.calls[-1]
