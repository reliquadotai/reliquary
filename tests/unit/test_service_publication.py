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
    INDEX_CACHE, PAGE_SEGMENTS, SEGMENT_CACHE, ObservationPublisher, encode_segment, index_key, page_key,
    public_events, segment_key, verify_index, verify_segment,
)
from tests.unit.test_service_runtime_v2 import build, explore, group, runtime

SECRET_HOTKEY = "5MinerHotkeyThatMustNeverBePublished"


def keypair():
    return Keypair.create_from_seed("0x" + "03" * 32)


def verify(body, key, **kw):
    return verify_index(body, key.ss58_address, expected_run_id="r1", **kw)


def publisher(rt, puts, *, clock=lambda: 1000.7, fail=None, get=None):
    key = keypair()

    async def put(key_, body, content_type, cache_control):
        if fail is not None:
            fail(key_)
        puts.append((key_, body, content_type, cache_control))
    return ObservationPublisher(rt, run_id="r1", task_id="next-rl", wallet=SimpleNamespace(hotkey=key),
                                put=put, get=get, clock=clock), key


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
    index = verify(idx_body, key)
    entry = index["segments"][0]
    assert entry["key"] == seg_key and entry["first_seq"] == 1 and entry["size"] == len(seg_body)
    assert entry["sha256"] == hashlib.sha256(seg_body).hexdigest()
    assert index["run_id"] == "r1" and index["order_sha256"] == rt.contract.sha256 and index["schema"]
    assert index["kind"] == "head" and index["last_number"] == 1 and index["pages"] == []
    assert set(entry) == {"number", "key", "first_seq", "last_seq", "sha256", "size"}
    assert verify_segment(seg_body, entry)
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
        verify(json.dumps(document).encode(), key)
    other = Keypair.create_from_seed("0x" + "04" * 32)
    with pytest.raises(ValueError, match="another validator"):
        verify_index(puts[1][1], other.ss58_address, expected_run_id="r1")
    with pytest.raises(ValueError):
        verify(b"not json", key)
    with pytest.raises(ValueError, match="another run"):
        verify_index(puts[1][1], key.ss58_address, expected_run_id="r2")
    with pytest.raises(ValueError, match="stale"):
        verify(puts[1][1], key, min_last_number=2)
    assert verify(puts[1][1], key, min_last_number=1)
    # the domain prefix is part of the signed bytes: a signature over the bare JSON does not verify
    body = {k: v for k, v in document.items() if k != "signature"}
    from reliquary.protocol.release_contract import canonical_json_bytes
    document["signature"] = bytes(key.sign(canonical_json_bytes(body))).hex()
    with pytest.raises(ValueError, match="signature"):
        verify(json.dumps(document).encode(), key)


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
    assert verify(puts[2][1], key)["segments"][0]["sha256"] == hashlib.sha256(puts[0][1]).hexdigest()


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
    verify(puts[-1][1], key)


def test_a_later_segment_extends_the_index(tmp_path):
    rt = runtime(tmp_path)
    explore(rt)
    puts = []
    pub, key = publisher(rt, puts)
    asyncio.run(pub.flush())
    explore(rt, prompt=9, hotkey="later")
    asyncio.run(pub.flush())
    index = verify(puts[-1][1], key)
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


def test_boot_refuses_a_configured_publication_with_a_remote_signer(monkeypatch):
    from reliquary.validator.service import ValidationService
    check = ValidationService._require_publication_signer
    monkeypatch.setenv("RELIQUARY_OBSERVATIONS_BUCKET", "fake-public-bucket")
    with pytest.raises(ValueError, match="local hotkey.*signer operation"):
        check(SimpleNamespace(_service_runtime=object(), _signer_client=object()))
    check(SimpleNamespace(_service_runtime=object(), _signer_client=None))       # local hotkey: fine
    check(SimpleNamespace(_service_runtime=None, _signer_client=object()))       # legacy / corpus: inert
    monkeypatch.delenv("RELIQUARY_OBSERVATIONS_BUCKET")
    check(SimpleNamespace(_service_runtime=object(), _signer_client=object()))   # not configured: off


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


# --- Fix round 1: bounded head + immutable pages (R29), remote check (I-1), reasons, close (m-6) ---

def small_pages(monkeypatch, page=3):
    monkeypatch.setattr(publication, "PAGE_SEGMENTS", page)
    monkeypatch.setattr(publication, "SERVICE_OBSERVATION_SEGMENT_MAX_EVENTS", 1)  # one event per segment


def fill(rt, n, start=10):
    for i in range(n):
        explore(rt, prompt=start + i, hotkey=f"h{i}")


def test_pages_roll_over_the_head_stays_bounded_and_a_reader_verifies_across_pages(tmp_path, monkeypatch):
    small_pages(monkeypatch)
    rt = runtime(tmp_path)
    fill(rt, 7)
    puts = []
    pub, key = publisher(rt, puts)
    while asyncio.run(pub.flush()) is not None:
        pass
    keys = [p[0] for p in puts]
    page1, page2 = page_key("r1", 1, 3), page_key("r1", 4, 6)
    assert keys.count(page1) == keys.count(page2) == 1
    # pages are written before the head that first references them, with the immutable cache header
    first_head_with_page2 = next(i for i, (k, b, _, _) in enumerate(puts)
                                 if k == index_key("r1") and page2.encode() in b)
    assert keys.index(page2) < first_head_with_page2 and keys.index(page1) < keys.index(page2)
    assert {c for k, _, _, c in puts if k in (page1, page2)} == {SEGMENT_CACHE}
    head_bytes = [b for k, b, _, _ in puts if k == index_key("r1")][-1]
    head = verify(head_bytes, key, min_last_number=7)
    assert head["last_number"] == 7 and [s["number"] for s in head["segments"]] == [7]   # every segment after the last closed page
    assert [p["key"] for p in head["pages"]] == [page1, page2]
    # the reader walks the pages once: every segment 1..7 is reachable and verifies
    segments = {k: b for k, b, _, _ in puts if k.endswith(".gz")}
    seen = []
    for page in head["pages"]:
        raw = next(b for k, b, _, _ in puts if k == page["key"])
        assert hashlib.sha256(raw).hexdigest() == page["sha256"] and len(raw) == page["size"]
        body = verify(raw, key)
        assert body["kind"] == "page" and body["first_number"] == page["first_number"]
        seen += body["segments"]
    seen += head["segments"]
    assert [e["number"] for e in seen] == [1, 2, 3, 4, 5, 6, 7]
    for e in seen:
        assert verify_segment(segments[e["key"]], e)


def test_a_restart_rebuilds_byte_identical_pages(tmp_path, monkeypatch):
    small_pages(monkeypatch)
    rt = runtime(tmp_path)
    fill(rt, 4)
    puts = []
    pub, _ = publisher(rt, puts)
    while asyncio.run(pub.flush()) is not None:
        pass
    first = next(b for k, b, _, _ in puts if k == page_key("r1", 1, 3))
    rt.close()
    rt2 = build(tmp_path / "runtime.sqlite3")
    puts2 = []
    pub2, _ = publisher(rt2, puts2, clock=lambda: 9999.0)
    asyncio.run(pub2.flush())
    assert next(b for k, b, _, _ in puts2 if k == page_key("r1", 1, 3)) == first   # signature included


def test_a_restart_holds_no_page_bytes_between_digest_and_upload(tmp_path, monkeypatch):
    small_pages(monkeypatch)
    rt = runtime(tmp_path)
    fill(rt, 10)                                  # three closed pages
    puts = []
    pub, _ = publisher(rt, puts)
    while asyncio.run(pub.flush()) is not None:
        pass
    rt.close()
    rt2 = build(tmp_path / "runtime.sqlite3")
    puts2 = []
    pub2, _ = publisher(rt2, puts2)
    held = []
    real_put = pub2.put

    async def put(key_, body, content_type, cache_control):
        if key_ != index_key("r1") and key_.endswith(".json"):
            # while a page is being uploaded, no page bytes are buffered in the publisher
            held.append(any(isinstance(v, bytes) for c in vars(pub2).values() if isinstance(c, dict) for v in c.values()))
        await real_put(key_, body, content_type, cache_control)
    pub2.put = put
    asyncio.run(pub2.flush())
    pages = [k for k, *_ in puts2 if k.endswith(".json") and k != index_key("r1")]
    assert len(pages) == 3 and len(held) == 3
    assert held == [False, False, False]
    assert pub2._page_pending == set()
    assert [b for k, b, *_ in puts2 if k in pages] == [b for k, b, *_ in puts if k in pages]   # same bytes as before


def test_page_and_head_verification_refuses_malformed_documents(tmp_path, monkeypatch):
    small_pages(monkeypatch)
    rt = runtime(tmp_path)
    fill(rt, 4)
    puts = []
    pub, key = publisher(rt, puts)
    while asyncio.run(pub.flush()) is not None:
        pass
    wallet = SimpleNamespace(hotkey=key)
    page = verify(next(b for k, b, _, _ in puts if k == page_key("r1", 1, 3)), key)
    body = {k: v for k, v in page.items() if k not in ("kind",)}
    body.pop("signature", None)
    for mutate, message in (
        (lambda b: b["segments"].pop(1), "exactly"),
        (lambda b: b["segments"][1].update(first_seq=b["segments"][1]["first_seq"] + 1, last_seq=b["segments"][1]["last_seq"] + 1), "contiguous"),
        (lambda b: b["segments"][0].update(key="observations/run-r1/seg-000009.jsonl.gz"), "key"),
    ):
        broken = json.loads(json.dumps(body))
        mutate(broken)
        with pytest.raises(ValueError, match=message):
            verify(publication.sign_document(broken, wallet), key)


def test_segment_verification_checks_size_and_hash(tmp_path):
    entry = {"sha256": hashlib.sha256(b"abc").hexdigest(), "size": 3}
    with pytest.raises(ValueError):
        verify_segment(b"abd", entry)
    with pytest.raises(ValueError):
        verify_segment(b"abc", {**entry, "size": 4})


def test_plan_segment_never_parses_event_payloads(tmp_path):
    rt = runtime(tmp_path)
    explore(rt)
    rt.log.events = lambda **kw: pytest.fail("plan_segment must use a SQL range")
    plan = rt.plan_segment(max_events=1000, now=5.0)
    assert plan["first_seq"] == 1 and plan["last_seq"] >= 1


def remote_head(tmp_path, n, name="remote"):
    """(head bytes, key) published by another journal with n events."""
    rt = runtime(tmp_path / name)
    fill(rt, n)
    puts = []
    pub, key = publisher(rt, puts)
    asyncio.run(pub.flush())
    return puts[-1][1], key


def getter(raw, calls=None):
    async def get(key):
        if calls is not None:
            calls.append(key)
        return raw
    return get


def test_remote_check_absent_remote_proceeds_and_runs_once(tmp_path):
    rt = runtime(tmp_path)
    explore(rt)
    puts, calls = [], []
    pub, _ = publisher(rt, puts, get=getter(None, calls))
    asyncio.run(pub.flush())
    explore(rt, prompt=9, hotkey="later")
    asyncio.run(pub.flush())
    assert calls == [index_key("r1")] and len(puts) == 4 and not pub.disabled


def test_remote_check_disables_publication_when_the_remote_is_ahead(tmp_path, caplog):
    raw, _ = remote_head(tmp_path, 1)
    rt = runtime(tmp_path / "restored")            # an older database: it has nothing committed
    explore(rt)
    puts = []
    pub, _ = publisher(rt, puts, get=getter(raw))
    with caplog.at_level("CRITICAL"):
        assert asyncio.run(pub.flush()) is None
    assert pub.disabled and puts == [] and "DISABLED" in caplog.text
    assert asyncio.run(pub.flush()) is None and puts == []
    asyncio.run(asyncio.wait_for(pub.run(asyncio.Event(), interval=0.01), 2))   # the loop ends, no raise


def test_remote_check_disables_when_a_common_segment_differs(tmp_path):
    raw, key = remote_head(tmp_path, 1)
    rt = runtime(tmp_path / "other")
    fill(rt, 1, start=30)
    puts = []
    pub, _ = publisher(rt, puts)
    asyncio.run(pub.flush())                       # local segment 1 now exists and differs from the remote's
    puts.clear()
    pub2, _ = publisher(rt, puts, get=getter(raw))
    asyncio.run(pub2.flush())
    assert pub2.disabled and puts == []


def test_remote_check_disables_on_a_forged_remote_index_and_proceeds_on_an_equal_one(tmp_path):
    rt = runtime(tmp_path)
    explore(rt)
    puts = []
    pub, key = publisher(rt, puts)
    asyncio.run(pub.flush())
    head = puts[-1][1]
    forged = json.loads(head)
    forged["last_number"] = 0
    pub2, _ = publisher(rt, [], get=getter(json.dumps(forged).encode()))
    asyncio.run(pub2.flush())
    assert pub2.disabled
    puts2 = []
    pub3, _ = publisher(rt, puts2, get=getter(head))   # same journal restarted: nothing to disable
    asyncio.run(pub3.flush())
    assert not pub3.disabled and puts2[-1][0] == index_key("r1")


def test_an_error_reading_the_remote_is_retried_not_disabling(tmp_path):
    rt = runtime(tmp_path)
    explore(rt)
    puts = []

    async def broken(key):
        raise OSError("r2 down")
    pub, _ = publisher(rt, puts, get=broken)
    with pytest.raises(OSError):
        asyncio.run(pub.flush())
    assert not pub.disabled and puts == []
    pub.get = getter(None)
    asyncio.run(pub.flush())
    assert len(puts) == 2


def test_every_reason_the_runtime_can_emit_is_in_the_published_allow_list(tmp_path):
    from reliquary.services.exploration import REFUSAL_REASONS, UNAUDITED_REASONS
    emitted = set(REFUSAL_REASONS) | set(UNAUDITED_REASONS) | {"order_inactive", "exploration_disabled",
                                                              "token_limit", "trained", "unaudited"}
    out = public_events([{"type": "observation", "id": "a", "reason": r} for r in sorted(emitted)], flush_at=1.0)
    assert {e["reason"] for e in out} <= publication.PUBLIC_REASONS
    assert publication.PUBLIC_REASONS - {"refused"} <= emitted | {"unaudited"}
    assert public_events([{"reason": "something_new"}], flush_at=1.0)[0]["reason"] == "refused"
    # a reason the runtime really emits today
    rt = runtime(tmp_path)
    from tests.unit.test_service_runtime_v2 import MATH
    gid, candidate = group(rt)
    rt.record_exploration(environment=MATH, prompt_idx=7, hotkey="hk", window=1, rewards=[0] * len(candidate["seeds"]),
                          group_id=gid, candidate=candidate, token_count=10 ** 9, now=120.0)
    puts = []
    pub, _ = publisher(rt, puts)
    asyncio.run(pub.flush())
    reasons = {e.get("reason") for e in lines(puts[0][1])} - {None}
    assert reasons == {"token_limit"} <= publication.PUBLIC_REASONS


def test_close_waits_for_a_running_thread_call_and_refuses_later_ones(tmp_path):
    import time
    rt = runtime(tmp_path)
    explore(rt)
    pub, _ = publisher(rt, [])
    done = []
    real = rt.plan_segment

    def slow(**kw):
        time.sleep(0.3)
        out = real(**kw)
        done.append(1)
        return out
    rt.plan_segment = slow

    async def scenario():
        task = asyncio.create_task(pub.flush())
        await asyncio.sleep(0.05)
        await pub.close()
        finished = list(done)
        task.cancel()
        with pytest.raises(BaseException):
            await task
        return finished
    assert asyncio.run(scenario()) == [1]
    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(pub._thread(lambda: None))


def test_stopping_the_service_closes_the_publisher_before_the_runtime():
    from reliquary.validator.service import ValidationService
    order = []

    class P:
        async def close(self):
            order.append("publisher")

    async def forever():
        await asyncio.sleep(0)
    async def scenario():
        me = SimpleNamespace(_observation_task=asyncio.create_task(forever()), _observation_stop=asyncio.Event(),
                             _observation_publisher=P())
        await ValidationService._stop_observation_publication(me)
    asyncio.run(scenario())
    assert order == ["publisher"]


# --- Fix round 2 (N-1 .. N-5) ---

def fake_segments(n):
    return [{"number": i, "first_seq": i, "last_seq": i, "sha256": "%064x" % i, "size": 1} for i in range(1, n + 1)]


class _StubRuntime:
    """Just enough runtime for the publisher's head/page code (no database)."""
    def __init__(self, n):
        self.n, self.pages, self.page_reads = n, {}, 0
        self.contract = SimpleNamespace(sha256="0" * 64)

    def published_segments(self, *, first_number=1, last_number=None):
        return [s for s in fake_segments(self.n) if first_number <= s["number"] <= (last_number or 2 ** 62)]

    def index_page(self, first, build):
        self.page_reads += 1
        if first not in self.pages:
            self.pages[first] = build()
        return self.pages[first]

    def last_published_number(self):
        return self.n


def stub_publisher(rt, key=None):
    key = key or keypair()
    return ObservationPublisher(rt, run_id="r1", task_id="t", wallet=SimpleNamespace(hotkey=key),
                                put=None), key


@pytest.mark.parametrize("page", [5, None])
def test_pages_and_head_cover_every_number_exactly_once(monkeypatch, page):
    if page is not None:
        monkeypatch.setattr(publication, "PAGE_SEGMENTS", page)
    size = publication.PAGE_SEGMENTS
    top = 3 * size + 2
    lasts = range(1, top + 1)
    rt = _StubRuntime(0)
    pub, key = stub_publisher(rt)
    for last in lasts:
        rt.n = last
        head_bytes, head = pub._head(last)
        body = verify_index(head_bytes, key.ss58_address, expected_run_id="r1", min_last_number=last)
        assert body["kind"] == "head" and len(body["segments"]) < size
        numbers = []
        for entry in body["pages"]:
            raw = pub._page_bytes((entry["first_number"] - 1) // size)
            publication.verify_page_entry(raw, entry)
            page_body = verify_index(raw, key.ss58_address, expected_run_id="r1")
            assert page_body["kind"] == "page"
            numbers += [e["number"] for e in page_body["segments"]]
        numbers += [e["number"] for e in body["segments"]]
        assert numbers == list(range(1, last + 1))   # union exactly 1..last, no duplicate, no gap


def test_verify_index_refuses_a_head_that_leaves_a_gap_or_a_whole_page(monkeypatch):
    monkeypatch.setattr(publication, "PAGE_SEGMENTS", 5)
    rt = _StubRuntime(12)
    pub, key = stub_publisher(rt)
    _, head = pub._head(12)
    wallet = SimpleNamespace(hotkey=key)

    def check(mutate, message):
        broken = json.loads(json.dumps(head))
        mutate(broken)
        with pytest.raises(ValueError, match=message):
            verify_index(publication.sign_document(broken, wallet), key.ss58_address, expected_run_id="r1")
    check(lambda b: b["segments"].pop(0), "continue its pages")                 # gap after the last page
    check(lambda b: b.update(pages=[], segments=head["segments"]), "continue its pages")   # first entry must be 1 w/o page
    check(lambda b: b["pages"].pop(), "continue its pages")
    rt.n = 4
    _, no_page = pub._head(4)
    broken = json.loads(json.dumps(no_page))
    broken["segments"].pop(0)
    with pytest.raises(ValueError, match="continue its pages"):
        verify_index(publication.sign_document(broken, wallet), key.ss58_address, expected_run_id="r1")


def test_a_closed_page_is_read_once_per_process(monkeypatch):
    monkeypatch.setattr(publication, "PAGE_SEGMENTS", 5)
    rt = _StubRuntime(0)
    pub, _ = stub_publisher(rt)
    for last in range(1, 23):
        rt.n = last
        pub._head(last)
    assert rt.page_reads == 4        # pages 0..3, once each, not once per flush


def test_a_failed_remote_read_does_not_mark_the_check_done_and_ten_failures_are_critical(tmp_path, caplog):
    rt = runtime(tmp_path)
    explore(rt)
    puts = []

    async def broken(key):
        raise OSError("r2 down")
    pub, _ = publisher(rt, puts, get=broken)
    with caplog.at_level("CRITICAL"):
        for _ in range(9):
            with pytest.raises(OSError):
                asyncio.run(pub.flush())
        assert "consecutive" not in caplog.text
        with pytest.raises(OSError):
            asyncio.run(pub.flush())
    assert not pub._checked and "10 consecutive failures" in caplog.text and puts == []
    # a remote that is ahead, read after the failures, still disables (the check really ran after them)
    raw, _ = remote_head(tmp_path, 1)
    puts2 = []
    pub2, _ = publisher(rt, puts2, get=broken)
    with pytest.raises(OSError):
        asyncio.run(pub2.flush())
    assert not pub2._checked and not pub2.disabled
    pub2.get = getter(raw)
    assert asyncio.run(pub2.flush()) is None
    assert pub2.disabled and puts2 == []
    # and an absent remote, read after a failure, lets the same journal proceed
    puts3 = []
    pub3, _ = publisher(rt, puts3, get=broken)
    with pytest.raises(OSError):
        asyncio.run(pub3.flush())
    pub3.get = getter(None)
    asyncio.run(pub3.flush())
    assert pub3._checked and not pub3.disabled and puts3


def paged_remote(tmp_path, n, monkeypatch):
    """(head bytes, {key: bytes}) published by another journal that filled n events with pages of 3."""
    small_pages(monkeypatch)
    rt = runtime(tmp_path / "remote")
    fill(rt, n)
    puts = []
    pub, key = publisher(rt, puts)
    while asyncio.run(pub.flush()) is not None:
        pass
    store = {k: b for k, b, _, _ in puts}
    return store[index_key("r1")], store, rt


def store_getter(store):
    async def get(key):
        return store.get(key)
    return get


def test_remote_check_reads_and_adopts_the_remote_pages(tmp_path, monkeypatch):
    head, store, remote_rt = paged_remote(tmp_path, 7, monkeypatch)
    # same journal, but its page table was lost: the remote bytes are adopted, nothing is rebuilt
    restored = runtime(tmp_path / "restored")
    for number in range(1, 8):
        seg = remote_rt.published_segments(first_number=number, last_number=number)[0]
        restored.db.execute("INSERT INTO service_segments(number, first_seq, last_seq, flush_at, sha256, size, windows, checkpoints, committed) "
                            "VALUES(?,?,?,?,?,?,?,?,1)", (seg["number"], seg["first_seq"], seg["last_seq"], 0.0, seg["sha256"], seg["size"], "[]", "[]"))
    restored.db.commit()
    puts = []
    pub, _ = publisher(restored, puts, get=store_getter(store))
    asyncio.run(pub.flush())
    assert not pub.disabled
    assert restored.index_page(1, lambda: pytest.fail("adopted")) == store[page_key("r1", 1, 3)]
    assert restored.index_page(4, lambda: pytest.fail("adopted")) == store[page_key("r1", 4, 6)]


def test_remote_check_disables_when_a_remote_page_is_missing_forged_or_differs(tmp_path, monkeypatch):
    head, store, remote_rt = paged_remote(tmp_path, 7, monkeypatch)
    page1 = page_key("r1", 1, 3)
    cases = {"missing": {k: v for k, v in store.items() if k != page1},
             "forged": {**store, page1: store[page1].replace(b'"size":', b'"size":1')},
             "other signed page": {**store, page1: store[page_key("r1", 4, 6)]}}
    for name, bad in cases.items():
        pub, _ = publisher(remote_rt, [], get=store_getter(bad))
        asyncio.run(pub.flush())
        assert pub.disabled, name
    body = {k: v for k, v in json.loads(store[page1]).items() if k != "signature"}
    resigned = publication.sign_document(body, SimpleNamespace(hotkey=keypair()))
    assert resigned != store[page1]
    # the remote is the journal's own: not disabled; then its local page bytes differ from the remote's
    pub, _ = publisher(remote_rt, [], get=store_getter(store))
    asyncio.run(pub.flush())
    assert not pub.disabled
    with remote_rt.db:
        remote_rt.db.execute("UPDATE service_index_pages SET body=? WHERE first_number=1", (resigned,))
    pub, _ = publisher(remote_rt, [], get=store_getter(store))
    asyncio.run(pub.flush())
    assert pub.disabled
    # a valid re-signed page whose bytes are not the ones the head hashes, with no local copy to compare to
    with remote_rt.db:
        remote_rt.db.execute("DELETE FROM service_index_pages")
    pub, _ = publisher(remote_rt, [], get=store_getter({**store, page1: resigned}))
    asyncio.run(pub.flush())
    assert pub.disabled


def test_close_keeps_waiting_for_a_thread_call_that_outlives_the_timeout(tmp_path):
    import time
    rt = runtime(tmp_path)
    explore(rt)
    pub, _ = publisher(rt, [])
    finished = []
    real = rt.plan_segment

    def slow(**kw):
        time.sleep(0.4)
        out = real(**kw)
        finished.append(1)
        return out
    rt.plan_segment = slow

    async def scenario():
        task = asyncio.create_task(pub.flush())
        await asyncio.sleep(0.05)
        await pub.close(timeout=0.05)
        seen = list(finished)
        task.cancel()
        with pytest.raises(BaseException):
            await task
        return seen
    assert asyncio.run(scenario()) == [1]    # close() returned only after the running call ended


def test_verify_index_refuses_a_head_holding_a_whole_page(monkeypatch):
    monkeypatch.setattr(publication, "PAGE_SEGMENTS", 7)
    rt = _StubRuntime(5)
    pub, key = stub_publisher(rt)
    head_bytes, _ = pub._head(5)
    assert verify_index(head_bytes, key.ss58_address, expected_run_id="r1")["last_number"] == 5
    monkeypatch.setattr(publication, "PAGE_SEGMENTS", 5)       # now five entries are a whole page
    with pytest.raises(ValueError, match="whole page"):
        verify_index(head_bytes, key.ss58_address, expected_run_id="r1")


def test_r30_window_aborted_is_public_on_settle_events_only_when_set():
    assert "window_aborted" in publication.PUBLIC_KEYS
    plain = {"type": "settle", "id": "a" * 64, "window": 1, "status": "trained", "proof": "proven", "ts": 5.0}
    flagged = {**plain, "window_aborted": True}
    out = publication.public_events([plain, flagged], flush_at=100.9)
    assert out[0] == {**plain, "ts": 100.0} and "window_aborted" not in out[0]
    assert out[1]["window_aborted"] is True
