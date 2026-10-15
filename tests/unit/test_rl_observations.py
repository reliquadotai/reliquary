"""The async RL scan log (``validator/rl_observations.py``, ``protocol/rl_scan_log.py``): every scan of the lease book
published as signed segments under ``rl/<run>/observations/`` (the window format, another root), served with
presigned URLs by ``GET /rl/observations``, read back and verified by the miner, and folded into its planners."""
from __future__ import annotations

import asyncio
import gzip
import json
import sqlite3
from types import SimpleNamespace

import pytest

bt = pytest.importorskip("bittensor")

from reliquary.protocol.rl_scan_log import read_scan_log, scans_as_page  # noqa: E402
from reliquary.services import publication as pub  # noqa: E402
from reliquary.validator.rl_lease_quota import QuotaController  # noqa: E402
from reliquary.validator.rl_leases import LeaseBook, LeaseBookConfig  # noqa: E402
from reliquary.validator.rl_observations import ObservationBoard, scan_publisher  # noqa: E402

VALIDATOR = bt.Keypair.create_from_uri("//Bob")
OTHER = bt.Keypair.create_from_uri("//Eve")
ENV = "math"
ROOT = "rl/smoke/observations"


def book_at(path) -> LeaseBook:
    return LeaseBook(path, LeaseBookConfig(expiry_s={ENV: 600.0}, cooldown_of=lambda env: 2),
                     quota=QuotaController(), clock=lambda: 1000.0)


class Bucket:
    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.cache: dict[str, str] = {}
        self.crash_after: str | None = None      # this key's upload lands, then the process "dies"

    async def put(self, key, body, content_type, cache_control):
        if cache_control == pub.SEGMENT_CACHE and key in self.objects:
            assert self.objects[key] == body, f"immutable {key} rewritten"
        self.objects[key], self.cache[key] = body, cache_control
        if key == self.crash_after:
            self.crash_after = None
            raise RuntimeError("crash after the upload, before the commit")

    async def get(self, key):
        return self.objects.get(key)


def publisher_for(book, bucket, wallet=VALIDATOR, clock=lambda: 2000.0):
    return scan_publisher(book, run="smoke", task_id="rl-async", wallet=SimpleNamespace(hotkey=wallet),
                          put=bucket.put, get=bucket.get, clock=clock)


def scan(book, prompt, rewards, verdict, k=0):
    book.record_outcome(ENV, prompt, k, rewards, verdict, now=1500.0 + prompt)


async def drain(publisher):
    while await publisher.flush() is not None:
        pass


def reader_for(board, bucket):
    async def ask(after):
        return json.loads(json.dumps(board.answer(after)))    # the wire

    async def fetch(url):
        return bucket.objects[url.removeprefix("mem://")]
    return ask, fetch


async def test_every_scan_is_published_signed_and_read_back(tmp_path):
    book, bucket = book_at(tmp_path / "rl.sqlite3"), Bucket()
    publisher = publisher_for(book, bucket)
    board = ObservationBoard(publisher, presign=lambda key: f"mem://{key}")
    assert board.answer() is None                                       # nothing published yet: 503
    await drain(publisher)
    assert bucket.objects == {}
    scan(book, 3, [1.0] * 8 + [0.0] * 8, "in_zone")
    scan(book, 4, [1.0] * 16, "too_easy")
    await drain(publisher)
    scan(book, 3, [0.0] * 16, "too_hard", k=1)
    await drain(publisher)
    assert sorted(bucket.objects) == [f"{ROOT}/index.json", f"{ROOT}/seg-000001.jsonl.gz",
                                      f"{ROOT}/seg-000002.jsonl.gz"]
    assert bucket.cache[f"{ROOT}/index.json"] == pub.INDEX_CACHE
    assert bucket.cache[f"{ROOT}/seg-000001.jsonl.gz"] == pub.SEGMENT_CACHE
    head = pub.verify_index(bucket.objects[f"{ROOT}/index.json"], VALIDATOR.ss58_address, expected_run_id="smoke",
                            root=ROOT)
    assert head["last_number"] == 2 and [s["first_seq"] for s in head["segments"]] == [1, 3]
    with pytest.raises(ValueError):                                     # another validator's key does not verify
        pub.verify_index(bucket.objects[f"{ROOT}/index.json"], OTHER.ss58_address, expected_run_id="smoke", root=ROOT)
    first = json.loads(gzip.decompress(bucket.objects[f"{ROOT}/seg-000001.jsonl.gz"]).splitlines()[0])
    assert first == {"type": "scan", "id": 1, "env": ENV, "prompt_idx": 3, "checkpoint": 0,
                     "rewards": [1.0] * 8 + [0.0] * 8, "verdict": "in_zone", "first_scan": True,
                     "state": "cooldown", "until_k": 2, "ts": 1503.0}       # no hotkey, no tokens

    answer = board.answer(1)
    assert answer["root"] == ROOT and set(answer["urls"]) == {f"{ROOT}/seg-000002.jsonl.gz"}
    events, last = await read_scan_log(*reader_for(board, bucket), VALIDATOR.ss58_address)
    assert last == 2 and [e["id"] for e in events] == [1, 2, 3]
    assert await read_scan_log(*reader_for(board, bucket), VALIDATOR.ss58_address, after=2) == ([], 2)
    with pytest.raises(ValueError):
        await read_scan_log(*reader_for(board, bucket), OTHER.ss58_address)
    bucket.objects[f"{ROOT}/seg-000002.jsonl.gz"] = gzip.compress(b"{}\n")   # a swapped segment is caught
    with pytest.raises(ValueError):
        await read_scan_log(*reader_for(board, bucket), VALIDATOR.ss58_address)

    page = scans_as_page(events)
    assert page["cursor"] == 3
    assert {(p["prompt_idx"], p["state"], p["until_k"]) for p in page["prompts"]} == {
        (3, "cooldown", 3), (4, "blacklisted", None)}                   # each prompt's state after its last scan
    by_prompt = sorted(page["prompts"], key=lambda p: p["prompt_idx"])
    assert sorted(book.prompt_state()["prompts"], key=lambda p: p["prompt_idx"]) == by_prompt   # = the route's
    book.close()


async def test_closed_pages_and_a_restart_rebuild_the_same_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(pub, "PAGE_SEGMENTS", 2)
    path, bucket = tmp_path / "rl.sqlite3", Bucket()
    book = book_at(path)
    publisher = publisher_for(book, bucket)
    for prompt in range(5):                                             # one segment per flush
        scan(book, prompt, [0.0] * 16, "too_hard")
        await drain(publisher)
    scan(book, 9, [0.0] * 16, "too_hard")
    bucket.crash_after = f"{ROOT}/seg-000006.jsonl.gz"
    with pytest.raises(RuntimeError):
        await publisher.flush()
    book.close()

    book = book_at(path)            # restart: segment 6 rebuilt byte for byte (Bucket.put asserts it), then 7
    scan(book, 8, [0.0] * 16, "too_hard")
    publisher = publisher_for(book, bucket, clock=lambda: 4000.0)
    await drain(publisher)
    assert publisher.head_bytes is not None and not publisher.disabled
    head = pub.verify_index(publisher.head_bytes, VALIDATOR.ss58_address, expected_run_id="smoke", root=ROOT)
    assert head["last_number"] == 7 and [p["key"] for p in head["pages"]] == [
        f"{ROOT}/index-000001-000002.json", f"{ROOT}/index-000003-000004.json", f"{ROOT}/index-000005-000006.json"]
    assert [(s["number"], s["first_seq"], s["last_seq"]) for s in head["segments"]] == [(7, 7, 7)]
    assert [s["first_seq"] for s in publisher.runtime.published_segments(first_number=6, last_number=6)] == [6]

    board = ObservationBoard(publisher, presign=lambda key: f"mem://{key}")
    assert set(board.answer(5)["urls"]) == {page["key"] for page in head["pages"]} | {
        f"{ROOT}/seg-000006.jsonl.gz", f"{ROOT}/seg-000007.jsonl.gz"}
    events, last = await read_scan_log(*reader_for(board, bucket), VALIDATOR.ss58_address)
    assert last == 7 and [e["prompt_idx"] for e in events] == [0, 1, 2, 3, 4, 9, 8]
    book.close()


async def test_a_lease_book_from_before_the_scan_log_gains_its_columns(tmp_path):
    path = tmp_path / "old.sqlite3"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE rl_prompt_scans(seq INTEGER PRIMARY KEY AUTOINCREMENT, environment TEXT NOT NULL, "
               "prompt_idx INTEGER NOT NULL, k INTEGER NOT NULL, rewards TEXT NOT NULL, verdict TEXT NOT NULL, "
               "first_scan INTEGER NOT NULL, at REAL NOT NULL)")
    db.execute("INSERT INTO rl_prompt_scans(environment, prompt_idx, k, rewards, verdict, first_scan, at) "
               "VALUES('math', 1, 0, '[0.0]', 'too_hard', 1, 10.0)")
    db.commit()
    db.close()
    book, bucket = book_at(path), Bucket()
    scan(book, 2, [1.0] * 16, "too_easy")
    publisher = publisher_for(book, bucket)
    await drain(publisher)
    events = [json.loads(line) for line in gzip.decompress(bucket.objects[f"{ROOT}/seg-000001.jsonl.gz"]).splitlines()]
    assert [(e["prompt_idx"], e["state"]) for e in events] == [(1, None), (2, "blacklisted")]
    book.close()


async def test_the_miner_bootstraps_its_planners_from_the_log(tmp_path):
    from reliquary.miner.rl_async import AsyncMiner, AsyncMinerConfig, DryRunTransport, FakeGroupGenerator, PromptPlanner

    book, bucket = book_at(tmp_path / "rl.sqlite3"), Bucket()
    publisher = publisher_for(book, bucket)
    scan(book, 3, [1.0] * 16, "too_easy")
    scan(book, 5, [0.0] * 16, "too_hard")
    await drain(publisher)
    board = ObservationBoard(publisher, presign=lambda key: f"mem://{key}")
    ask, fetch = reader_for(board, bucket)

    class Transport(DryRunTransport):
        validator_hotkey = VALIDATOR.ss58_address

        async def observations(self, after):
            return await ask(after)

        async def fetch(self, url):
            return await fetch(url)

    transport = Transport()
    planner = PromptPlanner(ENV, (0, 10))
    miner = AsyncMiner(AsyncMinerConfig(environments=(ENV,)), transport=transport, generator=FakeGroupGenerator(),
                       planners={ENV: planner})
    assert await miner.bootstrap_state() == 2
    assert planner.cursor == 2 and not planner.leasable(3) and not planner.leasable(5) and planner.leasable(4)
    await miner.refresh_state()
    assert transport.state_calls == [2]                                 # the route continues after the log
    assert await AsyncMiner(AsyncMinerConfig(environments=(ENV,)), transport=DryRunTransport(),
                            generator=FakeGroupGenerator(), planners={ENV: PromptPlanner(ENV, (0, 10))}
                            ).bootstrap_state() == 0                    # no log: the route from 0
    book.close()


def test_the_route_serves_the_head_or_503(tmp_path):
    from fastapi.testclient import TestClient

    from reliquary.validator.rl_async_service import mount_rl_async
    from reliquary.validator.server import ValidatorServer
    from tests.unit.test_rl_async_service import components

    rl = components(tmp_path)
    server = ValidatorServer()
    mount_rl_async(server, rl, validator_hotkey=VALIDATOR.ss58_address)
    client = TestClient(server.app)
    assert client.get("/rl/observations").status_code == 503
    bucket = Bucket()
    rl.observations = ObservationBoard(publisher_for(rl.book, bucket), presign=lambda key: f"mem://{key}")
    rl.book.record_outcome(next(iter(rl.book.config.expiry_s)), 1, 0, [0.0] * 16, "too_hard")
    asyncio.run(drain(rl.observations.publisher))
    response = client.get("/rl/observations", params={"after": "0"})
    assert response.status_code == 200 and response.headers["cache-control"] == "private, max-age=15"
    assert response.json()["index"]["last_number"] == 1 and len(response.json()["urls"]) == 1
    assert client.get("/rl/observations", params={"after": "x"}).status_code == 422
    rl.close()
