"""The executor registry reads through one long-lived client: the order control
re-reads every registration every 30 s, and a client built per object (it parses
botocore's service model) held the control's loop at a full core."""

from __future__ import annotations

import asyncio
import json

from reliquary.infrastructure import corpus_executor_store as store
from tests.unit.corpus_split_harness import LATENCY_ENV, FileS3


def test_a_registry_read_and_its_heartbeats_build_one_client(monkeypatch, tmp_path):
    monkeypatch.setenv(LATENCY_ENV, "0,0,0,0,0")
    built = []

    def factory(**kwargs):
        built.append(kwargs)
        return FileS3(tmp_path)

    monkeypatch.setattr(store, "get_s3_client", factory)

    async def run():
        seed = FileS3(tmp_path)
        for i in range(20):
            await seed.put_object(Bucket="reliquary", Key=f"{store.EXECUTOR_PREFIX}ox-{i}.json",
                                  Body=json.dumps({"executor_id": f"ox-{i}", "status": "active"}))
        documents = await store.list_executors()
        await store.record_heartbeat("ox-3", at=1.0)
        await store.record_heartbeat("ox-4", at=2.0)
        return documents, await store.read_executor("ox-3")

    documents, beat = asyncio.run(run())
    assert {d["executor_id"] for d in documents} == {f"ox-{i}" for i in range(20)}
    assert beat["last_heartbeat"] == 1.0
    assert built == [{}]


def test_explicit_credentials_get_a_client_of_their_own(monkeypatch, tmp_path):
    monkeypatch.setenv(LATENCY_ENV, "0,0,0,0,0")
    built = []

    def factory(**kwargs):
        built.append(kwargs)
        return FileS3(tmp_path)

    monkeypatch.setattr(store, "get_s3_client", factory)
    asyncio.run(store.list_executors(access_key_id="k"))
    assert built == [{"access_key_id": "k"}]
