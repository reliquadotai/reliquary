"""SplitDeliverySink: HTTP-routed keys over HTTP, evaluation files to the platform bucket."""

import asyncio

import pytest

from reliquary.corpus.delivery import HTTPDeliverySink, SplitDeliverySink

SUBNET = "deliveries/subnet-0e0c148b-60b2-4cea-8c19-ab3b20d7b6e5/manifest.json"
EVAL = "evaluations/order-eval-0e0c148b-60b2-4cea-8c19-ab3b20d7b6e5-1/graded.parquet"


class Recorder:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        async def call(key, *args):
            self.calls.append((name, key))
            return {"ok": True} if name == "get_json" else None
        return call


def _split():
    http = HTTPDeliverySink(url="https://platform.example", secret="s" * 40)
    sent, stored = Recorder(), Recorder()
    http.put_json = sent.put_json
    http.get_json = sent.get_json
    return SplitDeliverySink(http=http, bucket=stored), sent, stored


def test_keys_the_http_sink_routes_stay_on_http_and_evaluation_files_go_to_the_bucket():
    split, sent, stored = _split()
    asyncio.run(split.put_json(SUBNET, {"a": 1}))
    asyncio.run(split.put_json(EVAL.replace(".parquet", ".json"), {"b": 2}))
    asyncio.run(split.put_bytes(EVAL, b"x"))
    assert asyncio.run(split.get_json(SUBNET)) == {"ok": True}
    assert sent.calls == [("put_json", SUBNET), ("get_json", SUBNET)]
    assert stored.calls == [("put_json", EVAL.replace(".parquet", ".json")), ("put_bytes", EVAL)]


def test_it_supports_evaluations_and_keeps_the_http_sink_limits():
    split, _, _ = _split()
    assert split.evaluation_supported is True
    assert split.max_file_bytes == HTTPDeliverySink.max_file_bytes
    assert split.requires_job_contract_pin is True


def test_bytes_never_go_over_http():
    split, _, _ = _split()
    with pytest.raises(ValueError):
        asyncio.run(split.put_bytes(SUBNET, b"x"))
