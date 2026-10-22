"""Opt-in miner hardware declarations: ``POST /miner/setup`` (signed, stored, never scored) and its summary."""
from __future__ import annotations

import asyncio

import pytest

bt = pytest.importorskip("bittensor")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from reliquary.protocol.miner_setup import (  # noqa: E402
    MinerSetupReport, MinerSetupRequest, build_miner_setup_binding, collect_miner_setup, load_instance_id,
    query_gpus,
)
from reliquary.validator.miner_setup import MinerSetupStore, build_miner_setup_router, summarize  # noqa: E402

MINER = bt.Keypair.create_from_uri("//Alice")
VALIDATOR = bt.Keypair.create_from_uri("//Bob")
EVE = bt.Keypair.create_from_uri("//Eve")
NOW = 1_800_000_000
INSTANCE = "ab" * 16
DEVICES = [
    {"index": 0, "name": "NVIDIA H100 80GB HBM3", "memory_mib": 81559, "uuid": "GPU-aaaa"},
    {"index": 1, "name": "NVIDIA H100 80GB HBM3", "memory_mib": 81559, "uuid": "GPU-bbbb"},
]


def _report(instance=INSTANCE, devices=DEVICES, visible=None, note=""):
    return collect_miner_setup(hotkey=MINER.ss58_address, instance_id=instance, note=note, devices=devices,
                               visible=visible)


def _body(report, *, at=NOW, signer=MINER, hotkey=None, audience=None):
    hotkey = hotkey or MINER.ss58_address
    binding = build_miner_setup_binding(hotkey=hotkey, at=at, report=report,
                                        validator_hotkey=audience or VALIDATOR.ss58_address)
    return MinerSetupRequest(miner_hotkey=hotkey, at=at, report=report,
                             signature=signer.sign(binding).hex()).model_dump(mode="json")


@pytest.fixture
def harness(tmp_path):
    clock = {"now": float(NOW)}
    registered = {MINER.ss58_address}

    async def registration(hotkey):
        return None if hotkey in registered else "not_registered"

    store = MinerSetupStore(tmp_path / "setups.sqlite3")
    app = FastAPI()
    app.include_router(build_miner_setup_router(store, validator_hotkey=VALIDATOR.ss58_address,
                                                registration=registration, clock=lambda: clock["now"]))
    return TestClient(app), store, clock


def test_a_signed_declaration_is_stored(harness):
    client, store, _ = harness
    response = client.post("/miner/setup", json=_body(_report(note="1 box")))
    assert response.status_code == 200, response.text
    [row] = store.rows()
    assert row["hotkey"] == MINER.ss58_address
    assert [g["name"] for g in row["report"]["gpus"]] == ["NVIDIA H100 80GB HBM3"] * 2
    assert row["report"]["note"] == "1 box"
    # The raw UUID never leaves the miner.
    assert "GPU-aaaa" not in str(row)


@pytest.mark.parametrize("mutate, status", [
    (lambda r: {**_body(r), "signature": ""}, 401),
    (lambda r: _body(r, signer=EVE), 403),                      # signed by someone else
    (lambda r: _body(r, audience=EVE.ss58_address), 403),       # signed for another validator
    (lambda r: _body(r, at=NOW - 3600), 409),                   # stale
    (lambda r: {**_body(r), "report": {**r.model_dump(mode="json"), "note": "edited"}}, 403),
])
def test_refusals(harness, mutate, status):
    client, store, _ = harness
    assert client.post("/miner/setup", json=mutate(_report())).status_code == status
    assert store.rows() == []


def test_an_unregistered_hotkey_is_refused(harness):
    client, store, _ = harness
    body = _body(_report(), signer=EVE, hotkey=EVE.ss58_address)
    assert client.post("/miner/setup", json=body).status_code == 403
    assert store.rows() == []


def test_a_replay_or_a_burst_is_refused_then_a_later_report_replaces(harness):
    client, store, clock = harness
    first = _body(_report())
    assert client.post("/miner/setup", json=first).status_code == 200
    assert client.post("/miner/setup", json=first).status_code == 429
    clock["now"] += 30
    assert client.post("/miner/setup", json=_body(_report(), at=NOW + 30)).status_code == 429
    clock["now"] += 600
    later = _report(devices=DEVICES[:1])
    assert client.post("/miner/setup", json=_body(later, at=NOW + 630)).status_code == 200
    [row] = store.rows()
    assert len(row["report"]["gpus"]) == 1


def test_summary_counts_each_card_once_across_processes():
    # Two processes on one box, each pinned to one card, plus a restart re-reporting card 0.
    a = _report(instance="01" * 16, visible="0").model_dump(mode="json")
    b = _report(instance="02" * 16, visible="GPU-bbbb").model_dump(mode="json")
    again = _report(instance="03" * 16, visible="0").model_dump(mode="json")
    old = _report(instance="04" * 16, devices=[{**DEVICES[0], "uuid": "GPU-old"}]).model_dump(mode="json")
    rows = [{"hotkey": "hk", "instance_id": r["instance_id"], "at": NOW, "received_at": t, "report": r}
            for r, t in ((a, NOW), (b, NOW), (again, NOW), (old, NOW - 3 * 86400))]
    [line] = summarize(rows, now=NOW)
    assert line["instances"] == 3
    assert line["gpu_count"] == 2
    assert line["gpus"] == {"NVIDIA H100 80GB HBM3": 2}


def test_query_gpus_parses_nvidia_smi():
    class Done:
        stdout = "0, NVIDIA A100-SXM4-80GB, 81920, GPU-1234\n1, NVIDIA A100-SXM4-80GB, 81920, GPU-5678\n"

    devices = query_gpus(run=lambda *a, **k: Done())
    assert [d["index"] for d in devices] == [0, 1]
    assert devices[1] == {"index": 1, "name": "NVIDIA A100-SXM4-80GB", "memory_mib": 81920, "uuid": "GPU-5678"}


def test_query_gpus_without_nvidia_smi_is_empty():
    def missing(*a, **k):
        raise FileNotFoundError

    assert query_gpus(run=missing) == []


def test_instance_id_survives_a_restart(tmp_path):
    path = tmp_path / "state" / "id"
    first = load_instance_id(path)
    assert load_instance_id(path) == first
    MinerSetupReport(instance_id=first)


def test_the_miner_side_sends_a_report_the_route_accepts(harness):
    from reliquary.miner.miner_setup import send_miner_setup

    client, store, _ = harness

    class Wallet:
        class hotkey:
            ss58_address = MINER.ss58_address
            sign = staticmethod(MINER.sign)

    class Http:
        async def post(self, url, json, timeout):
            assert url == "http://v/miner/setup"
            return client.post("/miner/setup", json=json)

    import reliquary.miner.miner_setup as module

    original = module.time.time
    module.time.time = lambda: NOW
    try:
        ok = asyncio.run(send_miner_setup(Http(), validator_url="http://v/", validator_hotkey=VALIDATOR.ss58_address,
                                          wallet=Wallet, report=_report()))
    finally:
        module.time.time = original
    assert ok
    assert len(store.rows()) == 1


def test_concurrent_replays_are_stored_once(harness):
    import httpx

    client, store, _ = harness
    body = _body(_report())

    async def burst():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=client.app), base_url="http://v") as http:
            return await asyncio.gather(*(http.post("/miner/setup", json=body) for _ in range(16)))

    responses = asyncio.run(burst())
    assert [r.status_code for r in responses].count(200) == 1
    assert all(r.status_code in (200, 429) for r in responses)
    assert len(store.rows()) == 1
