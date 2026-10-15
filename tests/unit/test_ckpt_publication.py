"""Checkpoint publication: trainer-side deltas and anchors, the signed index on
GET /rl/checkpoint, and the miner's download plan with anchor fallback."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest
import torch

from reliquary.miner.ckpt_sync import CheckpointSync, IndexSignatureError, ObjectMismatch
from reliquary.shared.ckpt_delta import checkpoint_sha256
from reliquary.shared.ckpt_index import (
    NoDownloadPath,
    index_key,
    plan_download,
    verify_index_signature,
)
from reliquary.trainer.ckpt_publisher import DeltaPublisher
from reliquary.validator.rl_checkpoint import RlCheckpointBoard, presigner, rl_checkpoint_response
from tests.unit.test_ckpt_delta import bits_equal, chain, clone

HOTKEY = "5ValidatorHotkey"


def fake_sign(payload: bytes) -> bytes:
    return hashlib.sha256(b"secret" + payload).digest()


def fake_verify(hotkey: str, payload: bytes, signature: bytes) -> bool:
    return hotkey == HOTKEY and signature == fake_sign(payload)


class MemoryStore:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def put_bytes(self, key, data):
        self.objects[key] = bytes(data)

    def put_file(self, key, path):
        self.objects[key] = Path(path).read_bytes()

    def get_bytes(self, key):
        return self.objects.get(key)

    def delete(self, key):
        del self.objects[key]


def make_board(store, run="run1", **kw):
    return RlCheckpointBoard(
        load_index=lambda: store.get_bytes(index_key(run)), sign_fn=fake_sign,
        validator_hotkey=HOTKEY, presign=lambda key: f"mem://{key}", refresh_seconds=0, **kw)


def make_sync(store, tmp_path):
    def download(url, dest):
        dest.write_bytes(store.objects[url.removeprefix("mem://")])
    return CheckpointSync(validator_hotkey=HOTKEY, work_dir=tmp_path / "work",
                          download=download, verify_signature=fake_verify)


@pytest.fixture
def published(tmp_path):
    states = chain(10)
    store = MemoryStore()
    pub = DeltaPublisher(run="run1", store=store, staging_dir=tmp_path / "staging",
                         anchor_interval=4, cumulative_stride=2, max_cumulative_bases=2,
                         keep_anchors=2)
    for version, state in enumerate(states):
        index = pub.publish(version, state)
    return states, store, index


def answer(store):
    return asyncio.run(make_board(store).answer())


def test_publisher_layout_anchors_deltas_cumulatives_and_pruning(published):
    states, store, index = published
    assert index["latest"] == {"version": 10, "sha256": checkpoint_sha256(states[10])}
    assert [a["version"] for a in index["anchors"]] == [4, 8]  # anchor 0 pruned
    # Per-step deltas since the oldest kept anchor; the cumulative from the base
    # at anchor 8 (the base at 10 starts empty); older cumulatives pruned.
    spans = sorted((d["from"], d["to"]) for d in index["deltas"])
    assert spans == sorted([(k, k + 1) for k in range(4, 10)] + [(8, 10)])
    # Every object the index names is in the bucket, nothing else is but the index.
    keys = {f["key"] for a in index["anchors"] for f in a["files"]} | {
        d["key"] for d in index["deltas"]}
    assert keys | {index_key("run1")} == set(store.objects)
    stored = json.loads(store.objects[index_key("run1")])
    assert stored["latest"]["version"] == 10


def test_publisher_refuses_going_backwards(published, tmp_path):
    states, store, _ = published
    pub = DeltaPublisher(run="run1", store=store, staging_dir=tmp_path / "s2")
    pub.publish(11, states[10])
    with pytest.raises(ValueError):
        pub.publish(11, states[10])


def test_restarted_publisher_keeps_old_entries_and_anchors_first(published, tmp_path):
    states, store, _ = published
    pub = DeltaPublisher(run="run1", store=store, staging_dir=tmp_path / "s2",
                         anchor_interval=4, keep_anchors=3)
    index = pub.publish(11, states[10])
    assert [a["version"] for a in index["anchors"]] == [4, 8, 11]
    assert any(d["to"] == 10 for d in index["deltas"])


@pytest.mark.parametrize("start", [None, 4, 5, 6, 7, 8, 9, 10])
def test_miner_rebuilds_the_latest_bit_exactly_from_any_version(published, tmp_path, start):
    states, store, _ = published
    state = None if start is None else clone(states[start])
    rebuilt, version, plan = make_sync(store, tmp_path).sync(answer(store), state, start)
    assert version == 10
    assert bits_equal(rebuilt, states[10])
    if start is not None and start >= 8:
        assert plan.anchor is None


def test_miner_too_old_for_any_delta_takes_the_anchor(published, tmp_path):
    states, store, _ = published
    rebuilt, _, plan = make_sync(store, tmp_path).sync(answer(store), clone(states[1]), 1)
    assert plan.anchor is not None
    assert bits_equal(rebuilt, states[10])


def test_hash_mismatch_falls_back_to_the_anchor(published, tmp_path):
    states, store, _ = published
    drifted = clone(states[9])
    drifted["layers.0.norm.weight"].view(torch.int16)[0] ^= 1  # ones: no delta rewrites it
    rebuilt, _, plan = make_sync(store, tmp_path).sync(answer(store), drifted, 9)
    assert plan.anchor is not None
    assert bits_equal(rebuilt, states[10])


def test_tampered_object_is_caught_before_apply(published, tmp_path):
    states, store, index = published
    key = next(d["key"] for d in index["deltas"] if (d["from"], d["to"]) == (9, 10))
    store.objects[key] = store.objects[key][:-1] + bytes([store.objects[key][-1] ^ 1])
    rebuilt, _, plan = make_sync(store, tmp_path).sync(answer(store), clone(states[9]), 9)
    assert plan.anchor is not None and bits_equal(rebuilt, states[10])
    anchor_key = index["anchors"][-1]["files"][0]["key"]
    store.objects[anchor_key] = b"garbage"
    with pytest.raises(ObjectMismatch):
        make_sync(store, tmp_path).sync(answer(store), None, None)


def test_miner_refuses_an_index_not_signed_by_its_validator(published, tmp_path):
    _, store, _ = published
    response = answer(store)
    response["index"]["latest"]["version"] = 99
    with pytest.raises(IndexSignatureError):
        make_sync(store, tmp_path).sync(response, None, None)
    response = answer(store)
    response["validator_hotkey"] = "5Other"
    with pytest.raises(IndexSignatureError):
        make_sync(store, tmp_path).sync(response, None, None)


def test_up_to_date_miner_downloads_nothing(published, tmp_path):
    states, store, _ = published
    state = clone(states[10])
    rebuilt, version, plan = make_sync(store, tmp_path).sync(answer(store), state, 10)
    assert plan.empty and rebuilt is state and version == 10


def _index(anchors, deltas, latest=10):
    return {"latest": {"version": latest, "sha256": "h"},
            "anchors": [{"version": v, "sha256": "a", "files": [{"key": f"a{v}", "size": s}]}
                        for v, s in anchors],
            "deltas": [{"from": f, "to": t, "key": f"d{f}-{t}", "size": s}
                       for f, t, s in deltas]}


def test_plan_picks_the_cheapest_route_by_bytes():
    steps = [(k, k + 1, 10) for k in range(5, 10)]
    index = _index([(5, 1000)], steps + [(5, 10, 30)])
    plan = plan_download(index, 7)  # 30 bytes either way: fewer steps wins
    assert [(d["from"], d["to"]) for d in plan.deltas] == [(5, 10)]
    index = _index([(5, 1000)], steps + [(5, 10, 45)])
    assert len(plan_download(index, 7).deltas) == 3
    assert plan_download(index, 2).anchor["version"] == 5  # too old: anchor + cheapest
    assert plan_download(index, None).bytes == 1045
    # A small recent anchor beats a long chain even for a miner holding a state.
    plan = plan_download(_index([(9, 5)], steps), 6)
    assert plan.anchor["version"] == 9 and plan.bytes == 15


def test_plan_without_route_raises():
    with pytest.raises(NoDownloadPath):
        plan_download(_index([], [(5, 6, 1)]), 2)


def test_board_signs_once_and_presigns_every_object(published):
    _, store, index = published
    calls = []

    def sign(payload):
        calls.append(payload)
        return fake_sign(payload)

    board = RlCheckpointBoard(load_index=lambda: store.get_bytes(index_key("run1")), sign_fn=sign,
                              validator_hotkey=HOTKEY, presign=lambda k: f"mem://{k}",
                              refresh_seconds=0)
    first = asyncio.run(board.answer())
    second = asyncio.run(board.answer())
    assert len(calls) == 1
    assert verify_index_signature(first["index"], first["signature"], HOTKEY,
                                  verify_fn=fake_verify)
    assert set(first["urls"]) == {f["key"] for a in index["anchors"] for f in a["files"]} | {
        d["key"] for d in index["deltas"]}
    assert second["signature"] == first["signature"]


def test_board_keeps_the_last_index_when_the_bucket_goes_backwards_or_fails(published):
    _, store, _ = published
    raw = store.get_bytes(index_key("run1"))
    older = json.loads(raw)
    older["latest"]["version"] = 3
    feed = [raw, json.dumps(older).encode(), RuntimeError("r2 down")]

    def load():
        item = feed.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    board = RlCheckpointBoard(load_index=load, sign_fn=fake_sign, validator_hotkey=HOTKEY,
                              presign=lambda k: k, refresh_seconds=0)
    for _ in range(3):
        assert asyncio.run(board.answer())["index"]["latest"]["version"] == 10


def test_route_answers_503_without_an_index():
    empty = RlCheckpointBoard(load_index=lambda: None, sign_fn=fake_sign,
                              validator_hotkey=HOTKEY, presign=lambda k: k)
    assert asyncio.run(rl_checkpoint_response(None)).status_code == 503
    assert asyncio.run(rl_checkpoint_response(empty)).status_code == 503


def test_route_serves_the_signed_index(published):
    _, store, _ = published
    response = asyncio.run(rl_checkpoint_response(make_board(store)))
    body = json.loads(response.body)
    assert response.status_code == 200 and body["validator_hotkey"] == HOTKEY
    assert response.headers["cache-control"] == "private, no-store"


def test_presigned_url_is_local_and_never_carries_the_secret(monkeypatch):
    from reliquary.infrastructure.storage import public_checkpoint_bucket, public_checkpoint_client

    monkeypatch.setenv("R2_PUBLIC_ENDPOINT", "https://example.invalid")
    monkeypatch.setenv("R2_PUBLIC_BUCKET", "reliquary-public")
    monkeypatch.setenv("R2_PUBLIC_ACCESS_KEY_ID", "AKIDTEST")
    monkeypatch.setenv("R2_PUBLIC_SECRET_ACCESS_KEY", "SUPERSECRETVALUE")
    url = presigner(public_checkpoint_client(), public_checkpoint_bucket(), ttl=300)(
        "rl/run1/deltas/00000001-00000002.rqd")
    assert url.startswith("https://example.invalid/reliquary-public/rl/run1/deltas/")
    assert "X-Amz-Expires=300" in url and "SUPERSECRETVALUE" not in url


def test_public_client_requires_an_endpoint(monkeypatch):
    from reliquary.infrastructure.storage import public_checkpoint_client

    monkeypatch.delenv("R2_PUBLIC_ENDPOINT", raising=False)
    with pytest.raises(RuntimeError):
        public_checkpoint_client()


def test_cumulatives_past_the_size_cap_are_dropped(tmp_path):
    states = chain(6)
    store = MemoryStore()
    pub = DeltaPublisher(run="run1", store=store, staging_dir=tmp_path / "staging",
                         anchor_interval=100, cumulative_stride=2, max_cumulative_bytes=1)
    for version, state in enumerate(states):
        index = pub.publish(version, state)
    assert all(d["to"] - d["from"] == 1 for d in index["deltas"])
    rebuilt, _, plan = make_sync(store, tmp_path).sync(answer(store), clone(states[2]), 2)
    assert plan.anchor is None and len(plan.deltas) == 4
    assert bits_equal(rebuilt, states[6])


def _model_dir(tmp_path):
    folder = tmp_path / "hf-model"
    folder.mkdir()
    (folder / "config.json").write_text('{"model_type": "qwen3"}')
    (folder / "tokenizer.json").write_text('{"version": "1.0"}')
    (folder / "chat_template.jinja").write_text("{{ messages }}")
    (folder / "model.safetensors").write_bytes(b"weights are not model files")
    (folder / "model.safetensors.index.json").write_text("{}")
    (folder / ".cache").write_text("hidden")
    return folder


def test_anchors_carry_the_model_files_and_the_miner_keeps_them(tmp_path):
    states = chain(6)
    store = MemoryStore()
    model = _model_dir(tmp_path)
    pub = DeltaPublisher(run="run1", store=store, staging_dir=tmp_path / "staging", anchor_interval=2,
                         keep_anchors=2, model_dir=model)
    for version, state in enumerate(states):
        index = pub.publish(version, state)
    names = ["chat_template.jinja", "config.json", "tokenizer.json"]
    assert [a["version"] for a in index["anchors"]] == [4, 6]
    for anchor in index["anchors"]:
        assert [f["name"] for f in anchor["model_files"]] == names
    # pruned anchors take their model files with them; the board presigns the kept ones
    assert not any("/anchors/00000000/" in key or "/anchors/00000002/" in key for key in store.objects)
    response = answer(store)
    assert {f["key"] for a in index["anchors"] for f in a["model_files"]} <= set(response["urls"])
    sync = CheckpointSync(validator_hotkey=HOTKEY, work_dir=tmp_path / "work", verify_signature=fake_verify,
                          download=lambda url, dest: dest.write_bytes(store.objects[url.removeprefix("mem://")]),
                          model_dir=tmp_path / "miner-model")
    rebuilt, version, _ = sync.sync(response, None, None)
    assert version == 6 and bits_equal(rebuilt, states[6])
    assert sorted(p.name for p in (tmp_path / "miner-model").iterdir()) == names
    assert (tmp_path / "miner-model" / "config.json").read_text() == '{"model_type": "qwen3"}'
    assert sync.sync_model_files(response["index"], response["urls"]) == []      # unchanged: nothing fetched
    (tmp_path / "miner-model" / "tokenizer.json").write_text("stale")
    assert sync.sync_model_files(response["index"], response["urls"]) == ["tokenizer.json"]
    key = index["anchors"][-1]["model_files"][1]["key"]
    store.objects[key] = b"tampered"
    (tmp_path / "miner-model" / "config.json").unlink()
    with pytest.raises(ObjectMismatch):
        sync.sync_model_files(response["index"], response["urls"])
    assert not (tmp_path / "miner-model" / "config.json").exists()


def test_a_model_dir_without_config_or_with_unsafe_names_is_refused(tmp_path):
    from reliquary.shared.ckpt_index import model_file_name_ok
    from reliquary.trainer.ckpt_publisher import model_files

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "tokenizer.json").write_text("{}")
    with pytest.raises(ValueError):
        model_files(empty)
    assert not any(model_file_name_ok(n) for n in ("../config.json", "a/b", ".hidden", "x.safetensors", ""))
    assert model_file_name_ok("tokenizer_config.json")
