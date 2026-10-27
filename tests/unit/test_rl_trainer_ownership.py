"""A trainer takeover fences state changes and immutable checkpoint publication."""
from __future__ import annotations

import json

import pytest
import torch
from fastapi.testclient import TestClient

from reliquary.shared.ckpt_index import index_key
from reliquary.trainer.ckpt_publisher import DeltaPublisher
from reliquary.trainer.rl_link import TrainerLink, TrainerLinkError
from reliquary.validator.rl_async_service import build_rl_trainer_app
from reliquary.validator.rl_trainer_owner import (
    EPOCH_HEADER,
    LEASE_SECONDS,
    OWNER_HEADER,
    TrainerOwner,
)
from tests.unit.test_rl_async_service import TOKEN, MemoryStore, components


def test_owner_claim_is_durable_idempotent_and_requires_expiry_for_takeover(tmp_path):
    rl = components(tmp_path)
    now = [100.0]
    rl.book.clock = lambda: now[0]
    owner = TrainerOwner(rl.book)
    first = owner.claim("a" * 32)
    assert owner.claim("a" * 32)["epoch"] == first["epoch"] == 1
    with pytest.raises(RuntimeError, match="busy"):
        owner.claim("b" * 32)
    assert TrainerOwner(rl.book).claim("a" * 32)["epoch"] == 1
    now[0] += LEASE_SECONDS
    second = owner.claim("b" * 32)
    assert second["epoch"] == 2
    for operation in (owner.heartbeat, owner.release):
        with pytest.raises(RuntimeError, match="lost"):
            operation("a" * 32, first["epoch"])
    owner.release("b" * 32, second["epoch"])
    assert owner.claim("c" * 32)["epoch"] == 3
    rl.close()


@pytest.mark.parametrize("route", ["adopt", "take", "take_older", "take_exhausted", "release", "requeue",
                                       "report", "audits", "audit", "publish_index"])
def test_mutations_require_the_current_owner_in_addition_to_the_token(tmp_path, route):
    rl = components(tmp_path)
    client = TestClient(build_rl_trainer_app(rl, token=TOKEN))
    auth = {"Authorization": f"Bearer {TOKEN}"}
    assert client.get("/rl/trainer/config", headers=auth).status_code == 200
    assert client.post(f"/rl/trainer/{route}", json={}, headers=auth).status_code == 409
    rl.close()


def test_an_old_epoch_cannot_requeue_or_adopt_after_takeover(tmp_path):
    rl = components(tmp_path)
    now = [100.0]
    rl.book.clock = lambda: now[0]
    client = TestClient(build_rl_trainer_app(rl, token=TOKEN))
    auth = {"Authorization": f"Bearer {TOKEN}"}
    first = client.post("/rl/trainer/owner/claim", json={"owner": "a" * 32}, headers=auth).json()
    old = {**auth, OWNER_HEADER: first["owner"], EPOCH_HEADER: str(first["epoch"])}
    assert client.post("/rl/trainer/adopt", json={"k": 0}, headers=old).status_code == 200
    now[0] += LEASE_SECONDS
    second = client.post("/rl/trainer/owner/claim", json={"owner": "b" * 32}, headers=auth).json()
    current = {**auth, OWNER_HEADER: second["owner"], EPOCH_HEADER: str(second["epoch"])}
    for route, body in (("requeue", {}), ("adopt", {"k": 1})):
        assert client.post(f"/rl/trainer/{route}", json=body, headers=old).status_code == 409
    assert client.post("/rl/trainer/adopt", json={"k": 1}, headers=current).status_code == 200
    assert rl.book.latest_checkpoint == 1
    rl.close()


def test_checkpoint_objects_are_isolated_and_old_epoch_cannot_move_the_index(tmp_path):
    rl = components(tmp_path)
    now = [100.0]
    rl.book.clock = lambda: now[0]
    store = rl.checkpoint_store = MemoryStore()
    client = TestClient(build_rl_trainer_app(rl, token=TOKEN))
    old = TrainerLink("http://testserver", TOKEN, client=client, clock=lambda: now[0], patience_s=0)
    old.claim_ownership()
    publisher = DeltaPublisher(run="smoke", store=store, staging_dir=tmp_path / "old",
                               writer_id=old.writer_id, publish_index=old.publish_index,
                               assert_owner=old.assert_ownership)
    publisher.publish(0, {"w": torch.zeros(1, dtype=torch.bfloat16)})
    old_keys = set(store.objects) - {index_key("smoke")}
    assert old_keys and all("/writers/1-" in key for key in old_keys)
    now[0] += LEASE_SECONDS
    current = TrainerLink("http://testserver", TOKEN, client=client, clock=lambda: now[0], patience_s=0)
    current.claim_ownership()
    newer = DeltaPublisher(run="smoke", store=store, staging_dir=tmp_path / "new",
                           writer_id=current.writer_id, publish_index=current.publish_index,
                           assert_owner=current.assert_ownership)
    index = newer.publish(1, {"w": torch.ones(1, dtype=torch.bfloat16)})
    canonical = store.get_bytes(index_key("smoke"))
    assert any("/writers/2-" in f["key"] for a in index["anchors"] for f in a["files"])
    # Even a process with an inaccurate local deadline cannot replace the authoritative pointer.
    old._owner_deadline = now[0] + LEASE_SECONDS
    with pytest.raises(TrainerLinkError) as refused:
        publisher.publish(1, {"w": torch.full((1,), 9, dtype=torch.bfloat16)})
    assert refused.value.status == 409
    assert store.get_bytes(index_key("smoke")) == canonical
    assert json.loads(canonical)["latest"]["version"] == 1
    for key in old_keys:
        assert store.objects[key] is not None
    rl.close()


def test_index_retry_is_idempotent_but_existing_version_cannot_change(tmp_path):
    rl = components(tmp_path)
    store = rl.checkpoint_store = MemoryStore()
    link = TrainerLink("http://testserver", TOKEN, client=TestClient(build_rl_trainer_app(rl, token=TOKEN)),
                       patience_s=0)
    link.claim_ownership()
    publisher = DeltaPublisher(run="smoke", store=store, staging_dir=tmp_path,
                               writer_id=link.writer_id, publish_index=link.publish_index,
                               assert_owner=link.assert_ownership)
    index = publisher.publish(0, {"w": torch.zeros(1, dtype=torch.bfloat16)})
    link.publish_index(index)
    changed = {**index, "latest": {**index["latest"], "sha256": "a" * 64}}
    with pytest.raises(TrainerLinkError) as refused:
        link.publish_index(changed)
    assert refused.value.status == 422
    assert json.loads(store.get_bytes(index_key("smoke"))) == index
    rl.close()


def test_index_storage_does_not_inherit_bulk_transfer_retries(monkeypatch):
    import boto3

    from reliquary.infrastructure.storage import public_checkpoint_client

    configs = []
    monkeypatch.setenv("R2_PUBLIC_ENDPOINT", "https://storage.invalid")
    monkeypatch.setattr(boto3, "client", lambda *a, **kw: configs.append(kw["config"]))
    public_checkpoint_client(metadata_only=True)
    public_checkpoint_client()
    metadata, bulk = configs
    assert (metadata.connect_timeout, metadata.read_timeout) == (1, 2)
    assert metadata.retries["total_max_attempts"] == 1
    assert (bulk.connect_timeout, bulk.read_timeout) == (15, 120)
    assert bulk.retries["max_attempts"] == 5


def test_a_delayed_claim_response_cannot_extend_the_local_lease(tmp_path):
    rl = components(tmp_path)
    now = [0.0]
    client = TestClient(build_rl_trainer_app(rl, token=TOKEN))

    class Delayed:
        def request(self, *args, **kwargs):
            response = client.request(*args, **kwargs)
            now[0] += LEASE_SECONDS + 1
            return response

    link = TrainerLink("http://testserver", TOKEN, client=Delayed(), clock=lambda: now[0], patience_s=0)
    with pytest.raises(TrainerLinkError, match="ownership lost"):
        link.claim_ownership()
    rl.close()


def test_initial_checkpoint_published_before_adoption_is_recovered(tmp_path):
    from reliquary.trainer.async_cli import resume_weights
    from tests.unit.test_ckpt_publication import HOTKEY, make_board, make_sync

    store = MemoryStore()
    initial = {"w": torch.ones(2, dtype=torch.bfloat16)}
    DeltaPublisher(run="smoke", store=store, staging_dir=tmp_path / "initial").publish(0, initial)
    rl = components(tmp_path, board=make_board(store, run="smoke"))
    rl.checkpoint_store = store
    link = TrainerLink("http://testserver", TOKEN, client=TestClient(build_rl_trainer_app(rl, token=TOKEN)),
                       patience_s=0)
    link.claim_ownership()
    assert rl.book.latest_checkpoint is None
    state, version, _ = resume_weights(link, validator_hotkey=HOTKEY, work_dir=tmp_path / "resume",
                                       sync=make_sync(store, tmp_path / "sync"))
    assert version == 0 and rl.book.latest_checkpoint == 0
    assert torch.equal(state["w"], initial["w"])
    next_writer = DeltaPublisher(run="smoke", store=store, staging_dir=tmp_path / "next",
                                 writer_id=link.writer_id, publish_index=link.publish_index,
                                 assert_owner=link.assert_ownership)
    next_writer.publish(1, {"w": initial["w"] + 1})
    assert json.loads(store.get_bytes(index_key("smoke")))["latest"]["version"] == 1
    rl.close()


def test_a_truly_fresh_run_does_not_retry_a_missing_checkpoint(tmp_path):
    from reliquary.trainer.async_cli import resume_weights

    rl = components(tmp_path)
    link = TrainerLink("http://testserver", TOKEN, client=TestClient(build_rl_trainer_app(rl, token=TOKEN)),
                       patience_s=0)
    assert resume_weights(link, validator_hotkey="v", work_dir=tmp_path) is None
    assert rl.book.latest_checkpoint is None
    rl.close()
