"""A trainer takeover fences state changes and immutable checkpoint publication."""
from __future__ import annotations

import hashlib
import io
import json
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from reliquary.shared.ckpt_index import index_key
from reliquary.trainer.ckpt_publisher import DeltaPublisher, R2ObjectStore
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


class ConditionalBucket:
    class NoSuchKey(Exception):
        pass

    exceptions = SimpleNamespace(NoSuchKey=NoSuchKey)

    def __init__(self):
        self.objects = {}
        self.pending = []
        self.puts = []
        self.next_outcome = None

    def get_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise self.NoSuchKey(Key)
        data = self.objects[Key]
        return {"Body": io.BytesIO(data), "ETag": '"' + hashlib.sha256(data).hexdigest() + '"'}

    def commit(self, request):
        from botocore.exceptions import ClientError

        current = self.objects.get(request["Key"])
        etag = None if current is None else '"' + hashlib.sha256(current).hexdigest() + '"'
        if (("IfMatch" in request and request["IfMatch"] != etag)
                or (request.get("IfNoneMatch") == "*" and current is not None)):
            raise ClientError({"Error": {"Code": "PreconditionFailed"},
                               "ResponseMetadata": {"HTTPStatusCode": 412}}, "PutObject")
        self.objects[request["Key"]] = request["Body"]

    def put_object(self, **request):
        self.puts.append(request)
        outcome, self.next_outcome = self.next_outcome, None
        if outcome == "pending":
            self.pending.append(request)
        else:
            self.commit(request)
        if outcome in {"pending", "committed"}:
            raise TimeoutError("write response unavailable")


def checkpoint_index(version):
    return {"format": "reliquary-ckpt-index/1", "run": "smoke",
            "latest": {"version": version, "sha256": str(version) * 64}, "anchors": [], "deltas": []}


@pytest.mark.parametrize("initial", [None, 0])
def test_timed_out_index_cannot_commit_after_a_new_owner_publishes(tmp_path, initial):
    from botocore.exceptions import ClientError

    rl = components(tmp_path)
    now = [100.0]
    rl.book.clock = lambda: now[0]
    bucket = ConditionalBucket()
    key = index_key("smoke")
    if initial is not None:
        bucket.objects[key] = json.dumps(checkpoint_index(initial)).encode()
    rl.checkpoint_store = R2ObjectStore(bucket, "public")
    client = TestClient(build_rl_trainer_app(rl, token=TOKEN))
    old = TrainerLink("http://testserver", TOKEN, client=client, clock=lambda: now[0], patience_s=0)
    old.claim_ownership()
    bucket.next_outcome = "pending"
    with pytest.raises(TrainerLinkError):
        old.publish_index(checkpoint_index(1))
    assert len(bucket.pending) == len(bucket.puts) == 1
    now[0] += LEASE_SECONDS
    current = TrainerLink("http://testserver", TOKEN, client=client, clock=lambda: now[0], patience_s=0)
    current.claim_ownership()
    current.publish_index(checkpoint_index(2))
    with pytest.raises(ClientError, match="PreconditionFailed"):
        bucket.commit(bucket.pending[0])
    assert json.loads(bucket.objects[key]) == checkpoint_index(2)
    assert len(bucket.puts) == 2
    rl.close()


def test_index_response_loss_after_commit_is_reconciled_without_another_write(tmp_path):
    rl = components(tmp_path)
    bucket = ConditionalBucket()
    rl.checkpoint_store = R2ObjectStore(bucket, "public")
    link = TrainerLink("http://testserver", TOKEN, client=TestClient(build_rl_trainer_app(rl, token=TOKEN)),
                       patience_s=0)
    bucket.next_outcome = "committed"
    link.publish_index(checkpoint_index(0))
    assert len(bucket.puts) == 1
    assert bucket.puts[0]["IfNoneMatch"] == "*"
    assert "IfMatch" not in bucket.puts[0]
    assert json.loads(bucket.objects[index_key("smoke")]) == checkpoint_index(0)
    rl.close()


def test_index_retry_after_an_unknown_write_never_repeats_an_unconditional_put(tmp_path):
    from botocore.exceptions import ClientError

    rl = components(tmp_path)
    bucket = ConditionalBucket()
    key = index_key("smoke")
    original = json.dumps(checkpoint_index(0)).encode()
    bucket.objects[key] = original
    rl.checkpoint_store = R2ObjectStore(bucket, "public")
    link = TrainerLink("http://testserver", TOKEN, client=TestClient(build_rl_trainer_app(rl, token=TOKEN)),
                       patience_s=0)
    bucket.next_outcome = "pending"
    with pytest.raises(TrainerLinkError):
        link.publish_index(checkpoint_index(1))
    link.publish_index(checkpoint_index(1))
    expected = '"' + hashlib.sha256(original).hexdigest() + '"'
    assert [request["IfMatch"] for request in bucket.puts] == [expected, expected]
    with pytest.raises(ClientError, match="PreconditionFailed"):
        bucket.commit(bucket.pending[0])
    assert json.loads(bucket.objects[key]) == checkpoint_index(1)
    rl.close()


def test_a_concurrent_index_change_is_never_replaced_after_a_conditional_failure(tmp_path):
    class RacingBucket(ConditionalBucket):
        def put_object(self, **request):
            self.objects[request["Key"]] = json.dumps(checkpoint_index(2)).encode()
            super().put_object(**request)

    rl = components(tmp_path)
    bucket = RacingBucket()
    key = index_key("smoke")
    bucket.objects[key] = json.dumps(checkpoint_index(0)).encode()
    rl.checkpoint_store = R2ObjectStore(bucket, "public")
    link = TrainerLink("http://testserver", TOKEN, client=TestClient(build_rl_trainer_app(rl, token=TOKEN)),
                       patience_s=0)
    with pytest.raises(TrainerLinkError):
        link.publish_index(checkpoint_index(1))
    with pytest.raises(TrainerLinkError) as refused:
        link.publish_index(checkpoint_index(1))
    assert refused.value.status == 422
    assert len(bucket.puts) == 1
    assert json.loads(bucket.objects[key]) == checkpoint_index(2)
    rl.close()


def test_index_read_without_an_etag_fails_before_any_write():
    class MissingEtag(ConditionalBucket):
        def get_object(self, **request):
            self.body = io.BytesIO(b"{}")
            return {"Body": self.body}

    bucket = MissingEtag()
    with pytest.raises(RuntimeError, match="requires an ETag"):
        R2ObjectStore(bucket, "public").get_bytes_with_etag("index")
    assert bucket.body.closed and not bucket.puts


def test_r2_conditional_writes_use_the_sdk_condition_parameters():
    import boto3
    from botocore.stub import Stubber

    client = boto3.client("s3", region_name="auto", endpoint_url="https://storage.invalid",
                          aws_access_key_id="key", aws_secret_access_key="secret")
    store = R2ObjectStore(client, "public")
    with Stubber(client) as stub:
        stub.add_response("put_object", {}, {"Bucket": "public", "Key": "index", "Body": b"a",
                                              "IfNoneMatch": "*"})
        stub.add_response("put_object", {}, {"Bucket": "public", "Key": "index", "Body": b"b",
                                              "IfMatch": '"previous"'})
        store.put_bytes_if_match("index", b"a", None)
        store.put_bytes_if_match("index", b"b", '"previous"')
        stub.assert_no_pending_responses()


def test_index_storage_does_not_inherit_bulk_transfer_retries(monkeypatch):
    import boto3

    from reliquary.infrastructure.storage import public_checkpoint_client

    configs = []
    monkeypatch.setenv("R2_PUBLIC_ENDPOINT", "https://storage.invalid")
    monkeypatch.setattr(boto3, "client", lambda *a, **kw: configs.append(kw["config"]))
    public_checkpoint_client(metadata_only=True)
    public_checkpoint_client()
    metadata, bulk = configs
    assert (metadata.connect_timeout, metadata.read_timeout) == (2, 5)
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
