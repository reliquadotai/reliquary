import pytest
from fastapi.testclient import TestClient

from reliquary.customer_env.host import RemoteRewardClient, WorkerPool, create_app
from reliquary.customer_env.runner import RewardInfrastructureError, ScoreResult
from tests.unit.test_customer_env_qualify import fake_worker
from tests.unit.test_customer_env_runner import item

PREFIX = "registry.reliqua.ai/customer-envs/"
IMAGE = PREFIX + "acct/my-env@sha256:" + "c" * 64


@pytest.fixture
def served():
    opened = []

    def open_client(image):
        opened.append(image)
        return fake_worker()(image)

    pool = WorkerPool(open_client, allowed_prefix=PREFIX, max_warm=1)
    client = TestClient(create_app(pool))
    yield client, opened
    pool.close()


def remote(client, image=IMAGE, fresh=False):
    return RemoteRewardClient("http://testserver", image, fresh=fresh, transport=client._transport)


def test_scores_and_freezes_through_one_warm_worker(served):
    client, opened = served
    rc = remote(client)
    assert rc.score([item("t1", "2"), item("t1", "RAISE")])[0] == ScoreResult(1.0)
    rows, refused = rc.freeze_all()
    assert len(rows) == 5 and len(refused) == 1
    assert opened == [IMAGE]


def test_fresh_asks_for_a_new_sandbox_each_call(served):
    client, opened = served
    rc = remote(client, fresh=True)
    rc.score([item("t1", "2")])
    rc.score([item("t1", "2")])
    assert opened == [IMAGE, IMAGE]


@pytest.mark.parametrize("image", ["docker.io/library/python@sha256:" + "a" * 64, PREFIX + "acct/x:latest",
                                   PREFIX + "acct/x"])
def test_only_our_registry_images_by_digest_run(served, image):
    client, _ = served
    with pytest.raises(RewardInfrastructureError, match="reward_host_refused: 422"):
        remote(client, image).score([item("t1", "2")])


def test_a_worker_failure_is_a_503_read_as_infrastructure(served):
    client, _ = served
    with pytest.raises(RewardInfrastructureError, match="worker_died"):
        remote(client).score([item("t1", "CRASH")])


def test_a_host_without_a_builder_does_not_build(served):
    client, _ = served
    assert client.post("/v1/build", content=b"x", headers={"x-env-id": "nope"}).status_code == 404
