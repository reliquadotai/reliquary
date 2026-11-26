import asyncio
import hashlib
import json
from urllib.parse import quote

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from reliquary.customer_env import env_id
from reliquary.customer_env.admin import CustomerEnvs, add_routes
from reliquary.customer_env.qualify import Qualifier
from reliquary.customer_env.registry import Registry
from reliquary.customer_env.runner import ScoreResult
from reliquary.customer_env.scoring import CustomerScorer, reference
from tests.unit.test_customer_env_package import archive, good
from tests.unit.test_customer_env_qualify import EXAMPLES_OK, LOCK, Builder, fake_worker
from tests.unit.test_order_uploads import MODEL, REVISION, TOK

DATA = archive(good(**{"requirements.lock": LOCK, "examples.jsonl": EXAMPLES_OK}))
SHA = hashlib.sha256(DATA).hexdigest()
KEY = f"customer-envs/acct/my-env/{SHA}.tar.gz"


class Bucket:
    def __init__(self, objects=None):
        self.objects = dict(objects or {})

    async def get_bytes(self, key):
        return self.objects.get(key)

    async def create_bytes(self, key, body):
        assert self.objects.get(key, body) == body
        self.objects[key] = body
        return True

    async def put_bytes(self, key, body):
        self.objects[key] = body


@pytest.fixture
def app(tmp_path):
    registry = Registry(str(tmp_path / "envs.sqlite"))
    envs = CustomerEnvs(registry, lambda read, store: Qualifier(registry, read_object=read, builder=Builder(),
                                                                open_worker=fake_worker(), store=store))
    platform, subnet = Bucket({KEY: DATA}), Bucket()
    router = APIRouter(prefix="/admin/v1")
    add_routes(router, envs, deliveries=platform, eval_store=subnet,
               load_tokenizer=lambda model, revision: (TOK, 1000))
    application = FastAPI()
    application.include_router(router)
    return TestClient(application), registry, subnet


def wait_qualified(client, env):
    for _ in range(600):
        body = client.get(f"/admin/v1/customer-envs/{quote(env, safe='')}").json()
        if body["status"] in ("qualified", "refused"):
            return body
        asyncio.run(asyncio.sleep(0.1))
    raise AssertionError(body)


def test_submit_qualify_list_and_order_sets(app):
    client, registry, subnet = app
    with client:
        answer = client.post("/admin/v1/customer-envs", json={"account": "acct", "name": "my-env",
                                                              "object_key": KEY, "sha256": SHA}).json()
        env = str(env_id("acct", "my-env", SHA))
        assert answer == {"env_id": env, "status": "queued"}
        record = wait_qualified(client, env)
        assert (record["status"], record["uses"]) == ("qualified", ["rl", "batch", "eval"])
        assert "object_key" not in record
        listed = client.get("/admin/v1/customer-envs/accounts/acct").json()["versions"]
        assert [v["env_id"] for v in listed] == [env]
        assert client.get("/admin/v1/customer-envs/accounts/other").json() == {"versions": []}

        sets = client.post("/admin/v1/customer-envs/order-sets", json={
            "env_id": env, "account": "acct", "model": MODEL, "revision": REVISION, "purpose": "eval",
            "sampling": {"temperature": 1.0, "max_tokens": 16}, "start": 1, "end": 3})
        assert sets.status_code == 200, sets.text
        (group,) = sets.json()["groups"]
        grading_key = next(k for k in subnet.objects if k.endswith(f"{group['set_id']}/grading.jsonl"))
        grading = [json.loads(x) for x in subnet.objects[grading_key].splitlines()]
        assert [g["grader"] for g in grading] == [env, env]
        assert [json.loads(g["reference"])["key"] for g in grading] == ["t1", "t2"]

        refused = client.post("/admin/v1/customer-envs/order-sets", json={
            "env_id": env, "account": "other", "model": MODEL, "revision": REVISION, "purpose": "eval"})
        assert refused.status_code == 422 and "not_yours" in refused.text

        assert client.delete(f"/admin/v1/customer-envs/{quote(env, safe='')}").json()["status"] == "disabled"
        again = client.post("/admin/v1/customer-envs/order-sets", json={
            "env_id": env, "account": "acct", "model": MODEL, "revision": REVISION, "purpose": "eval"})
        assert again.status_code == 422 and "not_qualified" in again.text


def test_an_account_qualifies_at_most_three_versions_at_once(app):
    client, registry, _ = app
    for i in range(3):
        registry.submit(env_id("acct", f"env-{i}", f"{i}" * 64), "customer-envs/x")
    refused = client.post("/admin/v1/customer-envs", json={"account": "acct", "name": "my-env",
                                                           "object_key": KEY, "sha256": SHA})
    assert refused.status_code == 409 and refused.json()["detail"] == "too_many_qualifications"


@pytest.mark.parametrize("body", [
    {"account": "acct", "name": "my-env", "object_key": "batch-inputs/x", "sha256": SHA},
    {"account": "acct", "name": "my-env", "object_key": "customer-envs/../x", "sha256": SHA},
    {"account": "acct", "name": "My_Env", "object_key": KEY, "sha256": SHA},
])
def test_bad_submissions_are_refused(app, body):
    client, _, _ = app
    assert client.post("/admin/v1/customer-envs", json=body).status_code == 422


def test_unconfigured_routes_answer_503():
    router = APIRouter(prefix="/admin/v1")
    add_routes(router, None, deliveries=Bucket(), eval_store=Bucket())
    application = FastAPI()
    application.include_router(router)
    client = TestClient(application)
    assert client.get("/admin/v1/customer-envs/accounts/acct").status_code == 503


def test_eval_grading_scores_a_customer_row_on_the_reward_host_only():
    from reliquary.eval.grading import _Graders

    env = str(env_id("acct", "my-env", "a" * 64))
    seen = []

    class Client:
        def score(self, items):
            seen.extend(items)
            return [ScoreResult(1.0) if i["answer"].strip() == "4" else ScoreResult(None, "timeout") for i in items]

        def close(self):
            pass

    scorer = CustomerScorer(lambda e: "img@sha256:" + "c" * 64, lambda image: Client())
    graders = _Graders(lambda *_: pytest.fail("no environment opens for a customer row"), None,
                       customer_scorer=scorer)
    row = {"problem_id": "p", "custom_id": "p", "grader": env, "reference": reference("t2", "b" * 64)}
    assert graders.grade(row, "<think>hmm</think> 4") == (1.0, "", False)
    assert seen == [{"key": "t2", "prompt_sha256": "b" * 64, "answer": " 4"}]
    value, detail, _ = graders.grade(row, "5")
    assert value is None and detail == "customer_grader: timeout"
    bad = graders.grade({**row, "reference": "x"}, "4")
    assert bad[0] is None and "reference" in bad[1]
