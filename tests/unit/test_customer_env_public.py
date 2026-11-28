"""A public customer env: its owner publishes a qualified version that names a license, any account orders it,
and its source (where an imported env comes from) goes with it."""
import hashlib
from urllib.parse import quote

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from reliquary.customer_env import check_order_env, env_id
from reliquary.customer_env.admin import CustomerEnvs, add_routes
from reliquary.customer_env.pack import _with_manifest, pack, prime_source
from reliquary.customer_env.package import PackageInvalid, read_package
from reliquary.customer_env.qualify import Qualifier
from reliquary.customer_env.registry import Registry, resolve
from tests.unit.test_customer_env_admin import Bucket, wait_qualified
from tests.unit.test_customer_env_pack import fake_run
from tests.unit.test_customer_env_package import archive, good
from tests.unit.test_customer_env_qualify import EXAMPLES_OK, LOCK, Builder, fake_worker
from tests.unit.test_order_uploads import MODEL, REVISION, TOK

SOURCE = ('\n[tool.reliquary.source]\nhub = "huggingface"\nowner = "FineEnvs"\nname = "endless-terminals"\n'
          'url = "https://huggingface.co/datasets/FineEnvs/endless-terminals"\nrevision = "3f2a9c1"\n'
          'license = "Apache-2.0"\n')
PYPROJECT = '[project]\nname = "my-env"\nversion = "1"\n[tool.reliquary]\ntaskset = "my-env"\n'


def package(pyproject: str) -> bytes:
    return archive(good(**{"pyproject.toml": pyproject, "requirements.lock": LOCK, "examples.jsonl": EXAMPLES_OK}))


def test_the_source_and_license_are_read_and_frozen_in_the_sha256():
    pkg = read_package(package(PYPROJECT + SOURCE))
    assert pkg.source == {"hub": "huggingface", "owner": "FineEnvs", "name": "endless-terminals",
                          "url": "https://huggingface.co/datasets/FineEnvs/endless-terminals",
                          "revision": "3f2a9c1", "license": "Apache-2.0"}
    assert pkg.license == "Apache-2.0"
    assert read_package(package(PYPROJECT + SOURCE)).sha256 != read_package(package(PYPROJECT)).sha256


@pytest.mark.parametrize("pyproject, license", [
    (PYPROJECT, None),
    (PYPROJECT.replace('taskset = "my-env"', 'taskset = "my-env"\nlicense = "MIT"') + SOURCE, "MIT"),
    (PYPROJECT.replace('version = "1"', 'version = "1"\nlicense = "BSD-3-Clause"'), "BSD-3-Clause"),
    (PYPROJECT.replace('version = "1"', 'version = "1"\nlicense = {text = "MIT"}'), "MIT"),
    # A full license text in [project] names none we can show: the version is private-only, not refused.
    (PYPROJECT.replace('version = "1"', 'version = "1"\nlicense = {text = "Permission is hereby granted, ' +
                       'free of charge, to any person obtaining a copy of this software" }'), None),
])
def test_the_package_license_comes_first_then_its_source_then_its_project(pyproject, license):
    assert read_package(package(pyproject)).license == license


@pytest.mark.parametrize("source", [
    'hub = "github"\nowner = "a"\nname = "b"\n',
    'hub = "huggingface"\nname = "b"\n',
    'hub = "huggingface"\nowner = "a"\nname = "b"\nurl = "http://example.com"\n',
    'hub = "huggingface"\nowner = "a"\nname = "b"\nstars = 3\n',
    'hub = "huggingface"\nowner = "a b"\nname = "b"\n',
])
def test_a_bad_source_is_refused(source):
    with pytest.raises(PackageInvalid, match="source"):
        read_package(package(PYPROJECT + "\n[tool.reliquary.source]\n" + source))


def test_pack_records_a_prime_env_source_and_the_license(tmp_path):
    src = tmp_path / "wordle"
    src.mkdir()
    (src / "pyproject.toml").write_text('[project]\nname = "wordle"\nversion = "0.3.1"\n')
    examples = tmp_path / "examples.jsonl"
    examples.write_text(EXAMPLES_OK)
    provenance = prime_source("will/wordle", src)
    assert provenance == {"hub": "prime_hub", "owner": "will", "name": "wordle", "revision": "0.3.1",
                          "url": "https://app.primeintellect.ai/dashboard/environments/will/wordle"}
    pkg = read_package(pack(src, examples, taskset="wordle", license="MIT", provenance=provenance,
                            run=fake_run([])))
    assert (pkg.license, pkg.source) == ("MIT", provenance)
    assert prime_source("will/wordle@0.1.0", src)["revision"] == "0.1.0"


def test_a_manifest_already_there_gets_the_source_and_its_license_beside_it():
    text = _with_manifest(PYPROJECT, None, None, license="MIT", source={"hub": "prime_hub", "owner": "a",
                                                                        "name": "b"})
    pkg = read_package(package(text))
    assert (pkg.license, pkg.source) == ("MIT", {"hub": "prime_hub", "owner": "a", "name": "b", "license": "MIT"})
    with pytest.raises(ValueError, match="name the license there"):
        _with_manifest(PYPROJECT, None, None, license="MIT")


def test_another_account_may_order_only_a_public_version():
    env = str(env_id("owner", "my-env", "a" * 64))
    with pytest.raises(ValueError, match="not_yours"):
        check_order_env(env, account="other", product="eval", uses=["eval"])
    assert check_order_env(env, account="other", product="eval", uses=["eval"], public=True).account == "owner"
    with pytest.raises(ValueError, match="not_qualified_for"):
        check_order_env(env, account="other", product="rl", uses=["eval"], public=True)


def test_publishing_rules(tmp_path):
    clock = iter(range(1, 100))
    registry = Registry(str(tmp_path / "envs.sqlite"), clock=lambda: next(clock))
    env = env_id("owner", "my-env", "a" * 64)
    registry.submit(env, "customer-envs/x")
    with pytest.raises(ValueError, match="not_qualified"):
        registry.publish(str(env), account="owner", public=True, listed=None)
    registry.update(str(env), status="qualified", uses=["rl", "batch", "eval"])
    with pytest.raises(ValueError, match="license_missing"):
        registry.publish(str(env), account="owner", public=True, listed=None)
    registry.update(str(env), license="MIT")
    with pytest.raises(ValueError, match="not_yours"):
        registry.publish(str(env), account="other", public=True, listed=None)
    with pytest.raises(ValueError, match="not_public"):
        registry.publish(str(env), account="owner", public=None, listed=True)
    with pytest.raises(ValueError, match="not_yours"):
        resolve(registry, str(env), account="other", product="eval")

    record = registry.publish(str(env), account="owner", public=True, listed=None)
    assert (record.public, record.listed) == (True, True) and record.published_at
    assert [r.env_id for r in registry.catalog()] == [str(env)]
    assert resolve(registry, str(env), account="other", product="rl").env_id == str(env)

    assert registry.publish(str(env), account="owner", public=None, listed=False).listed is False
    assert registry.catalog() == []
    assert resolve(registry, str(env), account="other", product="rl").public  # unlisted, still ordered
    with pytest.raises(ValueError, match="public_is_final"):
        registry.publish(str(env), account="owner", public=False, listed=None)


@pytest.fixture
def app(tmp_path):
    data = package(PYPROJECT + SOURCE)
    sha = hashlib.sha256(data).hexdigest()
    key = f"customer-envs/owner/my-env/{sha}.tar.gz"
    registry = Registry(str(tmp_path / "envs.sqlite"))
    envs = CustomerEnvs(registry, lambda read, store: Qualifier(registry, read_object=read, builder=Builder(),
                                                                open_worker=fake_worker(), store=store))
    router = APIRouter(prefix="/admin/v1")
    add_routes(router, envs, deliveries=Bucket({key: data}), eval_store=Bucket(),
               load_tokenizer=lambda model, revision: (TOK, 1000))
    application = FastAPI()
    application.include_router(router)
    return TestClient(application), sha, key


def test_admin_publish_catalog_and_an_order_from_another_account(app):
    client, sha, key = app
    with client:
        client.post("/admin/v1/customer-envs", json={"account": "owner", "name": "my-env", "object_key": key,
                                                     "sha256": sha})
        env = str(env_id("owner", "my-env", sha))
        record = wait_qualified(client, env)
        assert record["status"] == "qualified", record
        assert (record["license"], record["source"]["owner"], record["public"]) == ("Apache-2.0", "FineEnvs", False)
        assert client.get("/admin/v1/customer-envs/catalog").json() == {"versions": []}

        order = {"env_id": env, "account": "other", "model": MODEL, "revision": REVISION, "purpose": "eval"}
        assert client.post("/admin/v1/customer-envs/order-sets", json=order).status_code == 422

        refused = client.post("/admin/v1/customer-envs/visibility",
                              json={"env_id": env, "account": "other", "public": True})
        assert refused.status_code == 409 and "not_yours" in refused.text
        published = client.post("/admin/v1/customer-envs/visibility",
                                json={"env_id": env, "account": "owner", "public": True})
        assert published.status_code == 200 and published.json()["listed"] is True
        (listed,) = client.get("/admin/v1/customer-envs/catalog").json()["versions"]
        assert listed["env_id"] == env and listed["source"]["hub"] == "huggingface"
        assert "object_key" not in listed

        assert client.post("/admin/v1/customer-envs/order-sets", json=order).status_code == 200
        gone = client.delete(f"/admin/v1/customer-envs/{quote(env, safe='')}")
        assert gone.status_code == 409 and "unlist it instead" in gone.text
        assert client.post("/admin/v1/customer-envs/visibility",
                           json={"env_id": env, "account": "owner", "public": False}).status_code == 409
        unknown = client.post("/admin/v1/customer-envs/visibility",
                              json={"env_id": str(env_id("owner", "x", "b" * 64)), "account": "owner",
                                    "listed": False})
        assert unknown.status_code == 404
