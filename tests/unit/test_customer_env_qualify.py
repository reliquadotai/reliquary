import hashlib
import json
import sys

import pytest

from reliquary.customer_env import env_id
from reliquary.customer_env.image import BuildFailed, dockerfile
from reliquary.customer_env.qualify import Qualifier
from reliquary.customer_env.registry import Registry, resolve, subnet_key
from reliquary.customer_env.runner import RewardClient, WorkerProcess
from tests.unit.test_customer_env_package import EXAMPLES, PYPROJECT, archive, good
from tests.unit.test_customer_env_runner import FAKE, ROOT

IMAGE = "registry.reliqua.ai/customer-envs/acct/my-env@sha256:" + "c" * 64
LOCK = f"numpy==2.0.0 --hash=sha256:{'a' * 64}\n"
EXAMPLES_OK = "\n".join(json.dumps(e) for e in ({"key": "t1", "answer": "2", "expected_score": 1.0},
                                                {"key": "t1", "answer": "3", "expected_score": 0.0}))


class Store:
    def __init__(self):
        self.objects = {}

    def create_bytes(self, key, body):
        assert key not in self.objects or self.objects[key] == body
        self.objects[key] = body


class Builder:
    def __init__(self, fail=None):
        self.fail, self.contexts = fail, []

    def build_archive(self, data, env):
        import tempfile
        from pathlib import Path

        from reliquary.customer_env.image import write_context
        from reliquary.customer_env.package import read_package

        with tempfile.TemporaryDirectory() as tmp:
            context = write_context(Path(tmp), data, read_package(data))
            self.contexts.append(sorted(p.name for p in context.iterdir()))
        if self.fail:
            raise BuildFailed(self.fail)
        return IMAGE


def fake_worker(env=None):
    def open_worker(image):
        assert image == IMAGE
        argv = ["env", f"PYTHONPATH={ROOT}", *(f"{k}={v}" for k, v in (env or {}).items()), sys.executable, FAKE]
        return RewardClient(WorkerProcess(argv, start_timeout_s=30), per_item_timeout_s=5)
    return open_worker


def setup(tmp_path, files=None, builder=None, open_worker=None, data=None):
    data = data or archive(files or good(**{"requirements.lock": LOCK, "examples.jsonl": EXAMPLES_OK}))
    sha = hashlib.sha256(data).hexdigest()
    env = env_id("acct", "my-env", sha)
    registry = Registry(str(tmp_path / "envs.sqlite"))
    record, created = registry.submit(env, f"customer-envs/acct/my-env/{sha}.tar.gz")
    assert created
    store = Store()
    q = Qualifier(registry, read_object=lambda key: data, builder=builder or Builder(),
                  open_worker=open_worker or fake_worker(), store=store)
    return q, registry, store, env


def test_a_good_package_qualifies_for_every_use_and_publishes_its_rows(tmp_path):
    builder = Builder()
    q, registry, store, env = setup(tmp_path, builder=builder)
    record = q.run(str(env))
    assert (record.status, record.reason, record.uses) == ("qualified", None, ["rl", "batch", "eval"])
    assert record.image_digest == IMAGE and record.prompt_set["count"] == 5 and record.prompt_set["refused"] == 1
    assert [e["score"] for e in record.examples] == [1.0, 0.0]
    assert builder.contexts == [["Dockerfile", "manifest.json", "package"]]
    rows = [json.loads(x) for x in store.objects[subnet_key(env, "prompts.jsonl")].splitlines()]
    assert [r["key"] for r in rows] == ["t0", "t1", "t2", "t3", "t4"]
    assert hashlib.sha256(store.objects[subnet_key(env, "prompts.jsonl")]).hexdigest() == record.prompt_set["sha256"]
    assert json.loads(store.objects[subnet_key(env, "record.json")])["status"] == "qualified"
    assert resolve(registry, str(env), account="acct", product="rl").env_id == str(env)
    with pytest.raises(ValueError, match="not_yours"):
        resolve(registry, str(env), account="other", product="rl")


@pytest.mark.parametrize("files, builder, worker_env, reason", [
    (good(**{"requirements.lock": "x>=1\n"}), None, None, "package_invalid"),
    (good(**{"requirements.lock": f"verifiers==0.3 --hash=sha256:{'a' * 64}\n"}), None, None, "package_invalid"),
    (None, Builder(fail="pip: no matching distribution"), None, "build_failed"),
    (None, None, {"FAKE_LOAD_FAIL": "1"}, "start_failed"),
    (good(**{"requirements.lock": LOCK, "examples.jsonl": EXAMPLES_OK.replace('"t1", "answer": "3"',
                                                                               '"t9", "answer": "3"')}),
     None, None, "examples_failed"),
    (good(**{"requirements.lock": LOCK, "examples.jsonl": EXAMPLES_OK.replace('"answer": "3"', '"answer": "2"')}),
     None, None, "examples_failed"),
    (good(**{"requirements.lock": LOCK, "examples.jsonl": EXAMPLES_OK.replace('"answer": "3"',
                                                                               '"answer": "HANG"')}),
     None, None, "examples_failed"),
    (good(**{"requirements.lock": LOCK, "examples.jsonl": EXAMPLES_OK,
             "pyproject.toml": PYPROJECT + 'kind = "agentic"\n'}), None, None, "agentic_not_supported"),
])
def test_each_failure_stops_with_its_reason(tmp_path, files, builder, worker_env, reason):
    q, registry, store, env = setup(tmp_path, files=files, builder=builder, open_worker=fake_worker(worker_env))
    record = q.run(str(env))
    assert (record.status, record.reason) == ("refused", reason), record.detail
    assert subnet_key(env, "record.json") not in store.objects
    with pytest.raises(ValueError, match="not_qualified"):
        resolve(registry, str(env), account="acct", product="eval")


def test_a_failure_of_ours_is_infrastructure_and_is_queued_again(tmp_path):
    def no_sandbox(image):
        return RewardClient(WorkerProcess(["/nonexistent/docker"]))

    q, registry, store, env = setup(tmp_path, open_worker=no_sandbox)
    record = q.run(str(env))
    assert (record.status, record.reason) == ("refused", "infrastructure")
    again, created = registry.submit(env, record.object_key)
    assert created and again.status == "queued"


def test_a_nondeterministic_grader_serves_evaluations_only(tmp_path):
    calls = {"n": 0}

    def flaky(image):
        calls["n"] += 1
        return fake_worker()(image) if calls["n"] == 1 else _Shifted(fake_worker()(image))

    q, registry, store, env = setup(tmp_path, open_worker=flaky)
    record = q.run(str(env))
    assert (record.status, record.reason, record.uses) == ("qualified", "nondeterministic", ["eval"])
    with pytest.raises(ValueError, match="not_qualified_for: rl"):
        resolve(registry, str(env), account="acct", product="rl")


class _Shifted:
    def __init__(self, inner):
        self.inner = inner

    def score(self, items):
        from reliquary.customer_env.runner import ScoreResult

        return [ScoreResult(1.0 - r.score) for r in self.inner.score(items)]

    def close(self):
        self.inner.close()


def test_a_tampered_archive_is_refused(tmp_path):
    data = archive(good(**{"requirements.lock": LOCK, "examples.jsonl": EXAMPLES_OK}))
    q, registry, store, env = setup(tmp_path, data=data)
    q._read = lambda key: data + b"x"
    assert q.run(str(env)).reason == "package_invalid"


def test_the_dockerfile_is_ours_and_installs_only_hash_pinned_dependencies():
    text = dockerfile("base@sha256:" + "0" * 64)
    assert "--require-hashes --no-deps" in text and "USER 65534:65534" in text
    assert text.startswith("FROM base@sha256:")


def test_resubmitting_a_version_is_idempotent(tmp_path):
    q, registry, store, env = setup(tmp_path)
    q.run(str(env))
    record, created = registry.submit(env, "customer-envs/whatever")
    assert not created and record.status == "qualified"
    assert [r.env_id for r in registry.for_account("acct")] == [str(env)] and registry.in_flight("acct") == 0
    assert EXAMPLES  # the package test's examples stay importable
