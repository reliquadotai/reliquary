"""The real path, with Docker: pack the toy env, build its image on our base, qualify it in sealed workers,
and check what the customer's code can reach from inside.

    docker build -f docker/customer-env/Dockerfile.base -t reliquary-customer-env-base:local .
    RELIQUARY_DOCKER_TESTS=1 [RELIQUARY_DOCKER_RUNTIME=runsc] pytest tests/unit/test_customer_env_docker.py
"""
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from reliquary.customer_env import env_id
from reliquary.customer_env.image import BuildFailed, write_context
from reliquary.customer_env.pack import pack
from reliquary.customer_env.package import read_package
from reliquary.customer_env.qualify import Qualifier
from reliquary.customer_env.registry import Registry
from reliquary.customer_env.runner import RewardClient, WorkerProcess, sealed_argv

pytestmark = pytest.mark.skipif(os.environ.get("RELIQUARY_DOCKER_TESTS") != "1", reason="needs docker and the base")
BASE = os.environ.get("RELIQUARY_CUSTOMER_ENV_BASE", "reliquary-customer-env-base:local")
RUNTIME = os.environ.get("RELIQUARY_DOCKER_RUNTIME", "runc")
TOY = Path(__file__).resolve().parents[1] / "fixtures" / "customer_env" / "toy-add"


class LocalBuilder:
    def build_archive(self, data, env):
        with tempfile.TemporaryDirectory() as tmp:
            context = write_context(Path(tmp), data, read_package(data), base_image=BASE)
            done = subprocess.run(["docker", "build", "-q", "-t", f"customer-env-test:{env.sha256[:12]}",
                                   str(context)], capture_output=True, text=True, timeout=1800)
            if done.returncode:
                raise BuildFailed(done.stderr[-2000:])
            return f"local/customer-env-test@{done.stdout.strip()}"


def sealed(image: str) -> list[str]:
    argv = sealed_argv(image, runtime=RUNTIME)
    argv[-1] = image.split("@", 1)[1]  # a local image has no repository digest: run it by id
    return argv


class Store:
    def __init__(self):
        self.objects = {}

    def create_bytes(self, key, body):
        self.objects[key] = body


@pytest.fixture(scope="module")
def qualified(tmp_path_factory):
    data = pack(TOY, TOY / "examples.jsonl")
    env = env_id("acct", "toy-add", read_package(data).sha256)
    registry = Registry(str(tmp_path_factory.mktemp("reg") / "envs.sqlite"))
    registry.submit(env, "customer-envs/x")
    record = Qualifier(registry, read_object=lambda key: data, builder=LocalBuilder(), store=Store(),
                       open_worker=lambda image: RewardClient(WorkerProcess(sealed(image), start_timeout_s=180),
                                                              per_item_timeout_s=60)).run(str(env))
    return record


def test_the_toy_env_qualifies_for_every_use(qualified):
    assert (qualified.status, qualified.reason, qualified.uses) == ("qualified", None, ["rl", "batch", "eval"])
    assert qualified.prompt_set["count"] == 10
    assert [e["score"] for e in qualified.examples] == [1.0, 1.0, 0.0]
    assert qualified.cost["scoring_ms_per_item"] < 1000


def test_the_customer_code_reaches_nothing(qualified):
    argv = sealed(qualified.image_digest)
    image = argv.pop()
    probe = """
import os, socket, subprocess
def probe(fn):
    try:
        fn()
        return "ALLOWED"
    except Exception:
        return "blocked"
print(probe(lambda: socket.create_connection(("1.1.1.1", 53), timeout=3)),
      probe(lambda: open("/usr/local/lib/python3.12/site-packages/reliquary/x.py", "w")),
      probe(lambda: open("/opt/reliquary/manifest.json", "a")),
      probe(lambda: (open("/tmp/x.sh", "w").write("#!/bin/sh"), os.chmod("/tmp/x.sh", 0o755),
                     subprocess.run(["/tmp/x.sh"], check=True))),
      probe(lambda: os.setuid(0)), os.getuid())
"""
    done = subprocess.run([*argv, "--entrypoint", "python", image, "-I", "-c", probe], capture_output=True,
                          text=True, timeout=120)
    assert done.stdout.split() == ["blocked"] * 5 + ["65534"], done.stderr
