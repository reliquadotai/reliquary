"""The activation checks of an env added to a running run."""
import asyncio
import errno
import hashlib
import os
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

pytest.importorskip("reliquary_sandbox_verifiers")

from reliquary.protocol.service_contract import EPISODE_BUDGET_FIELDS
from reliquary.sandbox.fleet import HEARTBEAT_MAX_AGE_S, Fleet
from tests.unit.episode_v2_fixtures import (
    ADDED,
    ADDED_IDENTITY,
    ADDED_PACKAGE,
    added_block,
    episode_block,
    episode_contract,
)
from tests.unit.sandbox_fixtures import (
    ADDRESS,
    NOW,
    capacity_report,
    machine_document,
    signer,
)
from tests.unit.test_sandbox_fleet import Clock, make
from tests.unit.test_servable_tasks import SECOND

BLOCK = added_block()
OPTIONS = BLOCK["catalogue"]["options"]
IMAGE_A = "registry.example/t@sha256:" + "a" * 64
IMAGE_B = "registry.example/t@sha256:" + "b" * 64
ALLOWED = frozenset({ADDED_PACKAGE})
ORDER = episode_contract()
SANDBOX = BLOCK["environment"]["episode"]["sandbox_env"]
VALIDITY = 7200                    # the token validity the issuer would give the env's sessions
GIB = 1024**3
CAPS = {"max_calls": 512, "max_cpu_s": 3600, "max_call_timeout_s": 600, "max_wall_s": 14400,
        "max_memory_bytes": 8 * GIB, "max_pids": 1024, "max_disk_bytes": 10 * GIB, "max_argument_bytes": 122880,
        "output_cap_bytes": 65536, "max_token_validity_s": 7 * 86400, "max_transcript_bytes": 8 * 1024**2}
BOOT = {"verifiers": frozenset({("0.1", "0" * 64, frozenset({"verifiers"}))})}
CURRENT = {**BOOT, "reliquary-terminal": frozenset({("0.3.0", "1" * 64, frozenset({"reliquary_terminal"}))})}


def attestation(**overrides):
    """Shaped like a sandbox conformance gate report."""
    value = {"gate": True, "passed": True, "problems": [], "fake": False, "sample_size": 32, "repeat": 2,
             "env_package": ADDED_IDENTITY, "entry": f"verifiers:{ADDED_PACKAGE}", "split": "tmax",
             "options": dict(OPTIONS), "gate_reasons": [], "env": "conformance-env",
             "image_lock_sha256": BLOCK["catalogue"]["image_lock_sha256"]}
    value.update(overrides)
    return value


class Source:
    """What the checks read of a BridgedEpisodeSource."""

    def __init__(self, **overrides):
        self.identity = overrides.get("identity", ADDED_IDENTITY)
        self._installed = overrides.get("installed", self.identity)
        self.lock_sha256 = overrides.get("lock_sha256", BLOCK["catalogue"]["image_lock_sha256"])
        self.lock_images = frozenset(overrides.get("lock_images", {IMAGE_A, IMAGE_B}))
        self._manifest = overrides.get("manifest", BLOCK["environment"]["version"])
        self._rows = overrides.get("rows", BLOCK["environment"]["dataset"]["rows"])
        self._images = overrides.get("images", lambda index: IMAGE_A if index % 2 else IMAGE_B)
        self.options_sha256 = "1" * 64
        self.module_name = overrides.get("module", "reliquary_terminal")
        self.resolved = 0

    def manifest_sha256(self):
        return self._manifest

    def installed_identity(self):
        return self._installed

    def task_image(self, index):
        self.resolved += 1
        return self._images(index)

    def __len__(self):
        return self._rows


class Prepared(Source):
    """A source whose tasks declare prepare timeouts."""

    def __init__(self, prepare, **overrides):
        super().__init__(**overrides)
        self._prepare = prepare
        self.prepare_reads = []

    def prepare_timeout_s(self, index):
        self.prepare_reads.append(index)
        return self._prepare(index)


# --- names: a plain distribution and a plain import module, its own -------------------------------------------
def _named(package):
    identity = package + "+g827598fa320ad040.vfb2"
    return added_block(catalogue={"package": package}, episode={"env_package": identity})


# --- machines ------------------------------------------------------------------------------------------------------
def serving_fleet(tmp_path, *, present, images, locked=2, **report):
    tmp_path.mkdir(parents=True, exist_ok=True)
    fleet, machine, served, _ = make(tmp_path)
    lock = BLOCK["catalogue"]["image_lock_sha256"]
    served[ADDRESS] = capacity_report(
        machine, at=NOW, env_packages={SANDBOX: ADDED_IDENTITY},
        env_options_sha256={SANDBOX: "1" * 64},
        env_image_locks={SANDBOX: {"sha256": lock, "images": locked, "present": present}},
        images=images, **report)
    asyncio.run(fleet.poll_once())
    fleet.served = served
    return fleet


def two_machine_fleet(tmp_path):
    keys = tmp_path / "two"
    keys.mkdir(parents=True)
    first, second = signer(keys, "a", "ka"), signer(keys, "b", "kb")
    documents = [machine_document(first), machine_document(second, machine_id="machine-2", address=SECOND)]
    served, clock = {}, Clock()

    async def read_documents():
        return documents

    async def fetch(address):
        return served.get(address)

    fleet = Fleet(read_documents=read_documents, fetch_report=fetch, clock=clock)
    asyncio.run(fleet.refresh_directory())
    lock = BLOCK["catalogue"]["image_lock_sha256"]

    def report(signed, at, *, images, present=2, package=ADDED_IDENTITY, lock_sha256=lock, **change):
        kw = dict(at=at, env_packages={SANDBOX: package}, env_options_sha256={SANDBOX: "1" * 64},
                  env_image_locks={SANDBOX: {"sha256": lock_sha256, "images": 2, "present": present}}, images=images,
                  **change)
        if signed is second:
            kw.update(machine_id="machine-2", address=SECOND)
        return capacity_report(signed, **kw)

    return fleet, clock, served, first, second, report


# --- the whole activation ---------------------------------------------------------------------------------------
def stub_fleet(complete=True):
    return SimpleNamespace(serving=lambda *a, **k: ["m"], reported_packages=lambda env: [],
                           images_served=lambda *a, **k: frozenset({IMAGE_A, IMAGE_B}),
                           view_complete=lambda *a, **k: complete)


def test_serving_takes_budgets_and_validity_together(tmp_path):
    fleet = serving_fleet(tmp_path, present=2, images=[IMAGE_A])
    with pytest.raises(ValueError):
        fleet.serving(SANDBOX, ADDED_IDENTITY, budgets={}, now=NOW)
    with pytest.raises(ValueError):
        fleet.view_complete(SANDBOX, ADDED_IDENTITY, validity_s=1, now=NOW)


def test_the_open_path_places_no_session_on_a_machine_under_another_lock(tmp_path):
    lock = BLOCK["catalogue"]["image_lock_sha256"]
    budgets = dict(BLOCK["environment"]["episode"]["budgets"])
    place = dict(image=IMAGE_A, env=SANDBOX, env_package=ADDED_IDENTITY, budgets=budgets, validity_s=VALIDITY,
                 now=NOW, env_options_sha256="1" * 64)
    fleet = serving_fleet(tmp_path / "same", present=2, images=[IMAGE_A])
    assert fleet.pick(**place, env_lock_sha256=lock) is not None
    other = serving_fleet(tmp_path / "other", present=2, images=[IMAGE_A], locked=2)
    other_lock = "9" * 64
    assert other.pick(**place, env_lock_sha256=other_lock) is None
    assert other.could_place(**place, env_lock_sha256=other_lock) is False
    assert other.servable(**place, env_lock_sha256=other_lock) == 0
    assert other.nearest(**place, env_lock_sha256=other_lock) is None
    assert other.pick(**place) is not None                       # no lock asked: placed as before
    # a machine that reports no lock for the env is not judged by it (an older gateway)
    (tmp_path / "plain").mkdir()
    plain, machine, served, _ = make(tmp_path / "plain")
    served[ADDRESS] = capacity_report(machine, at=NOW, env_packages={SANDBOX: ADDED_IDENTITY},
                                      env_options_sha256={SANDBOX: "1" * 64}, images=[IMAGE_A])
    asyncio.run(plain.poll_once())
    assert plain.pick(**place, env_lock_sha256=other_lock) is not None


def test_a_bridged_task_carries_its_lock_to_the_placement():
    from reliquary.sandbox.sessions import _reserving

    assert _reserving(SimpleNamespace(reserve=None, env_lock_sha256=None)) == {}
    assert _reserving(SimpleNamespace(reserve={"cpu_millis": 1}, env_lock_sha256="9" * 64)) == {
        "reserve": {"cpu_millis": 1}, "env_lock_sha256": "9" * 64}


# --- installed distributions -----------------------------------------------------------------------------------------
def _dist(name, version, files, record="rec"):
    return SimpleNamespace(metadata={"Name": name}, version=version, read_text=lambda file: record,
                           files=[PurePosixPath(f) for f in files])


# --- a module already loaded in this process ---------------------------------------------------------------------------
def test_a_module_loaded_at_another_identity_needs_a_restart():
    from reliquary.environment.bridged_episode import loaded_identity_problem

    loaded = {}
    assert loaded_identity_problem("reliquary_terminal", ADDED_IDENTITY, loaded) is None
    assert loaded_identity_problem("reliquary_terminal", ADDED_IDENTITY, loaded) is None
    problem = loaded_identity_problem("reliquary_terminal", "reliquary-terminal==0.3.0+gdead.vfb2", loaded)
    assert "validator restart required" in problem
    assert loaded_identity_problem(None, "x", loaded) is None


def test_the_bridged_source_refuses_a_rebuild_at_another_identity(monkeypatch):
    from reliquary.environment import bridged_episode

    monkeypatch.setattr(bridged_episode, "_LOADED_IDENTITIES", {"reliquary_terminal": "x==0+g0.vfb2"})
    module = SimpleNamespace(__name__="reliquary_terminal")
    monkeypatch.setattr("reliquary_sandbox_service.episodes.bridged_env_package_of", lambda m: ADDED_IDENTITY)
    env = SimpleNamespace(module=module, lock=SimpleNamespace(images={"a": IMAGE_A}), sandbox_task=None)
    config = bridged_episode.BridgedEnvConfig.from_catalogue(ADDED, {**BLOCK["catalogue"], "image_lock_sha256": None})
    with pytest.raises(ValueError, match="validator restart required"):
        bridged_episode.BridgedEpisodeSource(config, env=env)


