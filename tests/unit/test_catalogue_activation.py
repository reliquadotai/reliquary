"""The activation checks of an env added to a running run."""
import asyncio
import os
from types import SimpleNamespace

import pytest

pytest.importorskip("reliquary_sandbox_verifiers")

from reliquary.protocol.service_contract import EPISODE_BUDGET_FIELDS
from reliquary.sandbox.fleet import HEARTBEAT_MAX_AGE_S, Fleet
from reliquary.validator.catalogue_activation import (
    ADDED_PACKAGES_FILE,
    NotYet,
    activation_problem,
    allowed_packages,
    attestation_problem,
    budget_problem,
    identities_problem,
    install_problem,
    machines_problem,
    module_problem,
    name_problem,
    package_problem,
)
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

BLOCK = added_block()
OPTIONS = BLOCK["catalogue"]["options"]
IMAGE_A = "registry.example/t@sha256:" + "a" * 64
IMAGE_B = "registry.example/t@sha256:" + "b" * 64
ALLOWED = frozenset({ADDED_PACKAGE})
ORDER = episode_contract()
SANDBOX = BLOCK["environment"]["episode"]["sandbox_env"]
SECOND = "http://10.0.0.6:8080"


def attestation(**overrides):
    """Shaped like a sandbox conformance gate report."""
    value = {"gate": True, "passed": True, "problems": [], "fake": False, "sample_size": 32, "repeat": 2,
             "env_package": ADDED_IDENTITY, "entry": f"verifiers:{ADDED_PACKAGE}", "split": "tmax",
             "options": dict(OPTIONS), "gate_reasons": [], "env": "conformance-env"}
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


def test_a_gate_report_of_this_exact_env_attests():
    assert attestation_problem(attestation(), BLOCK) is None


@pytest.mark.parametrize("change, why", [
    ({"gate": False}, "not a gate"), ({"passed": False}, "did not pass"), ({"problems": ["x"]}, "problems"),
    ({"fake": True}, "fake"), ({"sample_size": 4}, "sample"), ({"env_package": "x==1+g0.vfb2"}, "env package"),
    ({"entry": "verifiers:reliquary-terminal==0.2.0"}, "taskset"), ({"split": "rl"}, "split"),
    ({"options": {"tools": ["bash", "edit"]}}, "options"),
])
def test_any_other_report_does_not(change, why):
    assert why in attestation_problem(attestation(**change), BLOCK)


def test_the_attestation_must_be_an_object():
    assert "object" in attestation_problem(["gate"], BLOCK)


def test_the_install_must_be_the_entrys():
    assert install_problem(Source(), BLOCK) is None
    assert "installed" in install_problem(Source(identity="x==1+g0.vfb2"), BLOCK)
    assert "lock" in install_problem(Source(lock_sha256="0" * 64), BLOCK)
    assert "manifest" in install_problem(Source(manifest="0" * 64), BLOCK)
    assert "rows" in install_problem(Source(rows=7), BLOCK)


# --- names: a plain distribution and a plain import module, its own -------------------------------------------
def _named(package):
    identity = package + "+g827598fa320ad040.vfb2"
    return added_block(catalogue={"package": package}, episode={"env_package": identity})


def test_a_plain_distribution_name_is_accepted():
    assert name_problem(BLOCK) is None
    assert name_problem(_named("Reliquary_Terminal2==0.3.0")) is None


@pytest.mark.parametrize("package", [
    "reliquary.terminal==0.3.0",          # pip's name for reliquary-terminal, a dotted module
    "reliquary--terminal==0.3.0",         # pip's name for reliquary-terminal, module reliquary__terminal
    "reliquary_-terminal==0.3.0",
    "reliquary-terminal-==0.3.0",
    "class==0.3.0",                       # a keyword is no module
    "9terminal==0.3.0",                   # nor a leading digit
])
def test_a_distribution_whose_module_is_not_a_plain_identifier_is_refused(package):
    assert name_problem(_named(package)) is not None


def test_a_dotted_entry_identity_is_refused_too():
    block = added_block(episode={"env_package": "reliquary.terminal==0.3.0+g827598fa320ad040.vfb2"})
    assert "reliquary.terminal" in name_problem(block)


def test_the_added_package_has_its_own_import_module():
    assert module_problem(Source(), BLOCK, {"swe": Source(module="reliquary_swe", identity="reliquary-swe==1")},
                          ORDER) is None
    assert "not reliquary_terminal" in module_problem(Source(module="reliquary_swe"), BLOCK, {}, ORDER)
    assert "swe" in module_problem(Source(), BLOCK, {"swe": Source(module="reliquary_terminal")}, ORDER)


def test_a_module_imported_under_the_verifiers_namespace_is_its_taskset_module():
    source = Source(module="verifiers.v1.tasksets.reliquary_terminal")
    assert module_problem(source, BLOCK, {}, ORDER) is None
    assert module_problem(Source(module="verifiers.v1.tasksets.other"), BLOCK, {}, ORDER) is not None


def test_a_module_an_active_env_or_the_order_derives_from_its_distribution_is_refused():
    # an active env whose source names no module: its distribution names it
    other = Source(module=None, identity="Reliquary.Terminal==0.1.0+g0.vfb2")
    assert "swe" in module_problem(Source(), BLOCK, {"swe": other}, ORDER)
    order = episode_contract(episode=episode_block(env_package="reliquary_terminal==0.1.0+g0.vfb2"))
    assert "order" in module_problem(Source(), BLOCK, {}, order)


# --- budgets: every per-env budget bounded by the order's ------------------------------------------------------
def test_an_added_env_within_the_orders_budgets_is_accepted():
    assert budget_problem(BLOCK, ORDER) is None


@pytest.mark.parametrize("field", ["max_episode_tokens", "max_turns", "max_tokens_per_turn"])
def test_each_episode_limit_above_the_orders_is_refused(field):
    episode = BLOCK["environment"]["episode"]
    block = added_block(episode={field: episode[field] + 1})
    assert field in budget_problem(block, ORDER)


@pytest.mark.parametrize("field", EPISODE_BUDGET_FIELDS)
def test_each_session_budget_above_the_orders_is_refused(field):
    budgets = dict(BLOCK["environment"]["episode"]["budgets"])
    assert budget_problem(added_block(episode={"budgets": budgets}), ORDER) is None
    budgets[field] += 1
    assert f"budgets.{field}" in budget_problem(added_block(episode={"budgets": budgets}), ORDER)


def test_a_budget_the_activation_does_not_know_is_refused():
    budgets = {**BLOCK["environment"]["episode"]["budgets"], "gpus": 1}
    assert "gpus" in budget_problem(added_block(episode={"budgets": budgets}), ORDER)


def test_an_order_without_an_episode_env_bounds_nothing_so_nothing_is_added():
    order = SimpleNamespace(order_environments={"math": {"share_bps": 10000}})
    assert "no episode" in budget_problem(BLOCK, order)


def test_the_bound_is_the_orders_not_the_views():
    view = SimpleNamespace(order_environments=ORDER.order_environments,
                           environments={**ORDER.environments, "big": {"episode": episode_block(max_turns=64)}})
    assert "max_turns" in budget_problem(added_block(episode={"max_turns": 9}), view)


# --- identities, allowlist ---------------------------------------------------------------------------------------
def test_installing_the_new_env_must_not_move_an_active_one():
    assert identities_problem({"swe": Source(identity="a==1+g1.vfb2")}) is None
    problem = identities_problem({"swe": Source(identity="a==1+g1.vfb2", installed="a==1+g2.vfb2")})
    assert "swe" in problem and "a==1+g2.vfb2" in problem


def test_only_an_owner_allowed_package_is_added(tmp_path):
    assert allowed_packages(tmp_path) == frozenset()
    (tmp_path / ADDED_PACKAGES_FILE).write_text(f"# reviewed 10-10\n{ADDED_PACKAGE}\n\n")
    os.chmod(tmp_path / ADDED_PACKAGES_FILE, 0o600)
    assert allowed_packages(tmp_path) == ALLOWED
    assert package_problem(BLOCK, ALLOWED) is None
    assert "allow" in package_problem(BLOCK, frozenset())


def test_an_allowlist_others_can_write_allows_nothing(tmp_path):
    (tmp_path / ADDED_PACKAGES_FILE).write_text(f"{ADDED_PACKAGE}\n")
    os.chmod(tmp_path / ADDED_PACKAGES_FILE, 0o666)
    assert allowed_packages(tmp_path) == frozenset()


# --- machines ------------------------------------------------------------------------------------------------------
def serving_fleet(tmp_path, *, present, images, locked=2):
    tmp_path.mkdir(parents=True, exist_ok=True)
    fleet, machine, served, _ = make(tmp_path)
    lock = BLOCK["catalogue"]["image_lock_sha256"]
    served[ADDRESS] = capacity_report(
        machine, at=NOW, env_packages={SANDBOX: ADDED_IDENTITY},
        env_options_sha256={SANDBOX: "1" * 64},
        env_image_locks={SANDBOX: {"sha256": lock, "images": locked, "present": present}},
        images=images)
    asyncio.run(fleet.poll_once())
    return fleet


def test_the_servable_scan_is_reused_while_the_fleets_images_did_not_change(tmp_path):
    fleet = serving_fleet(tmp_path, present=2, images=[IMAGE_A, IMAGE_B])
    source, memo = Source(), {}
    assert machines_problem(fleet, BLOCK, source=source, now=NOW, memo=memo) is None
    scanned = source.resolved
    assert scanned > 0
    assert machines_problem(fleet, BLOCK, source=source, now=NOW, memo=memo) is None
    assert source.resolved == scanned                                     # the boundary reused the prestage's scan


def test_a_machine_must_serve_it_with_enough_locked_images_and_enough_servable_tasks(tmp_path):
    fleet = serving_fleet(tmp_path / "a", present=0, images=[])
    problem = machines_problem(fleet, BLOCK, source=Source(), now=NOW)
    assert problem is not None and not isinstance(problem, NotYet)
    assert "0 machine" in problem and ADDED_IDENTITY in problem
    fleet = serving_fleet(tmp_path / "b", present=2, images=[IMAGE_A, IMAGE_B])
    assert machines_problem(fleet, BLOCK, source=Source(), now=NOW) is None
    # a lock of thousands of images: a threshold present, and the servable tasks counted from the reports
    many = frozenset(f"registry.example/t@sha256:{i:064x}" for i in range(5_000))
    fleet = serving_fleet(tmp_path / "c", present=32, images=[IMAGE_A], locked=5_000)
    big = Source(lock_images=many, rows=1_000)
    problem = machines_problem(fleet, BLOCK, source=Source(lock_images=many, rows=1_000, images=lambda i: IMAGE_B),
                               now=NOW)
    assert "servable" in problem and not isinstance(problem, NotYet)
    assert machines_problem(fleet, BLOCK, source=big, now=NOW) is None      # every other task is on IMAGE_A


def test_a_task_whose_image_cannot_be_read_is_not_servable(tmp_path):
    fleet = serving_fleet(tmp_path, present=2, images=[IMAGE_A, IMAGE_B])

    def broken(index):
        raise KeyError(index)

    problem = machines_problem(fleet, BLOCK, source=Source(images=broken), now=NOW)
    assert "servable" in problem and "could not be read" in problem


# --- an incomplete or stale fleet view is "not yet", never a final refusal -----------------------------------------
def test_a_directory_never_read_is_not_yet(tmp_path):
    async def read_documents():
        return []

    async def fetch(address):
        return None

    fleet = Fleet(read_documents=read_documents, fetch_report=fetch, clock=Clock())
    problem = machines_problem(fleet, BLOCK, source=Source(), now=NOW)
    assert isinstance(problem, NotYet)


def test_no_report_heard_yet_is_not_yet(tmp_path):
    fleet, _machine, _served, _clock = make(tmp_path)
    assert isinstance(machines_problem(fleet, BLOCK, source=Source(), now=NOW), NotYet)


def test_a_stale_directory_is_not_yet(tmp_path):
    fleet = serving_fleet(tmp_path, present=2, images=[IMAGE_A, IMAGE_B])
    later = NOW + 10 * 86400                                                # far past the directory's max age
    assert isinstance(machines_problem(fleet, BLOCK, source=Source(), now=later), NotYet)


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

    def report(signed, at, *, images, present=2, package=ADDED_IDENTITY):
        kw = dict(at=at, env_packages={SANDBOX: package}, env_options_sha256={SANDBOX: "1" * 64},
                  env_image_locks={SANDBOX: {"sha256": lock, "images": 2, "present": present}}, images=images)
        if signed is second:
            kw.update(machine_id="machine-2", address=SECOND)
        return capacity_report(signed, **kw)

    return fleet, clock, served, first, second, report


def test_a_serving_machine_that_missed_its_heartbeat_makes_a_shortfall_not_yet(tmp_path):
    fleet, clock, served, first, second, report = two_machine_fleet(tmp_path)
    # the first machine serves the env without any of its images; the second holds them
    served[ADDRESS], served[SECOND] = (report(first, NOW, images=[], present=0),
                                       report(second, NOW, images=[IMAGE_A, IMAGE_B]))
    asyncio.run(fleet.poll_once())
    assert machines_problem(fleet, BLOCK, source=Source(), now=NOW) is None
    clock.now = NOW + HEARTBEAT_MAX_AGE_S + 10
    served[ADDRESS], served[SECOND] = report(first, clock.now, images=[], present=0), None
    asyncio.run(fleet.poll_once())
    assert "machine-2" not in fleet.drained
    problem = machines_problem(fleet, BLOCK, source=Source(), now=clock.now)
    assert isinstance(problem, NotYet) and "0 machine" in problem


def test_a_serving_machine_with_too_few_tasks_beside_a_stale_one_is_not_yet(tmp_path):
    fleet, clock, served, first, second, report = two_machine_fleet(tmp_path)
    served[ADDRESS], served[SECOND] = (report(first, NOW, images=[IMAGE_A]),
                                       report(second, NOW, images=[IMAGE_A, IMAGE_B]))
    asyncio.run(fleet.poll_once())
    clock.now = NOW + HEARTBEAT_MAX_AGE_S + 10
    served[ADDRESS], served[SECOND] = report(first, clock.now, images=[IMAGE_A]), None
    asyncio.run(fleet.poll_once())
    only_b = Source(images=lambda index: IMAGE_B)
    problem = machines_problem(fleet, BLOCK, source=only_b, now=clock.now)
    assert isinstance(problem, NotYet) and "servable" in problem


def test_a_drained_serving_machine_makes_a_shortfall_not_yet(tmp_path):
    fleet, _clock, served, first, second, report = two_machine_fleet(tmp_path)
    served[ADDRESS], served[SECOND] = (report(first, NOW, images=[], present=0),
                                       report(second, NOW, images=[IMAGE_A, IMAGE_B]))
    asyncio.run(fleet.poll_once())
    fleet.drained.add("machine-2")
    assert isinstance(machines_problem(fleet, BLOCK, source=Source(), now=NOW), NotYet)


def test_a_complete_fresh_view_gives_a_final_answer(tmp_path):
    fleet, _clock, served, first, second, report = two_machine_fleet(tmp_path)
    # both fresh; one serves another identity (not this env's install), the other holds none of the images
    served[ADDRESS], served[SECOND] = (report(first, NOW, images=[], present=0),
                                       report(second, NOW, images=[IMAGE_A], package="other==1+g0.vfb2"))
    asyncio.run(fleet.poll_once())
    problem = machines_problem(fleet, BLOCK, source=Source(), now=NOW)
    assert problem is not None and not isinstance(problem, NotYet)


def test_a_stale_machine_of_another_identity_does_not_hold_the_answer(tmp_path):
    fleet, clock, served, first, second, report = two_machine_fleet(tmp_path)
    served[ADDRESS], served[SECOND] = (report(first, NOW, images=[], present=0),
                                       report(second, NOW, images=[IMAGE_A], package="other==1+g0.vfb2"))
    asyncio.run(fleet.poll_once())
    clock.now = NOW + HEARTBEAT_MAX_AGE_S + 10
    served[ADDRESS], served[SECOND] = report(first, clock.now, images=[], present=0), None
    asyncio.run(fleet.poll_once())
    problem = machines_problem(fleet, BLOCK, source=Source(), now=clock.now)
    assert problem is not None and not isinstance(problem, NotYet)


# --- the whole activation ---------------------------------------------------------------------------------------
def stub_fleet(complete=True):
    return SimpleNamespace(serving=lambda *a, **k: ["m"], reported_packages=lambda env: [],
                           images_served=lambda *a, **k: frozenset({IMAGE_A, IMAGE_B}),
                           view_complete=lambda *a, **k: complete)


def test_activation_runs_the_checks_in_order(tmp_path):
    built = SimpleNamespace(source=Source())
    kwargs = dict(build=lambda config: built, fleet=stub_fleet(), allowed=ALLOWED, active_sources={}, order=ORDER)
    assert activation_problem(ADDED, BLOCK, attestation(), **kwargs) == (None, built)
    problem, environment = activation_problem(ADDED, BLOCK, attestation(gate=False), **kwargs)
    assert environment is None and problem.startswith("the attestation")
    problem, _ = activation_problem(ADDED, BLOCK, attestation(), **{**kwargs, "allowed": frozenset()})
    assert "allow" in problem

    def broken(config):
        raise ImportError("no reliquary-terminal")

    problem, _ = activation_problem(ADDED, BLOCK, attestation(), **{**kwargs, "build": broken})
    assert "not installed on this validator" in problem
    moved = {"swe": Source(identity="a==1+g1.vfb2", installed="a==1+g2.vfb2", module="a")}
    problem, _ = activation_problem(ADDED, BLOCK, attestation(), **{**kwargs, "active_sources": moved})
    assert "swe" in problem
    clash = {"swe": Source(identity="a==1+g1.vfb2", module="reliquary_terminal")}
    problem, _ = activation_problem(ADDED, BLOCK, attestation(), **{**kwargs, "active_sources": clash})
    assert "module" in problem
    problem, _ = activation_problem(ADDED, BLOCK, attestation(), **{**kwargs, "build": lambda c: SimpleNamespace(
        source=Source(rows=7))})
    assert "rows" in problem


def test_activation_refuses_names_and_budgets_before_building_anything():
    def build(config):
        pytest.fail("nothing is imported for a refused entry")

    kwargs = dict(build=build, fleet=stub_fleet(), allowed=frozenset({"reliquary.terminal==0.3.0"}),
                  active_sources={}, order=ORDER)
    problem, environment = activation_problem(ADDED, _named("reliquary.terminal==0.3.0"), attestation(), **kwargs)
    assert environment is None and "reliquary.terminal" in problem
    big = added_block(episode={"max_turns": 9})
    problem, _ = activation_problem(ADDED, big, attestation(), **{**kwargs, "allowed": ALLOWED})
    assert "max_turns" in problem


def test_activation_passes_a_not_yet_through(tmp_path):
    built = SimpleNamespace(source=Source())
    fleet = SimpleNamespace(serving=lambda *a, **k: [], reported_packages=lambda env: [],
                            images_served=lambda *a, **k: frozenset(), view_complete=lambda *a, **k: False)
    problem, environment = activation_problem(ADDED, BLOCK, attestation(), build=lambda config: built, fleet=fleet,
                                              allowed=ALLOWED, active_sources={}, order=ORDER)
    assert isinstance(problem, NotYet) and environment is None
