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
from reliquary.validator.catalogue_activation import (
    ADDED_PACKAGES_FILE,
    NotYet,
    activation_problem,
    allowed_packages,
    attestation_problem,
    before_import_problem,
    budget_problem,
    identities_problem,
    install_problem,
    install_set_problem,
    installed_distributions,
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


def test_the_added_module_is_its_own_top_level_module_exactly():
    source = Source(module="verifiers.v1.tasksets.reliquary_terminal")
    assert "not reliquary_terminal" in module_problem(source, BLOCK, {}, ORDER)
    assert module_problem(Source(module="Reliquary_Terminal"), BLOCK, {}, ORDER) is not None


def test_an_active_module_under_the_verifiers_namespace_is_its_taskset_module():
    active = {"swe": Source(module="verifiers.v1.tasksets.reliquary_terminal", identity="other==1+g0.vfb2")}
    assert "swe" in module_problem(Source(), BLOCK, active, ORDER)


@pytest.mark.parametrize("package", ["json==0.3.0", "asyncio==0.3.0", "typing==1.0"])
def test_a_module_named_like_the_standard_library_is_refused(package):
    assert "standard-library" in name_problem(_named(package))


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


def test_the_servable_scan_is_reused_while_the_fleets_images_did_not_change(tmp_path):
    fleet = serving_fleet(tmp_path, present=2, images=[IMAGE_A, IMAGE_B])
    source, memo = Source(), {}
    assert machines_problem(fleet, BLOCK, validity_s=VALIDITY, source=source, now=NOW, memo=memo) is None
    scanned = source.resolved
    assert scanned > 0
    assert machines_problem(fleet, BLOCK, validity_s=VALIDITY, source=source, now=NOW, memo=memo) is None
    assert source.resolved == scanned                                     # the boundary reused the prestage's scan


def test_a_machine_must_serve_it_with_enough_locked_images_and_enough_servable_tasks(tmp_path):
    fleet = serving_fleet(tmp_path / "a", present=0, images=[])
    problem = machines_problem(fleet, BLOCK, validity_s=VALIDITY, source=Source(), now=NOW)
    assert problem is not None and not isinstance(problem, NotYet)
    assert "0 machine" in problem and ADDED_IDENTITY in problem
    fleet = serving_fleet(tmp_path / "b", present=2, images=[IMAGE_A, IMAGE_B])
    assert machines_problem(fleet, BLOCK, validity_s=VALIDITY, source=Source(), now=NOW) is None
    # a lock of thousands of images: a threshold present, and the servable tasks counted from the reports
    many = frozenset(f"registry.example/t@sha256:{i:064x}" for i in range(5_000))
    fleet = serving_fleet(tmp_path / "c", present=32, images=[IMAGE_A], locked=5_000)
    big = Source(lock_images=many, rows=1_000)
    problem = machines_problem(fleet, BLOCK, validity_s=VALIDITY, source=Source(lock_images=many, rows=1_000, images=lambda i: IMAGE_B),
                               now=NOW)
    assert "servable" in problem and not isinstance(problem, NotYet)
    assert machines_problem(fleet, BLOCK, validity_s=VALIDITY, source=big, now=NOW) is None      # every other task is on IMAGE_A


def test_a_task_whose_image_cannot_be_read_is_not_servable(tmp_path):
    fleet = serving_fleet(tmp_path, present=2, images=[IMAGE_A, IMAGE_B])

    def broken(index):
        raise KeyError(index)

    problem = machines_problem(fleet, BLOCK, validity_s=VALIDITY, source=Source(images=broken), now=NOW)
    assert "servable" in problem and "could not be read" in problem


# --- an incomplete or stale fleet view is "not yet", never a final refusal -----------------------------------------
def test_a_directory_never_read_is_not_yet(tmp_path):
    async def read_documents():
        return []

    async def fetch(address):
        return None

    fleet = Fleet(read_documents=read_documents, fetch_report=fetch, clock=Clock())
    problem = machines_problem(fleet, BLOCK, validity_s=VALIDITY, source=Source(), now=NOW)
    assert isinstance(problem, NotYet)


def test_no_report_heard_yet_is_not_yet(tmp_path):
    fleet, _machine, _served, _clock = make(tmp_path)
    assert isinstance(machines_problem(fleet, BLOCK, validity_s=VALIDITY, source=Source(), now=NOW), NotYet)


def test_a_stale_directory_is_not_yet(tmp_path):
    fleet = serving_fleet(tmp_path, present=2, images=[IMAGE_A, IMAGE_B])
    later = NOW + 10 * 86400                                                # far past the directory's max age
    assert isinstance(machines_problem(fleet, BLOCK, validity_s=VALIDITY, source=Source(), now=later), NotYet)


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


def test_a_serving_machine_that_missed_its_heartbeat_makes_a_shortfall_not_yet(tmp_path):
    fleet, clock, served, first, second, report = two_machine_fleet(tmp_path)
    # the first machine serves the env without any of its images; the second holds them
    served[ADDRESS], served[SECOND] = (report(first, NOW, images=[], present=0),
                                       report(second, NOW, images=[IMAGE_A, IMAGE_B]))
    asyncio.run(fleet.poll_once())
    assert machines_problem(fleet, BLOCK, validity_s=VALIDITY, source=Source(), now=NOW) is None
    clock.now = NOW + HEARTBEAT_MAX_AGE_S + 10
    served[ADDRESS], served[SECOND] = report(first, clock.now, images=[], present=0), None
    asyncio.run(fleet.poll_once())
    assert "machine-2" not in fleet.drained
    problem = machines_problem(fleet, BLOCK, validity_s=VALIDITY, source=Source(), now=clock.now)
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
    problem = machines_problem(fleet, BLOCK, validity_s=VALIDITY, source=only_b, now=clock.now)
    assert isinstance(problem, NotYet) and "servable" in problem


def test_a_drained_serving_machine_makes_a_shortfall_not_yet(tmp_path):
    fleet, _clock, served, first, second, report = two_machine_fleet(tmp_path)
    served[ADDRESS], served[SECOND] = (report(first, NOW, images=[], present=0),
                                       report(second, NOW, images=[IMAGE_A, IMAGE_B]))
    asyncio.run(fleet.poll_once())
    fleet.drained.add("machine-2")
    assert isinstance(machines_problem(fleet, BLOCK, validity_s=VALIDITY, source=Source(), now=NOW), NotYet)


def test_a_complete_fresh_view_gives_a_final_answer(tmp_path):
    fleet, _clock, served, first, second, report = two_machine_fleet(tmp_path)
    # both fresh; one serves another identity (not this env's install), the other holds none of the images
    served[ADDRESS], served[SECOND] = (report(first, NOW, images=[], present=0),
                                       report(second, NOW, images=[IMAGE_A], package="other==1+g0.vfb2"))
    asyncio.run(fleet.poll_once())
    problem = machines_problem(fleet, BLOCK, validity_s=VALIDITY, source=Source(), now=NOW)
    assert problem is not None and not isinstance(problem, NotYet)


def test_a_stale_machine_of_another_identity_does_not_hold_the_answer(tmp_path):
    fleet, clock, served, first, second, report = two_machine_fleet(tmp_path)
    served[ADDRESS], served[SECOND] = (report(first, NOW, images=[], present=0),
                                       report(second, NOW, images=[IMAGE_A], package="other==1+g0.vfb2"))
    asyncio.run(fleet.poll_once())
    clock.now = NOW + HEARTBEAT_MAX_AGE_S + 10
    served[ADDRESS], served[SECOND] = report(first, clock.now, images=[], present=0), None
    asyncio.run(fleet.poll_once())
    problem = machines_problem(fleet, BLOCK, validity_s=VALIDITY, source=Source(), now=clock.now)
    assert problem is not None and not isinstance(problem, NotYet)


# --- the whole activation ---------------------------------------------------------------------------------------
def stub_fleet(complete=True):
    return SimpleNamespace(serving=lambda *a, **k: ["m"], reported_packages=lambda env: [],
                           images_served=lambda *a, **k: frozenset({IMAGE_A, IMAGE_B}),
                           view_complete=lambda *a, **k: complete)


def test_activation_runs_the_checks_in_order(tmp_path):
    built = SimpleNamespace(source=Source())
    kwargs = dict(build=lambda config: built, fleet=stub_fleet(), allowed=ALLOWED, active_sources={}, order=ORDER,
                  boot=BOOT, current=CURRENT, validity_s=VALIDITY, loaded={})
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
                  active_sources={}, order=ORDER, boot=BOOT, current=CURRENT, validity_s=VALIDITY, loaded={})
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
                                              allowed=ALLOWED, active_sources={}, order=ORDER, boot=BOOT,
                                              current=CURRENT, validity_s=VALIDITY, loaded={})
    assert isinstance(problem, NotYet) and environment is None


# --- only machines the open path could place the env's sessions on count -------------------------------------------
@pytest.mark.parametrize("report, why", [
    ({"caps": {**CAPS, "max_memory_bytes": GIB}}, "caps below its memory budget"),
    ({"caps": {**CAPS, "max_wall_s": 60}}, "caps below its wall budget"),
    ({"caps": {**CAPS, "max_token_validity_s": VALIDITY - 1}}, "tokens shorter than the issuer's"),
    ({"tools_version": "reliquary-tools/0"}, "an old tools version"),
])
def test_a_machine_the_open_path_cannot_place_on_does_not_count(tmp_path, report, why):
    fleet = serving_fleet(tmp_path, present=2, images=[IMAGE_A, IMAGE_B], **report)
    problem = machines_problem(fleet, BLOCK, validity_s=VALIDITY, source=Source(), now=NOW)
    assert problem is not None and not isinstance(problem, NotYet) and "0 machine" in problem, why
    budgets = dict(BLOCK["environment"]["episode"]["budgets"])
    assert fleet.could_place(image=IMAGE_A, env=SANDBOX, env_package=ADDED_IDENTITY, budgets=budgets,
                             validity_s=VALIDITY, now=NOW, env_options_sha256="1" * 64) is False


def test_a_machine_with_room_for_the_envs_budgets_counts(tmp_path):
    fleet = serving_fleet(tmp_path, present=2, images=[IMAGE_A, IMAGE_B], caps=dict(CAPS))
    assert machines_problem(fleet, BLOCK, validity_s=VALIDITY, source=Source(), now=NOW) is None


def test_serving_takes_budgets_and_validity_together(tmp_path):
    fleet = serving_fleet(tmp_path, present=2, images=[IMAGE_A])
    with pytest.raises(ValueError):
        fleet.serving(SANDBOX, ADDED_IDENTITY, budgets={}, now=NOW)
    with pytest.raises(ValueError):
        fleet.view_complete(SANDBOX, ADDED_IDENTITY, validity_s=1, now=NOW)


def test_a_stale_machine_that_could_never_host_it_does_not_hold_the_answer(tmp_path):
    fleet, clock, served, first, second, report = two_machine_fleet(tmp_path)
    served[ADDRESS], served[SECOND] = (report(first, NOW, images=[], present=0),
                                       report(second, NOW, images=[IMAGE_A, IMAGE_B],
                                              caps={**CAPS, "max_memory_bytes": GIB}))
    asyncio.run(fleet.poll_once())
    clock.now = NOW + HEARTBEAT_MAX_AGE_S + 10
    served[ADDRESS], served[SECOND] = report(first, clock.now, images=[], present=0), None
    asyncio.run(fleet.poll_once())
    problem = machines_problem(fleet, BLOCK, validity_s=VALIDITY, source=Source(), now=clock.now)
    assert problem is not None and not isinstance(problem, NotYet)


# --- the env's image lock --------------------------------------------------------------------------------------------
def test_images_held_under_another_lock_are_not_servable(tmp_path):
    fleet, _clock, served, first, second, report = two_machine_fleet(tmp_path)
    # the first holds the lock's threshold but only IMAGE_A; the second holds IMAGE_B under ANOTHER lock
    served[ADDRESS], served[SECOND] = (report(first, NOW, images=[IMAGE_A]),
                                       report(second, NOW, images=[IMAGE_B], lock_sha256="9" * 64))
    asyncio.run(fleet.poll_once())
    only_b = Source(images=lambda index: IMAGE_B)
    problem = machines_problem(fleet, BLOCK, validity_s=VALIDITY, source=only_b, now=NOW)
    assert problem is not None and "servable" in problem and not isinstance(problem, NotYet)
    lock = BLOCK["catalogue"]["image_lock_sha256"]
    assert IMAGE_B in fleet.images_served(SANDBOX, ADDED_IDENTITY, now=NOW)            # without the lock: counted
    assert IMAGE_B not in fleet.images_served(SANDBOX, ADDED_IDENTITY, lock_sha256=lock, now=NOW)


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


# --- the fleet's view: each branch of an incomplete one ---------------------------------------------------------------
def test_a_changed_image_set_is_counted_again(tmp_path):
    only_b, memo = Source(images=lambda index: IMAGE_B), {}
    holds_b = serving_fleet(tmp_path / "b", present=2, images=[IMAGE_B])
    assert machines_problem(holds_b, BLOCK, validity_s=VALIDITY, source=only_b, now=NOW, memo=memo) is None
    holds_a = serving_fleet(tmp_path / "a", present=2, images=[IMAGE_A])
    problem = machines_problem(holds_a, BLOCK, validity_s=VALIDITY, source=only_b, now=NOW, memo=memo)
    assert problem is not None and "servable" in problem


def test_each_tasks_image_is_read_once_per_manifest(tmp_path):
    only_b, memo = Source(images=lambda index: IMAGE_B), {}
    holds_a = serving_fleet(tmp_path / "a", present=2, images=[IMAGE_A])
    assert machines_problem(holds_a, BLOCK, validity_s=VALIDITY, source=only_b, now=NOW, memo=memo) is not None
    read = only_b.resolved
    assert read == len(only_b)                                    # every task read once: none is servable
    holds_b = serving_fleet(tmp_path / "b", present=2, images=[IMAGE_B])
    assert machines_problem(holds_b, BLOCK, validity_s=VALIDITY, source=only_b, now=NOW, memo=memo) is None
    assert only_b.resolved == read                                # counted again, never read again
    other = Source(images=lambda index: IMAGE_B, manifest="0" * 64)
    machines_problem(holds_b, BLOCK, validity_s=VALIDITY, source=other, now=NOW, memo=memo)
    assert other.resolved > 0                                     # another manifest: read again
    manifest, images = memo["tasks"]
    assert manifest == "0" * 64 and isinstance(images, tuple)


def test_a_serving_machine_whose_resources_cannot_be_read_makes_a_shortfall_not_yet(tmp_path):
    fleet = serving_fleet(tmp_path, present=2, images=[IMAGE_A, IMAGE_B])   # its report has no resources block
    fleet.resources_required = True
    problem = machines_problem(fleet, BLOCK, validity_s=VALIDITY, source=Source(), now=NOW)
    assert isinstance(problem, NotYet) and "0 machine" in problem


def test_a_silent_poll_after_reports_makes_a_shortfall_not_yet(tmp_path):
    fleet = serving_fleet(tmp_path, present=0, images=[])
    problem = machines_problem(fleet, BLOCK, validity_s=VALIDITY, source=Source(), now=NOW)
    assert problem is not None and not isinstance(problem, NotYet)
    fleet.served[ADDRESS] = None                                  # our egress: no machine heard at all
    asyncio.run(fleet.poll_once())
    assert isinstance(machines_problem(fleet, BLOCK, validity_s=VALIDITY, source=Source(), now=NOW), NotYet)


# --- installed distributions -----------------------------------------------------------------------------------------
def _dist(name, version, files, record="rec"):
    return SimpleNamespace(metadata={"Name": name}, version=version, read_text=lambda file: record,
                           files=[PurePosixPath(f) for f in files])


def test_installed_distributions_reads_names_versions_records_and_top_level_names():
    found = installed_distributions(lambda: [
        _dist("Reliquary_Terminal", "0.3.0", ["reliquary_terminal/__init__.py", "reliquary_terminal-0.3.0.dist-info/RECORD",
                                              "../../bin/tool", "__pycache__/x.pyc", "evil.pth", "extra.py",
                                              "native.cpython-312-x86_64-linux-gnu.so"])])
    digest = hashlib.sha256(b"rec").hexdigest()
    assert found == {"reliquary-terminal": frozenset(
        {("0.3.0", digest, frozenset({"reliquary_terminal", "evil.pth", "extra", "native"}))})}


def test_installed_distributions_reads_this_venv():
    found = installed_distributions()
    assert "pytest" in found


def test_the_boot_set_plus_the_added_distribution_is_accepted():
    assert install_set_problem(BLOCK, BOOT, CURRENT) is None
    assert install_set_problem(BLOCK, CURRENT, CURRENT) is None      # installed before this validator started


@pytest.mark.parametrize("current, why", [
    ({**CURRENT, "verifiers": frozenset({("0.2", "0" * 64, frozenset({"verifiers"}))})}, "verifiers (changed)"),
    ({**CURRENT, "requests": frozenset({("2", "0" * 64, frozenset({"requests"}))})}, "requests (new)"),
    ({"reliquary-terminal": CURRENT["reliquary-terminal"]}, "verifiers (gone)"),
    ({**BOOT, "reliquary-terminal": frozenset({("0.3.1", "1" * 64, frozenset({"reliquary_terminal"}))})}, "0.3.1"),
    ({**BOOT, "reliquary-terminal": frozenset({("0.3.0", "1" * 64, frozenset({"reliquary_terminal", "utils"}))})},
     "top-level"),
    ({**BOOT, "reliquary-terminal": frozenset({("0.3.0", "1" * 64, frozenset({"reliquary_terminal", "x.pth"}))})},
     "top-level"),
    ({**BOOT, "reliquary-terminal": frozenset({("0.3.0", "1" * 64, frozenset({"reliquary_terminal"})),
                                               ("0.3.0", "2" * 64, frozenset({"reliquary_terminal"}))})}, "2 times"),
    (BOOT, "0 times"),
])
def test_any_other_install_is_refused(current, why):
    assert why in install_set_problem(BLOCK, BOOT, current)


def test_a_module_another_distribution_installs_is_refused():
    boot = {**BOOT, "shim": frozenset({("1", "3" * 64, frozenset({"shim", "reliquary_terminal"}))})}
    assert "shim" in install_set_problem(BLOCK, boot, {**boot, **{k: v for k, v in CURRENT.items() if k != "verifiers"},
                                                       "verifiers": BOOT["verifiers"]})


def test_a_distribution_changed_since_boot_needs_a_restart():
    changed = {**BOOT, "reliquary-terminal": frozenset({("0.3.0", "9" * 64, frozenset({"reliquary_terminal"}))})}
    assert "restart" in install_set_problem(BLOCK, CURRENT, changed)


def test_the_checks_before_any_import_run_in_order():
    assert before_import_problem(BLOCK, ALLOWED, ORDER, boot=BOOT, current=CURRENT) is None
    assert "allow" in before_import_problem(BLOCK, frozenset(), ORDER, boot=BOOT, current=CURRENT)
    assert "not plain" in before_import_problem(_named("a.b==0.3.0"), frozenset({"a.b==0.3.0"}), ORDER, boot=BOOT,
                                                current=CURRENT)
    assert "max_turns" in before_import_problem(added_block(episode={"max_turns": 9}), ALLOWED, ORDER, boot=BOOT,
                                                current=CURRENT)
    assert "moved" in before_import_problem(BLOCK, ALLOWED, ORDER, boot={}, current=CURRENT)


def test_activation_refuses_a_moved_install_before_building():
    def build(config):
        pytest.fail("nothing is imported for a refused entry")

    moved = {**CURRENT, "verifiers": frozenset({("0.2", "0" * 64, frozenset({"verifiers"}))})}
    problem, _ = activation_problem(ADDED, BLOCK, attestation(), build=build, fleet=stub_fleet(), allowed=ALLOWED,
                                    active_sources={}, order=ORDER, boot=BOOT, current=moved, validity_s=VALIDITY,
                                    loaded={})
    assert "verifiers" in problem


# --- the allowlist may pin the full identity -------------------------------------------------------------------------
def test_an_allowlist_line_may_pin_the_full_identity():
    assert package_problem(BLOCK, frozenset({ADDED_IDENTITY})) is None
    pinned = frozenset({ADDED_PACKAGE, ADDED_PACKAGE + "+g0000000000000000.vfb2"})
    assert "identity" in package_problem(BLOCK, pinned)


# --- a module already loaded in this process ---------------------------------------------------------------------------
def test_a_module_loaded_at_another_identity_needs_a_restart():
    from reliquary.environment.bridged_episode import loaded_identity_problem

    loaded = {}
    assert loaded_identity_problem("reliquary_terminal", ADDED_IDENTITY, loaded) is None
    assert loaded_identity_problem("reliquary_terminal", ADDED_IDENTITY, loaded) is None
    problem = loaded_identity_problem("reliquary_terminal", "reliquary-terminal==0.3.0+gdead.vfb2", loaded)
    assert "validator restart required" in problem
    assert loaded_identity_problem(None, "x", loaded) is None


def test_activation_refuses_new_code_under_a_module_already_loaded():
    second = "reliquary-terminal==0.3.0+g0000000000000001.vfb2"
    block = added_block(episode={"env_package": second})
    loaded = {}
    kwargs = dict(fleet=stub_fleet(), allowed=ALLOWED, active_sources={}, order=ORDER, boot=BOOT, current=CURRENT,
                  validity_s=VALIDITY, loaded=loaded)
    first, _ = activation_problem(ADDED, BLOCK, attestation(), build=lambda c: SimpleNamespace(source=Source()),
                                  **kwargs)
    assert first is None
    problem, environment = activation_problem(
        ADDED, block, attestation(env_package=second),
        build=lambda c: SimpleNamespace(source=Source(identity=second)), **kwargs)
    assert environment is None and "validator restart required" in problem


def test_the_bridged_source_refuses_a_rebuild_at_another_identity(monkeypatch):
    from reliquary.environment import bridged_episode

    monkeypatch.setattr(bridged_episode, "_LOADED_IDENTITIES", {"reliquary_terminal": "x==0+g0.vfb2"})
    module = SimpleNamespace(__name__="reliquary_terminal")
    monkeypatch.setattr("reliquary_sandbox_service.episodes.bridged_env_package_of", lambda m: ADDED_IDENTITY)
    env = SimpleNamespace(module=module, lock=SimpleNamespace(images={"a": IMAGE_A}), sandbox_task=None)
    config = bridged_episode.BridgedEnvConfig.from_catalogue(ADDED, {**BLOCK["catalogue"], "image_lock_sha256": None})
    with pytest.raises(ValueError, match="validator restart required"):
        bridged_episode.BridgedEpisodeSource(config, env=env)


# --- an I/O error is "not yet" ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("error", [OSError(errno.EIO, "disk"), ConnectionError("hub"), TimeoutError()])
def test_a_build_that_fails_on_io_is_not_yet(error):
    def build(config):
        raise error

    problem, environment = activation_problem(ADDED, BLOCK, attestation(), build=build, fleet=stub_fleet(),
                                              allowed=ALLOWED, active_sources={}, order=ORDER, boot=BOOT,
                                              current=CURRENT, validity_s=VALIDITY, loaded={})
    assert isinstance(problem, NotYet) and environment is None


def test_an_identity_that_cannot_be_read_now_is_not_yet():
    class Unreadable(Source):
        def installed_identity(self):
            raise OSError(errno.EBUSY, "busy")

    problem = identities_problem({"swe": Unreadable(identity="a==1+g1.vfb2")})
    assert isinstance(problem, NotYet)
    moved = identities_problem({"swe": Unreadable(identity="a==1+g1.vfb2"),
                                "z": Source(identity="b==1+g1.vfb2", installed="b==1+g2.vfb2")})
    assert "z" in moved and not isinstance(moved, NotYet)
