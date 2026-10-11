"""Schedule requests that add an env to a running run: the request, its activation checks, the runtime's view."""
import hashlib
import json
import os
import sqlite3

import pytest

from reliquary.protocol.service_schedule import initial_schedule, next_schedule
from reliquary.services.admission_policy import parse_service_announcement
from reliquary.services.runtime import protocol_slot_geometry
from reliquary.services.schedule import (
    ADD_WAIT_MAX_S, ScheduleNotYet, ScheduleRefused, ScheduleRequestStore, build_request, check_request,
    main, raise_activation_problem,
)
from reliquary.services.settlement import settle_window, validate_service_archive_v2
from tests.unit.episode_v2_fixtures import (
    ADDED, ADDED_IDENTITY, ADDED_PACKAGE, EPISODE, WINDOW_BEACON, added_block, episode_contract, episode_runtime,
)
from tests.unit.service_v2_fixtures import MATH

# A conformance gate report of exactly the added env (the format of the sandbox's conformance --report).
ATTESTATION = json.dumps({"gate": True, "passed": True, "problems": [], "fake": False, "sample_size": 32,
                          "env_package": ADDED_IDENTITY, "entry": f"verifiers:{ADDED_PACKAGE}", "split": "tmax",
                          "options": added_block()["catalogue"]["options"]}, sort_keys=True).encode()
SHARES = {MATH: 4000, EPISODE: 3000, ADDED: 3000}


def block():
    return added_block(catalogue={"conformance_sha256": hashlib.sha256(ATTESTATION).hexdigest()})


def request(contract, current, **kwargs):
    kwargs.setdefault("add", {ADDED: {**block(), "attestation": "conformance.json"}})
    return build_request(order_sha256=contract.sha256, revision=current.revision + 1, **kwargs)


def reader(name):
    assert name == "conformance.json"
    return ATTESTATION, json.loads(ATTESTATION)


def accept(name, added, attestation):
    return None


def test_a_request_adds_an_env_and_may_switch_it_on():
    contract = episode_contract()
    current = initial_schedule(contract)
    changed = check_request(request(contract, current, active=[MATH, EPISODE, ADDED], shares=SHARES),
                            contract=contract, current=current, catalogue_check=accept, read_attestation=reader)
    assert ADDED in changed.active_environments() and changed.added_environments() == {ADDED: block()}


def test_an_attestation_that_is_not_the_entrys_is_refused():
    contract = episode_contract()
    current = initial_schedule(contract)
    with pytest.raises(ScheduleRefused, match="attestation"):
        check_request(request(contract, current), contract=contract, current=current, catalogue_check=accept,
                      read_attestation=lambda name: (b"{}", {}))


def test_an_attestation_that_cannot_be_read_is_refused():
    contract = episode_contract()
    current = initial_schedule(contract)

    def missing(name):
        raise ValueError("attestation conformance.json is not in the request folder")

    with pytest.raises(ScheduleRefused, match="cannot be read"):
        check_request(request(contract, current), contract=contract, current=current, catalogue_check=accept,
                      read_attestation=missing)


def test_without_an_activation_check_nothing_is_added():
    contract = episode_contract()
    current = initial_schedule(contract)
    with pytest.raises(ScheduleRefused, match="cannot add"):
        check_request(request(contract, current), contract=contract, current=current, read_attestation=reader)


def test_the_activation_checks_reason_is_the_refusal():
    contract = episode_contract()
    current = initial_schedule(contract)

    def refuse(name, added, attestation):
        raise ScheduleRefused(f"cannot add {name}: no machine serves it")

    with pytest.raises(ScheduleRefused, match="no machine serves it"):
        check_request(request(contract, current), contract=contract, current=current, catalogue_check=refuse,
                      read_attestation=reader)


def test_the_activation_check_is_called_with_the_block_and_the_attestation():
    contract = episode_contract()
    current = initial_schedule(contract)
    seen = []
    check_request(request(contract, current), contract=contract, current=current, read_attestation=reader,
                  catalogue_check=lambda name, added, attestation: seen.append((name, added, attestation)))
    assert seen == [(ADDED, block(), json.loads(ATTESTATION))]


@pytest.mark.parametrize("attestation", ["../x.json", ".hidden", "a/b.json", "", None])
def test_the_attestation_is_a_plain_file_name_of_the_folder(attestation):
    contract = episode_contract()
    current = initial_schedule(contract)
    bad = request(contract, current, add={ADDED: {**block(), "attestation": attestation}})
    with pytest.raises(ScheduleRefused, match="attestation"):
        check_request(bad, contract=contract, current=current, catalogue_check=accept, read_attestation=reader)


@pytest.mark.parametrize("add", [None, {}, [], {ADDED: {"environment": {}}}])
def test_a_malformed_add_is_refused(add):
    contract = episode_contract()
    current = initial_schedule(contract)
    bad = request(contract, current)
    bad["add"] = add
    with pytest.raises(ScheduleRefused, match="add"):
        check_request(bad, contract=contract, current=current, catalogue_check=accept, read_attestation=reader)


def test_a_cooldown_out_of_bounds_for_an_env_added_by_the_same_request_is_refused():
    contract = episode_contract()
    current = initial_schedule(contract)
    with pytest.raises(ScheduleRefused, match="outside the contract's bounds"):
        check_request(request(contract, current, cooldowns={ADDED: 999_999_999}), contract=contract,
                      current=current, catalogue_check=accept, read_attestation=reader)


def test_a_request_without_add_keeps_its_exact_fields():
    contract = episode_contract()
    plain = build_request(order_sha256=contract.sha256, revision=1, cooldowns={MATH: 10})
    assert "add" not in plain
    assert set(plain) == {"schema", "request_id", "order_sha256", "revision", "active", "shares", "cooldowns"}


def test_an_unknown_field_is_still_refused():
    contract = episode_contract()
    current = initial_schedule(contract)
    bad = build_request(order_sha256=contract.sha256, revision=1, cooldowns={MATH: 10})
    bad["extra"] = 1
    with pytest.raises(ScheduleRefused, match="fields"):
        check_request(bad, contract=contract, current=current)


def test_a_final_refusal_of_another_env_wins_over_a_pending_activation():
    from tests.unit.service_v2_fixtures import CODE

    contract = episode_contract(envs=(MATH, EPISODE, CODE),
                                shares={MATH: 5000, EPISODE: 5000, CODE: 0})
    current = initial_schedule(contract)

    def later(name, added, attestation):
        raise ScheduleNotYet("the fleet's view is stale")

    req = request(contract, current, active=[MATH, EPISODE, CODE], shares={MATH: 4000, EPISODE: 3000, CODE: 3000})
    with pytest.raises(ScheduleRefused, match="not installed"):
        check_request(req, contract=contract, current=current, catalogue_check=later, read_attestation=reader,
                      installed_version=lambda name: None)


# ---------------------------------------------------------------- the activation answer's two kinds

def test_a_not_yet_problem_raises_not_yet_and_a_plain_one_refuses():
    from reliquary.validator.catalogue_activation import NotYet

    raise_activation_problem(ADDED, None)
    with pytest.raises(ScheduleNotYet, match="stale"):
        raise_activation_problem(ADDED, NotYet("the fleet's view is stale"))
    with pytest.raises(ScheduleRefused, match="0 machine"):
        raise_activation_problem(ADDED, "0 machine(s) serve it")
    assert not issubclass(ScheduleNotYet, ValueError)       # never caught as a refusal by accident


# ---------------------------------------------------------------- the runtime

def folder_with_attestation(tmp_path):
    os.chmod(tmp_path, 0o700)
    (tmp_path / "conformance.json").write_bytes(ATTESTATION)
    os.chmod(tmp_path / "conformance.json", 0o600)


def runtime_with_added(tmp_path, *, active=True):
    rt = episode_runtime(tmp_path)
    folder_with_attestation(tmp_path)
    store = ScheduleRequestStore(tmp_path)
    switch = {"active": [MATH, EPISODE, ADDED], "shares": SHARES} if active else {}
    store.write(request(rt.contract, rt.schedule, **switch))
    frozen = rt.apply_pending_schedule_request(store, window=1, catalogue_check=accept)
    return rt, store, frozen


def test_the_runtime_adds_at_the_next_window_and_serves_the_view(tmp_path):
    rt, store, frozen = runtime_with_added(tmp_path)
    assert ADDED not in frozen.to_dict()["environments"]                 # window 1 keeps its schedule (R6)
    assert store.status()["status"] == "applied"
    assert rt.contract.sha256 == rt.order_contract.sha256
    assert rt.contract.episode_policy(ADDED).split == "tmax"
    assert ADDED not in rt.order_contract.environments
    picks, slots = protocol_slot_geometry()
    rt.open_window(2, pools={MATH: 0.4, EPISODE: 0.3, ADDED: 0.3}, picks_target=picks, batch_slots=slots)
    announcement = rt.announcement(window=2, randomness=WINDOW_BEACON)
    contract, schedule = parse_service_announcement(announcement)
    assert ADDED in contract.episode_environments and ADDED in schedule.active_environments()
    assert rt.seed_pool(environment=ADDED, prompt_idx=3, window=2) is not None


def test_the_view_follows_a_schedule_applied_directly(tmp_path):
    rt = episode_runtime(tmp_path)
    assert type(rt.contract) is type(rt.order_contract)
    changed = next_schedule(rt.order_contract, rt.schedule, add={ADDED: block()})
    assert rt.apply_schedule(changed, request_id="direct", window=1)
    assert ADDED in rt.contract.environments


def test_the_view_is_the_latest_schedules_after_a_refused_request(tmp_path):
    rt, store, _ = runtime_with_added(tmp_path, active=False)
    store.write(build_request(order_sha256=rt.contract.sha256, revision=99, cooldowns={MATH: 10}))
    rt.apply_pending_schedule_request(store, window=2)
    assert store.status()["status"] == "refused"
    assert ADDED in rt.contract.environments


def test_the_runtime_refuses_an_add_without_an_activation_check(tmp_path):
    rt = episode_runtime(tmp_path)
    folder_with_attestation(tmp_path)
    store = ScheduleRequestStore(tmp_path)
    store.write(request(rt.contract, rt.schedule))
    rt.apply_pending_schedule_request(store, window=1)
    assert store.status()["status"] == "refused" and "cannot add" in store.status()["detail"]
    assert ADDED not in rt.contract.environments


def test_an_env_added_inactive_is_switched_on_later_by_the_cli(tmp_path, capsys):
    rt, store, _ = runtime_with_added(tmp_path, active=False)
    assert ADDED in rt.schedule.added_environments() and ADDED not in rt.schedule.active_environments()
    main(["--folder", str(tmp_path), "set", "--active", f"{MATH},{EPISODE},{ADDED}",
          "--share", f"{MATH}=4000", "--share", f"{EPISODE}=3000", "--share", f"{ADDED}=3000"])
    written = json.loads(capsys.readouterr().out)["written"]
    assert ADDED in written["active"] and "add" not in written
    rt.apply_pending_schedule_request(store, window=2, catalogue_check=accept,
                                      installed_version=lambda name: rt.contract.environments[name]["version"])
    assert ADDED in rt.schedule.active_environments()


def test_the_cli_set_refuses_a_cooldown_out_of_bounds_for_an_added_env(tmp_path, capsys):
    runtime_with_added(tmp_path, active=False)
    with pytest.raises(SystemExit):
        main(["--folder", str(tmp_path), "set", "--cooldown", f"{ADDED}=999999999"])
    assert "outside the contract's bounds" in capsys.readouterr().err


def test_a_restarted_runtime_serves_the_same_view(tmp_path):
    from reliquary.services.runtime import ServiceRuntime
    from tests.unit.service_v2_fixtures import qualification_v2

    rt, _, _ = runtime_with_added(tmp_path)
    rt.close()
    contract = episode_contract()
    again = ServiceRuntime(tmp_path / "runtime.sqlite3", contract, qualification_v2(contract))
    assert again.contract.episode_policy(ADDED) is not None


def test_an_archive_with_an_added_env_validates_from_its_schedule_alone():
    contract = episode_contract()
    schedule = next_schedule(contract, initial_schedule(contract), add={ADDED: block()},
                             active=[MATH, EPISODE, ADDED], shares=SHARES)
    picks, slots = 2, 4
    envelope = {"order_sha256": contract.sha256, "schedule": schedule.to_dict(), "schedule_sha256": schedule.sha256,
                "pools": {MATH: 0.4, EPISODE: 0.3, ADDED: 0.3}, "picks_target": picks, "batch_slots": slots,
                "checkpoint": {"checkpoint_n": 1, "repo": "models/test", "revision": "d" * 40, "sha256": "b" * 64}}
    archive = {"window_start": 3, "window_status": "complete", "rewards_by_hotkey": {},
               "batch": [{"hotkey": "5Hot", "env_name": ADDED, "prompt_idx": 0}]}
    record = settle_window(archive=archive, envelope=envelope, contract=contract, exploration={}, aborted=False)
    validate_service_archive_v2(record, contract, cap=1.0, picks_target=picks, batch_slots=slots)


def test_the_cli_writes_an_add_request(tmp_path, capsys):
    pytest.importorskip("reliquary_sandbox_verifiers")      # the CLI checks the attestation's options digest
    rt = episode_runtime(tmp_path)
    folder_with_attestation(tmp_path)
    entry = tmp_path / "entry.json"
    entry.write_text(json.dumps(block()))
    main(["--folder", str(tmp_path), "add", ADDED, "--entry", str(entry), "--attestation", "conformance.json"])
    written = json.loads(capsys.readouterr().out)["written"]
    assert written["add"][ADDED]["attestation"] == "conformance.json"
    assert written["revision"] == rt.schedule.revision + 1


def test_the_cli_refuses_an_attestation_of_another_env(tmp_path, capsys):
    pytest.importorskip("reliquary_sandbox_verifiers")
    episode_runtime(tmp_path)
    other = json.loads(ATTESTATION) | {"split": "other"}
    raw = json.dumps(other, sort_keys=True).encode()
    os.chmod(tmp_path, 0o700)
    (tmp_path / "conformance.json").write_bytes(raw)
    os.chmod(tmp_path / "conformance.json", 0o600)
    entry = tmp_path / "entry.json"
    entry.write_text(json.dumps(added_block(catalogue={"conformance_sha256": hashlib.sha256(raw).hexdigest()})))
    with pytest.raises(SystemExit):
        main(["--folder", str(tmp_path), "add", ADDED, "--entry", str(entry), "--attestation", "conformance.json"])
    assert "another split" in capsys.readouterr().err


# ---------------------------------------------------------------- an activation that cannot be settled yet

def not_yet(name, added, attestation):
    raise ScheduleNotYet("the fleet's view is stale or incomplete")


def pending_runtime(tmp_path):
    rt = episode_runtime(tmp_path)
    folder_with_attestation(tmp_path)
    store = ScheduleRequestStore(tmp_path)
    req = request(rt.contract, rt.schedule)
    store.write(req)
    return rt, store, req


def recorded(rt, request_id):
    return rt.db.execute("SELECT status FROM service_schedule_requests WHERE request_id=?",
                         (request_id,)).fetchone()


def test_not_yet_leaves_the_request_pending_and_shows_why(tmp_path):
    rt, store, req = pending_runtime(tmp_path)
    schedule = rt.apply_pending_schedule_request(store, window=1, now=1_000.0, catalogue_check=not_yet)
    assert schedule.revision == 0 and rt.schedule.revision == 0 and ADDED not in rt.contract.environments
    status = store.status()
    assert status["request_id"] == req["request_id"] and status["status"] == "waiting"
    assert "stale or incomplete" in status["detail"]
    assert recorded(rt, req["request_id"]) is None                       # never recorded as refused
    rt.apply_pending_schedule_request(store, window=2, now=1_060.0, catalogue_check=accept)
    assert store.status()["status"] == "applied" and ADDED in rt.contract.environments
    assert recorded(rt, req["request_id"]) == ("applied",)


def test_a_pending_request_is_unhandled_for_the_cli(tmp_path, capsys):
    rt, store, req = pending_runtime(tmp_path)
    rt.apply_pending_schedule_request(store, window=1, now=1_000.0, catalogue_check=not_yet)
    main(["--folder", str(tmp_path), "show"])
    shown = json.loads(capsys.readouterr().out)
    assert shown["pending_request"]["request_id"] == req["request_id"]
    assert shown["last_request"]["status"] == "waiting"
    with pytest.raises(SystemExit):
        main(["--folder", str(tmp_path), "set", "--cooldown", f"{MATH}=10"])
    assert "not been handled" in capsys.readouterr().err


def test_not_yet_past_its_bound_is_a_final_refusal(tmp_path):
    rt, store, req = pending_runtime(tmp_path)
    rt.apply_pending_schedule_request(store, window=1, now=1_000.0, catalogue_check=not_yet, add_wait_max_s=600)
    rt.apply_pending_schedule_request(store, window=2, now=1_600.0, catalogue_check=not_yet, add_wait_max_s=600)
    assert store.status()["status"] == "waiting"
    rt.apply_pending_schedule_request(store, window=3, now=1_601.0, catalogue_check=not_yet, add_wait_max_s=600)
    status = store.status()
    assert status["status"] == "refused" and "600" in status["detail"] and "stale or incomplete" in status["detail"]
    assert recorded(rt, req["request_id"]) == ("refused",)
    rt.apply_pending_schedule_request(store, window=4, now=1_700.0, catalogue_check=accept)   # final: never applied
    assert ADDED not in rt.contract.environments and store.status()["status"] == "refused"


def test_the_wait_bound_survives_a_restart(tmp_path):
    from reliquary.services.runtime import ServiceRuntime
    from tests.unit.service_v2_fixtures import qualification_v2

    rt, store, _ = pending_runtime(tmp_path)
    rt.apply_pending_schedule_request(store, window=1, now=1_000.0, catalogue_check=not_yet, add_wait_max_s=600)
    rt.close()
    contract = episode_contract()
    again = ServiceRuntime(tmp_path / "runtime.sqlite3", contract, qualification_v2(contract))
    again.apply_pending_schedule_request(store, window=2, now=1_601.0, catalogue_check=not_yet, add_wait_max_s=600)
    assert store.status()["status"] == "refused"


def test_the_default_wait_bound(tmp_path):
    rt, store, _ = pending_runtime(tmp_path)
    rt.apply_pending_schedule_request(store, window=1, now=1_000.0, catalogue_check=not_yet)
    rt.apply_pending_schedule_request(store, window=2, now=1_000.0 + ADD_WAIT_MAX_S, catalogue_check=not_yet)
    assert store.status()["status"] == "waiting"
    rt.apply_pending_schedule_request(store, window=3, now=1_001.0 + ADD_WAIT_MAX_S, catalogue_check=not_yet)
    assert store.status()["status"] == "refused"


def test_a_replaced_request_starts_its_own_wait(tmp_path):
    rt, store, _ = pending_runtime(tmp_path)
    rt.apply_pending_schedule_request(store, window=1, now=1_000.0, catalogue_check=not_yet, add_wait_max_s=600)
    second = request(rt.contract, rt.schedule)
    store.write(second)
    rt.apply_pending_schedule_request(store, window=2, now=1_590.0, catalogue_check=not_yet, add_wait_max_s=600)
    rt.apply_pending_schedule_request(store, window=3, now=1_700.0, catalogue_check=not_yet, add_wait_max_s=600)
    assert store.status()["request_id"] == second["request_id"] and store.status()["status"] == "waiting"
    waits = rt.db.execute("SELECT request_id FROM service_schedule_waits").fetchall()
    assert waits == [(second["request_id"],)]                             # the replaced one's wait is gone


def test_a_settled_request_leaves_no_wait_behind(tmp_path):
    rt, store, _ = pending_runtime(tmp_path)
    rt.apply_pending_schedule_request(store, window=1, now=1_000.0, catalogue_check=not_yet)
    rt.apply_pending_schedule_request(store, window=2, now=1_010.0, catalogue_check=accept)
    assert rt.db.execute("SELECT COUNT(*) FROM service_schedule_waits").fetchone() == (0,)


def test_the_wait_bound_is_an_operator_setting():
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-c", "from reliquary.services.schedule import ADD_WAIT_MAX_S as v; print(v)"],
                         env={**os.environ, "RELIQUARY_SCHEDULE_ADD_WAIT_MAX_S": "120"}, capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == "120"
    bad = subprocess.run([sys.executable, "-c", "import reliquary.services.schedule"],
                         env={**os.environ, "RELIQUARY_SCHEDULE_ADD_WAIT_MAX_S": "0"}, capture_output=True, text=True,
                         check=False)
    assert bad.returncode != 0 and "RELIQUARY_SCHEDULE_ADD_WAIT_MAX_S" in bad.stderr


def test_without_an_add_request_no_wait_is_written(tmp_path):
    rt = episode_runtime(tmp_path)
    store = ScheduleRequestStore(tmp_path)
    store.write(build_request(order_sha256=rt.contract.sha256, revision=1, cooldowns={MATH: 10}))
    rt.apply_pending_schedule_request(store, window=1)
    assert store.status()["status"] == "applied"
    with sqlite3.connect(tmp_path / "runtime.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM service_schedule_waits").fetchone() == (0,)


# ---------------------------------------------------------------- fix round

@pytest.mark.parametrize("windows", [0, 100_001])
def test_an_added_entrys_own_cooldown_outside_the_bounds_is_refused(windows):
    contract = episode_contract()
    current = initial_schedule(contract)
    assert contract.advice_policy["max_windows"] == 100_000
    entry = added_block(catalogue={"conformance_sha256": hashlib.sha256(ATTESTATION).hexdigest()},
                        cooldown_windows=windows)
    req = request(contract, current, add={ADDED: {**entry, "attestation": "conformance.json"}})
    with pytest.raises(ScheduleRefused, match=f"cannot add {ADDED}: its cooldown {windows} is outside"):
        check_request(req, contract=contract, current=current, catalogue_check=accept, read_attestation=reader)
    # The request's own cooldowns map, within the bounds, decides instead.
    req = request(contract, current, add={ADDED: {**entry, "attestation": "conformance.json"}},
                  cooldowns={ADDED: 10})
    changed = check_request(req, contract=contract, current=current, catalogue_check=accept,
                            read_attestation=reader)
    assert changed.cooldown_windows(ADDED) == 10


def test_switching_on_an_env_added_earlier_checks_its_install(tmp_path):
    rt, store, _ = runtime_with_added(tmp_path, active=False)
    pinned = rt.contract.environments[ADDED]["version"]
    installed = {}                                              # what this validator has, changed below

    def switch_on():
        store.write(build_request(order_sha256=rt.contract.sha256, revision=rt.schedule.revision + 1,
                                  active=[MATH, EPISODE, ADDED], shares=SHARES))
        rt.apply_pending_schedule_request(store, window=2, catalogue_check=accept, installed_version=installed.get)
        return store.status()

    status = switch_on()
    assert status["status"] == "refused" and "not installed" in status["detail"]
    installed[ADDED] = "0" * 64
    status = switch_on()
    assert status["status"] == "refused" and "installed version differs" in status["detail"]
    assert ADDED not in rt.schedule.active_environments()
    installed[ADDED] = pinned
    assert switch_on()["status"] == "applied" and ADDED in rt.schedule.active_environments()


SECOND = "tmax_v3"


def two_adds(contract, current):
    second = added_block(catalogue={"conformance_sha256": hashlib.sha256(ATTESTATION).hexdigest(),
                                    "package": "reliquary-other==0.1.0"},
                         episode={"sandbox_env": "reliquary-other",
                                  "env_package": "reliquary-other==0.1.0+g0123456789abcdef.vfb2"})
    return request(contract, current, add={ADDED: {**block(), "attestation": "conformance.json"},
                                           SECOND: {**second, "attestation": "conformance.json"}})


@pytest.mark.parametrize("order", [(ADDED, SECOND), (SECOND, ADDED)])
def test_with_several_adds_a_final_refusal_wins_and_names_its_env(order):
    contract = episode_contract()
    current = initial_schedule(contract)
    waiting, refused = order
    seen = []

    def check(name, added, attestation):
        seen.append(name)
        if name == waiting:
            raise ScheduleNotYet("the fleet's view is stale")
        raise ScheduleRefused("0 machine(s) serve it")

    with pytest.raises(ScheduleRefused, match=f"cannot add {refused}: 0 machine"):
        check_request(two_adds(contract, current), contract=contract, current=current, catalogue_check=check,
                      read_attestation=reader)
    assert sorted(seen) == sorted([ADDED, SECOND])


def test_with_several_adds_all_waiting_the_request_waits_with_every_reason():
    contract = episode_contract()
    current = initial_schedule(contract)

    def check(name, added, attestation):
        raise ScheduleNotYet(f"cannot add {name} yet: stale")

    with pytest.raises(ScheduleNotYet) as caught:
        check_request(two_adds(contract, current), contract=contract, current=current, catalogue_check=check,
                      read_attestation=reader)
    assert ADDED in str(caught.value) and SECOND in str(caught.value)


def safe_folder(tmp_path):
    folder = tmp_path / "requests"
    folder.mkdir(mode=0o700)
    return folder


def test_the_attestation_is_read_safely_a_symlink(tmp_path):
    folder = safe_folder(tmp_path)
    target = tmp_path / "elsewhere.json"
    target.write_bytes(ATTESTATION)
    os.chmod(target, 0o600)
    (folder / "conformance.json").symlink_to(target)
    with pytest.raises(ValueError, match="cannot be opened safely"):
        ScheduleRequestStore(folder).read_attestation("conformance.json")


def test_the_attestation_is_read_safely_not_a_regular_file(tmp_path):
    import threading

    folder = safe_folder(tmp_path)
    fifo = folder / "conformance.json"
    os.mkfifo(fifo, 0o600)
    outcome: dict = {}

    def read():
        try:
            outcome["value"] = ScheduleRequestStore(folder).read_attestation("conformance.json")
        except BaseException as exc:   # noqa: BLE001 - handed to the test's thread
            outcome["error"] = exc

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    reader.join(5)
    if reader.is_alive():
        # The read blocked on the FIFO (a blocking open): a writer that closes at once releases it, then fail fast.
        os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
        reader.join(5)
        pytest.fail("reading the attestation blocked on a FIFO instead of refusing it")
    assert isinstance(outcome.get("error"), ValueError), outcome
    assert "not a regular file" in str(outcome["error"])


def test_the_attestation_is_read_safely_writable_by_others(tmp_path):
    folder = safe_folder(tmp_path)
    (folder / "conformance.json").write_bytes(ATTESTATION)
    os.chmod(folder / "conformance.json", 0o602)
    with pytest.raises(ValueError, match="writable by group or others"):
        ScheduleRequestStore(folder).read_attestation("conformance.json")


def test_the_attestation_is_read_safely_bounded(tmp_path, monkeypatch):
    from reliquary.services import schedule

    folder = safe_folder(tmp_path)
    (folder / "conformance.json").write_bytes(ATTESTATION)
    os.chmod(folder / "conformance.json", 0o600)
    assert ScheduleRequestStore(folder).read_attestation("conformance.json")[0] == ATTESTATION
    monkeypatch.setattr(schedule, "ATTESTATION_MAX_BYTES", len(ATTESTATION) - 1)
    with pytest.raises(ValueError, match="larger than"):
        ScheduleRequestStore(folder).read_attestation("conformance.json")


@pytest.mark.parametrize("code", ["EIO", "EMFILE", "EAGAIN"])
def test_a_transient_attestation_read_error_waits(code):
    import errno

    contract = episode_contract()
    current = initial_schedule(contract)

    def failing(name):
        raise OSError(getattr(errno, code), os.strerror(getattr(errno, code)))

    with pytest.raises(ScheduleNotYet, match="cannot be read now"):
        check_request(request(contract, current), contract=contract, current=current, catalogue_check=accept,
                      read_attestation=failing)


def test_a_lasting_attestation_read_error_is_final():
    import errno

    contract = episode_contract()
    current = initial_schedule(contract)

    def failing(name):
        raise OSError(errno.EACCES, "Permission denied")

    with pytest.raises(ScheduleRefused, match="cannot be read"):
        check_request(request(contract, current), contract=contract, current=current, catalogue_check=accept,
                      read_attestation=failing)


def test_the_store_lets_a_transient_open_error_through(tmp_path, monkeypatch):
    import errno

    from reliquary.services import schedule

    folder = safe_folder(tmp_path)
    (folder / "conformance.json").write_bytes(ATTESTATION)
    os.chmod(folder / "conformance.json", 0o600)
    real_open = os.open

    def fake_open(path, flags, *args, **kwargs):
        if path == "conformance.json":
            raise OSError(errno.EMFILE, "Too many open files")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(schedule.os, "open", fake_open)
    with pytest.raises(OSError) as caught:
        ScheduleRequestStore(folder).read_attestation("conformance.json")
    assert not isinstance(caught.value, ValueError) and caught.value.errno == errno.EMFILE
    # The request file keeps its old behaviour: any open error is a ValueError.
    with pytest.raises(ValueError, match="cannot be opened safely"):
        schedule._read_json_file(folder, "conformance.json")


def test_a_clock_that_goes_back_never_lengthens_the_wait(tmp_path):
    rt, store, _ = pending_runtime(tmp_path)
    rt.apply_pending_schedule_request(store, window=1, now=10_000.0, catalogue_check=not_yet, add_wait_max_s=600)
    rt.apply_pending_schedule_request(store, window=2, now=5_000.0, catalogue_check=not_yet, add_wait_max_s=600)
    status = store.status()
    assert status["status"] == "waiting" and "unsettled in 600 s" in status["detail"]
    rt.apply_pending_schedule_request(store, window=3, now=10_601.0, catalogue_check=not_yet, add_wait_max_s=600)
    assert store.status()["status"] == "refused"


def test_a_request_removed_while_waiting_is_withdrawn(tmp_path):
    rt, store, req = pending_runtime(tmp_path)
    rt.apply_pending_schedule_request(store, window=1, now=1_000.0, catalogue_check=not_yet)
    store.request_path.unlink()
    rt.apply_pending_schedule_request(store, window=2, now=1_010.0, catalogue_check=not_yet)
    status = store.status()
    assert status["request_id"] == req["request_id"] and status["status"] == "withdrawn"
    assert rt.db.execute("SELECT COUNT(*) FROM service_schedule_waits").fetchone() == (0,)
    assert recorded(rt, req["request_id"]) is None


def test_no_request_and_no_wait_writes_nothing(tmp_path):
    rt = episode_runtime(tmp_path)
    store = ScheduleRequestStore(tmp_path)
    rt.apply_pending_schedule_request(store, window=1)
    assert store.status() is None


class TrackedLock:
    """An RLock that remembers whether it was ever fully released since ``mark``."""

    def __init__(self):
        import threading

        self._lock, self.depth, self.released = threading.RLock(), 0, False

    def acquire(self, *args, **kwargs):
        got = self._lock.acquire(*args, **kwargs)
        if got:
            self.depth += 1
        return got

    def release(self):
        self.depth -= 1
        if self.depth == 0:
            self.released = True
        self._lock.release()

    __enter__ = acquire

    def __exit__(self, *exc):
        self.release()


def watch_view_refresh(rt):
    """Each view rebuild records whether the lock was released between the last commit and the rebuild."""
    import contextlib

    lock, gaps, txn, refresh = TrackedLock(), [], rt._txn, rt._refresh_contract_view
    rt.lock = lock

    @contextlib.contextmanager
    def tracked_txn():
        with txn():
            yield
        lock.released = lock.depth == 0                      # the commit has just released it, or not

    def tracked_refresh():
        gaps.append(lock.released)
        refresh()

    rt._txn, rt._refresh_contract_view = tracked_txn, tracked_refresh
    return gaps


def test_the_view_is_rebuilt_before_the_lock_is_released(tmp_path):
    rt = episode_runtime(tmp_path)
    folder_with_attestation(tmp_path)
    gaps = watch_view_refresh(rt)
    store = ScheduleRequestStore(tmp_path)
    store.write(request(rt.contract, rt.schedule))
    rt.apply_pending_schedule_request(store, window=1, catalogue_check=accept)
    changed = next_schedule(rt.order_contract, rt.schedule, cooldowns={MATH: 12})
    rt.apply_schedule(changed, request_id="direct", window=1)
    assert gaps == [False, False] and ADDED in rt.contract.environments


def test_a_malformed_request_without_add_keeps_its_refusal_text():
    contract = episode_contract()
    current = initial_schedule(contract)
    bad = build_request(order_sha256=contract.sha256, revision=1, cooldowns={MATH: 10})
    bad["extra"] = 1
    expected = ("a schedule request has exactly the fields ['active', 'cooldowns', 'order_sha256', 'request_id', "
                "'revision', 'schema', 'shares']")
    for value in (bad, {k: v for k, v in bad.items() if k not in ("extra", "shares")}, ["not", "a", "dict"]):
        with pytest.raises(ScheduleRefused) as caught:
            check_request(value, contract=contract, current=current)
        assert str(caught.value) == expected
