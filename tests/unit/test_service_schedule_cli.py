"""Live schedule operator command (decisions E, G; R6, R7): CLI, request store, runtime application."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from reliquary.protocol.release_contract import canonical_json_bytes
from reliquary.services.schedule import REQUEST_SCHEMA, ScheduleRequestStore, build_request
from tests.unit.service_v2_fixtures import CODE, MATH, contract_v2
from tests.unit.test_service_runtime_v2 import (
    POOL, archive, audited, build, explore, open_window, runtime,
)

ROOT = Path(__file__).resolve().parents[2]


def math_only_runtime(tmp_path):
    """An order that declares CODE with share 0 (inactive at launch, R7) and runs MATH alone."""
    contract = contract_v2(shares={MATH: 10000, CODE: 0})
    rt = build(tmp_path / "runtime.sqlite3", contract)
    rt.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision="d" * 40)
    open_window(rt, 1, pools={MATH: POOL})
    return rt


def pinned(contract):
    return lambda name: contract.environments[name]["version"]


def submit(rt, store, **kw):
    """A request as the CLI would write it for the current schedule."""
    return store.submit(order_sha256=rt.contract.sha256, revision=rt.schedule.revision + 1, **kw)


def raw_request(rt, **override):
    request = build_request(order_sha256=rt.contract.sha256, revision=rt.schedule.revision + 1,
                            cooldowns={MATH: 12})
    request.update(override)
    return request


def cli(folder, *args, check=True):
    return subprocess.run([sys.executable, "-m", "reliquary.services.schedule", "--folder", str(folder), *args],
                          capture_output=True, text=True, check=check, cwd=ROOT)


# ---------------------------------------------------------------- runtime application

def test_request_applies_once_at_the_next_window(tmp_path):
    rt = runtime(tmp_path)
    store = ScheduleRequestStore(tmp_path)
    request = submit(rt, store, cooldowns={MATH: 1741})
    schedule = rt.apply_pending_schedule_request(store, window=2)
    assert schedule.cooldown_windows(MATH) == 1741 and schedule.revision == 1
    assert store.status()["status"] == "applied" and store.status()["request_id"] == request["request_id"]
    assert rt.apply_pending_schedule_request(store, window=3).revision == 1   # the file is still there
    assert rt.schedule.revision == 1


def test_refused_request_is_reported_and_keeps_the_schedule(tmp_path):
    rt = runtime(tmp_path)
    store = ScheduleRequestStore(tmp_path)
    submit(rt, store, active=[MATH], shares={MATH: 9000})
    schedule = rt.apply_pending_schedule_request(store, window=2)
    assert schedule.revision == 0 and rt.schedule.revision == 0
    assert store.status()["status"] == "refused" and "10000" in store.status()["detail"]


def test_deactivation_applies_next_window_and_current_envelope_still_settles(tmp_path):
    rt = runtime(tmp_path)
    paid = explore(rt, env=CODE, hotkey="x")
    audited(rt, paid)
    store = ScheduleRequestStore(tmp_path)
    submit(rt, store, active=[MATH], shares={MATH: 10000})
    # Applied during window 1: window 1 keeps its frozen envelope, window 2 gets the new schedule.
    assert rt.apply_pending_schedule_request(store, window=1).active_environments() == (CODE, MATH)
    assert rt.schedule.active_environments() == (MATH,)
    assert rt.envelope(1)["schedule"]["revision"] == 0 and set(rt.envelope(1)["pools"]) == {MATH, CODE}
    settled = rt.reconcile_archive(archive(1))
    assert settled["rewards_by_hotkey"]["x"] > 0
    assert CODE in settled["service_pools_by_environment"]
    assert settled["service_schedule"]["revision"] == 0
    with pytest.raises(ValueError):
        open_window(rt, 2, pools={MATH: POOL, CODE: POOL})
    assert open_window(rt, 2, pools={MATH: POOL})["schedule"]["revision"] == 1
    assert rt.apply_pending_schedule_request(store, window=2).active_environments() == (MATH,)


def test_open_window_keeps_its_schedule_whatever_a_late_request_does(tmp_path):
    rt = runtime(tmp_path)
    store = ScheduleRequestStore(tmp_path)
    submit(rt, store, cooldowns={MATH: 99})
    assert rt.apply_pending_schedule_request(store, window=1).cooldown_windows(MATH) == 50   # R6: frozen
    assert rt.window_schedule(1).revision == 0
    assert rt.window_schedule(2).cooldown_windows(MATH) == 99


@pytest.mark.parametrize("case", [
    "other_order", "replay", "gap", "unknown_env", "cooldown_low", "cooldown_high", "bad_schema", "extra_field",
    "bool_revision", "share_for_inactive", "string_cooldown", "nothing", "active_not_list", "shares_not_sum",
])
def test_runtime_validates_a_hand_written_request_and_refuses_without_applying(tmp_path, case):
    from reliquary.validator.control import write_json
    rt = runtime(tmp_path)
    store = ScheduleRequestStore(tmp_path)
    request = raw_request(rt)
    cooldown_bounds = rt.contract.advice_policy
    request.update({
        "other_order": {"order_sha256": "0" * 64},
        "replay": {"revision": 0},
        "gap": {"revision": 2},
        "unknown_env": {"cooldowns": {"nope": 5}},
        "cooldown_low": {"cooldowns": {MATH: cooldown_bounds["min_windows"] - 1}},
        "cooldown_high": {"cooldowns": {MATH: cooldown_bounds["max_windows"] + 1}},
        "bad_schema": {"schema": "service-schedule-request/v0"},
        "extra_field": {"surprise": 1},
        "bool_revision": {"revision": True},
        "share_for_inactive": {"active": [MATH], "shares": {MATH: 10000, CODE: 100}},
        "string_cooldown": {"cooldowns": {MATH: "12"}},
        "nothing": {"cooldowns": None},
        "active_not_list": {"active": "reliquary_dapo_math_v1"},
        "shares_not_sum": {"shares": {MATH: 5000, CODE: 4000}},
    }[case])
    write_json(store.request_path, request)
    assert rt.apply_pending_schedule_request(store, window=2).revision == 0
    assert rt.schedule.revision == 0
    status = store.status()
    assert status["status"] == "refused" and status["detail"]
    word = {"replay": "replayed", "gap": "gap", "other_order": "another order", "unknown_env": "unknown",
            "cooldown_low": "bounds", "cooldown_high": "bounds", "shares_not_sum": "10000"}.get(case)
    assert word is None or word in status["detail"], status["detail"]
    # the operator can recover with a fresh, valid request
    submit(rt, store, cooldowns={MATH: 12})
    assert rt.apply_pending_schedule_request(store, window=2).revision == 1


def test_refused_request_is_not_reevaluated_and_does_not_block_the_next(tmp_path):
    rt = runtime(tmp_path)
    store = ScheduleRequestStore(tmp_path)
    first = submit(rt, store, cooldowns={MATH: 10**9})
    rt.apply_pending_schedule_request(store, window=2)
    stamp = store.status()["at"]
    rt.apply_pending_schedule_request(store, window=3)
    assert store.status()["at"] == stamp and store.status()["request_id"] == first["request_id"]
    second = submit(rt, store, cooldowns={MATH: 20})
    assert rt.apply_pending_schedule_request(store, window=4).cooldown_windows(MATH) == 20
    assert store.status()["request_id"] == second["request_id"]


def test_activation_needs_the_env_installed_at_the_pinned_version(tmp_path):
    rt = math_only_runtime(tmp_path)
    contract = rt.contract
    assert rt.schedule.active_environments() == (MATH,)
    store = ScheduleRequestStore(tmp_path)
    activate = dict(active=[MATH, CODE], shares={MATH: 6000, CODE: 4000})

    submit(rt, store, **activate)
    assert rt.apply_pending_schedule_request(store, window=2, installed_version=lambda n: None).revision == 0
    assert "not installed" in store.status()["detail"]

    submit(rt, store, **activate)
    assert rt.apply_pending_schedule_request(store, window=2, installed_version=lambda n: "e" * 64).revision == 0
    assert "version" in store.status()["detail"] and store.status()["status"] == "refused"

    submit(rt, store, **activate)
    schedule = rt.apply_pending_schedule_request(store, window=2, installed_version=pinned(contract))
    assert schedule.active_environments() == (CODE, MATH) and store.status()["status"] == "applied"


def test_default_install_check_is_the_registry(tmp_path):
    # The unit-test order pins fake versions, so the real registry cannot match: activation is refused.
    rt = math_only_runtime(tmp_path)
    store = ScheduleRequestStore(tmp_path)
    submit(rt, store, active=[MATH, CODE], shares={MATH: 6000, CODE: 4000})
    assert rt.apply_pending_schedule_request(store, window=2).revision == 0
    assert store.status()["status"] == "refused"


def test_applied_revision_survives_a_restart_and_is_not_applied_twice(tmp_path):
    rt = runtime(tmp_path)
    store = ScheduleRequestStore(tmp_path)
    request = submit(rt, store, active=[MATH], shares={MATH: 10000})
    rt.apply_pending_schedule_request(store, window=1)
    contract = rt.contract
    rt.close()

    rt = build(tmp_path / "runtime.sqlite3", contract)
    rt.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision="d" * 40)
    assert rt.schedule.revision == 1
    assert rt.apply_pending_schedule_request(store, window=2).revision == 1   # same file, same request id
    assert rt.schedule.revision == 1
    assert store.status()["request_id"] == request["request_id"]
    assert open_window(rt, 2, pools={MATH: POOL})["schedule"]["revision"] == 1


def test_status_file_is_restored_after_a_crash_between_commit_and_report(tmp_path):
    rt = runtime(tmp_path)
    store = ScheduleRequestStore(tmp_path)
    request = submit(rt, store, cooldowns={MATH: 12})
    rt.apply_pending_schedule_request(store, window=2)
    store.status_path.unlink()
    assert rt.apply_pending_schedule_request(store, window=3).revision == 1
    assert store.status()["status"] == "applied" and store.status()["request_id"] == request["request_id"]


def test_unsafe_request_files_are_refused_without_being_followed(tmp_path):
    rt = runtime(tmp_path)
    folder = tmp_path / "ops"
    store = ScheduleRequestStore(folder)
    folder.mkdir(mode=0o700)
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(raw_request(rt)))
    outside.chmod(0o600)
    os.symlink(outside, store.request_path)
    assert rt.apply_pending_schedule_request(store, window=2).revision == 0
    assert store.status()["status"] == "refused" and "safely" in store.status()["detail"]
    store.request_path.unlink()

    for body, word in ((b"x" * 70_000, "larger"), (b'{"a":1,"a":2}', "strict JSON"), (b"[1]", "object"),
                       (b'{"revision": NaN}', "strict JSON"), (b"\xff\xfe", "strict JSON")):
        store.request_path.write_bytes(body)
        store.request_path.chmod(0o600)
        assert rt.apply_pending_schedule_request(store, window=2).revision == 0
        assert word in store.status()["detail"], body[:10]
    store.request_path.chmod(0o666)
    assert rt.apply_pending_schedule_request(store, window=2).revision == 0
    assert "writable" in store.status()["detail"]
    assert rt.schedule.revision == 0


def test_store_writes_atomically_and_leaves_no_temporary_file(tmp_path):
    store = ScheduleRequestStore(tmp_path / "ops")
    request = store.submit(order_sha256="a" * 64, revision=1, cooldowns={MATH: 3})
    assert sorted(p.name for p in (tmp_path / "ops").iterdir()) == ["schedule-request.json"]
    assert (tmp_path / "ops" / "schedule-request.json").stat().st_mode & 0o777 == 0o600
    assert store.take() == request and request["schema"] == REQUEST_SCHEMA and store.take()["revision"] == 1


def test_a_store_failure_never_raises_into_the_validator_loop(tmp_path):
    rt = runtime(tmp_path)

    class Broken:
        def take(self):
            raise RuntimeError("disk on fire")

    assert rt.apply_pending_schedule_request(Broken(), window=2).revision == 0


# ---------------------------------------------------------------- CLI

def test_cli_set_writes_a_request_file(tmp_path):
    rt = runtime(tmp_path)      # stays open: the CLI reads the database read-only next to the live writer
    out = cli(tmp_path, "set", "--cooldown", f"{MATH}=12")
    request = json.loads((tmp_path / "schedule-request.json").read_text())
    assert request["cooldowns"] == {MATH: 12} and request["active"] is None and request["shares"] is None
    assert request["order_sha256"] == rt.contract.sha256 and request["revision"] == 1
    assert json.loads(out.stdout)["written"] == request
    assert rt.apply_pending_schedule_request(ScheduleRequestStore(tmp_path), window=2).cooldown_windows(MATH) == 12


@pytest.mark.parametrize("args,word", [
    (["--cooldown", f"{MATH}=0"], "bounds"),
    (["--active", MATH, "--share", f"{MATH}=9000"], "10000"),
    (["--cooldown", "nope=5"], "unknown"),
    (["--active", f"{MATH},{CODE}", "--share", f"{MATH}=6000", "--share", f"{CODE}=3000"], "10000"),
    ([], "changes nothing"),
    (["--cooldown", f"{MATH}=abc"], "env=integer"),
])
def test_cli_set_refuses_an_invalid_request_before_writing_it(tmp_path, args, word):
    runtime(tmp_path)
    result = cli(tmp_path, "set", *args, check=False)
    assert result.returncode != 0 and word in result.stderr
    assert not (tmp_path / "schedule-request.json").exists()


def test_cli_refuses_to_activate_an_env_that_is_not_installed(tmp_path):
    math_only_runtime(tmp_path)
    result = cli(tmp_path, "set", "--active", f"{MATH},{CODE}", "--share", f"{MATH}=6000", "--share", f"{CODE}=4000",
                 check=False)
    assert result.returncode != 0 and "cannot activate" in result.stderr
    assert not (tmp_path / "schedule-request.json").exists()


def test_cli_needs_a_folder_and_a_runtime(tmp_path):
    assert cli(tmp_path / "missing", "show", check=False).returncode != 0
    env = {k: v for k, v in os.environ.items() if k != "RELIQUARY_SERVICE_SCHEDULE_FOLDER"}
    result = subprocess.run([sys.executable, "-m", "reliquary.services.schedule", "show"], capture_output=True,
                            text=True, env=env, cwd=ROOT)
    assert result.returncode != 0 and "--folder" in result.stderr


def test_cli_show_reads_the_runtime_read_only(tmp_path):
    rt = runtime(tmp_path)
    rt.close()
    before = (tmp_path / "runtime.sqlite3").stat().st_mtime_ns
    shown = json.loads(cli(tmp_path, "show").stdout)
    assert shown["schedule"]["revision"] == 0 and set(shown["schedule"]["environments"]) == {MATH, CODE}
    assert shown["pending_request"] is None and shown["last_request"] is None
    assert (tmp_path / "runtime.sqlite3").stat().st_mtime_ns == before


def test_cli_show_prints_pending_request_status_and_advice(tmp_path):
    rt = runtime(tmp_path)
    rt.refresh_cooldown_advice(window=1, populations={MATH: 1000, CODE: 1000})
    cli(tmp_path, "set", "--cooldown", f"{MATH}=12")
    shown = json.loads(cli(tmp_path, "show").stdout)
    assert shown["pending_request"]["cooldowns"] == {MATH: 12} and shown["last_request"] is None
    assert set(shown["cooldown_advice"]) == {MATH, CODE}
    assert shown["cooldown_advice"][MATH]["status"] == "insufficient_data"
    assert "recommendation only" in shown["cooldown_advice_note"]
    rt.apply_pending_schedule_request(ScheduleRequestStore(tmp_path), window=2)
    shown = json.loads(cli(tmp_path, "show").stdout)
    assert shown["pending_request"] is None and shown["last_request"]["status"] == "applied"
    assert shown["schedule"]["revision"] == 1 and shown["schedule"]["environments"][MATH]["cooldown_windows"] == 12


# ---------------------------------------------------------------- cooldown advice hook (Task 8's runtime half)

def advisor(monkeypatch, fn):
    from reliquary.services import cooldown_advisor
    monkeypatch.setattr(cooldown_advisor, "recommend_cooldown", fn)


def test_runtime_stores_advice_and_never_changes_the_schedule(tmp_path):
    from tests.unit.test_service_runtime_v2 import train
    rt = runtime(tmp_path)
    for prompt in range(3):
        train(rt, prompt=prompt)
    before = rt.schedule
    advice = rt.refresh_cooldown_advice(window=2, populations={MATH: 1000})
    assert advice[MATH]["status"] == "insufficient_data" and advice[CODE]["status"] == "insufficient_data"
    assert rt.schedule == before and rt.schedule.sha256 == before.sha256
    assert rt.cooldown_advice()[MATH]["first_scans"] == 3
    assert advice[MATH]["current_windows"] == 50


def test_advice_previous_comes_from_the_table_and_survives_a_restart(tmp_path, monkeypatch):
    from reliquary.services import cooldown_advisor
    real, seen = cooldown_advisor.recommend_cooldown, []

    def spy(**kw):
        seen.append(kw["previous"])
        return {**real(**kw), "ema_windows": 7.5, "recommended_windows": 8}

    advisor(monkeypatch, spy)
    rt = runtime(tmp_path)
    rt.refresh_cooldown_advice(window=2, populations={MATH: 10, CODE: 10})
    assert seen[:2] == [None, None]
    rt.refresh_cooldown_advice(window=3, populations={MATH: 10, CODE: 10})
    assert seen[2]["ema_windows"] == 7.5 and seen[2]["recommended_windows"] == 8
    contract = rt.contract
    rt.close()
    rt = build(tmp_path / "runtime.sqlite3", contract)
    rt.refresh_cooldown_advice(window=4, populations={MATH: 10, CODE: 10})
    assert seen[4]["recommended_windows"] == 8


def test_an_advisor_exception_is_stored_as_an_error_with_the_note_and_never_raised(tmp_path, monkeypatch):
    from reliquary.services import cooldown_advisor
    real, calls = cooldown_advisor.recommend_cooldown, {"n": 0}

    def flaky(**kw):
        calls["n"] += 1
        if calls["n"] > 2:
            raise OverflowError("int too large to convert to float")
        return {**real(**kw), "ema_windows": 7.5, "recommended_windows": 8}

    advisor(monkeypatch, flaky)
    rt = runtime(tmp_path)
    rt.refresh_cooldown_advice(window=2, populations={MATH: 10, CODE: 10})
    advice = rt.refresh_cooldown_advice(window=3, populations={MATH: 10**500, CODE: 10})   # raises OverflowError
    for env in (MATH, CODE):
        assert advice[env]["status"] == "error" and advice[env]["reasons"] == ["OverflowError"]
        assert "Recommendation only" in advice[env]["note"]
        assert advice[env]["recommended_windows"] == 8 and advice[env]["ema_windows"] == 7.5   # state kept
    assert rt.cooldown_advice()[MATH]["status"] == "error" and rt.schedule.revision == 0


def test_advice_is_informational_in_the_archive_and_never_enters_settlement(tmp_path, monkeypatch):
    advisor(monkeypatch, lambda **kw: {"status": "ok", "reasons": [], "recommended_windows": kw["population"]})
    from reliquary.services.runtime import FROZEN_ARCHIVE_FIELDS, protocol_slot_geometry
    from reliquary.services.settlement import validate_service_archive_v2
    from tests.unit.test_service_runtime_v2 import frozen
    assert "service_cooldown_advice" not in FROZEN_ARCHIVE_FIELDS
    rt = runtime(tmp_path)
    rt.refresh_cooldown_advice(window=1, populations={MATH: 5, CODE: 5})
    first = rt.reconcile_archive(archive(1))
    assert set(first["service_cooldown_advice"]) == {MATH, CODE}
    rt.refresh_cooldown_advice(window=2, populations={MATH: 6000, CODE: 5})
    assert rt.cooldown_advice() != first["service_cooldown_advice"]       # the live advice moved on...
    second = rt.reconcile_archive(archive(1))
    assert canonical_json_bytes(second) == canonical_json_bytes(first)    # ...the window's snapshot did not (I2)
    assert frozen(first) == frozen(second)                                  # money does not follow the advice
    picks, slots = protocol_slot_geometry()
    second["service_cooldown_advice"] = {"garbage": float("nan")}
    validate_service_archive_v2(second, rt.contract, cap=1.0, picks_target=picks, batch_slots=slots)


def test_the_advice_snapshot_of_a_window_survives_a_restart_and_a_refresh(tmp_path, monkeypatch):
    advisor(monkeypatch, lambda **kw: {"status": "ok", "reasons": [], "recommended_windows": kw["population"]})
    rt = runtime(tmp_path)
    rt.refresh_cooldown_advice(window=1, populations={MATH: 5, CODE: 5})
    first = rt.reconcile_archive(archive(1))
    contract = rt.contract
    rt.close()
    rt = build(tmp_path / "runtime.sqlite3", contract)
    rt.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision="d" * 40)
    rt.refresh_cooldown_advice(window=2, populations={MATH: 777, CODE: 888})
    again = rt.reconcile_archive(archive(1))
    assert canonical_json_bytes(again) == canonical_json_bytes(first)
    assert again["service_cooldown_advice"][MATH]["recommended_windows"] == 5
    assert rt.cooldown_advice()[MATH]["recommended_windows"] == 777


def test_an_advice_refresh_never_raises_even_if_consumption_or_the_table_fails(tmp_path):
    rt = runtime(tmp_path)

    def boom():
        raise RuntimeError("consumption unreadable")

    rt.measured_consumption = boom
    advice = rt.refresh_cooldown_advice(window=2, populations={MATH: 5, CODE: 5})
    assert {a["status"] for a in advice.values()} == {"error"}
    del rt.measured_consumption
    rt.db.execute("DROP TABLE service_cooldown_advice")
    advice = rt.refresh_cooldown_advice(window=3, populations={MATH: 5, CODE: 5})     # the upsert fails: logged
    assert set(advice) == {MATH, CODE}


# ---------------------------------------------------------------- review fixes (I1, M1, M3, M4)

def test_cli_set_refuses_to_destroy_an_unhandled_pending_request(tmp_path):
    rt = runtime(tmp_path)
    first = json.loads(cli(tmp_path, "set", "--cooldown", f"{MATH}=12").stdout)["written"]
    result = cli(tmp_path, "set", "--cooldown", f"{MATH}=13", check=False)
    assert result.returncode != 0 and first["request_id"] in result.stderr and "--replace" in result.stderr
    assert json.loads((tmp_path / "schedule-request.json").read_text()) == first        # untouched
    replaced = cli(tmp_path, "set", "--cooldown", f"{MATH}=13", "--replace")
    assert first["request_id"] in replaced.stderr and '"cooldowns"' in replaced.stderr
    new = json.loads((tmp_path / "schedule-request.json").read_text())
    assert new["request_id"] != first["request_id"] and new["cooldowns"] == {MATH: 13}
    assert json.loads(replaced.stdout)["replaced"] == first
    # Once the validator has handled the request, a new one needs no --replace.
    rt.apply_pending_schedule_request(ScheduleRequestStore(tmp_path), window=2)
    assert cli(tmp_path, "set", "--cooldown", f"{MATH}=14", check=False).returncode == 0


@pytest.mark.parametrize("target", ["folder", "file"])
@pytest.mark.parametrize("fault", ["group_writable", "foreign_owner"])
def test_request_folder_and_file_must_be_owned_and_not_writable_by_others(tmp_path, monkeypatch, target, fault):
    rt = runtime(tmp_path)
    folder = tmp_path / "ops"
    store = ScheduleRequestStore(folder)
    folder.mkdir()
    folder.chmod(0o700)
    request = submit(rt, store, cooldowns={MATH: 12})
    victim = folder if target == "folder" else store.request_path
    real_geteuid = os.geteuid
    if fault == "group_writable":
        victim.chmod(0o770 if target == "folder" else 0o660)
    else:
        monkeypatch.setattr(os, "geteuid", lambda: real_geteuid() + 1)
    # runtime: a recorded refusal, no crash, nothing applied
    assert rt.apply_pending_schedule_request(store, window=2).revision == 0
    assert rt.schedule.revision == 0
    monkeypatch.undo()
    status = json.loads(store.status_path.read_text())
    detail = status["detail"]
    assert status["status"] == "refused" and ("writable" in detail or "owned" in detail)
    # CLI: a clear refusal, exit non-zero, request file untouched
    before = store.request_path.read_bytes()
    if fault == "foreign_owner":
        result = _cli_as_foreign_user(folder, ["set", "--cooldown", f"{MATH}=13", "--replace"])
    else:
        result = cli(folder, "--db", str(tmp_path / "runtime.sqlite3"), "set", "--cooldown", f"{MATH}=13",
                     "--replace", check=False)
    assert result.returncode != 0 and ("writable" in result.stderr or "owned" in result.stderr), result.stderr
    assert store.request_path.read_bytes() == before


def _cli_as_foreign_user(folder, command):
    code = ("import os,sys; os.geteuid = lambda: os.getuid() + 1; "
            "from reliquary.services.schedule import main; main(sys.argv[1:])")
    return subprocess.run([sys.executable, "-c", code, "--folder", str(folder), "--db",
                           str(folder.parent / "runtime.sqlite3"), *command], capture_output=True, text=True,
                          cwd=ROOT)


def test_foreign_owner_is_named_in_the_refusal(tmp_path, monkeypatch):
    rt = runtime(tmp_path)
    store = ScheduleRequestStore(tmp_path)
    submit(rt, store, cooldowns={MATH: 12})
    real_geteuid = os.geteuid
    monkeypatch.setattr(os, "geteuid", lambda: real_geteuid() + 1)
    with pytest.raises(ValueError, match="owned"):
        store.take()


def test_pairs_reject_non_ascii_digits():
    from reliquary.services.schedule import _pairs
    assert _pairs(["a=12"], "--share") == {"a": 12}
    with pytest.raises(SystemExit):
        _pairs(["a=\u0663"], "--share")          # ARABIC-INDIC DIGIT THREE passes str.isdigit()
    with pytest.raises(SystemExit):
        _pairs(["a=\u00b2"], "--share")          # superscript two: isdigit() but int() fails


def test_cli_without_a_schedule_row_gives_an_operator_message_not_a_traceback(tmp_path):
    import sqlite3
    runtime(tmp_path).close()
    db = sqlite3.connect(tmp_path / "runtime.sqlite3")
    db.execute("DELETE FROM service_schedules")
    db.commit()
    db.close()
    result = cli(tmp_path, "show", check=False)
    assert result.returncode != 0 and "no schedule" in result.stderr and "Traceback" not in result.stderr
