"""The controller supervisor: one RL controller per training task, and the map file the front reads."""
from __future__ import annotations

import json
from types import SimpleNamespace

from reliquary.validator.rl_supervisor import Supervisor, subprocess_spawn  # noqa: F401


class Proc:
    def __init__(self) -> None:
        self.code: int | None = None
        self.signals: list[str] = []

    def poll(self):
        return self.code

    def terminate(self) -> None:
        self.signals.append("term")

    def kill(self) -> None:
        self.signals.append("kill")
        self.code = -9


class Rig:
    def __init__(self, tmp_path, **kwargs) -> None:
        self.now = 1000.0
        self.entries: dict = {}
        self.final: set[str] = set()
        self.spawned: list[tuple[str, int]] = []
        self.procs: dict[str, Proc] = {}
        self.map_path = tmp_path / "map.json"
        self.sup = Supervisor(lambda: self.entries, self.spawn, ports=range(9100, 9103), map_path=self.map_path,
                              clock=lambda: self.now, final_of=lambda entry: entry.task_id in self.final, **kwargs)

    def spawn(self, task_id: str, port: int, entry=None) -> Proc:
        self.spawned.append((task_id, port))
        self.procs[task_id] = Proc()
        return self.procs[task_id]

    def add(self, task_id: str, *, status: str = "active", training: bool = True) -> None:
        self.entries[task_id] = SimpleNamespace(task_id=task_id, status=status,
                                                params={"training_order": {}} if training else {})

    def map(self) -> dict:
        return json.loads(self.map_path.read_text())


def test_an_active_training_task_gets_a_controller_and_a_port(tmp_path):
    rig = Rig(tmp_path)
    rig.add("t-a")
    rig.add("plain", training=False)
    rig.sup.tick()
    rig.sup.tick()
    assert rig.spawned == [("t-a", 9100)]
    assert rig.map()["tasks"] == {"t-a": {"port": 9100, "run": "t-a"}}


def test_a_retired_task_stops_after_its_last_settle(tmp_path):
    rig = Rig(tmp_path)
    rig.add("t-a")
    rig.sup.tick()
    rig.entries["t-a"].status = "retired"
    rig.now += 15
    rig.sup.tick()
    assert rig.procs["t-a"].signals == []              # not final: it keeps settling
    assert "t-a" in rig.map()["tasks"]
    rig.final.add("t-a")
    rig.now += 15
    rig.sup.tick()
    assert rig.procs["t-a"].signals == ["term"]
    assert rig.map()["tasks"] == {}
    rig.now += 61
    rig.sup.tick()
    assert rig.procs["t-a"].signals == ["term", "kill"]
    rig.now += 15
    rig.sup.tick()
    assert len(rig.spawned) == 1                       # retired: never restarted
    rig.add("t-b")
    rig.sup.tick()
    assert rig.spawned[-1] == ("t-b", 9100)            # the port was freed


def test_a_dead_controller_restarts_with_backoff(tmp_path):
    rig = Rig(tmp_path)
    rig.add("t-a")
    rig.sup.tick()
    waits = []
    for _ in range(5):
        rig.procs["t-a"].code = 1
        died = rig.now
        count = len(rig.spawned)
        rig.sup.tick()
        assert "t-a" not in rig.map()["tasks"]
        while len(rig.spawned) == count:
            rig.now += 5
            rig.sup.tick()
        waits.append(rig.now - died)
    assert [round(w) for w in waits] == [30, 60, 120, 240, 480]
    assert rig.spawned[-1] == ("t-a", 9100)
    rig.procs["t-a"].code = 1
    for _ in range(2):
        count = len(rig.spawned)
        while len(rig.spawned) == count:
            rig.now += 5
            rig.sup.tick()
        rig.procs["t-a"].code = 1
    assert rig.sup.backoff_of("t-a") == 600            # capped


def test_the_map_names_a_default_task(tmp_path):
    rig = Rig(tmp_path)
    rig.sup.tick()
    assert rig.map() == {"default": None, "tasks": {}}
    rig.add("t-a")
    rig.sup.tick()
    rig.now += 15
    rig.add("t-b")
    rig.sup.tick()
    assert rig.map()["default"] == "t-a"               # the oldest running
    (tmp_path / "x").mkdir()
    pinned = Rig(tmp_path / "x", default_task="t-b")
    pinned.add("t-a")
    pinned.add("t-b")
    pinned.sup.tick()
    assert pinned.map()["default"] == "t-b"


def test_an_unreadable_registry_changes_nothing(tmp_path):
    rig = Rig(tmp_path)
    rig.add("t-a")
    rig.sup.tick()

    def broken():
        raise RuntimeError("r2 down")

    rig.sup._registry_reader = broken
    rig.sup.tick()
    assert rig.procs["t-a"].signals == [] and "t-a" in rig.map()["tasks"]


def test_the_spawn_binds_loopback_and_announces_nothing(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr("subprocess.Popen", lambda cmd, **kw: seen.update(cmd=cmd, env=kw["env"]) or Proc())
    subprocess_spawn(["reliquary", "validate"], tmp_path, {"PATH": "/bin"})("t-a", 9100)
    assert "--external-ip" not in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index("--http-host") + 1] == "127.0.0.1"
    assert seen["cmd"][seen["cmd"].index("--http-port") + 1] == "9100"
    assert seen["env"]["RELIQUARY_RL_RUN"] == "t-a"
    assert seen["env"]["RELIQUARY_TASK_ID"] == "t-a" and seen["env"]["RELIQUARY_RL_MODE"] == "async"


def test_a_retired_task_not_yet_final_is_restarted_when_its_controller_died(tmp_path):
    rig = Rig(tmp_path)
    rig.add("t-a")
    rig.sup.tick()
    rig.entries["t-a"].status = "retired"
    rig.procs["t-a"].code = 1
    for _ in range(8):
        rig.now += 5
        rig.sup.tick()
    assert len(rig.spawned) == 2
    # a fresh supervisor (deploy) picks up retired-not-final tasks, and not final ones
    fresh = Rig(tmp_path / "f") if (tmp_path / "f").mkdir() is None else None
    fresh.entries = rig.entries
    fresh.add("t-done", status="retired")
    fresh.final.add("t-done")
    fresh.sup.tick()
    assert [t for t, _ in fresh.spawned] == ["t-a"]


def test_a_failing_spawn_or_map_write_leaves_running_controllers_alone(tmp_path):
    rig = Rig(tmp_path)
    rig.add("t-a")
    rig.sup.tick()
    rig.add("t-b")

    def boom(task_id, port):
        raise RuntimeError("fork failed")

    rig.sup._spawn = boom
    rig.sup.tick()
    assert rig.procs["t-a"].signals == [] and "t-a" in rig.map()["tasks"]
    rig.sup._spawn = rig.spawn
    rig.sup._map_path = tmp_path / "missing" / "deep" / "map.json"
    (tmp_path / "missing").write_text("a file, not a directory")
    rig.now += 100
    rig.sup.tick()                                      # the write fails; no exception
    assert rig.procs["t-a"].signals == []


def test_a_second_supervisor_on_the_same_state_dir_is_refused(tmp_path):
    import pytest

    from reliquary.validator.rl_supervisor import StateLocked, lock_state_dir

    first = lock_state_dir(tmp_path)
    with pytest.raises(StateLocked):
        lock_state_dir(tmp_path)
    first.close()
    lock_state_dir(tmp_path).close()


def test_the_map_names_each_tasks_model(tmp_path):
    from reliquary.protocol.profiles import PROFILES

    rig = Rig(tmp_path)
    rig.add("t-a")
    rig.entries["t-a"].profile_id, rig.entries["t-a"].contract = "qwen35-4b-auction-v3", None
    rig.add("t-b")                                   # no profile it can resolve: no model named
    rig.sup.tick()
    profile = PROFILES["qwen35-4b-auction-v3"]
    tasks = rig.map()["tasks"]
    assert tasks["t-a"]["model"] == {"id": profile.model_id, "revision": profile.model_revision}
    assert "model" not in tasks["t-b"]


GUMBEL = "qwen35-4b-reliquary-suite-v10-gumbel-dev1"


def _spawned(monkeypatch, tmp_path, entry, env=None):
    seen = {}
    monkeypatch.setattr("subprocess.Popen", lambda cmd, **kw: seen.update(cmd=cmd, env=kw["env"]) or Proc())
    base = {"PATH": "/bin", "RELIQUARY_RL_TRAINER_TOKEN": "s" * 32, "RELIQUARY_RL_TRAINER_BIND": "127.0.0.1:8889",
            "RELIQUARY_STATE_DIR": "/root/shared", **(env or {})}
    subprocess_spawn(["reliquary", "validate"], tmp_path, base)(entry.task_id, 9100, entry)
    return seen


def _entry(task_id, profile_id=GUMBEL, contract=None):
    return SimpleNamespace(task_id=task_id, status="active", params={"training_order": {}}, profile_id=profile_id,
                           contract=contract)


def test_each_child_has_its_own_state_dir_sets_no_weights_and_holds_no_trainer_secret(tmp_path, monkeypatch):
    a = _spawned(monkeypatch, tmp_path, _entry("t-a"))
    b = _spawned(monkeypatch, tmp_path, _entry("t-b"))
    assert a["env"]["RELIQUARY_STATE_DIR"] == str(tmp_path / "t-a")
    assert b["env"]["RELIQUARY_STATE_DIR"] == str(tmp_path / "t-b")
    for seen in (a, b):
        assert "--no-set-weights" in seen["cmd"]
        assert "RELIQUARY_RL_TRAINER_TOKEN" not in seen["env"] and "RELIQUARY_RL_TRAINER_BIND" not in seen["env"]


def test_a_child_runs_its_tasks_profile_and_model(tmp_path, monkeypatch):
    from reliquary.protocol import profiles

    profile = profiles.PROFILES[GUMBEL]
    seen = _spawned(monkeypatch, tmp_path, _entry("t-a"), env={"RELIQUARY_PROTOCOL_PROFILE": "qwen35-2b-auction-v2",
                                                                "RELIQUARY_TASK_CONTRACT": "/elsewhere.json"})
    assert seen["env"]["RELIQUARY_PROTOCOL_PROFILE"] == GUMBEL and "RELIQUARY_TASK_CONTRACT" not in seen["env"]
    assert seen["cmd"][seen["cmd"].index("--checkpoint") + 1] == profile.model_id
    monkeypatch.setattr(profiles.os, "environ", seen["env"])
    assert profiles.resolve_protocol_profile().profile_id == GUMBEL

    carried = _spawned(monkeypatch, tmp_path, _entry("t-c", contract=profile.to_generation_contract()))
    path = carried["env"]["RELIQUARY_TASK_CONTRACT"]
    assert path.startswith(str(tmp_path / "t-c"))
    monkeypatch.setattr(profiles.os, "environ", carried["env"])
    assert profiles.resolve_protocol_profile().model_id == profile.model_id
    assert carried["cmd"][carried["cmd"].index("--checkpoint") + 1] == profile.model_id


def test_the_default_task_stays_while_it_restarts(tmp_path):
    rig = Rig(tmp_path)
    rig.add("t-b")
    rig.sup.tick()
    rig.now += 15
    rig.add("t-a")
    rig.sup.tick()
    assert rig.map()["default"] == "t-b"
    rig.procs["t-b"].code = 1
    rig.now += 5
    rig.sup.tick()
    assert "t-b" not in rig.map()["tasks"] and rig.map()["default"] == "t-b"   # bare paths: 503 while it restarts
    rig.now += 31
    rig.sup.tick()
    assert rig.map()["default"] == "t-b" and "t-b" in rig.map()["tasks"]
    rig.entries["t-b"].status = "retired"
    rig.final.add("t-b")
    rig.now += 15
    rig.sup.tick()
    assert rig.map()["default"] == "t-a"                                         # stopped for good: the next


def test_the_default_is_the_first_task_seen_across_supervisor_restarts(tmp_path):
    rig = Rig(tmp_path, order_path=tmp_path / "order.json")
    rig.add("t-z")
    rig.sup.tick()
    rig.add("t-a")
    rig.now += 15
    rig.sup.tick()
    fresh = Rig(tmp_path, order_path=tmp_path / "order.json")
    fresh.entries = rig.entries
    fresh.sup.tick()
    assert fresh.map()["default"] == "t-z"                                       # not the first by name


def test_a_retired_task_without_a_bill_logs_an_error_every_hour_after_a_day(tmp_path, caplog):
    import logging

    billed: set[str] = set()
    rig = Rig(tmp_path, has_spend=lambda entry: entry.task_id in billed)
    rig.add("t-a")
    rig.add("t-b")
    rig.sup.tick()
    rig.entries["t-a"].status = rig.entries["t-b"].status = "retired"
    billed.add("t-b")
    with caplog.at_level(logging.ERROR, logger="reliquary.validator.rl_supervisor"):
        for _ in range(int(26 * 3600 / 15) + 1):
            rig.now += 15
            rig.sup.tick()
    errors = [r for r in caplog.records if r.levelno == logging.ERROR and "t-a" in r.getMessage()]
    assert len(errors) == 3                                                       # at 24 h, 25 h, 26 h
    assert not [r for r in caplog.records if r.levelno == logging.ERROR and "t-b" in r.getMessage()]
    assert rig.procs["t-a"].signals == []                                         # never stopped for it
