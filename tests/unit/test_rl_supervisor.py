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

    def spawn(self, task_id: str, port: int) -> Proc:
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
