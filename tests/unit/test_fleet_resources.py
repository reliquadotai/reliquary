"""RL placement by CPU and memory: the validator mirrors the gateway's reservation of each session's agent box."""
import asyncio
import math
from types import SimpleNamespace

import pytest

pytest.importorskip("reliquary_sandbox.attest")

from reliquary.sandbox.fleet import Placement, reservation_need  # noqa: E402
from reliquary.sandbox.rl_capacity import PRESENT_WINDOW_S  # noqa: E402
from reliquary.sandbox.tasks import ResolvedTask, SweTaskResolver, task_cpu_millis  # noqa: E402
from tests.unit.sandbox_fixtures import (  # noqa: E402
    ADDRESS, GIB, MACHINE, NOW, capacity_report, machine_document, signer,
)
from tests.unit.test_rl_engagements import (  # noqa: E402
    HOTKEY, build as build_rl, close as close_rl, engagement, open_ as open_rl,
)
from tests.unit.test_sandbox_fleet import PICK, make  # noqa: E402
from tests.unit.test_sandbox_sessions import build as build_corpus, open_ as open_corpus  # noqa: E402

RESOURCES = {"cpu_millis": 2_000, "memory_bytes": 16 * GIB, "used_cpu_millis": 0, "used_memory_bytes": 0,
             "free_cpu_millis": 2_000, "free_memory_bytes": 16 * GIB, "max_box_cpu_millis": 1_000,
             "boxes_per_episode": 1, "cpu_overcommit_permille": 1_000, "sessions": []}
RESERVE = {"cpu_millis": 1_000, "memory_bytes": 2 * GIB}


def fleet_with(tmp_path, at=NOW, **resources):
    fleet, machine, served, clock = make(tmp_path)
    clock.now = at
    served[ADDRESS] = capacity_report(machine, at=at, resources={**RESOURCES, **resources})
    asyncio.run(fleet.poll_once())
    return fleet, machine, served, clock


def report(fleet, machine, served, clock, at, **resources):
    clock.now = at
    served[ADDRESS] = capacity_report(machine, at=at, resources={**RESOURCES, **resources})
    asyncio.run(fleet.poll_once())


# --- the need, and the one CPU rounding rule shared with the gateway ------------------------------------------

def test_the_need_mirrors_the_gateways_reservation():
    assert reservation_need(RESOURCES, {"cpu_millis": 500, "memory_bytes": GIB}) == (500, GIB)
    assert reservation_need(RESOURCES, {"cpu_millis": None, "memory_bytes": GIB}) == (1_000, GIB)
    assert reservation_need(RESOURCES, {"cpu_millis": 10, "memory_bytes": GIB}) == (100, GIB)     # 0.1 CPU floor
    assert reservation_need(RESOURCES, {"cpu_millis": 16_000, "memory_bytes": GIB}) is None       # never clamped
    assert reservation_need({**RESOURCES, "boxes_per_episode": 2}, {"cpu_millis": 500, "memory_bytes": GIB}) == (
        1_000, 2 * GIB)


def test_task_cpus_are_rounded_up_to_a_millicpu():
    assert task_cpu_millis(None) is None and task_cpu_millis(float("nan")) is None
    assert task_cpu_millis(True) is None
    assert [task_cpu_millis(c) for c in (0.5, 2, 2.0, 1.2355, 2.0004, 0.0001, 0)] == [500, 2000, 2000, 1236, 2001, 1, 1]
    assert task_cpu_millis(math.inf) > 10 ** 12


def _gateway_refuses(cpus, box_nano):
    """The gateway's box-size check (422), as it is written there."""
    return cpus * 1_000_000_000 > box_nano


@pytest.mark.parametrize("box_nano", [2_000_000_000, 1_000_000_000, 2_000_400_000, 1_234_567_891])
@pytest.mark.parametrize("cpus", [0.05, 0.5, 1.0, 1.2345678, 1.2345679, 1.2355, 2.0, 2.0004, 2.0004001, 2.0000001,
                                  1.000000001, 3.0])
def test_a_task_the_gateway_refuses_for_its_box_is_never_placed_there(cpus, box_nano):
    resources = {**RESOURCES, "max_box_cpu_millis": box_nano // 1_000_000}
    need = reservation_need(resources, {"cpu_millis": task_cpu_millis(cpus), "memory_bytes": GIB})
    if _gateway_refuses(cpus, box_nano):
        assert need is None
    elif box_nano % 1_000_000 == 0:
        assert need is not None                          # a whole-millis box: exactly the gateway's answer
    if need is not None:
        gateway_millis = max(100_000_000, round(min(box_nano, cpus * 1_000_000_000))) // 1_000_000
        assert gateway_millis <= need[0] <= gateway_millis + 1     # never below what the gateway reserves


# --- held reservations -----------------------------------------------------------------------------------------

def test_an_issued_reservation_is_held_until_a_report_shows_its_session(tmp_path):
    fleet, machine, served, clock = fleet_with(tmp_path, free_cpu_millis=1_000)
    assert fleet.pick(now=NOW, reserve=RESERVE, **PICK).machine_id == MACHINE
    fleet.note_issued(MACHINE, NOW, reserve=RESERVE, hold_s=PRESENT_WINDOW_S, session_id="s-1")
    assert fleet.pick(now=NOW, reserve=RESERVE, **PICK) is None
    # a newer report that does not show the session yet (its token not presented): still held, never zero times
    report(fleet, machine, served, clock, NOW + 5, free_cpu_millis=1_000)
    assert fleet.pick(now=NOW + 5, reserve=RESERVE, **PICK) is None
    # the report lists the session: its `used` counts it, the validator's copy is dropped (never twice)
    report(fleet, machine, served, clock, NOW + 10, free_cpu_millis=1_000, used_cpu_millis=1_000, sessions=["s-1"])
    assert fleet.pick(now=NOW + 10, reserve=RESERVE, **PICK) is not None
    # gone from a later report (closed on its gateway): never counted again
    report(fleet, machine, served, clock, NOW + 15, free_cpu_millis=2_000)
    assert fleet.pick(now=NOW + 15, reserve=RESERVE, **PICK) is not None


def test_a_report_listing_a_session_says_its_miner_had_the_token(tmp_path):
    fleet, machine, served, clock = fleet_with(tmp_path)
    assert not fleet.session_reported("s-1")
    report(fleet, machine, served, clock, NOW + 10, sessions=["s-1"])
    report(fleet, machine, served, clock, NOW + 20, sessions=[])            # gone from the next report: still seen
    assert fleet.session_reported("s-1") and not fleet.session_reported("s-2")


def test_a_reservation_is_dropped_when_its_session_leaves_the_book(tmp_path):
    fleet, _, _, _ = fleet_with(tmp_path, free_cpu_millis=1_000)
    fleet.note_issued(MACHINE, NOW, reserve=RESERVE, hold_s=PRESENT_WINDOW_S, session_id="s-1")
    fleet.note_settled("s-0")                                   # another session: nothing changes
    assert fleet.pick(now=NOW, reserve=RESERVE, **PICK) is None
    fleet.note_settled("s-1")                                   # closed before any report saw it
    assert fleet.pick(now=NOW, reserve=RESERVE, **PICK) is not None


def test_a_reservation_is_held_until_a_report_made_after_its_present_window(tmp_path):
    fleet, machine, served, clock = fleet_with(tmp_path, free_cpu_millis=1_000)
    fleet.note_issued(MACHINE, NOW, reserve=RESERVE, hold_s=60, session_id="s-2")
    # the window has passed, but the latest report was made inside it: the gateway may have taken the token at
    # second 59 and not reported it yet: still held
    report(fleet, machine, served, clock, NOW + 40, free_cpu_millis=1_000)
    clock.now = NOW + 65
    assert fleet.pick(now=NOW + 65, reserve=RESERVE, **PICK) is None
    # a report made after the window does not list it: the gateway refuses that token now, the hold ends
    report(fleet, machine, served, clock, NOW + 66, free_cpu_millis=1_000)
    assert fleet.pick(now=NOW + 66, reserve=RESERVE, **PICK) is not None


def test_slots_still_bound_an_rl_placement(tmp_path):
    fleet, machine, served, clock = make(tmp_path)
    served[ADDRESS] = capacity_report(machine, at=NOW, free=0, resources=RESOURCES)
    asyncio.run(fleet.poll_once())
    assert fleet.pick(now=NOW, reserve=RESERVE, **PICK) is None             # the gateway refuses `no free slot`


def test_memory_bounds_placement(tmp_path):
    fleet, _, _, _ = fleet_with(tmp_path, free_memory_bytes=3 * GIB)
    assert fleet.pick(now=NOW, reserve={"cpu_millis": 100, "memory_bytes": 4 * GIB}, **PICK) is None
    assert fleet.pick(now=NOW, reserve={"cpu_millis": 100, "memory_bytes": GIB}, **PICK) is not None
    assert fleet.could_place(now=NOW, reserve={"cpu_millis": 100, "memory_bytes": 4 * GIB}, **PICK) is True


def test_a_box_smaller_than_the_task_declares_is_never_used(tmp_path):
    fleet, _, _, _ = fleet_with(tmp_path)
    big = {"cpu_millis": 4_000, "memory_bytes": GIB}
    assert fleet.pick(now=NOW, reserve=big, **PICK) is None
    assert fleet.could_place(now=NOW, reserve=big, **PICK) is False            # never: refused at once
    assert fleet.nearest(now=NOW, reserve=big, **PICK) is None                # and never held for it
    assert fleet.could_place(now=NOW, reserve={"cpu_millis": 1_000, "memory_bytes": GIB}, **PICK) is True
    assert fleet.could_place(now=NOW, reserve={"cpu_millis": 1_000, "memory_bytes": 32 * GIB}, **PICK) is False
    assert fleet.could_place(now=NOW, **PICK) is True                          # no reserve: slots, unchanged


def test_a_machine_without_resources_gets_no_rl_session_once_resources_are_required(tmp_path, caplog):
    fleet, machine, served, _ = make(tmp_path)
    served[ADDRESS] = capacity_report(machine, at=NOW)
    asyncio.run(fleet.poll_once())
    reserve = {"cpu_millis": 100, "memory_bytes": GIB}
    assert fleet.pick(now=NOW, **PICK) is not None
    assert fleet.pick(now=NOW, reserve=reserve, **PICK) is not None     # not required: slots (test fleets)
    fleet.resources_required = True                                     # the production RL wiring
    with caplog.at_level("ERROR", logger="reliquary.sandbox.fleet"):
        assert fleet.pick(now=NOW, reserve=reserve, **PICK) is None
        fleet.pick(now=NOW, reserve=reserve, **PICK)
    assert sum("reports no valid resources" in r.message for r in caplog.records) == 1   # once per machine
    # a configuration miss of ours: retryable (cannot be judged), never final for the seed
    assert fleet.could_place(now=NOW, reserve=reserve, **PICK) is None
    assert fleet.nearest(now=NOW, reserve=reserve, **PICK) is None
    assert fleet.pick(now=NOW, **PICK) is not None                      # corpus: slots, unchanged


def test_the_production_wiring_requires_resources(tmp_path):
    from tests.unit.test_rl_episode_wiring import _services

    assert _services(tmp_path / "a", require_resources=True)[0].fleet.resources_required is True
    assert _services(tmp_path / "b")[0].fleet.resources_required is False


@pytest.mark.parametrize("broken", [{"free_cpu_millis": "2000"}, {"boxes_per_episode": 0}, {"cpu_millis": -1},
                                    {"max_box_cpu_millis": 0}, {"free_memory_bytes": True}])
def test_a_malformed_resources_block_places_nothing(tmp_path, broken):
    fleet, _, _, _ = fleet_with(tmp_path, **broken)
    assert fleet.pick(now=NOW, reserve={"cpu_millis": 100, "memory_bytes": GIB}, **PICK) is None


# --- the least loaded, and the machine held for the queue's head -------------------------------------------------

def two_machines(tmp_path):
    one, two = signer(tmp_path, "a", "ka"), signer(tmp_path, "b", "kb")
    documents = [machine_document(one, machine_id="m-a", address="http://a:1"),
                 machine_document(two, machine_id="m-b", address="http://b:1")]
    fleet, _, served, _ = make(tmp_path, documents=documents)
    # m-a: many slots, little CPU; m-b: one slot, much CPU
    served["http://a:1"] = capacity_report(one, at=NOW, machine_id="m-a", address="http://a:1", free=4,
                                           resources={**RESOURCES, "free_cpu_millis": 500})
    served["http://b:1"] = capacity_report(two, at=NOW, machine_id="m-b", address="http://b:1", free=1,
                                           resources={**RESOURCES, "free_cpu_millis": 1_600})
    asyncio.run(fleet.poll_once())
    return fleet


def test_the_least_loaded_by_cpu_and_memory_is_picked_and_held(tmp_path):
    fleet = two_machines(tmp_path)
    small = {"cpu_millis": 400, "memory_bytes": GIB}
    assert fleet.pick(now=NOW, **PICK) == Placement("m-a", "http://a:1")              # slots (corpus)
    assert fleet.pick(now=NOW, reserve=small, **PICK) == Placement("m-b", "http://b:1")
    assert fleet.pick(now=NOW, reserve=small, exclude={"m-b"}, **PICK) == Placement("m-a", "http://a:1")
    big = {"cpu_millis": 1_000, "memory_bytes": GIB}
    assert fleet.pick(now=NOW, reserve=big, exclude={"m-b"}, **PICK) is None           # m-a lacks 500 millis
    # held for a big head waiter: the machine closest to fitting it by CPU and memory, not by slots
    fleet.note_issued("m-b", NOW, reserve={"cpu_millis": 1_000, "memory_bytes": GIB}, hold_s=120, session_id="x")
    assert fleet.pick(now=NOW, reserve=big, **PICK) is None                             # m-b: 600 left
    assert fleet.nearest(now=NOW, reserve=big, **PICK) == "m-b"                         # short 400, m-a short 500
    fleet.note_issued("m-b", NOW, reserve={"cpu_millis": 500, "memory_bytes": GIB}, hold_s=120, session_id="y")
    assert fleet.nearest(now=NOW, reserve=big, **PICK) == "m-a"                         # m-b now short 900
    fleet.note_settled("x")
    fleet.note_settled("y")
    assert fleet.nearest(now=NOW, reserve=big, **PICK) == "m-b"
    assert fleet.nearest(now=NOW, **PICK) == "m-a"                                      # slots: unchanged


# --- the issuer ------------------------------------------------------------------------------------------------

def test_rl_terms_carry_the_tasks_cpus_and_memory(tmp_path):
    env = build_rl(tmp_path)
    grant = open_rl(env, 0)
    picked = env.issuer._fleet.picked[-1]
    memory = env.issuer._fleet.picked[-1]["budgets"]["memory_bytes"]
    assert picked["reserve"] == {"cpu_millis": None, "memory_bytes": memory}
    kw = env.issuer._fleet.issued_kw[-1]
    record = env.book.get(grant.session_id)
    assert kw == {"reserve": {"cpu_millis": None, "memory_bytes": memory},
                  "hold_s": min(record.expires_at - record.issued_at, PRESENT_WINDOW_S),
                  "session_id": grant.session_id}
    assert kw["hold_s"] == PRESENT_WINDOW_S                      # a token longer than its present window
    close_rl(env, grant, "graded")
    assert env.issuer._fleet.settled[-1] == grant.session_id      # the fleet drops its hold


def test_lapsed_and_voided_sessions_drop_their_holds(tmp_path):
    env = build_rl(tmp_path)
    grant = open_rl(env, 0)

    async def void():
        env.issuer.void_machine(MACHINE)
        await env.issuer.drain(5)
    asyncio.run(void())
    assert env.issuer._fleet.settled == [grant.session_id]
    second = open_rl(env, 1)
    record = env.book.get(second.session_id)
    env.issuer._clock.now = record.expires_at + 10_000
    asyncio.run(env.issuer.maintain())
    assert env.issuer._fleet.settled == [grant.session_id, second.session_id]


def test_corpus_placement_keeps_its_exact_call(tmp_path):
    env = build_corpus(tmp_path)
    open_corpus(env)
    assert "reserve" not in env.fleet.picked[-1]
    assert env.fleet.issued_kw[-1] == {}


class Recorder:
    def __init__(self):
        self.calls = []

    def pick(self, **kw):
        self.calls.append(("pick", kw))

    def could_place(self, **kw):
        self.calls.append(("could_place", kw))
        return True

    def nearest(self, **kw):
        self.calls.append(("nearest", kw))
        return "m-1"


class HoldingQueue:
    def __init__(self):
        self.held, self.missed_rooms = None, []

    def should_hold(self, ticket):
        return True

    def hold(self, ticket, target):
        self.held = target

    def missed(self, ticket, *, room=None):
        self.missed_rooms.append(room)


def test_the_queue_looks_for_room_and_holds_a_machine_by_the_reservation(tmp_path):
    env = build_rl(tmp_path)
    terms = asyncio.run(env.issuer._engagements["rl_precommit"].terms(HOTKEY, engagement(env.precommit, 0)))
    fleet, queue = Recorder(), HoldingQueue()
    env.issuer._fleet, env.issuer._queue = fleet, queue
    assert env.issuer._no_room_now(terms, frozenset({"m-2"})) is True
    env.issuer._missed(object(), terms, "room")
    assert queue.held == "m-1"
    assert [name for name, _ in fleet.calls] == ["pick", "could_place", "nearest"]
    assert all(kw["reserve"] == terms.reserve for _, kw in fleet.calls)
    assert fleet.calls[0][1]["exclude"] == frozenset({"m-2"})


def test_the_resolver_reads_the_tasks_cpus():
    task = SimpleNamespace(image="i@sha256:" + "a" * 64, limits=SimpleNamespace(cpus=0.5, memory_bytes=GIB,
                                                                                    disk_bytes=None, pids=None,
                                                                                    wall_s=None,
                                                                                    per_call_timeout_s=None,
                                                                                    max_calls=None))
    resolved = asyncio.run(SweTaskResolver("train", lambda split, index: task, options_sha256="1" * 64).resolve(0))
    assert resolved == ResolvedTask(task.image, {"memory_bytes": GIB}, "1" * 64, cpu_millis=500)
    task.limits.cpus = None
    resolved = asyncio.run(SweTaskResolver("train", lambda split, index: task, options_sha256="1" * 64).resolve(0))
    assert resolved.cpu_millis is None
