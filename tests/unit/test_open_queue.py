"""The fair open queue of RL sandbox sessions at saturation."""
import asyncio
from types import SimpleNamespace

import pytest

attest = pytest.importorskip("reliquary_sandbox.attest")

from reliquary.miner.rl_episode_client import RL_FINAL_REFUSALS, RL_THROTTLES  # noqa: E402
from reliquary.miner.signed_episode import SignedSweEpisodeRunner  # noqa: E402
from reliquary.protocol.sandbox_session import SessionRefused  # noqa: E402
from reliquary.sandbox.fleet import Placement  # noqa: E402
from reliquary.sandbox.routes import REFUSAL_STATUS  # noqa: E402
from reliquary.sandbox.sessions import Grant, _placement_class  # noqa: E402
from tests.unit.sandbox_fixtures import ADDRESS, MACHINE  # noqa: E402
from tests.unit.test_rl_engagements import HOTKEY, build, close, engagement, open_  # noqa: E402
from tests.unit.test_sandbox_fleet import make  # noqa: E402

A, B = ("env-a", 500, 1), ("env-b", 500, 1)          # placement classes (env, cpu_millis, memory_bytes)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


# --- the queue alone -------------------------------------------------------------------------------------------

# --- the issuer with the queue ---------------------------------------------------------------------------------

def opening(env, seed, request_id):
    return asyncio.ensure_future(env.issuer.open(hotkey=HOTKEY, request_id=request_id,
                                                 engagement=engagement(env.precommit, seed)))


async def until(predicate, seconds=2.0):
    for _ in range(int(seconds / 0.01)):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never met")


class Refusals:
    def __init__(self):
        self.seen = []

    def opened(self, record):
        pass

    def closed(self, record, final):
        pass

    def refused(self, hotkey, engagement, refusal):
        self.seen.append(refusal.reason)


def test_without_a_queue_no_room_is_refused_at_once_as_before(tmp_path):
    env = build(tmp_path)
    env.issuer._fleet.placement = None
    refused = asyncio.run(env.issuer.open(hotkey=HOTKEY, request_id="a" * 32,
                                          engagement=engagement(env.precommit, 0)))
    assert refused.reason == "sandbox_capacity" and "why" not in refused.detail


def test_the_queue_refusals_are_retried_by_the_miner():
    assert REFUSAL_STATUS["sandbox_saturated"] == REFUSAL_STATUS["open_queue_full"] == 503
    assert REFUSAL_STATUS["open_queue_key_full"] == REFUSAL_STATUS["open_queued"] == 429
    assert REFUSAL_STATUS["task_unplaceable"] == 409


def test_the_2c_miner_reopens_under_saturation_and_drops_an_unplaceable_seed():
    assert {"sandbox_saturated", "open_queue_full"} <= RL_THROTTLES
    assert "task_unplaceable" in RL_FINAL_REFUSALS
    runner = SimpleNamespace(validator_throttles=RL_THROTTLES, _backoff=0, _unavailable=0,
                             _hold_opens=lambda seconds: None)
    for _ in range(12):                              # more than MAX_UNAVAILABLE in a row
        answer = SignedSweEpisodeRunner._note_refusal(
            runner, SessionRefused("sandbox_saturated", {}, status=503, retry_after=10))
        assert answer.reason == "sandbox_saturated"  # never validator_unavailable


def test_the_poll_listener_fires_on_an_early_return_too(tmp_path):
    fleet, _machine, _served, _clock = make(tmp_path)        # no machine answers: poll_once returns early
    calls = []
    fleet.on_polled = lambda: calls.append(1)
    asyncio.run(fleet.poll_once())
    assert calls == [1]


# --- the fleet cannot always judge, coalesced sweeps, the head's hold, the room checks ----------------------

from dataclasses import fields  # noqa: E402

from reliquary.sandbox.fleet import HEARTBEAT_MAX_AGE_S  # noqa: E402
from reliquary.sandbox.sessions import SandboxPolicy, SessionBook, SessionIssuer, _tier  # noqa: E402
from tests.unit.sandbox_fixtures import capacity_report, machine_document, signer  # noqa: E402
from tests.unit.test_sandbox_fleet import PICK  # noqa: E402
from tests.unit.test_sandbox_sessions import Clock as IssuerClock  # noqa: E402


def test_the_fleet_answers_cannot_judge_for_our_outages_and_never_only_from_fresh_reports(tmp_path):
    fleet, machine, served, clock = make(tmp_path)
    now = clock.now
    assert fleet.could_place(**PICK, now=now) is None              # a start: no report accepted yet
    served[ADDRESS] = capacity_report(machine, at=now)
    asyncio.run(fleet.poll_once())
    assert fleet.could_place(**PICK, now=now) is True
    assert fleet.could_place(**{**PICK, "image": "other@sha256:" + "b" * 64}, now=now) is False   # never
    fleet.drained.add(MACHINE)                                       # its holder drained for a moment
    assert fleet.could_place(**PICK, now=now) is None
    fleet.drained.discard(MACHINE)
    assert fleet.could_place(**PICK, now=now + HEARTBEAT_MAX_AGE_S + 1) is None    # every report stale
    served.clear()                                                   # no machine answers: our egress
    asyncio.run(fleet.poll_once())
    assert fleet._egress_suspect and fleet.could_place(**PICK, now=now) is None


def test_the_fleet_never_and_unknown_follow_what_each_machine_last_reported(tmp_path):
    fleet, machine, served, clock = make(tmp_path)
    served[ADDRESS] = capacity_report(machine, at=clock.now, images=())
    asyncio.run(fleet.poll_once())
    assert fleet.could_place(**PICK, now=clock.now) is False         # fresh, and it does not serve the image
    fleet.drained.add(MACHINE)
    assert fleet.could_place(**PICK, now=clock.now) is None          # no fresh report left: cannot judge


def test_a_drained_or_stale_image_holder_is_cannot_judge_beside_a_fresh_machine_without_it(tmp_path):
    one, two = signer(tmp_path, "a", "ka"), signer(tmp_path, "b", "kb")
    documents = [machine_document(one, machine_id="m-a", address="http://a:1"),
                 machine_document(two, machine_id="m-b", address="http://b:1")]
    fleet, _, served, clock = make(tmp_path, documents=documents)
    served["http://a:1"] = capacity_report(one, at=clock.now, machine_id="m-a", address="http://a:1")
    served["http://b:1"] = capacity_report(two, at=clock.now, machine_id="m-b", address="http://b:1", images=())
    asyncio.run(fleet.poll_once())
    assert fleet.could_place(**PICK, now=clock.now) is True
    fleet.drained.add("m-a")                         # the image's only holder drained; m-b fresh, without it
    assert fleet.could_place(**PICK, now=clock.now) is None
    fleet.drained.discard("m-a")
    fleet._reports["m-a"]["at"] = clock.now - HEARTBEAT_MAX_AGE_S - 5         # its report went stale
    assert fleet.could_place(**PICK, now=clock.now) is None


def test_without_a_queue_refused_opens_count_as_before(tmp_path):
    env = build(tmp_path)
    env.issuer._fleet.placement = None
    assert open_(env, 0).reason == "sandbox_capacity" and len(env.issuer._refused_opens[HOTKEY]) == 1


def test_reservations_are_classed_by_tier():
    assert [_tier(v) for v in (1, 1000, 1024, 1500, 2048, 3000)] == [1, 1024, 1024, 2048, 2048, 4096]
    assert _tier(None) is None and _tier(0) == 0

    def cls(cpu, mem):
        return _placement_class(SimpleNamespace(env="e", image="i", reserve={"cpu_millis": cpu, "memory_bytes": mem}))
    assert cls(1500, 3 << 30) == cls(1800, (7 << 30) // 2)     # per-task reservations share a few classes
    assert cls(1500, 3 << 30) != cls(4000, 3 << 30)


# --- a fleet with CPUs: big and small reservations -------------------------------------------------------------

class Rooms:
    """Machines with CPUs; an open's need is its image's CPUs, taken when its session opens and freed when the
    test ends it."""

    def __init__(self, cpus, need):
        self.free, self.need, self.picks = dict(cpus), dict(need), 0
        self.ended = None

    def directory_ready(self, now=None):
        return True

    def pick(self, *, image, exclude=(), **kw):
        self.picks += 1
        fits = [m for m, free in self.free.items() if m not in exclude and free >= self.need[image]]
        return Placement(max(fits, key=lambda m: (self.free[m], m)), ADDRESS) if fits else None

    def could_place(self, **kw):
        return True

    def nearest(self, **kw):
        return max(self.free, key=lambda m: (self.free[m], m))

    def note_issued(self, machine_id, at, **kw):
        pass

    # the session listener: the open takes its CPUs
    def opened(self, record):
        image = record.engagement.split(":")[0]
        self.free[record.machine_id] -= self.need[image]
        if self.ended is not None:
            self.ended(record.machine_id, self.need[image])

    def closed(self, record, final):
        pass


class Opens:
    kind = "fake"

    def __init__(self, template, need):
        self.template, self.need = template, need

    async def terms(self, hotkey, engagement):
        terms = SimpleNamespace(**{f.name: getattr(self.template, f.name) for f in fields(self.template)})
        terms.image, terms.engagement = engagement["image"], f"{engagement['image']}:{engagement['n']}"
        terms.exclusive, terms.still_valid = False, None
        terms.reserve = {"cpu_millis": 1000 * self.need[terms.image], "memory_bytes": 1 << 30}
        return terms


async def rooms_issuer(tmp_path, rooms, q):
    env = build(tmp_path)
    template = await env.issuer._engagements["rl_precommit"].terms(HOTKEY, engagement(env.precommit, 0))
    policy = SandboxPolicy(max_live_per_hotkey=10 ** 6, max_opens_per_hour=10 ** 6,
                           max_refused_opens_per_minute=10 ** 6)
    validator = signer(tmp_path, "v2", "v2")
    return SessionIssuer(book=SessionBook(policy), store=env.issuer._store, fleet=rooms, signer=validator,
                         token_verifier=env.issuer._token_verifier, engagements={"fake": Opens(template, rooms.need)},
                         policy=policy, clock=IssuerClock(), events=rooms, queue=q)


def open_one(issuer, hotkey, image, n):
    return asyncio.ensure_future(issuer.open(hotkey=hotkey, request_id=f"{n:032x}",
                                             engagement={"kind": "fake", "image": image, "n": n}))


class VirtualTimeLoop(asyncio.SelectorEventLoop):
    """An event loop on a virtual clock: when nothing is ready it jumps to its next timer instead of sleeping, so a
    timed simulation runs the same way whatever the machine's load (no wall clock anywhere)."""

    def __init__(self):
        super().__init__()
        self._now = 0.0
        real = self._selector.select

        def select(timeout=None):
            events = real(0)
            if not events and timeout is None:
                # Nothing ready, no timer: every task waits on another. Fail now, never hang to the file's timeout.
                raise RuntimeError("the simulation deadlocked: nothing ready and no timer pending")
            if not events and timeout:
                self._now += timeout
            return events
        self._selector.select = select

    def time(self):
        return self._now


def run_virtual(coroutine):
    loop = VirtualTimeLoop()
    try:
        return loop.run_until_complete(coroutine)
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()


def test_a_deadlocked_simulation_fails_at_once():
    async def stuck():
        await asyncio.get_running_loop().create_future()           # nobody ever sets it, and no timer is pending

    with pytest.raises(RuntimeError, match="deadlocked"):
        run_virtual(stuck())


