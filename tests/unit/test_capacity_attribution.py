"""A saturation gives a capacity verdict only when others held the capacity."""
import asyncio

import pytest

attest = pytest.importorskip("reliquary_sandbox.attest")

from reliquary.protocol.service_episode import rl_engagement  # noqa: E402
from reliquary.sandbox.sessions import LIVE, SandboxPolicy, SessionBook  # noqa: E402
from tests.unit.sandbox_fixtures import MACHINE, NOW  # noqa: E402
from tests.unit.test_open_queue import Refusals, queue  # noqa: E402
from tests.unit.test_rl_engagements import HOTKEY, build, engagement, open_  # noqa: E402
from tests.unit.test_rl_episode_wiring import _record  # noqa: E402

RECEIPT = {"document": {"kind": "open_refusal", "machine_id": MACHINE, "key_id": "k1", "session_id": "s",
                        "token_sha256": "a" * 64, "reason": "no_free_slot", "at": 1}, "signature": "sig"}


def saturating(env, *, operators=None, **queue_kwargs):
    queue_kwargs.setdefault("wait_s", 0.05)
    env.issuer._queue = q = queue(**queue_kwargs)
    env.issuer._events = listener = Refusals()
    env.issuer._operator_of = (operators or {}).get
    return q, listener


def others_hold(env, n, hotkey="5Other", machine=MACHINE):
    for k in range(n):
        record = _record(f"o-{hotkey}-{machine}-{k}", sha="f" * 64, hotkey=hotkey, state=LIVE, seed=k,
                         issued_at=NOW - 60, expires_at=NOW + 600)
        record.machine_id = machine
        env.book.add(record)


def refused(env, seed):
    env.issuer._fleet.placement = None                  # no room now; the machine could take it later
    return asyncio.run(env.issuer.open(hotkey=HOTKEY, request_id=f"{seed:032x}",
                                       engagement=engagement(env.precommit, seed)))


def test_a_saturation_of_the_miners_own_making_gives_no_verdict(tmp_path):
    env = build(tmp_path)
    _q, listener = saturating(env)
    others_hold(env, 4, hotkey=HOTKEY)                   # its OTHER sessions (another precommit) fill the machine
    open_(env, 0)
    assert refused(env, 1).reason == "sandbox_saturated"
    assert listener.seen == []


def test_its_own_single_group_never_makes_the_saturation_self_caused(tmp_path):
    env = build(tmp_path)
    _q, listener = saturating(env)
    for seed in range(3):
        open_(env, seed)                                 # only this precommit's sessions on the machine
    others_hold(env, 2)
    assert refused(env, 3).reason == "sandbox_saturated"
    assert listener.seen == ["sandbox_saturated"]


def test_a_saturation_others_caused_gives_one(tmp_path):
    env = build(tmp_path)
    _q, listener = saturating(env)
    for seed in range(2):
        open_(env, seed)
    others_hold(env, 6)
    assert refused(env, 2).reason == "sandbox_saturated"
    assert listener.seen == ["sandbox_saturated"]


def test_the_operators_other_hotkeys_are_its_own(tmp_path):
    env = build(tmp_path)
    _q, listener = saturating(env, operators={HOTKEY: "cold", "5Sybil": "cold"})
    open_(env, 0)
    others_hold(env, 6, hotkey="5Sybil")
    assert refused(env, 2).reason == "sandbox_saturated"
    assert listener.seen == []


def test_only_the_machines_that_could_take_the_open_are_read(tmp_path):
    env = build(tmp_path)
    _q, listener = saturating(env)
    others_hold(env, 6, hotkey=HOTKEY, machine="elsewhere")   # its load on a machine that cannot take this task
    others_hold(env, 2)
    assert refused(env, 0).reason == "sandbox_saturated"
    assert listener.seen == ["sandbox_saturated"]


@pytest.mark.parametrize("waiting, seen", [(HOTKEY, []), ("5Other", ["open_queue_full"])])
def test_a_full_queue_counts_only_when_others_fill_it(tmp_path, waiting, seen):
    env = build(tmp_path)
    q, listener = saturating(env, max_waiting=1, wait_s=5.0)
    q.join("5Holder", cls=("elsewhere", None, None))     # holds the turn
    q.join(waiting, cls=("elsewhere", None, None))       # fills the queue
    assert refused(env, 0).reason == "open_queue_full"
    assert listener.seen == seen
    assert q.keys() == [waiting]


def test_an_open_no_machine_could_ever_take_keeps_its_verdict_whatever_the_load(tmp_path):
    env = build(tmp_path)
    _q, listener = saturating(env)
    others_hold(env, 6, hotkey=HOTKEY)
    env.issuer._fleet.placeable = False
    assert refused(env, 0).reason == "task_unplaceable"
    assert listener.seen == ["task_unplaceable"]


def test_a_failing_attribution_gives_no_verdict(tmp_path):
    env = build(tmp_path)
    _q, listener = saturating(env)
    others_hold(env, 6)

    def broken(**kw):
        raise RuntimeError("fleet")

    env.issuer._fleet.servable_machines = broken
    assert refused(env, 0).reason == "sandbox_saturated"
    assert listener.seen == []


@pytest.mark.parametrize("holder", [HOTKEY, "5Other"])
def test_a_signed_machine_full_refusal_is_ours_whoever_filled_the_machine(tmp_path, holder):
    env = build(tmp_path)
    env.issuer._record_faults = True
    env.issuer._open_refusal_check = lambda receipt, record: True
    grant = open_(env, 0)
    others_hold(env, 3, hotkey=holder)
    asyncio.run(env.issuer.close(hotkey=HOTKEY, session_id=grant.session_id, reason="open_failed",
                                 transcript=None, open_refusal=RECEIPT))
    assert env.book.get(grant.session_id).fault == "open_refused:no_free_slot"


def test_load_the_book_does_not_hold_is_never_the_miners(tmp_path):
    env = build(tmp_path)
    _q, listener = saturating(env)
    others_hold(env, 1, hotkey=HOTKEY)                   # its other precommit: the only RL session on the machine
    env.issuer._fleet.reported_load = lambda machines: 7 if MACHINE in machines else 0   # a corpus job fills it
    assert refused(env, 0).reason == "sandbox_saturated"
    assert listener.seen == ["sandbox_saturated"]


@pytest.mark.parametrize("entitled, seen", [(8, ["sandbox_saturated"]), (3, [])])
def test_within_its_allocation_a_saturation_is_never_the_miners(tmp_path, entitled, seen):
    env = build(tmp_path)
    _q, listener = saturating(env)
    others_hold(env, 4, hotkey=HOTKEY)                   # 4 held: within an allocation of 8, beyond one of 3
    env.issuer._entitled = lambda hotkey: entitled if hotkey == HOTKEY else 0
    assert refused(env, 0).reason == "sandbox_saturated"
    assert listener.seen == seen


@pytest.mark.parametrize("sybil_entitled, seen", [(4, ["sandbox_saturated"]), (0, [])])
def test_the_entitlement_is_the_operators_hotkeys_together(tmp_path, sybil_entitled, seen):
    env = build(tmp_path)
    _q, listener = saturating(env, operators={HOTKEY: "cold", "5Sybil": "cold"})
    others_hold(env, 3, hotkey=HOTKEY)
    others_hold(env, 3, hotkey="5Sybil")                 # 6 held by the operator
    env.issuer._entitled = {HOTKEY: 2, "5Sybil": sybil_entitled}.get
    assert refused(env, 0).reason == "sandbox_saturated"
    assert listener.seen == seen


def test_a_full_queue_of_its_own_groups_tickets_is_not_its_doing(tmp_path):
    env = build(tmp_path)
    q, listener = saturating(env, max_waiting=2, wait_s=5.0)
    q.join("5Holder", cls=("elsewhere", None, None))     # holds the turn
    q.join(HOTKEY, cls=("elsewhere", None, None), engagement=rl_engagement(1, env.precommit.sha256, 5))
    q.join("5Other", cls=("elsewhere", None, None))
    assert refused(env, 0).reason == "open_queue_full"
    assert listener.seen == ["open_queue_full"]           # its own group's ticket is not its other load


def test_attribution_failures_are_counted(tmp_path):
    env = build(tmp_path)
    _q, _listener = saturating(env)

    def broken(**kw):
        raise RuntimeError("fleet")

    env.issuer._fleet.servable_machines = broken
    refused(env, 0)
    refused(env, 1)
    assert env.issuer.attribution_failures == 2


def test_another_signed_refusal_is_ours_whoever_holds_the_machine(tmp_path):
    env = build(tmp_path)
    env.issuer._record_faults = True
    env.issuer._open_refusal_check = lambda receipt, record: True
    grant = open_(env, 0)
    others_hold(env, 3, hotkey=HOTKEY)
    receipt = {**RECEIPT, "document": {**RECEIPT["document"], "reason": "open_failed"}}
    asyncio.run(env.issuer.close(hotkey=HOTKEY, session_id=grant.session_id, reason="open_failed",
                                 transcript=None, open_refusal=receipt))
    assert env.book.get(grant.session_id).fault == "open_refused:open_failed"


def test_the_book_counts_who_holds_a_machine_from_an_index():
    book = SessionBook(SandboxPolicy())
    for n, hotkey in enumerate(("5A", "5A", "5B")):
        record = _record(f"s-{n}", sha="f" * 64, hotkey=hotkey, state=LIVE, seed=n, issued_at=NOW - 60,
                         expires_at=NOW + 600)
        record.machine_id = MACHINE
        book.add(record)
    assert book.held_on((MACHINE,), NOW) == {"5A": 2, "5B": 1}
    assert book.held_on(("other",), NOW) == {}
    assert book.held_on_precommit((MACHINE,), NOW, book.get("s-0").engagement) == 3
    assert book.held_on_precommit(("other",), NOW, book.get("s-0").engagement) == 0
    book.get("s-0").state = "closed"
    assert book.held_on((MACHINE,), NOW) == {"5A": 1, "5B": 1}
    book.prune(NOW + 3 * 86_400)                          # pruned records leave the machine index
    assert "s-0" not in book._by_machine.get(MACHINE, set())


@pytest.mark.parametrize("free, listed, load", [(4, 0, 0), (1, 0, 3), (4, 2, 2), (3, 3, 3)])
def test_a_machines_reported_load_is_its_used_slots_or_listed_sessions(tmp_path, free, listed, load):
    from tests.unit.sandbox_fixtures import ADDRESS, capacity_report
    from tests.unit.test_fleet_resources import RESOURCES
    from tests.unit.test_sandbox_fleet import make

    fleet, machine, served, clock = make(tmp_path)
    clock.now = NOW
    served[ADDRESS] = capacity_report(machine, at=NOW, free=free,
                                      resources={**RESOURCES, "sessions": [f"s-{n}" for n in range(listed)]})
    asyncio.run(fleet.poll_once())
    assert fleet.reported_load((MACHINE,)) == load
    assert fleet.reported_load(("other",)) == 0
