"""Saturation metrics and the operator's ALERTs; the fleet size the quota shares is counted in sessions the
machines' slots, CPU and memory hold; new machines are picked up without a restart."""
import asyncio
from types import SimpleNamespace

import pytest

attest = pytest.importorskip("reliquary_sandbox.attest")

from reliquary.sandbox.rl_capacity import (  # noqa: E402
    CAPACITY_VERDICT_ALERT_SESSIONS,
    CHECKER_BUSY_ALERT_PER_MIN,
    SATURATION_ALERT_FOR_S,
    SATURATION_ALERT_REFUSALS_PER_MIN,
    SATURATION_ALERT_WAIT_S,
    SessionTiers,
    YieldStats,
    sessions_of,
)
from reliquary.sandbox.rl_metrics import SaturationMonitor  # noqa: E402
from reliquary.sandbox.rl_quota import YieldQuota  # noqa: E402
from tests.unit.sandbox_fixtures import (  # noqa: E402
    ADDRESS,
    BUDGETS,
    ENV,
    ENV_PACKAGE,
    GIB,
    IMAGE,
    MACHINE,
    NOW,
    capacity_report,
    machine_document,
    signer,
)
from tests.unit.test_rl_episode_wiring import _services  # noqa: E402
from tests.unit.test_sandbox_fleet import PICK, make  # noqa: E402


class Clock:
    def __init__(self):
        self.now = 1_000.0

    def __call__(self):
        return self.now


# --- the snapshot ---------------------------------------------------------------------------------------------

def test_the_snapshot_reports_waits_refusals_and_use_per_entity_and_env():
    clock = Clock()
    monitor = SaturationMonitor(clock=clock)
    for seconds in (1.0, 2.0, 3.0, 20.0):
        monitor.record_wait(seconds)
    monitor.record_refusal("sandbox_saturated", "reliquary_swe", "rl:1:a:0")
    monitor.record_refusal("quota_live", "reliquary_swe", "rl:1:a:1")
    for seconds in (0.2, 0.3, 0.4, 2.0):
        monitor.check_time(seconds)
    monitor.checker_busy("5Hot", "a" * 64)

    class Quota:
        def usage(self, now):
            return {"main": 10, "newcomer": 3, "low_yield": 1}

    class Fleet:
        def capacity_by_env(self, now=None, **kw):
            return {"swe": {"slots": 40, "free": 2, "image_sets": {"a" * 64: {"slots": 40, "free": 2}}}}

    snap = monitor.snapshot(quota=Quota(), fleet=Fleet())
    assert snap["queue"]["waits"] == 4 and snap["queue"]["wait_p50"] in (2.0, 3.0) and snap["queue"]["wait_p90"] == 20.0
    assert snap["refused"] == {"reliquary_swe": {"sandbox_saturated": 1, "quota_live": 1}}
    assert snap["used"] == {"main": 10, "newcomer": 3, "low_yield": 1}
    assert snap["fleet"]["swe"]["image_sets"]["a" * 64]["slots"] == 40
    assert snap["checks"]["count"] == 4 and snap["checks"]["p90_s"] == 2.0 and snap["checks"]["checker_busy"] == 1
    clock.now += 400                                                     # the window is five minutes
    assert monitor.snapshot()["refused"] == {} and monitor.snapshot()["checks"]["count"] == 0


# --- the grow-fleet ALERT -------------------------------------------------------------------------------------

def saturate(monitor, clock, minutes, *, free_fraction, distinct=True, reason="sandbox_saturated"):
    fired = []
    for tick in range(int(minutes * 6)):
        for n in range(SATURATION_ALERT_REFUSALS_PER_MIN // 6 + 1):
            engagement = f"rl:1:{tick}:{n}" if distinct else "rl:1:same:0"
            monitor.record_refusal(reason, "swe", engagement)
        clock.now += 10
        fired.append(monitor.check(free_fraction=free_fraction))
    return [text for text in fired if text]


def test_an_alert_fires_once_when_the_full_fleet_stays_saturated():
    clock = Clock()
    monitor = SaturationMonitor(clock=clock)
    alerts = saturate(monitor, clock, SATURATION_ALERT_FOR_S / 60 + 1, free_fraction=0.0)
    assert len(alerts) == 1 and "grow the sandbox fleet" in alerts[0]
    clock.now += 2 * SATURATION_ALERT_FOR_S
    assert monitor.check(free_fraction=0.0) is None and not monitor._alerted
    # re-armed: a new saturation alerts again
    assert len(saturate(monitor, clock, SATURATION_ALERT_FOR_S / 60 + 1, free_fraction=0.0)) == 1


def test_retries_of_one_seed_count_once():
    clock = Clock()
    monitor = SaturationMonitor(clock=clock)
    assert saturate(monitor, clock, 10, free_fraction=0.0, distinct=False) == []
    assert monitor.fleet_refusals(clock.now) == 1


@pytest.mark.parametrize("reason", ["quota_pool", "quota_live", "quota_opens", "quota_pool_opens",
                                    "task_unplaceable", "environment_unserved", "directory_unavailable"])
def test_only_fleet_capacity_refusals_ask_for_machines(reason):
    clock = Clock()
    monitor = SaturationMonitor(clock=clock)
    assert saturate(monitor, clock, 10, free_fraction=0.0, reason=reason) == []
    assert len(saturate(SaturationMonitor(clock=clock), clock, 10, free_fraction=0.0, reason="open_queue_full")) == 1


def test_a_fleet_with_free_sessions_raises_no_alert():
    clock = Clock()
    monitor = SaturationMonitor(clock=clock)
    for _ in range(100):
        monitor.record_wait(SATURATION_ALERT_WAIT_S * 10)
    assert saturate(monitor, clock, 10, free_fraction=0.30) == []
    assert saturate(monitor, clock, 10, free_fraction=None) == []          # no fleet known: nothing to grow


def test_long_waits_alone_alert_while_the_fleet_is_full():
    clock = Clock()
    monitor = SaturationMonitor(clock=clock)
    fired = []
    for _ in range(int(SATURATION_ALERT_FOR_S / 10) + 2):
        monitor.record_wait(SATURATION_ALERT_WAIT_S * 2)
        clock.now += 10
        fired.append(monitor.check(free_fraction=0.01))
    assert len([text for text in fired if text]) == 1


def test_a_short_burst_raises_no_alert():
    clock = Clock()
    monitor = SaturationMonitor(clock=clock)
    for n in range(100):
        monitor.record_refusal("sandbox_saturated", "swe", f"rl:1:x:{n}")
    clock.now += 10
    assert monitor.check(free_fraction=0.0) is None
    clock.now += SATURATION_ALERT_FOR_S
    assert monitor.check(free_fraction=0.0) is None


# --- what feeds the monitor ------------------------------------------------------------------------------------

def test_the_queue_tells_how_long_each_ticket_waited():
    from reliquary.sandbox.open_queue import OpenQueue

    clock, waits = Clock(), []
    q = OpenQueue(rank=lambda h: (h, 1.0, False), clock=clock, on_wait=lambda seconds, cls: waits.append(seconds))
    first, second = q.join("5A", cls="c"), q.join("5B", cls="c")
    clock.now += 4
    q.done(first, placed=True)
    clock.now += 3
    q.done(second)
    q.done(second)                                                  # idempotent: told once
    assert waits == [4.0, 7.0]


def test_the_issuer_tells_every_refused_open_with_its_env_and_engagement(tmp_path):
    from tests.unit.test_capacity_attribution import refused, saturating
    from tests.unit.test_rl_engagements import build

    env = build(tmp_path)
    saturating(env)
    seen = []
    env.issuer._events = SimpleNamespace(open_refused=lambda *args: seen.append(args))
    assert refused(env, 0).reason == "sandbox_saturated"
    env.current["window"] = 2                                       # the precommit's window is over: its terms refuse
    terms_refusal = refused(env, 1)
    assert seen[0] == ("sandbox_saturated", ENV, f"rl:1:{env.precommit.sha256}:0", IMAGE)
    assert seen[1] == (terms_refusal.reason, None, None, None)


# --- the check slots ------------------------------------------------------------------------------------------

def test_busy_check_slots_alert_with_the_slots_the_measured_check_time_needs():
    clock = Clock()
    monitor = SaturationMonitor(clock=clock)
    fired = []
    for tick in range(int(SATURATION_ALERT_FOR_S / 10) + 8):     # the first minute fills the rate
        for n in range(CHECKER_BUSY_ALERT_PER_MIN // 6 + 1):
            monitor.checker_busy("5Hot", f"{tick}-{n}")
            monitor.checker_busy("5Hot", f"{tick}-{n}")             # a retry of the same group counts once
        for _ in range(10):
            monitor.check_time(1.5)
        clock.now += 10
        fired.append(monitor.check_checker(running=3))
    alerts = [text for text in fired if text]
    assert len(alerts) == 1 and "MAX_CHECKS_RUNNING" in alerts[0] and "p90 1.50 s" in alerts[0]
    # rate = (60 checks + 12 busy groups) / 60 s, p90 1.5 s: ceil(1.2 x 1.5) + 1 = 3 slots at least
    assert "at least 3" in alerts[0]


def test_a_few_busy_refusals_raise_no_check_alert():
    clock = Clock()
    monitor = SaturationMonitor(clock=clock)
    for tick in range(100):
        for _ in range(CHECKER_BUSY_ALERT_PER_MIN):
            monitor.checker_busy("5Hot", "same")                    # one group retried: one a minute
        clock.now += 10
        assert monitor.check_checker(running=3) is None


def test_the_intake_reports_its_check_times_and_busy_refusals(tmp_path, monkeypatch):
    import threading

    from reliquary.validator import episode_intake
    from tests.unit.test_episode_intake import admit, world

    monitor = SaturationMonitor(clock=Clock())
    w = world(tmp_path, intake_kwargs={"check_metrics": monitor})
    assert admit(w)[1] is not None
    assert len(monitor._checks) == 1 and monitor._checks[0][1] >= 0
    monkeypatch.setattr(episode_intake, "_CHECKS_RUNNING", threading.BoundedSemaphore(1))
    assert episode_intake._CHECKS_RUNNING.acquire(blocking=False)          # another check runs
    w.prepared.reject_reason = w.prepared.reject_stage = None
    prepared, claim = admit(w)
    assert claim is None and prepared.reject_stage == "episode_checker_busy"
    assert [group for _at, group in monitor._busy] == [f"5Hot:{w.group.precommit.sha256}"]


# --- an env that loses every machine; capacity verdicts per hotkey --------------------------------------------

def test_an_env_losing_every_machine_is_alerted_again_after_a_recovery():
    monitor = SaturationMonitor(clock=Clock())
    assert monitor.check_envs({"swe": True, "math": True}) == ([], [])
    assert monitor.check_envs({"swe": False, "math": True}) == ([], [])          # one round: a late heartbeat
    alerts, recovered = monitor.check_envs({"swe": False, "math": True})       # two in a row: lost
    assert len(alerts) == 1 and "swe lost every sandbox machine" in alerts[0] and recovered == []
    assert monitor.check_envs({"swe": False, "math": True}) == ([], [])          # once while it lasts
    assert monitor.check_envs({"swe": True, "math": True}) == ([], ["swe"])
    monitor.check_envs({"swe": False, "math": True})
    alerts, _ = monitor.check_envs({"swe": False, "math": True})
    assert len(alerts) == 1                                                     # a new loss: a new ALERT
    fresh = SaturationMonitor(clock=Clock())                                    # the boot already alerted it
    assert fresh.check_envs({"swe": False}, known_lost={"swe"}) == ([], [])


def test_one_late_heartbeat_of_a_one_machine_env_is_no_loss():
    monitor = SaturationMonitor(clock=Clock())
    for _ in range(10):
        assert monitor.check_envs({"swe": False}) == ([], [])
        assert monitor.check_envs({"swe": True}) == ([], [])


class VerdictLedger:
    """The ledger's counts since start: verdicts and (neutral, decided) sessions per hotkey."""

    def __init__(self, counts):
        self.counts = counts                                      # hotkey -> [neutral, decided]

    def capacity_counts(self):
        return {hotkey: (1, neutral) for hotkey, (neutral, _decided) in self.counts.items()}

    def session_counts(self):
        return {hotkey: tuple(value) for hotkey, value in self.counts.items()}


def test_a_hotkey_living_on_capacity_verdicts_is_alerted_once_until_it_falls_back():
    clock = Clock()
    monitor = SaturationMonitor(clock=clock)
    n = CAPACITY_VERDICT_ALERT_SESSIONS
    ledger = VerdictLedger({"5Many": [n, n], "5Few": [3, 0], "5Big": [n, 20 * n]})
    alerts = monitor.check_capacity_verdicts(ledger)
    assert len(alerts) == 1 and "5Many" in alerts[0]                # 50 %; 5Few too few; 5Big under 25 %
    assert monitor.check_capacity_verdicts(ledger) == []
    ledger.counts["5Many"][1] += 100 * n
    assert monitor.check_capacity_verdicts(ledger) == []            # its share fell: re-armed
    ledger.counts["5Many"][0] += 40 * n
    assert len(monitor.check_capacity_verdicts(ledger)) == 1
    rows = monitor.snapshot(ledger=ledger)["capacity_verdicts"]
    assert [row["hotkey"] for row in rows] == ["5Many", "5Big", "5Few"]


def test_an_honest_key_with_rare_verdicts_over_days_is_never_alerted():
    """Both counts over the same 24 h: occasional verdicts never add up against a capped decided count (since start,
    5 days give 240 neutral sessions, against the last 200 decided: 55 %)."""
    from reliquary.sandbox.rl_capacity import CAPACITY_VERDICT_SAMPLE_S

    clock = Clock()
    monitor = SaturationMonitor(clock=clock)
    ledger = VerdictLedger({"5Honest": [0, 0]})
    for _hour in range(24 * 5):                                     # five days: 2 neutral, 40 decided an hour
        ledger.counts["5Honest"][0] += 2
        ledger.counts["5Honest"][1] += 40
        for _ in range(3_600 // CAPACITY_VERDICT_SAMPLE_S):
            clock.now += CAPACITY_VERDICT_SAMPLE_S
            assert monitor.check_capacity_verdicts(ledger) == []
    (_hotkey, _verdicts, neutral, decided, share), = monitor._rows(ledger)
    assert 48 <= neutral <= 52 and decided >= 900 and share < 0.06


def test_a_burst_of_verdicts_within_a_day_is_alerted_and_forgotten_after_it():
    from reliquary.sandbox.rl_capacity import CAPACITY_VERDICT_SAMPLE_S, USEFUL_HORIZON_S

    clock = Clock()
    monitor = SaturationMonitor(clock=clock)
    ledger = VerdictLedger({"5K": [0, 0]})
    monitor.check_capacity_verdicts(ledger)
    ledger.counts["5K"] = [2 * CAPACITY_VERDICT_ALERT_SESSIONS, 10]
    assert len(monitor.check_capacity_verdicts(ledger)) == 1
    for _ in range(int(USEFUL_HORIZON_S / CAPACITY_VERDICT_SAMPLE_S) + 2):
        clock.now += CAPACITY_VERDICT_SAMPLE_S
        monitor.check_capacity_verdicts(ledger)
    assert monitor._rows(ledger) == [] and monitor._verdict_alerted == set()        # out of the window: re-armed


# --- the fleet size, in sessions the resources hold (never the bare slots) ------------------------------------

BOX = {"cpu_millis": 8_000, "memory_bytes": 64 * GIB, "used_cpu_millis": 0, "used_memory_bytes": 0,
       "free_cpu_millis": 8_000, "free_memory_bytes": 64 * GIB, "max_box_cpu_millis": 4_000,
       "boxes_per_episode": 1, "cpu_overcommit_permille": 1_000, "present_window_s": 120, "sessions": []}


def resource_fleet(tmp_path, *, capacity=8, free=8, required=True, **report):
    tmp_path.mkdir(parents=True, exist_ok=True)
    fleet, machine, served, _clock = make(tmp_path)
    fleet.resources_required = required
    report.setdefault("resources", BOX)
    served[ADDRESS] = capacity_report(machine, at=NOW, capacity=capacity, free=free, **report)
    asyncio.run(fleet.poll_once())
    return fleet


def tiers_for(cpu_millis=None, memory_bytes=4 * GIB, resolved=True):
    policy = SimpleNamespace(sandbox_env=ENV, env_package=ENV_PACKAGE,
                             budgets_dict=lambda: {"memory_bytes": memory_bytes})
    tiers = SessionTiers({"swe": policy})
    if resolved:
        tiers.observe("swe", SimpleNamespace(cpu_millis=cpu_millis, limits={}))
    return tiers


class Ledger:
    version = 1

    def snapshot(self):
        return {"5Main": YieldStats(useful=200, decided=200, useful_24h=200, useful_s_24h=86_400.0 * 50)}


def main_allocation(capacity):
    clock = Clock()
    quota = YieldQuota(ledger=Ledger(), pool_seeds=2, group_size=1, capacity=capacity, clock=clock)
    for seed in range(20):                                         # a MAIN key asking for far more than the fleet
        quota.refusal("5Main", int(clock()), precommit_sha256=f"{seed:064x}", seed_index=0, environment="swe")
    clock.now += 1.0                                               # past the allocation's refresh
    return quota.allocation()


def test_an_8_slot_machine_with_4_cpu_boxes_on_8_cpus_counts_2_sessions_and_the_quota_never_allocates_more(tmp_path):
    fleet = resource_fleet(tmp_path)
    tiers = tiers_for(cpu_millis=None)                             # tasks declaring no CPUs take the whole box
    reservations = tiers.reservations()
    assert fleet.session_capacity() == 8                           # the bare slots (what the quota used to share)
    assert fleet.session_capacity(reservations=reservations) == 2
    alloc = main_allocation(lambda: fleet.session_capacity(reservations=reservations))
    assert sessions_of(alloc.main["5Main"]) + sessions_of(alloc.newcomer) + sessions_of(alloc.low_yield) <= 2
    assert sessions_of(alloc.main["5Main"]) == 2
    # ... and placement hosts exactly that many
    reserve = reservations[(ENV, ENV_PACKAGE)]
    placed = []
    for n in range(3):
        placement = fleet.pick(now=NOW, reserve=reserve, **PICK)
        if placement is not None:
            fleet.note_issued(placement.machine_id, NOW, reserve=reserve, hold_s=120, session_id=f"s-{n}")
            placed.append(placement)
    assert len(placed) == 2
    assert fleet.session_room(reservations=reservations) == (2, 0)


@pytest.mark.parametrize("cpu_millis, memory_gib, sessions", [
    (4_000, 4, 2), (1_000, 4, 8), (500, 4, 8), (1_000, 16, 4), (8_000, 4, 2)])
def test_the_fleet_size_is_the_binding_of_slots_cpu_and_memory(tmp_path, cpu_millis, memory_gib, sessions):
    fleet = resource_fleet(tmp_path)
    tiers = tiers_for(cpu_millis=cpu_millis, memory_bytes=memory_gib * GIB)   # 8 000: above the box, counted at it
    assert fleet.session_capacity(reservations=tiers.reservations()) == sessions


def test_the_tier_is_the_largest_task_seen_and_a_whole_box_before_any(tmp_path):
    fleet = resource_fleet(tmp_path)
    tiers = tiers_for(resolved=False)
    assert fleet.session_capacity(reservations=tiers.reservations()) == 2          # nothing resolved: a box each
    tiers.observe("swe", SimpleNamespace(cpu_millis=1_000, limits={}))
    assert fleet.session_capacity(reservations=tiers.reservations()) == 8
    tiers.observe("swe", SimpleNamespace(cpu_millis=2_000, limits={}))
    tiers.observe("swe", SimpleNamespace(cpu_millis=1_000, limits={}))             # never shrinks back
    assert fleet.session_capacity(reservations=tiers.reservations()) == 4
    tiers.observe("swe", SimpleNamespace(cpu_millis=1_000, limits={"memory_bytes": 32 * GIB}))
    assert fleet.session_capacity(reservations=tiers.reservations()) == 2          # a task raising its memory
    tiers.observe("swe", SimpleNamespace(cpu_millis=None, limits={}))
    assert tiers.reservation("swe")["cpu_millis"] is None                          # a box task: the box from now on


def test_a_machine_counts_only_for_the_envs_it_serves_with_their_images(tmp_path):
    reservations = tiers_for(cpu_millis=1_000).reservations()
    other = resource_fleet(tmp_path / "a", env_packages={ENV: "reliquary-swe==9.9"})            # another package
    assert other.session_capacity(reservations=reservations) == 0
    imageless = resource_fleet(tmp_path / "b", env_image_locks={ENV: {"sha256": "b" * 64, "images": 3,
                                                                      "present": 0}})
    assert imageless.session_capacity(reservations=reservations) == 0
    assert imageless.capacity_by_env(reservations=reservations)[ENV]["machines"] == 0
    unread = resource_fleet(tmp_path / "c", resources={**BOX, "present_window_s": 600})       # invalid resources
    assert unread.session_capacity(reservations=reservations) == 0


def test_a_machine_serving_two_envs_counts_at_the_larger_reservation(tmp_path):
    fleet = resource_fleet(tmp_path, env_packages={ENV: ENV_PACKAGE, "reliquary-math": "math==1"})
    small = tiers_for(cpu_millis=1_000).reservations()
    both = {**small, ("reliquary-math", "math==1"): {"cpu_millis": 4_000, "memory_bytes": 4 * GIB}}
    assert fleet.session_capacity(reservations=small) == 8
    assert fleet.session_capacity(reservations=both) == 2
    view = fleet.capacity_by_env(reservations=both)
    assert view[ENV]["slots"] == 8 and view["reliquary-math"]["slots"] == 2


def test_without_required_resources_a_machine_reporting_none_counts_its_slots(tmp_path):
    fleet = resource_fleet(tmp_path, required=False, resources=None, capacity=6, free=6)
    assert fleet.session_capacity(reservations=tiers_for(cpu_millis=4_000).reservations()) == 6


def test_the_operators_view_per_env_and_image_set(tmp_path):
    lock = {ENV: {"sha256": "c" * 64, "images": 3, "present": 3}}
    fleet = resource_fleet(tmp_path, free=5, env_image_locks=lock,
                           resources={**BOX, "free_cpu_millis": 3_000, "used_cpu_millis": 5_000})
    view = fleet.capacity_by_env(reservations=tiers_for(cpu_millis=1_000).reservations())
    assert view[ENV]["slots"] == 8 and view[ENV]["free"] == 3 and view[ENV]["machines"] == 1
    assert view[ENV]["image_sets"] == {"c" * 64: {"machines": 1, "slots": 8, "free": 3}}
    plain = fleet.capacity_by_env()                                    # slots only, every env reported
    assert plain[ENV]["slots"] == 8 and plain[ENV]["free"] == 5


# --- box sizes in the saturation attribution ------------------------------------------------------------------

def test_one_4_cpu_session_weighs_four_1_cpu_ones_in_the_attribution(tmp_path):
    from tests.unit.test_capacity_attribution import others_hold, refused, saturating
    from tests.unit.test_rl_engagements import HOTKEY, build

    def scenario(path, weighted):
        env = build(path)
        _q, listener = saturating(env)
        others_hold(env, 1, hotkey=HOTKEY)                # its OTHER precommit: one 4-CPU session (half the machine)
        others_hold(env, 3)                               # others: three 1-CPU sessions (3/8 of it)
        if weighted:
            env.issuer._fleet.session_share = lambda machine, reserve: 1 / 8      # a slot (1 CPU) of 8 CPUs
            env.issuer._fleet.load_share = lambda machines: 7 / 8 if MACHINE in machines else 0.0
            env.issuer._weights[f"o-{HOTKEY}-{MACHINE}-0"] = 4 / 8
        assert refused(env, 0).reason == "sandbox_saturated"
        return listener.seen

    assert scenario(tmp_path / "count", weighted=False) == ["sandbox_saturated"]   # 1 of 4 sessions: others'
    assert scenario(tmp_path / "share", weighted=True) == []                        # half the machine: its own


def test_the_fleet_measures_a_sessions_share_and_the_machines_load(tmp_path):
    fleet = resource_fleet(tmp_path, free=6, resources={**BOX, "free_cpu_millis": 2_000, "used_cpu_millis": 6_000})
    assert fleet.session_share(MACHINE, {"cpu_millis": 4_000, "memory_bytes": 4 * GIB}) == 0.5
    assert fleet.session_share(MACHINE, {"cpu_millis": 500, "memory_bytes": 4 * GIB}) == 1 / 8     # a slot at least
    assert fleet.session_share(MACHINE, None) == 1 / 8
    assert fleet.load_share((MACHINE,)) == 0.75                     # CPU: 6 of 8 (slots: 2 of 8)
    assert fleet.session_share("other", None) is None and fleet.load_share(("other",)) == 0.0


def test_the_issuer_weighs_each_session_it_places_and_forgets_it_when_it_settles(tmp_path):
    from tests.unit.test_rl_engagements import build, close, open_

    env = build(tmp_path)
    env.issuer._fleet.session_share = lambda machine, reserve: 0.25
    env.issuer._fleet.load_share = lambda machines: 0.0
    grant = open_(env, 0)                                  # RL terms carry the session's reservation
    assert env.issuer._weights == {grant.session_id: 0.25}
    close(env, grant, reason="open_failed")
    assert env.issuer._weights == {}


# --- new machines picked up live -------------------------------------------------------------------------------

def test_a_machine_added_to_the_fleet_is_used_without_a_restart(tmp_path):
    one = signer(tmp_path, "m", "k1")
    two = signer(tmp_path, "m2", "k2")
    documents = [machine_document(one)]
    fleet, _machine, served, _clock = make(tmp_path, documents=documents)
    fleet.resources_required = True
    served[ADDRESS] = capacity_report(one, at=NOW, capacity=8, free=8, resources=BOX)
    asyncio.run(fleet.poll_once())
    reservations = tiers_for(cpu_millis=1_000).reservations()
    quota_clock = Clock()
    quota = YieldQuota(ledger=Ledger(), pool_seeds=2, group_size=1, clock=quota_clock,
                       capacity=lambda: fleet.session_capacity(reservations=reservations))
    before = quota.allocation().capacity
    assert before == 8
    documents.append(machine_document(two, machine_id="m-2", address="http://m2:1"))
    served["http://m2:1"] = capacity_report(two, at=NOW, machine_id="m-2", address="http://m2:1", capacity=8,
                                            free=8, resources={**BOX, "cpu_millis": 4_000, "free_cpu_millis": 4_000})
    asyncio.run(fleet.refresh_directory())
    asyncio.run(fleet.poll_once())
    assert fleet.session_capacity(reservations=reservations) == 12
    assert quota.allocation().capacity == 8                         # cached for a second at most
    quota_clock.now += 1.0
    assert quota.allocation().capacity == 12                        # then the new machine counts, no restart
    assert fleet.pick(now=NOW, image=IMAGE, env=ENV, env_package=ENV_PACKAGE, budgets=BUDGETS, validity_s=4500,
                      reserve=reservations[(ENV, ENV_PACKAGE)], exclude={MACHINE}).machine_id == "m-2"


# --- the wiring --------------------------------------------------------------------------------------------------

def test_the_wiring_builds_the_monitor_and_serves_the_cached_snapshot_behind_a_token(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from reliquary.validator import rl_sandbox_wiring, server

    assert rl_sandbox_wiring.OBSERVATIONS_TOKEN_ENV == server.SERVICE_OBSERVATIONS_TOKEN_ENV
    monkeypatch.delenv(rl_sandbox_wiring.RL_SATURATION_TOKEN_ENV, raising=False)
    monkeypatch.delenv(server.SERVICE_OBSERVATIONS_TOKEN_ENV, raising=False)
    services, _ = _services(tmp_path, yield_path=tmp_path / "rl-yield.sqlite3")
    assert isinstance(services.saturation, SaturationMonitor)
    assert services.queue._on_wait == services.saturation.record_wait
    assert services.saturation in services.issuer._events.listeners
    assert services.intake._check_metrics is services.saturation
    app = FastAPI()
    for router in services.routers:
        app.include_router(router)
    client = TestClient(app)
    calls = []
    real = services.saturation.snapshot
    services.saturation.snapshot = lambda **kw: calls.append(1) or real(**kw)
    assert client.get("/rl/saturation").status_code == 404                  # no token configured: nothing shown
    monkeypatch.setenv(server.SERVICE_OBSERVATIONS_TOKEN_ENV, "obs-token")
    assert client.get("/rl/saturation").status_code == 401
    assert client.get("/rl/saturation", headers={"Authorization": "Bearer wrong"}).status_code == 401
    good = {"Authorization": "Bearer obs-token"}
    assert client.get("/rl/saturation", headers=good).status_code == 503     # before the first background round
    services.check_saturation()
    services.saturation_cache = services.saturation_snapshot()               # what saturation_forever does
    computed = len(calls)
    for _ in range(5):
        body = client.get("/rl/saturation", headers=good).json()
    assert len(calls) == computed                                            # requests compute nothing
    assert set(body) >= {"queue", "refused", "used", "fleet", "checks", "envs_lost", "capacity_verdicts",
                         "attribution_failures", "reservations", "saturated"}
    assert body["used"] == {"main": 0, "newcomer": 0, "low_yield": 0, "capacity": 0}
    name = services.environments[0]
    assert body["reservations"][name]["seeded"] is False
    monkeypatch.setenv(rl_sandbox_wiring.RL_SATURATION_TOKEN_ENV, "own-token")   # its own token wins
    assert client.get("/rl/saturation", headers=good).status_code == 401
    assert client.get("/rl/saturation", headers={"Authorization": "Bearer own-token"}).status_code == 200
    jobs = services.background()
    assert any(job.__qualname__.endswith("saturation_forever") for job in jobs)
    for job in jobs:
        job.close()
    plain, _ = _services(tmp_path / "plain")
    assert plain.saturation is None and len(plain.routers) == 2


def test_the_background_round_refreshes_the_cached_snapshot(tmp_path):
    services, _ = _services(tmp_path, yield_path=tmp_path / "rl-yield.sqlite3")

    async def one_round():
        job = asyncio.ensure_future(services.saturation_forever(3600))
        await asyncio.sleep(0.05)
        job.cancel()
        await asyncio.gather(job, return_exceptions=True)
    asyncio.run(one_round())
    assert services.cached_saturation() is not None and "at" in services.cached_saturation()


def test_every_task_the_views_resolve_is_observed(tmp_path):
    services, _ = _services(tmp_path, yield_path=tmp_path / "rl-yield.sqlite3")
    name = services.environments[0]
    assert all(reserve["cpu_millis"] is None for reserve in services.tiers.reservations().values())
    seen = []
    services.tiers.observe = lambda env, task: seen.append((env, task))
    task = asyncio.run(services.views[name].resolve_task(0))
    assert seen == [(name, task)]
    assert services.resolvers[name] is services.views[name].resolve_task       # the boot's options probe too
    assert services.tiers._policies is services.policies                        # an env added later counts


def test_the_tiers_are_seeded_at_start_from_the_envs_own_tasks(tmp_path):
    from reliquary.sandbox.tasks import ResolvedTask

    services, _ = _services(tmp_path, yield_path=tmp_path / "rl-yield.sqlite3")
    name = services.environments[0]
    asked = []

    async def resolve(index):
        asked.append(index)
        return ResolvedTask(IMAGE, {}, cpu_millis=4_000 if index % 100 == 0 else 1_000)   # 1 % big tasks
    services.raw_resolvers[name] = resolve
    services.task_counts[name] = 10_000
    asyncio.run(services.seed_tiers())
    assert len(asked) == 512 and min(asked) == 0 and max(asked) > 9_000             # spread over the range
    assert services.tiers.reservation(name)["cpu_millis"] == 1_000                  # the 95th percentile
    services.tiers.observe(name, ResolvedTask(IMAGE, {}, cpu_millis=4_000))        # a rare big precommit
    services.tiers.observe(name, ResolvedTask(IMAGE, {}, cpu_millis=None))
    assert services.tiers.reservation(name)["cpu_millis"] == 1_000                  # never moves it
    view = services.tiers.view()[name]
    assert view["seeded"] and view["tasks"] == 512 and view["above"] == 2


def test_an_env_whose_tasks_cannot_be_counted_stays_learnt(tmp_path):
    services, _ = _services(tmp_path, yield_path=tmp_path / "rl-yield.sqlite3")
    name = services.environments[0]
    services.task_counts[name] = None
    asyncio.run(services.seed_tiers())
    assert not services.tiers.seeded(name)


def test_the_saturation_round_alerts_an_env_that_lost_its_machines_and_rearms_the_open_path(tmp_path):
    services, _ = _services(tmp_path, yield_path=tmp_path / "rl-yield.sqlite3")
    asyncio.run(services.fleet.refresh_directory())
    name = services.environments[0]
    serving = {"now": False}
    services.fleet.serving = lambda *a, **kw: ["m"] if serving["now"] else []
    services.issuer._unserved_alerted.add((services.served[name][0], "pkg", None))
    services.check_saturation()
    alerts = services.check_saturation()
    assert any("lost every sandbox machine" in text for text in alerts)
    serving["now"] = True
    assert services.check_saturation() == []
    assert services.issuer._unserved_alerted == set()      # the open path's ALERT fires again on a new loss
    serving["now"] = False
    services.check_saturation()
    assert any("lost every sandbox machine" in text for text in services.check_saturation())


def test_a_stale_directory_alerts_no_env(tmp_path):
    services, _ = _services(tmp_path, yield_path=tmp_path / "rl-yield.sqlite3")       # the directory never read
    assert not services.fleet.directory_ready()
    for _ in range(3):
        assert not any("lost every sandbox machine" in text for text in services.check_saturation())


# --- per env and per image set: one saturated env alerts while the rest of the fleet idles -------------------

def mixed_fleet(free_x, free_y=300, lock_rows=None):
    return {"env-x": {"machines": 2, "slots": 16, "free": free_x, "image_sets": lock_rows or {}},
            "env-y": {"machines": 98, "slots": 784, "free": free_y, "image_sets": {}}}


def test_a_saturated_env_alerts_while_the_rest_of_the_fleet_idles():
    clock = Clock()
    monitor = SaturationMonitor(clock=clock)
    fired, fleet_wide = [], []
    for tick in range(int(SATURATION_ALERT_FOR_S / 10) + 8):       # the first minute fills the rate
        for n in range(SATURATION_ALERT_REFUSALS_PER_MIN // 6 + 1):
            monitor.record_refusal("sandbox_saturated", "env-x", f"rl:1:{tick}:{n}", IMAGE)
        clock.now += 10
        fired += monitor.check_saturated(mixed_fleet(free_x=0))
        fleet_wide.append(monitor.check(free_fraction=300 / 800))          # the fleet as a whole is 37 % free
    assert len(fired) == 1 and "env env-x saturated" in fired[0] and not any(fleet_wide)
    assert monitor.snapshot()["saturated"] == ["env-x"]
    clock.now += 120
    assert monitor.check_saturated(mixed_fleet(free_x=0)) == [] and monitor.snapshot()["saturated"] == []


def test_waits_of_one_env_alert_for_that_env_only():
    clock = Clock()
    monitor = SaturationMonitor(clock=clock)
    fired = []
    for _ in range(int(SATURATION_ALERT_FOR_S / 10) + 2):
        monitor.record_wait(SATURATION_ALERT_WAIT_S * 2, ("env-x", 1024, 1 << 32))
        clock.now += 10
        fired += monitor.check_saturated(mixed_fleet(free_x=0, free_y=0))
    assert len(fired) == 1 and "env-x" in fired[0]


def test_a_saturated_image_set_alerts_by_the_refusals_of_its_images():
    clock = Clock()
    monitor = SaturationMonitor(clock=clock)
    locks = {"a" * 64: {"machines": 1, "slots": 8, "free": 0}, "b" * 64: {"machines": 1, "slots": 8, "free": 8}}
    images = {"a" * 64: frozenset({"img-a"}), "b" * 64: frozenset({"img-b"})}
    fired = []
    for tick in range(int(SATURATION_ALERT_FOR_S / 10) + 8):       # the first minute fills the rate
        for n in range(SATURATION_ALERT_REFUSALS_PER_MIN // 6 + 1):
            monitor.record_refusal("sandbox_saturated", "env-x", f"rl:1:{tick}:{n}", "img-a")
        clock.now += 10
        fired += monitor.check_saturated(mixed_fleet(free_x=8, lock_rows=locks),       # the env is half free
                                         images_of=lambda env, lock: images[lock])
    assert len(fired) == 1 and f"image set {'a' * 12} of env env-x" in fired[0]


# --- the per-env cap of a MAIN hotkey's allocation ---------------------------------------------------------------

def test_a_main_keys_groups_on_a_scarce_env_stay_within_its_share_of_that_env():
    from reliquary.sandbox.routes import REFUSAL_STATUS
    from reliquary.sandbox.rl_yield import CAPACITY_REASONS
    from reliquary.sandbox.sessions import CAPACITY_REFUSALS, LIVE
    from tests.unit.test_rl_episode_wiring import _record

    clock = Clock()
    clock.now = float(NOW)
    caps = {"env-x": 16, "env-y": 784}
    quota = YieldQuota(ledger=Ledger(), pool_seeds=2, group_size=1, capacity=lambda: 800, clock=clock,
                       env_capacities=lambda: dict(caps), env_names={"x": "env-x", "env-x": "env-x",
                                                                     "y": "env-y", "env-y": "env-y"})
    for seed in range(400):                                          # it asks for 400 sessions: allocated 400
        quota.refusal("5Main", NOW, precommit_sha256=f"{seed:064x}", seed_index=0, environment="x")
    for n in range(8):                                               # 8 groups (M = 1) already on env-x
        record = _record(f"s-{n}", sha=f"{n:064x}", hotkey="5Main", state=LIVE, issued_at=NOW, expires_at=NOW + 600)
        record.env = "env-x"
        quota.opened(record)
    clock.now += 1.0
    assert sessions_of(quota.allocation().main["5Main"]) == 400
    refused = quota.refusal("5Main", NOW + 1, precommit_sha256="e" * 64, seed_index=0, environment="x")
    assert refused.reason == "quota_env" and refused.detail["max"] == 8 and refused.retry_after   # ceil(400 x 16/800)
    assert quota.refusal("5Main", NOW + 1, precommit_sha256="f" * 64, seed_index=0, environment="y") is None
    # a seed of a started group is never cut
    assert quota.refusal("5Main", NOW + 1, precommit_sha256=f"{0:064x}", seed_index=1, environment="x") is None
    # retryable, never a capacity verdict, never counted by the grow-fleet ALERT
    assert REFUSAL_STATUS["quota_env"] == 429
    assert "quota_env" not in CAPACITY_REFUSALS and "quota_env" not in CAPACITY_REASONS
    from reliquary.sandbox.rl_metrics import FLEET_REASONS
    assert "quota_env" not in FLEET_REASONS
    caps["env-x"] = 800                                              # a homogeneous fleet: no per-env cap
    clock.now += 1.0
    assert quota.refusal("5Main", NOW + 2, precommit_sha256="e" * 64, seed_index=0, environment="x") is None


# --- the corpus attribution and the tiers' keys ---------------------------------------------------------------

def test_a_session_without_a_reservation_is_attributed_by_counts_as_before(tmp_path):
    from tests.unit.test_rl_engagements import build

    env = build(tmp_path)
    env.issuer._fleet.servable_machines = lambda **kw: ("m",)
    env.issuer._fleet.session_share = lambda machine, reserve: 1.0
    env.issuer._fleet.load_share = lambda machines: 0.0
    seen = []
    real = env.issuer._own_load
    env.issuer._own_load = lambda *a, **kw: seen.append(kw.get("by_share")) or real(*a, **kw)
    from reliquary.sandbox.sessions import Refusal

    terms = SimpleNamespace(image=IMAGE, env=ENV, env_package=ENV_PACKAGE, budgets=BUDGETS, env_options_sha256=None,
                            engagement="corpus:j:0", reserve=None, prepare_timeout_s=None)
    env.issuer._self_caused("5Hot", terms, Refusal("sandbox_saturated", {}))
    assert seen == [False]
    assert real("5Hot", ("m",), NOW, "corpus:j:0") == (0, 0)        # session counts: integers, the 13b rule


def test_two_envs_on_one_sandbox_env_at_two_packages_keep_both_reservations():
    def policy(package, memory):
        return SimpleNamespace(sandbox_env=ENV, env_package=package, budgets_dict=lambda: {"memory_bytes": memory})
    tiers = SessionTiers({"a": policy("pkg==1", 2 * GIB), "b": policy("pkg==2", 4 * GIB)})
    assert set(tiers.reservations()) == {(ENV, "pkg==1"), (ENV, "pkg==2")}
    assert tiers.reservations()[(ENV, "pkg==2")]["memory_bytes"] == 4 * GIB


def test_the_start_seeds_the_tiers_before_serving(tmp_path):
    services, _ = _services(tmp_path, yield_path=tmp_path / "rl-yield.sqlite3")
    name = services.environments[0]
    asyncio.run(services.start())
    try:
        assert services.tiers.seeded(name) and services.tiers.view()[name]["tasks"] == 512     # of its 1 000 tasks
    finally:
        services.ledger.close()
