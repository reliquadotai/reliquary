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
from tests.unit.test_sandbox_fleet import PICK, make  # noqa: E402


class Clock:
    def __init__(self):
        self.now = 1_000.0

    def __call__(self):
        return self.now


# --- the snapshot ---------------------------------------------------------------------------------------------

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


# --- what feeds the monitor ------------------------------------------------------------------------------------

# --- the check slots ------------------------------------------------------------------------------------------

# --- an env that loses every machine; capacity verdicts per hotkey --------------------------------------------

class VerdictLedger:
    """The ledger's counts since start: verdicts and (neutral, decided) sessions per hotkey."""

    def __init__(self, counts):
        self.counts = counts                                      # hotkey -> [neutral, decided]

    def capacity_counts(self):
        return {hotkey: (1, neutral) for hotkey, (neutral, _decided) in self.counts.items()}

    def session_counts(self):
        return {hotkey: tuple(value) for hotkey, value in self.counts.items()}


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

# --- the wiring --------------------------------------------------------------------------------------------------

# --- per env and per image set: one saturated env alerts while the rest of the fleet idles -------------------

def mixed_fleet(free_x, free_y=300, lock_rows=None):
    return {"env-x": {"machines": 2, "slots": 16, "free": free_x, "image_sets": lock_rows or {}},
            "env-y": {"machines": 98, "slots": 784, "free": free_y, "image_sets": {}}}


# --- the per-env cap of a MAIN hotkey's allocation ---------------------------------------------------------------

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


