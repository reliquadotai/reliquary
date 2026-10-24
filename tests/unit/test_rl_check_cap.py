"""The episode intake's in-flight check cap follows the groups a hotkey can submit (yield quota)."""
import asyncio
from types import SimpleNamespace

import pytest

attest = pytest.importorskip("reliquary_sandbox.attest")

from reliquary.validator import episode_intake  # noqa: E402
from reliquary.validator.episode_intake import EpisodeGroupIntake  # noqa: E402
from tests.unit.test_episode_intake import CONTRACT, EPISODE, NOW, world  # noqa: E402
from tests.unit.test_rl_episode_wiring import _services  # noqa: E402


def intake(checks_allowed=None, **kwargs):
    sessions = SimpleNamespace(policy=SimpleNamespace(claim_ttl_s=10**6))
    return EpisodeGroupIntake(checkers={}, precommits=lambda sha: None, directory=lambda now: None,
                              token_verifier=None, sessions=sessions, seen=frozenset,
                              checks_allowed=checks_allowed, **kwargs)


def groups(n):
    """A hook: every hotkey can submit ``n`` groups."""
    return lambda hotkey, received: frozenset(f"p{k}" for k in range(n))


def per_hotkey(counts):
    return lambda hotkey, received: frozenset(f"{hotkey}-p{k}" for k in range(counts.get(hotkey, 0)))


def submittable(i, hotkey):
    return None if i._checks_allowed is None else len(i._checks_allowed(hotkey, NOW))


def started(i, hotkey, operator, n):
    sub = submittable(i, hotkey)
    return sum(i._rate_refusal(hotkey, i._owner(hotkey, operator), operator=operator, submittable=sub) is None
               for _ in range(n))


def checked_in_a_minute(i, hotkey, n):
    """Checks started (each ended at once, so only the per-minute budget can refuse) out of ``n``."""
    count = 0
    for _ in range(n):
        if i._rate_refusal(hotkey, i._owner(hotkey, None), submittable=submittable(i, hotkey)) is None:
            i._check_ended(i._owner(hotkey, None))
            count += 1
    return count


def test_one_operators_hotkeys_together_leave_a_slot_to_the_others():
    i = intake(groups(2))
    assert started(i, "5A", "cold", 5) + started(i, "5B", "cold", 5) == i._max_running - 1
    assert i._rate_refusal("5B", "5B", operator="cold", submittable=2) == ("episode_checks_in_flight",
                                                             {"operator_in_flight": 2, "max": 2})
    assert started(i, "5C", "warm", 1) == 1


def test_an_operators_slot_comes_back_when_any_of_its_checks_ends():
    i = intake(groups(2))
    assert started(i, "5A", "cold", 1) + started(i, "5B", "cold", 1) == 2
    i._check_ended("5A")
    assert i._op_in_flight == {"cold": 1} and i._op_of == {"5B": ["cold"]}
    assert started(i, "5C", "cold", 2) == 1
    i._check_ended("5B")
    i._check_ended("5C")
    assert i._op_in_flight == {} and i._op_of == {} and i._in_flight == {}


def test_without_the_hook_the_cap_is_the_per_operator_one():
    i = intake()
    assert i._owner("5A", "cold") == "cold"
    assert started(i, "5A", "cold", 1) + started(i, "5B", "cold", 5) == 2
    assert i._op_in_flight == {} and i._op_of == {}               # the operator count is the hook's only


def test_a_hotkey_checks_as_many_groups_at_once_as_it_can_submit():
    i = intake(per_hotkey({"5Big": 3}), max_checks_running=5)
    assert i._owner("5Big", "cold") == "5Big"                   # keyed by hotkey: the operator is not read
    assert started(i, "5Big", "cold", 10) == 3
    assert i._rate_refusal("5Big", "5Big", submittable=3)[0] == "episode_checks_in_flight"
    assert started(i, "5Junk", "cold", 5) == 0                  # nothing to submit: no check slot


def test_ten_hotkeys_get_the_check_slots_one_hotkey_with_their_groups_gets():
    one = intake(groups(10), max_checks_running=20)
    ten = intake(groups(1), max_checks_running=20)
    alone = started(one, "5One", None, 20)
    split = sum(started(ten, f"5S{n}", None, 2) for n in range(10))
    assert alone == split == 10


def test_one_check_slot_is_always_left_to_the_others():
    i = intake(groups(5))
    assert started(i, "5Big", None, 10) == i._max_running - 1 == episode_intake.MAX_CHECKS_RUNNING - 1
    assert i._rate_refusal("5Big", "5Big", submittable=5) == ("episode_checks_in_flight", {"in_flight": 2, "max": 2})
    assert started(i, "5Other", None, 1) == 1


def test_a_finished_check_frees_its_slot():
    i = intake(groups(1))
    assert started(i, "5A", None, 2) == 1
    i._check_ended("5A")
    assert started(i, "5A", None, 1) == 1


def test_the_per_minute_budget_scales_with_the_hotkeys_allocation():
    scale = {"5Big": 3.0}
    i = intake(groups(5), max_checks_per_minute=2, checks_scale=lambda hotkey: scale.get(hotkey, 1.0))
    assert checked_in_a_minute(i, "5Big", 20) == 6
    assert i._rate_refusal("5Big", "5Big", submittable=5) == ("episode_rate", {"per_minute": 6, "max": 6})
    assert checked_in_a_minute(i, "5Small", 20) == 2               # scale 1: the intake's own budget
    # The budget holds no slot: a big budget still holds at most max_running - 1 checks at once.
    j = intake(groups(5), checks_scale=lambda hotkey: 100.0)
    assert started(j, "5Big", None, 10) == j._max_running - 1


def test_without_a_scale_the_budget_is_the_same_for_every_hotkey():
    i = intake(groups(5), max_checks_per_minute=2)
    assert checked_in_a_minute(i, "5Big", 20) == 2


def test_an_intake_of_its_own_slots_checks_on_them_only(tmp_path, monkeypatch):
    import threading

    hook = {}
    w = world(tmp_path, intake_kwargs={"checks_allowed": lambda hotkey, received: hook["groups"],
                                       "max_checks_running": 2})
    hook["groups"] = {w.group.precommit.sha256}
    module_slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(episode_intake, "_CHECKS_RUNNING", module_slots)
    assert module_slots.acquire(blocking=False)                 # the module's slot is taken: not ours
    assert w.intake._max_running == 2
    assert w.intake._slots.acquire(blocking=False) and w.intake._slots.acquire(blocking=False)

    def admit():
        w.prepared.reject_reason = w.prepared.reject_stage = None
        w.prepared.episode_pending = True
        return asyncio.run(w.intake.admit(environment=EPISODE, prepared=w.prepared, received=NOW + 100,
                                          contract=CONTRACT))

    prepared, claim = admit()
    assert claim is None and prepared.reject_stage == "episode_checker_busy"
    assert w.intake._in_flight == {} and w.intake._op_in_flight == {}
    w.intake._slots.release()
    w.intake._slots.release()
    prepared, claim = admit()
    assert claim is not None and prepared.reject_reason is None
    assert w.intake._in_flight == {} and w.intake._op_in_flight == {} and w.intake._op_of == {}
    module_slots.release()


def test_an_intake_needs_two_check_slots_at_least():
    with pytest.raises(ValueError, match="max_checks_running"):
        intake(groups(1), max_checks_running=1)


def hooked(tmp_path, monkeypatch, **intake_kwargs):
    """An intake whose hook answers ``hook["groups"]`` (or raises ``hook["raise"]``), and its checker's calls."""
    hook = {}

    def allowed(hotkey, received):
        if "raise" in hook:
            raise hook["raise"]
        return hook["groups"]

    w = world(tmp_path, intake_kwargs={"checks_allowed": allowed, **intake_kwargs})
    hook["groups"] = {w.group.precommit.sha256}
    checker = w.intake._checkers[EPISODE]
    real, calls = checker.check, []

    def check(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(checker, "check", check)
    return w, hook, calls


def admit_again(w, operator=None):
    w.prepared.reject_reason = w.prepared.reject_stage = None
    w.prepared.episode_pending = True
    return asyncio.run(w.intake.admit(environment=EPISODE, prepared=w.prepared, received=NOW + 100,
                                      contract=CONTRACT, operator=operator))


def test_a_group_of_a_precommit_the_hotkey_cannot_submit_takes_no_check_slot(tmp_path, monkeypatch):
    w, hook, calls = hooked(tmp_path, monkeypatch, max_checks_per_minute=1)
    hook["groups"] = {"f" * 64}                          # another group is submittable, not this one
    prepared, claim = admit_again(w)
    assert claim is None and (prepared.reject_reason, prepared.reject_stage) == (
        episode_intake.RejectReason.RATE_LIMITED, "episode_checks_in_flight")
    assert "episode_checks_in_flight" in episode_intake.RETRYABLE_STAGES
    assert calls == [] and w.outcomes.calls == []       # no check ran, no verdict
    assert w.intake._in_flight == {} and w.intake._recent == {}     # no slot, no per-minute check spent
    hook["groups"] = {"f" * 64, w.group.precommit.sha256}
    prepared, claim = admit_again(w)
    assert claim is not None and calls == [1]


def test_an_operators_hotkeys_share_its_bound_through_admit(tmp_path, monkeypatch):
    w, hook, calls = hooked(tmp_path, monkeypatch)
    intake_of = w.intake
    hook["groups"] = {w.group.precommit.sha256, "a" * 64}
    assert started(intake_of, "5Other", "cold", 5) == 2                  # 2 = MAX_CHECKS_RUNNING - 1
    prepared, claim = admit_again(w, operator="cold")
    assert claim is None and prepared.reject_stage == "episode_checks_in_flight" and calls == []
    assert [call["stage"] for call in w.outcomes.calls] == ["episode_checks_in_flight"]     # the ledger: no verdict
    prepared, claim = admit_again(w, operator="warm")
    assert claim is not None and calls == [1]
    assert intake_of._op_in_flight == {"cold": 2} and intake_of._in_flight == {"5Other": 2}


def test_a_raising_hook_refuses_the_group_without_a_verdict_or_a_slot(tmp_path, monkeypatch):
    w, hook, calls = hooked(tmp_path, monkeypatch)
    hook["raise"] = RuntimeError("index broken")
    prepared, claim = admit_again(w)
    assert claim is None and (prepared.reject_reason, prepared.reject_stage) == (
        episode_intake.RejectReason.RATE_LIMITED, "episode_checks_in_flight")
    assert calls == [] and w.outcomes.calls == []
    assert w.intake._in_flight == {} and w.intake._op_in_flight == {} and w.intake._recent == {}
    del hook["raise"]
    assert admit_again(w)[1] is not None


def test_a_raising_budget_scale_keeps_the_base_budget(tmp_path, monkeypatch):
    def broken(hotkey):
        raise RuntimeError("index broken")

    w, _hook, calls = hooked(tmp_path, monkeypatch, checks_scale=broken, max_checks_per_minute=1)
    prepared, claim = admit_again(w)
    assert claim is not None and calls == [1]
    asyncio.run(w.intake.settle(claim, accepted=False))
    prepared, claim = admit_again(w)
    assert claim is None and prepared.reject_stage == "episode_rate"           # the base budget of 1
    assert w.intake._in_flight == {} and w.intake._op_in_flight == {}


def graded_group(services, hotkey, precommit, *, state, expires_at, closed_at):
    from tests.unit.test_rl_episode_wiring import _record

    m = services.quota._m_env("reliquary-swe")
    records = []
    for seed in range(m):
        record = _record(f"{hotkey}-{seed}", sha=precommit, hotkey=hotkey, state=state, seed=seed,
                         expires_at=expires_at)
        record.closed_status, record.closed_at = "graded", closed_at
        services.book.add(record)
        services.quota.opened(record)
        records.append(record)
    return records


def test_a_lapsed_group_counts_exactly_when_the_claim_would_take_it(tmp_path):
    import time

    from reliquary.sandbox.rl_quota import _LATE_GRADED_S
    from reliquary.sandbox.sessions import GRADING_GRACE_S, LAPSED

    services, _ = _services(tmp_path, yield_path=tmp_path / "rl-yield.sqlite3")
    now = int(time.time())
    deadline = now - 30                                # its sessions lapsed half a minute ago
    precommit = "c" * 64
    records = graded_group(services, "5Late", precommit, state=LAPSED, expires_at=deadline - GRADING_GRACE_S,
                           closed_at=deadline + 1)
    hook = services.intake._checks_allowed
    for received, claimable in ((deadline - 5, True), (deadline, True), (deadline + 1, False)):
        assert (precommit in hook("5Late", received)) is claimable
        refusal = services.issuer._claim_locked(records[0].session_id, "5Late", received, now)
        assert (refusal is None) is claimable
        if refusal is None:
            services.book.release_claim(records[0].session_id)
    # A group received by the deadline is admitted within its admission wall, while the index still keeps it.
    assert _LATE_GRADED_S >= episode_intake.ADMISSION_DEADLINE_S


def test_the_wiring_hands_the_quotas_submittable_groups_to_the_intake(tmp_path):
    import time

    from reliquary.sandbox.rl_capacity import MAX_CHECKS_RUNNING
    from reliquary.sandbox.sessions import CLOSED_GRADED

    services, _ = _services(tmp_path, yield_path=tmp_path / "rl-yield.sqlite3")
    now = int(time.time())
    assert services.intake._checks_allowed("5Nobody", now) == frozenset()
    graded_group(services, "5Ready", "d" * 64, state=CLOSED_GRADED, expires_at=now + 600, closed_at=now - 5)
    assert services.intake._checks_allowed("5Ready", now) == {"d" * 64}
    assert services.intake._checks_scale == services.quota.cap_scale
    assert services.intake._max_running == MAX_CHECKS_RUNNING and services.intake._slots is not None
    assert services.intake._owner("5A", "cold") == "5A"
    plain, _ = _services(tmp_path / "plain")
    assert plain.intake._checks_allowed is None and plain.intake._checks_scale is None
    assert plain.intake._slots is None and plain.intake._owner("5A", "cold") == "cold"
