# tests/unit/test_service_runtime_v2.py
"""ServiceRuntime v2: run log + exploration ledger + settlement behind one frozen window envelope."""
import json
import sqlite3
from pathlib import Path

import pytest

from reliquary.constants import M_ROLLOUTS
from reliquary.protocol.seed_pool import pool_from_service_policy
from reliquary.protocol.service_contract import ServiceContract
from reliquary.protocol.service_schedule import next_schedule
from reliquary.protocol.submission import ServicePolicyAnnouncement
from reliquary.services import runtime as runtime_module
from reliquary.services.runtime import (
    FROZEN_ARCHIVE_FIELDS, AuditOutcome, ServicePolicyLimit, ServiceRuntime, protocol_slot_geometry, validate_service_archive,
)
from reliquary.services.settlement import SettlementError, validate_service_archive_v2
from tests.unit.service_v2_fixtures import CODE, MATH, contract_v2, contract_v2_dict, qualification_v2

PICKS, SLOTS = protocol_slot_geometry()
T = PICKS * SLOTS                       # training slots of one env in one window
POOL = 0.25
PRICE = 0.15 * POOL / T                 # nominal exploration entitlement
CAP_COUNT = int(0.10 * T / 0.15 + 1e-9)  # whole entitlements under the 10 % cap
ZERO = [0.0] * M_ROLLOUTS
ONES = [1.0] * M_ROLLOUTS
HALF = [1.0] * (M_ROLLOUTS // 2) + [0.0] * (M_ROLLOUTS - M_ROLLOUTS // 2)
BEACON = "cd" * 32
WINDOW_BEACON = "ab" * 32


def drand_round(instant: float) -> int:
    """Test drand clock: one round every 3 s."""
    return int(instant) // 3


def build(path, contract=None, **kw):
    contract = contract or contract_v2(**kw)
    return ServiceRuntime(path, contract, qualification_v2(contract), now=0, drand_round_at=drand_round)


def open_window(rt, window, pools=None, now=0.0):
    envelope = rt.open_window(window, pools=pools or {MATH: POOL, CODE: POOL}, picks_target=PICKS, batch_slots=SLOTS,
                              now=now)
    rt.announcement(window=window, randomness=WINDOW_BEACON)
    return envelope


def runtime(tmp_path, contract=None, **kw):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rt = build(tmp_path / "runtime.sqlite3", contract, **kw)
    rt.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision="d" * 40)
    open_window(rt, 1)
    return rt


def reward_contract(**reward):
    value = contract_v2_dict()
    value["policies"]["reward"].update(reward)
    return ServiceContract.from_dict(value)


def group(rt, *, env=MATH, prompt=7, window=1, seeds=None):
    """(group id, candidate) of a subset of the window's announced pool, as the batcher derives them."""
    pool = rt.seed_pool(environment=env, prompt_idx=prompt, window=window)
    selection = pool.selection(list(range(M_ROLLOUTS)) if seeds is None else list(seeds))
    return selection.sha256, {"pool_sha256": selection.pool_sha256, "seeds": list(selection.seeds)}


def explore(rt, *, prompt=7, hotkey="hk", env=MATH, rewards=ZERO, window=1, now=120.0, seeds=None, **kw):
    group_id, candidate = group(rt, env=env, prompt=prompt, window=window, seeds=seeds)
    return rt.record_exploration(environment=env, prompt_idx=prompt, hotkey=hotkey, window=window, rewards=rewards,
                                 group_id=group_id, candidate=candidate, token_count=100, now=now, **kw)


def train(rt, *, prompt=7, hotkey="t", env=MATH, rewards=HALF, window=1, now=5.0, seeds=None):
    group_id, candidate = group(rt, env=env, prompt=prompt, window=window, seeds=seeds)
    return rt.record_training(environment=env, prompt_idx=prompt, hotkey=hotkey, window=window, rewards=rewards,
                              group_id=group_id, candidate=candidate, token_count=10, now=now)


def draw(rt, window=1, now=10_000.0, beacon=BEACON, **kw):
    return rt.resolve_draws(window, beacon_for_round=lambda r: beacon, now=now, **kw)


def audited(rt, result, *, passed=True, window=1, now=10_000.0):
    draw(rt, window, now=now)
    return rt.record_audit(result["observation_id"], passed=passed, now=now)


def archive(window=1, batch=(), **extra):
    rows = [{"hotkey": hotkey, "env_name": env, "prompt_idx": prompt} for hotkey, env, prompt in batch]
    return {"window_start": window, "window_status": "complete", "batch": rows, "rewards_by_hotkey": {}, **extra}


def events(rt, kind=None, identity=None):
    return [e for _, e in rt.events(limit=10_000)
            if (kind is None or e["type"] == kind) and (identity is None or e["id"] == identity)]


def frozen(result):
    return json.dumps({key: result[key] for key in FROZEN_ARCHIVE_FIELDS}, sort_keys=True)


# ---------------------------------------------------------------- observation path

def test_first_scan_is_entitled_at_15_percent_of_a_training_group(tmp_path):
    rt = runtime(tmp_path)
    result = explore(rt)
    assert result["first_scan"] and result["entitled"] and result["inserted"]
    assert result["status"] == "exploration_pending" and result["reason"] is None
    assert result["amount"] == pytest.approx(PRICE, rel=1e-12)
    assert result["forced_audit"] is True
    (event,) = events(rt, "observation")
    assert event["status"] == "exploration_pending" and event["proof"] == "pending" and "hotkey" not in event
    assert event["candidate"]["seeds"] == list(range(M_ROLLOUTS))


def test_draw_round_is_the_validator_clock_arrival_round_plus_two(tmp_path):
    rt = runtime(tmp_path)
    assert explore(rt, prompt=1, now=120.0)["draw_round"] == 42            # arrival defaults to now
    assert explore(rt, prompt=2, now=120.0, arrived_at=90.0)["draw_round"] == 32
    # An arrival "in the future" is clamped to the validator's now: it cannot push the draw away.
    assert explore(rt, prompt=3, now=120.0, arrived_at=9_999.0)["draw_round"] == 42
    with pytest.raises(TypeError):  # a caller cannot hand in a round of its own
        explore(rt, prompt=4, arrival_round=1)


def test_two_hotkeys_on_one_never_scanned_prompt_same_subset_first_arrival_wins(tmp_path):
    rt = runtime(tmp_path)
    assert group(rt)[0] == group(rt)[0]                   # every reference miner shares this group id
    first = explore(rt, hotkey="a")
    second = explore(rt, hotkey="b")                       # same env, prompt, window, seeds 0..M-1
    assert first["entitled"] and first["first_scan"]
    assert second["observation_id"] != first["observation_id"]   # R1: two observations
    assert not second["entitled"] and not second["first_scan"] and second["amount"] == 0.0
    assert (second["status"], second["reason"]) == ("exploration_unpaid", "already_scanned")
    (published,) = events(rt, "observation", second["observation_id"])
    assert published["status"] == "exploration_unpaid" and published["reason"] == "already_scanned"
    # Another subset of the same prompt is another group id, and still not a first scan.
    other = explore(rt, hotkey="c", seeds=range(M_ROLLOUTS, 2 * M_ROLLOUTS))
    assert group(rt, seeds=range(M_ROLLOUTS, 2 * M_ROLLOUTS))[0] != group(rt)[0]
    assert (other["entitled"], other["reason"]) == (False, "already_scanned")
    assert len(rt.ledger.rows(1, environment=MATH)) == 1


def test_same_miner_retry_is_idempotent(tmp_path):
    rt = runtime(tmp_path)
    first = explore(rt, hotkey="a")
    again = explore(rt, hotkey="a", now=500.0)
    assert again["observation_id"] == first["observation_id"] and again["inserted"] is False
    assert again["entitled"] and again["draw_round"] == first["draw_round"]
    assert len(events(rt)) == 1 and len(rt.ledger.rows(1, environment=MATH)) == 1


def test_training_scan_blocks_later_exploration_pay(tmp_path):
    rt = runtime(tmp_path)
    trained = train(rt)
    assert trained["first_scan"] and trained["inserted"]
    late = explore(rt)
    assert (late["entitled"], late["reason"]) == (False, "already_scanned")


def test_observations_survive_adoption(tmp_path):
    rt = runtime(tmp_path)
    explore(rt)
    rt.adopt(checkpoint_n=1, repo="models/test", revision="f" * 40, sha256="e" * 64)
    open_window(rt, 2)
    later = explore(rt, window=2, hotkey="other")
    assert (later["first_scan"], later["reason"]) == (False, "already_scanned")
    assert rt.announcement(window=2, randomness=WINDOW_BEACON)["checkpoint"]["revision"] == "f" * 40
    assert rt.announcement(window=1, randomness=WINDOW_BEACON)["checkpoint"]["revision"] == "d" * 40
    assert events(rt, "observation", later["observation_id"])[0]["checkpoint"] == "f" * 40


def test_not_an_observation_is_a_clean_logged_refusal(tmp_path, monkeypatch):
    rt = runtime(tmp_path)
    errors = []
    monkeypatch.setattr(runtime_module.logger, "error", lambda *args: errors.append(args))
    group_id, candidate = group(rt)
    base = dict(environment=MATH, prompt_idx=7, hotkey="a", window=1, group_id=group_id, candidate=candidate,
                token_count=10, now=120.0)
    bad = [
        dict(base, rewards=ZERO[:-1], candidate=dict(candidate, seeds=candidate["seeds"])),   # incomplete group
        dict(base, rewards=ZERO, candidate=dict(candidate, seeds=candidate["seeds"][:-1])),   # seeds / rewards
        dict(base, rewards=ZERO, candidate=dict(candidate, pool_sha256="0" * 64)),            # another pool
        dict(base, rewards=ZERO, group_id="1" * 64),                                          # not the selection
        dict(base, rewards=ZERO, candidate=None),
        dict(base, rewards=[float("nan")] * M_ROLLOUTS),
        dict(base, rewards=[2.0] * M_ROLLOUTS),
        dict(base, rewards=ZERO, prompt_idx=1000),                                            # outside the dataset
        dict(base, rewards=ZERO, hotkey=""),
    ]
    for kwargs in bad:
        with pytest.raises(ServicePolicyLimit):
            rt.record_exploration(**kwargs)
        with pytest.raises(ServicePolicyLimit):
            rt.record_training(**kwargs)
    assert len(errors) == 2 * len(bad)                    # R10: never silent
    assert events(rt) == [] and rt.ledger.rows(1, environment=MATH) == []
    assert not rt.db.in_transaction
    # Same identity with other evidence: refused, the stored observation is untouched.
    first = rt.record_exploration(**dict(base, rewards=ZERO))
    with pytest.raises(ServicePolicyLimit, match="different evidence"):
        rt.record_exploration(**dict(base, rewards=ONES))
    assert any(first["observation_id"] in str(args) for args in errors)
    assert len(events(rt)) == 1


def test_a_prompt_outside_the_ordered_dataset_is_refused(tmp_path):
    rt = runtime(tmp_path)                                 # fixture datasets have 1000 rows
    assert explore(rt, prompt=999)["entitled"] is True
    for prompt in (1000, 10**6):
        with pytest.raises(ServicePolicyLimit, match="outside the ordered dataset"):
            explore(rt, prompt=prompt)
        with pytest.raises(ServicePolicyLimit, match="outside the ordered dataset"):
            train(rt, prompt=prompt)
    assert len(events(rt)) == 1


def test_a_window_without_envelope_or_an_env_outside_it_takes_nothing(tmp_path):
    rt = runtime(tmp_path, envs=(MATH, CODE))
    group_id, candidate = group(rt)
    for window in (2, 99):
        with pytest.raises(ServicePolicyLimit, match="envelope"):
            rt.record_exploration(environment=MATH, prompt_idx=7, hotkey="a", window=window, rewards=ZERO,
                                  group_id=group_id, candidate=candidate, token_count=1, now=120.0)
    with pytest.raises(ServicePolicyLimit, match="not active"):
        rt.record_exploration(environment="reliquary_science_v1", prompt_idx=7, hotkey="a", window=1, rewards=ZERO,
                              group_id=group_id, candidate=candidate, token_count=1, now=120.0)
    assert events(rt) == []


def test_no_drand_clock_means_no_entitlement(tmp_path):
    def broken(_):
        raise RuntimeError("drand chain info is not known yet")
    contract = contract_v2()
    rt = ServiceRuntime(tmp_path / "r.sqlite3", contract, qualification_v2(contract), now=0, drand_round_at=broken)
    rt.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision="d" * 40)
    open_window(rt, 1)
    with pytest.raises(ServicePolicyLimit, match="draw round"):
        explore(rt)
    assert events(rt) == []


def test_runtime_refusals_are_published_unpaid_with_their_reason(tmp_path):
    rt = runtime(tmp_path, contract=reward_contract(max_tokens_per_group=50))
    over = explore(rt, prompt=1)                           # helper sends 100 tokens
    assert (over["status"], over["reason"], over["first_scan"]) == ("exploration_unpaid", "token_limit", False)
    assert rt.log.is_scanned(MATH, 1) is False
    rt2 = runtime(tmp_path / "x", exploration=0)
    off = explore(rt2, prompt=1)
    assert (off["status"], off["reason"]) == ("exploration_unpaid", "exploration_disabled")


def test_inactive_order_refuses_training_and_unpays_exploration(tmp_path):
    rt = runtime(tmp_path)
    with pytest.raises(ServicePolicyLimit, match="inactive"):
        train(rt, now=10**9)
    late = explore(rt, now=10**9)
    assert (late["entitled"], late["reason"]) == (False, "order_inactive")
    assert rt.log.is_scanned(MATH, 7) is False


def test_exploration_does_not_consume_the_order_budget(tmp_path):
    value = contract_v2_dict()
    value["limits"]["max_groups"] = 2
    rt = runtime(tmp_path, contract=ServiceContract.from_dict(value))
    for prompt in range(5):
        explore(rt, prompt=prompt, hotkey="spam")
    assert rt.active(now=121.0) is True                   # unpaid/unproven groups cannot close the order
    train(rt, prompt=50, now=122.0)
    train(rt, prompt=51, now=123.0)
    assert rt.active(now=124.0) is False


# ---------------------------------------------------------------- audits

def test_failed_audit_forfeits_bans_and_reopens_the_prompt(tmp_path):
    rt = runtime(tmp_path)
    first = explore(rt, hotkey="cheat", prompt=7)
    other = explore(rt, hotkey="cheat", prompt=8, env=CODE)
    assert set(draw(rt)) == {first["observation_id"], other["observation_id"]}
    outcome = rt.record_audit(first["observation_id"], passed=False, now=10_000.0)
    assert outcome.kind == "failed" and outcome.failed and outcome.applied
    forfeited = outcome.forfeited
    assert set(forfeited) == {first["observation_id"], other["observation_id"]}   # the window, every env
    assert rt.exploration_banned("cheat", now=10_001.0) and rt.exploration_banned("cheat", now=10_000.0 + 86_399)
    assert not rt.exploration_banned("cheat", now=10_000.0 + 86_401)
    (settle,) = events(rt, "settle", first["observation_id"])
    assert (settle["status"], settle["proof"]) == ("exploration_forfeited", "failed")
    (collateral,) = events(rt, "settle", other["observation_id"])
    assert (collateral["status"], collateral["proof"]) == ("exploration_forfeited", "pending")
    assert rt.log.is_scanned(MATH, 7) is False and rt.log.is_scanned(CODE, 8) is False
    assert rt.queued_audits(1) == []                      # a forfeited row is not worth an audit
    honest = explore(rt, hotkey="honest", prompt=7, now=10_002.0)
    assert honest["entitled"] is True and honest["first_scan"] is True


def test_banned_hotkey_is_published_unpaid_and_holds_no_first_scan(tmp_path):
    rt = runtime(tmp_path)
    audited(rt, explore(rt, hotkey="cheat"), passed=False)
    banned = explore(rt, hotkey="cheat", prompt=8, now=10_030.0)
    assert (banned["entitled"], banned["status"], banned["reason"]) == (False, "exploration_unpaid", "banned")
    assert events(rt, "observation", banned["observation_id"])[0]["reason"] == "banned"
    assert rt.log.is_scanned(MATH, 8) is False            # released: someone else can be paid for it
    assert explore(rt, hotkey="honest", prompt=8, now=10_031.0)["entitled"] is True


def test_over_cap_is_published_unpaid_and_holds_no_first_scan(tmp_path):
    rt = runtime(tmp_path)
    results = [explore(rt, prompt=prompt, hotkey=f"h{prompt % 6}") for prompt in range(CAP_COUNT + 1)]
    assert all(r["entitled"] for r in results[:CAP_COUNT])
    last = results[-1]
    assert (last["entitled"], last["status"], last["reason"], last["first_scan"]) == \
        (False, "exploration_unpaid", "cap", False)
    assert events(rt, "observation", last["observation_id"])[0]["reason"] == "cap"
    assert rt.log.is_scanned(MATH, CAP_COUNT) is False
    assert explore(rt, prompt=0, env=CODE)["entitled"] is True          # the cap is per env
    open_window(rt, 2)
    assert explore(rt, prompt=CAP_COUNT, window=2, now=130.0)["entitled"] is True   # and per window


def test_passed_audit_publishes_audited_but_never_paid_before_finalize(tmp_path):
    rt = runtime(tmp_path)
    result = explore(rt)
    assert audited(rt, result) == AuditOutcome("passed") and audited(rt, result).passed
    (settle,) = events(rt, "settle", result["observation_id"])
    assert (settle["status"], settle["proof"]) == ("exploration_pending", "audited")
    assert not any(e["status"] == "exploration_paid" for e in events(rt))


def test_resolve_draws_uses_the_beacon_of_exactly_that_round_and_never_a_future_one(tmp_path):
    rt = runtime(tmp_path)
    early = explore(rt, prompt=1, now=120.0)               # draw round 42
    late = explore(rt, prompt=2, now=300.0)                # draw round 102
    asked = []

    def fetch(round_id):
        asked.append(round_id)
        return {"round": round_id + 1, "randomness": BEACON}            # "latest", not this round
    assert rt.resolve_draws(1, beacon_for_round=fetch, now=200.0) == []  # clock at round 66
    assert asked == [42]                                   # round 102 does not exist yet: never asked
    assert rt.pending_draw_rounds(1) == [42, 102]          # and the wrong-round beacon was ignored
    assert rt.resolve_draws(1, beacon_for_round=lambda r: "xyz", now=200.0) == []
    assert rt.resolve_draws(1, beacon_for_round=lambda r: {"round": r, "randomness": BEACON}, now=200.0) == \
        [early["observation_id"]]
    assert rt.pending_draw_rounds(1) == [102] and rt.pending_draw_rounds(1, environment=CODE) == []
    # A stored beacon is THE beacon of its round: a later, different answer is not used.
    stored = rt.db.execute("SELECT round, randomness FROM service_draw_beacons").fetchall()
    assert stored == [(42, BEACON)]
    assert rt.resolve_draws(1, beacon_for_round=lambda r: "ee" * 32, now=400.0) == [late["observation_id"]]
    assert dict(rt.db.execute("SELECT round, randomness FROM service_draw_beacons").fetchall()) == \
        {42: BEACON, 102: "ee" * 32}
    same_round = explore(rt, prompt=5, now=500.0, arrived_at=120.0)       # a later row of round 42
    asked.clear()
    assert rt.resolve_draws(1, beacon_for_round=lambda r: asked.append(r) or "11" * 32, now=600.0) == \
        [same_round["observation_id"]]
    assert asked == []                                     # round 42 is not fetched twice


def test_default_beacon_source_is_the_verified_drand_beacon(tmp_path, monkeypatch):
    rt = runtime(tmp_path)
    result = explore(rt)
    calls = []
    monkeypatch.setattr("reliquary.infrastructure.drand.get_verified_beacon",
                        lambda round_id: calls.append(round_id) or {"round": round_id, "signature": "", "randomness": BEACON})
    assert rt.resolve_draws(1, now=10_000.0) == [result["observation_id"]]
    assert calls == [42]


def test_audit_queue_runs_hotkeys_past_probation_first_then_by_draw_round(tmp_path):
    rt = runtime(tmp_path, contract=reward_contract(new_hotkey_audit_groups=1, audit_bps=10000))
    audited(rt, explore(rt, hotkey="old", prompt=0), now=200.0)          # "old" leaves probation
    new_early = explore(rt, hotkey="new", prompt=1, now=300.0)           # in probation, earliest round
    old_late = explore(rt, hotkey="old", prompt=2, now=330.0)
    old_later = explore(rt, hotkey="old", prompt=3, env=CODE, now=360.0)
    new_late = explore(rt, hotkey="new", prompt=4, now=390.0)
    assert rt.exploration_backlog(1) == {"pending_draw": 4, "queued": 0, "queued_past_probation": 0}
    draw(rt)
    queue = rt.queued_audits(1)
    assert [row["observation_id"] for row in queue] == [
        old_late["observation_id"], old_later["observation_id"], new_early["observation_id"], new_late["observation_id"]]
    assert [row["past_probation"] for row in queue] == [True, True, False, False]
    assert queue[1]["environment"] == CODE and queue[0]["hotkey"] == "old" and queue[0]["forced"] is False
    assert [row["observation_id"] for row in rt.queued_audits(1, environment=CODE)] == [old_later["observation_id"]]
    assert rt.exploration_backlog(1) == {"pending_draw": 0, "queued": 4, "queued_past_probation": 2}
    rt.record_audit(old_late["observation_id"], passed=True, now=10_001.0)
    assert old_late["observation_id"] not in [row["observation_id"] for row in rt.queued_audits(1)]


def test_a_verdict_that_cannot_be_applied_is_logged_and_never_raises(tmp_path, monkeypatch):
    rt = runtime(tmp_path, contract=reward_contract(new_hotkey_audit_groups=0, audit_bps=0))
    errors = []
    monkeypatch.setattr(runtime_module.logger, "error", lambda *args: errors.append(args))
    not_drawn = explore(rt)
    draw(rt)                                               # audit_bps 0, no probation: not drawn
    assert rt.ledger.rows(1, environment=MATH)[0]["audit"] == "not_drawn"
    assert rt.record_audit(not_drawn["observation_id"], passed=False, now=10_000.0) == AuditOutcome("not_applied")
    assert rt.record_audit("f" * 64, passed=True, now=10_000.0) == AuditOutcome("not_applied")
    assert not AuditOutcome("not_applied").applied
    assert len(errors) == 2 and not_drawn["observation_id"] in str(errors[0]) and "f" * 64 in str(errors[1])
    assert not rt.exploration_banned("hk", now=10_001.0) and events(rt, "settle") == []
    assert not rt.db.in_transaction

    def busy(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(runtime_module, "apply_exploration_verdict", busy)
    assert rt.record_audit(not_drawn["observation_id"], passed=True, now=10_000.0).kind == "not_applied"
    assert len(errors) == 3 and not rt.db.in_transaction


def test_finalize_unpays_what_was_never_audited_and_releases_its_first_scan(tmp_path):
    rt = runtime(tmp_path)
    queued = explore(rt, prompt=1, now=120.0)
    pending = explore(rt, prompt=2, now=9_000.0)
    paid = explore(rt, prompt=3, hotkey="other", now=120.0)
    rt.resolve_draws(1, beacon_for_round=lambda r: BEACON, now=200.0)
    rt.record_audit(paid["observation_id"], passed=True, now=201.0)
    released = rt.finalize_exploration(1, environment=MATH, now=9_100.0)
    assert set(released) == {queued["observation_id"], pending["observation_id"]}
    for result in (queued, pending):
        settle = events(rt, "settle", result["observation_id"])[-1]
        assert (settle["status"], settle["proof"]) == ("exploration_unpaid", "unproven")
    assert rt.log.is_scanned(MATH, 1) is False and rt.log.is_scanned(MATH, 2) is False
    assert rt.log.is_scanned(MATH, 3) is True
    assert not rt.exploration_banned("hk", now=9_101.0)   # unaudited is never sanctioned
    count = len(events(rt))
    assert rt.finalize_exploration(1, environment=MATH, now=9_200.0) == released and len(events(rt)) == count
    late = explore(rt, prompt=9, now=9_300.0)
    assert (late["entitled"], late["reason"]) == (False, "finalized")
    assert explore(rt, prompt=9, env=CODE, now=9_300.0)["entitled"] is True      # R2: the other env is open
    assert rt.queued_audits(1) == [] and rt.pending_draw_rounds(1, environment=MATH) == []


# ---------------------------------------------------------------- settlement

def test_reconcile_pays_from_the_envelope_self_validates_and_is_idempotent(tmp_path):
    rt = runtime(tmp_path)
    paid = explore(rt, hotkey="x")
    audited(rt, paid)
    won = train(rt, prompt=20, hotkey="a")
    lost = train(rt, prompt=21, hotkey="b")
    rt.finalize_exploration(1, environment=MATH, now=10_050.0)
    given = archive(batch=[("a", MATH, 20)], rewards_by_hotkey={"a": POOL / T})
    first = rt.reconcile_archive(given, now=10_100.0)
    assert first["rewards_by_hotkey"] == {"a": pytest.approx(POOL / T), "x": pytest.approx(PRICE)}
    assert first["service_exploration_by_environment"] == {MATH: {"x": 1}}
    assert first["service_pools_by_environment"] == {CODE: POOL, MATH: POOL}
    assert (first["service_picks_target"], first["service_batch_slots"]) == (PICKS, SLOTS)
    assert first["service_scale_by_environment"] == {CODE: 1.0, MATH: 1.0}
    assert first["service_training_recomputed_delta"] == pytest.approx(0.0, abs=1e-15)
    assert first["service_schedule_sha256"] == rt.envelope(1)["schedule_sha256"]
    # What weight replay will do, after a JSON round trip, under the protocol geometry.
    validate_service_archive_v2(json.loads(json.dumps(first)), rt.contract, cap=1.0, picks_target=PICKS, batch_slots=SLOTS)
    before = len(events(rt))
    again = rt.reconcile_archive(given, now=11_000.0)
    assert json.dumps(again, sort_keys=True) == json.dumps(first, sort_keys=True)
    assert frozen(rt.reconcile_archive(first, now=12_000.0)) == frozen(first)   # fed back enriched: same money
    assert len(events(rt)) == before
    final = {e["id"]: (e["status"], e["proof"]) for e in events(rt, "settle")}
    assert final[paid["observation_id"]] == ("exploration_paid", "audited")
    assert final[won["observation_id"]] == ("trained", "proven")
    assert final[lost["observation_id"]] == ("proven_unpaid", "proven")
    # Another batch or another disposition for a settled window: the frozen money, never a raise (I2).
    assert frozen(rt.reconcile_archive(archive(batch=[("a", MATH, 20), ("b", MATH, 21)]))) == frozen(first)
    assert frozen(rt.reconcile_archive(given, aborted=True)) == frozen(first)
    assert len(events(rt)) == before
    with pytest.raises(ServicePolicyLimit, match="settled"):
        explore(rt, prompt=30, now=12_001.0)
    with pytest.raises(ServicePolicyLimit, match="settled"):
        train(rt, prompt=31, now=12_001.0)


def test_reconcile_finalizes_what_the_caller_did_not_and_never_pays_an_unaudited_row(tmp_path):
    rt = runtime(tmp_path)
    queued = explore(rt, hotkey="x")
    draw(rt)                                               # drawn, never audited, never finalized by the caller
    result = rt.reconcile_archive(archive(), now=10_100.0)
    assert result["rewards_by_hotkey"] == {} and result["service_exploration_by_environment"] == {}
    settle = events(rt, "settle", queued["observation_id"])[-1]
    assert (settle["status"], settle["proof"]) == ("exploration_unpaid", "unproven")
    assert rt.log.is_scanned(MATH, 7) is False
    assert rt.ledger.is_finalized(1, environment=MATH) and rt.ledger.is_finalized(1, environment=CODE)


def test_full_window_scales_training_and_exploration_alike_and_the_archive_validates(tmp_path):
    rt = runtime(tmp_path)
    for prompt in range(CAP_COUNT):
        explore(rt, prompt=prompt, hotkey=f"x{prompt % 6}")
    draw(rt)
    for row in rt.queued_audits(1):
        rt.record_audit(row["observation_id"], passed=True, now=10_000.0)
    rt.finalize_exploration(1, environment=MATH, now=10_050.0)
    batch = [("a", MATH, 100 + i) for i in range(T)] + [("b", CODE, 100 + i) for i in range(T // 2)]
    result = rt.reconcile_archive(archive(batch=batch), now=10_100.0)
    scale = 1 / (1 + CAP_COUNT * 0.15 / T)
    assert result["service_scale_by_environment"][MATH] == pytest.approx(scale, rel=1e-12) and scale < 1
    assert result["service_scale_by_environment"][CODE] == 1.0
    rewards = result["rewards_by_hotkey"]
    assert rewards["a"] == pytest.approx(POOL * scale, rel=1e-12)                 # training scaled...
    explorers = sum(rewards[f"x{i}"] for i in range(6))
    assert explorers == pytest.approx(CAP_COUNT * PRICE * scale, rel=1e-12)       # ...explorers too
    assert rewards["a"] + explorers == pytest.approx(POOL, rel=1e-12)   # env pool conserved
    assert rewards["b"] == pytest.approx(POOL / 2, rel=1e-12)                     # the other env is untouched
    validate_service_archive_v2(json.loads(json.dumps(result)), rt.contract, cap=1.0, picks_target=PICKS, batch_slots=SLOTS)
    validate_service_archive(json.loads(json.dumps(result)), rt.contract, cap=1.0)   # replay's name, protocol geometry
    with pytest.raises(SettlementError, match="geometry"):
        validate_service_archive(result, rt.contract, cap=1.0, picks_target=1)


def test_late_failed_audit_after_finalize_bans_and_changes_no_archive(tmp_path):
    rt = runtime(tmp_path)
    paid = explore(rt, hotkey="x", prompt=1, now=120.0)
    unaudited = explore(rt, hotkey="x", prompt=2, now=150.0)
    draw(rt)
    rt.record_audit(paid["observation_id"], passed=True, now=10_000.0)
    rt.finalize_exploration(1, environment=MATH, now=10_050.0)
    given = archive()
    first = rt.reconcile_archive(given, now=10_100.0)
    assert first["rewards_by_hotkey"] == {"x": pytest.approx(PRICE)}
    rows, count = rt.ledger.rows(1, environment=MATH), len(events(rt))
    late = rt.record_audit(unaudited["observation_id"], passed=False, now=10_200.0)
    assert late == AuditOutcome("failed", ())                            # applied (it bans), nothing to forfeit
    assert rt.exploration_banned("x", now=10_201.0)                      # the sanction that is left (R3)
    assert rt.record_audit(unaudited["observation_id"], passed=True, now=10_210.0) == AuditOutcome("not_applied")   # nor a late pass
    assert rt.ledger.rows(1, environment=MATH) == rows and len(events(rt)) == count
    assert rt.log.is_scanned(MATH, 1) is True                            # the paid first scan is not released
    assert json.dumps(rt.reconcile_archive(given, now=10_300.0), sort_keys=True) == json.dumps(first, sort_keys=True)


def test_aborted_window_pays_no_exploration_and_gives_its_first_scans_back(tmp_path):
    rt = runtime(tmp_path)
    passed = explore(rt, hotkey="x", prompt=1)
    audited(rt, passed)
    train(rt, prompt=20, hotkey="a")
    result = rt.reconcile_archive(archive(batch=[("a", MATH, 20)]), aborted=True, now=10_100.0)
    assert result["window_status"] == "aborted" and result["service_exploration_by_environment"] == {}
    assert "x" not in result["rewards_by_hotkey"]
    settle = events(rt, "settle", passed["observation_id"])[-1]
    assert (settle["status"], settle["proof"]) == ("exploration_unpaid", "audited")
    assert not any(e["status"] in ("exploration_paid", "trained") for e in events(rt, "settle"))
    assert rt.log.is_scanned(MATH, 1) is False
    validate_service_archive_v2(result, rt.contract, cap=1.0, picks_target=PICKS, batch_slots=SLOTS)
    later = rt.reconcile_archive(archive(batch=[("a", MATH, 20)]), aborted=False)   # frozen as aborted (I2)
    assert later["window_status"] == "aborted" and frozen(later) == frozen(result)
    # An archive that says "aborted" is aborted, whatever the keyword.
    rt2 = runtime(tmp_path / "second")
    audited(rt2, explore(rt2, hotkey="x"))
    assert rt2.reconcile_archive(archive(window_status="aborted"))["rewards_by_hotkey"] == {}


def test_reconcile_self_checks_the_archive_and_a_failed_check_freezes_nothing(tmp_path, monkeypatch):
    rt = runtime(tmp_path)
    audited(rt, explore(rt, hotkey="x"))
    seen = []

    def refuse(record, contract, **kw):
        seen.append(kw)
        raise SettlementError("forged")
    with monkeypatch.context() as patch:
        patch.setattr(runtime_module, "validate_service_archive_v2", refuse)
        with pytest.raises(SettlementError, match="forged"):
            rt.reconcile_archive(archive(), now=10_100.0)
    assert seen == [{"cap": 1.0, "picks_target": PICKS, "batch_slots": SLOTS}]   # the protocol geometry
    assert rt.db.execute("SELECT COUNT(*) FROM service_settled").fetchone()[0] == 0
    assert not rt.ledger.is_finalized(1, environment=MATH) and events(rt, "settle")[-1]["status"] == "exploration_pending"
    assert rt.reconcile_archive(archive(), now=10_200.0)["rewards_by_hotkey"] == {"x": pytest.approx(PRICE)}


def test_a_paid_group_without_env_or_string_hotkey_freezes_nothing(tmp_path):
    rt = runtime(tmp_path)
    for row in ({"hotkey": "a"}, {"hotkey": 5, "env_name": MATH}, {"hotkey": "a", "env_name": "reliquary_science_v1"}):
        with pytest.raises(SettlementError, match="outside the window envelope"):
            rt.reconcile_archive({"window_start": 1, "batch": [row], "rewards_by_hotkey": {}})
    assert rt.db.execute("SELECT COUNT(*) FROM service_settled").fetchone()[0] == 0
    assert not rt.ledger.is_finalized(1, environment=MATH) and not rt.db.in_transaction
    assert explore(rt)["entitled"] is True                # the window is still live


# ---------------------------------------------------------------- envelope, schedule, announcement

def test_envelope_freezes_exactly_the_active_envs_and_the_protocol_geometry(tmp_path):
    rt = build(tmp_path / "r.sqlite3")
    with pytest.raises(ValueError, match="checkpoint"):
        rt.open_window(1, pools={MATH: POOL, CODE: POOL}, picks_target=PICKS, batch_slots=SLOTS)
    rt.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision="d" * 40)
    for pools in ({MATH: POOL}, {MATH: POOL, CODE: POOL, "reliquary_science_v1": 0.1}, {MATH: 0.51, CODE: POOL},
                  {MATH: float("nan"), CODE: POOL}, {MATH: -0.1, CODE: POOL}):
        with pytest.raises(ValueError):
            rt.open_window(1, pools=pools, picks_target=PICKS, batch_slots=SLOTS)
    for picks, slots in ((1, SLOTS), (PICKS, SLOTS + 1), (PICKS + 1, SLOTS)):
        with pytest.raises(ValueError, match="geometry"):
            rt.open_window(1, pools={MATH: POOL, CODE: POOL}, picks_target=picks, batch_slots=slots)
    envelope = rt.open_window(1, pools={MATH: POOL, CODE: POOL}, picks_target=PICKS, batch_slots=SLOTS)
    assert set(envelope) == {"order_sha256", "schedule", "schedule_sha256", "pools", "picks_target",
                             "batch_slots", "checkpoint"}
    assert envelope["pools"] == {CODE: POOL, MATH: POOL} and envelope["order_sha256"] == rt.contract.sha256
    assert envelope["checkpoint"] == {"checkpoint_n": 0, "repo": "models/test", "revision": "d" * 40, "sha256": "b" * 64}
    assert rt.open_window(1, pools={MATH: POOL, CODE: POOL}, picks_target=PICKS, batch_slots=SLOTS) == envelope
    with pytest.raises(ValueError, match="already frozen"):
        rt.open_window(1, pools={MATH: POOL, CODE: 0.2}, picks_target=PICKS, batch_slots=SLOTS)
    assert rt.envelope(1) == envelope


def test_schedule_change_applies_to_the_next_window_and_a_deactivated_env_still_settles(tmp_path):
    rt = runtime(tmp_path)
    before = explore(rt, env=CODE, prompt=1, hotkey="x")
    changed = next_schedule(rt.contract, rt.schedule, active=(MATH,), shares={MATH: 10000})
    assert rt.apply_schedule(changed, request_id="r1", window=1) is True
    assert rt.apply_schedule(changed, request_id="r1", window=1) is False
    assert rt.schedule.active_environments() == (MATH,)
    # The running window keeps its frozen schedule, pools and announcement (R6).
    assert rt.envelope(1)["schedule"]["revision"] == 0
    assert rt.envelope(1)["schedule"]["environments"][CODE]["active"] == 1
    assert rt.window_schedule(1).active_environments() == (CODE, MATH)
    assert set(rt.envelope(1)["pools"]) == {MATH, CODE}
    assert rt.announcement(window=1, randomness=WINDOW_BEACON)["schedule"]["revision"] == 0
    assert rt.open_window(1, pools={MATH: POOL, CODE: POOL}, picks_target=PICKS, batch_slots=SLOTS) == rt.envelope(1)
    after = explore(rt, env=CODE, prompt=2, hotkey="x", now=130.0)       # still admitted and entitled
    assert after["entitled"] is True
    draw(rt)
    for result in (before, after):
        rt.record_audit(result["observation_id"], passed=True, now=10_000.0)
    rt.finalize_exploration(1, environment=CODE, now=10_050.0)
    result = rt.reconcile_archive(archive(batch=[("c", CODE, 50)]), now=10_100.0)
    assert result["rewards_by_hotkey"] == {"c": pytest.approx(POOL / T), "x": pytest.approx(2 * PRICE)}
    assert result["service_schedule"]["revision"] == 0 and set(result["service_pools_by_environment"]) == {MATH, CODE}
    validate_service_archive_v2(result, rt.contract, cap=1.0, picks_target=PICKS, batch_slots=SLOTS)
    # The next window is built from the new schedule, and only from it.
    with pytest.raises(ValueError, match="exactly the active"):
        rt.open_window(2, pools={MATH: POOL, CODE: POOL}, picks_target=PICKS, batch_slots=SLOTS)
    envelope = rt.open_window(2, pools={MATH: 0.5}, picks_target=PICKS, batch_slots=SLOTS)
    assert envelope["schedule"]["revision"] == 1 and set(envelope["pools"]) == {MATH}
    rt.announcement(window=2, randomness=WINDOW_BEACON)
    with pytest.raises(ServicePolicyLimit, match="not active"):
        explore(rt, env=CODE, prompt=3, window=2, now=10_200.0)
    with pytest.raises(ValueError, match="increase by one"):
        rt.apply_schedule(changed, request_id="r2", window=2)
    with pytest.raises(ValueError, match="another schedule"):
        rt.apply_schedule(next_schedule(rt.contract, rt.schedule, cooldowns={MATH: 3}), request_id="r1", window=2)


def test_announcement_is_v2_and_stable_within_a_window(tmp_path):
    rt = runtime(tmp_path)
    first = rt.announcement(window=1, randomness=WINDOW_BEACON, environment=MATH)
    ServicePolicyAnnouncement(**first)                     # the protocol's own v2 check
    assert set(first) == {"contract", "schedule", "checkpoint", "supported_capabilities", "pool_epoch",
                          "pool_randomness"}
    assert first["contract"]["schema"] == "service-contract/v2"
    assert "public-seed-pool/v3" in first["supported_capabilities"]
    assert "exploration-first-scan/v1" in first["supported_capabilities"]
    assert "public-group-pool/v1" not in first["supported_capabilities"]
    assert first["pool_epoch"] == 1 and first["pool_randomness"] == WINDOW_BEACON
    assert first["schedule"] == rt.envelope(1)["schedule"] and first["checkpoint"] == rt.envelope(1)["checkpoint"]
    # Nothing that happens during the window changes what the batcher holds for it.
    rt.apply_schedule(next_schedule(rt.contract, rt.schedule, cooldowns={MATH: 3}), request_id="r1", window=1)
    rt.adopt(checkpoint_n=1, repo="models/test", revision="f" * 40, sha256="e" * 64)
    assert rt.announcement(window=1, randomness="99" * 32) == first      # a later beacon does not replace it
    assert rt.announcement(window=1, randomness=WINDOW_BEACON, environment=CODE) == first
    pool = pool_from_service_policy(first, environment=MATH, prompt_idx=7, checkpoint_hash="d" * 40)
    assert pool.sha256 == rt.seed_pool(environment=MATH, prompt_idx=7, window=1).sha256
    # The next window announces its own epoch, beacon, schedule and checkpoint.
    rt.open_window(5, pools={MATH: POOL, CODE: POOL}, picks_target=PICKS, batch_slots=SLOTS)
    later = rt.announcement(window=5, randomness="99" * 32)
    assert (later["pool_epoch"], later["pool_randomness"]) == (5, "99" * 32)
    assert later["schedule"]["revision"] == 1 and later["checkpoint"]["revision"] == "f" * 40
    with pytest.raises(ServicePolicyLimit):
        rt.announcement(window=6, randomness=WINDOW_BEACON)
    with pytest.raises(ServicePolicyLimit):
        rt.announcement(window=1, randomness=WINDOW_BEACON, environment="reliquary_science_v1")
    with pytest.raises(ValueError):
        rt.announcement(window=1, randomness="")


def test_a_group_before_the_window_is_announced_has_no_pool(tmp_path):
    rt = build(tmp_path / "r.sqlite3")
    rt.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision="d" * 40)
    rt.open_window(1, pools={MATH: POOL, CODE: POOL}, picks_target=PICKS, batch_slots=SLOTS)
    with pytest.raises(ServicePolicyLimit, match="announced"):
        rt.record_exploration(environment=MATH, prompt_idx=7, hotkey="a", window=1, rewards=ZERO, group_id="1" * 64,
                              candidate={"pool_sha256": "0" * 64, "seeds": list(range(M_ROLLOUTS))}, token_count=1,
                              now=120.0)


# ---------------------------------------------------------------- persistence

def test_restart_mid_window_keeps_the_envelope_the_pending_draws_and_the_queued_audits(tmp_path):
    path = tmp_path / "runtime.sqlite3"
    rt = runtime(tmp_path)
    queued_a = explore(rt, hotkey="a", prompt=1, now=120.0)
    queued_b = explore(rt, hotkey="b", prompt=2, env=CODE, now=150.0)
    pending = explore(rt, hotkey="a", prompt=3, now=9_000.0)
    rt.resolve_draws(1, beacon_for_round=lambda r: BEACON, now=200.0)
    rt.apply_schedule(next_schedule(rt.contract, rt.schedule, active=(MATH,), shares={MATH: 10000}),
                      request_id="r1", window=1)
    envelope, announced = rt.envelope(1), rt.announcement(window=1, randomness=WINDOW_BEACON)
    salt_ids = [queued_a["observation_id"], queued_b["observation_id"], pending["observation_id"]]
    rt.close()

    rt = build(path)
    rt.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision="d" * 40)
    assert rt.envelope(1) == envelope
    # The restart path: pools rebuilt from the window's own (frozen) schedule give the frozen envelope back.
    assert rt.window_schedule(1).active_environments() == (CODE, MATH)
    assert rt.open_window(1, pools={MATH: POOL, CODE: POOL}, picks_target=PICKS, batch_slots=SLOTS) == envelope
    assert rt.announcement(window=1, randomness="77" * 32) == announced
    assert rt.pending_draw_rounds(1) == [pending["draw_round"]]
    assert [row["observation_id"] for row in rt.queued_audits(1)] == salt_ids[:2]
    assert rt.exploration_backlog(1) == {"pending_draw": 1, "queued": 2, "queued_past_probation": 0}
    # Same run salt: a retry after the restart is the same observation, not a second one.
    assert explore(rt, hotkey="a", prompt=1, now=9_050.0)["observation_id"] == queued_a["observation_id"]
    assert rt.record_audit(queued_a["observation_id"], passed=True, now=9_060.0) == AuditOutcome("passed")
    assert rt.record_audit(queued_b["observation_id"], passed=True, now=9_060.0) == AuditOutcome("passed")
    assert rt.resolve_draws(1, beacon_for_round=lambda r: BEACON, now=9_100.0) == [pending["observation_id"]]
    rt.record_audit(pending["observation_id"], passed=True, now=9_110.0)
    for env in (MATH, CODE):
        rt.finalize_exploration(1, environment=env, now=9_120.0)
    result = rt.reconcile_archive(archive(), now=9_130.0)
    assert result["rewards_by_hotkey"] == {"a": pytest.approx(2 * PRICE), "b": pytest.approx(PRICE)}
    first = json.dumps(result, sort_keys=True)
    rt.close()

    rt = build(path)                                       # a settled window survives too
    assert json.dumps(rt.reconcile_archive(archive(), now=9_999.0), sort_keys=True) == first
    assert rt.schedule.active_environments() == (MATH,)
    rt.close()


def test_one_wal_connection_is_shared_by_log_and_ledger(tmp_path):
    rt = runtime(tmp_path)
    assert rt.log.db is rt.db and rt.ledger.db is rt.db
    assert rt.db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert not rt.db.in_transaction
    explore(rt)
    assert not rt.db.in_transaction
    other = sqlite3.connect(Path(tmp_path) / "runtime.sqlite3")
    assert other.execute("SELECT COUNT(*) FROM exploration_entitlements").fetchone()[0] == 1   # committed
    other.close()


def test_v1_adaptive_training_and_bad_qualifications_are_refused(tmp_path):
    v1 = json.loads((Path(__file__).parents[1] / "fixtures" / "service_contract_v1.json").read_text())
    v1["service_kind"] = "adaptive_training"
    v1["policies"]["checkpoint"] = {"kind": "trainer-driven/v1", "task_scoped": 1}
    v1 = ServiceContract.from_dict(v1)                     # still a valid v1 document...
    with pytest.raises(ValueError, match="service-contract/v2"):   # ...that the RL runtime refuses
        ServiceRuntime(tmp_path / "v1.sqlite3", v1, {"schema": "service-runtime-qualification/v2", "qualified": True,
                                                    "qualification_id": "q", "contract_sha256": v1.sha256,
                                                    "group_size": M_ROLLOUTS, "forced_seed_report_sha256": "f" * 64})
    assert not (tmp_path / "v1.sqlite3").exists()
    contract = contract_v2()
    for index, overrides in enumerate((
            {"contract_sha256": "0" * 64}, {"forced_seed_report_sha256": "x"}, {"qualified": False},
            {"schema": "service-runtime-qualification/v1"}, {"group_size": M_ROLLOUTS + 1})):
        with pytest.raises(ValueError, match="qualification"):
            ServiceRuntime(tmp_path / f"q{index}.sqlite3", contract, qualification_v2(contract, **overrides))


def test_another_order_cannot_reuse_the_run_journal(tmp_path):
    contract = contract_v2()
    ServiceRuntime(tmp_path / "r.sqlite3", contract, qualification_v2(contract)).close()
    other = contract_v2(cooldown_windows=51)
    with pytest.raises(ValueError, match="another order"):
        ServiceRuntime(tmp_path / "r.sqlite3", other, qualification_v2(other))
    ServiceRuntime(tmp_path / "r.sqlite3", contract, qualification_v2(contract)).close()   # the owner still can
    legacy = sqlite3.connect(tmp_path / "v1.sqlite3")
    legacy.execute("CREATE TABLE service_contexts(id TEXT PRIMARY KEY)")
    legacy.commit()
    legacy.close()
    with pytest.raises(ValueError, match="another order"):
        ServiceRuntime(tmp_path / "v1.sqlite3", contract, qualification_v2(contract))


def test_checkpoint_lineage_is_append_only_and_restorable(tmp_path):
    rt = runtime(tmp_path)
    with pytest.raises(ValueError, match="repository"):
        rt.adopt(checkpoint_n=1, repo="models/other", revision="f" * 40, sha256="e" * 64)
    with pytest.raises(ValueError, match="lineage"):
        rt.ensure_checkpoint(checkpoint_n=1, repo="models/test", revision="f" * 40)
    rt.adopt(checkpoint_n=1, repo="models/test", revision="f" * 40, sha256="e" * 64)
    with pytest.raises(ValueError, match="another identity"):
        rt.adopt(checkpoint_n=1, repo="models/test", revision="f" * 40, sha256="a" * 64)
    assert rt.checkpoint == {"checkpoint_n": 1, "repo": "models/test", "revision": "f" * 40, "sha256": "e" * 64}
    assert rt.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision="d" * 40)["revision"] == "d" * 40
    assert rt.ensure_checkpoint(checkpoint_n=1, repo="models/test", revision="f" * 40)["sha256"] == "e" * 64


# ---------------------------------------------------------------- consumption, removed v1 surface

def test_consumption_is_measured_per_env_and_smoothed(tmp_path):
    from types import SimpleNamespace as G
    rt = runtime(tmp_path)
    assert rt.measured_consumption() == {}
    assert rt.record_consumption(1) == {}                  # no journal facts: no invented throughput
    stride = 2

    def journal(key, math_prompts, code_prompts):
        rt.record_training_journal(key, b"payload-%d" % key, is_tombstone=False, stride=stride,
                                   batches={MATH: [G(prompt_idx=p) for p in math_prompts],
                                            CODE: [G(prompt_idx=p) for p in code_prompts]})
    journal(2, [1, 2], [1]); journal(3, [2, 3], [])        # window 1: 3 math prompts, 1 code prompt
    assert rt.record_consumption(3) == {CODE: 1.0, MATH: 3.0}
    journal(4, [1], [1, 2, 3]); journal(5, [1], [4])       # window 2: 1 math, 4 code
    smoothed = rt.record_consumption(5)                    # EMA, smoothing_bps = 3000
    assert smoothed == {CODE: pytest.approx(0.3 * 4 + 0.7 * 1), MATH: pytest.approx(0.3 * 1 + 0.7 * 3)}
    assert rt.record_consumption(5) == smoothed and rt.measured_consumption() == smoothed
    with pytest.raises(ValueError, match="backwards"):
        rt.record_consumption(4)
    rt.close()
    rt = build(tmp_path / "runtime.sqlite3")
    assert rt.measured_consumption() == smoothed           # survives a restart
    journal(8, [1], [])                                    # window 3 never journaled, window 4 half journaled
    assert rt.record_consumption(9) == {} and rt.measured_consumption() == {}


def test_removed_v1_entry_points_fail_closed_and_name_their_task(tmp_path):
    rt = runtime(tmp_path)
    for call in (lambda: rt.view, lambda: rt.row_ids, lambda: rt.prepare_view(window=1),
                 lambda: rt.training_pool({MATH: 1.0}), lambda: rt.record_verified({}), lambda: rt.snapshot()):
        with pytest.raises(NotImplementedError, match="wired in Task 1"):
            call()
    assert not hasattr(rt, "mark_unaudited")              # it would defeat the per-hotkey audit horizon


# ---------------------------------------------------------------- round 4

def test_n3_a_window_finalized_then_found_aborted_gives_its_prompts_back_and_they_pay_later(tmp_path):
    rt = runtime(tmp_path)
    first = explore(rt, hotkey="x", prompt=1)
    audited(rt, first)
    rt.finalize_exploration(1, environment=MATH, now=10_050.0)   # the batcher finalizes before it knows
    assert rt.log.is_scanned(MATH, 1) is True and rt.ledger.payable(1, environment=MATH) == {"x": 1}
    result = rt.reconcile_archive(archive(), aborted=True, now=10_100.0)
    assert result["window_status"] == "aborted" and result["service_exploration_by_environment"] == {}
    assert rt.ledger.payable(1, environment=MATH) == {}
    assert [r["status"] for r in rt.ledger.rows(1, environment=MATH)] == ["unpaid"]
    assert rt.log.is_scanned(MATH, 1) is False
    assert events(rt, "settle", first["observation_id"])[-1]["status"] == "exploration_unpaid"
    # a later window: the same prompt is a first scan again, and it is paid there
    open_window(rt, 2)
    again = explore(rt, hotkey="y", prompt=1, window=2, now=130.0)
    assert again["first_scan"] and again["entitled"]
    audited(rt, again, window=2)
    rt.finalize_exploration(2, environment=MATH, now=10_150.0)
    later = rt.reconcile_archive(archive(window=2), now=10_200.0)
    assert later["service_exploration_by_environment"] == {MATH: {"y": 1}}
    assert later["rewards_by_hotkey"] == {"y": pytest.approx(PRICE)}


def test_n3_an_aborted_finalize_through_the_runtime_is_idempotent(tmp_path):
    rt = runtime(tmp_path)
    audited(rt, explore(rt, hotkey="x", prompt=1))
    rt.finalize_exploration(1, environment=MATH, now=10_050.0)
    first = rt.reconcile_archive(archive(), aborted=True, now=10_100.0)
    again = rt.reconcile_archive(archive(), aborted=True, now=10_200.0)
    assert frozen(first) == frozen(again)


def test_n2_n11_a_late_failure_forfeits_the_other_open_env_and_says_so(tmp_path):
    rt = runtime(tmp_path)
    math = explore(rt, hotkey="cheat", prompt=1)
    code = explore(rt, hotkey="cheat", prompt=2, env=CODE)
    draw(rt)
    rt.finalize_exploration(1, environment=MATH, now=10_050.0)    # math row: drawn, never audited
    assert rt.log.is_scanned(CODE, 2)
    outcome = rt.record_audit(math["observation_id"], passed=False, now=10_100.0)
    assert outcome == AuditOutcome("failed", (code["observation_id"],)) and outcome.failed
    assert rt.exploration_banned("cheat", now=10_101.0)
    assert not rt.log.is_scanned(CODE, 2)
    assert events(rt, "settle", code["observation_id"])[-1]["status"] == "exploration_forfeited"
    assert rt.record_audit(math["observation_id"], passed=True, now=10_110.0) == AuditOutcome("not_applied")
    assert rt.record_audit(code["observation_id"], passed=True, now=10_120.0).kind == "not_applied"


# ---------------------------------------------------------------- Task 7 review: I1, R17, I2, M1-M5

def settle_event(rt, result):
    return events(rt, "settle", result["observation_id"])[-1]


def scan_holder(rt, prompt, env=MATH):
    row = rt.db.execute("SELECT first_id FROM run_scans WHERE environment=? AND prompt_idx=?", (env, prompt)).fetchone()
    return None if row is None else row[0]


def release_unaudited(rt, held):
    rt.finalize_exploration(1, environment=MATH, now=10_050.0)          # never drawn, never audited


def release_failed_audit(rt, held):
    assert audited(rt, held, passed=False).failed


def release_horizon(rt, held):
    # "a" leaves probation, then holds a drawn row never audited: its later not-drawn row is unpaid too.
    assert [r["audit"] for r in rt.ledger.rows(1, environment=MATH)] == ["passed", "queued", "not_drawn"]
    rt.finalize_exploration(1, environment=MATH, now=10_050.0)
    assert [r["audit"] for r in rt.ledger.rows(1, environment=MATH)] == ["passed", "unaudited", "unaudited"]


def release_aborted(rt, held):
    rt.reconcile_archive(archive(), aborted=True, now=10_050.0)


def release_late_failure(rt, held):
    # the hotkey's CODE row was finalized drawn-unaudited; its late failure forfeits the open MATH row
    code = explore(rt, hotkey="a", prompt=90, env=CODE)
    draw(rt)
    rt.finalize_exploration(1, environment=CODE, now=10_040.0)
    outcome = rt.record_audit(code["observation_id"], passed=False, now=10_050.0)
    assert outcome.failed and held["observation_id"] in outcome.forfeited


@pytest.mark.parametrize("release", [release_unaudited, release_failed_audit, release_horizon, release_aborted,
                                     release_late_failure])
def test_i1_a_trained_prompt_stays_scanned_whatever_releases_the_exploration_that_saw_it_first(tmp_path, release):
    """I1: A's exploration holds the scan of P, B trains P (another window), A ends unpaid -> P stays scanned."""
    contract = reward_contract(new_hotkey_audit_groups=1, audit_bps=0) if release is release_horizon else None
    rt = runtime(tmp_path, contract=contract)
    if release is release_horizon:
        passing = explore(rt, hotkey="a", prompt=50, now=100.0)
        explore(rt, hotkey="a", prompt=51, now=100.0)                    # drawn (probation), never audited
        draw(rt, now=200.0)
        assert rt.record_audit(passing["observation_id"], passed=True, now=200.0).passed   # out of probation
    held = explore(rt, hotkey="a", prompt=7, now=300.0)
    if release is release_horizon:
        draw(rt)
    assert held["entitled"] and scan_holder(rt, 7) == held["observation_id"]
    open_window(rt, 2)
    trained = train(rt, hotkey="b", prompt=7, window=2, now=310.0)
    assert trained["inserted"] and trained["first_scan"] is False
    release(rt, held)
    audit, status = rt.ledger.state(held["observation_id"])
    assert status != "reserved" or audit == "unaudited"                 # A's row ended unpaid
    assert rt.log.is_scanned(MATH, 7) is True                           # NOT "never scanned" again
    assert scan_holder(rt, 7) == trained["observation_id"]              # re-seated on the training observation
    assert rt.log.first_scan_stats(MATH)[0] >= 1 and rt.log.first_scan_stats(MATH)[1] == 1   # counted, in-zone
    again = explore(rt, hotkey="c", prompt=7, window=2, now=10_060.0)
    assert (again["entitled"], again["reason"]) == (False, "already_scanned")
    # ...and it stays so once the training window is settled as trained
    rt.reconcile_archive(archive(window=2, batch=[("b", MATH, 7)]), now=10_100.0)
    open_window(rt, 3)
    assert explore(rt, hotkey="d", prompt=7, window=3, now=10_200.0)["reason"] == "already_scanned"


def test_i1_without_any_training_observation_a_released_prompt_is_free_again(tmp_path):
    rt = runtime(tmp_path)
    held = explore(rt, hotkey="a", prompt=7)
    rt.finalize_exploration(1, environment=MATH, now=10_050.0)
    assert rt.log.is_scanned(MATH, 7) is False and scan_holder(rt, 7) is None
    assert settle_event(rt, held)["status"] == "exploration_unpaid" and "reason" not in settle_event(rt, held)


def test_i1_the_scan_is_seated_on_the_earliest_surviving_training_observation(tmp_path):
    rt = runtime(tmp_path)
    held = explore(rt, hotkey="a", prompt=7)
    open_window(rt, 2)
    open_window(rt, 3)
    first = train(rt, hotkey="b", prompt=7, window=2, now=310.0)
    second = train(rt, hotkey="c", prompt=7, window=3, now=320.0)
    rt.finalize_exploration(1, environment=MATH, now=10_050.0)
    assert scan_holder(rt, 7) == first["observation_id"]
    # window 2 aborts: it trained nothing, its observation stops counting; window 3's takes the seat
    rt.reconcile_archive(archive(window=2), aborted=True, now=10_100.0)
    assert scan_holder(rt, 7) == second["observation_id"] and rt.log.is_scanned(MATH, 7)
    assert settle_event(rt, first)["status"] == "proven_unpaid"
    assert held["observation_id"] != scan_holder(rt, 7)


def test_an_aborted_window_trains_nothing_so_its_training_observations_do_not_keep_the_prompt(tmp_path):
    rt = runtime(tmp_path)
    alone = train(rt, hotkey="b", prompt=7)                             # holds the scan of 7
    held = explore(rt, hotkey="a", prompt=8)
    train(rt, hotkey="b", prompt=8, now=130.0)                          # 8: exploration first, then trained
    assert scan_holder(rt, 7) == alone["observation_id"]
    result = rt.reconcile_archive(archive(batch=[("b", MATH, 7), ("b", MATH, 8)]), aborted=True, now=10_100.0)
    assert result["window_status"] == "aborted"
    assert {e["status"] for e in events(rt, "settle") if e["proof"] == "proven"} == {"proven_unpaid"}
    assert settle_event(rt, held)["status"] == "exploration_unpaid"
    assert rt.log.is_scanned(MATH, 7) is False and rt.log.is_scanned(MATH, 8) is False
    open_window(rt, 2)
    for prompt in (7, 8):                                               # both are first scans again
        later = explore(rt, hotkey="c", prompt=prompt, window=2, now=10_200.0)
        assert later["entitled"] and later["first_scan"]


def test_a_proven_group_left_out_of_the_batch_does_not_keep_the_prompt_a_trained_one_does(tmp_path):
    rt = runtime(tmp_path)
    won = train(rt, hotkey="a", prompt=20)
    lost = train(rt, hotkey="b", prompt=21)
    assert rt.log.is_scanned(MATH, 21)                                  # scanned while it may still be trained
    assert explore(rt, hotkey="x", prompt=21)["reason"] == "already_scanned"
    rt.reconcile_archive(archive(batch=[("a", MATH, 20)]), now=10_100.0)
    assert settle_event(rt, won)["status"] == "trained" and settle_event(rt, lost)["status"] == "proven_unpaid"
    assert rt.log.is_scanned(MATH, 20) is True and rt.log.is_scanned(MATH, 21) is False
    open_window(rt, 2)
    assert explore(rt, hotkey="x", prompt=20, window=2, now=10_200.0)["reason"] == "already_scanned"
    assert explore(rt, hotkey="x", prompt=21, window=2, now=10_200.0)["entitled"] is True


LOW, HIGH = range(M_ROLLOUTS), range(M_ROLLOUTS, 2 * M_ROLLOUTS)


@pytest.mark.parametrize("trainer", ["x", "other"], ids=["same-hotkey-both-lanes", "two-hotkeys"])
def test_r17_exploration_on_a_prompt_trained_in_the_same_window_is_unpaid_without_sanction(tmp_path, trainer):
    """One operator, 32 public seeds of P: an all-fail subset sent as exploration, an in-zone subset
    sent for training. Paid: the training group only (never 1.15 groups)."""
    rt = runtime(tmp_path)
    probe = explore(rt, hotkey="x", prompt=7, seeds=LOW)                # exploration FIRST: entitled for now
    clean = explore(rt, hotkey="x", prompt=8)
    assert probe["entitled"] and probe["first_scan"]
    trained = train(rt, hotkey=trainer, prompt=7, seeds=HIGH, now=130.0)
    assert trained["inserted"] and trained["first_scan"] is False
    draw(rt)
    assert [row["prompt_idx"] for row in rt.queued_audits(1)] == [8]    # the doomed row needs no audit
    assert rt.exploration_backlog(1)["queued"] == 1
    assert rt.record_audit(clean["observation_id"], passed=True, now=10_000.0).passed
    released = rt.finalize_exploration(1, environment=MATH, now=10_050.0)
    assert released == [probe["observation_id"]]
    settle = settle_event(rt, probe)
    assert (settle["status"], settle["reason"]) == ("exploration_unpaid", "trained")
    assert rt.ledger.payable(1, environment=MATH) == {"x": 1}           # prompt 8 only
    assert rt.ledger.rows(1, environment=MATH)[0]["status"] == "trained"
    assert not rt.exploration_banned("x", now=10_051.0)                 # no sanction
    assert rt.ledger.passed_audits("x") == 1                            # probation neither lost nor gained
    assert scan_holder(rt, 7) == trained["observation_id"]              # the scan moved to the training group
    result = rt.reconcile_archive(archive(batch=[(trainer, MATH, 7)]), now=10_100.0)
    assert result["service_exploration_by_environment"] == {MATH: {"x": 1}}
    total = sum(result["rewards_by_hotkey"].values())
    assert total == pytest.approx(POOL / T + PRICE)                     # one training group + prompt 8
    settle = settle_event(rt, probe)                                    # the final status says why, still
    assert (settle["status"], settle["reason"]) == ("exploration_unpaid", "trained")
    assert not any(e["status"] == "exploration_paid" and e["id"] == probe["observation_id"] for e in events(rt))
    validate_service_archive_v2(result, rt.contract, cap=1.0, picks_target=PICKS, batch_slots=SLOTS)


def test_r17_training_first_then_exploration_is_refused_at_once(tmp_path):
    rt = runtime(tmp_path)
    train(rt, hotkey="x", prompt=7, seeds=HIGH)
    for hotkey in ("x", "other"):
        late = explore(rt, hotkey=hotkey, prompt=7, seeds=LOW, now=130.0)
        assert (late["entitled"], late["status"], late["reason"]) == (False, "exploration_unpaid", "already_scanned")
    assert rt.ledger.rows(1, environment=MATH) == []


def test_r17_a_voided_row_sets_no_audit_horizon(tmp_path):
    rt = runtime(tmp_path, contract=reward_contract(new_hotkey_audit_groups=1, audit_bps=0))
    seasoned = explore(rt, hotkey="x", prompt=1, now=100.0)             # forced (probation)
    doomed = explore(rt, hotkey="x", prompt=7, now=110.0)               # forced too: drawn, will stay queued
    draw(rt)
    assert rt.record_audit(seasoned["observation_id"], passed=True, now=10_000.0).passed
    behind = explore(rt, hotkey="x", prompt=9, now=10_010.0)            # past probation, later round: not drawn
    draw(rt, now=20_000.0)
    train(rt, hotkey="t", prompt=7, now=20_010.0)
    assert [r["audit"] for r in rt.ledger.rows(1, environment=MATH)] == ["passed", "queued", "not_drawn"]
    released = rt.finalize_exploration(1, environment=MATH, now=20_050.0)
    assert released == [doomed["observation_id"]]                       # NOT the row behind it
    assert rt.ledger.payable(1, environment=MATH) == {"x": 2}
    assert settle_event(rt, doomed)["reason"] == "trained" and behind["entitled"]
    assert not rt.exploration_banned("x", now=20_051.0)


def test_r17_also_catches_a_training_group_recorded_after_the_env_was_finalized(tmp_path):
    rt = runtime(tmp_path)
    probe = explore(rt, hotkey="x", prompt=7, seeds=LOW)
    audited(rt, probe)
    rt.finalize_exploration(1, environment=MATH, now=10_050.0)
    assert rt.ledger.payable(1, environment=MATH) == {"x": 1}
    train(rt, hotkey="x", prompt=7, seeds=HIGH, now=10_060.0)           # the window is not settled yet
    result = rt.reconcile_archive(archive(batch=[("x", MATH, 7)]), now=10_100.0)
    assert result["service_exploration_by_environment"] == {}
    assert result["rewards_by_hotkey"] == {"x": pytest.approx(POOL / T)}
    settle = settle_event(rt, probe)
    assert (settle["status"], settle["proof"], settle["reason"]) == ("exploration_unpaid", "audited", "trained")


def test_same_hotkey_with_two_subsets_of_one_prompt_is_two_observations_and_one_first_scan(tmp_path):
    rt = runtime(tmp_path)
    first = explore(rt, hotkey="x", prompt=7, seeds=LOW)
    second = explore(rt, hotkey="x", prompt=7, seeds=HIGH, now=130.0)
    assert second["observation_id"] != first["observation_id"] and second["inserted"]
    assert first["entitled"] and (second["entitled"], second["reason"]) == (False, "already_scanned")
    assert len(rt.ledger.rows(1, environment=MATH)) == 1 and len(events(rt, "observation")) == 2
    assert events(rt, "observation", second["observation_id"])[0]["candidate"]["seeds"] == list(HIGH)


def test_m1_a_pass_on_a_row_forfeited_meanwhile_publishes_nothing(tmp_path):
    rt = runtime(tmp_path)
    bad = explore(rt, hotkey="cheat", prompt=1)
    good = explore(rt, hotkey="cheat", prompt=2)
    draw(rt)
    assert rt.record_audit(bad["observation_id"], passed=False, now=10_000.0).failed
    assert settle_event(rt, good)["status"] == "exploration_forfeited"
    count = len(events(rt))
    assert rt.record_audit(good["observation_id"], passed=True, now=10_001.0) == AuditOutcome("not_applied")
    assert len(events(rt)) == count                                     # no pending/audited after the forfeit
    assert settle_event(rt, good)["status"] == "exploration_forfeited"


def test_m1_the_runtime_itself_never_publishes_pending_for_a_row_that_is_not_entitled(tmp_path, monkeypatch):
    rt = runtime(tmp_path)
    bad = explore(rt, hotkey="cheat", prompt=1)
    good = explore(rt, hotkey="cheat", prompt=2)
    draw(rt)
    rt.record_audit(bad["observation_id"], passed=False, now=10_000.0)
    count = len(events(rt))
    monkeypatch.setattr(runtime_module, "apply_exploration_verdict", lambda *a, **k: ("passed", []))
    rt.record_audit(good["observation_id"], passed=True, now=10_001.0)   # even if the ledger said "passed"
    assert len(events(rt)) == count


HUGE = 10**400


def test_m3_a_huge_integer_is_a_clean_refusal_everywhere(tmp_path):
    rt = runtime(tmp_path)
    group_id, candidate = group(rt)
    base = dict(environment=MATH, prompt_idx=7, hotkey="a", window=1, rewards=ZERO, group_id=group_id,
                candidate=candidate, token_count=10, now=120.0)
    bad = [dict(base, now=HUGE), dict(base, rewards=[HUGE] * M_ROLLOUTS), dict(base, token_count=HUGE),
           dict(base, window=HUGE), dict(base, prompt_idx=HUGE), dict(base, now=-HUGE)]
    for kwargs in bad:
        with pytest.raises(ServicePolicyLimit):
            rt.record_exploration(**kwargs)
        with pytest.raises(ServicePolicyLimit):
            rt.record_training(**kwargs)
        assert not rt.db.in_transaction
    for arrival in (HUGE, -HUGE, float("inf"), "soon", True):
        with pytest.raises(ServicePolicyLimit):
            rt.record_exploration(**base, arrived_at=arrival)
    for pools in ({MATH: HUGE, CODE: POOL}, {MATH: -HUGE, CODE: POOL}):
        with pytest.raises(ValueError):
            rt.open_window(2, pools=pools, picks_target=PICKS, batch_slots=SLOTS, now=1.0)
    with pytest.raises(ValueError):
        rt.open_window(2, pools={MATH: POOL, CODE: POOL}, picks_target=PICKS, batch_slots=SLOTS, now=HUGE)
    with pytest.raises(ValueError):
        rt.active(now=HUGE)
    assert rt.record_audit("f" * 64, passed=True, now=HUGE) == AuditOutcome("not_applied")
    assert events(rt) == [] and not rt.db.in_transaction
    assert rt.record_exploration(**base)["entitled"] is True            # the runtime is intact


def test_m4_windows_open_in_increasing_order_and_a_frozen_one_reopens(tmp_path):
    rt = runtime(tmp_path)                                              # window 1
    first = rt.envelope(1)
    changed = next_schedule(rt.contract, rt.schedule, active=(MATH,), shares={MATH: 10000})
    rt.apply_schedule(changed, request_id="r1", window=1)
    five = rt.open_window(5, pools={MATH: 0.5}, picks_target=PICKS, batch_slots=SLOTS, now=10.0)
    assert five["schedule"]["revision"] == 1
    for window in (2, 4):                                               # an old window born with the new schedule
        with pytest.raises(ValueError, match="increasing order"):
            rt.open_window(window, pools={MATH: 0.5}, picks_target=PICKS, batch_slots=SLOTS, now=11.0)
        with pytest.raises(ServicePolicyLimit):
            rt.envelope(window)
    # restart: an already frozen window, even an old one, gives its frozen envelope back
    assert rt.open_window(1, pools={MATH: POOL, CODE: POOL}, picks_target=PICKS, batch_slots=SLOTS, now=12.0) == first
    assert rt.open_window(5, pools={MATH: 0.5}, picks_target=PICKS, batch_slots=SLOTS, now=12.0) == five
    assert rt.open_window(6, pools={MATH: 0.5}, picks_target=PICKS, batch_slots=SLOTS, now=13.0)["schedule"]["revision"] == 1


def test_m5_an_arrival_before_the_window_opened_is_refused(tmp_path):
    path = tmp_path / "runtime.sqlite3"
    rt = build(path)
    rt.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision="d" * 40)
    open_window(rt, 1, now=1_000.0)
    with pytest.raises(ServicePolicyLimit, match="precedes the opening"):
        explore(rt, prompt=1, now=1_200.0, arrived_at=999.0)            # would pick a round already public
    with pytest.raises(ServicePolicyLimit, match="precedes the opening"):
        explore(rt, prompt=1, now=990.0)                                # the clamp to now cannot go below it either
    assert events(rt) == [] and rt.ledger.rows(1, environment=MATH) == []
    assert explore(rt, prompt=1, now=1_200.0, arrived_at=1_000.0)["draw_round"] == drand_round(1_000.0) + 2
    # the opening instant is the FIRST one, and it survives a restart and a reopen
    rt.close()
    rt = build(path)
    open_window(rt, 1, now=5_000.0)
    assert rt.db.execute("SELECT opened_at FROM service_windows WHERE window=1").fetchone()[0] == 1_000.0
    assert explore(rt, prompt=2, now=5_100.0, arrived_at=1_500.0)["entitled"] is True
    with pytest.raises(ServicePolicyLimit, match="precedes the opening"):
        explore(rt, prompt=3, now=5_100.0, arrived_at=999.9)


# ---------------------------------------------------------------- I2: the first settlement is the truth

def test_i2_a_crash_after_settlement_then_a_recovery_that_rebuilds_the_window_aborted_cannot_brick(tmp_path, monkeypatch):
    path = tmp_path / "runtime.sqlite3"
    rt = runtime(tmp_path)
    paid = explore(rt, hotkey="x", prompt=1)
    audited(rt, paid)
    train(rt, prompt=20, hotkey="a")
    rt.finalize_exploration(1, environment=MATH, now=10_050.0)
    first = rt.reconcile_archive(archive(batch=[("a", MATH, 20)]), now=10_100.0)   # settled, not aborted...
    assert first["rewards_by_hotkey"] == {"a": pytest.approx(POOL / T), "x": pytest.approx(PRICE)}
    count = len(events(rt))
    rt.close()                                                          # ...then the validator dies

    rt = build(path)                                                    # restart: fill_closed_recovery.recover
    errors = []
    monkeypatch.setattr(runtime_module.logger, "error", lambda *args: errors.append(args))
    rebuilt = {"window_start": 1, "window_status": "aborted", "batch": [], "rewards_by_hotkey": {},
               "failure_stage": "active_window_recovery"}
    for _ in range(2):                                                  # at every start, not only the first
        recovered = rt.reconcile_archive(rebuilt, aborted=True, now=20_000.0)
        assert frozen(recovered) == frozen(first)                       # byte-identical frozen money
        assert recovered["window_status"] == "complete"                 # the frozen disposition: NOT aborted
        assert recovered["failure_stage"] == "active_window_recovery" and recovered["batch"] == []
        assert recovered["service_training_recomputed_delta"] == first["service_training_recomputed_delta"]
    assert len(errors) == 2 and "already settled" in errors[0][0]
    assert len(events(rt)) == count and not rt.db.in_transaction        # nothing published, nothing written
    assert rt.log.is_scanned(MATH, 1) and rt.log.is_scanned(MATH, 20)   # paid scans are not given back
    assert rt.db.execute("SELECT aborted FROM service_settled WHERE window=1").fetchone()[0] == 0
    # a recovered-partial archive (same disposition, another batch) and one that cannot even be settled
    partial = rt.reconcile_archive({**rebuilt, "window_status": "recovered_partial"}, now=20_100.0)
    assert frozen(partial) == frozen(first) and partial["window_status"] == "recovered_partial"
    broken = rt.reconcile_archive({"window_start": 1, "batch": [{"hotkey": 5}], "rewards_by_hotkey": {}}, now=20_200.0)
    assert frozen(broken) == frozen(first) and len(errors) == 4
    # the same archive as the first time: same bytes, and no error
    assert json.dumps(rt.reconcile_archive(archive(batch=[("a", MATH, 20)]), now=20_300.0), sort_keys=True) == \
        json.dumps(first, sort_keys=True)
    assert len(errors) == 4
    rt.close()


def test_i2_a_window_frozen_as_aborted_stays_aborted(tmp_path, monkeypatch):
    rt = runtime(tmp_path)
    audited(rt, explore(rt, hotkey="x", prompt=1))
    train(rt, prompt=20, hotkey="a")
    first = rt.reconcile_archive(archive(), aborted=True, now=10_100.0)
    errors = []
    monkeypatch.setattr(runtime_module.logger, "error", lambda *args: errors.append(args))
    later = rt.reconcile_archive(archive(batch=[("a", MATH, 20)]), now=10_200.0)
    assert later["window_status"] == "aborted" and frozen(later) == frozen(first) and later["rewards_by_hotkey"] == {}
    assert len(errors) == 1 and rt.log.is_scanned(MATH, 20) is False


# ---------------------------------------------------------------- restart

def test_restart_reuses_the_stored_draw_beacon_whatever_the_new_beacon_source_says(tmp_path):
    path = tmp_path / "runtime.sqlite3"
    contract = reward_contract(new_hotkey_audit_groups=0, audit_bps=5000)   # the beacon decides
    rt = runtime(tmp_path, contract=contract)
    for prompt in range(12):
        explore(rt, prompt=prompt, hotkey=f"h{prompt % 6}", now=120.0)       # all at draw round 42
    half = [prompt for prompt in range(6)]
    # Draw the first six rows only (the others are held back as if their env pass had not run yet).
    rt.db.execute("UPDATE exploration_entitlements SET draw_round=99 WHERE prompt_idx>=6")
    rt.db.commit()
    rt.resolve_draws(1, beacon_for_round=lambda r: BEACON, now=200.0)
    before = {r["prompt_idx"]: r["audit"] for r in rt.ledger.rows(1, environment=MATH) if r["prompt_idx"] in half}
    assert set(before.values()) <= {"queued", "not_drawn"}
    rt.db.execute("UPDATE exploration_entitlements SET draw_round=42 WHERE prompt_idx>=6")
    rt.db.commit()
    ids = {r["prompt_idx"]: r["observation_id"] for r in rt.ledger.rows(1, environment=MATH)}
    rt.close()

    from reliquary.services.exploration import audit_selected
    rt = build(path, contract)
    asked = []
    other = "ee" * 32                                                       # the reopened source disagrees
    assert other != BEACON
    rt.resolve_draws(1, beacon_for_round=lambda r: asked.append(r) or other, now=10_000.0)
    assert asked == []                                                      # round 42 is stored: never asked again
    assert rt.db.execute("SELECT round, randomness FROM service_draw_beacons").fetchall() == [(42, BEACON)]
    after = {r["prompt_idx"]: r["audit"] for r in rt.ledger.rows(1, environment=MATH)}
    assert {p: after[p] for p in half} == before                            # drawn rows did not move
    expected = {p: ("queued" if audit_selected(beacon_randomness=BEACON, observation_id=ids[p], audit_bps=5000,
                                               forced=False) else "not_drawn") for p in range(12)}
    assert after == expected                                                # every draw used the STORED beacon
    with_other = {p: ("queued" if audit_selected(beacon_randomness=other, observation_id=ids[p], audit_bps=5000,
                                                 forced=False) else "not_drawn") for p in range(12)}
    assert with_other != expected                                           # (the other beacon draws differently)
    rt.close()


def test_restart_between_finalize_and_reconcile_settles_what_was_finalized(tmp_path):
    path = tmp_path / "runtime.sqlite3"
    rt = runtime(tmp_path)
    paid = explore(rt, hotkey="x", prompt=1, now=120.0)
    lost = explore(rt, hotkey="y", prompt=2, now=9_000.0)                # never drawn
    rt.resolve_draws(1, beacon_for_round=lambda r: BEACON, now=200.0)
    rt.record_audit(paid["observation_id"], passed=True, now=201.0)
    train(rt, prompt=20, hotkey="a")
    for env in (MATH, CODE):
        rt.finalize_exploration(1, environment=env, now=9_100.0)
    count = len(events(rt))
    rt.close()

    rt = build(path)
    assert rt.ledger.is_finalized(1, environment=MATH) and rt.ledger.payable(1, environment=MATH) == {"x": 1}
    assert rt.queued_audits(1) == [] and rt.pending_draw_rounds(1) == []
    assert rt.record_audit(lost["observation_id"], passed=True, now=9_200.0) == AuditOutcome("not_applied")
    result = rt.reconcile_archive(archive(batch=[("a", MATH, 20)]), now=9_300.0)
    assert result["rewards_by_hotkey"] == {"a": pytest.approx(POOL / T), "x": pytest.approx(PRICE)}
    assert settle_event(rt, paid)["status"] == "exploration_paid"
    assert settle_event(rt, lost)["status"] == "exploration_unpaid"
    assert len(events(rt)) == count + 2                                  # exploration_paid + trained, nothing twice
    assert rt.log.is_scanned(MATH, 1) and not rt.log.is_scanned(MATH, 2)
    rt.close()


def test_a_legacy_sampling_env_records_and_pays_without_a_pool_selection(tmp_path):
    value = contract_v2_dict()
    value["environments"][CODE]["sampling"] = {"kind": "legacy/v1"}
    rt = runtime(tmp_path, contract=ServiceContract.from_dict(value))
    assert rt.seed_pool(environment=CODE, prompt_idx=3, window=1) is None
    assert rt.seed_pool(environment=MATH, prompt_idx=3, window=1) is not None
    base = dict(environment=CODE, prompt_idx=3, window=1, group_id="1" * 64, candidate=None, token_count=10)
    probe = rt.record_exploration(**base, hotkey="x", rewards=ZERO, now=120.0)
    assert probe["entitled"] and probe["first_scan"]
    assert events(rt, "observation", probe["observation_id"])[0]["candidate"] is None
    won = rt.record_training(**dict(base, prompt_idx=4, group_id="2" * 64), hotkey="a", rewards=HALF, now=121.0)
    assert won["inserted"] and won["first_scan"]
    with pytest.raises(ServicePolicyLimit, match="no public seed pool"):   # a selection where there is no pool
        rt.record_exploration(**dict(base, prompt_idx=5, candidate=group(rt)[1]), hotkey="x", rewards=ZERO, now=122.0)
    audited(rt, probe)
    result = rt.reconcile_archive(archive(batch=[("a", CODE, 4)]), now=10_100.0)
    assert result["rewards_by_hotkey"] == {"a": pytest.approx(POOL / T), "x": pytest.approx(PRICE)}
    validate_service_archive_v2(result, rt.contract, cap=1.0, picks_target=PICKS, batch_slots=SLOTS)
