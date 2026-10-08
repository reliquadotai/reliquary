# tests/unit/test_service_exploration_lane.py
"""The exploration lane in the batcher (decision C): lane choice, ledger, audit plan, seal drain.

Everything here runs a REAL ``ServiceRuntime`` (SQLite) behind a real ``GrpoWindowBatcher``; only the
GPU proof (``_verify_expensive``) and the drand clock are replaced. M_ROLLOUTS is the local
profile's; the tests are written in terms of the constants.
"""
import asyncio
import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from reliquary import constants
from reliquary.constants import M_ROLLOUTS, SERVICE_EXPLORATION_AUDIT_INFLIGHT, SERVICE_EXPLORATION_AUDIT_PRIORITY
from reliquary.protocol.service_contract import ServiceContract
from reliquary.protocol.submission import RejectReason
from reliquary.services import runtime as runtime_module
from reliquary.services.runtime import ServiceRuntime, protocol_slot_geometry
from reliquary.validator import batcher as batcher_module
from reliquary.validator.batcher import (
    GrpoWindowBatcher, PendingSubmission, _ProvenGroup, audit_scheduler_environment,
)
from reliquary.validator.fill_window import FillState
from reliquary.validator.proof_scheduler import (
    GlobalProofScheduler, ProofDecision, ProofDecisionStatus, ProofPlan, RankedProof,
)
from tests.unit.service_v2_fixtures import CODE, MATH, contract_v2, contract_v2_dict, qualification_v2
from tests.unit.test_grpo_window_batcher import FakeEnv, _make_batcher

PICKS, SLOTS = protocol_slot_geometry()
POOL = 0.25
ZERO = [0.0] * M_ROLLOUTS
HALF = [1.0] * (M_ROLLOUTS // 2) + [0.0] * (M_ROLLOUTS - M_ROLLOUTS // 2)
# In zone as observed, but the one failure is uncertain: it may have been a success, i.e. a uniform group.
FRAGILE = [1.0] * (M_ROLLOUTS - 1) + [0.0]
LAST = M_ROLLOUTS - 1
BEACON = "cd" * 32
WINDOW_BEACON = "ab" * 32
BINARY = (0.0, 1.0)
CLOCK = {"round": 1_000}


class MathEnv(FakeEnv):
    name = MATH


@pytest.fixture(autouse=True)
def _drand(monkeypatch):
    """A controllable drand: the round clock, and a beacon for every round (the draw itself is
    the runtime's; ``tests/unit/test_service_runtime_v2.py`` covers its determinism)."""
    CLOCK["round"] = 1_000
    monkeypatch.setattr(runtime_module, "_verified_beacon", lambda round_id: BEACON)


def make_runtime(tmp_path, contract=None):
    contract = contract or contract_v2()
    tmp_path.mkdir(parents=True, exist_ok=True)
    rt = ServiceRuntime(tmp_path / "runtime.sqlite3", contract, qualification_v2(contract), now=time.time() - 10,
                        drand_round_at=lambda instant: CLOCK["round"])
    rt.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision="d" * 40)
    rt.open_window(1, pools={MATH: POOL, CODE: POOL}, picks_target=PICKS, batch_slots=SLOTS,
                   now=time.time() - 5)
    rt.announcement(window=1, randomness=WINDOW_BEACON)
    return rt


def reward_contract(**reward):
    value = contract_v2_dict()
    value["policies"]["reward"].update(reward)
    return ServiceContract.from_dict(value)


def make_batcher(rt, *, scheduler=None, **kwargs):
    b = _make_batcher(window_start=1, env=MathEnv(), **kwargs)
    b.service_runtime, b.service_environment = rt, MATH
    b.service_policy = rt.announcement(window=1, randomness=WINDOW_BEACON, environment=MATH)
    b.current_checkpoint_hash = "d" * 40
    b.fill_state = FillState(budgets={MATH: 64}, picks_target=PICKS)
    b.training_plans = []
    b._extend_proof_plan = lambda candidates: b.training_plans.extend(candidates)
    if scheduler is None:
        scheduler = MagicMock()
        scheduler.submit.return_value.decisions.return_value = ()
    b._proof_scheduler = scheduler
    return b


def make_pending(rt, *, prompt=7, hotkey="hk", rewards=ZERO, seeds=None, purpose="exploration", **fields):
    pool = rt.seed_pool(environment=MATH, prompt_idx=prompt, window=1)
    selection = pool.selection(list(range(M_ROLLOUTS)) if seeds is None else list(seeds))
    rollouts = [SimpleNamespace(commit={"tokens": [1] + [2] * 10, "rollout": {"prompt_length": 1}})
                for _ in range(M_ROLLOUTS)]
    request = SimpleNamespace(miner_hotkey=hotkey, service_binding={"purpose": purpose},
                              pool_selection=selection.to_dict(), rollouts=rollouts, _grading_refundable=False)
    root = f"{prompt}:{hotkey}".encode().ljust(32, b"\0")
    values = dict(hotkey=hotkey, prompt_idx=prompt, request=request, rewards=list(rewards), drand_round=3,
                  merkle_root=root, selection_digest=root, attainable_rewards=BINARY,
                  telemetry=SimpleNamespace(t_body_completed=500.0, arrival_drand_round=None),
                  decision_ts=600.0)
    values.update(fields)
    return PendingSubmission(**values)


def arrive(b, pending):
    b._submit_arrival_proof(pending)
    b.flush_service_admissions()
    return b.difficulty_auction_metadata_by_id[id(pending)]


def tick(b, *, drain=False):
    b.service_exploration_tick(drain=drain)


def events(rt, identity=None):
    return [e for _, e in rt.events(limit=10_000) if identity is None or e["id"] == identity]


def ready(rt):
    """Let every pending draw fall due (the drand clock moves past every draw round)."""
    CLOCK["round"] += 100


# ---------------------------------------------------------------- Review Focus 1 and 5

def test_exploration_never_reserves_training_capacity(tmp_path):
    rt = make_runtime(tmp_path)
    b = make_batcher(rt)
    before = b.fill_state.snapshot()
    for prompt in range(3):
        pending = make_pending(rt, prompt=prompt)
        row = arrive(b, pending)
        assert row["status"] == "exploration_pending"
        assert pending.request._grading_refundable is True         # the productive admission budget is refunded
    after = b.fill_state.snapshot()
    assert after["in_flight"] == before["in_flight"] and after["admitted"] == before["admitted"]
    assert b.training_plans == []                                  # and the training proof plan never saw them
    b._proof_scheduler.submit.assert_not_called()


def test_two_hotkeys_on_one_never_scanned_prompt_in_one_window_one_is_entitled(tmp_path):
    rt = make_runtime(tmp_path)
    b = make_batcher(rt)
    first = arrive(b, make_pending(rt, hotkey="a"))
    second = arrive(b, make_pending(rt, hotkey="b"))
    assert first["status"] == "exploration_pending" and first["exploration_fraction"] > 0
    assert second["status"] == "already_scanned" and second["exploration_fraction"] == 0.0
    assert second["service_unpaid_reason"] == "already_scanned"
    published = {e["id"]: e for e in events(rt)}
    assert published[second["service_observation_id"]]["reason"] == "already_scanned"
    assert published[first["service_observation_id"]]["status"] == "exploration_pending"


# ---------------------------------------------------------------- lane from the vector

def test_a_miner_cannot_route_an_in_zone_vector_to_exploration(tmp_path):
    rt = make_runtime(tmp_path)
    b = make_batcher(rt)
    pending = make_pending(rt, rewards=HALF, purpose="exploration")
    b._submit_arrival_proof(pending)
    b.flush_service_admissions()
    assert pending.service_lane == "training"
    assert len(b.training_plans) == 1                              # it goes to the full proof pipeline
    assert events(rt) == []                                        # and is NOT an exploration observation
    assert b.fill_state.snapshot()["in_flight"][MATH] == 1


def test_a_uniform_vector_declared_for_training_is_exploration(tmp_path):
    rt = make_runtime(tmp_path)
    b = make_batcher(rt)
    pending = make_pending(rt, rewards=ZERO, purpose="training")
    row = arrive(b, pending)
    assert pending.service_lane == "exploration" and row["status"] == "exploration_pending"
    assert b.training_plans == [] and b.fill_state.snapshot()["in_flight"][MATH] == 0


def test_an_incomplete_group_is_no_observation(tmp_path):
    rt = make_runtime(tmp_path)
    b = make_batcher(rt)
    pending = make_pending(rt, rewards=ZERO[:-1])
    b._submit_arrival_proof(pending)
    assert b.difficulty_auction_metadata_by_id[id(pending)]["status"] == "utility_ineligible"
    assert events(rt) == [] and b.training_plans == []


def test_r23_an_in_zone_group_that_is_not_robust_is_published_unproven_and_is_not_a_scan(tmp_path):
    rt = make_runtime(tmp_path)
    b = make_batcher(rt)
    pending = make_pending(rt, rewards=FRAGILE, uncertain_indices=(LAST,), purpose="training")
    row = arrive(b, pending)
    assert row["status"] == "service_unproven_published" and row["service_unpaid_reason"] == "not_robust"
    assert b.training_plans == [] and b.fill_state.snapshot()["in_flight"][MATH] == 0     # never trained
    (event,) = events(rt)
    assert (event["status"], event["reason"], event["proof"]) == ("exploration_unpaid", "not_robust", "unproven")
    assert event["uncertain"] == [LAST]                                     # its uncertain position is flagged
    assert rt.ledger.rows(1, environment=MATH) == []                        # no entitlement, nothing paid
    assert not rt.log.is_scanned(MATH, 7)                                   # NOT a scan
    # so the prompt can still be a paid first scan for someone else
    other = arrive(b, make_pending(rt, hotkey="other", prompt=7, rewards=ZERO))
    assert other["status"] == "exploration_pending"


def test_an_uncertain_but_terminated_rollout_is_zeroed_before_classification_and_paid(tmp_path):
    """R24: the published vector is the classified and paid one; a terminated unboxed rollout does not block pay."""
    rt = make_runtime(tmp_path)
    b = make_batcher(rt)
    rewards = ZERO[:]
    pending = make_pending(rt, rewards=rewards, uncertain_indices=(2,))
    row = arrive(b, pending)
    assert row["status"] == "exploration_pending"
    (event,) = events(rt)
    assert event["rewards_bps"] == [0] * M_ROLLOUTS and event["uncertain"] == [2]


def test_r21_two_capped_rollouts_the_robust_rule_decides_training_and_exploration_is_unpaid_truncated(tmp_path):
    rt = make_runtime(tmp_path)
    b = make_batcher(rt)
    # Training: two capped failures of an otherwise balanced group. No count limit; the robust rule
    # (every joint assignment still in zone) admits it.
    training = make_pending(rt, prompt=1, rewards=HALF, truncated_indices=(LAST, LAST - 1),
                            uncertain_indices=(LAST, LAST - 1), purpose="training")
    b._submit_arrival_proof(training)
    assert training.service_lane == "training" and len(b.training_plans) == 1
    # Exploration: any capped rollout makes the group an unpaid observation, never sanctioned.
    capped = make_pending(rt, prompt=2, hotkey="slow", truncated_indices=(LAST, LAST - 1),
                          uncertain_indices=(LAST, LAST - 1))
    row = arrive(b, capped)
    assert row["status"] == "exploration_unpaid" and row["service_unpaid_reason"] == "truncated"
    assert rt.ledger.rows(1, environment=MATH) == []
    assert not rt.log.is_scanned(MATH, 2)                                    # no first-scan slot taken
    assert not rt.exploration_banned("slow")
    (event,) = [e for e in events(rt) if e["id"] == row["service_observation_id"]]
    assert event["reason"] == "truncated" and event["uncertain"] == [LAST - 1, LAST]


def test_a_cut_group_does_not_burn_the_prompt_for_an_honest_one(tmp_path):
    rt = make_runtime(tmp_path)
    b = make_batcher(rt)
    arrive(b, make_pending(rt, hotkey="slow", truncated_indices=(0,), uncertain_indices=(0,)))
    honest = arrive(b, make_pending(rt, hotkey="honest"))
    assert honest["status"] == "exploration_pending"


# ---------------------------------------------------------------- admission details

def test_the_arrival_stamp_is_the_validator_clock_of_the_complete_body_never_a_miner_value(tmp_path):
    rt = make_runtime(tmp_path)
    seen = []
    real = rt.record_exploration
    rt.record_exploration = lambda **kw: (seen.append(kw), real(**kw))[1]
    b = make_batcher(rt)
    arrive(b, make_pending(rt, prompt=1, drand_round=999_999,
                           telemetry=SimpleNamespace(t_body_completed=1234.5, arrival_drand_round=7)))
    arrive(b, make_pending(rt, prompt=2, telemetry=None, decision_ts=777.0))
    assert [kw["arrived_at"] for kw in seen] == [1234.5, 777.0]
    assert all("arrival_round" not in kw and "draw_round" not in kw for kw in seen)


def test_the_pool_and_group_id_come_from_the_announced_selection_not_from_the_batcher_randomness(tmp_path):
    rt = make_runtime(tmp_path)
    b = make_batcher(rt)
    b.randomness = "zz-garbage-not-the-pool-beacon"        # the GRAIL randomness: unrelated to the seed pool
    pending = make_pending(rt, seeds=range(1, 2 * M_ROLLOUTS, 2))
    row = arrive(b, pending)
    assert row["status"] == "exploration_pending"
    (event,) = events(rt)
    assert event["candidate"]["seeds"] == list(range(1, 2 * M_ROLLOUTS, 2))
    # a selection of another pool is refused cleanly by the runtime, nothing recorded
    wrong = make_pending(rt, prompt=9)
    wrong.request.pool_selection = {**wrong.request.pool_selection, "pool_sha256": "0" * 64}
    row = arrive(b, wrong)
    assert row["status"] == "service_policy_limit" and len(events(rt)) == 1


def test_the_runtime_is_never_called_on_the_callers_thread(tmp_path):
    rt = make_runtime(tmp_path)
    threads = []
    real = rt.record_exploration
    rt.record_exploration = lambda **kw: (threads.append(threading.get_ident()), real(**kw))[1]
    b = make_batcher(rt)
    arrive(b, make_pending(rt))
    assert threads and threads[0] != threading.get_ident()


def test_a_refused_observation_is_logged_at_error_with_its_id_and_never_silent(tmp_path, caplog):
    rt = make_runtime(tmp_path)
    b = make_batcher(rt)
    pending = make_pending(rt)
    pending.request.pool_selection = {**pending.request.pool_selection, "pool_sha256": "0" * 64}   # not the announced pool
    with caplog.at_level(logging.ERROR, logger="reliquary"):
        row = arrive(b, pending)
    assert row["status"] == "service_policy_limit"
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("exploration group refused" in m and "observation=" in m and "observation=None" not in m
               for m in errors)
    assert events(rt) == []


def test_a_banned_hotkeys_declared_exploration_is_refused_before_grading_and_a_routed_one_is_published_unpaid(tmp_path):
    rt = make_runtime(tmp_path)
    b = make_batcher(rt)
    first = arrive(b, make_pending(rt, hotkey="cheat"))
    ready(rt)
    tick(b)
    rt.record_audit(first["service_observation_id"], passed=False)
    assert rt.exploration_banned("cheat")
    declared = SimpleNamespace(miner_hotkey="cheat", service_binding={"purpose": "exploration"})
    assert b._service_admission_refusal(declared) == (RejectReason.RATE_LIMITED, "service_exploration_banned")
    assert b._service_admission_refusal(SimpleNamespace(miner_hotkey="cheat", service_binding={"purpose": "training"})) is None
    assert b._service_admission_refusal(SimpleNamespace(miner_hotkey="fine", service_binding={"purpose": "exploration"})) is None
    assert "service_exploration_banned" in batcher_module._NON_PRODUCTIVE_ADMISSION_STAGES
    # the same hotkey sending a uniform vector declared as training still lands in the exploration lane
    row = arrive(b, make_pending(rt, hotkey="cheat", prompt=11, purpose="training"))
    assert row["status"] == "exploration_banned" and row["exploration_fraction"] == 0.0


def test_both_real_reservation_sites_refuse_a_banned_declared_exploration(tmp_path, monkeypatch):
    from reliquary.constants import FORCED_SEED_PROTOCOL_VERSION, PROTOCOL_PROFILE_ID
    rt = make_runtime(tmp_path)
    b = make_batcher(rt)
    first = arrive(b, make_pending(rt, hotkey="cheat"))
    ready(rt)
    tick(b)
    rt.record_audit(first["service_observation_id"], passed=False)
    monkeypatch.setattr("reliquary.services.admission_policy.validate_submission_policy", lambda request, policy: rt.contract)

    def request(hotkey, purpose):
        return SimpleNamespace(miner_hotkey=hotkey, service_binding={"purpose": purpose}, window_start=1,
                               checkpoint_hash="d" * 40, protocol_version=FORCED_SEED_PROTOCOL_VERSION,
                               generation_profile_id=PROTOCOL_PROFILE_ID, prompt_idx=7, _grading_refundable=False)
    banned = b.reserve_prepared_identity(request("cheat", "exploration"), [])
    assert banned == (False, RejectReason.RATE_LIMITED, "service_exploration_banned")
    b.try_reserve_logical_group = lambda r: (True, None)
    assert b.reserve_prepared_identity(request("cheat", "training"), [])[2] != "service_exploration_banned"
    inline = request("cheat", "exploration")
    response = b._accept_locked(inline)
    assert response.accepted is False and response.reason is RejectReason.RATE_LIMITED
    assert inline._grading_refundable is True                  # the stage is non-productive: the budget comes back


def test_late_exploration_after_the_finalize_is_not_recorded(tmp_path):
    rt = make_runtime(tmp_path)
    b = make_batcher(rt)
    b.finalize_service_exploration()
    row = arrive(b, make_pending(rt))
    assert row["status"] == "exploration_window_closed" and events(rt) == []


# ---------------------------------------------------------------- training lane

def _decision(job_id, verified, status=ProofDecisionStatus.PASSED):
    return ProofDecision(job_id=job_id, rank=1, prompt_key=("prompt", 1), status=status, device_id=None,
                         started_at=None, finished_at=None, value=verified)


def _proved(b, pending, verified, monkeypatch, rt):
    monkeypatch.setattr("reliquary.services.admission_policy.validate_submission_policy",
                        lambda request, policy: rt.contract)
    job = f"1:{MATH}:arrival:1"
    b._arrival_proof_meta[job] = (None, 0, "", pending)
    b._open_proof_plan_handle = SimpleNamespace(decisions=lambda: (_decision(job, verified),), done=lambda: False)
    b.fill_state.reserve(MATH)
    b._reconcile_fill_state_decisions(MATH)
    return job


def test_every_batched_training_group_has_a_recorded_observation_first(tmp_path, monkeypatch):
    rt = make_runtime(tmp_path)
    b = make_batcher(rt)
    pending = make_pending(rt, hotkey="t", rewards=HALF, purpose="training")
    verified = SimpleNamespace(hotkey="t")
    b.difficulty_auction_metadata_by_id[id(pending)] = {"rank": 1, "status": "proof_pending"}
    _proved(b, pending, verified, monkeypatch, rt)
    (group,) = b._proven_groups[MATH]
    (event,) = events(rt)
    assert event["status"] == "trained" or event["verdict"] == "in-zone"
    assert pending.service_observation_id == event["id"]
    assert b.difficulty_auction_metadata_by_id[id(pending)]["service_observation_id"] == event["id"]
    assert rt.log.is_scanned(MATH, 7)


def test_a_training_group_refused_by_the_runtime_is_not_batched_and_says_so_loudly(tmp_path, monkeypatch, caplog):
    rt = make_runtime(tmp_path)
    b = make_batcher(rt)
    rt.reconcile_archive({"window_start": 1, "window_status": "complete", "batch": [], "rewards_by_hotkey": {}})
    pending = make_pending(rt, hotkey="t", rewards=HALF, purpose="training")
    b.difficulty_auction_metadata_by_id[id(pending)] = {"rank": 1, "status": "proof_pending"}
    with caplog.at_level(logging.ERROR, logger="reliquary"):
        _proved(b, pending, SimpleNamespace(hotkey="t"), monkeypatch, rt)
    assert b._proven_groups.get(MATH, []) == []                       # never silently batched
    assert b.difficulty_auction_metadata_by_id[id(pending)]["status"] == "service_policy_limit"
    assert b.fill_state.snapshot()["in_flight"][MATH] == 0           # the capacity it held is back
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("NOT batched" in m and "observation=" in m and "observation=None" not in m for m in errors)


def test_the_training_record_is_made_on_the_proof_worker_and_the_reconciliation_reuses_it(tmp_path, monkeypatch):
    rt = make_runtime(tmp_path)
    b = make_batcher(rt)
    calls = []
    real = rt.record_training
    rt.record_training = lambda **kw: (calls.append(threading.get_ident()), real(**kw))[1]
    monkeypatch.setattr("reliquary.services.admission_policy.validate_submission_policy",
                        lambda request, policy: rt.contract)
    pending = make_pending(rt, hotkey="t", rewards=HALF, purpose="training")
    verified = SimpleNamespace(hotkey="t")
    worker = threading.Thread(target=lambda: b._prerecord_service_training(pending, verified))
    worker.start(); worker.join()
    assert len(calls) == 1 and calls[0] != threading.get_ident()
    assert b._service_training_receipt(pending, verified) == pending.service_observation_id
    assert len(calls) == 1                                           # reused, not recorded twice
    assert b._service_training_receipts == {}


def test_the_training_record_is_not_made_for_a_forensic_proof(tmp_path, monkeypatch):
    rt = make_runtime(tmp_path)
    b = make_batcher(rt)
    rt.record_training = MagicMock(side_effect=AssertionError("a forensic proof is no training group"))
    pending = make_pending(rt, hotkey="t", rewards=HALF, purpose="training")
    b._verify_expensive = lambda p, model=None, audit=False: SimpleNamespace(hotkey="t")
    b._execute_scheduled_proof(pending, model=None, count_operator_debt=False)
    rt.record_training.assert_not_called()


# ---------------------------------------------------------------- audit plan

def _seasoned_contract():
    return reward_contract(new_hotkey_audit_groups=1, audit_bps=10000)


def test_drawn_groups_go_to_a_separate_low_priority_audit_plan(tmp_path):
    rt = make_runtime(tmp_path)
    b = make_batcher(rt)
    arrive(b, make_pending(rt, hotkey="a", prompt=1))
    ready(rt)
    tick(b)
    (call,) = b._proof_scheduler.submit.call_args_list
    plan = call.args[0]
    assert isinstance(plan, ProofPlan)
    assert plan.priority == SERVICE_EXPLORATION_AUDIT_PRIORITY == 5          # training is 0
    assert plan.environment == audit_scheduler_environment(MATH) != MATH     # one live plan per scheduler env
    assert plan.open_ended and plan.allow_shortfall and not plan.complete_all
    assert plan.required_passes > plan.max_attempts                        # observational: the target is never met
    assert plan.dispatch_deadline_at is not None and plan.dispatch_deadline_at < plan.deadline_at
    (candidate,) = plan.candidates
    assert candidate.counts_toward_target is False and candidate.resources == ()
    assert candidate.payload.audit is True and candidate.payload.count_operator_debt is False
    assert candidate.payload.pending.hotkey == "a"


def test_audits_are_handed_over_a_few_at_a_time_past_probation_hotkeys_first(tmp_path):
    rt = make_runtime(tmp_path, _seasoned_contract())
    b = make_batcher(rt)
    seasoned = arrive(b, make_pending(rt, hotkey="old", prompt=1))
    ready(rt)
    tick(b)                                                                    # the plan exists now
    rt.record_audit(seasoned["service_observation_id"], passed=True)           # "old" is past probation (1 pass)
    b._audit_open.clear()                                                      # that audit has finished
    b._proof_scheduler.reset_mock()
    new = [arrive(b, make_pending(rt, hotkey="new", prompt=p)) for p in (2, 3, 4)]
    again = arrive(b, make_pending(rt, hotkey="old", prompt=5))
    ready(rt)
    tick(b)
    (call,) = b._proof_scheduler.extend.call_args_list
    candidates = call.args[1]
    assert len(candidates) == SERVICE_EXPLORATION_AUDIT_INFLIGHT               # bounded in flight
    assert candidates[0].payload.pending.hotkey == "old"                       # past probation first
    assert candidates[1].payload.pending.hotkey == "new"
    assert [c.rank for c in candidates] == sorted(c.rank for c in candidates)
    # in flight is full: another tick hands over nothing more until audits finish
    b._proof_scheduler.extend.reset_mock()
    tick(b)
    b._proof_scheduler.extend.assert_not_called()
    handed = {id(c.payload.pending) for c in candidates}
    statuses = sorted(r["status"] for r in b.difficulty_auction_metadata_by_id.values())
    # the first audit's row (finished above, its row not refreshed) + the two handed over, no more
    assert statuses.count("exploration_audit_queued") == 3
    assert b.difficulty_auction_metadata_by_id[id(candidates[0].payload.pending)]["status"] == "exploration_audit_queued"
    assert len(handed) == 2 and again is not None and len(new) == 3


def test_probation_hotkeys_get_a_share_of_the_audits_while_the_window_runs(tmp_path):
    rt = make_runtime(tmp_path, _seasoned_contract())
    b = make_batcher(rt)
    first = arrive(b, make_pending(rt, hotkey="old", prompt=1))
    ready(rt)
    tick(b)
    rt.record_audit(first["service_observation_id"], passed=True)              # "old" is past probation
    b._audit_open.clear()
    for prompt in range(10, 14):
        arrive(b, make_pending(rt, hotkey="old", prompt=prompt))
    arrive(b, make_pending(rt, hotkey="new", prompt=20))
    ready(rt)
    rt.resolve_draws(1, environment=MATH)
    queue = rt.queued_audits(1, environment=MATH)
    assert queue[-1]["hotkey"] == "new" and not queue[-1]["past_probation"]    # strict order puts it last...
    b._audit_submitted.add(first["service_observation_id"])
    b._audit_turn = constants.SERVICE_EXPLORATION_PROBATION_EVERY - 1           # ...but the share lets it in
    assert [row["hotkey"] for row in b._audit_candidates(False, 1)] == ["new"]
    b._audit_turn = constants.SERVICE_EXPLORATION_PROBATION_EVERY - 1
    assert [row["hotkey"] for row in b._audit_candidates(True, 1)] == ["old"]   # a drain is strict
    b._audit_turn = 0
    assert [row["hotkey"] for row in b._audit_candidates(False, 1)] == ["old"]  # the other turns serve the seasoned


def test_the_audit_plan_exists_only_when_a_scheduler_does_and_tells_when_it_cannot_run(tmp_path, caplog):
    rt = make_runtime(tmp_path)
    b = make_batcher(rt)
    b._proof_scheduler = None
    arrive(b, make_pending(rt))
    ready(rt)
    with caplog.at_level(logging.ERROR, logger="reliquary"):
        tick(b)
        tick(b)
    assert sum("no proof scheduler" in r.getMessage() for r in caplog.records) == 1
    b.finalize_service_exploration()
    (row,) = rt.ledger.rows(1, environment=MATH)
    assert row["audit"] == "unaudited"                                          # unpaid, never sanctioned


# ---------------------------------------------------------------- audit verdicts

def _drawn(tmp_path, hotkey="hk", contract=None):
    rt = make_runtime(tmp_path, contract)
    b = make_batcher(rt)
    pending = make_pending(rt, hotkey=hotkey)
    arrive(b, pending)
    ready(rt)
    tick(b)
    return rt, b, pending


def _audit(b, pending, *, verified, stage=None):
    def prove(p, model=None, audit=False):
        assert audit is True                  # an audit never applies the training signal rule
        p.proof_reject_stage = stage
        return verified
    b._verify_expensive = prove
    return b._execute_exploration_audit(pending, model=None)


def test_a_passing_audit_pays_and_a_failing_one_forfeits_and_bans_through_the_runtime_outcome(tmp_path):
    rt, b, passing = _drawn(tmp_path, "good")
    _audit(b, passing, verified=SimpleNamespace(hotkey="good"))
    row = b.difficulty_auction_metadata_by_id[id(passing)]
    assert row["status"] == "exploration_audit_passed"
    assert not rt.exploration_banned("good")
    cheat = make_pending(rt, hotkey="cheat", prompt=8)
    other = make_pending(rt, hotkey="cheat", prompt=9)
    arrive(b, cheat); arrive(b, other)
    ready(rt)
    tick(b)
    _audit(b, cheat, verified=None, stage="grail")
    assert b.difficulty_auction_metadata_by_id[id(cheat)]["status"] == "exploration_forfeited"
    assert b.difficulty_auction_metadata_by_id[id(other)]["status"] == "exploration_forfeited"   # its other groups too
    assert b.difficulty_auction_metadata_by_id[id(other)]["exploration_fraction"] == 0.0
    assert rt.exploration_banned("cheat")
    assert not rt.exploration_banned("good")


@pytest.mark.parametrize("stage", ["service_contract", "service_proof_capability"])
def test_a_proof_that_could_not_judge_the_group_is_no_verdict(tmp_path, stage):
    rt, b, pending = _drawn(tmp_path, "hk")
    _audit(b, pending, verified=None, stage=stage)
    assert not rt.exploration_banned("hk")
    # R25: the validator could not judge it: unaudited, reason validator_lost (no horizon), unpaid, no ban
    assert b.difficulty_auction_metadata_by_id[id(pending)]["status"] == "exploration_unaudited"
    (entry,) = rt.ledger.rows(1, environment=MATH)
    assert entry["audit"] == "unaudited" and entry["status"] == "reserved"
    assert rt.ledger.unaudited_reason(pending.service_observation_id) == "validator_lost"
    assert b.proof_failure_debt("hk") == 0                    # M7: a validator-side stage never charges the miner


@pytest.mark.parametrize("stage, debt", [("service_contract", 0), ("service_proof_capability", 0), ("grail", 1),
                                         ("termination", 1)])
def test_m7_only_a_real_proof_failure_adds_hotkey_debt_never_an_inconclusive_stage(tmp_path, stage, debt):
    rt = make_runtime(tmp_path)
    b = make_batcher(rt)
    b._reject(RejectReason.GRAIL_FAIL, hotkey="hk", prompt_idx=7, reject_stage=stage)
    assert b.proof_failure_debt("hk") == debt


def _cap_found_by_the_proof(b):
    def prove(p, model=None, audit=False):
        assert audit is True
        p.truncated_indices = (LAST,)             # what `_verify_expensive` merges when the proof finds a cut
        return SimpleNamespace(hotkey=p.hotkey)
    b._verify_expensive = prove


def test_r26_a_cap_the_proof_finds_on_a_rollout_admission_saw_terminated_is_a_failed_audit_and_a_ban(tmp_path, monkeypatch):
    monkeypatch.setattr("reliquary.shared.modeling.resolve_eos_token_ids", lambda model, tokenizer: {99})
    rt, b, pending = _drawn(tmp_path, "forger")
    pending.request.rollouts[LAST].commit["tokens"] = [1] + [2] * 10 + [99]      # the forged trailing EOS
    _cap_found_by_the_proof(b)
    b._execute_exploration_audit(pending, model=None)
    assert rt.exploration_banned("forger")
    row = b.difficulty_auction_metadata_by_id[id(pending)]
    assert row["status"] == "exploration_forfeited" and row["exploration_fraction"] == 0.0
    (entry,) = rt.ledger.rows(1, environment=MATH)
    assert entry["audit"] == "failed" and entry["status"] == "forfeited"
    b.finalize_service_exploration()
    assert rt.ledger.payable(1, environment=MATH) == {}


def test_r26_a_cap_the_proof_finds_when_admission_had_no_eos_ids_is_no_verdict_validator_lost(tmp_path, monkeypatch):
    monkeypatch.setattr("reliquary.shared.modeling.resolve_eos_token_ids", lambda model, tokenizer: set())
    rt, b, pending = _drawn(tmp_path, "hk")
    pending.request.rollouts[LAST].commit["tokens"] = [1] + [2] * 10 + [99]
    _cap_found_by_the_proof(b)
    b._execute_exploration_audit(pending, model=None)
    assert not rt.exploration_banned("hk")                                # no evidence of a forgery: no sanction
    row = b.difficulty_auction_metadata_by_id[id(pending)]
    assert row["status"] == "exploration_unaudited" and row["exploration_fraction"] == 0.0
    (entry,) = rt.ledger.rows(1, environment=MATH)
    assert entry["audit"] == "unaudited" and rt.ledger.unaudited_reason(pending.service_observation_id) == "validator_lost"
    b.finalize_service_exploration()
    assert rt.ledger.payable(1, environment=MATH) == {}
    assert b.difficulty_auction_metadata_by_id[id(pending)]["exploration_fraction"] == 0.0


def test_r26_a_cap_on_a_rollout_without_a_trailing_eos_is_no_evidence_of_a_forgery_even_with_eos_ids(tmp_path, monkeypatch):
    monkeypatch.setattr("reliquary.shared.modeling.resolve_eos_token_ids", lambda model, tokenizer: {99})
    rt, b, pending = _drawn(tmp_path, "hk")                               # the default tokens end in 2, not in an EOS
    _cap_found_by_the_proof(b)
    b._execute_exploration_audit(pending, model=None)
    assert not rt.exploration_banned("hk")
    assert rt.ledger.unaudited_reason(pending.service_observation_id) == "validator_lost"


def test_a_forged_termination_stays_a_failed_audit_on_the_service_path(tmp_path):
    rt, b, pending = _drawn(tmp_path, "forger")
    _audit(b, pending, verified=None, stage="termination")
    assert rt.exploration_banned("forger")


def test_the_legacy_cap_count_limit_does_not_apply_to_the_service_path_but_does_to_legacy(monkeypatch):
    import inspect
    source = inspect.getsource(GrpoWindowBatcher._verify_expensive)
    assert "service_contract is None\n                        and truncated_count > max_truncated_per_submission" in source


def test_an_audit_verdict_that_is_not_applied_is_retried_while_the_row_waits_and_acted_on(tmp_path, monkeypatch):
    rt, b, pending = _drawn(tmp_path, "hk")
    identity = pending.service_observation_id
    outcomes = iter([runtime_module.AuditOutcome("not_applied"), None])
    real = rt.record_audit
    calls = []

    def flaky(oid, *, passed):
        calls.append(oid)
        nxt = next(outcomes)
        return nxt if nxt is not None else real(oid, passed=passed)
    rt.record_audit = flaky
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert b._apply_audit_verdict(identity, True).passed
    assert len(calls) == 2
    # a verdict that is moot (the env is finalized) is not retried
    rt2, b2, pending2 = _drawn(tmp_path / "second", "hk")
    b2.finalize_service_exploration()
    seen = []
    real2 = rt2.record_audit
    rt2.record_audit = lambda oid, *, passed: (seen.append(1), real2(oid, passed=passed))[1]
    assert not b2._apply_audit_verdict(pending2.service_observation_id, True).applied
    assert len(seen) == 1


# ---------------------------------------------------------------- audits really run, before the seal

def _real_scheduler(batcher_getter, verdicts):
    """A real GlobalProofScheduler whose proof callable runs the batcher's audit callable."""
    def proof(invocation):
        payload = invocation.candidate.payload
        out = payload.execute(None)
        from reliquary.validator.proof_scheduler import ProofExecution
        return ProofExecution(passed=out is not None, value=out, reason=None if out is not None else "proof_rejected")
    return GlobalProofScheduler(devices=("d0",), environments=(MATH, audit_scheduler_environment(MATH)),
                                proof_callable=proof, checkpoint_revision="d" * 40)


def _wait(predicate, seconds=10.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_audits_run_as_they_are_drawn_and_free_a_probation_slot_before_any_seal(tmp_path):
    from reliquary.constants import PROBATION_PENDING_LIMIT
    rt = make_runtime(tmp_path)
    scheduler = _real_scheduler(None, None)
    try:
        b = make_batcher(rt, scheduler=scheduler)
        b._verify_expensive = lambda p, model=None, audit=False: SimpleNamespace(hotkey=p.hotkey)
        rows = [arrive(b, make_pending(rt, hotkey="newbie", prompt=p)) for p in range(1, PROBATION_PENDING_LIMIT + 1)]
        refused = arrive(b, make_pending(rt, hotkey="newbie", prompt=50))
        assert all(r["status"] == "exploration_pending" for r in rows)
        assert refused["status"] == "exploration_unpaid" and refused["service_unpaid_reason"] == "probation_limit"
        ready(rt)
        assert _wait(lambda: (tick(b), all(
            b.difficulty_auction_metadata_by_id[id(p)]["status"] == "exploration_audit_passed"
            for p in [r for r in b._exploration_pending.values()])
            and len(b._audit_submitted) == PROBATION_PENDING_LIMIT)[1])
        # nothing sealed, nothing finalized: the passed audits freed the probation slots
        assert not rt.ledger.is_finalized(1, environment=MATH)
        retry = arrive(b, make_pending(rt, hotkey="newbie", prompt=51))
        assert retry["status"] == "exploration_pending"
    finally:
        scheduler.close()


def test_the_audit_plan_runs_behind_a_training_plan_of_the_same_env(tmp_path):
    """Priority 5 vs 0 on distinct scheduler environments: both plans can be live at once."""
    rt = make_runtime(tmp_path)
    order = []
    release = threading.Event()

    def proof(invocation):
        from reliquary.validator.proof_scheduler import ProofExecution
        order.append(invocation.plan_id)
        if invocation.plan_id.endswith("fill-closed"):
            release.wait(5)
        return ProofExecution(passed=True, value=SimpleNamespace(hotkey="x"))
    scheduler = GlobalProofScheduler(devices=("d0",), environments=(MATH, audit_scheduler_environment(MATH)),
                                     proof_callable=proof, checkpoint_revision="d" * 40)
    try:
        training = scheduler.submit(ProofPlan(
            plan_id="1:m:fill-closed", environment=MATH, checkpoint_revision="d" * 40, required_passes=1,
            candidates=(RankedProof(job_id="t1", rank=1, prompt_key=("prompt", 1), payload=object()),
                        RankedProof(job_id="t2", rank=2, prompt_key=("prompt", 2), payload=object())),
            deadline_at=time.monotonic() + 60, priority=0, open_ended=True, allow_shortfall=True))
        audit = scheduler.submit(ProofPlan(
            plan_id="1:m:exploration-audit", environment=audit_scheduler_environment(MATH),
            checkpoint_revision="d" * 40, required_passes=99, allow_shortfall=True, open_ended=True,
            candidates=(RankedProof(job_id="a1", rank=1, prompt_key=("a", 1), payload=object(),
                                    counts_toward_target=False),),
            deadline_at=time.monotonic() + 60, dispatch_deadline_at=time.monotonic() + 50, priority=5))
        assert _wait(lambda: order)
        assert order == ["1:m:fill-closed"]            # the audit waits while training holds the only device
        release.set()
        assert _wait(lambda: len(order) >= 2)
        assert order[0] == "1:m:fill-closed" and "1:m:exploration-audit" in order
    finally:
        release.set()
        scheduler.close()


# ---------------------------------------------------------------- seal drain, finalize, reconcile

def _service(rt, batchers, scheduler_log=None):
    from reliquary.validator.service import ValidationService
    service = ValidationService.__new__(ValidationService)
    service._service_runtime = rt
    service._active_batchers = {b.service_environment: b for b in batchers}
    return service


def _short_bounds(monkeypatch, drain=1.5, probation=0.5):
    monkeypatch.setattr(constants, "SERVICE_EXPLORATION_DRAIN_SECONDS", drain)
    monkeypatch.setattr(constants, "SERVICE_EXPLORATION_DRAIN_PROBATION_SECONDS", probation)


@pytest.mark.asyncio
async def test_the_seal_drains_every_past_probation_audit_then_finalizes_then_reconciles(tmp_path, monkeypatch):
    rt = make_runtime(tmp_path, _seasoned_contract())
    scheduler = _real_scheduler(None, None)
    try:
        b = make_batcher(rt, scheduler=scheduler)
        b._verify_expensive = lambda p, model=None, audit=False: SimpleNamespace(hotkey=p.hotkey)
        first = arrive(b, make_pending(rt, hotkey="old", prompt=1))
        ready(rt)
        tick(b)
        assert _wait(lambda: rt.queued_audits(1) == [] and b.audits_in_flight() == 0)
        rows = [arrive(b, make_pending(rt, hotkey="old", prompt=p)) for p in range(2, 9)]
        ready(rt)
        order = []
        for name in ("close_service_exploration", "finalize_service_exploration"):
            real = getattr(b, name)
            setattr(b, name, lambda real=real, name=name: (order.append(name), real())[1])
        service = _service(rt, [b])
        real_reconcile = rt.reconcile_archive
        rt.reconcile_archive = lambda *a, **k: (order.append("reconcile"), real_reconcile(*a, **k))[1]
        _short_bounds(monkeypatch, drain=20.0)
        await service._drain_service_exploration([b])
        order.append("drained")
        await service._settle_service_archive(rt, {"window_start": 1, "window_status": "complete",
                                                   "batch": [], "rewards_by_hotkey": {}})
        # EVERY queued audit of the past-probation hotkey ran before the finalize
        assert rt.queued_audits(1) == [] or rt.ledger.is_finalized(1, environment=MATH)
        statuses = [b.difficulty_auction_metadata_by_id[id(p)]["status"] for p in b._exploration_pending.values()]
        assert set(statuses) == {"exploration_audit_passed"} and len(statuses) == 8
        assert order[0] == "close_service_exploration"
        assert order.index("finalize_service_exploration") < order.index("reconcile")
        assert order.index("drained") < order.index("reconcile")
        assert first["status"] in {"exploration_pending", "exploration_audit_passed"}
    finally:
        scheduler.close()


@pytest.mark.asyncio
async def test_a_stuck_audit_hits_the_bound_and_the_seal_finalizes_anyway(tmp_path, monkeypatch, caplog):
    rt = make_runtime(tmp_path, _seasoned_contract())
    stuck = threading.Event()
    b = make_batcher(rt)         # MagicMock scheduler: nothing ever runs, every audit is "stuck"
    seasoned = arrive(b, make_pending(rt, hotkey="old", prompt=1))
    ready(rt)
    tick(b)
    rt.record_audit(seasoned["service_observation_id"], passed=True)       # "old" is past probation
    b._audit_submitted.add(seasoned["service_observation_id"])
    for prompt in (2, 3):
        arrive(b, make_pending(rt, hotkey="old", prompt=prompt))
    ready(rt)
    service = _service(rt, [b])
    _short_bounds(monkeypatch, drain=1.0, probation=0.2)
    started = time.monotonic()
    with caplog.at_level(logging.ERROR, logger="reliquary"):
        await service._drain_service_exploration([b])
    assert 0.9 <= time.monotonic() - started < 8.0                        # bounded, never forever
    assert any("drain hit its bound" in r.getMessage() for r in caplog.records)
    assert rt.ledger.is_finalized(1, environment=MATH)                     # finalized anyway
    rows = {r["observation_id"]: r for r in rt.ledger.rows(1, environment=MATH)}
    stuck_rows = [r for r in rows.values() if r["hotkey"] == "old" and r["audit"] != "passed"]
    assert stuck_rows and all(r["audit"] == "unaudited" for r in stuck_rows)     # unpaid, never sanctioned
    assert not rt.exploration_banned("old")
    assert rt.ledger.payable(1, environment=MATH) in ({}, {"old": 1})            # only the audited group can be paid
    statuses = {b.difficulty_auction_metadata_by_id[id(p)]["status"] for p in b._exploration_pending.values()}
    assert "exploration_unaudited" in statuses


@pytest.mark.asyncio
async def test_the_drain_waits_at_most_two_drand_rounds_for_pending_draws(tmp_path, monkeypatch):
    rt = make_runtime(tmp_path)
    b = make_batcher(rt)
    arrive(b, make_pending(rt))
    # the clock never reaches the draw round: the draw stays pending
    CLOCK["round"] = 1_000
    service = _service(rt, [b])
    monkeypatch.setattr(constants, "SERVICE_EXPLORATION_DRAIN_SECONDS", 60.0)
    monkeypatch.setattr("reliquary.infrastructure.drand.get_current_chain", lambda *a, **k: {"period": 0.4})
    started = time.monotonic()
    await service._drain_service_exploration([b])
    waited = time.monotonic() - started
    assert 0.6 <= waited < 6.0                                              # ~ 2 rounds + margin, not the 60 s bound
    (entry,) = rt.ledger.rows(1, environment=MATH)
    assert entry["audit"] == "unaudited"


@pytest.mark.asyncio
async def test_the_drain_is_inert_without_a_service_runtime():
    from reliquary.validator.service import ValidationService
    service = ValidationService.__new__(ValidationService)
    service._service_runtime = None
    legacy = _make_batcher()
    await service._drain_service_exploration([legacy])        # returns at once; touches nothing
    await service._service_exploration_tick([legacy])


# ---------------------------------------------------------------- batch pairs join the trained set (R17 defence)

def test_the_batchs_env_prompt_pairs_void_exploration_on_the_same_prompt(tmp_path):
    rt = make_runtime(tmp_path)
    b = make_batcher(rt)
    row = arrive(b, make_pending(rt, hotkey="e", prompt=5))
    assert row["status"] == "exploration_pending"
    # the log never saw a training observation of prompt 5 (a lost one), but the batch trains it
    result = rt.reconcile_archive({"window_start": 1, "window_status": "complete", "rewards_by_hotkey": {},
                                   "batch": [{"hotkey": "t", "env_name": MATH, "prompt_idx": 5}]})
    (entry,) = rt.ledger.rows(1, environment=MATH)
    assert entry["status"] == "trained"
    assert result["service_exploration_by_environment"] == {}
    # nothing else is touched: a batch row on another prompt voids nothing
    rt2 = make_runtime(tmp_path / "other")
    b2 = make_batcher(rt2)
    arrive(b2, make_pending(rt2, hotkey="e", prompt=5))
    result = rt2.reconcile_archive({"window_start": 1, "window_status": "complete", "rewards_by_hotkey": {},
                                    "batch": [{"hotkey": "t", "env_name": MATH, "prompt_idx": 6},
                                              {"hotkey": "t", "env_name": CODE, "prompt_idx": 5}]})
    (entry,) = rt2.ledger.rows(1, environment=MATH)
    assert entry["status"] == "reserved"
    assert result["service_training_by_environment"].get(MATH) is not None


# ---------------------------------------------------------------- shape of the batch rows

def test_batch_rows_equal_the_journal_receipts_rows(tmp_path):
    """The archive's batch rows and the journal's accounting rows carry the same (env, hotkey, prompt) keys,
    so a recovery re-settlement has the same digest as the seal's."""
    from reliquary.validator.fill_closed_recovery import accounting_rows
    group = SimpleNamespace(hotkey="t", prompt_idx=5, sigma=0.5, eos_tokens=1, claimed_checkpoint_hash="d" * 40,
                            merkle_root_bytes=b"\0" * 32, selection_digest=b"\0" * 32, rollout_hashes=[], rollouts=[])
    receipts = accounting_rows({MATH: [group]}, batch_index=0)
    rt = make_runtime(tmp_path)
    seal_rows = [{"hotkey": group.hotkey, "prompt_idx": group.prompt_idx, "env_name": MATH}]
    base = {"window_start": 1, "window_status": "complete", "rewards_by_hotkey": {}}
    one = rt.reconcile_archive({**base, "batch": seal_rows})
    again = rt.reconcile_archive({**base, "batch": receipts})
    import json
    assert json.dumps({k: one[k] for k in runtime_module.FROZEN_ARCHIVE_FIELDS}, sort_keys=True) == \
        json.dumps({k: again[k] for k in runtime_module.FROZEN_ARCHIVE_FIELDS}, sort_keys=True)
    digest = rt.db.execute("SELECT payload FROM service_settled").fetchone()[0]
    assert "batch_sha256" in digest


# ---------------------------------------------------------------- R22

def test_r22_a_missing_box_away_from_the_by_type_default_is_refused_at_boot(tmp_path):
    bad = contract_v2(missing_box="graded")                    # math must be "uncertain"
    with pytest.raises(ValueError, match="missing_box"):
        ServiceRuntime(tmp_path / "bad.sqlite3", bad, qualification_v2(bad), now=time.time())
    wrong_other = contract_v2(envs=(MATH, CODE), missing_box="uncertain")   # code must be "graded"
    with pytest.raises(ValueError, match="missing_box"):
        ServiceRuntime(tmp_path / "bad2.sqlite3", wrong_other, qualification_v2(wrong_other), now=time.time())
    make_runtime(tmp_path / "ok")                              # the defaults boot


def test_r22_task_config_refuses_it_at_resolution(monkeypatch):
    from reliquary.validator import task_config
    entry = SimpleNamespace(task_id="rl", service_contract=contract_v2(missing_box="graded").to_dict())
    with pytest.raises(task_config.TaskConfigError, match="missing_box"):
        task_config._service_env_caps(entry, cap=1.0)


# ---------------------------------------------------------------- legacy is untouched

def test_a_legacy_batcher_reaches_none_of_the_service_lane_code(monkeypatch):
    b = _make_batcher()
    assert b.service_runtime is None and b.service_policy is None
    names = ("_service_lane_of", "_admit_service_observation", "_record_service_observation",
             "_prerecord_service_training", "_record_service_training", "_service_training_receipt",
             "_submit_audits", "_execute_exploration_audit")
    for name in names:
        monkeypatch.setattr(b, name, MagicMock(side_effect=AssertionError(f"{name} ran for a legacy task")))
    from reliquary.validator.fill_window import FillState as FS
    from tests.unit.test_prove_on_arrival import _pending_stub
    b.fill_state = FS(budgets={"openmathinstruct": 4}, picks_target=16)
    extended = []
    b._extend_proof_plan = lambda candidates: extended.extend(candidates)
    pending = _pending_stub(1, rewards=[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pending.request = SimpleNamespace(rollouts=[None] * M_ROLLOUTS)
    b._submit_arrival_proof(pending)
    assert extended and pending.service_lane is None
    b.service_exploration_tick()
    b.finalize_service_exploration()
    b.close_service_exploration()
    assert b._exploration_pending == {} and b._exploration_executor is None
    b._verify_expensive = lambda p, model=None, audit=False: SimpleNamespace(hotkey="x")
    assert b._execute_scheduled_proof(pending, model=None, count_operator_debt=True) is not None


# ---------------------------------------------------------------- Fix round 1: I2 (R26), the non-training bound

@pytest.fixture
def fill_closed(monkeypatch):
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)


def _prepared(rt, b, *, hotkey, prompt, rewards=ZERO):
    """A graded body through the REAL commit (`accept_prepared_submission`): same request as `make_pending`."""
    from reliquary.validator.admission import PreparedSubmission
    pending = make_pending(rt, prompt=prompt, hotkey=hotkey, rewards=rewards)
    request = pending.request
    request.prompt_idx, request.drand_round, request.merkle_root = prompt, 3, pending.merkle_root.hex()
    request._logical_group_reservation, request._retain_payload, request._payload_bytes = None, False, 1000
    return PreparedSubmission(request=request, completion_texts=[], rewards=list(rewards), rollout_hashes=[],
                              selection_digest=pending.selection_digest, prompt_content_sha256="a" * 64,
                              target_content_sha256="b" * 64, attainable_rewards=BINARY)


def _retain(b, prepared):
    """What the HTTP worker does after the commit returns: the request's bytes move to the retained bucket."""
    request = prepared.request
    with b._proof_admission_lock:
        if not request._retain_payload:           # exactly the in-flight release: a cleared flag frees the bytes
            return
        b._retained_payload_reservations[id(request)] = (request, request.miner_hotkey, request._payload_bytes)
        b._retained_payload_bytes += request._payload_bytes
        b._payload_bytes_by_hotkey[request.miner_hotkey] = b._payload_bytes_by_hotkey.get(request.miner_hotkey, 0) + 1000


def test_i2_a_sybil_flood_of_non_training_groups_is_refused_at_the_limit_before_anything_is_retained(fill_closed, tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "SERVICE_NON_TRAINING_PER_HOTKEY_WINDOW_ENV", 3)
    rt = make_runtime(tmp_path)
    b = make_batcher(rt)
    answers = []
    for prompt in range(1, 6):
        prepared = _prepared(rt, b, hotkey="sybil", prompt=prompt)
        response = b.accept_prepared_submission(prepared)
        answers.append((response.accepted, response.reason))
        if response.accepted:
            assert prepared.request._retain_payload is True
        else:
            assert prepared.request._retain_payload is False            # refused BEFORE the payload is retained
    assert answers[:3] == [(True, RejectReason.ACCEPTED)] * 3
    assert answers[3:] == [(False, RejectReason.RATE_LIMITED)] * 2
    assert b.rejected_submissions[-1].reject_stage == "service_non_training_limit"
    assert len(b._pending) == 3
    other = b.accept_prepared_submission(_prepared(rt, b, hotkey="honest", prompt=9))
    assert other.accepted                                                  # another hotkey is not affected
    training = b.accept_prepared_submission(_prepared(rt, b, hotkey="sybil", prompt=10, rewards=HALF))
    assert training.accepted                                               # honest training from the same hotkey too
    assert b._non_training_by_hotkey == {"sybil": 3, "honest": 1}


def test_i2_the_per_prompt_cap_applies_to_non_training_groups_too(fill_closed, tmp_path, monkeypatch):
    monkeypatch.setattr(batcher_module, "MAX_SUBMISSIONS_PER_PROMPT", 2)
    rt = make_runtime(tmp_path)
    b = make_batcher(rt)
    results = [b.accept_prepared_submission(_prepared(rt, b, hotkey=f"hk{i}", prompt=5)) for i in range(3)]
    assert [r.accepted for r in results] == [True, True, False]
    assert results[2].reason is RejectReason.PROMPT_FULL
    assert b.rejected_submissions[-1].reject_stage == "prompt_capacity"
    assert b.accept_prepared_submission(_prepared(rt, b, hotkey="hk9", prompt=6)).accepted   # another prompt is free


def test_i2_the_limit_is_a_service_constant_and_the_guard_keeps_legacy_admission_unchanged(monkeypatch):
    assert constants.SERVICE_NON_TRAINING_PER_HOTKEY_WINDOW_ENV == 32
    legacy = _make_batcher()
    pending = SimpleNamespace(hotkey="hk", prompt_idx=1, service_lane=None)
    assert legacy._service_non_training_refusal(pending) is None
    legacy._note_service_non_training(pending)
    assert legacy._non_training_by_hotkey == {} and legacy._non_training_per_prompt == {}


def test_i2_the_bytes_of_a_group_that_is_not_entitled_are_released_at_once(fill_closed, tmp_path):
    rt = make_runtime(tmp_path)
    b = make_batcher(rt)
    first = _prepared(rt, b, hotkey="a", prompt=7)
    assert b.accept_prepared_submission(first).accepted
    _retain(b, first)
    b.flush_service_admissions()
    assert b._retained_payload_bytes == 1000                              # entitled (first scan): held until its draw
    second = _prepared(rt, b, hotkey="b", prompt=7)                       # the same never-scanned prompt: "already_scanned"
    assert b.accept_prepared_submission(second).accepted
    _retain(b, second)
    b.flush_service_admissions()
    assert b._retained_payload_bytes == 1000 and second.request._retain_payload is False   # only the entitled one is held
    assert b._payload_bytes_by_hotkey.get("b", 0) == 0
    assert id(second.request) not in b._retained_payload_reservations


def test_i2_when_the_worker_has_not_retained_yet_the_flag_makes_it_release_directly(fill_closed, tmp_path):
    rt = make_runtime(tmp_path)
    b = make_batcher(rt)
    prepared = _prepared(rt, b, hotkey="b", prompt=7)
    other = _prepared(rt, b, hotkey="a", prompt=7)
    assert b.accept_prepared_submission(other).accepted
    b.flush_service_admissions()
    assert b.accept_prepared_submission(prepared).accepted
    b.flush_service_admissions()                                          # recorded (and released) before the worker retained
    assert prepared.request._retain_payload is False                      # so the in-flight release frees the bytes itself


def test_i2_an_entitled_group_keeps_its_bytes_until_the_draw_says_it_is_not_audited_and_a_drawn_one_until_its_audit(fill_closed, tmp_path):
    rt = make_runtime(tmp_path, reward_contract(new_hotkey_audit_groups=1, audit_bps=0))   # the first group is forced, the rest are not drawn
    b = make_batcher(rt)
    b._verify_expensive = lambda p, model=None, audit=False: SimpleNamespace(hotkey=p.hotkey)

    def admit(prompt):
        prepared = _prepared(rt, b, hotkey="old", prompt=prompt)
        assert b.accept_prepared_submission(prepared).accepted
        _retain(b, prepared)
        b.flush_service_admissions()
        return prepared

    first = admit(1)
    assert b._retained_payload_bytes == 1000                              # entitled, waiting for the draw: held
    ready(rt)
    tick(b)
    (entry,) = rt.exploration_rows(1, environment=MATH)
    assert entry["audit"] == "queued" and b._retained_payload_bytes == 1000   # drawn and queued: keeps its tokens
    b._execute_exploration_audit(b._exploration_pending[entry["observation_id"]], model=None)
    assert b._retained_payload_bytes == 0                                 # its audit concluded: released
    admit(2), admit(3)                                                    # past probation now: not drawn
    assert b._retained_payload_bytes == 2000
    ready(rt)
    tick(b)
    audits = sorted(r["audit"] for r in rt.exploration_rows(1, environment=MATH))
    assert audits == ["not_drawn", "not_drawn", "passed"]
    assert b._retained_payload_bytes == 0                                 # not drawn: nothing will audit them, released


# ---------------------------------------------------------------- Fix round 1: I4 (no dead wait) and I3/R25 (reasons)

def _lost_setup(tmp_path, scheduler):
    """A seasoned hotkey with three drawn rows waiting for an audit that the validator will not be able to run."""
    rt = make_runtime(tmp_path, _seasoned_contract())
    b = make_batcher(rt, scheduler=scheduler)
    seasoned = arrive(b, make_pending(rt, hotkey="old", prompt=1))
    ready(rt)
    tick(b)
    rt.record_audit(seasoned["service_observation_id"], passed=True)
    b._audit_submitted.add(seasoned["service_observation_id"])
    for prompt in (2, 3, 4):
        arrive(b, make_pending(rt, hotkey="old", prompt=prompt))
    ready(rt)
    return rt, b, seasoned["service_observation_id"]


async def _drain_fast(rt, b, monkeypatch):
    service = _service(rt, [b])
    _short_bounds(monkeypatch, drain=30.0, probation=0.2)
    started = time.monotonic()
    await service._drain_service_exploration([b])
    return time.monotonic() - started


def _assert_lost(rt, b, passed_id):
    assert rt.ledger.is_finalized(1, environment=MATH)
    rows = [r for r in rt.ledger.rows(1, environment=MATH) if r["observation_id"] != passed_id]
    assert len(rows) == 3 and all(r["audit"] == "unaudited" for r in rows)
    assert {rt.ledger.unaudited_reason(r["observation_id"]) for r in rows} == {"validator_lost"}   # no horizon (R25)
    assert not rt.exploration_banned("old")
    for r in rows:
        settles = [e for e in events(rt, r["observation_id"]) if e["type"] == "settle"]
        assert settles[-1]["status"] == "exploration_unpaid" and settles[-1]["reason"] == "unaudited"


def _stop_scheduler():
    scheduler = MagicMock()
    scheduler.submit.return_value.decisions.return_value = ()
    return scheduler


@pytest.mark.asyncio
async def test_i4_no_proof_scheduler_the_drain_does_not_wait_and_the_rows_are_validator_lost(tmp_path, monkeypatch, caplog):
    rt, b, passed_id = _lost_setup(tmp_path, None)
    b._proof_scheduler = None
    assert b.audits_can_progress() is False
    with caplog.at_level(logging.ERROR, logger="reliquary"):
        waited = await _drain_fast(rt, b, monkeypatch)
    assert waited < 5.0                                                   # not the 30 s bound
    assert any("audits cannot run" in r.getMessage() for r in caplog.records)
    _assert_lost(rt, b, passed_id)


@pytest.mark.asyncio
async def test_i4_the_audit_plan_cannot_take_work_the_drain_does_not_wait_and_the_rows_are_validator_lost(tmp_path, monkeypatch):
    from reliquary.validator.proof_scheduler import ProofPlanClosed
    scheduler = _stop_scheduler()
    scheduler.submit.side_effect = ProofPlanClosed("no longer accepts work")
    rt, b, passed_id = _lost_setup(tmp_path, scheduler)
    waited = await _drain_fast(rt, b, monkeypatch)
    assert waited < 5.0 and b._audit_unavailable_logged
    _assert_lost(rt, b, passed_id)


@pytest.mark.asyncio
async def test_i4_the_audit_plan_was_retired_the_drain_does_not_wait_and_the_rows_are_validator_lost(tmp_path, monkeypatch):
    scheduler = _stop_scheduler()
    rt, b, passed_id = _lost_setup(tmp_path, scheduler)
    b._audit_submitted.update(r["observation_id"] for r in rt.queued_audits(1))     # all handed over...
    b._audit_handle = scheduler.submit.return_value
    scheduler.submit.return_value.done.return_value = True                         # ...and the plan is retired
    waited = await _drain_fast(rt, b, monkeypatch)
    assert waited < 5.0
    _assert_lost(rt, b, passed_id)


@pytest.mark.asyncio
async def test_i4_the_dispatch_deadline_passed_the_drain_does_not_wait_and_the_rows_are_validator_lost(tmp_path, monkeypatch):
    rt, b, passed_id = _lost_setup(tmp_path, _stop_scheduler())
    base = b._time_fn
    b._time_fn = lambda: base() + 10 ** 6                                           # far past window_opened_at + bound
    waited = await _drain_fast(rt, b, monkeypatch)
    assert waited < 5.0
    _assert_lost(rt, b, passed_id)


@pytest.mark.asyncio
async def test_i4_the_checkpoint_was_swapped_the_drain_does_not_wait_and_the_rows_are_validator_lost(tmp_path, monkeypatch):
    from reliquary.validator.proof_scheduler import CheckpointNotReady
    scheduler = _stop_scheduler()
    scheduler.submit.side_effect = CheckpointNotReady("plan requires another checkpoint")
    scheduler.active_checkpoint_revision = "e" * 40                                 # the scheduler moved on
    rt, b, passed_id = _lost_setup(tmp_path, scheduler)
    waited = await _drain_fast(rt, b, monkeypatch)
    assert waited < 5.0
    _assert_lost(rt, b, passed_id)


@pytest.mark.asyncio
async def test_i4_a_transient_checkpoint_not_ready_is_still_waited_for_up_to_the_bound(tmp_path, monkeypatch):
    from reliquary.validator.proof_scheduler import CheckpointNotReady
    scheduler = _stop_scheduler()
    scheduler.submit.side_effect = CheckpointNotReady("devices not ready")          # same revision: transient
    rt, b, passed_id = _lost_setup(tmp_path, scheduler)
    service = _service(rt, [b])
    _short_bounds(monkeypatch, drain=1.0, probation=0.2)
    started = time.monotonic()
    await service._drain_service_exploration([b])
    assert 0.9 <= time.monotonic() - started < 8.0                                  # the bound, as before
    rows = [r for r in rt.ledger.rows(1, environment=MATH) if r["observation_id"] != passed_id]
    # the validator could have audited them: the audit plan was running, so the horizon reason applies
    assert {rt.ledger.unaudited_reason(r["observation_id"]) for r in rows} == {"unaudited"}


@pytest.mark.asyncio
async def test_r25_the_drain_bound_reached_with_the_plan_running_leaves_rows_that_set_the_horizon(tmp_path, monkeypatch):
    rt, b, passed_id = _lost_setup(tmp_path, _stop_scheduler())                     # MagicMock plan: running, never completes
    service = _service(rt, [b])
    _short_bounds(monkeypatch, drain=1.0, probation=0.2)
    await service._drain_service_exploration([b])
    rows = [r for r in rt.ledger.rows(1, environment=MATH) if r["observation_id"] != passed_id]
    assert rows and {rt.ledger.unaudited_reason(r["observation_id"]) for r in rows} == {"unaudited"}


def test_r25_a_proof_that_never_ran_to_a_verdict_is_validator_lost(tmp_path):
    scheduler = _stop_scheduler()
    rt, b, passed_id = _lost_setup(tmp_path, scheduler)
    b._audit_open.clear()                                                          # (the passed row's slot is free)
    tick(b)                                                                        # hands the audits over
    open_ids = {j: i for j, i in b._audit_open.items() if i != passed_id}
    assert len(open_ids) >= 2
    job_id, identity = next(iter(open_ids.items()))
    scheduler.submit.return_value.decisions.return_value = (
        SimpleNamespace(job_id=job_id, status=ProofDecisionStatus.ERROR),)
    b._reconcile_audit_decisions()
    assert rt.ledger.unaudited_reason(identity) == "validator_lost"


def test_i3_the_finalize_failure_path_forces_validator_lost_even_while_the_plan_runs(tmp_path):
    rt, b, passed_id = _lost_setup(tmp_path, _stop_scheduler())
    assert b.audits_can_progress() is True
    b.finalize_service_exploration(audits_could_run=False)                         # service.py's window-failed path
    rows = [r for r in rt.ledger.rows(1, environment=MATH) if r["observation_id"] != passed_id]
    assert {rt.ledger.unaudited_reason(r["observation_id"]) for r in rows} == {"validator_lost"}


def test_m5_every_row_status_the_batcher_publishes_for_a_non_trained_group_is_a_known_service_status():
    import inspect
    import re
    from reliquary.validator.service import _SERVICE_EXPLORATION_STATUSES
    source = inspect.getsource(GrpoWindowBatcher)
    used = set(re.findall(r'_set_service_row\(\s*\w+,\s*"([a-z_]+)"', source))
    used |= set(re.findall(r'^\s*"[a-z_]+": "((?:exploration|service)_[a-z_]+|already_scanned)",?$', source, re.M))
    used |= set(re.findall(r'return "((?:exploration|service)_[a-z_]+)"', source))
    assert {"exploration_unavailable", "service_record_failed", "service_policy_limit"} <= used
    assert used - _SERVICE_EXPLORATION_STATUSES == set()


def test_i2_both_commit_sites_decide_the_lane_and_refuse_before_the_payload_is_retained():
    import inspect
    for method in (GrpoWindowBatcher.accept_prepared_submission, GrpoWindowBatcher._accept_locked):
        source = inspect.getsource(method)
        gate = source.index("self._service_non_training_refusal(pending)")
        assert gate < source.index("request._retain_payload = True") < source.index("self._pending.append(pending)")
        assert source.rindex("self.confirm_logical_group_reservation(request)") > gate
        assert source.index("self._note_service_non_training(pending)") > source.index("self._pending.append(pending)")


def test_i4_audits_cannot_progress_without_a_scheduler_even_before_any_audit_was_handed_over(tmp_path):
    rt = make_runtime(tmp_path)
    b = make_batcher(rt)
    b._proof_scheduler = None
    assert b._audit_unavailable_logged is False and b.audits_can_progress() is False
    b._proof_scheduler = MagicMock()
    assert b.audits_can_progress() is True


@pytest.mark.asyncio
async def test_r25_what_the_finalize_stop_releases_is_judged_by_the_finalize_not_marked_lost(tmp_path):
    scheduler = _stop_scheduler()
    rt, b, passed_id = _lost_setup(tmp_path, scheduler)
    b._audit_open.clear()
    tick(b)
    open_ids = {j: i for j, i in b._audit_open.items() if i != passed_id}
    job_id, identity = next(iter(open_ids.items()))
    scheduler.submit.return_value.decisions.return_value = (
        SimpleNamespace(job_id=job_id, status=ProofDecisionStatus.NOT_NEEDED),)     # released by stop_dispatch
    b.finalize_service_exploration()                                               # the plan was running
    assert rt.ledger.unaudited_reason(identity) == "unaudited"                     # horizon reason, not validator_lost


# ---------------------------------------------------------------- Fix round 2: B1 (memory), B2 (audit crash), B2b (boundary)

def _token_lists(request):
    return [r.commit["tokens"] for r in request.rollouts]


def test_b1_a_released_non_training_group_holds_no_token_list_and_a_queued_one_keeps_them(fill_closed, tmp_path):
    rt = make_runtime(tmp_path, reward_contract(new_hotkey_audit_groups=1, audit_bps=0))
    b = make_batcher(rt)
    b._verify_expensive = lambda p, model=None, audit=False: SimpleNamespace(hotkey=p.hotkey)

    def admit(hotkey, prompt):
        prepared = _prepared(rt, b, hotkey=hotkey, prompt=prompt)
        assert b.accept_prepared_submission(prepared).accepted
        _retain(b, prepared)
        b.flush_service_admissions()
        return prepared

    first = admit("old", 1)
    dup = admit("other", 1)                               # already scanned: not entitled, released at once
    assert dup.request.rollouts == []                      # on the OBJECT, not only the counters
    assert len(first.request.rollouts) == M_ROLLOUTS      # entitled, waiting for its draw: kept
    ready(rt)
    tick(b)
    (entry,) = rt.exploration_rows(1, environment=MATH)
    assert entry["audit"] == "queued" and len(first.request.rollouts) == M_ROLLOUTS   # drawn and queued: kept
    queued = b._exploration_pending[entry["observation_id"]]
    b._execute_exploration_audit(queued, model=None)
    assert queued.request.rollouts == []                   # audit concluded: gone
    admit("old", 2), admit("old", 3)                       # past probation: not drawn
    assert any(r.request.rollouts for r in b._exploration_pending.values())
    ready(rt)
    tick(b)
    for pending in list(b._pending) + list(b._exploration_pending.values()):
        assert pending.request.rollouts == []              # nothing in the window's structures pins a token list
    assert not any(_token_lists(p.request) for p in b._pending)
    assert b._retained_payload_bytes == 0


def test_b1_the_forensic_sample_skips_an_emptied_non_training_entry(fill_closed, tmp_path):
    rt = make_runtime(tmp_path)
    b = make_batcher(rt)
    training = make_pending(rt, prompt=2, hotkey="t", rewards=HALF, prompt_content_sha256="a" * 64)
    training.service_lane = "training"
    emptied = make_pending(rt, prompt=3, hotkey="x", prompt_content_sha256="b" * 64)
    emptied.service_lane = "exploration"
    emptied.request.rollouts = []
    b._pending = [emptied, training]
    b.seal_randomness, b._proof_wall_started_at = "ee" * 32, 0.0
    seen = []
    b._proof_scheduler = MagicMock()
    b._prove_forensic_scheduled = lambda sample: (seen.extend(p for p, _ in sample), [])[1]
    b._prove_forensic_sample()
    assert seen == [training]                              # the emptied observation is never sampled
    assert b._pending == [emptied, training]               # untouched: the entry is skipped, not dropped


def test_b2_an_exception_inside_the_audit_proof_is_that_rows_horizon_and_no_ban(tmp_path):
    rt, b, pending = _drawn(tmp_path)

    def boom(p, model=None, audit=False):
        raise RuntimeError("payload makes the forward pass crash")
    b._verify_expensive = boom
    assert b._execute_exploration_audit(pending, model=None) is None        # does not propagate
    identity = pending.service_observation_id
    assert rt.ledger.unaudited_reason(identity) == "unaudited"              # the horizon, not validator_lost
    assert not rt.exploration_banned("hk")
    assert b.difficulty_auction_metadata_by_id[id(pending)]["status"] == "exploration_unaudited"
    assert not b._audit_open


def test_b2_a_crashing_payload_does_not_fault_the_scheduler_and_another_hotkeys_audit_concludes(tmp_path):
    rt = make_runtime(tmp_path, _seasoned_contract())
    scheduler = _real_scheduler(None, None)
    try:
        b = make_batcher(rt, scheduler=scheduler)
        rows = {hk: arrive(b, make_pending(rt, hotkey=hk, prompt=i)) for i, hk in enumerate(("cheat", "honest"), 1)}
        bad_id = rows["cheat"]["service_observation_id"]

        def prove(p, model=None, audit=False):
            if p.service_observation_id == bad_id:
                raise ValueError("crafted payload")
            return SimpleNamespace(hotkey=p.hotkey)
        b._verify_expensive = prove
        ready(rt)
        assert _wait(lambda: (tick(b), rt.ledger.unaudited_reason(bad_id) == "unaudited"
                              and b.difficulty_auction_metadata_by_id[id(b._exploration_pending[
                                  rows["honest"]["service_observation_id"]])]["status"] == "exploration_audit_passed")[1])
        from reliquary.validator.proof_scheduler import SchedulerState
        assert scheduler.state is SchedulerState.RUNNING          # the crash did not fault the proof plane
        assert not rt.exploration_banned("cheat") and not rt.exploration_banned("honest")
        assert b.audits_can_progress()
    finally:
        scheduler.close()


def test_b2_a_scheduler_level_error_without_a_run_stays_validator_lost_but_a_run_error_is_the_horizon(tmp_path):
    scheduler = _stop_scheduler()
    rt, b, passed_id = _lost_setup(tmp_path, scheduler)
    b._audit_open.clear()
    tick(b)
    open_ids = {j: i for j, i in b._audit_open.items() if i != passed_id}
    (j1, i1), (j2, i2) = list(open_ids.items())[:2]
    scheduler.submit.return_value.decisions.return_value = (
        SimpleNamespace(job_id=j1, status=ProofDecisionStatus.ERROR, started_at=None),
        SimpleNamespace(job_id=j2, status=ProofDecisionStatus.ERROR, started_at=12.0))
    b._reconcile_audit_decisions()
    assert rt.ledger.unaudited_reason(i1) == "validator_lost"
    assert rt.ledger.unaudited_reason(i2) == "unaudited"
    assert not rt.exploration_banned("old")


def test_b2_an_audit_queued_behind_a_forfeit_is_skipped_not_proven_on_an_emptied_request(tmp_path):
    rt, b, pending = _drawn(tmp_path)
    b._release_observation_payload(pending)                 # (a forfeited sibling is released while its job waits)
    b._verify_expensive = lambda *a, **k: pytest.fail("a released audit must not be proven")
    assert b._execute_exploration_audit(pending, model=None) is None


@pytest.mark.asyncio
async def test_b2b_window_sealed_at_full_length_drain_bound_with_the_plan_running_is_the_horizon(tmp_path, monkeypatch):
    rt, b, passed_id = _lost_setup(tmp_path, _stop_scheduler())      # MagicMock plan: running, never completes
    base = b._time_fn()
    opened = b.window_opened_at
    state = {"late": False}
    real = b.exploration_drain_state

    def drain_state():
        out = real()
        state["late"] = True                                    # from now on the clock is past the dispatch deadline
        return out
    b.exploration_drain_state = drain_state
    from reliquary.constants import FILL_CLOSED_MAX_SECONDS
    monkeypatch.setattr(b, "_time_fn", lambda: opened + FILL_CLOSED_MAX_SECONDS + (10 ** 6 if state["late"] else 0))
    assert b.audits_can_progress() is True                      # first tick: inside the dispatch window
    service = _service(rt, [b])
    _short_bounds(monkeypatch, drain=30.0, probation=0.2)
    await service._drain_service_exploration([b])
    assert b.audits_can_progress() is False                     # at the end the clock alone says "cannot"
    rows = [r for r in rt.ledger.rows(1, environment=MATH) if r["observation_id"] != passed_id]
    assert rows and {rt.ledger.unaudited_reason(r["observation_id"]) for r in rows} == {"unaudited"}
