"""Out-of-zone audits judged by the async trainer (``trainer.audits``): the same diagnostics and verdict as the
validator's proof forward, through the trainer's own model (plain head or the FSDP2 fused head), and the loop that
takes jobs from the validator and reports them."""
from __future__ import annotations

import pytest

import dataclasses

from reliquary.trainer.audits import TrainerAudits, audit_group, group_diagnostics, rollout_diagnostics
from reliquary.validator import rl_audit
from reliquary.validator.rl_submit import AuditResult
from tests.unit.test_rl_audit import TinyLM, _group, _transitions, gumbel  # noqa: F401  (gumbel: autouse fixture)


def _forged(table):
    return _group(table, override_rollout=2, override=lambda j, last: (last * 7 + 3) % 16 if j == 30 else None)


@pytest.mark.parametrize("make", [_group, _forged, lambda t: _group(t, nonce_for_draw="cd" * 32)])
def test_the_trainer_forward_judges_as_the_proof_forward(make):
    table = _transitions()
    group = make(table)
    model = TinyLM(table)
    assert audit_group(model, group) == rl_audit.audit_forward(model)(group)
    for i in range(len(group.completions)):
        assert rollout_diagnostics(model, group, i) == rl_audit.rollout_draw_diagnostics(model, group, i)


def test_the_fused_head_projects_the_audit_rows_inside_its_own_call():
    from reliquary.trainer.distributed import FusedLogprobHead

    table = _transitions()
    group = _forged(table)
    plain = rl_audit.audit_forward(TinyLM(table))(group)
    model = TinyLM(table)
    calls = []
    model.lm_head.__class__ = FusedLogprobHead           # what fsdp2's shard_model installs
    original = FusedLogprobHead.forward

    def counted(self, *args, **kw):
        calls.append(kw.get("audit") is not None)
        return original(self, *args, **kw)

    FusedLogprobHead.forward = counted
    try:
        assert audit_group(model, group) == plain
    finally:
        FusedLogprobHead.forward = original
    assert calls == [True]                               # one head call for the packed group: one gather under FSDP2


class Source:
    def __init__(self, jobs, gone=()):
        self.jobs, self.gone, self.reported, self.asked = list(jobs), set(gone), {}, []

    def take_audits(self, limit):
        self.asked.append(limit)
        return self.jobs[:limit]

    def report_audit(self, lease_id, result):
        if lease_id in self.gone:
            raise LookupError(lease_id)
        self.reported[lease_id] = result
        self.jobs = [j for j in self.jobs if j.lease_id != lease_id]


class Job:
    def __init__(self, lease_id):
        self.lease_id = lease_id


def test_the_trainer_loop_takes_judges_and_reports_within_its_budget():
    t = [0.0]
    judged = []

    def audit_fn(model, group):
        judged.append(group.lease_id)
        t[0] += 10.0
        return AuditResult(True, None, (1.0,) * 16)

    source = Source([Job(f"L{i}") for i in range(5)], gone={"L1"})
    audits = TrainerAudits(source, audit_fn=audit_fn, clock=lambda: t[0])
    assert audits.run(object(), limit=4, budget_s=25.0) == 3              # the 4th waits: 30 s spent > 25 s
    assert judged == ["L0", "L1", "L2"] and set(source.reported) == {"L0", "L2"}   # L1 concluded meanwhile
    assert audits.metrics() == {"async/audits": 3.0, "async/audit_s": 30.0}
    assert audits.metrics() == {"async/audits": 0.0, "async/audit_s": 0.0}
    assert audits.run(object(), limit=0, budget_s=25.0) == 0 and source.asked == [4]


def _ragged(group, lengths):
    """The group with rollout i cut to ``lengths[i]`` tokens (mixed lengths: padded micro-batches)."""
    completions = tuple(c[:n] for c, n in zip(group.completions, lengths))
    return dataclasses.replace(group, completions=completions)


@pytest.mark.parametrize("make", [_group, _forged, lambda t: _group(t, nonce_for_draw="cd" * 32)])
@pytest.mark.parametrize("budget", [10_000, 120, 40])
def test_the_packed_group_forward_gives_each_rollouts_own_diagnostics(make, budget):
    """Packed into right-padded micro-batches under any token budget (one rollout over it bins alone), every rollout
    keeps exactly the diagnostics of its own forward, and the verdict is the proof forward's."""
    table = _transitions()
    full = make(table)
    group = _ragged(full, [len(full.completions[0]) - (i * 5) % 23 for i in range(16)])
    model = TinyLM(table)
    assert group_diagnostics(model, group, token_budget=budget) == [
        rollout_diagnostics(model, group, i) for i in range(16)]
    assert audit_group(model, group) == rl_audit.audit_forward(model)(group)


def test_audits_are_taken_a_few_at_a_time_so_the_budget_leaves_little_downloaded(monkeypatch):
    from reliquary.trainer import audits as audits_module

    monkeypatch.setattr(audits_module, "AUDIT_CHUNK", 2)
    t = [0.0]

    def audit_fn(model, group):
        t[0] += 10.0
        return AuditResult(True, None, (1.0,) * 16)

    source = Source([Job(f"L{i}") for i in range(9)])
    audits = TrainerAudits(source, audit_fn=audit_fn, clock=lambda: t[0])
    assert audits.run(object(), limit=7, budget_s=1000.0) == 7
    assert source.asked == [2, 2, 2, 1]
    source.asked.clear()
    assert audits.run(object(), limit=7, budget_s=15.0) == 2 and source.asked == [2]


class Sized(Job):
    def __init__(self, lease_id, rollouts=16):
        super().__init__(lease_id)
        self.completions = ((1,),) * rollouts


def _batched(log, t, step=10.0, verdict=lambda g: AuditResult(True, None, (1.0,) * 16)):
    def audit_fn(model, group):                     # the per-group path must not be used when ``many`` exists
        raise AssertionError("judged one group at a time")

    def many(model, groups):
        log.append([g.lease_id for g in groups])
        t[0] += step
        return [verdict(g) for g in groups]

    audit_fn.many = many
    return audit_fn


def test_batched_audits_pack_groups_up_to_the_rollouts_per_call():
    t, calls = [0.0], []
    source = Source([Sized("A"), Sized("B"), Sized("C", 40), Sized("D"), Sized("E", 80), Sized("F")])
    audits = TrainerAudits(source, audit_fn=_batched(calls, t), clock=lambda: t[0], max_rollouts_per_call=64)
    assert audits.run(object(), limit=10, budget_s=1000.0) == 6
    # consecutive groups while they fit; a group over the cap goes alone
    assert calls == [["A", "B"], ["C", "D"], ["E"], ["F"]]
    assert set(source.reported) == {"A", "B", "C", "D", "E", "F"}


def test_batched_audits_report_the_same_verdicts_as_group_by_group():
    verdict = {f"L{i}": AuditResult(i % 3 != 0, None, (1.0,) * 16) for i in range(9)}

    def one(model, group):
        return verdict[group.lease_id]

    t = [0.0]
    by_group = Source([Sized(f"L{i}") for i in range(9)], gone={"L4"})
    TrainerAudits(by_group, audit_fn=one, clock=lambda: t[0]).run(object(), limit=9, budget_s=1e9)
    calls = []
    batched = Source([Sized(f"L{i}") for i in range(9)], gone={"L4"})
    TrainerAudits(batched, audit_fn=_batched(calls, t, verdict=lambda g: verdict[g.lease_id]),
                  clock=lambda: t[0]).run(object(), limit=9, budget_s=1e9)
    # L4 is concluded meanwhile: it stays at the queue's head and takes the last slot of the round, in both modes
    assert batched.reported == by_group.reported and len(by_group.reported) == 7
    assert calls == [["L0", "L1", "L2", "L3"], ["L4", "L5", "L6", "L7"], ["L4"]]


def test_batched_audits_stop_between_calls_when_the_budget_is_spent():
    t, calls = [0.0], []
    source = Source([Sized(f"L{i}") for i in range(12)])
    audits = TrainerAudits(source, audit_fn=_batched(calls, t, step=30.0), clock=lambda: t[0])
    assert audits.run(object(), limit=12, budget_s=50.0) == 8                # 2 calls: 60 s > 50 s
    assert len(calls) == 2 and audits.metrics() == {"async/audits": 8.0, "async/audit_s": 60.0,
                                                "async/audit_calls": 2.0}
    with pytest.raises(ValueError):
        TrainerAudits(source, max_rollouts_per_call=0)
