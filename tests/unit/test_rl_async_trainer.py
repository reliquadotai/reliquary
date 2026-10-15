"""The async trainer loop (P4, rl-async-design.md §4.6-4.9): batches at the env mix from the TrainingBuffer, the
training forward as verification (floor mask before the backward, pooled group judgment with a back-out pass),
report_training, checkpoint publication and adoption, price signals; the window path untouched."""
from __future__ import annotations

import contextlib
import sqlite3
import threading
from types import SimpleNamespace

import pytest
import torch

from reliquary.environment import forced_sampling as fs
from reliquary.trainer.async_trainer import (
    AsyncTrainer, AsyncTrainerConfig, DrawGate, mix_quotas, training_group,
)
from reliquary.validator import training, verifier
from reliquary.validator.rl_buffer import SubmittedGroup, TrainingBuffer
from reliquary.validator.rl_ledger import paid_tokens
from tests.unit.test_substitution_loss import _GUMBEL, _VOCAB, _Model, _logits_at

N_TOK, ROLLOUTS = 24, 4


@pytest.fixture(autouse=True)
def _substitution(monkeypatch):
    monkeypatch.setattr(training, "KL_BETA", 0.0)
    monkeypatch.setattr("reliquary.constants.TRAINING_LOSS", "substitution")
    monkeypatch.setattr(verifier, "draw_settings", lambda: _GUMBEL)
    training.reset_training_state()
    yield
    training.reset_training_state()


class Book:
    """The lease book's surface the buffer and the loop use."""

    def __init__(self, latest=3):
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.lock = threading.RLock()
        self.latest_checkpoint = latest
        self.adopted = []

    @contextlib.contextmanager
    def transaction(self):
        with self.lock:
            yield
            self.db.commit()

    def adopt_checkpoint(self, k):
        self.adopted.append(k)
        self.latest_checkpoint = k


class Subs:
    def __init__(self, buffer):
        self.buffer, self.reports = buffer, {}

    def report_training(self, lease_id, **kw):
        assert self.buffer._remove(lease_id)
        self.reports[lease_id] = kw


class Prices:
    def __init__(self):
        self.waits, self.ticks = [], []

    def note_wait(self, env, seconds):
        self.waits.append((env, seconds))

    def tick(self, trained, depth):
        self.ticks.append((dict(trained), dict(depth)))


class Publisher:
    def __init__(self):
        self.versions = []

    def publish(self, version, state):
        self.versions.append(version)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def submitted(model, lease, env="math", k=3, arrived=0.0, forged=(), rewards=None, n=N_TOK, prompt_idx=7):
    """A buffered group whose rollouts follow the model's Gumbel pick for the lease's uniforms (forged: random)."""
    base = SubmittedGroup(lease_id=lease, hotkey="A", environment=env, prompt_idx=prompt_idx, checkpoint=k,
                          nonce="cd" * 32, arrived_at=arrived, rewards=tuple(rewards or [1.0, 0.0] * (ROLLOUTS // 2)),
                          truncated=(False,) * ROLLOUTS, prompt_tokens=(1, 2, 3),
                          completions=tuple((0,) * n for _ in range(ROLLOUTS)), force_spans=(None,) * ROLLOUTS)
    gen = torch.Generator().manual_seed(hash(lease) % 1000)
    completions = []
    for i in range(ROLLOUTS):
        us, tokens = base.draw_uniforms(i), [3]
        for j in range(n):
            if i in forged:
                tokens.append(int(torch.randint(0, _VOCAB, (1,), generator=gen)))
            else:
                tokens.append(int(fs.pick_rows(_logits_at(model, tokens[-1]), [us[j]], draw=fs.DRAW_GUMBEL_MAX)[0]))
        completions.append(tuple(tokens[1:]))
    return SubmittedGroup(**{**base.__dict__, "completions": tuple(completions)})


def grads_of(model, plan, gate=None):
    for p in model.parameters():
        p.grad = None
    training._accumulate_grouped_grads(model, None, plan, torch.device("cpu"), budget=10_000, atomic=False,
                                       kl_stats=training._new_kl_stats(), gate=gate)
    if gate is not None:
        undo = gate.judge(plan)
        if undo:
            training._accumulate_grouped_grads(model, None, undo, torch.device("cpu"), budget=10_000,
                                               atomic=False, gate=gate.replay())
    return [torch.zeros_like(p) if p.grad is None else p.grad.clone() for p in model.parameters()]


def gate_for(groups):
    # 4 rollouts a group here: one bad rollout pools to ~0.75 (16 in production: 0.92 against 0.90)
    return DrawGate([[r._async_slot for r in g.rollouts] for g in groups], group_floor=0.7)


# --- batch assembly ----------------------------------------------------------------------------------------------
def test_quotas_follow_the_mix():
    assert mix_quotas({"a": 0.5, "b": 0.25, "c": 0.25}, 8) == {"a": 4, "b": 2, "c": 2}
    assert mix_quotas({"a": 0.5, "b": 0.5, "c": 0.0}, 3) == {"a": 2, "b": 1}
    assert sum(mix_quotas({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}, 10).values()) == 10


def test_a_short_env_waits_bounded_then_proceeds_freshest_first():
    model, book, clock = _Model(), Book(), Clock()
    buffer = TrainingBuffer(book)
    for i in range(6):
        buffer.push(submitted(model, f"a{i}", "a", k=2 + i % 2, arrived=float(i), n=2))
    buffer.push(submitted(model, "b0", "b", n=2))
    prices = Prices()
    trainer = AsyncTrainer(model, config=AsyncTrainerConfig({"a": 0.5, "b": 0.25, "c": 0.25}, 8, wait_s=2.0,
                                                            poll_s=0.5),
                           buffer=buffer, submissions=Subs(buffer), book=book, prices=prices, clock=clock,
                           sleep=clock.sleep)
    taken, waits = trainer.assemble()
    assert [g.lease_id for g in taken["a"]] == ["a5", "a3", "a1", "a4"]      # k=3 first, then latest arrival
    assert [g.lease_id for g in taken["b"]] == ["b0"] and taken["c"] == []
    assert waits == {"b": 2.0, "c": 2.0} and prices.waits == [("b", 2.0), ("c", 2.0)]
    assert buffer.depth() == {"a": 2}


def test_groups_past_the_max_lag_are_dropped_unpaid_without_a_forward():
    model, book = _Model(), Book(latest=10)
    buffer = TrainingBuffer(book)
    buffer.push(submitted(model, "old", k=1, n=2))
    buffer.push(submitted(model, "new", k=9, n=2))
    subs = Subs(buffer)
    trainer = AsyncTrainer(model, config=AsyncTrainerConfig({"math": 1.0}, 4, max_lag=8), buffer=buffer,
                           submissions=subs, book=book)
    assert trainer.drop_stale() == 1
    assert subs.reports["old"] == {"rollout_match": [0.0] * ROLLOUTS, "accepted": False, "failure_class": None}
    assert buffer.depth() == {"math": 1}


# --- the gate ----------------------------------------------------------------------------------------------------
def test_a_rollout_under_the_floor_leaves_the_loss_and_the_pay():
    model = _Model()
    group = training_group(submitted(model, "g", forged={3}), slot0=0, index=0)
    advantages = [1.0, -1.0, 1.0, -1.0]
    plan = [(group, advantages, 1.0 / (ROLLOUTS * N_TOK))]
    gate = gate_for([group])
    got = grads_of(model, plan, gate)
    assert gate.admitted == [1, 1, 1, 0] and gate.accepted == [True] and gate.undone == 0
    honest = SimpleNamespace(rollouts=group.rollouts[:3], prompt_idx=7)
    want = grads_of(model, [(honest, advantages[:3], 1.0 / (ROLLOUTS * N_TOK))], DrawGate([[0, 1, 2]]))
    assert all(torch.allclose(a, b, atol=1e-7) for a, b in zip(got, want))
    assert gate.rate(3) < 0.5 and gate.rate(0) == 1.0
    detail = {"rollout_match": [gate.rate(s) for s in range(4)],
              "completion_tokens": [N_TOK] * 4,
              "matched_tokens": [gate.tallies[s][3] if gate.admitted[s] else 0 for s in range(4)]}
    assert paid_tokens(detail, 0.95) == 3 * N_TOK


def test_a_failed_group_is_backed_out_before_the_step():
    model = _Model()
    good = training_group(submitted(model, "good"), slot0=0, index=0)
    bad = training_group(submitted(model, "bad", forged={0, 1, 2}), slot0=ROLLOUTS, index=1)
    scale = 1.0 / (2 * ROLLOUTS * N_TOK)
    plan = [(good, [1.0, -1.0, 1.0, -1.0], scale), (bad, [1.0, -1.0, 1.0, -1.0], scale)]
    gate = gate_for([good, bad])
    got = grads_of(model, plan, gate)
    assert gate.accepted == [True, False] and gate.undone == 1 and gate.admitted[ROLLOUTS + 3] == 1
    want = grads_of(model, plan[:1], DrawGate([[0, 1, 2, 3]]))
    # bf16 autocast: the back-out cancels to ~1e-4 on grads of ~3e-2 (without it the gap is ~3e-2)
    assert all(torch.allclose(a, b, atol=3e-4) for a, b in zip(got, want))


def test_window_mode_is_unchanged_without_a_gate():
    """No gate: the window path's substitution grads and draw counters, as before (pi_old from the claim)."""
    model = _Model()
    group = training_group(submitted(model, "g", forged={3}), slot0=0, index=0)
    plan = [(group, [1.0, -1.0, 1.0, -1.0], 1.0 / (ROLLOUTS * N_TOK))]
    stats = training._new_kl_stats()
    for p in model.parameters():
        p.grad = None
    training._accumulate_grouped_grads(model, None, plan, torch.device("cpu"), budget=10_000, atomic=False,
                                       kl_stats=stats)
    assert stats["draw_rollout_count"] == 4 and stats["ppo_token_count"] == ROLLOUTS * N_TOK
    assert group.rollouts[3]._training_draw_counts[1] < group.rollouts[3]._training_draw_counts[0]


# --- the loop ----------------------------------------------------------------------------------------------------
def loop(model, book, buffer, **config):
    clock = Clock()
    publisher, prices, subs = Publisher(), Prices(), Subs(buffer)
    trainer = AsyncTrainer(model, config=AsyncTrainerConfig({"math": 1.0}, 2, wait_s=0.0, group_floor=0.7, **config),
                           buffer=buffer, submissions=subs, book=book, prices=prices, publisher=publisher,
                           clock=clock, sleep=clock.sleep)
    return trainer, subs, publisher, prices


def test_one_step_reports_every_group_and_publishes_the_next_checkpoint():
    model, book = _Model(), Book(latest=3)
    buffer = TrainingBuffer(book)
    buffer.push(submitted(model, "good", forged={2}, arrived=1.0))
    buffer.push(submitted(model, "bad", forged={0, 1, 2}, arrived=0.0))
    trainer, subs, publisher, prices = loop(model, book, buffer)
    before = [p.detach().clone() for p in model.parameters()]
    assert trainer.run_once() == "trained"
    trainer.close()
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    good, bad = subs.reports["good"], subs.reports["bad"]
    assert good["accepted"] and good["matched_tokens"] == [N_TOK, N_TOK, 0, N_TOK]
    assert good["rollout_match"][2] < 0.95 <= min(good["rollout_match"][:2])
    assert not bad["accepted"] and bad["failure_class"] == "statistical"          # lag 0: evidence
    assert publisher.versions == [4] and book.adopted == [4] and trainer.version == 4
    assert prices.ticks == [({"math": 3 * N_TOK}, {})]
    assert trainer.last_step["async/groups_accepted"] == 1 and trainer.last_step["async/groups_undone"] == 1
    assert buffer.depth() == {}


def test_publication_every_n_steps_and_hf_backup_every_m(monkeypatch):
    model, book = _Model(), Book(latest=None)
    buffer = TrainingBuffer(book)
    backups = []
    backup = SimpleNamespace(submit=lambda step, save: backups.append(step))

    def fake_step(model, batches, *, gate, **_):
        groups = [g for b in batches for g in b]
        for g in groups:
            gate.admit(g.rollouts, [(30, 30, N_TOK, N_TOK)] * len(g.rollouts))
        gate.judge([])
        return model

    trainer, subs, publisher, prices = loop(model, book, buffer, steps_per_checkpoint=2, hf_backup_every=3)
    trainer._train_step, trainer.backup = fake_step, backup
    assert publisher.versions == [0] and book.adopted == [0]                       # fresh book: initial weights
    for step in range(6):
        buffer.push(submitted(model, f"g{step}", k=trainer.version, n=2))
        assert trainer.run_once() == "trained"
    trainer.close()
    assert publisher.versions == [0, 1, 2, 3] and book.adopted == [0, 1, 2, 3]
    assert backups == [3, 6]
    assert all(r["accepted"] for r in subs.reports.values())


def test_a_skipped_step_pays_nothing_and_a_crash_requeues():
    model, book = _Model(), Book()
    buffer = TrainingBuffer(book)
    buffer.push(submitted(model, "g", n=2))
    trainer, subs, publisher, _ = loop(model, book, buffer)

    def skipped(*_, **__):
        raise training.TrainingStepSkipped("gradient_spike", 1e9)

    trainer._train_step = skipped
    assert trainer.run_once() == "skipped"
    assert subs.reports["g"]["accepted"] is False and subs.reports["g"]["failure_class"] is None
    buffer.push(submitted(model, "h", n=2))

    def crash(*_, **__):
        raise RuntimeError("optimizer")

    trainer._train_step = crash
    with pytest.raises(RuntimeError):
        trainer.run_once()
    assert buffer.depth() == {"math": 1} and publisher.versions == []
    assert trainer.version == 3
    trainer.close()


def test_nothing_buffered_waits():
    model, book = _Model(), Book()
    trainer, _, publisher, prices = loop(model, book, TrainingBuffer(book))
    assert trainer.run_once() == "waited" and publisher.versions == [] and prices.ticks == []
    trainer.close()
