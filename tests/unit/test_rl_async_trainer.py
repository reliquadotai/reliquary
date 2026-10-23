"""The async trainer loop (P4, rl-async-design.md §4.6-4.9): batches at the env mix from the TrainingBuffer, the
training forward as verification (floor mask before the backward, pooled group judgment with a back-out pass),
report_training, checkpoint publication and adoption, price signals; the window path untouched."""
from __future__ import annotations

import contextlib
import sqlite3
import threading
import zlib
from types import SimpleNamespace

import pytest
import torch

from reliquary.environment import forced_sampling as fs
from reliquary.trainer.async_trainer import (
    AsyncTrainer, AsyncTrainerConfig, DrawGate, training_group,
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


class Publisher:
    def __init__(self):
        self.versions = []

    def publish(self, version, state):
        self.versions.append(version)


class Clock:
    def __init__(self, on_sleep=None):
        self.t, self.on_sleep = 0.0, on_sleep

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s
        if self.on_sleep is not None:
            self.on_sleep()


def submitted(model, lease, env="math", k=3, arrived=0.0, forged=(), rewards=None, n=N_TOK, prompt_idx=7, noisy=()):
    """A buffered group whose rollouts follow the model's Gumbel pick for the lease's uniforms (forged: random;
    noisy: every 4th token random, honest drift at ~0.8)."""
    base = SubmittedGroup(lease_id=lease, hotkey="A", environment=env, prompt_idx=prompt_idx, checkpoint=k,
                          nonce="cd" * 32, arrived_at=arrived, rewards=tuple(rewards or [1.0, 0.0] * (ROLLOUTS // 2)),
                          truncated=(False,) * ROLLOUTS, prompt_tokens=(1, 2, 3),
                          completions=tuple((0,) * n for _ in range(ROLLOUTS)), force_spans=(None,) * ROLLOUTS)
    gen = torch.Generator().manual_seed(zlib.crc32(lease.encode()) % 1000)
    completions = []
    for i in range(ROLLOUTS):
        us, tokens = base.draw_uniforms(i), [3]
        for j in range(n):
            if i in forged or (i in noisy and j % 4 == 3):
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
def test_the_batch_is_any_env_in_arrival_order_and_waits_until_full():
    model, book = _Model(), Book()
    buffer = TrainingBuffer(book)
    buffer.push(submitted(model, "a0", "a", k=3, arrived=2.0, n=2))
    buffer.push(submitted(model, "b0", "b", k=2, arrived=0.0, n=2))
    buffer.push(submitted(model, "a1", "a", k=2, arrived=1.0, n=2))
    late = iter([submitted(model, "c0", "c", arrived=5.0, n=2)])
    clock = Clock(on_sleep=lambda: [buffer.push(g) for g in late])
    trainer = AsyncTrainer(model, config=AsyncTrainerConfig(4, poll_s=0.5), buffer=buffer,
                           submissions=Subs(buffer), book=book, clock=clock, sleep=clock.sleep)
    assert [g.lease_id for g in trainer.assemble()] == ["b0", "a1", "a0", "c0"]    # oldest first, no mix
    assert clock.t == 0.5 and buffer.depth() == {}


def test_groups_past_the_max_lag_are_dropped_unpaid_without_a_forward():
    model, book = _Model(), Book(latest=10)
    buffer = TrainingBuffer(book)
    buffer.push(submitted(model, "old", k=1, n=2))
    buffer.push(submitted(model, "new", k=9, n=2))
    subs = Subs(buffer)
    trainer = AsyncTrainer(model, config=AsyncTrainerConfig(4, max_lag=8), buffer=buffer,
                           submissions=subs, book=book)
    assert trainer.drop_stale() == 1
    assert subs.reports["old"] == {"rollout_match": [0.0] * ROLLOUTS, "accepted": False,
                                   "failure_class": None, "judged": False}
    assert buffer.depth() == {"math": 1}


# --- the gate ----------------------------------------------------------------------------------------------------
def test_a_rollout_under_the_floor_leaves_the_loss_and_the_pay():
    model = _Model()
    group = training_group(submitted(model, "g", noisy={3}), slot0=0, index=0)
    advantages = [1.0, -1.0, 1.0, -1.0]
    plan = [(group, advantages, 1.0 / (ROLLOUTS * N_TOK))]
    gate = gate_for([group])
    got = grads_of(model, plan, gate)
    assert gate.admitted == [1, 1, 1, 0] and gate.accepted == [True] and gate.undone == 0
    honest = SimpleNamespace(rollouts=group.rollouts[:3], prompt_idx=7)
    want = grads_of(model, [(honest, advantages[:3], 1.0 / (ROLLOUTS * N_TOK))], DrawGate([[0, 1, 2]]))
    assert all(torch.allclose(a, b, atol=1e-7) for a, b in zip(got, want))
    assert gate.forgery_floor <= gate.rate(3) < gate.rollout_floor and gate.rate(0) == 1.0
    detail = {"rollout_match": [gate.rate(s) for s in range(4)],
              "completion_tokens": [N_TOK] * 4,
              "matched_tokens": [gate.tallies[s][3] if gate.admitted[s] else 0 for s in range(4)]}
    assert paid_tokens(detail, gate.rollout_floor) == 3 * N_TOK


def test_one_rollout_under_the_forgery_floor_rejects_its_group_whatever_the_pool():
    """A 16/16 group made in-zone by one forged rollout ([EOS], random tokens): the honest rest pools over the
    group floor, yet the group is rejected (evidence, unpaid), its admitted rollouts backed out."""
    model = _Model()
    group = training_group(submitted(model, "g", forged={3}), slot0=0, index=0)
    plan = [(group, [1.0, -1.0, 1.0, -1.0], 1.0 / (ROLLOUTS * N_TOK))]
    gate = DrawGate([[0, 1, 2, 3]], group_floor=0.7)
    grads_of(model, plan, gate)
    assert gate.rate(3) < gate.forgery_floor and gate.pooled(0) >= 0.7
    assert gate.accepted == [False] and gate.undone == 1


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
def loop(model, book, buffer, batch=2, **config):
    clock = Clock()
    publisher, subs = Publisher(), Subs(buffer)
    trainer = AsyncTrainer(model, config=AsyncTrainerConfig(batch, group_floor=0.7, **config),
                           buffer=buffer, submissions=subs, book=book, publisher=publisher,
                           clock=clock, sleep=clock.sleep)
    return trainer, subs, publisher


def test_one_step_reports_every_group_and_publishes_the_next_checkpoint():
    model, book = _Model(), Book(latest=3)
    buffer = TrainingBuffer(book)
    buffer.push(submitted(model, "good", noisy={2}, arrived=1.0))
    buffer.push(submitted(model, "bad", forged={0, 1, 2}, arrived=0.0))
    trainer, subs, publisher = loop(model, book, buffer)
    before = [p.detach().clone() for p in model.parameters()]
    assert trainer.run_once() == "trained"
    trainer.close()
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    good, bad = subs.reports["good"], subs.reports["bad"]
    assert good["accepted"] and good["matched_tokens"] == [N_TOK, N_TOK, 0, N_TOK]
    assert good["rollout_match"][2] < 0.95 <= min(good["rollout_match"][:2])
    assert not bad["accepted"] and bad.get("failure_class") is None              # unpaid, never sanctioned
    assert publisher.versions == [4] and book.adopted == [4] and trainer.version == 4
    assert buffer.drain_tick().taken == 2 * ROLLOUTS * N_TOK              # the prices read the trainer's speed here
    assert trainer.last_step["async/groups_accepted"] == 1 and trainer.last_step["async/groups_undone"] == 1
    assert buffer.depth() == {}


def test_publication_every_step_and_hf_backup_every_m(monkeypatch):
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

    trainer, subs, publisher = loop(model, book, buffer, batch=1, hf_backup_every=3)
    trainer._train_step, trainer.backup = fake_step, backup
    assert publisher.versions == [0] and book.adopted == [0]                       # fresh book: initial weights
    for step in range(6):
        buffer.push(submitted(model, f"g{step}", k=trainer.version, n=2))
        assert trainer.run_once() == "trained"
    trainer.close()
    assert publisher.versions == list(range(7)) and book.adopted == list(range(7))
    assert backups == [3, 6]
    assert all(r["accepted"] for r in subs.reports.values())


def test_the_step_survives_a_restart_so_backup_names_never_repeat():
    model, book = _Model(), Book(latest=None)
    buffer = TrainingBuffer(book)
    backups = []
    backup = SimpleNamespace(submit=lambda step, save: backups.append(step))

    def fake_step(model, batches, *, gate, **_):
        for g in (g for b in batches for g in b):
            gate.admit(g.rollouts, [(30, 30, N_TOK, N_TOK)] * len(g.rollouts))
        gate.judge([])
        return model

    steps = []
    for life in range(3):                                # the trainer restarts twice on the same book
        trainer, *_ = loop(model, book, buffer, batch=1, hf_backup_every=2)
        trainer._train_step, trainer.backup = fake_step, backup
        assert trainer.steps == trainer.version == 3 * life
        for i in range(3):
            buffer.push(submitted(model, f"g{life}-{i}", k=trainer.version, n=2))
            assert trainer.run_once() == "trained"
            steps.append(trainer.last_step["step"])
            assert trainer.last_step["step"] == trainer.version
        trainer.close()
    assert steps == list(range(1, 10))
    assert backups == [2, 4, 6, 8]


def test_hf_backup_freezes_the_actual_scheduler_step_with_the_weights(tmp_path):
    model, book = _Model(), Book(latest=3)
    buffer = TrainingBuffer(book)
    queued, saved = [], []
    trainer, *_ = loop(model, book, buffer, batch=1, hf_backup_every=1)
    trainer.global_step_hint = 120
    trainer.backup = SimpleNamespace(submit=lambda step, save: queued.append((step, save)))
    trainer._backup_save = lambda state, folder, *, lr_schedule_step: saved.append((state, lr_schedule_step))
    try:
        buffer.push(submitted(model, "first", k=trainer.version))
        assert trainer.run_once() == "trained"
        first_state = {key: value.detach().clone() for key, value in trainer.model.state_dict().items()}
        buffer.push(submitted(model, "second", k=trainer.version))
        assert trainer.run_once() == "trained"
        assert training.current_lr_schedule_step() == 122
        assert [step for step, _ in queued] == [4, 5]
        queued[0][1](tmp_path)
        assert saved[0][1] == 121  # neither checkpoint number 4 nor the later scheduler position 122
        for key, value in first_state.items():
            assert torch.equal(saved[0][0][key], value)
    finally:
        trainer.close()


def test_a_skipped_step_pays_nothing_and_a_crash_requeues():
    model, book = _Model(), Book()
    buffer = TrainingBuffer(book)
    buffer.push(submitted(model, "g", n=2))
    trainer, subs, publisher = loop(model, book, buffer, batch=1)

    def skipped(*_, **__):
        raise training.TrainingStepSkipped("gradient_spike", 1e9)

    trainer._train_step = skipped
    assert trainer.run_once() == "skipped"
    assert subs.reports["g"]["accepted"] is False and subs.reports["g"]["failure_class"] is None
    assert subs.reports["g"]["judged"] is False
    buffer.push(submitted(model, "h", n=2))

    def crash(*_, **__):
        raise RuntimeError("optimizer")

    trainer._train_step = crash
    with pytest.raises(RuntimeError):
        trainer.run_once()
    assert buffer.depth() == {"math": 1} and publisher.versions == []
    assert trainer.version == 3
    trainer.close()


def test_a_buffer_file_from_before_the_attempt_count_opens_and_serves():
    model, book = _Model(), Book()
    book.db.execute("CREATE TABLE rl_buffer(lease_id TEXT PRIMARY KEY, environment TEXT NOT NULL, k INTEGER NOT NULL, "
                    "arrived_at REAL NOT NULL, hotkey TEXT NOT NULL, state TEXT NOT NULL, body BLOB NOT NULL)")
    buffer = TrainingBuffer(book)
    buffer.push(submitted(model, "g", n=2))
    assert [g.lease_id for g in buffer.take(1)] == ["g"] and buffer.take_exhausted(1, 10) == []
    buffer.release(["g"])
    assert [g.lease_id for g in buffer.take_exhausted(1, 10)] == ["g"]


def test_more_than_one_step_per_checkpoint_is_refused():
    with pytest.raises(ValueError, match="steps_per_checkpoint must be 1"):
        AsyncTrainerConfig(4, steps_per_checkpoint=2)


def test_a_group_that_breaks_every_step_is_dropped_and_the_others_train():
    """A step that raises reports nothing; its groups come back (a first return is served like a fresh group: a restart
    is no failure), from their second return on one per batch, and a group handed out ``max_attempts`` times without
    a report leaves untrained: no crash loop."""
    model, book = _Model(), Book()
    buffer = TrainingBuffer(book)
    fresh = iter(range(100))
    clock = Clock(on_sleep=lambda: buffer.push(submitted(model, f"f{next(fresh)}", n=2, arrived=100.0)))
    for i, lease in enumerate(["p", "a", "b", "c"]):
        buffer.push(submitted(model, lease, n=2, arrived=float(i)))
    subs, crashes = Subs(buffer), []
    trainer = AsyncTrainer(model, config=AsyncTrainerConfig(2, max_attempts=3), buffer=buffer, submissions=subs,
                           book=book, publisher=Publisher(), clock=clock, sleep=clock.sleep)

    def step(model, batches, *, gate, **_):
        groups = [g for b in batches for g in b]
        if any(g.lease_id == "p" for g in groups):
            crashes.append([g.lease_id for g in groups])
            raise RuntimeError("CUDA out of memory")
        for g in groups:
            gate.admit(g.rollouts, [(30, 30, N_TOK, N_TOK)] * len(g.rollouts))
        gate.judge([])
        return model

    trainer._train_step = step
    for _ in range(20):
        if "p" in subs.reports:
            break
        try:
            trainer.run_once()
        except RuntimeError:
            pass
    trainer.close()
    assert len(crashes) == 3 and subs.reports["p"]["accepted"] is False
    assert subs.reports["p"].get("failure_class") is None                    # unpaid, never sanctioned
    assert crashes[0] == crashes[1] == ["p", "a"]                            # back once: served like fresh groups
    assert "p" in crashes[2] and "a" not in crashes[2]                        # suspects: one per batch
    assert all(subs.reports[lease]["accepted"] for lease in ("a", "b", "c"))  # a shared two crashes, then trained
    assert trainer.last_step.get("async/exhausted_dropped", 0) in (0, 1)


def test_a_take_whose_answer_was_lost_is_requeued_before_the_next_step():
    model, book = _Model(), Book()
    buffer = TrainingBuffer(book)
    buffer.push(submitted(model, "g", n=2))
    assert [g.lease_id for g in buffer.take(1)] == ["g"]          # taken, answer lost: the trainer never saw it
    reopened = TrainingBuffer(book)
    assert reopened.depth() == {}                                 # controller reopen does not recover the trainer
    trainer, subs, _ = loop(model, book, reopened, batch=1)
    assert trainer.run_once() == "trained" and "g" in subs.reports
    trainer.close()


def test_one_wandb_row_per_step_with_the_step_timings(monkeypatch):
    from unittest.mock import MagicMock

    from reliquary.validator import telemetry

    fake_wandb = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)
    monkeypatch.setattr(telemetry, "_enabled", True)
    monkeypatch.setattr(telemetry, "_held", None)
    model, book = _Model(), Book(latest=3)
    buffer = TrainingBuffer(book)
    buffer.push(submitted(model, "good", arrived=1.0))
    buffer.push(submitted(model, "other", arrived=0.0))
    trainer, *_ = loop(model, book, buffer)
    trainer.global_step_hint = 3                         # the LR schedule position; the wandb step is the step
    assert trainer.run_once() == "trained"
    trainer.close()
    fake_wandb.log.assert_called_once()
    row, step = fake_wandb.log.call_args.args[0], fake_wandb.log.call_args.kwargs["step"]
    assert step == 4                                     # resumed at checkpoint 3: the 4th step
    assert "train/lr" in row and row["async/groups_accepted"] == 2
    assert {"async/wait_s", "async/train_s", "async/step_s", "async/snapshot_s", "async/publish_wait_s"} <= set(row)
    assert row["async/completion_tokens"] == 2 * ROLLOUTS * N_TOK


def test_the_gate_refuses_a_token_far_from_the_pick_and_keeps_the_suspects_for_the_regrade():
    """Every mismatch carries its Gumbel gap: over ``gap_tolerance`` the rollout leaves the loss and the pay; over
    ``substitute_gap`` the (position, pick) goes to the validator, which grades the rollout again with it."""
    model = _Model()
    group = training_group(submitted(model, "g", noisy={3}), slot0=0, index=0)
    plan = [(group, [1.0, -1.0, 1.0, -1.0], 1.0 / (ROLLOUTS * N_TOK))]
    lenient = DrawGate([[0, 1, 2, 3]], rollout_floor=0.0, group_floor=0.0, forgery_floor=0.0,
                       gap_tolerance=float("inf"), substitute_gap=0.0)
    grads_of(model, plan, lenient)
    assert lenient.admitted == [1, 1, 1, 1] and lenient.max_gap[:3] == [0.0] * 3 and lenient.max_gap[3] > 0
    assert lenient.substitutions[:3] == [[], [], []] and lenient.substitutions[3]
    tokens = group.rollouts[3].commit["tokens"][3:]                          # the completion after the 3 prompt tokens
    assert all(p % 4 == 3 and pick != tokens[p] for p, pick in lenient.substitutions[3])   # the random tokens only
    metrics = lenient.metrics()
    assert metrics["async/rollouts_regraded"] == 1 and metrics["async/gap_max"] == lenient.max_gap[3]
    strict = DrawGate([[0, 1, 2, 3]], rollout_floor=0.0, group_floor=0.0, forgery_floor=0.0,
                      gap_tolerance=lenient.max_gap[3] / 2)
    grads_of(model, plan, strict)
    assert strict.admitted == [1, 1, 1, 0] and strict.metrics()["async/rollouts_over_gap_tolerance"] == 1



def test_a_pick_that_ends_the_rollout_goes_to_the_regrade_whatever_its_gap():
    import torch

    from reliquary.validator.training import _DrawRows, _draw_suspects

    eos = 7
    rows = _DrawRows(targets=torch.tensor([5, eos, 9, eos]), drawn=torch.tensor([True, True, True, True]),
                     match=torch.tensor([True, False, False, True]), stochastic=torch.tensor([True] * 4),
                     entropy=torch.zeros(4), gap=torch.tensor([0.0, 0.2, 0.3, 0.0]))
    (gap, pairs, last), = _draw_suspects(rows, [4], [[10, 11, 12, 13]], [10], substitute_gap=3.0,
                                         stop_ids=frozenset({eos}))
    assert pairs == [(1, eos)] and gap == pytest.approx(0.3)                 # small gaps: only the skipped end


def test_a_rollout_cut_short_by_an_eos_far_from_the_pick_is_refused():
    """A final EOS is checked on its own (the re-grade cannot see a cut continuation): its gap must be at most
    ``terminal_gap``. Here every rollout's last token stands for the EOS; the noisy rollout's last token is a random
    one."""
    model = _Model()
    group = training_group(submitted(model, "g", noisy={3}), slot0=0, index=0)
    plan = [(group, [1.0, -1.0, 1.0, -1.0], 1.0 / (ROLLOUTS * N_TOK))]
    eos = {r.commit["tokens"][-1] for r in group.rollouts}
    loose = dict(rollout_floor=0.0, group_floor=0.0, forgery_floor=0.0, gap_tolerance=float("inf"))
    gate = DrawGate([[0, 1, 2, 3]], **loose, terminal_gap=0.5, eos_ids=eos)
    grads_of(model, plan, gate)
    assert gate.bad_terminal == [0, 0, 0, 1] and gate.admitted == [1, 1, 1, 0]
    assert gate.metrics()["async/rollouts_bad_termination"] == 1
    off = DrawGate([[0, 1, 2, 3]], **loose, terminal_gap=0.0)                     # no EOS ids: no rule
    grads_of(model, plan, off)
    assert off.bad_terminal == [0, 0, 0, 0]


def test_audits_are_judged_before_each_step_and_while_the_batch_fills():
    """The trainer judges the validator's out-of-zone audits on its own weights: one bounded round before each step,
    more while it waits for a full batch."""
    model, book = _Model(), Book()
    buffer = TrainingBuffer(book)
    rounds = []

    class Audits:
        def run(self, m, *, limit, budget_s):
            rounds.append((m is trainer.model, limit, budget_s))
            return 1 if len(rounds) < 3 else 0           # two rounds with work, then the queue is empty

        def metrics(self):
            return {"async/audits": float(len(rounds)), "async/audit_s": 0.0}

    clock = Clock(on_sleep=lambda: buffer.push(submitted(model, f"g{int(clock.t)}", n=2)))
    subs = Subs(buffer)
    trainer = AsyncTrainer(model, config=AsyncTrainerConfig(1, audits_per_step=4, audit_budget_s=30.0),
                           buffer=buffer, submissions=subs, book=book, publisher=Publisher(), clock=clock,
                           sleep=clock.sleep, audits=Audits())

    def step(model, batches, *, gate, **_):
        for g in (g for b in batches for g in b):
            gate.admit(g.rollouts, [(30, 30, N_TOK, N_TOK)] * len(g.rollouts))
        gate.judge([])
        return model

    trainer._train_step = step
    assert trainer.run_once() == "trained"
    trainer.close()
    assert rounds[0] == (True, 4, 30.0) and len(rounds) == 3        # before the step, then while waiting
    with pytest.raises(ValueError):
        AsyncTrainerConfig(1, audits_per_step=-1)


def test_the_trainer_never_takes_the_cudnn_attention_backend():
    import torch

    from reliquary.trainer.async_cli import disable_cudnn_attention

    before = torch.backends.cuda.cudnn_sdp_enabled()
    try:
        disable_cudnn_attention(torch)
        assert torch.backends.cuda.cudnn_sdp_enabled() is False
    finally:
        torch.backends.cuda.enable_cudnn_sdp(before)


@pytest.mark.parametrize("forwarded", [True, False])
def test_a_forwarded_zero_match_group_is_reported_as_judged(forwarded):
    model, book = _Model(), Book()
    buffer = TrainingBuffer(book)
    buffer.push(submitted(model, "zero-match"))
    trainer, subs, _ = loop(model, book, buffer, batch=1)

    def step(model, batches, *, gate, **_):
        if forwarded:
            for g in (g for b in batches for g in b):
                gate.admit(g.rollouts, [(64, 0, 64, 0)] * len(g.rollouts))
        gate.judge([])
        return model

    trainer._train_step = step
    assert trainer.run_once() == "trained"
    trainer.close()
    report = subs.reports["zero-match"]
    assert report["accepted"] is False and report["rollout_match"] == [0.0] * ROLLOUTS
    assert report["judged"] is forwarded and report.get("failure_class") is None


def test_drift_rows_split_matches_by_position_entropy_and_gap():
    t = torch.tensor
    rows = training._DrawRows(
        targets=t([0, 0, 0, 0, 0]), drawn=t([True, True, True, True, False]),
        match=t([True, False, True, False, False]), stochastic=t([True, True, True, False, False]),
        entropy=t([0.1, 1.0, 2.0, 0.0, 0.0]), gap=t([0.0, 0.3, 0.0, 9.0, 0.0]))
    first, second = training._rollout_drift_rows(rows, [3, 2])
    assert len(first) == training.DRIFT_WIDTH
    assert first[:2] == pytest.approx([3.1, 3.0]) and second[:2] == [0.0, 2.0]
    # position buckets: every row sits under 1k tokens -> bucket 0 holds (stochastic, matches)
    assert first[2:4] == [3.0, 2.0] and sum(first[4:10]) == 0
    # entropy buckets <0.5, 0.5-1.5, >=1.5 (stochastic rows only)
    assert first[10:16] == [1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    gaps = first[16:]
    assert gaps[1] == 1.0 and sum(gaps) == 1.0                       # one mismatch, gap 0.3 in (0.25, 0.5]
    assert second[16 + 8] == 1.0                                     # gap 9 in (8, 12]


def test_the_gate_breaks_drift_down_by_lag_and_env():
    model = _Model()
    groups = [training_group(submitted(model, name, noisy=noisy), slot0=4 * i, index=i)
              for i, (name, noisy) in enumerate((("fresh", ()), ("stale", {0, 1, 2, 3})))]
    plan = [(g, [1.0, -1.0, 1.0, -1.0], 1.0 / (2 * ROLLOUTS * N_TOK)) for g in groups]
    gate = DrawGate([[0, 1, 2, 3], [4, 5, 6, 7]], rollout_floor=0.0, group_floor=0.0, forgery_floor=0.0,
                    gap_tolerance=float("inf"), envs=["math", "code"], lags=[0, 5])
    grads_of(model, plan, gate)
    m = gate.metrics()
    assert m["async/lag_max"] == 5 and m["async/by_lag/0/groups"] == 1 and m["async/by_lag/3plus/groups"] == 1
    assert m["async/by_lag/0/match_rate"] == 1.0 and m["async/by_lag/0/substitution_rate"] == 0.0
    assert m["async/by_lag/3plus/match_rate"] < 1.0 and m["async/by_lag/3plus/substitution_rate"] > 0.0
    assert m["async/by_env/code/rollouts"] == 4 and m["async/by_env/math/regraded_ratio"] == 0.0
    assert m["async/drift/match_by_position/lt1k"] < 1.0 and m["async/drift/mismatched_tokens"] > 0
    assert 0 < m["async/drift/mismatch_gap_p99"] and m["async/rollout_gap_p99"] >= m["async/rollout_gap_p50"]
    assert {"async/by_lag/0/entropy", "async/by_env/math/groups_rejected_ratio"} <= set(m)


def test_the_wandb_row_carries_the_drift_breakdowns(monkeypatch):
    from unittest.mock import MagicMock

    from reliquary.validator import telemetry

    fake_wandb = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)
    monkeypatch.setattr(telemetry, "_enabled", True)
    monkeypatch.setattr(telemetry, "_held", None)
    model, book = _Model(), Book(latest=3)
    buffer = TrainingBuffer(book)
    buffer.push(submitted(model, "a", k=3, arrived=1.0))
    buffer.push(submitted(model, "b", k=2, env="code", arrived=0.0))
    trainer, *_ = loop(model, book, buffer)
    assert trainer.run_once() == "trained"
    trainer.close()
    row = fake_wandb.log.call_args.args[0]
    assert row["async/by_lag/0/groups"] == 1 and row["async/by_lag/1/groups"] == 1 and row["async/lag_max"] == 1
    assert row["async/by_env/code/completion_tokens_mean"] == N_TOK and row["async/by_env/math/truncated_ratio"] == 0


def test_the_batch_uniforms_are_each_rollouts_draw_uniforms():
    from reliquary.trainer.async_trainer import batch_uniforms, training_group

    groups = [SubmittedGroup(lease_id=f"L{j}", hotkey="A", environment="math", prompt_idx=7 + j, checkpoint=3,
                             nonce="cd" * 32, arrived_at=0.0, rewards=(1.0, 0.0) * (ROLLOUTS // 2),
                             truncated=(False,) * ROLLOUTS, prompt_tokens=(1, 2, 3),
                             completions=tuple((0,) * (4 + i + j) for i in range(ROLLOUTS)),
                             force_spans=(None,) * ROLLOUTS) for j in range(3)]
    got = batch_uniforms(groups)
    assert got == [[g.draw_uniforms(i) for i in range(ROLLOUTS)] for g in groups]
    built = training_group(groups[1], slot0=0, index=1, uniforms=got[1])
    assert [r._forced_draw_uniforms for r in built.rollouts] == got[1]


def test_the_fused_head_takes_its_draws_through_fsdp_as_one_object(monkeypatch):
    """FSDP2's pre-forward walks and rebuilds every list argument element by element; the draws (one per row)
    reach the head wrapped, and the head unwraps them."""
    from reliquary.trainer import distributed
    from reliquary.validator import training

    seen = {}

    class Head:
        fused_selected_logprobs = True

        def __call__(self, hidden, targets, draws, chunk=None):
            seen["draws"] = draws
            return "ok"

    draws = [0.25, None, 0.5]
    assert training._selected_logprobs_from_hidden(torch.zeros(3, 4), torch.zeros(3, dtype=torch.long), Head(),
                                                   draws=draws) == "ok"
    assert isinstance(seen["draws"], distributed.OpaqueArg) and seen["draws"].value is draws

    captured = {}
    monkeypatch.setattr(training, "_selected_logprobs_from_hidden",
                        lambda hidden, targets, project, **kw: captured.setdefault("draws", kw["draws"]))
    head = distributed.FusedLogprobHead(4, 5, bias=False)
    head(torch.zeros(3, 4), torch.zeros(3, dtype=torch.long), distributed.OpaqueArg(draws))
    assert captured["draws"] is draws
