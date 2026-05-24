"""train_step — basic interface tests (no real model required).

These tests cover the stable interface contract: empty-batch is a no-op,
the model reference is always returned, and a non-empty batch with no
valid rollouts still returns cleanly. The full math is tested in
test_training_grpo.py and test_training_rollout_loss.py.
"""

import logging
from unittest.mock import MagicMock, patch

from reliquary.validator.training import train_step, reset_training_state


def test_train_step_with_empty_batch():
    reset_training_state()
    model = MagicMock()
    result = train_step(model=model, batches=[], ref_model=MagicMock())
    assert result is model


def test_train_step_empty_batch_logs(caplog):
    reset_training_state()
    caplog.set_level(logging.INFO, logger="reliquary.validator.training")
    train_step(model=MagicMock(), batches=[], ref_model=MagicMock())
    assert any("empty batch" in rec.message for rec in caplog.records)


def test_train_step_returns_model_on_all_degenerate_groups():
    """If every group is degenerate (all-same reward), no optimizer step
    and the original model is still returned."""
    reset_training_state()

    import torch
    import reliquary.validator.training as _t

    # Build a tiny linear model so _lazy_init and device resolution work.
    model = torch.nn.Linear(2, 2)

    # All rollouts have identical reward → advantages all zero → skipped.
    rollout = MagicMock()
    rollout.reward = 1.0
    group = MagicMock()
    group.rollouts = [rollout] * 4
    group.prompt_idx = 0

    import copy
    ref = copy.deepcopy(model)
    result = train_step(model=model, batches=[[group]], ref_model=ref)
    assert result is model

    # Optimizer state should have been initialised but step not taken
    # (n_processed == 0 → early return before optimizer.step).
    assert _t._optimizer is not None
    # Verify the optimizer step count is still 0.
    assert _t._scheduler.last_epoch == 0


def test_train_step_forwards_metrics_to_telemetry(monkeypatch):
    """When train_step completes a successful step, it calls
    telemetry.log_training_step with the GRPO metrics dict and the
    window_index as the step."""
    from unittest.mock import MagicMock
    import torch

    import reliquary.validator.training as _t
    from reliquary.validator import telemetry

    _t.reset_training_state()

    captured = {}

    def fake_log(metrics, step):
        captured["metrics"] = metrics
        captured["step"] = step

    monkeypatch.setattr(telemetry, "log_training_step", fake_log)

    # Build a tiny model and a batch with a non-degenerate group so a
    # real optimizer step runs. We reuse the same minimal rollout shape
    # the other stub tests use but with varied rewards.
    model = torch.nn.Linear(2, 2)

    def _mk_rollout(reward, prompt_len=1):
        r = MagicMock()
        r.reward = float(reward)
        r.tokens = [0, 1]
        r.commit = {"rollout": {"prompt_length": prompt_len, "token_logprobs": [0.0]}}
        return r

    group = MagicMock()
    group.rollouts = [_mk_rollout(1.0), _mk_rollout(0.0)]
    group.prompt_idx = 0

    # The test doesn't care whether the _rollout_loss forward pass
    # actually runs; monkeypatch it to return loss tensors connected to
    # model parameters (so loss.backward() works) but numerically zero.
    # Must return fresh tensors each call — the graph is freed after each
    # backward pass.
    def fake_loss(model, ref_model, rollout, advantage, device):
        p = next(model.parameters())
        zero_loss = (p.sum() * 0.0)
        return zero_loss, zero_loss.detach()

    monkeypatch.setattr(_t, "_rollout_loss", fake_loss)

    import copy
    ref = copy.deepcopy(model)
    _t.train_step(model=model, batches=[[group]], ref_model=ref, window_index=7)

    assert captured["step"] == 7
    m = captured["metrics"]
    for key in (
        "train/lr", "train/ppo_loss", "train/kl", "train/grad_norm",
        "train/rollouts_processed", "train/rollouts_total",
        "train/valid_rollout_ratio",
        "rewards/mean", "rewards/std", "rewards/min", "rewards/max",
        "batch/n_groups", "batch/n_degenerate_groups", "batch/degenerate_ratio",
    ):
        assert key in m, f"missing metric {key}"
    assert m["batch/n_groups"] == 1
    assert m["batch/n_degenerate_groups"] == 0
    assert m["rewards/min"] == 0.0
    assert m["rewards/max"] == 1.0


def test_train_step_counts_degenerate_groups(monkeypatch):
    """A batch of only degenerate groups reports n_degenerate_groups ==
    n_groups and does not emit metrics (no successful step — early
    return before the metrics branch)."""
    from unittest.mock import MagicMock
    import torch

    import reliquary.validator.training as _t
    from reliquary.validator import telemetry

    _t.reset_training_state()

    called = []
    monkeypatch.setattr(
        telemetry, "log_training_step",
        lambda metrics, step: called.append((metrics, step)),
    )

    model = torch.nn.Linear(2, 2)
    rollout = MagicMock()
    rollout.reward = 1.0
    group = MagicMock()
    group.rollouts = [rollout] * 4
    group.prompt_idx = 0

    import copy
    ref = copy.deepcopy(model)
    _t.train_step(model=model, batches=[[group]], ref_model=ref, window_index=3)

    # No successful step → no metrics emitted.
    assert called == []
