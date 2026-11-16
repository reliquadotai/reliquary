"""A same-run restart resumes Adam from the per-rank snapshot instead of re-warming the LR."""
import pytest
import torch

import reliquary.validator.training as training
from reliquary.constants import LR_RESTART_REWARMUP_WINDOWS
from reliquary.validator import optimizer_state


@pytest.fixture(autouse=True)
def _fresh(tmp_path, monkeypatch):
    monkeypatch.setenv(optimizer_state.STATE_DIR_ENV, str(tmp_path))
    monkeypatch.delenv("RELIQUARY_REQUIRE_OPTIMIZER_RESUME", raising=False)
    training.reset_training_state()
    yield
    training.reset_training_state()


def _model(dtype=torch.float32):
    torch.manual_seed(0)
    return torch.nn.Linear(8, 8).to(dtype)


def _steps(model, n, hint):
    assert training._lazy_init(model, global_step_hint=hint)
    for i in range(n):
        model(torch.randn(4, 8, dtype=next(model.parameters()).dtype) * (i + 1)).float().pow(2).sum().backward()
        training._optimizer.step()
        training._scheduler.step()
        training._save_optimizer_state()
        training._optimizer.zero_grad()
    training._state_writer.wait()


def _state(optimizer, params):
    return [{k: v.clone() for k, v in optimizer.state[p].items()} for p in params]


def _published(model):
    """What the trainer resumes from: the bf16 weights of the checkpoint."""
    return {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}


def _resumed(published, dtype=torch.float32):
    model = _model(dtype)
    model.load_state_dict({k: v.to(dtype) for k, v in published.items()})
    return model


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_restart_restores_moments_and_skips_rewarmup(dtype, strict, monkeypatch):
    if strict:
        monkeypatch.setenv("RELIQUARY_REQUIRE_OPTIMIZER_RESUME", "1")
    model = _model(dtype)
    _steps(model, 12, hint=0)
    params, optimizer = training._stepped()
    want_state, want_masters = _state(optimizer, params), [p.detach().clone() for p in params]
    published = _published(model)
    training.reset_training_state()

    resumed = _resumed(published, dtype)
    assert training._lazy_init(resumed, global_step_hint=12)
    params, optimizer = training._stepped()
    for got, want in zip(_state(optimizer, params), want_state):
        assert got.keys() == want.keys()
        for key in want:
            assert torch.equal(got[key], want[key]), key
    for got, want in zip(params, want_masters):
        assert torch.equal(got.detach(), want)          # the fp32 masters, not the bf16 rounding
    lam = training._scheduler.lr_lambdas[0]
    assert lam(12) == 1.0 and lam(13) == 1.0


def test_no_snapshot_rewarms():
    model = _model()
    _steps(model, 12, hint=0)
    published = _published(model)
    training.reset_training_state()
    for path in optimizer_state.state_dir().iterdir():
        path.unlink()
    assert training._lazy_init(_resumed(published), global_step_hint=12)
    assert training._scheduler.lr_lambdas[0](12) == pytest.approx(1 / (LR_RESTART_REWARMUP_WINDOWS + 1))
    assert not training._optimizer.state


@pytest.mark.parametrize("missing", ["snapshot", "weights", "step", "another_rank"])
def test_required_resume_refuses_missing_or_incompatible_state_before_any_update(monkeypatch, missing):
    model = _model()
    _steps(model, 12, hint=0)
    published = _published(model)
    training.reset_training_state()
    hint = 12
    if missing == "snapshot":
        for path in optimizer_state.state_dir().iterdir():
            path.unlink()
    elif missing == "weights":
        published["weight"] = published["weight"] + 1
    elif missing == "step":
        hint = 11
        for path in optimizer_state.state_dir().glob("*step00000011.pt"):
            path.unlink()
    else:
        monkeypatch.setattr(training, "_all_ranks_agree", lambda local_ok: False)
    monkeypatch.setenv("RELIQUARY_REQUIRE_OPTIMIZER_RESUME", "1")
    resumed = _resumed(published)
    before = [p.detach().clone() for p in resumed.parameters()]
    with pytest.raises(RuntimeError, match="matching optimizer state is required"):
        training._lazy_init(resumed, global_step_hint=hint)
    assert training._scheduler is None and training._optimizer_model_id is None
    assert not training._optimizer.state
    assert all(torch.equal(a, b) for a, b in zip(before, resumed.parameters()))


def test_required_resume_keeps_genuinely_fresh_runs_valid(monkeypatch):
    monkeypatch.setenv("RELIQUARY_REQUIRE_OPTIMIZER_RESUME", "1")
    assert training._lazy_init(_model(), global_step_hint=0)
    assert not training._optimizer.state and training._scheduler is not None


def test_snapshot_of_other_weights_is_refused():
    model = _model()
    _steps(model, 12, hint=0)
    other = _published(model)
    other["weight"] = other["weight"] + 1
    training.reset_training_state()
    assert training._lazy_init(_resumed(other), global_step_hint=12)
    assert training._scheduler.lr_lambdas[0](12) < 1.0
    assert not training._optimizer.state


def test_snapshot_at_other_step_is_refused():
    """Crash after a step that was never published: the checkpoint is one step behind the last snapshot."""
    model = _model()
    _steps(model, 12, hint=0)
    published = _published(model)
    _steps(model, 1, hint=0)
    training.reset_training_state()
    assert training._lazy_init(_resumed(published), global_step_hint=12)
    assert training._optimizer.state                     # the step-12 snapshot is still kept
    training.reset_training_state()
    assert training._lazy_init(_resumed(published), global_step_hint=11)
    assert not training._optimizer.state


def test_keeps_last_snapshots_only(tmp_path):
    _steps(_model(), 5, hint=0)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["rank00-of01-step00000004.pt", "rank00-of01-step00000005.pt"]


def test_off_without_dir(monkeypatch, tmp_path):
    monkeypatch.delenv(optimizer_state.STATE_DIR_ENV)
    _steps_no_wait = _model()
    assert training._lazy_init(_steps_no_wait, global_step_hint=0)
    training._save_optimizer_state()
    assert training._state_writer is None and not list(tmp_path.iterdir())


def _sharded_resume(state_dir, steps_before, steps_after, restart):
    """FSDP2 on gloo: train, optionally restart from the bf16 checkpoint plus the snapshots, train on."""
    import os

    import torch.distributed as dist

    from reliquary.trainer.distributed import gather_full_state_dict, init_distributed, shard_model
    from reliquary.validator import training
    from tests.unit.test_train_step_sharded import _batches
    from tests.unit.trainer_dist_helpers import tiny_qwen35

    os.environ[optimizer_state.STATE_DIR_ENV] = state_dir
    ctx = init_distributed(device_type="cpu")
    training.reset_training_state()
    training.KL_BETA = 0.0
    training.configure_distributed(ctx)
    model = tiny_qwen35()
    batches = [_batches(model, seed=s) for s in range(steps_before + steps_after)]
    model = shard_model(model, ctx, param_dtype=torch.float32)
    for batch in batches[:steps_before]:
        training.train_step(model, batch, ref_model=None, global_step_hint=0)
    restored = None
    if restart:
        published = [gather_full_state_dict(model, ctx)]
        dist.broadcast_object_list(published, src=0)
        training.reset_training_state()
        model = tiny_qwen35()
        model.load_state_dict({k: v.float() for k, v in published[0].items()})
        model = shard_model(model, ctx, param_dtype=torch.float32)
        assert training._lazy_init(model, global_step_hint=steps_before)
        restored = bool(training._optimizer.state)
    for batch in batches[steps_before:]:
        training.train_step(model, batch, ref_model=None, global_step_hint=steps_before)
    training._state_writer.wait()
    full = {}
    for name, tensor in model.state_dict().items():
        full[name] = tensor.full_tensor().clone()
    return full, restored


def test_sharded_restart_continues_exactly(tmp_path):
    from tests.unit.trainer_dist_helpers import run_ranks

    straight = run_ranks(_sharded_resume, 2, str(tmp_path / "a"), 2, 1, False)
    resumed = run_ranks(_sharded_resume, 2, str(tmp_path / "b"), 2, 1, True)
    for rank in (0, 1):
        assert resumed[rank][1] is True
        for name, tensor in straight[rank][0].items():
            assert torch.equal(tensor, resumed[rank][0][name]), name


def _forward_then_step(state_dir):
    """As in production: the sharded model runs forwards (pi_old, audits) before its first train step."""
    import os

    from torch.distributed.tensor import DTensor

    from reliquary.trainer.distributed import init_distributed, shard_model
    from reliquary.validator import training
    from reliquary.validator.training import _batched_completion_logprobs
    from tests.unit.test_train_step_sharded import _batches
    from tests.unit.trainer_dist_helpers import tiny_qwen35

    os.environ[optimizer_state.STATE_DIR_ENV] = state_dir
    ctx = init_distributed(device_type="cpu")
    training.reset_training_state()
    training.KL_BETA = 0.0
    training.configure_distributed(ctx)
    model = tiny_qwen35()
    batch = _batches(model, seed=0)
    model = shard_model(model, ctx, param_dtype=torch.float32)
    with torch.no_grad():                                    # leaves the unsharded weights registered
        x = torch.tensor([batch[0][0].rollouts[0].tokens])
        _batched_completion_logprobs(model, x, torch.ones_like(x), [8], [x.shape[1]])
    before = {}
    for name, p in model.named_parameters():
        before[name] = (p.full_tensor() if isinstance(p, DTensor) else p).detach().clone()
    training.train_step(model, batch, ref_model=None, global_step_hint=0)
    training._state_writer.wait()
    params, _ = training._stepped()
    unchanged = [name for name, p in model.named_parameters()
                 if torch.equal(before[name], (p.full_tensor() if isinstance(p, DTensor) else p).detach())]
    return unchanged, all(isinstance(p, DTensor) for p in params), sorted(os.listdir(state_dir))


def test_the_optimizer_steps_the_shards_after_a_forward_left_weights_unsharded(tmp_path):
    """teutonic-rl-async-v1 2026-10-24: the optimizer held the base's and the head's unsharded weights (FSDP2
    registers them after a forward); embed_tokens, model.norm and lm_head never moved and the snapshot failed."""
    from tests.unit.trainer_dist_helpers import run_ranks

    out = run_ranks(_forward_then_step, 2, str(tmp_path))
    for rank in (0, 1):
        unchanged, sharded, files = out[rank]
        assert unchanged == [] and sharded
        assert any(f.startswith(f"rank{rank:02d}-") for f in files)


def _required_resume_missing_one_rank(state_dir):
    import os
    import torch.distributed as dist
    from reliquary.trainer.distributed import gather_full_state_dict, init_distributed, shard_model
    from tests.unit.test_train_step_sharded import _batches
    from tests.unit.trainer_dist_helpers import tiny_qwen35

    os.environ[optimizer_state.STATE_DIR_ENV] = state_dir
    os.environ['RELIQUARY_REQUIRE_OPTIMIZER_RESUME'] = '1'
    ctx = init_distributed(device_type='cpu')
    training.reset_training_state()
    training.KL_BETA = 0.0
    training.configure_distributed(ctx)
    model = tiny_qwen35()
    batches = _batches(model, seed=0)
    model = shard_model(model, ctx, param_dtype=torch.float32)
    training.train_step(model, batches, ref_model=None, global_step_hint=0)
    training._state_writer.wait()
    published = [gather_full_state_dict(model, ctx)]
    dist.broadcast_object_list(published, src=0)
    training.reset_training_state()
    if ctx.rank == 1:
        optimizer_state._path(optimizer_state.state_dir(), ctx.rank, ctx.world_size, 1).unlink()
    dist.barrier()
    model = tiny_qwen35()
    model.load_state_dict({k: v.float() for k, v in published[0].items()})
    model = shard_model(model, ctx, param_dtype=torch.float32)
    try:
        training._lazy_init(model, global_step_hint=1)
    except RuntimeError as exc:
        return ('matching optimizer state is required' in str(exc), training._scheduler is None,
                not training._optimizer.state)
    return False, False, False


def test_required_resume_aborts_every_rank_when_one_snapshot_is_missing(tmp_path):
    from tests.unit.trainer_dist_helpers import run_ranks

    out = run_ranks(_required_resume_missing_one_rank, 2, str(tmp_path))
    assert out == {0: (True, True, True), 1: (True, True, True)}
