from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

import reliquary.validator.batcher as batcher_module
from reliquary.validator.batcher import GrpoWindowBatcher, ValidSubmission
from reliquary.validator.difficulty_auction import (
    ShadowSubmission,
    select_shadow_auction as real_select_shadow_auction,
)


class MathEnv:
    name = "openmathinstruct"

    def __len__(self):
        return 100

    def get_problem(self, prompt_idx):
        return {"prompt": f"p{prompt_idx}", "ground_truth": "a"}


class CodeEnv(MathEnv):
    name = "opencodeinstruct"


def _batcher(env=None):
    return GrpoWindowBatcher(
        window_start=500,
        env=env or MathEnv(),
        model=SimpleNamespace(),
        completion_text_fn=lambda _rollout: "",
        verify_commitment_proofs_fn=lambda *_args, **_kwargs: None,
        verify_signature_fn=lambda *_args, **_kwargs: True,
        drand_round_check_enabled=False,
    )


def _submission(hotkey, prompt_idx, drand_round, k):
    digest = hotkey.encode().ljust(32, b"\x00")[:32]
    return ValidSubmission(
        hotkey=hotkey,
        prompt_idx=prompt_idx,
        merkle_root_bytes=digest,
        selection_digest_bytes=digest,
        drand_round=drand_round,
        rollouts=[SimpleNamespace(reward=1.0)] * k
        + [SimpleNamespace(reward=0.0)] * (8 - k),
    )


def test_shadow_does_not_change_production_fcfs_selection_or_rewards():
    batcher = _batcher()
    easy = [
        _submission(f"easy{i}", i, drand_round=1, k=6)
        for i in range(8)
    ]
    hard_late = _submission("hard", 99, drand_round=9, k=2)
    batcher._valid = easy + [hard_late]
    batcher.valid_count = len(batcher._valid)

    production_batch, rewards = batcher.seal_batch()

    assert {submission.hotkey for submission in production_batch} == {
        f"easy{i}" for i in range(8)
    }
    assert rewards == {f"easy{i}": pytest.approx(1 / 8) for i in range(8)}
    shadow = batcher.difficulty_auction_shadow
    assert shadow["mode"] == "observation_only"
    assert shadow["production_changed"] is False
    assert shadow["status"] == "computed"
    assert shadow["shadow_selected_count"] == 8
    assert shadow["selection_overlap_count"] == 7
    hard_row = next(
        row for row in shadow["candidates"] if row["hotkey"] == "hard"
    )
    assert hard_row["shadow_selected"] is True
    assert hard_row["production_selected"] is False


def test_shadow_failure_cannot_fail_production_seal(monkeypatch):
    batcher = _batcher()
    batcher._valid = [_submission("miner", 1, drand_round=1, k=4)]
    batcher.valid_count = 1

    def _explode(*_args, **_kwargs):
        raise RuntimeError("shadow-only failure")

    monkeypatch.setattr(batcher_module, "select_shadow_auction", _explode)

    production_batch, rewards = batcher.seal_batch()

    assert [submission.hotkey for submission in production_batch] == ["miner"]
    assert rewards == {"miner": pytest.approx(1 / 8)}
    assert batcher.difficulty_auction_shadow["status"] == "error"
    assert batcher.difficulty_auction_shadow["error_type"] == "RuntimeError"


def test_shadow_receives_only_detached_immutable_values(monkeypatch):
    batcher = _batcher()
    live_submission = _submission("miner", 1, drand_round=1, k=4)
    batcher._valid = [live_submission]
    batcher.valid_count = 1

    def _inspect(pool, **kwargs):
        assert isinstance(pool, tuple)
        assert len(pool) == 1
        candidate = pool[0]
        assert isinstance(candidate, ShadowSubmission)
        assert candidate.source_id == id(live_submission)
        assert candidate is not live_submission
        assert candidate.rewards == (1.0,) * 4 + (0.0,) * 4
        with pytest.raises(FrozenInstanceError):
            candidate.prompt_idx = 99
        return real_select_shadow_auction(pool, **kwargs)

    monkeypatch.setattr(batcher_module, "select_shadow_auction", _inspect)

    production_batch, rewards = batcher.seal_batch()

    assert production_batch == [live_submission]
    assert rewards == {"miner": pytest.approx(1 / 8)}
    assert live_submission.prompt_idx == 1


def test_code_environment_is_explicitly_out_of_scope():
    batcher = _batcher(CodeEnv())
    batcher._valid = [_submission("miner", 1, drand_round=1, k=4)]
    batcher.valid_count = 1

    production_batch, _rewards = batcher.seal_batch()

    assert [submission.hotkey for submission in production_batch] == ["miner"]
    assert (
        batcher.difficulty_auction_shadow["status"]
        == "out_of_scope_environment"
    )
