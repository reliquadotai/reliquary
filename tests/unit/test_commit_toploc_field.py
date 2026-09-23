"""The RL commit may carry TOPLOC proofs; it may never carry the thresholds."""

import pytest
from pydantic import ValidationError

from reliquary.protocol.submission import CommitModel
from tests.unit.test_grpo_window_batcher import _make_commit


def test_a_commit_without_proofs_is_unchanged():
    assert CommitModel.model_validate(_make_commit()).toploc_proofs is None


def test_a_commit_carries_bounded_proofs():
    commit = _make_commit()
    commit["toploc_proofs"] = ["A" * 344]
    assert CommitModel.model_validate(commit).toploc_proofs == ["A" * 344]


def test_proofs_are_bounded_by_the_commit_length():
    commit = _make_commit()           # 36 tokens
    commit["toploc_proofs"] = ["A" * 344] * 3
    with pytest.raises(ValidationError):
        CommitModel.model_validate(commit)


def test_a_miner_cannot_send_the_thresholds():
    commit = _make_commit()
    commit["toploc_spec"] = {"scheme": "toploc-v1", "mode": "enforce", "exp_mismatch_threshold": 128}
    with pytest.raises(ValidationError):
        CommitModel.model_validate(commit)
