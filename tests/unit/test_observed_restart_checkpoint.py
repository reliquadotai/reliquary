"""An observed-live controller may restart onto the checkpoint its worker has
already adopted, but only one its own durable state recorded."""

from types import SimpleNamespace

import pytest

from reliquary.validator import observed_proof_rollout as o


def _pool(checkpoint=SimpleNamespace(checkpoint_n=7, revision="a" * 40)):
    return SimpleNamespace(health=SimpleNamespace(checkpoint=checkpoint))


def test_without_the_flag_nothing_is_resumed(monkeypatch):
    monkeypatch.delenv("RELIQUARY_PROOF_RESUME_ADOPTED_CHECKPOINT", raising=False)
    assert o.observed_restart_checkpoint(_pool()) is None


def test_the_flag_without_durable_state_refuses(monkeypatch):
    monkeypatch.setenv("RELIQUARY_PROOF_RESUME_ADOPTED_CHECKPOINT", "1")
    monkeypatch.setattr(o, "observed_live_requested", lambda: True)
    monkeypatch.delenv("RELIQUARY_STATE_DIR", raising=False)
    with pytest.raises(ValueError, match="durable state"):
        o.observed_restart_checkpoint(_pool())


def test_a_checkpoint_the_controller_never_recorded_refuses(monkeypatch, tmp_path):
    monkeypatch.setenv("RELIQUARY_PROOF_RESUME_ADOPTED_CHECKPOINT", "1")
    monkeypatch.setenv("RELIQUARY_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(o, "observed_live_requested", lambda: True)
    with pytest.raises(ValueError, match="absent from durable controller state"):
        o.observed_restart_checkpoint(_pool())
