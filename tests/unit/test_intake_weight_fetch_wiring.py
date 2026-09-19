"""The controller skips the checkpoint weights only when a remote plane proves.

The shadow pool compares a local proof against the remote one, so it needs the
weights on the controller. It must keep them: this is what the wiring guards.
"""

from types import SimpleNamespace

import pytest

import reliquary.shared.training_payload as training_payload
import reliquary.validator.checkpoint_intake as checkpoint_intake
from reliquary.validator.remote_proof import RemoteProofPool, ShadowProofPool
from reliquary.validator.service import ValidationService


class _Store:
    repo_id = "org/repo"

    def current_manifest(self):
        return None


@pytest.fixture
def captured(monkeypatch):
    calls = []

    def _intake(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(checkpoint_intake, "CheckpointIntake", _intake)
    monkeypatch.setattr(checkpoint_intake, "default_r2_client", lambda: object())
    monkeypatch.setattr(training_payload, "active_training_identity", lambda: {})
    return calls


def _service(**attrs):
    return SimpleNamespace(_checkpoint_intake=None, _checkpoint_store=_Store(), **attrs)


def test_only_the_pure_remote_pool_counts_as_network_proof():
    # _network_proof is `is_remote is True`; shadow must not qualify.
    assert RemoteProofPool.is_remote is True
    assert ShadowProofPool.is_remote is False


def test_remote_proof_plane_does_not_fetch_weights(captured):
    ValidationService._detached_intake_ref(_service(_network_proof=True))
    assert captured[-1]["fetch_weights"] is False


def test_local_or_shadow_proof_keeps_fetching_weights(captured):
    ValidationService._detached_intake_ref(_service(_network_proof=False))
    assert captured[-1]["fetch_weights"] is True


def test_unknown_proof_topology_fails_safe_to_fetching_weights(captured):
    ValidationService._detached_intake_ref(_service())
    assert captured[-1]["fetch_weights"] is True
