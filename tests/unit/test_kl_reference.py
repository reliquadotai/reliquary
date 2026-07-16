"""Fixed KL-reference lifecycle and reproducibility contract."""

from __future__ import annotations

import copy
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

import reliquary.validator.service as service_mod


class _FakeEnv:
    name = "fake"

    def __len__(self):
        return 100

    def get_problem(self, _index):
        return {"prompt": "p", "ground_truth": "a"}

    def compute_reward(self, _problem, _completion):
        return 0.0


def _service(model, *, load_model_fn=None):
    return service_mod.ValidationService(
        wallet=MagicMock(hotkey=MagicMock(ss58_address="x")),
        model=model,
        tokenizer=MagicMock(),
        env=_FakeEnv(),
        netuid=99,
        load_model_fn=load_model_fn,
    )


def test_fixed_kl_reference_is_pinned_frozen_and_observable(
    monkeypatch, tmp_path,
):
    revision = "a" * 40
    snapshot = tmp_path / "snapshots" / revision
    snapshot.mkdir(parents=True)
    download_calls = []

    def fake_snapshot_download(**kwargs):
        download_calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(service_mod, "KL_BASE_MODEL", f"owner/repo@{revision}")
    monkeypatch.setattr(service_mod, "KL_BETA_EXPLICIT", True)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", fake_snapshot_download
    )

    train = nn.Linear(4, 4)
    svc = _service(train, load_model_fn=lambda _path: copy.deepcopy(train))

    assert download_calls == [{"repo_id": "owner/repo", "revision": revision}]
    assert svc.base_ref_model is not None
    assert svc.base_ref_model is not svc.verify_model
    assert not svc.base_ref_model.training
    assert all(not p.requires_grad for p in svc.base_ref_model.parameters())
    assert svc.kl_reference_state == {
        "schema_version": 1,
        "mode": "fixed",
        "beta": service_mod.KL_BETA,
        "requested_model": f"owner/repo@{revision}",
        "repo_id": "owner/repo",
        "requested_revision": revision,
        "resolved_revision": revision,
        "loaded": True,
        "device": "cpu",
        "storage_bytes": 80,
    }
    assert (
        svc.server._health_payload().training_kl_reference
        == svc.kl_reference_state
    )

    # Publishing refreshes verify_model, but the fixed anchor must not move.
    anchor_before = copy.deepcopy(svc.base_ref_model.state_dict())
    with torch.no_grad():
        for parameter in svc.train_model.parameters():
            parameter.add_(1.0)
    svc.verify_model.load_state_dict(svc.train_model.state_dict())
    for name, value in svc.base_ref_model.state_dict().items():
        assert torch.equal(value, anchor_before[name])


@pytest.mark.parametrize(
    "spec",
    [
        "owner/repo",
        "owner/repo@main",
        "owner/repo@abc123",
        "@" + "a" * 40,
    ],
)
def test_fixed_kl_reference_rejects_mutable_or_malformed_specs(
    monkeypatch, spec,
):
    monkeypatch.setattr(service_mod, "KL_BASE_MODEL", spec)
    with pytest.raises(ValueError, match="KL_BASE_MODEL"):
        _service(nn.Linear(2, 2))


def test_fixed_kl_reference_load_failure_is_fatal(monkeypatch):
    revision = "b" * 40
    monkeypatch.setattr(service_mod, "KL_BASE_MODEL", f"owner/repo@{revision}")
    monkeypatch.setattr(service_mod, "KL_BETA_EXPLICIT", True)

    def fail_download(**_kwargs):
        raise ConnectionError("HF down")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fail_download)

    with pytest.raises(RuntimeError, match="failed to load required fixed"):
        _service(nn.Linear(2, 2))


def test_fixed_kl_reference_rejects_unexpected_resolved_sha(
    monkeypatch, tmp_path,
):
    requested = "b" * 40
    resolved = "c" * 40
    snapshot = tmp_path / "snapshots" / resolved
    snapshot.mkdir(parents=True)
    monkeypatch.setattr(
        service_mod, "KL_BASE_MODEL", f"owner/repo@{requested}"
    )
    monkeypatch.setattr(service_mod, "KL_BETA_EXPLICIT", True)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **_kwargs: str(snapshot),
    )

    with pytest.raises(RuntimeError, match="failed to load required fixed"):
        _service(nn.Linear(2, 2), load_model_fn=lambda _path: nn.Linear(2, 2))


def test_default_kl_reference_remains_rolling(monkeypatch):
    monkeypatch.setattr(service_mod, "KL_BASE_MODEL", "")
    svc = _service(nn.Linear(2, 2))

    assert svc.base_ref_model is None
    assert svc.kl_reference_state["mode"] == "rolling"
    assert svc.kl_reference_state["loaded"] is True


def test_fixed_kl_reference_requires_explicit_beta(monkeypatch):
    revision = "d" * 40
    monkeypatch.setattr(service_mod, "KL_BASE_MODEL", f"owner/repo@{revision}")
    monkeypatch.setattr(service_mod, "KL_BETA_EXPLICIT", False)

    with pytest.raises(ValueError, match="explicit RELIQUARY_KL_BETA"):
        _service(nn.Linear(2, 2))
