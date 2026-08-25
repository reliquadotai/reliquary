"""Where the validator's OWN model replicas belong.

Measured on the live validator 2026-08-25 (window 32796, isolated plane,
detached trainer): the main process held 31.4 GB of the 80 GB card while
``nvidia-smi pmon`` reported ``sm = 0`` across 76 consecutive samples
spanning a full proof burst. Three replicas were resident — ``train_model``
(never trained, and holding weights the code explicitly refuses to read —
see the detached-mode guard in ``_ensure_proof_scheduler_ready``),
``verify_model`` (refreshed at every swap, never executed), and the
worker's — and exactly one of them ran a kernel.

Under an isolated proof plane this process neither trains (the detached
trainer owns that; isolation requires it) nor proves (every proof runs in a
worker that loads its own replica). These tests pin that its pair stays off
the proof device, and that the in-process plane still fails closed when no
replica is available on a configured device.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch.nn as nn

import reliquary.validator.service as service_mod
from reliquary.validator.proof_worker import (
    ProofModelProxy,
    validator_replica_device,
)


class _FakeEnv:
    name = "fake"

    def __len__(self):
        return 100

    def get_problem(self, _index):
        return {"prompt": "p", "ground_truth": "a"}

    def compute_reward(self, _problem, _completion):
        return 0.0


def _service(model, **kwargs):
    return service_mod.ValidationService(
        wallet=MagicMock(hotkey=MagicMock(ss58_address="x")),
        model=model,
        tokenizer=MagicMock(),
        env=_FakeEnv(),
        netuid=99,
        **kwargs,
    )


def test_an_isolated_plane_keeps_the_validator_replicas_off_the_gpu():
    assert validator_replica_device(isolated_plane=True) == "cpu"


def test_no_isolated_plane_keeps_the_validator_replicas_on_the_gpu():
    assert validator_replica_device(isolated_plane=False) == "cuda:0"


def test_resume_load_places_the_replica_where_the_plane_says(monkeypatch):
    """``load_validator_replica`` is the resume path; it must not pin cuda:0."""
    import reliquary.shared.modeling as modeling

    placed: list[str] = []

    class _Model:
        def to(self, device):
            placed.append(device)
            return self

        def eval(self):
            return self

    monkeypatch.setattr(
        modeling, "load_text_generation_model", lambda *a, **k: _Model()
    )
    service_mod.load_validator_replica("/x", isolated_plane=True)
    service_mod.load_validator_replica("/x", isolated_plane=False)

    assert placed == ["cpu", "cuda:0"]


def test_a_service_with_no_worker_pool_resumes_onto_the_gpu(monkeypatch):
    """The flag alone is not the question — is there actually a plane?

    RELIQUARY_PROOF_PROCESS_ISOLATION can be on while no plane is built: a
    protocol profile below v3 resolves NO proof devices, so the pool and the
    scheduler are never created and the batcher proves in-process against
    this replica. Deciding on the flag alone would leave a
    flash-attention-2 model on the CPU and prove every rollout there, with
    no error to say so.
    """
    import reliquary.shared.modeling as modeling

    placed: list[str] = []

    class _Model:
        def to(self, device):
            placed.append(device)
            return self

        def eval(self):
            return self

    monkeypatch.setattr(
        modeling, "load_text_generation_model", lambda *a, **k: _Model()
    )
    no_plane = _service(nn.Linear(4, 4))
    no_plane._load_model_fn("/x")
    assert placed == ["cuda:0"]

    placed.clear()
    with_plane = _service(
        nn.Linear(4, 4),
        proof_devices=("cuda:0",),
        proof_models={"cuda:0": ProofModelProxy(device_id="cuda:0")},
        proof_worker_pool=MagicMock(),
    )
    with_plane._load_model_fn("/x")
    assert placed == ["cpu"]


def test_replica_loader_forwards_the_pinned_revision(monkeypatch):
    """The CLI's boot load pins a base-model revision; the resume load does
    not. One loader serves both so the device decision cannot drift between
    them — the CLI call site is then a delegation with no logic of its own.
    """
    import reliquary.shared.modeling as modeling

    seen: list[dict] = []

    class _Model:
        def to(self, device):
            seen.append({"device": device})
            return self

        def eval(self):
            return self

    def _load(source, **kwargs):
        seen.append({"source": source, **kwargs})
        return _Model()

    monkeypatch.setattr(modeling, "load_text_generation_model", _load)
    service_mod.load_validator_replica(
        "owner/repo", isolated_plane=True, revision="b" * 40
    )

    assert seen[0]["source"] == "owner/repo"
    assert seen[0]["revision"] == "b" * 40
    assert seen[1]["device"] == "cpu"


def test_isolated_service_proves_through_the_proxy_not_its_own_replica():
    """The wiring must survive a verify_model that is no longer on the GPU.

    ``ValidationService.__init__`` falls back to ``self.verify_model`` for a
    configured proof device when no replica is supplied. With an isolated
    plane the proxies are supplied, so the fallback must not fire — otherwise
    moving the pair to the CPU would silently route proofs at a replica that
    cannot serve them.
    """
    proxy = ProofModelProxy(device_id="cuda:0")
    svc = _service(
        nn.Linear(4, 4),  # on the CPU, as an isolated plane leaves it
        proof_devices=("cuda:0",),
        proof_models={"cuda:0": proxy},
        proof_worker_pool=MagicMock(),
    )

    assert svc._proof_models["cuda:0"] is proxy
    assert svc._proof_models["cuda:0"] is not svc.verify_model


def test_in_process_plane_still_refuses_a_device_with_no_replica():
    """Fail closed: a CPU-only replica cannot serve a cuda proof device."""
    with pytest.raises(RuntimeError, match="no model replica"):
        _service(nn.Linear(4, 4), proof_devices=("cuda:0",))
