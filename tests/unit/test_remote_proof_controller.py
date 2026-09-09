"""CPU-controller startup/adoption uses metadata, with no local model load."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from reliquary.validator.proof_worker import ProofModelProxy, ProofWorkerUnavailable

REV = "a" * 40


class MetadataPool:
    is_remote = True
    devices = ("cuda:0",)
    runtime_fingerprint = {}

    def __init__(self):
        self.bound = self.installed = None
        self.fail = False
        self.calls = []

    def bind_checkpoint(self, n, repo, revision):
        self.bound = (n, repo, revision)

    def revision(self, device):
        return self.installed

    def reload(self, device, path, revision, repo):
        assert self.bound == (7, repo, revision)
        self.calls.append(("adopt", revision))
        if self.fail:
            raise ProofWorkerUnavailable("GPU adoption failed")
        self.installed = revision

    def assert_ready(self):
        if self.installed != REV:
            raise ProofWorkerUnavailable("not adopted")

    def readiness_snapshot(self):
        return {"ready": self.installed == REV}

    def verifier_for_window(self, window, env, revision):
        def verify(commit, model, randomness, **kwargs):
            assert isinstance(model, ProofModelProxy)
            self.calls.append(("prove", window, env, revision, model.device_id))
            return "complete-proof"
        return verify


class Environment:
    name = "fake"
    def __len__(self):
        return 100
    def get_problem(self, _index):
        return {"prompt": "p", "ground_truth": "a"}
    def compute_reward(self, _problem, _completion):
        return 0.


@pytest.fixture
def controller(monkeypatch, tmp_path):
    import reliquary.constants as constants
    import reliquary.validator.service as module
    import huggingface_hub
    from reliquary.validator.checkpoint_profile import write_checkpoint_profile

    monkeypatch.setattr(constants, "DETACHED_TRAINER", True)
    monkeypatch.setattr(module, "KL_BASE_MODEL", "")
    monkeypatch.setattr(module.telemetry, "init", lambda **kw: None)
    monkeypatch.setattr(module, "load_validator_replica", lambda *a, **kw: pytest.fail("local weights loaded"))
    monkeypatch.setattr(module, "FILL_CLOSED_ENABLED", False)
    pool = MetadataPool()
    proxy = ProofModelProxy("cuda:0", SimpleNamespace(eos_token_id=2), SimpleNamespace())
    service = module.ValidationService(
        wallet=MagicMock(hotkey=MagicMock(ss58_address="test-controller")), model=proxy,
        tokenizer=MagicMock(), env=Environment(), netuid=99,
        proof_worker_pool=pool, proof_devices=pool.devices,
        proof_models={"cuda:0": proxy}, resume_from=f"sha:{REV}", hf_repo_id="test/checkpoints")
    write_checkpoint_profile(tmp_path, {"lr_schedule_step": 12})
    downloads = []
    def download(**kwargs):
        downloads.append(kwargs)
        return str(tmp_path)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(
        list_repo_commits=lambda **kwargs: [SimpleNamespace(commit_id=REV, title="checkpoint 7 (test)")]))
    def install(n, revision):
        assert pool.installed == revision, "manifest advertised before GPU acknowledgement"
        pool.calls.append(("manifest", revision))
        return SimpleNamespace(checkpoint_n=n, revision=revision)
    service._checkpoint_store = SimpleNamespace(repo_id="test/checkpoints", install_external=install)
    service.server = MagicMock()
    try:
        yield service, pool, downloads
    finally:
        service.proof_scheduler.close(timeout=2)


def test_resume_metadata_only_and_adopt_before_publication(controller):
    service, pool, downloads = controller
    assert service.train_model is None
    assert isinstance(service.verify_model, ProofModelProxy)
    asyncio.run(service._apply_resume_from())
    assert pool.calls == [("adopt", REV), ("manifest", REV)]
    assert service._verify_model_checkpoint_revision == REV
    assert service._checkpoint_n == 7
    assert service._resumed_lr_schedule_step == 12
    assert downloads[0]["revision"] == REV
    assert "allow_patterns" in downloads[0]
    assert all("safetensors" not in p and "bin" not in p for p in downloads[0]["allow_patterns"])
    assert service.proof_scheduler.checkpoint_ready(REV)


def test_failed_adoption_leaves_manifest_unpublished(controller):
    service, pool, _ = controller
    pool.fail = True
    with pytest.raises(ProofWorkerUnavailable):
        asyncio.run(service._apply_resume_from())
    service.server.set_current_checkpoint.assert_not_called()
    assert service._verify_model_checkpoint_revision is None
    assert not service.proof_scheduler.checkpoint_ready(REV)


def test_scheduled_forensic_and_legacy_batchers_use_same_bound_network_verifier(controller, monkeypatch):
    import reliquary.validator.service as module
    service, pool, _ = controller
    asyncio.run(service._apply_resume_from())
    captured = []
    def open_window(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace()
    monkeypatch.setattr(module, "open_grpo_window", open_window)
    service._checkpoint_store.current_manifest = lambda: SimpleNamespace(revision=REV)
    service._build_window_batchers(42)
    assert len(captured) == 1
    arguments = captured[0]
    assert arguments["proof_scheduler"] is service.proof_scheduler
    assert isinstance(arguments["model"], ProofModelProxy)
    # Forensic / legacy calls use the batcher model; scheduled calls supply a
    # chosen pool proxy. Both reach the very same immutable-window closure.
    for model in (arguments["model"], service._proof_models["cuda:0"]):
        assert arguments["verify_commitment_proofs_fn"]({}, model, "ab") == "complete-proof"
    assert pool.calls[-2:] == [("prove", 42, "fake", REV, "cuda:0")] * 2


def test_cli_remote_boot_never_resolves_cuda_or_loads_local_weights(monkeypatch):
    from typer.testing import CliRunner
    import reliquary.cli.main as cli
    import reliquary.constants as constants
    import reliquary.infrastructure.chain as chain
    import reliquary.shared.modeling as modeling
    import reliquary.validator.remote_proof as remote
    import reliquary.validator.service as service_module
    import reliquary.validator.weight_only as weights
    import bittensor

    monkeypatch.setenv("RELIQUARY_PROOF_EXECUTOR_MODE", "remote")
    monkeypatch.setattr(constants, "DETACHED_TRAINER", True)
    monkeypatch.setattr(constants, "KL_BASE_MODEL", "")
    monkeypatch.setattr(cli, "_resolve_cli_environment_mix", lambda _v: [("fake", 1)])
    monkeypatch.setattr(cli, "_v3_activation_checkpoint_revision", lambda *a: REV)
    monkeypatch.setattr(cli, "_configured_proof_device_identities", lambda *a: pytest.fail("CUDA topology on CPU controller"))
    monkeypatch.setattr(modeling, "load_tokenizer", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(modeling, "load_text_generation_model", lambda *a, **kw: pytest.fail("local model loaded"))
    monkeypatch.setattr(service_module, "load_validator_replica", lambda *a, **kw: pytest.fail("local replica loaded"))
    monkeypatch.setattr(bittensor, "Wallet", lambda **kw: SimpleNamespace())
    async def subtensor():
        return SimpleNamespace()
    monkeypatch.setattr(chain, "get_subtensor", subtensor)
    pool = MetadataPool()
    pool.start = lambda: None
    pool.proxies = lambda: {"cuda:0": ProofModelProxy("cuda:0")}
    pool.qualify = lambda revision: {"revision": revision}
    monkeypatch.setattr(remote.RemoteProofPool, "from_environment", lambda **kw: pool)
    seen = []
    class Service:
        def __init__(self, wallet, model, tokenizer, **kwargs):
            assert isinstance(model, ProofModelProxy)
            assert kwargs["proof_worker_pool"] is pool
            assert kwargs["proof_devices"] == ("cuda:0",)
            seen.append(kwargs)
        async def run(self, subtensor):
            pass
    class WeightSetter:
        def __init__(self, **kwargs):
            pass
        async def run(self):
            pass
    monkeypatch.setattr(service_module, "ValidationService", Service)
    monkeypatch.setattr(weights, "WeightOnlyValidator", WeightSetter)
    result = CliRunner().invoke(cli.app, ["validate", "--resume-from", f"sha:{REV}"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert len(seen) == 1


def test_capacity_qualifier_checks_network_scope_in_every_sample(tmp_path):
    import importlib.util
    import json
    from pathlib import Path
    from reliquary.validator.remote_proof_protocol import RemoteProofMeasurement, transport_hash

    script = Path(__file__).resolve().parents[2] / "scripts/qualify_proof_capacity.py"
    spec = importlib.util.spec_from_file_location("remote_capacity_test", script)
    qualifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qualifier)
    marker = RemoteProofMeasurement(worker_id="proof-test", transport_sha256=transport_hash()).model_dump()
    rows = [dict(environment=environment, seconds=.1, proof_passed=True,
        profile_id=qualifier.PROTOCOL_PROFILE_ID, model_revision=qualifier.PROTOCOL_MODEL_REVISION,
        software_revision=REV, checkpoint_revision=REV, runtime_fingerprint_hash="b" * 64,
        hardware_class="test-only", device_uuid="test-gpu",
        rollout_count=qualifier.ROLLOUTS_PER_PROOF,
        completion_token_lengths=[qualifier.MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV[environment]] * qualifier.ROLLOUTS_PER_PROOF,
        remote_proof=marker)
        for environment in qualifier.ENVIRONMENTS for _ in range(20)]
    path = tmp_path / "samples.jsonl"
    kwargs = dict(software_revision=REV, checkpoint_revision=REV, runtime_fingerprint_hash="b" * 64,
        hardware_class="test-only", benchmark_device_count=1, remote_proof=marker)
    path.write_text("\n".join(json.dumps(row) for row in rows))
    qualifier._load_samples(path, **kwargs)
    del rows[-1]["remote_proof"]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="remote proof measurement"):
        qualifier._load_samples(path, **kwargs)


def test_worker_adoption_binds_real_published_number_profile_and_loaded_oid(monkeypatch, tmp_path):
    import huggingface_hub
    from reliquary.validator.remote_proof_server import ProofBackend
    from reliquary.validator.remote_proof_protocol import CheckpointBinding
    from reliquary.validator.checkpoint_profile import active_checkpoint_profile, write_checkpoint_profile

    profile = active_checkpoint_profile()
    # The test's historical default profile has no hash. The production
    # endpoint is V5+, where write_checkpoint_profile already includes one.
    profile.setdefault("generation_contract_sha256", "b" * 64)
    from reliquary.validator.checkpoint_profile import CHECKPOINT_PROFILE_NAME
    import json
    (tmp_path / CHECKPOINT_PROFILE_NAME).write_text(json.dumps(profile))
    cp = CheckpointBinding(profile_id=profile["profile_id"],
        generation_contract_sha256=profile["generation_contract_sha256"],
        training_run_id=profile["training_run_id"], checkpoint_n=7,
        repo_id="test/checkpoints", revision=REV)
    commits = [SimpleNamespace(commit_id=REV, title="checkpoint 7 (test)")]
    def api(**kwargs):
        assert kwargs == {"token": False}
        return SimpleNamespace(list_repo_commits=lambda **kw: commits)
    monkeypatch.setattr(huggingface_hub, "HfApi", api)
    def download(repo, name, **kwargs):
        assert (repo, name, kwargs) == (cp.repo_id, CHECKPOINT_PROFILE_NAME,
                                       {"revision": REV, "token": False})
        return str(tmp_path / name)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    pool = MetadataPool()
    pool.bound = (7, cp.repo_id, REV)
    pool.describe = lambda device: {"revision": pool.installed}
    backend = ProofBackend(pool)
    backend.adopt(cp)
    assert pool.installed == REV
    with pytest.raises(ValueError, match="number"):
        backend.adopt(cp.model_copy(update={"checkpoint_n": 8}))
    commits.append(SimpleNamespace(commit_id="d" * 40, title="checkpoint 7 (duplicate)"))
    with pytest.raises(ValueError, match="ambiguous"):
        backend.adopt(cp)
    commits.pop()
    profile["training_run_id"] = "another-run"
    (tmp_path / CHECKPOINT_PROFILE_NAME).write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="training_run_id"):
        backend.adopt(cp)
    # Actual process acknowledgement, not parent revision cache, is decisive.
    profile["training_run_id"] = cp.training_run_id
    (tmp_path / CHECKPOINT_PROFILE_NAME).write_text(json.dumps(profile))
    pool.describe = lambda device: {"revision": None}
    with pytest.raises(ProofWorkerUnavailable, match="acknowledge"):
        backend.adopt(cp)


def test_server_publishes_gpu_fingerprint_without_recollecting_cpu_runtime(monkeypatch):
    import reliquary.validator.server as module
    from reliquary.shared.runtime_fingerprint import runtime_profile_hash

    server = module.ValidatorServer.__new__(module.ValidatorServer)
    server._active_batchers = {}
    server._reset_window_scoped_state = lambda: None
    runtime = {"cuda_available": True, "gpu_name": "test-GPU"}
    runtime["profile_hash"] = runtime_profile_hash(runtime)
    server.set_proof_runtime_fingerprint(runtime)
    monkeypatch.setattr(module, "collect_runtime_fingerprint", lambda **kwargs: pytest.fail("CPU runtime substituted"))
    server.set_active_batchers({"fake": SimpleNamespace(model=ProofModelProxy("cuda:0"))})
    assert server._runtime_fingerprint == runtime
    with pytest.raises(ValueError):
        server.set_proof_runtime_fingerprint({"cuda_available": False})


def test_remote_unavailability_is_visible_before_new_window(controller):
    from reliquary.validator.service import FatalProofPlaneError
    service, pool, _ = controller
    asyncio.run(service._apply_resume_from())
    service._checkpoint_store.current_manifest = lambda: SimpleNamespace(revision=REV)
    assert service._proof_scheduler_health_snapshot()["remote_proof"]["ready"] is True
    pool.installed = None
    snapshot = service._proof_scheduler_health_snapshot()
    assert snapshot["remote_proof"]["ready"] is False
    assert "remote_proof_unavailable" in snapshot["degraded_reasons"]
    with pytest.raises(FatalProofPlaneError):
        asyncio.run(service._ensure_proof_scheduler_ready())
