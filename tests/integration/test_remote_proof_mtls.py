"""Real HTTP + TLS, with only the GPU-owning backend replaced on CPU CI."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import socket
import ssl
import subprocess
import threading
import time

import httpx
import pytest
import uvicorn

from reliquary.shared.runtime_fingerprint import runtime_profile_hash
from reliquary.validator.proof_capacity import compute_proof_path_hash
from reliquary.validator.proof_worker import ProofWorkerUnavailable
from reliquary.validator.remote_proof import RemoteProofPool
from reliquary.validator.remote_proof_protocol import (
    AdoptionRequest, CheckpointBinding, ProofInput, ProofRequest, ProofValues, digest,
)
from reliquary.validator.remote_proof_server import create_proof_app
from reliquary.validator.verifier import ProofResult

REV = "a" * 40
IDENTITY = dict(profile_id="test-proof-profile", generation_contract_sha256="b" * 64,
                training_run_id="test-run", repo_id="test/checkpoints")


def kernel_result():
    return ProofResult(
        all_passed=True, passed=3, checked=3, has_sparse_outputs=True,
        p_stop=.1, challenge_lp_indices=[1, 2], challenge_lp_values=[-1., -2.],
        completion_chosen_probs=[.2, .3], completion_argmax_probs=[.4, .5],
        completion_argmax_ids=[2, 3], seed_n_positions=2, seed_n_stochastic=2,
        seed_n_match=2, terminal_pick_ok=True, terminal_pick_cdf_miss=0.,
        natural_close_pick_ok=False, natural_close_pick_cdf_miss=.1,
    )


class CPUProofBackend:
    """Only this test dependency stands in for GPU allocation/kernel calls."""
    devices = ("cuda:0",)

    def __init__(self):
        self.revision = None
        self.calls = 0
        self.adoptions = 0
        self.delay = 0
        self.fail = False
        self.partial_adoption = False
        self.started = threading.Event()
        self.finished = threading.Event()
        self.runtime = {"schema_version": 2, "cuda_available": True,
                        "gpu_name": "FAKE GPU FOR CPU TRANSPORT TEST ONLY"}
        self.runtime["profile_hash"] = runtime_profile_hash(self.runtime)

    def describe(self, device):
        return dict(device_id=device, physical_device="cuda:0",
                    hardware_class="FAKE GPU FOR CPU TRANSPORT TEST ONLY", device_uuid="gpu-test-0",
                    revision=self.revision, runtime=dict(self.runtime),
                    config={"eos_token_id": 3}, generation_config={"eos_token_id": 3})

    def adopt(self, checkpoint):
        self.adoptions += 1
        self.revision = None if self.partial_adoption else checkpoint.revision

    def prove(self, request):
        self.calls += 1
        self.started.set()
        try:
            time.sleep(self.delay)
            if self.fail:
                raise ProofWorkerUnavailable("injected GPU failure")
            assert request.checkpoint.revision == self.revision
            return kernel_result()
        finally:
            self.finished.set()


@pytest.fixture(scope="module")
def pki(tmp_path_factory):
    output = tmp_path_factory.mktemp("remote-proof") / "dedicated-pki"
    script = Path(__file__).resolve().parents[2] / "scripts/generate_signer_pki.sh"
    subprocess.run([str(script), str(output), "127.0.0.1"], check=True,
                   capture_output=True, text=True)
    return output


@contextmanager
def endpoint(pki, backend, *, tamper=None, timeout=2.):
    app = create_proof_app(backend=backend, worker_id="proof-test", **IDENTITY,
                          software_revision="c" * 40, proof_path_hash=compute_proof_path_hash())
    if tamper:
        from starlette.responses import Response

        @app.middleware("http")
        async def change_response(request, call_next):
            response = await call_next(request)
            if request.url.path != "/v1/prove" or response.status_code != 200:
                return response
            body = b"".join([part async for part in response.body_iterator])
            value = json.loads(body)
            tamper(value)
            return Response(json.dumps(value), media_type="application/json")

    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    config = uvicorn.Config(app, log_level="error", access_log=False,
        ssl_certfile=str(pki / "signer/server.crt"), ssl_keyfile=str(pki / "signer/server.key"),
        ssl_ca_certs=str(pki / "signer/ca.crt"), ssl_cert_reqs=ssl.CERT_REQUIRED)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(.01)
    assert server.started
    client = RemoteProofPool(base_url=f"https://127.0.0.1:{listener.getsockname()[1]}",
        ca_path=str(pki / "signer-client/ca.crt"), cert_path=str(pki / "signer-client/client.crt"),
        key_path=str(pki / "signer-client/client.key"), expected_worker_id="proof-test",
        **IDENTITY, request_timeout=timeout, reload_timeout=2.)
    try:
        client.start()
        client.bind_checkpoint(7, IDENTITY["repo_id"], REV)
        client.reload("cuda:0", "/path/never/sent/to/worker", REV, IDENTITY["repo_id"])
        yield client
    finally:
        client.close(force=True)
        server.should_exit = True
        thread.join(10)
        listener.close()
        assert not thread.is_alive()


def payload():
    return ProofInput(tokens=[1, 2, 3], commitments=[{"sketch": 0}] * 3,
        rollout={"prompt_length": 1, "completion_length": 2},
        randomness="ab" * 32, seed_u_values=[.1, .2])


def request_for(client, **changes):
    body = payload()
    values = dict(job_id="job-1", attempt=0, worker_id="proof-test",
        session_id=client.health.session_id, device_id="cuda:0",
        runtime_hash=client.runtime_fingerprint["profile_hash"], checkpoint=client._checkpoint,
        window=42, environment="openmathinstruct", expires_at_ms=int((time.time() + 5) * 1000),
        content_sha256=digest(body.model_dump()), payload=body)
    values.update(changes)
    return ProofRequest(**values)


def prove(client):
    value = payload()
    verify = client.verifier_for_window(42, "openmathinstruct", REV)
    return verify(value.commit(), client.proxies()["cuda:0"], value.randomness,
                  seed_u_values=value.seed_u_values)


def test_real_mtls_sparse_result_and_adoption(pki):
    backend = CPUProofBackend()
    with endpoint(pki, backend) as client:
        assert client.assert_ready().checkpoint.checkpoint_n == 7
        assert ProofValues.from_kernel(prove(client)) == ProofValues.from_kernel(kernel_result())
        assert backend.calls == 1
        assert client.revision("cuda:0") == REV
        assert client.runtime_fingerprint["gpu_name"].startswith("FAKE")
        no_identity = ssl.create_default_context(cafile=str(pki / "signer/ca.crt"))
        with pytest.raises(httpx.HTTPError):
            httpx.get(str(client._client.base_url).rstrip("/") + "/v1/health", verify=no_identity,
                      timeout=1, trust_env=False)


def test_retry_same_bytes_executes_once_and_rebinding_refused(pki):
    backend = CPUProofBackend()
    with endpoint(pki, backend) as client:
        request = request_for(client)
        first = client._request("POST", "/v1/prove", request, timeout=2)
        assert client._request("POST", "/v1/prove", request, timeout=2) == first
        assert backend.calls == 1
        with pytest.raises(ProofWorkerUnavailable, match="409"):
            client._request("POST", "/v1/prove", request.model_copy(update={"window": 43}), timeout=2)


@pytest.mark.parametrize("field,value", [
    ("window", 43), ("attempt", 1), ("job_id", "other-job"),
    ("content_sha256", "e" * 64), ("runtime_hash", "e" * 64),
    ("session_id", "other-session"), ("device_id", "cuda:1"),
])
def test_client_rejects_reassigned_or_stale_response(pki, field, value):
    backend = CPUProofBackend()
    with endpoint(pki, backend, tamper=lambda data: data.update({field: value})) as client:
        with pytest.raises(ProofWorkerUnavailable, match="binding"):
            prove(client)
        assert client.revision("cuda:0") is None


def test_rejects_old_checkpoint_result_and_boolean_only_result(pki):
    for tamper in (
        lambda data: data["checkpoint"].update(revision="d" * 40),
        lambda data: data.update(result={"all_passed": True}),
        lambda data: data["result"].update(has_sparse_outputs=False),
    ):
        with endpoint(pki, CPUProofBackend(), tamper=tamper) as client:
            with pytest.raises(ProofWorkerUnavailable):
                prove(client)


def test_server_refuses_expired_unknown_runtime_and_checkpoint(pki):
    backend = CPUProofBackend()
    with endpoint(pki, backend) as client:
        for changes in (
            {"expires_at_ms": 1}, {"device_id": "cuda:9"},
            {"runtime_hash": "f" * 64},
            {"checkpoint": CheckpointBinding(**IDENTITY, checkpoint_n=6, revision="f" * 40)},
        ):
            with pytest.raises(ProofWorkerUnavailable):
                client._request("POST", "/v1/prove", request_for(client, **changes), timeout=2)
        assert backend.calls == 0


def test_partial_adoption_never_becomes_ready_and_cannot_regress(pki):
    backend = CPUProofBackend()
    with endpoint(pki, backend) as client:
        backend.partial_adoption = True
        client.bind_checkpoint(8, IDENTITY["repo_id"], "d" * 40)
        with pytest.raises(ProofWorkerUnavailable):
            client.reload("cuda:0", None, "d" * 40, IDENTITY["repo_id"])
        assert client.revision("cuda:0") is None
        old = AdoptionRequest(worker_id=client.worker_id, session_id=client.health.session_id,
                               checkpoint=CheckpointBinding(**IDENTITY, checkpoint_n=7, revision=REV))
        with pytest.raises(ProofWorkerUnavailable, match="409"):
            client._request("POST", "/v1/adopt", old, timeout=2)
        backend.partial_adoption = False
        client.reload("cuda:0", None, "d" * 40, IDENTITY["repo_id"])
        assert client.assert_ready().checkpoint.checkpoint_n == 8


def test_disconnect_retains_gpu_slot_and_blocks_adoption(pki):
    backend = CPUProofBackend()
    backend.delay = .4
    with endpoint(pki, backend) as client:
        request = request_for(client)
        with pytest.raises(ProofWorkerUnavailable):
            client._request("POST", "/v1/prove", request, timeout=.05)
        assert backend.started.wait(1)
        with pytest.raises(ProofWorkerUnavailable, match="503"):
            client._request("POST", "/v1/prove", request_for(client, job_id="job-2"), timeout=1)
        adoption = AdoptionRequest(worker_id=client.worker_id, session_id=client.health.session_id,
                                   checkpoint=client._checkpoint)
        with pytest.raises(ProofWorkerUnavailable, match="503"):
            client._request("POST", "/v1/adopt", adoption, timeout=1)
        assert backend.finished.wait(2)
        assert backend.calls == 1


def test_gpu_timeout_and_fault_are_infrastructure_errors(pki):
    backend = CPUProofBackend()
    with endpoint(pki, backend, timeout=.05) as client:
        backend.delay = .2
        with pytest.raises(ProofWorkerUnavailable):
            prove(client)
        assert client.revision("cuda:0") is None
    backend = CPUProofBackend()
    with endpoint(pki, backend) as client:
        backend.fail = True
        from reliquary.validator.proof_scheduler import (
            GlobalProofScheduler, ProofDecisionStatus, ProofPlan, RankedProof,
            ProofPlanOutcome, SchedulerState,
        )
        scheduler = GlobalProofScheduler(devices=client.devices,
            environments=("openmathinstruct",), proof_callable=lambda _inv: prove(client))
        try:
            scheduler.mark_device_ready("cuda:0", REV)
            scheduler.resume(REV)
            result = scheduler.submit(ProofPlan(plan_id="test", environment="openmathinstruct",
                checkpoint_revision=REV, candidates=[RankedProof("job", 1, "prompt", payload=None, resources=(("miner", 1),))],
                required_passes=1, deadline_at=time.monotonic() + 5)).result(timeout=5)
            assert result.outcome is ProofPlanOutcome.CAPACITY_ABORTED
            assert scheduler.state is SchedulerState.FAULTED
            assert all(d.status is not ProofDecisionStatus.REJECTED for d in result.decisions)
        finally:
            scheduler.close()


def test_hardware_change_and_worker_restart_require_requalification(pki):
    backend = CPUProofBackend()
    with endpoint(pki, backend) as client:
        backend.runtime["gpu_name"] = "REPLACED GPU"
        backend.runtime["profile_hash"] = runtime_profile_hash(backend.runtime)
        with pytest.raises(ProofWorkerUnavailable, match="changed"):
            client.assert_ready()
        assert client.revision("cuda:0") is None


def test_wire_refuses_pickle_nonfinite_duplicate_keys_and_digest_mismatch():
    from reliquary.validator.remote_proof_protocol import ProofHealth
    for body in (b"\x80\x04pickle", b'{"protocol":"a","protocol":"b"}', b'{"x":NaN}'):
        with pytest.raises((ValueError, UnicodeError)):
            ProofHealth.read(body)
    with pytest.raises(ValueError):
        ProofInput(tokens=[1], commitments=[{"sketch": 0}], rollout={}, randomness="ab",
                   seed_u_values=[float("nan")])


def test_real_cpu_kernel_round_trip_preserves_every_sparse_field(pki):
    """The production kernel can run on a tiny CPU model without a GPU fake."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from reliquary.validator.verifier import verify_commitment_proofs

    config = AutoConfig.for_model("qwen3", vocab_size=256, hidden_size=64,
        intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=256, eos_token_id=2)
    config._attn_implementation = "eager"
    torch.manual_seed(1234)
    model = AutoModelForCausalLM.from_config(config).to(torch.float32).eval()
    data = payload()
    expected = verify_commitment_proofs(data.commit(), model, data.randomness,
                                        seed_u_values=data.seed_u_values)

    class KernelBackend(CPUProofBackend):
        def prove(self, request):
            return verify_commitment_proofs(request.payload.commit(), model,
                request.payload.randomness, seed_u_values=request.payload.seed_u_values)

    with endpoint(pki, KernelBackend()) as client:
        result = prove(client)
        assert ProofValues.from_kernel(result) == ProofValues.from_kernel(expected)


def test_shadow_results_and_failures_cannot_change_local_decision(pki):
    from reliquary.validator.remote_proof import ShadowProofPool
    from reliquary.validator.proof_worker import ProofModelProxy

    class Local:
        devices = ("cuda:0",)
        def call(self, *_args):
            return kernel_result()
        def close(self, **_kwargs):
            pass

    backend = CPUProofBackend()
    with endpoint(pki, backend) as client:
        pool = ShadowProofPool(Local(), client)
        verify = pool.verifier_for_window(42, "openmathinstruct", REV)
        backend.delay = .3
        data = payload()
        try:
            started = time.monotonic()
            first = verify(data.commit(), ProofModelProxy("cuda:0"), data.randomness,
                           seed_u_values=data.seed_u_values)
            assert time.monotonic() - started < .2
            assert first.all_passed
            # A second local job never queues behind the shadow GPU.
            assert verify(data.commit(), ProofModelProxy("cuda:0"), data.randomness,
                          seed_u_values=data.seed_u_values).all_passed
            assert pool.dropped == 1
            assert backend.finished.wait(2)
            deadline = time.monotonic() + 2
            while pool.compared < 1 and time.monotonic() < deadline:
                time.sleep(.01)
            assert (pool.compared, pool.diverged) == (1, 0)
            # Even local schema drift / scheduling shutdown is non-authoritative.
            pool._executor.shutdown()
            assert verify(data.commit(), ProofModelProxy("cuda:0"), data.randomness,
                          seed_u_values=data.seed_u_values).all_passed
            assert pool.unavailable == 1
        finally:
            pool.close()


def test_wrong_ca_expired_certificate_and_worker_restart_fail_closed(pki, tmp_path):
    # Issue an already expired leaf under the same CA, proving validity time
    # is enforced independently of CA membership and HTTP worker_id.
    cert = tmp_path / "expired.crt"
    csr = tmp_path / "expired.csr"
    ext = tmp_path / "expired.ext"
    ext.write_text("basicConstraints=critical,CA:FALSE\nextendedKeyUsage=clientAuth\n")
    subprocess.run(["openssl", "req", "-new", "-key", str(pki / "signer-client/client.key"),
        "-subj", "/CN=expired-controller", "-out", str(csr)], check=True, capture_output=True)
    (tmp_path / "index").write_text("")
    (tmp_path / "serial").write_text("99\n")
    configuration = tmp_path / "ca.conf"
    configuration.write_text(f"""[ca]
default_ca=test
[test]
database={tmp_path / 'index'}
new_certs_dir={tmp_path}
serial={tmp_path / 'serial'}
private_key={pki / 'ca/ca.key'}
certificate={pki / 'ca/ca.crt'}
default_md=sha256
policy=policy
[policy]
commonName=supplied
""")
    issued = subprocess.run(["openssl", "ca", "-batch", "-notext", "-config", str(configuration),
        "-in", str(csr), "-startdate", "20000101000000Z", "-enddate", "20000102000000Z",
        "-extfile", str(ext), "-out", str(cert)], capture_output=True, text=True)
    assert issued.returncode == 0, issued.stderr
    with endpoint(pki, CPUProofBackend()) as client:
        tls = ssl.create_default_context(cafile=str(pki / "ca/ca.crt"))
        tls.load_cert_chain(str(cert), str(pki / "signer-client/client.key"))
        with httpx.Client(verify=tls, trust_env=False) as expired_client:
            with pytest.raises(httpx.HTTPError):
                expired_client.get(str(client._client.base_url) + "/v1/health", timeout=1)
        with httpx.Client(verify=ssl.create_default_context(), trust_env=False) as wrong_ca:
            with pytest.raises(httpx.HTTPError):
                wrong_ca.get(str(client._client.base_url) + "/v1/health", timeout=1)
        client.health = client.health.model_copy(update={"session_id": "dead-process"})
        with pytest.raises(ProofWorkerUnavailable, match="restarted"):
            client.assert_ready()
        assert client.readiness_snapshot()["ready"] is False


def test_server_rejects_unknown_protocol_extra_fields_and_bad_digest(pki):
    with endpoint(pki, CPUProofBackend()) as client:
        value = request_for(client).model_dump()
        for mutation in ({"protocol": "v0"}, {"callback": "os.system"},
                         {"content_sha256": "0" * 64}, {"window": True}):
            response = client._client.post("/v1/prove", json={**value, **mutation})
            assert response.status_code == 422
        response = client._client.post("/v1/prove", content=b"x" * (16 * 1024 * 1024 + 1),
                                        headers={"content-type": "application/json"})
        assert response.status_code == 413


@pytest.mark.parametrize("fill_closed", [False, True])
def test_capacity_requires_network_measurements_and_actual_gpu_identity(pki, tmp_path, monkeypatch, fill_closed):
    import hashlib
    from reliquary import constants as c
    from reliquary.validator.remote_proof_protocol import RemoteProofMeasurement, transport_hash
    from reliquary.validator.proof_capacity import ProofCapacityQualificationError

    monkeypatch.setattr(c, "PROTOCOL_PROFILE_ID", IDENTITY["profile_id"])
    monkeypatch.setattr(c, "PROTOCOL_MODEL_REVISION", "1" * 40)
    monkeypatch.setattr(c, "MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV", {"openmathinstruct": 10})
    monkeypatch.setattr(c, "MAX_RANKED_PROOF_ATTEMPTS_PER_WINDOW", 1)
    monkeypatch.setattr(c, "FORENSIC_SAMPLE_PER_WINDOW", 0)
    monkeypatch.setattr(c, "MAX_PROOF_WALL_SECONDS", 240.)
    monkeypatch.setattr(c, "FILL_CLOSED_ENABLED", fill_closed)
    monkeypatch.setattr(c, "FILL_CLOSED_ADMISSION_BUDGET_PER_ENV", 512)
    monkeypatch.setattr(c, "FILL_CLOSED_MAX_SECONDS", 1800.)
    wall_seconds, proof_count = (1800., 512) if fill_closed else (240., 1)
    manifest_path = tmp_path / "capacity.json"
    with endpoint(pki, CPUProofBackend()) as client:
        manifest = dict(schema_version=3, profile_id=IDENTITY["profile_id"],
            model_revision="1" * 40, software_revision="c" * 40, checkpoint_revision=REV,
            samples_sha256="2" * 64, runtime_fingerprint_hash=client.runtime_fingerprint["profile_hash"],
            hardware_class=client.health.slots[0].hardware_class, benchmark_device_count=1,
            benchmark_device_uuids=["gpu-test-0"], proof_wall_seconds=wall_seconds, headroom_fraction=.2,
            proofs_per_environment={"openmathinstruct": proof_count}, p95_seconds_per_proof={"openmathinstruct": .01},
            p95_seconds_per_proof_by_environment_and_device={"openmathinstruct": {"gpu-test-0": .01}},
            sample_count_by_environment={"openmathinstruct": 20},
            sample_count_by_environment_and_device={"openmathinstruct": {"gpu-test-0": 20}},
            minimum_samples_per_device_per_environment=20,
            minimum_completion_tokens_by_environment={"openmathinstruct": 9},
            measured_at="2026-09-09T00:00:00Z", qualified=True, proof_path_hash=compute_proof_path_hash())
        def write():
            raw = json.dumps(manifest).encode()
            manifest_path.write_bytes(raw)
            monkeypatch.setenv("RELIQUARY_PROOF_CAPACITY_MANIFEST", str(manifest_path))
            monkeypatch.setenv("RELIQUARY_PROOF_CAPACITY_MANIFEST_SHA256", hashlib.sha256(raw).hexdigest())
        write()
        with pytest.raises(ValueError):
            client.qualify(REV)  # Local GPU measurements cannot qualify remote execution.
        manifest["remote_proof"] = RemoteProofMeasurement(worker_id="proof-test", transport_sha256=transport_hash()).model_dump()
        write()
        assert client.qualify(REV)["qualified"] is True
        if fill_closed:
            # Legacy wall and attempt limits cannot qualify a fill window,
            # even with authentic remote measurements and the correct GPU.
            manifest["proof_wall_seconds"] = 240.
            write()
            with pytest.raises(ProofCapacityQualificationError, match="wall"):
                client.qualify(REV)
            manifest["proof_wall_seconds"] = wall_seconds
            manifest["proofs_per_environment"]["openmathinstruct"] = 1
            write()
            with pytest.raises(ProofCapacityQualificationError, match="reserve enough proofs"):
                client.qualify(REV)
            manifest["proofs_per_environment"]["openmathinstruct"] = proof_count
        manifest["benchmark_device_uuids"] = ["gpu-someone-else"]
        write()
        with pytest.raises(ProofCapacityQualificationError, match="UUID"):
            client.qualify(REV)
        manifest["remote_proof"]["worker_id"] = "wrong-worker"
        write()
        with pytest.raises(ProofWorkerUnavailable, match="measured"):
            client.qualify(REV)


def test_another_role_ca_and_wrong_server_name_are_refused(pki, tmp_path):
    other = tmp_path / "unrelated-role-pki"
    script = Path(__file__).resolve().parents[2] / "scripts/generate_signer_pki.sh"
    subprocess.run([str(script), str(other), "127.0.0.1"], check=True, capture_output=True)
    with endpoint(pki, CPUProofBackend()) as client:
        wrong_role = ssl.create_default_context(cafile=str(pki / "ca/ca.crt"))
        wrong_role.load_cert_chain(str(other / "signer-client/client.crt"), str(other / "signer-client/client.key"))
        with httpx.Client(verify=wrong_role, trust_env=False) as http:
            with pytest.raises(httpx.HTTPError):
                http.get(str(client._client.base_url) + "/v1/health", timeout=1)
        correct_role = ssl.create_default_context(cafile=str(pki / "ca/ca.crt"))
        correct_role.load_cert_chain(str(pki / "signer-client/client.crt"), str(pki / "signer-client/client.key"))
        with socket.create_connection(("127.0.0.1", client._client.base_url.port), timeout=1) as sock:
            with pytest.raises(ssl.SSLCertVerificationError):
                correct_role.wrap_socket(sock, server_hostname="different-proof-host.invalid")


def test_health_probe_cannot_invalidate_a_new_adoption(pki):
    backend = CPUProofBackend()
    with endpoint(pki, backend) as client:
        entered = threading.Event()
        release = threading.Event()
        held_once = False
        def health_descriptions():
            nonlocal held_once
            descriptions = [backend.describe("cuda:0")]
            if not held_once:
                held_once = True
                entered.set()
                assert release.wait(2)
            return descriptions
        backend.health_descriptions = health_descriptions
        errors = []
        def run(fn):
            try:
                fn()
            except Exception as exc:
                errors.append(exc)
        health = threading.Thread(target=run, args=(client.assert_ready,))
        health.start()
        assert entered.wait(2)
        next_revision = "d" * 40
        client.bind_checkpoint(8, IDENTITY["repo_id"], next_revision)
        adoption = threading.Thread(target=run, args=(lambda: client.reload(
            "cuda:0", None, next_revision, IDENTITY["repo_id"]),))
        adoption.start()
        try:
            time.sleep(.03)
            assert backend.adoptions == 1  # Waits until old health reply is consumed.
        finally:
            release.set()
            health.join(3)
            adoption.join(3)
        assert not errors
        assert not health.is_alive() and not adoption.is_alive()
        assert client.revision("cuda:0") == next_revision
        assert client.assert_ready().checkpoint.checkpoint_n == 8


def test_health_serializes_native_hf_configuration(pki):
    from transformers import GenerationConfig, Qwen3Config

    class NativeMetadataBackend(CPUProofBackend):
        def describe(self, device):
            value = super().describe(device)
            value["config"] = Qwen3Config().to_dict()
            value["generation_config"] = GenerationConfig(
                exponential_decay_length_penalty=(8, 1.01),
            ).to_dict()
            return value

    backend = NativeMetadataBackend()
    assert set(backend.describe("cuda:0")["config"]["id2label"]) == {0, 1}
    with endpoint(pki, backend) as client:
        assert client.health.config["id2label"] == {"0": "LABEL_0", "1": "LABEL_1"}
        assert client.health.config["hidden_size"] == Qwen3Config().hidden_size
        assert client.health.generation_config["exponential_decay_length_penalty"] == [8, 1.01]
    # The process-local description retains the native HF representation.
    assert set(backend.describe("cuda:0")["config"]["id2label"]) == {0, 1}


def test_worker_utility_workload_setting_is_bound_to_controller(pki):
    from reliquary.validator.utility_telemetry import utility_telemetry_enabled
    with endpoint(pki, CPUProofBackend()) as client:
        assert client.health.utility_telemetry_enabled == utility_telemetry_enabled()
        changed=client.health.model_copy(update={'utility_telemetry_enabled':not utility_telemetry_enabled()})
        with pytest.raises(ProofWorkerUnavailable,match='utility telemetry setting differs'):
            client._validate_health(changed)
