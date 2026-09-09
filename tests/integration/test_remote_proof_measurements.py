"""Real mTLS measurements with explicit fake GPU and group-gate fixtures."""
from __future__ import annotations

import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from reliquary.validator.batcher import PendingSubmission, ValidSubmission, _ScheduledProofPayload
from reliquary.validator.proof_measurements import ProofMeasurements
from reliquary.validator.proof_scheduler import GlobalProofScheduler, ProofPlan, RankedProof
from reliquary.validator.proof_worker import ProofWorkerUnavailable
from reliquary.validator.service import ValidationService
from tests.integration.test_remote_proof_mtls import (
    CPUProofBackend, IDENTITY, REV, endpoint, payload, pki,
)


def group(client, *, mode="pass", gate_delay=0., index=0):
    from reliquary.constants import M_ROLLOUTS
    request = SimpleNamespace(window_start=42, rollouts=[object() for _ in range(M_ROLLOUTS)])
    pending = PendingSubmission(hotkey="test", prompt_idx=index, request=request,
        rewards=[0.] * M_ROLLOUTS, drand_round=1, merkle_root=b"a" * 32, selection_digest=b"a" * 32)
    batcher = SimpleNamespace(window_start=42, env=SimpleNamespace(name="openmathinstruct"))
    verify = client.verifier_for_window(42, "openmathinstruct", REV)
    def execute(pending, *, model, count_operator_debt):
        value = payload()
        for _ in range(M_ROLLOUTS - (mode == "partial")):
            verify(value.commit(), model, value.randomness, seed_u_values=value.seed_u_values)
        # This deterministic group-gate fixture establishes that the measured
        # interval encloses post-network validator work too. It is not GRAIL
        # correctness/latency evidence; the backend is explicitly fake.
        time.sleep(gate_delay)
        return None if mode == "reject" else ValidSubmission(hotkey="test", prompt_idx=index,
            merkle_root_bytes=b"a" * 32, rollouts=request.rollouts)
    batcher._execute_scheduled_proof = execute
    return RankedProof(job_id=f"test:{index}", rank=index, prompt_key=index,
                       payload=_ScheduledProofPayload(batcher, pending))


@pytest.fixture(autouse=True)
def test_profile(monkeypatch):
    import reliquary.constants as c
    monkeypatch.setattr(c, "PROTOCOL_PROFILE_ID", IDENTITY["profile_id"])


def run_group(client, output, candidate):
    recorder = ProofMeasurements(output, client)
    context = SimpleNamespace(_proof_models=client.proxies(), _proof_measurements=recorder)
    with GlobalProofScheduler(devices=client.devices, environments=("openmathinstruct",),
            checkpoint_revision=REV,
            proof_callable=lambda invocation: ValidationService._execute_scheduled_proof(context, invocation)) as scheduler:
        plan = ProofPlan(plan_id="test-plan", environment="openmathinstruct", checkpoint_revision=REV,
                         candidates=[candidate], required_passes=1, deadline_at=time.monotonic() + 5)
        return scheduler.submit_many([plan])[0].result(timeout=6)


def test_actual_network_group_measurement_includes_all_rollouts_and_post_gates(pki, tmp_path):
    from reliquary.constants import M_ROLLOUTS
    backend = CPUProofBackend()
    backend.delay = .005
    path = tmp_path / "measurements.jsonl"
    with endpoint(pki, backend) as client:
        result = run_group(client, path, group(client, gate_delay=.08))
        assert result.winner_job_ids == ("test:0",)
        assert backend.calls == M_ROLLOUTS
        row = json.loads(path.read_text())
        assert row["seconds"] >= .08 + M_ROLLOUTS * .005
        assert row["proof_passed"] and row["complete_remote_group"]
        assert len(row["wire_receipts"]) == row["rollout_count"] == M_ROLLOUTS
        assert row["completion_token_lengths"] == [2] * M_ROLLOUTS
        assert row["checkpoint_n"] == 7 and row["checkpoint_revision"] == REV
        assert row["runtime_fingerprint_hash"] == client.runtime_fingerprint["profile_hash"]
        assert row["device_uuid"] == "gpu-test-0"
        assert row["remote_proof"]["measurement_scope"] == "validator-end-to-end-mtls"
        assert path.stat().st_mode & 0o777 == 0o600
        assert all(receipt["checkpoint"] == client._adopted.model_dump() for receipt in row["wire_receipts"])
        assert all("tokens" not in receipt and "payload" not in receipt for receipt in row["wire_receipts"])


@pytest.mark.parametrize("mode", ["partial", "reject", "infrastructure"])
def test_partial_rejected_and_infrastructure_groups_never_become_passing_samples(pki, tmp_path, mode):
    backend = CPUProofBackend()
    backend.fail = mode == "infrastructure"
    path = tmp_path / "measurements.jsonl"
    with endpoint(pki, backend) as client:
        run_group(client, path, group(client, mode=mode))
        row = json.loads(path.read_text())
        assert row["proof_passed"] is False
        if mode == "infrastructure":
            assert row["infrastructure_error_type"] == "ProofWorkerUnavailable"
            assert row["wire_receipts"] == []
        elif mode == "partial":
            assert row["complete_remote_group"] is False


def test_private_collector_is_explicit_and_refuses_relabelled_local_mode(tmp_path, monkeypatch):
    monkeypatch.delenv("RELIQUARY_PRIVATE_REMOTE_PROOF_MEASUREMENTS", raising=False)
    assert ProofMeasurements.from_environment(None) is None
    monkeypatch.setenv("RELIQUARY_PRIVATE_REMOTE_PROOF_MEASUREMENTS", str(tmp_path / "out"))
    with pytest.raises(ValueError, match="authoritative remote"):
        ProofMeasurements.from_environment(SimpleNamespace(is_remote=True))


def test_measurement_file_cannot_be_reused_or_replaced_by_symlink(pki, tmp_path):
    with endpoint(pki, CPUProofBackend()) as client:
        path = tmp_path / "out"
        recorder = ProofMeasurements(path, client)
        with pytest.raises(FileExistsError):
            ProofMeasurements(path, client)
        victim = tmp_path / "other"
        victim.write_text("unchanged")
        path.unlink()
        path.symlink_to(victim)
        context = SimpleNamespace(_proof_models=client.proxies(), _proof_measurements=recorder)
        from reliquary.validator.proof_scheduler import ProofInvocation
        invocation = ProofInvocation("cuda:0", "test", "openmathinstruct", REV, group(client), time.monotonic() + 5)
        with pytest.raises(ProofWorkerUnavailable, match="persisted"):
            ValidationService._execute_scheduled_proof(context, invocation)
        assert victim.read_text() == "unchanged"


def test_cached_transport_retry_is_one_authenticated_rollout_receipt(pki, tmp_path):
    from reliquary.constants import M_ROLLOUTS
    backend = CPUProofBackend()
    with endpoint(pki, backend) as client:
        request = client._request
        def repeat_same_request(method, path, body=None, **kwargs):
            if path == "/v1/prove":
                request(method, path, body, **kwargs)  # First response was lost.
            return request(method, path, body, **kwargs)
        client._request = repeat_same_request
        path = tmp_path / "samples"
        run_group(client, path, group(client))
        row = json.loads(path.read_text())
        assert row["proof_passed"]
        assert backend.calls == len(row["wire_receipts"]) == M_ROLLOUTS


def test_corpus_refuses_wrong_checkpoint_and_unsigned_request_before_admission(pki, monkeypatch):
    import scripts.measure_remote_proof_capacity as benchmark
    import reliquary.validator.service as service
    from tests.unit.test_grpo_window_batcher import _request
    monkeypatch.setattr(service, "open_grpo_window", lambda *a, **kw: pytest.fail("invalid corpus reached admission"))
    with endpoint(pki, CPUProofBackend()) as client:
        request = _request()
        request.generation_profile_id = IDENTITY["profile_id"]
        row = {"environment": "openmathinstruct", "randomness": "ab" * 32,
               "request": request.model_dump(mode="json")}
        arguments = dict(pool=client, environments={"openmathinstruct": object()}, tokenizer=None, index=0)
        with pytest.raises(ValueError, match="adopted"):
            benchmark.prepare_candidate(row, **arguments)
        row["request"]["checkpoint_hash"] = REV
        with pytest.raises(ValueError, match="signature"):
            benchmark.prepare_candidate(row, **arguments)


def test_isolated_harness_reuses_service_scheduler_and_refuses_duplicate_corpus(pki, tmp_path, monkeypatch):
    import scripts.measure_remote_proof_capacity as benchmark
    with endpoint(pki, CPUProofBackend()) as client:
        client.qualify = lambda *args: pytest.fail("benchmark must not depend on prior capacity")
        monkeypatch.setattr(benchmark, "prepare_candidate", lambda row, **kwargs:
            ("openmathinstruct", group(client, index=kwargs["index"])))
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text("".join(json.dumps({"test_fixture": n}) + "\n" for n in range(20)))
        output = tmp_path / "measurements"
        report = benchmark.measure(corpus, output=output, pool=client, tokenizer=None,
            environments={"openmathinstruct": object()}, timeout=30)
        assert report["groups"] == 20 and report["qualified"] is False
        assert len(output.read_text().splitlines()) == 20
        import scripts.qualify_proof_capacity as qualifier
        # Only the fixture's token cap/environment is synthetic. The existing
        # 20-sample/device gate parses the producer's actual JSONL unchanged.
        monkeypatch.setattr(qualifier, "ENVIRONMENTS", ("openmathinstruct",))
        monkeypatch.setattr(qualifier, "PROTOCOL_PROFILE_ID", IDENTITY["profile_id"])
        monkeypatch.setattr(qualifier, "MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV", {"openmathinstruct": 2})
        first = json.loads(output.read_text().splitlines()[0])
        samples, devices, _ = qualifier._load_samples(output,
            software_revision=client.health.software_revision, checkpoint_revision=REV,
            runtime_fingerprint_hash=client.runtime_fingerprint["profile_hash"],
            hardware_class=client.health.slots[0].hardware_class, benchmark_device_count=1,
            remote_proof=first["remote_proof"])
        assert len(samples["openmathinstruct"][devices[0]]) == 20
        corpus.write_text('{"test_fixture":1}\n{ "test_fixture": 1 }\n')
        with pytest.raises(ValueError, match="duplicate"):
            benchmark.measure(corpus, output=tmp_path / "duplicate-output", pool=client,
                tokenizer=None, environments={"openmathinstruct": object()}, timeout=10)
        assert not (tmp_path / "duplicate-output").exists()
