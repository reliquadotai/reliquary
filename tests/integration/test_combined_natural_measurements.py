"""Real mTLS fixture: preserve all attempts without relabelling CPU test proofs."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.measure_remote_proof_capacity as benchmark
from reliquary.protocol.submission import BatchSubmissionResponse, RejectReason
from tests.integration.test_remote_proof_measurements import group
from tests.integration.test_remote_proof_mtls import (
    IDENTITY,
    CPUProofBackend,
    endpoint,
)
from tests.integration.test_remote_proof_mtls import (
    pki as pki,  # noqa: PLC0414 -- pytest fixture re-export
)


@pytest.fixture(autouse=True)
def profile(monkeypatch):
    import reliquary.constants as c
    monkeypatch.setattr(c, "PROTOCOL_PROFILE_ID", IDENTITY["profile_id"])


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def run_fixture(client, tmp_path, monkeypatch, modes, *, combined=True):
    def prepare(row, **kwargs):
        # Explicit fixture latency establishes the ledger times admission;
        # it makes no claim about real GPU or real grader performance.
        time.sleep(.01)
        if row["mode"] == "zone":
            raise benchmark.AdmissionRejected(SimpleNamespace(reason=RejectReason.OUT_OF_ZONE))
        if row["mode"] == "bad_signature":
            raise benchmark.AdmissionRejected(SimpleNamespace(reason=RejectReason.BAD_SIGNATURE))
        candidate = group(client, mode=row["mode"], index=kwargs["index"])
        if row["mode"] == "reject":
            candidate.payload.pending.reject_response = BatchSubmissionResponse(
                accepted=False, reason=RejectReason.GRAIL_FAIL)
        return "openmathinstruct", candidate

    monkeypatch.setattr(benchmark, "prepare_candidate", prepare)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"environment": "openmathinstruct", "mode": mode,
                                      "fixture_index": i}) + "\n" for i, mode in enumerate(modes)))
    return benchmark.measure(corpus, output=tmp_path / "samples", pool=client, tokenizer=None,
        environments={"openmathinstruct": object()}, timeout=10, combined_natural=combined)


def test_combined_keeps_all_normal_rejects_and_binds_only_complete_real_passes(pki, tmp_path, monkeypatch):
    with endpoint(pki, CPUProofBackend()) as client:
        report = run_fixture(client, tmp_path, monkeypatch, ["pass", "zone", "reject", "pass"])
    assert report["qualified"] is False
    assert report["full_http_or_grader_capacity_qualified"] is False
    assert report["admitted_groups"] == 3 and report["passing_groups"] == 2
    assert report["admission_rejections_by_reason"] == {"out_of_zone": 1}
    assert report["proof_rejections_by_reason"] == {"grail_fail": 1}
    assert report["numerical_rejections_requires_review"] == {"grail_fail": 1}
    attempts, raw, samples = rows(report["attempt_ledger"]), rows(report["proof_attempts"]), rows(report["samples"])
    assert attempts[0]["event"] == "run_started" and attempts[-1]["event"] == "run_complete"
    admissions = [row for row in attempts if row["event"] == "admission"]
    decisions = [row for row in attempts if row["event"] == "proof"]
    assert len(admissions) == 4 and all(row["admission_seconds"] >= .01 for row in admissions)
    assert len(decisions) == len(raw) == 3
    assert {row["job_id"] for row in samples} == {"test:0", "test:3"}
    assert next(row for row in raw if row["job_id"] == "test:2")["proof_passed"] is False
    selection = {"input_sha256", "attempt_ledger_sha256", "proof_attempts_sha256", "corpus_sha256"}
    for row in samples:
        assert row["proof_passed"] is True and row["complete_remote_group"] is True
        original = next(value for value in raw if value["job_id"] == row["job_id"])
        assert {key: value for key, value in row.items() if key not in selection} == original
        admission = next(value for value in admissions if value.get("job_id") == row["job_id"])
        assert row["input_sha256"] == admission["input_sha256"]
        for field, file in (("attempt_ledger_sha256", report["attempt_ledger"]),
                            ("proof_attempts_sha256", report["proof_attempts"]),
                            ("corpus_sha256", report["corpus"])):
            assert row[field] == report[field] == hashlib.sha256(Path(file).read_bytes()).hexdigest()
    assert all(Path(report[key]).stat().st_mode & 0o777 == 0o600
               for key in ("attempt_ledger", "proof_attempts", "samples"))


@pytest.mark.parametrize("failure", ["partial", "infrastructure", "bad_signature"])
def test_no_final_samples_on_partial_proof_infrastructure_or_non_zone_admission_error(pki, tmp_path, monkeypatch, failure):
    backend = CPUProofBackend()
    backend.fail = failure == "infrastructure"
    with endpoint(pki, backend) as client, pytest.raises((ValueError, RuntimeError)):
        run_fixture(client, tmp_path, monkeypatch, ["pass", failure])
    assert not (tmp_path / "samples").exists()
    attempts = rows(tmp_path / "samples.attempts.jsonl")
    assert attempts[-1]["event"] == "run_failed"
    if failure != "bad_signature":
        assert (tmp_path / "samples.proof-attempts.jsonl").stat().st_size > 0
    else:
        assert attempts[-2]["reason"] == "bad_signature"


def test_default_mode_still_refuses_any_proof_reject(pki, tmp_path, monkeypatch):
    with endpoint(pki, CPUProofBackend()) as client, pytest.raises(RuntimeError, match="rejected/failed"):
        run_fixture(client, tmp_path, monkeypatch, ["pass", "reject"], combined=False)
    assert len(rows(tmp_path / "samples")) == 2  # Existing raw evidence contract.
    assert not (tmp_path / "samples.attempts.jsonl").exists()


def test_all_out_of_zone_supply_cannot_become_passing_evidence(pki, tmp_path, monkeypatch):
    with endpoint(pki, CPUProofBackend()) as client, pytest.raises(ValueError, match="no admitted"):
        run_fixture(client, tmp_path, monkeypatch, ["zone", "zone"])
    assert not (tmp_path / "samples").exists()
    admissions = [row for row in rows(tmp_path / "samples.attempts.jsonl") if row["event"] == "admission"]
    assert len(admissions) == 2 and all(row["reason"] == "out_of_zone" for row in admissions)
