from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from reliquary.validator.proof_capacity import (
    ProofCapacityQualification,
    ProofCapacityQualificationError,
    load_proof_capacity_qualification,
    resolve_cuda_proof_devices,
)


def test_capacity_qualifier_help_runs_from_source_checkout(tmp_path):
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "qualify_proof_capacity.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--benchmark-device-count" in completed.stdout


@pytest.mark.parametrize(
    (
        "profile_id",
        "model_revision",
        "rollout_count",
        "representative_length",
        "expected_proofs",
    ),
    (
        (
            "qwen35-4b-auction-v3",
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            8,
            14_746,
            18,
        ),
        (
            "qwen3-4b-base-dapo-v4",
            "906bfd4b4dc7f14ee4320094d8b41684abff8539",
            16,
            7_373,
            34,
        ),
        (
            "qwen3-4b-base-dapo-reasoning-v5",
            "906bfd4b4dc7f14ee4320094d8b41684abff8539",
            16,
            7_373,
            34,
        ),
    ),
)
def test_capacity_qualifier_binds_full_representative_evidence(
    tmp_path,
    profile_id,
    model_revision,
    rollout_count,
    representative_length,
    expected_proofs,
):
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "qualify_proof_capacity.py"
    )
    samples_path = tmp_path / "samples.jsonl"
    output_path = tmp_path / "proof-capacity.json"
    software_revision = "b" * 40
    checkpoint_revision = "c" * 40
    rows = []
    for environment in ("openmathinstruct", "opencodeinstruct"):
        for _index in range(20):
            rows.append({
                "environment": environment,
                "seconds": 1.0,
                "proof_passed": True,
                "profile_id": profile_id,
                "model_revision": model_revision,
                "software_revision": software_revision,
                "checkpoint_revision": checkpoint_revision,
                "runtime_fingerprint_hash": "e" * 64,
                "hardware_class": "NVIDIA H100 80GB HBM3",
                "device_uuid": "GPU-EXACT",
                "rollout_count": rollout_count,
                "completion_token_lengths": (
                    [representative_length] * rollout_count
                ),
            })
    payload = (
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    ).encode()
    samples_path.write_bytes(payload)
    env = {
        **dict(os.environ),
        "RELIQUARY_PROTOCOL_PROFILE": profile_id,
    }

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(samples_path),
            "--output",
            str(output_path),
            "--software-revision",
            software_revision,
            "--checkpoint-revision",
            checkpoint_revision,
            "--runtime-fingerprint-hash",
            "e" * 64,
            "--hardware-class",
            "NVIDIA H100 80GB HBM3",
            "--benchmark-device-count",
            "1",
            "--measured-at",
            "2026-07-31T00:00:00Z",
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads(output_path.read_text())
    assert manifest["schema_version"] == 3
    assert manifest["benchmark_device_uuids"] == ["gpu-exact"]
    assert manifest["samples_sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["proofs_per_environment"] == {
        "openmathinstruct": expected_proofs,
        "opencodeinstruct": expected_proofs,
    }


def _manifest(**overrides):
    device_count = int(overrides.pop("benchmark_device_count", 9))
    value = {
        "schema_version": 3,
        "profile_id": "qwen35-4b-auction-v3",
        "model_revision": "a" * 40,
        "software_revision": "b" * 40,
        "checkpoint_revision": "c" * 40,
        "samples_sha256": "d" * 64,
        "runtime_fingerprint_hash": "e" * 64,
        "hardware_class": "NVIDIA H100 80GB HBM3",
        "benchmark_device_count": device_count,
        "benchmark_device_uuids": [
            f"gpu-{index}" for index in range(device_count)
        ],
        "proof_wall_seconds": 240.0,
        "headroom_fraction": 0.2,
        "proofs_per_environment": {
            "openmathinstruct": 18,
            "opencodeinstruct": 18,
        },
        "p95_seconds_per_proof": {
            "openmathinstruct": 60.0,
            "opencodeinstruct": 30.0,
        },
        "p95_seconds_per_proof_by_environment_and_device": {
            "openmathinstruct": {
                f"gpu-{index}": 60.0 for index in range(device_count)
            },
            "opencodeinstruct": {
                f"gpu-{index}": 30.0 for index in range(device_count)
            },
        },
        "sample_count_by_environment": {
            "openmathinstruct": 20 * device_count,
            "opencodeinstruct": 20 * device_count,
        },
        "sample_count_by_environment_and_device": {
            "openmathinstruct": {
                f"gpu-{index}": 20 for index in range(device_count)
            },
            "opencodeinstruct": {
                f"gpu-{index}": 20 for index in range(device_count)
            },
        },
        "minimum_samples_per_device_per_environment": 20,
        "minimum_completion_tokens_by_environment": {
            "openmathinstruct": 14_746,
            "opencodeinstruct": 29_492,
        },
        "measured_at": "2026-07-31T00:00:00Z",
        "qualified": True,
    }
    value.update(overrides)
    return value


def _validate(
    manifest: dict,
    *,
    devices: int = 9,
    software_revision: str | None = "b" * 40,
    proof_path_hash: str | None = None,
):
    qualification = ProofCapacityQualification.from_mapping(manifest)
    return qualification.validate(
        profile_id="qwen35-4b-auction-v3",
        model_revision="a" * 40,
        software_revision=software_revision,
        proof_path_hash=proof_path_hash,
        checkpoint_revision="c" * 40,
        runtime_fingerprint_hash="e" * 64,
        configured_devices=tuple(f"cuda:{i}" for i in range(devices)),
        configured_hardware=tuple(
            "NVIDIA H100 80GB HBM3" for _ in range(devices)
        ),
        configured_device_uuids=tuple(
            f"gpu-{index}" for index in range(devices)
        ),
        proof_wall_seconds=240.0,
        minimum_proofs_per_environment=18,
        minimum_completion_tokens_per_environment={
            "openmathinstruct": 14_746,
            "opencodeinstruct": 29_492,
        },
    )


def test_capacity_uses_both_environments_and_headroom():
    # 18*60 + 18*30 = 1620 device-seconds. At 20% headroom one H100
    # contributes 192 seconds, so nine devices are required.
    report = _validate(_manifest(), devices=9)

    assert report["minimum_device_count"] == 9
    assert report["required_device_seconds"] == 1620.0
    assert report["available_device_seconds"] == 1728.0
    assert report["configured_device_uuids"] == [
        f"gpu-{index}" for index in range(9)
    ]


def test_capacity_rejects_a_smaller_fleet():
    with pytest.raises(
        ProofCapacityQualificationError,
        match="requires 9, has 8",
    ):
        _validate(_manifest(benchmark_device_count=8), devices=8)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"profile_id": "wrong"}, "profile mismatch"),
        ({"model_revision": "c" * 40}, "model revision mismatch"),
        ({"software_revision": "d" * 40}, "software revision mismatch"),
        ({"qualified": False}, "did not qualify"),
        ({"headroom_fraction": 0.0}, "headroom_fraction"),
        (
            {
                "proofs_per_environment": {
                    "openmathinstruct": 17,
                    "opencodeinstruct": 18,
                }
            },
            "does not reserve enough proofs",
        ),
        (
            {
                "minimum_completion_tokens_by_environment": {
                    "openmathinstruct": 14_745,
                    "opencodeinstruct": 29_492,
                }
            },
            "not representative",
        ),
        (
            {
                "sample_count_by_environment": {
                    "openmathinstruct": 19,
                    "opencodeinstruct": 20,
                }
            },
            "too few proof samples",
        ),
        (
            {"minimum_samples_per_device_per_environment": 19},
            "20 samples per GPU",
        ),
        (
            {
                "sample_count_by_environment_and_device": {
                    "openmathinstruct": {
                        **{
                            f"gpu-{index}": 20
                            for index in range(1, 9)
                        },
                        "gpu-0": 19,
                    },
                    "opencodeinstruct": {
                        f"gpu-{index}": 20 for index in range(9)
                    },
                }
            },
            "too few per-GPU samples",
        ),
        (
            {"runtime_fingerprint_hash": "f" * 64},
            "runtime fingerprint mismatch",
        ),
        (
            {
                "p95_seconds_per_proof": {
                    "openmathinstruct": 59.0,
                    "opencodeinstruct": 30.0,
                }
            },
            "worst GPU p95",
        ),
    ],
)
def test_capacity_contract_mismatches_fail_closed(override, message):
    with pytest.raises(ProofCapacityQualificationError, match=message):
        _validate(_manifest(**override))


def test_capacity_rejects_partial_or_missing_runtime_revision():
    for revision in ("b" * 12, None):
        with pytest.raises(
            ProofCapacityQualificationError,
            match="software revision mismatch",
        ):
            _validate(_manifest(), software_revision=revision)


def test_capacity_rejects_non_boolean_qualification():
    with pytest.raises(
        ProofCapacityQualificationError,
        match="invalid proof-capacity manifest",
    ):
        ProofCapacityQualification.from_mapping(
            _manifest(qualified="false")
        )


def test_capacity_rejects_runtime_topology_or_gpu_uuid_mismatch():
    qualification = ProofCapacityQualification.from_mapping(_manifest())
    kwargs = {
        "profile_id": "qwen35-4b-auction-v3",
        "model_revision": "a" * 40,
        "software_revision": "b" * 40,
        "checkpoint_revision": "c" * 40,
        "runtime_fingerprint_hash": "e" * 64,
        "configured_devices": tuple(f"cuda:{i}" for i in range(9)),
        "configured_hardware": tuple(
            "NVIDIA H100 80GB HBM3" for _ in range(9)
        ),
        "configured_device_uuids": tuple(
            ["gpu-other", *[f"gpu-{index}" for index in range(1, 9)]]
        ),
        "proof_wall_seconds": 240.0,
        "minimum_proofs_per_environment": 18,
        "minimum_completion_tokens_per_environment": {
            "openmathinstruct": 14_746,
            "opencodeinstruct": 29_492,
        },
    }
    with pytest.raises(
        ProofCapacityQualificationError,
        match="GPU UUIDs differ",
    ):
        qualification.validate(**kwargs)


class _FakeCuda:
    def __init__(self):
        self._uuids = ("GPU-A", "GPU-B")

    def device_count(self):
        return len(self._uuids)

    def get_device_name(self, index):
        return "NVIDIA H100 80GB HBM3"

    def get_device_properties(self, index):
        return type("Properties", (), {"uuid": self._uuids[index]})()


def test_cuda_device_resolution_canonicalizes_and_rejects_aliases():
    resolved = resolve_cuda_proof_devices(
        ("cuda:00", "cuda:1"),
        cuda=_FakeCuda(),
    )
    assert [identity.device_id for identity in resolved] == [
        "cuda:0",
        "cuda:1",
    ]
    assert [identity.device_uuid for identity in resolved] == [
        "gpu-a",
        "gpu-b",
    ]

    with pytest.raises(
        ProofCapacityQualificationError,
        match="explicit cuda:<index>",
    ):
        resolve_cuda_proof_devices(("cuda",), cuda=_FakeCuda())

    with pytest.raises(
        ProofCapacityQualificationError,
        match="duplicate CUDA indices",
    ):
        resolve_cuda_proof_devices(
            ("cuda:0", "cuda:00"),
            cuda=_FakeCuda(),
        )


def test_manifest_loader_requires_exact_digest(tmp_path):
    payload = json.dumps(
        _manifest(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    path = tmp_path / "proof-capacity.json"
    path.write_bytes(payload)

    loaded = load_proof_capacity_qualification(
        path,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert loaded.profile_id == "qwen35-4b-auction-v3"

    with pytest.raises(
        ProofCapacityQualificationError,
        match="SHA-256 mismatch",
    ):
        load_proof_capacity_qualification(
            path,
            expected_sha256="0" * 64,
        )


def test_runtime_fingerprint_mismatch_fails_closed_by_default():
    with pytest.raises(
        ProofCapacityQualificationError, match="runtime fingerprint"
    ):
        _validate(_manifest(runtime_fingerprint_hash="f" * 64))


def test_faster_runtime_switch_carries_qualification_over(monkeypatch):
    from reliquary import constants as C

    monkeypatch.setattr(C, "PROOF_CAPACITY_ACCEPT_FASTER_RUNTIME", True)
    report = _validate(_manifest(runtime_fingerprint_hash="f" * 64))
    assert report["qualified"] is True
    assert report["runtime_fingerprint_carried_over_from"] == "f" * 64


def test_faster_runtime_switch_keeps_malformed_hash_fail_closed(monkeypatch):
    from reliquary import constants as C

    monkeypatch.setattr(C, "PROOF_CAPACITY_ACCEPT_FASTER_RUNTIME", True)
    with pytest.raises(
        ProofCapacityQualificationError, match="runtime fingerprint"
    ):
        _validate(_manifest(runtime_fingerprint_hash="not-a-sha"))


def test_matching_fingerprint_reports_no_carryover():
    report = _validate(_manifest())
    assert report["runtime_fingerprint_carried_over_from"] is None


def test_faster_runtime_switch_covers_proof_path_change(monkeypatch):
    from reliquary import constants as C

    # Different image AND different proof path: fail-closed by default...
    with pytest.raises(
        ProofCapacityQualificationError, match="software revision"
    ):
        _validate(
            _manifest(software_revision="a" * 40, proof_path_hash="1" * 64),
            software_revision="b" * 40,
            proof_path_hash="2" * 64,
        )
    # ...carried over as a lower bound with the operator switch.
    monkeypatch.setattr(C, "PROOF_CAPACITY_ACCEPT_FASTER_RUNTIME", True)
    report = _validate(
        _manifest(software_revision="a" * 40, proof_path_hash="1" * 64),
        software_revision="b" * 40,
        proof_path_hash="2" * 64,
    )
    assert report["qualified"] is True
    assert report["qualification_carried_over_from"] == "a" * 40
