from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from reliquary.validator.proof_capacity import (
    ProofCapacityQualification,
    ProofCapacityQualificationError,
    load_proof_capacity_qualification,
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


def _manifest(**overrides):
    value = {
        "schema_version": 1,
        "profile_id": "qwen35-4b-auction-v3",
        "model_revision": "a" * 40,
        "software_revision": "b" * 40,
        "hardware_class": "NVIDIA H100 80GB HBM3",
        "benchmark_device_count": 1,
        "proof_wall_seconds": 240.0,
        "headroom_fraction": 0.2,
        "proofs_per_environment": {
            "openmathinstruct": 10,
            "opencodeinstruct": 10,
        },
        "p95_seconds_per_proof": {
            "openmathinstruct": 60.0,
            "opencodeinstruct": 30.0,
        },
        "measured_at": "2026-07-31T00:00:00Z",
        "qualified": True,
    }
    value.update(overrides)
    return value


def _validate(
    manifest: dict,
    *,
    devices: int = 5,
):
    qualification = ProofCapacityQualification.from_mapping(manifest)
    return qualification.validate(
        profile_id="qwen35-4b-auction-v3",
        model_revision="a" * 40,
        software_revision="b" * 12,
        configured_devices=tuple(f"cuda:{i}" for i in range(devices)),
        configured_hardware=tuple(
            "NVIDIA H100 80GB HBM3" for _ in range(devices)
        ),
        proof_wall_seconds=240.0,
        minimum_proofs_per_environment=10,
    )


def test_capacity_uses_both_environments_and_headroom():
    # 10*60 + 10*30 = 900 device-seconds. At 20% headroom one H100
    # contributes 192 seconds, so five devices are required.
    report = _validate(_manifest(), devices=5)

    assert report["minimum_device_count"] == 5
    assert report["required_device_seconds"] == 900.0
    assert report["available_device_seconds"] == 960.0


def test_capacity_rejects_a_smaller_fleet():
    with pytest.raises(
        ProofCapacityQualificationError,
        match="requires 5, has 4",
    ):
        _validate(_manifest(), devices=4)


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
                    "openmathinstruct": 8,
                    "opencodeinstruct": 10,
                }
            },
            "does not reserve enough proofs",
        ),
    ],
)
def test_capacity_contract_mismatches_fail_closed(override, message):
    with pytest.raises(ProofCapacityQualificationError, match=message):
        _validate(_manifest(**override))


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
