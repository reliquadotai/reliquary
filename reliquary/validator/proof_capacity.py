"""Fail-closed qualification for the auction-v3 proof fleet."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


_ENVIRONMENTS = ("openmathinstruct", "opencodeinstruct")


class ProofCapacityQualificationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ProofCapacityQualification:
    schema_version: int
    profile_id: str
    model_revision: str
    software_revision: str
    hardware_class: str
    benchmark_device_count: int
    proof_wall_seconds: float
    headroom_fraction: float
    proofs_per_environment: Mapping[str, int]
    p95_seconds_per_proof: Mapping[str, float]
    measured_at: str
    qualified: bool

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "ProofCapacityQualification":
        try:
            return cls(
                schema_version=int(value["schema_version"]),
                profile_id=str(value["profile_id"]),
                model_revision=str(value["model_revision"]),
                software_revision=str(value["software_revision"]),
                hardware_class=str(value["hardware_class"]),
                benchmark_device_count=int(
                    value["benchmark_device_count"]
                ),
                proof_wall_seconds=float(value["proof_wall_seconds"]),
                headroom_fraction=float(value["headroom_fraction"]),
                proofs_per_environment={
                    str(name): int(count)
                    for name, count in dict(
                        value["proofs_per_environment"]
                    ).items()
                },
                p95_seconds_per_proof={
                    str(name): float(seconds)
                    for name, seconds in dict(
                        value["p95_seconds_per_proof"]
                    ).items()
                },
                measured_at=str(value["measured_at"]),
                qualified=bool(value["qualified"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProofCapacityQualificationError(
                "invalid proof-capacity manifest"
            ) from exc

    def validate(
        self,
        *,
        profile_id: str,
        model_revision: str,
        software_revision: str | None,
        configured_devices: Sequence[str],
        configured_hardware: Sequence[str],
        proof_wall_seconds: float,
        minimum_proofs_per_environment: int,
    ) -> dict[str, Any]:
        if self.schema_version != 1:
            raise ProofCapacityQualificationError(
                "unsupported proof-capacity manifest schema"
            )
        if not self.qualified:
            raise ProofCapacityQualificationError(
                "proof-capacity benchmark did not qualify"
            )
        if self.profile_id != profile_id:
            raise ProofCapacityQualificationError(
                "proof-capacity profile mismatch"
            )
        if self.model_revision != model_revision:
            raise ProofCapacityQualificationError(
                "proof-capacity model revision mismatch"
            )
        if software_revision and not (
            self.software_revision.startswith(software_revision)
            or software_revision.startswith(self.software_revision)
        ):
            raise ProofCapacityQualificationError(
                "proof-capacity software revision mismatch"
            )
        if self.benchmark_device_count <= 0:
            raise ProofCapacityQualificationError(
                "benchmark_device_count must be positive"
            )
        if (
            not math.isfinite(self.proof_wall_seconds)
            or self.proof_wall_seconds <= 0
            or not math.isclose(
                self.proof_wall_seconds,
                proof_wall_seconds,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ProofCapacityQualificationError(
                "proof-capacity wall does not match runtime"
            )
        if (
            not math.isfinite(self.headroom_fraction)
            or not 0.0 < self.headroom_fraction < 0.5
        ):
            raise ProofCapacityQualificationError(
                "headroom_fraction must be within (0, 0.5)"
            )
        if not configured_devices or (
            len(configured_devices) != len(set(configured_devices))
        ):
            raise ProofCapacityQualificationError(
                "configured proof devices must be non-empty and unique"
            )
        normalized_hardware = {
            str(name).strip().casefold()
            for name in configured_hardware
            if str(name).strip()
        }
        if normalized_hardware != {self.hardware_class.casefold()}:
            raise ProofCapacityQualificationError(
                "configured proof hardware differs from benchmark"
            )

        required_device_seconds = 0.0
        for environment in _ENVIRONMENTS:
            proof_count = self.proofs_per_environment.get(environment)
            p95_seconds = self.p95_seconds_per_proof.get(environment)
            if (
                proof_count is None
                or proof_count < minimum_proofs_per_environment
            ):
                raise ProofCapacityQualificationError(
                    f"{environment} benchmark does not reserve enough proofs"
                )
            if (
                p95_seconds is None
                or not math.isfinite(p95_seconds)
                or p95_seconds <= 0
                or p95_seconds >= proof_wall_seconds
            ):
                raise ProofCapacityQualificationError(
                    f"{environment} p95 proof latency is invalid"
                )
            required_device_seconds += proof_count * p95_seconds

        usable_seconds_per_device = proof_wall_seconds * (
            1.0 - self.headroom_fraction
        )
        minimum_devices = math.ceil(
            required_device_seconds / usable_seconds_per_device
        )
        if len(configured_devices) < minimum_devices:
            raise ProofCapacityQualificationError(
                "configured proof fleet is below measured capacity: "
                f"requires {minimum_devices}, has {len(configured_devices)}"
            )

        return {
            "qualified": True,
            "profile_id": self.profile_id,
            "model_revision": self.model_revision,
            "software_revision": self.software_revision,
            "hardware_class": self.hardware_class,
            "configured_device_count": len(configured_devices),
            "minimum_device_count": minimum_devices,
            "required_device_seconds": required_device_seconds,
            "available_device_seconds": (
                len(configured_devices) * usable_seconds_per_device
            ),
            "headroom_fraction": self.headroom_fraction,
            "measured_at": self.measured_at,
        }


def load_proof_capacity_qualification(
    path: str | Path,
    *,
    expected_sha256: str,
) -> ProofCapacityQualification:
    manifest_path = Path(path)
    payload = manifest_path.read_bytes()
    expected = expected_sha256.strip().lower()
    if (
        len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ProofCapacityQualificationError(
            "proof-capacity manifest SHA-256 must be 64 lowercase hex chars"
        )
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ProofCapacityQualificationError(
            "proof-capacity manifest SHA-256 mismatch"
        )
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ProofCapacityQualificationError(
            "proof-capacity manifest is not valid JSON"
        ) from exc
    if not isinstance(decoded, dict):
        raise ProofCapacityQualificationError(
            "proof-capacity manifest must be a JSON object"
        )
    return ProofCapacityQualification.from_mapping(decoded)


__all__ = [
    "ProofCapacityQualification",
    "ProofCapacityQualificationError",
    "load_proof_capacity_qualification",
]
