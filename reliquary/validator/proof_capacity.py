"""Fail-closed qualification for the auction-v3 proof fleet."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CUDA_DEVICE_RE = re.compile(r"^cuda:(\d+)$")


class ProofCapacityQualificationError(RuntimeError):
    pass


def capacity_budget() -> dict[str, Any]:
    """Match the active scheduler's budget and qualified drain horizon.

    FillState admits at most its monotone budget (including failed proofs);
    fill windows bypass the seal-time auction/forensic proof plan entirely.
    Strict service may admit for one such horizon before draining; its later
    settlement deadline is not a larger proof-capacity requirement.
    """
    from reliquary import constants as c
    if c.FILL_CLOSED_ENABLED:
        budget = {"mode": "fill_closed", "proofs_per_environment": c.FILL_CLOSED_ADMISSION_BUDGET_PER_ENV,
            "wall_seconds": c.FILL_CLOSED_MAX_SECONDS,
            "target_groups_per_environment": c.FILL_CLOSED_TARGET_GROUPS_PER_ENV,
            "picks_per_window": c.FILL_CLOSED_PICKS_PER_WINDOW}
        if c.FILL_CLOSED_BOUNDED_PROOFS:
            budget.update(mode="fill_closed_bounded",
                drain_seconds=c.FILL_CLOSED_PROOF_DRAIN_SECONDS,
                dispatch_seconds=c.FILL_CLOSED_PROOF_DISPATCH_SECONDS,
                precommit_seconds=c.FILL_CLOSED_PRECOMMIT_SECONDS,
                upload_grace_seconds=c.SUBMISSION_UPLOAD_GRACE_SECONDS)
        return budget
    return {"mode": "seal_time_auction",
        "proofs_per_environment": c.MAX_RANKED_PROOF_ATTEMPTS_PER_WINDOW + c.FORENSIC_SAMPLE_PER_WINDOW,
        "wall_seconds": c.MAX_PROOF_WALL_SECONDS}


# The files whose bytes determine what one proof costs on the GPU. Pinning
# these (plus the parameter values below) lets a qualification legitimately
# survive an image deploy that does not touch the proof path — the full
# ``software_revision`` pin turned EVERY deploy into a re-benchmark, including
# HTTP-only ones. Anything imported by these files that changes proof cost
# must either live in this list or be captured as a parameter value; when in
# doubt, add the file — a false invalidation costs a 15-minute benchmark, a
# false carry-over costs a mis-sized fleet.
PROOF_PATH_FILES = (
    "reliquary/validator/verifier.py",
    "reliquary/validator/proof_scheduler.py",
    "reliquary/shared/forward.py",
    "reliquary/protocol/grail_verifier.py",
    "reliquary/protocol/crypto.py",
    "reliquary/environment/forced_sampling.py",
)


def _live_proof_parameters() -> Mapping[str, Any]:
    from reliquary import constants

    return {
        "CHALLENGE_K": constants.CHALLENGE_K,
        "LAYER_INDEX": constants.LAYER_INDEX,
        "M_ROLLOUTS": constants.M_ROLLOUTS,
        "T_PROTO": constants.T_PROTO,
        "TOP_K_PROTO": constants.TOP_K_PROTO,
        "TOP_P_PROTO": constants.TOP_P_PROTO,
        "MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV": dict(
            constants.MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV
        ),
    }


def compute_proof_path_hash(
    *,
    repo_root: str | Path | None = None,
    parameters: Mapping[str, Any] | None = None,
) -> str:
    """SHA-256 over the proof-path files' bytes and proof parameter values.

    ``repo_root``/``parameters`` are injectable for tests; production callers
    use the installed tree and the live protocol constants.
    """
    root = (
        Path(repo_root)
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    digest = hashlib.sha256()
    for relative in PROOF_PATH_FILES:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        try:
            digest.update((root / relative).read_bytes())
        except OSError as exc:
            raise ProofCapacityQualificationError(
                f"proof-path file unreadable: {relative}"
            ) from exc
        digest.update(b"\x00")
    values = parameters if parameters is not None else _live_proof_parameters()
    digest.update(
        json.dumps(values, sort_keys=True, default=str).encode("utf-8")
    )
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ProofDeviceIdentity:
    device_id: str
    hardware_class: str
    device_uuid: str


def _is_canonical_cuda_index(device: Any) -> bool:
    """A bare, canonical ``cuda:<index>`` — never a slot.

    Capacity is a claim about physical cards: ``available_device_seconds`` is
    the wall clock times this count. Slot ids must fail here so extra processes
    can never inflate a qualified budget.
    """
    match = _CUDA_DEVICE_RE.fullmatch(str(device))
    if match is None:
        return False
    return str(device) == f"cuda:{int(match.group(1))}"


def physical_proof_device(slot_id: str) -> str:
    """The CUDA device a proof slot runs on: ``cuda:0#2`` -> ``cuda:0``.

    Slot ids exist so the pool can hold one process per slot and the scheduler
    one dispatch thread per slot; every kernel still lands on the physical
    device named before the ``#``.
    """
    return str(slot_id).strip().split("#", 1)[0]


def expand_proof_slots(
    devices: Sequence[str], slots_per_device: int,
) -> tuple[str, ...]:
    """One id per proof process.

    ``slots_per_device == 1`` returns the bare device ids, so a deployment
    that does not opt in sees byte-identical identifiers everywhere they are
    recorded (scheduler telemetry, capacity manifests, archives).
    """
    count = int(slots_per_device)
    if count < 1:
        raise ValueError("a proof device needs at least one proof slot")
    cleaned = tuple(str(device).strip() for device in devices)
    if count == 1:
        return cleaned
    return tuple(
        f"{device}#{slot}" for device in cleaned for slot in range(count)
    )


def resolve_cuda_proof_devices(
    raw_devices: Sequence[str],
    *,
    cuda: Any,
) -> tuple[ProofDeviceIdentity, ...]:
    """Resolve explicit CUDA indices to canonical physical identities.

    Deliberately physical-only, slot syntax refused: what this returns is
    handed to ``ProofCapacityQualification.validate``, which counts cards. Ask
    for more processes with ``RELIQUARY_PROOF_SLOTS_PER_DEVICE``, which widens
    the plane *after* this resolution and leaves capacity alone.
    """

    available = int(cuda.device_count())
    resolved: list[ProofDeviceIdentity] = []
    device_ids: set[str] = set()
    device_uuids: set[str] = set()
    for raw_device in raw_devices:
        value = str(raw_device).strip()
        match = _CUDA_DEVICE_RE.fullmatch(value)
        if match is None:
            raise ProofCapacityQualificationError(
                "proof devices must use explicit cuda:<index> syntax "
                "(for several processes per card set "
                "RELIQUARY_PROOF_SLOTS_PER_DEVICE)"
            )
        index = int(match.group(1))
        if index < 0 or index >= available:
            raise ProofCapacityQualificationError(
                f"proof device {value!r} is outside the visible CUDA fleet"
            )
        device_id = f"cuda:{index}"
        if device_id in device_ids:
            raise ProofCapacityQualificationError(
                "configured proof devices resolve to duplicate CUDA indices"
            )
        hardware_class = str(cuda.get_device_name(index)).strip()
        properties = cuda.get_device_properties(index)
        device_uuid = str(getattr(properties, "uuid", "")).strip().casefold()
        if not hardware_class:
            raise ProofCapacityQualificationError(
                f"proof device {device_id!r} has no hardware identity"
            )
        if not device_uuid:
            raise ProofCapacityQualificationError(
                f"proof device {device_id!r} has no stable GPU UUID"
            )
        if device_uuid in device_uuids:
            raise ProofCapacityQualificationError(
                "configured proof devices resolve to the same physical GPU"
            )
        device_ids.add(device_id)
        device_uuids.add(device_uuid)
        resolved.append(
            ProofDeviceIdentity(
                device_id=device_id,
                hardware_class=hardware_class,
                device_uuid=device_uuid,
            )
        )
    return tuple(resolved)


@dataclass(frozen=True, slots=True)
class ProofCapacityQualification:
    schema_version: int
    profile_id: str
    model_revision: str
    software_revision: str
    checkpoint_revision: str
    samples_sha256: str
    runtime_fingerprint_hash: str
    hardware_class: str
    benchmark_device_count: int
    benchmark_device_uuids: tuple[str, ...]
    proof_wall_seconds: float
    headroom_fraction: float
    proofs_per_environment: Mapping[str, int]
    p95_seconds_per_proof: Mapping[str, float]
    p95_seconds_per_proof_by_environment_and_device: Mapping[
        str,
        Mapping[str, float],
    ]
    sample_count_by_environment: Mapping[str, int]
    sample_count_by_environment_and_device: Mapping[
        str,
        Mapping[str, int],
    ]
    minimum_samples_per_device_per_environment: int
    minimum_completion_tokens_by_environment: Mapping[str, int]
    measured_at: str
    qualified: bool
    # Optional: content hash of the proof path at benchmark time. Absent on
    # legacy manifests, which keep the strict same-image behavior.
    proof_path_hash: str | None = None
    combined_evidence: Mapping[str, Any] | None = None
    # Strict remains the default for all existing manifests. Bounded service
    # qualifies a drain envelope, never completion of the admission ceiling.
    service_mode: str = "strict"

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "ProofCapacityQualification":
        try:
            qualified = value["qualified"]
            if not isinstance(qualified, bool):
                raise TypeError("qualified must be a boolean")
            return cls(
                combined_evidence=value.get("combined_evidence"),
                service_mode=str(value.get("service_mode", "strict")),
                schema_version=int(value["schema_version"]),
                profile_id=str(value["profile_id"]),
                model_revision=str(value["model_revision"]),
                software_revision=str(value["software_revision"]),
                checkpoint_revision=str(value["checkpoint_revision"]),
                samples_sha256=str(value["samples_sha256"]),
                runtime_fingerprint_hash=str(
                    value["runtime_fingerprint_hash"]
                ),
                hardware_class=str(value["hardware_class"]),
                benchmark_device_count=int(
                    value["benchmark_device_count"]
                ),
                benchmark_device_uuids=tuple(
                    str(device_uuid).strip().casefold()
                    for device_uuid in value["benchmark_device_uuids"]
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
                p95_seconds_per_proof_by_environment_and_device={
                    str(environment): {
                        str(device_uuid).strip().casefold(): float(seconds)
                        for device_uuid, seconds in dict(device_values).items()
                    }
                    for environment, device_values in dict(
                        value[
                            "p95_seconds_per_proof_by_environment_and_device"
                        ]
                    ).items()
                },
                sample_count_by_environment={
                    str(name): int(count)
                    for name, count in dict(
                        value["sample_count_by_environment"]
                    ).items()
                },
                sample_count_by_environment_and_device={
                    str(environment): {
                        str(device_uuid).strip().casefold(): int(count)
                        for device_uuid, count in dict(device_values).items()
                    }
                    for environment, device_values in dict(
                        value[
                            "sample_count_by_environment_and_device"
                        ]
                    ).items()
                },
                minimum_samples_per_device_per_environment=int(
                    value["minimum_samples_per_device_per_environment"]
                ),
                minimum_completion_tokens_by_environment={
                    str(name): int(count)
                    for name, count in dict(
                        value["minimum_completion_tokens_by_environment"]
                    ).items()
                },
                measured_at=str(value["measured_at"]),
                qualified=qualified,
                proof_path_hash=(
                    str(value["proof_path_hash"])
                    if value.get("proof_path_hash") is not None
                    else None
                ),
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
        checkpoint_revision: str,
        runtime_fingerprint_hash: str,
        proof_path_hash: str | None = None,
        configured_devices: Sequence[str],
        configured_hardware: Sequence[str],
        configured_device_uuids: Sequence[str],
        proof_wall_seconds: float,
        minimum_proofs_per_environment: int,
        minimum_completion_tokens_per_environment: Mapping[str, int],
        configured_slots: Mapping[str, str] | None = None,
        maximum_context_tokens: int | None = None,
        full_completion_tokens_per_environment: Mapping[str, int] | None = None,
    ) -> dict[str, Any]:
        if self.schema_version not in (3, 4):
            raise ProofCapacityQualificationError(
                "unsupported proof-capacity manifest schema"
            )
        budget = capacity_budget()
        bounded = budget["mode"] == "fill_closed_bounded"
        if self.service_mode != ("bounded" if bounded else "strict"):
            raise ProofCapacityQualificationError("proof service mode differs from runtime")
        if bounded and self.schema_version != 4:
            raise ProofCapacityQualificationError("bounded fill service requires combined schema 4 evidence")
        if self.schema_version == 4:
            if (proof_wall_seconds != budget["wall_seconds"]
                    or minimum_proofs_per_environment != budget["proofs_per_environment"]):
                raise ProofCapacityQualificationError("combined capacity caller does not match the active scheduler budget")
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
        if (
            software_revision is None
            or _COMMIT_SHA_RE.fullmatch(software_revision) is None
            or _COMMIT_SHA_RE.fullmatch(self.software_revision) is None
        ):
            raise ProofCapacityQualificationError(
                "proof-capacity software revision mismatch"
            )
        if self.schema_version == 4 and (
            self.software_revision != software_revision
            or self.runtime_fingerprint_hash != runtime_fingerprint_hash
        ):
            raise ProofCapacityQualificationError("combined capacity requires the exact measured runtime")
        qualification_carried_over_from: str | None = None
        if self.software_revision != software_revision:
            # Carry-over: a different image is acceptable iff the proof path
            # it ships is byte-identical to the one benchmarked. Legacy
            # manifests (no stored hash) and callers that cannot compute
            # their own hash keep the strict fail-closed behavior.
            if (
                self.proof_path_hash is None
                or proof_path_hash is None
                or _SHA256_RE.fullmatch(self.proof_path_hash) is None
                or _SHA256_RE.fullmatch(proof_path_hash) is None
            ):
                raise ProofCapacityQualificationError(
                    "proof-capacity software revision mismatch"
                )
            if self.proof_path_hash != proof_path_hash:
                from reliquary.constants import (
                    PROOF_CAPACITY_ACCEPT_FASTER_RUNTIME,
                )

                # Same lower-bound argument as the runtime-fingerprint switch:
                # an operator-asserted faster proof path keeps the stored
                # capacity valid as a floor. Fail-closed without the switch.
                if not PROOF_CAPACITY_ACCEPT_FASTER_RUNTIME:
                    raise ProofCapacityQualificationError(
                        "proof-capacity software revision mismatch"
                    )
            qualification_carried_over_from = self.software_revision
        if (
            _COMMIT_SHA_RE.fullmatch(checkpoint_revision) is None
            or _COMMIT_SHA_RE.fullmatch(self.checkpoint_revision) is None
            or self.checkpoint_revision != checkpoint_revision
        ):
            raise ProofCapacityQualificationError(
                "proof-capacity checkpoint revision mismatch"
            )
        if _SHA256_RE.fullmatch(self.samples_sha256) is None:
            raise ProofCapacityQualificationError(
                "proof-capacity sample digest must be lowercase SHA-256"
            )
        if (
            _SHA256_RE.fullmatch(runtime_fingerprint_hash) is None
            or _SHA256_RE.fullmatch(self.runtime_fingerprint_hash) is None
        ):
            raise ProofCapacityQualificationError(
                "proof-capacity runtime fingerprint mismatch"
            )
        runtime_fingerprint_carried_over_from: str | None = None
        if self.runtime_fingerprint_hash != runtime_fingerprint_hash:
            from reliquary.constants import (
                PROOF_CAPACITY_ACCEPT_FASTER_RUNTIME,
            )

            # A runtime change (kernel upgrade, dependency bump) invalidates
            # the benchmark fingerprint. With the operator switch asserting
            # the new runtime is AT LEAST AS FAST on the proof path, the
            # stored capacity stays valid as a conservative lower bound — it
            # was measured under the slower runtime, so it can only
            # under-promise. Anything else stays fail-closed.
            if not PROOF_CAPACITY_ACCEPT_FASTER_RUNTIME:
                raise ProofCapacityQualificationError(
                    "proof-capacity runtime fingerprint mismatch"
                )
            runtime_fingerprint_carried_over_from = (
                self.runtime_fingerprint_hash
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
        if any(
            not _is_canonical_cuda_index(device)
            for device in configured_devices
        ):
            raise ProofCapacityQualificationError(
                "configured proof devices must be canonical CUDA indices"
            )
        if not (
            len(configured_devices)
            == len(configured_hardware)
            == len(configured_device_uuids)
            == self.benchmark_device_count
        ):
            raise ProofCapacityQualificationError(
                "runtime proof fleet topology differs from benchmark"
            )
        benchmark_uuids = tuple(
            device_uuid.strip().casefold()
            for device_uuid in self.benchmark_device_uuids
        )
        runtime_uuids = tuple(
            str(device_uuid).strip().casefold()
            for device_uuid in configured_device_uuids
        )
        if (
            len(benchmark_uuids) != self.benchmark_device_count
            or len(set(benchmark_uuids)) != len(benchmark_uuids)
            or not all(benchmark_uuids)
            or len(set(runtime_uuids)) != len(runtime_uuids)
            or set(runtime_uuids) != set(benchmark_uuids)
        ):
            raise ProofCapacityQualificationError(
                "configured proof GPU UUIDs differ from benchmark"
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

        combined_bounds = None
        if self.schema_version == 4:
            from reliquary.validator.proof_capacity_combined import validate_combined_evidence
            try:
                combined_bounds = validate_combined_evidence(
                    self.combined_evidence, device_uuids=benchmark_uuids,
                    environments=minimum_completion_tokens_per_environment,
                    maximum_context_tokens=maximum_context_tokens,
                    completion_caps=full_completion_tokens_per_environment,
                    minimum_samples=self.minimum_samples_per_device_per_environment,
                    natural_p95=self.p95_seconds_per_proof_by_environment_and_device,
                    natural_samples_sha256=self.samples_sha256, configured_slots=configured_slots,
                )
            except (ValueError, TypeError, KeyError, OSError) as exc:
                raise ProofCapacityQualificationError(f"invalid combined capacity evidence: {exc}") from exc
            if self.headroom_fraction < 0.2:
                raise ProofCapacityQualificationError("combined capacity requires at least 20% headroom")
        elif self.combined_evidence is not None:
            raise ProofCapacityQualificationError("combined evidence requires schema 4")
        required_device_seconds = 0.0
        if self.minimum_samples_per_device_per_environment < 20:
            raise ProofCapacityQualificationError(
                "proof-capacity benchmark needs 20 samples per GPU and environment"
            )
        required_environments = tuple(
            minimum_completion_tokens_per_environment
        )
        if not required_environments:
            raise ProofCapacityQualificationError(
                "proof-capacity qualification requires an active environment"
            )
        for environment in required_environments:
            proof_count = self.proofs_per_environment.get(environment)
            p95_seconds = self.p95_seconds_per_proof.get(environment)
            if (
                proof_count is None
                or proof_count < minimum_proofs_per_environment
            ):
                raise ProofCapacityQualificationError(
                    f"{environment} benchmark does not reserve enough proofs"
                )
            sample_count = self.sample_count_by_environment.get(environment)
            minimum_sample_count = (
                self.minimum_samples_per_device_per_environment
                * self.benchmark_device_count
            )
            if sample_count is None or sample_count < minimum_sample_count:
                raise ProofCapacityQualificationError(
                    f"{environment} benchmark has too few proof samples"
                )
            per_device_sample_count = (
                self.sample_count_by_environment_and_device.get(
                    environment,
                    {},
                )
            )
            if set(per_device_sample_count) != set(benchmark_uuids):
                raise ProofCapacityQualificationError(
                    f"{environment} sample counts do not cover every GPU"
                )
            if any(
                count < self.minimum_samples_per_device_per_environment
                for count in per_device_sample_count.values()
            ):
                raise ProofCapacityQualificationError(
                    f"{environment} benchmark has too few per-GPU samples"
                )
            if sample_count != sum(per_device_sample_count.values()):
                raise ProofCapacityQualificationError(
                    f"{environment} sample counts do not conserve"
                )
            per_device_p95 = (
                self.p95_seconds_per_proof_by_environment_and_device.get(
                    environment,
                    {},
                )
            )
            if set(per_device_p95) != set(benchmark_uuids):
                raise ProofCapacityQualificationError(
                    f"{environment} benchmark does not cover every GPU"
                )
            if any(
                not math.isfinite(seconds)
                or seconds <= 0
                or seconds >= proof_wall_seconds
                for seconds in per_device_p95.values()
            ):
                raise ProofCapacityQualificationError(
                    f"{environment} per-GPU p95 proof latency is invalid"
                )
            required_completion_tokens = (
                minimum_completion_tokens_per_environment.get(environment)
            )
            measured_completion_tokens = (
                self.minimum_completion_tokens_by_environment.get(environment)
            )
            if (
                required_completion_tokens is None
                or measured_completion_tokens is None
                or measured_completion_tokens < (1 if combined_bounds is not None else required_completion_tokens)
            ):
                raise ProofCapacityQualificationError(
                    f"{environment} benchmark is not representative of "
                    "the protocol completion cap"
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
            if not math.isclose(
                p95_seconds,
                max(per_device_p95.values()),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ProofCapacityQualificationError(
                    f"{environment} aggregate p95 is not the worst GPU p95"
                )
            required_device_seconds += proof_count * (
                max(combined_bounds[environment].values()) if combined_bounds is not None else p95_seconds
            )

        usable_seconds_per_device = proof_wall_seconds * (
            1.0 - self.headroom_fraction
        )
        minimum_devices = math.ceil(
            required_device_seconds / usable_seconds_per_device
        )
        bounded_group_seconds = None
        if bounded:
            bounded_group_seconds = max(
                value for devices in combined_bounds.values() for value in devices.values()
            ) / (1.0 - self.headroom_fraction)
            if bounded_group_seconds > budget["drain_seconds"]:
                raise ProofCapacityQualificationError(
                    "measured group bound with headroom exceeds bounded proof drain margin"
                )
        elif len(configured_devices) < minimum_devices:
            raise ProofCapacityQualificationError(
                "configured proof fleet is below measured capacity: "
                f"requires {minimum_devices}, has {len(configured_devices)}"
            )

        return {
            "qualified": True,
            "schema_version": self.schema_version,
            "service_mode": self.service_mode,
            "window_budget": budget,
            "all_admitted_proofs_qualified": not bounded,
            "guaranteed_picks_per_window": 0 if bounded else None,
            "qualified_group_seconds_with_headroom": bounded_group_seconds,
            "combined_seconds_per_proof_bound": combined_bounds,
            "qualification_carried_over_from": qualification_carried_over_from,
            "profile_id": self.profile_id,
            "model_revision": self.model_revision,
            "software_revision": self.software_revision,
            "checkpoint_revision": self.checkpoint_revision,
            "samples_sha256": self.samples_sha256,
            "runtime_fingerprint_hash": self.runtime_fingerprint_hash,
            "runtime_fingerprint_carried_over_from": (
                runtime_fingerprint_carried_over_from
            ),
            "hardware_class": self.hardware_class,
            "configured_device_count": len(configured_devices),
            "configured_device_uuids": sorted(runtime_uuids),
            "minimum_device_count": None if bounded else minimum_devices,
            "required_device_seconds": None if bounded else required_device_seconds,
            "all_admission_demand_device_seconds": required_device_seconds,
            "available_device_seconds": (
                len(configured_devices) * usable_seconds_per_device
            ),
            "headroom_fraction": self.headroom_fraction,
            "proofs_per_environment": dict(self.proofs_per_environment),
            "p95_seconds_per_proof_by_environment_and_device": {
                environment: dict(values)
                for environment, values in (
                    self.p95_seconds_per_proof_by_environment_and_device.items()
                )
            },
            "sample_count_by_environment": dict(
                self.sample_count_by_environment
            ),
            "sample_count_by_environment_and_device": {
                environment: dict(values)
                for environment, values in (
                    self.sample_count_by_environment_and_device.items()
                )
            },
            "minimum_samples_per_device_per_environment": (
                self.minimum_samples_per_device_per_environment
            ),
            "minimum_completion_tokens_by_environment": dict(
                self.minimum_completion_tokens_by_environment
            ),
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
    "PROOF_PATH_FILES",
    "compute_proof_path_hash",
    "ProofCapacityQualification",
    "ProofCapacityQualificationError",
    "ProofDeviceIdentity",
    "load_proof_capacity_qualification",
    "resolve_cuda_proof_devices",
]
