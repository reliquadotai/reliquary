"""Offline native evaluation admission; never allocates or qualifies GPU execution."""

from __future__ import annotations

from email.parser import BytesParser
import math
from pathlib import Path
import re
import time
import zipfile

from . import affine_native_evaluation as native
from .affine import _directory, _private_file
from .affine_scoring import checksum, read_json, require


def runtime_inventory(requirements, wheels, versions):
    """Check the supplied wheel inventory; image resolution still needs pip check."""
    requirements, wheels = Path(requirements), Path(wheels)
    _private_file(requirements)
    _directory(wheels)
    require(requirements.stat().st_size <= 1024**2, "runtime lock too large")
    text = requirements.read_text().replace("\\\n", " ")
    pins = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(
            r"([A-Za-z0-9_.-]+)==([^\s;]+)((?:\s+--hash=sha256:[0-9a-f]{64})+)", line
        )
        require(
            match is not None, "runtime lock requires exact versions and SHA256 hashes"
        )
        name, version, hashes = match.groups()
        key = re.sub(r"[-_.]+", "-", name).lower()
        require(key not in pins, "duplicate runtime pin")
        pins[key] = (version, set(re.findall(r"sha256:([0-9a-f]{64})", hashes)))
    require(0 < len(pins) <= 512, "bounded complete runtime inventory required")
    require(
        all(
            pins.get(re.sub(r"[-_.]+", "-", name).lower(), (None,))[0] == version
            for name, version in versions.items()
        ),
        "signed native package fingerprint differs",
    )
    files, found = {}, set()
    members = list(wheels.iterdir())
    require(len(members) <= 512, "runtime wheel inventory too large")
    for path in members:
        _private_file(path)
        require(
            path.name.endswith(".whl") and path.stat().st_size <= 16 * 1024**3,
            "only bounded runtime wheels allowed",
        )
        with zipfile.ZipFile(path) as archive:
            metadata = [
                v
                for v in archive.infolist()
                if v.filename.endswith(".dist-info/METADATA")
            ]
            require(
                len(metadata) == 1 and metadata[0].file_size <= 1024**2,
                "wheel metadata unavailable",
            )
            values = BytesParser().parsebytes(archive.read(metadata[0]))
        name = re.sub(r"[-_.]+", "-", values.get("Name", "")).lower()
        version = values.get("Version")
        digest = checksum(path)
        require(
            name in pins
            and name not in found
            and pins[name][0] == version
            and digest in pins[name][1],
            "runtime wheel differs from the lock",
        )
        found.add(name)
        files[path.name] = dict(sha256=digest, bytes=path.stat().st_size)
    missing = sorted(set(pins) - found)
    return dict(
        requirementsSha256=checksum(requirements),
        wheels=files,
        missingPackages=missing,
        wheelInventoryVerified=not missing,
        dependencyResolutionVerified=False,
        runtimeImageQualified=False,
    )


def preflight(bundle, config_path, *, now=None):
    config_path = Path(config_path)
    _private_file(config_path)
    c = read_json(config_path, 65536)
    require(
        isinstance(c, dict)
        and set(c)
        == {
            "schema",
            "maxSeconds",
            "rateMicroUsdPerHour",
            "maxCostMicroUsd",
            "providerCeilingMicroUsdPerGpuHour",
            "gpuCount",
            "companyCostCeilingMicroUsd",
            "image",
            "requirementsFile",
            "wheelDirectory",
        }
        and c["schema"] == "affine-native-preflight/v1",
        "exact preflight fields required",
    )
    for key, low, high in (
        ("maxSeconds", 2, 86400),
        ("rateMicroUsdPerHour", 1, 10**9),
        ("maxCostMicroUsd", 1, 10**12),
        ("providerCeilingMicroUsdPerGpuHour", 1, 10**9),
        ("gpuCount", 1, 1),
        ("companyCostCeilingMicroUsd", 1, 10**12),
    ):
        require(
            type(c[key]) is int and low <= c[key] <= high,
            "invalid bounded preflight amount",
        )
    require(
        isinstance(c["image"], str)
        and re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", c["image"]),
        "digest-pinned runtime image required",
    )
    d, job, _, _ = native.load_bundle(bundle)
    now = time.time() if now is None else now
    require(
        type(now) in (int, float) and math.isfinite(now), "invalid observation time"
    )
    inventory = runtime_inventory(
        c["requirementsFile"], c["wheelDirectory"], job["runtime_versions"]
    )
    charge = (c["maxSeconds"] * c["rateMicroUsdPerHour"] + 3599) // 3600
    provider = (
        c["maxSeconds"] * c["gpuCount"] * c["providerCeilingMicroUsdPerGpuHour"] + 3599
    ) // 3600
    blockers = []
    if d["assurance"] == "fixture":
        blockers.append("fixture_only")
    if not job["created_at"] <= now or now + c["maxSeconds"] > d["nativeJobExpiresAt"]:
        blockers.append("original_job_lifetime_insufficient")
    if (
        not d["limits"]["finalizationReserveSeconds"]
        < c["maxSeconds"]
        <= d["limits"]["maxSeconds"]
    ):
        blockers.append("allocation_duration_incompatible")
    if charge > c["maxCostMicroUsd"]:
        blockers.append("customer_budget_insufficient")
    if provider > c["companyCostCeilingMicroUsd"]:
        blockers.append("company_compute_budget_insufficient")
    if not inventory["wheelInventoryVerified"]:
        blockers.append("runtime_wheels_missing")
    return dict(
        schema="affine-native-preflight-result/v1",
        jobDigest=d["jobDigest"],
        image=c["image"],
        maxSeconds=c["maxSeconds"],
        maxReservedMicroUsd=charge,
        maxProviderComputeMicroUsd=provider,
        offlineReady=not blockers,
        blockers=blockers,
        runtimeInventory=inventory,
        costCoverage="allocation_gpu_time_only",
        totalRentalCostQualified=False,
        qualificationRequired=[
            "linux_amd64_python312_wheel_compatibility",
            "resolved_runtime_image",
            "native_h200_parity",
            "account_and_capacity_approval",
        ],
        nativeExecutionReady=False,
        activated=False,
    )
