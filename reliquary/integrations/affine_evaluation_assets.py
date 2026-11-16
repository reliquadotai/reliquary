"""Check a local checkpoint export against an authenticated Affine opening."""

from __future__ import annotations

import hashlib
from itertools import islice
import os
from pathlib import Path

from .affine import _private_file, _private_path
from .affine_checkpoint import _member, _stamp
from .affine_epoch import canonical, signed
from .affine_evaluation import HEX, SCHEMA
from .affine_scoring import checksum, read_json, require, write_json
from reliquary.protocol.release_contract import canonical_sha256


def check_assets(config_path: Path, output: Path) -> dict:
    """Authenticate an opening and local bytes; never fetch or activate anything."""
    require(
        Path(config_path).is_absolute() and Path(output).is_absolute(),
        "absolute private paths required",
    )
    _private_file(Path(config_path))
    config = read_json(config_path, limit=1024**2)
    paths = {"preparedOffer", "manifestEnvelope", "modelSnapshot", "modelDirectory"}
    require(
        isinstance(config, dict)
        and set(config) == paths | {"schema", "authority"}
        and config["schema"] == "affine-evaluation-assets/v1"
        and isinstance(config["authority"], str)
        and HEX.fullmatch(config["authority"]),
        "invalid local asset check configuration",
    )
    for name in paths:
        require(
            isinstance(config[name], str) and Path(config[name]).is_absolute(),
            "absolute private asset paths required",
        )
        if name != "modelDirectory":
            _private_file(Path(config[name]))
    offer = read_json(config["preparedOffer"], limit=256 * 1024)
    require(
        isinstance(offer, dict)
        and offer.get("schema") == SCHEMA
        and offer.get("availability") == "withdrawn"
        and offer.get("compatibility") == "independent-subnet-evaluation"
        and all(
            offer.get(key) is False
            for key in (
                "nativeParityVerified",
                "nativeTaskMappingVerified",
                "gpuQualificationVerified",
                "publicationVerified",
                "publicModelLiveVerified",
            )
        )
        and offer.get("offerDigest")
        == canonical_sha256(
            {key: value for key, value in offer.items() if key != "offerDigest"}
        ),
        "invalid frozen preparation offer",
    )
    snapshot = read_json(config["modelSnapshot"], limit=1024**2)
    require(
        checksum(config["modelSnapshot"]) == offer.get("modelSnapshotSha256")
        and isinstance(snapshot, dict)
        and snapshot.get("id") == offer["quoteRequest"]["model"]
        and snapshot.get("sha") == offer["quoteRequest"]["revision"]
        and snapshot.get("private") is False
        and snapshot.get("gated") is False,
        "local public model snapshot differs from prepared offer",
    )
    envelope = read_json(config["manifestEnvelope"], limit=32 * 1024**2)
    try:
        manifest = signed(envelope, config["authority"])
    except Exception:
        raise ValueError("invalid native opening signature") from None
    manifest_sha = hashlib.sha256(canonical(manifest)).hexdigest()
    native = offer["nativeSuite"]
    require(
        manifest_sha == native["manifestSha256"],
        "native opening differs from prepared offer",
    )
    suite = native["suite"]
    require(
        isinstance(manifest.get("evaluation_sets"), list),
        "original signed evaluation suite required",
    )
    matches = [
        row
        for row in manifest["evaluation_sets"]
        if isinstance(row, dict) and row.get("name") == suite["name"]
    ]
    require(
        len(matches) == 1 and canonical(matches[0]) == canonical(suite),
        "original signed evaluation suite differs",
    )
    checkpoint = manifest.get("checkpoint")
    require(
        isinstance(checkpoint, dict)
        and set(checkpoint) in ({"id", "files"}, {"id", "files", "read_urls"}),
        "exact native checkpoint descriptor required",
    )
    files = checkpoint.get("files")
    require(
        isinstance(files, dict) and 1 <= len(files) <= 256,
        "bounded native checkpoint inventory required",
    )
    urls = checkpoint.get("read_urls", {})
    require(
        isinstance(urls, dict)
        and (not urls or set(urls) == set(files))
        and all(isinstance(value, str) for value in urls.values()),
        "native checkpoint transport inventory differs",
    )
    for name, digest in files.items():
        require(
            isinstance(name, str)
            and Path(name).name == name
            and not name.startswith(".")
            and isinstance(digest, str)
            and HEX.fullmatch(digest),
            "invalid native checkpoint filename or hash",
        )
    require(
        checkpoint.get("id") == hashlib.sha256(canonical(files)).hexdigest()
        and "config.json" in files
        and any(name.endswith(".safetensors") for name in files),
        "native checkpoint content identifier differs",
    )
    directory = Path(config["modelDirectory"])
    _private_path(directory)
    require(
        directory.is_dir()
        and directory.stat().st_uid == os.getuid()
        and not directory.stat().st_mode & 0o077,
        "dedicated owner-private checkpoint export required",
    )
    members = list(islice(directory.iterdir(), 257))
    require(
        {path.name for path in members} == set(files) and len(members) == len(files),
        "local checkpoint inventory differs",
    )
    stamps = {path.name: _stamp(path.stat(follow_symlinks=False)) for path in members}
    observed = {path.name: _member(path) for path in sorted(members)}
    require(
        all(observed[name]["sha256"] == digest for name, digest in files.items()),
        "local checkpoint bytes differ",
    )
    require(
        observed["config.json"]["sha256"] == offer.get("modelConfigSha256"),
        "prepared model config differs from checkpoint bytes",
    )
    require(
        {path.name for path in islice(directory.iterdir(), 257)} == set(files)
        and all(
            _stamp(path.stat(follow_symlinks=False)) == stamps[path.name]
            for path in members
        ),
        "local checkpoint changed during verification",
    )
    receipt = {
        "schema": "affine-evaluation-assets-result/v1",
        "stage": "local_assets_checked",
        "offerDigest": offer["offerDigest"],
        "authority": config["authority"],
        "manifestSha256": manifest_sha,
        "checkpointId": checkpoint["id"],
        "model": snapshot["id"],
        "revision": snapshot["sha"],
        "localInventoryDigest": canonical_sha256(observed),
        "fileCount": len(observed),
        "localCheckpointBytesVerified": True,
        "openingSignatureVerified": True,
        "publicModelLiveVerified": False,
        "nativeTaskMappingVerified": False,
        "nativeParityVerified": False,
        "gpuQualificationVerified": False,
        "activated": False,
    }
    write_json(output, receipt)
    return {
        key: receipt[key]
        for key in (
            "schema",
            "stage",
            "offerDigest",
            "fileCount",
            "localCheckpointBytesVerified",
            "openingSignatureVerified",
            "nativeParityVerified",
            "activated",
        )
    }
