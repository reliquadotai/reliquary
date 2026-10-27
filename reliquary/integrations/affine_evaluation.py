"""Prepare an independent subnet evaluation from frozen, reviewed operator assets."""

from __future__ import annotations

import re
from pathlib import Path

from reliquary.eval import protocol
from reliquary.eval.prompt_source import single_turn_messages
from reliquary.eval.storage import _verified
from reliquary.integrations.affine import _private_file
from reliquary.integrations.affine_scoring import (
    checksum,
    read_json,
    require,
    write_json,
)
from reliquary.protocol.release_contract import canonical_sha256
from reliquary.shared.strict_json import strict_json_loads

SCHEMA = "affine-subnet-evaluation-offer/v1"
HEX = re.compile(r"[0-9a-f]{64}")
REVISION = re.compile(r"[0-9a-f]{40}")
REPO = re.compile(r"[A-Za-z0-9][\w.-]{0,95}/[A-Za-z0-9][\w.-]{0,95}")
REVIEW_FIELDS = {
    "sourceApproved",
    "rightsApproved",
    "overlapReviewed",
    "publicModelApproved",
    "promptsVisibleToMinersApproved",
}


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def _object(value, fields):
    return isinstance(value, dict) and set(value) == set(fields)


def _hash(value):
    return isinstance(value, str) and HEX.fullmatch(value)


def prepare(config_path: Path, output: Path) -> dict:
    """Freeze a private offer; never publish a set, qualify a GPU or place an order."""
    from reliquary.constants import SUPPORTED_ARCHITECTURES

    _private_file(Path(config_path))
    config = read_json(config_path, limit=1024**2)
    require(
        _object(
            config,
            {
                "schema",
                "setDirectory",
                "modelSnapshot",
                "modelConfig",
                "nativeSuite",
                "request",
                "review",
            },
        )
        and config["schema"] == "affine-subnet-evaluation-intake/v1",
        "invalid evaluation intake",
    )
    review = config["review"]
    require(
        _object(review, REVIEW_FIELDS)
        and all(value is True for value in review.values()),
        "source, rights, overlap, public model and prompt visibility require review",
    )
    for name in ("setDirectory", "modelSnapshot", "modelConfig"):
        require(
            isinstance(config[name], str) and Path(config[name]).is_absolute(),
            "intake paths must be absolute",
        )
    directory = Path(config["setDirectory"])
    for name in ("set.json", "prompts.jsonl", "grading.jsonl"):
        path = directory / name
        _private_file(path)
        require(
            path.stat().st_size <= 64 * 1024**2,
            "evaluation set exceeds preparation bound",
        )
    card, _, prompts, grading = _verified(directory)
    require(card == read_json(directory / "set.json"), "set JSON changed")
    digest = protocol.validate_set(card, current_grader=True)
    require(
        digest is not None and card.get("interaction", "single_turn") == "single_turn",
        "frozen single-turn protocol required",
    )
    envelope = {
        key: card[key] for key in ("protocol", "protocol_json", "protocol_sha256")
    }
    protocol.validate_execution({card["set_id"]: envelope}, {card["set_id"]: digest})
    body = card["protocol"]
    source, overlap = body["source"], body["disjointness"]
    require(
        isinstance(source.get("data"), dict)
        and source["data"]
        and all(_hash(value) for value in source["data"].values())
        and isinstance(source.get("licenses"), list)
        and source["licenses"]
        and all(
            _object(item, {"id", "sha256"})
            and isinstance(item["id"], str)
            and item["id"]
            and _hash(item["sha256"])
            for item in source["licenses"]
        )
        and _object(overlap, {"disjoint", "evidence_sha256"})
        and type(overlap["disjoint"]) is bool
        and _hash(overlap["evidence_sha256"]),
        "reviewed source, rights and overlap pins required",
    )
    rows = [
        strict_json_loads(line)
        for line in prompts.decode().splitlines()
        if line.strip()
    ]
    grade_rows = [
        strict_json_loads(line)
        for line in grading.decode().splitlines()
        if line.strip()
    ]
    require(
        _integer(card["count"], 1, 5000)
        and len(rows) == len(grade_rows) == card["count"]
        and [row["problem_id"] for row in rows]
        == [row["problem_id"] for row in grade_rows]
        and len({row["problem_id"] for row in rows}) == len(rows),
        "invalid frozen problem coverage",
    )
    for row in rows:
        require(
            single_turn_messages(row.get("messages")) is not None,
            "subnet evaluation requires single-turn text prompts",
        )
    for name in ("modelSnapshot", "modelConfig"):
        _private_file(Path(config[name]))
    snapshot = read_json(config["modelSnapshot"], limit=1024**2)
    model_config = read_json(config["modelConfig"], limit=1024**2)
    require(
        isinstance(snapshot, dict)
        and isinstance(snapshot.get("id"), str)
        and REPO.fullmatch(snapshot["id"])
        and ".." not in snapshot["id"]
        and isinstance(snapshot.get("sha"), str)
        and REVISION.fullmatch(snapshot["sha"])
        and snapshot.get("private") is False
        and snapshot.get("gated") is False,
        "resolved public ungated model snapshot required",
    )
    architectures = (
        model_config.get("architectures") if isinstance(model_config, dict) else None
    )
    require(
        isinstance(architectures, list)
        and architectures
        and isinstance(architectures[0], str)
        and architectures[0] in SUPPORTED_ARCHITECTURES,
        "model architecture is not supported by subnet evaluation",
    )
    native = config["nativeSuite"]
    require(
        _object(native, {"manifestSha256", "sourceRevision", "suite", "taskMapping"})
        and _hash(native["manifestSha256"])
        and isinstance(native["sourceRevision"], str)
        and REVISION.fullmatch(native["sourceRevision"]),
        "invalid native suite provenance",
    )
    suite = native["suite"]
    require(
        _object(suite, {"name", "env_id", "indices", "seed", "repeats"})
        and all(
            isinstance(suite[name], str) and 0 < len(suite[name]) <= 128
            for name in ("name", "env_id")
        )
        and isinstance(suite["indices"], list)
        and len(suite["indices"]) == len(rows)
        and all(_integer(index, 0, 10**9) for index in suite["indices"])
        and len(set(suite["indices"])) == len(rows)
        and _integer(suite["seed"], 0, 2**31 - 1)
        and _integer(suite["repeats"], 1, 16),
        "invalid native suite",
    )
    mapping = native["taskMapping"]
    require(
        isinstance(mapping, list) and len(mapping) == len(rows),
        "native task mapping required",
    )
    for index, row, entry in zip(suite["indices"], rows, mapping, strict=True):
        require(
            _object(entry, {"nativeIndex", "taskSha256", "problemId", "promptSha256"})
            and _integer(entry["nativeIndex"], 0, 10**9)
            and entry["nativeIndex"] == index
            and _hash(entry["taskSha256"])
            and entry["problemId"] == row["problem_id"]
            and entry["promptSha256"] == canonical_sha256(row["messages"]),
            "native provenance mapping differs from frozen prompts",
        )
    request = config["request"]
    require(
        _object(
            request, {"problems", "samples", "max_new_tokens", "thinking", "sampling"}
        )
        and _integer(request["problems"], 1, len(rows))
        and _integer(request["samples"], 1, 64)
        and _integer(request["max_new_tokens"], 16, 131072)
        and type(request["thinking"]) is bool,
        "invalid bounded subnet request",
    )
    sampling = request["sampling"]
    require(
        _object(sampling, {"temperature", "top_p", "top_k", "seed"})
        and type(sampling["temperature"]) in (float, int)
        and 0 < sampling["temperature"] <= 2
        and type(sampling["top_p"]) in (float, int)
        and 0.01 <= sampling["top_p"] <= 1
        and (sampling["top_k"] == -1 or _integer(sampling["top_k"], 1, 1000))
        and _integer(sampling["seed"], 0, 2**31 - 1),
        "invalid subnet sampling",
    )
    offer = {
        "schema": SCHEMA,
        "availability": "withdrawn",
        "compatibility": "independent-subnet-evaluation",
        "nativeParityVerified": False,
        "nativeTaskMappingVerified": False,
        "gpuQualificationVerified": False,
        "publicationVerified": False,
        "publicModelLiveVerified": False,
        "promptsVisibleToMiners": True,
        "modelArchitecture": architectures[0],
        "modelSnapshotSha256": checksum(config["modelSnapshot"]),
        "modelConfigSha256": checksum(config["modelConfig"]),
        "setProtocolSha256": digest,
        "setFiles": {
            name: checksum(directory / name)
            for name in ("set.json", "prompts.jsonl", "grading.jsonl")
        },
        "nativeSuite": native,
        "review": review,
        "quoteRequest": {
            "model": snapshot["id"],
            "revision": snapshot["sha"],
            "mode": "subnet",
            "envs": [
                {
                    "env": card["env"],
                    "set_id": card["set_id"],
                    "problems": request["problems"],
                    "samples": request["samples"],
                }
            ],
            "max_new_tokens": request["max_new_tokens"],
            "thinking": request["thinking"],
            "sampling": sampling,
        },
    }
    offer["offerDigest"] = canonical_sha256(offer)
    write_json(output, offer)
    return {
        key: offer[key]
        for key in (
            "schema",
            "offerDigest",
            "availability",
            "compatibility",
            "nativeParityVerified",
            "gpuQualificationVerified",
            "publicationVerified",
        )
    }
