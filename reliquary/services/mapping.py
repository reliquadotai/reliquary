"""Mapping/export artifacts with explicit observation confidence."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from reliquary.protocol.release_contract import canonical_json_bytes, canonical_sha256
from reliquary.protocol.service_contract import ServiceContract
from reliquary.services.observations import observation_id, observation_signal, validate_observation


def environment_version(card: dict) -> str:
    return card.get("environment_manifest_sha256") or canonical_sha256({
        "source": card["source"], "taskset": card.get("taskset"),
        "prompt_template_id": card.get("prompt_template_id"), "grading_sha256": card["grading_sha256"],
    })


def validate_grading_context(contract: ServiceContract, sets: dict, provenance: dict) -> None:
    value = contract.to_dict()
    if value["service_kind"] != "dataset_mapping" or value["policies"]["sampling"]["kind"] != "legacy/v1" or value["scoring"]["kind"] != "environment-reward/v1":
        raise ValueError("this grading adapter supports frozen legacy-sampling environment-reward mapping only")
    if list(sets) != [value["dataset"]["id"]]:
        raise ValueError("mapping needs its one qualified dataset")
    card, rows = sets[value["dataset"]["id"]]
    if card["prompts_sha256"] != value["dataset"]["sha256"]:
        raise ValueError("mapping dataset digest mismatch")
    if card["env"] != value["environment"]["id"] or environment_version(card) != value["environment"]["version"]:
        raise ValueError("mapping environment version mismatch")
    for name, source in (("repo", "model"), ("revision", "revision"), ("sha256", "checkpoint_sha256")):
        if value["checkpoint"][name] != provenance.get(source):
            raise ValueError(f"mapping checkpoint {name} mismatch or unavailable")
    if provenance.get("generation_contract_sha256") != value["generation_contract_sha256"]:
        raise ValueError("mapping generation contract mismatch or unavailable")
    if len(rows) > value["limits"]["max_groups"]:
        raise ValueError("mapping population exceeds budget")


def export_grading_mapping(directory: Path, contract: ServiceContract, sets: dict, report: dict,
                           *, generation_verified: bool) -> list[Path]:
    import pyarrow.parquet as pq

    validate_grading_context(contract, sets, report["provenance"])
    graded_path = directory / "graded.parquet"
    source_digest = hashlib.sha256(graded_path.read_bytes()).hexdigest()
    by_problem: dict[str, list] = {}
    for row in pq.read_table(graded_path, columns=["problem_id", "sample_index", "score", "tokens"]).to_pylist():
        by_problem.setdefault(row["problem_id"], []).append(row)
    set_id = contract.to_dict()["dataset"]["id"]
    sample_count = report["provenance"]["sets"][0]["samples"]
    version_drift = any(s.get("grader_version_drift") for s in report["provenance"]["sets"])
    observations = []
    for problem in sets[set_id][1]:
        rows = sorted(by_problem.get(problem["problem_id"], []), key=lambda r: r["sample_index"])
        rewards = [None if version_drift or r["score"] is None else round(r["score"] * 10000) for r in rows]
        observations.append({
            "schema": "prompt-observation/v1", "context_sha256": contract.context_sha256,
            "row_id": problem["problem_id"], "group_id": "evaluation-samples",
            "expected_samples": sample_count, "sample_ids": [f"sample-{r['sample_index']}" for r in rows],
            "rewards_bps": rewards, "tokens": [r["tokens"] for r in rows], "window": 0,
            "verification": {"generation": "verified" if generation_verified else "unverified",
                             "sampling": "unverified", "grading": "error" if version_drift else "graded"},
            "source_sha256": source_digest,
        })
    manifest = export_mapping(directory, contract, observations, expected_rows=len(sets[set_id][1]),
                              group_semantics="evaluation-samples-per-problem")
    manifest["provenance"] = {"source_kind": "evaluation-grading", "graded_sha256": source_digest,
                              "rl_group_comparable": False}
    (directory / "mapping-manifest.json").write_bytes(canonical_json_bytes(manifest))
    return [directory / "mapping.jsonl", directory / "mapping-manifest.json"]


def mapped_group(row: dict, contract: ServiceContract) -> dict:
    value = validate_observation(row, contract)
    return {**value, "schema": "mapped-group/v1", "observation_id": observation_id(value),
            "classification": asdict(observation_signal(value, contract))}


def mapping_artifact(contract: ServiceContract, observations: Iterable[dict], *, expected_rows: int,
                     group_semantics: str = "declared-group", source_manifest_sha256: str | None = None) -> tuple[bytes, dict]:
    if type(expected_rows) is not int or expected_rows < 1:
        raise ValueError("expected_rows must be positive")
    groups, ids = [], set()
    token_total = 0
    limits = contract.to_dict()["limits"]
    for row in observations:
        mapped = mapped_group(row, contract)
        if mapped["observation_id"] in ids:
            raise ValueError("duplicate observation in mapping")
        ids.add(mapped["observation_id"])
        token_total += sum(mapped["tokens"])
        if len(groups) >= limits["max_groups"] or token_total > limits["max_tokens"]:
            raise ValueError("mapping exceeds its ordered budget")
        groups.append(mapped)
    groups.sort(key=lambda r: (r["row_id"], r["group_id"]))
    rows = {r["row_id"] for r in groups}
    if len(rows) > expected_rows:
        raise ValueError("mapping contains more rows than declared population")
    body = b"".join(canonical_json_bytes(row) + b"\n" for row in groups)
    value = contract.to_dict()
    counts = Counter(r["classification"]["category"] for r in groups)
    complete = len(rows) == expected_rows and all(r["classification"]["category"] != "unknown" for r in groups)
    manifest = {
        "schema": "dataset-mapping/v1", "contract_sha256": contract.sha256,
        "context_sha256": contract.context_sha256, "dataset": value["dataset"],
        "checkpoint": value["checkpoint"], "environment": value["environment"], "scoring": value["scoring"],
        "groups": len(groups), "rows": len(rows), "expected_rows": expected_rows,
        "complete": complete, "category_counts": dict(sorted(counts.items())), "tokens": token_total,
        "group_semantics": group_semantics,
        "generation_verified": bool(groups) and all(r["verification"]["generation"] == "verified" for r in groups),
        "sampling_verified": bool(groups) and all(r["verification"]["sampling"] == "verified" for r in groups),
        "source_manifest_sha256": source_manifest_sha256,
        "files": {"mapping.jsonl": {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}},
    }
    return body, manifest


def export_mapping(directory: Path, contract: ServiceContract, observations: Iterable[dict], **kwargs) -> dict:
    body, manifest = mapping_artifact(contract, observations, **kwargs)
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in (("mapping.jsonl", body), ("mapping-manifest.json", canonical_json_bytes(manifest))):
        target = directory / name
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f"refusing to replace different artifact {name}")
        if not target.exists():
            temp = directory / f".{name}.tmp"
            temp.write_bytes(content)
            temp.replace(target)
    return manifest
