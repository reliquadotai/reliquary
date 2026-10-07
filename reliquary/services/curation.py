"""Immutable source-preserving curation, with explicit generation/sampling gates."""
from __future__ import annotations
import hashlib
import json
from reliquary.protocol.service_contract import ServiceContract
from reliquary.services.observations import observation_signal


def curate_rows(source_bytes: bytes, mapping_bytes: bytes, manifest: dict, contract: ServiceContract,
                *, categories: set[str] | None = None, require_sampling: bool = False) -> tuple[bytes, dict]:
    """Export a new slice; source rows stay intact and unknown never means K=0."""
    categories = {"in-zone"} if categories is None else categories
    allowed = {"in-zone", "uniform-low", "uniform-high", "uniform-intermediate", "diverse-below-threshold"}
    if not categories or not categories <= allowed:
        raise ValueError("unknown is not a curated training category")
    if manifest.get("schema") != "dataset-mapping/v1" or manifest.get("context_sha256") != contract.context_sha256:
        raise ValueError("mapping context mismatch")
    if hashlib.sha256(source_bytes).hexdigest() != contract.to_dict()["dataset"]["sha256"]:
        raise ValueError("source dataset digest mismatch")
    if hashlib.sha256(mapping_bytes).hexdigest() != manifest.get("files", {}).get("mapping.jsonl", {}).get("sha256"):
        raise ValueError("mapping file digest mismatch")
    if not manifest.get("complete") or not manifest.get("generation_verified"):
        raise ValueError("curation requires a complete generation-verified mapping")
    if require_sampling and not manifest.get("sampling_verified"):
        raise ValueError("automatic eligibility requires sampling verification")
    selected = set()
    for raw in mapping_bytes.splitlines():
        mapped = json.loads(raw)
        observation = {k: mapped[k] for k in ("context_sha256", "row_id", "group_id", "expected_samples", "sample_ids", "rewards_bps", "tokens", "window", "verification", "source_sha256")}
        observation["schema"] = "prompt-observation/v1"
        signal = observation_signal(observation, contract)
        if observation["verification"]["generation"] != "verified":
            raise ValueError("curation requires generation verification for every mapped row")
        if require_sampling and observation["verification"]["sampling"] != "verified":
            raise ValueError("automatic eligibility requires sampling verification for every mapped row")
        if signal.category == "unknown":
            raise ValueError("complete mapping contains an unknown group")
        if signal.category in categories:
            selected.add(observation["row_id"])
    output, seen = [], set()
    for line in source_bytes.splitlines(keepends=True):
        row = json.loads(line)
        identifier = row.get("row_id", row.get("problem_id"))
        if not isinstance(identifier, str) or identifier in seen:
            raise ValueError("source rows need unique row_id/problem_id")
        seen.add(identifier)
        if identifier in selected:
            output.append(line)
    if not selected <= seen:
        raise ValueError("mapped rows absent from source dataset")
    body = b"".join(output)
    return body, {"schema": "curated-dataset/v1", "context_sha256": contract.context_sha256,
                  "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                  "mapping_sha256": hashlib.sha256(mapping_bytes).hexdigest(),
                  "sha256": hashlib.sha256(body).hexdigest(), "rows": len(output),
                  "categories": sorted(categories), "sampling_verified": manifest["sampling_verified"]}
