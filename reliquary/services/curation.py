"""Immutable source-preserving curation, with explicit generation/sampling gates."""
from __future__ import annotations
import hashlib
import json
from reliquary.protocol.service_contract import ServiceContract
from reliquary.services.exploration import STATUS_FORFEITED
from reliquary.services.heldout_guard import refuse_held_out
from reliquary.services.observations import observation_signal


def audit_passed(ledger, observation_id: str) -> bool:
    """Read-only: did the exploration audit of this observation PASS? ``ledger`` is an
    ``ExplorationLedger``; a group sampled but not drawn, drawn and not audited, or whose hotkey
    forfeited its earnings (audit failed on another group), is not."""
    state = ledger.state(observation_id)
    return state is not None and state[0] == "passed" and state[1] != STATUS_FORFEITED


def generation_proven(mapped: dict, observation: dict, ledger=None) -> bool:
    """R11: a group counts as generation-verified when it was fully proven, or when it is an
    exploration group whose audit passed (looked up by its run observation id)."""
    if observation["verification"]["generation"] == "verified":
        return True
    if ledger is None:
        return False
    return any(isinstance(key, str) and audit_passed(ledger, key)
               for key in (mapped.get("run_observation_id"), mapped.get("observation_id")))


def curate_rows(source_bytes: bytes, mapping_bytes: bytes, manifest: dict, contract: ServiceContract,
                *, set_card: dict, categories: set[str] | None = None, require_sampling: bool = False,
                ledger=None) -> tuple[bytes, dict]:
    """Export a new slice; source rows stay intact and unknown never means K=0.

    A row is taken only when every one of its groups is in ``categories``. Held-out
    evaluation sets are refused first (``set_card`` is the source set's card)."""
    refuse_held_out(contract, set_card)
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
    if not manifest.get("complete") or not (manifest.get("generation_verified") or ledger is not None):
        raise ValueError("curation requires a complete generation-verified mapping")
    if require_sampling and not manifest.get("sampling_verified"):
        raise ValueError("automatic eligibility requires sampling verification")
    groups_by_row: dict[str, list[str]] = {}
    for raw in mapping_bytes.splitlines():
        if not raw.strip():
            continue
        mapped = json.loads(raw)
        observation = {k: mapped[k] for k in ("context_sha256", "row_id", "group_id", "expected_samples", "sample_ids", "rewards_bps", "tokens", "window", "verification", "source_sha256")}
        observation["schema"] = "prompt-observation/v1"
        signal = observation_signal(observation, contract)
        if not generation_proven(mapped, observation, ledger):
            raise ValueError("curation requires generation verification for every mapped row")
        if require_sampling and observation["verification"]["sampling"] != "verified":
            raise ValueError("automatic eligibility requires sampling verification for every mapped row")
        if signal.category == "unknown":
            raise ValueError("complete mapping contains an unknown group")
        groups_by_row.setdefault(observation["row_id"], []).append(signal.category)
    selected = {row for row, cats in groups_by_row.items() if all(c in categories for c in cats)}
    mixed = sum(1 for cats in groups_by_row.values()
                if any(c in categories for c in cats) and not all(c in categories for c in cats))
    output, seen = [], set()
    for line in source_bytes.splitlines(keepends=True):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("source rows must be JSON objects")
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
                  "categories": sorted(categories), "sampling_verified": manifest["sampling_verified"],
                  "mixed_rows": mixed, "generation_audit": manifest.get("generation_audit", "full")}
