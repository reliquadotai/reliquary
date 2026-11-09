"""Durable, context-bound observations; no payment side effects."""

from __future__ import annotations

import json

from reliquary.protocol.release_contract import canonical_json_bytes, canonical_sha256
from reliquary.protocol.service_contract import ServiceContract, _integer, _identifier, _object, _sha
from reliquary.services.scoring import classify_signal

OBSERVATION_SCHEMA = "prompt-observation/v1"
FIELDS = {"schema", "context_sha256", "row_id", "group_id", "expected_samples", "sample_ids",
          "rewards_bps", "tokens", "window", "verification", "source_sha256"}


def validate_observation(row: dict, contract: ServiceContract) -> dict:
    _object(row, FIELDS, "observation")
    if row["schema"] != OBSERVATION_SCHEMA or row["context_sha256"] != contract.context_sha256:
        raise ValueError("observation context mismatch")
    for name in ("row_id", "group_id"):
        _identifier(row[name], name)
    _sha(row["source_sha256"], "source_sha256")
    count = _integer(row["expected_samples"], "expected_samples", 2, 65536)
    if any(not isinstance(row[name], list) for name in ("sample_ids", "rewards_bps", "tokens")):
        raise ValueError("sample_ids, rewards_bps and tokens must be arrays")
    n = len(row["sample_ids"])
    if n > count or len(row["rewards_bps"]) != n or len(row["tokens"]) != n:
        raise ValueError("sample arrays differ or exceed expected slots")
    for sid in row["sample_ids"]:
        _identifier(sid, "sample_id")
    if len(set(row["sample_ids"])) != n:
        raise ValueError("duplicate sample id")
    for reward in row["rewards_bps"]:
        if reward is not None:
            _integer(reward, "reward_bps", 0, 10000)
    for tokens in row["tokens"]:
        _integer(tokens, "tokens", 0)
    _integer(row["window"], "window", 0)
    verification = _object(row["verification"], {"generation", "sampling", "grading"}, "verification")
    if any(verification[name] not in ("verified", "unverified") for name in ("generation", "sampling")) or verification["grading"] not in ("graded", "error"):
        raise ValueError("unknown verification status")
    return json.loads(canonical_json_bytes(row))


def observation_id(row: dict) -> str:
    """A slot identity cannot be resubmitted with different rewards or evidence."""
    return canonical_sha256({k: row[k] for k in ("context_sha256", "row_id", "group_id")})


def observation_signal(row: dict, contract: ServiceContract):
    validated = validate_observation(row, contract)
    rewards = [r / 10000 if r is not None else None for r in validated["rewards_bps"]]
    if row["verification"]["grading"] != "graded":
        rewards = [None] * len(rewards)
    return classify_signal(rewards, expected=row["expected_samples"],
                           sigma_min_bps=contract.to_dict()["scoring"]["sigma_min_bps"])


