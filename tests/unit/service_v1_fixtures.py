"""Shared service-contract/v1 and observation builders."""

import hashlib
import json
from pathlib import Path

from reliquary.protocol.service_contract import ServiceContract


def contract(source=None):
    value = json.loads((Path(__file__).parents[1] / "fixtures/service_contract_v1.json").read_text())
    if source is not None:
        value["dataset"]["sha256"] = hashlib.sha256(source).hexdigest()
    return ServiceContract.from_dict(value)


def observation(c, row_id="row-1", rewards=(0, 10000), **kw):
    return {"schema": "prompt-observation/v1", "context_sha256": c.context_sha256,
            "row_id": row_id, "group_id": "group-0", "expected_samples": 2,
            "sample_ids": [f"sample-{i}" for i in range(len(rewards))], "rewards_bps": list(rewards),
            "tokens": [10] * len(rewards), "window": 1,
            "verification": {"generation": "verified", "sampling": "unverified", "grading": "graded"},
            "source_sha256": "f" * 64, **kw}
