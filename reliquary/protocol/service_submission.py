"""Explicit service intent bound to both the envelope and each rollout."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

SUBMISSION_SCHEMA = "service-submission/v1"
ROLLOUT_SCHEMA = "service-rollout/v1"
PROOF_VERSION = "service-group-proof/v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class ServiceBinding:
    contract_sha256: str
    purpose: str

    def __post_init__(self) -> None:
        if not isinstance(self.contract_sha256, str) or not _SHA.fullmatch(self.contract_sha256):
            raise ValueError("lowercase service contract SHA-256 required")
        if not isinstance(self.purpose, str) or self.purpose not in ("training", "exploration"):
            raise ValueError("unsupported service submission purpose")

    @classmethod
    def from_dict(cls, value: Any) -> ServiceBinding:
        if not isinstance(value, dict) or set(value) != {"schema", "contract_sha256", "purpose"} or value["schema"] != SUBMISSION_SCHEMA:
            raise ValueError("unknown service submission binding")
        return cls(value["contract_sha256"], value["purpose"])

    def to_dict(self) -> dict:
        return {"schema": SUBMISSION_SCHEMA, "contract_sha256": self.contract_sha256,
                "purpose": self.purpose}

    def rollout_binding(self, rollout_index: int) -> dict:
        if type(rollout_index) is not int or not 0 <= rollout_index <= 63:
            raise ValueError("bounded canonical rollout index required")
        return {"schema": ROLLOUT_SCHEMA, "contract_sha256": self.contract_sha256,
                "purpose": self.purpose, "rollout_index": rollout_index}


def parse_service_rollout_binding(value: Any) -> tuple[ServiceBinding, int]:
    if not isinstance(value, dict) or set(value) != {"schema", "contract_sha256", "purpose", "rollout_index"} or value["schema"] != ROLLOUT_SCHEMA:
        raise ValueError("unknown service rollout binding")
    binding = ServiceBinding(value["contract_sha256"], value["purpose"])
    index = value["rollout_index"]
    binding.rollout_binding(index)
    return binding, index


