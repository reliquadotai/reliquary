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


def validate_service_rollout_bindings(binding: ServiceBinding | dict,
                                     commits: list[dict], *, signed_episodes: bool = False) -> None:
    """Require the exact envelope intent and original index on every rollout. ``signed_episodes``
    (plan 2C): every rollout carries a signed episode; otherwise none may carry any episode."""
    if isinstance(binding, dict):
        binding = ServiceBinding.from_dict(binding)
    if not isinstance(binding, ServiceBinding) or not 2 <= len(commits) <= 64:
        raise ValueError("bounded complete service group required")
    for index, commit in enumerate(commits):
        metadata = commit.get("rollout")
        if not isinstance(metadata, dict):
            raise ValueError("service group bindings require single-turn rollouts")
        episode = metadata.get("episode")
        if signed_episodes:
            from reliquary.protocol.submission import is_signed_episode

            if not is_signed_episode(metadata):
                raise ValueError("this environment takes signed episodes")
        elif episode is not None:
            raise ValueError("service group bindings require single-turn rollouts")
        actual, original_index = parse_service_rollout_binding(metadata.get("service_binding"))
        if actual != binding or original_index != index:
            raise ValueError("service rollout intent differs from its envelope")
