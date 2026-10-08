import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from reliquary.admin.service import GradeEvaluation

PROVENANCE = {"model": "m", "revision": "r", "model_sha": "x", "sampling": {}, "thinking": False,
              "max_new_tokens": 8, "vllm_version": "v", "gpu": "g", "pod_provider_id": "p"}


def _contract():
    return json.loads((Path(__file__).parents[1] / "fixtures/service_contract_v1.json").read_text())


def test_uploads_cannot_carry_a_service_contract():
    body = {"source": "uploads", "set_ids": ["s"], "completion_keys": ["k"], "problems_per_set": {"s": 1},
            "samples_per_set": {"s": 2}, "provenance": PROVENANCE}
    GradeEvaluation(**body)  # the same body without a contract is valid, so the refusal is the contract's
    with pytest.raises(ValidationError, match="cannot be verified"):
        GradeEvaluation(**body, service_contract=_contract())


def test_job_grading_may_carry_a_service_contract():
    GradeEvaluation(source="job", job_id="j", set_ids=["s"], problems_per_set={"s": 1},
                    samples_per_set={"s": 2}, service_contract=_contract())
