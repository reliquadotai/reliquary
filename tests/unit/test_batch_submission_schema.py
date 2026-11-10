"""Pydantic schemas for v2 GRPO market submissions."""

import pytest
from pydantic import ValidationError

from reliquary.constants import M_ROLLOUTS
from reliquary.protocol.submission import RejectReason, WindowState


def test_new_reject_reasons_exist():
    """Schema/Token/Termination validators emit dedicated reject codes."""
    assert RejectReason.BAD_SCHEMA.value == "bad_schema"
    assert RejectReason.BAD_TOKENS.value == "bad_tokens"
    assert RejectReason.TOKENS_MISMATCH.value == "tokens_mismatch"
    assert RejectReason.BAD_TERMINATION.value == "bad_termination"
    assert RejectReason.REWARD_DISTRIBUTION.value == "reward_distribution"
    assert RejectReason.REWARD_SHAPE_SUSPICIOUS.value == "reward_shape_suspicious"


def _announcement_dict(**mutate):
    from reliquary.protocol.service_contract import SUPPORTED_V2_CAPABILITIES, ServiceContract
    from reliquary.protocol.service_schedule import initial_schedule
    from tests.unit.service_v2_fixtures import contract_v2_dict
    value = contract_v2_dict()
    for env in value["environments"].values():
        env["sampling"].update(mutate)
    contract = ServiceContract.from_dict(value)
    return {"contract": contract.to_dict(), "schedule": initial_schedule(contract).to_dict(),
            "checkpoint": {"checkpoint_n": 3, "repo": "models/test", "revision": "d" * 40, "sha256": "e" * 64},
            "supported_capabilities": sorted(SUPPORTED_V2_CAPABILITIES), "pool_epoch": 5, "pool_randomness": "ab" * 32}


def test_announcement_with_a_seed_pool_group_size_other_than_m_is_refused():
    from reliquary.protocol.submission import ServicePolicyAnnouncement
    assert ServicePolicyAnnouncement.model_validate(_announcement_dict()).pool_epoch == 5      # positive control
    other = M_ROLLOUTS + 1
    with pytest.raises(ValidationError, match="group_size must be M_ROLLOUTS"):
        ServicePolicyAnnouncement.model_validate(_announcement_dict(group_size=other, pool_seeds=2 * other))


