"""v2.1 schema extensions: state in GrpoBatchState, checkpoint_hash in BatchSubmissionRequest."""

import pytest
from pydantic import ValidationError

from reliquary.constants import M_ROLLOUTS
from reliquary.protocol.submission import WindowState


def test_behavioural_reject_reasons_exist():
    """Logprob + Distribution validator ports add two new reject codes."""
    from reliquary.protocol.submission import RejectReason
    assert RejectReason.LOGPROB_MISMATCH.value == "logprob_mismatch"
    assert RejectReason.DISTRIBUTION_SUSPICIOUS.value == "distribution_suspicious"
