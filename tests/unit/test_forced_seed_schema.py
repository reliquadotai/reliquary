from reliquary.protocol.submission import RejectReason, BatchSubmissionRequest


def test_seed_mismatch_reason_exists():
    assert RejectReason.SEED_MISMATCH.value == "seed_mismatch"


def test_protocol_version_defaults_zero_and_accepts_int():
    fields = BatchSubmissionRequest.model_fields
    assert "protocol_version" in fields
    assert fields["protocol_version"].default == 0
